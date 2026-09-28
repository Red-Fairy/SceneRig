"""SAM3D Worker for 3D Reconstruction from Masked Images.

This script uses SAM3D to reconstruct a 3D mesh from an image and its
segmentation mask, then transforms the mesh vertices to world coordinates
and exports as GLB format.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d", "notebook"))
sys.path.append(os.path.join(ROOT, "utils", "third_party", "sam3d"))

# Must run before importing `inference`: notebook/inference.py reads CONDA_PREFIX
# (sets CUDA_HOME from it) at import time. Direct launchers use the conda env's
# Python binary, but that alone does not guarantee CONDA_PREFIX is set.
if "CONDA_PREFIX" not in os.environ:
    python_bin = sys.executable
    conda_env = os.path.dirname(os.path.dirname(python_bin))
    os.environ["CONDA_PREFIX"] = conda_env

from inference import Inference, load_image

# GLB mesh (Y-up) -> pose frame (Z-up), a PROPER rotation (det +1). Matches REST3D's
# reference (`_R_ZUP_TO_YUP.T`). The earlier `R_flip_z @ R_yup_to_zup` chain was a det -1
# REFLECTION that MIRRORED every reconstructed object left/right (wallet zip on the wrong
# side, AirPods flipped, etc.) -- see CHANGELOG 2026-06-30.
R_YUP_TO_ZUP: torch.Tensor = torch.tensor(
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=torch.float32
)


def load_pointmap_p3d(path: str) -> torch.Tensor:
    """Load a MoGE camera-space point map ``(H, W, 3)`` from ``path`` and rotate it into
    the PyTorch3D camera frame the SAM3D point-map condition was trained on.

    This reproduces the transform ``InferencePipelinePointMap.compute_pointmap`` applies
    to its INTERNAL (MoGE-v1) estimate before embedding it -- the pipeline's ``pointmap=``
    path deliberately skips that rotation (it assumes the caller already supplies points in
    PyTorch3D frame). So pre-applying it here lets us condition on the scene's already-
    computed MoGE-2 point map (``<out>/moge/points.npy``, MoGE camera frame: +X right,
    +Y down, +Z forward) instead of re-running depth internally, in the right convention.
    Reusing ``camera_to_pytorch3d_camera`` keeps a single source of truth for the frame.
    """
    from sam3d_objects.pipeline.inference_pipeline_pointmap import (
        camera_to_pytorch3d_camera,
    )

    pts = torch.from_numpy(
        np.load(path).astype(np.float32)
    )  # (H, W, 3), MoGE cam frame
    h, w, _ = pts.shape
    rotation = camera_to_pytorch3d_camera(device="cpu").rotation
    pts = Transform3d().rotate(rotation).transform_points(pts.reshape(-1, 3))
    return pts.reshape(h, w, 3)


def transform_mesh_vertices(
    vertices: np.ndarray,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Transform mesh vertices from model space to world coordinates (REST3D convention).

    Steps (matching REST3D's `make_scene_untextured_mesh`):
    1. Y-up GLB mesh -> Z-up pose frame via ``R_YUP_TO_ZUP`` (proper rotation, det +1 -- no
       left/right mirror).
    2. Up/down fix: ``R_mat`` row 2 is where the object's local +Z lands; if its Y-component
       is negative the reconstruction is inverted, so apply a 180deg flip (det +1; axis from
       ``R_mat[1, 2]``) BEFORE posing so the object rests upright. Uniform per-object rule --
       only flips objects whose predicted pose is inverted. (Does NOT fix ~90deg articulated
       cases like an open laptop, which need a separate resolver.)
    3. Apply the model's local-to-camera pose (scale, rotation, translation).

    Args:
        vertices: Mesh vertices as numpy array (N, 3).
        rotation: Quaternion rotation tensor (the predicted local-to-camera pose).
        translation: Translation vector tensor.
        scale: Scale factor tensor.

    Returns:
        Transformed vertices as torch tensor (N, 3).
    """
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  # Add batch dimension [1, N, 3]
    vertices = vertices @ R_YUP_TO_ZUP.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    if float(R_mat[2, 1]) < 0:  # local +Z lands with -Y -> inverted
        if float(R_mat[1, 2]) > 0:
            vertices[..., 0] *= -1.0  # 180deg about Y (negate X, Z)
            vertices[..., 2] *= -1.0
        else:
            vertices[..., 1] *= -1.0  # 180deg about X (negate Y, Z)
            vertices[..., 2] *= -1.0
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
        .rotate(R_mat)
        .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)
    return vertices_world[0]  # Remove batch dimension


def main() -> None:
    """Run SAM3D reconstruction on a masked image and export as GLB."""
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--mask", required=True, help="Path to mask npy file")
    p.add_argument("--config", required=True, help="Path to SAM3D config file")
    p.add_argument("--glb", required=True, help="Path for output GLB file")
    p.add_argument(
        "--info", required=False, help="Path to save JSON output (instead of stdout)"
    )
    p.add_argument(
        "--pristine",
        required=False,
        help="Path to save the UNtransformed output['glb'] (canonical, upright; for the "
        "rule-out fallback on weird poses)",
    )
    p.add_argument(
        "--pointmap",
        required=False,
        help="Path to a MoGE camera-space point map .npy (H,W,3) to use as the SAM3D "
        "condition instead of re-estimating depth internally (see load_pointmap_p3d).",
    )
    args = p.parse_args()

    inference = Inference(args.config, compile=False)
    image = load_image(args.image)
    # Load mask from npy file
    mask = np.load(args.mask)
    mask = mask > 0
    pointmap = load_pointmap_p3d(args.pointmap) if args.pointmap else None
    output = inference(image, mask, seed=42, pointmap=pointmap)

    mesh = output["glb"]
    vertices = mesh.vertices
    if args.pristine:  # save BEFORE the pose transform mutates vertices
        os.makedirs(os.path.dirname(args.pristine), exist_ok=True)
        mesh.export(args.pristine)

    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()

    vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
    mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(args.glb), exist_ok=True)
    mesh.export(args.glb)

    def _pose_lists(p):  # serialize an intermediate ICP pose dict (or None)
        if p is None:
            return None
        return {
            "translation": p["translation"][0].cpu().float().tolist(),
            "rotation": p["rotation"].squeeze().cpu().float().tolist(),
            "scale": p["scale"][0].cpu().float().tolist(),
        }

    # Prepare output data
    translation_data = {
        "glb_path": args.glb,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.tolist(),
        # Poses bracketing SAM3D's shape-ICP step (None if optimizer returned early).
        "icp_pre": _pose_lists(output.get("pose_pre_icp")),
        "icp_post": _pose_lists(output.get("pose_post_icp")),
        "icp_applied": (
            None if output.get("flag_icp") is None else bool(output.get("flag_icp"))
        ),
    }

    # Write to file if --info provided, otherwise print to stdout for backward compatibility
    if args.info:
        os.makedirs(os.path.dirname(args.info), exist_ok=True)
        with open(args.info, "w") as f:
            json.dump(translation_data, f, indent=2)
    else:
        print(json.dumps(translation_data))


if __name__ == "__main__":
    main()
