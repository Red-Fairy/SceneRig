"""Persistent SAM3D reconstruction server.

Loads the heavy SAM3D pipeline once, then answers many (image, mask) -> GLB
requests over a newline-delimited JSON protocol on stdin/stdout. Spawning
``sam3d_worker.py`` per object re-loads the ~13 GB pipeline every time; a
persistent server amortizes that across all instances of a scene.

Protocol (one JSON object per line):
  request  -> {"image": <path>, "mask": <mask.npy>, "glb": <out.glb>,
               "info": <out.json|optional>, "seed": <int|optional>,
               "pointmap": <points.npy|optional>,
               "pristine": <untransformed-canonical.glb|optional>}
             {"cmd": "ping"} | {"cmd": "shutdown"}
  response <- {"ok": true, "glb_path": <path>, "translation": [...],
               "rotation": [...], "scale": [...], "n_vertices": N,
               "icp_pre": <pose|null>, "icp_post": <pose|null>,
               "icp_applied": <bool|null>}
             {"ok": false, "reason": "error", "error": <msg>}
             {"ready": true}                                   (after model load)

Runs in the dependency-isolated SAM3D micromamba env (see lib/utils/_path.py).
The mesh transform mirrors ``sam3d_worker.transform_mesh_vertices`` (reused
directly) so server and worker produce identical output. Everything except the
protocol goes to stderr so stdout stays pure JSON.
"""

import json
import os
import sys

import numpy as np

_protocol_fd = os.dup(sys.stdout.fileno())
_real_stdout = os.fdopen(_protocol_fd, "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

# sam3d_worker sets CONDA_PREFIX + sys.path and imports the SAM3D `inference`
# module at import time; reuse its Inference/load_image/transform_mesh_vertices.
sys.path.insert(0, os.path.dirname(__file__))
import sam3d_worker as w  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CONFIG = os.path.join(
    ROOT, "utils", "third_party", "sam3d", "checkpoints", "hf", "pipeline.yaml"
)


def _respond(obj):
    _real_stdout.write(json.dumps(obj) + "\n")
    _real_stdout.flush()


def _pose_lists(p):
    """Serialize an intermediate ICP pose dict (cpu tensors from _decompose_pose) to
    plain lists, matching the top-level translation/rotation/scale layout. None -> None."""
    if p is None:
        return None
    return {
        "translation": p["translation"][0].cpu().float().tolist(),
        "rotation": p["rotation"].squeeze().cpu().float().tolist(),
        "scale": p["scale"][0].cpu().float().tolist(),
    }


def _reconstruct(inference, req):
    image = w.load_image(req["image"])
    mask = np.load(req["mask"]) > 0
    pointmap = w.load_pointmap_p3d(req["pointmap"]) if req.get("pointmap") else None
    output = inference(image, mask, seed=int(req.get("seed", 42)), pointmap=pointmap)
    mesh = output["glb"]
    S = output["scale"][0].cpu().float()
    T = output["translation"][0].cpu().float()
    R = output["rotation"].squeeze().cpu().float()
    if req.get("pristine"):  # UNtransformed canonical mesh (rule-out fallback)
        os.makedirs(os.path.dirname(req["pristine"]), exist_ok=True)
        mesh.export(req["pristine"])
    verts = w.transform_mesh_vertices(mesh.vertices, R, T, S)
    mesh.vertices = verts.cpu().numpy().astype(np.float32)
    glb = req["glb"]
    os.makedirs(os.path.dirname(glb), exist_ok=True)
    mesh.export(glb)
    info = {
        "glb_path": glb,
        "translation": T.tolist(),
        "rotation": R.tolist(),
        "scale": S.tolist(),
        "n_vertices": int(len(mesh.vertices)),
        # Poses bracketing SAM3D's shape-ICP step (camera space; same
        # translation/rotation-quat/scale layout as above). None if the layout
        # optimizer returned early (occlusion / no target points).
        "icp_pre": _pose_lists(output.get("pose_pre_icp")),
        "icp_post": _pose_lists(output.get("pose_post_icp")),
        "icp_applied": (
            None if output.get("flag_icp") is None else bool(output.get("flag_icp"))
        ),
    }
    if req.get("info"):
        os.makedirs(os.path.dirname(req["info"]), exist_ok=True)
        with open(req["info"], "w") as f:
            json.dump(info, f, indent=2)
    return info


def main():
    config = os.environ.get("SAM3D_CONFIG", DEFAULT_CONFIG)
    inference = w.Inference(config, compile=False)
    _respond({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            _respond({"ok": False, "reason": "error", "error": "bad json"})
            continue
        cmd = req.get("cmd")
        if cmd == "shutdown":
            break
        if cmd == "ping":
            _respond({"ok": True, "pong": True})
            continue
        try:
            info = _reconstruct(inference, req)
            _respond({"ok": True, **info})
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the server
            _respond({"ok": False, "reason": "error", "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
