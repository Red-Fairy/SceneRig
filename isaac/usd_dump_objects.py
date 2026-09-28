"""Dump each resolved GRASE object root from USD to npz (run with the Isaac venv).

    $SCENERIG_ISAAC_PYTHON isaac/usd_dump_objects.py \
        <scene_visual.usdc> <out.npz> <placement.json> <object_identity.json>

World-frame vertices/faces per explicitly resolved object root, keyed by the USD prim name
(so downstream collider files match the exported scene exactly). The accompanying identity
manifest records the pipeline name, exported name, and absolute root path. Colliders must be
built from THIS geometry, not the placed GLBs: agent stages (register/composition) move
objects in the blend after placement, so GLB poses can be stale — 0708_phys_real8334's glue
bottle was 24 cm from its GLB, which put its collider inside a neighbor's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

try:
    from .export_identity import (
        ExportIdentityError,
        build_identity_manifest,
        usd_identifier,
        write_identity_manifest,
    )
    from .mesh_triangulation import triangulate_mesh_faces
except ImportError:  # direct ``python isaac/usd_dump_objects.py`` execution
    from export_identity import (
        ExportIdentityError,
        build_identity_manifest,
        usd_identifier,
        write_identity_manifest,
    )
    from mesh_triangulation import triangulate_mesh_faces


def _pipeline_names(placement_path):
    data = json.loads(Path(placement_path).read_text())
    names = [
        record.get("mesh_name")
        for record in data.get("objects", [])
        if record.get("mesh_name")
    ]
    if not names:
        raise ExportIdentityError(f"no pipeline mesh names in {placement_path}")
    return names


def _nested_expected_roots(root, pipeline_names):
    """Return expected object-looking prims found below a top-level object root."""
    expected = {usd_identifier(name): name for name in pipeline_names}
    nested = []
    for top in root.GetChildren():
        if not top.GetName().startswith("obj_"):
            continue
        for prim in Usd.PrimRange(top):
            if prim == top:
                continue
            pipeline_name = expected.get(prim.GetName())
            if pipeline_name:
                nested.append((pipeline_name, str(prim.GetPath()), str(top.GetPath())))
    return nested


PART_MARKER = "_collision_part_"  # blender_collision_source authored part children


def main(src, out, placement_path, manifest_path):
    stage = Usd.Stage.Open(str(src))
    if stage is None:
        raise RuntimeError(f"could not open USD stage {src}")
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot().GetChildren()[0]
    pipeline_names = _pipeline_names(placement_path)
    nested = _nested_expected_roots(root, pipeline_names)
    if nested:
        detail = ", ".join(
            f"{name} at {path} under {parent}" for name, path, parent in nested
        )
        raise ExportIdentityError(
            f"nested pipeline object roots are forbidden: {detail}"
        )

    roots = [
        (prim.GetName(), str(prim.GetPath()))
        for prim in root.GetChildren()
        if prim.GetName().startswith("obj_")
    ]
    manifest = build_identity_manifest(pipeline_names, roots)
    mapped_paths = {record["prim_path"] for record in manifest["objects"]}
    unmatched = sorted(path for _, path in roots if path not in mapped_paths)
    if unmatched:
        raise ExportIdentityError(
            f"unmatched exported obj_* roots are forbidden: {unmatched}"
        )

    cache = UsdGeom.XformCache()
    data, names = {}, []
    for record in manifest["objects"]:
        obj = record["usd_name"]
        object_root = stage.GetPrimAtPath(record["prim_path"])
        meshes = [prim for prim in Usd.PrimRange(object_root) if prim.IsA(UsdGeom.Mesh)]
        if not meshes:
            raise ExportIdentityError(
                f"exported object root {record['prim_path']} has no descendant meshes"
            )
        for prim in meshes:
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
            M = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            world = pts @ M[:3, :3].T + M[:3, 3]
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            if m.GetHoleIndicesAttr().Get():
                raise ValueError(
                    f"{prim.GetPath()}: meshes with authored holeIndices are unsupported"
                )
            tris = triangulate_mesh_faces(
                world, counts, idx, mesh_name=str(prim.GetPath())
            )
            if PART_MARKER in prim.GetName():  # authored part: separate cook input
                # index from the name (``<obj>_collision_part_NNN``): USD child order
                # is not the union's operand order
                key = f"part_{obj}_{int(prim.GetName().rsplit('_', 1)[1])}"
                if key in data:
                    raise ExportIdentityError(f"duplicate authored part prim {prim.GetPath()}")
                data[key] = (world, np.asarray(tris, dtype=np.int64))
                continue
            if obj in data:  # multi-mesh body: merge within this explicit root only
                v0, f0 = data[obj]
                data[obj] = (
                    np.vstack([v0, world]),
                    np.vstack([f0, np.asarray(tris) + len(v0)]),
                )
            else:
                data[obj] = (world, np.asarray(tris, dtype=np.int64))
                names.append(obj)
    flat = {"names": np.asarray(names)}
    for n, (v, f) in data.items():
        flat[f"v_{n}"] = v  # bodies as v_<obj>; authored parts as v_part_<obj>_<i>
        flat[f"f_{n}"] = np.asarray(f, dtype=np.int64)
        if n.startswith("part_"):
            obj = n[len("part_"):].rsplit("_", 1)[0]
            flat[f"n_parts_{obj}"] = np.asarray(int(flat.get(f"n_parts_{obj}", 0)) + 1)
    np.savez(out, **flat)
    # Publish the identity contract only after every mapped root produced geometry and
    # the dump completed. A failed export must not leave a plausible but partial manifest.
    write_identity_manifest(manifest_path, manifest)
    print(f"USD_DUMP_OK {out} objects={len(names)} identity={manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
