"""Decompose a scene's objects into CoACD collider parts (run with ./.venv).

    # export chain (authoritative): from the exported USD's own geometry
    .venv/bin/python isaac/build_collision.py --from-dump <visual_meshes.npz> --out <dir>

    # legacy: from the placed GLBs in placement.json
    .venv/bin/python isaac/build_collision.py <exp_or_task_dir> [--out <dir>]

Writes one world-frame parts file per object: <out>/<prim_name>.npz, which
``isaac/isaac_add_physics.py`` authors as the USD collision prims.

The --from-dump mode consumes ``isaac/usd_dump_objects.py`` output and MUST be used
for the export chain: agent stages (register/composition) move objects in the blend
after placement, so placed-GLB poses can be stale — 0708_phys_real8334's glue bottle
sat 24 cm from its GLB, which put its GLB-derived collider inside a neighbor's.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.tools.geometry.collision import (  # noqa: E402
    decompose_authored_parts,
    decompose_glb,
    decompose_world_mesh,
)


def find_task_dir(d: Path) -> Path:
    for c in (d, d / "scene"):
        if (c / "placement.json").exists():
            return c
    raise SystemExit(f"no placement.json under {d}")


def _supports_near(start: Path) -> set[str]:
    """Support-object mesh names from the nearest scene_graph.json at or above
    ``start`` — these get the finer SUPPORT_* CoACD budget. {} = all default."""
    from lib.tools.geometry.physics import support_mesh_names

    for d in (start, *start.parents[:3]):
        gp = d / "scene_graph.json"
        if gp.exists():
            try:
                return support_mesh_names(json.loads(gp.read_text()).get("nodes", []))
            except Exception:  # noqa: BLE001 - budget selection is best-effort
                return set()
    return set()


def _from_dump(
    dump: str,
    name: str,
    out_npz: str,
    support: bool = False,
    frame_glb: str | None = None,
) -> int:
    d = np.load(dump, allow_pickle=False)
    n_parts = int(d[f"n_parts_{name}"]) if f"n_parts_{name}" in d else 0
    if n_parts and not support:  # authored object: hull each part (collision.py)
        part_meshes = [(d[f"v_part_{name}_{i}"], d[f"f_part_{name}_{i}"]) for i in range(n_parts)]
        return decompose_authored_parts(
            part_meshes, d[f"v_{name}"], d[f"f_{name}"], out_npz, frame_glb
        )
    return decompose_world_mesh(
        d[f"v_{name}"], d[f"f_{name}"], out_npz, support, frame_glb
    )


def _pristine_by_name(start: Path) -> dict[str, str]:
    """mesh_name -> pristine GLB path from the nearest placement.json (best-effort);
    the exact local-frame overshoot clamp source (collision._pristine_frame)."""
    for d in (start, *start.parents[:3]):
        pp = d / "placement.json"
        if pp.exists():
            try:
                # placement paths are GRASE-root-relative (the pipeline's cwd
                # convention); this script runs from anywhere, so resolve them —
                # same treatment as mesh_glb in the GLB branch below.
                return {
                    o["mesh_name"]: str(
                        Path(o["pristine_glb"])
                        if Path(o["pristine_glb"]).is_absolute()
                        else REPO_ROOT / o["pristine_glb"]
                    )
                    for o in json.loads(pp.read_text()).get("objects", [])
                    if o.get("mesh_name") and o.get("pristine_glb")
                }
            except Exception:  # noqa: BLE001 - the clamp is an enhancement
                return {}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("exp_dir", nargs="?", help="exp/task dir (GLB mode)")
    ap.add_argument("--from-dump", default=None, help="visual_meshes.npz from usd_dump_objects.py")
    ap.add_argument("--out", default=None, help="output dir (default <task>/isaac/collision)")
    a = ap.parse_args()

    ctx = multiprocessing.get_context("spawn")  # fork deadlocks under CoACD threads
    if a.from_dump:
        out = Path(a.out or Path(a.from_dump).parent / "collision")
        out.mkdir(parents=True, exist_ok=True)
        names = [str(n) for n in np.load(a.from_dump)["names"]]
        sup = _supports_near(Path(a.from_dump).resolve().parent)
        prs = _pristine_by_name(Path(a.from_dump).resolve().parent)
        args = [
            (a.from_dump, n, str(out / f"{n}.npz"), n in sup, prs.get(n))
            for n in names
        ]
        with ProcessPoolExecutor(max_workers=min(4, max(1, len(args))), mp_context=ctx) as ex:
            counts = list(ex.map(_from_dump, *zip(*args)))
        for n, c in zip(names, counts):
            print(f"  {n}: {c} parts")
        print(f"COLLISION_OK {out} objects={len(names)}")
        return

    task = find_task_dir(Path(a.exp_dir).resolve())
    out = Path(a.out) if a.out else task / "isaac/collision"
    out.mkdir(parents=True, exist_ok=True)
    sup = _supports_near(task)
    tasks = []
    for o in json.load(open(task / "placement.json")).get("objects", []):
        glb, name = o.get("mesh_glb"), o.get("mesh_name")
        if not (glb and name):
            continue
        gp = Path(glb) if Path(glb).is_absolute() else REPO_ROOT / glb
        if gp.exists():
            pr = o.get("pristine_glb")
            if pr and not Path(pr).is_absolute():
                pr = str(REPO_ROOT / pr)
            tasks.append((str(gp), str(out / f"{name}.npz"), name in sup, pr))
    with ProcessPoolExecutor(max_workers=min(4, max(1, len(tasks))), mp_context=ctx) as ex:
        counts = list(ex.map(decompose_glb, *zip(*tasks)))
    for (glb, npz, *_), n in zip(tasks, counts):
        print(f"  {Path(npz).stem}: {n} parts")
    print(f"COLLISION_OK {out} objects={len(tasks)}")


if __name__ == "__main__":
    main()
