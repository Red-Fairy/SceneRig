"""Convert a GRASE experiment's final.blend into an Isaac-ready scene.usd.

    python isaac/blend_to_isaac.py <exp_dir> [--skip-verify]
    # e.g. python isaac/blend_to_isaac.py output/static_scene/0701_v2_real8334

Pipeline (the driver runs in the repo `.venv`; it imports numpy/scipy/trimesh helpers):
  1. repo Blender (headless): final.blend -> visual USD plus collision-source USD;
     authored roots use the harness's shared Boolean union at the current pose.
  2. Isaac venv python: resolve placement names to independent USD roots, write
     scene/isaac/object_identity.json, and dump their world meshes to visual_meshes.npz
     [usd_dump_objects.py]; validate paired geometry/provenance before cooking.
  3. repo .venv python: CoACD collider parts -> scene/isaac/collision/*.npz  [build_collision.py --from-dump]
     (from current evaluated collision geometry: GLB poses can be stale)
  4. seed physics_overrides.json from pose_changes.json using mass/COM stabilization helpers
  5. translate the preprocess VLM physics estimate into Isaac object names/masses [translate_vlm_physics.py]
  6. Isaac venv python: stamp rigid bodies/colliders  -> scene/isaac/scene.usd [isaac_add_physics.py]
  7. Isaac venv python: 5 s headless settle test      -> scene/isaac/verify_report.json [isaac_verify_settle.py]
     (5 mm drift tolerance: the pipeline settle ran the same engine + colliders)

Output: <exp_dir>/scene/isaac/scene.usd — reference it from Isaac Lab via UsdFileCfg.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from .export_identity import load_identity_manifest
except ImportError:  # direct ``python isaac/blend_to_isaac.py`` execution
    from export_identity import load_identity_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS = REPO_ROOT / "isaac"
BLENDER = REPO_ROOT / "lib/utils/third_party/blender-4.5/blender"
ISAAC_PY = Path(
    os.environ.get("SCENERIG_ISAAC_PYTHON", "lib/utils/third_party/isaac/venv/bin/python")
)
GRASE_PY = REPO_ROOT / ".venv/bin/python"  # CoACD lives in the repo venv


def run(cmd, marker, timeout=1800):
    # timeout kills the child before raising: an Isaac step that crashes pre-close
    # hangs on Kit's non-daemon threads holding GPU memory, and an unbounded wait
    # here would wedge the whole export behind it.
    print("+", " ".join(str(c) for c in cmd), flush=True)
    try:
        out = subprocess.run(
            [str(c) for c in cmd], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"step timed out after {timeout} s (missing {marker})")
    sys.stdout.write(out.stdout[-2000:])
    if out.returncode != 0 or marker not in out.stdout:
        sys.stderr.write(out.stderr[-3000:])
        sys.exit(f"step failed (missing {marker})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", help="experiment dir containing scene/final/final.blend")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument(
        "--drift-tol",
        type=float,
        default=0.005,
        help="verify drift tolerance (m); poses were settled in this same engine with "
        "the same colliders, so anything beyond mm-scale is a regression",
    )
    ap.add_argument("--isaac-python", default=str(ISAAC_PY))
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    if not exp.is_absolute():
        exp = REPO_ROOT / exp
    blend = exp / "scene/final/final.blend"
    assert blend.exists(), f"no final.blend at {blend}"
    out_dir = exp / "scene/isaac"
    out_dir.mkdir(exist_ok=True)
    visual = out_dir / "scene_visual.usdc"
    collision_source = out_dir / "scene_collision.usdc"
    collision_request = out_dir / "collision_export_request.json"
    scene = out_dir / "scene.usd"
    dump = out_dir / "visual_meshes.npz"
    identity = out_dir / "object_identity.json"
    collision = out_dir / "collision"
    placement = exp / "scene/placement.json"
    assert placement.exists(), f"no placement.json at {placement}"

    from isaac.collision_export_contract import build_collision_export_request

    request = build_collision_export_request(exp)
    collision_request.write_text(json.dumps(request, indent=2) + "\n")

    run(
        [
            BLENDER,
            "-b",
            blend,
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python",
            SCRIPTS / "isaac_export_usd.py",
            "--",
            visual,
            collision_source,
            collision_request,
        ],
        "ISAAC_COLLISION_SOURCE_OK",
    )
    run(
        [
            args.isaac_python,
            SCRIPTS / "usd_dump_objects.py",
            collision_source,
            dump,
            placement,
            identity,
        ],
        "USD_DUMP_OK",
    )
    run(
        [args.isaac_python, SCRIPTS / "collision_source_validation.py", exp],
        "ISAAC_COLLISION_VALIDATED",
    )
    run(
        [
            GRASE_PY,
            SCRIPTS / "build_collision.py",
            "--from-dump",
            dump,
            "--out",
            collision,
        ],
        "COLLISION_OK",
    )
    # Seed physics_overrides.json from the preprocess settle records: objects the settle
    # ladder could only stand via CoM/flatten stabilization get the same treatment at
    # export, without auto_stabilize having to re-discover them. The CoM is re-solved on
    # the EXPORT-pose colliders (the recorded com_world is stale after later stages move
    # the object); flatten/friction/damping carry over. auto_stabilize can still refine.
    pc_path = exp / "scene/physics/pose_changes.json"
    ov_path = out_dir / "physics_overrides.json"
    if pc_path.exists():
        from isaac_auto_stabilize import solve_com

        pc = json.loads(pc_path.read_text())
        wants = {n: r["physics_overrides"] for n, r in (pc.get("objects") or {}).items()
                 if r.get("physics_overrides")}  # fmt: skip
        # pose_changes speaks PIPELINE names; collision files + stamped prims speak
        # exported USD names. The dump step recorded the one-to-one bridge.
        manifest = load_identity_manifest(identity)
        name_map = {
            record["pipeline_name"]: record["usd_name"]
            for record in manifest["objects"]
        }
        seeded = {}
        for pn, ov in wants.items():
            prim = name_map.get(pn)
            npz = collision / f"{prim}.npz" if prim else None
            if not (prim and npz.exists()):
                print(f"[blend_to_isaac] override seed for {pn}: no mapped prim/npz "
                      "— skipped")  # fmt: skip
                continue
            try:
                com, r, h, h_uni, inertia = solve_com(str(npz), 12.0)
            except Exception:  # noqa: BLE001 - degenerate footprint: skip the seed
                continue
            seeded[prim] = {
                "com_world": com,
                "flatten_base_mm": ov.get("flatten_base_mm", 8.0),
                "friction": ov.get("friction", 0.9),
                "angular_damping": ov.get("angular_damping", 1.5),
                # must accompany com_world (self-inconsistent rigid body otherwise
                # — see isaac_auto_stabilize.solve_com's docstring)
                "diagonal_inertia": inertia["diagonal_inertia"],
                "principal_axes": inertia["principal_axes"],
                "solver_mass_kg": float(inertia["mass"]),
                "_note": f"seeded from preprocess settle (pipeline name {pn})",
            }
        if seeded:
            ov_path.write_text(json.dumps(seeded, indent=1))
            print(
                f"[blend_to_isaac] seeded {len(seeded)} physics overrides from settle"
            )
    # VLM physics: translate the preprocess estimate (pipeline names, estimation-
    # time size anchors) into scene/isaac/physics_vlm.json (prim names, mass
    # rescaled to the dump geometry) — consumed by isaac_add_physics below.
    if (exp / "scene/physics/physics_vlm.json").exists():
        run(
            [GRASE_PY, SCRIPTS / "translate_vlm_physics.py", exp],
            "VLM_TRANSLATE_OK",
        )
    run(
        [args.isaac_python, SCRIPTS / "isaac_add_physics.py", visual, scene, collision],
        "ISAAC_PHYSICS_OK",
    )
    if not args.skip_verify:
        run(
            [
                args.isaac_python,
                SCRIPTS / "isaac_verify_settle.py",
                scene,
                out_dir / "verify_report.json",
                str(args.drift_tol),
            ],
            "ISAAC_VERIFY_OK",
        )
    print(f"DONE {scene}")


if __name__ == "__main__":
    main()
