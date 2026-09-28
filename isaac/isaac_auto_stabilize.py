"""Auto-stabilize objects that are unstable under default physics parameters.

    $SCENERIG_ISAAC_PYTHON isaac/isaac_auto_stabilize.py <exp_dir> \
        [--theta 12] [--flatten-mm 8] [--tol 0.005]

Rationale: the input photo shows every object at rest, so an object that falls over
under uniform-density sim has a provably wrong mass model — correcting it toward the
observed behavior is data-driven. Loop:

  round 0: stamp DEFAULT physics (no overrides), 5 s settle  -> unstable set
  round 1: per unstable object, author a COM solving tip-threshold >= theta:
           keep natural xy when edge margin is adequate, otherwise move minimally
           toward the Chebyshev center to MARGIN_FLOOR_M; z uses the achieved
           margin/tan(theta), floored at 0.3*natural height; also write consistent
           mass/inertia/principal axes + flatten_base_mm/friction/damping
  round 2: same with theta*2 (stronger, tumbler-adjacent)
  round 3: kinematic — pose held exactly (settle-capped poses physics can't hold)

Writes <exp>/scene/isaac/physics_overrides.json; each round re-stamps + re-settles.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "isaac"
SLICE_MM = 8.0
sys.path.insert(0, str(REPO_ROOT))  # for lib.tools.geometry.mass_properties
sys.path.insert(0, str(SCRIPTS))  # for physics_config: physics.py loads THIS file via
# spec_from_file_location, which does NOT put isaac/ on sys.path (a bare
# "from physics_config import ..." then raises ModuleNotFoundError and the whole
# incremental settle is skipped). Running it as a script masks this, since sys.path[0]
# is already isaac/ then.


def run(cmd, marker, timeout=1800):
    # timeout kills the child before raising: an Isaac step that crashes pre-close
    # hangs on Kit's non-daemon threads (see blend_to_isaac.run).
    try:
        out = subprocess.run(
            [str(c) for c in cmd], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"step timed out after {timeout} s (missing {marker})")
    if marker not in out.stdout:
        sys.stderr.write(out.stdout[-1500:] + out.stderr[-1500:])
        sys.exit(f"step failed (missing {marker})")


def chebyshev_center(pts2d):
    """Deepest interior point of the convex hull of pts2d, and its edge clearance."""
    hull = ConvexHull(pts2d)
    a, b = hull.equations[:, :2], -hull.equations[:, 2]  # A x <= b
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    # maximize r  s.t.  A x + ||A_i|| r <= b
    res = linprog(c=[0, 0, -1], A_ub=np.hstack([a, norms]), b_ub=b, bounds=[(None, None)] * 2 + [(0, None)])
    return res.x[:2], res.x[2]


from physics_config import DENSITY  # noqa: E402  (one source — see that module)
# Minimal footprint-edge clearance for a PROJECTED CoM xy (the leaning class).
MARGIN_FLOOR_M = 0.005
# CoM height floor as a fraction of the natural (uniform-density) height: real
# base-weighted objects (weeble toys, bottles with settled contents) have CoMs
# around 20-35% of their height — below ~30% the virtual mass distribution stops
# corresponding to any plausible object. Without this floor the margin formula
# could park the CoM at an ABSOLUTE ~23mm (or ~5mm on the export paths, which
# skip the sliver guard) regardless of object height. When the floor binds, the
# achieved tip threshold falls below theta — acceptable, because every consumer
# verifies the candidate EMPIRICALLY (ladder drop / auto-stabilize re-settle)
# and falls back to the toppled pose when an honestly-bounded CoM cannot stand.
COM_HEIGHT_FLOOR_FRAC = 0.3


def solve_com(npz_path, theta_deg):
    """Returns (com_world, r_cheb, h, h_uniform, inertia).

    CoM xy is MINIMALLY displaced (2026-07-21): keep the NATURAL (uniform-density)
    centroid xy whenever its footprint-edge margin is adequate — a well-based
    object keeps its real, empirically-robust balance point (the unconditional
    Chebyshev jump moved it ~13 mm downhill on 0720_compfix_real8219's plush and
    lost a marginal edge landing that the natural point survived). Only when the
    natural xy is marginal/outside the footprint (the leaning class the
    stabilization exists for) is it pulled toward the Chebyshev center — just far
    enough to reach MARGIN_FLOOR_M of edge clearance, not all the way. z is then
    lowered against the ACHIEVED margin: z = zmin + min(h_uniform,
    margin/tan(theta)), FLOORED at COM_HEIGHT_FLOOR_FRAC * h_uniform — the
    override may claim a weighted base, never an impossible one; when the floor
    binds, the empirical drop test (not the theta target) decides, and an object
    that cannot stand with an honest CoM topples and keeps its fallen pose.

    ``inertia`` is ``{"mass", "diagonal_inertia", "principal_axes"}`` — the
    physically self-consistent (mass, inertia-about-com_world) pair that MUST
    accompany the CoM override (see mass_properties.py: a centerOfMass override
    with no matching inertia tensor is a self-inconsistent rigid body — it
    produced a real launch-on-contact failure, 0720_compfix_real8219)."""
    from lib.tools.geometry.mass_properties import (
        assembly_natural_properties,
        stabilized_mass_properties,
    )

    d = np.load(npz_path)
    parts = [(d[f"v{i}"], d[f"f{i}"].reshape(-1, 3)) for i in range(int(d["n"]))]
    allv = np.vstack([v for v, _ in parts])
    zmin = allv[:, 2].min()
    foot = allv[allv[:, 2] < zmin + SLICE_MM / 1000.0][:, :2]
    (cx, cy), r = chebyshev_center(foot)
    # signed distance of a point to the footprint hull's nearest edge (+ inside)
    fhull = ConvexHull(foot)
    fa, fb = fhull.equations[:, :2], -fhull.equations[:, 2]
    fn = np.linalg.norm(fa, axis=1)

    def margin_of(p):
        return float(np.min((fb - fa @ np.asarray(p, dtype=float)) / fn))

    _, nat_com, _ = assembly_natural_properties(parts, DENSITY)
    h_uniform = float(nat_com[2] - zmin)
    nat_xy = np.asarray(nat_com[:2], dtype=float)
    cheb = np.array([cx, cy], dtype=float)
    m_min = min(r, MARGIN_FLOOR_M)
    if margin_of(nat_xy) >= m_min:
        xy = nat_xy  # natural balance point is fine — do not move it
    else:
        # walk nat -> cheb just far enough to reach m_min of margin. margin is
        # concave along the segment (min of affine functions) and margin(cheb)=r
        # >= m_min, so the first-crossing bisection finds the minimal t.
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if margin_of(nat_xy + mid * (cheb - nat_xy)) >= m_min:
                hi = mid
            else:
                lo = mid
        xy = nat_xy + hi * (cheb - nat_xy)
    h = min(h_uniform, max(margin_of(xy), 0.0) / math.tan(math.radians(theta_deg)))
    h = max(h, COM_HEIGHT_FLOOR_FRAC * h_uniform)  # never a physically absurd weeble
    com = [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(zmin + h), 4)]
    inertia = stabilized_mass_properties(parts, DENSITY, com)
    return com, r, h, h_uniform, inertia


def stamp_and_settle(exp, isaac_py, tol):
    iso = exp / "scene/isaac"
    run([isaac_py, SCRIPTS / "isaac_add_physics.py", iso / "scene_visual.usdc",
         iso / "scene.usd", iso / "collision"], "ISAAC_PHYSICS_OK")
    run([isaac_py, SCRIPTS / "isaac_verify_settle.py", iso / "scene.usd",
         iso / "verify_report.json", str(tol)], "ISAAC_VERIFY_")
    r = json.loads((iso / "verify_report.json").read_text())
    unstable = {
        name: {
            "drift_m": body.get("drift_m"),
            "rotation_drift_deg": body.get("rotation_drift_deg"),
            "fell_through": body.get("fell_through"),
            "support": body.get("support"),
        }
        for name, body in r["bodies"].items()
        if not body.get("physics_pass", False)
    }
    return unstable, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir")
    ap.add_argument("--theta", type=float, default=12.0, help="target tip threshold, deg")
    ap.add_argument("--flatten-mm", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=0.005)
    ap.add_argument(
        "--isaac-python",
        default=os.environ.get(
            "SCENERIG_ISAAC_PYTHON", "lib/utils/third_party/isaac/venv/bin/python"
        ),
    )
    args = ap.parse_args()

    exp = Path(args.exp_dir)
    if not exp.is_absolute():
        exp = REPO_ROOT / exp
    ov_path = exp / "scene/isaac/physics_overrides.json"

    ov_path.write_text("{}")  # round 0: default physics
    unstable, report = stamp_and_settle(exp, args.isaac_python, args.tol)
    print(f"round 0 (defaults): unstable = {unstable or 'none'}", flush=True)
    if report.get("simulation_status") == "dynamic_verified":
        print(f"AUTO_STABILIZE_OK overrides={ov_path}")
        return 0
    if not unstable:
        print(f"AUTO_STABILIZE_FAIL overrides={ov_path}")
        return 2

    overrides = {}
    for rnd, theta in [(1, args.theta), (2, args.theta * 2), (3, None)]:
        for name in unstable:
            if theta is None:
                overrides[name] = {"kinematic": True}
                continue
            npz = exp / f"scene/isaac/collision/{name}.npz"
            com, r, h, h_uni, inertia = solve_com(npz, theta)
            overrides[name] = {
                "com_world": com, "flatten_base_mm": args.flatten_mm,
                "angular_damping": 1.5, "friction": 0.9,
                # accompany the CoM with its physically-consistent inertia tensor
                # (a CoM-only override is a self-inconsistent rigid body that can
                # launch on contact — 0720_compfix_real8219)
                "diagonal_inertia": inertia["diagonal_inertia"],
                "principal_axes": inertia["principal_axes"],
                # density-250 reference mass: isaac_add_physics rescales the
                # inertia with it when a VLM/override mass is authored
                "solver_mass_kg": float(inertia["mass"]),
                "_note": f"auto: theta={theta}deg r_cheb={r*1000:.1f}mm "
                         f"h={h*1000:.1f}mm (uniform {h_uni*1000:.1f}mm)",
            }
        ov_path.write_text(json.dumps(overrides, indent=1))
        unstable, report = stamp_and_settle(exp, args.isaac_python, args.tol)
        print(f"round {rnd} ({'kinematic' if theta is None else f'theta={theta}'}): "
              f"unstable = {unstable or 'none'}", flush=True)
        if report.get("physics_pass") is True:
            break
        if not unstable:
            print(f"AUTO_STABILIZE_FAIL overrides={ov_path}")
            return 2

    report = json.loads((exp / "scene/isaac/verify_report.json").read_text())
    status = report.get("simulation_status", "unstable")
    marker = (
        "OK"
        if status == "dynamic_verified"
        else "KINEMATIC_FALLBACK"
        if status == "stable_with_kinematic_fallback"
        else "FAIL"
    )
    print(f"AUTO_STABILIZE_{marker} overrides={ov_path}")
    if status == "dynamic_verified":
        return 0
    if status == "stable_with_kinematic_fallback":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
