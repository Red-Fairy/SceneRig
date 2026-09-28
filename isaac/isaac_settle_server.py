"""Persistent PhysX settle server for the incremental (DFS) scene build.

    /fsx/rundongluo/isaac/venv/bin/python isaac/isaac_settle_server.py

One SimulationApp per scene; JSON-RPC over stdin/stdout (one JSON object per line;
non-JSON stdout lines are Isaac log noise the client skips — same contract as
register_blender_server). The scene is PERSISTENT: a 100 m table slab at z=0 plus
every committed object as a STATIC collider; exactly one object is dynamic at a time.

Protocol (requests -> responses):
  {"cmd":"add","name":n,"npz":path,"com":[x,y,z]?,"friction":f?,"damping":d?,
   "flatten_mm":m?,"diagonal_inertia":[3]?,"principal_axes":[x,y,z,w]?,"mass":kg?}
                                          -> {"ok":true}   (parts at their npz world pose;
                                              a com override MUST carry its matching
                                              inertia — CoM-only is self-inconsistent;
                                              "mass" = explicit VLM mass, else uniform
                                              DENSITY; rides uniform scales at s^3)
  {"cmd":"swap","name":n,"npz":path,...}  -> {"ok":true}   (replace parts: pristine/CoM rung)
  {"cmd":"add_static","name":n,"npz":path} -> {"ok":true}  (root surface: collider in every
                                              sim incl. scoped ladders, never dynamic)
  {"cmd":"drop","name":n,"scene":[..]?,"budget":"micro"?,"ancestors":[..]?}
                                          -> {"ok":true,"R":3x3,"t":[..],"tilt_deg":..,
                                              "cum_tilt_deg":..,"lift_mm":..,
                                              "release_dz_mm":..,"down_cleared":bool,
                                              "disp_xy_mm":..,"disp_xy_m":[dx,dy],
                                              "rollable":bool,"converged":bool,
                                              "drop_continued":bool,"steps_used":int}
                                              (release from ~contact: lift-to-clear out of a
                                              penetration, else SNAP-DOWN to first contact
                                              when the pose floats (release_dz_mm signed; no
                                              free-fall). With ``ancestors``: a big lift
                                              forced by a NON-ancestor prefers a clear pose
                                              BELOW (chair-under-table) over lifting on top.
                                              ``scene`` limits the statics — LADDER drops
                                              isolate vs ancestors only; omitted = full
                                              committed scene; delta is world->world vs the
                                              parts BEFORE the drop; budget "micro" caps sim
                                              at 2 s for the composition winner-settle)
  {"cmd":"move_group","members":[..],"deltas":{n:4x4},"budget":"micro"?,
   "exclude":[..]?,"pen_ignore":[..]?,"free":[..]?,"sequential":bool?,
   "free_static":bool?}
                                          -> {"ok":true,"D":4x4,"tilt_deg":..,
                                              "cum_tilt_deg":{n:..},"lift_mm":..,
                                              "pen_before_mm":{n:..},"pen_after_mm":{n:..},
                                              "disp_mm":{n:..},"rollable":{n:..},
                                              "body_pen":{n:..},"converged":bool,
                                              "free":{n:{D,tilt_deg,cum_tilt_deg}}?}
                                              (free = individual dynamic bodies at their
                                              current poses — the carry-decision probe)
                                              (atomic: apply carry deltas, weld members into
                                              ONE rigid body, settle, bake D into each; total
                                              blend delta per member = D @ C_member;
                                              pen_*_mm = members' overlap depth vs every
                                              other body before/after — the client's
                                              penetration gate; pen_ignore names skipped)
  {"cmd":"settle_set","bodies":[..],"deltas":{n:4x4},"free":[..]?,
   "groups":[[..],..]?,"budget":"micro"?}
                                          -> {"ok":true,"D":{n:4x4},"tilt_deg":{n:..},
                                              "cum_tilt_deg":{n:..},"disp_mm":{n:..},
                                              "rollable":{n:..},"lift_mm":..,
                                              "pen_before_mm":{n:..},"pen_after_mm":{n:..},
                                              "body_pen":{n:..},"converged":bool}
                                              (SEVERAL moved hierarchies settled in ONE
                                              sim as individual dynamic bodies — no weld —
                                              with the rest of the scene static; union
                                              lift-to-clear vs the non-set bodies; free =
                                              contact-dependents, dynamic at current poses)
  {"cmd":"scale_resettle","parent":n,"delta":4x4,"free":[..]?,
   "absent":[..]?,"budget":"micro"?}
                                          -> {"ok":true,"D":{n:4x4},"tilt_deg":..,
                                              "cum_tilt_deg":{n:..},"disp_mm":{n:..},
                                              "rollable":{n:..},"lift_mm":..,
                                              "pen_before_mm":{n:..},"pen_after_mm":{n:..},
                                              "body_pen":{n:..},"converged":bool}
                                              (destination-support order: lateral contacts stay
                                              present while parent settles; true cargo is absent,
                                              then re-seated support-first; a final no-lift local
                                              all-dynamic closure lets both contact sides react)
  {"cmd":"contacts","tol":m?,"pairs_only":bool?} -> {"ok":true,"pairs":[{"a","b","depth_mm"?}]}
                                              pairs_only skips the depth scan (no depth_mm)
                                              (hull-overlap pairs at current poses — the
                                              composition rules gate)
  {"cmd":"clearance","name":n,"scene"?}   -> {"ok":true,"dz":..}  (lift-to-clear height at
                                              the current pose — the ICP overlap budget)
  {"cmd":"clear_along","name":n,"cap":m,"exclude":[..]?,"budget":"micro"?}
                                          -> {"ok":true,"cleared":bool,"push_mm":..,
                                              "R","t","tilt_deg","cum_tilt_deg","lift_mm"}
                                              (topple-retry: SLIDE the body to the nearest xy in
                                              a cap-radius DISK where it clears the committed
                                              bodies, then settle from contact — replaces the
                                              lift+drop that topples tall/thin objects; exclude =
                                              own support chain; cleared=false => vertical-drop
                                              baseline)
  {"cmd":"transform","name":n,"M":4x4}    -> {"ok":true}   (bake an external correction, e.g.
                                              ICP similarity incl. scale, into the parts)
  {"cmd":"penetration","members":[..],"deltas":{n:4x4}}
                                          -> {"ok":true,"pen":{...}}
  {"cmd":"resting_on","name":n}       -> {"ok":true,"support_body":n|null}
  {"cmd":"supports_of","name":n}      -> {"ok":true,"supports":[{...,"relation":
                                              "strict_below"|"clear_below_or_container"|
                                              "ambiguous_same_level"}]}
  {"cmd":"remove","name":n}               -> {"ok":true}
  {"cmd":"pose","name":n}                 -> {"ok":true,"total":4x4,
                                              "cum_tilt_deg":..}  (cumulative delta vs add)
  {"cmd":"certify","dxy_cap":m?,"tilt_cap_deg":d?}
                                          -> {"ok":true,"drift":{n:{dxy,dz,tilt_deg,pinned?}},
                                              "converged":bool,
                                              "total":{n:4x4}}  (unfreeze ALL, short free sim,
                                              bake; the final simulation-ready guarantee.
                                              drift = CENTROID displacement. With caps set,
                                              an object drifting past them is PINNED at its
                                              composed pose and the sim re-runs without it —
                                              recorded, never baked)
  {"cmd":"certify_sequential","dxy_cap":m?,"tilt_cap_deg":d?,
   "order":[..]?,"repair":bool?}             -> {"ok":true,"drift":{...},"total":{...}}
  {"cmd":"ping"}                          -> {"ok":true,"pid":..}  (liveness probe)
  {"cmd":"reset"}                         -> {"ok":true}   (clear the object registry; the
                                              next add starts a fresh scene — a reused warm
                                              server is indistinguishable from a fresh boot)
  {"cmd":"shutdown"}                      -> {"ok":true}

Transport: stdin/stdout by default. With ``--port-file <path>`` the server instead
listens on 127.0.0.1:<ephemeral> and atomically writes {"port","pid"} to <path>, so
later pipeline stages (which run in different OS processes) can reconnect to the SAME
SimulationApp instead of paying a fresh ~30-60 s Kit boot. One connection is served at
a time; a NEW connection preempts the current one (stages are strictly sequential — a
lingering idle client from a dying stage just sees EOF). The {"ready":true} stdout
line is still printed so the spawning client's boot-wait is transport-agnostic.

Being a daemon, it self-reaps two ways (see main_tcp): ``--owner-pid <pid>`` exits once
the owning RUN is gone, and GRASE_ISAAC_IDLE_TIMEOUT (default 1800 s, 0 disables) exits
after that long with no client connected. Without these a killed run stranded a
SimulationApp holding ~11 GB across every visible GPU until someone noticed it in
nvidia-smi — the client-side shutdown is a Python ``finally``, which SIGKILL skips.

Committed objects are frozen by construction (static colliders re-authored from baked
world-space parts), so a later release can never disturb them — no dominoes. The
lift-to-clear search uses convex-hull half-space tests (scipy) with AABB prefilter:
raise the probe in 2 mm steps until no vertex-level overlap with any committed part.
Releases are from ~contact in BOTH directions: a floating pose (MoGE z error) is
snapped DOWN to first contact before release (settle_geometry.drop_dz — exact
intervals, no free-fall: 0725_arr_room1 chair_1 free-fell 400 mm and toppled 92 deg,
falsely taking the pristine rung).
"""

from __future__ import annotations

import json
import math
import os
import select
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaacsim import SimulationApp  # noqa: E402
from lib.tools.geometry.settle_geometry import (  # noqa: E402
    Hull,
    clear_dist,
    clear_dz,
    drop_dz,
    flatten_base,
    hulls_overlap,  # noqa: F401 - re-exported for parity with settle_geometry
    load_parts,
    min_clear_dist,
    pair_clear_dist,
    support_relation,
)

app = SimulationApp({"headless": True})

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from physics_config import (  # noqa: E402
    ANGULAR_REST_SPEED_DEG_S,
    CONTACT_OFFSET,
    DENSITY,
    EXPORT_VELOCITY_REST_POLICY,
    LINEAR_REST_SPEED_M_S,
    OBJ_FRICTION,
    REST_OFFSET,
    SURF_FRICTION,
    apply_scene_tuning,
    log_protocol,
    read_body_speeds,
)
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

DT = 1.0 / 120.0
MAX_STEPS, MIN_STEPS, REST_STEPS = 600, 30, 20
MICRO_STEPS = 240  # "micro" budget: winner-settles start near rest; cap tail latency
POS_EPS, ANG_EPS = 5e-5, 0.005
NUDGE = 0.002  # release height above the collision-free z (m)
LIFT_STEP = 0.002  # lift-to-clear increment (m)
LIFT_MAX = 1.0
# Down-preference gate for cmd_drop (fires only when the caller passes its
# ``ancestors``): a lift this big forced by a NON-ancestor is the chair-under-
# table pattern — resolve it downward instead of parking the object on top.
# Mirrors the client's HCLEAR_LIFT_MM reactive gate (physics.py).
DOWN_PREF_MIN_LIFT_MM = 30.0

_B = np.array(
    [[sx, sy, sz] for sx in (-50, 50) for sy in (-50, 50) for sz in (-0.5, 0.0)]
)
_BF = np.array(
    [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
     (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
)  # fmt: skip


def _respond(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()










def _quat_to_mat(q):
    """[x,y,z,w] quaternion -> 3x3 rotation matrix."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def _mat_to_quat(M):
    """3x3 rotation matrix -> [x,y,z,w] quaternion (Shepperd's method)."""
    t = float(np.trace(M))
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return [
            float((M[2, 1] - M[1, 2]) / s), float((M[0, 2] - M[2, 0]) / s),
            float((M[1, 0] - M[0, 1]) / s), 0.25 * s,
        ]  # fmt: skip
    i = int(np.argmax(np.diag(M)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(1.0 + M[i, i] - M[j, j] - M[k, k], 1e-12)) * 2
    q = [0.0, 0.0, 0.0, float((M[k, j] - M[j, k]) / s)]
    q[i] = 0.25 * s
    q[j] = float((M[j, i] + M[i, j]) / s)
    q[k] = float((M[k, i] + M[i, k]) / s)
    return q


class Obj:
    def __init__(self, name, parts, com=None, friction=None, damping=None,
                 static=False, rollable=None, diag_inertia=None, axes_quat=None,
                 mass=None):
        self.name = name
        self.parts = parts  # world-space (current pose baked in)
        self.com = None if com is None else np.asarray(com, dtype=float)
        # Explicit (VLM-estimated) mass in kg; None -> uniform DENSITY body.
        # Rides uniform scales at s^3 in _apply_delta.
        self.mass = None if mass is None else float(mass)
        # Explicit inertia accompanying a CoM override (a CoM-only override is a
        # self-inconsistent rigid body — 0720_compfix_real8219 launch-on-contact).
        # diag_inertia: principal moments about the CoM; axes: world-frame
        # principal basis, kept as a matrix so _apply_delta can rotate it along.
        self.diag_inertia = (
            None if diag_inertia is None else np.asarray(diag_inertia, dtype=float)
        )
        self.axes = None if axes_quat is None else _quat_to_mat(axes_quat)
        self.friction = friction
        self.damping = damping
        self.static = static  # root surface: collider in every sim, never dynamic
        self.total = np.eye(4)  # cumulative world delta since add()
        self.hulls = [Hull(v) for v, _ in parts]
        # Rollable (a LYING cylinder: rolling is benign, tilt is meaningless —
        # judge it by displacement instead). Client-supplied (VLM, state-aware);
        # fallback for pre-field runs: orientation-aware extents at the ADD pose —
        # long horizontal axis + round cross-section (z within 1.5x the small
        # horizontal dim). A STANDING marker (long axis vertical) is NOT rollable.
        if rollable is None:
            v = np.vstack([p for p, _ in parts])
            ext = v.max(0) - v.min(0)
            rollable = bool(
                max(ext[0], ext[1]) >= 2.0 * ext[2]
                and min(ext[0], ext[1]) <= 1.5 * ext[2]
            )
        self.rollable = bool(rollable)

    def rebuild_hulls(self):
        self.hulls = [Hull(v) for v, _ in self.parts]

    def centroid(self):
        return np.vstack([v for v, _ in self.parts]).mean(0)


_objs: dict[str, Obj] = {}
_world = None
_prims: dict[str, object] = {}


def _material(stage, path, friction):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    return mat


def _author(stage, root, parts, mat, dynamic=False, com=None, damping=None,
            diag_inertia=None, axes=None, mass_kg=None):
    xform = UsdGeom.Xform.Define(stage, root)
    prim = xform.GetPrim()
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        if mass_kg is not None:  # explicit (VLM) mass; PhysX derives the rest
            mass.CreateMassAttr(float(mass_kg))
        else:
            mass.CreateDensityAttr(DENSITY)
        if com is not None:
            mass.CreateCenterOfMassAttr(tuple(float(c) for c in com))
            if diag_inertia is not None:
                # the physically-consistent tensor that MUST accompany a CoM
                # override (CoM-only = self-inconsistent rigid body: launched a
                # plush on edge contact, 0720_compfix_real8219)
                mass.CreateDiagonalInertiaAttr(
                    tuple(float(v) for v in diag_inertia)
                )
                if axes is not None:
                    qx, qy, qz, qw = _mat_to_quat(np.asarray(axes, dtype=float))
                    mass.CreatePrincipalAxesAttr(
                        Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz))
                    )
        px_body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        if damping is not None:
            px_body.CreateAngularDampingAttr(float(damping))
    for i, (v, f) in enumerate(parts):
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/part_{i:03d}")
        mesh.CreatePointsAttr([tuple(p) for p in np.asarray(v, dtype=float)])
        mesh.CreateFaceVertexCountsAttr([3] * len(f))
        mesh.CreateFaceVertexIndicesAttr([int(x) for x in np.asarray(f).reshape(-1)])
        p = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(p)
        UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set("convexHull")
        px = PhysxSchema.PhysxCollisionAPI.Apply(p)
        px.CreateContactOffsetAttr(CONTACT_OFFSET)
        px.CreateRestOffsetAttr(REST_OFFSET)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, materialPurpose="physics")
    return prim


def _rebuild_scene(dynamic=None, scene_names=None, exclude=None, free=None):
    """Author the scene fresh: slab + the listed statics (None = every committed
    object), plus ``dynamic`` (a list of names) welded into ONE rigid body — a
    single-element list is a normal drop; several elements weld a carried group
    (parent + cargo) so it settles as the loaded assembly with no self-collision.
    ``free`` names are authored as INDIVIDUAL dynamic bodies (not welded, not
    static): the carry-decision probe moves a parent alone with its children
    free, so a child losing its perch is OBSERVED (topple) instead of frozen.
    ``scene_names`` scopes the LADDER's isolated stability drops (slab + ancestors
    only — a cluttered neighborhood must not masquerade as intrinsic instability);
    ``static`` surfaces are always included regardless. A fresh stage per (re)build
    sidesteps PhysX prim-mutation edge cases; authoring ~a dozen compound bodies is
    milliseconds next to the sim itself."""
    global _world, _prims
    World.clear_instance()
    stage_utils.create_new_stage()
    _world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = _world.stage
    apply_scene_tuning(stage)  # no-op unless GRASE_PHYSX_TUNED=1
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    surf_mat = _material(stage, "/World/PhysMatSurf", SURF_FRICTION)
    _author(stage, "/World/ground", [(_B, _BF)], surf_mat)
    _prims = {}
    dyn_set = set(dynamic or ())
    free_set = set(free or ())
    excl = set(exclude or ())  # rotate_180: descendants stay put AND stay out of
    # the sim — stationary cargo would otherwise pin the flipping parent
    include = None if scene_names is None else set(scene_names)
    for i, (name, o) in enumerate(_objs.items()):
        if name in dyn_set or name in excl or name in free_set:
            continue
        if include is not None and name not in include and not o.static:
            continue
        mat = _material(stage, f"/World/PhysMat_{i:03d}", SURF_FRICTION)
        _prims[name] = _author(stage, f"/World/obj_{i:03d}", o.parts, mat)
    for i, name in enumerate(free or ()):
        o = _objs[name]
        mat = _material(stage, f"/World/PhysMatFree_{i:03d}",
                        o.friction or OBJ_FRICTION)  # fmt: skip
        _prims[f"__free_{name}__"] = _author(
            stage, f"/World/free_{i:03d}", o.parts, mat, dynamic=True,
            com=o.com, damping=o.damping,
            diag_inertia=o.diag_inertia, axes=o.axes, mass_kg=o.mass,
        )  # fmt: skip
    if dynamic:
        lead = _objs[dynamic[0]]  # the moved parent leads friction/damping
        parts = [pt for m in dynamic for pt in _objs[m].parts]
        mat = _material(stage, "/World/PhysMatDyn", lead.friction or OBJ_FRICTION)
        # >1 member: com=None so uniform density over ALL parts yields the loaded
        # assembly's true CoM (a heavy mug shifts where the tray rests) — and no
        # explicit inertia either (the weld's tensor comes from density over the
        # combined shapes, consistent by construction). Explicit (VLM) mass is
        # likewise SOLO-only: summing per-member masses under a uniform-density
        # CoM/inertia would be a mismatched triple.
        solo = len(dynamic) == 1
        _prims["__dyn__"] = _author(
            stage, "/World/dyn", parts, mat, dynamic=True,
            com=lead.com if solo else None, damping=lead.damping,
            diag_inertia=lead.diag_inertia if solo else None,
            axes=lead.axes if solo else None,
            mass_kg=lead.mass if solo else None,
        )  # fmt: skip
    _world.reset()


def _pose(prim, cache=None):
    return np.array((cache or UsdGeom.XformCache()).GetLocalToWorldTransform(prim)).T


def _validate_rest_policy(rest_policy: str | None) -> None:
    """Reject an unsupported explicit stopping policy before rebuilding a scene."""
    if rest_policy not in (None, EXPORT_VELOCITY_REST_POLICY):
        raise ValueError(f"unsupported rest policy: {rest_policy!r}")


def _body_speed_report(prims) -> dict:
    """Sample live velocities, including verified PhysX sleep, before stopping."""
    report = {}
    for prim in prims:
        row = read_body_speeds(prim)
        report[str(prim.GetPath())] = {
            **row,
            "below_limits": bool(
                row["valid"]
                and row["linear_velocity_m_s"] < LINEAR_REST_SPEED_M_S
                and row["angular_velocity_deg_s"] < ANGULAR_REST_SPEED_DEG_S
            ),
        }
    return report


def _rest_result(objects: dict, converged: bool, steps: int, quiet: int) -> dict:
    """Bounded evidence for one velocity-aware simulation, without a step trace."""
    return {
        "status": "passed" if converged else "not_converged",
        "physics_hz": 1.0 / DT,
        "steps_used": steps,
        "quiet_steps": quiet,
        "thresholds": {
            "linear_velocity_m_s": LINEAR_REST_SPEED_M_S,
            "angular_velocity_deg_s": ANGULAR_REST_SPEED_DEG_S,
        },
        "objects": objects,
        "offending_objects": sorted(
            name for name, row in objects.items() if not row["below_limits"]
        ),
    }


def _run_to_rest_phases(prims, budgets, *, rest_policy=None, rest_report=None):
    """Step one live world through consecutive budgets until every prim is quiet.

    The pose cache, quiet counter, rigid-body velocity, and PhysX contact/solver state
    all survive a budget boundary. ``continued`` says a second phase was entered;
    ``steps_used`` is the total across phases. The world is stopped exactly once, at
    convergence or after every budget is exhausted.

    The optional rest policy additionally requires live finite speeds below the
    export defaults. ``rest_report`` receives the final measurements before stop;
    omitting the policy preserves the existing pose-only stopping condition.
    """
    _validate_rest_policy(rest_policy)
    cache = UsdGeom.XformCache()
    prev = [_pose(p, cache) for p in prims]
    M = prev
    quiet = 0
    converged = not prims
    continued = False
    steps_used = 0
    speeds = {}
    for phase, max_steps in enumerate(budgets):
        if converged:
            break
        continued = continued or phase > 0
        for _ in range(max_steps):
            _world.step(render=False)
            cache = UsdGeom.XformCache()
            M = [_pose(p, cache) for p in prims]
            dp = max(
                float(np.linalg.norm(m[:3, 3] - q[:3, 3])) for m, q in zip(M, prev)
            )
            da = 0.0
            for m, q in zip(M, prev):
                c = (float(np.trace(m[:3, :3] @ q[:3, :3].T)) - 1.0) / 2.0
                da = max(da, math.acos(max(-1.0, min(1.0, c))))
            prev = M
            speed_quiet = True
            if rest_policy is not None:
                speeds = _body_speed_report(prims)
                speed_quiet = all(row["below_limits"] for row in speeds.values())
            quiet = (
                quiet + 1
                if steps_used >= MIN_STEPS
                and dp < POS_EPS
                and da < ANG_EPS
                and speed_quiet
                else 0
            )
            steps_used += 1
            if quiet >= REST_STEPS:
                converged = True
                break
        if converged:
            break
    if rest_policy is not None and rest_report is not None:
        rest_report.update(_rest_result(speeds, converged, steps_used, quiet))
    _world.stop()
    return M, converged, continued, steps_used


def _run_to_rest(prims, max_steps=MAX_STEPS):
    """Run one bounded phase; return a mid-motion snapshot on non-convergence."""
    M, converged, _, _ = _run_to_rest_phases(prims, (max_steps,))
    return M, converged


def _delta(M):
    U, _, Vt = np.linalg.svd(M[:3, :3])
    return U @ Vt, M[:3, 3].copy()


def _tilt(R):
    # Normalize uniform scale out of the block before reading the cosine —
    # composition scale deltas ride o.total (s*I, det=s^3), so a raw R[2,2]
    # reads acos(s*cos(tilt)): a 0.70 resize of an UPRIGHT body read 45deg and
    # tripped the capsize cap (0725_comfix2_room1 vase), and every later read
    # of a scaled body kept the fake baseline. Same cbrt(det) idiom as
    # _apply_delta's inertia rescale; no-op for rigid deltas (det=1).
    Rm = np.asarray(R, dtype=float)
    s = np.cbrt(max(float(np.linalg.det(Rm)), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, float(Rm[2, 2]) / s))))


def _blockers(name, scene_names=None):
    """Hulls of every body ``name`` must clear: the listed scene objects
    (None = all committed) plus statics, which always block clearance."""
    include = None if scene_names is None else set(scene_names)
    return [h for n, obj in _objs.items()
            if n != name and (obj.static or include is None or n in include)
            for h in obj.hulls]  # fmt: skip


def _zmin(o):
    return min(float(v[:, 2].min()) for v, _ in o.parts)


def _clear_dz(name, scene_names=None):
    """Smallest +z shift at which ``name`` overlaps none of the listed statics
    (None = all committed) and sits above the slab."""
    o = _objs[name]
    others = _blockers(name, scene_names)
    base = max(0.0, -_zmin(o))  # never start below the slab top
    if base:
        o_shifted = [h.translated([0.0, 0.0, base]) for h in o.hulls]
        return base + clear_dz(o_shifted, others, step=LIFT_STEP, cap=LIFT_MAX)
    return clear_dz(o.hulls, others, step=LIFT_STEP, cap=LIFT_MAX)


def _release_dz(name, scene_names=None):
    """Signed release shift for cmd_drop: +lift to clear a penetration (the
    legacy lift-to-clear), else -drop to FIRST CONTACT when the pose floats
    (never below the slab). Killing the free-fall keeps the ladder's tilt gate
    a statement about the PLACEMENT, not the drop height (0725_arr_room1
    chair_1: a 400 mm MoGE float free-fell, toppled 92 deg, and wrongly took
    the pristine rung)."""
    up = _clear_dz(name, scene_names)
    if up > 0.0:
        return up
    o = _objs[name]
    zmin = _zmin(o)
    if zmin <= 0.0:
        return 0.0
    return -drop_dz(o.hulls, _blockers(name, scene_names), cap=zmin)


def _down_escape(name, scene_names=None):
    """Downward resolution of a NON-ancestor overlap (the chair-under-table
    pattern): the smallest -z shift that clears every blocker, then a further
    quasi-static drop to first contact — hull bottom kept at/above the slab
    throughout. None when no clear pose exists below (the caller falls back to
    lift-to-clear and the reactive clear_along gates then own it)."""
    o = _objs[name]
    zmin = _zmin(o)
    if zmin <= 0.0:
        return None
    others = _blockers(name, scene_names)
    esc = clear_dist(o.hulls, others, (0.0, 0.0, -1.0), step=LIFT_STEP, cap=zmin)
    if esc >= zmin:  # never cleared before the slab stopped it
        return None
    shifted = [h.translated([0.0, 0.0, -esc]) for h in o.hulls]
    return esc + drop_dz(shifted, others, cap=zmin - esc)


def _apply_delta(o, R, t):
    o.parts = [(v @ np.asarray(R).T + t, f) for v, f in o.parts]
    if o.com is not None:
        # the CoM override is a world-frame point riding the body — leaving it
        # behind after a move gives the stabilized rung a wrong lever arm
        o.com = np.asarray(R) @ o.com + t
    if o.axes is not None or o.diag_inertia is not None or o.mass is not None:
        # the explicit inertia rides the body too: principal axes rotate with the
        # orthonormal part; under a uniform scale s the moments scale as s^5
        # (mass ~ s^3, gyration length^2 ~ s^2 — keeps the authored tensor
        # consistent with the density-derived mass PhysX recomputes per authoring)
        Rm = np.asarray(R, dtype=float)
        s = float(np.cbrt(max(np.linalg.det(Rm), 1e-12)))
        if o.axes is not None:
            o.axes = (Rm / s) @ o.axes
        if o.diag_inertia is not None and abs(s - 1.0) > 1e-9:
            o.diag_inertia = o.diag_inertia * s**5
        if o.mass is not None and abs(s - 1.0) > 1e-9:
            o.mass = o.mass * s**3  # explicit mass rides scale at s^3 too
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = R, t
    o.total = D @ o.total
    Rm = np.asarray(R, dtype=float)
    if np.abs(Rm @ Rm.T - np.eye(3)).max() < 1e-9:
        # rigid delta: transform the cached hulls analytically (T1a) — qhull
        # only re-runs on geometry/scale changes, never per pose change
        o.hulls = [h.transformed_rigid(Rm, t) for h in o.hulls]
    else:
        o.rebuild_hulls()  # scaled delta: re-derive (keeps unit-norm rows)


def _steps(budget):
    return MICRO_STEPS if budget == "micro" else MAX_STEPS


def cmd_drop(name, scene_names=None, budget=None, ancestors=None):
    o = _objs[name]
    c_pre = o.centroid()  # pre-release: the shift is pure z, so xy disp is drop-only
    # signed release shift: +lift out of penetration, -snap-down to contact when
    # floating (no free-fall: released ~2mm above first contact either way)
    dz = _release_dz(name, scene_names)
    down_cleared = False
    if (ancestors is not None and dz * 1000.0 > DOWN_PREF_MIN_LIFT_MM
            and _clear_dz(name, list(ancestors)) <= 0.0):
        # big lift forced by a NON-ancestor (clear of its own support chain and
        # the root statics): resolve the overlap DOWNWARD when a clear pose
        # exists below — lifting would park it on top of the neighbor
        down = _down_escape(name, scene_names)
        if down is not None:
            dz, down_cleared = -down, True
    dz += NUDGE
    _apply_delta(o, np.eye(3), np.array([0.0, 0.0, dz]))
    _rebuild_scene(dynamic=[name], scene_names=scene_names)
    # F0b (2026-08-08): a cap hit is MID-MOTION, not a rest. Continue the SAME
    # live rigid body for one full extra budget: no stage rebuild, stop/play cycle,
    # velocity reset, or loss of contact/solver state at the phase boundary.
    Ms, conv, continued, steps_used = _run_to_rest_phases(
        [_prims["__dyn__"]], (_steps(budget), MAX_STEPS)
    )
    R, t = _delta(Ms[0])
    _apply_delta(o, R, t)
    # net delta vs the PRE-RELEASE parts: shift by dz then sim (R,t) => R@(v+dz) + t
    net_t = R @ np.array([0.0, 0.0, dz]) + t
    d = (o.centroid() - c_pre)[:2]
    return {
        "ok": True,
        "R": R.tolist(),
        "t": net_t.tolist(),
        "tilt_deg": _tilt(R),
        "cum_tilt_deg": _tilt(o.total[:3, :3]),  # vs the ORIGINAL placed pose
        # lift_mm keeps its historical meaning (penetration-forced +lift; the
        # client's HCLEAR gates key on it); the signed release is reported apart
        "lift_mm": max(dz - NUDGE, 0.0) * 1000.0,
        "release_dz_mm": (dz - NUDGE) * 1000.0,
        "down_cleared": down_cleared,
        # class-aware stability signal: a ROLLABLE body (lying cylinder) is judged
        # by how far it rolled/slid, not by tilt — rolling in place is benign.
        # disp_xy_m is the drift VECTOR (m): the ladder's roll-back rung inverts
        # it to restore the placed xy while keeping the settled rotation.
        "disp_xy_mm": float(np.linalg.norm(d) * 1000.0),
        "disp_xy_m": [float(d[0]), float(d[1])],
        "rollable": o.rollable,
        # False = no rest found even after the continuation (poses are a snapshot)
        "converged": bool(conv),
        "drop_continued": continued,
        "steps_used": steps_used,
    }


HCLEAR_STEP = 0.005  # horizontal search grid step (m)
HCLEAR_HULL_BUDGET = 1500  # max hull-confirm tests on AABB-overlapping cells


def _aabb(parts):
    v = np.vstack([p for p, _ in parts])
    return v.min(0), v.max(0)


def cmd_resting_on(name, tol=0.008):
    """The OBJECT directly BENEATH ``name`` that supports it: the highest committed body
    whose XY footprint overlaps ``name`` and whose top is at/below ``name``'s base within
    ``tol``. Returns {support_body, static, support_top_z, footprint_frac}; support_body is
    None when ``name`` rests only on the slab. A LATERAL neighbor (the keyboard a mug leans
    on) has its top ABOVE the mug's base, so it is NOT returned — only true under-supports.
    Used to detect an object physics-stacked on a NON-parent (pliers on a box)."""
    o = _objs.get(name)
    if o is None:
        return {"ok": True, "support_body": None}
    olo, ohi = _aabb(o.parts)
    o_area = max((ohi[0] - olo[0]) * (ohi[1] - olo[1]), 1e-9)
    best, best_top, best_frac, best_static = None, -1e30, 0.0, False
    for n, b in _objs.items():
        if n == name:
            continue
        blo, bhi = _aabb(b.parts)
        ox = min(ohi[0], bhi[0]) - max(olo[0], blo[0])
        oy = min(ohi[1], bhi[1]) - max(olo[1], blo[1])
        if ox <= 0 or oy <= 0:
            continue  # no XY footprint overlap
        if bhi[2] > olo[2] + tol:
            continue  # not beneath name's base (lateral or above)
        if bhi[2] > best_top:
            best, best_top = n, float(bhi[2])
            best_frac, best_static = (ox * oy) / o_area, bool(b.static)
    return {
        "ok": True,
        "support_body": best,
        "support_top_z": (best_top if best is not None else None),
        "static": best_static,
        "footprint_frac": round(best_frac, 3),
    }


def cmd_supports_of(name, tol=0.008, min_frac=0.1):
    """Every committed body that ``name`` rests ON or is CONTAINED BY — its whole
    support/container set, not just the single highest under-body. A body qualifies
    when its XY footprint overlaps ``name`` by >= ``min_frac`` of name's footprint AND
    its BASE is at/below name's base within ``tol`` (it extends underneath name).
    Unlike :func:`cmd_resting_on` this does NOT require the body's TOP to be below
    name's base, so a RIMMED tray/bowl whose rim rises above the object sitting in its
    well still counts as that object's container (resting_on misses it — the rim top
    is above the base). Returns {supports:[{name, static, footprint_frac, top_z,
    relation}]}, highest top first. ``relation`` is ``strict_below``,
    ``clear_below_or_container``, or ``ambiguous_same_level``; the last class remains
    useful support telemetry but must NOT be protected from contact-dependent re-settle.
    The first two protect an object's support from being freed as a false dependent (a
    fork must never free its tray) AND let the settle's atop-a-non-parent detection find
    a support that resting_on's strict test misses (wendy1 scissors wedged on a tilted
    box: box top above scissors base)."""
    o = _objs.get(name)
    if o is None:
        return {"ok": True, "supports": []}
    olo, ohi = _aabb(o.parts)
    o_area = max((ohi[0] - olo[0]) * (ohi[1] - olo[1]), 1e-9)
    out = []
    for n, b in _objs.items():
        if n == name:
            continue
        blo, bhi = _aabb(b.parts)
        ox = min(ohi[0], bhi[0]) - max(olo[0], blo[0])
        oy = min(ohi[1], bhi[1]) - max(olo[1], blo[1])
        if ox <= 0 or oy <= 0:
            continue  # no XY footprint overlap
        if blo[2] > olo[2] + tol:
            continue  # body starts ABOVE name's base -> perched on / lateral, not under
        frac = (ox * oy) / o_area
        if frac >= min_frac:
            out.append({
                "name": n, "static": bool(b.static),
                "footprint_frac": round(frac, 3), "top_z": float(bhi[2]),
                "relation": support_relation(olo, blo, bhi, frac, tol),
            })  # fmt: skip
    out.sort(key=lambda r: r["top_z"], reverse=True)
    return {"ok": True, "supports": out}


def _hull_overlaps(name, exclude):
    """Does ``name`` overlap any committed body except itself / ``exclude`` (its
    own support chain)? bbox prefilter, then convex-hull test. Uses the object's
    CURRENT o.hulls (the caller sets o.parts + rebuild_hulls per candidate)."""
    o = _objs[name]
    skip = set(exclude) | {name}
    olo, ohi = _aabb(o.parts)
    for n, obj in _objs.items():
        if n in skip:
            continue
        blo, bhi = _aabb(obj.parts)
        if (olo > bhi).any() or (ohi < blo).any():
            continue  # bbox-disjoint: cannot overlap
        if any(hulls_overlap(a, b, 0.0) for a in o.hulls for b in obj.hulls):
            return True
    return False


def cmd_clear_along(name, cap, exclude=None, budget=None, **_):
    """Topple-retry: instead of lifting a toppling tall/thin object vertically to
    clear a neighbor and dropping it (which topples it), SLIDE it horizontally to
    the NEAREST xy within a ``cap``-radius disk where it clears the committed
    bodies, then settle from contact. The CLIENT reverts the prior vertical drop
    first, so this starts from the upright pose. A DISK search (not a fixed ray):
    a camera-ray line can miss an off-axis opening — the 0715 wendy1 marker's
    clear band was in -x while its camera ray pointed -y, so the ray found
    nothing. ``exclude`` names the object's own support chain (never slid off its
    parent). Coarse AABB pass finds candidate cells cheaply; the nearest ones are
    hull-confirmed in order (rebuild_hulls only for those few). EVERY path ends in
    a settled ``cmd_drop`` (full drop fields): if no cell clears within the cap it
    vertical-drops to the baseline fallen pose. The search touches o.parts/hulls
    only (never o.total until the winning slide), so a miss leaves o.total clean."""
    o = _objs[name]
    excl = set(exclude or ())
    start = [(v.copy(), f) for v, f in o.parts]  # upright restore point
    hulls0 = list(o.hulls)  # candidate cells translate these; parts stay put
    olo, ohi = _aabb(start)
    others = [(n, *_aabb(ob.parts)) for n, ob in _objs.items()
              if n not in (excl | {name})]  # fmt: skip
    steps = int(cap / HCLEAR_STEP)

    def _aabb_clear(dx, dy):  # cheap coarse test at offset (dx,dy)
        lo = olo + [dx, dy, 0.0]
        hi = ohi + [dx, dy, 0.0]
        for _, blo, bhi in others:
            if not ((lo > bhi).any() or (hi < blo).any()):
                return False
        return True

    # ALL disk cells, nearest-first. AABB-disjoint is a cheap ACCEPT (disjoint boxes
    # => disjoint hulls); AABB-OVERLAPPING cells are NOT rejected but HULL-tested --
    # an elongated object wedged among wide-AABB neighbors (a mic on a splayed tripod)
    # clears with a few-cm slide even though its whole box never escapes theirs within
    # the cap. Old code only hull-tested AABB-disjoint cells, so the truly-clear
    # near-neighbor cells were never tried (abc3 microphone_1/stand_1 -> cleared=False
    # in a 10cm disk that physically had room). Bounded hull budget so a genuinely
    # boxed-in object still gives up cheaply.
    cells = sorted(
        (math.hypot(i * HCLEAR_STEP, j * HCLEAR_STEP), i * HCLEAR_STEP, j * HCLEAR_STEP)
        for i in range(-steps, steps + 1) for j in range(-steps, steps + 1)
        if math.hypot(i * HCLEAR_STEP, j * HCLEAR_STEP) <= cap
    )  # fmt: skip
    best = None
    hull_left = HCLEAR_HULL_BUDGET
    for dist, dx, dy in cells:  # nearest-first; truly-nearest clear cell wins
        if _aabb_clear(dx, dy):
            best = (dist, dx, dy)  # disjoint boxes => guaranteed clear, no hull test
            break
        if hull_left <= 0:
            continue
        hull_left -= 1
        o.hulls = [h.translated([dx, dy, 0.0]) for h in hulls0]
        if not _hull_overlaps(name, excl):
            best = (dist, dx, dy)
            break
    o.hulls = hulls0  # reset before the real move (parts were never touched)
    if best is not None and (best[1] or best[2]):
        _apply_delta(o, np.eye(3), np.array([best[1], best[2], 0.0]))
    # settle from contact (or baseline fall); exclude IS the ancestor chain, so
    # a failed slide's fallback drop still gets the down-preference
    r = cmd_drop(name, budget=budget, ancestors=sorted(excl))
    r["cleared"] = best is not None
    r["push_mm"] = math.hypot(best[1], best[2]) * 1000.0 if best else 0.0
    return r


def _member_penetration_mm(members, skip):
    """Deepest hull overlap of any member vs each OTHER body (statics included) at
    the current poses: {other_name: depth_mm}. Depth = min +z lift to clear, capped
    at 0.2 m — values at the cap mean "deep". Kept z-only ON PURPOSE (unlike the
    rules gate's cmd_contacts, which switched to min_clear_dist): the move commit
    compares pen_before vs pen_after in the SAME metric under a delta rule, so the
    lateral overstatement cancels, and PEN_CAP_MM/PEN_NEW_MM were tuned against
    this metric. ``skip`` names are not measured (welded members, excluded
    children, pen_ignore)."""
    out = {}
    for n, o in _objs.items():
        if n in skip:
            continue
        depth = 0.0
        for m in members:
            om = _objs[m]
            if any(hulls_overlap(h, s, 0.0) for h in om.hulls for s in o.hulls):
                depth = max(
                    depth, clear_dz(om.hulls, o.hulls, step=LIFT_STEP, cap=0.2)
                )
        if depth > 0.0:
            out[n] = depth * 1000.0
    return out


# Delta penetration rule, server-side mirror of the client's gate constants
# (composition_physics.PEN_CAP_MM / PEN_NEW_MM): a body counts as newly wedged when
# it ends deeper than the cap AND worsened by more than the floor.
_PEN_CAP_MM, _PEN_NEW_MM = 8.0, 5.0


def _srv_pen_worst(before, after):
    worst = None
    for other, a in after.items():
        w = a - before.get(other, 0.0)
        if a > _PEN_CAP_MM and w > _PEN_NEW_MM:
            if worst is None or w > worst[1]:
                worst = (other, w, a)
    return worst


def _containers_of(name, min_frac=0.5):
    """Bodies that CONTAIN ``name``: supports whose hull top rises above name's
    base (a rim/well reaching around it) and whose footprint covers name
    substantially. Exempts a tray/bowl seating from the per-body pen gate — fat
    well-filling hulls overlap 60-100 mm at a mesh-tangent rest, so gating on
    them would reject every correct container placement. KNOWN DEGENERACY: a
    deep co-level entanglement FAKES this signature (a body wedged 50 mm into a
    peer reads base-under + high frac), so a wedged pair involving a plausible
    container is exempt too. Accepted because the entangled-rest class is
    prevented UPSTREAM by the phase mechanics (one dynamic body per sim,
    lift-to-clear vs statics — no simultaneous piles), and mesh-evidence gates
    (BVH rules gate, mesh-corroborated certify) sit behind this hull-space
    backstop un-exempted."""
    o = _objs.get(name)
    if o is None:
        return set()
    olo, ohi = _aabb(o.parts)
    out = set()
    for r in cmd_supports_of(name, min_frac=min_frac)["supports"]:
        if r["top_z"] > olo[2] + 0.005:  # rim rises above the base: containment
            out.add(r["name"])
    return out


def _pair_depth_mm(a, b):
    """Symmetric pair depth: the SMALLER of the two z-lifts that separate the
    pair. clear_dz from one side alone overstates a plain resting stack — pushing
    the UNDER body up through the one resting on it reads the full stack height
    (croissant resting ON a donut read 52-72 mm from the donut's side,
    0720_settlefix_abc1). A true entanglement is deep from BOTH sides."""
    if not any(hulls_overlap(h, s, 0.0) for h in a.hulls for s in b.hulls):
        return 0.0
    return min(
        clear_dz(a.hulls, b.hulls, step=LIFT_STEP, cap=0.2),
        clear_dz(b.hulls, a.hulls, step=LIFT_STEP, cap=0.2),
    ) * 1000.0


def _per_body_pen(members, free, excl):
    """Per-body penetration maps for the client's per-body gate/report: each MEMBER
    measured with its weld siblings skipped (welded overlap is bogus), each FREE
    body measured against EVERYTHING else — this is what the old blanket pen_skip
    hid (0720_orinit3_abc1: a freed croissant ended 54 mm inside a donut, unmeasured
    and reported 'settled cleanly'). Depth is the SYMMETRIC pair depth (see
    ``_pair_depth_mm``); a body's CONTAINERS are exempt (``_containers_of`` — a
    rimmed-tray seating is by-design deep in fat hull space; mesh-REAL burials are
    still caught by the BVH rules gate and the mesh-corroborated certify)."""
    out = {}
    mset = set(members) | set(excl)
    for b in members:
        skip = (mset - {b}) | {b} | _containers_of(b)
        out[b] = {
            n: d for n, o in _objs.items()
            if n not in skip and (d := _pair_depth_mm(_objs[b], o)) > 0.0
        }  # fmt: skip
    for f in free or ():
        skip = {f} | set(excl) | _containers_of(f)
        out[f] = {
            n: d for n, o in _objs.items()
            if n not in skip and (d := _pair_depth_mm(_objs[f], o)) > 0.0
        }  # fmt: skip
    return out


_DEFINITE_SUPPORT_RELATIONS = {"strict_below", "clear_below_or_container"}


def _support_names(name):
    """Definite under-support/container names used for settle ORDERING.

    ``cmd_supports_of`` deliberately also reports ``ambiguous_same_level`` for
    telemetry.  That class is a lateral lean/contact candidate, not cargo: treating
    it as a rider puts the neighbour in ``exclude`` while its mate settles (the
    0809 breakfast bagels), so the mate is never tested against the leaner's
    collider.  Keep the broad RPC response, but use only definite vertical support
    relations wherever this server decides who may be temporarily absent.
    """
    return {
        r["name"]
        for r in cmd_supports_of(name)["supports"]
        if r.get("relation") in _DEFINITE_SUPPORT_RELATIONS
    }


def _free_order(free):
    """Support-first topological order among the free bodies (a freed support
    settles before its freed rider), name-sorted for determinism."""
    fs = set(free)
    deps = {f: _support_names(f) & fs for f in free}
    out, seen = [], set()

    def _emit(f):
        if f in seen:
            return
        seen.add(f)
        for s in sorted(deps[f]):
            _emit(s)
        out.append(f)

    for f in sorted(free):
        _emit(f)
    return out


def _dependent_free(free, members, sup_before):
    """Free bodies resting (TRANSITIVELY) on a member per the pre-commit support
    sets: they settle AFTER the members. Dropping a rider before its moving
    support means it falls through the absent support and the member then lands
    ON TOP of it (the keyboard-over-mug inversion); transitivity covers stacked
    cargo (cup on saucer on the moving tray)."""
    dep: set = set()
    base = set(members)
    changed = True
    while changed:
        changed = False
        for f in free:
            if f not in dep and sup_before.get(f, set()) & (base | dep):
                dep.add(f)
                changed = True
    return dep


def _drop_free_sequential(free, absent, budget, sup_before,
                          skip_on_support_kept=True, clear_always=()):
    """Re-settle free bodies ONE AT A TIME, support-first, each with its OWN
    lift-to-clear. ``absent`` names are removed from every sim and blocker set
    (group 1 runs with the members + member-dependents absent; group 3 with
    nothing extra absent). With ``skip_on_support_kept`` (group 1 — the
    independent free bodies), a body that kept every pre-commit support is
    SKIPPED: it never moved, so only a LOSS can invalidate its rest — the member
    arriving on/in it never forces a re-drop (the abc1 tray inversion). Group 3
    (member-dependents) passes False: they always re-seat on the member's new
    pose. One body per sim — no simultaneous piles (the 0720_orinit3_abc1 t36
    genesis). Returns ({name: record}, converged)."""
    free_out, converged = {}, True
    absent = set(absent)
    for f in _free_order(free):
        o = _objs[f]
        lost = set(sup_before.get(f, set())) - _support_names(f)
        reason = (
            f"support_lost:{','.join(sorted(lost))}" if lost
            else None if skip_on_support_kept
            else "member_dependent"
        )  # fmt: skip
        if reason is None:
            free_out[f] = {
                "D": np.eye(4).tolist(), "tilt_deg": 0.0,
                "cum_tilt_deg": _tilt(o.total[:3, :3]), "disp_mm": 0.0,
                "rollable": o.rollable, "skipped": True,
            }  # fmt: skip
            continue
        c_pre = o.centroid()
        zmin = min(float(v[:, 2].min()) for v, _ in o.parts)
        base = max(0.0, -zmin)
        fhulls = o.hulls
        if base:
            fhulls = [h.translated([0.0, 0.0, base]) for h in fhulls]
        # NEVER lift a body over its OWN riders/contents: a re-dropped container
        # whose clearance saw the member seated in its well lifted OVER it and
        # re-dropped ON TOP, inverting the stack + cascading support_lost re-drops
        # of every rider (0720_settlefix_abc1 edit #3: tray perched on its
        # pastries). Riders stay as colliders in the SIM (normal resting contact);
        # they just don't force the release height up.
        # ``clear_always`` (the members, in group 3) can never be exempted: the
        # member settled with its dependents ABSENT, so it cannot genuinely rest
        # on one — but a deep entanglement fakes the rider signature (e2 replay:
        # the settled member read as the entangled free body's rider, was dropped
        # from its clearance, and the pair re-entangled 46mm).
        riders = {r for r, ob in _objs.items()
                  if r != f and not ob.static and r not in set(clear_always)
                  and f in _support_names(r)}  # fmt: skip
        others = [h for n, ob in _objs.items()
                  if n != f and n not in absent and n not in riders
                  for h in ob.hulls]  # fmt: skip
        dz = clear_dz(fhulls, others, step=LIFT_STEP, cap=LIFT_MAX)
        if dz >= LIFT_MAX:
            dz = 0.0  # can't clear (wall overlap): release in place
        up = np.array([0.0, 0.0, base + dz + NUDGE])
        _apply_delta(o, np.eye(3), up)
        _rebuild_scene(dynamic=[f], exclude=list(absent))
        Ms, conv = _run_to_rest([_prims["__dyn__"]], _steps(budget))
        converged = converged and conv
        Rf, tf = _delta(Ms[0])
        _apply_delta(o, Rf, tf)
        Df = np.eye(4)
        Df[:3, :3], Df[:3, 3] = Rf, Rf @ up + tf  # lift then settle, vs synced
        free_out[f] = {
            "D": Df.tolist(), "tilt_deg": _tilt(Rf),
            "cum_tilt_deg": _tilt(o.total[:3, :3]),
            "disp_mm": float(np.linalg.norm(o.centroid() - c_pre) * 1000.0),
            "rollable": o.rollable, "reason": reason,
        }  # fmt: skip
    return free_out, converged


def cmd_move_group(members, deltas, budget=None, exclude=None, pen_ignore=None,
                   free=None, sequential=False, free_static=False):
    """Atomic carry + weld-settle: apply each member's 4x4 carry delta, lift the
    UNION to clear, settle all members as one compound rigid body, bake the single
    settle delta D into every member. Returns D (vs the post-carry parts) so the
    client's total blend delta per member is D @ C_member. ``exclude`` names are
    absent from the sim entirely (rotate_180: children deliberately stay put — as
    stationary colliders they'd pin the flipping parent under its own cargo).
    Also reports the members' interpenetration depths vs every other body before
    and after (pen_before_mm/pen_after_mm) — the client's penetration gate; a
    wedged body can settle quiet-and-upright, so tilt/lift alone cannot see it.
    ``pen_ignore`` names are skipped in that measurement (scale moves re-seat
    siblings right after, so member-vs-member overlap there is transient).
    ``free`` names are the bodies to re-settle around the move. With
    ``free_static`` (the COMMIT path) the commit runs in DESTINATION SUPPORT
    ORDER: (1) independent free bodies settle first, one at a time, with the
    members + member-dependents absent (skip-if-no-support-lost); (2) the member
    weld lift-to-clears the settled scene — independents BLOCK the lift, so the
    member lands ON its destination container, never through an absent one —
    and drops; (3) free bodies transitively riding a member re-seat last,
    unconditionally, on its new pose; (4) the member weld and every free body run
    one no-lift reciprocal closure as independent rigid bodies.  Lateral leaners
    therefore remain colliders throughout and both sides may react before the
    pose is returned. Nothing ever drops before the thing it will rest on, so
    container/cargo stack inversions cannot occur. Without
    ``free_static`` (the carry-decision PROBE) free bodies are individual
    dynamic bodies left at their current poses, invisible to the members' lift —
    a child losing its perch topples observably. Their rest deltas come back
    under ``free``. Also returns per-body penetration maps (``body_pen``) for
    members AND free bodies — the blanket pen_skip once hid a freed croissant
    ending 54 mm inside a donut (0720_orinit3_abc1) — and ``converged`` (False =
    some sim ended at the step cap mid-motion)."""
    excl = set(exclude or ())
    free = list(free or ())
    pen_skip = set(members) | excl | set(pen_ignore or ()) | set(free)
    pen_before = _member_penetration_mm(members, pen_skip)
    body_before = _per_body_pen(members, free, excl)
    sup_before = {f: _support_names(f) for f in free}
    for m in members:
        if _objs[m].static:
            raise ValueError(f"{m} is a static surface")
        C = np.asarray(deltas[m], dtype=float)
        _apply_delta(_objs[m], C[:3, :3], C[:3, 3])
    # pre-settle centroids (members: post-carry, pre-lift; free: current pose) —
    # the settle displacement is the class-aware stability metric for rollables
    c_pre = {n: _objs[n].centroid() for n in list(members) + free}

    def _lift_members(blocker_skip):
        """Lift the member union to clear every body NOT in ``blocker_skip``
        (never below the slab top; wall overlaps release in place)."""
        hulls = [h for m in members for h in _objs[m].hulls]
        zmin = min(
            float(v[:, 2].min()) for m in members for v, _ in _objs[m].parts
        )
        base = max(0.0, -zmin)
        if base:
            hulls = [h.translated([0.0, 0.0, base]) for h in hulls]
        others = [
            h for n, o in _objs.items() if n not in blocker_skip
            for h in o.hulls
        ]  # fmt: skip
        dz = clear_dz(hulls, others, step=LIFT_STEP, cap=LIFT_MAX)
        if dz >= LIFT_MAX:
            # never clears — composition scenes include WALLS as statics, and an
            # xy overlap with a wall can't be lifted away. Release in place and
            # let PhysX depenetration own it instead of a 1 m tower drop.
            dz = 0.0
        lift = base + dz + NUDGE
        up = np.array([0.0, 0.0, lift])
        for m in members:
            _apply_delta(_objs[m], np.eye(3), up)
        return lift, up

    def _free_rec(f, Rf, tf):
        _apply_delta(_objs[f], Rf, tf)
        Df = np.eye(4)
        Df[:3, :3], Df[:3, 3] = Rf, tf
        return {
            "D": Df.tolist(),
            "tilt_deg": _tilt(Rf),  # THIS settle's attitude change = topple test
            "cum_tilt_deg": _tilt(_objs[f].total[:3, :3]),
            "disp_mm": float(np.linalg.norm(_objs[f].centroid() - c_pre[f]) * 1000.0),
            "rollable": _objs[f].rollable,
        }

    free_out = {}
    conv_all = True
    if free and free_static:
        # COMMIT: destination support order (2026-07-20 user design).
        #   Group 1 — independent free bodies settle FIRST, with the members AND
        #   the member-dependents ABSENT (a container re-seats with nothing in
        #   its well to lift over); skip-if-no-support-lost keeps untouched
        #   bodies exactly put.
        #   Group 2 — members (welded) lift-to-clear the settled scene and drop.
        #   Group 3 — member-dependent free bodies (transitively riding a
        #   member) re-seat LAST, unconditionally, on the member's new pose.
        dep = _dependent_free(free, members, sup_before)
        indep = [f for f in free if f not in dep]
        out1, conv1 = _drop_free_sequential(
            indep, excl | set(members) | dep, budget, sup_before
        )
        conv_all = conv_all and conv1
        lift, up = _lift_members(set(members) | excl | dep)
        _rebuild_scene(dynamic=list(members), exclude=list(excl | dep))
        Ms, conv = _run_to_rest([_prims["__dyn__"]], _steps(budget))
        conv_all = conv_all and conv
        R, t = _delta(Ms[0])
        for m in members:
            _apply_delta(_objs[m], R, t)
        out3, conv3 = _drop_free_sequential(
            sorted(dep), excl, budget, sup_before,
            skip_on_support_kept=False, clear_always=members,
        )
        conv_all = conv_all and conv3
        free_out = {**out1, **out3}

        # Reciprocal local closure: direct lateral contacts were present as
        # statics while the member weld dropped, and true cargo has now re-seated.
        # Make the weld + every free/contact body dynamic together, without a new
        # lift, so neither side is frozen in an untested lean.  The member weld
        # stays a compound here, preserving the carry API's one common D.
        _rebuild_scene(dynamic=list(members), exclude=list(excl), free=free)
        prims = [_prims["__dyn__"]] + [
            _prims[f"__free_{f}__"] for f in free
        ]
        Ms, conv4 = _run_to_rest(prims, _steps(budget))
        conv_all = conv_all and conv4
        R4, t4 = _delta(Ms[0])
        for m in members:
            _apply_delta(_objs[m], R4, t4)
        # Compose the closure into the common member settle.
        D2 = np.eye(4)
        D2[:3, :3], D2[:3, 3] = R, R @ up + t
        D4 = np.eye(4)
        D4[:3, :3], D4[:3, 3] = R4, t4
        D_final = D4 @ D2
        R, t = D_final[:3, :3], D_final[:3, 3] - D_final[:3, :3] @ up
        for f, M in zip(free, Ms[1:]):
            Rf, tf = _delta(M)
            _apply_delta(_objs[f], Rf, tf)
            Df4 = np.eye(4)
            Df4[:3, :3], Df4[:3, 3] = Rf, tf
            fr = free_out[f]
            fr["D"] = (Df4 @ np.asarray(fr["D"], dtype=float)).tolist()
            fr["tilt_deg"] = max(float(fr.get("tilt_deg", 0.0)), _tilt(Rf))
            fr["cum_tilt_deg"] = _tilt(_objs[f].total[:3, :3])
            fr["disp_mm"] = float(
                np.linalg.norm(_objs[f].centroid() - c_pre[f]) * 1000.0
            )
    elif free and sequential:
        # SEQUENTIAL probe: settle the moved parent ALONE first (children absent),
        # then settle each child ONE AT A TIME onto the now-static settled scene, so
        # the parent's own motion/lift and sibling collisions can't perturb a child
        # mid-fall. The topple decision is unchanged (a child the parent slid out
        # from under still falls), but the STAY-poses that get committed are cleaner.
        lift, up = _lift_members(set(members) | excl | set(free))
        _rebuild_scene(dynamic=list(members), exclude=list(excl | set(free)))
        Ms, conv = _run_to_rest([_prims["__dyn__"]], _steps(budget))
        conv_all = conv_all and conv
        R, t = _delta(Ms[0])
        for m in members:
            _apply_delta(_objs[m], R, t)
        for f in free:  # f dynamic; settled members + siblings + others all static
            _rebuild_scene(dynamic=[f], exclude=list(excl))
            Ms, conv = _run_to_rest([_prims["__dyn__"]], _steps(budget))
            conv_all = conv_all and conv
            Rf, tf = _delta(Ms[0])
            free_out[f] = _free_rec(f, Rf, tf)
    else:
        lift, up = _lift_members(set(members) | excl | set(free))
        _rebuild_scene(dynamic=list(members), exclude=exclude, free=free)
        prims = [_prims["__dyn__"]] + [_prims[f"__free_{f}__"] for f in free]
        Ms, conv = _run_to_rest(prims, _steps(budget))
        conv_all = conv_all and conv
        R, t = _delta(Ms[0])
        for m in members:
            _apply_delta(_objs[m], R, t)
        for f, M in zip(free, Ms[1:]):
            Rf, tf = _delta(M)
            free_out[f] = _free_rec(f, Rf, tf)
    D = np.eye(4)
    D[:3, :3], D[:3, 3] = R, R @ up + t  # net vs post-carry (pre-lift) parts
    body_after = _per_body_pen(members, free, excl)
    return {
        "ok": True,
        "D": D.tolist(),
        "tilt_deg": _tilt(R),
        "cum_tilt_deg": {m: _tilt(_objs[m].total[:3, :3]) for m in members},
        # settle displacement per member (mm, centroid, vs the post-carry pose)
        # + rollability: rollable members are gated on displacement, not tilt
        "disp_mm": {
            m: float(np.linalg.norm(_objs[m].centroid() - c_pre[m]) * 1000.0)
            for m in members
        },
        "rollable": {m: _objs[m].rollable for m in members},
        "lift_mm": (lift - NUDGE) * 1000.0,
        "pen_before_mm": pen_before,
        "pen_after_mm": _member_penetration_mm(members, pen_skip),
        "body_pen": {
            b: {"before": body_before.get(b, {}), "after": body_after.get(b, {})}
            for b in list(members) + free
        },
        "body_stacked": _body_stacked(members, free, excl, body_after),
        "converged": conv_all,
        **({"free": free_out} if free else {}),
    }


def _zrange(o):
    vs = np.vstack([v for v, _ in o.parts])
    return float(vs[:, 2].min()), float(vs[:, 2].max())


def _body_stacked(members, free, excl, pen):
    """For every measured pair in ``pen`` (body -> {other: depth_mm}), whether the pair is
    STACKED (one body's bottom at/above the other's top within 1 cm) rather than side by
    side — the client's sibling-vs-stacked test for its hull allowance (audit F-M11)."""
    out = {}
    for b, others in pen.items():
        zb = _zrange(_objs[b])
        row = {}
        for n in others:
            zn = _zrange(_objs[n])
            row[n] = bool(zb[0] >= zn[1] - 0.01 or zn[0] >= zb[1] - 0.01)
        out[b] = row
    return out


def cmd_settle_set(bodies, deltas, free=None, budget=None, groups=None):
    """Joint settle of SEVERAL independently moved hierarchies — the gpt6_v1 freeform
    ``execute_and_evaluate`` that moved more than one object (2026-09-15 owner rule:
    only the moved objects' hierarchies simulate; everything else is a static
    collider). Apply each body's 4x4 world delta (its requested pose; the client
    carries descendants), lift the UNION of ``bodies`` straight up until it clears
    every body NOT in the set (relative arrangement preserved; a wall overlap that
    never clears releases in place, as in move_group), then run ONE sim with every
    body in ``bodies`` + ``free`` (contact-dependents resting on them, at their
    current poses, no lift) as INDIVIDUAL dynamic rigid bodies against the rest of
    the committed scene as statics, and bake each settle into its total. No weld:
    two bodies the agent left overlapping depenetrate against each other instead of
    being frozen into one compound, and every body is measured against every other
    (``body_pen``), so the client's penetration gate sees a body wedged in its
    co-moved neighbor. Returns per-body D (net vs the post-delta pose, lift
    included: the client's total blend delta per body = D @ C_body), this-settle
    tilt, cumulative tilt, settle displacement, rollability, ``lift_mm``,
    ``pen_before_mm``/``pen_after_mm`` (bodies vs the rest, move_group metric),
    ``body_pen`` and ``converged``."""
    bodies = list(bodies)
    free = [f for f in (free or ()) if f not in bodies]
    for b in bodies:
        if _objs[b].static:
            raise ValueError(f"{b} is a static surface")
    pen_skip = set(bodies) | set(free)
    pen_before = _member_penetration_mm(bodies, pen_skip)
    body_before = _per_body_pen([], bodies + free, set())  # each vs everything
    for b in bodies:
        C = np.asarray(deltas[b], dtype=float)
        _apply_delta(_objs[b], C[:3, :3], C[:3, 3])
    names = bodies + free
    c_pre = {n: _objs[n].centroid() for n in names}
    # Per-HIERARCHY lift-to-clear (audit F-M3): each moved hierarchy is lifted only as
    # far as IT needs to clear the non-set bodies, so one body's wall overlap (never
    # clears -> release in place) or big lift no longer drops every co-moved body from
    # the same height. Groups default to one body each.
    groups = [list(g) for g in (groups or [[b] for b in bodies])]
    others = [h for n, o in _objs.items() if n not in pen_skip for h in o.hulls]
    up_of, lift_max = {}, 0.0
    for g in groups:
        hulls = [h for b in g for h in _objs[b].hulls]
        zmin = min(float(v[:, 2].min()) for b in g for v, _ in _objs[b].parts)
        base = max(0.0, -zmin)
        if base:
            hulls = [h.translated([0.0, 0.0, base]) for h in hulls]
        dz = clear_dz(hulls, others, step=LIFT_STEP, cap=LIFT_MAX)
        if dz >= LIFT_MAX:
            dz = 0.0  # an xy overlap with a wall never clears: release in place
        lift = base + dz + NUDGE
        up = np.array([0.0, 0.0, lift])
        for b in g:
            _apply_delta(_objs[b], np.eye(3), up)
            up_of[b] = up
        lift_max = max(lift_max, lift)
    lift = lift_max
    _rebuild_scene(free=names)
    prims = [_prims[f"__free_{n}__"] for n in names]
    Ms, converged = _run_to_rest(prims, _steps(budget))
    unconverged = []
    if not converged:  # name the bodies still moving (audit F-M4)
        speeds = _body_speed_report(prims)
        unconverged = [
            n for n, prim in zip(names, prims)
            if not speeds[str(prim.GetPath())]["below_limits"]
        ]  # fmt: skip
    D, tilt, cum, disp, roll = {}, {}, {}, {}, {}
    for n, M in zip(names, Ms):
        R, t = _delta(M)
        _apply_delta(_objs[n], R, t)
        Dn = np.eye(4)
        Dn[:3, :3] = R
        Dn[:3, 3] = (R @ up_of[n] + t) if n in up_of else t  # net vs post-delta pose
        D[n] = Dn.tolist()
        tilt[n] = _tilt(R)
        cum[n] = _tilt(_objs[n].total[:3, :3])
        disp[n] = float(np.linalg.norm(_objs[n].centroid() - c_pre[n]) * 1000.0)
        roll[n] = _objs[n].rollable
    body_after = _per_body_pen([], names, set())
    return {
        "ok": True,
        "D": D,
        "tilt_deg": tilt,
        "cum_tilt_deg": cum,
        "disp_mm": disp,
        "rollable": roll,
        "lift_mm": (lift - NUDGE) * 1000.0,
        "pen_before_mm": pen_before,
        "pen_after_mm": _member_penetration_mm(bodies, pen_skip),
        "body_pen": {
            n: {"before": body_before.get(n, {}), "after": body_after.get(n, {})}
            for n in names
        },
        "body_stacked": _body_stacked([], names, set(), body_after),
        "converged": bool(converged),
        "unconverged_bodies": unconverged,
    }


def cmd_scale_resettle(parent, delta, free=None, budget=None, absent=None):
    """Scale-specific commit (NOT a carry), in the same DESTINATION SUPPORT ORDER
    as the move commit: (1) lateral contacts remain in place; (2) the resized
    parent lift-to-clears and drops AGAINST their colliders while only true cargo
    is absent; (3) that cargo re-seats last, support-first; (4) parent, lateral
    contacts, and cargo run one no-lift all-dynamic local closure. Returns, per
    body, the rest delta D
    (parent's D is vs its POST-SCALE pre-lift pose so the client can do
    ``D @ scale``; a free body's D is vs its synced pose),
    tilt/displacement/rollability, the parent lift, the stack's interpenetration
    before/after (legacy gate), per-body penetration maps (``body_pen``), and
    ``converged``."""
    free = list(free or ())
    absent = [a for a in (absent or ()) if a in set(free)]
    stack = [parent] + free
    skip = set(stack)
    pen_before = _member_penetration_mm(stack, skip)
    body_before = _per_body_pen([parent], free, set())
    sup_before = {f: _support_names(f) for f in free}
    dep = _dependent_free(free, [parent], sup_before) | set(absent)

    if _objs[parent].static:
        raise ValueError(f"{parent} is a static surface")
    C = np.asarray(delta, dtype=float)
    _apply_delta(_objs[parent], C[:3, :3], C[:3, 3])

    # Response deltas are measured from these poses: parent AFTER the candidate
    # scale, every free body at its synced pose.  Computing them once at the end
    # lets the reciprocal closure below move every body independently without
    # losing the scale_resettle API's per-body semantics.
    total_base = {n: _objs[n].total.copy() for n in stack}
    centroid_base = {n: _objs[n].centroid() for n in stack}

    # -- Group 1: keep lateral contacts in place ----------------------------------
    # ``free - dep`` is the lateral/contact class after _support_names filters out
    # ambiguous_same_level.  Do NOT settle it with the parent absent: the pair's
    # mutual contact is exactly the constraint being preserved.  It remains a
    # static collider during group 2, then both sides become dynamic in group 4.
    conv_all = True

    # -- Group 2: parent alone (scale in place, lift-to-clear, drop) ---------------
    blockers_skip = {parent} | dep
    others = [h for n, o in _objs.items() if n not in blockers_skip for h in o.hulls]
    zmin = min(float(v[:, 2].min()) for v, _ in _objs[parent].parts)
    base = max(0.0, -zmin)  # never start below the slab top
    phulls = _objs[parent].hulls
    if base:
        phulls = [h.translated([0.0, 0.0, base]) for h in phulls]
    dz = clear_dz(phulls, others, step=LIFT_STEP, cap=LIFT_MAX)
    if dz >= LIFT_MAX:
        dz = 0.0  # can't clear a wall vertically -> release in place, sim depenetrates
    lift = base + dz + NUDGE
    _apply_delta(_objs[parent], np.eye(3), np.array([0.0, 0.0, lift]))
    # Only true cargo is absent.  Lateral leaners in ``indep`` are authored as
    # ordinary static colliders, so the resized parent must settle against them.
    _rebuild_scene(dynamic=[parent], exclude=list(dep))
    Ms, conv = _run_to_rest([_prims["__dyn__"]], _steps(budget))
    conv_all = conv_all and conv
    Rp, tp = _delta(Ms[0])
    _apply_delta(_objs[parent], Rp, tp)
    # -- Group 3: dependents re-seat last, unconditionally --------------------------
    if dep:
        _out3, conv_b = _drop_free_sequential(
            sorted(dep), set(), budget, sup_before,
            skip_on_support_kept=False, clear_always=[parent],
        )
        conv_all = conv_all and conv_b

    # -- Group 4: reciprocal local closure ----------------------------------------
    # Group 2 proved the parent against the lateral bodies, and group 3 re-seated
    # true cargo, but neither proves that ALL affected bodies remain at rest when
    # allowed to react.  Run the exact local analogue of final certification: no
    # lift, no exclusions, and one independent dynamic rigid body per member.
    # This is where the breakfast bagel's 9-degree response must happen -- inside
    # composition feedback, not later in the otherwise-unseen final certify.
    if free:
        _rebuild_scene(dynamic=[parent], free=free)
        prims = [_prims["__dyn__"]] + [
            _prims[f"__free_{f}__"] for f in free
        ]
        Ms, conv_c = _run_to_rest(prims, _steps(budget))
        conv_all = conv_all and conv_c
        for n, M in zip(stack, Ms):
            Rn, tn = _delta(M)
            _apply_delta(_objs[n], Rn, tn)

    D = {
        n: _objs[n].total @ np.linalg.inv(total_base[n])
        for n in stack
    }
    step_tilts = {n: _tilt(D[n][:3, :3]) for n in stack}
    out = {
        "ok": True,
        "D": {n: D[n].tolist() for n in stack},
        "tilt_deg": max(step_tilts.values(), default=_tilt(Rp)),
        "cum_tilt_deg": {
            n: _tilt(_objs[n].total[:3, :3]) for n in stack
        },
        "disp_mm": {
            n: float(
                np.linalg.norm(_objs[n].centroid() - centroid_base[n]) * 1000.0
            )
            for n in stack
        },
        "rollable": {n: _objs[n].rollable for n in stack},
        "lift_mm": (lift - NUDGE) * 1000.0,
    }

    out["pen_before_mm"] = pen_before
    out["pen_after_mm"] = _member_penetration_mm(stack, skip)
    body_after = _per_body_pen([parent], free, set())
    out["body_pen"] = {
        b: {"before": body_before.get(b, {}), "after": body_after.get(b, {})}
        for b in stack
    }
    out["converged"] = conv_all
    return out


def cmd_penetration(members, deltas):
    """DRY-RUN feasibility probe for the pre-physics move clamp: apply each member's
    4x4 ``delta``, measure the deepest overlap of the members vs every OTHER body
    (statics included; member-vs-member skipped — they move as a rigid compound),
    then EXACTLY restore the pre-probe state. Returns {other_name: depth_mm}. No
    physics is run — this is a geometric hull test at the candidate pose."""
    # snapshot EVERYTHING _apply_delta mutates — com/axes/inertia/mass ride the
    # body too; restoring only the geometry leaked the probed translation into
    # the CoM override, making every stabilized object tip on all later settles
    # (0725_tipfix2_room1 vase: 86-110 deg on every composition move)
    snap = {
        m: ([(v.copy(), f) for v, f in _objs[m].parts], _objs[m].total.copy(),
            list(_objs[m].hulls),
            None if _objs[m].com is None else _objs[m].com.copy(),
            None if _objs[m].axes is None else _objs[m].axes.copy(),
            None if _objs[m].diag_inertia is None
            else _objs[m].diag_inertia.copy(),
            _objs[m].mass)
        for m in members
    }
    for m in members:
        C = np.asarray(deltas[m], dtype=float)
        _apply_delta(_objs[m], C[:3, :3], C[:3, 3])
    pen = _member_penetration_mm(members, set(members))
    for m in members:  # exact restore (no accumulated float drift)
        o = _objs[m]
        (o.parts, o.total, o.hulls, o.com, o.axes, o.diag_inertia,
         o.mass) = snap[m]
    return {"ok": True, "pen": pen}


def cmd_contacts(tol=0.0, pairs_only=False):
    """Pairwise hull-overlap report at the CURRENT poses (the composition rules
    gate): overlapping pairs + how deep. Depth = the smallest single-direction
    push (+-x, +-y, +z) of the non-static member that separates the pair
    (min_clear_dist) -- the old z-only lift read a hairline LATERAL kiss between
    two lying utensils as the neighbor's full height (0720_leanfix_abc1 false
    FAIL). Static-static pairs are skipped. ``tol`` dilates the overlap test so
    near-contacts within ``tol`` are reported too -- the dependency query
    (_contact_deps) passes a few mm to catch leaners that don't quite
    interpenetrate. ``pairs_only`` skips the per-pair depth scan and omits
    ``depth_mm`` entirely (never a fake 0.0): the dependency query reads only
    the pair NAMES, and the depth scans were ~57 of a 59 s stacked-scene sweep.
    Gated on the EXPLICIT flag, not on tol != 0 — the depth-consuming callers
    (rules gate, certify) keep their path regardless of tolerance."""
    names = list(_objs)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            oa, ob = _objs[a], _objs[b]
            if oa.static and ob.static:
                continue
            if not any(hulls_overlap(h, s, 0.0, tol) for h in oa.hulls for s in ob.hulls):
                continue
            if pairs_only:
                pairs.append({"a": a, "b": b})
                continue
            probe, other = (ob, oa) if oa.static else (oa, ob)
            if tol != 0.0:
                depth = clear_dz(probe.hulls, other.hulls, step=LIFT_STEP, cap=0.2)
            elif other.static:
                depth = min_clear_dist(probe.hulls, other.hulls, step=LIFT_STEP, cap=0.2)
            else:
                # object<->object: SYMMETRIC — registry order (sorted names) must not
                # pick the probe; a container pushed out of its cargo reads the escape
                # distance (plate<->tomato 44 mm vs 4-6 mm, 0904 fruits_plate_to_bowl).
                depth = pair_clear_dist(probe.hulls, other.hulls, step=LIFT_STEP, cap=0.2)
            pairs.append({"a": a, "b": b, "depth_mm": depth * 1000.0})
    return {"ok": True, "pairs": pairs}


def _certify_pass(
    pinned, *, include_support_proxy=True, rest_policy=None, rest_report=None
):
    """One all-dynamic free sim (pinned names authored as statics at their current
    poses). Returns {name: (R, t)} settle deltas for the objects that ran dynamic —
    nothing is applied; the caller decides."""
    global _world
    _validate_rest_policy(rest_policy)
    World.clear_instance()
    stage_utils.create_new_stage()
    _world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    stage = _world.stage
    apply_scene_tuning(stage)  # no-op unless GRASE_PHYSX_TUNED=1
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    surf_mat = _material(stage, "/World/PhysMatSurf", SURF_FRICTION)
    if include_support_proxy:
        _author(stage, "/World/ground", [(_B, _BF)], surf_mat)
    prims, names = [], []
    for i, (name, o) in enumerate(_objs.items()):
        if o.static or name in pinned:
            _author(
                stage,
                f"/World/obj_{i:03d}",
                o.parts,
                _material(stage, f"/World/PhysMat_{i:03d}", SURF_FRICTION),
            )
            continue
        mat = _material(stage, f"/World/PhysMat_{i:03d}", o.friction or OBJ_FRICTION)
        prims.append(_author(
            stage, f"/World/obj_{i:03d}", o.parts, mat, dynamic=True,
            com=o.com, damping=o.damping,
            diag_inertia=o.diag_inertia, axes=o.axes, mass_kg=o.mass,
        ))  # fmt: skip
        names.append(name)
    _world.reset()
    if rest_policy is None:
        Ms, converged = _run_to_rest(prims)
    else:
        measured = {}
        Ms, converged, _, _ = _run_to_rest_phases(
            prims, (MAX_STEPS,), rest_policy=rest_policy, rest_report=measured
        )
        measured["objects"] = {
            name: measured["objects"][str(prim.GetPath())]
            for name, prim in zip(names, prims)
        }
        measured["offending_objects"] = sorted(
            name for name, row in measured["objects"].items()
            if not row["below_limits"]
        )
        if rest_report is not None:
            rest_report.update(measured)
    return {name: _delta(M) for name, M in zip(names, Ms)}, converged


def _com_drift(o, R, t):
    """Drift as CENTROID displacement, not the delta's translation — under a large
    rotation t is a pivot artifact (0715 abc3: a 123-deg roll read as 'dxy=1007mm'
    when the flashlight really moved 322mm)."""
    c = np.vstack([v for v, _ in o.parts]).mean(0)
    d = np.asarray(R) @ c + t - c
    return {
        "dxy": float(np.linalg.norm(d[:2])),
        "dz": float(d[2]),
        "tilt_deg": _tilt(np.asarray(R)),
    }


def cmd_certify(
    dxy_cap=None, tilt_cap_deg=None, *, support_policy=None, rest_policy=None
):
    """Unfreeze ALL objects, free-sim to rest, bake the drift. With caps set, an
    object drifting past them is PINNED — kept at its composed pose, authored as a
    static — and the sim re-runs without it, so a capsize/ejection is RECORDED
    (drift entry + pinned:true) instead of baked into the deliverable. Without
    caps (the preprocess certify) behavior is the old single pass.

    ``support_policy="actual_surfaces_v1"`` explicitly omits the global proxy,
    requires registered static colliders, and acknowledges their exact names.
    An absent policy retains the legacy proxy and response shape. This is a
    per-request policy only; it never changes later simulations' defaults.

    ``rest_policy="export_velocity_v1"`` additionally requires the existing quiet
    streak to satisfy the export speed limits. The response acknowledges it and
    includes bounded per-object measurements under ``rest``. Without it, legacy
    stopping and response fields remain unchanged.
    """
    _validate_rest_policy(rest_policy)
    if support_policy is not None and support_policy != "actual_surfaces_v1":
        raise ValueError(f"unsupported certification support_policy={support_policy!r}")
    actual_surfaces = support_policy == "actual_surfaces_v1"
    surfaces = sorted(n for n, o in _objs.items() if o.static) if actual_surfaces else []
    if actual_surfaces and not surfaces:
        raise ValueError("actual_surfaces_v1 certification requires static colliders")
    if not _objs:
        result = {"ok": True, "drift": {}, "total": {}, "converged": True}
        if rest_policy is not None:
            result.update(rest_policy=rest_policy, rest=_rest_result({}, True, 0, 0))
        return result
    pinned, drift = set(), {}
    rest = {}
    for _ in range(1 + (3 if dxy_cap or tilt_cap_deg else 0)):
        if rest_policy is not None:
            deltas, converged = _certify_pass(
                pinned, include_support_proxy=not actual_surfaces,
                rest_policy=rest_policy, rest_report=rest,
            )
        elif actual_surfaces:
            deltas, converged = _certify_pass(pinned, include_support_proxy=False)
        else:
            deltas, converged = _certify_pass(pinned)
        viol = []
        for name, (R, t) in deltas.items():
            d = _com_drift(_objs[name], R, t)
            drift[name] = d
            # rollable objects roll in place at certify too (a lying mic, a
            # round pastry): displacement still caps them, tilt does not
            tilt_bad = (
                tilt_cap_deg
                and d["tilt_deg"] > tilt_cap_deg
                and not _objs[name].rollable
            )
            if (dxy_cap and d["dxy"] > dxy_cap) or tilt_bad:
                viol.append(name)
        if not viol:
            break
        for name in viol:
            pinned.add(name)
            drift[name]["pinned"] = True  # keeps the FIRST violating measurement
    for name, (R, t) in deltas.items():
        if name not in pinned:  # a last-pass violator must not be baked either
            _apply_delta(_objs[name], R, t)
    result = {
        "ok": True,
        "drift": drift,
        "converged": converged,  # False = step cap hit mid-motion, not a rest
        "total": {n: o.total.tolist() for n, o in _objs.items()},
    }
    if actual_surfaces:
        result.update(
            support_policy="actual_surfaces_v1", support_proxy=False, surfaces=surfaces
        )
    if rest_policy is not None:
        result.update(rest_policy=rest_policy, rest=rest)
    return result


def cmd_validate_surfaces(*, rest_policy=None):
    """Diagnostic free settle on actual static meshes, with NO tabletop proxy.

    Registry poses/totals are never modified and no drift is baked or pinned.
    The next simulation reconstructs its stage from that unchanged registry.
    An explicit rest policy uses the same velocity-aware condition as certify.
    """
    _validate_rest_policy(rest_policy)
    surfaces = sorted(n for n, o in _objs.items() if o.static)
    if not surfaces:
        result = {
            "ok": True,
            "status": "unavailable",
            "reason": "no actual static colliders",
            "support_proxy": False,
            "surfaces": [],
        }
        if rest_policy is not None:
            result.update(rest_policy=rest_policy, rest=_rest_result({}, False, 0, 0))
        return result
    rest = {}
    if rest_policy is None:
        deltas, converged = _certify_pass(set(), include_support_proxy=False)
    else:
        deltas, converged = _certify_pass(
            set(), include_support_proxy=False,
            rest_policy=rest_policy, rest_report=rest,
        )
    result = {
        "ok": True,
        "status": "measured",
        "support_proxy": False,
        "surfaces": surfaces,
        "converged": converged,
        "drift": {n: _com_drift(_objs[n], R, t) for n, (R, t) in deltas.items()},
    }
    if rest_policy is not None:
        result.update(rest_policy=rest_policy, rest=rest)
    return result


def cmd_certify_sequential(dxy_cap=None, tilt_cap_deg=None, order=None, repair=None):
    """SEQUENTIAL certify: settle each object onto the scene BUILT UP SO FAR, in
    ``order`` (support/DFS: parents before children), instead of the all-dynamic joint
    free sim of :func:`cmd_certify`. A joint sim lets a marginally stable object be
    knocked over by a settling neighbor (false topple); the build-up removes that
    coupling. Crucially each object is dropped with ONLY the already-committed objects
    present (via ``scene_names``) — later objects, INCLUDING this object's own cargo,
    are ABSENT. Authoring the cargo static while dropping its support would pin the
    support DOWN into the slab (a rimmed tray shoved under the table by the food sitting
    in it: abc1/abc2 — sequential dropped it to -4.5mm vs a joint/build-up -0.1mm);
    the build-up lets each support settle free before its cargo lands, matching the
    preprocess incremental settle. Each object keeps its embedded CoM/damping (set at
    ``add``). An object drifting past ``dxy_cap``/``tilt_cap_deg`` is PINNED (kept at
    its composed pose); pinned or committed, it is then present (static) for later
    drops. Returns the same {drift, total} shape as cmd_certify.

    ``repair`` names get cmd_drop's lift-to-clear prepended to their own build-up
    drop: a plain settle cannot push a buried body out (0720_orinit3_abc1), so the
    caller flags mesh-corroborated deep pre-contact members and each escapes its
    wedge against the COMMITTED scene only — a wedged support drops onto the bare
    scene with its cargo absent, instead of being lifted out from under statically
    frozen cargo and landing propped against it (0722_bulk_groot2: the standalone
    pre-certify re-place baked a tray at 36deg into the deliverable, past the pin).
    The repaired drop is measured/capped vs the PRE-LIFT composed pose like any
    other; a repair still past the caps pins back to the composed pose."""
    if not _objs:
        return {"ok": True, "drift": {}, "total": {}}
    seq = [n for n in (order or list(_objs)) if n in _objs and not _objs[n].static]
    seq += [n for n, o in _objs.items() if not o.static and n not in seq]  # any missed
    repair = set(repair or ())
    drift = {}
    committed = []  # already-settled objects: the ONLY non-static bodies each drop sees
    for name in seq:
        o = _objs[name]
        pen_pre = _member_penetration_mm([name], {name})
        lift = 0.0
        if name in repair:
            lift = _clear_dz(name, committed) + NUDGE
            _apply_delta(o, np.eye(3), np.array([0.0, 0.0, lift]))
        _rebuild_scene(dynamic=[name], scene_names=committed)  # build-up: cargo absent
        Ms, _ = _run_to_rest([_prims["__dyn__"]])
        R, t = _delta(Ms[0])
        # (R, t) is vs the LIFTED start; dxy/tilt vs the pre-lift pose are identical
        # (the lift is pure +z), only dz shifts by the lift
        d = _com_drift(o, R, t)
        if lift:
            d["dz"] = float(d["dz"] + lift)
            d["repair_lift_mm"] = round(lift * 1000.0, 1)
        drift[name] = d
        tilt_bad = (
            tilt_cap_deg and d["tilt_deg"] > tilt_cap_deg and not o.rollable
        )
        if (dxy_cap and d["dxy"] > dxy_cap) or tilt_bad:
            drift[name]["pinned"] = True  # keep composed pose (do NOT apply the settle)
            if lift:
                _apply_delta(o, np.eye(3), np.array([0.0, 0.0, -lift]))  # undo the lift
        else:
            _apply_delta(o, R, t)  # commit; later objects land on this settled pose
            # penetration-aware pin (rollables NOT exempt — penetration is
            # pose-independent, unlike the tilt cap): a certify drop that ends the
            # object newly wedged (delta rule, 0720_orinit3_abc1) reverts to the
            # composed pose instead of baking the burial.
            worst = _srv_pen_worst(pen_pre, _member_penetration_mm([name], {name}))
            if worst is not None:
                _apply_delta(o, R.T, -(R.T @ t))  # exact inverse of the settle
                if lift:
                    _apply_delta(o, np.eye(3), np.array([0.0, 0.0, -lift]))
                drift[name]["pinned"] = True
                drift[name]["pen_other"] = worst[0]
                drift[name]["pen_after_mm"] = round(worst[2], 1)
        committed.append(name)  # present (static) for later drops, pinned or not
    return {
        "ok": True,
        "drift": drift,
        "total": {n: o.total.tolist() for n, o in _objs.items()},
    }


def main():
    log_protocol("settle", settle_hz=1.0 / DT)
    _respond({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "shutdown":
            _respond({"ok": True})
            break
        _respond(_handle(req))
    app.close()


def _handle(req):
    """Dispatch one request (any cmd except shutdown) -> response dict."""
    cmd = req.get("cmd")
    try:
        if cmd in ("add", "swap", "add_static"):
            # swap = replace geometry (pristine / stabilized rung); `total` resets
            # to identity — the new npz IS the new baseline the client bakes onto.
            parts = load_parts(req["npz"])
            if req.get("flatten_mm"):
                parts = flatten_base(parts, float(req["flatten_mm"]))
            _objs[req["name"]] = Obj(
                req["name"], parts, com=req.get("com"),
                friction=req.get("friction"), damping=req.get("damping"),
                static=cmd == "add_static", rollable=req.get("rollable"),
                diag_inertia=req.get("diagonal_inertia"),
                axes_quat=req.get("principal_axes"),
                mass=req.get("mass"),
            )  # fmt: skip
            return {"ok": True}
        elif cmd == "drop":
            return cmd_drop(req["name"], req.get("scene"), req.get("budget"),
                            req.get("ancestors"))
        elif cmd == "clear_along":
            return cmd_clear_along(
                req["name"], req["cap"],
                req.get("exclude"), req.get("budget"))
        elif cmd == "move_group":
            return cmd_move_group(
                req["members"], req["deltas"], req.get("budget"),
                req.get("exclude"), req.get("pen_ignore"),
                req.get("free"), req.get("sequential", False),
                req.get("free_static", False))
        elif cmd == "settle_set":
            return cmd_settle_set(
                req["bodies"], req["deltas"], req.get("free"), req.get("budget"),
                req.get("groups"))
        elif cmd == "scale_resettle":
            return cmd_scale_resettle(
                req["parent"], req["delta"], req.get("free"),
                req.get("budget"), req.get("absent"))
        elif cmd == "contacts":
            return cmd_contacts(req.get("tol", 0.0), req.get("pairs_only", False))
        elif cmd == "penetration":
            return cmd_penetration(req["members"], req["deltas"])
        elif cmd == "resting_on":
            return cmd_resting_on(req["name"])
        elif cmd == "supports_of":
            return cmd_supports_of(req["name"])
        elif cmd == "clearance":
            return {"ok": True, "dz": _clear_dz(req["name"], req.get("scene"))}
        elif cmd == "transform":
            M = np.asarray(req["M"], dtype=float)
            _apply_delta(_objs[req["name"]], M[:3, :3], M[:3, 3])
            return {"ok": True}
        elif cmd == "remove":
            _objs.pop(req["name"], None)
            return {"ok": True}
        elif cmd == "pose":
            o = _objs[req["name"]]
            return {"ok": True, "total": o.total.tolist(),
                    "cum_tilt_deg": _tilt(o.total[:3, :3])}
        elif cmd == "certify":
            if "rest_policy" in req:
                return cmd_certify(
                    req.get("dxy_cap"), req.get("tilt_cap_deg"),
                    support_policy=req.get("support_policy"),
                    rest_policy=req["rest_policy"],
                )
            if "support_policy" in req:
                return cmd_certify(
                    req.get("dxy_cap"), req.get("tilt_cap_deg"),
                    support_policy=req["support_policy"],
                )
            return cmd_certify(req.get("dxy_cap"), req.get("tilt_cap_deg"))
        elif cmd == "validate_surfaces":
            if "rest_policy" in req:
                return cmd_validate_surfaces(rest_policy=req["rest_policy"])
            return cmd_validate_surfaces()
        elif cmd == "certify_sequential":
            return cmd_certify_sequential(
                req.get("dxy_cap"), req.get("tilt_cap_deg"), req.get("order"),
                req.get("repair"))
        elif cmd == "ping":
            return {"ok": True, "pid": os.getpid()}
        elif cmd == "reset":
            # The registry is the ONLY cross-command state (the USD stage is
            # re-authored from _objs on every sim), so clearing it makes the
            # warm server indistinguishable from a fresh boot.
            _objs.clear()
            return {"ok": True}
        else:
            return {"ok": False, "error": f"unknown cmd {cmd}"}
    except Exception as exc:  # noqa: BLE001 - keep the server alive on bad input
        return {"ok": False, "error": str(exc)[:300]}


def _send(conn, obj):
    try:
        conn.sendall((json.dumps(obj) + "\n").encode())
    except OSError:
        pass  # peer vanished mid-response; the select loop reaps it next pass


WATCHDOG_TICK = 5.0  # select() timeout: how often the two liveness checks run
IDLE_TIMEOUT = float(os.environ.get("GRASE_ISAAC_IDLE_TIMEOUT", 1800.0))


def _owner_alive(owner_pid):
    """True if the run that owns this server is still around (or unknown)."""
    if not owner_pid:
        return True
    try:
        os.kill(owner_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, just not ours
    return True


def main_tcp(port_file: str, owner_pid: int = 0, idle_timeout: float = IDLE_TIMEOUT):
    """Serve the same protocol over 127.0.0.1 so later pipeline stages (other OS
    processes) reconnect to this SimulationApp instead of booting their own.

    Two self-reaping guards, since the client-side teardown is a Python ``finally``
    that SIGKILL/OOM-kill/eviction skips entirely (leaked servers pinned ~11 GB
    across all 8 GPUs until noticed by hand). Both are checked only on an IDLE
    select() tick, so neither can interrupt an in-flight rpc:
      * ``--owner-pid``: exit once the process that owns the RUN is gone. Not the
        parent — under a standalone main.py the spawner is a per-stage exec.py MCP
        child that dies before certify reuses the server (see SettleClient).
      * ``idle_timeout``: exit after this long with NO client connected (0 = never).
        Backstop for callers that never shut the server down at all. Must stay
        comfortably above the longest legitimate inter-stage gap — a stage that
        holds the socket while doing Blender work is NOT idle and never trips it.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    tmp = port_file + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"port": srv.getsockname()[1], "pid": os.getpid()}, f)
    os.replace(tmp, port_file)  # atomic: a connecting client never sees a torn file
    log_protocol("settle", settle_hz=1.0 / DT)
    _respond({"ready": True})  # spawner boot-waits on stdout, same as pipe mode
    conn, buf, stop = None, b"", False
    idle_since = time.monotonic()
    try:
        while not stop:
            readable = select.select(
                [srv] + ([conn] if conn else []), [], [], WATCHDOG_TICK
            )[0]
            if not readable:  # idle tick: the only place the guards may fire
                if not _owner_alive(owner_pid):
                    print(f"owner pid {owner_pid} gone; exiting", file=sys.stderr,
                          flush=True)  # fmt: skip
                    break
                if (
                    conn is None
                    and idle_timeout
                    and time.monotonic() - idle_since > idle_timeout
                ):
                    print(f"idle {idle_timeout:.0f}s with no client; exiting",
                          file=sys.stderr, flush=True)  # fmt: skip
                    break
                continue
            if srv in readable:
                new, _ = srv.accept()
                if conn is not None:
                    conn.close()  # latest client wins; a stale peer sees EOF
                conn, buf = new, b""
                continue
            data = conn.recv(65536)
            if not data:
                conn.close()
                conn, buf = None, b""
                idle_since = time.monotonic()  # start the no-client clock
                continue
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if req.get("cmd") == "shutdown":
                    _send(conn, {"ok": True})
                    stop = True
                    break
                _send(conn, _handle(req))
    finally:
        if conn is not None:
            conn.close()
        srv.close()
        try:
            os.remove(port_file)
        except OSError:
            pass
        app.close()


if __name__ == "__main__":
    if "--port-file" in sys.argv:
        _owner = (
            int(sys.argv[sys.argv.index("--owner-pid") + 1])
            if "--owner-pid" in sys.argv
            else 0  # 0 = no owner known: idle_timeout is then the only guard
        )
        main_tcp(sys.argv[sys.argv.index("--port-file") + 1], owner_pid=_owner)
    else:
        main()
