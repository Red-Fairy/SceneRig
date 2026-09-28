"""Headless Isaac Sim settle test for a physics-stamped GRASE scene USD.

    $SCENERIG_ISAAC_PYTHON isaac/isaac_verify_settle.py \
        <scene.usd> <report.json> [drift_tol_m]

Loads the scene, simulates 5 s of physics, and reports per-object
translation/rotation drift, final velocity, dynamic/kinematic status, and an
AABB support candidate, with measured PhysX contact fallback when that search
finds none. Dynamic verification requires every identity-mapped
body to be present, dynamic, settled, supported, and under all drift thresholds.
"""

import hashlib
import json
import math
import os
import sys
import traceback
from importlib import metadata

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from physics_config import (  # noqa: E402
    ANGULAR_REST_SPEED_DEG_S,
    LINEAR_REST_SPEED_M_S,
    apply_scene_tuning,
    log_protocol,
    read_body_speeds,
)
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402
from support_contact import ContactSupport  # noqa: E402

# The gate's physics rate. 60 Hz is `World()`'s default and what every report
# under output/ was produced at. isaac_yam runs 120 Hz everywhere (eval, replay,
# datagen, render) and recon's own settle server is already 1/120 -- so at 60 the
# gate is COARSER than the settle it certifies. Raising it is a protocol change
# that renders 411 existing verify_report.json non-comparable, so it is an
# explicit opt-in rather than a silent default flip.
#
# Duration is held at 5 s across rates: steps scale with the rate.
VERIFY_HZ = float(os.environ.get("GRASE_VERIFY_HZ", "60"))
VERIFY_SECONDS = 5.0
STEPS = int(round(VERIFY_HZ * VERIFY_SECONDS))


def _runtime_versions():
    def first(*names):
        for name in names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return None

    return {
        "isaac_sim": first("isaacsim"),
        "physx": first("isaacsim-extscache-physics", "omni-physx"),
    }


def main():
    scene_usd, report_path = sys.argv[1], sys.argv[2]
    drift_tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005  # m
    rotation_tol_deg = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    linear_velocity_tol = (
        float(sys.argv[5]) if len(sys.argv) > 5 else LINEAR_REST_SPEED_M_S
    )
    angular_velocity_tol_deg = (
        float(sys.argv[6]) if len(sys.argv) > 6 else ANGULAR_REST_SPEED_DEG_S
    )

    log_protocol("verify", verify_hz=VERIFY_HZ)
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / VERIFY_HZ)
    apply_scene_tuning(world.stage)  # no-op unless GRASE_PHYSX_TUNED=1
    stage_utils.add_reference_to_stage(scene_usd, "/World/scene")
    stage = world.stage
    xf = UsdGeom.XformCache()
    contact_support = ContactSupport()
    pending_contacts = []
    geometry_cache = {}

    def collider_states(*, refresh_geometry=False):
        states = {}
        cache = UsdGeom.XformCache()
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            path = str(prim.GetPath())
            ancestor = prim
            while (
                ancestor
                and ancestor.IsValid()
                and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI)
            ):
                ancestor = ancestor.GetParent()
            dynamic = bool(ancestor and ancestor.IsValid())
            if refresh_geometry or path not in geometry_cache:
                geometry_cache[path] = hashlib.sha256(
                    repr(
                        (
                            prim.GetTypeName(),
                            [
                                (name, prim.GetAttribute(name).Get())
                                for name in (
                                    "points",
                                    "faceVertexCounts",
                                    "faceVertexIndices",
                                    "size",
                                    "radius",
                                    "height",
                                    "axis",
                                    "physics:approximation",
                                )
                            ],
                        )
                    ).encode()
                ).hexdigest()
            states[path] = {
                "owner": ancestor.GetName() if dynamic else path,
                "actor_path": str(ancestor.GetPath()) if dynamic else path,
                "static": not dynamic,
                "matrix": np.asarray(cache.GetLocalToWorldTransform(prim)).T,
                "geometry": geometry_cache[path],
            }
        return states

    def on_contacts(headers, data):
        try:
            for header in headers:
                pending_contacts.append(
                    {
                        "type": "lost" if int(header.type) == 1 else "contact",
                        "actors": [
                            str(PhysicsSchemaTools.intToSdfPath(p))
                            for p in (header.actor0, header.actor1)
                        ],
                        "colliders": [
                            str(PhysicsSchemaTools.intToSdfPath(p))
                            for p in (header.collider0, header.collider1)
                        ],
                        "contacts": [
                            {
                                "position": list(c.position),
                                "normal": list(c.normal),
                                "impulse": list(c.impulse),
                                "separation": c.separation,
                            }
                            for c in data[
                                header.contact_data_offset : header.contact_data_offset
                                + header.num_contact_data
                            ]
                        ],
                    }
                )
        except Exception as exc:
            # Missing native evidence must not turn an unsupported body into a pass.
            contact_support.error = f"contact report unavailable: {exc}"

    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
    contact_subscription = (
        get_physx_simulation_interface().subscribe_contact_report_events(on_contacts)
    )
    world.reset()

    identity_path = os.path.join(os.path.dirname(scene_usd), "object_identity.json")
    expected = []
    identity_error = None
    if os.path.isfile(identity_path):
        with open(identity_path) as stream:
            expected = sorted(
                record["usd_name"] for record in json.load(stream).get("objects", [])
            )
    else:
        identity_error = f"identity manifest missing: {identity_path}"
    if not expected and identity_error is None:
        identity_error = "identity manifest contains no required bodies"

    def quat(matrix):
        value = matrix.ExtractRotationQuat()
        imag = value.GetImaginary()
        return [
            float(value.GetReal()),
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        ]

    def rotation_delta_deg(a, b):
        dot = min(1.0, abs(sum(x * y for x, y in zip(a, b))))
        return math.degrees(2.0 * math.acos(dot))

    def body_bounds(prim):
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = aligned.GetMin(), aligned.GetMax()
        return [float(v) for v in lo], [float(v) for v in hi]

    def body_states():
        states = {}
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(prim)
                matrix = xf.GetLocalToWorldTransform(prim)
                states[prim.GetName()] = {
                    "position": [float(v) for v in matrix.ExtractTranslation()],
                    "rotation_wxyz": quat(matrix),
                    **read_body_speeds(prim),
                    "kinematic": bool(api.GetKinematicEnabledAttr().Get()),
                    "bounds": body_bounds(prim),
                }
        return states

    before = body_states()
    if not before:
        raise RuntimeError("scene contains no rigid bodies to verify")
    floor_z = (
        min(state["position"][2] for state in before.values()) - 1.0
    )  # anything 1 m under the lowest body fell through

    for step in range(1, STEPS + 1):
        world.step(render=False)
        if pending_contacts and contact_support.error is None:
            try:
                contact_support.update(pending_contacts, collider_states(), step)
            except Exception as exc:
                contact_support.error = f"contact evidence unavailable: {exc}"
        pending_contacts.clear()

    xf.Clear()
    after = body_states()
    final_colliders = collider_states(refresh_geometry=True)
    del contact_subscription

    # Candidate support tops include other rigid bodies and non-rigid geometry.
    supports = []
    rigid_paths = {
        prim.GetPath()
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    for name, state in after.items():
        supports.append((name, state["bounds"]))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        ancestor = prim
        under_rigid = False
        while ancestor and ancestor.IsValid():
            if ancestor.GetPath() in rigid_paths:
                under_rigid = True
                break
            ancestor = ancestor.GetParent()
        if not under_rigid:
            try:
                supports.append((str(prim.GetPath()), body_bounds(prim)))
            except Exception:
                pass

    def support_for(name, bounds):
        lo, _ = bounds
        candidates = []
        for other, (support_lo, support_hi) in supports:
            if other == name:
                continue
            overlap_x = min(bounds[1][0], support_hi[0]) - max(lo[0], support_lo[0])
            overlap_y = min(bounds[1][1], support_hi[1]) - max(lo[1], support_lo[1])
            gap = lo[2] - support_hi[2]
            if overlap_x > 0 and overlap_y > 0 and -0.02 <= gap <= 0.03:
                candidates.append((abs(gap), other, gap))
        if not candidates:
            return None
        _, other, gap = min(candidates)
        return {"name": other, "vertical_gap_m": round(gap, 5)}

    motion_ok = {}
    measurements = {}
    for name, state0 in before.items():
        state1 = after.get(name)
        if state1 is None:
            continue
        drift = (
            sum((a - b) ** 2 for a, b in zip(state1["position"], state0["position"]))
            ** 0.5
        )
        rotation_drift = rotation_delta_deg(
            state0["rotation_wxyz"], state1["rotation_wxyz"]
        )
        fell = state1["position"][2] < floor_z
        measurements[name] = (drift, rotation_drift, fell)
        motion_ok[name] = bool(
            drift < drift_tol
            and rotation_drift < rotation_tol_deg
            and not fell
            and state1["valid"]
            and state1["linear_velocity_m_s"] < linear_velocity_tol
            and state1["angular_velocity_deg_s"] < angular_velocity_tol_deg
        )
    sleeping = {name: state["sleeping"] for name, state in after.items()}
    report, physics_ok, dynamic_ready = {}, True, True
    missing = sorted(set(expected) - set(after))
    unexpected = sorted(set(after) - set(expected)) if expected else []
    for name, state0 in before.items():
        state1 = after.get(name)
        if state1 is None:
            physics_ok = dynamic_ready = False
            continue
        p0, p1 = state0["position"], state1["position"]
        drift, rotation_drift, fell = measurements[name]
        support = support_for(name, state1["bounds"])
        if support is None:
            support = contact_support.support_for(
                name, final_colliders, motion_ok, sleeping, STEPS
            )
        body_physics_ok = motion_ok[name] and support is not None
        body_dynamic_ready = body_physics_ok and not state1["kinematic"]
        report[name] = {
            "drift_m": round(drift, 4),
            "rotation_drift_deg": round(rotation_drift, 3),
            "fell_through": fell,
            "kinematic": state1["kinematic"],
            "linear_velocity_m_s": (
                round(state1["linear_velocity_m_s"], 5) if state1["valid"] else None
            ),
            "angular_velocity_deg_s": (
                round(state1["angular_velocity_deg_s"], 5) if state1["valid"] else None
            ),
            "velocity_valid": state1["valid"],
            "sleeping": state1["sleeping"],
            "velocity_source": state1["velocity_source"],
            "raw_usd_linear_velocity_m_s": state1["raw_usd_linear_velocity_m_s"],
            "raw_usd_angular_velocity_deg_s": state1["raw_usd_angular_velocity_deg_s"],
            **(
                {"velocity_error": state1["velocity_error"]}
                if "velocity_error" in state1
                else {}
            ),
            "support": support,
            "physics_pass": body_physics_ok,
            "dynamic_ready": body_dynamic_ready,
            "before": [round(v, 4) for v in p0],
            "after": [round(v, 4) for v in p1],
        }
        physics_ok &= body_physics_ok
        dynamic_ready &= body_dynamic_ready
    physics_ok &= not missing and not unexpected and identity_error is None
    dynamic_ready &= physics_ok
    kinematic = sorted(name for name, state in after.items() if state["kinematic"])
    simulation_status = (
        "dynamic_verified"
        if dynamic_ready
        else "stable_with_kinematic_fallback"
        if physics_ok and kinematic
        else "unstable"
    )

    with open(report_path, "w") as f:
        json.dump(
            {
                "pass": dynamic_ready,
                "physics_pass": physics_ok,
                "dynamic_ready": dynamic_ready,
                "simulation_status": simulation_status,
                "runtime_versions": _runtime_versions(),
                "steps": STEPS,
                "physics_hz": VERIFY_HZ,
                "thresholds": {
                    "drift_tol_m": drift_tol,
                    "rotation_tol_deg": rotation_tol_deg,
                    "linear_velocity_tol_m_s": linear_velocity_tol,
                    "angular_velocity_tol_deg_s": angular_velocity_tol_deg,
                },
                "missing_bodies": missing,
                "unexpected_bodies": unexpected,
                "identity_error": identity_error,
                "contact_evidence": {
                    "events": contact_support.event_count,
                    "error": contact_support.error,
                },
                "kinematic_bodies": kinematic,
                "bodies": report,
            },
            f,
            indent=1,
        )
    print(
        f"ISAAC_VERIFY_{'OK' if dynamic_ready else 'FAIL'} "
        f"status={simulation_status} bodies={len(report)} "
        f"report={report_path}",
        flush=True,
    )
    return dynamic_ready


# os._exit on BOTH paths, never app.close(): Kit's non-daemon threads keep the
# process alive after the main thread dies (and close() itself can hang headless),
# so a crashed verify would linger holding GPU memory while the spawner blocks on
# its pipe forever.
try:
    _ok = main()
except Exception:
    traceback.print_exc()
    os._exit(1)
os._exit(0 if _ok else 2)
