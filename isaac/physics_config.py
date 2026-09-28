"""One place for GRASE's PhysX constants — material defaults and scene tuning.

Before this module the material heuristics were declared three times (``DENSITY``
in ``isaac_settle_server``, ``isaac_add_physics`` and ``isaac_auto_stabilize``;
friction twice), held together only by comments reading "same as the settle
stage". Nothing failed if they drifted. Structure borrowed from isaac_yam's
``core/physics.py``, whose docstring records what two constants that had to agree
actually cost.

Pure stdlib at import time — ``pxr`` is imported inside ``apply_scene_tuning`` —
so the repo venv, the Isaac venv and the tests can all read the same numbers.
"""

from __future__ import annotations

import math
import os
from typing import Any

# --------------------------------------------------------------- materials --
#: kg/m^3. Light plastic/wood heuristic, used when the VLM estimate has no mass
#: for an object. Precedence is overrides > VLM > this.
DENSITY = 250.0
#: Default friction when neither an override nor a VLM material lookup applies.
OBJ_FRICTION = 0.6
SURF_FRICTION = 0.8
#: PhysX floats cm-scale objects at its own defaults; these are GRASE's.
CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0

# Shared speed limits; the exporter's explicit CLI overrides remain independent.
LINEAR_REST_SPEED_M_S = 0.01
ANGULAR_REST_SPEED_DEG_S = 1.0
EXPORT_VELOCITY_REST_POLICY = "export_velocity_v1"


def _finite_velocity_norm(value) -> float | None:
    """Absent or malformed velocity telemetry cannot establish rest."""
    if value is None:
        return None
    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError):
        return None
    if len(vector) != 3 or not all(math.isfinite(v) for v in vector):
        return None
    magnitude = math.hypot(*vector)
    return magnitude if math.isfinite(magnitude) else None


def rigid_body_sleeping(prim) -> bool:
    """Query live PhysX; USD velocity attributes retain pre-sleep values."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, UsdUtils

    stage_id = UsdUtils.StageCache.Get().GetId(prim.GetStage()).ToLongInt()
    body_id = PhysicsSchemaTools.sdfPathToInt(prim.GetPath())
    return bool(get_physx_simulation_interface().is_sleeping(stage_id, body_id))


def read_body_speeds(prim) -> dict:
    """Read m/s and degrees/s, correcting only verified sleeping bodies to zero."""
    from pxr import UsdPhysics

    api = UsdPhysics.RigidBodyAPI(prim)
    raw_linear = _finite_velocity_norm(api.GetVelocityAttr().Get())
    raw_angular = _finite_velocity_norm(api.GetAngularVelocityAttr().Get())
    error = None
    try:
        sleeping = rigid_body_sleeping(prim)
    except Exception as exc:  # Native telemetry failure must never imply rest.
        sleeping = None
        error = str(exc)[:300]
    if sleeping is True:
        linear, angular, source = 0.0, 0.0, "physx_sleep_zero"
    elif sleeping is False:
        linear, angular, source = raw_linear, raw_angular, "usd_awake"
    else:
        linear, angular, source = None, None, "unavailable"
    return {
        "linear_velocity_m_s": linear,
        "angular_velocity_deg_s": angular,
        "valid": linear is not None and angular is not None,
        "sleeping": sleeping,
        "velocity_source": source,
        "raw_usd_linear_velocity_m_s": raw_linear,
        "raw_usd_angular_velocity_deg_s": raw_angular,
        **({"velocity_error": error} if error is not None else {}),
    }


PHYSX_SCENE_TUNING: dict[str, Any] = {
    "solver_type": 1,  # TGS
    "min_position_iteration_count": 32,
    "min_velocity_iteration_count": 1,
    "bounce_threshold_velocity": 0.2,
    "gpu_collision_stack_size": 2**30,
    "gpu_heap_capacity": 2**30,
    "gpu_temp_buffer_capacity": 2**30,
}

TUNING_ENV_VAR = "GRASE_PHYSX_TUNED"


def tuning_enabled() -> bool:
    return os.environ.get(TUNING_ENV_VAR, "0") == "1"


def apply_scene_tuning(stage, *, force: bool = False) -> dict[str, Any]:
    ""
    if not (force or tuning_enabled()):
        return {}

    from pxr import PhysxSchema, UsdPhysics  # noqa: PLC0415 - Isaac-venv only

    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        raise RuntimeError(
            "no UsdPhysics.Scene on the stage; call apply_scene_tuning after "
            "World() has authored /physicsScene"
        )
    api = PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    api.CreateSolverTypeAttr("TGS" if PHYSX_SCENE_TUNING["solver_type"] == 1 else "PGS")
    api.CreateMinPositionIterationCountAttr(
        PHYSX_SCENE_TUNING["min_position_iteration_count"]
    )
    api.CreateMinVelocityIterationCountAttr(
        PHYSX_SCENE_TUNING["min_velocity_iteration_count"]
    )
    api.CreateBounceThresholdAttr(PHYSX_SCENE_TUNING["bounce_threshold_velocity"])
    api.CreateGpuCollisionStackSizeAttr(PHYSX_SCENE_TUNING["gpu_collision_stack_size"])
    api.CreateGpuHeapCapacityAttr(PHYSX_SCENE_TUNING["gpu_heap_capacity"])
    api.CreateGpuTempBufferCapacityAttr(PHYSX_SCENE_TUNING["gpu_temp_buffer_capacity"])
    return dict(PHYSX_SCENE_TUNING)


# ------------------------------------------------------- protocol reporting --
#: What a scored GRASE physics run is supposed to be. A run that departs from it
#: is REPORTED, not refused -- probing at other settings is legitimate; comparing
#: the number to a published one afterwards is not. Borrowed from isaac_yam's
#: ``PhysicsTuning.off_protocol``, whose reasoning applies verbatim: the flags
#: carry values, and a value that merely differs looks deliberate.
SCORED_PROTOCOL: dict[str, Any] = {
    "settle_hz": 120.0,
    "verify_hz": 60.0,
    "physx_tuned": False,
    "density": DENSITY,
    "obj_friction": OBJ_FRICTION,
    "surf_friction": SURF_FRICTION,
    "contact_offset": CONTACT_OFFSET,
    "rest_offset": REST_OFFSET,
}


def observed_protocol(*, settle_hz: float | None = None,
                      verify_hz: float | None = None) -> dict[str, Any]:
    """The protocol a run is actually built with, for logging and comparison."""
    obs = {
        "physx_tuned": tuning_enabled(),
        "density": DENSITY,
        "obj_friction": OBJ_FRICTION,
        "surf_friction": SURF_FRICTION,
        "contact_offset": CONTACT_OFFSET,
        "rest_offset": REST_OFFSET,
    }
    if settle_hz is not None:
        obs["settle_hz"] = float(settle_hz)
    if verify_hz is not None:
        obs["verify_hz"] = float(verify_hz)
    if tuning_enabled():
        obs["physx"] = dict(PHYSX_SCENE_TUNING)
    return obs


def off_protocol(observed: dict[str, Any]) -> list[str]:
    """How a run departs from :data:`SCORED_PROTOCOL`, phrase by phrase.

    Empty means on-protocol. Only keys the observation actually carries are
    checked, so a settle-only run is not reported as diverging from a verify
    rate it never read.
    """
    off: list[str] = []
    for key, want in SCORED_PROTOCOL.items():
        if key not in observed:
            continue
        got = observed[key]
        if got != want:
            off.append(f"{key}={got!r} (scored: {want!r})")
    return off


def log_protocol(tag: str, **rates: float) -> list[str]:
    """Print the run's protocol and any departures. Returns the departures."""
    obs = observed_protocol(**rates)
    off = off_protocol(obs)
    rate_str = " ".join(f"{k}={v:g}" for k, v in sorted(rates.items()) if v is not None)
    print(f"[{tag}] protocol: {rate_str} physx_tuned={obs['physx_tuned']}", flush=True)
    if off:
        print(f"[{tag}] OFF-PROTOCOL: {'; '.join(off)}", flush=True)
    return off
