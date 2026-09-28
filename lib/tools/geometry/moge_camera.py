""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

DEFAULT_SENSOR_MM = 36.0

# Source-view camera basis in GRASE world (camera at origin, proper rotation).
CAM_LOCATION = (0.0, 0.0, 0.0)
CAM_FORWARD = (0.0, -1.0, 0.0)  # view direction (look along -Y)
CAM_UP = (0.0, 0.0, 1.0)  # +Z up
CAM_RIGHT = (-1.0, 0.0, 0.0)  # image right is world -X (forced by det +1)


# --------------------------------------------------------------------------- #
# Coordinate conversion                                                        #
# --------------------------------------------------------------------------- #
def moge_to_world(p, R=None, T=None) -> tuple[float, float, float]:
    """MoGE camera-space point (OpenCV) -> GRASE world: (-x, -z, -y), optionally
    gravity-rotated by ``R`` then canonical-translated by ``T`` (so the chosen root
    surface lies at z=0). The camera, at the MoGE-frame origin, maps to ``T``."""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    v = np.array([-x, -z, -y])
    if R is not None:
        v = np.asarray(R) @ v
    if T is not None:
        v = v + np.asarray(T, dtype=v.dtype)
    return (float(v[0]), float(v[1]), float(v[2]))


def moge_points_to_world(points: np.ndarray, R=None, T=None) -> np.ndarray:
    """Vectorized (..., 3) OpenCV camera points -> GRASE world (-x, -z, -y), optionally
    gravity-rotated by ``R`` then canonical-translated by ``T`` (root surface -> z=0)."""
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    out = np.stack([-x, -z, -y], axis=-1)
    if R is not None:
        out = out @ np.asarray(R, dtype=out.dtype).T
    if T is not None:
        out = out + np.asarray(T, dtype=out.dtype)
    return out


# --------------------------------------------------------------------------- #
# Gravity alignment (from the supporting surface's MoGE normals)               #
# --------------------------------------------------------------------------- #
def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def estimate_gravity_up(
    normals: np.ndarray, mask: np.ndarray, iters: int = 2, thresh_deg: float = 25.0
) -> Optional[np.ndarray]:
    """Robust gravity-up (GRASE world, pre-alignment) from a supporting surface's
    MoGE normal map. ``normals`` (H,W,3) camera-frame unit normals, ``mask`` (H,W)
    bool. Averages the surface normals with iterative outlier rejection, maps to
    world, orients toward +Z. Returns a unit vector or None if too few normals."""
    n = np.asarray(normals, dtype=np.float64)[np.asarray(mask, bool)]
    n = n[np.isfinite(n).all(axis=1)]
    norm = np.linalg.norm(n, axis=1)
    n = n[norm > 1e-6] / norm[norm > 1e-6, None]
    if len(n) < 10:
        return None
    mean = _unit(n.mean(axis=0))
    for _ in range(iters):
        keep = (n @ mean) > math.cos(math.radians(thresh_deg))
        if keep.sum() < 10:
            break
        mean = _unit(n[keep].mean(axis=0))
    g = np.array([-mean[0], -mean[2], -mean[1]])  # camera-normal -> GRASE world dir
    g = _unit(g)
    return -g if g[2] < 0 else g  # orient up (+Z hemisphere)


def alignment_rotation(g) -> np.ndarray:
    """Minimal rotation (3x3, det +1) taking unit ``g`` onto +Z (Rodrigues)."""
    g = _unit(g)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(g, z)
    s = float(np.linalg.norm(v))
    c = float(np.dot(g, z))
    if s < 1e-8:  # already aligned or anti-aligned
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def gravity_roll_deg(g) -> float:
    """Source-camera ROLL (image-plane tilt, degrees) induced by aligning gravity ``g``.

    For the minimal ``alignment_rotation(g)``, the camera's right axis ends up with world-Z
    component exactly ``-g_x``, so the roll is ``arcsin(g_x)`` — ``g_x`` being the left-right
    component of gravity in the camera (pre-alignment) frame. Pitch/azimuth live in g_y, g_z."""
    g = _unit(np.asarray(g, float))
    return math.degrees(math.asin(max(-1.0, min(1.0, float(g[0])))))


# Camera zero-roll prior: a hand-held photo is normally upright, so an induced roll
# under this threshold is treated as surface-normal estimation noise and snapped out.
# Shared by BOTH canonicalization paths (preprocess._compute_canonical_transform and
# preprocess._synthetic_ground) — change it here, not at a call site.
DEROLL_MAX_DEG = 2.5


def trust_measured_camera_roll(mj: dict) -> bool:
    """Should the measured camera roll be KEPT instead of plumbed upright?

    The zero-roll prior above exists for HAND-HELD photos, where a small estimated roll
    is table-normal noise and the photographer was really upright. Neither premise holds
    once the run is given GT depth AND a calibrated intrinsics file: the depth is metric
    rather than a monocular guess, so the surface normal it yields is trustworthy, and
    the camera is a FIXED RIG whose small roll is a real, repeatable mounting offset.
    Snapping that to zero injects the very error it was meant to remove — and does so
    identically on every frame from that rig, so it never averages out.

    Requires BOTH signals (``backend == "gt"`` and an intrinsics FILE): a GT depth run
    that fell back to a FOV-only or MoGE-estimated principal point has no calibrated
    camera to trust, so it keeps the hand-held prior.
    """
    return (mj or {}).get("backend") == "gt" and str(
        (mj or {}).get("intrinsics_source") or ""
    ).startswith("file:")


def deroll_gravity(g, max_deg: float = DEROLL_MAX_DEG):
    """Snap a small camera ROLL out of the estimated gravity-up.

    A hand-held photo is normally upright (zero roll); a small left-right tilt in the
    estimated gravity is an artifact of the table-normal estimate. When the induced roll
    (:func:`gravity_roll_deg`) is under ``max_deg``, zero ``g_x`` — keeping pitch/azimuth
    (g_y, g_z) — so the source camera comes out exactly upright. Returns unit gravity.
    (Threshold raised 2.0 -> 2.5 after 0702_l2_raw3: a -2.37° estimated roll slipped just
    past the snap and gave the whole render a visible constant lean.)

    One pass is exact — no re-canonicalize loop: roll is ``asin(g_x)``, so zeroing
    ``g_x`` makes ``alignment_rotation`` of the result an exactly-upright camera."""
    g = _unit(np.asarray(g, float))
    if abs(gravity_roll_deg(g)) < max_deg:
        g = _unit(np.array([0.0, g[1], g[2]]))
    return g


# --------------------------------------------------------------------------- #
# FOV <-> lens                                                                 #
# --------------------------------------------------------------------------- #
def lens_from_fov(fov_deg: float, sensor_mm: float = DEFAULT_SENSOR_MM) -> float:
    """Blender lens (mm) for a horizontal FOV and sensor width: (s/2)/tan(fov/2)."""
    return (sensor_mm / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def fov_from_lens(lens_mm: float, sensor_mm: float = DEFAULT_SENSOR_MM) -> float:
    """Inverse of :func:`lens_from_fov` (degrees)."""
    return math.degrees(2.0 * math.atan((sensor_mm / 2.0) / lens_mm))


# --------------------------------------------------------------------------- #
# Camera config from MoGE                                                      #
# --------------------------------------------------------------------------- #
def camera_config_from_moge(
    mj: dict[str, Any], sensor_mm: float = DEFAULT_SENSOR_MM, R=None, T=None
) -> dict[str, Any]:
    """Build a Blender camera config from a parsed ``moge.json``.

    Returns lens/sensor/resolution + the pinhole pixel intrinsics (fx_px, fy_px,
    cx, cy) of the source view, ready for :func:`set_blender_camera` and
    :func:`project_world_to_pixel`. ``R`` is the gravity-alignment rotation (the camera
    is re-oriented so the world becomes gravity-up); ``T`` is the canonical translation
    (root surface -> z=0), which places the camera at ``T`` instead of the origin. Both
    leave the render identical (camera + scene co-transform).
    """
    w, h = int(mj["image_width"]), int(mj["image_height"])
    fov_x, fov_y = float(mj["fov_x_deg"]), float(mj["fov_y_deg"])
    lens = lens_from_fov(fov_x, sensor_mm)
    # Horizontal-fit sensor: fx_px = (W/2)/tan(fov_x/2); MoGE square pixels make
    # fy_px land on the same value, so vertical framing matches too.
    fx_px = (w / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    fy_px = (h / 2.0) / math.tan(math.radians(fov_y) / 2.0)
    k_norm = mj.get("intrinsics_norm")
    px_norm = float(k_norm[0][2]) if k_norm else 0.5
    py_norm = float(k_norm[1][2]) if k_norm else 0.5
    cx_px, cy_px = px_norm * w, py_norm * h
    # Blender's shift is expressed in units of the SENSOR-FIT dimension, which is the
    # WIDTH here (sensor_fit=HORIZONTAL) for BOTH axes — hence /w twice, not /h for y.
    # SIGNS ARE MEASURED, NOT DERIVED: shift moves the FRUSTUM, so image content travels
    # the opposite way, and x/y additionally disagree because pixel v grows downward while
    # Blender's +y is up. The two effects cancel on y and compound on x, giving the
    # asymmetric-looking pair below. Verified by rendering a marker placed exactly on the
    # optical axis and confirming it lands on (cx, cy): the naive
    # (+dx, -dy) pair put it at the MIRROR point in BOTH axes (measured 317.5, 165.0 vs
    # the correct 322.1, 194.7 on the robolab top camera). See moge_camera_test.py.
    shift_x = (w / 2.0 - cx_px) / w
    shift_y = (cy_px - h / 2.0) / w
    fwd, up = np.array(CAM_FORWARD), np.array(CAM_UP)
    if R is not None:
        Rm = np.asarray(R, dtype=np.float64)
        fwd, up = Rm @ fwd, Rm @ up
    location = [float(x) for x in T] if T is not None else list(CAM_LOCATION)
    return {
        "lens": lens,
        "sensor_mm": sensor_mm,
        "sensor_fit": "HORIZONTAL",
        "resolution_x": w,
        "resolution_y": h,
        "location": location,
        "forward": [float(x) for x in fwd],
        "up": [float(x) for x in up],
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "fx_px": fx_px,
        "fy_px": fy_px,
        "cx": cx_px,
        "cy": cy_px,
        "shift_x": shift_x,
        "shift_y": shift_y,
    }


def camera_config_from_lookat(
    location,
    look_at,
    lens: float,
    res_x: int,
    res_y: int,
    sensor_mm: float = DEFAULT_SENSOR_MM,
) -> dict[str, Any]:
    """Projection config for an arbitrary LOOK-AT camera (a novel view), mirroring
    :func:`camera_config_from_moge` but built from ``(location, look_at, lens)``. Matches the
    Blender ``to_track_quat('-Z','Y')`` orientation used by the renderer: view dir = look_at -
    location, up resolved toward world +Z. Ready for :func:`project_world_to_pixel`."""
    loc = np.asarray(location, dtype=np.float64)
    fwd = _unit(np.asarray(look_at, dtype=np.float64) - loc)
    z = np.array([0.0, 0.0, 1.0])
    up = z - float(z @ fwd) * fwd  # world +Z orthogonalised to the view dir
    if np.linalg.norm(up) < 1e-6:  # looking straight up/down -> stable up
        y = np.array([0.0, 1.0, 0.0])
        up = y - float(y @ fwd) * fwd
    up = _unit(up)
    w, h = int(res_x), int(res_y)
    fov_x = 2.0 * math.atan((sensor_mm / 2.0) / float(lens))
    fx_px = (w / 2.0) / math.tan(fov_x / 2.0)
    return {
        "lens": float(lens),
        "sensor_mm": sensor_mm,
        "sensor_fit": "HORIZONTAL",
        "resolution_x": w,
        "resolution_y": h,
        "location": [float(x) for x in loc],
        "forward": [float(x) for x in fwd],
        "up": [float(x) for x in up],
        "fx_px": fx_px,
        "fy_px": fx_px,
        # A look-at camera is SYNTHETIC (novel views / orbits): it has no calibrated
        # sensor, so its principal point is the image centre and its shift is zero. Stated
        # explicitly so set_blender_camera clears any shift left by a source-view camera.
        "cx": w / 2.0,
        "cy": h / 2.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
    }


# --------------------------------------------------------------------------- #
# Reprojection (verification + occlusion/placement helper)                     #
# --------------------------------------------------------------------------- #
def project_world_to_pixel(p_world, cfg: dict[str, Any]) -> tuple[float, float, float]:
    """Project a GRASE-world point to source-view pixel (u, v) + depth.

    Pinhole through the source camera basis (origin, -Y forward, +Z up). ``depth``
    is the distance along the view direction; (u, v) only valid when depth > 0.
    """
    loc = np.asarray(cfg.get("location", (0.0, 0.0, 0.0)), dtype=np.float64)
    p = np.array([float(p_world[0]), float(p_world[1]), float(p_world[2])]) - loc
    # Camera basis: view dir = forward, up = up, image-right (OpenCV +X) = -(up x fwd).
    f = _unit(cfg.get("forward", (0.0, -1.0, 0.0)))
    up = _unit(cfg.get("up", (0.0, 0.0, 1.0)))
    right = -np.cross(up, f)
    depth = float(p @ f)
    if depth <= 0:
        return (math.nan, math.nan, depth)
    x_cv = float(p @ right)
    y_cv = -float(p @ up)  # image-down = -up
    u = cfg["fx_px"] * x_cv / depth + cfg["cx"]
    v = cfg["fy_px"] * y_cv / depth + cfg["cy"]
    return (u, v, depth)


def pixel_to_world_ray(
    u: float, v: float, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibrated world-space ray through source pixel ``(u, v)``.

    This is the exact inverse of :func:`project_world_to_pixel`: it uses the full
    camera basis and the calibrated principal point rather than assuming a centred,
    upright look-at camera.  The returned origin and direction are float64 arrays and
    the direction is unit length.
    """

    try:
        fx = float(cfg["fx_px"])
        fy = float(cfg["fy_px"])
        cx = float(cfg["cx"])
        cy = float(cfg["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("camera config has invalid pixel intrinsics") from exc
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
        raise ValueError("camera config intrinsics must be finite with positive focal lengths")

    origin = np.asarray(cfg.get("location", CAM_LOCATION), dtype=np.float64)
    forward = _unit(cfg.get("forward", CAM_FORWARD))
    up = _unit(cfg.get("up", CAM_UP))
    if origin.shape != (3,) or forward.shape != (3,) or up.shape != (3,):
        raise ValueError("camera location/forward/up must be three-vectors")
    if not (
        np.isfinite(origin).all()
        and np.isfinite(forward).all()
        and np.isfinite(up).all()
    ):
        raise ValueError("camera location/forward/up must be finite")
    # Match project_world_to_pixel exactly: OpenCV image-right is -(up x forward),
    # while image-down is -up. Re-orthogonalise up so a slightly rounded JSON basis
    # cannot introduce a systematic inverse-projection error.
    right = _unit(-np.cross(up, forward))
    up = _unit(np.cross(right, forward))
    x_cv = (float(u) - cx) / fx
    y_cv = (float(v) - cy) / fy
    direction = _unit(forward + x_cv * right - y_cv * up)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-12:
        raise ValueError("pixel produces a degenerate camera ray")
    return origin, direction


def intersect_pixel_ray_with_plane(
    u: float,
    v: float,
    cfg: dict[str, Any],
    *,
    plane_point=(0.0, 0.0, 0.0),
    plane_normal=(0.0, 0.0, 1.0),
    minimum_incidence: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Intersect a calibrated pixel ray with a world plane.

    Returns ``(point, ray_distance, incidence)`` where ``incidence`` is the absolute
    dot product between the unit ray and unit plane normal.  A non-forward or grazing
    intersection raises :class:`ValueError`; callers can use the returned incidence as
    a stability diagnostic without reimplementing camera conventions.
    """

    origin, direction = pixel_to_world_ray(u, v, cfg)
    point = np.asarray(plane_point, dtype=np.float64)
    normal = _unit(plane_normal)
    if point.shape != (3,) or normal.shape != (3,):
        raise ValueError("plane point/normal must be three-vectors")
    if not np.isfinite(point).all() or not np.isfinite(normal).all():
        raise ValueError("plane point/normal must be finite")
    denom = float(direction @ normal)
    incidence = abs(denom)
    if incidence < max(float(minimum_incidence), 0.0):
        raise ValueError("pixel ray is too nearly parallel to the plane")
    distance = float((point - origin) @ normal / denom)
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("pixel ray does not meet the plane in front of the camera")
    hit = origin + distance * direction
    return hit, distance, incidence


# --------------------------------------------------------------------------- #
# Blender applier (runs inside Blender)                                         #
# --------------------------------------------------------------------------- #
def set_blender_camera(cfg: dict[str, Any], camera=None, scene=None):
    """Apply a camera config to a Blender camera + render settings (lazy bpy)."""
    import bpy  # noqa: PLC0415
    from mathutils import Matrix, Vector  # noqa: PLC0415

    scene = scene or bpy.context.scene
    if camera is None:
        camera = scene.camera
    cam = camera.data
    cam.type = "PERSP"
    cam.sensor_fit = cfg["sensor_fit"]
    cam.sensor_width = cfg["sensor_mm"]
    cam.lens = max(float(cfg["lens"]), 1.0)
    # Principal-point offset. Always assigned (default 0.0) rather than only when non-zero:
    # this applier is also pointed at REUSED camera datablocks, and a stale shift left from
    # a calibrated source view would silently mis-frame an otherwise centred camera.
    cam.shift_x = float(cfg.get("shift_x", 0.0))
    cam.shift_y = float(cfg.get("shift_y", 0.0))
    camera.location = Vector(cfg["location"])
    # Build the orientation from BOTH forward and up so a gravity-rotated camera is
    # set exactly (the local frame: forward = -Z, up = +Y, right = +X).
    f = Vector(cfg["forward"]).normalized()
    u = Vector(cfg["up"]).normalized()
    zl = -f
    xl = u.cross(zl).normalized()
    yl = zl.cross(xl)
    camera.rotation_euler = Matrix((xl, yl, zl)).transposed().to_euler()
    scene.render.resolution_x = cfg["resolution_x"]
    scene.render.resolution_y = cfg["resolution_y"]
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    return camera
