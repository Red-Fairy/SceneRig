"""MoGE preprocessing for the initializer: geometry + per-object metric placement.

Runs once per scene, BEFORE the agent pipeline:

  1. depth estimate   -> ``<out>/moge/`` (MoGE-2 by default; user-provided
     metric depth and camera intrinsics are also supported)
  2. agentic masking  -> ``<out>/masks/masks.json`` (per-instance masks + points)
  3. placement table  -> ``<out>/placement.json`` (per object: world center + metric
     size + screen bbox), the constraints the initializer seeds objects from
  4. camera lock      -> writes the MoGE source-view camera into the shared .blend

The placement builder is pure (unit-tested); ``apply_camera_to_blend`` and
``preprocess_scene`` shell out to MoGE/SAM3/Molmo/Blender.
"""

from __future__ import annotations

import contextvars
import functools
import json
import os
import subprocess
import sys
import tempfile
import threading
import time as _time
import weakref
from pathlib import Path
from typing import Any, Optional

import numpy as np

from lib.tools.geometry import moge_camera as mc
from lib.tools.geometry import scene_graph as sgm
from lib.tools.geometry.agentic_mask import _binarize, _dilate1
from lib.tools.geometry.surface_relations import floor_under_furniture_support

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


_ACTIVE_PSEUDO_GT_PROCESS: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "grase_active_preprocess_pseudo_gt", default=None
)
_ACTIVE_MESH_PRESPAWN: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "grase_active_preprocess_mesh_prespawn", default=None
)


def _prepare_pseudo_gt_regeneration(
    pseudo_gt_dir: Path, rejection_reason: str, nvs_backend: str = "sharp_gpt"
) -> Optional[Path]:
    ""

    scene_dir = pseudo_gt_dir.parent
    archive: Optional[Path] = None
    if os.path.lexists(pseudo_gt_dir):
        archive = scene_dir / "pseudo_gt_reuse_rejected"
        suffix = 2
        while os.path.lexists(archive):
            archive = scene_dir / f"pseudo_gt_reuse_rejected_{suffix}"
            suffix += 1
        pseudo_gt_dir.rename(archive)
    pseudo_gt_dir.mkdir(parents=True, exist_ok=False)
    audit = {
        "requested_reuse": True,
        "action": "regenerate",
        "nvs_backend": nvs_backend,
        "reason": rejection_reason,
        "rejected_bundle": str(archive) if archive is not None else None,
    }
    audit_path = scene_dir / "pseudo_gt_reuse_fallback.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(temporary, audit_path)
    return archive


def _terminate_and_reap_process(proc: Any) -> None:
    """Best-effort terminate/reap without masking the caller's real failure."""

    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        else:
            # ``poll`` reaps a completed ``Popen`` child. Custom process adapters
            # may not, so a zero-time wait keeps this helper explicit.
            proc.wait(timeout=0)
    except Exception:  # noqa: BLE001 - preserve the caller's original exception
        try:
            proc.kill()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - process may already be gone
            pass


def _pseudo_gt_failure_boundary(func):
    """Terminate active pseudo-GT work before a preprocess exception escapes.

    A weakref finalizer is only a leak backstop: an exception traceback retains the
    function frame and can therefore retain its sentinel indefinitely. This wrapper
    is the actual unwind boundary and is context-local so parallel callers cannot
    terminate one another's subprocesses.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        token = _ACTIVE_PSEUDO_GT_PROCESS.set(None)
        try:
            return func(*args, **kwargs)
        except BaseException:
            _terminate_and_reap_process(_ACTIVE_PSEUDO_GT_PROCESS.get())
            raise
        finally:
            _ACTIVE_PSEUDO_GT_PROCESS.reset(token)

    return wrapped


def _cleanup_mesh_prespawn(state: Any, *, join_timeout: float = 30.0) -> None:
    """Cancel, join, and close a prespawned mesh server without masking failures.

    ``Sam3dServer`` construction happens in a background thread.  The shared state and
    lock close both races at an exceptional unwind: a server already published here is
    closed by the caller, while one that finishes after the bounded join observes
    ``cancel_requested`` and closes itself before it can become available to the main
    pipeline.
    """

    if not isinstance(state, dict):
        return
    lock = state.get("lock")
    try:
        if lock is not None:
            with lock:
                state["cancel_requested"] = True
        else:
            state["cancel_requested"] = True
    except Exception:  # noqa: BLE001 - cleanup must preserve the primary failure
        pass

    thread = state.get("thread")
    try:
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout)
    except Exception:  # noqa: BLE001 - an unstarted/failed thread has no server to reap
        pass

    server = None
    try:
        if lock is not None:
            with lock:
                server = state.get("srv")
                state["srv"] = None
        else:
            server = state.get("srv")
            state["srv"] = None
    except Exception:  # noqa: BLE001 - malformed state is best-effort cleanup only
        server = state.get("srv")
    if server is not None:
        try:
            server.close()
        except Exception:  # noqa: BLE001 - preserve the caller's original exception
            pass


def _mesh_prespawn_failure_boundary(func):
    """Reap a successfully started SAM3D prespawn on every exceptional unwind."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        token = _ACTIVE_MESH_PRESPAWN.set(None)
        try:
            return func(*args, **kwargs)
        except BaseException:
            _cleanup_mesh_prespawn(_ACTIVE_MESH_PRESPAWN.get())
            raise
        finally:
            _ACTIVE_MESH_PRESPAWN.reset(token)

    return wrapped


def _resize_mask(b: np.ndarray, h: int, w: int) -> np.ndarray:
    if b.shape == (h, w):
        return b
    from PIL import Image

    return np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((w, h))) > 127


def _screen_bbox(b: np.ndarray) -> list[float]:
    """Normalized [u_min, v_min, u_max, v_max] of a binary mask."""
    h, w = b.shape
    ys, xs = np.where(b)
    if xs.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(xs.min()) / (w - 1),
        float(ys.min()) / (h - 1),
        float(xs.max()) / (w - 1),
        float(ys.max()) / (h - 1),
    ]


def _depth_consistency(
    mask, world_points, cam_pos, win=5, kmad=4.0, floor=0.02, min_keep=0.6
):
    """Prune mask pixels whose RADIAL depth (distance from camera) deviates from the local
    windowed median — LingBot completion spikes and silhouette depth-bleed. Depth-domain, so
    a smooth thin object (marker) keeps its tips, unlike a 3D density/isolation filter which
    deletes exactly the sparse extreme points that define a thin object's size. Tuned on the
    redetect object set (plush/monitor/keyboard/box/mug/tray/placemat/book/marker/scissors/
    laptop/stand): median 0% clean shrink, thin marker ~5%, injected depth-spike Δ < 1.5%.
    The ``floor`` (2 cm) is the operative threshold at tabletop depths; ``kmad`` guards deeper
    scenes. Falls back to the input mask if it would remove more than ``1 - min_keep``."""
    from scipy.ndimage import median_filter

    finite = np.isfinite(world_points).all(axis=2)
    both = mask & finite
    if int(both.sum()) < 20:
        return mask
    cam = np.asarray(cam_pos, dtype=np.float64)
    d = np.linalg.norm(world_points - cam, axis=2)
    d = np.where(finite, d, float(np.median(d[finite])))
    dev = np.abs(d - median_filter(d, size=win))
    din = dev[both]
    thr = max(floor, kmad * float(np.median(np.abs(din - np.median(din)))) * 1.4826)
    kept = mask & (dev <= thr)
    if int(kept.sum()) < min_keep * int(mask.sum()):  # safety: never gut the object
        return mask
    return kept


def object_world_box(
    mask: np.ndarray,
    world_points: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
    cam_pos=None,
    depth_filter: bool = False,
    depth_filter_min_keep: float = 0.6,
):
    """Robust world-space center + size of a masked object from MoGE points.

    Returns (center[x,y,z], size[sx,sy,sz], depth) using ``lo``/``hi`` percentiles
    to shrug off mask-edge / depth outliers. ``depth`` is distance along -Y (the
    view direction). Returns (None, None, None) if the mask covers no valid point.

    ``cam_pos`` (world coords) enables the SLIVER-DISTRUST trim. For a THIN mask,
    boundary pixels whose depth belongs to the BACKGROUND are a large fraction of
    the points — and MoGE depth is SMOOTH across edges, so the bleed forms a
    continuous tail that survives the percentile band, inflating the extent along
    the viewing ray and dragging the median center backward (0709 wendy1 marker#1:
    a pen-thin marker measured 0.24m deep, toppled, and shipped as a giant pristine
    mesh 3 runs in a row). Thinness is measured on the mask itself: the fraction of
    pixels within 3px of the boundary (bleed magnitude is proportional to it).
    Masks >= 25% edge get the trim: points farther than max(3*MAD, 2cm) from the
    median longitudinal coordinate (along the mean viewing ray) are cut. Bulky
    masks (a coffee machine at ~6% edge, a keyboard at ~22%) are never touched, so
    a genuine far side cannot be lost. Validated on 0709_eval3: marker#1
    0.24 -> 0.10m, abc1 spoon 0.15 -> 0.10m, bread-roll tray bleed cut; machine /
    keyboard / boxes / books byte-identical.
    """
    pts = _object_world_points(
        mask,
        world_points,
        cam_pos=cam_pos,
        depth_filter=depth_filter,
        depth_filter_min_keep=depth_filter_min_keep,
    )
    if pts is None:
        return None, None, None
    center = np.median(pts, axis=0)
    p_lo = np.percentile(pts, lo, axis=0)
    p_hi = np.percentile(pts, hi, axis=0)
    size = p_hi - p_lo
    depth = float(-center[1])  # view direction is -Y
    return [float(x) for x in center], [float(x) for x in size], depth


def _object_world_points(
    mask, world_points, cam_pos=None, depth_filter=False, depth_filter_min_keep=0.6
):
    ""
    h, w = world_points.shape[:2]
    b = _resize_mask(_binarize(mask), h, w)
    if depth_filter and cam_pos is not None:
        b = _depth_consistency(b, world_points, cam_pos, min_keep=depth_filter_min_keep)
    sel = b & np.isfinite(world_points).all(axis=2)
    pts = world_points[sel]
    if pts.size == 0:
        return None
    if cam_pos is not None and len(pts) >= 8:
        er = b.copy()
        for _ in range(3):
            er = ~_dilate1(~er)
        edge_frac = 1.0 - er.sum() / max(b.sum(), 1)
        if edge_frac >= 0.25:
            cam = np.asarray(cam_pos, dtype=np.float64)
            ray = pts.mean(axis=0) - cam
            n = np.linalg.norm(ray)
            if n > 1e-6:
                ray = ray / n
                t = (pts - cam) @ ray
                mad = float(np.median(np.abs(t - np.median(t))))
                keep = np.abs(t - np.median(t)) <= max(3.0 * mad, 0.02)
                if keep.sum() >= 4:
                    pts = pts[keep]
    return pts


def oriented_world_box(
    mask: np.ndarray,
    world_points: np.ndarray,
    lo: float = 2.0,
    hi: float = 98.0,
    cam_pos=None,
    min_aniso: float = 1.3,
    depth_filter: bool = False,
    depth_filter_min_keep: float = 0.6,
):
    """Ground-plane ORIENTED box of a masked object from MoGE world points.

    Returns ``(size[major, minor, height], yaw_deg)`` -- a box oriented in the xy
    (gravity) plane with Z kept vertical. Unlike :func:`object_world_box` (a world-axis
    box that inflates the depth of a YAWED object, e.g. a Kleenex box at ~50deg reads as a
    near-square footprint), this fits the footprint orientation, so its diagonal reflects
    the object's true metric size. Used for scale (``place_glb``), not orientation.

    Orientation is PCA on the xy footprint; extents are the ``lo``/``hi`` percentile band
    along the principal axes (robust to mask-edge / depth outliers, matching the
    axis-aligned box). If the footprint is near-isotropic (eigenvalue ratio < ``min_aniso``)
    the yaw is ill-conditioned, so it degenerates to axis-aligned (yaw 0) -- harmless
    because a near-square footprint's size barely changes with orientation. Returns
    ``(None, None)`` if the mask covers no valid point.
    """
    pts = _object_world_points(
        mask,
        world_points,
        cam_pos=cam_pos,
        depth_filter=depth_filter,
        depth_filter_min_keep=depth_filter_min_keep,
    )
    if pts is None:
        return None, None
    xy = pts[:, :2]
    c = np.median(xy, axis=0)
    evals, evecs = np.linalg.eigh(np.cov((xy - c).T))
    if evals[0] <= 1e-12 or evals[1] / evals[0] < min_aniso:
        axes = np.eye(2)  # near-square: yaw unreliable -> axis-aligned
    else:
        axes = np.stack([evecs[:, 1], evecs[:, 0]])  # rows: [major, minor]
    proj = (xy - c) @ axes.T
    ext = np.percentile(proj, hi, axis=0) - np.percentile(proj, lo, axis=0)
    z = float(np.percentile(pts[:, 2], hi) - np.percentile(pts[:, 2], lo))
    yaw = float(np.degrees(np.arctan2(axes[0, 1], axes[0, 0])))
    return [float(abs(ext[0])), float(abs(ext[1])), z], yaw


def build_placement_table(
    masks_json_path: str,
    points_npy: str,
    moge_json_path: Optional[str] = None,
    R=None,
    T=None,
    same_size_categories: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Per-object metric placement constraints for the initializer.

    For every masked instance: world ``center`` + ``size`` (metres) from its MoGE
    points (converted to GRASE world, gravity-aligned by ``R`` and canonical-translated by
    ``T`` so the root surface is at z=0), ``depth``, and the normalized ``screen_bbox``.

    ``same_size_categories`` is retained only for call compatibility.  Same-size intent is
    resolved once, after final mask pruning, into ``masks.json.same_size_resolution``;
    placement copies the exact group identity stamped on each retained instance.  The
    compatibility ``same_size`` flag deliberately remains false here: it becomes true only
    after ``generate_meshes`` successfully normalizes enough usable meshes in the group.
    """
    del same_size_categories  # policy resolution belongs to the final-mask stage
    masks = json.load(open(masks_json_path))
    world = mc.moge_points_to_world(np.load(points_npy), R, T)
    h, w = world.shape[:2]
    # Camera position in this frame: the camera sits at the pre-canonical origin,
    # which R maps to 0 and T translates to T (identity when un-canonicalized).
    cam = np.asarray(T, dtype=np.float64) if T is not None else np.zeros(3)
    table: list[dict[str, Any]] = []
    for r in masks.get("instances", []):
        if r.get("kind") == "root_surface":
            continue  # surfaces are scene-graph roots (primitives), not placed objects
        mp = r.get("mask_path")
        if not mp or not os.path.exists(mp):
            continue
        mask = np.load(mp)
        screen_mask = mask  # for screen_bbox: must stay in the ORIGINAL image frame
        obj_world = world
        depth_filter = True
        dfilter_min_keep = 0.9
        rd = r.get("redetect")
        if rd and os.path.exists(rd.get("mask_path", "")):
            # Redetected instance (generative resegment): the full-object mask from the
            # edited image, with LingBot-refined points where available (the edit hole
            # has no valid reference depth) — so center/size cover e.g. a monitor's
            # revealed stand and base.
            mask = np.load(rd["mask_path"])
            if rd.get("points_npy") and os.path.exists(rd["points_npy"]):
                obj_world = mc.moge_points_to_world(np.load(rd["points_npy"]), R, T)
                # LingBot-completed points carry small depth noise — the prune (already
                # ON for every object above) matters doubly here, at its original
                # tuning (min_keep 0.6, scripts/tune_robust_box.py).
                dfilter_min_keep = 0.6
            # Border completion re-frames the mask (pad-shifted, extends past the frame),
            # so its pixels don't map to original screen coords — keep the original mask
            # for the screen bbox. A same-frame removal redetect uses the full mask.
            if not rd.get("border"):
                screen_mask = mask
        center, size, depth = object_world_box(
            mask,
            obj_world,
            cam_pos=cam,
            depth_filter=depth_filter,
            depth_filter_min_keep=dfilter_min_keep,
        )
        if center is None:
            continue
        # Oriented footprint box: rotation-robust metric size for scaling (place_glb).
        # `size` stays the world-axis box (unchanged semantics for every other consumer);
        # `obb_size`/`obb_yaw` are additive. yaw is stored for diagnostics only -- object
        # orientation still comes from SAM3D's pose.
        obb_size, obb_yaw = oriented_world_box(
            mask,
            obj_world,
            cam_pos=cam,
            depth_filter=depth_filter,
            depth_filter_min_keep=dfilter_min_keep,
        )
        entry = {
            "category": r["category"],
            "instance": r["instance"],
            "description": r.get("description"),
            # VLM state-aware rollability (lying cylinder rolls; standing one
            # doesn't) — the composition physics gates judge rollable objects by
            # displacement instead of tilt. None on pre-field runs (server falls
            # back to an orientation-aware extents heuristic).
            "rollable": r.get("rollable"),
            # Compatibility flag consumed by physics/composition.  A registry candidate
            # is not a size lock yet; mesh normalization activates it atomically later.
            "same_size": False,
            "center": [round(c, 4) for c in center],
            "size": [round(s, 4) for s in size],
            "obb_size": [round(s, 4) for s in obb_size] if obb_size else None,
            "obb_yaw": round(obb_yaw, 2) if obb_yaw is not None else None,
            "depth": round(depth, 4),
            "screen_bbox": [
                round(v, 4)
                for v in _screen_bbox(_resize_mask(_binarize(screen_mask), h, w))
            ],
            "mask_path": mp,
        }
        # Preserve the resolver's stable group identity without renaming the canonical
        # category/id.  These fields are candidates until mesh normalization succeeds.
        for field in (
            "same_size_group_id",
            "same_size_group_source",
            "same_size_group_minimum_members",
            "same_size_group_display_name",
            "same_size_group_status",
        ):
            if r.get(field) is not None:
                entry[field] = r[field]
        if rd:
            entry["redetect"] = rd
        table.append(entry)
    return table


_SAME_SIZE_GROUP_FIELDS = (
    "same_size_group_id",
    "same_size_group_source",
    "same_size_group_minimum_members",
    "same_size_group_display_name",
    "same_size_group_status",
)


def _same_size_object_id(record: dict[str, Any]) -> str:
    """Return the canonical instance id without normalizing/renaming its category."""
    return f"{record.get('category')}#{record.get('instance')}"


def _retained_mask_ids(records: list[dict[str, Any]]) -> set[str]:
    """Ids with a present, nonempty retained segmentation mask (eligibility gate 1)."""
    retained: set[str] = set()
    for record in records:
        if record.get("kind") == "root_surface":
            continue
        mask_path = record.get("mask_path")
        if not mask_path or not os.path.isfile(mask_path):
            continue
        try:
            if np.asarray(np.load(mask_path)).any():
                retained.add(_same_size_object_id(record))
        except (OSError, ValueError):
            continue
    return retained


def _stamp_same_size_groups(
    records: list[dict[str, Any]],
    registry: dict[str, Any],
    *,
    activated_group_ids: Optional[set[str]] = None,
) -> None:
    """Project active registry membership onto masks/placement records.

    The registry is authoritative; stale proposer booleans and stale group fields are
    removed on every pass.  ``activated_group_ids`` is supplied only after mesh scaling:
    until then membership is a candidate and the compatibility flag stays false.
    """
    from lib.tools.geometry.same_size import active_same_size_members

    member_groups = active_same_size_members(registry)
    groups = {
        str(group.get("group_id")): group
        for group in registry.get("groups", [])
        if isinstance(group, dict) and group.get("group_id")
    }
    # Automatic groups the auditor confirmed but the run did not lock stay visible on the
    # rows (status ``detected_unlocked``); they are never in ``activated_group_ids``.
    for group_id, group in groups.items():
        if group.get("status") == "detected_unlocked":
            for member_id in group.get("members", []):
                member_groups.setdefault(str(member_id), group_id)
    activated_group_ids = activated_group_ids or set()
    # Final registries use the audit status ``applied``.  Projection is explicit here so
    # it does not depend on whether the module's eligibility helper recognizes that
    # post-normalization status.
    for group_id in activated_group_ids:
        group = groups.get(str(group_id))
        if group:
            for member_id in group.get("active_members", []):
                member_groups[str(member_id)] = group_id
    for record in records:
        record["same_size"] = False
        record.pop("same_size_applied", None)
        if record.pop("same_size_scale_locked", False):
            # Never unlock an unrelated asset/registration lock.
            if not record.get("registration_locked"):
                record["scale_locked"] = False
        for field in _SAME_SIZE_GROUP_FIELDS:
            record.pop(field, None)

        object_id = _same_size_object_id(record)
        group_id = member_groups.get(object_id)
        group = groups.get(str(group_id)) if group_id else None
        if not group:
            continue
        record.update(
            {
                "same_size_group_id": group_id,
                "same_size_group_source": group.get("source"),
                "same_size_group_minimum_members": group.get("minimum_members"),
                "same_size_group_display_name": group.get("display_name"),
                "same_size_group_status": group.get("status"),
            }
        )
        if group_id in activated_group_ids:
            record["same_size"] = True
            record["same_size_applied"] = True
            record["scale_locked"] = True
            record["same_size_scale_locked"] = True


def _write_same_size_registry(
    masks_json_path: str,
    payload: dict[str, Any],
    registry: dict[str, Any],
    *,
    activated_group_ids: Optional[set[str]] = None,
) -> None:
    """Persist the registry and its derived per-instance compatibility projection."""
    from lib.tools.geometry.same_size import REGISTRY_KEY

    payload[REGISTRY_KEY] = registry
    for proposed in payload.get("objects", []):
        if not isinstance(proposed, dict):
            continue
        proposed.pop("same_size", None)
        for proposed_instance in proposed.get("instances", []):
            if isinstance(proposed_instance, dict):
                proposed_instance.pop("same_size", None)
    _stamp_same_size_groups(
        payload.get("instances", []),
        registry,
        activated_group_ids=activated_group_ids,
    )
    with open(masks_json_path, "w") as f:
        json.dump(payload, f, indent=2)


def _log_ineligible_same_size_groups(registry: dict[str, Any], stage: str) -> None:
    """Make threshold losses visible without turning automatic hints into failures."""
    latest_validation = next(
        (
            validation
            for validation in reversed(registry.get("validations", []))
            if isinstance(validation, dict) and validation.get("stage") == stage
        ),
        {},
    )
    latest_counts = {
        str(row.get("group_id")): int(row.get("surviving_member_count") or 0)
        for row in latest_validation.get("groups", [])
        if isinstance(row, dict) and row.get("group_id")
    }
    for group in registry.get("groups", []):
        minimum = int(group.get("minimum_members") or 0)
        active_count = latest_counts.get(
            str(group.get("group_id")), len(set(group.get("active_members", [])))
        )
        if active_count >= minimum:
            continue
        level = (
            "WARNING" if group.get("source") in {"user_explicit", "manual"} else "info"
        )
        print(
            f"[same_size] {level}: {group.get('group_id')} is inactive at {stage}: "
            f"{active_count}/{minimum} eligible members"
        )


def _resolve_same_size_after_mask_pruning(
    masks_json_path: str,
    image_path: str,
    requested_terms: Optional[list[str]],
    model: str,
) -> dict[str, Any]:
    """Resolve manual or automatic groups over the FINAL retained mask inventory.

    A nonempty user list is an allow-list (manual min=2); an empty list runs the Opus
    visual auditor (automatic min=3).  The initial proposer is intentionally absent from
    this path.  Manual ambiguity/unmatched terms are persisted and then raised visibly.
    """
    from lib.tools.geometry.agentic_mask import _make_vlm
    from lib.tools.geometry.same_size import (
        build_retained_inventory,
        invalid_inventory_registry,
        resolve_same_size_registry,
        revalidate_same_size_registry,
    )

    payload = json.load(open(masks_json_path))
    terms = [str(term).strip() for term in (requested_terms or []) if str(term).strip()]
    # The prompts were calibrated on Opus-5, but the calls follow the run's configured
    # preprocess model so non-Anthropic lanes need no Anthropic credentials.
    policy_model = model
    vlm_holder: dict[str, Any] = {}

    def vlm(system: str, parts: list, max_tokens: int = 700) -> str:
        # Deterministically resolved manual requests (and automatic scenes with no
        # count-eligible candidates) should not even need to construct a model client.
        if "call" not in vlm_holder:
            vlm_holder["call"] = _make_vlm(policy_model, effort="medium")
        return vlm_holder["call"](system, parts, max_tokens=max_tokens)

    try:
        inventory = build_retained_inventory(
            payload.get("objects", []), payload.get("instances", [])
        )
        registry = resolve_same_size_registry(
            inventory,
            user_terms=terms,
            image_path=image_path,
            manual_resolver_vlm=vlm if terms else None,
            automatic_auditor_vlm=None if terms else vlm,
            model=policy_model,
        )
    except ValueError as exc:
        # A malformed retained inventory (e.g. two categories colliding only under the
        # strict normalizer, or a '#' in a category) must not kill a finished ~1 h
        # preprocess for an OPTIONAL automatic audit. Explicit user terms stay fatal:
        # silently dropping a manual request would violate user intent.
        if terms:
            raise
        print(f"[same_size] WARNING: automatic resolution skipped: {exc}")
        registry = invalid_inventory_registry(terms, policy_model, str(exc))
    retained_ids = _retained_mask_ids(payload.get("instances", []))
    registry = revalidate_same_size_registry(
        registry, retained_ids, stage="retained_masks"
    )
    _log_ineligible_same_size_groups(registry, "retained_masks")
    _write_same_size_registry(masks_json_path, payload, registry)

    if terms:
        incomplete = [
            resolution
            for resolution in registry.get("term_resolutions", [])
            if resolution.get("verdict") != "matched"
        ]
        if incomplete or registry.get("errors"):
            details = "; ".join(
                f"{item.get('term')!r}: {item.get('verdict')}"
                + (f" ({item.get('reason')})" if item.get("reason") else "")
                for item in incomplete
            )
            raise RuntimeError(
                "same-size user request could not be resolved completely; "
                f"see masks.json.same_size_resolution: {details or registry.get('errors')}"
            )
    return registry


def _revalidate_same_size_placement(
    masks_json_path: str, table: list[dict[str, Any]]
) -> dict[str, Any]:
    """Drop groups that fall below their manual/automatic threshold at placement."""
    from lib.tools.geometry.same_size import REGISTRY_KEY, revalidate_same_size_registry

    payload = json.load(open(masks_json_path))
    registry = payload.get(REGISTRY_KEY)
    if not isinstance(registry, dict):
        raise RuntimeError(
            "same-size resolution is missing before placement; rerun full preprocessing"
        )
    registry = revalidate_same_size_registry(
        registry,
        {_same_size_object_id(record) for record in table},
        stage="placement",
    )
    _log_ineligible_same_size_groups(registry, "placement")
    _stamp_same_size_groups(table, registry)
    _write_same_size_registry(masks_json_path, payload, registry)
    return registry


# --------------------------------------------------------------------------- #
# Blender camera lock                                                          #
# --------------------------------------------------------------------------- #
_CAM_SCRIPT = r"""
import json, sys
import bpy
sys.path.insert(0, {repo!r})
from lib.tools.geometry import moge_camera as mc
mj = json.load(open({moge!r}))
_g = mj.get("gravity") or {{}}
R = _g.get("R")  # gravity-alignment rotation
T = _g.get("T")  # canonical translation (root surface -> z=0; camera moves to T)
bpy.ops.wm.open_mainfile(filepath={blend!r})
cam = bpy.context.scene.camera
if cam is None:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
mc.set_blender_camera(mc.camera_config_from_moge(mj, R=R, T=T), camera=cam)
bpy.ops.wm.save_as_mainfile(filepath={out!r})
print("CAMERA_LOCKED", round(cam.data.lens, 2))
"""


def apply_camera_to_blend(
    blend_path: str,
    moge_json_path: str,
    blender_cmd: str,
    out_blend: Optional[str] = None,
) -> None:
    """Write the MoGE source-view camera into ``blend_path`` (in place by default)."""
    out_blend = out_blend or blend_path
    script = _CAM_SCRIPT.format(
        repo=REPO_ROOT, moge=moge_json_path, blend=blend_path, out=out_blend
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    try:
        subprocess.run(
            [blender_cmd, "--background", "--factory-startup", "--python", script_path],
            check=True,
            cwd=REPO_ROOT,
        )
    finally:
        os.unlink(script_path)


# --------------------------------------------------------------------------- #
# Scene graph (VLM support prior + MoGE geometric resolution)                  #
# --------------------------------------------------------------------------- #
def _mask_world_points(mask_path: str, world: np.ndarray) -> np.ndarray:
    h, w = world.shape[:2]
    m = _resize_mask(_binarize(np.load(mask_path)), h, w)
    return world[m & np.isfinite(world).all(axis=2)]


def _mask_frac(mask_path: str) -> float:
    """Fraction of the REFERENCE FRAME a mask covers, at the mask's own resolution —
    the same number ``coverage_report`` calls ``photo frac`` (its ``skip_photo_below``
    decides "photo mask too small to compare"). Deliberately NOT measured on the
    depth-resized mask so the two stay comparable."""
    return float((np.load(mask_path) > 0).mean())


def _node_id(r: dict[str, Any]) -> str:
    return f"{r['category']}#{r['instance']}"


def _map_support(
    obj: dict, surfaces: list[dict], objects: list[dict], scores: dict[str, float]
) -> Optional[str]:
    ""
    sup = obj.get("support")
    measured_surfaces = [s for s in surfaces if "_inliers" in s]
    dominant_pool = measured_surfaces or surfaces
    dominant = (
        max(dominant_pool, key=lambda s: s.get("_inliers", 0))["id"]
        if dominant_pool
        else None
    )
    if not sup:
        return dominant
    sup = str(sup).strip().lower()
    by_id = {
        n["id"].lower(): n["id"] for n in surfaces + objects if n["id"] != obj["id"]
    }
    if sup in by_id:  # exact instance id given by the VLM
        return by_id[sup]
    cands = [
        n
        for n in surfaces + objects  # legacy: bare category -> best instance
        if n["category"].lower() == sup and n["id"] != obj["id"]
    ]
    if cands:
        return max(cands, key=lambda n: scores.get(n["id"], 0.0))["id"]
    return dominant


_SUPPORT_SYSTEM = (
    "Assess only the VISUAL EVIDENCE for direct physical support in this marked "
    "photo. Red region A is the object and blue region B is the candidate support. "
    "Return ON only when the image gives positive evidence that an upward-facing "
    "part of B lies under A's lower contact and carries some of A's downward weight. "
    "Support need not be exclusive: A may also lean against or touch another object. "
    "Return NOT_ON only when the image gives positive evidence that B is merely "
    "beside, in front of, behind, or laterally touching A, or that A continues to "
    "another lower support. Otherwise return UNCERTAIN. In particular, projected "
    "overlap or an object emerging above a foreground container/occluder is not "
    "positive support evidence when A's lower contact is hidden; do not guess from "
    "semantic plausibility. Colors identify masks, not physical edges. Reply with "
    'JSON only: {"verdict":"on"}, {"verdict":"not_on"}, or '
    '{"verdict":"uncertain"}.'
)

_SUPPORT_GEOMETRY_SYSTEM = (
    "The photo-only support referee found the contact visually ambiguous. Resolve "
    "whether blue region B directly supports red region A using the marked photo and "
    "the supplied approximate depth heights. B supports A when it bears A's downward "
    "weight from below; support may be partial or non-exclusive. Heights can be noisy "
    "by a couple of centimeters, so use them only to distinguish materially different "
    'vertical arrangements. Reply with JSON only: {"on": true} or {"on": false}.'
)


def _ask_support(
    vlm,
    image_path: str,
    masks_dir: Path,
    o_node: dict,
    c_node: dict,
    default_on: bool,
    evidence: str = "",
) -> dict[str, Any]:
    """Visual-first support referee with a geometry-only ambiguity fallback.

    The primary call receives ONE marked crop, instance ids/categories only, and no
    proposer description or depth prose.  That independence matters: the descriptions
    are the claims being adjudicated, and universally injected monocular heights made an
    obvious bagel-on-plate relation read false.  A ternary visual verdict lets a fully
    hidden contact (the dolls statue behind its holder) say ``uncertain`` instead of
    guessing.  Only then does a second call receive ``evidence``.

    The returned record always has a final boolean ``on`` plus audit fields.  On any
    crop/IO/parse failure, ``default_on`` preserves the old asymmetric no-change rule:
    True keeps a declared parent (T2), False declines a geometric nomination (T1).
    """
    from lib.tools.geometry.agentic_mask import (
        _binarize,
        _img_part,
        _pair_overlay,
        _union_crop,
        require_json_contract,
        slugify,
        vlm_json,
    )

    visual_verdict = "uncertain"
    geometry_fallback_used = False
    try:
        ma = _binarize(np.load(o_node["mask_path"]))
        mb = _binarize(np.load(c_node["mask_path"]))
        tag = f"{slugify(o_node['id'])}_{slugify(c_node['id'])}"
        ov = str(masks_dir / f"_sup_{tag}_ov.png")
        marked = str(masks_dir / f"_sup_{tag}_marked.png")
        _pair_overlay(image_path, ma, mb, ov)
        _union_crop(ov, ma, mb, marked)
        a_id = o_node.get("id") or o_node["category"]
        b_id = c_node.get("id") or c_node["category"]
        txt = f"Red A={a_id}. Blue B={b_id}. What does this image establish?"
        visual = vlm_json(
            vlm,
            _SUPPORT_SYSTEM,
            [
                {"type": "text", "text": txt},
                _img_part(marked),
            ],
            label=f"support-visual {a_id}/{b_id}",
            default={"verdict": "uncertain"},
            validate=lambda reply: require_json_contract(
                reply,
                enum_fields={"verdict": {"on", "not_on", "uncertain"}},
            ),
        )
        visual_verdict = visual["verdict"]
        if visual_verdict != "uncertain":
            return {
                "on": visual_verdict == "on",
                "visual_verdict": visual_verdict,
                "geometry_fallback_used": False,
                "geometry_answer": None,
            }

        if not evidence:
            return {
                "on": default_on,
                "visual_verdict": "uncertain",
                "geometry_fallback_used": False,
                "geometry_answer": None,
            }
        geometry_fallback_used = True
        geometry = vlm_json(
            vlm,
            _SUPPORT_GEOMETRY_SYSTEM,
            [
                {
                    "type": "text",
                    "text": (
                        f"Red A={a_id}. Blue B={b_id}. Approximate sensor evidence: "
                        f"{evidence} Does B rest directly beneath and support A?"
                    ),
                },
                _img_part(marked),
            ],
            label=f"support-geometry {a_id}/{b_id}",
            default={"on": default_on},
            validate=lambda reply: require_json_contract(reply, bool_fields=("on",)),
        )
        geometry_answer = bool(geometry["on"])
        return {
            "on": geometry_answer,
            "visual_verdict": "uncertain",
            "geometry_fallback_used": geometry_fallback_used,
            "geometry_answer": geometry_answer,
        }
    except Exception as e:  # noqa: BLE001 - the referee must never kill the graph
        print(f"[scene_graph] support referee failed for {o_node.get('id')} ({e})")
        return {
            "on": default_on,
            "visual_verdict": visual_verdict,
            "geometry_fallback_used": geometry_fallback_used,
            "geometry_answer": None,
        }


def _authoritative_scene_inventory(
    masks: dict[str, Any],
) -> list[tuple[dict[str, Any], bool]]:
    """Return the final upstream inventory in stable order.

    The boolean marks records from ``unmasked_root_surfaces``.  A duplicate id is
    always an upstream artifact error, including a collision between the masked and
    deliberately-unmasked sections; silently coalescing the records would make it
    impossible for downstream code to know which record was authoritative.
    """
    inventory: list[tuple[dict[str, Any], bool]] = []
    seen: dict[str, str] = {}
    for section, unmasked in (
        ("instances", False),
        ("unmasked_root_surfaces", True),
    ):
        rows = masks.get(section, [])
        if not isinstance(rows, list):
            raise ValueError(f"scene inventory field {section!r} must be a list")
        for index, rec in enumerate(rows):
            if not isinstance(rec, dict):
                raise ValueError(f"{section}[{index}] must be an object")
            category = rec.get("category")
            if not isinstance(category, str) or not category.strip():
                raise ValueError(f"{section}[{index}] has no non-empty category")
            instance = rec.get("instance")
            if (
                not isinstance(instance, int)
                or isinstance(instance, bool)
                or instance < 0
            ):
                raise ValueError(
                    f"{section}[{index}] ({category!r}) has invalid instance "
                    f"{instance!r}; expected integer >= 0"
                )
            nid = _node_id(rec)
            kind = rec.get("kind", "root_surface" if unmasked else None)
            if kind not in {"root_surface", "object"}:
                raise ValueError(
                    f"retained scene instance {nid!r} has invalid kind {kind!r}"
                )
            if unmasked and kind != "root_surface":
                raise ValueError(
                    f"unmasked_root_surfaces contains non-root instance {nid!r}"
                )
            if nid in seen:
                raise ValueError(
                    f"duplicate retained scene instance id {nid!r}: "
                    f"{seen[nid]} and {section}[{index}]"
                )
            seen[nid] = f"{section}[{index}]"
            inventory.append((rec, unmasked))
    return inventory


def _semantic_inventory_node(rec: dict[str, Any], unmasked: bool) -> dict[str, Any]:
    """Create an identity-only graph node without consulting depth or a mask."""
    return {
        "id": _node_id(rec),
        "category": rec["category"],
        "kind": "root_surface" if unmasked else rec["kind"],
        "support": rec.get("support"),
        "rollable": rec.get("rollable"),
        "description": rec.get("description"),
        "mask_path": None if unmasked else rec.get("mask_path"),
    }


def _validate_and_apply_graph_topology(
    nodes: dict[str, dict[str, Any]], parents: dict[str, Optional[str]]
) -> list[str]:
    """Apply an exact support map and enforce the one-hierarchy contract.

    Inventory identity has already been frozen.  This function may only add the
    derived ``parent``/``children`` fields and mirror an object's exact parent into
    ``support``; it cannot turn an object into a graph root or discard a node.
    """
    object_ids = [nid for nid, n in nodes.items() if n.get("kind") == "object"]
    root_ids = [nid for nid, n in nodes.items() if n.get("kind") == "root_surface"]
    expected, supplied = set(object_ids), set(parents)
    if expected != supplied:
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        raise ValueError(
            "scene graph parent map does not cover the retained object inventory: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for oid in object_ids:
        pid = parents[oid]
        if not isinstance(pid, str) or not pid:
            raise ValueError(
                f"retained object {oid!r} has no exact support id after upstream "
                "support finalization"
            )
        if pid not in nodes:
            raise ValueError(
                f"retained object {oid!r} has dangling support {pid!r}; "
                "support must name a retained scene instance"
            )
        if pid == oid:
            raise ValueError(f"retained object {oid!r} cannot support itself")

    # Validate before mutating nodes, and report the actual cycle rather than silently
    # severing an edge and inventing a parentless ordinary-object root.
    done: set[str] = set()
    for oid in object_ids:
        if oid in done:
            continue
        order: list[str] = []
        position: dict[str, int] = {}
        cur = oid
        while cur in expected and cur not in done:
            if cur in position:
                cycle = order[position[cur] :] + [cur]
                raise ValueError("retained object support cycle: " + " -> ".join(cycle))
            position[cur] = len(order)
            order.append(cur)
            cur = parents[cur]
        done.update(order)

    for node in nodes.values():
        node.pop("parent", None)
        node["children"] = []
    for oid in object_ids:
        pid = parents[oid]
        nodes[oid]["parent"] = pid
        nodes[oid]["support"] = pid
        nodes[pid]["children"].append(oid)

    for oid in object_ids:
        node = nodes[oid]
        if node.get("parent") != node.get("support"):
            raise AssertionError(f"parent/support mismatch for retained object {oid}")
    inverse = {
        nid: sorted(oid for oid in object_ids if nodes[oid].get("parent") == nid)
        for nid in nodes
    }
    for nid, node in nodes.items():
        if sorted(node["children"]) != inverse[nid]:
            raise AssertionError(f"children are not the inverse of parent for {nid}")
    return root_ids


def _construct_scene_graph(
    masks_json_path: str,
    points_npy: str,
    up=(0, 0, 1),
    eps: float = 0.04,
    decide=None,
    R=None,
    T=None,
    vlm=None,
    *,
    resolve_supports: bool,
    persist_supports: bool,
) -> dict[str, Any]:
    """Shared measurement machinery for support finalization and graph building.

    ``resolve_supports=True`` is used only by :func:`finalize_object_supports`, the
    upstream semantic normalization/referee pass.  ``False`` is the read-only graph
    builder path and consumes exact finalized supports without reinterpretation.
    """
    masks = json.load(open(masks_json_path))
    inventory = _authoritative_scene_inventory(masks)
    # Identity is unconditional and precedes every mask/depth operation.  Measurement
    # failures below only leave optional geometry fields absent.
    nodes = {
        _node_id(rec): _semantic_inventory_node(rec, unmasked)
        for rec, unmasked in inventory
    }
    surface_nodes = [n for n in nodes.values() if n["kind"] == "root_surface"]
    object_nodes = [n for n in nodes.values() if n["kind"] == "object"]
    surfaces: list[dict[str, Any]] = []  # measured roots only
    objects: list[dict[str, Any]] = []  # measured objects only
    world = mc.moge_points_to_world(np.load(points_npy), R, T)

    for r in masks.get("instances", []):
        node = nodes[_node_id(r)]
        mp = r.get("mask_path")
        if not mp or not os.path.exists(mp):
            continue
        try:
            pts = _mask_world_points(mp, world)
        except (OSError, ValueError, EOFError) as e:
            print(
                f"[scene_graph] no geometry for {node['id']} "
                f"(mask could not be read: {e})"
            )
            continue
        if len(pts) < 30:
            continue
        if r["kind"] == "root_surface":
            plane = sgm.fit_plane(pts, up=up)
            if plane is None:
                continue
            ext = sgm.footprint_bbox(pts, up)
            pn, pkind = sgm.plumb_plane(plane["normal"])
            n_out = pn if pkind != "ambiguous" and pn else plane["normal"]
            node["plane"] = {
                "normal": [float(x) for x in n_out],
                "normal_raw": [float(x) for x in plane["normal"]],
                "plumb": pkind,
                # keep n.x + d = 0 through the measured centroid for the snapped normal
                "d": float(-np.dot(np.asarray(n_out, float), plane["centroid"])),
                "inlier_frac": plane["inlier_frac"],
                "mask_frac": _mask_frac(mp),
            }
            node["extent"] = [[float(x) for x in ext[0]], [float(x) for x in ext[1]]]
            node["world_center"] = [float(x) for x in plane["centroid"]]
            node["span"] = [float(ext[1][0] - ext[0][0]), float(ext[1][1] - ext[0][1])]
            node["_plane"], node["_extent"] = plane, ext
            node["_inliers"] = plane["inliers"]
            surfaces.append(node)
        else:
            bt = sgm.object_base_top(pts, up=up)
            if bt is None:
                continue
            node["_bt"] = bt
            node["world_center"] = [float(x) for x in bt["centroid"]]
            objects.append(node)

    masks_dir = Path(masks_json_path).parent
    image_path = masks_dir.parent / "input.png"
    adjudicate = (
        resolve_supports
        and vlm is not None
        and os.environ.get("GRASE_SUPPORT_ADJUDICATE", "1") != "0"
        and image_path.exists()
    )
    budget = [4]  # per-scene cap across BOTH triggers; overflow logs, never calls
    adjudications: list[dict[str, Any]] = []

    def _cand_bt(n: dict) -> Optional[dict]:
        bt = n.get("_bt")
        if bt is None:
            return None
        return {
            "base_h": bt["base_h"],
            "top_h": bt["top_h"],
            "top_bbox": bt["top_bbox"],
        }

    _upv = np.asarray(up, float)

    def _evidence(o: dict, c: dict) -> str:
        """Height facts for the referee — the crop often HIDES the contact region
        (the dolls statue's base is fully occluded by the pen holder it merely
        stands behind, and it visually reads as sitting IN it), while geometry
        holds exactly the deciding fact. Heights are relative to the main surface
        (z=0 in the canonical frame); A's base uses the MEDIAN of its base band
        (robust to the mask bleed these cases carry)."""
        try:
            a = float(np.median(o["_bt"]["base_pts"] @ _upv))
            cb, ct = c["_bt"]["base_h"], c["_bt"]["top_h"]
            return (
                f"Depth heights above the main surface: A's base ~{a * 100:.0f}cm; "
                f"B spans {cb * 100:.0f}-{ct * 100:.0f}cm. If A rested on/in B, "
                f"A's base would be near {ct * 100:.0f}cm (on top) or within "
                f"{cb * 100:.0f}-{ct * 100:.0f}cm (inside)."
            )
        except Exception:  # noqa: BLE001 - evidence is optional
            return ""

    def _t1_decide(obj: str, pv, pg: str, sc: dict) -> str:
        """Resolve a missed stack when geometry strongly contradicts the declared parent."""
        if not adjudicate or budget[0] <= 0:
            return pv
        pgn = nodes.get(pg)
        if pgn is None or pgn.get("kind") == "root_surface" or not pgn.get("mask_path"):
            return pv
        if sc.get(pg, 0.0) < 0.2 or (sc.get(pv, 0.0) if pv else 0.0) > 0.05:
            return pv
        o, pvn = nodes.get(obj), nodes.get(pv) if pv else None
        if o is None:
            return pv
        cb = _cand_bt(pvn) if pvn is not None else None
        if cb is not None and sgm.in_container(o["_bt"], cb, eps):
            return pv  # declared parent is a container holding the object
        budget[0] -= 1
        decision = _ask_support(
            vlm,
            str(image_path),
            masks_dir,
            o,
            pgn,
            default_on=False,
            evidence=_evidence(o, pgn),
        )
        on = decision["on"]
        chosen = pg if on else pv
        adjudications.append(
            {
                "obj": obj,
                "trigger": "t1",
                "declared": pv,
                "candidate": pg,
                "score_pv": round(sc.get(pv, 0.0) if pv else 0.0, 3),
                "score_pg": round(sc.get(pg, 0.0), 3),
                "answer": bool(on),
                "applied_parent": chosen,
                "visual_verdict": decision["visual_verdict"],
                "geometry_fallback_used": decision["geometry_fallback_used"],
                "geometry_answer": decision["geometry_answer"],
            }
        )
        print(
            f"[scene_graph] support referee (t1): {obj} on {pg}? -> "
            f"{'yes -> re-parented' if on else f'no, kept {pv}'}"
        )
        return chosen

    scores: dict[str, dict[str, float]] = {}
    vlm_parents: dict[str, Optional[str]] = {}
    if resolve_supports:
        measured_object_ids = {n["id"] for n in objects}
        for o in object_nodes:
            sc: dict[str, float] = {}
            if o["id"] in measured_object_ids:
                sc.update(
                    {
                        s["id"]: sgm.support_score(
                            o["_bt"],
                            {
                                "kind": "surface",
                                "plane": s["_plane"],
                                "extent": s["_extent"],
                            },
                            eps,
                        )
                        for s in surfaces
                    }
                )
                for c in objects:
                    if c["id"] != o["id"]:
                        sc[c["id"]] = sgm.support_score(
                            o["_bt"],
                            {
                                "kind": "object",
                                "top_h": c["_bt"]["top_h"],
                                "top_bbox": c["_bt"]["top_bbox"],
                                "base_h": c["_bt"]["base_h"],
                            },
                            eps,
                        )
            scores[o["id"]] = sc
            vlm_parents[o["id"]] = _map_support(o, surface_nodes, object_nodes, sc)

        decide_cb = (
            decide if decide is not None else (_t1_decide if adjudicate else None)
        )
        res = sgm.resolve_parents(vlm_parents, scores, decide_cb)
    else:
        # The builder consumes exact upstream facts.  Do not use depth to repair,
        # nominate, demote, or case-normalize a semantic parent here.
        res = {
            "parents": {o["id"]: o.get("support") for o in object_nodes},
            "flags": [],
        }

    if adjudicate:
        # T2 — implausible declared OBJECT parent (dolls bust statue#0: declared on
        # the pen holder it merely stands BEHIND; score(pv)=0.000 while its base sat
        # 0.0cm off the table plane — and no flag ever fired because the best score
        # was tau_weak-muted). Needs no geometric winner: only proof the declared
        # relation is impossible plus a concrete contact alternative to fall to.
        # Runs BEFORE the _bt/_plane pop (primitives must be alive) and before the
        # parent->support mirror, so the one-hierarchy invariant holds for free.
        def _t2_alternative(o: dict, pv: str, sc: dict) -> Optional[str]:
            best = None  # key (0=surface first, -score): argmax score, surfaces on tie
            o_bt = o["_bt"]
            for s in surfaces:
                # Root alternatives use the MEDIAN base distance and no coverage
                # clause: the dolls statue's mask bleed put 89% of its base band
                # off-table and 10% of it 15cm BELOW the plane, so every
                # low-percentile/bbox statistic fails on exactly the corrupted
                # masks T2 exists for. The median survives <50% corruption, and a
                # height-matched ROOT is the conservative catch-all — the same
                # place _map_support falls for unmappable supports.
                n_, d_ = s["_plane"]["normal"], s["_plane"]["d"]
                gap = float(np.median(o_bt["base_pts"] @ n_ + d_))
                if abs(gap) > eps:
                    continue
                key = (0, -sc.get(s["id"], 0.0))
                if best is None or key < best[0]:
                    best = (key, s["id"])
            for c in objects:
                if c["id"] in (o["id"], pv):
                    continue
                if abs(o_bt["base_h"] - c["_bt"]["top_h"]) > eps:
                    continue
                if (
                    sgm._bbox_overlap_frac(o_bt["foot_bbox"], c["_bt"]["top_bbox"])
                    < 0.3
                ):
                    continue
                key = (1, -sc.get(c["id"], 0.0))
                if best is None or key < best[0]:
                    best = (key, c["id"])
            return best[1] if best else None

        done = {a["obj"] for a in adjudications}
        t2: list[tuple[float, str, str, str]] = []
        for o in objects:
            oid = o["id"]
            if oid in done:
                continue
            pv = res["parents"].get(oid)
            pvn = nodes.get(pv) if pv else None
            if pvn is None or pvn.get("kind") == "root_surface":
                continue  # root-parented is the default safe state — T1's business
            sc = scores.get(oid, {})
            if sc.get(pv, 0.0) > 0.05:
                continue  # declared parent is geometrically possible: VLM stands
            cb = _cand_bt(pvn)
            if cb is None or not pvn.get("mask_path"):
                continue  # unmeasurable parent: skip, conservatively keep the VLM
            if sgm.in_container(o["_bt"], cb, eps):
                continue  # inside a container: the declared parent is right
            alt = _t2_alternative(o, pv, sc)
            if alt is None:
                continue  # nothing plausible to fall to -> unactionable
            t2.append((sc.get(pv, 0.0), oid, pv, alt))
        for _, oid, pv, alt in sorted(t2, key=lambda t: t[0]):  # most impossible first
            if budget[0] <= 0:
                print(f"[scene_graph] support budget exhausted; skipping {oid}")
                break
            budget[0] -= 1
            decision = _ask_support(
                vlm,
                str(image_path),
                masks_dir,
                nodes[oid],
                nodes[pv],
                default_on=True,
                evidence=_evidence(nodes[oid], nodes[pv]),
            )
            on = decision["on"]
            if not on:
                res["parents"][oid] = alt
            adjudications.append(
                {
                    "obj": oid,
                    "trigger": "t2",
                    "declared": pv,
                    "alt": alt,
                    "score_pv": round(scores[oid].get(pv, 0.0), 3),
                    "answer": bool(on),
                    "applied_parent": res["parents"][oid],
                    "visual_verdict": decision["visual_verdict"],
                    "geometry_fallback_used": decision["geometry_fallback_used"],
                    "geometry_answer": decision["geometry_answer"],
                }
            )
            print(
                f"[scene_graph] support referee (t2): {oid} on {pv}? -> "
                f"{'yes, kept' if on else 'no -> ' + alt}"
            )

    # Validate the complete retained topology before any upstream write-back.  This
    # catches a dangling/null/cyclic resolver result without leaving masks.json in a
    # partially finalized state.
    roots = _validate_and_apply_graph_topology(nodes, res["parents"])

    if persist_supports:
        if not resolve_supports:
            raise ValueError("persist_supports requires resolve_supports=True")
        adjudicated_ids = {
            a["obj"] for a in adjudications if a["applied_parent"] != a["declared"]
        }
        changed: dict[str, Optional[str]] = {}
        for r in masks.get("instances", []):
            oid = _node_id(r)
            if r.get("kind") != "object" or oid not in res["parents"]:
                continue
            pid = res["parents"][oid]
            if r.get("support") != pid:
                changed[oid] = pid
                r["support"] = pid
            if oid in adjudicated_ids:
                r["support_adjudicated"] = True
        if changed or adjudicated_ids:
            with open(masks_json_path, "w") as f:
                json.dump(masks, f, indent=2)
            print(f"[preprocess] finalized object supports -> masks.json: {changed}")

    for n in nodes.values():
        for k in ("_bt", "_plane", "_extent", "_inliers"):
            n.pop(k, None)
    return {
        "up": list(up),
        "roots": roots,
        "flags": res["flags"],
        "support_adjudications": adjudications,
        # Kept as an empty compatibility field for consumers that display this old
        # diagnostic.  No retained root can be "unfittable" out of the graph now.
        "roots_unfittable": [],
        "nodes": list(nodes.values()),
    }


def finalize_object_supports(
    masks_json_path: str,
    points_npy: str,
    up=(0, 0, 1),
    eps: float = 0.04,
    decide=None,
    R=None,
    T=None,
    vlm=None,
) -> dict[str, Any]:
    """Normalize/adjudicate object supports and persist exact ids upstream.

    This is the only pass allowed to change support claims.  The orchestrator first runs
    it before semantic-main pruning (with the optional referee), then runs a deterministic
    no-VLM pass after every inventory mutation and before :func:`build_scene_graph`.
    Geometry and the optional support referee operate only on measurable nodes;
    unmeasurable objects still participate and retain/normalize their semantic parent.
    """
    graph = _construct_scene_graph(
        masks_json_path,
        points_npy,
        up=up,
        eps=eps,
        decide=decide,
        R=R,
        T=T,
        vlm=vlm,
        resolve_supports=True,
        persist_supports=True,
    )
    return {
        "flags": graph["flags"],
        "support_adjudications": graph["support_adjudications"],
    }


def build_scene_graph(
    masks_json_path: str,
    points_npy: str,
    up=(0, 0, 1),
    eps: float = 0.04,
    R=None,
    T=None,
) -> dict[str, Any]:
    """Losslessly build a support graph from the finalized upstream inventory.

    Every retained ``instances`` or ``unmasked_root_surfaces`` record becomes exactly
    one node before any mask/depth operation.  Existing metric geometry is attached
    opportunistically; missing geometry never changes identity.  Supports are consumed
    as exact upstream ids and topology errors fail explicitly.  The function never
    writes ``masks.json`` and never calls a VLM.
    """
    return _construct_scene_graph(
        masks_json_path,
        points_npy,
        up=up,
        eps=eps,
        R=R,
        T=T,
        resolve_supports=False,
        persist_supports=False,
    )


def _render_canonical_root(
    image_path: str, cands: list[dict], chosen_id: str, out_png: Optional[str] = None
) -> Optional[str]:
    ""
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    arr = np.asarray(im).astype(np.float32)
    cols = [(255, 80, 80), (80, 200, 120), (90, 150, 250), (240, 200, 60)]
    centers = []
    for n, c in enumerate(cands):
        m = c["mask"]
        if m.shape != (h, w):
            m = _resize_mask(m, h, w)
        arr[m] = 0.5 * arr[m] + 0.5 * np.array(cols[n % len(cols)], np.float32)
        ys, xs = np.where(m)
        centers.append((int(xs.mean()), int(ys.mean())) if xs.size else (5, 5))
    img = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    draw = ImageDraw.Draw(img)
    for c, (cx, cy) in zip(cands, centers):
        tag = f"{c['id']}  <- GROUND" if c["id"] == chosen_id else c["id"]
        draw.text((cx, cy), tag, fill=(255, 255, 255))
    out = out_png or str(Path(image_path).with_name("canonical_root.png"))
    img.save(out)
    return out


def main_support_id(masks: dict) -> Optional[str]:
    ""
    insts = {
        f"{r['category']}#{r['instance']}".lower(): r
        for r in masks.get("instances", [])
    }
    roots = {k for k, r in insts.items() if r.get("kind") == "root_surface"}
    votes: dict[str, int] = {}
    for k, r in insts.items():
        if k in roots:
            continue
        cur, seen = (r.get("support") or "").strip().lower(), set()
        while cur and cur not in roots and cur in insts and cur not in seen:
            seen.add(cur)
            cur = (insts[cur].get("support") or "").strip().lower()
        if cur in roots:
            votes[cur] = votes.get(cur, 0) + 1
    if votes:
        return max(votes.items(), key=lambda kv: kv[1])[0]
    # No-vote scenes are unusual, but their identity must not depend on hash seed.
    return min(roots) if roots else None


def stamp_main_support(graph: dict, masks_payload: dict) -> str:
    """Persist exactly one support-chain winner into the current graph contract."""
    main_root = main_support_id(masks_payload)
    graph_nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
    matches = [n for n in graph_nodes if n.get("id") == main_root]
    if main_root is None:
        raise RuntimeError("support-chain vote produced no main support")
    if len(matches) != 1 or matches[0].get("kind") != "root_surface":
        # Name the unmeasurable roots when there are any: the usual cause is that the
        # vote winner was dropped from the graph by build_scene_graph (a room scene's
        # floor with too few finite depth points), which is otherwise invisible here.
        unfit = graph.get("roots_unfittable") or []
        raise RuntimeError(
            "support-chain main support must name exactly one root_surface node: "
            f"{main_root!r}"
            + (
                f" (unmeasurable root(s) absent from the graph: {'; '.join(unfit)})"
                if unfit
                else ""
            )
        )
    graph["main_support_id"] = main_root
    sf = scene_form(masks_payload)
    for node in graph_nodes:
        node.pop("main_support", None)
        if node.get("id") == main_root:
            node["main_support"] = True
            node["form"] = "table" if sf == "closeup:table" else "tabletop"
    return main_root


def scene_form(masks: dict) -> str:
    ""
    r = masks.get("routing") or {}
    f = str(r.get("form") or "").strip().lower()
    if f not in ("room", "closeup:table", "closeup:tabletop"):
        if str(r.get("mode") or "").strip().lower() == "room":
            f = "room"
        else:
            for o in masks.get("objects", []):
                for it in o.get("instances", []):
                    legacy = (
                        (it.get("form") or "").strip().lower()
                        if isinstance(it, dict)
                        else ""
                    )
                    if legacy in ("tabletop", "table"):
                        f = f"closeup:{legacy}"
                        break
                if f in ("closeup:table", "closeup:tabletop"):
                    break
            else:
                f = "closeup:table"  # same safe default as the router

    if f == "closeup:tabletop":
        main_id = main_support_id(masks)
        if floor_under_furniture_support(masks.get("relationships") or [], main_id):
            return "closeup:table"
    return f


def scene_form_is_table(masks: dict) -> bool:
    """True when a BASE/legs must exist under the main support (the rules gate's
    connected-base check). Room scenes anchor on the floor, so they are not 'table'."""
    return scene_form(masks) == "closeup:table"


_FLOOR_CATS = ("floor", "ground")


def _adjudicate_surface_relationships(
    masks_json_path: str, world: np.ndarray | None
) -> dict[str, Any]:
    """Adjudicate and persist relationship authority before any relationship consumer.

    This is deliberately a file-level helper because preprocessing mutates ``masks.json``
    several times (prune, wall merge, sliver cleanup).  It is safe to call again after
    those mutations: verdicts are recomputed, provenance/id fields survive, and the audit
    list accumulates claims whose endpoints are subsequently pruned.
    """

    from lib.tools.geometry.surface_relations import adjudicate_relationships

    md = json.load(open(masks_json_path))
    rels = md.get("relationships") or []
    reverted_id = str((md.get("work_surface_reverted") or {}).get("id") or "")
    if reverted_id:
        for rel in rels:
            if (
                rel.get("type") == "under"
                and rel.get("b") == reverted_id
                and not rel.get("provenance")
            ):
                rel["provenance"] = [
                    {
                        "source": "geometry_preprocess",
                        "action": "derived_work_surface_reversion",
                    }
                ]
    roots = {
        f"{r['category']}#{r['instance']}".lower()
        for r in md.get("instances", [])
        if r.get("kind") == "root_surface"
    }
    id2m: dict[str, np.ndarray] = {}
    contact_occluder_masks: list[np.ndarray] = []
    if world is not None:
        hh, ww = world.shape[:2]
        for r in md.get("instances", []):
            path = r.get("mask_path")
            if not path or not os.path.exists(path):
                continue
            resized = _resize_mask(_binarize(np.load(path)), hh, ww)
            if r.get("kind") != "root_surface":
                # Non-root masks explain holes/interruptions in an apparent support-wall
                # seam.  They may suppress image compatibility, but never manufacture it.
                contact_occluder_masks.append(resized)
                continue
            rid = f"{r['category']}#{r['instance']}".lower()
            id2m[rid] = resized
    summary = adjudicate_relationships(
        rels,
        roots,
        id2m,
        world,
        contact_occluder_masks=contact_occluder_masks,
    )
    md["relationships"] = rels
    md["relationship_adjudication"] = summary
    _reconcile_relationship_form(md)

    # Keep a scene-wide audit even when hard-only connectivity pruning later removes
    # an endpoint.  This is diagnostic data, never a source of geometry authority.
    prior = {
        str(r.get("relationship_id")): r
        for r in md.get("relationship_audit", [])
        if isinstance(r, dict) and r.get("relationship_id")
    }
    for rel in rels:
        prior[str(rel.get("relationship_id"))] = rel
    md["relationship_audit"] = list(prior.values())
    with open(masks_json_path, "w") as f:
        json.dump(md, f, indent=2)
    return summary


def _reconcile_relationship_form(md: dict[str, Any]) -> None:
    """Make ``routing.form`` agree with adjudicated floor-under-main authority.

    A valid UNDER claim is policy-authoritative at phase 0, before finite-mask geometry is
    available.  Once adjudication normalizes ``under(floor, main furniture)`` this helper
    promotes a bare tabletop route to the full furniture form.  It records the prior router
    form so a later post-merge adjudication can reversibly remove the promotion if endpoint
    validity is lost.
    """

    from lib.tools.geometry.surface_relations import floor_under_furniture_support

    routing = md.get("routing")
    if not isinstance(routing, dict):
        return
    current = str(routing.get("form") or "").strip().lower()
    if current == "room":
        return
    main_id = main_support_id(md)
    floor_id = floor_under_furniture_support(md.get("relationships") or [], main_id)
    if floor_id and current == "closeup:tabletop":
        routing.setdefault("form_without_relationship_adjudication", current)
        routing["form"] = "closeup:table"
        routing["form_source"] = "relationship_adjudication"
        routing["form_reason"] = (
            f"policy-authoritative hard under({floor_id}, {main_id})"
        )
        return

    controlled_sources = {"floor_under_main_support", "relationship_adjudication"}
    if (
        not floor_id
        and current == "closeup:table"
        and str(routing.get("form_source") or "") in controlled_sources
    ):
        prior = (
            str(routing.get("form_without_relationship_adjudication") or "")
            .strip()
            .lower()
        )
        raw = str(md.get("scene_kind") or "").strip().lower()
        if prior not in {"closeup:tabletop", "closeup:table"}:
            prior = {
                "tabletop": "closeup:tabletop",
                "closeup:tabletop": "closeup:tabletop",
            }.get(raw, "")
        if prior == "closeup:tabletop":
            routing["form"] = prior
            routing["form_source"] = "relationship_adjudication_restored"
            routing["form_reason"] = (
                "no adjudicated hard floor-under-main relationship remains; "
                "restored the pre-relationship router form"
            )


def _sync_relationship_form_to_moge(
    masks_json_path: str, moge_json_path: str
) -> str | None:
    """Persist the effective post-adjudication form in MoGE's gravity contract."""

    if not os.path.exists(moge_json_path):
        return None
    md = json.load(open(masks_json_path))
    effective = scene_form(md)
    mj = json.load(open(moge_json_path))
    gravity = mj.get("gravity")
    if not isinstance(gravity, dict):
        return None
    gravity["form"] = effective
    routing = md.get("routing") or {}
    gravity["form_source"] = str(routing.get("form_source") or "routing")
    with open(moge_json_path, "w") as f:
        json.dump(mj, f, indent=2)
    return effective


def _fail_closed_relationship_adjudication(
    masks_json_path: str, error: Exception
) -> None:
    ""

    from lib.tools.geometry.surface_relations import RELATIONSHIP_TYPES

    md = json.load(open(masks_json_path))
    failure_reason = (
        f"relationship adjudication unavailable: {type(error).__name__}: {error}"
    )
    root_ids = {
        f"{r['category']}#{r['instance']}".strip().lower()
        for r in md.get("instances", [])
        if r.get("kind") == "root_surface"
    }
    relationships = [
        rel for rel in (md.get("relationships") or []) if isinstance(rel, dict)
    ]
    md["relationships"] = relationships
    for index, rel in enumerate(relationships):
        rel["relationship_id"] = str(rel.get("relationship_id") or f"rel-{index:03d}")
        rel_type = str(rel.get("type") or "").strip().lower()
        a = str(rel.get("a") or "").strip().lower()
        b = str(rel.get("b") or "").strip().lower()
        rel["type"], rel["a"], rel["b"] = rel_type, a, b
        valid_relationship = (
            rel_type in RELATIONSHIP_TYPES
            and bool(a)
            and bool(b)
            and a != b
            and a in root_ids
            and b in root_ids
        )
        valid_under = valid_relationship and rel_type == "under"
        if valid_under:
            a_cat = a.rsplit("#", 1)[0]
            b_cat = b.rsplit("#", 1)[0]
            if b_cat in _FLOOR_CATS and a_cat not in _FLOOR_CATS:
                rel["a"], rel["b"] = b, a
            rel["status"] = "confirmed"
            rel["enforcement"] = "hard"
            measurements = rel.get("measurements")
            if not isinstance(measurements, dict):
                measurements = rel["measurements"] = {}
            measurements["under_policy_authoritative"] = True
            policy_reason = (
                "valid UNDER is policy-authoritative; finite-mask geometry is "
                "diagnostic only"
            )
        elif valid_relationship:
            rel["status"] = "unverified"
            rel["enforcement"] = "advisory"
            policy_reason = None
        else:
            rel["status"] = "rejected"
            rel["enforcement"] = "none"
            policy_reason = (
                "malformed, self-referential, or unresolved relationship cannot receive "
                "fallback authority"
            )
        reasons = rel.get("adjudication_reasons")
        if not isinstance(reasons, list):
            reasons = rel["adjudication_reasons"] = []
        if failure_reason not in reasons:
            reasons.append(failure_reason)
        if policy_reason is not None and policy_reason not in reasons:
            reasons.append(policy_reason)
        provenance = rel.get("provenance")
        if not isinstance(provenance, list):
            provenance = rel["provenance"] = []
        record = {
            "source": "geometry_preprocess",
            "action": (
                "under_policy_confirmed_after_adjudication_failure"
                if valid_under
                else (
                    "adjudication_failed_closed"
                    if valid_relationship
                    else "invalid_relationship_rejected_after_adjudication_failure"
                )
            ),
            "reason": policy_reason or failure_reason,
        }
        if record not in provenance:
            provenance.append(record)
    _reconcile_relationship_form(md)
    with open(masks_json_path, "w") as f:
        json.dump(md, f, indent=2)


def _prune_to_main_support(
    masks_json_path: str,
    main_id: Optional[str],
    scene_kind: str = "closeup",
) -> None:
    ""
    if not main_id:
        return
    main_id = str(main_id).strip().lower()
    masks = json.load(open(masks_json_path))
    insts = masks.get("instances", [])
    rels = masks.get("relationships", []) or []
    from lib.tools.geometry.surface_relations import (
        RELATIONSHIP_TYPES,
        relationship_is_hard,
    )

    root_ids = {
        f"{r['category']}#{r['instance']}".strip().lower()
        for r in insts
        if r.get("kind") == "root_surface"
    }
    adj: dict[str, set] = {}
    relationship_endpoint_ids: set[str] = set()
    for r in rels:
        if not isinstance(r, dict):
            continue
        rel_type = str(r.get("type") or "").strip().lower()
        a = str(r.get("a") or "").strip().lower()
        b = str(r.get("b") or "").strip().lower()
        if (
            rel_type not in RELATIONSHIP_TYPES
            or not a
            or not b
            or a == b
            or a not in root_ids
            or b not in root_ids
        ):
            continue
        # Identity and authority are deliberately orthogonal.  Even a rejected claim is
        # evidence that the proposer/auditor referred to two successfully segmented scene
        # roots; its enforcement verdict must not silently delete either surface.
        relationship_endpoint_ids.update((a, b))
        if relationship_is_hard(r):
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    keep_surf, frontier = {main_id}, [main_id]  # surfaces related to the main support
    while frontier:
        for nb in adj.get(frontier.pop(), ()):
            if nb not in keep_surf:
                keep_surf.add(nb)
                frontier.append(nb)
    endpoint_only_ids = relationship_endpoint_ids - keep_surf
    if endpoint_only_ids:
        print(
            "[preprocess] retained root relationship endpoint identity outside the "
            "main hard-connected component: " + ", ".join(sorted(endpoint_only_ids))
        )
    by_id = {f"{r['category']}#{r['instance']}".strip().lower(): r for r in insts}
    sup = {
        nid: (str(r.get("support")).strip().lower() if r.get("support") else None)
        for nid, r in by_id.items()
    }
    room = scene_kind == "room"
    kept_floor_ids = {
        nid
        for nid, r in by_id.items()
        if r.get("kind") == "root_surface"
        and nid in keep_surf
        and str(r["category"]).strip().lower() in _FLOOR_CATS
    }

    listed_cats = {str(r["category"]).strip().lower() for r in insts} | {
        str(o.get("category", "")).strip().lower() for o in masks.get("objects", [])
    }

    def _chain(oid: str) -> str:
        """'main' | 'floor' (kept floor root) | 'dangling' | 'other'."""
        cur, seen = oid, set()
        while True:
            if cur == main_id:
                return "main"
            if cur not in sup:
                # id the proposer never listed: an INDEX SLIP on a listed category
                # is recoverable (geometry re-parents); an absent category is an
                # intentionally excluded surface -> drop.
                cat = cur.rsplit("#", 1)[0].strip().lower()
                return "dangling" if cat in listed_cats else "other"
            nxt = sup[cur]
            if nxt is None:
                if cur == oid:
                    return "dangling"
                return "floor" if cur in kept_floor_ids else "other"  # listed root
            if nxt in seen:
                return "other"  # cycle
            seen.add(nxt)
            cur = nxt

    object_status: dict[str, str] = {}
    kept_object_ids: set[str] = set()
    dangling_kept: list[str] = []
    for r in insts:
        if r.get("kind") == "root_surface":
            continue
        nid = f"{r['category']}#{r['instance']}".strip().lower()
        status = _chain(nid)
        object_status[nid] = status
        if status == "main" or (status == "floor" and room):
            kept_object_ids.add(nid)
        elif status == "dangling":
            # The final upstream support pass will assign an exact retained parent.
            r["support"] = None
            sup[nid] = None
            dangling_kept.append(nid)
            kept_object_ids.add(nid)

    # A kept object makes its complete declared support chain part of the retained
    # inventory. Root-surface support metadata is not a graph parent edge, but it still
    # participates in this semantic reachability walk: floor -> table -> mug must not
    # keep the mug while deleting the table it explicitly rests on.
    required_support_ids: set[str] = set()
    for oid in kept_object_ids:
        cur = sup.get(oid)
        seen: set[str] = set()
        while cur in by_id and cur not in seen:
            required_support_ids.add(cur)
            seen.add(cur)
            cur = sup.get(cur)

    kept, pruned = [], []
    for r in insts:
        nid = f"{r['category']}#{r['instance']}".strip().lower()
        if r.get("kind") == "root_surface":
            (
                kept
                if nid in keep_surf
                or nid in relationship_endpoint_ids
                or nid in required_support_ids
                else pruned
            ).append(r)
        elif nid in kept_object_ids:
            kept.append(r)
        else:
            pruned.append(r)
    if dangling_kept:
        print(
            "[preprocess] dangling support id(s) — kept with support cleared, "
            "geometric parent resolution decides: " + ", ".join(sorted(dangling_kept))
        )
    if not pruned:
        if dangling_kept:  # persist the nulled supports even when nothing dropped
            masks["instances"] = kept
            with open(masks_json_path, "w") as f:
                json.dump(masks, f, indent=2)
        return
    masks["instances"] = kept
    kept_root_ids = {
        f"{r['category']}#{r['instance']}".strip().lower()
        for r in kept
        if r.get("kind") == "root_surface"
    }
    masks["relationships"] = [
        r
        for r in rels  # drop edges to pruned surfaces
        if isinstance(r, dict)
        and str(r.get("a") or "").strip().lower() in kept_root_ids
        and str(r.get("b") or "").strip().lower() in kept_root_ids
    ]
    masks.setdefault("dropped", []).extend(
        {
            "category": r["category"],
            "instance": r["instance"],
            "description": r.get("description"),
            "reason": "off_main_support",
        }
        for r in pruned
    )
    with open(masks_json_path, "w") as f:
        json.dump(masks, f, indent=2)
    print(
        f"[preprocess] pruned {len(pruned)} off-main-support instance(s): "
        + ", ".join(sorted({r["category"] for r in pruned}))
    )


def _prune_disconnected(masks_json_path: str) -> None:
    """Connectivity prune, post-resegment: drop objects whose mask is fully disconnected
    from the root surface they rest on (``agentic_mask.drop_disconnected_from_support``).
    Runs AFTER generative re-segmentation so a partially-detected object (a monitor whose
    stand got no mask) is judged on its redetect-COMPLETED mask, not the partial one. A
    BORDER redetect mask lives on the outpainted canvas (misaligned with the support's
    mask), so border records keep the ORIGINAL in-frame mask for the test — mirroring
    ``build_placement_table``'s screen_bbox rule. Rewrites masks.json only when something
    dropped; a dropped entry keeps its full instance record (overlay/redetect) for the
    demo."""
    from lib.tools.geometry.agentic_mask import (
        InstanceMask,
        drop_disconnected_from_support,
    )

    masks = json.load(open(masks_json_path))
    recs: list[InstanceMask] = []
    by_key: dict[tuple[str, int], dict] = {}
    for r in masks.get("instances", []):
        rd = r.get("redetect") or {}
        mp = r.get("mask_path")
        if (
            rd.get("mask_path")
            and os.path.exists(rd["mask_path"])
            and not rd.get("border")
        ):
            mp = rd["mask_path"]
        recs.append(
            InstanceMask(
                category=r["category"],
                instance=r["instance"],
                kind=r.get("kind", "object"),
                support=r.get("support"),
                description=r.get("description"),
                mask_path=mp,
                overlay_path=r.get("overlay_path"),
            )
        )
        by_key[(r["category"], r["instance"])] = r
    dropped: list[dict[str, Any]] = []
    kept = drop_disconnected_from_support(recs, dropped)
    if not dropped:
        return
    kept_keys = {(k.category, k.instance) for k in kept}
    masks["instances"] = [
        r for r in masks["instances"] if (r["category"], r["instance"]) in kept_keys
    ]
    for d in dropped:
        orig = by_key[(d["category"], d["instance"])]
        masks.setdefault("dropped", []).append({**orig, "reason": d["reason"]})
    with open(masks_json_path, "w") as f:
        json.dump(masks, f, indent=2)
    print(
        f"[preprocess] connectivity prune (post-resegment): dropped {len(dropped)} "
        "instance(s): " + ", ".join(f"{d['category']}#{d['instance']}" for d in dropped)
    )


COPLANAR_WALL_ANG_DEG = 10.0  # horizontal-normal angle (mod 180)
COPLANAR_WALL_OFFSET = 0.20  # m, max of the mutual centroid-to-plane distances
_SLIVER_PLANE_INLIER = 0.6  # fraction of a SLIVER wall's points that must lie inside a
# trusted keeper's plane band before it may be absorbed (see _merge_coplanar_walls)
_SLIVER_MIN_PTS = 50  # a sliver needs at least this many finite points to be judged
_SLIVER_VETO_DEG = 45.0  # a sliver's OWN fit may VETO a merge (never authorize one): a
def _drop_sliver_roots(
    masks_json_path: str, exclude_ids: set[str] | None = None
) -> list[str]:
    """Drop non-wall root surfaces below the reliable-plane mask threshold.

    A sub-2% WALL remains a real scene-graph identity, including its mask and
    relationships, but its mask is too small to trust for plane orientation, yaw
    anchoring, or detailed mask-size grading.  The MAIN support (``exclude_ids``)
    and floor/ground categories are also always retained.  Supports pointing at a
    genuinely dropped non-wall root are nulled and its relationships are removed.
    Kill switch ``GRASE_DROP_SLIVER_ROOTS=0``.
    """
    if os.environ.get("GRASE_DROP_SLIVER_ROOTS", "1") == "0":
        return []
    from lib.tools.geometry.scene_graph import MIN_PLANE_MASK_FRAC

    md = json.load(open(masks_json_path))
    logs: list[str] = []
    retained_walls: list[str] = []
    keep: list = []
    for r in md.get("instances", []):
        rid = f"{r['category']}#{r['instance']}"
        cat = str(r.get("category", "")).lower()
        if (
            r.get("kind") != "root_surface"
            or rid in (exclude_ids or set())
            or "floor" in cat
            or "ground" in cat
            or not r.get("mask_path")
            or not os.path.exists(r["mask_path"])
        ):
            keep.append(r)
            continue
        frac = float((_binarize(np.load(r["mask_path"])) > 0).mean())
        if frac >= MIN_PLANE_MASK_FRAC:
            keep.append(r)
            continue
        if cat == "wall":
            keep.append(r)
            retained_walls.append(
                f"{rid} ({frac:.2%} of frame; identity retained, plane untrusted)"
            )
            continue
        md.setdefault("dropped", []).append({**r, "reason": f"sliver_mask {frac:.3%}"})
        logs.append(f"{rid} ({frac:.2%} of frame)")
        for x in md.get("instances", []):
            if (x.get("support") or "").lower() == rid.lower():
                x["support"] = None
        md["relationships"] = [
            rel
            for rel in md.get("relationships", [])
            if rid not in (rel.get("a"), rel.get("b"))
        ]
    if logs or retained_walls:
        md["instances"] = keep
        if logs:
            md["sliver_roots_dropped"] = md.get("sliver_roots_dropped", []) + logs
        if retained_walls:
            prior = md.setdefault("sliver_walls_retained", [])
            prior.extend(x for x in retained_walls if x not in prior)
        json.dump(md, open(masks_json_path, "w"), indent=2)
        if logs:
            print(f"[preprocess] sliver roots dropped: {'; '.join(logs)}")
        if retained_walls:
            print(
                "[preprocess] sliver walls retained (plane untrusted): "
                + "; ".join(retained_walls)
            )
    return logs


def _merge_coplanar_walls(
    masks_json_path: str, world: np.ndarray, exclude_ids: set[str] | None = None
) -> list[str]:
    ""
    if os.environ.get("GRASE_COPLANAR_WALLS", "1") == "0":
        return []
    from lib.tools.geometry import scene_graph as sgm
    from lib.tools.geometry.agentic_mask import make_overlay

    md = json.load(open(masks_json_path))
    hh, ww = world.shape[:2]
    walls = []  # (rid, rec, plane, mask_area) — trusted; may FORM a group
    slivers = []  # (rid, rec, pts, mask_area) — untrusted; may only JOIN one
    for r in md.get("instances", []):
        rid = f"{r['category']}#{r['instance']}"
        if (
            r.get("kind") != "root_surface"
            or "floor" in r["category"].lower()
            or rid in (exclude_ids or set())
            or not r.get("mask_path")
            or not os.path.exists(r["mask_path"])
        ):
            continue
        native_mask = _binarize(np.load(r["mask_path"]))
        m = _resize_mask(native_mask, hh, ww)
        pts = world[m & np.isfinite(world).all(axis=2)]
        if float((native_mask > 0).mean()) < sgm.MIN_PLANE_MASK_FRAC:
            if len(pts) >= _SLIVER_MIN_PTS:  # judged by its POINTS in tier 2, not a fit
                slivers.append((rid, r, pts, int(m.sum())))
            continue
        if len(pts) < 100:
            continue
        pl = sgm.fit_plane(pts, up=(0, 0, 1))
        if pl is None or abs(pl["normal"][2]) > sgm.PLUMB_WALL_MAX_NZ:
            continue  # not a plumb wall (or a corrupt vertical fit on a support)
        walls.append((rid, r, pl, int(m.sum())))
    if not walls:  # tier 2 needs at least one trusted plane to test against
        return []

    parent = {w[0]: w[0] for w in walls}

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _hn(n):
        n = np.asarray(n, float)
        h = float(np.hypot(n[0], n[1]))
        return n[:2] / h if h > 1e-6 else None

    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            (ra, _, pa, _), (rb, _, pb, _) = walls[i], walls[j]
            ha, hb = _hn(pa["normal"]), _hn(pb["normal"])
            if ha is None or hb is None:
                continue
            ang = np.degrees(np.arccos(min(1.0, abs(float(ha @ hb)))))
            off = max(
                abs(float(np.asarray(pa["normal"]) @ pb["centroid"] + pa["d"])),
                abs(float(np.asarray(pb["normal"]) @ pa["centroid"] + pb["d"])),
            )
            if ang < COPLANAR_WALL_ANG_DEG and off < COPLANAR_WALL_OFFSET:
                parent[_find(ra)] = _find(rb)

    groups: dict[str, list] = {}
    for w in walls:
        groups.setdefault(_find(w[0]), []).append(w)
    for grp in groups.values():
        grp.sort(key=lambda w: -w[3])  # keeper = largest TRUSTED mask, index 0

    # Tier 2: slivers join an existing group; they are appended AFTER the sort above so
    # one can never become a keeper, and each is tested only against trusted planes.
    sliver_notes: dict[str, str] = {}
    if slivers and os.environ.get("GRASE_COPLANAR_WALLS_SLIVER", "1") != "0":
        for rid, rec, pts, area in slivers:
            # The sliver's own fit is a VETO only — it can block a merge, never authorize
            # one. Without it, proximity alone absorbed a near-perpendicular neighbour.
            spl = sgm.fit_plane(pts, up=(0, 0, 1))
            if spl is None:
                continue  # no fit to veto with -> retain the wall (the safe default)
            sn = np.asarray(spl["normal"], float)
            best = None  # (sort key, group id, inlier fraction)
            for gid, grp in groups.items():
                kn = np.asarray(grp[0][2]["normal"], float)
                cos = abs(float(sn @ kn)) / (
                    (np.linalg.norm(sn) * np.linalg.norm(kn)) or 1.0
                )
                if np.degrees(np.arccos(min(1.0, cos))) > _SLIVER_VETO_DEG:
                    continue  # a different wall, whatever the residuals say
                kpl = grp[0][2]
                resid = np.abs(pts @ kn + kpl["d"])
                frac = float((resid < COPLANAR_WALL_OFFSET).mean())
                if frac < _SLIVER_PLANE_INLIER:
                    continue
                key = (-frac, float(np.median(resid)))
                if best is None or key < best[0]:
                    best = (key, gid, frac)
            if best is not None:
                groups[best[1]].append((rid, rec, None, area))
                sliver_notes[rid] = f"sliver, {best[2]:.0%} of points in plane"

    logs: list[str] = []
    image_path = str(Path(masks_json_path).parent.parent / "input.png")
    for grp in groups.values():
        if len(grp) < 2:
            continue
        kid, krec = grp[0][0], grp[0][1]
        keep = _binarize(np.load(krec["mask_path"])) > 0
        for rid, rec, _pl, _a in grp[1:]:
            vm = _binarize(np.load(rec["mask_path"])) > 0
            if vm.shape != keep.shape:
                vm = _resize_mask(vm, keep.shape[0], keep.shape[1])
            keep = keep | vm
            md["instances"] = [
                x for x in md["instances"] if f"{x['category']}#{x['instance']}" != rid
            ]
            md.setdefault("dropped", []).append(
                {**rec, "reason": f"coplanar_with {kid}"}
            )
            for x in md["instances"]:
                if (x.get("support") or "").lower() == rid.lower():
                    x["support"] = kid
            for rel in md.get("relationships") or []:
                for k in ("a", "b"):
                    if (rel.get(k) or "").lower() == rid.lower():
                        rel[k] = kid
            note = sliver_notes.get(rid)
            logs.append(f"{rid} -> {kid}" + (f" ({note})" if note else ""))
        md["relationships"] = [
            rel
            for rel in md.get("relationships") or []
            if (rel.get("a") or "").lower() != (rel.get("b") or "").lower()
        ]
        np.save(krec["mask_path"], keep.astype(np.uint8) * 255)
        if krec.get("overlay_path") and os.path.exists(image_path):
            try:
                make_overlay(image_path, keep, krec["overlay_path"])
            except Exception:  # noqa: BLE001 - overlay is demo-only
                pass
    if logs:
        md["coplanar_walls_merged"] = md.get("coplanar_walls_merged", []) + logs
        with open(masks_json_path, "w") as f:
            json.dump(md, f, indent=2)
        print(f"[preprocess] coplanar walls merged: {'; '.join(logs)}")
    return logs


def _regenerate_composite(masks_json_path: str, image_path: str) -> None:
    ""
    from lib.tools.geometry.agentic_mask import InstanceMask, make_composite

    try:
        md = json.load(open(masks_json_path))
        recs = []
        for r in md.get("instances", []):
            rec = InstanceMask(
                category=r["category"],
                instance=r["instance"],
                kind=r.get("kind", "object"),
                support=r.get("support"),
                description=r.get("description"),
                point=r.get("point"),
                mask_path=r.get("mask_path"),
            )
            recs.append(rec)
        make_composite(
            image_path, recs, str(Path(masks_json_path).parent / "composite.png")
        )
        print(f"[preprocess] composite regenerated from {len(recs)} final instances")
    except Exception as e:  # noqa: BLE001 - a demo artifact must not fail the run
        print(f"WARNING [preprocess]: composite regeneration failed ({e})")


def _revert_occluded_work_surface(
    masks_json_path: str, coverage_thresh: float = 0.30
) -> None:
    """Room-track backstop: when the router/proposer made the WORK SURFACE an object
    but its mask is mostly COVERED by the other object masks (>= ``coverage_thresh``
    of it), SAM3D would have to hallucinate the very surface everything rests on —
    flip the scene back to FULL closeup semantics: the work surface becomes a root
    surface, the scene-level form flag becomes 'closeup:table' (it stands elevated on
    a visible floor, so its base counts), an ``under`` floor->work-surface relationship is added
    so the surface-keep rule still connects the scene, and ``scene_kind`` /
    ``routing.mode`` flip back to 'closeup' so the downstream prune treats the scene
    exactly like today's pipeline (floor furniture drops — the settle proxies every
    root-supported object onto the z=0 slab, which is the TABLE top here, so keeping
    chairs in a reverted scene would beach them at table height). Masks untouched.
    No-op unless masks.json records a room-track run (routing.mode == 'room')."""
    masks = json.load(open(masks_json_path))
    if (masks.get("routing") or {}).get("mode") != "room":
        return
    insts = masks.get("instances", [])
    by_id = {f"{r['category']}#{r['instance']}": r for r in insts}
    sup = {nid: (r.get("support") or None) for nid, r in by_id.items()}
    children: dict[str, list[str]] = {}
    for nid, s in sup.items():
        if s:
            children.setdefault(s, []).append(nid)
    floor_ids = {
        nid
        for nid, r in by_id.items()
        if r.get("kind") == "root_surface" and r["category"].lower() in _FLOOR_CATS
    }
    ws = [
        (nid, r)
        for nid, r in by_id.items()
        if r.get("kind") != "root_surface"
        and sup.get(nid) in floor_ids
        and children.get(nid)
    ]
    if not ws:
        return
    ws_id, ws_rec = ws[0]
    mp = ws_rec.get("mask_path")
    if not mp or not os.path.exists(mp):
        return
    ws_mask = _binarize(np.load(mp))
    others = np.zeros_like(ws_mask)
    for nid, r in by_id.items():
        if nid == ws_id or r.get("kind") == "root_surface":
            continue
        omp = r.get("mask_path")
        if omp and os.path.exists(omp):
            others |= _resize_mask(_binarize(np.load(omp)), *ws_mask.shape)
    area = int(ws_mask.sum())
    coverage = float((ws_mask & others).sum() / area) if area else 1.0
    if coverage < coverage_thresh:
        return
    floor_id = sup[ws_id]
    ws_rec["kind"] = "root_surface"  # support stays the floor id (like a closeup table)
    ws_cat, ws_k = ws_rec["category"], ws_rec["instance"]
    for o in masks.get("objects", []):
        if o.get("category") != ws_cat:
            continue
        for k, it in enumerate(o.get("instances", [])):
            if isinstance(it, dict) and k == ws_k:
                it["kind"] = "root_surface"
    rels = masks.setdefault("relationships", [])
    if not any(
        r.get("type") == "under" and r.get("a") == floor_id and r.get("b") == ws_id
        for r in rels
    ):
        # Coverage makes the *routing form* below authoritative.  The mandatory
        # adjudication pass validates the endpoints and makes this UNDER policy-hard;
        # finite-mask contact remains diagnostic rather than a source of authority.
        rels.append({"type": "under", "a": floor_id, "b": ws_id})
    masks["work_surface_reverted"] = {
        "id": ws_id,
        "coverage": round(coverage, 3),
        "threshold": coverage_thresh,
        "router_scene_kind": masks.get("scene_kind"),
    }
    masks["scene_kind"] = "closeup:table"  # full closeup semantics from here on
    r = masks.setdefault("routing", {})
    r["mode"] = "closeup"
    # elevated above a visible floor -> the full piece, base included
    r["form"] = "closeup:table"
    with open(masks_json_path, "w") as f:
        json.dump(masks, f, indent=2)
    print(
        f"[preprocess] work surface {ws_id} is {coverage:.0%} covered "
        f"(>= {coverage_thresh:.0%}) -> reverted to root_surface (closeup semantics)"
    )


def _accept_up_normal(g, pts):
    """Accept ``g`` as a support surface's up-normal in the RAW (camera-oriented)
    frame: pass when it points up (``g[2] >= 0.6``); else — a steep / top-down photo
    moves the tabletop normal off +Z — accept only when it clearly faces the camera
    (world origin), sign-fixed toward it. The two 0.6 thresholds overlap in camera
    pitch, so level photos never reach the face branch and can't be sign-flipped.
    Returns the (possibly sign-fixed) normal, or None (not a support plane)."""
    if g is None:
        return None
    g = np.asarray(g, float)
    if g[2] >= 0.6:
        return g
    c = pts.mean(axis=0) if len(pts) else None
    face = (
        float(np.dot(g, -c) / np.linalg.norm(c))
        if c is not None and np.linalg.norm(c) > 1e-6
        else 0.0
    )
    if abs(face) < 0.6:
        return None
    return -g if face < 0 else g


_GROW_BAND = 0.015  # m: coplanarity band for absorbing a component
_GROW_COMP_MIN = 200  # px: ignore smaller components
_GROW_FRAC = 0.6  # fraction of a component's points inside the band to absorb it
_WITNESS_INLIER = 0.3  # dominant-plane quality bar for a child witness
_COMP_MIN_PX = 500  # px: root components smaller than this don't vote
_COMP_INLIER = 0.5  # a root component must BE a plane to vote
_CLUSTER_DEG = 15.0  # direction-cluster radius (parallel evidence: <=10.4 deg
# measured across MoGE curvature; wrong-plane evidence: >=73 deg)
_Z_GATE_DEG = 25.0  # root component parallel enough to anchor the z height
_CHILD_OVERRIDE_ROOT_SHARE = 0.2  # child-only cluster wins only when root planar
# evidence totals < 20% of its weight


def _grown_plane_fit(m, point, world_raw):
    """Part A. Returns ``(grown_mask, pts, plane)`` — the mask trimmed to the
    Molmo-pointed component plus every component coplanar with it, its finite
    points, and their RANSAC plane — or ``(m, pts, plane-or-None)`` (full-mask
    behavior) when SciPy/point/components are unavailable."""
    fin = np.isfinite(world_raw).all(axis=2)
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001 - growth is an enhancement, never a gate
        pts = world_raw[m & fin]
        return m, pts, (sgm.fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None)
    lab, n = ndimage.label(m)
    if n <= 1:
        pts = world_raw[m & fin]
        return m, pts, (sgm.fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None)
    h, w = m.shape
    cid = 0
    if point:
        px = min(max(int(round(float(point[0]) * (w - 1))), 0), w - 1)
        py = min(max(int(round(float(point[1]) * (h - 1))), 0), h - 1)
        cid = int(lab[py, px])
    if cid == 0:  # no point / point off-mask -> largest component
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        cid = int(np.argmax(sizes)) + 1
    comp = lab == cid
    seed_pts = world_raw[comp & fin]
    pl = sgm.fit_plane(seed_pts, up=(0, 0, 1)) if len(seed_pts) >= 100 else None
    if pl is None:  # seed too small to fit -> keep the full-mask behavior
        pts = world_raw[m & fin]
        return m, pts, (sgm.fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None)
    nrm, cen = np.asarray(pl["normal"], float), np.asarray(pl["centroid"], float)
    grown = comp.copy()
    for i in range(1, n + 1):
        if i == cid:
            continue
        ci = lab == i
        if int(ci.sum()) < _GROW_COMP_MIN:
            continue
        p = world_raw[ci & fin]
        if len(p) < 50:
            continue
        d = np.abs((p - cen) @ nrm)
        if float((d <= _GROW_BAND).mean()) >= _GROW_FRAC:
            grown |= ci
    pts = world_raw[grown & fin]
    return grown, pts, (sgm.fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 100 else None)


def _ndeg(a, b):
    """Angle in degrees between two directions (sign-agnostic)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _direction_consensus(full_mask, insts, root_id, world_raw):
    """Part B (see the constants block above). Vote over the root mask's planar
    components + the children's dominant planes. Returns ``(g, witness_id, anchor)``
    — the winning direction, the donating child's id (None when a root component
    donated), and the union mask of root components parallel enough to anchor the
    z height (None -> flush-below fallback) — or ``None`` when there is nothing to
    vote with (caller keeps its screening fit)."""
    fin = np.isfinite(world_raw).all(axis=2)
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001 - consensus is an enhancement, never a gate
        return None
    lab, n = ndimage.label(full_mask)
    hyps = []  # (weight, normal, child_id | None, comp_id | None)
    comp_fits = {}  # comp_id -> (normal, mask)
    for i in range(1, n + 1):
        ci = lab == i
        if int(ci.sum()) < _COMP_MIN_PX:
            continue
        p = world_raw[ci & fin]
        pl = sgm.fit_plane(p, up=(0, 0, 1)) if len(p) >= 100 else None
        if not pl or pl["inlier_frac"] < _COMP_INLIER:
            continue
        nc = _accept_up_normal(np.asarray(pl["normal"], float), p)
        if nc is None:
            continue
        hyps.append((pl["inlier_frac"] * len(p), nc, None, i))
        comp_fits[i] = (nc, ci)
    for r in insts:
        if (r.get("support") or "").strip().lower() != root_id.lower():
            continue
        mp = r.get("mask_path")
        if not mp or not os.path.exists(mp):
            continue
        h, w = full_mask.shape
        m = _resize_mask(_binarize(np.load(mp)), h, w)
        p = world_raw[m & fin]
        pl = sgm.fit_plane(p, up=(0, 0, 1)) if len(p) >= 100 else None
        if not pl or pl["inlier_frac"] < _WITNESS_INLIER:
            continue
        nc = _accept_up_normal(np.asarray(pl["normal"], float), p)
        if nc is None:
            continue
        hyps.append((pl["inlier_frac"] * len(p), nc, _node_id(r), None))
    if not hyps:
        return None
    hyps.sort(key=lambda x: -x[0])
    clusters = []  # [rep_normal, total_weight, members]
    for wgt, nc, child, cid in hyps:
        for cl in clusters:
            if _ndeg(cl[0], nc) <= _CLUSTER_DEG:
                cl[1] += wgt
                cl[2].append((wgt, nc, child, cid))
                break
        else:
            clusters.append([nc, wgt, [(wgt, nc, child, cid)]])
    clusters.sort(key=lambda c: -c[1])
    rooted = [c for c in clusters if any(m[3] is not None for m in c[2])]
    childonly = [c for c in clusters if all(m[3] is None for m in c[2])]
    root_total = sum(w for w, _, _, cid in hyps if cid is not None)
    if rooted:
        winner = rooted[0]
        if childonly and root_total < _CHILD_OVERRIDE_ROOT_SHARE * childonly[0][1]:
            winner = childonly[0]  # the root mask is effectively junk
    else:
        winner = clusters[0]
    top = max(winner[2], key=lambda m: m[0])
    g, witness_id = top[1], top[2]
    anchor = None
    for i, (nc, ci) in comp_fits.items():
        if _ndeg(nc, g) <= _Z_GATE_DEG:
            anchor = ci if anchor is None else (anchor | ci)
    return g, witness_id, anchor


def _synthetic_ground(insts, normals, points_npy, world_raw, h, w, deroll=True):
    """Fallback when NO masked horizontal root surface qualifies as the ground — the
    support was unmaskable, or everything rests in/on a container OBJECT (bridge5's
    bin). The downstream contract still requires a level z=0 plane UNDER the scene
    (the initializer's main-support build and the robot harness), so
    synthesize one: gravity comes from the dominant RANSAC plane of the best DONOR
    instance (the one most other instances rest on — a container's floor dominates
    its own mask — else the largest mask), and the ground sits FLUSH below the lowest
    instance base, so the container/objects rest on it. Returns (R, T, info), or
    (None, None, None) when no instance has usable points (nothing to reconstruct).
    ``info['root']`` is None: no masked surface owns z=0, which also keeps the
    off-main-support prune a no-op."""
    sup_counts: dict[str, int] = {}
    for r in insts:
        s = (r.get("support") or "").strip().lower()
        if s:
            sup_counts[s] = sup_counts.get(s, 0) + 1
    donors = []
    for r in insts:
        mp = r.get("mask_path")
        if not mp or not os.path.exists(mp):
            continue
        m = _resize_mask(_binarize(np.load(mp)), h, w)
        donors.append((sup_counts.get(_node_id(r).lower(), 0), int(m.sum()), r, m))
    donors.sort(key=lambda d: (d[0], d[1]), reverse=True)
    g, donor = None, None
    for _, _, r, m in donors:
        pts = world_raw[m & np.isfinite(world_raw).all(axis=2)]
        if len(pts) < 100:
            continue
        pl = sgm.fit_plane(pts, up=(0, 0, 1))
        # 0.3 (not the candidates' 0.5): a container mask spans floor + walls, so the
        # dominant (floor) plane legitimately holds well under half the points.
        raw_g = (
            np.asarray(pl["normal"], float)
            if pl and pl["inlier_frac"] >= 0.3
            else mc.estimate_gravity_up(normals, m)
        )
        g = _accept_up_normal(raw_g, pts)
        if g is not None:
            donor = r
            break
    if g is None:
        return None, None, None
    roll_pre = mc.gravity_roll_deg(g)
    if deroll:
        derolled = abs(roll_pre) < mc.DEROLL_MAX_DEG
        g = mc.deroll_gravity(g)
    else:
        derolled = False  # calibrated GT rig: the measured roll is real

    R = mc.alignment_rotation(g)
    world = mc.moge_points_to_world(np.load(points_npy), R)
    bases = []
    for _, _, r, m in donors:
        zs = world[m & np.isfinite(world).all(axis=2)][:, 2]
        if len(zs) >= 30:
            bases.append(float(np.percentile(zs, 5)))  # robust per-instance base
    if not bases:
        return None, None, None
    z_ground = min(bases)
    info = {
        "root": None,
        "surface": "synthetic_ground",
        "synthetic": True,
        "donor": _node_id(donor),
        "g_world": [float(x) for x in g],
        "z_root": z_ground,
        "form": None,
        "roll_deg_pre": round(float(roll_pre), 3),
        "derolled": bool(derolled),
    }
    return R, np.array([0.0, 0.0, -z_ground]), info


def _choose_ground(
    cands: list[dict], main_id: Optional[str], scene_kind: str
) -> tuple[dict, bool]:
    """The scene's GROUND reference among the horizontal candidates, as
    ``(chosen, matched_main_support)``.

    Primary: the support-chain-resolved main support from ``main_support_id``, matched
    exactly. Being category-agnostic, it grounds closeups on table/desk/workbench/
    cutting-board/stove alike and room scenes on the floor.

    Tiebreak when that support is absent or names a surface that failed the candidate
    filters (2/344): a room's ground IS the floor, whereas in closeup a floor sits UNDER
    the work surface and must not become z=0 — so prefer the opposite class per mode, and
    let area settle any remainder. Deterministic; no VLM. (It replaced a VLM tie-breaker
    that never ran — that needed a form failure AND >1 candidate, and the pool has never
    held more than one; see ``_render_canonical_root``.)"""
    if main_id:
        for c in cands:
            if c["id"] == main_id:
                return c, True
    floors = [c for c in cands if c["category"].lower() in _FLOOR_CATS]
    prefer = floors if scene_kind == "room" else [c for c in cands if c not in floors]
    return max(prefer or cands, key=lambda c: c["area"]), False


def _compute_canonical_transform(
    masks_json_path: str,
    normal_npy: str,
    points_npy: str,
    image_path: str,
    deroll: bool = True,
):
    ""
    if not os.path.exists(normal_npy):
        return None, None, None
    normals = np.load(normal_npy)
    h, w = normals.shape[:2]
    world_raw = mc.moge_points_to_world(
        np.load(points_npy), None
    )  # GRASE world, pre-align
    masks_data = json.load(open(masks_json_path))
    insts = masks_data.get("instances", [])
    main_id = main_support_id(masks_data)  # support-chain resolved main support
    # 2-way mode for the ground tiebreak: scene_kind is now the router's 3-way verdict
    # ('room' | 'closeup:table' | 'closeup:tabletop'), and _choose_ground only asks
    # "is this a room?" (room ground IS the floor; in closeup a floor sits UNDER the
    # work surface and must not become z=0).
    scene_kind = "room" if scene_form(masks_data) == "room" else "closeup"
    # The ground reference must be a surface OBJECTS rest on (z=0 is the resting plane).
    # Now that every root is masked, exclude roots no object rests on
    # (a back wall, or a floor beneath a table) from the choice — else z=0 could land on the
    # floor while objects sit on the table.
    obj_support_ids = {
        (r.get("support") or "").strip().lower()
        for r in insts
        if r.get("kind") != "root_surface" and r.get("support")
    }
    cands = []
    for r in insts:
        if r.get("kind") != "root_surface":
            continue
        if obj_support_ids and _node_id(r).lower() not in obj_support_ids:
            continue  # no object rests on it -> not the ground
        mp = r.get("mask_path")
        if not mp or not os.path.exists(mp):
            continue
        m_full = _resize_mask(_binarize(np.load(mp)), h, w)
        # Part A: fit on the Molmo-pointed component grown by coplanar components
        # only — same-material contaminants (a wood pegboard behind a wood table)
        # never join the fit. Screening only: the CHOSEN root's final direction and
        # z anchor come from the point-free consensus below.
        m, pts, pl = _grown_plane_fit(m_full, r.get("point"), world_raw)
        if pl and pl["inlier_frac"] >= 0.5:  # trust the point fit when well-supported
            g = np.asarray(pl["normal"], float)
        else:  # fallback: MoGE normal-map average (over the grown mask)
            g = mc.estimate_gravity_up(normals, m)
        # Up-or-camera-facing gate (steep / top-down photos — abc1): see
        # _accept_up_normal. A normal that is neither is not the support.
        g = _accept_up_normal(g, pts)
        if g is None:
            continue
        cands.append(
            {
                "id": _node_id(r),
                "category": r["category"],
                "g": g,
                "mask": m,
                "mask_full": m_full,  # all components: the consensus votes on these
                "area": int(m.sum()),
            }
        )
    if not cands:
        # No masked horizontal root surface holds an object (unmaskable support, or a
        # container-object scene like bridge5's bin) -> synthesize a level ground
        # flush below everything rather than shipping a null gravity.
        return _synthetic_ground(
            insts, normals, points_npy, world_raw, h, w, deroll=deroll
        )
    chosen, matched_main = _choose_ground(cands, main_id, scene_kind)
    if not matched_main:
        print(
            f"WARNING [preprocess.canonical]: no usable main-support candidate; picked "
            f"{chosen['id']} by the {scene_kind} tiebreak ({len(cands)} candidate(s))"
        )
    if scene_kind == "room" and not any(
        c["category"].lower() in _FLOOR_CATS for c in cands
    ):
        # Not an enforced invariant: the room proposer is TOLD the floor is the scene's
        # anchor, but nothing validates it, and a listed-but-unmasked floor never reaches
        # the pool. Say so instead of silently grounding on something else.
        print(
            f"WARNING [preprocess.canonical]: room scene has no masked floor among the "
            f"ground candidates ({[c['id'] for c in cands]}) — grounding on "
            f"{chosen['id']}"
        )
    try:  # debug/demo overlay only — never block canonicalization on it
        _render_canonical_root(
            image_path,
            cands,
            chosen["id"],
            str(Path(masks_json_path).parent / "canonical_root.png"),
        )
    except Exception as e:  # noqa: BLE001
        print(f"WARNING [preprocess.canonical]: root overlay failed ({e}); continuing")
    # Part B: point-free direction consensus over the chosen root's components +
    # its children's planes (see the constants block). The z height anchors to the
    # root components parallel to the winning direction; when the root has none
    # (fully mis-masked support), it falls back to FLUSH BELOW all instance bases.
    witness_id, z_source = None, "root_component"
    res = _direction_consensus(chosen["mask_full"], insts, chosen["id"], world_raw)
    if res is not None:
        chosen["g"], witness_id, anchor = res
        if anchor is not None:
            chosen["mask"] = anchor
        else:
            z_source = "flush_below"
    # Camera zero-roll prior: an induced roll under DEROLL_MAX_DEG is estimation noise
    # (a hand-held photo is upright) — zero it in the camera frame and re-derive R.
    # SKIPPED for a calibrated GT-depth run: there the roll is a real rig offset and the
    # metric depth makes the normal trustworthy (see mc.trust_measured_camera_roll).
    roll_pre = mc.gravity_roll_deg(chosen["g"])
    if deroll:
        derolled = abs(roll_pre) < mc.DEROLL_MAX_DEG
        chosen["g"] = mc.deroll_gravity(chosen["g"])
        print(
            f"[preprocess.canonical] camera roll {roll_pre:+.2f}deg -> "
            + ("snapped upright" if derolled else f"kept (>= {mc.DEROLL_MAX_DEG}deg)")
        )
    else:
        derolled = False
        print(
            f"[preprocess.canonical] camera roll {roll_pre:+.2f}deg -> KEPT "
            "(GT depth + calibrated intrinsics: measured roll is a real rig offset)"
        )
    R = mc.alignment_rotation(chosen["g"])  # gravity-up (identity if already aligned)
    world = mc.moge_points_to_world(np.load(points_npy), R)
    fin_w = np.isfinite(world).all(axis=2)
    if z_source == "flush_below":
        # Partial hallucination: the ground plane sits flush below every instance
        # base (a container's floor height is INSIDE it, never the ground height).
        bases = []
        for r in insts:
            mp = r.get("mask_path")
            if not mp or not os.path.exists(mp):
                continue
            zs = world[_resize_mask(_binarize(np.load(mp)), h, w) & fin_w][:, 2]
            if len(zs) >= 30:
                bases.append(float(np.percentile(zs, 5)))
        z_root = min(bases) if bases else 0.0
    else:
        # canonical z: RANSAC the anchor region's plane in the gravity-aligned world.
        pts = world[chosen["mask"] & fin_w]
        plane = sgm.fit_plane(pts, up=(0, 0, 1)) if len(pts) >= 30 else None
        z_root = (
            float(plane["centroid"][2])
            if plane
            else (float(np.median(pts[:, 2])) if len(pts) else 0.0)
        )
    T = np.array([0.0, 0.0, -z_root])
    info = {
        "root": chosen["id"],
        "surface": chosen["category"],
        "g_world": [float(x) for x in chosen["g"]],
        "z_root": z_root,
        "z_source": z_source,
        # camera-roll audit trail (pre-deroll roll + whether the zero-roll prior fired)
        "roll_deg_pre": round(float(roll_pre), 3),
        "derolled": bool(derolled),
        # SCENE-level flag now (routing.form): 'room' | 'closeup:table' |
        # 'closeup:tabletop'. It no longer varies per surface, so it is recorded here
        # as the scene's form rather than a lookup on the chosen ground's id.
        "form": scene_form(masks_data),
    }
    if witness_id:
        info["g_witness"] = witness_id  # direction donated by this child's plane
    return R, T, info


# --------------------------------------------------------------------------- #
# SAM3D mesh generation (shape from SAM3D, placement from MoGE)                #
# --------------------------------------------------------------------------- #
def _moge_pointmap_path(out_dir: str, backend: str) -> Optional[str]:
    """The scene depth-backend point map to condition SAM3D on, or ``None``.

    Default for the ``sam3d`` backend: reuse the scene's already-computed point map
    (``<out>/moge/points.npy``), supplied by MoGE-2 or user-provided depth, instead of letting
    SAM3D re-estimate depth (MoGE-v1) internally. One geometry source for the whole
    pipeline, and it skips SAM3D's per-object MoGE-v1 pass. Set ``GRASE_SAM3D_POINTMAP=0``
    to force the old v1-internal baseline. Returns ``None`` for non-sam3d backends (DSO
    has no point-map condition) or when the point-map file is missing."""
    if backend != "sam3d":
        return None
    if os.environ.get("GRASE_SAM3D_POINTMAP", "1").lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return None
    p = Path(out_dir) / "moge" / "points.npy"
    return str(p) if p.exists() else None


def unify_same_size_scales(
    table: list[dict[str, Any]],
    masks_json_path: str,
    *,
    auto_lock: bool = False,
) -> list[str]:
    """Give every eligible registry group ONE shared physical size.

    ``auto_lock`` gates the
    ``vlm_auto`` groups: the visual auditor still runs and its groups stay in the
    registry / on the rows with status ``detected_unlocked``, but their meshes are
    normalized and scale-locked only with ``--same-size-auto-lock``. Manual
    (``user_explicit``) groups always apply.

    Target = per-axis MEDIAN of the instances' metric-pristine ``*_pcand.glb`` OBB
    extents (``[long, mid, short]``, metres) — the median rejects occlusion-shrunk
    outliers. Each pristine OBB frame is frozen before deformation; one anisotropic scale
    tensor is applied there and transported through the verified pristine-to-posed rigid
    transform to the placed mesh. A newly optimized post-scale OBB is never authoritative:
    irregular or near-symmetric geometry can make that solver choose a different frame.

    Membership is keyed by ``same_size_group_id``, not category. Manual groups require two
    usable pristine/posed pairs and automatic groups require three. Only after every pair
    in a group stages and validates are ``same_size`` and ``scale_locked`` activated.
    Native reconstruction caches are never touched. Returns the stable group ids it unified
    (for logging/tests)."""
    from lib.tools.geometry.mesh_place import (
        placed_glb_obb_frame,
        rescale_corresponding_glbs_to_extents,
    )
    from lib.tools.geometry.same_size import (
        NORMALIZATION_VERSION,
        REGISTRY_KEY,
        active_same_size_members,
        revalidate_same_size_registry,
    )

    # Clear only locks created by a previous same-size pass. Registration/asset locks are
    # independent and must survive. A registry candidate is never active by default.
    for o in table:
        o["same_size"] = False
        o.pop("same_size_applied", None)
        if o.pop("same_size_scale_locked", False) and not o.get("registration_locked"):
            o["scale_locked"] = False

    # Read extents before eligibility: a present path whose geometry cannot be measured
    # is not a usable mesh and must count as missing at this third/final threshold gate.
    candidate_pairs: dict[str, list[tuple[dict[str, Any], dict[str, np.ndarray]]]] = {}
    for o in table:
        group_id = o.get("same_size_group_id")
        mesh_glb = o.get("mesh_glb")
        pristine_glb = o.get("pristine_glb")
        if (
            group_id
            and mesh_glb
            and os.path.exists(mesh_glb)
            and pristine_glb
            and os.path.exists(pristine_glb)
            and not o.get("scale_locked")
            and o.get("mesh_provider") != "asset_bank"
        ):
            frame = placed_glb_obb_frame(pristine_glb)
            if frame is not None:
                candidate_pairs.setdefault(str(group_id), []).append((o, frame))

    registry_payload = json.load(open(masks_json_path))
    registry = registry_payload.get(REGISTRY_KEY)
    if not isinstance(registry, dict):
        raise RuntimeError(
            "same-size resolution is missing at mesh normalization; "
            "rerun full preprocessing"
        )
    registry = revalidate_same_size_registry(
        registry,
        {
            _same_size_object_id(o)
            for pairs in candidate_pairs.values()
            for o, _ in pairs
        },
        stage="usable_mesh",
    )
    _log_ineligible_same_size_groups(registry, "usable_mesh")
    registry["auto_lock_enabled"] = bool(auto_lock)
    unlocked_auto: set[str] = set()
    if not auto_lock:
        for group in registry.get("groups", []):
            if group.get("source") == "vlm_auto" and group.get("status") == "active":
                group["status"] = "detected_unlocked"
                unlocked_auto.add(str(group.get("group_id")))
                print(
                    f"[same_size] {group.get('group_id')}: vlm_auto group detected but NOT "
                    "locked (pass --same-size-auto-lock to normalize automatic groups)"
                )
    active_members = active_same_size_members(registry)
    for group_id in unlocked_auto:
        candidate_pairs.pop(group_id, None)

    # Remove candidate metadata from rows that failed the usable-mesh count. Keeping a
    # stale group id would let a downstream consumer accidentally reactivate the group.
    # Detected-but-unlocked automatic groups keep their metadata (the VLM's flag stays
    # visible) with the compatibility flag false.
    for o in table:
        object_id = _same_size_object_id(o)
        group_id = o.get("same_size_group_id")
        if group_id in unlocked_auto:
            o["same_size"] = False
            o["same_size_group_status"] = "detected_unlocked"
            continue
        if group_id and active_members.get(object_id) != group_id:
            for field in _SAME_SIZE_GROUP_FIELDS:
                o.pop(field, None)

    unified: list[str] = []
    failures: dict[str, str] = {}
    targets: dict[str, list[float]] = {}
    for group_id, pairs in candidate_pairs.items():
        pairs = [
            (o, frame)
            for o, frame in pairs
            if active_members.get(_same_size_object_id(o)) == group_id
        ]
        if not pairs:
            continue
        target = np.median(
            np.stack([frame["sorted_extents"] for _, frame in pairs]), axis=0
        )  # [long, mid, short]
        staged_outputs: list[tuple[str, str]] = []  # (temporary, final)
        backups: dict[str, str] = {}
        try:
            achieved: dict[str, dict[str, Any]] = {}
            seen_outputs: set[str] = set()
            for o, frame in pairs:
                posed = str(o["mesh_glb"])
                canonical = str(o["pristine_glb"])
                if posed == canonical:
                    raise RuntimeError(
                        f"posed and metric-pristine GLBs share one path: {posed}"
                    )
                pair_paths = (canonical, posed)
                if any(path in seen_outputs for path in pair_paths):
                    duplicate = next(
                        path for path in pair_paths if path in seen_outputs
                    )
                    raise RuntimeError(f"duplicate GLB path in group: {duplicate}")
                seen_outputs.update(pair_paths)

                staged_by_final: dict[str, str] = {}
                for glb in pair_paths:
                    # Never rewrite a member in place while another member can still
                    # fail. Stage every GLB beside its final path and validate it first.
                    with tempfile.NamedTemporaryFile(
                        dir=str(Path(glb).parent),
                        prefix=f".{Path(glb).stem}.same_size.",
                        suffix=Path(glb).suffix,
                        delete=False,
                    ) as tmp:
                        staged = tmp.name
                    os.unlink(staged)
                    staged_outputs.append((staged, glb))
                    staged_by_final[glb] = staged

                result = rescale_corresponding_glbs_to_extents(
                    canonical,
                    posed,
                    staged_by_final[canonical],
                    staged_by_final[posed],
                    target,
                    canonical_frame=frame,
                )
                achieved[_same_size_object_id(o)] = result["posed"]
            if len(achieved) != len(pairs):
                raise RuntimeError(
                    f"normalized {len(achieved)}/{len(pairs)} placed meshes"
                )
            # Commit only after the whole group is valid. Originals first move to sibling
            # backups so an os.replace failure can restore every already-committed member.
            for staged, final in staged_outputs:
                with tempfile.NamedTemporaryFile(
                    dir=str(Path(final).parent),
                    prefix=f".{Path(final).stem}.same_size_backup.",
                    suffix=Path(final).suffix,
                    delete=False,
                ) as backup_file:
                    backup = backup_file.name
                os.unlink(backup)
                backups[final] = backup
                os.replace(final, backup)
                os.replace(staged, final)
        except Exception as exc:  # noqa: BLE001 - fail closed, never apply a partial lock
            # Restore originals for any replacement that committed before the failure.
            for final, backup in backups.items():
                if os.path.exists(backup):
                    os.replace(backup, final)
            for staged, _final in staged_outputs:
                if os.path.exists(staged):
                    os.unlink(staged)
            failures[group_id] = str(exc)
            for o, _ in pairs:
                for field in _SAME_SIZE_GROUP_FIELDS:
                    o.pop(field, None)
            print(f"[same_size] {group_id}: normalization failed ({exc})")
            continue
        else:
            for backup in backups.values():
                if os.path.exists(backup):
                    os.unlink(backup)

        for o, _e in pairs:
            res = achieved[_same_size_object_id(o)]
            o["size"] = [round(s, 4) for s in res["size"]]
            o["obb_size"] = [round(s, 4) for s in res["obb_sorted"]]
            o["same_size"] = True
            o["same_size_applied"] = True
            o["same_size_group_status"] = "applied"
            o["scale_locked"] = True
            o["same_size_scale_locked"] = True
        unified.append(group_id)
        targets[group_id] = [round(float(t), 6) for t in target]
        print(
            f"[same_size] {group_id}: unified {len(pairs)} instances to shared OBB box "
            f"{[round(float(t), 3) for t in target]}"
        )

    activated = set(unified)
    for group in registry.get("groups", []):
        group_id = str(group.get("group_id") or "")
        if group_id in activated:
            group["status"] = "applied"
            group["normalization"] = {
                "stage": "usable_mesh",
                "target_obb_extents": targets[group_id],
                "member_count": len(group.get("active_members", [])),
                "measurement_frame": "metric_pristine_frozen_obb",
                "normalization_version": NORMALIZATION_VERSION,
                "scaled_outputs": ["mesh_glb", "pristine_glb"],
            }
        elif group_id in failures:
            group["status"] = "normalization_failed"
            group["active_members"] = []
            group["normalization"] = {
                "stage": "usable_mesh",
                "error": failures[group_id],
            }
    _write_same_size_registry(
        masks_json_path,
        registry_payload,
        registry,
        activated_group_ids=activated,
    )
    return unified


def generate_meshes(
    image_path: str,
    table: list[dict[str, Any]],
    out_dir: str,
    sam3d=None,
    R=None,
    *,
    masks_json_path: str,
    same_size_auto_lock: bool = False,
) -> list[dict[str, Any]]:
    """Reconstruct each unresolved object and place it at its metric transform.

    Generated rows receive ``mesh_glb`` + ``mesh_name`` using the downstream contract.
    The public SceneRig pipeline uses SAM3D for object reconstruction. The server is
    injectable via ``sam3d`` for tests and for the preprocessing prespawn path."""
    from lib.tools.geometry.agentic_mask import (
        Sam3dServer,
        slugify,
    )
    from lib.tools.geometry.mesh_place import (
        apply_world_rotation_to_glb,
        place_glb,
    )
    from lib.utils._path import SAM3D_PY

    meshes_dir = Path(out_dir) / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    # Condition SAM3D on the scene depth backend's existing point map by default (see
    # _moge_pointmap_path): one geometry source, and it skips SAM3D's internal MoGE-v1 pass.
    pointmap_npy = _moge_pointmap_path(out_dir, "sam3d")
    recon = sam3d  # injected server (kept named `sam3d` for back-compat)
    own = None  # server is started lazily below, only if a mesh must be reconstructed
    # Pass 1: reconstruct (or reuse) each object's raw + pristine mesh and read its pose.
    items = []  # (o, slug, raw, pristine, placed, rotation|None)
    try:
        for o in table:
            mp = o.get("mask_path")
            # Redetected instance (generative resegment): condition the mesh model on
            # the object's OWN edited image (occluders removed) + the full-object
            # re-segmented mask, under a distinct raw-cache tag.
            rd = o.get("redetect") or {}
            obj_image = image_path
            obj_pointmap = pointmap_npy  # depth condition for reconstruction
            rd_suffix = ""
            if rd.get("mask_path") and os.path.exists(rd["mask_path"]):
                mp = rd["mask_path"]
                obj_image = rd.get("edited_image") or image_path
                rd_suffix = "_rd"
                # Condition SAM3D on the object's OWN LingBot-refined depth point map (it
                # matches the edited image, dims and content) — passing None makes SAM3D fall
                # back to its internal MoGE-v1 estimate, less accurate than our refined depth.
                # If a redetect has no completed depth (vital-only reseg on the ORIGINAL
                # image), the scene point map still matches iff the edited image IS the
                # reference; otherwise skip (edited image with no matching depth).
                rdp = rd.get("points_npy")
                if rdp and os.path.exists(rdp):
                    obj_pointmap = rdp
                elif obj_image != image_path:
                    obj_pointmap = None
            if not mp or not os.path.exists(mp):
                continue
            slug = f"{slugify(o['category'])}_{o['instance']}"
            raw = meshes_dir / f"{slug}_raw{rd_suffix}.glb"
            pristine = meshes_dir / f"{slug}_pristine{rd_suffix}.glb"
            info = meshes_dir / f"{slug}_info{rd_suffix}.json"
            placed = meshes_dir / f"{slug}.glb"
            rotation = None
            if not (raw.exists() and raw.stat().st_size > 0):
                if recon is None:
                    own = recon = Sam3dServer(
                        SAM3D_PY, log_path=str(meshes_dir / "sam3d_server.log")
                    )
                resp = recon.reconstruct(
                    obj_image,
                    mp,
                    str(raw),
                    info_json=str(info),
                    pristine_glb=str(pristine),
                    pointmap_npy=obj_pointmap,  # refined MoGE-2 for redetects (see above)
                )
                if not resp:
                    continue
                rotation = resp.get("rotation")
            if (
                rotation is None and info.exists()
            ):  # cached raw: recover pose from sidecar
                try:
                    rotation = json.load(open(info)).get("rotation")
                except Exception:  # noqa: BLE001
                    rotation = None
            items.append((o, slug, raw, pristine, placed, rotation))
    finally:
        if own is not None:
            own.close()

    def _place_pristine180(pristine_glb, out_glb, center, size, use_obb=False):
        # Place the upright canonical (pristine) mesh with NO SAM3D pose and NO gravity tilt
        # (canonical Y-up -> Z-up only), anchored to the MoGE center/size, THEN yaw it 180deg about
        # world-up. The pristine loses azimuth (defaults to the SAM3D canonical facing), which for
        # the common case (an open laptop) points the screen BACKWARDS; the 180deg flip faces it
        # forward (validated on 8226). Used only by the toppling fallback.
        place_glb(
            str(pristine_glb),
            out_glb,
            center,
            size,
            R=None,
            backend="canonical_y_up",
            use_obb=use_obb,
        )
        apply_world_rotation_to_glb(
            out_glb,
            out_glb,
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],  # 180deg about world +Z
            center,
        )

    # Pass 2: place each object at the MoGE transform (raw SAM3D pose, always).
    for o, slug, raw, pristine, placed, _ in items:
        has_pristine = pristine.exists() and pristine.stat().st_size > 0
        # Scale by the oriented footprint box when available (rotation-robust; see
        # oriented_world_box / place_mesh_vertices). Falls back to the world-axis box on
        # older placement tables. Orientation is unchanged (still SAM3D's pose).
        scale_size = o.get("obb_size") or o["size"]
        use_obb = o.get("obb_size") is not None
        place_glb(
            str(raw),
            str(placed),
            o["center"],
            scale_size,
            R=R,
            backend="sam3d",
            use_obb=use_obb,
        )
        # Toppling fallback candidate: the SAME pristine+180 placement, dropped by the physics
        # settle ONLY if the raw placement topples (>45deg). None if no pristine mesh.
        if has_pristine:
            pcand = str(meshes_dir / f"{slug}_pcand.glb")
            _place_pristine180(
                pristine, pcand, o["center"], scale_size, use_obb=use_obb
            )
            o["pristine_glb"] = pcand
        else:
            o["pristine_glb"] = None
        o["mesh_glb"] = str(placed)
        o["mesh_name"] = f"obj_{slug}"
        o["mesh_provider"] = "sam3d"

    # Resolve the third/final eligibility gate over meshes that actually materialized,
    # then normalize each stable registry group. Compatibility locks activate on success.
    unify_same_size_scales(table, masks_json_path=masks_json_path, auto_lock=same_size_auto_lock)
    return table


# Reads (entries_json, blend) from argv to avoid str.format collisions with the
# f-string braces below.
_IMPORT_SCRIPT = r"""
import json, sys
import bpy
entries_path, blend, clearflag, basepath = sys.argv[-4:]
entries = json.load(open(entries_path))
bpy.ops.wm.open_mainfile(filepath=blend)
if clearflag.startswith("clear"):
    # empty wrappers) before re-importing. Filtering to type=="MESH" left the empty
    # parents behind, so every re-import (physics, ground-rest) leaked an "obj_*_1.NNN"
    # empty -- enough of them to overflow the scene-info object cap and hide table/wall.
    stale = [o for o in bpy.data.objects if o.name.startswith("obj_")]
    for o in stale:
        bpy.data.objects.remove(o, do_unlink=True)
    if clearflag == "clear_warn" and stale:  # the FIRST import expects an EMPTY blend
        print("WARNING [preprocess.import]: the shared blend already held %d obj_* before the "
              "first SAM3D import -- a polluted empty_scene.blend template or a reused blend. "
              "Cleared them (clear=True) so NO duplicates result, but fix the blend source." %
              len(stale))
n = 0
for e in entries:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=e["mesh_glb"])
    new = [o for o in bpy.data.objects if o not in before]
    for i, o in enumerate(new):
        o.name = e["mesh_name"] if i == 0 else "%s_%d" % (e["mesh_name"], i)
        # Rename the glTF-default material(s) ("Material_0", "Material_0.001", ...) to be
        # object-linked (obj_<name>_mat). Otherwise the texture agent can't tell a
        # reconstructed asset's material from a root surface and may grab e.g. "Material_0"
        # thinking it is the wall -- clearing it and wiping that asset's baked texture (a
        # black coffee machine rendered flat white). obj_*_mat is left alone downstream.
        if o.type == "MESH" and o.data is not None:
            for j, m in enumerate(o.data.materials):
                if m is not None:
                    m.name = ("%s_mat" % o.name) if j == 0 else ("%s_mat_%d" % (o.name, j))
    n += 1
bpy.ops.wm.save_as_mainfile(filepath=blend)
if basepath != "-":
    # Blend-frame BASELINE: matrix_world of every object as preprocess hands it over
    # (the glTF Y-up->Z-up conversion, identical for all, plus whatever the importer
    # chose). The delivered-vs-placed measure needs it to separate the COMPOSITION
    # delta (M_final @ M_base^-1) from this baseline; nothing can recover it from the
    # final blend once the agent has moved things.
    json.dump({e["mesh_name"]: [list(r) for r in bpy.data.objects[e["mesh_name"]].matrix_world]
               for e in entries if e["mesh_name"] in bpy.data.objects},
              open(basepath, "w"), indent=2)
print("MESHES_IMPORTED", n)
"""


def import_meshes_to_blend(
    blend_path: str,
    entries: list[dict[str, Any]],
    blender_cmd: str,
    clear: bool = False,
    warn_if_cleared: bool = False,
    base_matrix_path: Optional[str] = None,
) -> None:
    ""
    if not entries:
        return
    flag = ("clear_warn" if warn_if_cleared else "clear") if clear else "noclear"
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as ef:
        json.dump(entries, ef)
        entries_path = ef.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_IMPORT_SCRIPT)
        script_path = f.name
    try:
        subprocess.run(
            [
                blender_cmd,
                "--background",
                "--factory-startup",
                "--python",
                script_path,
                "--",
                entries_path,
                blend_path,
                flag,
                base_matrix_path or "-",
            ],
            check=True,
            cwd=REPO_ROOT,
        )
    finally:
        os.unlink(script_path)
        os.unlink(entries_path)


# --------------------------------------------------------------------------- #
# Full preprocessing orchestrator                                             #
# --------------------------------------------------------------------------- #
@_mesh_prespawn_failure_boundary
@_pseudo_gt_failure_boundary
def preprocess_scene(
    image_path: str,
    out_dir: str,
    blender_cmd: Optional[str] = None,
    blend_path: Optional[str] = None,
    model: str = "claude-opus-5",
    ignore_objects: Optional[list[str]] = None,
    same_size_categories: Optional[list[str]] = None,
    gt_depth: Optional[str] = None,
    gt_fov_x_deg: Optional[float] = None,
    vlm_physics: bool = True,
    vlm_physics_jobs: int = 5,
) -> dict[str, Any]:
    """MoGE estimate -> masking -> generative resegmentation -> placement -> meshes -> camera lock -> mesh
    import. Returns paths. SAM3D is the object mesh model. MoGE-2 is the default
    monocular metric-depth model. ``gt_depth`` (a metric Z-depth .npy aligned with
    the input image) bypasses MoGE entirely: gt_depth_estimate ingests it into the
    same ``<out>/moge/`` artifacts (intrinsics from an ``intrinsics.json`` next to
    the depth, ``gt_fov_x_deg`` override, else a MoGE-2 FOV-only fallback).
    The physics settle is PhysX + CoACD colliders, run incrementally against the
    persistent Isaac server. ``vlm_physics`` (default ON) estimates per-object
    material/mass/friction with the VLM right after reconstruction
    (physics/physics_vlm.json), consumed by the settle ladder, composition physics
    and the Isaac export."""
    depth_label = "gt" if gt_depth else "moge2"
    from lib.tools.geometry.agentic_mask import (
        Sam3Server,
        segment_scene,
    )
    from lib.utils._path import SAM3_PY

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _progress_path = out / "preprocess_progress.json"
    _progress_lock = threading.Lock()
    _progress_stop = threading.Event()
    _progress: dict[str, Any] = {
        "version": 1,
        "state": "running",
        "active_steps": ["depth", "segmentation"],
        "completed_steps": [],
        "total_steps": [
            "depth",
            "segmentation",
            "resegment",
            "canonicalize",
            "placement+graph",
            "meshes",
            *(["vlm_physics"] if vlm_physics else []),
            *(["settle"] if blender_cmd else []),
            *(["camera_lock"] if blender_cmd and blend_path else []),
            "pseudo_gt",
        ],
        "updated_at": _time.time(),
    }

    def _write_progress_unlocked() -> None:
        _progress["updated_at"] = _time.time()
        tmp = _progress_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(_progress, indent=2))
        os.replace(tmp, _progress_path)

    def _write_progress() -> None:
        with _progress_lock:
            _write_progress_unlocked()

    def _progress_advance(
        completed: str | list[str], active: str | list[str] | None
    ) -> None:
        completed_list = [completed] if isinstance(completed, str) else completed
        with _progress_lock:
            for step in completed_list:
                if step not in _progress["completed_steps"]:
                    _progress["completed_steps"].append(step)
            if active is None:
                _progress["active_steps"] = []
            else:
                _progress["active_steps"] = (
                    [active] if isinstance(active, str) else active
                )
            _write_progress_unlocked()

    def _heartbeat() -> None:
        while not _progress_stop.wait(30):
            _write_progress()

    class _ProgressLifetime:
        pass

    _progress_lifetime = _ProgressLifetime()

    def _abandon_progress() -> None:
        """Stop a leaked heartbeat and expose an exceptional preprocess exit."""
        _progress_stop.set()
        with _progress_lock:
            if _progress["state"] != "running":
                return
            _progress["state"] = "failed"
            _progress["active_steps"] = []
            _progress["error"] = "preprocessing exited before completion"
            _write_progress_unlocked()

    _progress_finalizer = weakref.finalize(_progress_lifetime, _abandon_progress)

    _write_progress()
    threading.Thread(target=_heartbeat, daemon=True).start()
    # Per-step wall clock: each _tick(label) closes the step started at the previous
    # tick. Written to stage_timings.json as "preprocess_steps" (the runner merges its
    # "preprocess" total in on top; the demo's timing panel renders both).
    _steps: dict[str, float] = {}
    _t_tick = [_time.time()]

    def _tick(label: str) -> None:
        now = _time.time()
        _steps[label] = round(_steps.get(label, 0.0) + now - _t_tick[0], 2)
        _t_tick[0] = now

    from lib.utils.common import normalize_input_image

    image_path = normalize_input_image(image_path, str(out / "input.png"))
    pseudo_gt_dir = out / "pseudo_gt"
    moge_json = out / "moge" / "moge.json"
    points_npy = out / "moge" / "points.npy"
    _parallel: dict[
        str, dict
    ] = {}  # background steps -> {secs, hidden, wait, ok, after, span}
    _depth = {"proc": None, "spawn": 0.0, "joined": moge_json.exists()}
    if not moge_json.exists():
        # Depth runs as a BACKGROUND subprocess: segmentation's server spawns +
        # proposer call (~90-120s) are depth-free, which hides the ~70s estimate.
        # Joined (points_ready) inside segment_scene right before its first points
        # use — and explicitly below for the cached-masks path. Unlike pseudo-GT,
        # depth is REQUIRED downstream, so a failed build raises.
        if gt_depth:
            script = "gt_depth_estimate.py"
            extra = ["--depth", gt_depth]
            if gt_fov_x_deg is not None:
                extra += ["--fov-x-deg", str(gt_fov_x_deg)]
        else:
            script = "moge_estimate.py"
            extra = []
        (out / "moge").mkdir(parents=True, exist_ok=True)
        _depth["spawn"] = _time.time()
        with open(out / "moge" / "estimate.log", "w") as _dlog:
            _depth["proc"] = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(REPO_ROOT, "lib", "tools", "geometry", script),
                    "--image",
                    image_path,
                    "--out-dir",
                    str(out),
                    *extra,
                ],
                stdout=_dlog,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": REPO_ROOT},
            )
        print(f"[preprocess] depth ({depth_label}) launched in the background")
    else:
        cached = json.load(open(moge_json)).get("backend", "moge2")
        if cached != depth_label:
            print(
                f"WARNING [preprocess]: cached {moge_json} was computed with depth "
                f"backend {cached!r} but {depth_label!r} was requested — reusing "
                f"the cache (delete <out>/moge/ to re-estimate)"
            )

    def _join_depth() -> None:
        """Wait for the background depth estimate (idempotent). Raises on failure —
        every later stage needs the point map."""
        if _depth["joined"]:
            return
        t0 = _time.time()
        rc = _depth["proc"].wait(timeout=600)
        wait_s = _time.time() - t0
        _depth["joined"] = True
        if rc != 0 or not moge_json.exists():
            tail = ""
            try:
                tail = "".join(open(out / "moge" / "estimate.log").readlines()[-6:])
            except OSError:
                pass
            raise RuntimeError(f"background depth estimate failed (rc={rc}):\n{tail}")
        wall_s = max(moge_json.stat().st_mtime - _depth["spawn"], 0.0)
        _parallel["depth"] = {
            "secs": round(wall_s, 2),
            "hidden": round(max(wall_s - wait_s, 0.0), 2),
            "wait": round(wait_s, 2),
            "ok": True,
            "after": "depth",
            "span": 1,
        }
        print(
            f"[preprocess] depth ready ({wall_s:.0f}s in background, "
            f"{wait_s:.0f}s waited)"
        )

    _tick("depth")  # spawn cost only; the wall time lands in preprocess_parallel
    masks_json = out / "masks" / "masks.json"
    # Reuse cached masks when a prior run already segmented this scene (mirrors the
    # MoGE + raw-mesh idempotency); re-masking is the slow VLM + SAM3 step. The
    # scene graph + gravity are recomputed from the masks below, deterministically.
    need_segment = not (
        masks_json.exists() and json.load(open(masks_json)).get("instances")
    )
    # Keep ONE SAM3 server warm across initial segmentation AND generative
    # re-segmentation: they run back-to-back with nothing GPU-heavy between, so sharing
    # avoids a load->teardown->reload of the SAM3 backbone. Create it when segmentation
    # is not cached and close it before canonicalize/mesh/settle. With cached masks,
    # generative_resegment lazily creates its own server when needed.
    shared_sam3 = None
    if need_segment:
        # segment_scene normally mkdirs masks/ itself; we spawn the server first, so
        # ensure the log dir exists before the server opens its log file.
        (out / "masks").mkdir(parents=True, exist_ok=True)
        shared_sam3 = Sam3Server(
            SAM3_PY, log_path=str(out / "masks" / "sam3_server.log")
        )
    try:
        if need_segment:
            segment_scene(
                image_path,
                str(out),
                points_npy=str(points_npy),
                points_ready=_join_depth,
                model=model,
                ignore_objects=ignore_objects,
                sam3=shared_sam3,
                same_size_categories=same_size_categories,
                room_mode=False,
            )

        _join_depth()  # cached-masks path never enters segment_scene's join point
        _tick("segmentation")
        _progress_advance(
            ["depth", "segmentation"],
            "resegment",
        )

        # Generative re-segmentation: pairwise occlusion DAG (touching pairs,
        # hierarchy-gated) + per-instance vital-part check; redetect = occluded-by-objects
        # OR vital-part-missing -> LanPaint + Qwen-Image-Edit MASKED occluder removal (+
        # transitive support descendants), SAM3 re-segmentation (part-aware prompt), and
        # LingBot depth completion over the edit hole. Records live in masks.json under each
        # instance's `redetect` key; downstream stages (placement, meshes, ICP, register)
        # key on that record.
        try:
            from lib.tools.geometry.depth_refine import refine_depth
            from lib.tools.geometry.generative_resegment import (
                generative_resegment as _run_generative_resegment,
            )

            records = _run_generative_resegment(str(out), model=model, sam3=shared_sam3)
            data_mj = json.load(open(masks_json))
            by_id = {
                f"{r['category']}#{r['instance']}": r
                for r in data_mj.get("instances", [])
            }
            for rid, rec in records.items():
                if rec.get("hole_mask") and not rec.get("points_npy"):
                    slug_rd = rid.replace("#", "").replace(" ", "_")
                    resp = refine_depth(
                        str(out), rec["edited_image"], rec["hole_mask"], slug_rd
                    )
                    if resp:
                        rec["points_npy"] = resp["out_points"]
                        by_id[rid]["redetect"] = rec
            with open(masks_json, "w") as f:
                json.dump(data_mj, f, indent=2)
            if records:
                print(
                    f"[preprocess] generative resegment: {len(records)} object(s) "
                    f"redetected ({', '.join(records)})"
                )
        except Exception as e:  # noqa: BLE001 - resegment is best-effort
            print(f"WARNING [preprocess]: generative resegment failed: {e}")
        _tick("resegment")
        _progress_advance("resegment", "canonicalize")
    finally:
        if shared_sam3 is not None:
            shared_sam3.close()

    # P1 (audit 07-28 §11.3): prespawn the ~163 s SAM3D boot in the background so it
    # hides behind the CPU/API-bound steps between here and generate_meshes (tier-3
    # merge VLM calls, connectivity prune, canonicalize, placement+graph). Placement
    # is deliberately AFTER the resegment block: its Qwen edit server needs ~40 GB
    # and fails boot next to large residents (07-27 "exited before ready"), so the
    # SAM3D must never coexist with it — same invariant as the shared-SAM3 close
    # above. A failed prespawn (or a fully mesh-cached rerun, which wastes one idle
    # boot) degrades to generate_meshes' original lazy in-loop spawn.
    _mesh_srv: dict[str, Any] = {
        "srv": None,
        "thread": None,
        "cancel_requested": False,
        "lock": threading.Lock(),
    }
    _mesh_spawn: Optional[threading.Thread] = None
    def _boot_mesh_server() -> None:
        try:
            from lib.tools.geometry.agentic_mask import Sam3dServer
            from lib.utils._path import SAM3D_PY

            (out / "meshes").mkdir(parents=True, exist_ok=True)
            server = Sam3dServer(
                SAM3D_PY, log_path=str(out / "meshes" / "sam3d_server.log")
            )
            with _mesh_srv["lock"]:
                cancelled = bool(_mesh_srv["cancel_requested"])
                if not cancelled:
                    _mesh_srv["srv"] = server
            if cancelled:
                server.close()
        except Exception as e:  # noqa: BLE001 - degrade to the lazy in-loop spawn
            print(f"[preprocess] SAM3D prespawn failed ({e}); meshes will boot inline")

    _mesh_spawn = threading.Thread(target=_boot_mesh_server, daemon=True)
    _mesh_srv["thread"] = _mesh_spawn
    # Register before start so an interrupt in the start call itself still reaches
    # the deterministic unwind boundary. The background closure shares this exact
    # state object and self-closes if cancellation wins the publication race.
    _ACTIVE_MESH_PRESPAWN.set(_mesh_srv)
    _mesh_spawn.start()
    print("[preprocess] SAM3D mesh server prespawned in the background (P1)")

    try:
        from lib.tools.geometry.agentic_mask import tier3_component_merge

        tier3_component_merge(str(masks_json), image_path, model=model)
    except Exception as e:  # noqa: BLE001 - best-effort, mirrors the resegment block
        print(f"WARNING [preprocess]: tier-3 component merge failed: {e}")

    _prune_disconnected(str(masks_json))

    # Room-mode occlusion revert (BEFORE canonicalization): a room-track proposal whose
    # work surface is mostly covered by objects is flipped back to the closeup
    # decomposition (table = root surface again). It updates scene-level routing.form and
    # routing.mode, which changes the support-chain canonical-root resolution.
    _revert_occluded_work_surface(str(masks_json))

    # Phase 0 runs before canonical-root selection because that path calls scene_form().
    # It validates/canonicalizes UNDER early enough for the policy-hard relation to promote
    # a tabletop route to the full furniture form.  With no world frame yet, every other
    # geometry-dependent VLM claim remains advisory; phase 1 below replaces those
    # provisional verdicts with metric evidence after canonicalization.
    try:
        phase0 = _adjudicate_surface_relationships(str(masks_json), None)
        counts = phase0.get("counts_by_enforcement", {})
        print(
            "[preprocess] relationship adjudication (pre-canonical): "
            f"{counts.get('hard', 0)} hard, {counts.get('advisory', 0)} advisory, "
            f"{counts.get('none', 0)} rejected"
        )
    except Exception as e:  # noqa: BLE001 - fail advisory except policy-hard UNDER
        print(f"[preprocess] pre-canonical relationship adjudication failed ({e})")
        try:
            _fail_closed_relationship_adjudication(str(masks_json), e)
        except Exception as fallback_error:  # noqa: BLE001 - primary failure is logged above
            print(
                "[preprocess] WARNING: relationship fail-closed write also failed "
                f"({fallback_error})"
            )

    # Canonicalize: gravity-align (+Z up) AND translate so the support-chain-selected main
    # support (floor/table) lies at z=0; the camera moves to T. Root resolution is
    # deterministic (no VLM). Stored in moge.json + threaded through placement / scene
    # graph / meshes / camera.
    # The zero-roll prior is a HAND-HELD assumption. A calibrated GT-depth run has a
    # fixed rig whose small roll is real and repeatable, and metric depth to measure it
    # with, so plumbing it upright would bake in a per-rig error. Decided from the
    # depth backend + intrinsics provenance that gt_depth_estimate already recorded.
    _mj_pre = json.load(open(moge_json))
    _keep_roll = mc.trust_measured_camera_roll(_mj_pre)
    R, T, canon = _compute_canonical_transform(
        str(masks_json),
        str(out / "moge" / "normal.npy"),
        str(points_npy),
        image_path,
        deroll=not _keep_roll,
    )

    # Finalize object support semantics before the main-support identity and pruning
    # become irreversible.  The first transform is only a provisional metric frame for
    # support scoring/referee evidence.  If exact-id/null/bare-category normalization
    # changes the support-chain winner, recompute the canonical frame from those final
    # upstream facts instead of failing after expensive downstream work.
    from lib.tools.geometry.agentic_mask import _make_vlm

    support_main_before = main_support_id(json.load(open(masks_json)))
    support_resolution = finalize_object_supports(
        str(masks_json),
        str(points_npy),
        R=R,
        T=T,
        vlm=(
            _make_vlm(model, effort="high")
            if os.environ.get("GRASE_SUPPORT_ADJUDICATE", "1") != "0"
            else None
        ),
    )
    support_main_after = main_support_id(json.load(open(masks_json)))
    if support_main_after != support_main_before:
        print(
            "[preprocess] finalized object supports changed the main-support "
            f"candidate {support_main_before!r} -> {support_main_after!r}; "
            "recomputing the canonical frame"
        )
        R, T, canon = _compute_canonical_transform(
            str(masks_json),
            str(out / "moge" / "normal.npy"),
            str(points_npy),
            image_path,
            deroll=not _keep_roll,
        )
    mj = json.load(open(moge_json))
    mj["gravity"] = {
        # Audit trail: WHY the roll was or was not plumbed, alongside roll_deg_pre.
        "deroll_applied": not _keep_roll,
        "R": R.tolist() if R is not None else None,
        "T": T.tolist() if T is not None else None,
        **(canon or {}),
    }
    with open(moge_json, "w") as f:
        json.dump(mj, f, indent=2)

    # Safety prune to the main support's scene (placement / scene-graph / meshes read the
    # pruned masks.json). The proposer is told to focus already; this catches over-inclusion.
    # world_c (canonical points) feeds the against backstop below.
    world_c = (
        mc.moge_points_to_world(np.load(points_npy), R, T) if R is not None else None
    )

    # Phase 1 relationship adjudication MUST precede relationship-driven pruning and
    # form reconciliation.  The canonical world provides metric evidence that replaces
    # phase 0's provisional non-UNDER verdicts.
    try:
        rel_summary = _adjudicate_surface_relationships(str(masks_json), world_c)
        counts = rel_summary.get("counts_by_enforcement", {})
        print(
            "[preprocess] relationship adjudication (pre-prune): "
            f"{counts.get('hard', 0)} hard, {counts.get('advisory', 0)} advisory, "
            f"{counts.get('none', 0)} rejected"
        )
        try:
            effective_form = _sync_relationship_form_to_moge(
                str(masks_json), str(moge_json)
            )
            if canon is not None and effective_form is not None:
                canon["form"] = effective_form
        except Exception as sync_error:  # noqa: BLE001 - audit copy is best-effort
            print(f"[preprocess] relationship form sync failed ({sync_error})")
    except Exception as e:  # noqa: BLE001 - retain phase-0 authority on best-effort failure
        print(
            f"[preprocess] relationship adjudication failed ({e}); "
            "keeping phase-0 authority"
        )
    protected_root_ids = {
        value for value in (support_main_after, (canon or {}).get("root")) if value
    }
    _prune_to_main_support(
        str(masks_json),
        support_main_after,
        scene_kind=json.load(open(masks_json)).get("scene_kind", "closeup"),
    )

    # Geometric `against` backstop: a support measurably flush to a wall gets its
    # `perpendicular` upgraded to `against` (yaw pin + flush constraint at the rules
    # gate). Needs the canonical frame to tell walls from supports; skipped when
    # canonicalization failed. Best-effort: a failure here keeps the proposer's rels.
    if R is not None:
        try:
            # ONE WALL PER PLANE: merge coplanar wall roots BEFORE the relationship
            # backstops and the graph build, so everything downstream (against/
            # corner rels, planes, prompt anchors) sees a single wall per plane.
            _merge_coplanar_walls(
                str(masks_json),
                world_c,
                exclude_ids=protected_root_ids,
            )
        except Exception as e:  # noqa: BLE001 - a merge failure keeps the split walls
            print(f"[preprocess] coplanar-wall merge failed ({e}); keeping walls as-is")
        try:
            # A sub-2% wall keeps its identity/relationships but loses geometric
            # authority; other non-anchor sliver roots keep the August 10 drop rule.
            _drop_sliver_roots(str(masks_json), exclude_ids=protected_root_ids)
        except Exception as e:  # noqa: BLE001 - a drop failure keeps the slivers
            print(f"[preprocess] sliver-root drop failed ({e}); keeping slivers")
        try:
            rel_summary = _adjudicate_surface_relationships(str(masks_json), world_c)
            counts = rel_summary.get("counts_by_enforcement", {})
            print(
                "[preprocess] relationship adjudication (final): "
                f"{counts.get('hard', 0)} hard, {counts.get('advisory', 0)} advisory, "
                f"{counts.get('none', 0)} rejected"
            )
            try:
                effective_form = _sync_relationship_form_to_moge(
                    str(masks_json), str(moge_json)
                )
                if canon is not None and effective_form is not None:
                    canon["form"] = effective_form
            except Exception as sync_error:  # noqa: BLE001 - audit copy is best-effort
                print(f"[preprocess] relationship form sync failed ({sync_error})")
        except Exception as e:  # noqa: BLE001 - keep phase-1 verdicts on failure
            print(f"[preprocess] final relationship adjudication failed ({e})")

    # All inventory-changing passes are complete. Validate every declared mask now,
    # then run one deterministic no-VLM support finalization over the exact retained
    # set. This resolves an upstream prune/sliver-induced null parent before the
    # read-only builder and proves that the semantic main identity remained stable.
    from lib.tools.geometry.inventory_contract import (
        validate_retained_mask_artifacts,
    )

    final_masks_payload = json.load(open(masks_json))
    validate_retained_mask_artifacts(final_masks_payload, out)
    final_support_resolution = finalize_object_supports(
        str(masks_json),
        str(points_npy),
        R=R,
        T=T,
        vlm=None,
    )
    final_support_main = main_support_id(json.load(open(masks_json)))
    if final_support_main != support_main_after:
        raise RuntimeError(
            "final upstream inventory changed the semantic main-support identity: "
            f"before_pruning={support_main_after!r}, "
            f"after_pruning={final_support_main!r}, "
            f"canonical_root={(canon or {}).get('root')!r}"
        )
    support_resolution["flags"] = final_support_resolution["flags"]

    _tick("canonicalize")
    _progress_advance("canonicalize", ["placement+graph", "pseudo_gt"])

    # Pseudo-GT process state is initialized here, but launch is deliberately delayed
    # until graph/placement/constraint validation has passed.  It still overlaps the
    # expensive reconstruction/physics path without wasting a GPU on an invalid scene.
    _pgt_proc = None
    _pgt_spawn = 0.0

    # build_scene_graph is deliberately read-only and lossless: support normalization
    # already ran upstream, before canonicalization/pruning were finalized, and the
    # builder now consumes the exact remaining inventory/support facts.
    graph = build_scene_graph(
        str(masks_json),
        str(points_npy),
        R=R,
        T=T,
    )
    retained_node_ids = {str(node.get("id")) for node in graph.get("nodes", [])}
    retained_nodes = {str(node.get("id")): node for node in graph.get("nodes", [])}
    graph["flags"] = [
        row
        for row in support_resolution["flags"]
        if row.get("obj") in retained_node_ids
    ]
    graph["support_adjudications"] = [
        row
        for row in support_resolution["support_adjudications"]
        if row.get("obj") in retained_node_ids
        and retained_nodes[row["obj"]].get("support") == row.get("applied_parent")
    ]
    # Rebuild composite.png from the FINAL instance list — after every masks.json
    # mutation (resegment, merges, prunes, coplanar walls, referee write-back), so
    # the demo's segmentation image and its 0-based labels always match the object
    # list (hangr: label 26 was the since-pruned wall#2).
    _regenerate_composite(str(masks_json), image_path)
    # Carry adjudicated surface RELATIONSHIPS + their audit summary to the initializer.
    # Only hard records are constraints; advisory records remain visible hypotheses and
    # rejected records remain audit-only.
    relationships_payload = json.load(open(masks_json))
    graph["relationships"] = relationships_payload.get("relationships", [])
    graph["relationship_adjudication"] = relationships_payload.get(
        "relationship_adjudication", {}
    )
    graph["relationship_audit"] = relationships_payload.get("relationship_audit", [])
    # Persist the support-chain vote winner as the ONE authoritative identity consumed
    # by initializer text and its tinted overlay.  Canonicalization usually grounds the
    # same root, but it is allowed a geometric fallback and therefore is not an identity
    # selector.  Writing both graph-level id and one node flag makes disagreement or a
    # partially staged artifact detectable instead of silently selecting nearest-z.
    masks_payload = json.load(open(masks_json))
    stamp_main_support(graph, masks_payload)
    from lib.tools.geometry.inventory_contract import validate_scene_graph_inventory

    # Identity/topology/relationship parity is cheap and must fail before placement,
    # yaw compilation, reconstruction, physics, or background GPU work begins.
    validate_scene_graph_inventory(masks_payload, graph, allow_runtime_additions=False)

    same_size_registry = _resolve_same_size_after_mask_pruning(
        str(masks_json), image_path, same_size_categories, model
    )
    print(
        "[preprocess] same-size resolution: "
        f"{same_size_registry.get('mode')} mode, "
        f"{len(same_size_registry.get('groups', []))} group(s)"
    )
    table = build_placement_table(
        str(masks_json),
        str(points_npy),
        str(moge_json),
        R=R,
        T=T,
    )
    _revalidate_same_size_placement(str(masks_json), table)
    from lib.tools.geometry.inventory_contract import validate_placement_inventory

    validate_placement_inventory(graph, table)
    with open(out / "scene_graph.json", "w") as f:
        json.dump(graph, f, indent=2)
    # Freeze source-only yaw evidence after the FINAL retained masks, support identity,
    # calibrated camera, and canonical point map are stable.  This artifact never reads
    # the generated Blender scene: runtime root transactions may recompile constraints
    # against it, but must not recompute or move its source target.
    from lib.tools.geometry.main_support_yaw_observation import (
        build_main_support_yaw_observation,
        write_main_support_yaw_observation,
    )

    final_masks_payload = json.load(open(masks_json))
    camera_config = (
        mc.camera_config_from_moge(mj, R=R, T=T)
        if R is not None and T is not None
        else None
    )
    yaw_observation = build_main_support_yaw_observation(
        out,
        graph=graph,
        masks_payload=final_masks_payload,
        world_points=world_c,
        camera_config=camera_config,
        input_path=image_path,
    )
    yaw_observation_path = write_main_support_yaw_observation(out, yaw_observation)
    print(f"[preprocess] main-support yaw observation: {yaw_observation_path}")
    from lib.tools.geometry.relationship_constraints import (
        write_initializer_constraints,
    )

    constraints_path = write_initializer_constraints(
        out, graph, yaw_observation=yaw_observation
    )
    print(f"[preprocess] initializer constraints: {constraints_path}")
    _tick("placement+graph")
    _progress_advance(
        "placement+graph",
        ["meshes", "pseudo_gt"],
    )

    # Adopt the P1-prespawned server (join returns as soon as the boot thread —
    # itself bounded by the server's ready_timeout — finishes). generate_meshes
    # never closes an injected server, so close it here as soon as meshing is
    # done: the ~13 GB must be gone before vlm_physics renders and settle.
    if _mesh_spawn is not None:
        _mesh_spawn.join()
    try:
        generate_meshes(
            image_path,
            table,
            str(out),
            R=R,
            sam3d=_mesh_srv["srv"],
            masks_json_path=str(masks_json),
            same_size_auto_lock=False,
        )
    finally:
        _cleanup_mesh_prespawn(_mesh_srv)
        _ACTIVE_MESH_PRESPAWN.set(None)
    from lib.tools.geometry.inventory_contract import validate_object_materialization

    validate_object_materialization(graph, table, artifact_root=out)
    _tick("meshes")
    _progress_advance(
        "meshes",
        ["vlm_physics", "pseudo_gt"]
        if vlm_physics
        else (["settle", "pseudo_gt"] if blender_cmd else ["pseudo_gt"]),
    )

    # Pseudo-GT novel views launch only after graph, placement, constraints, and every
    # retained mesh have passed their exact boundaries. It still overlaps VLM physics
    # and the normally-longer settle path, so invalid scenes cannot orphan GPU work.
    from lib.tools.geometry.pseudo_gt_contract import (
        PseudoGTContractError,
        load_complete_pseudo_gt,
    )

    _pgt_ready = False
    if not _pgt_ready:
        _pgt_nvs_backend = os.environ.get("GRASE_NVS_BACKEND", "sharp_gpt")
        pseudo_gt_dir.mkdir(parents=True, exist_ok=True)
        _pgt_spawn = _time.time()
        with open(pseudo_gt_dir / "build.log", "w") as _pgt_log:
            _pgt_proc = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(
                        REPO_ROOT, "lib", "tools", "geometry", "novel_view.py"
                    ),
                    "--pseudo-gt",
                    "--moge-dir",
                    str(out / "moge"),
                    "--image",
                    image_path,
                    "--out-dir",
                    str(pseudo_gt_dir),
                    "--nvs-backend",
                    _pgt_nvs_backend,
                ],
                stdout=_pgt_log,
                stderr=subprocess.STDOUT,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": REPO_ROOT},
            )
            # Register before leaving the log-file context: even a close-time I/O
            # failure or an immediately delivered interrupt must reach the outer
            # exception boundary with the launched child already discoverable.
            _ACTIVE_PSEUDO_GT_PROCESS.set(_pgt_proc)
        print("[preprocess] pseudo-GT build launched in the background")

    def _abort_pseudo_gt() -> None:
        """Stop background GPU work before propagating a fatal later-stage error."""

        _terminate_and_reap_process(_pgt_proc)
        _ACTIVE_PSEUDO_GT_PROCESS.set(None)

    if vlm_physics:
        # VLM physics estimation (material + mass + table friction) right after
        # reconstruction, BEFORE settle — the ladder then drops bodies with the
        # estimated friction/mass instead of the flat 250 kg/m^3 / 0.6 defaults.
        # Canonical file: physics/physics_vlm.json (pipeline names, OBB-extents
        # mass anchor); every consumer (settle, composition, isaac export) falls
        # back to the heuristic defaults when it is missing.
        from lib.tools.geometry.physics_estimate import (
            estimate_scene,
            objects_from_table,
            write_failed_manifest,
        )

        expected_physics_objects = [
            {"name": o["mesh_name"]} for o in table if o.get("mesh_name")
        ]
        try:
            estimate_scene(
                objects_from_table(table, out),
                out / "physics/physics_vlm.json",
                model=model,
                jobs=vlm_physics_jobs,
                expected_names=[o["name"] for o in expected_physics_objects],
            )
        except Exception as e:  # noqa: BLE001 - estimation must not abort preprocess
            print(
                f"[preprocess] vlm physics estimation failed ({e}); "
                "heuristic defaults downstream"
            )
            write_failed_manifest(
                expected_physics_objects,
                out / "physics/physics_vlm.json",
                model,
                str(e),
            )
        _tick("vlm_physics")
        _progress_advance(
            "vlm_physics",
            ["settle", "pseudo_gt"] if blender_cmd else ["pseudo_gt"],
        )
    else:
        from lib.tools.geometry.physics_estimate import (
            objects_from_table,
            write_disabled_manifest,
        )

        write_disabled_manifest(
            [{"name": o["mesh_name"]} for o in table if o.get("mesh_name")],
            out / "physics/physics_vlm.json",
            model=model,
        )

    # Physics settle (before init): persistent Isaac server, DFS build-up in support
    # order with interleaved per-object ICP + re-drop + certify, baking the settling
    # rotation + grounding z-shift into the placed GLB + placement table so the
    # initializer sees objects in stable resting poses. ``disable_icp`` skips the
    # photo pose-alignment (settle + certify still run).
    if blender_cmd:
        from lib.tools.geometry.inventory_contract import InventoryContractError
        from lib.tools.geometry.physics import incremental_settle

        try:
            inc = incremental_settle(
                table,
                graph,
                blender_cmd,
                str(out),
                R=R,
                T=T,
                disable_icp=False,
                icp_freeze=[],
            )
            fell = sum(1 for r in inc.values() if r.get("fell"))
            print(
                f"[preprocess] incrementally settled {len(inc)} objects"
                + (f" ({fell} accepted fallen)" if fell else "")
            )
        except InventoryContractError:
            # Inventory loss is not a recoverable physics failure.  Continuing would
            # hand the agents a smaller object set than the final masks/graph promise.
            _abort_pseudo_gt()
            raise
        except Exception as e:  # noqa: BLE001 - ordinary settle failure keeps placement
            print(
                f"[preprocess] incremental settle failed ({e}); keeping MoGE placement"
            )
        _tick("settle")
        _progress_advance(
            "settle",
            ["camera_lock", "pseudo_gt"]
            if blender_cmd and blend_path
            else ["pseudo_gt"],
        )
    _timings_path = out / "stage_timings.json"
    try:
        _existing = json.load(open(_timings_path)) if _timings_path.exists() else {}
    except Exception:  # noqa: BLE001
        _existing = {}
    _existing["preprocess_steps"] = _steps
    if _parallel:
        _existing.setdefault("preprocess_parallel", {}).update(_parallel)
    with open(_timings_path, "w") as f:
        json.dump(_existing, f, indent=2)
    placement_path = out / "placement.json"
    with open(placement_path, "w") as f:
        json.dump({"objects": table}, f, indent=2)
    if blender_cmd and blend_path:
        try:
            apply_camera_to_blend(blend_path, str(moge_json), blender_cmd)
            entries = [
                {"mesh_glb": o["mesh_glb"], "mesh_name": o["mesh_name"]}
                for o in table
                if o.get("mesh_glb")
            ]
            # clear=True makes the first import idempotent if an output directory is reused.
            import_meshes_to_blend(
                blend_path,
                entries,
                blender_cmd,
                clear=True,
                warn_if_cleared=True,
                base_matrix_path=str(out / "physics" / "blend_base.json"),
            )
        except BaseException:
            _abort_pseudo_gt()
            raise
        _progress_advance("camera_lock", ["pseudo_gt"])
    # Join the background pseudo-GT build. When the overlap window (placement/
    # meshes/settle/pose_match/camera) exceeded the build time, the wait here is
    # ~0 — the honest critical-path cost. Recorded for the demo's timing panel:
    # preprocess_steps.pseudo_gt_wait = residual on the critical path;
    # preprocess_parallel.pseudo_gt = the subprocess's own wall time.
    if _pgt_proc is not None:
        _t_join = _time.time()
        try:
            rc = _pgt_proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            _pgt_proc.kill()
            _pgt_proc.wait(timeout=10)
            rc = -1
        wait_s = _time.time() - _t_join
        cams = pseudo_gt_dir / "cameras.json"
        end = cams.stat().st_mtime if cams.exists() else _time.time()
        wall_s = max(end - _pgt_spawn, 0.0)
        validation_error = None
        if rc == 0:
            try:
                load_complete_pseudo_gt(
                    pseudo_gt_dir,
                    source_image=image_path,
                    rebase_paths=True,
                )
                _pgt_ready = True
            except PseudoGTContractError as exc:
                validation_error = str(exc)
        if _pgt_ready:
            print(
                f"[preprocess] pseudo-GT ready ({wall_s:.0f}s in background, "
                f"{wait_s:.0f}s on the critical path)"
            )
        else:
            detail = (
                f"; incomplete bundle: {validation_error}" if validation_error else ""
            )
            print(
                f"[preprocess] pseudo-GT build failed (rc={rc}, see "
                f"{pseudo_gt_dir / 'build.log'}{detail}); composition falls back to "
                "source view"
            )
        try:
            _tj = json.load(open(_timings_path)) if _timings_path.exists() else {}
        except Exception:  # noqa: BLE001
            _tj = {}
        _tj.setdefault("preprocess_steps", {})["pseudo_gt_wait"] = round(wait_s, 2)
        _keys = [k for k in _steps if not k.endswith("_wait")]
        _span = (
            len(_keys) - _keys.index("canonicalize") - 1
            if "canonicalize" in _keys
            else 0
        )
        _tj.setdefault("preprocess_parallel", {})["pseudo_gt"] = {
            "secs": round(wall_s, 2),
            "hidden": round(max(wall_s - wait_s, 0.0), 2),
            "wait": round(wait_s, 2),
            "ok": _pgt_ready,
            "after": "canonicalize",
            "span": _span,
        }
        with open(_timings_path, "w") as f:
            json.dump(_tj, f, indent=2)
    # The child has been waited/reaped (or no build was launched). Clear the
    # function-level exception boundary before later artifact/progress validation.
    _ACTIVE_PSEUDO_GT_PROCESS.set(None)

    # Enforce the lossless inventory contract in the preprocessing API itself, not
    # only in the CLI runner.  This catches direct library callers and ensures the
    # progress marker cannot say ``complete`` for a graph/placement/mesh divergence.
    from lib.tools.geometry.inventory_contract import validate_scene_artifacts

    validate_scene_artifacts(
        out, allow_runtime_additions=False, allow_runtime_inventory=False
    )

    if _pgt_ready:
        _progress_advance("pseudo_gt", None)
    with _progress_lock:
        if not _pgt_ready:
            _progress["skipped_steps"] = ["pseudo_gt"]
        _progress["state"] = "complete"
        _progress["active_steps"] = []
        _write_progress_unlocked()
    _progress_stop.set()
    _progress_finalizer.detach()

    return {
        "moge_json": str(moge_json),
        "masks_json": str(out / "masks" / "masks.json"),
        "main_support_yaw_observation_json": str(yaw_observation_path),
        "initializer_constraints_json": str(constraints_path),
        "placement_json": str(placement_path),
        "objects": table,
    }
