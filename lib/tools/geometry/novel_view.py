""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

# GPT-Image inpaint is an external API call: retry a few times (transient failures are common)
# before degrading to the raw splat, so a momentary hiccup doesn't silently produce a pseudo-GT
# that is just the point-cloud render.
_GPT_ATTEMPTS = 3
_GPT_BACKOFF_S = 2.0

from lib.tools.geometry import moge_camera as mc
from lib.tools.geometry.gpt_inpaint import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    POLISH_PROMPT,
    complete,
    polish,
)
from lib.tools.geometry.pseudo_gt_contract import (
    PSEUDO_GT_VIEWS_AZ_EL as NOVEL_VIEWS_AZ_EL,
)
from lib.tools.geometry.pseudo_gt_contract import (
    view_tag,
)
from lib.tools.geometry.render_pointcloud import _parse_pose, render_view

_POLISH_MAX_SIDE = 2048
_DRIFT_FLAG = 25.0

# Derived, so consumers (the blender exec tool, prompts) never hardcode the views and drift
# from NOVEL_VIEWS_AZ_EL above. AZIMUTHS/ELEVATIONS feed the tool-arg enums; the two NOTEs are
# the shared prose describing the choice + what pseudo-GT means.
NOVEL_VIEW_AZIMUTHS: list[float] = sorted({az for az, _ in NOVEL_VIEWS_AZ_EL})
NOVEL_VIEW_ELEVATIONS: list[float] = sorted({el for _, el in NOVEL_VIEWS_AZ_EL})

# Shared tail: which views exist, what the offsets mean, and which reference comes back.
# True for BOTH notes below, since both describe what happens once a view IS chosen.
_NOVEL_VIEW_BODY: str = (
    "It must be one of the "
    f"{len(NOVEL_VIEWS_AZ_EL)} pre-rendered views: "
    + ", ".join(
        (f"({az:g},{el:g})=source" if (az, el) == (0.0, 0.0) else f"({az:g},{el:g})")
        for az, el in NOVEL_VIEWS_AZ_EL
    )
    + " (each varies ONE axis — do NOT combine a nonzero azimuth with a nonzero elevation). These (azimuth, elevation) values are ORBIT OFFSETS (deltas) FROM the reference view, not "
    "absolute angles — (0,0)=source IS the reference view. The scene is then rendered from that "
    "camera and, when trustworthy, the matching reference image is returned "
    "alongside: at the SOURCE view (0,0) the reference is the REAL ground-truth target photo "
    "(exact); at any NOVEL view it is that view's PSEUDO-GROUND-TRUTH (a generative completion "
    "of the lifted point cloud, a soft target). A completion flagged for excessive drift or "
    "generation failure is withheld and the render comes back with a warning instead. "
    "When a reference is present, compare your render to it and adjust to match. "
    "A novel view's pseudo-GT is returned ONLY with the render that requested it and is not kept "
    "in your context — call again if you need to compare against that view later."
)

# execute_and_evaluate: a viewpoint is always resolved (omitted -> (0,0)), so the paired
# reference is unconditional.
NOVEL_VIEW_NOTE: str = (
    " You MAY also pass a viewpoint via (azimuth, elevation); it DEFAULTS to (0,0)=source "
    "(the reference view) if omitted. " + _NOVEL_VIEW_BODY
)

# render_current_scene: the reference is CONDITIONAL — omitting BOTH args renders the locked
# source view alone. Cheap on purpose: the source-view target photo is already pinned at the
# top of the agent's context, so the bare render is the one case where the pair adds nothing.
NOVEL_VIEW_NOTE_ON_REQUEST: str = (
    " Pass a viewpoint via (azimuth, elevation) to get that view's REFERENCE image alongside your "
    "render; a lone azimuth or elevation implies 0 on the other axis. Omit BOTH and you get a "
    "bare render of the source view with NO reference attached — compare it against the target "
    "photo at the top of this conversation. " + _NOVEL_VIEW_BODY
)

# MoGE OpenCV frame: +X right, +Y down, +Z forward, so world-up is -Y.
_MOGE_UP = np.array([0.0, -1.0, 0.0])


def _rot_axis(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix about a unit ``axis`` by ``deg`` degrees."""
    a = axis / (np.linalg.norm(axis) or 1.0)
    t = np.radians(deg)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * (k @ k)


def _moge_up(R) -> np.ndarray:
    ""
    if R is None:
        return _MOGE_UP
    g = np.asarray(R, dtype=np.float64)[2]  # g_world = main-support normal
    return np.array([-g[0], -g[2], -g[1]])


def _axis_plane_pivot(R, T) -> "np.ndarray | None":
    """Orbit pivot = where the camera's viewing axis meets the main-support plane (the table
    point at the centre of the view). In the MoGE frame the optical axis is ``(0,0,1)`` and the
    support plane is ``up . p = z_root`` with ``up = M @ g_world`` and ``z_root = -T[2]``, so the
    intersection is ``(0, 0, z_root / up_z)``. None (caller falls back) when gravity is unknown,
    the axis is ~parallel to the plane (no usable hit), or the plane lies behind the camera."""
    if R is None or T is None:
        return None
    up = _moge_up(R)
    uz = float(up[2])
    if abs(uz) < 1e-6:  # axis ~parallel to the plane
        return None
    t = (
        -float(np.asarray(T, dtype=np.float64)[2]) / uz
    )  # z_root / up_z  (z_root = -T[2])
    if not np.isfinite(t) or t <= 0:  # plane behind / at the camera
        return None
    return np.array([0.0, 0.0, t])


def _root_mask(moge_dir: str, root_id) -> "np.ndarray | None":
    """Boolean mask of the main supporting surface (``root_id`` = ``'category#inst'``), read
    from ``<scene>/masks/``. None if it can't be resolved (caller falls back)."""
    if not root_id:
        return None
    masks_dir = Path(moge_dir).parent / "masks"
    try:
        instances = json.load(open(masks_dir / "masks.json")).get("instances", [])
    except Exception:  # noqa: BLE001 - no masks -> caller falls back to the all-points pivot
        return None
    for inst in instances:
        if f"{inst.get('category')}#{inst.get('instance')}" == root_id:
            p = Path(inst.get("mask_path", ""))
            if not p.exists():
                p = masks_dir / p.name  # mask_path is repo-relative; resolve
            return (np.load(p) > 0) if p.exists() else None
    return None


def _root_pivot(
    points: np.ndarray, root_mask: np.ndarray, up: np.ndarray, min_pts: int = 200
) -> "np.ndarray | None":
    """Orbit pivot = robust centre of the MAIN-SUPPORT surface in the MoGE frame. Median of the
    root mask's points, after rejecting the points farthest from the support plane (object /
    mask-edge depth outliers that leaked into the mask), so the pivot snaps ONTO the table.
    None when the mask is too small / absent (caller falls back to the all-points median)."""
    m = root_mask & np.isfinite(points).all(axis=2)
    pts = points[m].astype(np.float64)
    if len(pts) < min_pts:
        return None
    n = up / (np.linalg.norm(up) or 1.0)
    c = np.median(pts, axis=0)
    off = np.abs((pts - c) @ n)  # |distance| to the plane through c
    keep = off <= max(float(np.percentile(off, 80)), 1e-9)  # drop the farthest-off 20%
    return np.median(pts[keep], axis=0)  # centroid of the on-plane inliers


def _scene_pivot(moge_dir: str) -> np.ndarray:
    """All-points fallback pivot in the MoGE OpenCV frame: median of valid points."""
    pts = np.load(Path(moge_dir) / "points.npy").astype(np.float64)
    flat = pts.reshape(-1, 3)
    flat = flat[np.isfinite(flat).all(axis=1)]
    return np.median(flat, axis=0) if len(flat) else np.array([0.0, 0.0, 1.0])


def _choose_pivot(axis_pivot, centroid, k: float = 1.5):
    """Pick the orbit pivot. Prefer the view-centred ``axis_pivot`` (camera axis ∩ support plane),
    but the axis∩plane hit blows up at GRAZING angles (small elevation -> the optical axis is ~
    parallel to the table, so the hit runs far behind the scene). Fall back to the bounded support
    centroid when the axis pivot is implausibly far (‖axis‖ > k·‖centroid‖) or absent."""
    if axis_pivot is None:
        return centroid
    if centroid is None:
        return axis_pivot
    return (
        axis_pivot
        if (float(np.linalg.norm(axis_pivot)) <= k * float(np.linalg.norm(centroid)))
        else centroid
    )


def novel_view_cameras(moge_dir: str, R=None, T=None) -> list[dict]:
    """The fixed pseudo-GT cameras, each expressed in BOTH frames so the point-cloud
    render (MoGE OpenCV) and the Blender scene render (GRASE canonical) view the *same*
    viewpoint. Orbits the reference camera by (az, el) about the MAIN-SUPPORT plane normal
    (gravity up, from ``R``) around the pivot where the camera's viewing axis meets that plane
    (the table point at the centre of the view) — so the views level around the table rather
    than tilting with the camera. Falls back to the support-mask centre, then the all-points
    median, when gravity is unavailable.

    Returns one dict per view: ``{az, el, tag, cam2world_moge (4x4 list), location,
    look_at, lens}`` where ``location``/``look_at`` are canonical (gravity-aligned, with
    ``R``/``T`` applied) for the Blender camera and ``lens`` is the source focal length.
    """
    with open(Path(moge_dir) / "moge.json") as f:
        mj = json.load(f)
    lens = mc.lens_from_fov(float(mj["fov_x_deg"]))
    res_x, res_y = (
        int(mj["image_width"]),
        int(mj["image_height"]),
    )  # source aspect ratio
    up = _moge_up(R)  # orbit about the main-support normal (gravity up)
    # Orbit pivot = where the camera's viewing axis meets the support plane (the table point at the
    # view centre) WHEN that hit is sane; at grazing angles it runs far behind the scene, so cap it
    # against the bounded support-mask centre (then the all-points median when the mask is absent).
    points = np.load(Path(moge_dir) / "points.npy").astype(np.float64)
    root_mask = _root_mask(moge_dir, (mj.get("gravity") or {}).get("root"))
    centroid = (
        _root_pivot(points, root_mask, up)
        if root_mask is not None and root_mask.shape == points.shape[:2]
        else None
    )
    if centroid is None:
        centroid = _scene_pivot(moge_dir)
    pivot = _choose_pivot(_axis_plane_pivot(R, T), centroid)
    eye0 = np.zeros(3)  # reference camera sits at the MoGE-frame origin
    back0 = eye0 - pivot  # pivot -> camera, the orbit radius vector
    fwd0 = np.array(
        [0.0, 0.0, 1.0]
    )  # reference forward (MoGE frame IS the source camera)
    d0 = float(np.linalg.norm(back0)) or 1.0
    out: list[dict] = []
    for az, el in NOVEL_VIEWS_AZ_EL:
        Q = _rot_axis(up, az)  # swing horizontally
        # Horizontal tilt axis oriented so +elevation raises the camera (looks DOWN at the
        # scene) and -elevation lowers it: cross(back, up), not cross(up, back).
        right = np.cross(Q @ back0, up)
        if np.linalg.norm(right) > 1e-9:
            Q = _rot_axis(right, el) @ Q  # then tilt vertically
        eye = pivot + Q @ back0
        cam2world = np.eye(4)
        cam2world[:3, :3] = Q  # Q @ identity: the rotated source-camera orientation
        cam2world[:3, 3] = eye
        out.append(
            {
                "az": az,
                "el": el,
                "tag": view_tag(az, el),
                "cam2world_moge": cam2world.tolist(),
                "location": list(mc.moge_to_world(eye, R, T)),
                "look_at": list(mc.moge_to_world(eye + Q @ fwd0 * d0, R, T)),
                "lens": float(lens),
                "res_x": res_x,
                "res_y": res_y,  # source aspect so renders match the pseudo-GT
            }
        )
    return out


def _sharp_renders(moge_dir: str, image_path: str, out_dir: str) -> None:
    """Raw SHARP renders for every novel camera in ``out_dir``/cameras.json.

    Shells out to the SHARP venv (own py3.13 env, like the other third_party
    backends); one subprocess per scene, one model load, all views. Raises on
    failure so the caller can fall back to the splat backend.
    """
    import os
    import subprocess
    import sys

    from lib.utils._path import SHARP_PY

    repo_root = str(Path(__file__).resolve().parents[3])
    subprocess.run(
        [
            SHARP_PY,
            str(Path(repo_root) / "lib" / "tools" / "geometry" / "sharp_render.py"),
            "--image", image_path,
            "--moge-json", str(Path(moge_dir) / "moge.json"),
            "--cameras-json", str(Path(out_dir) / "cameras.json"),
            "--out-dir", str(out_dir),
        ],
        check=True,
        timeout=900,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONPATH": repo_root},
    )


def _polish_view(vdir: Path, prompt: str, model: str) -> None:
    """One view: downscale the raw render, gpt-polish it (with retries), composite
    nothing — the output IS the completion — then measure observed-content drift.

    Failure semantics mirror the splat backend: after ``_GPT_ATTEMPTS`` the raw
    render is kept as the completion artifact and ``gpt_fallback.txt`` flags it.
    Drift beyond ``_DRIFT_FLAG`` writes ``drift_flag.txt``. Both states are recorded
    as untrusted in cameras.json, so the agent never receives the bad reference.
    """
    from PIL import Image

    completed = vdir / "completed.png"
    raw = Image.open(vdir / "render_raw.png").convert("RGB")
    src_size = raw.size
    if max(raw.size) > _POLISH_MAX_SIDE:
        sc = _POLISH_MAX_SIDE / max(raw.size)
        raw = raw.resize((round(raw.width * sc), round(raw.height * sc)), Image.LANCZOS)
    inp = vdir / "polish_input.png"
    raw.save(inp)

    marker = vdir / "gpt_fallback.txt"
    if marker.exists():
        marker.unlink()
    for stale in (vdir / "drift.json", vdir / "drift_flag.txt"):
        if stale.exists():
            stale.unlink()
    last_err: Exception | None = None
    for attempt in range(1, _GPT_ATTEMPTS + 1):
        try:
            polish(str(inp), str(completed), prompt=prompt, model=model)
            last_err = None
            break
        except Exception as e:  # noqa: BLE001 - external image API; retry then degrade
            last_err = e
            print(
                f"[pseudo_gt] gpt polish failed for {vdir.name} "
                f"(attempt {attempt}/{_GPT_ATTEMPTS}): {type(e).__name__}: {e}"
            )
            if attempt < _GPT_ATTEMPTS:
                time.sleep(_GPT_BACKOFF_S * attempt)
    if last_err is not None:
        shutil.copy(vdir / "render_raw.png", completed)
        marker.write_text(
            "gpt polish failed; completed.png is the raw SHARP render (NOT a GPT "
            f"completion).\nlast error: {type(last_err).__name__}: {last_err}\n"
        )
        return

    # completed_srcres.png at the source resolution for consumers that need it;
    # completed.png keeps the API's native output like the splat backend does.
    out = Image.open(completed).convert("RGB")
    out.resize(src_size, Image.LANCZOS).save(vdir / "completed_srcres.png")

    hole = np.asarray(Image.open(vdir / "hole_mask.png").convert("L").resize(raw.size)) < 128
    g = np.asarray(out.resize(raw.size), dtype=np.float64)
    drift = float(np.abs(g - np.asarray(raw, dtype=np.float64))[~hole].mean())
    (vdir / "drift.json").write_text(json.dumps({"keep_drift": round(drift, 2)}))
    if drift > _DRIFT_FLAG:
        (vdir / "drift_flag.txt").write_text(
            f"observed-content drift {drift:.1f}/255 exceeds {_DRIFT_FLAG}: the polish "
            "likely re-imagined parts of the visible scene, not just the holes.\n"
        )
        print(f"[pseudo_gt] DRIFT FLAG {vdir.name}: {drift:.1f}/255")


def build_pseudo_gt(
    moge_dir: str,
    image_path: str,
    out_dir: str,
    R=None,
    T=None,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    point_radius: int = 1,
    backend: str = "gpt_splat",
) -> str:
    ""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cams = novel_view_cameras(moge_dir, R, T)

    if backend == "sharp_gpt":
        from concurrent.futures import ThreadPoolExecutor

        from PIL import Image

        # Poses first: sharp_render reads cameras.json (re-dumped with pseudo_gt
        # paths at the end, same as the splat backend).
        with open(out / "cameras.json", "w") as f:
            json.dump({"views": cams}, f, indent=2)
        try:
            _sharp_renders(moge_dir, image_path, str(out))
        except Exception as e:  # noqa: BLE001 - degrade to the proven splat backend
            print(
                f"[pseudo_gt] SHARP render failed ({type(e).__name__}: {e}); "
                "falling back to the gpt_splat backend"
            )
            backend = "gpt_splat"
        else:
            novel = []
            for cam in cams:
                vdir = out / cam["tag"]
                vdir.mkdir(parents=True, exist_ok=True)
                completed = str(vdir / "completed.png")
                if cam["az"] == 0.0 and cam["el"] == 0.0:
                    Image.open(image_path).convert("RGB").save(completed)
                    cam["pseudo_gt_trusted"] = True
                else:
                    novel.append(vdir)
                cam["pseudo_gt"] = completed
            # The 4 novel views are independent API calls: run them concurrently
            # (sequential gpt was ~250 s/scene; the slowest single call bounds this).
            with ThreadPoolExecutor(max_workers=len(novel) or 1) as ex:
                list(ex.map(lambda v: _polish_view(v, POLISH_PROMPT, model), novel))
            for cam in cams:
                if cam.get("pseudo_gt_trusted") is True:
                    continue
                vdir = out / cam["tag"]
                drift_data = {}
                try:
                    drift_data = json.loads((vdir / "drift.json").read_text())
                except (OSError, ValueError, TypeError):
                    pass
                if drift_data.get("keep_drift") is not None:
                    cam["keep_drift"] = drift_data["keep_drift"]
                if (vdir / "gpt_fallback.txt").exists():
                    cam["pseudo_gt_trusted"] = False
                    cam["pseudo_gt_untrusted_reason"] = "generation_failed"
                elif (vdir / "drift_flag.txt").exists():
                    cam["pseudo_gt_trusted"] = False
                    cam["pseudo_gt_untrusted_reason"] = "excessive_observed_content_drift"
                else:
                    cam["pseudo_gt_trusted"] = True
            with open(out / "cameras.json", "w") as f:
                json.dump({"views": cams}, f, indent=2)
            return str(out)

    for cam in cams:
        vdir = out / cam["tag"]
        vdir.mkdir(parents=True, exist_ok=True)
        completed = str(vdir / "completed.png")
        if cam["az"] == 0.0 and cam["el"] == 0.0:  # reference == source image
            from PIL import Image

            Image.open(image_path).convert("RGB").save(completed)
            cam["pseudo_gt"] = completed
            cam["pseudo_gt_trusted"] = True
            continue
        render = render_view(
            moge_dir,
            image_path,
            str(vdir),
            np.asarray(cam["cam2world_moge"]),
            point_radius=point_radius,
        )
        marker = vdir / "gpt_fallback.txt"
        if marker.exists():
            marker.unlink()  # clear any stale marker on rebuild
        last_err: Exception | None = None
        for attempt in range(1, _GPT_ATTEMPTS + 1):
            try:
                complete(
                    render.render_png,
                    render.mask_png,
                    completed,
                    prompt=prompt,
                    model=model,
                )
                last_err = None
                break
            except Exception as e:  # noqa: BLE001 - external image API; retry then degrade
                last_err = e
                print(
                    f"[pseudo_gt] GPT-Image complete failed for {cam['tag']} "
                    f"(attempt {attempt}/{_GPT_ATTEMPTS}): {type(e).__name__}: {e}"
                )
                if attempt < _GPT_ATTEMPTS:
                    time.sleep(_GPT_BACKOFF_S * attempt)  # linear backoff
        if last_err is not None:
            # Every attempt failed -> fall back to the raw splat, but make the degraded state
            # VISIBLE: a silent copy made the splat masquerade as a GPT completion (the demo
            # then showed the point-cloud render and the "GPT" image as identical).
            print(
                f"[pseudo_gt] all {_GPT_ATTEMPTS} attempts failed for {cam['tag']}; using the "
                "raw splat render as the pseudo-GT (flagged in gpt_fallback.txt)"
            )
            shutil.copy(render.render_png, completed)
            marker.write_text(
                "GPT-Image inpaint failed; completed.png is the raw splat render (NOT a GPT "
                f"completion).\nlast error: {type(last_err).__name__}: {last_err}\n"
            )
        cam["pseudo_gt"] = completed
        cam["pseudo_gt_trusted"] = last_err is None
        if last_err is not None:
            cam["pseudo_gt_untrusted_reason"] = "generation_failed"
    with open(out / "cameras.json", "w") as f:
        json.dump({"views": cams}, f, indent=2)
    return str(out)


def run(
    moge_dir: str,
    image_path: str,
    out_dir: str,
    cam2world,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL,
    size: str = "auto",
    point_radius: int = 1,
) -> dict:
    """Render the new view then complete it; return the artifact paths."""
    render = render_view(
        moge_dir, image_path, out_dir, cam2world, point_radius=point_radius
    )
    completed = complete(
        render.render_png,
        render.mask_png,
        f"{out_dir}/completed.png",
        prompt=prompt,
        model=model,
        size=size,
    )
    return {**render.__dict__, "completed_png": completed}


def main() -> None:
    p = argparse.ArgumentParser(
        description="MoGE cloud -> novel-view render -> GPT complete"
    )
    p.add_argument("--moge-dir", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pose", default=None, help="16 floats, row-major cam2world")
    p.add_argument("--pose-file", default=None, help=".npy or .json 4x4 cam2world")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--size", default="auto")
    p.add_argument("--point-radius", type=int, default=1)
    p.add_argument(
        "--nvs-backend",
        choices=("gpt_splat", "sharp_gpt"),
        default="gpt_splat",
        help="gpt_splat = MoGE point-cloud splat + masked GPT fill (legacy); "
        "sharp_gpt = SHARP 3DGS raw render + maskless GPT polish, novel views "
        "completed in PARALLEL. sharp_gpt degrades to gpt_splat if SHARP fails.",
    )
    p.add_argument(
        "--pseudo-gt",
        action="store_true",
        help="Build the full pseudo-GT orbit set (build_pseudo_gt) instead of a "
        "single view; R/T are read from <moge-dir>/moge.json gravity. Used by "
        "preprocess to run the GPT-completion calls as a BACKGROUND subprocess "
        "overlapped with meshes/settle.",
    )
    args = p.parse_args()

    if args.pseudo_gt:
        mj = json.load(open(Path(args.moge_dir) / "moge.json"))
        g = mj.get("gravity") or {}
        R = np.array(g["R"]) if g.get("R") is not None else None
        T = np.array(g["T"]) if g.get("T") is not None else None
        built = build_pseudo_gt(
            args.moge_dir, args.image, args.out_dir, R=R, T=T, backend=args.nvs_backend
        )
        print(json.dumps({"built": built}))
        return

    cam2world = _parse_pose(args.pose, args.pose_file)
    result = run(
        args.moge_dir,
        args.image,
        args.out_dir,
        cam2world,
        prompt=args.prompt,
        model=args.model,
        size=args.size,
        point_radius=args.point_radius,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
