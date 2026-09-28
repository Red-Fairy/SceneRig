"""Overlay for the top-down BEV render used by the initializer/composition agents.

The BEV is an ORTHOGRAPHIC top-down render centred on the az/el pivot over a ``size_m`` square.
The render camera looks down -Z with up +Y; the base image is ROTATED 180 deg before drawing
so the final BEV is CAMERA-ALIGNED: **image-up = world -Y** (deeper into the scene — the
reference camera, which looks roughly along -Y, sits near the bottom looking UP the image) and
**image-right = world -X** (the camera's right; matches the photo, where world +X is toward
image-LEFT). On top of the flat per-surface tint we draw:
surface AABB outlines + id labels (so a thin vertical wall — a line from above — is visible), a
colour legend, the reference camera as a dot + view arrow (clamped to the frame edge when it sits
outside the square, with its distance), and the world +X/+Y axes. Pure PIL; the projection helpers
are unit-tested without Blender.
"""

from __future__ import annotations

import json
from typing import Any


def bev_pixel(
    x: float, y: float, pivot, size_m: float, res: int
) -> tuple[float, float]:
    """World XY -> BEV pixel, camera-aligned (180 deg vs the raw ortho render):
    image-right = world -X, image-up = world -Y — both match the reference photo's directions."""
    px = res / 2.0 - (float(x) - float(pivot[0])) / size_m * res
    py = res / 2.0 + (float(y) - float(pivot[1])) / size_m * res
    return px, py


def clamp_to_frame(
    px: float, py: float, res: int, pad: float = 3.0
) -> tuple[float, float, bool]:
    """Clamp a point to the frame border along the ray from the image centre (for an off-frame
    camera). Returns ``(px, py, inside)``; ``inside`` is False when it had to be clamped."""
    if 0.0 <= px <= res and 0.0 <= py <= res:
        return px, py, True
    cx = cy = res / 2.0
    dx, dy = px - cx, py - cy
    if dx == 0.0 and dy == 0.0:
        return cx, cy, True
    ts = []
    if dx > 0:
        ts.append((res - pad - cx) / dx)
    elif dx < 0:
        ts.append((pad - cx) / dx)
    if dy > 0:
        ts.append((res - pad - cy) / dy)
    elif dy < 0:
        ts.append((pad - cy) / dy)
    t = min(t for t in ts if t > 0)
    return cx + dx * t, cy + dy * t, False


def _font(sz: int):
    from PIL import ImageFont

    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", sz)
    except Exception:  # noqa: BLE001 - bundled font may be absent
        try:
            return ImageFont.load_default(sz)
        except Exception:  # noqa: BLE001
            return ImageFont.load_default()


def draw_bev(base_png: str, data: dict[str, Any], out_png: str) -> str:
    """Draw the overlay on ``base_png`` (the flat tinted render) using the sidecar ``data`` and save
    to ``out_png``. ``data``: ``{size_m, pivot:[x,y,z], res, surfaces:[{name,color[r,g,b],aabb:[xmin,
    ymin,xmax,ymax],centroid:[x,y]}], camera:{pos:[x,y,z], forward:[fx,fy,fz]}}``."""
    import math

    from PIL import Image, ImageDraw

    im = Image.open(base_png).convert("RGBA")
    # The ortho render has +X right / +Y up; rotate 180 deg so the BEV is camera-aligned
    # (image-up = -Y, image-right = -X — the reference camera's own orientation).
    im = im.transpose(Image.ROTATE_180)
    res = int(data.get("res") or im.size[0])
    if im.size != (res, res):
        im = im.resize((res, res))
    pivot, size_m = data["pivot"], float(data["size_m"])
    surfaces = data.get("surfaces", [])
    dr = ImageDraw.Draw(im, "RGBA")
    fs = max(15, int(res * 0.022))
    lfs = max(18, int(res * 0.026))  # legend text
    sfont, lfont = _font(fs), _font(lfs)

    def W(x, y):
        return bev_pixel(x, y, pivot, size_m, res)

    def otext(xy, s, fnt, fill):
        o = max(2, int(fnt.size * 0.14))
        for ox, oy in (
            (-o, 0),
            (o, 0),
            (0, -o),
            (0, o),
            (-o, -o),
            (o, o),
            (-o, o),
            (o, -o),
        ):
            dr.text((xy[0] + ox, xy[1] + oy), s, fill=(0, 0, 0, 255), font=fnt)
        dr.text(xy, s, fill=fill, font=fnt)

    # --- surfaces: AABB outline for the horizontal ones (table/floor) only. A wall renders as its
    # true (often diagonal) footprint band, whose AXIS-ALIGNED bbox is a big misleading rectangle,
    # so we do NOT outline walls. Per-surface id labels are drawn LAST using placement rules
    # that avoid centroid collisions; the legend below also lists every surface. ---
    for s in surfaces:
        if not s["name"].startswith("wall"):
            col = tuple(int(c) for c in s["color"]) + (255,)
            xmin, ymin, xmax, ymax = s["aabb"]
            p0, p1 = W(xmin, ymin), W(xmax, ymax)
            box = [
                min(p0[0], p1[0]),
                min(p0[1], p1[1]),
                max(p0[0], p1[0]),
                max(p0[1], p1[1]),
            ]
            dr.rectangle(box, outline=col, width=max(2, int(res * 0.006)))

    # --- reference camera: dot + view arrow, clamped to the edge if off-frame ---
    cam = data.get("camera") or {}
    if cam.get("pos"):
        cpx, cpy = W(cam["pos"][0], cam["pos"][1])
        cx, cy, inside = clamp_to_frame(cpx, cpy, res)
        fwd = cam.get("forward") or [0.0, -1.0, 0.0]
        n = math.hypot(fwd[0], fwd[1]) or 1.0
        ax, ay = -fwd[0] / n, fwd[1] / n  # camera-aligned: image x = -X, image y = +Y
        tip = (cx + ax * res * 0.12, cy + ay * res * 0.12)
        r = max(5, int(res * 0.012))
        dr.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(255, 220, 40, 255),
            outline=(0, 0, 0, 255),
            width=2,
        )
        dr.line(
            [(cx, cy), tip], fill=(255, 220, 40, 255), width=max(3, int(res * 0.008))
        )
        for dth in (2.6, -2.6):
            hx = tip[0] + math.cos(math.atan2(ay, ax) + dth) * r * 2.2
            hy = tip[1] + math.sin(math.atan2(ay, ax) + dth) * r * 2.2
            dr.line(
                [tip, (hx, hy)],
                fill=(255, 220, 40, 255),
                width=max(3, int(res * 0.008)),
            )
        dist = math.hypot(cam["pos"][0] - pivot[0], cam["pos"][1] - pivot[1])
        otext(
            (cx + r + 3, cy - fs),
            "camera" if inside else f"camera {dist:.1f} m",
            sfont,
            (255, 220, 40, 255),
        )

    # --- world +X / +Y axes widget (bottom-left; camera-aligned: +X LEFT, +Y DOWN) ---
    ox, oy, al = int(res * 0.19), int(res * 0.84), int(res * 0.09)
    dr.line(
        [(ox, oy), (ox - al, oy)],
        fill=(235, 70, 70, 255),
        width=max(3, int(res * 0.007)),
    )
    dr.line(
        [(ox, oy), (ox, oy + al)],
        fill=(70, 215, 95, 255),
        width=max(3, int(res * 0.007)),
    )
    otext((ox - al - fs * 1.6, oy - fs * 0.5), "+X", sfont, (235, 70, 70, 255))
    otext((ox - fs * 0.6, oy + al + 2), "+Y", sfont, (70, 215, 95, 255))

    # --- legend + scale caption, ON the image in the (usually empty) top-left corner. Outlined
    # text (no backing box) so it reads over any surface behind it. ---
    lx, ly = int(res * 0.02), int(res * 0.02)
    otext((lx, ly), "BEV (top-down)", lfont, (255, 255, 255, 255))
    ly += int(lfs * 1.5)
    otext((lx, ly), f"{size_m:.1f} m across", lfont, (205, 205, 210, 255))
    ly += int(lfs * 1.6)
    for s in surfaces:
        col = tuple(int(c) for c in s["color"]) + (255,)
        dr.rectangle(
            [lx, ly + int(lfs * 0.1), lx + lfs, ly + lfs],
            fill=col,
            outline=(255, 255, 255, 255),
            width=2,
        )
        otext(
            (lx + lfs + 8, ly + int(lfs * 0.05)), s["name"], lfont, (255, 255, 255, 255)
        )
        ly += int(lfs * 1.5)

    # --- per-surface id labels LAST, so they sit on TOP of the bboxes/arrows and stay readable.
    # Walls at their MIDDLE; horizontal surfaces at their bbox TOP-LEFT; a floor/ceiling whose
    # top-left is OFF-frame is skipped (it's in the legend), else placed on top like the tables. ---
    for s in surfaces:
        col = tuple(int(c) for c in s["color"]) + (255,)
        name = s["name"]
        if name.startswith("wall"):
            cx, cy = W(s["centroid"][0], s["centroid"][1])
            otext((cx - fs * 0.3 * len(name), cy - fs * 0.6), name, sfont, col)
        else:
            xmin, ymin, xmax, ymax = s["aabb"]
            tlx, tly = W(xmin, ymax)  # image top-left of the (axis-aligned) bbox
            if (name.startswith("floor") or name.startswith("ceiling")) and not (
                0 <= tlx <= res and 0 <= tly <= res
            ):
                continue  # floor label off-frame -> skip
            m = int(res * 0.006) + 3  # clear the bbox outline + a ~3px margin
            otext((tlx + m, tly + m), name, sfont, col)

    im.convert("RGB").save(out_png)
    return out_png


def draw_from_sidecar(base_png: str, sidecar_json: str, out_png: str) -> str:
    return draw_bev(base_png, json.load(open(sidecar_json)), out_png)
