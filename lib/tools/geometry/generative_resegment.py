"""Generative re-segmentation: occlusion DAG + vital-part check -> edit + re-detect.

Image-to-3D models (TRELLIS.2, DSO, SAM3D) condition on a single object crop and expect
the object to be COMPLETE; a partially occluded object yields a truncated mesh. This
module owns the whole detect -> decide -> re-segment pipeline over a segmented scene:

  1. OCCLUSION DETECTION: touching mask pairs (few-px dilation overlap + a boundary
     median-depth prior) are each adjudicated by one pairwise VLM call (zoomed clean +
     red/blue-marked crops) -> directed occluder->occluded edges. A mask touching the
     image frame is flagged ``border`` geometrically (no VLM).
  2. HIERARCHY GATE (pure code): a support-ANCESTOR cannot occlude its descendants —
     a table is never in front of the cup standing on it — so such edges are dropped
     (logged, not silent). Child->parent edges stay: a cup DOES cover its table.
  3. CROP CHECK: one VLM call per instance (full photo, red-marked zoom, mask
     cut-out on gray) with the DAG-verified occluders given as context; it judges
     (a) whether a structural VITAL part (a monitor's stand, a table's legs) is
     hidden or missing from the cut-out and (b) crop SEVERITY — how much of the
     object's potentially-visible extent the cut-out shows (heavy <=> under 75%).
     Occluder attribution is NOT its job; the pairwise judge owns topology, this
     call owns magnitude, code owns policy.
  4. REDETECT gate: edit path fires for ``occluded-by-objects AND (crop-heavy OR
     vital-part-missing)`` or when a non-trivial occluder carve (``carved_px``)
     forces recovery; vital-only (segmentation miss) -> reseg-original.
     - occluded: removal set = in-edge occluders + their transitive
       support-descendants -> LanPaint + Qwen-Image-Edit MASKED inpaint over the
       occluder-union hole ("Remove {...} from the image"; LanPaint regenerates the
       masked region coherently from its surroundings) -> SAM3 re-segmentation on the
       edited image (part-aware prompt when a vital part is named, sibling masks
       carved out, removed occluders' region exempt).
     - vital-only (no removable occluder — a segmentation miss): SAM3 part-aware
       re-segmentation on the ORIGINAL image; no edit. Frame-cut parts are NOT
       vital-flagged; purely frame-cut objects use the border-completion arm instead.
     - border-only: outpaint the missing side, refine its depth with LingBot, then
       re-segment the completed object and retain its expanded point grid.

Artifacts under ``<scene>/masks/``: ``generative_resegment.json`` (pairs, edges kept/
dropped, vital verdicts, redetect set), ``_occl_*`` pair crops, ``_vital_*`` instance
crops, ``edited/<slug>.png`` (+ ``<slug>_hole.png`` for LingBot depth invalidation),
``<slug>_redetect.npy``, and border-arm ``*_border.png`` / ``*_border_points.npy``.
The per-instance ``redetect`` record persisted into
masks.json keeps the schema downstream consumes (placement / meshes / ICP / register /
demo): {mask_path, seg_prompt, score, edited_image, hole_mask, removed, prompt,
vital_part}. Always invoked by ``preprocess_scene`` after segmentation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from lib.tools.geometry.agentic_mask import (
    DEFAULT_MODEL,
    _dilate1,
    _img_part,
    _make_vlm,
    _pair_overlay,
    _union_crop,
    require_json_contract,
    vlm_json,
)
from lib.tools.geometry.border_complete import (
    border_complete,
    border_cut_side,
    edge_contact_ratio,
)
from lib.tools.geometry.resegmentation_audit import (
    audit_cache_matches,
    audit_resegmentation,
    prepare_evidence,
)

HOLE_DILATE_PX = 12

_OCCL_SYSTEM = """You judge OCCLUSION between two objects in a scene photo.
You see the full image, then a zoomed close-up of the two objects, then the same
close-up with object A tinted/outlined RED and object B tinted/outlined BLUE.

Answer: is one object PARTIALLY HIDDEN behind the other (part of it invisible because
the other object is in front of it), or are they merely ADJACENT (both fully visible,
just close together or touching in 2D)?

Rules:
- Occlusion means SOME part of the object is invisible because the other object sits
  in front of it — ANY amount counts, including the patch of a surface covered by an
  object resting on top of it (a mug standing on a book DOES occlude the book).
  Shadows and reflections do NOT count.
- Answer "none" ONLY when both objects are FULLY visible: they touch or sit side by
  side in the image without either covering any part of the other.
- If BOTH lose part of their visible extent to the other, report the one losing MORE.
- You judge WHO covers WHOM only — how MUCH is hidden is judged elsewhere.

Return STRICT JSON only:
{"reason": "<1-2 sentences>",
 "occluded": "A" | "B" | "none",
 "missing_part": "<which part is hidden, e.g. 'left half', 'lower-right corner'>"}"""

_VITAL_SYSTEM = """You check whether a structural VITAL PART of an object is missing
from its segmentation. You see: the full photo; a zoomed close-up with the object
tinted/outlined RED; and the object's cut-out on a gray background (its mask applied —
exactly the image a 3D reconstruction model would receive).

A VITAL PART is a structural part the object needs to stand or function — a monitor's
stand/base, a table's legs, a lamp's foot, a mug's handle-side body. Flag it when the
part is hidden by ANOTHER OBJECT, or visible in the photo but absent from the cut-out
— EVEN IF the hidden fraction is small: a reconstruction from this crop would be
unusable (e.g. a screen slab with no base cannot stand).

Parts cut off by the IMAGE FRAME are not ``vital_part_missing``. Do NOT count frame
truncation as a missing vital part; report left/right truncation separately through
``beyond_frame_fraction`` below. An object merely sitting near the image edge is not
truncated.

Parts that are simply NOT VISIBLE FROM THIS VIEWPOINT — the object's far side, its
back, or parts hidden behind the object's OWN body (a teddy bear whose legs are tucked
behind/under its torso, a mug whose handle faces away) — are NOT missing: the
reconstruction model completes unseen sides on its own (self-occlusion). Flag a part
ONLY when it would be VISIBLE from this camera were the blocker absent.

The test is a COUNTERFACTUAL: with the listed blockers REMOVED, what would this
camera see? A face covered by an object RESTING ON this one WOULD be visible from
this camera if the blocker were absent — count it as hidden. NEVER classify a face
as "not visible from this camera anyway" when the only thing hiding it is the
blocker itself. Self-occlusion means hidden by the object's OWN body given its own
pose — nothing else. Examples: a bottom book whose ENTIRE top face lies under
another book shows only its spine — LOW visible_fraction (a reconstruction would get
a spine sliver); a book merely carrying a small jar loses a notch of its top — HIGH
visible_fraction.

The message may list KNOWN OCCLUDERS (verified by a separate pass) — use them to
locate what they hide; do not second-guess the list.

Also estimate ``visible_fraction``: of everything this object WOULD show from this
camera with all blockers absent (ignore its far side and anything beyond the image
frame), what fraction is actually present in the cut-out? A book whose top face is
covered by another book resting on it shows only its spine: that is a LOW fraction
(well under half), even though the hidden face is "just one side". An object with a
small notch taken out by a neighbour is a HIGH fraction. Estimate it carefully and
consistently from the supplied views.

Separately, judge whether the object is ROLLABLE in its CURRENT pose in the
photo: is it TIPPED OVER — resting on a side of its body that is NOT how it
normally rests — or resting on a ROUNDED part of its own body so a nudge would
roll it? FIRST compare the object's current orientation to how that object
normally rests. Judge the 3D pose, not the 2D image direction: depending on
the camera the image may look at the scene from the side or straight down.
rollable is true when EITHER:
  (a) the object is TIPPED OVER onto its side or face instead of its natural
      base — a bottle, cup, mug, marker, pen, microphone, or box lying on its
      side (whether that side is rounded OR flat: a shoe on its side or a
      cereal box on its face count too); or
  (b) the object rests on a ROUNDED part of its own body — a cylindrical pen,
      marker, or bottle on its side, a ball, an apple, an orange, an egg, a
      croissant, a banana on its curve — so it could roll if nudged. For
      ROUNDED contact, "that is how it always rests" is NEVER a reason to
      answer false: a pen always lies on its side, and it still rolls — it IS
      rollable.
The surface UNDERNEATH is IRRELEVANT — a bottle lying on a flat table IS
rollable; "it rests on a flat surface" is never a reason to answer false.
Beware the ELONGATION TRAP, which applies at EVERY downward camera angle —
straight down, steeply tilted, or shallow. A STANDING pen, marker, bottle or
can photographed from above ALWAYS projects as an elongated, usually diagonal
shape, because you are seeing its length foreshortened rather than its end.
Apparent elongation is therefore NEVER evidence that an object is lying, and
neither is a visible side profile: a standing marker under a tilted camera
shows its whole side.

Decide instead from WHERE THE BODY MEETS THE SURFACE, which is visible at any
angle:
  - STANDING: the object touches the surface at ONE END only — a small contact
    patch — and the rest of the body rises away from it. The far end is off the
    surface. Its cast shadow leaves that contact point and runs to one side of
    the body, and the near end typically looks LARGER than the far end.
  - LYING: the WHOLE LENGTH of the body rests on the surface. BOTH ends sit at
    surface height and the shadow hugs the body along its entire length.
The RED outline in the close-up is drawn ON TOP of the object's own base and
often covers both the contact patch and the near part of its shadow. When the
marking hides the contact region, judge it from the FULL IMAGE instead — the
shadow there is unmarked.

For an ELONGATED object (pen, marker, bottle, can, tube) also run this LENGTH
CHECK, which needs neither the contact patch nor the shadow. Estimate how long
the object would appear IF IT WERE LYING FLAT at that spot, using a neighbour
of known size as a ruler (a mug is roughly 9 cm across, a dinner plate 26 cm, a
whiteboard marker 13-14 cm long), then compare with its ACTUAL apparent length:
  - apparent length close to the object's true length -> LYING;
  - apparent length a small FRACTION of it (often barely more than the object's
    own width) -> STANDING, foreshortened because you are looking down its axis
    rather than across it.
A marker spanning only a third of a nearby mug's width is standing, not a
miniature marker lying down.
When the contact region is genuinely hidden AND the length check is
inconclusive, prefer STANDING for an object whose category normally stands —
calling a standing object rollable removes the only guard against it toppling.
false only when the object meets the support through a FLAT face, base, edge,
legs, or stand of its own, in its normal rest orientation: anything standing
upright (a tripod or stand whose splayed legs touch the ground is STANDING
even if its column is short, angled, or folded; a STANDING marker is not
rollable — the same marker on its side is); FLAT-bottomed or slab-like
objects resting flat as they always do — cutlery (spoons, forks, knives),
spatulas, books, plates, keyboards, trays, remotes, power banks, phones,
folded glasses, a case with a domed top resting on its flat bottom,
keys/keychains, a donut or cookie flat on its face; and anything held,
clipped, or attached to something. If occlusion hides the contact and the
pose, default to false. This is about the state in THIS image, not the
object category.

Separately from visible_fraction, estimate ``beyond_frame_fraction``: of the
object's FULL physical extent facing this camera, what fraction lies BEYOND the
image frame (sliced off by the LEFT or RIGHT image edge)? 0.0 when the object is
entirely inside the frame — merely sitting near or lightly touching the edge is
0.0. Judge from the silhouette: a monitor whose left third is off-screen is
~0.33; a notebook losing a thin sliver past the edge is ~0.05.

Return STRICT JSON only:
{"reason": "<1-2 sentences>",
 "beyond_frame_fraction": <float 0-1>,
 "visible_fraction": <float 0-1>,
 "vital_part_missing": true | false,
 "vital_part": "<the structural part missing from the cut-out, or ''>",
 "contact": "one_end" | "full_length" | "unclear",
 "rollable": true | false}

``contact`` records HOW the object meets the surface and must be decided BEFORE
rollable: "one_end" (it stands on one end/base), "full_length" (its whole body
rests along the surface), or "unclear". An object whose contact is "one_end" is
STANDING and is NOT rollable. It is a reasoning step, not a stored field —
committing to it before answering is what keeps the two consistent."""


def _validate_occlusion_reply(out: dict) -> dict:
    value = out.get("occluded")
    if not isinstance(value, str):
        raise ValueError("occluded must be a JSON string")
    normalized = value.strip().upper()
    if normalized not in {"A", "B", "NONE"}:
        raise ValueError("occluded must be A, B, or none")
    out["occluded"] = "none" if normalized == "NONE" else normalized
    if not isinstance(out.get("missing_part", ""), str):
        raise ValueError("missing_part must be a JSON string")
    return out


def _validate_vital_reply(out: dict) -> dict:
    require_json_contract(
        out,
        bool_fields=("vital_part_missing", "rollable"),
        unit_interval_fields=("visible_fraction", "beyond_frame_fraction"),
    )
    if not isinstance(out.get("vital_part"), str):
        raise ValueError("vital_part must be a JSON string")
    return out


def _rid(rec: dict) -> str:
    return f"{rec['category']}#{rec['instance']}"


def _slug(rid: str) -> str:
    return rid.replace("#", "").replace(" ", "_")


def _fit(m: np.ndarray, shape) -> np.ndarray:
    if m.shape == shape:
        return m
    from PIL import Image

    return (
        np.asarray(
            Image.fromarray(m.astype("uint8") * 255).resize((shape[1], shape[0]))
        )
        > 127
    )


def _dilate(m: np.ndarray, px: int) -> np.ndarray:
    for _ in range(px):
        m = _dilate1(m)
    return m


# --------------------------------------------------------------------------- #
# Stage 1 — occlusion detection                                                #
# --------------------------------------------------------------------------- #
def candidate_pairs(
    objs: list[dict],
    masks: dict[str, np.ndarray],
    depth: Optional[np.ndarray],
    valid: Optional[np.ndarray],
    dilate_px: int = 3,
    min_contact_px: int = 20,
) -> list[dict]:
    """Touching object pairs + the boundary depth prior.

    For each pair whose ``dilate_px``-dilated masks overlap by >= ``min_contact_px``,
    sample each object's own pixels inside the contact zone and compare median depth:
    the nearer side is the prior occluder. ``front`` is None when depth is missing or
    the gap is within noise (< 5% of the nearer depth)."""
    dil = {i: _dilate(masks[i], dilate_px) for i in masks}
    out = []
    for x in range(len(objs)):
        for y in range(x + 1, len(objs)):
            a, b = _rid(objs[x]), _rid(objs[y])
            contact = dil[a] & dil[b]
            n_contact = int(contact.sum())
            if n_contact < min_contact_px:
                continue
            front, da, db = None, None, None
            if depth is not None:
                ok = np.isfinite(depth) & (valid if valid is not None else True)
                sa = depth[contact & masks[a] & ok]
                sb = depth[contact & masks[b] & ok]
                if sa.size >= 5 and sb.size >= 5:
                    da, db = float(np.median(sa)), float(np.median(sb))
                    if abs(da - db) > 0.05 * min(da, db):
                        front = a if da < db else b
            out.append(
                {
                    "a": a,
                    "b": b,
                    "contact_px": n_contact,
                    "depth_a": da,
                    "depth_b": db,
                    "front_prior": front,
                }
            )
    return out


def judge_pair(
    vlm: Callable,
    image_path: str,
    pair: dict,
    rec_a: dict,
    rec_b: dict,
    masks: dict[str, np.ndarray],
    masks_dir: Path,
    n: int,
) -> dict:
    """One VLM call: adjacency vs occlusion for a touching pair. Saves the clean +
    marked zoom crops (demo-inspectable, same pattern as the merge decisions)."""
    a, b = pair["a"], pair["b"]
    tag = f"{a}__{b}".replace("#", "").replace(" ", "_")
    clean = str(masks_dir / f"_occl_{n}_{tag}_clean.png")
    marked = str(masks_dir / f"_occl_{n}_{tag}_marked.png")
    tmp = str(masks_dir / "_occl_pair.png")
    _pair_overlay(image_path, masks[a], masks[b], tmp)
    _union_crop(image_path, masks[a], masks[b], clean)
    _union_crop(tmp, masks[a], masks[b], marked)
    hint = ""
    if pair.get("front_prior"):
        near = "A (RED)" if pair["front_prior"] == a else "B (BLUE)"
        hint = (
            f" Depth estimate suggests {near} is NEARER the camera along their shared "
            "boundary (so the other would be the hidden one) — verify this visually."
        )
    txt = (
        f"Object A (RED) = '{rec_a['category']}': {rec_a['description']}. "
        f"Object B (BLUE) = '{rec_b['category']}': {rec_b['description']}.{hint}"
    )
    # default {} -> occluded "none": one flaky reply skips this pair's recovery
    # instead of cancelling resegment for the whole scene (RC6).
    out = vlm_json(
        vlm,
        _OCCL_SYSTEM,
        [
            {"type": "text", "text": "Full image (context):"},
            _img_part(image_path),
            {"type": "text", "text": txt + " Zoomed close-up (no marks):"},
            _img_part(clean),
            {"type": "text", "text": "Same close-up, A red, B blue:"},
            _img_part(marked),
        ],
        label=f"judge_pair {a}/{b}",
        default={},
        validate=_validate_occlusion_reply,
    )
    occ = str(out.get("occluded", "none")).strip().upper()
    rec = {
        **pair,
        "clean_crop": clean,
        "marked_crop": marked,
        "reason": out.get("reason"),
        "missing_part": out.get("missing_part"),
        "occluded": None,
        "occluder": None,
    }
    if occ in ("A", "B"):
        rec["occluded"] = a if occ == "A" else b
        rec["occluder"] = b if occ == "A" else a
    return rec


# --------------------------------------------------------------------------- #
# Stage 2 — hierarchy gate                                                     #
# --------------------------------------------------------------------------- #
def filter_hierarchy(
    edges: list[dict], support: dict[str, Optional[str]]
) -> tuple[list[dict], list[dict]]:
    """Drop edges whose occluder is a support-ANCESTOR of the occluded object.

    A surface/object is never in front of something resting (transitively) on it —
    such edges are VLM hallucinations. Child->parent edges stay (a cup legitimately
    covers the patch of table it stands on). Cycle-safe ancestor walk. Returns
    (kept, dropped) with dropped edges annotated ``drop_reason="hierarchy"``."""

    def ancestors(rid: str) -> set:
        out, cur = set(), support.get(rid)
        while cur and cur not in out:
            out.add(cur)
            cur = support.get(cur)
        return out

    kept, dropped = [], []
    for e in edges:
        if e["occluder"] in ancestors(e["occluded"]):
            dropped.append({**e, "drop_reason": "hierarchy"})
        else:
            kept.append(e)
    return kept, dropped


MARK_RED = ((255.0, 0.0, 0.0), "RED")
MARK_GREEN = ((0.0, 220.0, 60.0), "GREEN")


def _marked_crop(
    img: np.ndarray, m: np.ndarray, out: str, pad_frac=0.35, mark=MARK_RED
) -> str:
    ""
    from PIL import Image

    rgb = np.array(mark[0], float)
    arr = img.astype(np.float32).copy()
    ring = m.copy()
    for _ in range(3):
        ring = _dilate1(ring)
    ring = ring & ~m
    arr[m] = 0.8 * arr[m] + 0.2 * rgb
    arr[ring] = rgb
    from lib.tools.geometry.agentic_mask import clamp_crop_aspect

    ys, xs = np.where(m)
    py, px = (
        int((ys.max() - ys.min()) * pad_frac) + 8,
        int((xs.max() - xs.min()) * pad_frac) + 8,
    )
    y0, y1 = max(0, ys.min() - py), min(img.shape[0], ys.max() + py)
    x0, x1 = max(0, xs.min() - px), min(img.shape[1], xs.max() + px)
    x0, y0, x1, y1 = clamp_crop_aspect(
        int(x0), int(y0), int(x1), int(y1), img.shape[1], img.shape[0]
    )
    Image.fromarray(arr[y0:y1, x0:x1].clip(0, 255).astype("uint8")).save(out)
    return out


def _masked_crop(img: np.ndarray, m: np.ndarray, out: str, pad=8) -> str:
    """The reconstruction-model view: mask applied, composited on mid-gray."""
    from PIL import Image

    ys, xs = np.where(m)
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    c, mc = img[y0:y1, x0:x1].astype(np.float32), m[y0:y1, x0:x1]
    gray = np.full_like(c, 128.0)
    gray[mc] = c[mc]
    Image.fromarray(gray.clip(0, 255).astype("uint8")).save(out)
    return out


def check_vital(
    vlm: Callable,
    image_path: str,
    rid: str,
    rec: dict,
    mask: np.ndarray,
    masks_dir: Path,
    occluders: list,
    border: bool,
    heavy_below: float = 0.75,
) -> dict:
    """One VLM call on the CONDITIONING CROP: is a structural vital part missing,
    and how damaged is the crop? The DAG-verified ``occluders`` (+ the border flag)
    are given as context. ``visible_fraction`` is a VLM anchor (crop-relative:
    of what this camera COULD see, how much the cut-out shows — a book under another
    book is a spine sliver = LOW, regardless of "only one side" being hidden);
    severity is DERIVED here (heavy <=> fraction < ``heavy_below``); the raw
    fraction is persisted for display/audit only. Severity normally gates the edit path:
    heavy-or-vital occluded objects buy a removal edit, while a non-trivial ``carved_px``
    override can force the edit regardless of severity (0709: real8334 spent 6 edits on genuinely
    slight clutter for 1 useful record, while the gpt1 book — "slight" by whole-object
    area — is a spine-only crop that NEEDS recovery)."""
    from PIL import Image

    img = np.asarray(Image.open(image_path).convert("RGB"))
    tag = _slug(rid)
    marked = _marked_crop(img, mask, str(masks_dir / f"_vital_{tag}_marked.png"))
    cut = _masked_crop(img, mask, str(masks_dir / f"_vital_{tag}_cut.png"))
    known = ""
    if occluders:
        parts = [
            f"{o[0]} (hides: {o[1]})"
            if isinstance(o, (tuple, list)) and len(o) > 1 and o[1]
            else (o[0] if isinstance(o, (tuple, list)) else o)
            for o in occluders
        ]
        known = f"KNOWN OCCLUDERS covering parts of this object: {', '.join(parts)}.\n"
    if border:
        known += (
            "The object's mask touches the IMAGE BORDER: do not classify frame "
            "truncation as a missing vital part; report it through "
            "beyond_frame_fraction instead.\n"
        )
    # default {} -> not vital, fraction 1.0 (slight): an unparseable reply must not
    # buy an edit, and must not cancel resegment for the whole scene (RC6).
    out = vlm_json(
        vlm,
        _VITAL_SYSTEM,
        [
            {"type": "text", "text": "Full image (context):"},
            _img_part(image_path),
            {
                "type": "text",
                "text": f"Object in question: '{rid}': "
                f"{rec.get('description', '')}.\n{known}"
                "Zoomed close-up (object tinted/outlined red):",
            },
            _img_part(marked),
            {"type": "text", "text": "The object's mask cut-out on gray:"},
            _img_part(cut),
        ],
        label=f"check_vital {rid}",
        default={},
        validate=_validate_vital_reply,
    )
    vital = bool(out.get("vital_part_missing", False))
    try:
        frac = float(out.get("visible_fraction"))
    except (TypeError, ValueError):
        frac = 1.0  # conservative: an unparseable anchor must not buy an edit
        print(
            f"[generative_resegment] {rid}: visible_fraction unparseable; treating as slight"
        )
    try:
        beyond = float(out.get("beyond_frame_fraction"))
    except (TypeError, ValueError):
        beyond = 0.0  # conservative: an unparseable reply must not buy an edit
    return {
        "vital_part_missing": vital,
        "vital_part": (out.get("vital_part") or "") if vital else "",
        "severity": "heavy" if frac < heavy_below else "slight",
        # persisted for DISPLAY/audit (demo shows "~N% hidden"); the gate never
        # reads it — severity above is the only policy input
        "visible_fraction": round(frac, 2),
        "beyond_frame_fraction": round(beyond, 2),
        # state-aware rollability (a LYING marker rolls, a STANDING one does
        # not): consumed by the composition physics gates — a rolling object is
        # judged by DISPLACEMENT, not tilt (a lying mic rolling 150 deg in place
        # is benign but reads as a capsize). Default False on parse failure.
        "rollable": bool(out.get("rollable", False)),
        "reason": out.get("reason"),
        "marked_crop": marked,
        "cut_crop": cut,
    }


# --------------------------------------------------------------------------- #
# Stage 4 — redetect execution                                                 #
# --------------------------------------------------------------------------- #
def trivial_seam(seam_px: int, target_area: int) -> bool:
    ""
    min_px = int(os.environ.get("GRASE_REDETECT_MIN_CONTACT_PX", "150"))
    min_frac = float(os.environ.get("GRASE_REDETECT_MIN_CONTACT_FRAC", "0.03"))
    return seam_px < min_px and seam_px < min_frac * max(target_area, 1)


def occlusion_seam(target: np.ndarray, occluder_masks: list[np.ndarray]) -> int:
    """Dilated-contact seam between the target and its would-be-removed occluders
    (same 3px dilation as candidate_pairs, so the numbers are comparable)."""
    t = _dilate(target > 0, 3)
    seam = 0
    for om in occluder_masks:
        seam += int((t & _dilate(_fit(om > 0, t.shape), 3)).sum())
    return seam


def build_removal_set(
    target: str, direct_occluders: list[str], support: dict[str, Optional[str]]
) -> list[str]:
    """Occluder ids to erase for ``target``: direct occluders (sentinels dropped) plus
    everything transitively SUPPORTED by a removed object (never the target itself)."""
    closure = {o for o in direct_occluders if o in support and o != target}
    grew = True
    while grew:
        grew = False
        for rid, sup in support.items():
            if sup in closure and rid not in closure and rid != target:
                closure.add(rid)
                grew = True
    return sorted(closure)


def _natural_join(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def build_occluder_names(occluders: list[str], desc: dict[str, str]) -> list[str]:
    """Removal-prompt names for ``occluders``, per category: COLLAPSE a category to its
    (singular/plural) name when EVERY scene instance of that category is being removed
    ("croissants") -- no same-category object survives, so a verbose per-object list only
    hurts (a long positional enumeration makes the edit model regenerate one; 0717 abc1
    tray left a donut). Otherwise KEEP each occluder's individual description -- a book
    occluding another book stays "gray book on top" so the surviving target book is not
    erased too ("remove all books" would delete it). ``desc`` keys are every scene
    instance; its values are the per-object descriptions. The target is never an occluder,
    so a same-category occluder can never see its whole category listed -> always keeps
    individual descriptions (book-on-book stays disambiguated)."""
    scene_by_cat: dict[str, set] = {}
    for rid in desc:
        scene_by_cat.setdefault(rid.split("#")[0], set()).add(rid)
    occ_by_cat: dict[str, list] = {}
    for o in occluders:
        occ_by_cat.setdefault(o.split("#")[0], []).append(o)
    names, seen = [], set()
    for o in occluders:  # first-seen category order, occluder order within a category
        cat = o.split("#")[0]
        if cat in seen:
            continue
        seen.add(cat)
        occ = occ_by_cat[cat]
        if set(occ) == scene_by_cat.get(
            cat, set()
        ):  # whole category removed -> collapse
            names.append(cat if len(occ) == 1 else f"{cat}s")
        else:  # a same-category object survives -> keep individual descriptions
            names.extend(desc.get(x, x) for x in occ)
    return names


def build_removal_prompt(
    occluders: list[str],
    desc: dict[str, str],
    *,
    target_category: Optional[str] = None,
) -> str:
    """Shared masked-edit instruction for ordinary and border redetection.

    The erased silhouette can cover either an underlying support or a remaining
    object hidden behind a foreground blocker. The prompt therefore describes the
    counterfactual scene rather than assuming every edited pixel is empty table.
    ``target_category`` is included only for the foreign-blocker case, where the
    edit must be explicitly told not to paint background over the target reveal.
    """
    names = _natural_join(build_occluder_names(occluders, desc))
    target_note = (
        f" If their removal reveals part of the remaining {target_category}, "
        f"reconstruct that part of the {target_category}."
        if target_category
        else ""
    )
    return (
        f"Remove only the {names} from the image. Restore the scene as it would appear "
        f"with those objects absent.{target_note} Preserve every remaining object; "
        "where no remaining object is behind the removed objects, continue the existing "
        "support or background surface. Leave no trace of the removed objects and do "
        "not add or duplicate anything."
    )


def remove_objects(
    scene_dir: str,
    target: str,
    occluders: list[str],
    masks: dict[str, np.ndarray],
    desc: dict[str, str],
    qwen: Any,
    target_category: Optional[str] = None,
) -> dict[str, str]:
    """LanPaint + Qwen-Image-Edit MASKED inpaint removing ``occluders``; returns the
    redetect record stub {edited_image, hole_mask}. LanPaint regenerates the masked
    region coherently from its surroundings, so a proper mask is safe: the occluder
    union dilated ``HOLE_DILATE_PX`` px (a few px past the soft edge -> clean
    boundary removal). Mask convention is white=keep / black=edit. The same region is
    also saved as an alpha hole (``<slug>_hole.png``) for LingBot depth invalidation."""
    from PIL import Image

    scene = Path(scene_dir)
    out_dir = scene / "masks" / "edited"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(scene / "input.png").convert("RGBA")
    w, h = img.size
    hole = np.zeros((h, w), bool)
    for o in occluders:
        hole |= _fit(masks[o], (h, w))
    for _ in range(
        HOLE_DILATE_PX
    ):  # a few px past the occluder edge -> no boundary halo
        hole = _dilate1(hole)
    tag = _slug(target)
    # LanPaint edit mask: white(255)=keep, black(0)=regenerate over the removal region.
    edit_mask_path = out_dir / f"{tag}_editmask.png"
    Image.fromarray(np.where(hole, 0, 255).astype("uint8"), "L").save(edit_mask_path)
    # Alpha hole (alpha==0 = removed region) for LingBot depth re-completion.
    rgba = np.array(img)
    rgba[..., 3] = np.where(hole, 0, 255)
    hole_path = out_dir / f"{tag}_hole.png"
    Image.fromarray(rgba, "RGBA").save(hole_path)
    prompt = build_removal_prompt(occluders, desc, target_category=target_category)
    edited_path = out_dir / f"{tag}.png"
    resp = qwen.inpaint(
        str(scene / "input.png"), str(edit_mask_path), prompt, str(edited_path)
    )
    if not resp or not resp.get("ok"):
        raise RuntimeError(f"qwen inpaint failed: {(resp or {}).get('error')}")
    return {
        "edited_image": str(edited_path),
        "hole_mask": str(hole_path),
        "removed": occluders,
        "prompt": prompt,
    }


PART_UNION_MAX_GROWTH = float(os.environ.get("GRASE_RESEG_MAX_GROWTH", "8.0"))


def resegment(
    scene_dir: str,
    target: str,
    category: str,
    edited_image: str,
    original_mask: np.ndarray,
    vital_part: str = "",
    sam3: Any = None,
    min_iou_vs_original: float = 0.25,
    exclude_mask: Optional[np.ndarray] = None,
    synonyms: Optional[list[str]] = None,
    point: Optional[list[float]] = None,
    foreign_occluders: Optional[list[np.ndarray]] = None,
) -> Optional[dict[str, str]]:
    ""
    from lib.tools.geometry.agentic_mask import Sam3Server
    from lib.utils._path import SAM3_PY

    scene = Path(scene_dir)
    prompts = [category]
    if vital_part:
        prompts.insert(0, f"{category} including {vital_part}")
    syn_prompts = [
        s
        for s in (synonyms or [])
        if s and s.lower() != category.lower() and s not in prompts
    ]
    out_npy = scene / "masks" / f"{_slug(target)}_redetect.npy"
    own = None
    if sam3 is None:
        # the vital-only path (no removable occluder) reaches here before any edit
        # created masks/edited/ — the server's log open needs the dir to exist
        (scene / "masks" / "edited").mkdir(parents=True, exist_ok=True)
        own = sam3 = Sam3Server(
            SAM3_PY, log_path=str(scene / "masks" / "edited" / "sam3_redetect.log")
        )
    state: dict = {"orig": None, "grown": None, "excl": None}

    def _prep(shape):
        # original dilated a few px for the adjacency test; exclude mask carved
        # down to pixels OUTSIDE the original — computed once, on the first
        # detection's frame (rungs may run at different edited-image scales).
        if state["orig"] is None:
            o = _fit(original_mask, shape)
            g = o
            for _ in range(10):
                g = _dilate1(g)
            state["excl"] = (
                _fit(exclude_mask, shape) & ~o if exclude_mask is not None else None
            )
            state["orig"], state["grown"] = o, g
        return state["orig"], state["grown"], state["excl"]

    best, best_score, best_prompt = None, -1.0, ""
    n_seen = 0  # candidates judged across all rungs, for the outcome log

    def _judge(m, s, tag):
        nonlocal best, best_score, best_prompt, n_seen
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[0]
        m = m > 0
        orig, grown, excl = _prep(m.shape)
        if excl is not None:
            # a detection spanning ADJACENT instances passes the target-IoU gate
            # and wins on area (bridge3: one mask swallowed both plush toys) —
            # carve the siblings' pixels out before judging
            m = m & ~excl
        n_seen += 1
        iou = (m & orig).sum() / max((m | orig).sum(), 1)
        if iou >= min_iou_vs_original:
            cand = m  # overlaps the visible part -> trust it as the object
        elif (m & grown).any():
            # Disjoint but ADJACENT: SAM3 returned only the missing PART (the
            # part-aware prompt sometimes parses as the part itself, e.g. a
            # stand-only mask for "monitor including stand/base") -> the full
            # object is that part unioned with the visible mask. A PART cannot be
            # arbitrarily larger than the object it completes: see
            # PART_UNION_MAX_GROWTH (the tracker rung otherwise donates the whole
            # supporting surface here).
            cand = m | orig
            if int(cand.sum()) > PART_UNION_MAX_GROWTH * max(int(orig.sum()), 1):
                print(
                    f"[resegment] {target}: rejected a part-union candidate "
                    f"{int(cand.sum()) / max(int(orig.sum()), 1):.0f}x the original "
                    f"(> {PART_UNION_MAX_GROWTH:.0f}x) from {tag!r} — a part cannot "
                    "be that much larger than the object"
                )
                return
        else:
            return  # disjoint and far away: a different object
        # support-subtree clip (see docstring): swallowing >80% of a foreign
        # removed occluder's footprint = claiming the look-alike fill, not reveal
        for f in foreign_occluders or []:
            ff = _fit(f, cand.shape) if f.shape != cand.shape else (f > 0)
            fs = int(ff.sum())
            if fs and int((cand & ff).sum()) > 0.8 * fs:
                cand = cand & ~(ff & ~orig)
                print(
                    f"[resegment] {target}: candidate swallowed a foreign removed "
                    f"occluder ({fs}px, supported by a third object) — clipped"
                )
        # prefer the fullest object; never settle for less than the original
        if cand.sum() >= orig.sum() and (best is None or cand.sum() > best.sum()):
            best, best_score, best_prompt = cand, float(s), tag

    def _grew() -> bool:
        return (
            best is not None
            and state["orig"] is not None
            and best.sum() > 1.02 * state["orig"].sum()
        )

    try:
        for prompt in prompts:
            all_masks, scores = sam3.segment_all(edited_image, prompt, str(out_npy))
            if all_masks is None or not len(all_masks):
                continue
            for m, s in zip(all_masks, scores):
                _judge(m, s, prompt)
            if _grew():
                break  # this rung already grew the mask; skip the remaining rungs
        # tier-3 analogue: the instance's own Molmo point prompts the tracker.
        # Normalized coords, so the (possibly rescaled) edited frame needs no math.
        if point is not None and hasattr(sam3, "segment_points") and not _grew():
            pm, ps = sam3.segment_points(edited_image, [list(point)], str(out_npy))
            if pm is not None:
                _judge(pm, ps, "point:tracker")
        # synonyms LAST (see docstring: hypernym hazard) — only when the precise rungs
        # found nothing that grew the mask.
        if not _grew():
            for prompt in syn_prompts:
                all_masks, scores = sam3.segment_all(edited_image, prompt, str(out_npy))
                if all_masks is None or not len(all_masks):
                    continue
                for m, s in zip(all_masks, scores):
                    _judge(m, s, prompt)
                if _grew():
                    break
    finally:
        if own is not None:
            own.close()
    orig = state["orig"]
    if best is None or (orig is not None and best.sum() <= 1.02 * orig.sum()):
        if orig is None:
            print(
                f"[resegment] {target}: no detections for any rung "
                f"({len(prompts) + len(syn_prompts)} text prompt(s)"
                f"{' + point' if point is not None else ''})"
            )
        elif best is None:
            print(
                f"[resegment] {target}: {n_seen} candidate(s), none acceptable "
                "(disjoint from the original or smaller than it)"
            )
        else:
            print(
                f"[resegment] {target}: best candidate "
                f"{best.sum() / max(orig.sum(), 1):.2f}x the original — no gain, "
                "keeping the original mask (no-op: the edit revealed nothing)"
            )
        if os.path.exists(out_npy):
            os.remove(out_npy)  # no gain over the original mask
        return None
    np.save(out_npy, best)
    return {"mask_path": str(out_npy), "seg_prompt": best_prompt, "score": best_score}


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def generative_resegment(
    scene_dir: str,
    model: str = DEFAULT_MODEL,
    vlm: Any = None,
    qwen: Any = None,
    sam3: Any = None,
) -> dict[str, dict]:
    """The full pipeline over one scene (idempotent). Returns {rid: redetect_record};
    records are also persisted into masks.json under each instance's ``redetect``
    key, and the full run report to ``masks/generative_resegment.json``."""
    scene = Path(scene_dir)
    masks_dir = scene / "masks"
    masks_json = masks_dir / "masks.json"
    image_path = str(scene / "input.png")
    data = json.load(open(masks_json))
    objs = []
    for r in data.get("instances", []):
        if r.get("kind") == "root_surface" or not r.get("mask_path"):
            continue
        mp = r["mask_path"]
        if not os.path.exists(mp):  # stored repo-root-relative; fall back to basename
            mp = str(masks_dir / os.path.basename(mp))
            if not os.path.exists(mp):
                continue
        objs.append({**r, "mask_path": mp})
    from PIL import Image

    w, h = Image.open(image_path).size
    masks = {_rid(r): _fit(np.load(r["mask_path"]) > 0, (h, w)) for r in objs}
    by_id = {_rid(r): r for r in objs}
    insts = {
        f"{r['category']}#{r['instance']}": r
        for r in data.get("instances", [])
        if r.get("mask_path")
    }
    support = {rid: r.get("support") for rid, r in insts.items()}
    desc = {rid: r.get("description") or r["category"] for rid, r in insts.items()}
    # proposer synonyms per category (reseg's synonym rung) — objects may be absent
    # in minimal/bench masks.json files, so default to no synonyms.
    syns = {
        o["category"]: [s for s in (o.get("synonyms") or []) if s]
        for o in data.get("objects", [])
        if isinstance(o, dict) and o.get("category")
    }

    depth = valid = None
    pts_npy = scene / "moge" / "points.npy"
    if pts_npy.exists():
        pts = np.load(pts_npy)
        d = np.linalg.norm(pts, axis=-1).astype(np.float32)
        if d.shape == (h, w):
            depth = d
            vm = scene / "moge" / "mask.npy"
            if vm.exists():
                valid = _fit(np.load(vm) > 0, (h, w))

    vlm = vlm or _make_vlm(
        model, os.environ.get("GRASE_RESEGMENT_VLM_EFFORT", "medium")
    )

    # Stage 1: touching pairs -> pairwise VLM -> edges; border flags geometrically.
    pairs = candidate_pairs(objs, masks, depth, valid)
    judged = [
        judge_pair(
            vlm, image_path, p, by_id[p["a"]], by_id[p["b"]], masks, masks_dir, n
        )
        for n, p in enumerate(pairs)
    ]
    edges = [j for j in judged if j["occluder"]]
    border = {
        i: bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
        for i, m in masks.items()
    }

    # Stage 2: hierarchy gate.
    edges, hier_dropped = filter_hierarchy(edges, support)
    if hier_dropped:
        print(
            "[generative_resegment] hierarchy gate dropped: "
            + "; ".join(f"{e['occluder']} -/-> {e['occluded']}" for e in hier_dropped)
        )
    occluders_of: dict[str, list[str]] = {}
    hides: dict[
        str, list
    ] = {}  # occluded -> [(occluder, missing_part)] for the crop check
    for e in edges:
        occluders_of.setdefault(e["occluded"], []).append(e["occluder"])
        hides.setdefault(e["occluded"], []).append(
            (e["occluder"], e.get("missing_part") or "")
        )

    # Stage 3: vital-part check per instance (DAG occluders given as context).
    vitals = {
        rid: check_vital(
            vlm,
            image_path,
            rid,
            by_id[rid],
            masks[rid],
            masks_dir,
            hides.get(rid, []),
            border[rid],
        )
        for rid in masks
        if masks[rid].any()
    }

    def _mask(rid: str) -> np.ndarray:
        mp = insts[rid]["mask_path"]
        if not os.path.exists(mp):
            mp = str(masks_dir / os.path.basename(mp))
        return np.load(mp) > 0

    def _others_mask(rid: str, removed: tuple = ()) -> Optional[np.ndarray]:
        acc = None
        base = _mask(rid)
        for other in insts:
            if (
                other == rid
                or other in removed
                or insts[other].get("kind") == "root_surface"
            ):
                continue
            m = _fit(_mask(other), base.shape)
            acc = m if acc is None else (acc | m)
        return acc

    # Persist state-aware rollability on every instance (masks.json is written
    # below either way) — the composition physics gates read it via placement.
    for rid, v in vitals.items():
        if rid in insts:
            insts[rid]["rollable"] = bool(v.get("rollable", False))

    # Stage 4: union gate — occluded-by-objects OR vital-part-missing.
    # LanPaint+Qwen edit server (occluder removal + border outpaint): load once, serve
    # every edit this pass, close at the end. Created lazily on first edit so scenes with
    # nothing to redetect never pay the model-load cost.
    _own_qwen = [None]
    # One SAM3 server shared across every redetect/border-completion this pass (F7):
    # created lazily on first use, so a scene with nothing to redetect pays no load.
    # When the caller injects ``sam3`` (e.g. preprocess sharing the initial-segmentation
    # server), that one is used and this stays None.
    _own_sam3 = [None]

    def _sam3():
        if sam3 is not None:
            return sam3
        if _own_sam3[0] is None:
            from lib.tools.geometry.agentic_mask import Sam3Server
            from lib.utils._path import SAM3_PY

            (masks_dir / "edited").mkdir(
                parents=True, exist_ok=True
            )  # log dir must exist
            _own_sam3[0] = Sam3Server(
                SAM3_PY, log_path=str(masks_dir / "edited" / "sam3_redetect.log")
            )
        return _own_sam3[0]

    def _qwen():
        if qwen is not None:
            return qwen
        if _own_qwen[0] is None:
            from lib.tools.geometry.qwen_edit import QwenEditServer

            (masks_dir / "edited").mkdir(
                parents=True, exist_ok=True
            )  # log dir must exist
            _own_qwen[0] = QwenEditServer(
                log_path=str(masks_dir / "edited" / "qwen.log")
            )
        return _own_qwen[0]

    records = {}
    skipped_trivial: dict[str, dict] = {}  # Layer 1 audit
    audits: dict[str, dict] = {}
    audit_effort = os.environ.get("GRASE_RESEGMENT_VLM_EFFORT", "medium")

    def _audit_dir(rid: str, rec: dict) -> Path:
        mode = (
            "border"
            if rec.get("border")
            else "removal"
            if rec.get("removed")
            else "resegment_only"
        )
        return masks_dir / "audits" / _slug(rid) / mode

    def _audit(rid: str, rec: dict) -> bool:
        verdict = audit_resegmentation(
            vlm=vlm,
            scene=scene,
            instance=insts[rid],
            record=rec,
            out_dir=_audit_dir(rid, rec),
            model=model,
            effort=audit_effort,
        )
        audits[rid] = verdict
        if verdict["accepted"]:
            rec["audit"] = verdict
            return True
        # Keep diagnostic images/masks on disk, but do not promote the candidate.
        insts[rid].pop("redetect", None)
        print(
            f"[generative_resegment] {rid}: quality audit {verdict['status']} — "
            "keeping original image/mask; candidate retained under masks/edited"
        )
        return False

    def _reuse_prior(rid: str) -> bool:
        prior = insts[rid].get("redetect")
        if not prior or not os.path.exists(prior.get("mask_path", "")):
            return False
        audit = prior.get("audit")
        if not audit:
            # Explicit grandfathering, not a newly audited pass: migration/re-audit
            # is an offline operation and must not silently spend API tokens here.
            audits[rid] = {"status": "legacy_unverified", "accepted": False}
            print(
                f"[generative_resegment] {rid}: reusing LEGACY UNVERIFIED redetect; "
                "not evaluated by the before/after auditor"
            )
            records[rid] = prior
            return True
        evidence = prepare_evidence(
            scene=scene,
            instance=insts[rid],
            record=prior,
            out_dir=_audit_dir(rid, prior),
        )
        if audit_cache_matches(audit, evidence, model, audit_effort) and audit.get(
            "accepted"
        ):
            audits[rid] = audit
            records[rid] = prior
        else:
            audits[rid] = {
                "status": "unverified",
                "accepted": False,
                "error": "Cached audit no longer matches inputs/settings; explicit re-audit required",
            }
            insts[rid].pop("redetect", None)
            print(
                f"[generative_resegment] {rid}: stale/unaccepted audit; keeping original image/mask"
            )
        return True

    skipped_border: dict[str, dict] = {}  # border-magnitude gate audit
    min_beyond = float(os.environ.get("GRASE_BORDER_MIN_BEYOND", "0.1"))
    min_contact = float(os.environ.get("GRASE_BORDER_MIN_CONTACT", "0.85"))
    try:
        for rid in masks:
            occl = sorted(set(occluders_of.get(rid, [])))
            carve_forced = False  # set by the carved-container override below
            v = vitals.get(rid) or {}
            vital_part = v.get("vital_part") or ""
            heavy = v.get("severity") == "heavy"
            vital = bool(v.get("vital_part_missing"))
            side = border_cut_side(masks[rid])
            beyond = float(v.get("beyond_frame_fraction") or 0.0)
            contact = edge_contact_ratio(masks[rid], side) if side else 0.0
            if side and beyond < min_beyond and contact < min_contact:
                print(
                    f"[generative_resegment] {rid}: touches {side} edge but only "
                    f"~{beyond:.0%} beyond frame (< {min_beyond:.0%}) and edge contact "
                    f"{contact:.2f} (< {min_contact:.2f}); skipping border completion"
                )
                skipped_border[rid] = {
                    "side": side,
                    "beyond_frame_fraction": beyond,
                    "edge_contact_ratio": round(contact, 2),
                }
                side = None
            elif side and beyond < min_beyond:
                print(
                    f"[generative_resegment] {rid}: edge contact {contact:.2f} >= "
                    f"{min_contact:.2f} — border completion fires on GEOMETRY "
                    f"(vlm beyond ~{beyond:.0%} under-fired)"
                )
            if side:
                if _reuse_prior(rid):
                    continue
                removal = build_removal_set(rid, occl, support)
                try:
                    brec = border_complete(
                        str(scene),
                        rid,
                        side,
                        removal,
                        masks,
                        desc,
                        qwen=_qwen(),
                        sam3=_sam3(),
                    )
                except Exception as e:  # noqa: BLE001 - one failed edit must not abort the pass
                    print(
                        f"[generative_resegment] {rid}: border completion failed ({e})"
                    )
                    brec = None
                if brec and not _audit(rid, brec):
                    brec = None
                if brec:
                    records[rid] = brec
                    insts[rid]["redetect"] = brec
                    print(f"[generative_resegment] {rid}: border-completed ({side})")
                    continue
                # else fall through to the standard occl/vital path
            if not occl and not vital:
                continue
            if occl and not (heavy or vital):
                carved = int(insts[rid].get("carved_px") or 0)
                if carved and not trivial_seam(carved, int((_mask(rid) > 0).sum())):
                    carve_forced = True
                    print(
                        f"[generative_resegment] {rid}: crop damage is slight but "
                        f"{carved}px were carved out (occupied by an occluder) — "
                        "forcing redetect"
                    )
                else:
                    # Occluded, but the conditioning crop is barely damaged: an edit
                    # cannot gain enough to justify its cost/risk. Recorded (report +
                    # demo badge), not redetected.
                    print(
                        f"[generative_resegment] {rid}: occluded but crop damage is "
                        "slight; skipping redetect"
                    )
                    continue
            if _reuse_prior(rid):
                continue
            removal = build_removal_set(rid, occl, support)
            if removal and heavy and not vital:
                # Layer 1: objective seam floor overriding the (subjective) heavy
                # verdict — a trivial target/occluder seam bounds the possible reveal,
                # so no VLM visibility estimate justifies the edit's cost or its
                # look-alike-regeneration risk (FP egg#3: 104px/2.1%). Vital claims
                # are exempt (a vital part can hide behind a narrow seam) — Layer 2
                # still verifies whatever the edit produces.
                seam = occlusion_seam(_mask(rid), [_mask(o) for o in removal])
                area = int((_mask(rid) > 0).sum())
                if trivial_seam(seam, area):
                    print(
                        f"[generative_resegment] {rid}: occluder seam is trivial "
                        f"({seam}px, {seam / max(area, 1) * 100:.1f}% of target) — "
                        "skipping redetect"
                    )
                    skipped_trivial[rid] = {"seam_px": seam, "target_area": area}
                    continue
            rec = {"vital_part": vital_part}
            if removal:
                # Members outside the target's support subtree are foreground blockers,
                # not items resting on the target. Tell the edit explicitly to reveal the
                # target in that case; the same partition is reused below for mask clipping.
                subtree = {o for o in removal if support.get(o) == rid}
                _grew_sub = True
                while _grew_sub:
                    _grew_sub = False
                    for o in removal:
                        if o not in subtree and support.get(o) in subtree:
                            subtree.add(o)
                            _grew_sub = True
                foreign_ids = [o for o in removal if o not in subtree]
                try:
                    rec.update(
                        remove_objects(
                            str(scene),
                            rid,
                            removal,
                            {o: _mask(o) for o in removal},
                            desc,
                            _qwen(),
                            target_category=(
                                insts[rid]["category"] if foreign_ids else None
                            ),
                        )
                    )
                except Exception as e:  # noqa: BLE001 - one blocked/failed edit must not
                    # abort the whole pass
                    print(f"[generative_resegment] {rid}: edit failed ({e}); skipping")
                    continue
                # Foreign removed occluders = removal-set members OUTSIDE the target's
                # support subtree (supported by the target directly, or transitively
                # through other removed occluders). Their footprints are viewpoint
                # occlusion, not resting contact — reseg must not claim them wholesale.
                foreign = [masks[o] for o in foreign_ids if o in masks]
                seg = resegment(
                    str(scene),
                    rid,
                    insts[rid]["category"],
                    rec["edited_image"],
                    _mask(rid),
                    vital_part=vital_part,
                    sam3=_sam3(),
                    exclude_mask=_others_mask(rid, removed=tuple(removal)),
                    synonyms=syns.get(insts[rid]["category"]),
                    point=insts[rid].get("point"),
                    foreign_occluders=foreign,
                )
                if seg is None and carve_forced:
                    # Carve-forced redetects exist for the CLEAN IMAGE, not an amodal
                    # mask — SAM3 failing to re-ground the category on the edited image
                    # (this notebook needed the tier-3 tracker originally) must not
                    # throw the successful occluder removal away. Undo the carve for
                    # the occluder-free frame: original mask + the removed occluders'
                    # footprints (>80% of each sat ON this object by the carve
                    # precondition). The before/after quality auditor still gates it.
                    um = _mask(rid) > 0
                    for o in removal:
                        um = um | (_fit(_mask(o), um.shape) > 0)
                    from PIL import Image as _EImg

                    ew, eh = _EImg.open(rec["edited_image"]).size
                    if um.shape != (eh, ew):
                        um = _fit(um, (eh, ew))
                    fb = masks_dir / f"{_slug(rid)}_carvefb.npy"
                    np.save(fb, um.astype(np.uint8) * 255)
                    seg = {"mask_path": str(fb), "carve_fallback": True}
                    print(
                        f"[generative_resegment] {rid}: reseg failed on the edited "
                        "image — keeping original+occluder mask (carve-forced fallback)"
                    )
                if seg is None:
                    print(
                        f"[generative_resegment] {rid}: re-segmentation failed; keeping "
                        "the original mask (no redetect record)"
                    )
                    continue
                rec.update(seg)
            elif vital_part:
                # No removable occluder (segmentation miss / border cut): re-segment the
                # ORIGINAL image with the part-aware prompt; no edit.
                rec["edited_image"] = str(scene / "input.png")
                seg = resegment(
                    str(scene),
                    rid,
                    insts[rid]["category"],
                    rec["edited_image"],
                    _mask(rid),
                    vital_part=vital_part,
                    sam3=_sam3(),
                    exclude_mask=_others_mask(rid),
                    synonyms=syns.get(insts[rid]["category"]),
                    point=insts[rid].get("point"),
                )
                if seg is None:
                    print(
                        f"[generative_resegment] {rid}: part-aware re-segmentation on "
                        "the original image found no gain; keeping the original mask"
                    )
                    continue
                rec.update(seg)
            else:
                continue  # border-only occlusion, no vital part: nothing to recover
            if not _audit(rid, rec):
                continue
            records[rid] = rec
            insts[rid]["redetect"] = rec
    finally:
        if _own_qwen[0] is not None:
            _own_qwen[0].close()
        if _own_sam3[0] is not None:
            _own_sam3[0].close()
    with open(masks_json, "w") as f:
        json.dump(data, f, indent=2)
    report = {
        "pairs": judged,
        "edges": edges,
        "dropped_edges": hier_dropped,
        "vitals": vitals,
        "border": border,
        "redetected": sorted(records),
        "skipped_trivial_seam": skipped_trivial,
        "resegmentation_audits": audits,
        "rejected_multi_instance": {
            rid: audit
            for rid, audit in audits.items()
            if (audit.get("verdict") or {})
            .get("checks", {})
            .get("target_identity_and_count")
            == "fail"
        },
        "skipped_border_slight": skipped_border,
    }
    with open(masks_dir / "generative_resegment.json", "w") as f:
        json.dump(report, f, indent=2)
    return records
