"""Per-object instance masking via MolmoPoint + SAM3 (inspired by REST3D A.1).

On the single input image:
  1. A VLM proposes an open-vocabulary object and root-surface list with
     disambiguating per-instance descriptions.
  2. For each category, including root surfaces:
       - SAM3 text prompts return candidate masks for that category.
       - MolmoPoint-8B points to each instance's detailed description.
       - The containing mask is selected, with crop+text and tracker-point fallbacks.
     A mask confirmed by a Molmo point is a verified instance: two independent
     models agreeing replaces the old text-refine + VLM-verify loop.
  3. If a MoGE point map is supplied, each confirmed instance also yields the
     median camera-space 3D point + depth (for metric placement downstream).

Both heavy models run as persistent servers in their isolated venvs
(see lib/utils/_path.py): SAM3 in the sam3 venv, MolmoPoint in the molmo venv.
The VLM object-list call goes through the project's OpenAI-compatible client.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from lib.tools.geometry.surface_relations import (
    parse_relationships,
    relationship_glossary,
)

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DEFAULT_MODEL = "claude-opus-5"
SAM3_SERVER = os.path.join(REPO_ROOT, "lib", "tools", "sam3d", "sam3_server.py")
MOLMO_SERVER = os.path.join(REPO_ROOT, "lib", "tools", "geometry", "molmo_server.py")
SAM3D_SERVER = os.path.join(REPO_ROOT, "lib", "tools", "sam3d", "sam3d_server.py")

# Surfaces: listed for scene context but not reconstructed as SAM3D assets.
# Matched on the HEAD noun (last word) so "wooden table" -> surface but
# "desk mat" / "background object" stay objects.
SURFACE_HEADS = frozenset(
    {
        "floor",
        "table",
        "desk",
        "wall",
        "ground",
        "ceiling",
        "backdrop",
        "background",
        "countertop",
        "tabletop",
        "floorboards",
        "counter",
        "panel",
        "barrier",
        "divider",
    }
)


def is_duplicate_mask(
    mb: np.ndarray,
    claimed: list,
    kinds: list,
    dedup_iou: float,
    kind: Optional[str] = None,
) -> bool:
    ""
    return any(
        _mask_iou(mb, c) > dedup_iou
        for c, k in zip(claimed, kinds)
        if kind is None or k == kind
    )


def salvage_eligible(reason: str) -> bool:
    """Drop reasons the Molmo-collapse salvage may retry (single source of truth).

    ROOT surfaces record ``root_unsegmented`` where objects record ``unsegmented``; the
    old exact-equality test excluded every ladder-failed root while collapsed/superseded
    roots stayed eligible. Applicability is decided by the salvage's own point test, not
    by the reason string."""
    r = str(reason or "")
    return r in ("unsegmented", "root_unsegmented") or r.startswith(
        ("collapsed_into", "superseded_by")
    )


def is_surface(name: str) -> bool:
    tokens = re.findall(r"[a-z]+", name.lower())
    if not tokens:
        return False
    return tokens[-1] in SURFACE_HEADS or name.strip().lower() in SURFACE_HEADS


# A bare supporting-surface noun ("table") often fails to ground when the real
# surface is e.g. a butcher-block/cutting-board; retry these synonyms (in order)
# for a SUPPORTING root surface whose own category text grounded nothing.
_SURFACE_SYNONYMS = ("tabletop", "countertop", "wooden table")
_SUPPORT_HEADS = frozenset(
    {
        "table",
        "tabletop",
        "desk",
        "counter",
        "countertop",
        "worktop",
        "bench",
        "board",
        "block",  # cutting board / butcher block
    }
)


def _is_support_head(name: str) -> bool:
    """True if ``name``'s head noun denotes a horizontal SUPPORTING surface (table /
    counter / cutting-board), i.e. one the surface-synonym retry should apply to."""
    tokens = re.findall(r"[a-z]+", name.lower())
    return bool(tokens) and tokens[-1] in _SUPPORT_HEADS


@dataclass(kw_only=True, slots=True)
class InstanceMask:
    category: str
    instance: int
    kind: str = "object"  # "object" | "root_surface"
    support: Optional[str] = None  # category this rests on/attaches to (VLM prior)
    description: Optional[str] = (
        None  # VLM's detailed pointing prompt for this instance
    )
    point: Optional[list[float]] = None  # normalized (u, v) Molmo point
    score: Optional[float] = None  # SAM3 mask score
    mask_path: Optional[str] = None
    overlay_path: Optional[str] = None
    point_cam: Optional[list[float]] = None  # median MoGE camera-space point
    depth: Optional[float] = None
    tier: Optional[int] = (
        None  # masking tier: 1 full-image text, 2 crop+text, 3 tracker point
    )
    source: Optional[str] = (
        None  # provenance: None = proposer instance; "sam3_recount" = added by the
        # under-count fallback (SAM3-detected + Molmo-corroborated), see _recount
    )
    # Compatibility projection only.  The proposer/masker always writes False; the final
    # retained-inventory registry sets it True only after a group is successfully normalized.
    same_size: bool = False
    restored_px: Optional[int] = None  # pixels given back by restore_severed_components
    # (occluder-severed fragments keep_own_components wrongly trimmed); None = untouched.
    carved_px: Optional[int] = None  # pixels removed from this mask by carve_overlaps
    # (the post-merge cleanup that separates a swallowed neighbour); None = untouched.
    overlap_with: Optional[list[str]] = None  # ids this instance's dedup was DEFERRED
    # against (dedup -> merge -> carve reorder): merge_objects consumes this list so
    # deferred pairs never Rule-A auto-merge; carve recomputes pairs from final masks.

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "instance": self.instance,
            "kind": self.kind,
            "support": self.support,
            "description": self.description,
            "point": self.point,
            "score": self.score,
            "mask_path": self.mask_path,
            "overlay_path": self.overlay_path,
            "point_cam": self.point_cam,
            "depth": self.depth,
            "tier": self.tier,
            "source": self.source,
            "same_size": self.same_size,
            "restored_px": self.restored_px,
            "carved_px": self.carved_px,
            "overlap_with": self.overlap_with,
        }


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                   #
# --------------------------------------------------------------------------- #
def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower()[:48] or "object"


def parse_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response (tolerates fences and
    surrounding prose). Scans each ``{``/``[`` start with ``raw_decode`` and returns the first
    BALANCED, valid payload — the old greedy ``\\{.*\\}`` regex grabbed from the first ``{`` to
    the LAST ``}``, so valid JSON followed by prose containing a brace (or a second object)
    raised mid-decode and killed the run (RC6, INIT_FAILURE_ANALYSIS_0702.md). Raises
    ``ValueError`` (with a snippet) when nothing in the text parses."""
    if text is None:
        raise ValueError("empty response")
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                return dec.raw_decode(t, i)[0]
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no valid JSON found in response: {t[:200]}")


_RAISE = object()  # sentinel: vlm_json without a default re-raises on exhaustion


def vlm_json(
    vlm: Callable[..., str],
    system: str,
    parts: list,
    *,
    max_tokens: int = 700,
    retries: int = 3,
    label: str = "vlm_json",
    default: Any = _RAISE,
    validate: Optional[Callable[[dict], dict]] = None,
) -> Any:
    ""
    err = ""
    for attempt in range(retries):
        retry_parts = parts + (
            [
                {
                    "type": "text",
                    "text": (
                        f"Your previous reply was invalid ({err}). "
                        "Return ONLY the single JSON object — no prose, no code fences."
                    ),
                }
            ]
            if err
            else []
        )
        try:
            out = parse_json(vlm(system, retry_parts, max_tokens=max_tokens))
            if not isinstance(out, dict):
                raise ValueError(f"expected a JSON object, got {type(out).__name__}")
            if validate is not None:
                out = validate(out)
                if not isinstance(out, dict):
                    raise ValueError("JSON validator must return an object")
            return out
        except (ValueError, json.JSONDecodeError) as e:
            err = str(e)[:300]
            print(f"[{label}] bad VLM JSON (attempt {attempt + 1}/{retries}): {err}")
    if default is _RAISE:
        raise ValueError(f"{label}: unparseable VLM JSON {retries}x — last: {err}")
    print(f"[{label}] giving up after {retries} attempts; falling back to {default!r}")
    return default


def require_json_contract(
    out: dict,
    *,
    bool_fields: tuple[str, ...] = (),
    int_fields: tuple[str, ...] = (),
    unit_interval_fields: tuple[str, ...] = (),
    enum_fields: Optional[dict[str, set[str]]] = None,
) -> dict:
    """Validate the small typed JSON contracts used by preprocessing VLM calls.

    Python truthiness is deliberately forbidden here: ``"false"`` is a non-empty
    string and therefore truthy, so accepting it as a boolean silently reverses a
    model decision. Validators raise ``ValueError`` so ``vlm_json`` can re-ask using
    its normal invalid-reply path.
    """
    for field in bool_fields:
        if type(out.get(field)) is not bool:
            raise ValueError(f"{field} must be a JSON boolean")
    for field in int_fields:
        if type(out.get(field)) is not int:
            raise ValueError(f"{field} must be a JSON integer")
    for field in unit_interval_fields:
        value = out.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a number in [0, 1]")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be a finite number in [0, 1]")
        out[field] = value
    for field, allowed in (enum_fields or {}).items():
        value = out.get(field)
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{field} must be one of {sorted(allowed)}")
    return out


def _binarize(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] in (1, 3) else mask[0]
    return mask > 0


def _nonempty_masks(masks: np.ndarray, scores: list):
    """Drop all-zero masks (+ their scores) from a SAM3 ``(N,H,W)`` stack; ``(None, [])`` if
    none remain. SAM3 can report a positive-score detection whose mask is empty after the
    server's >0.5 threshold — returning it as a valid instance silently poisoned tier-1
    (the point path already guards this, so the two entry points were inconsistent). F3."""
    keep = [i for i in range(len(masks)) if int((masks[i] > 0).sum()) > 0]
    if not keep:
        return None, []
    if len(keep) == len(masks):
        return masks, scores
    return masks[keep], [scores[i] for i in keep if i < len(scores)]


_PLACEMENT_CLAUSE = re.compile(
    r"\s+(?:standing|lying|sitting|resting|leaning|tipped|placed|positioned|located"
    r"|propped|stacked|tucked|hanging|growing|rising|extending)\b.*$",
    re.IGNORECASE,
)
_TRAILING_PAREN = re.compile(r"\s*(\([^)]*\))\s*$")

_BARE_PREP_JOIN = re.compile(
    r"(\b(?:at|near|on)\s+the\s+[a-z\- ]{2,24}?)\s+(?:inside|in|within)\s+(the\b)",
    re.IGNORECASE,
)


def rejoin_with_of(q: str) -> str:
    """ "... at the back left inside the bowl" -> "... at the back left of the bowl".
    Returns ``q`` unchanged when the positional+bare-preposition shape is absent."""
    out = _BARE_PREP_JOIN.sub(r"\1 of \2", q, count=1)
    return out if out != q else q


def strip_placement_clause(q: str) -> str:
    """Head noun phrase for the Molmo rung-3 retry ("red mug standing upright in the
    middle of the table" -> "red mug"), preserving a trailing parenthetical. Returns
    ``q`` unchanged when there is nothing to cut."""
    m = _TRAILING_PAREN.search(q)
    suffix = f" {m.group(1)}" if m else ""
    body = q[: m.start()] if m else q
    head = _PLACEMENT_CLAUSE.sub("", body).strip()
    if not head or head + suffix.strip() == q:
        return q
    return head + suffix


def point_in_mask(mask: np.ndarray, u: float, v: float) -> bool:
    """Is normalized point (u, v) inside the (H, W) binary mask?"""
    b = _binarize(mask)
    h, w = b.shape
    px = min(max(int(round(u * (w - 1))), 0), w - 1)
    py = min(max(int(round(v * (h - 1))), 0), h - 1)
    return bool(b[py, px])


def keep_own_components(mask: np.ndarray, own_points) -> np.ndarray:
    """Trim an object mask to ONLY the connected component(s) that hold this instance's own Molmo
    point(s), dropping every other component — whether it carries a neighbour's point (a swallowed
    object, e.g. a 'power bank' mask that also grabbed the key fob) OR no point at all (a stray
    fragment the SAM blob attached, e.g. the charging plug fused onto the power bank). A single
    object is one connected region, so anything not under our own point is not us.

    ``own_points`` are this instance's normalized ``[u, v]`` points in [0, 1] (Molmo returns a list,
    so a genuinely multi-part object with a point per part keeps every pointed component). Best-
    effort: returns the binarized input unchanged if SciPy is unavailable, the mask is one
    component, or the point lands off every component (so an ambiguous point never erases a mask)."""
    b = _binarize(mask)
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001 - cleanup is best-effort; fall back to the point-count guard
        return b
    lab, n = ndimage.label(b)
    if n <= 1:
        return b
    h, w = b.shape

    def _comp(uv) -> int:
        u, v = uv
        return int(lab[min(int(v * h), h - 1), min(int(u * w), w - 1)])

    own = {_comp(p) for p in own_points} - {0}
    if not own:
        return b  # point off every component -> leave mask intact
    return b & np.isin(lab, list(own))  # keep ONLY this object's own-point component(s)


def restore_severed_components(
    results: list["InstanceMask"],
    trimmed: dict[tuple, np.ndarray],
    inst_points: dict[tuple, list],
    image_path: Optional[str] = None,
    min_px: int = 50,
    max_overlap: float = 0.10,
    bridge_halo_px: int = 3,
) -> int:
    ""
    if not trimmed:
        return 0
    try:
        from scipy import ndimage
    except Exception:  # noqa: BLE001 - cleanup is best-effort
        return 0
    obj_masks: dict[tuple, np.ndarray] = {}
    for r in results:
        if r.kind == "object" and r.mask_path and os.path.exists(r.mask_path):
            obj_masks[(r.category, r.instance)] = _binarize(np.load(r.mask_path))
    eight = np.ones((3, 3), bool)
    n_restored = 0
    for key, removed in trimmed.items():
        rec = next(
            (r for r in results if (r.category, r.instance) == key), None
        )  # dropped/collapsed since the trim -> nothing to restore
        if rec is None or rec.kind == "root_surface":
            continue
        own = obj_masks.get(key)
        if own is None or own.shape != removed.shape:
            continue
        others = [m for k, m in obj_masks.items() if k != key and m.shape == own.shape]
        if not others:
            continue  # no occluder can exist -> nothing can bridge
        others_union = np.zeros_like(own)
        for m in others:
            others_union |= m
        grown = others_union
        for _ in range(max(1, bridge_halo_px)):
            grown = _dilate1(grown)
        bridge = own | removed | grown
        lab_b, _ = ndimage.label(bridge, structure=eight)
        own_ids = set(np.unique(lab_b[own])) - {0}
        h, w = own.shape
        foreign = [p for k, pl in inst_points.items() if k != key for p in (pl or [])]
        lab_r, n = ndimage.label(removed)
        restored = np.zeros_like(own)
        for k in range(1, n + 1):
            comp = lab_r == k
            npx = int(comp.sum())
            if npx < min_px:
                continue
            if any(  # (i) another instance's point -> swallowed neighbour, not us
                comp[min(int(v * h), h - 1), min(int(u * w), w - 1)] for u, v in foreign
            ):
                continue
            if int((comp & others_union).sum()) > max_overlap * npx:  # (ii)
                continue
            if not (set(np.unique(lab_b[comp])) - {0}) & own_ids:  # (iii) no bridge
                continue
            restored |= comp
        if not restored.any():
            continue
        merged = own | restored
        np.save(rec.mask_path, merged.astype(np.uint8) * 255)
        obj_masks[key] = merged
        rec.restored_px = int(restored.sum())
        if image_path and rec.overlay_path:
            make_overlay(image_path, merged.astype(np.uint8) * 255, rec.overlay_path)
        n_restored += 1
        print(
            f"[masking] restored severed component(s) ({rec.restored_px}px) for "
            f"{key[0]}#{key[1]} — reconnected through another object's mask"
        )
    return n_restored


def clamp_crop_aspect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    w: int,
    h: int,
    lo: float = 0.5,
    hi: float = 2.0,
) -> tuple[int, int, int, int]:
    """Keep a crop's aspect (height/width) inside [lo, hi] by GROWING the shorter
    side with neighboring image region — padded symmetrically, shifting one-sided at
    image borders (the crop_box convention), capped at the image dimension. An
    extreme sliver crop (a 10:1 book spine / upright marker) otherwise reaches the
    VLM at degenerate resolution after the API's own rescale."""
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return x0, y0, x1, y1
    if ch > hi * cw:  # too tall -> widen to ratio == hi
        need = min(int(-(-ch // hi)), w)
        x0 -= (need - cw) // 2
        x1 = x0 + need
        if x0 < 0:
            x0, x1 = 0, need
        if x1 > w:
            x1, x0 = w, w - need
    elif ch < lo * cw:  # too wide -> heighten to ratio == lo
        need = min(int(-(-(cw * lo) // 1)), h)
        y0 -= (need - ch) // 2
        y1 = y0 + need
        if y0 < 0:
            y0, y1 = 0, need
        if y1 > h:
            y1, y0 = h, h - need
    return x0, y0, x1, y1


def crop_box(
    cx: float, cy: float, w: int, h: int, fw: float = 1 / 3, fh: float = 1 / 3
) -> tuple[int, int, int, int]:
    """A ``(h*fh) x (w*fw)`` crop centered on pixel ``(cx, cy)``, clamped inside the
    image by shifting the opposite edge (the crop keeps its size and never leaves the
    image). Returns ``(x0, y0, x1, y1)``."""
    cw, ch = int(round(w * fw)), int(round(h * fh))
    x0 = int(round(cx - cw / 2))
    x1 = x0 + cw
    if x0 < 0:
        x0, x1 = 0, cw
    if x1 > w:
        x1, x0 = w, w - cw
    y0 = int(round(cy - ch / 2))
    y1 = y0 + ch
    if y0 < 0:
        y0, y1 = 0, ch
    if y1 > h:
        y1, y0 = h, h - ch
    return x0, y0, x1, y1


def _centroid_in(mask: np.ndarray, win: tuple[int, int, int, int]) -> bool:
    """True if ``mask``'s pixel centroid falls inside window ``(x0, y0, x1, y1)``."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return False
    x0, y0, x1, y1 = win
    return x0 <= xs.mean() < x1 and y0 <= ys.mean() < y1


def best_mask_for_point(
    masks: np.ndarray, u: float, v: float, nest_thresh: float = 0.8
) -> Optional[int]:
    """Index of the SAM3 mask that owns normalized point (u, v), or None.

    Start from the smallest mask containing the point, but if it is
    >= ``nest_thresh`` contained in a larger candidate (part/whole of the same
    object, e.g. SAM3 splitting a robot arm into upper + full), prefer the larger
    (whole) mask. Distinct instances that only partially overlap keep the smaller.
    """
    n = masks.shape[0]
    candidates = [i for i in range(n) if point_in_mask(masks[i], u, v)]
    if not candidates:
        return None
    bmasks = [masks[i] > 0 for i in candidates]
    area = {i: int(b.sum()) for i, b in zip(candidates, bmasks)}
    bmap = dict(zip(candidates, bmasks))
    best = min(candidates, key=lambda i: area[i])
    improved = True
    while improved:  # walk up the nesting chain to the whole-object mask
        improved = False
        for j in candidates:
            if (
                area[j] > area[best]
                and area[best] > 0
                and (int((bmap[best] & bmap[j]).sum()) / area[best] >= nest_thresh)
            ):
                best, improved = j, True
                break
    return best


def assign_points_to_masks(
    masks: np.ndarray,
    scores: list[float],
    points: list[list[float]],
    nest_thresh: float = 0.8,
) -> list[dict[str, Any]]:
    """Match Molmo points to SAM3 instance masks by containment.

    masks: (N, H, W). Returns one record per confirmed mask (a mask containing
    >=1 point): {mask_idx, point, score}. First point wins per mask.
    """
    confirmed: dict[int, list[float]] = {}
    for u, v in points:
        best = best_mask_for_point(masks, u, v, nest_thresh)
        if best is not None:
            confirmed.setdefault(best, [u, v])
    return [
        {
            "mask_idx": i,
            "point": confirmed[i],
            "score": float(scores[i]) if i < len(scores) else None,
        }
        for i in sorted(confirmed)
    ]


def make_overlay(image_path: str, mask: np.ndarray, out_path: str) -> None:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    b = _binarize(mask)
    m = np.asarray(Image.fromarray((b.astype("uint8")) * 255).resize(img.size)) > 127
    arr = np.array(img).astype(np.float32)
    arr[m] = 0.5 * arr[m] + 0.5 * np.array([255.0, 40.0, 40.0])
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(out_path)


_PALETTE = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
    (174, 199, 232),
    (255, 187, 120),
    (152, 223, 138),
    (255, 152, 150),
    (197, 176, 213),
    (196, 156, 148),
    (247, 182, 210),
    (199, 199, 199),
    (219, 219, 141),
    (158, 218, 229),
]


def make_composite(
    image_path: str, instances: list["InstanceMask"], out_path: str
) -> None:
    """Tint every instance mask in its own colour + draw a numbered point anchor,
    so the verifier can audit the whole segmentation at a glance."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    for i, r in enumerate(instances):
        if not r.mask_path or not os.path.exists(r.mask_path):
            continue
        b = _binarize(np.load(r.mask_path))
        if b.shape != (h, w):
            b = (
                np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((w, h)))
                > 127
            )
        arr[b] = 0.5 * arr[b] + 0.5 * np.array(_PALETTE[i % len(_PALETTE)], float)
    comp = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    d = ImageDraw.Draw(comp)
    for i, r in enumerate(instances):
        if not r.point:
            continue
        x, y = r.point[0] * w, r.point[1] * h
        c = _PALETTE[i % len(_PALETTE)]
        d.ellipse(
            [x - 8, y - 8, x + 8, y + 8], fill=c, outline=(255, 255, 255), width=2
        )
        d.text((x + 10, y - 7), str(i), fill=(255, 255, 255))
    comp.save(out_path)


# NAMEABLE tints for the surfaces-only overlay the verifier audits. The id list names
# each surface's colour, so the verifier can map a tinted region back to an id instead of
# guessing from position — there are only ever a handful of root surfaces, so plain colour
# words are unambiguous where the 20-tone tab20 palette of `_PALETTE` is not.
_SURFACE_COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (230, 30, 30)),
    ("green", (40, 200, 60)),
    ("blue", (40, 100, 235)),
    ("yellow", (245, 220, 40)),
    ("magenta", (235, 60, 215)),
    ("cyan", (45, 220, 225)),
    ("orange", (255, 145, 20)),
    ("purple", (150, 65, 220)),
]


def make_surface_composite(
    image_path: str, surfaces: list["InstanceMask"], out_path: str
) -> dict[str, str]:
    """Tint ONLY the root surfaces, each in a NAMED colour; return ``{id: colour}``.

    The verifier corrects surfaces and relationships and never touches an object, so object
    tints were pure distraction — worse, they drowned out the very thing it must judge: a
    wall mask spanning the whole background reads as "no wall is masked" once a dozen object
    tints sit on top of it (0731_init_foodpack re-added an already-masked wall#0). Objects
    stay as ordinary photo pixels, so occlusion still reads correctly."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    arr = np.array(img).astype(np.float32)
    colors: dict[str, str] = {}
    for i, r in enumerate(surfaces):
        if not r.mask_path or not os.path.exists(r.mask_path):
            continue
        name, rgb = _SURFACE_COLORS[i % len(_SURFACE_COLORS)]
        colors[f"{r.category}#{r.instance}"] = name
        b = _binarize(np.load(r.mask_path))
        if b.shape != (h, w):
            b = (
                np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((w, h)))
                > 127
            )
        arr[b] = 0.5 * arr[b] + 0.5 * np.array(rgb, float)
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(out_path)
    return colors


def object_point_from_mask(
    mask: np.ndarray, points: np.ndarray
) -> tuple[Optional[list[float]], Optional[float]]:
    """Median camera-space 3D point + depth of the masked pixels (MoGE points)."""
    b = _binarize(mask)
    h, w = points.shape[:2]
    if b.shape != (h, w):
        from PIL import Image

        b = np.array(Image.fromarray(b.astype("uint8") * 255).resize((w, h))) > 127
    sel = b & np.isfinite(points).all(axis=2)
    pts = points[sel]
    if pts.size == 0:
        return None, None
    med = np.median(pts, axis=0)
    return [float(x) for x in med], float(med[2])


# --------------------------------------------------------------------------- #
# Persistent model servers                                                     #
# --------------------------------------------------------------------------- #
class _JsonServer:
    """Spawn a stdin/stdout JSON-line server and wait for {"ready": true}."""

    def __init__(
        self, py: str, script: str, log_path: Optional[str], ready_timeout: float
    ):
        self._log = open(log_path, "w") if log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            [py, script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
            cwd=REPO_ROOT,
        )
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"{script} exited before ready")
            try:
                if json.loads(line).get("ready"):
                    return
            except json.JSONDecodeError:
                continue
        raise RuntimeError(f"{script} not ready in time")

    def _rpc(self, req: dict) -> Optional[dict]:
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def close(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            self.proc.kill()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()


class Sam3Server(_JsonServer):
    """SAM3 text -> all instance masks (+scores)."""

    def __init__(
        self, py: str, log_path: Optional[str] = None, ready_timeout: float = 600.0
    ):
        super().__init__(py, SAM3_SERVER, log_path, ready_timeout)

    def segment_all(self, image_path: str, category: str, out_npy: str):
        """Returns (masks (N,H,W) uint8, scores) or (None, []) if no detection."""
        Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
        resp = self._rpc(
            {
                "image": image_path,
                "object": category,
                "out": out_npy,
                "all_instances": True,
            }
        )
        if not resp or not resp.get("ok") or not os.path.exists(out_npy):
            return None, []
        raw = np.load(out_npy)
        masks, scores = _nonempty_masks(raw, resp.get("scores", []))
        if masks is None:
            return None, []  # every detection was all-zero after threshold (F3)
        if len(masks) != len(raw):  # keep the on-disk stack consistent with the return
            np.save(out_npy, masks)
        return masks, scores

    def segment_points(
        self,
        image_path: str,
        points: list[list[float]],
        out_npy: str,
        labels: Optional[list[int]] = None,
    ) -> tuple[Optional[np.ndarray], float]:
        """Interactive (SAM2-style) point segmentation -> the WHOLE object at the click(s).
        ``points`` are normalized [u, v] (1 for single-click, 2+ to refine). Returns
        ``(mask HxW uint8, iou_quality)`` or ``(None, 0.0)``."""
        Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
        req = {
            "image": image_path,
            "out": out_npy,
            "points": [[float(u), float(v)] for u, v in points],
        }
        if labels:
            req["labels"] = labels
        resp = self._rpc(req)
        if not resp or not resp.get("ok") or not os.path.exists(out_npy):
            return None, 0.0
        m = np.load(out_npy)
        return (
            (m, float(resp.get("iou", 0.0))) if int((m > 0).sum()) > 0 else (None, 0.0)
        )

    def segment_point_candidates(
        self,
        image_path: str,
        points: list[list[float]],
        out_npy: str,
    ) -> tuple[Optional[np.ndarray], list[float]]:
        ""
        Path(out_npy).parent.mkdir(parents=True, exist_ok=True)
        resp = self._rpc(
            {
                "image": image_path,
                "out": out_npy,
                "points": [[float(u), float(v)] for u, v in points],
                "all_candidates": True,
            }
        )
        if not resp or not resp.get("ok") or not os.path.exists(out_npy):
            return None, []
        stack = np.load(out_npy)
        if stack.ndim == 2:
            stack = stack[None]
        return stack, [float(x) for x in resp.get("ious", [])]


class MolmoServer(_JsonServer):
    """MolmoPoint 'point to <object>' -> normalized (u,v) per instance."""

    def __init__(
        self, py: str, log_path: Optional[str] = None, ready_timeout: float = 900.0
    ):
        super().__init__(py, MOLMO_SERVER, log_path, ready_timeout)

    def point(self, image_path: str, obj: str) -> list[list[float]]:
        ""
        q = re.sub(r"^(the|a|an)\s+", "", obj.strip(), flags=re.IGNORECASE)
        resp = self._rpc({"image": image_path, "object": q})
        pts = (resp or {}).get("points", []) if (resp or {}).get("ok") else []
        if not pts and "," in q:
            q2 = q.split(",")[0].strip()
            if q2 and q2 != q:
                resp = self._rpc({"image": image_path, "object": q2})
                pts = (resp or {}).get("points", []) if (resp or {}).get("ok") else []
        if not pts:
            q3 = strip_placement_clause(q)
            if q3 != q:
                resp = self._rpc({"image": image_path, "object": q3})
                p3 = (resp or {}).get("points", []) if (resp or {}).get("ok") else []
                if len(p3) == 1:
                    pts = p3
        if not pts:
            q4 = rejoin_with_of(q)
            if q4 != q:
                resp = self._rpc({"image": image_path, "object": q4})
                p4 = (resp or {}).get("points", []) if (resp or {}).get("ok") else []
                if len(p4) == 1:
                    pts = p4
        return [[float(u), float(v)] for u, v in pts]


class Sam3dServer(_JsonServer):
    """SAM3D (image + mask) -> reconstructed mesh GLB. Loads the ~13 GB pipeline
    once; far faster than re-spawning sam3d_worker per object."""

    def __init__(
        self, py: str, log_path: Optional[str] = None, ready_timeout: float = 900.0
    ):
        super().__init__(py, SAM3D_SERVER, log_path, ready_timeout)

    def reconstruct(
        self,
        image_path: str,
        mask_npy: str,
        out_glb: str,
        info_json: Optional[str] = None,
        pristine_glb: Optional[str] = None,
        pointmap_npy: Optional[str] = None,
    ) -> Optional[dict]:
        """Returns the info dict ({glb_path, translation, rotation, scale, ...}) or
        None on failure. ``pristine_glb`` (optional) also saves the UNtransformed canonical
        mesh, used as the upright fallback for ruled-out (weird-pose) objects.
        ``pointmap_npy`` (optional) is a MoGE camera-space point map ``(H, W, 3)`` used as
        the reconstruction condition instead of SAM3D re-estimating depth internally."""
        req = {"image": image_path, "mask": mask_npy, "glb": out_glb}
        if info_json:
            req["info"] = info_json
        if pristine_glb:
            req["pristine"] = pristine_glb
        if pointmap_npy:
            req["pointmap"] = pointmap_npy
        resp = self._rpc(req)
        if not resp:
            raise RuntimeError("SAM3D server exited without a response")
        if not resp.get("ok"):
            raise RuntimeError(f"SAM3D reconstruction failed: {resp.get('error', 'unknown error')}")
        return resp


# --------------------------------------------------------------------------- #
# VLM object-list proposer                                                     #
# --------------------------------------------------------------------------- #
def _make_vlm(model: str, effort: str = "high") -> Callable[[str, list], str]:
    ""
    from lib.utils.common import build_client, get_model_response

    client = build_client(model)

    def vlm(system: str, user_parts: list, max_tokens: int = 700) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_parts},
        ]
        resp = get_model_response(
            client,
            {"model": model, "messages": messages, "max_tokens": max_tokens},
            effort=effort,
        )
        return resp.choices[0].message.content or ""

    return vlm


def _img_part(path: str) -> dict:
    from lib.utils.common import get_image_base64

    return {"type": "image_url", "image_url": {"url": get_image_base64(path)}}


_PROPOSE_SYSTEM = (
    "You analyze an image for 3D scene reconstruction. There is ONE scene to reconstruct: "
    "the MAIN SUPPORT — the primary surface the in-focus objects rest on (usually the "
    "table / desk / counter / cutting-board nearest the camera) — and ONLY what belongs to "
    "it. A CONTAINER holding the in-focus objects (a bin, tray, basket, box, bowl, plate) "
    "is NEVER the main support: it is an OBJECT resting ON the main support, and the "
    "objects inside it rest on the container. The main support is the table/counter/floor "
    "BENEATH the container — list that surface even when the container hides most of it "
    "(only corners or edges visible).\n"
    "FOCUS — list ONLY:\n"
    "  (a) OBJECTS resting ON the main support, or stacked on those — including "
    "furniture/fixtures STANDING ON the main support (a shelf unit, rack, stand), "
    "even when empty.\n"
    "  (b) the ROOT SURFACES that form the main support's setting: the main support "
    "itself; EVERY wall or panel that bounds it — check ALL sides (the back wall, a LEFT "
    "wall, a RIGHT wall, and any front panel / half-wall / divider in front of or beside "
    "the support), listing each as its own instance; and the floor/ground ONLY if the main "
    "support's legs/base reach down to it.\n"
    "IGNORE everything else as if absent from the scene: objects lying on the floor or "
    "held by / sitting on a person; a SECOND table/desk standing elsewhere (NOT on the "
    "main support) and anything on it; people; "
    "and distant or background surfaces that are not part of the main support's immediate "
    "setting.\n"
    "List every kept object grouped by category, describing EACH instance separately. "
    "Each category must appear exactly ONCE — merge duplicates before responding.\n"
    "  - \"category\": a short, common, SINGULAR noun for segmentation (e.g. 'book', "
    "'mug', 'pen') — no brand names, model numbers, colors, or positions.\n"
    '  - "instances": a JSON list with ONE entry per visible instance of that category. '
    "Each entry is an object with:\n"
    '      * "description": a phrase precise enough for a pointing model to locate THAT '
    "exact instance. Write it as ONE flowing noun phrase that starts DIRECTLY with the "
    "noun — never with 'the'/'a'/'an' — with NO commas or appended clauses, and use a "
    "PARTICIPLE for spatial relations: 'metal spoon resting inside the steel pot', "
    "'gray book on top of the black book near the right edge'. NEVER 'the metal spoon "
    "inside the steel pot' (leading article) and NEVER 'gray book on top of the stack, "
    "right side' (comma clause) — the pointing model returns NOTHING for both forms. "
    "Disambiguate by plain VISUAL cues it can "
    "SEE (position, relative size, color, spatial relationship), plus the object's "
    "commonly-recognized identity when helpful (e.g. 'iPhone', 'MacBook'). But do NOT lean "
    "on UNUSUAL, hard-to-read identifiers — obscure brand names or printed/quoted text on "
    "the object (titles, labels, logos): a pointing model does not reliably read fine "
    "print, so describe the object by its visible appearance instead.\n"
    '      * "kind": "root_surface" for the main support, a wall, or the floor/ground; '
    '"object" for any movable, separately-nameable item — including a '
    "container (bin / tray / basket / bowl / box) that holds other objects: its support "
    "is the surface beneath it, and its contents' support is the container's id. EACH "
    "wall is a SEPARATE instance — two walls that meet at a corner or are perpendicular "
    "(a back wall and a side wall) are TWO distinct 'wall' instances, NEVER merged.\n"
    '      * "support": the IDENTIFIER of the single instance THIS one directly rests on '
    "/ attaches to. Before assigning, LOOK at what the object DIRECTLY rests on: if it "
    "sits on a smaller platform / stand / riser / tray / dock rather than directly on "
    "the main support, LIST that platform as its own object instance and use ITS id — "
    'never skip over it to the surface below. Every instance is "category#k" (k = 0-based position in that '
    "category's list: the first book is book#0, the main support is table#0). A mug on the "
    "desk -> 'table#0'; a top book on a bottom book -> 'book#1'; a wall clock -> 'wall#0'. "
    "Use null ONLY for the root surfaces themselves. The id MUST point at an instance in "
    'YOUR OWN output: k is the 0-based position within that category\'s "instances" list, '
    "so it must be smaller than the number of instances you list for that category — NEVER "
    "a global counter or a guessed index. Before responding, verify every support id "
    "resolves to a listed instance. Order each category's instances "
    "consistently so ids are stable. If an object actually rests on a surface you are NOT "
    "listing (e.g. a second desk you're excluding), DROP that object too — never reassign it "
    "to the main support.\n"
    '      * "synonyms": for an OBJECT, a JSON list of 1-3 ALTERNATIVE common SINGULAR '
    'nouns a generic detector might use (corn -> ["corn cob", "maize"]; specs -> '
    '["glasses"]). Use [] for root surfaces.\n'
    "GRANULARITY — segment whole, separately-nameable objects, NOT their parts: a keychain "
    "of several keys/fobs is ONE 'keychain' (do NOT split it); charms on a bracelet, "
    "buttons on a remote stay part of their whole. A container and contents INSERTED "
    "into it that form ONE standing arrangement — flowers in a vase, a plant in its "
    "pot, a candle in its holder — end up as ONE object, but DO list the container "
    "AND the inserted contents as SEPARATE instances (each needs its own mask; a "
    "later pass merges the arrangement into the container). Give separate instances only to items "
    "picked up individually (two stacked books). When unsure, prefer FEWER, larger "
    "objects. Group identical items under ONE category with multiple instances.\n"
    'Also give "relationships": a JSON list of {"type", "a", "b"} between '
    'ROOT-SURFACE ids ("a"/"b" are "category#k"), describing how the kept surfaces '
    "connect:\n" + relationship_glossary() + "\n"
    'Respond with ONLY a JSON object: {"objects": [{"category": str, "synonyms": '
    '[str], "instances": [{"description": str, "kind": str, '
    '"support": str|null}, ...]}, ...], "relationships": '
    '[{"type": str, "a": str, "b": str}, ...]}.'
)


def _shared_block(start: str, end: str) -> str:
    """Slice a field-definition block verbatim out of ``_PROPOSE_SYSTEM`` so the room
    proposer inherits the battle-tested rules (pointing-model description format,
    synonyms, granularity) from ONE source — the two prompts cannot drift."""
    i, j = _PROPOSE_SYSTEM.index(start), _PROPOSE_SYSTEM.index(end)
    assert i < j, f"shared block anchors out of order: {start[:40]!r}"
    return _PROPOSE_SYSTEM[i:j]


# Room-track proposer: ONE coherent worldview (no closeup branch) — the floor is the
# anchor root surface, walls are the only other root surfaces, and
# the WORK SURFACE (table/desk) is an ordinary OBJECT standing on the floor.
_PROPOSE_SYSTEM_ROOM = (
    "You analyze an image of an indoor ROOM for 3D scene reconstruction. The scene's "
    "GROUND is the FLOOR. The scene to reconstruct is the WORK SURFACE (the table / "
    "desk the in-focus objects rest on) standing on that floor, with its arrangement.\n"
    'ROOT SURFACES — list ONLY these as kind "root_surface":\n'
    "  - the FLOOR: the ANCHOR of the scene.\n"
    "  - EVERY wall or panel that bounds the room — check ALL sides (back, LEFT, "
    "RIGHT, any front panel / half-wall / divider), each as its own SEPARATE "
    "instance; two walls meeting at a corner are TWO instances, NEVER merged.\n"
    "NOTHING else is a root surface. The WORK SURFACE is NOT a root surface.\n"
    'OBJECTS — list as kind "object":\n'
    "  (a) the WORK SURFACE itself: a piece of furniture resting on the floor "
    "(support = the floor's id). Objects ON it use ITS id as support (a vase on the "
    "table -> support 'table#0'). A CONTAINER holding objects (bin, tray, basket, "
    "box, bowl, plate) is an OBJECT on whatever it rests on, and its contents rest "
    "on the container.\n"
    "  (b) objects resting ON the work surface, or stacked on those.\n"
    "  (c) FURNITURE standing on the FLOOR that is PART OF the work surface's OWN "
    "arrangement — a chair / stool / bench pulled up to or tucked under the work "
    "surface, facing it, within about one chair-length — support = the floor's id. "
    "Furniture farther away, in the background, or serving a different area of the "
    "room is IGNORED — when in doubt, leave it out.\n"
    "IGNORE everything else as if absent: loose small items on the floor (bags, "
    "boxes, shoes, cables, debris); objects held by / sitting on a person; a SECOND "
    "table / desk / shelf and anything on it; people; distant or background surfaces.\n"
    "List every kept object grouped by category, describing EACH instance separately. "
    "Each category must appear exactly ONCE — merge duplicates before responding.\n"
    "  - \"category\": a short, common, SINGULAR noun for segmentation (e.g. 'table', "
    "'chair', 'vase') — no brand names, model numbers, colors, or positions.\n"
    '  - "instances": a JSON list with ONE entry per visible instance. Each entry:\n'
    + _shared_block('      * "description"', '      * "kind"')
    + '      * "kind": "root_surface" ONLY for the floor or a wall; '
    '"object" for everything else INCLUDING the work surface and (c) furniture.\n'
    '      * "support": the IDENTIFIER of the single instance THIS one directly '
    'rests on / attaches to. Every instance is "category#k" (k = 0-based position '
    "in that category's OWN instances list — never a global counter, never an index "
    "beyond the list; verify every support id resolves to a listed instance before "
    "responding). The work surface and floor furniture -> the floor's id; an object "
    "on the work surface -> the work surface's id; stacked -> the object below. Use "
    "null ONLY for the root surfaces themselves. If an object rests on something you "
    "are NOT listing, DROP that object too — never reassign it.\n"
    + _shared_block('      * "synonyms"', "GRANULARITY")
    + _shared_block("GRANULARITY", 'Also give "relationships"')
    + 'Also give "relationships": a JSON list of {"type", "a", "b"} between '
    "ROOT-SURFACE ids ONLY — the floor and walls. The work surface and all other "
    "objects are NEVER endpoints (a wall standing ON the floor is 'under': floor "
    "under wall). Use only these types:\n" + relationship_glossary() + "\n"
    'Respond with ONLY a JSON object: {"objects": [{"category": str, "synonyms": '
    '[str], "instances": [{"description": str, "kind": str, '
    '"support": str|null}, ...]}, ...], "relationships": '
    '[{"type": str, "a": str, "b": str}, ...]}.'
)


# Top-level scene router: one cheap image-only call BEFORE the proposer decides which
# track (proposer + verifier pair) handles the scene. The OCCLUSION criterion lives
# here — a crowded work surface routes closeup, where the primitive main-support build
# is the right call. closeup is the stated (and coded) safe default: router failure
# collapses to today's pipeline, never away from it.
_ROUTER_SYSTEM = (
    "You route a single photo to a 3D scene-reconstruction pipeline AND classify its "
    "main support in one shot. Reply with EXACTLY one token: room, closeup:table, or "
    "closeup:tabletop.\n"
    "- room: an INDOOR room-scale view where the FLOOR is clearly visible as the "
    "ground of the scene, a WORK SURFACE (table / desk) stands on it as a distinct, "
    "MOSTLY VISIBLE piece of furniture (its top NOT crowded with objects), typically "
    "with companion furniture (chairs / stools) arranged around it. The room pipeline "
    "reconstructs the floor as the ground and every piece of furniture — including "
    "the table — as a separate 3D object.\n"
    "- closeup: a non-room view with an identifiable work surface and arrangement — "
    "a close view of a work surface and the objects on "
    "it; a workbench / lab / enclosure shot focused on the tabletop; a scene whose "
    "work surface is mostly COVERED by objects; or another view without a clear indoor "
    "floor-and-furniture arrangement but with a clear support and objects. The closeup "
    "pipeline builds the work surface as the flat stage of the scene. Split it by what "
    "holds the work surface UP:\n"
    "  * closeup:table — the top AND a structure BENEATH it are visible (legs / pedestal "
    "/ cabinet / cart / drawers), OR distinct floor/ground is visible BELOW the work "
    "surface. The structure COUNTS even when it looks like a SEPARATE unit the top merely "
    "sits on. If ground is visible below a table/desk/counter, choose closeup:table even "
    "when its legs or base are partly or fully occluded: the tabletop may be thin, but "
    "some connected structure must hold it above the ground.\n"
    "  * closeup:tabletop — truly NOTHING beneath the top is visible AND no floor or "
    "ground below it appears in the scene: the support reads as a bare PLANE, such as a "
    "countertop run or a view cropped tightly to the top.\n"
    "When unsure between the two closeup forms, reply closeup:table — building a base "
    "that is hidden costs less than leaving an elevated surface floating. When unsure "
    "whether the scene is a room at all, reply closeup:table (the safe default)."
)


def _norm_scene_kind(ans: str) -> str:
    ""
    s = (ans or "").strip().lower()
    if "room" in s and "closeup" not in s:
        return "room"
    if "tabletop" in s:
        return "closeup:tabletop"
    return "closeup:table"


_ROUTER_MAX_TOKENS = 400


def route_scene(vlm: Callable, image_path: str) -> str:
    """Router VLM call: classify the scene as
    'room'|'closeup:table'|'closeup:tabletop'. Best-effort — any failure returns a
    closeup default (today's pipeline), but an EMPTY reply is retried and reported
    rather than silently taken as a classification."""
    for attempt in (1, 2):
        try:
            # _img_part stays INSIDE the try: an unreadable image must fall back like any
            # other failure, never propagate (routing must never kill a run).
            ans = vlm(
                _ROUTER_SYSTEM,
                [{"type": "text", "text": "Route this scene."}, _img_part(image_path)],
                max_tokens=_ROUTER_MAX_TOKENS,
            )
        except Exception as e:  # noqa: BLE001 - routing must never kill a run
            print(f"[route_scene] router failed ({e}) -> closeup:table")
            return "closeup:table"
        if str(ans or "").strip():
            return _norm_scene_kind(ans)
        # An empty reply is a FAILURE, not a verdict. Never let it fall through to the
        # safe default without saying so — that silence is the whole bug.
        print(f"[route_scene] EMPTY router reply (attempt {attempt}/2)")
    print(
        "[route_scene] router returned nothing twice -> closeup:table "
        "(a DEFAULT, not a verdict — re-check if this scene looks like a room)"
    )
    return "closeup:table"


def _norm_instances(category: str, raw: list) -> list[dict[str, Any]]:
    ""
    cat_kind = "root_surface" if is_surface(category) else "object"
    out: list[dict[str, Any]] = []
    for it in raw:
        if isinstance(it, str):
            desc, kind, support = it.strip(), cat_kind, None
        elif isinstance(it, dict):
            desc = str(it.get("description", "")).strip()
            kind = str(it.get("kind", "")).strip().lower()
            kind = kind if kind in ("root_surface", "object") else cat_kind
            support = it.get("support")
            support = str(support).strip().lower() if support else None
        else:
            continue
        if desc:
            out.append({"description": desc, "kind": kind, "support": support})
    return out or [{"description": category, "kind": cat_kind, "support": None}]


def _norm_synonyms(raw: Any) -> list[str]:
    """1-3 alternative detector nouns (object categories only); tolerant of junk."""
    if not isinstance(raw, list):
        return []
    out = [str(s).strip() for s in raw if str(s).strip()]
    return out[:3]


def _norm_cat(name: str) -> str:
    """Canonical category key used for duplicate and overlap comparisons.

    Folds ``[_-]`` separators like same_size._normal_text does, so any category pair
    that would collide in the strict same-size inventory ("coffee-cup" vs "coffee cup")
    is merged here first instead of crashing preprocessing at the final step."""
    s = re.sub(r"[_-]+", " ", str(name).strip().lower())
    s = " ".join(s.split())
    return s[:-1] if len(s) > 3 and s.endswith("s") else s


def _merge_duplicate_categories(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ""
    by_cat: dict[str, dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for o in objects:
        o["same_size"] = False
        key = _norm_cat(o.get("category", ""))
        first = by_cat.get(key)
        if first is None:
            by_cat[key] = o
            out.append(o)
            continue
        first["instances"].extend(o.get("instances", []))
        first["synonyms"] = _norm_synonyms(
            list(dict.fromkeys(first.get("synonyms", []) + o.get("synonyms", [])))
        )
        print(f"[propose_objects] merged duplicate category entry: {o.get('category')}")
    return out


def propose_objects(
    vlm: Callable,
    image_path: str,
    ignore_objects: Optional[list[str]] = None,
    same_size_categories: Optional[list[str]] = None,
    mode: str = "closeup",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """VLM enumerates the scene -> ``(objects, relationships)``.

    ``mode`` picks the track: ``'closeup'`` (default — the MAIN-SUPPORT worldview,
    today's behavior) or ``'room'`` (``_PROPOSE_SYSTEM_ROOM``: the floor is the anchor,
    and the work surface is an ordinary OBJECT on it).
    ``objects``: per category a SAM3 ``category``, fallback ``synonyms`` (1-3 alt nouns for
    full-image/tier-1, tier-2, and recount retries), and per-instance ``instances``
    (pointing prompts for Molmo).
    ``relationships``: ``[{type, a, b}]`` between
    root-surface ids. Single shot — over-segmentation is fixed afterwards by ``merge_objects``,
    and off-main-support objects/surfaces by the prune in ``preprocess``.

    ``same_size_categories`` remains in the public signature for call-site compatibility,
    but same-size resolution happens only after the retained scene graph is stable.  The
    proposer never sees or applies it.
    """
    system = _PROPOSE_SYSTEM_ROOM if mode == "room" else _PROPOSE_SYSTEM
    ignore_note = ""
    if ignore_objects:
        ignore_note = (
            "\n\nALSO ignore these specific objects entirely, even if on the main support: "
            + ", ".join(str(o) for o in ignore_objects)
            + "."
        )
    parts = [
        {
            "type": "text",
            "text": "List the main-support scene's objects, surfaces, and "
            "their relationships." + ignore_note,
        },
        _img_part(image_path),
    ]
    err, err_kind = "", "parse"
    parsed_ok = False
    objects: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []
    for attempt in range(3):
        if not err:
            retry_parts = parts
        elif err_kind == "parse":
            retry_parts = parts + [
                {
                    "type": "text",
                    "text": (
                        f"Your previous reply was not valid JSON ({err}). "
                        "Return ONLY the JSON object — no prose, no code fences."
                    ),
                }
            ]
        else:
            retry_parts = parts + [{"type": "text", "text": err}]
        try:
            out = parse_json(vlm(system, retry_parts, max_tokens=16000))
        except (ValueError, json.JSONDecodeError) as e:
            err, err_kind = str(e)[:300], "parse"
            print(
                f"[propose_objects] unparseable VLM JSON (attempt {attempt + 1}/3): {err}"
            )
            continue
        parsed_ok = True
        # New schema: {"objects":[...], "relationships":[...]}; tolerate a bare list (objects only).
        raw_objs = out.get("objects") if isinstance(out, dict) else out
        raw_rels = out.get("relationships") if isinstance(out, dict) else None
        objects = []
        for o in raw_objs or []:
            if not isinstance(o, dict):
                continue
            cat = str(o.get("category", "")).strip()
            if not cat:
                continue
            obj = {
                "category": cat,
                "synonyms": _norm_synonyms(o.get("synonyms")),
                # Kept for serialized compatibility only.  A proposer reply must never
                # activate a downstream shared-size constraint.
                "same_size": False,
                "instances": _norm_instances(cat, o.get("instances") or []),
            }
            objects.append(obj)
        objects = _merge_duplicate_categories(objects)
        relationships = parse_relationships(raw_rels)
        # Preserve who asserted each claim.  Geometry preprocessing later decides
        # hard/advisory/rejected authority; the VLM is only the proposal source.
        for rel in relationships:
            rel["provenance"] = [{"source": "vlm_proposer", "action": "proposed"}]
            # No VLM claim is authoritative before metric preprocessing.
            rel["status"] = "unverified"
            rel["enforcement"] = "advisory"
        if not any(i.get("kind") == "object" for o in objects for i in o["instances"]):
            err_kind = "empty"
            err = (
                "Your previous reply listed ZERO movable objects. Re-check the image "
                "carefully for separately-nameable objects resting on the main support. "
                "If objects are present, list every one plus the root surfaces. If the "
                "scene genuinely contains no movable objects, keep the object list empty "
                "rather than inventing any. Return ONLY the JSON object."
            )
            print(
                f"[propose_objects] parsed reply lists zero object instances "
                f"(attempt {attempt + 1}/3); raw: {json.dumps(out)[:400]}"
            )
            continue
        break
    else:
        if not parsed_ok:
            raise ValueError(
                f"propose_objects: VLM returned unparseable JSON 3x — last: {err}"
            )
        # Parsed but still empty after 3 asks: return the empty proposal and let the
        # runner's established ZERO-instance guard decide the run's fate.
        print(
            "[propose_objects] zero object instances after 3 attempts — returning empty proposal"
        )
    return objects, relationships


# --------------------------------------------------------------------------- #
# Segmentation verifier (one pass, AFTER segment+merge): a second VLM audits the #
# ACTUAL segmentation (shown a surfaces-only overlay) — adds missing bounding      #
# surfaces (name-only, built visually) and fixes the relationships. It             #
# does NOT drop objects: off-support removal is left to the geometric             #
# mask-connectivity check (``drop_disconnected_from_support``), so a good object  #
# is never lost to a VLM misjudgment.                                             #
# --------------------------------------------------------------------------- #
_VERIFY_SYSTEM = (
    "You AUDIT the segmentation of a single-image 3D scene. There is ONE main support "
    "(a table/desk/counter) + its setting: objects ON it (or on its floor, when kept), "
    "and the walls/floor that BOUND it. "
    "Background and secondary surfaces (a second desk that is NOT the main support, a back table, "
    "the floor under a far-away item, a person) were INTENTIONALLY EXCLUDED. "
    "You audit ROOT SURFACES ONLY. You are shown the id list of the surfaces that ALREADY have "
    "a mask, and an OVERLAY in which each of them is tinted the colour the list names for it. "
    "Objects are NOT tinted and NOT listed — they are not yours to add, remove, move, or "
    'mention. Every id is a STRING "category#k" (e.g. "table#0", "wall#0"); ALWAYS refer to '
    "surfaces by that exact string id — NEVER by a colour or a bare integer. Return ONLY "
    "the requested JSON object:\n"
    '  "add_surfaces": CRITICAL — a root surface the segmenter FAILED to mask is completely '
    "ABSENT from the list and carries NO tint in the overlay. Compare the photo against the "
    "overlay and ADD every root surface that bounds or supports the main support but is missing: "
    "the back wall, the LEFT wall, the RIGHT wall, any front panel / half-wall / divider, and the "
    "floor/ground the main support stands on. Walls and floors are OFTEN unmasked — actively "
    "re-detect them; two walls meeting at a room corner are two SEPARATE instances. "
    "The id list is AUTHORITATIVE about what already exists: if a region is tinted, that surface "
    "is ALREADY masked — do NOT add a second copy of it under a new name, however large the "
    "tinted area is. A geometry hint is a candidate to reconcile against the list, never a "
    "licence to add a duplicate. "
    'Each entry is {"category": "wall"|"floor"|..., "description": "..."}. Add ONLY a surface '
    "that genuinely bounds or supports the main support; NEVER re-add an intentionally-excluded "
    "background surface (a second desk, a back table).\n"
    '  "relationships": ALWAYS return the COMPLETE corrected list of relationships between ROOT SURFACES ONLY '
    "(the main support, walls, floor — INCLUDING any surface you added above). "
    'Each is {"type","a","b"} where a and b are exact "category#k" ids '
    'from the list above or from your own additions (e.g. "table#0", "wall#0"), never colours. '
    "Use [] only when there are genuinely no valid root-surface relationships. FIX wrong types and ADD missing "
    'ones — e.g. a desk/table standing ON the floor is "under" (floor under desk), NOT '
    '"perpendicular". Use only these types:\n' + relationship_glossary() + "\n"
    'Respond with ONLY: {"add_surfaces": [...], "relationships": [...]}.'
)


_VERIFY_SYSTEM_ROOM = (
    "You AUDIT the segmentation of a single-image 3D ROOM scene. The scene is a "
    "ROOM: ONE floor (the ground) + the walls that BOUND it. The WORK SURFACE "
    "(table / desk), companion furniture (chairs), and all smaller items are "
    "OBJECTS standing on the floor or resting on each other — an object is NEVER a "
    "surface. Background and "
    "secondary surfaces (a second desk, a far-away area, a person) were "
    "INTENTIONALLY EXCLUDED. You audit ROOT SURFACES ONLY. You are shown the id list "
    "of the surfaces that ALREADY have a mask, and an OVERLAY in which each of them is "
    "tinted the colour the list names for it. Objects are NOT tinted and NOT listed — "
    "they are not yours to add, remove, move, re-classify, or mention. Every id is a "
    'STRING "category#k" (e.g. "floor#0", "wall#0"); ALWAYS refer to surfaces by that '
    "exact string id — NEVER by a colour or a bare integer. Return ONLY the requested "
    "JSON object:\n"
    '  "add_surfaces": CRITICAL — a root surface the segmenter FAILED to mask is '
    "completely ABSENT from the list and carries NO tint in the overlay. Compare the "
    "photo against the overlay and ADD every "
    "bounding surface missing from the list: the back wall, the LEFT wall, the "
    "RIGHT wall, any front panel / half-wall / divider, and the floor if missing. "
    "Walls are OFTEN unmasked — actively re-detect them; two walls meeting at a "
    "room corner are two SEPARATE instances. "
    "The id list is AUTHORITATIVE about what already exists: if a region is tinted, "
    "that surface is ALREADY masked — do NOT add a second copy of it under a new name, "
    "however large the tinted area is. A geometry hint is a candidate to reconcile "
    "against the list, never a licence to add a duplicate. "
    'Each entry is {"category": "wall"|"floor"|..., "description": "..."}. NEVER '
    "add the work surface or any furniture as a surface — they are objects.\n"
    '  "relationships": ALWAYS return the COMPLETE corrected list of relationships between ROOT '
    "SURFACES ONLY (the floor and walls, INCLUDING any surface you added above). "
    'Each is {"type","a","b"} with exact "category#k" ids from the list above or from '
    "your own additions, never colours. "
    "Use [] only when there are genuinely no valid root-surface relationships. "
    "FIX wrong types and ADD missing ones — a wall standing ON the floor is 'under' "
    "(floor under wall). Use only these types:\n" + relationship_glossary() + "\n"
    'Respond with ONLY: {"add_surfaces": [...], "relationships": [...]}.'
)


def _results_summary(
    results: list["InstanceMask"],
    relationships: list[dict[str, str]],
    colors: dict[str, str],
    main_support: Optional[str] = None,
) -> str:
    ""
    surf = []
    for r in results:
        if r.kind != "root_surface":
            continue
        tag = f"{r.category}#{r.instance}"
        c = colors.get(tag)
        surf.append(
            f"  {tag} ({r.description or r.category})"
            + (f" [tinted {c}]" if c else " [NOT tinted — mask missing]")
            + (" [MAIN SUPPORT]" if tag.lower() == (main_support or "") else "")
        )
    rels = [f"  {x['type']}: {x['a']} <-> {x['b']}" for x in relationships] or [
        "  (none)"
    ]
    return (
        "ROOT SURFACES ALREADY MASKED (id, description, overlay tint):\n"
        + ("\n".join(surf) or "  (none)")
        + "\nRELATIONSHIPS:\n"
        + "\n".join(rels)
    )


def _next_surface_idx(
    results: list["InstanceMask"], unmasked: list[dict[str, Any]], cat: Optional[str]
) -> int:
    """Next free instance index for category ``cat`` across kept results + unmasked roots, so a
    verifier-added surface never collides with an existing id (e.g. a second 'wall')."""
    used = [r.instance for r in results if r.category == cat]
    used += [u.get("instance", -1) for u in unmasked if u.get("category") == cat]
    return (max(used) + 1) if used else 0


def verify_segmentation(
    vlm: Callable,
    composite_path: str,
    objects: list[dict[str, Any]],
    results: list["InstanceMask"],
    relationships: list[dict[str, str]],
    dropped: list[dict[str, Any]],
    unmasked_roots: list[dict[str, Any]],
    hints: Optional[str] = None,
    image_path: Optional[str] = None,
    mode: str = "closeup",
    colors: Optional[dict[str, str]] = None,
) -> tuple[
    list["InstanceMask"],
    list[dict[str, str]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    ""
    review: dict[str, Any] = {
        "ran": True,
        "add_surfaces": [],
        "changed_rels": False,
        "rels_before": relationships,
        "rels_after": relationships,
        "hints": hints,
    }
    try:
        from lib.tools.geometry.preprocess import main_support_id

        main_id = main_support_id(
            {
                "instances": [
                    {
                        "category": r.category,
                        "instance": r.instance,
                        "kind": r.kind,
                        "support": r.support,
                    }
                    for r in results
                ]
            }
        )
        prompt = (
            "Audit the ROOT SURFACES of this segmented scene:"
            "\n\n" + _results_summary(results, relationships, colors or {}, main_id)
        )
        if hints:
            prompt += (
                "\n\nGEOMETRY HINTS (from depth/normals — confirm against the image):\n"
                + hints
            )
        # The CLEAN photo comes first: added-surface descriptions must describe the
        # photo's actual appearance. With only the tinted overlay to look at, the
        # verifier described abc1's (white) back wall as "the back green wall panel"
        # — the green was the neighbouring mask tint — and the initializer then
        # hunted for a green wall that does not exist.
        parts = [{"type": "text", "text": prompt}]
        if image_path and os.path.exists(image_path):
            parts += [
                {
                    "type": "text",
                    "text": "The ORIGINAL photo (ground truth — describe "
                    "added surfaces from THIS image; mask tints are NOT object colors):",
                },
                _img_part(image_path),
            ]
        parts += [
            {
                "type": "text",
                "text": "The SAME photo with every ALREADY-MASKED root "
                "surface tinted in the colour named above. Untinted pixels are not covered "
                "by an existing root-surface mask: they may be objects, background, or a "
                "MISSING root surface. Objects are never tinted, so do not classify them "
                "as surfaces:",
            },
            _img_part(composite_path),
        ]
        out = parse_json(
            vlm(
                _VERIFY_SYSTEM_ROOM if mode == "room" else _VERIFY_SYSTEM,
                parts,
                max_tokens=2000,
            )
        )
        if not isinstance(out, dict):
            return results, relationships, dropped, unmasked_roots, review
        # NOTE: the verifier no longer drops objects — off-support removal is the
        # geometric mask-connectivity check's job (drop_disconnected_from_support),
        # so a valid object is never lost to a VLM misjudgment. Any stray "drop"
        # field the model emits is ignored.
        # (1) record newly-added bounding surfaces as NAME-ONLY roots (renumbered ids, no
        # re-segmentation) -> the initializer builds them visually from the description.
        add = [s for s in (out.get("add_surfaces") or []) if isinstance(s, dict)]
        for s in add:
            cat = str(s.get("category", "")).strip().lower() or "wall"
            desc = str(s.get("description", "")).strip() or cat
            unmasked_roots.append(
                {
                    "category": cat,
                    "instance": _next_surface_idx(results, unmasked_roots, cat),
                    "description": desc,
                }
            )
        _missing_rels = object()
        raw_reply_rels = out.get("relationships", _missing_rels)
        new_rels = (
            list(relationships)
            if raw_reply_rels is _missing_rels
            else parse_relationships(raw_reply_rels)
        )
        if raw_reply_rels is not _missing_rels:
            # The audit VLM returns a complete list.  Attach provenance without
            # trusting any metadata it might have emitted: exact retained claims are
            # confirmations; same-endpoint type changes are corrections; wholly new
            # endpoint pairs are auditor proposals.
            def _pair_key(rel):
                return tuple(sorted((rel.get("a", ""), rel.get("b", ""))))

            old_exact = {(r.get("type"), *_pair_key(r)): r for r in relationships}
            old_pair = {_pair_key(r): r for r in relationships}
            for rel in new_rels:
                exact = old_exact.get((rel.get("type"), *_pair_key(rel)))
                prior = exact or old_pair.get(_pair_key(rel))
                provenance = list((prior or {}).get("provenance") or [])
                if not provenance and prior:
                    provenance.append({"source": "vlm_proposer", "action": "proposed"})
                if exact:
                    audit = {"source": "vlm_auditor", "action": "confirmed"}
                elif prior:
                    audit = {
                        "source": "vlm_auditor",
                        "action": "corrected_type",
                        "from": prior.get("type"),
                        "to": rel.get("type"),
                    }
                else:
                    audit = {"source": "vlm_auditor", "action": "proposed"}
                if audit not in provenance:
                    provenance.append(audit)
                rel["provenance"] = provenance
                rel["status"] = "unverified"
                rel["enforcement"] = "advisory"
        valid_ids = {
            f"{r.category}#{r.instance}".lower()
            for r in results
            if r.kind == "root_surface"
        } | {f"{u['category']}#{u['instance']}".lower() for u in unmasked_roots}

        def _repair(i: str) -> Optional[str]:
            if i in valid_ids:
                return i
            if "#" not in i:
                hits = [v for v in valid_ids if v.rsplit("#", 1)[0] == i]
                if len(hits) == 1:
                    return hits[0]
            return None

        repaired_rels, dropped_rels, kept_rels = [], [], []
        for r in new_rels:
            ra, rb = _repair(r["a"]), _repair(r["b"])
            if ra is None or rb is None:
                dropped_rels.append(r)
                bad = [x for x, fx in ((r["a"], ra), (r["b"], rb)) if fx is None]
                print(
                    f"[masking] dropped relationship {r['type']} {r['a']} <-> {r['b']}: "
                    f"{', '.join(bad)} not a known root surface "
                    f"(known: {', '.join(sorted(valid_ids))})"
                )
                continue
            if (ra, rb) != (r["a"], r["b"]):
                repaired_rels.append(dict(r))
                print(
                    f"[masking] repaired relationship ids: {r['type']} {r['a']} <-> "
                    f"{r['b']} -> {ra} <-> {rb}"
                )
                r = {**r, "a": ra, "b": rb}
            if r not in kept_rels:  # a repair may collide with an existing rel
                kept_rels.append(r)
        new_rels = kept_rels
        if (
            raw_reply_rels is not _missing_rels
            and bool(raw_reply_rels)
            and not new_rels
            and relationships
        ):
            print(
                f"[masking] verifier relationships collapsed to empty "
                f"({len(dropped_rels)} invalid-id drops); keeping the proposer's "
                f"{len(relationships)} relationship(s)"
            )
            new_rels = list(relationships)
            dropped_rels, repaired_rels = [], []
        review.update(
            {
                "add_surfaces": add,
                "changed_rels": new_rels != relationships,
                "rels_after": new_rels,
                "dropped_rels": dropped_rels,
                "repaired_rels": repaired_rels,
            }
        )
        return results, new_rels, dropped, unmasked_roots, review
    except Exception as e:  # noqa: BLE001 - audit is advisory; keep the un-audited segmentation
        review.update({"ran": False, "error": str(e)})
        return results, relationships, dropped, unmasked_roots, review


# --------------------------------------------------------------------------- #
# Geometric wall backstop: large flat surfaces ~perpendicular to the main       #
# support are candidate walls/panels the proposer may have missed (-> verifier).#
# --------------------------------------------------------------------------- #
def _ransac_planes(
    pts: np.ndarray, k: int, min_frac: float, iters: int = 200, thresh: float = 0.02
):
    """Up to ``k`` dominant planes by iterative RANSAC (largest first), each kept only if it
    holds >= ``min_frac`` of the points. Returns ``[(unit_normal, inlier_mask)]``."""
    rng = np.random.RandomState(0)
    remaining = np.ones(len(pts), bool)
    out = []
    for _ in range(k):
        idx = np.where(remaining)[0]
        if len(idx) < max(500, int(min_frac * len(pts))):
            break
        sub = pts[idx]
        best = None
        for _ in range(iters):
            s = sub[rng.choice(len(sub), 3, replace=False)]
            n = np.cross(s[1] - s[0], s[2] - s[0])
            ln = float(np.linalg.norm(n))
            if ln < 1e-9:
                continue
            n = n / ln
            inl = np.abs((sub - s[0]) @ n) < thresh
            if best is None or int(inl.sum()) > int(best[0].sum()):
                best = (inl, n)
        if best is None or int(best[0].sum()) < min_frac * len(pts):
            break
        full = np.zeros(len(pts), bool)
        full[idx[best[0]]] = True
        out.append((best[1], full))
        remaining[idx[best[0]]] = False
    return out


def _img_region(xs: np.ndarray, ys: np.ndarray, w: int, h: int) -> str:
    cx, cy = float(np.mean(xs)) / max(w, 1), float(np.mean(ys)) / max(h, 1)
    horiz = "left" if cx < 0.38 else "right" if cx > 0.62 else "center"
    vert = "upper " if cy < 0.33 else "lower " if cy > 0.66 else ""
    return vert + horiz


def _mask_to(b: np.ndarray, h: int, w: int) -> np.ndarray:
    """Binary mask resampled to (h, w) — masks are saved at image resolution, the point
    map may differ."""
    if b.shape == (h, w):
        return b
    from PIL import Image

    return np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((w, h))) > 127


def _support_normal(
    results: list["InstanceMask"], P: np.ndarray, min_frac: float = 0.25
) -> Optional[tuple[np.ndarray, str]]:
    """``(unit normal, id)`` of the MAIN support, fitted through its OWN masked points, or
    None when no masked root surface qualifies.

    The support is the masked root surface that the most objects rest on (``support``
    back-references, the same signal ``drop_disconnected_from_support`` resolves); ties and
    the no-object case fall back to the largest root-surface mask. Deliberately NOT
    "the largest plane in the scene" — see ``wall_hints``."""
    surf = {
        f"{r.category}#{r.instance}": r
        for r in results
        if r.kind == "root_surface" and r.mask_path and os.path.exists(r.mask_path)
    }
    if not surf:
        return None
    votes: dict[str, int] = {}
    for r in results:
        if r.kind == "root_surface":
            continue
        sid = (r.support or "").lower()
        for k in surf:
            if k.lower() == sid:
                votes[k] = votes.get(k, 0) + 1
    h, w = P.shape[:2]
    finite = np.isfinite(P).all(axis=2)

    def _area(k: str) -> int:
        try:
            return int(
                (_mask_to(_binarize(np.load(surf[k].mask_path)), h, w) & finite).sum()
            )
        except Exception:  # noqa: BLE001
            return 0

    best = max(surf, key=lambda k: (votes.get(k, 0), _area(k)))
    m = _mask_to(_binarize(np.load(surf[best].mask_path)), h, w) & finite
    pts = P[m]
    if len(pts) < 500:  # too little visible surface to fit a plane through
        return None
    if len(pts) > 40000:
        pts = pts[np.random.RandomState(0).choice(len(pts), 40000, replace=False)]
    fit = _ransac_planes(pts, 1, min_frac)
    return (fit[0][0], best) if fit else None


def wall_hints(
    points_npy: Optional[str],
    results: Optional[list["InstanceMask"]] = None,
    max_planes: int = 4,
    min_frac: float = 0.04,
) -> Optional[str]:
    ""
    try:
        if not points_npy or not os.path.exists(points_npy):
            return None
        P = np.load(points_npy)
        if P.ndim != 3:
            return None
        h, w = P.shape[:2]
        finite = np.isfinite(P).all(axis=2)
        ys, xs = np.where(finite)
        if len(xs) < 2000:
            return None
        sup = _support_normal(results or [], P)
        if sup is None:  # no masked support -> no trustworthy reference frame
            return None
        n0, sup_id = sup
        pts = P[ys, xs]
        if len(pts) > 40000:  # subsample for speed (deterministic)
            idx = np.random.RandomState(0).choice(len(pts), 40000, replace=False)
            pts, xs, ys = pts[idx], xs[idx], ys[idx]
        hints = []
        # every plane is eligible now, including the largest: the support is identified by
        # its mask, so a dominant wall is no longer excluded for being the reference.
        for n, mask in _ransac_planes(pts, max_planes, min_frac):
            if (
                abs(float(np.dot(n, n0))) > 0.5
            ):  # parallel-ish to the support -> not a wall (and skips the support itself)
                continue
            region = _img_region(xs[mask], ys[mask], w, h)
            hints.append(
                f"- a large flat surface on the {region} of the image, roughly "
                f"perpendicular to {sup_id} (the main support) — likely a wall/panel; "
                "list it if it bounds the scene."
            )
        return "\n".join(hints) if hints else None
    except Exception:  # noqa: BLE001 - the hint is advisory; never break masking
        return None


def normalize_objects(objects: list[Any]) -> list[dict[str, Any]]:
    ""
    out = []
    for o in objects:
        if isinstance(o, str):
            out.append(
                {
                    "category": o,
                    "synonyms": [],
                    "same_size": False,
                    "instances": _norm_instances(o, [o]),
                }
            )
        elif isinstance(o, dict) and str(o.get("category", "")).strip():
            cat = o["category"].strip()
            insts = _norm_instances(cat, o.get("instances") or [])
            cs = str(o["support"]).strip().lower() if o.get("support") else None
            if cs:
                for it in insts:
                    it["support"] = it["support"] or cs
            out.append(
                {
                    "category": cat,
                    "synonyms": _norm_synonyms(o.get("synonyms")),
                    # Same-size constraints are resolved after segmentation; never trust
                    # a raw/pre-given compatibility flag at masking time.
                    "same_size": False,
                    "instances": insts,
                }
            )
    return out


def objects_for_masking(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedup by category (merging per-instance entries). Root surfaces are kept
    (segmented for plane fits) — only the SAM3D mesh step skips them."""
    merged: dict[str, dict[str, Any]] = {}
    for o in objects:
        key = o["category"].lower()
        if key not in merged:
            merged[key] = {
                "category": o["category"],
                "synonyms": o.get("synonyms", []),
                "same_size": False,
                "instances": [],
            }
        merged[key]["instances"].extend(o["instances"])
    return list(merged.values())


# --------------------------------------------------------------------------- #
# Over-segmentation merger (replaces the propose/verify loop)                   #
# --------------------------------------------------------------------------- #
_MERGE_CRITERION = (
    "they are parts of ONE single physical object (the same object split into two masks). "
    "Parts joined into a single object — a cap on its stem, a lid on a jar, a handle or "
    "spout on a body, a head on a figure, a base under a toy — are ONE object EVEN IF they "
    "are different colours or materials. Only answer no if A and B are genuinely independent "
    "items that merely touch or overlap. CAUTION: two SEPARATE instances of the same "
    "category (two stuffed animals, two books, two cups) leaning on, stacked on, or "
    "overlapping each other are NOT one object — merge only when one region is an "
    "incomplete PART (a head, limb, cap, handle) that could not be a standalone item; if "
    "EACH region could be a complete object on its own, answer no. ONE exception: a "
    "standing ARRANGEMENT — contents INSERTED into a container so they stand as one "
    "unit (flowers in a vase, a plant in its pot, a candle in its holder) — merges "
    "into the CONTAINER's name even though each part looks complete; the container's "
    "own dedicated UNDER-DISH joins the arrangement too (a flowerpot standing on its "
    "drip saucer/drainage tray -> one object, the pot's name). Contents merely "
    "PLACED in/on something (fruit in a bowl, tools in a bin, an item on a tray or "
    "table) are NOT an arrangement — nor is a cup or teapot on a saucer (two "
    "objects): answer no"
)

_MERGE_SYSTEM = (
    "You are shown the full image for context, then a ZOOMED close-up of two candidate "
    "regions — first WITHOUT marks (so you can see what they are), then the SAME close-up "
    "with region A outlined/tinted RED and region B outlined/tinted BLUE. Decide whether A "
    "and B should be MERGED into a single object. Merge if " + _MERGE_CRITERION + ".\n"
    "If you merge, choose the single best category name for the combined object — it MUST "
    "be EXACTLY one of the two given names.\n"
    'Before deciding, commit to what each region IS: fill "a_part" and "b_part" '
    "with the specific part or object each region shows (e.g. 'the elephant's head' vs "
    "'a complete second plush toy') — if both read as complete objects, merge must be "
    "false.\n"
    'Respond with ONLY JSON: {"a_part": str, "b_part": str, "merge": bool, "keep": str} '
    '("keep" is the chosen name; use "" when merge is false).'
)

_T3_COMPONENT_RULE = (
    "\nCOMPONENT RULE: ALSO merge when one region is a FIXED COMPONENT of the other "
    "object — a stand, base plate, foot, mount, bracket, knob, or housing that is part "
    "of that object's own structure (a monitor's flat stand base, a lamp's foot, a mic "
    "stand's clamp knob). Such components are sometimes mis-detected as separate small "
    "items (a monitor's base plate read as a 'keyboard' or 'tray' — a REAL keyboard "
    "shows a visible KEY GRID; a slim slab at a device's foot with no visible keys is "
    "that device's base). A component CONTINUES INTO the other object's structure; a "
    "separate item merely sits NEXT TO it with its own complete outline. Objects that "
    "rest on, lean against, plug into, or sit beside the other object remain separate "
    "(a real keyboard in front of a monitor, a mouse or power adapter next to a laptop, "
    "a cup beside anything: NO merge).\n"
    "SUSPENSION RULE: an object that HOVERS above the ground/surface — background "
    "visible beneath it, and its own region contains NO leg, base, or support of its "
    "own reaching down — cannot stay in place unsupported: it MUST be attached to the "
    "structure that holds it (a stand, arm, pole, clamp, hook). If region A hovers and "
    "region B is the supporting structure at its position, they are ONE linked assembly: "
    "merge (even when the exact attachment point is thin, occluded, or missed by the "
    "masks). This rule NEVER applies to an object resting ON a surface or ON another "
    "object — resting objects support themselves and stay separate.\n"
)
_T3_ANCHOR = "\nIf you merge, choose"
_MERGE_SYSTEM_T3 = (
    _MERGE_SYSTEM.replace(_T3_ANCHOR, _T3_COMPONENT_RULE + _T3_ANCHOR)
    if _T3_ANCHOR in _MERGE_SYSTEM
    else _MERGE_SYSTEM + _T3_COMPONENT_RULE
)


def _validate_merge_reply(out: dict, categories: tuple[str, str]) -> dict:
    require_json_contract(out, bool_fields=("merge",))
    keep = out.get("keep")
    if not isinstance(keep, str):
        raise ValueError("keep must be a JSON string")
    if out["merge"]:
        by_lower = {c.lower(): c for c in categories}
        if keep.strip().lower() not in by_lower:
            raise ValueError(
                f"keep must be one of {list(categories)} when merge is true"
            )
        out["keep"] = by_lower[keep.strip().lower()]
    return out


def _down_mask(b: np.ndarray, target: int = 200) -> np.ndarray:
    """Downscale a bool mask so its short side is ~``target`` px — fast adjacency tests
    at a fixed 1/target spatial resolution (so a 1-px dilation == ~1/target of the side)."""
    h, w = b.shape
    if min(h, w) <= target:
        return b
    s = target / min(h, w)
    from PIL import Image

    rs = Image.fromarray(b.astype("uint8") * 255).resize(
        (max(1, round(w * s)), max(1, round(h * s)))
    )
    return np.asarray(rs) > 127


def _dilate1(b: np.ndarray) -> np.ndarray:
    """1-px 8-connected dilation (no wrap-around)."""
    out = b.copy()
    out[:-1] |= b[1:]
    out[1:] |= b[:-1]
    out[:, :-1] |= b[:, 1:]
    out[:, 1:] |= b[:, :-1]
    out[:-1, :-1] |= b[1:, 1:]
    out[1:, 1:] |= b[:-1, :-1]
    out[:-1, 1:] |= b[1:, :-1]
    out[1:, :-1] |= b[:-1, 1:]
    return out


def _union_crop(
    image_path: str, ma: np.ndarray, mb: np.ndarray, out: str, pad_frac: float = 0.35
) -> None:
    """Crop ``image_path`` to the padded bbox of mask A|B — a zoomed close-up of the pair so
    the merge VLM can see the junction (at full-scene scale two small adjacent objects are
    impossible to judge). Works on the original photo or on the red/blue marked overlay
    (same dimensions)."""
    from PIL import Image

    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    def _fit(m: np.ndarray) -> np.ndarray:
        return (
            np.asarray(Image.fromarray((m > 0).astype("uint8") * 255).resize((w, h)))
            > 127
            if m.shape != (h, w)
            else m > 0
        )

    ys, xs = np.where(_fit(ma) | _fit(mb))
    if len(xs) == 0:
        im.save(out)
        return
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    px, py = int((x1 - x0 + 1) * pad_frac), int((y1 - y0 + 1) * pad_frac)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(w, x1 + px + 1), min(h, y1 + py + 1)
    cx0, cy0, cx1, cy1 = clamp_crop_aspect(cx0, cy0, cx1, cy1, w, h)
    im.crop((cx0, cy0, cx1, cy1)).save(out)


def _pair_overlay(
    image_path: str, ma: np.ndarray, mb: np.ndarray, out_path: str
) -> None:
    """Mark mask A (red) and mask B (blue) on the FULL-BRIGHTNESS image with a light fill
    and a bright contour outline, so the merge VLM can still see what the object IS (a
    heavy opaque tint hid e.g. a mushroom's cap+stem and made it look like two blobs)."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    arr = np.array(img).astype(np.float32)

    def _fit(m: np.ndarray) -> np.ndarray:
        if m.shape == arr.shape[:2]:
            return m
        return (
            np.asarray(Image.fromarray(m.astype("uint8") * 255).resize(img.size)) > 127
        )

    def _ring(m: np.ndarray, w: int = 3) -> np.ndarray:
        d = m.copy()
        for _ in range(w):
            d = _dilate1(d)
        return d & ~m

    a, b = _fit(ma), _fit(mb)
    arr[a] = 0.80 * arr[a] + 0.20 * np.array([255.0, 0.0, 0.0])  # A: light red fill
    arr[b] = 0.80 * arr[b] + 0.20 * np.array([0.0, 90.0, 255.0])  # B: light blue fill
    arr[_ring(a)] = np.array([255.0, 0.0, 0.0])  # A: bright red outline
    arr[_ring(b)] = np.array([0.0, 90.0, 255.0])  # B: bright blue outline
    Image.fromarray(arr.clip(0, 255).astype("uint8")).save(out_path)


MERGE_IOU_AUTO = 0.45  # >= : dual-naming duplicate (mushroom<->toy), merge without VLM.
# 0 false fires on the 3,082 clean historical pairs at this bar.
MERGE_IOU_BAND = 0.25  # [BAND, AUTO): ask the VLM — recall safety for duplicates whose
MERGE_VESSELS = {
    s.strip().lower()
    for s in os.environ.get(
        "GRASE_MERGE_VESSELS", "vase,pot,holder,jar,stand,shelf,flowerpot"
    ).split(",")
    if s.strip()
}


def merge_rule(cat_a: str, cat_b: str, iou: float) -> str:
    ""
    a, b = cat_a.lower(), cat_b.lower()
    if iou >= MERGE_IOU_AUTO:
        return "merge"
    if a == b:
        return "skip"
    if (a in MERGE_VESSELS or any(s in a for s in MERGE_VESSELS)) != (
        b in MERGE_VESSELS or any(s in b for s in MERGE_VESSELS)
    ):
        return "vlm"
    if iou >= MERGE_IOU_BAND:
        return "vlm"
    return "skip"


def merge_objects(
    vlm: Callable,
    image_path: str,
    results: list["InstanceMask"],
    masks_dir: str,
    moge_points: Optional[np.ndarray] = None,
    should_merge: Optional[
        Callable[["InstanceMask", "InstanceMask"], Optional[bool]]
    ] = None,
) -> tuple[list["InstanceMask"], list[dict[str, Any]], list[dict[str, Any]]]:
    """Over-segmentation pass: for every pair of OBJECT masks that TOUCH (within ~1/200
    of the image's short side), run the v2 ``merge_rule`` cascade. Only ``vlm`` verdicts
    are judged by the VLM; an accepted merge unions the masks and the VLM names the
    survivor (it keeps its name + id), while the other is removed. Support references
    to the removed object follow the survivor; a survivor supported by the removed
    object inherits its external support instead of becoming self-supported.
    A grown survivor is re-tested against everyone. Root surfaces are never touched.

    ``should_merge`` is an optional hook (manual rules): return True/False to bypass the
    VLM for a pair, or None to defer to it. Mutates surviving recs
    (mask/overlay/point_cam/support) and returns
    ``(results_without_victims, merge_log, merge_decisions)`` (the last is every
    VLM-judged pair, for the web demo)."""
    from collections import deque

    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    def rid(r: "InstanceMask") -> str:
        return f"{r.category}#{r.instance}"

    nodes: dict[str, dict[str, Any]] = {}
    for r in results:
        if (
            r.kind == "root_surface"
            or not r.mask_path
            or not os.path.exists(r.mask_path)
        ):
            continue
        m = _binarize(np.load(r.mask_path))
        nodes[rid(r)] = {"rec": r, "mask": m, "ds": _down_mask(m)}
    merge_log: list[dict[str, Any]] = []
    merge_decisions: list[
        dict[str, Any]
    ] = []  # every VLM-judged pair (for the web demo)
    if len(nodes) < 2:
        return results, merge_log, merge_decisions

    def touch(i: str, j: str) -> bool:
        return bool((_dilate1(nodes[i]["ds"]) & nodes[j]["ds"]).any())

    pending: deque = deque()
    in_q: set = set()
    answered_no: set = set()

    def enqueue(i: str, j: str) -> None:
        p = frozenset((i, j))
        if i != j and p not in in_q:
            pending.append(p)
            in_q.add(p)

    ids = list(nodes)
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            if touch(ids[x], ids[y]):
                enqueue(ids[x], ids[y])

    changed: set = set()
    tmp_overlay = str(masks_dir / "_merge_pair.png")
    while pending:
        p = pending.popleft()
        in_q.discard(p)
        a, b = tuple(p)
        if a not in nodes or b not in nodes or p in answered_no or not touch(a, b):
            continue
        ra, rb = nodes[a]["rec"], nodes[b]["rec"]
        decision = should_merge(ra, rb) if should_merge else None
        keep = ""
        if decision is None and os.environ.get("GRASE_MERGE_RULES", "1") != "0":
            # v2 cascade: only 'vlm' outcomes fall through to the pair prompt below.
            _i = (nodes[a]["ds"] & nodes[b]["ds"]).sum()
            _u = (nodes[a]["ds"] | nodes[b]["ds"]).sum()
            verdict = merge_rule(ra.category, rb.category, _i / _u if _u else 0.0)
            if verdict == "merge" and (
                rid(rb) in (ra.overlap_with or []) or rid(ra) in (rb.overlap_with or [])
            ):
                verdict = "vlm"
            if verdict == "merge":
                decision = True
                # auto-merge survivor: the larger mask keeps its identity
                keep = (
                    rid(ra) if nodes[a]["ds"].sum() >= nodes[b]["ds"].sum() else rid(rb)
                )
            elif verdict == "skip":
                decision = False
        if decision is None:
            n = len(merge_decisions)
            tag = f"{rid(ra)}__{rid(rb)}".replace("#", "")
            clean_crop = str(masks_dir / f"_merge_{n}_{tag}_clean.png")
            zoom_crop = str(masks_dir / f"_merge_{n}_{tag}_marked.png")
            _pair_overlay(image_path, nodes[a]["mask"], nodes[b]["mask"], tmp_overlay)
            _union_crop(image_path, nodes[a]["mask"], nodes[b]["mask"], clean_crop)
            _union_crop(tmp_overlay, nodes[a]["mask"], nodes[b]["mask"], zoom_crop)
            txt = (
                f"Region A (red) = '{ra.category}': {ra.description}. "
                f"Region B (blue) = '{rb.category}': {rb.description}."
            )
            # A skipped merge just leaves two fragments (recoverable downstream) —
            # never worth killing the run over, hence the no-merge default.
            out = vlm_json(
                vlm,
                _MERGE_SYSTEM,
                [
                    {"type": "text", "text": "Full image (context):"},
                    _img_part(image_path),
                    {
                        "type": "text",
                        "text": txt + " Zoomed close-up of the two regions (no marks):",
                    },
                    _img_part(clean_crop),
                    {
                        "type": "text",
                        "text": "Same close-up, Region A red, Region B blue:",
                    },
                    _img_part(zoom_crop),
                ],
                label=f"merge_objects {rid(ra)}/{rid(rb)}",
                default={"merge": False},
                validate=lambda reply, cats=(ra.category, rb.category): (
                    _validate_merge_reply(reply, cats)
                ),
            )
            decision = bool(out.get("merge"))
            keep = str(out.get("keep", "")).strip()
            merge_decisions.append(
                {
                    "a": rid(ra),
                    "b": rid(rb),
                    "desc_a": ra.description,
                    "desc_b": rb.description,
                    "merge": decision,
                    "keep": keep or None,
                    "clean": os.path.basename(clean_crop),
                    "marked": os.path.basename(zoom_crop),
                }
            )
        if not decision:
            answered_no.add(p)
            continue
        # survivor = the VLM-named node (keeps its name+id); fallback = larger mask
        kl = keep.lower()
        if kl == ra.category.lower() and kl != rb.category.lower():
            surv, vic = a, b
        elif kl == rb.category.lower() and kl != ra.category.lower():
            surv, vic = b, a
        else:
            surv, vic = (
                (a, b)
                if int(nodes[a]["mask"].sum()) >= int(nodes[b]["mask"].sum())
                else (b, a)
            )
        nodes[surv]["mask"] = nodes[surv]["mask"] | nodes[vic]["mask"]
        nodes[surv]["ds"] = _down_mask(nodes[surv]["mask"])
        # Keep the support graph attached to live IDs, including across chained merges.
        survivor = nodes[surv]["rec"]
        victim_support = nodes[vic]["rec"].support
        if (victim_support or "").strip().lower() in {surv.lower(), vic.lower()}:
            victim_support = None
        for rec in results:
            if (rec.support or "").strip().lower() == vic.lower():
                rec.support = victim_support if rec is survivor else surv
        merge_log.append(
            {
                "kept": rid(nodes[surv]["rec"]),
                "merged": rid(nodes[vic]["rec"]),
                "keep_name": keep or None,
            }
        )
        changed.add(surv)
        del nodes[vic]
        # the victim is gone and the survivor grew -> re-test the survivor against all
        answered_no = {q for q in answered_no if vic not in q and surv not in q}
        for q in list(in_q):
            if vic in q or surv in q:
                in_q.discard(q)
        pending = deque(q for q in pending if vic not in q and surv not in q)
        for other in nodes:
            if other != surv and touch(surv, other):
                enqueue(surv, other)

    for sid in changed:
        if sid not in nodes:  # this survivor was later merged away
            continue
        r, m = nodes[sid]["rec"], nodes[sid]["mask"]
        np.save(r.mask_path, m.astype(np.uint8) * 255)
        if r.overlay_path:
            make_overlay(image_path, m, r.overlay_path)
        if moge_points is not None:
            r.point_cam, r.depth = object_point_from_mask(m, moge_points)

    alive = set(nodes)
    final = [
        r
        for r in results
        if r.kind == "root_surface"
        or not (r.mask_path and os.path.exists(r.mask_path))
        or rid(r) in alive
    ]
    return final, merge_log, merge_decisions


def tier3_component_merge(
    masks_json_path: str,
    image_path: str,
    model: str = DEFAULT_MODEL,
    vlm: Optional[Callable] = None,
    adj_k: int = 3,
) -> None:
    """Post-resegment merge pass for tier-3 objects.

    For each tier-3 object, every partner within ``adj_k`` dilations at the ~200 px
    downscale (relaxed vs the merge stage's 1 px: the abc3 cradle<->stand assembly
    sits at a 2 px gap) on REDETECT-aware masks is judged by the pair VLM with the
    COMPONENT + SUSPENSION rules appended (``_MERGE_SYSTEM_T3``). Runs in preprocess
    AFTER generative re-segmentation — the 8226 keyboard<->monitor pair only touches
    once the monitor's redetect reclaims its base — and BEFORE the connectivity
    prune, so the union is what gets judged there. On merge the survivor's ACTIVE
    mask file (redetect when present) absorbs the victim at that file's native
    resolution; the victim moves to ``dropped`` (reason ``merged_into <survivor>``,
    full record kept for the demo) and supports pointing at it re-point to the
    survivor. Every judged pair is appended to ``merge_decisions``/``merge_log``
    (same schema as the merge stage, ``stage='tier3_post_resegment'``) so the demo
    renders them unchanged. Best-effort per pair: a flaky VLM reply skips that pair
    only. Kill-switch: ``GRASE_T3_MERGE=0``. ``vlm`` is injectable for tests; when
    None it is built lazily only if a pair actually queues (scenes without tier-3
    objects pay nothing).
    """
    if os.environ.get("GRASE_T3_MERGE", "1") == "0":
        return
    masks_dir = Path(masks_json_path).parent
    data = json.load(open(masks_json_path))

    def _active_path(r: dict) -> Optional[str]:
        rd = r.get("redetect") or {}
        mp = (
            rd.get("mask_path")
            if rd.get("mask_path") and not rd.get("border")
            else r.get("mask_path")
        )
        return mp if mp and os.path.exists(mp) else None

    objs = [
        r
        for r in data.get("instances", [])
        if r.get("kind") != "root_surface" and _active_path(r)
    ]
    t3 = [r for r in objs if r.get("tier") == 3]
    if not t3:
        return

    def rid(r: dict) -> str:
        return f"{r['category']}#{r['instance']}"

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

    shape = None
    loaded: dict[str, tuple[dict, np.ndarray]] = {}
    for r in objs:
        m = _binarize(np.load(_active_path(r)))
        if shape is None:
            shape = m.shape
        loaded[rid(r)] = (r, _fit(m, shape))
    pairs: list[tuple[str, str]] = []
    queued: set = set()
    for r in t3:
        ka = rid(r)
        grown = _down_mask(loaded[ka][1])
        for _ in range(adj_k):
            grown = _dilate1(grown)
        for kb, (_rb, mb) in loaded.items():
            if kb == ka or frozenset((ka, kb)) in queued:
                continue
            if (grown & _down_mask(mb)).any():
                queued.add(frozenset((ka, kb)))
                pairs.append((ka, kb))
    if not pairs:
        return
    vlm = vlm or _make_vlm(model, effort="medium")
    dead: set = set()
    changed = False
    for n, (ka, kb) in enumerate(pairs):
        if ka in dead or kb in dead:
            continue
        ra, ma = loaded[ka]
        rb, mb = loaded[kb]
        tag = f"{ka}__{kb}".replace("#", "").replace(" ", "_")
        clean = str(masks_dir / f"_t3merge_{n}_{tag}_clean.png")
        marked = str(masks_dir / f"_t3merge_{n}_{tag}_marked.png")
        tmp = str(masks_dir / "_t3merge_pair.png")
        _pair_overlay(image_path, ma, mb, tmp)
        _union_crop(image_path, ma, mb, clean)
        _union_crop(tmp, ma, mb, marked)
        txt = (
            f"Region A (red) = '{ra['category']}': {ra.get('description')}. "
            f"Region B (blue) = '{rb['category']}': {rb.get('description')}."
        )
        out = vlm_json(
            vlm,
            _MERGE_SYSTEM_T3,
            [
                {"type": "text", "text": "Full image (context):"},
                _img_part(image_path),
                {
                    "type": "text",
                    "text": txt + " Zoomed close-up of the two regions (no marks):",
                },
                _img_part(clean),
                {
                    "type": "text",
                    "text": "Same close-up, Region A red, Region B blue:",
                },
                _img_part(marked),
            ],
            label=f"t3merge {ka}/{kb}",
            default={"merge": False, "keep": ""},
            validate=lambda reply, cats=(ra["category"], rb["category"]): (
                _validate_merge_reply(reply, cats)
            ),
        )
        decision = bool(out.get("merge"))
        keep = str(out.get("keep", "")).strip()
        data.setdefault("merge_decisions", []).append(
            {
                "a": ka,
                "b": kb,
                "desc_a": ra.get("description"),
                "desc_b": rb.get("description"),
                "merge": decision,
                "keep": keep or None,
                "clean": os.path.basename(clean),
                "marked": os.path.basename(marked),
                "stage": "tier3_post_resegment",
            }
        )
        changed = True
        if not decision:
            continue
        kl = keep.lower()
        if kl == ra["category"].lower() and kl != rb["category"].lower():
            surv, vic = ka, kb
        elif kl == rb["category"].lower() and kl != ra["category"].lower():
            surv, vic = kb, ka
        else:  # unmatched keep -> larger mask survives (merge_objects convention)
            surv, vic = (ka, kb) if int(ma.sum()) >= int(mb.sum()) else (kb, ka)
        rs, ms = loaded[surv]
        rv, _mv = loaded[vic]
        union = ms | loaded[vic][1]
        loaded[surv] = (rs, union)
        # union goes into the survivor's ACTIVE mask file (redetect when present),
        # at that file's native resolution (redetect masks can differ from the grid)
        target = _active_path(rs)
        native = _binarize(np.load(target)).shape
        np.save(target, _fit(union, native).astype(np.uint8) * 255)
        if rs.get("overlay_path"):
            try:
                make_overlay(image_path, union, rs["overlay_path"])
            except Exception:  # noqa: BLE001 - overlay is demo-only
                pass
        dead.add(vic)
        data["instances"] = [r for r in data["instances"] if rid(r) != vic]
        data.setdefault("dropped", []).append({**rv, "reason": f"merged_into {surv}"})
        for r in data["instances"]:
            if (r.get("support") or "").lower() == vic.lower():
                r["support"] = surv
        data.setdefault("merge_log", []).append(
            {
                "kept": surv,
                "merged": vic,
                "keep_name": keep or None,
                "stage": "tier3_post_resegment",
            }
        )
        print(f"[t3merge] merged {vic} into {surv} (a_part={out.get('a_part')!r})")
    if changed:
        with open(masks_json_path, "w") as f:
            json.dump(data, f, indent=2)


def carve_overlaps(masks_json_path: str, image_path: str) -> int:
    ""
    data = json.load(open(masks_json_path))
    masks_dir = Path(masks_json_path).parent

    def _active(r: dict) -> Optional[str]:
        """Resolvable mask path (redetects don't exist yet at segmentation time)."""
        mp = r.get("mask_path")
        if mp and not os.path.exists(mp):
            alt = str(masks_dir / os.path.basename(mp))
            mp = alt if os.path.exists(alt) else None
        return mp if mp and os.path.exists(mp) else None

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

    objs: list[tuple[dict, str]] = []
    for r in data.get("instances", []):
        if r.get("kind") == "root_surface":
            continue
        mp = _active(r)
        if mp is not None:
            objs.append((r, mp))
    if len(objs) < 2:
        return 0

    shape = None
    loaded: dict[str, np.ndarray] = {}
    meta: dict[str, tuple[dict, str]] = {}
    for r, mp in objs:
        rid = f"{r['category']}#{r['instance']}"
        m = _binarize(np.load(mp))
        if shape is None:
            shape = m.shape
        loaded[rid] = _fit(m, shape)
        meta[rid] = (r, mp)

    ids = sorted(loaded, key=lambda k: int(loaded[k].sum()))  # smallest first
    n_carved = 0
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1 :]:
            A, B = loaded[a_id], loaded[b_id]
            a = int(A.sum())
            if not a:
                continue
            inter = int((A & B).sum())
            if inter <= 0.8 * a:  # not the swallowed-neighbour shape
                continue
            ra, mpa = meta[a_id]
            rb, mpb = meta[b_id]
            keep = B & ~A
            if int(keep.sum()) < 0.5 * int(B.sum()):
                print(
                    f"[carve_overlaps] carving {a_id} would cost {b_id} "
                    f">=50% of itself — container reading wrong, skipping"
                )
                continue
            native = _binarize(np.load(mpb)).shape
            np.save(mpb, _fit(keep, native).astype(np.uint8) * 255)
            if rb.get("overlay_path"):
                try:
                    make_overlay(image_path, keep, rb["overlay_path"])
                except Exception:  # noqa: BLE001 - overlay is demo-only
                    pass
            loaded[b_id] = keep
            rb["carved_px"] = int(rb.get("carved_px") or 0) + inter
            data.setdefault("carve_log", []).append(
                {
                    "carved": b_id,
                    "kept_neighbour": a_id,
                    "carved_px": inter,
                    "stage": "carve_overlaps",
                }
            )
            n_carved += 1
            print(
                f"[carve_overlaps] {a_id} ({a}px) was {inter / a:.0%} inside "
                f"{b_id} and merge declined to union — carved {inter}px out of {b_id}"
            )
    if n_carved:
        with open(masks_json_path, "w") as f:
            json.dump(data, f, indent=2)
    return n_carved


def drop_disconnected_from_support(
    results: list[InstanceMask],
    dropped: list[dict[str, Any]],
    gap_frac: float = 0.03,
) -> list[InstanceMask]:
    """Drop any object whose mask is FULLY disconnected from the mask of the root surface it
    directly rests on — every part of it lies more than ``gap_frac`` of the image's short side away.
    A genuine resting object touches the surface it sits on (the surface mask has a notch where the
    object occludes it, so the two masks are adjacent); a mask floating a gap away is a mis-
    segmentation (e.g. a wall picture wrongly assigned to the table). Runs in preprocess
    (``_prune_disconnected``) AFTER generative re-segmentation, on redetect-completed masks
    when present — so a partial detection gets its vital-part completion before being judged.

    Proximity is tested at the merge stage's downsampled resolution (``_down_mask`` short side
    ~200 px + repeated ``_dilate1``), so the tolerance is IMAGE-RELATIVE — the same convention
    ``merge_objects`` uses to decide two masks touch (~3% of the short side keeps a monitor whose
    stand wasn't segmented, while still dropping a mask floating >10% away). Only objects whose
    ``support`` resolves to a masked root surface are checked — objects on another object, or on a
    name-only (unmasked) wall, are left alone. The dropped entry keeps ``overlay_path``/``mask_path``
    so the web demo can still show the rejected mask, but the object is removed from the returned
    results (and so from masks.json / the scene graph)."""
    surf = {
        f"{r.category}#{r.instance}": r for r in results if r.kind == "root_surface"
    }
    k = max(1, round(gap_frac * 200))  # ~200px short side -> k px == gap_frac of it

    def _grown(mask_path: str) -> np.ndarray:
        m = _down_mask(_binarize(np.load(mask_path)))
        for _ in range(k):
            m = _dilate1(m)
        return m

    def _align(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Nearest-resize a downsampled mask to ``shape`` when they disagree.
        ``_down_mask`` scales each mask from its OWN native dims, and a redetect mask
        (SAM3 on the LanPaint/Qwen-edited image) can be 1-2 px off the canonical input
        size — at ~200 px that lands (200,356) vs (200,354) and broke the ``&`` below
        on 3/8 scenes of the 0728_ab768 batch. A 1-2 px nearest resize is well inside
        this heuristic's image-relative tolerance (k px == gap_frac of the short side)."""
        if m.shape == shape:
            return m
        from PIL import Image

        resized = Image.fromarray(m.astype("uint8") * 255).resize((shape[1], shape[0]))
        return np.asarray(resized) > 127

    checked, exempt = [], []
    for r in results:
        s = surf.get((r.support or "").lower()) if r.kind != "root_surface" else None
        if s is None or not (
            r.mask_path
            and os.path.exists(r.mask_path)
            and s.mask_path
            and os.path.exists(s.mask_path)
        ):
            exempt.append(r)
            continue
        checked.append((r, s.mask_path, _down_mask(_binarize(np.load(r.mask_path)))))
    region: dict[str, np.ndarray] = {}  # per support: its grown mask + accepted objs
    for _, sp, _m in checked:
        if sp not in region:
            region[sp] = _grown(sp)
    accepted: set[int] = set()
    grew = True
    while grew:
        grew = False
        for i, (r, sp, om) in enumerate(checked):
            if i in accepted:
                continue
            om = _align(om, region[sp].shape)
            if bool((region[sp] & om).any()):
                accepted.add(i)
                reg = om  # extend the contact region with the accepted object
                for _ in range(k):
                    reg = _dilate1(reg)
                region[sp] = region[sp] | reg
                grew = True
    kept: list[InstanceMask] = list(exempt)
    for i, (r, sp, om) in enumerate(checked):
        if i in accepted:
            kept.append(r)
        else:
            dropped.append(
                {
                    "category": r.category,
                    "instance": r.instance,
                    "description": r.description,
                    "support": r.support,
                    "reason": f"disconnected from support (mask > {gap_frac:.0%} of the "
                    f"image from {r.support} and from every object on it)",
                    "overlay_path": r.overlay_path,
                    "mask_path": r.mask_path,
                }
            )
    # restore the caller's original ordering (exempt + checked were split)
    order = {id(r): i for i, r in enumerate(results)}
    kept.sort(key=lambda r: order[id(r)])
    return kept


# --------------------------------------------------------------------------- #
# Describe recount-added instances (one VLM call per re-processed category)      #
# --------------------------------------------------------------------------- #
_DESCRIBE_NEW_SYSTEM = (
    "You NAME newly-detected instances of ONE object category in a photo. Each NUMBERED "
    "bright box is a NEW instance that needs a name; GRAY boxes are instances of the SAME "
    "category that are ALREADY named (their names are printed) — your descriptions must "
    "NOT duplicate them and should say how each new one DIFFERS. For each numbered box "
    'write a "description": ONE flowing noun phrase that starts DIRECTLY with the noun '
    "(NEVER 'the'/'a'/'an'), with NO commas or appended clauses, disambiguating THAT "
    "instance by visible cues a pointing model can see (position, relative size, color, "
    "spatial relation to a nearby object). Do NOT rely on printed text / brand names / "
    'logos. Respond with ONLY: {"descriptions": [{"box": <int>, "description": '
    '"<noun phrase>"}, ...]} — one entry per numbered box.'
)


def _mask_bbox(mask_path: str, W: int, H: int) -> Optional[tuple[int, int, int, int]]:
    """Pixel bbox (x0, y0, x1, y1) of a saved mask at image size (W, H), or None."""
    if not mask_path or not os.path.exists(mask_path):
        return None
    b = _binarize(np.load(mask_path))
    if b.shape != (H, W):
        from PIL import Image

        b = np.asarray(Image.fromarray(b.astype("uint8") * 255).resize((W, H))) > 127
    ys, xs = np.where(b)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def describe_new_instances(
    vlm: Callable,
    image_path: str,
    results: list["InstanceMask"],
    objects: list[dict[str, Any]],
    masks_dir: Path,
) -> None:
    """Give every recount-added instance (``source == 'sam3_recount'``) a real, pointing-
    model-compatible description via ONE VLM call PER re-processed category (so it fires at
    most once per category — usually <=1x per input). The VLM sees the photo with the new
    instances' bboxes drawn bright + numbered and the already-named same-category instances
    drawn gray + labelled, and returns a description per box. Best-effort: on any failure or
    a missing box, falls back to a positional name (``'{cat} at {region}'``) so an appended
    instance is NEVER left with the placeholder. Mutates ``rec.description`` and mirrors it
    into ``objects[...]['instances'][pos]`` so masks.json's objects/instances agree."""
    from PIL import Image, ImageDraw

    new_by_cat: dict[str, list["InstanceMask"]] = {}
    for r in results:
        if getattr(r, "source", None) == "sam3_recount":
            new_by_cat.setdefault(r.category, []).append(r)
    if not new_by_cat:
        return
    obj_by_cat = {o["category"]: o for o in objects}
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    def _positional(r: "InstanceMask") -> str:
        if r.point:
            return f"{r.category} at {_img_region([r.point[0] * W], [r.point[1] * H], W, H)}"
        return r.category

    for cat, new_recs in new_by_cat.items():
        try:
            existing = [
                r
                for r in results
                if r.category == cat and getattr(r, "source", None) != "sam3_recount"
            ]
            annotated = img.copy()
            d = ImageDraw.Draw(annotated)
            for r in existing:  # gray boxes + current names (do-not-duplicate context)
                bb = _mask_bbox(r.mask_path, W, H)
                if bb:
                    d.rectangle(bb, outline=(150, 150, 150), width=3)
                    d.text(
                        (bb[0] + 3, bb[1] + 3),
                        (r.description or cat)[:24],
                        fill=(220, 220, 220),
                    )
            anchors = []
            for n, r in enumerate(new_recs, 1):  # bright numbered boxes
                bb = _mask_bbox(r.mask_path, W, H)
                if bb:
                    d.rectangle(bb, outline=(255, 40, 40), width=4)
                    d.text((bb[0] + 4, bb[1] + 4), str(n), fill=(255, 255, 0))
                    anchors.append(
                        f"{n}: {cat} near {_img_region([(bb[0] + bb[2]) / 2], [(bb[1] + bb[3]) / 2], W, H)}"
                    )
            ann_path = str(masks_dir / f"_describe_{slugify(cat)}.png")
            annotated.save(ann_path)
            parts = [
                {
                    "type": "text",
                    "text": f"Category: {cat}. Name each NUMBERED bright box.\n"
                    + "\n".join(anchors),
                },
                _img_part(ann_path),
            ]
            out, err = None, ""
            for _ in range(3):
                retry = parts + (
                    [
                        {
                            "type": "text",
                            "text": f"Previous reply was not valid "
                            f"JSON ({err}). Return ONLY the JSON object.",
                        }
                    ]
                    if err
                    else []
                )
                try:
                    out = parse_json(vlm(_DESCRIBE_NEW_SYSTEM, retry, max_tokens=800))
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    err = str(e)[:200]
            by_box: dict[int, str] = {}
            if isinstance(out, dict):
                for e in out.get("descriptions", []) or []:
                    if isinstance(e, dict) and "box" in e:
                        desc = (
                            re.sub(
                                r"^(the|a|an)\s+",
                                "",
                                str(e.get("description", "")).strip(),
                                flags=re.IGNORECASE,
                            )
                            .split(",")[0]
                            .strip()
                        )
                        if desc:
                            by_box[int(e["box"])] = desc
            for n, r in enumerate(new_recs, 1):
                desc = by_box.get(n) or _positional(r)
                r.description = desc
                oc = obj_by_cat.get(cat)
                if oc and 0 <= r.instance < len(oc.get("instances", [])):
                    oc["instances"][r.instance]["description"] = desc
        except Exception as e:  # noqa: BLE001 - naming is best-effort; keep positional fallback
            print(f"[masking] describe_new_instances failed for {cat}: {e}")
            for r in new_recs:
                if not r.description or "(recount #" in (r.description or ""):
                    r.description = _positional(r)


GROUP_SPLIT_MIN_FRAC = 0.08  # a member must be >= this fraction of the group mask (no slivers)
GROUP_SPLIT_CONTAIN = 0.80  # a member must lie >= this fraction inside the group mask
GROUP_SPLIT_COVER = 0.70  # the members together must explain >= this fraction of the group


def segment_scene(
    image_path: str,
    out_dir: str,
    objects: Optional[list[Any]] = None,
    model: str = DEFAULT_MODEL,
    points_npy: Optional[str] = None,
    points_ready: Optional[Callable[[], None]] = None,
    sam3: Optional[Sam3Server] = None,
    molmo: Optional[MolmoServer] = None,
    vlm: Optional[Callable] = None,
    ignore_objects: Optional[list[str]] = None,
    proposer_verify: bool = True,
    same_size_categories: Optional[list[str]] = None,
    room_mode: bool = False,
    mask_merge: bool = True,
) -> list[InstanceMask]:
    """VLM objects -> per category: SAM3 all-instance masks; per instance: Molmo
    points the detailed description, assigned to its containing mask.

    A top-level ROUTER call first classifies the scene 3-way (``scene_kind``:
    'room'|'closeup:table'|'closeup:tabletop', persisted in masks.json). With
    ``room_mode`` OFF (default), a 'room' verdict is demoted to closeup:table, while
    the table/tabletop form flag remains effective. With ``room_mode`` ON, a 'room'
    scene routes to the room proposer/verifier pair (floor = anchor root surface,
    the work surface an ordinary object on it).

    The proposer lists objects/surfaces/relationships. ``_segment_objects`` then produces at
    most one mask per instance (dropping any it can't localize), and ``merge_objects``
    collapses over-segmented pieces. When ``proposer_verify`` is enabled, a SECOND VLM pass
    audits that resulting segmentation AFTER segment+merge using a surfaces-only overlay: it
    may add missing bounding surfaces by name and correct relationships, but it never drops
    objects. ``objects`` may be given to skip proposing (a flat list of category strings is
    also accepted). Servers are injectable (tests); else started and torn down.
    """
    from lib.utils._path import MOLMO_PY, SAM3_PY

    masks_dir = Path(out_dir) / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    caller_vlm = vlm  # a caller-supplied vlm (tests) also drives the router below
    vlm = vlm or _make_vlm(model, effort="high")

    own_sam3 = own_molmo = None
    if sam3 is None:
        own_sam3 = sam3 = Sam3Server(
            SAM3_PY, log_path=str(masks_dir / "sam3_server.log")
        )
    if molmo is None:
        own_molmo = molmo = MolmoServer(
            MOLMO_PY, log_path=str(masks_dir / "molmo_server.log")
        )

    proposed = objects is None  # proposer ran -> the verifier audits it
    scene_kind, scene_form_flag, mode = "closeup:table", "closeup:table", "closeup"
    if proposed:
        scene_kind = route_scene(
            caller_vlm or _make_vlm(model, effort="medium"), image_path
        )
        scene_form_flag = (
            scene_kind
            if room_mode
            else ("closeup:table" if scene_kind == "room" else scene_kind)
        )
        mode = "room" if scene_form_flag == "room" else "closeup"
        print(f"[segment_scene] router: scene_kind={scene_kind} -> {mode} track")
        objects, relationships = propose_objects(
            vlm,
            image_path,
            ignore_objects=ignore_objects,
            same_size_categories=same_size_categories,
            mode=mode,
        )
    else:
        objects, relationships = normalize_objects(objects), []
    # First point of depth use is per-instance annotation inside the ladder below —
    # everything above (server spawns, proposer) is depth-free, so preprocess runs the
    # MoGE-2 estimate as a BACKGROUND subprocess and we join it here (the load was
    # deliberately moved from the top of this function to buy that overlap window).
    if points_ready is not None:
        points_ready()
    moge_points = (
        np.load(points_npy) if points_npy and os.path.exists(points_npy) else None
    )
    review = None
    composite_path = str(masks_dir / "composite.png")
    merge_log: list[dict[str, Any]] = []
    try:
        to_mask = objects_for_masking(objects)
        results, dropped, unmasked_roots = _segment_objects(
            image_path,
            to_mask,
            masks_dir,
            sam3,
            molmo,
            moge_points,
            vlm=vlm,
            defer_dedup=mask_merge,  # no merge stage -> dedup must not leave overlaps
        )
        # Name any recount-added instances (source='sam3_recount') before merge/verify and
        # masks.json read their descriptions (one VLM call per re-processed category).
        describe_new_instances(vlm, image_path, results, to_mask, masks_dir)
        if mask_merge:
            results, merge_log, merge_decisions = merge_objects(
                vlm, image_path, results, masks_dir, moge_points
            )
        else:
            merge_log, merge_decisions = [], []  # keep masks.json shape unchanged
        # NOTE: the geometric disconnected-from-support prune no longer runs here — it
        # moved to preprocess._prune_disconnected, AFTER generative re-segmentation, so a
        # partially-detected object (monitor with an unmasked stand) gets its vital-part
        # completion before being judged. The verifier below sees floating masks too.
        # Verifier (one pass, AFTER segment+merge): shown the surfaces-only overlay, it
        # adds missing bounding surfaces (name-only, built visually by the initializer)
        # and fixes relationships; it never drops objects.
        if proposed and proposer_verify:
            # The verifier gets its OWN surfaces-only overlay; `composite_path` stays the
            # full every-instance overlay served by the demo.
            surf_composite = os.path.join(
                os.path.dirname(composite_path), "composite_surfaces.png"
            )
            surf_colors = make_surface_composite(
                image_path,
                [r for r in results if r.kind == "root_surface"],
                surf_composite,
            )
            hints = None
            results, relationships, dropped, unmasked_roots, review = (
                verify_segmentation(
                    vlm,
                    surf_composite,
                    objects,
                    results,
                    relationships,
                    dropped,
                    unmasked_roots,
                    hints=hints,
                    image_path=image_path,
                    mode=mode,
                    colors=surf_colors,
                )
            )
    finally:
        if own_sam3 is not None:
            own_sam3.close()
        if own_molmo is not None:
            own_molmo.close()

    # Relationship proposals are intentionally still advisory here.  Keep the router's
    # one-shot form as the effective contract until metric preprocessing adjudicates the
    # claims; a raw floor-under-table hypothesis cannot itself force a full table/base.
    # ``scene_kind`` remains the raw router verdict for auditability.
    instance_payload = [r.to_dict() for r in results]
    if not proposed:
        form_source = "default"
        form_reason = (
            "pre-specified objects skipped routing; using safe closeup:table default"
        )
    elif scene_kind == "room" and not room_mode:
        form_source = "room_mode_demotion"
        form_reason = (
            "raw room verdict demoted to closeup:table because room mode is disabled"
        )
    else:
        form_source = "router"
        form_reason = "effective form matches the raw router verdict"
    make_composite(
        image_path, results, composite_path
    )  # final overlay (post drops/adds)

    payload = {
        "objects": objects,
        # Router verdict + which track actually ran (mode == 'room' only when the
        # flag was on AND the router said room). scene_kind is recorded even in
        # record-only runs so batches can be audited for routing stability.
        "scene_kind": scene_kind,
        "routing": {
            "room_mode_enabled": room_mode,
            "mode": mode,
            "form": scene_form_flag,
            "form_source": form_source,
            "form_reason": form_reason,
        },
        # VLM provenance is preserved, but every claim stays advisory until metric
        # preprocessing adjudicates its authority.
        "relationships": relationships,
        "proposal_review": review,  # verifier patch (added surfaces / drops / rel fixes)
        "categories_masked": list(dict.fromkeys(r.category for r in results)),
        "root_surfaces": list(
            dict.fromkeys(r.category for r in results if r.kind == "root_surface")
        ),
        "merge_log": merge_log,
        "merge_decisions": merge_decisions,  # every VLM-judged pair (web-demo inspection)
        "dropped": dropped,  # listed-but-unmasked instances (objects; also roots, reason=root_unsegmented)
        # name-only root surfaces the VERIFIER re-added from the image (walls / floor the
        # segmenter never masked) — kept so the scene graph + initializer build them visually.
        "unmasked_root_surfaces": unmasked_roots,
        "composite": composite_path,
        "instances": instance_payload,
    }
    with open(masks_dir / "masks.json", "w") as f:
        json.dump(payload, f, indent=2)
    if mask_merge:
        try:
            carve_overlaps(str(masks_dir / "masks.json"), image_path)
        except Exception as e:  # noqa: BLE001 - carve is cleanup, never fatal
            print(f"WARNING [masking]: carve_overlaps failed: {e}")
    return results


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    if a.shape != b.shape:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _cross_out(image_path: str, masks: list, out_path: str) -> str:
    """Photo copy with the given (already-claimed) instance masks grayed out and
    X-marked — the Molmo-collapse salvage points at "the OTHER <category>" on this
    image, turning a failed linguistic disambiguation into a visual one."""
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    H, W = arr.shape[:2]
    boxes = []
    for m in masks:
        mb = m > 0
        if mb.shape != (H, W):
            mb = (
                np.asarray(Image.fromarray(mb.astype("uint8") * 255).resize((W, H)))
                > 127
            )
        arr[mb] = 0.25 * arr[mb] + 0.75 * 128.0
        ys, xs = np.where(mb)
        if xs.size:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    out = Image.fromarray(arr.clip(0, 255).astype("uint8"))
    draw = ImageDraw.Draw(out)
    for x0, y0, x1, y1 in boxes:
        w = max(2, (x1 - x0) // 40)
        draw.line([(x0, y0), (x1, y1)], fill=(255, 0, 0), width=w)
        draw.line([(x0, y1), (x1, y0)], fill=(255, 0, 0), width=w)
    out.save(out_path)
    return out_path


def _sam3_instance_indices(
    masks: Optional[np.ndarray],
    min_area_frac: float = 0.0005,
    dedup_iou: float = 0.7,
) -> list[int]:
    if masks is None or not len(masks):
        return []
    h, w = masks.shape[1:]
    min_area = min_area_frac * h * w
    kept: list[int] = []
    for i in range(len(masks)):
        if int((masks[i] > 0).sum()) < min_area:
            continue
        if any(_mask_iou(masks[i], masks[j]) > dedup_iou for j in kept):
            continue
        kept.append(i)
    return kept


_T3_SCALE_SYSTEM = (
    "You referee a segmentation SCALE dispute. One object was clicked; the segmenter "
    "returned candidate regions of different extents around the SAME click. Answer one "
    "question: which candidate covers the WHOLE named object — all of it, and nothing "
    "else? Typical traps: a candidate that is only a printed logo/letter/label or one "
    "face/part of the object (too small), and a candidate that also swallows "
    "neighbouring objects or the surface the object rests on (too large). The named "
    "object may itself be large (e.g. a tabletop filling most of the frame): judge "
    "against the description, not against size. Reply with JSON only: "
    '{"choice": "A"} (or "B"/"C"), or {"choice": "none"} if no candidate covers '
    "exactly the whole object."
)


def _t3_adjudicate(
    vlm: Callable[..., str],
    image_path: str,
    masks_dir: Path,
    tag: str,
    cat: str,
    desc: str,
    survivors: list[tuple[int, np.ndarray, float]],
) -> Optional[int]:
    """One VLM call deciding a tier-3 scale dispute. ``survivors`` are the guard-passing
    candidates as (stack index, bool mask, area fraction), largest first, at most 3.
    All crops share one framing (the union bbox) so extents are comparable. Returns the
    chosen stack index, or None for 'none'/invalid (caller falls back to argmax)."""
    letters = "ABC"
    union = np.zeros_like(survivors[0][1])
    for _i, mb, _a in survivors:
        union = union | mb
    clean = str(masks_dir / f"_t3scale_{tag}_clean.png")
    _union_crop(image_path, union, union, clean)
    parts: list = [
        {"type": "text", "text": "Full image (context):"},
        _img_part(image_path),
        {
            "type": "text",
            "text": f"The clicked object — '{cat}': {desc}. "
            "Zoomed view of the disputed area (no marks):",
        },
        _img_part(clean),
    ]
    for letter, (_i, mb, area) in zip(letters, survivors):
        tmp = str(masks_dir / f"_t3scale_{tag}_{letter}_ov.png")
        marked = str(masks_dir / f"_t3scale_{tag}_{letter}.png")
        _pair_overlay(image_path, mb, np.zeros_like(mb), tmp)
        _union_crop(tmp, union, union, marked)
        parts += [
            {
                "type": "text",
                "text": f"Candidate {letter} (red, {100 * area:.1f}% of the image):",
            },
            _img_part(marked),
        ]

    def _validate_choice(reply: dict) -> dict:
        choice = reply.get("choice")
        if not isinstance(choice, str):
            raise ValueError("choice must be a JSON string")
        choice = choice.strip().upper()
        allowed = set(letters[: len(survivors)]) | {"NONE"}
        if choice not in allowed:
            raise ValueError(f"choice must be one of {sorted(allowed)}")
        reply["choice"] = "none" if choice == "NONE" else choice
        return reply

    out = vlm_json(
        vlm,
        _T3_SCALE_SYSTEM,
        parts,
        label=f"t3_scale {tag}",
        default={"choice": "none"},
        validate=_validate_choice,
    )
    choice = str(out.get("choice", "none")).strip().upper()
    if choice in letters[: len(survivors)]:
        return survivors[letters.index(choice)][0]
    return None


def _segment_objects(
    image_path: str,
    objects: list[dict[str, Any]],
    masks_dir: Path,
    sam3: Sam3Server,
    molmo: MolmoServer,
    moge_points: Optional[np.ndarray],
    dedup_iou: float = 0.7,
    min_iou: float = 0.6,  # tier-3 gate on SAM's *predicted* IoU (a conservative self-confidence,
    # not measured accuracy); 0.7 dropped good masks on cluttered scenes
    # (e.g. a workbench: soldering station 0.68, screwdriver holder 0.63).
    vlm: Optional[
        Callable[..., str]
    ] = None,  # tier-3 scale adjudicator; None = argmax only
    defer_dedup: bool = True,  # dedup DEFERS distinct-category contained pairs to
) -> tuple[list[InstanceMask], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tiered masking, text-first with two escalating fallbacks.
    Returns ``(results, dropped, unmasked_roots)``.

    - **Tier 1 (text):** SAM3 ``segment_all(category)`` on the FULL image; each instance
      is assigned to the mask at its Molmo point (root surfaces take the largest mask).
      The reliable primary path.
    - **Tier 2 (crop + text):** small objects the full-image detector misses. Crop a
      1/3 x 1/3 window centered on the object's Molmo point and re-run
      ``segment_all(category)`` on the zoomed crop (where the object is large enough to
      detect), then map the chosen mask back to full-image coordinates.
    - **Single-instance union:** in tiers 1 and 2, when a category has exactly ONE
      instance but the (category-specific) detector returns multiple masks, those masks
      are parts of one bound object (e.g. the keys/fobs of a single keychain) and are
      UNIONED into that one node (tier 1 unions masks whose centroid is near the object's
      point; the tier-2 crop is already tight, so it unions all crop masks).
    - **Tier 3 (tracker, all points):** SAM3 interactive predictor prompted with ALL of
      the instance's Molmo points at once (every point is foreground). Last resort. For a
      ROOT SURFACE (which has no point at tier 1 and no crop), this is its only fallback:
      Molmo points at the surface and the tracker recovers the contiguous region (e.g. a
      wooden cutting-board the "table" concept failed to ground).
    - still failing -> drop (surfaced in ``dropped``).

    Each accepted instance records the ``tier`` (1/2/3) it was solved at. The instance
    INDEX is the VLM listing position, so node ids ('book#0', 'book#1') stay stable."""
    results: list[InstanceMask] = []
    dropped: list[dict[str, Any]] = []
    # Always empty here; the verifier later appends name-only re-detected roots.
    unmasked_roots: list[dict[str, Any]] = []
    claimed: list[np.ndarray] = []  # accepted masks, for dedup
    cat_state: dict[str, tuple] = {}  # per-category (text_masks, used_text) for salvage
    # (cat, pos) -> pixels keep_own_components removed; the restore post-pass at the
    # end (restore_severed_components) gives back occluder-severed parts once every
    # claimed mask exists.
    trimmed: dict[tuple, np.ndarray] = {}
    from PIL import Image

    img = Image.open(image_path).convert("RGB")  # for tier-2 crops
    W, H = img.size

    # Hybrid root surfaces: ATTEMPT to mask EVERY root surface (not just object-parents).
    # A masked root yields a RANSAC plane (point+normal) the initializer can follow; a root
    # the segmentor can't localize is DROPPED here -- the verifier re-detects genuinely-missing
    # bounding surfaces from the image, so ``unmasked_roots`` is populated ONLY by the verifier.

    def _dup(mb: np.ndarray, kind: Optional[str] = None) -> bool:
        # `claimed` and `results` are index-parallel (same invariant _overlap_idx uses).
        return is_duplicate_mask(
            mb, claimed, [r.kind for r in results], dedup_iou, kind
        )

    t3_multi = os.environ.get("GRASE_T3_MULTIMASK", "1") != "0"

    def _t3_scale_pick(cat, pos, desc, pt, cguard, mp) -> Optional[np.ndarray]:
        """Multimask scale adjudication for a SINGLE-click tier-3 prompt. Returns the
        VLM-chosen candidate as a uint8 0/255 mask (already written to ``mp``), or
        None -> the caller falls back to the argmax path unchanged.

        Deterministic pre-filter first: a candidate colliding with a claimed mask
        (``_dup``) or swallowing >=3 foreign object points (``cguard``; roots pass
        None) is geometrically impossible and never reaches the VLM. The VLM is
        consulted ONLY on a genuine scale dispute among survivors (largest > 2x
        smallest) — same-scale survivors are jitter, and the argmax path decides."""
        if not (t3_multi and vlm is not None):
            return None
        if not hasattr(sam3, "segment_point_candidates"):
            return None  # older server/stub: argmax path only
        cp = str(masks_dir / f"_t3cands_{slugify(cat)}_{pos}.npy")
        stack, _ious = sam3.segment_point_candidates(image_path, [list(pt)], cp)
        if stack is None or len(stack) < 2:
            return None
        survivors = []
        for i in range(len(stack)):
            mb = stack[i] > 0
            if not mb.any() or _dup(mb) or (cguard is not None and cguard(mb)):
                continue
            survivors.append((i, mb, float(mb.mean())))
        if len(survivors) < 2:
            return None
        survivors.sort(key=lambda t: -t[2])
        if survivors[0][2] <= 2 * survivors[-1][2]:
            return None  # same scale — no dispute to referee
        pick = _t3_adjudicate(
            vlm,
            image_path,
            masks_dir,
            f"{slugify(cat)}_{pos}",
            cat,
            desc,
            survivors[:3],
        )
        if pick is None:
            return None
        chosen = (stack[pick] > 0).astype("uint8") * 255
        np.save(mp, chosen)
        print(
            f"[masking] {cat}#{pos}: tier-3 scale dispute -> VLM chose candidate "
            f"{pick} ({100 * float(chosen.mean() / 255):.1f}% of the image)"
        )
        return chosen

    syn_stacks: dict[tuple[str, str], Optional[np.ndarray]] = {}
    syn_state: dict[str, Optional[np.ndarray]] = {}
    syn_used: dict[str, set] = {}

    def _syn_stack(cat: str, syn: str) -> Optional[np.ndarray]:
        """SAM3 ``segment_all`` masks for one (category, synonym), grounded ONCE per
        run. ``None`` when the synonym grounds nothing."""
        key = (cat, syn)
        if key not in syn_stacks:
            sp = str(masks_dir / f"{slugify(cat)}_all_syn_{slugify(syn)}.npy")
            sm, _ = sam3.segment_all(image_path, syn, sp)
            syn_stacks[key] = sm if sm is not None and len(sm) else None
        return syn_stacks[key]

    def _syn_pool(obj: dict, cat: str) -> Optional[np.ndarray]:
        if cat in syn_state:
            return syn_state[cat]
        stacks = []
        for syn in obj.get("synonyms") or []:
            if not syn:
                continue
            sm = _syn_stack(cat, syn)
            if sm is not None:
                stacks.append(sm)
        pool = np.concatenate(stacks) if stacks else None
        syn_state[cat] = pool
        return pool

    def _overlap_idx(mb: np.ndarray, kind: str) -> Optional[int]:
        """Index of a claimed SAME-KIND mask that is the same object as ``mb`` — either
        high IoU, or one mask largely CONTAINED in the other (part-vs-whole, which
        symmetric IoU misses: e.g. a small 'mushroom' part inside the whole 'toy'). Only
        same-kind, so a small OBJECT sitting within a ROOT SURFACE's extent is not merged.
        Returns None if no same-object overlap."""
        a = int(mb.sum())
        for i, c in enumerate(claimed):
            if results[i].kind != kind:
                continue
            inter = int((mb & c).sum())
            if not inter:
                continue
            ca = int(c.sum())
            union = a + ca - inter
            iou = inter / union if union else 0.0
            contain = inter / min(a, ca) if min(a, ca) else 0.0
            if iou > dedup_iou or contain > 0.8:
                return i
        return None

    # Molmo points for EVERY instance (objects AND root surfaces), computed once up front and
    # reused across tiers (no second Molmo call). Objects use their point to pick the mask AT it;
    # a root surface uses it to disambiguate several same-category masks by DESCRIPTION (e.g. the
    # wall BEHIND the table vs the wall on the left) instead of blindly taking the largest. Two
    # guards below still count only OBJECT points (``all_obj_points``):
    #   C  (under-segmentation): reject an object mask covering >=3 object points (a blob
    #      that merged several neighbours, e.g. a "toy" mask swallowing the corn+banana).
    #   dedup guard: only collapse a part-vs-whole overlap when the fuller mask spans <3
    #      objects (a true over-split, e.g. mushroom-part inside the toy-whole), so a
    #      distinct neighbour is never absorbed.
    inst_points: dict[tuple[str, int], list] = {}
    obj_keys: set[tuple[str, int]] = set()
    for o in objects:
        for p, it in enumerate(o["instances"]):
            inst_points[(o["category"], p)] = molmo.point(image_path, it["description"])
            if it.get("kind") != "root_surface":
                obj_keys.add((o["category"], p))
    obj_point = {k: inst_points[k][0] for k in obj_keys if inst_points.get(k)}
    # (cat, pos) -> the instance keys that REST ON it (declared children). Excluded
    # from the C-guard counts ONLY — never from the dedup guard, where the spanned
    # points are exactly what stops a distinct neighbour from being absorbed.
    _id2key = {
        f"{o['category']}#{p}".lower(): (o["category"], p)
        for o in objects
        for p in range(len(o["instances"]))
    }
    children: dict[tuple, set] = {}
    for o in objects:
        for p, it in enumerate(o["instances"]):
            sk = _id2key.get((it.get("support") or "").strip().lower())
            if sk is not None:
                children.setdefault(sk, set()).add((o["category"], p))

    def _obj_points_in(
        mb: np.ndarray,
        only: Optional[set] = None,
        exclude: Optional[set] = None,
    ) -> int:
        """How many distinct OBJECT-instance Molmo points fall inside mask ``mb``.
        ``only`` restricts the count to those instance keys — the salvage pass counts
        settled instances + the one under retry, so the stale points of known-collapsed,
        still-unresolved siblings cannot re-trigger the C guard. ``exclude`` drops
        keys from the count — the C guard passes an instance's declared CHILDREN
        (a container's mask legitimately spans its own contents' points; bridge5's
        bin holds all 8 toys)."""
        n = 0
        for k, (u, v) in obj_point.items():
            if only is not None and k not in only:
                continue
            if exclude is not None and k in exclude:
                continue
            if mb[min(int(v * H), H - 1), min(int(u * W), W - 1)]:
                n += 1
        return n

    def _mask_one(
        obj: dict,
        cat: str,
        pos: int,
        inst: dict,
        text_masks,
        used_text: set,
        count_only: Optional[set] = None,
        record_drop: bool = True,
        allow_supersede: bool = True,
    ) -> bool:
        """Run the 3-tier ladder + guards for ONE instance; on acceptance mutates
        results/claimed (and dropped, unless ``record_drop`` is False) and returns
        True. Called by the main loop and re-called by the Molmo-collapse salvage
        (``count_only`` scopes the C guard there). ``allow_supersede=False`` (recount)
        forbids REPLACING an already-claimed same-kind mask: a new instance that overlaps
        an existing one is DROPPED, never swapped in (so a recount orphan can never
        overwrite a different-category instance, e.g. gpt1's notebook masked as a book)."""
        desc = inst["description"]
        kind = inst.get("kind", "object")
        mp = masks_dir / f"{slugify(cat)}_{pos}.npy"
        pts = inst_points.get((cat, pos), [])
        mask, tier, point = None, None, None

        if text_masks is not None and len(text_masks):
            if kind == "root_surface":
                c = None
                if pts:
                    c = best_mask_for_point(text_masks, pts[0][0], pts[0][1])
                    if (
                        c is None
                    ):  # NEAREST mask whose border is within ~W/30 of the point
                        from scipy.ndimage import distance_transform_edt

                        h, w = text_masks[0].shape[-2:]
                        tol_px = max(3, w // 30)
                        px = min(max(int(round(pts[0][0] * (w - 1))), 0), w - 1)
                        py = min(max(int(round(pts[0][1] * (h - 1))), 0), h - 1)
                        best_d = None
                        for i in range(len(text_masks)):
                            d = float(
                                distance_transform_edt(~_binarize(text_masks[i]))[
                                    py, px
                                ]
                            )
                            if d <= tol_px and (best_d is None or d < best_d):
                                best_d, c = d, i
                picked_by_point = c is not None
                if c is None and not pts:  # no point at all: keep the largest-mask path
                    c = max(
                        range(len(text_masks)),
                        key=lambda i: int((text_masks[i] > 0).sum()),
                    )
                if c is not None and c not in used_text:
                    mask, tier = text_masks[c], 1
                    if picked_by_point:
                        point = list(pts[0])
                    used_text.add(c)
            else:
                for pu, pv in pts:
                    c = best_mask_for_point(text_masks, pu, pv)
                    if c is None or c in used_text:
                        continue
                    used_text.add(c)
                    point, tier = [pu, pv], 1
                    if len(obj["instances"]) == 1 and len(text_masks) > 1:
                        win = crop_box(pu * W, pv * H, W, H)
                        um = text_masks[c] > 0
                        exempt = {(cat, pos)} | (children.get((cat, pos)) or set())
                        for j in range(len(text_masks)):
                            if (
                                j != c
                                and j not in used_text
                                and _centroid_in(text_masks[j], win)
                            ):
                                if _obj_points_in(
                                    text_masks[j] > 0,
                                    only=count_only,
                                    exclude=exempt,
                                ):
                                    print(
                                        f"[masking] {cat}#{pos}: union candidate "
                                        "holds another instance's point — skipped"
                                    )
                                    continue
                                um |= text_masks[j] > 0
                                used_text.add(j)
                        mask = um.astype(np.uint8) * 255
                    else:
                        mask = text_masks[c]
                    break

        if kind != "root_surface" and pts:
            if mask is None:
                pool = _syn_pool(obj, cat)
                if pool is not None and len(pool):
                    used = syn_used.setdefault(cat, set())
                    j = best_mask_for_point(pool, pts[0][0], pts[0][1])
                    if j is not None and j not in used:
                        used.add(j)
                        mask, tier, point = pool[j], 1, list(pts[0])
                        print(
                            f"[masking] {cat}#{pos}: point missed every base "
                            "candidate; synonym pool grounded it (tier 1)"
                        )
            elif tier == 1 and len(obj["instances"]) == 1:
                pool = _syn_pool(obj, cat)
                if pool is not None and len(pool):
                    used = syn_used.setdefault(cat, set())
                    j = best_mask_for_point(pool, point[0], point[1])
                    if j is not None and j not in used:
                        mb0, sb = mask > 0, pool[j] > 0
                        inter = int((mb0 & sb).sum())
                        if (
                            mb0.sum()
                            and inter / int(mb0.sum()) >= 0.8
                            and int(sb.sum()) >= 1.5 * int(mb0.sum())
                        ):
                            used.add(j)
                            mask = pool[j]
                            print(
                                f"[masking] {cat}#{pos}: base candidate is a PART "
                                f"({int(mb0.sum())}px) of a synonym WHOLE "
                                f"({int(sb.sum())}px); taking the whole (tier 1)"
                            )

        # C (under-segmentation guard): reject an object mask that spans >=3 distinct
        # object points (a blob merging neighbours, e.g. "toy" over corn+banana) so the
        # next tier can recover a tight, single-object mask. The instance's own declared
        # CHILDREN are exempt from the count: a container's mask rightfully spans its
        # contents (bridge5's bin holds all 8 toys), and SAM3's carve-outs of the
        # occluding contents are not guaranteed.
        if (
            mask is not None
            and kind != "root_surface"
            and _obj_points_in(
                mask > 0, only=count_only, exclude=children.get((cat, pos))
            )
            >= 3
        ):
            mask, tier, point = None, None, None

        if mask is None and pts and kind != "root_surface":
            u, v = pts[0]
            crop_path = str(masks_dir / f"{slugify(cat)}_{pos}_crop.png")
            crop_all = str(masks_dir / f"{slugify(cat)}_{pos}_crop_all.npy")

            def _win_clip(fm: np.ndarray, x0, y0, x1, y1) -> float:
                """Fraction of the window border covered by the mask, counting ONLY
                edges interior to the image (a frame-cut object legitimately rides
                the image border)."""
                edges = []
                if y0 > 0:
                    edges.append(fm[y0, x0:x1])
                if y1 < H:
                    edges.append(fm[y1 - 1, x0:x1])
                if x0 > 0:
                    edges.append(fm[y0:y1, x0])
                if x1 < W:
                    edges.append(fm[y0:y1, x1 - 1])
                return float(np.concatenate(edges).mean()) if edges else 0.0

            clipped_full = None
            for f in (1 / 3, 2 / 3):
                x0, y0, x1, y1 = crop_box(u * W, v * H, W, H, fw=f, fh=f)
                cw, ch = x1 - x0, y1 - y0
                img.crop((x0, y0, x1, y1)).save(crop_path)
                pu_c, pv_c = (
                    (u * W - x0) / cw,
                    (v * H - y0) / ch,
                )  # Molmo point in crop coords
                full = None
                for term in [cat] + [s for s in (obj.get("synonyms") or []) if s]:
                    cmasks, _ = sam3.segment_all(crop_path, term, crop_all)
                    if (
                        cmasks is None
                        or not len(cmasks)
                        or cmasks.shape[1:] != (ch, cw)
                    ):
                        continue
                    if len(obj["instances"]) == 1 and len(cmasks) > 1:
                        # ONE bound object split into parts: the crop is tight around it
                        # and the masks are category-specific, so union them all —
                        # except candidates holding ANOTHER instance's Molmo point
                        # (same guard as the tier-1 union; a stacked neighbour can sit
                        # entirely inside the crop). If the surviving union misses the
                        # own point, the invariant check below rejects the term.
                        exempt = {(cat, pos)} | (children.get((cat, pos)) or set())
                        cm = np.zeros((ch, cw), bool)
                        for k in range(len(cmasks)):
                            part = np.zeros((H, W), bool)
                            part[y0:y1, x0:x1] = cmasks[k] > 0
                            if _obj_points_in(part, only=count_only, exclude=exempt):
                                print(
                                    f"[masking] {cat}#{pos}: tier-2 union candidate "
                                    "holds another instance's point — skipped"
                                )
                                continue
                            cm |= cmasks[k] > 0
                    else:
                        ci = best_mask_for_point(cmasks, pu_c, pv_c)
                        if ci is None:
                            # The text term didn't ground on the object AT the Molmo point —
                            # it locked onto a neighbour that happens to fall in the crop. Do
                            # NOT fall back to the largest blob (that grabs the wrong object);
                            # skip the term and let tier 3 (point-prompted) recover the object.
                            continue
                        cm = cmasks[ci] > 0
                    # Invariant: an object's mask MUST contain its own Molmo point. A mask that
                    # doesn't is grounded on the wrong thing (e.g. SAM3 reading a fabric glasses
                    # case as the nearby container) -> reject so tier 3 can recover it.
                    if not point_in_mask(cm, pu_c, pv_c):
                        continue
                    cand = np.zeros((H, W), np.uint8)
                    cand[y0:y1, x0:x1] = cm.astype(np.uint8) * 255
                    if not _dup(cand > 0):
                        full = cand
                        break
                if full is None:
                    break  # nothing grounded in the window; growing won't invent it
                clip = _win_clip(full > 0, x0, y0, x1, y1)
                clipped_full = full
                if clip <= 0.05:
                    mask, tier, point = full, 2, [u, v]
                    break
                print(
                    f"[masking] {cat}#{pos}: tier-2 mask rides the crop-window edge "
                    f"({clip:.0%} of interior border, window {f:.2f}) — "
                    + ("growing the window" if f < 0.5 else "escalating")
                )
            if mask is None and clipped_full is not None:
                pool = _syn_pool(obj, cat)
                j = (
                    best_mask_for_point(pool, u, v)
                    if pool is not None and len(pool)
                    else None
                )
                used = syn_used.setdefault(cat, set())
                if j is not None and j not in used and not _dup(pool[j] > 0):
                    used.add(j)
                    mask, tier, point = pool[j], 2, [u, v]
                    print(
                        f"[masking] {cat}#{pos}: clipped at both windows; full-image "
                        "synonym candidate at the point wins"
                    )
                else:
                    mask, tier, point = clipped_full, 2, [u, v]
        if (
            mask is not None
            and kind != "root_surface"
            and _obj_points_in(
                mask > 0, only=count_only, exclude=children.get((cat, pos))
            )
            >= 3
        ):
            mask, tier, point = None, None, None  # C guard (see tier 1)

        # tier 3: SAM3 tracker prompted with ALL of the instance's Molmo points.
        # Single-click prompts run the scale adjudicator first (multimask candidates,
        # deterministic pre-filter, VLM on dispute); its pick needs no min_iou gate —
        # predicted IoU is meaningless across scales (see _T3_SCALE_SYSTEM rationale).
        if mask is None and pts and kind != "root_surface":
            m3c = (
                _t3_scale_pick(
                    cat,
                    pos,
                    desc,
                    pts[0],
                    lambda mb: (
                        _obj_points_in(
                            mb, only=count_only, exclude=children.get((cat, pos))
                        )
                        >= 3
                    ),
                    str(mp),
                )
                if len(pts) == 1
                else None
            )
            if m3c is not None:
                mask, tier, point = m3c, 3, list(pts[0])
            else:
                m3, iou3 = sam3.segment_points(
                    image_path, [list(p) for p in pts], str(mp)
                )
                if m3 is not None and iou3 >= min_iou and not _dup(m3 > 0):
                    mask, tier, point = m3, 3, list(pts[0])
        if (
            mask is not None
            and kind != "root_surface"
            and _obj_points_in(
                mask > 0, only=count_only, exclude=children.get((cat, pos))
            )
            >= 3
        ):
            mask, tier, point = None, None, None  # C guard (see tier 1)

        if mask is None and kind == "root_surface":
            rpts = pts
            if rpts:
                mrc = (
                    _t3_scale_pick(cat, pos, desc, rpts[0], None, str(mp))
                    if len(rpts) == 1
                    else None
                )
                if mrc is not None:
                    mask, tier, point = mrc, 3, list(rpts[0])
                else:
                    mr, _iour = sam3.segment_points(
                        image_path, [list(p) for p in rpts], str(mp)
                    )
                    if mr is not None and not _dup(mr > 0, kind):
                        mask, tier, point = mr, 3, list(rpts[0])

        # give up: an instance the segmentor couldn't localize is DROPPED -- whether an
        # object OR a root surface. We no longer carry a mask-less root as a name-only
        # ``unmasked_root``; instead the VLM verifier re-detects genuinely-missing bounding
        # surfaces (walls / floor) from the image and re-adds them (as name-only roots).
        if mask is None:
            if record_drop:
                dropped.append(
                    {
                        "category": cat,
                        "instance": pos,
                        "description": desc,
                        "reason": "root_unsegmented"
                        if kind == "root_surface"
                        else "unsegmented",
                        # grounding evidence for post-hoc debugging: [] means Molmo
                        # could not point at the description at all
                        "molmo_points": [list(pt) for pt in pts],
                    }
                )
            return False
        # Connected-component cleanup: a single object is one connected region, so keep ONLY
        # the component(s) under this object's own Molmo point and drop the rest — a swallowed
        # neighbour (power-bank mask grabbing the key fob) OR a stray fragment with no point
        # (the charging plug fused onto the power bank). The <3-point guard misses both.
        if kind != "root_surface" and pts:
            pre = _binarize(mask)
            kept_b = keep_own_components(mask, pts)
            removed = pre & ~kept_b
            if removed.any():
                trimmed[(cat, pos)] = removed  # candidate for the restore post-pass
            mask = kept_b.astype(np.uint8) * 255
        mb = mask > 0
        deferred_partner: Optional["InstanceMask"] = None
        # Same-object overlap (high IoU OR part-vs-whole containment). If this mask is
        # not larger than the existing one it's a redundant duplicate -> drop (it IS
        # represented by the kept mask). We RECORD the collapse rather than dropping it
        # silently: this is also the path where a DISTINCT object mis-grounded onto a
        # neighbour gets absorbed (the glasses-case-as-container bug), and a silent
        # ``continue`` made those losses invisible (``dropped`` stayed empty).
        di = _overlap_idx(mb, kind)
        if di is not None and kind != "root_surface":
            # Guard: only collapse the overlap if the fuller mask is a single object
            # (spans <3 object points). C should already prevent >=3-point masks; this
            # keeps the dedup from absorbing a distinct neighbour swallowed by a blob.
            fuller = mb if int(mb.sum()) >= int(claimed[di].sum()) else claimed[di]
            if _obj_points_in(fuller, only=count_only) >= 3:
                di = None
        if (
            di is not None
            and kind != "root_surface"
            and allow_supersede
            and defer_dedup
        ):
            c = claimed[di]
            inter = int((mb & c).sum())
            iou = inter / max(int(mb.sum()) + int(c.sum()) - inter, 1)
            if iou <= dedup_iou and _norm_cat(cat) != _norm_cat(results[di].category):
                deferred_partner = results[di]
                di = None
        if di is not None and (
            not allow_supersede or int(mb.sum()) <= int(claimed[di].sum())
        ):
            kept = results[di]
            if record_drop:
                dropped.append(
                    {
                        "category": cat,
                        "instance": pos,
                        "description": desc,
                        "reason": f"collapsed_into {kept.category}#{kept.instance}",
                    }
                )
            return False
        if point is None:  # surface / no point -> centroid
            ys, xs = np.where(mb)
            point = (
                [float(xs.mean()) / mask.shape[1], float(ys.mean()) / mask.shape[0]]
                if xs.size
                else [0.5, 0.5]
            )
        np.save(mp, mask)
        rec = InstanceMask(
            category=cat,
            instance=pos,
            kind=kind,
            support=inst.get("support"),
            description=desc,
            point=[float(point[0]), float(point[1])],
            score=None,
            # Mask records start unlocked.  Shared-size groups are resolved only after
            # segmentation/recount/merge/pruning has produced a stable inventory.
            same_size=False,
        )
        rec.mask_path = str(mp)
        rec.tier = tier
        if deferred_partner is not None:
            pid = f"{deferred_partner.category}#{deferred_partner.instance}"
            rec.overlap_with = [pid]
            deferred_partner.overlap_with = (deferred_partner.overlap_with or []) + [
                f"{cat}#{pos}"
            ]
            print(
                f"[masking] {cat}#{pos} overlaps {pid} (contained pair) — dedup "
                "DEFERRED to merge_objects; carve_overlaps separates any remainder"
            )
        ov = masks_dir / f"{slugify(cat)}_{pos}_overlay.png"
        make_overlay(image_path, mask, str(ov))
        rec.overlay_path = str(ov)
        if moge_points is not None:
            rec.point_cam, rec.depth = object_point_from_mask(mask, moge_points)
        if di is not None:  # fuller mask of the same object ->
            victim = results[di]
            vic_id = f"{victim.category}#{victim.instance}"
            new_id = f"{cat}#{pos}"
            dropped.append(
                {
                    "category": victim.category,
                    "instance": victim.instance,
                    "description": victim.description,
                    "reason": f"superseded_by {new_id}",
                }
            )
            print(
                f"[masking] {new_id} superseded {vic_id} (fuller mask of the same "
                f"object); support pointers healed"
            )
            if (rec.support or "") == vic_id:
                rec.support = victim.support  # inherit the victim's parent
            for r in results:
                if r is not victim and (r.support or "") == vic_id:
                    r.support = new_id
            for o2 in objects:
                for it2 in o2.get("instances", []):
                    if (it2.get("support") or "") == vic_id:
                        it2["support"] = victim.support if it2 is inst else new_id
            claimed[di] = mb  # supersede the smaller claimed one
            results[di] = rec
        else:
            claimed.append(mb)
            results.append(rec)
        return True

    def _split_singleton_group(obj: dict, cat: str, insts: list) -> bool:
        ""
        di = next(
            (
                i
                for i, r in enumerate(results)
                if r.category == cat and r.instance == 0 and r.kind == "object"
            ),
            None,
        )
        if di is None:
            return False
        text_masks, used_text = cat_state.get(cat, (None, set()))
        if text_masks is None or not len(text_masks):
            return False
        own = claimed[di]
        own_px = int(own.sum())
        if own_px == 0:
            return False
        members = []
        for i in _sam3_instance_indices(text_masks):
            m = text_masks[i] > 0
            mpx = int(m.sum())
            if mpx < GROUP_SPLIT_MIN_FRAC * own_px:
                continue
            if mpx > (1.0 - GROUP_SPLIT_MIN_FRAC) * own_px:
                continue  # the group mask itself (or a near-copy) is not a member
            if int((m & own).sum()) < GROUP_SPLIT_CONTAIN * mpx:
                continue
            members.append(i)
        if len(members) < 2:
            return False
        cover = np.zeros_like(own)
        for i in members:
            cover |= text_masks[i] > 0
        if int((cover & own).sum()) < GROUP_SPLIT_COVER * own_px:
            return False
        plural = list(
            molmo.point(image_path, f"all the {cat}s")
            or molmo.point(image_path, f"{cat}s")
            or []
        )
        witnessed = [
            i
            for i in members
            if any(point_in_mask(text_masks[i], pu, pv) for pu, pv in plural)
        ]
        if len(witnessed) < 2:
            return False
        rec = results[di]
        pt = obj_point.get((cat, 0)) or rec.point
        keeper = next(
            (i for i in witnessed if pt and point_in_mask(text_masks[i], pt[0], pt[1])),
            None,
        )
        if keeper is None:
            keeper = max(witnessed, key=lambda i: int((text_masks[i] > 0).sum()))
        mb = text_masks[keeper] > 0
        mask = mb.astype(np.uint8) * 255
        if rec.mask_path:
            np.save(rec.mask_path, mask)
        if rec.overlay_path:
            make_overlay(image_path, mask, rec.overlay_path)
        if moge_points is not None:
            rec.point_cam, rec.depth = object_point_from_mask(mask, moge_points)
        claimed[di] = mb
        # Retire the GROUP mask(s) from the stack in place (the array object is what
        # cat_state holds): the recount's orphan grounding would otherwise re-select the
        # union — every member's plural point lies inside it — and be rejected as a
        # duplicate of the keeper. Members other than the keeper become unclaimed.
        for g in range(len(text_masks)):
            if g in members:
                continue
            if _mask_iou(text_masks[g], own) > 0.5:
                text_masks[g] = 0
        used_text.difference_update(m for m in members if m != keeper)
        used_text.add(keeper)
        cat_state[cat] = (text_masks, used_text)
        print(
            f"[masking] recount group-split: {cat}#0 was the union of {len(members)} "
            f"SAM3 instances ({len(witnessed)} Molmo-witnessed); kept member {keeper}, "
            f"released the rest as orphans"
        )
        return True

    def _recount() -> list["InstanceMask"]:
        """Fallback for proposer UNDER-count of a repeated category. For each category the
        proposer listed with >=2 instances, count SAM3 masks over the UNION of the
        category prompt and its proposer synonyms (novel masks only — the category
        word alone under-segments when an instance answers to a synonym: abc4's
        handle-less tumbler is a "cup", never a "mug"); when the union holds any
        distinct UNCLAIMED mask (without comparing with the proposed count), retry
        those ORPHANED masks (ones no
        long-description Molmo point claimed in the main pass). An orphan becomes a
        new instance ONLY if a category-or-synonym-level Molmo 'all the {cat}s'
        point lands inside it (the SAM3+Molmo agreement rule) AND it does not
        overlap an already-claimed OBJECT mask (dedup + cross-category guard: the
        gpt1 notebook that SAM3 masks as a 'book' is rejected here). Global
        post-pass: runs after every category's tier-1 + salvage, so the overlap
        guard sees all claimed masks regardless of processing order. Appends recs
        tagged source='sam3_recount'. Logs one gate line per evaluated category."""
        added: list[InstanceMask] = []
        proposed_cats = {_norm_cat(o["category"]) for o in objects}
        for obj in objects:
            cat = obj["category"]
            insts = obj["instances"]
            if any(it.get("kind") == "root_surface" for it in insts):
                continue
            if len(insts) < 2:
                # Singleton categories recount ONLY through the group-split: when the
                # one proposed instance's mask is the union of >=2 distinct SAM3
                # instances that Molmo also sees as plural, the singleton keeps one
                # member and the others become orphans for the ordinary acceptance
                # below (plural point inside + no overlap with a claimed object).
                if not _split_singleton_group(obj, cat, insts):
                    continue
            text_masks, used_text = cat_state.get(cat, (None, set()))
            if text_masks is None or not len(text_masks):
                print(
                    f"[masking] recount gate: {cat} proposed={len(insts)} "
                    f"sam3=0 -> off (no SAM3 masks)"
                )
                continue
            base_kept = len(_sam3_instance_indices(text_masks))
            # SYNONYM UNION: the proposer's category word can under-segment (abc4:
            # a handle-less tumbler answers to "cup", not "mug", so SAM3("mug")
            # agreed with the under-proposal and the gate stayed shut). Extend the
            # stack with each synonym's segment_all masks, keeping only masks NOVEL
            # vs the stack (IoU <= dedup threshold). Appending at the END preserves
            # ``used_text`` index validity (every claimed index points into the
            # original prefix). A synonym naming another PROPOSED category is
            # skipped — its instances are already claimed elsewhere and would only
            # inflate the gate count.
            syn_note = ""
            used_syns: list[str] = []
            for syn in obj.get("synonyms") or []:
                if not syn or _norm_cat(syn) in proposed_cats:
                    continue
                sm = _syn_stack(cat, syn)  # shared cache: grounds once per (cat, syn)
                if sm is None:
                    continue
                novel = [
                    m for m in sm
                    if not any(_mask_iou(m, t) > dedup_iou for t in text_masks)
                ]  # fmt: skip
                if novel:
                    text_masks = np.concatenate([text_masks, np.stack(novel)])
                    used_syns.append(syn)
                syn_note += f" +{syn}:{len(novel)}novel"
            sam3_kept = _sam3_instance_indices(text_masks)
            # Gate on UNCLAIMED masks, not on the proposed count. When two instances ground onto
            # the SAME object the duplicate is dropped (collapsed_into / superseded_by)
            # and the real second object is left as an orphan while the counts TIE — so a
            # proposed-count gate shut on exactly the scenes that needed recovery
            # The recovery below was ALWAYS orphan-based;
            # only the gate asked a proposal question. Acceptance is unchanged and already
            # strict (a plural Molmo point must land INSIDE the mask, and it must not
            # duplicate or overlap any claimed object), so the gate only decides whether
            # we PAY for the plural point.
            orphans = [i for i in sam3_kept if i not in used_text]
            fire = bool(orphans)
            print(
                f"[masking] recount gate: {cat} proposed={len(insts)} "
                f"sam3={base_kept}{syn_note} union={len(sam3_kept)} "
                f"unclaimed={len(orphans)} -> {'FIRE' if fire else 'off'}"
            )
            if not fire:  # gate: every distinct SAM3 mask is already claimed
                continue
            # Corroboration points from the category AND every synonym that
            # contributed novel masks — Molmo asked for "all the mugs" may not
            # point at a tumbler that only answers to "cup". Points are pure
            # existence witnesses per orphan MASK; duplicates across queries are
            # harmless (dedup lives at the mask level: stack novelty above,
            # _dup/_overlap_idx below).
            plural = list(
                molmo.point(image_path, f"all the {cat}s")
                or molmo.point(image_path, f"{cat}s")
                or []
            )
            for syn in used_syns:
                plural += list(
                    molmo.point(image_path, f"all the {syn}s")
                    or molmo.point(image_path, f"{syn}s")
                    or []
                )
            if not plural:
                continue
            support = next(
                (it.get("support") for it in insts if it.get("support")), None
            )
            for i in orphans:
                mb = text_masks[i] > 0
                hit = next(
                    (
                        [pu, pv]
                        for pu, pv in plural
                        if point_in_mask(text_masks[i], pu, pv)
                    ),  # fmt: skip
                    None,
                )
                if hit is None:
                    continue  # SAM3 mask with no Molmo point -> not an instance
                if _dup(mb) or _overlap_idx(mb, "object") is not None:
                    continue  # already represented (dedup / another category, e.g. notebook)
                pos = len(insts)  # appended index -> stable id
                new_inst = {
                    "description": f"{cat} (recount #{pos})",
                    "kind": "object",
                    "support": support,
                    "form": None,
                }
                insts.append(new_inst)
                inst_points[(cat, pos)] = [hit]
                obj_point[(cat, pos)] = hit
                if _mask_one(
                    obj,
                    cat,
                    pos,
                    new_inst,
                    text_masks,
                    used_text,
                    record_drop=False,
                    allow_supersede=False,
                ):
                    rec = next(
                        (
                            r
                            for r in results
                            if r.category == cat and r.instance == pos
                        ),  # fmt: skip
                        None,
                    )
                    if rec is not None:
                        rec.source = "sam3_recount"
                        added.append(rec)
                        print(
                            f"[masking] recount: appended {cat}#{pos} "
                            f"(orphan SAM3 mask + Molmo '{cat}s' point)"
                        )
                        continue
                # masking failed -> undo the tentative slot
                insts.pop()
                inst_points.pop((cat, pos), None)
                obj_point.pop((cat, pos), None)
        return added

    for obj in objects:
        cat = obj["category"]
        # Always run the text pass — we now attempt to mask every category, roots included.
        allp = str(masks_dir / f"{slugify(cat)}_all.npy")
        text_masks, _ = sam3.segment_all(image_path, cat, allp)
        cat_is_root = any(it.get("kind") == "root_surface" for it in obj["instances"])
        if text_masks is None or len(text_masks) == 0:
            ladder = [s for s in (obj.get("synonyms") or []) if s]
            if cat_is_root and _is_support_head(cat):
                ladder += [s for s in _SURFACE_SYNONYMS if s not in ladder]
            for syn in ladder:
                tm, _ = sam3.segment_all(image_path, syn, allp)
                if tm is not None and len(tm):
                    text_masks = tm
                    break
        used_text: set[int] = set()
        for pos, inst in enumerate(
            obj["instances"]
        ):  # index == VLM position -> stable id
            _mask_one(obj, cat, pos, inst, text_masks, used_text)
        cat_state[cat] = (text_masks, used_text)

    def _pt_yx(pt):
        return min(int(pt[1] * H), H - 1), min(int(pt[0] * W), W - 1)

    suspects = []
    for d in dropped:
        if not salvage_eligible(d.get("reason")):
            continue
        k = (d["category"], d["instance"])
        p0 = (inst_points.get(k) or [None])[0]
        if p0 is None:
            print(
                f"[masking] salvage skipped {k[0]}#{k[1]}: Molmo returned no point "
                "for its description"
            )
            continue
        y, x = _pt_yx(p0)
        hit = any(c[y, x] for c in claimed)
        if not hit:
            allp = masks_dir / f"{slugify(d['category'])}_all.npy"
            if allp.exists():
                cands = np.load(allp)
                hit = any(
                    point_in_mask(cands[j] > 0, p0[0], p0[1]) for j in range(len(cands))
                )
        if hit:
            suspects.append(d)
    for d in suspects:
        cat, pos = d["category"], d["instance"]
        obj = next((o for o in objects if o["category"] == cat), None)
        if obj is None or pos >= len(obj["instances"]):
            continue
        inst = obj["instances"][pos]
        same_cat = [
            _binarize(np.load(r.mask_path))
            for r in results
            if r.category == cat and r.mask_path and os.path.exists(r.mask_path)
        ]
        probe = image_path
        if same_cat:
            probe = str(masks_dir / f"{slugify(cat)}_{pos}_salvage.png")
            _cross_out(image_path, same_cat, probe)
        newpts = molmo.point(probe, f"{inst['description']} (NOT a crossed-out one)")
        if not newpts:
            print(
                f"[masking] salvage gave up on {cat}#{pos}: re-point returned nothing"
            )
            continue
        y, x = _pt_yx(newpts[0])
        if any(c[y, x] for c in claimed):
            # fresh point still on a claimed object -> genuinely absent
            print(
                f"[masking] salvage gave up on {cat}#{pos}: fresh point still on a "
                "claimed mask"
            )
            continue
        inst_points[(cat, pos)] = newpts
        obj_point[(cat, pos)] = newpts[0]
        tm, ut = cat_state.get(cat, (None, set()))
        settled = {(r.category, r.instance) for r in results}
        if _mask_one(
            obj,
            cat,
            pos,
            inst,
            tm,
            ut,
            count_only=settled | {(cat, pos)},
            record_drop=False,
        ):
            dropped.remove(d)
            print(f"[masking] salvage recovered {cat}#{pos} (Molmo point collapse)")

    # Under-count fallback: append SAM3-detected + Molmo-corroborated instances the
    # proposer missed on repeated categories (runs last so its guards see every claim).
    _recount()
    # Give back occluder-severed components the per-instance trim removed blind (it
    # runs before the occluder is masked); criteria + provenance in
    # restore_severed_components. Salvage/recount ran, so every claim is final here.
    restore_severed_components(results, trimmed, inst_points, image_path=image_path)
    return results, dropped, unmasked_roots


def main() -> None:
    p = argparse.ArgumentParser(description="MolmoPoint+SAM3 per-instance masking")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--points-npy", default=None)
    p.add_argument("--objects", default=None, help="comma-separated; else VLM proposes")
    args = p.parse_args()
    objs = [o.strip() for o in args.objects.split(",")] if args.objects else None
    res = segment_scene(
        args.image,
        args.out_dir,
        objects=objs,
        model=args.model,
        points_npy=args.points_npy,
    )
    print(f"{len(res)} instances across {len(set(r.category for r in res))} categories")
    for r in res:
        d = f" depth={r.depth:.2f}" if r.depth else ""
        print(f"  {r.category} #{r.instance} score={r.score} pt={r.point}{d}")


if __name__ == "__main__":
    main()
