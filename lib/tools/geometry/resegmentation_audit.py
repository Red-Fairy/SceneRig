"""Before/after auditing of a candidate reconstruction image and instance mask.

The auditor never edits its source artifacts. A candidate is promoted only when
every applicable check passes; malformed/unavailable judgments are unverified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from lib.tools.geometry.agentic_mask import _img_part, vlm_json

# ML Rules override: this standalone project's environment does not include ursalog.
logger = logging.getLogger(__name__)

AUDITOR_VERSION = "resegmentation-quality-v3"
CHECKS = (
    "target_identity_and_count",
    "observed_geometry_preserved",
    "completion_plausible",
    "mask_fidelity",
    "relevant_occluder_removed",
)
MAX_TOKENS = 4000

AUDIT_SYSTEM = """You audit a candidate image-and-instance-mask pair for single-image
3D object reconstruction. Judge whether it is usable for the SAME target instance.
This is a BEFORE/AFTER comparison, not merely a duplicate-object count.

The four ALIGNED panels are: original RGB, candidate RGB, original masked cut-out,
candidate masked cut-out on gray. The cut-out is what reconstruction receives.
Cyan outlines in the original RGB mark the edit region when available; the other
panels have no tint. Blue-gray hatched pixels mean OUTSIDE THE ORIGINAL FRAME,
not observed background. The original mask is a reference, NOT ground truth:
correcting its segmentation errors is allowed. Judge visible evidence in RGB.

Evaluate exactly the REQUIRED CHECKS supplied by the caller:
- target_identity_and_count: the candidate contains the same target and no extra
  instance merged into it. Other categories do not count as target duplicates,
  but DO count as contamination under mask_fidelity. Parts of one object need
  not be connected. Do not demand closed separate silhouettes to notice duplicates.
- observed_geometry_preserved: originally visible target geometry, pose and major
  proportions survive. Reject a visible broad barrel becoming a string, disappearing
  observed parts, or major deformation. Ignore small resampling/lighting/texture
  changes and legitimate correction of erroneous original mask pixels.
  Assess image-edit damage in RGB; a mask omission is judged separately below.
  Distinguish the object's own structure from a SEPARATE support beneath it:
  a keyboard on a flat pad/tray does not need that support included in its mask.
  Mere contact or adjacency does not make a support part of the target.
- completion_plausible: recovered regions plausibly continue this object and
  visibly address the requested recovery. Reject grossly distorted or disconnected
  invented parts or obvious failure to recover the needed part. Hidden geometry has
  NO ground truth here: plausible continuation is a pass, not uncertain merely
  because its exact unseen shape cannot be known. Notches, blur and texture changes
  alone are normal edit artifacts, not a reason to fail.
- mask_fidelity: the candidate cut-out isolates the target's visible body in the
  candidate image. Reject substantial table/background, substantial unrelated-object pixels,
  or omission of substantial visible target parts. This is reconstruction-input
  usability, NOT pixel-perfect segmentation. Modest holes, notches, ragged edges,
  and small gaps between correctly identified parts are tolerable when the main
  object's shape remains clear and usable. In particular a recognizable recovered
  stand/base need not have a solid flawless mask: scattered holes and a rough base
  edge alone are a PASS, not fail or uncertain. Missing a major body region or
  merging a large background patch is still a failure. Judge scale relative to
  the target, not whether ANY incorrect pixel or missing small accessory exists.
  A compact missing corner is tolerable when the dominant body and recovered
  structure remain recognizable; do not promote such a local gap to a major-body
  failure merely because it includes part of a bezel or edge. Likewise modest
  edge-adjacent material (e.g. a thin clear cover/lip alongside a recovered book)
  need not invalidate an otherwise usable, well-isolated main body. Distinguish
  these local imperfections from losing a long body segment or including a large
  patch whose boundary has no corresponding object boundary in candidate RGB.
  Likewise a small amount of background retained inside a handle/opening is a
  tolerable interior-mask error when the object's main shape and handle remain
  recognizable in RGB. Do not equate it with swallowing a supporting surface.
- relevant_occluder_removed: judge whether remaining requested blockers materially
  defeat this target's useful recovery, not whether EVERY requested deletion is
  perfect. Substantial useful recovery may PASS with a small residual occlusion
  at an edge; this also applies when border completion succeeds but a minor blocker
  remains. Fail when a large needed body region remains hidden or the blocker is
  substantially included in the cut-out. Do not reject for unrelated image changes
  elsewhere, or because a removed blocker revealed background beside the target
  when that background is correctly excluded from the candidate mask.

There is NO maximum legitimate area-growth ratio: heavily occluded books, trays,
stands and border-cut objects can grow greatly. A high visible fraction also does
not rule out a genuinely missing structural part. For border completion compare
preservation only inside the observed frame; do not reject new out-of-frame parts
for lacking original evidence. Useful PARTIAL border recovery is acceptable: the
object can still extend beyond the expanded image, and a different edge can remain
cropped. Do not require every side to be fully completed. Assess whether recovered
parts are plausible and preserve a usable object, not whether the full object now
fits inside the frame. Do not assert true 3D accuracy from these 2D panels.
Partial recovery still requires actual recovery: a blank/gray added border strip
with no new target geometry is NOT a successful partial border completion.

Return strict JSON with exactly checks, issues, summary. checks maps EACH required
check to pass, fail, or uncertain; omit inapplicable checks. issues is a list of
{check, evidence}, with a short localized visual explanation for EVERY failed or
uncertain check (no issue is needed for a pass). summary is one short sentence.
Use uncertain only when these images cannot resolve an applicable quality check,
not as a substitute for judging plausible hidden geometry. Do not output an overall
accept flag: the caller derives that from your individual findings."""


class AuditIssue(BaseModel):
    """Localized evidence for a failed or unresolved quality check."""

    model_config = ConfigDict(extra="forbid", strict=True)
    check: Literal[
        "target_identity_and_count",
        "observed_geometry_preserved",
        "completion_plausible",
        "mask_fidelity",
        "relevant_occluder_removed",
    ]
    evidence: str = Field(min_length=1)


class AuditReply(BaseModel):
    """Strict transport schema; applicability is validated by the caller."""

    model_config = ConfigDict(extra="forbid", strict=True)
    checks: dict[str, Literal["pass", "fail", "uncertain"]]
    issues: list[AuditIssue]
    summary: str = Field(min_length=1)


def required_checks(mode: str, removed: list[str]) -> list[str]:
    """Return the checks applicable to an edit, border completion, or mask-only run."""
    if mode not in {"removal", "border", "resegment_only"}:
        raise ValueError(f"Unknown resegmentation audit {mode=}")
    checks = [CHECKS[0], CHECKS[3]]
    if mode != "resegment_only":
        checks.extend((CHECKS[1], CHECKS[2]))
    if removed:
        checks.append(CHECKS[4])
    return [check for check in CHECKS if check in checks]


def validate_reply(reply: dict, required: list[str]) -> dict:
    """Reject missing/extra checks and unexplained failures before applying policy."""
    parsed = AuditReply.model_validate(reply)
    if set(parsed.checks) != set(required):
        raise ValueError(
            f"Expected audit checks {required!r}, got {list(parsed.checks)!r}"
        )
    explained = {issue.check for issue in parsed.issues}
    for issue in parsed.issues:
        if parsed.checks.get(issue.check) not in {"fail", "uncertain"}:
            raise ValueError(
                f"Issue contradicts or names inapplicable check {issue.check!r}"
            )
    for check, status in parsed.checks.items():
        if status != "pass" and check not in explained:
            raise ValueError(f"Missing visual evidence for {check!r}: {status!r}")
    return parsed.model_dump()


def resolve_scene_asset(scene: Path, stored: str) -> Path:
    """Prefer staged scene-local files over absolute paths pointing at a legacy run."""
    path = Path(stored)
    if "/scene/" in path.as_posix():
        local = scene / path.as_posix().rsplit("/scene/", 1)[1]
        if local.is_file():
            return local
    if path.is_file():
        return path
    for local in (
        scene / stored,
        scene / "masks" / path.name,
        scene / "masks/edited" / path.name,
    ):
        if local.is_file():
            return local
    raise FileNotFoundError(
        f"Cannot resolve scene asset {stored!r} under {str(scene)!r}"
    )


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D instance mask, got {mask.shape=}")
    return (
        np.asarray(
            Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
                size, Image.Resampling.NEAREST
            )
        )
        > 0
    )


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_evidence(
    *,
    scene: Path,
    instance: dict,
    record: dict,
    out_dir: Path,
) -> dict:
    """Write aligned audit panels, preserving all source images, masks and records.

    Returns:
        JSON-serializable evidence metadata, input fingerprint, and panel path.
    """
    original_path = scene / "input.png"
    mask_path = resolve_scene_asset(scene, instance["mask_path"])
    edited_path = resolve_scene_asset(scene, record["edited_image"])
    candidate_path = resolve_scene_asset(scene, record["mask_path"])
    mode = (
        "border"
        if record.get("border")
        else "removal"
        if record.get("removed")
        else "resegment_only"
    )
    removed = list(record.get("removed") or [])
    original = Image.open(original_path).convert("RGB")
    edited = Image.open(edited_path).convert("RGB")
    original_mask = _resize_mask(np.load(mask_path, allow_pickle=False), original.size)
    candidate = _resize_mask(np.load(candidate_path, allow_pickle=False), edited.size)
    inputs = {
        "original_image": original_path,
        "original_mask": mask_path,
        "candidate_image": edited_path,
        "candidate_mask": candidate_path,
    }
    observed = np.ones((edited.height, edited.width), dtype=bool)
    edit_region = np.zeros_like(observed)
    if mode == "border":
        border = record["border"]
        width, height, xoff = (
            int(border["width"]),
            int(border["height"]),
            int(border["xoff"]),
        )
        if (
            (width, height) != edited.size
            or height != original.height
            or xoff < 0
            or xoff + original.width > width
        ):
            raise ValueError(
                f"Invalid border coordinate mapping: {border!r}, {original.size=}, {edited.size=}"
            )
        yy, xx = np.indices(observed.shape)
        before = np.empty((height, width, 3), dtype=np.uint8)
        before[:] = (66, 83, 102)
        before[(xx + yy) % 16 < 2] = (110, 129, 145)
        before[:, xoff : xoff + original.width] = np.asarray(original)
        aligned_mask = np.zeros_like(observed)
        aligned_mask[:, xoff : xoff + original.width] = original_mask
        observed[:] = False
        observed[:, xoff : xoff + original.width] = True
        edit_region = ~observed
    else:
        before = np.asarray(
            original.resize(edited.size, Image.Resampling.LANCZOS)
        ).copy()
        aligned_mask = _resize_mask(original_mask, edited.size)
    if record.get("hole_mask"):
        hole_path = resolve_scene_asset(scene, record["hole_mask"])
        inputs["edit_hole"] = hole_path
        hole = Image.open(hole_path).convert("RGBA").getchannel("A")
        edit_region |= (
            np.asarray(hole.resize(edited.size, Image.Resampling.NEAREST)) == 0
        )
    elif mode == "border":
        # The border depth hole includes an extra dilation; use the actual removal
        # edit mask for observed-frame evidence instead, when that artifact exists.
        slug = f"{instance['category']}{instance['instance']}".replace(" ", "_")
        removal_path = scene / "masks/edited" / f"{slug}_border_rm_mask.png"
        if removal_path.is_file():
            inputs["removal_edit_mask"] = removal_path
            removal = (
                np.asarray(
                    Image.open(removal_path)
                    .convert("L")
                    .resize(original.size, Image.Resampling.NEAREST)
                )
                == 0
            )
            edit_region[:, xoff : xoff + original.width] |= removal
    union = aligned_mask | candidate
    if not union.any() or not candidate.any():
        raise ValueError("Cannot audit an empty candidate mask")
    ys, xs = np.where(union)
    padding = max(16, int(max(np.ptp(xs), np.ptp(ys)) * 0.15))
    x0, x1 = (
        max(0, int(xs.min()) - padding),
        min(edited.width, int(xs.max()) + padding + 1),
    )
    y0, y1 = (
        max(0, int(ys.min()) - padding),
        min(edited.height, int(ys.max()) + padding + 1),
    )
    crop = np.s_[y0:y1, x0:x1]
    before_cut = np.where(aligned_mask[..., None], before, 128).astype(np.uint8)
    before_cut[~observed] = before[~observed]
    after = np.asarray(edited)
    after_cut = np.where(candidate[..., None], after, 128).astype(np.uint8)
    context = before.copy()
    edge = np.zeros_like(edit_region)
    edge[1:] |= edit_region[1:] != edit_region[:-1]
    edge[:, 1:] |= edit_region[:, 1:] != edit_region[:, :-1]
    context[edge & observed] = (0, 220, 240)
    panels = [
        Image.fromarray(arr[crop]) for arr in (context, after, before_cut, after_cut)
    ]
    scale = min(768 / panels[0].width, 768 / panels[0].height, 2.0)
    size = (
        max(1, round(panels[0].width * scale)),
        max(1, round(panels[0].height * scale)),
    )
    title_h = 30
    board = Image.new("RGB", (size[0] * 2, (size[1] + title_h) * 2), "#202020")
    draw = ImageDraw.Draw(board)
    labels = (
        "A: BEFORE RGB (cyan = edit boundary)",
        "B: CANDIDATE RGB",
        "C: BEFORE masked target",
        "D: CANDIDATE masked target",
    )
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        x, y = (index % 2) * size[0], (index // 2) * (size[1] + title_h)
        board.paste(panel.resize(size, Image.Resampling.LANCZOS), (x, y + title_h))
        draw.text((x + 5, y + 8), label, fill="white")
    metadata = {
        "auditor_version": AUDITOR_VERSION,
        "prompt_sha256": hashlib.sha256(AUDIT_SYSTEM.encode()).hexdigest(),
        "target": f"{instance['category']}#{instance['instance']}",
        "description": instance.get("description", ""),
        "mode": mode,
        "removed": removed,
        "requested_part": record.get("vital_part", ""),
        "border": record.get("border"),
        "required_checks": required_checks(mode, removed),
        "input_sha256": {key: _digest(path) for key, path in inputs.items()},
        "crop_xyxy": [x0, y0, x1, y1],
        "candidate_size": list(edited.size),
        "area_growth": float(candidate.sum() / max(aligned_mask.sum(), 1)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(metadata, sort_keys=True).encode()
    ).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_path = out_dir / "panel.png"
    board.save(panel_path)
    metadata.update(
        fingerprint=fingerprint,
        panel=str(panel_path),
        source_paths={key: str(path) for key, path in inputs.items()},
    )
    (out_dir / "evidence.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def audit_candidate(
    vlm: Callable,
    evidence: dict,
    *,
    model: str,
    effort: str = "medium",
) -> dict:
    """Audit prepared evidence once (bounded schema retries), retaining raw replies.

    Operational/model failures produce an explicit unverified result, never an
    accepted default. Programming errors in evidence construction remain visible.
    """
    context = {
        key: evidence[key]
        for key in (
            "target",
            "description",
            "mode",
            "removed",
            "requested_part",
            "required_checks",
            "border",
        )
    }
    parts = [
        {"type": "text", "text": json.dumps(context, indent=2)},
        _img_part(evidence["panel"]),
    ]
    raw_replies = []

    def recorded_vlm(system: str, user_parts: list, max_tokens: int) -> str:
        raw = vlm(system, user_parts, max_tokens=max_tokens)
        raw_replies.append(raw)
        return raw

    start = time.monotonic()
    error = None
    try:
        reply = vlm_json(
            recorded_vlm,
            AUDIT_SYSTEM,
            parts,
            max_tokens=MAX_TOKENS,
            label=f"resegmentation_audit {evidence['target']}",
            default=None,
            validate=lambda value: validate_reply(value, evidence["required_checks"]),
        )
    except Exception as exc:  # noqa: BLE001 - external API boundary; retain original on failure
        logger.exception(
            "Resegmentation auditor unavailable for %s", evidence["target"]
        )
        error = f"{type(exc).__name__}: {exc}"
        reply = None
    if reply is None:
        status = "unverified"
        error = error or "No schema-valid reply after three attempts"
    elif "fail" in reply["checks"].values():
        status = "rejected"
    elif "uncertain" in reply["checks"].values():
        status = "unverified"
    else:
        status = "accepted"
    return {
        "auditor_version": AUDITOR_VERSION,
        "fingerprint": evidence["fingerprint"],
        "model": model,
        "effort": effort,
        "max_tokens": MAX_TOKENS,
        "status": status,
        "accepted": status == "accepted",
        "verdict": reply,
        "error": error,
        "raw_replies": raw_replies,
        "attempts": len(raw_replies),
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "panel": evidence["panel"],
    }


def audit_resegmentation(
    *,
    vlm: Callable,
    scene: Path,
    instance: dict,
    record: dict,
    out_dir: Path,
    model: str,
    effort: str = "medium",
) -> dict:
    """Prepare, judge and persist one selected pipeline candidate without deleting it."""
    evidence = prepare_evidence(
        scene=scene, instance=instance, record=record, out_dir=out_dir
    )
    verdict = audit_candidate(vlm, evidence, model=model, effort=effort)
    (out_dir / "audit.json").write_text(json.dumps(verdict, indent=2) + "\n")
    return verdict


def audit_cache_matches(audit: dict, evidence: dict, model: str, effort: str) -> bool:
    """Match exact inputs and auditor settings; legacy records are not audited passes."""
    return (
        audit.get("auditor_version") == AUDITOR_VERSION
        and audit.get("fingerprint") == evidence["fingerprint"]
        and audit.get("model") == model
        and audit.get("effort") == effort
        and audit.get("max_tokens") == MAX_TOKENS
    )
