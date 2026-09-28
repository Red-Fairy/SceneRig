"""Prompt Builder for constructing agent prompts in the GRASE system.

This module provides utilities for building system and user prompts
for the Generator and Verifier agents, handling image encoding
and memory management.
"""

import json
import logging
import math
import os
import re
from typing import Any, Optional

from openai import OpenAI

from lib.prompts import get_system_prompt
from lib.prompts.static_scene.scopes import (
    initializer_object_repair_enabled,
    reconstruction_exclusions,
)
from lib.utils.common import AGENT_VLM_EDGE, display_path, get_image_base64

logger = logging.getLogger(__name__)

_PROMPT_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
_PROMPT_IMAGE_CAP = 8
_RUNTIME_ROOT_EDIT_CAP = 8
_RUNTIME_ROOT_TEXT_CAP = 320
_TARGET_ANCHOR_MIN_LENGTH_FRAME_DIAG = 0.015
_TARGET_ANCHOR_FRAME_EPSILON = 0.012
_TARGET_ANCHOR_CLUSTER_EPSILON = 0.02
_TARGET_ANCHOR_REJECT_REASONS = frozenset(
    {
        "boundary_support_low",
        "line_fit_residual_high",
        "occluder_contaminated",
        "top_face_support_low",
    }
)


def _main_support_target_mask_anchors(
    observation: Any,
) -> list[dict[str, Any]]:
    """Select bounded, deterministic 2-D anchors from source top-mask edges.

    ``main_support_yaw_observation.json`` already contains source-bound normalized
    endpoints fitted to the canonical top-face mask boundary.  Reusing those points
    keeps verifier grounding in the exact preprocessing evidence without introducing a
    second contour extractor at prompt time.  Frame-touching and short segments are
    intentionally usable here: they are weak for world-yaw inference, but often carry
    the most useful finite target-mask crossings.
    """
    if not isinstance(observation, dict):
        return []
    endpoints: list[tuple[float, float]] = []
    for segment in observation.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        try:
            length = float(segment.get("length_frame_diag"))
            boundary_support = float(segment.get("boundary_support_fraction"))
            top_support = float(segment.get("top_face_support_fraction"))
        except (TypeError, ValueError):
            continue
        reasons = set(segment.get("reason_codes") or [])
        if (
            not all(
                math.isfinite(value)
                for value in (length, boundary_support, top_support)
            )
            or length < _TARGET_ANCHOR_MIN_LENGTH_FRAME_DIAG
            or boundary_support < 0.8
            or top_support < 0.8
            or reasons & _TARGET_ANCHOR_REJECT_REASONS
        ):
            continue
        raw_endpoints = segment.get("endpoints_normalized")
        if not isinstance(raw_endpoints, list) or len(raw_endpoints) != 2:
            continue
        for raw in raw_endpoints:
            if not isinstance(raw, list) or len(raw) != 2:
                continue
            try:
                x, y = float(raw[0]), float(raw[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                continue
            # The validated edge artifact allows a tiny fitted-line overshoot.  The
            # verifier contract uses image coordinates, where the frame is [0, 1].
            endpoints.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y))))
    if not endpoints:
        return []

    frame_epsilon = _TARGET_ANCHOR_FRAME_EPSILON
    anchors: list[dict[str, Any]] = []
    interior = [
        point
        for point in endpoints
        if min(point[0], 1.0 - point[0], point[1], 1.0 - point[1]) > frame_epsilon
    ]
    if interior:
        x, y = min(interior, key=lambda point: (point[1], point[0]))
        anchors.append(
            {
                "anchor_id": "topmost_visible_boundary_point",
                "point_normalized": [round(x, 3), round(y, 3)],
            }
        )

    side_values: dict[str, list[float]] = {
        "left": [],
        "right": [],
        "top": [],
        "bottom": [],
    }
    for x, y in endpoints:
        if x <= frame_epsilon:
            side_values["left"].append(y)
        if x >= 1.0 - frame_epsilon:
            side_values["right"].append(y)
        if y <= frame_epsilon:
            side_values["top"].append(x)
        if y >= 1.0 - frame_epsilon:
            side_values["bottom"].append(x)

    def clustered(values: list[float]) -> list[float]:
        groups: list[list[float]] = []
        for value in sorted(values):
            if not groups or value - groups[-1][-1] > _TARGET_ANCHOR_CLUSTER_EPSILON:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [sum(group) / len(group) for group in groups]

    side_specs = (
        ("left", "upper", "lower"),
        ("right", "upper", "lower"),
        ("top", "left", "right"),
        ("bottom", "left", "right"),
    )
    for side, first_name, last_name in side_specs:
        values = clustered(side_values[side])
        if not values:
            continue
        selected = [("", values[0])]
        if len(values) > 1:
            selected = [(first_name, values[0]), (last_name, values[-1])]
        for qualifier, value in selected:
            anchor_id = (
                f"{side}_{qualifier}_frame_crossing"
                if qualifier
                else f"{side}_frame_crossing"
            )
            point = [0.0, value] if side == "left" else [1.0, value]
            if side == "top":
                point = [value, 0.0]
            elif side == "bottom":
                point = [value, 1.0]
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "point_normalized": [round(point[0], 3), round(point[1], 3)],
                }
            )
    # One interior extremum plus at most two crossings on each of four frame sides.
    return anchors[:9]


def scene_graph_revision(graph: Any) -> int:
    ""
    if not isinstance(graph, dict):
        return 0
    raw = graph.get("scene_graph_revision", graph.get("graph_revision", 0))
    try:
        revision = int(raw)
    except (TypeError, ValueError):
        return 0
    return max(revision, 0)


def root_surface_build_name(node: Any) -> Optional[str]:
    ""
    if not isinstance(node, dict):
        return None
    explicit = node.get("build_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    graph_id = node.get("id")
    if not isinstance(graph_id, str) or not graph_id:
        return None
    # Mirrors surface_build_name without importing geometry code into result-only
    # consumers.
    return graph_id.replace("#", "_")


def bounded_runtime_root_surface_edits(
    graph: Any, *, limit: int = _RUNTIME_ROOT_EDIT_CAP
) -> list[dict[str, Any]]:
    """Return bounded, revision-bound metadata for active initializer-added roots.

    The scene graph is authoritative: removed roots disappear from this handoff, while
    every active runtime root remains visible after ``end`` replaces the last tool
    response. Mesh, plane, and mask payloads remain in ``scene_graph.json``.
    """
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        return []
    revision = scene_graph_revision(graph)

    def compact_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text[:_RUNTIME_ROOT_TEXT_CAP] if text else None

    def compact_region(value: Any) -> Optional[list[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            region = [round(float(component), 5) for component in value]
        except (TypeError, ValueError):
            return None
        if not all(-0.01 <= component <= 1.01 for component in region):
            return None
        return region

    rows: list[dict[str, Any]] = []
    for node in graph["nodes"]:
        if not isinstance(node, dict) or node.get("kind") != "root_surface":
            continue
        source = str(node.get("source") or "")
        if not (
            node.get("runtime_added") is True
            or source
            in {
                "initializer_build_root_surface",
                "initializer_runtime",
                "build_root_surface",
            }
        ):
            continue
        row: dict[str, Any] = {
            "action": "add",
            "id": node.get("id"),
            "build_name": root_surface_build_name(node),
            "surface_type": node.get("category") or node.get("surface_type"),
            "description": compact_text(node.get("description")),
            "reason": compact_text(
                node.get("runtime_reason")
                or node.get("addition_reason")
                or node.get("reason")
            ),
            "target_region_norm": compact_region(
                node.get("target_region_norm") or node.get("authorized_region_norm")
            ),
            "added_attempt": node.get("added_attempt", node.get("attempt_idx")),
            "source": source or "initializer_runtime",
            "scene_graph_revision": revision,
        }
        rows.append({key: value for key, value in row.items() if value is not None})

    rows.sort(
        key=lambda row: (
            str(row.get("id") or ""),
            str(row.get("build_name") or ""),
        )
    )
    return rows[: max(0, int(limit))]


def _resolve_prompt_images(
    path: str, preferred_names: tuple[str, ...], cap: int = _PROMPT_IMAGE_CAP
) -> list[str]:
    """Resolve a directory input into a deterministic, bounded image list.

    The first existing, decodable preferred image preserves the historical
    single-image behavior. Otherwise regular files are extension-filtered, decoded by
    PIL, sorted case-insensitively with an original-name tiebreak, and capped. The
    environment knob may lower the requested cap but never raise the safety ceiling.
    """
    from PIL import Image

    directory = os.fspath(path)
    hard_cap = min(max(int(cap), 1), _PROMPT_IMAGE_CAP)
    raw_env_cap = os.environ.get("GRASE_PROMPT_DIRECTORY_IMAGE_CAP")
    if raw_env_cap is not None:
        try:
            env_cap = int(raw_env_cap)
            if env_cap < 1:
                raise ValueError
            hard_cap = min(hard_cap, env_cap)
        except ValueError:
            logger.warning(
                "Ignoring invalid GRASE_PROMPT_DIRECTORY_IMAGE_CAP=%r; expected a "
                "positive integer",
                raw_env_cap,
            )

    def decodable(candidate: str) -> bool:
        if not os.path.isfile(candidate):
            return False
        if os.path.splitext(candidate)[1].lower() not in _PROMPT_IMAGE_EXTENSIONS:
            return False
        try:
            with Image.open(candidate) as image:
                image.verify()
            return True
        except Exception:  # noqa: BLE001 - invalid user attachment is skipped
            logger.warning("Skipping undecodable prompt image: %s", candidate)
            return False

    entries = os.listdir(directory)
    for preferred in preferred_names:
        candidate = os.path.join(directory, preferred)
        if preferred in entries and decodable(candidate):
            return [candidate]

    candidates = sorted(entries, key=lambda name: (name.casefold(), name))
    valid = [os.path.join(directory, name) for name in candidates]
    valid = [candidate for candidate in valid if decodable(candidate)]
    if not valid:
        allowed = ", ".join(sorted(_PROMPT_IMAGE_EXTENSIONS))
        raise ValueError(
            f"No valid prompt images found in {directory!r}; allowed formats: {allowed}"
        )
    if len(valid) > hard_cap:
        logger.warning(
            "Prompt image directory %s contains %d valid images; using the first %d "
            "and omitting %d",
            directory,
            len(valid),
            hard_cap,
            len(valid) - hard_cap,
        )
    return valid[:hard_cap]


# Feedback-string patterns emitted by lib/tools/blender/exec.py (investigate/move/exec).
# The object-state table parses them back out of memory; keep in sync with the emitters.
_INV_VISIT = re.compile(r"Investigated .*?\(visit ([^)]*)\)")
_BUDGET_SUFFIX = re.compile(r"(?<!\d)(\d+)\s*/\s*(\d+)\s*$")
_INV_IOU = re.compile(r"- (\S+): (?:score (-?\d+\.\d+), )?silhouette IoU (\d+\.\d+)")
_MOVE_DELTA = re.compile(
    r"moved (\S+) along (\w+): score (-?\d+\.\d+) -> (-?\d+\.\d+), IoU (\d+\.\d+) -> (\d+\.\d+)"
)
# The rotation variants of the move-failure texts read "fine-yaw candidate" /
# "did not clear the acceptance bar" / "a fine-yaw improvement" (exec.py's
# rotation branches); the alternations keep one pattern per outcome across all
# aspects.
_MOVE_REJECTED = re.compile(
    r"(\S+) (\w+): the best (?:fine-yaw )?candidate improved the render match "
    r"but physics REJECTED it"
)
_MOVE_NOOP = re.compile(
    r"(\S+) (\w+): (?:no candidate improved the match"
    r"|the best fine-yaw candidate did not clear the acceptance bar)"
    r" \(IoU stays (\d+\.\d+)\)"
)
_MOVE_BUDGET = re.compile(r"\(Move budget: (\d+)/(\d+) used on (\S+?)\.\)")
_MOVE_CAP_DEFAULT = 5
_SIZE_PENDING_OPEN = re.compile(
    r"SIZE HINT for (\S+?):|\(size for (\S+?): a mismatch also measured"
    r"|SIZE PENDING for (\S+?):"
)
_SIZE_REMEASURE_STILL = re.compile(r"Size re-measured at the new pose: still ~")
_SIZE_REMEASURE_MATCH = re.compile(
    r"Size re-measured at the new pose: matches the photo mask"
)
_MOVE_CAP_HIT = re.compile(
    r"move rejected: '(\S+)' has used its full move budget \((\d+)/(\d+)\)"
)
_MOVE_CLAMPED = re.compile(
    r"(\S+) (\w+): the search found (?:an|a fine-yaw) improvement but the move "
    r"was rejected pre-physics"
)
# Yaw-gate refusal (phase C1 extended the gate to ALL rotations with a paired
# reliable reading, so this is now a common rotation outcome): the search found
# gain but the re-measured yaw failed the improvement demand. The feedback routes
# to a placement fix first — the cell must say so, and the object stays workable
# (see _status), or the table would read "rotation never tried" forever.
_MOVE_YAW_REFUSED = re.compile(
    r"(\S+) (\w+): the search found a better silhouette match but it "
    r"FAILED THE YAW CHECK"
)
_FLIP_APPLIED = re.compile(r"yaw-rotated (\S+) 180 deg")
_FLIP_REJECTED = re.compile(r"rotate_180 on (\S+) was REJECTED")
# A flip transaction rolled back because its durable snapshot failed: the one-shot
# stays consumed (exec.py's rollback text) but no _FLIP_* pattern above matches, so
# the aspect cell read "-" and the cheapest rediscovery was burning a round on the
# ONE-SHOT refusal.
_FLIP_ROLLBACK = re.compile(r"rotate_180 on (\S+) was rolled back")
_FLIP_SPENT = re.compile(
    r"rotate_180 already used on (\S+)"
)  # the retry refusal itself
_SIZE_LOCKED = re.compile(r"move rejected: '(\S+)' has a LOCKED size")
# Objects named in a settle note. The note now speaks the scene-graph dialect the
# move/investigate tools accept (croissant#2); _SETTLE_OBJ still reads the Blender mesh
# name older transcripts carry (obj_fork_0 -> fork#0). Like _INV_IOU, a spaced category
# is recovered by its last word only (coffee machine#0 -> machine#0).
_SETTLE_OID = re.compile(r"\b([A-Za-z0-9][\w-]*)#(\d+)\b")
_SETTLE_OBJ = re.compile(
    r"obj_([A-Za-z0-9_]+?)_(\d+)\b"
)  # legacy: obj_fork_0 -> fork#0
# Decision-time evidence pairs on the settle note's per-object line (P1/P3): the
# yaw before->after and the appearance-agreement before->after. They exist to
# steer THAT round's keep/undo and are stale the moment the round is collapsed,
# so the ledger trim strips them from any surviving line. (Their sentence-cased
# move-feedback twins — " Measured yaw: ..." / " Appearance agreement ..." —
# already fall to _trim's leading-clause rule.) NOT parsed into the OBJECT STATE
# table: evidence, not state.
_EDIT_EVIDENCE_PAIRS = re.compile(
    r"; (?:measured yaw: ~-?\d+deg -> ~-?\d+deg vs the photo"
    r"|appearance agreement vs photo: -?\d+\.\d+ -> -?\d+\.\d+ \([^)]*\))"
)


COMP_BLOCK_K = 4

MAX_VIEW_TOKENS = 120_000

_W2_SCALAR_KINDS = {
    "Your scene rendered from view": "render",
    "Reference-view render": "refview",
    "ALTERNATE CAMERA": "alternate",
}
_W2_GT_DUP = "GROUND-TRUTH target photo for the source view"
_W2_PGT = "PSEUDO-GROUND-TRUTH"
_W2_PATH_CAPTION = "Image loaded from local path: "


def _est_view_tokens(view: list[dict[str, Any]]) -> int:
    """Rough token size of a view, for the append-only runaway guard only.

    Deliberately crude: images are capped at AGENT_VLM_EDGE (768) so they cost ~590
    tokens each, and text is ~4 chars/token. It only has to be good enough to bound a
    runaway trajectory, never to predict a bill — the ledger in usage.jsonl is the
    authority on real token counts.
    """
    n = 0
    for m in view:
        content = m.get("content")
        if isinstance(content, str):
            n += len(content) // 4
            continue
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                n += 590
            else:
                n += len(str(part.get("text", ""))) // 4
    return n


def age_window_images(
    view: list[dict[str, Any]],
    protected: int,
    frozen_from: Optional[int] = None,
) -> list[dict[str, Any]]:
    ""
    first_assistant = next(
        (i for i, m in enumerate(view) if m.get("role") == "assistant"), len(view)
    )
    start = max(protected, first_assistant)

    end = len(view) if frozen_from is None else min(frozen_from, len(view))
    occ: list[tuple[int, int, str]] = []  # (msg_idx, part_idx, kind)
    for mi in range(start, end):
        content = view[mi].get("content")
        if not isinstance(content, list):
            continue
        texts = [
            p.get("text", "")
            for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        is_pgt = any(t.startswith(_W2_PGT) for t in texts)
        caption = ""
        for pi, part in enumerate(content):
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                caption = part.get("text", "")
                continue
            if part.get("type") != "image_url":
                continue
            if is_pgt:
                kind = "pgt_pair"
            elif caption.startswith(_W2_GT_DUP):
                kind = "gt_dup"
            else:
                kind = next(
                    (
                        k
                        for pfx, k in _W2_SCALAR_KINDS.items()
                        if caption.startswith(pfx)
                    ),
                    "other",
                )
            occ.append((mi, pi, kind))
    if not occ:
        return view

    keep: set[tuple[int, int]] = set()
    for kind in _W2_SCALAR_KINDS.values():
        k_occ = [o for o in occ if o[2] == kind]
        if k_occ:
            keep.add(k_occ[-1][:2])  # newest occurrence of the kind
    for kind in ("pgt_pair", "other"):
        k_occ = [o for o in occ if o[2] == kind]
        if k_occ:
            newest_mi = k_occ[-1][0]  # newest message of the kind, atomically
            keep.update(o[:2] for o in k_occ if o[0] == newest_mi)

    targets: dict[int, dict[int, str]] = {}
    for mi, pi, kind in occ:
        if (mi, pi) not in keep:
            targets.setdefault(mi, {})[pi] = kind
    if not targets:
        return view

    out = list(view)
    for mi, stubs in targets.items():
        parts = view[mi]["content"]
        new_content: list[dict[str, Any]] = []
        i = 0
        while i < len(parts):
            if i in stubs:
                path, consumed = "", False
                nxt = parts[i + 1] if i + 1 < len(parts) else None
                if (
                    isinstance(nxt, dict)
                    and nxt.get("type") == "text"
                    and nxt.get("text", "").startswith(_W2_PATH_CAPTION)
                ):
                    path = nxt["text"][len(_W2_PATH_CAPTION) :]
                    consumed = True
                if stubs[i] == "gt_dup":
                    stub = (
                        "[superseded image removed: duplicate of the target photo "
                        "in the FIRST message of this conversation]"
                    )
                else:
                    stub = (
                        f"[superseded {stubs[i]} image removed"
                        + (f" — was {path}" if path else "")
                        + "; newer evidence of this kind appears later in the "
                        "conversation]"
                    )
                new_content.append({"type": "text", "text": stub})
                i += 2 if consumed else 1
                continue
            new_content.append(parts[i])
            i += 1
        out[mi] = {**view[mi], "content": new_content}
    return out


def _group_rounds(rest: list[dict]) -> list[list[dict]]:
    """Group tail messages into tool rounds: each round starts at an assistant message."""
    rounds: list[list[dict]] = []
    for m in rest:
        if m.get("role") == "assistant" or not rounds:
            rounds.append([m])
        else:
            rounds[-1].append(m)
    return rounds


def _round_texts(grp: list[dict]) -> list[str]:
    """All text parts from a round's tool + user messages (not the assistant turn)."""
    texts: list[str] = []
    for m in grp[1:]:
        if m.get("role") not in ("tool", "user"):
            continue
        content = m.get("content")
        if isinstance(content, str):
            texts.append(content)
        else:
            texts.extend(
                c.get("text", "")
                for c in (content if isinstance(content, list) else [])
                if isinstance(c, dict) and c.get("type") == "text"
            )
    return texts


def _round_tool_name(grp: list[dict]) -> Optional[str]:
    tcs = grp[0].get("tool_calls") or []
    return tcs[0].get("function", {}).get("name") if tcs else None


def build_object_state_table(memory: list[dict[str, Any]]) -> str:
    """Per-object rollup of the composition pose loop, parsed from the full memory.

    The chronological ledger scatters an object's history ("xy no-improve at r26,
    rotation physics-rejected at r28, budget hit at r36") across dozens of lines; the
    agent demonstrably fails to reassemble it and re-visits exhausted objects
    (0722_gatefix_abc2: fork#0 at r7/r25/r35, knife#0 five times). This table is the
    lookup that scattered history denies it: latest IoU/score, budgets used, the
    latest outcome per move aspect, and how many execute_and_evaluate edits touched
    the object (and how many were undone). Returns "" when nothing is parseable."""
    state: dict[str, dict[str, Any]] = {}

    def budget_counts(v: Optional[str]) -> Optional[tuple[int, int]]:
        """Read the terminal ordinary-visit budget from terse or annotated text.

        Post-flip verification may legally exceed the normal investigate cap, so its
        feedback annotates (rather than increments) the ordinary count, for example
        ``mandatory post-flip verification; normal visits remain 3/3``.  Only the
        terminal numeric fraction is authoritative; arbitrary descriptive text is not
        interpreted as a budget.
        """
        if not isinstance(v, str):
            return None
        match = _BUDGET_SUFFIX.search(v.strip())
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def obj(o: str) -> dict[str, Any]:
        return state.setdefault(
            o, {"iou": None, "score": None, "inv": None, "moves": None, "moves_n": 0,
                "aspects": {}, "exec": 0, "undone": 0, "iou_r": -1, "keep_r": -1,
                "size_pending": False}
        )  # fmt: skip

    rounds = _group_rounds(memory[2:])
    for ridx, grp in enumerate(rounds):
        texts = _round_texts(grp)
        touched: set[str] = set()
        tool = _round_tool_name(grp)
        is_exec = tool == "execute_and_evaluate"
        # moves share the undo stack with exec edits (_push_edit_snapshot("move")), so
        # an applied move followed by undo_last_step is reverted too — its "improved"
        # delta and IoU must not survive into the table (they described a state that
        # no longer exists).
        undone = (
            tool in ("execute_and_evaluate", "move")
            and ridx + 1 < len(rounds)
            and (_round_tool_name(rounds[ridx + 1]) == "undo_last_step")
        )
        if tool == "move":
            tcs = grp[0].get("tool_calls") or []
            try:
                margs = json.loads(tcs[0]["function"]["arguments"]) if tcs else {}
            except Exception:  # noqa: BLE001 - ledger parsing is best-effort
                margs = {}
            tgt = str(margs.get("object") or "")
            refused = any(
                "move rejected" in t
                or "rotate_180 already used" in t
                or "blocked: a retained rotate_180" in t
                or "one-shot and move budget were not consumed" in t
                for t in texts
            )
            if tgt and not refused:
                obj(tgt)["moves_n"] += 1
        for t in texts:
            for entry in _INV_VISIT.findall(t):
                for part in entry.split(", "):
                    name, _, used = part.rpartition(": ")
                    counts = budget_counts(used)
                    if name and counts is not None:
                        obj(name)["inv"] = f"{counts[0]}/{counts[1]}"
            for name, score, iou in _INV_IOU.findall(t):
                if undone:
                    continue  # post-settle IoU of an edit the undo then reverted
                s = obj(name)
                s["iou"], s["iou_r"] = iou, ridx
                if score:
                    s["score"] = score
            for name, aspect, sc0, sc1, iou0, iou1 in _MOVE_DELTA.findall(t):
                if undone:
                    continue  # the applied move was undone; its delta/IoU are gone
                s = obj(name)
                s["iou"], s["score"], s["iou_r"] = iou1, sc1, ridx
                s["aspects"][aspect] = f"improved {iou0}->{iou1}"
            for name, aspect in _MOVE_REJECTED.findall(t):
                obj(name)["aspects"][aspect] = "physics-REJECTED"
            for name, aspect, iou in _MOVE_NOOP.findall(t):
                s = obj(name)
                s["iou"], s["iou_r"] = iou, ridx
                s["aspects"][aspect] = "no-improve"
            for used, cap, name in _MOVE_BUDGET.findall(t):
                obj(name)["moves"] = f"{used}/{cap}"
            for name, used, cap in _MOVE_CAP_HIT.findall(t):
                obj(name)["moves"] = f"{used}/{cap}"
            for name, aspect in _MOVE_CLAMPED.findall(t):
                obj(name)["aspects"][aspect] = "BLOCKED pre-physics"
            for name, aspect in _MOVE_YAW_REFUSED.findall(t):
                obj(name)["aspects"][aspect] = "yaw-gate REFUSED (fix placement first)"
            for name in _FLIP_APPLIED.findall(t):
                obj(name)["aspects"]["rotate_180"] = "SPENT (applied)"
            for name in _FLIP_REJECTED.findall(t):
                # F6a refund: a CLEAN physics rejection does not consume the
                # one-shot — the feedback's "The one-shot was NOT consumed" clause
                # is the contract key (exec.py rotate_180 branch). Never SPENT.
                obj(name)["aspects"]["rotate_180"] = (
                    "physics-REJECTED (one-shot refunded)"
                    if "The one-shot was NOT consumed" in t
                    else "SPENT (rejected)"
                )
            for name in _FLIP_SPENT.findall(t):
                # the refusal arrives AFTER the outcome it refuses; setdefault keeps
                # the richer applied/rejected reading instead of downgrading it
                obj(name)["aspects"].setdefault("rotate_180", "SPENT")
            for name in _FLIP_ROLLBACK.findall(t):
                obj(name)["aspects"]["rotate_180"] = "SPENT (rollback)"
            for name in _SIZE_LOCKED.findall(t):
                obj(name)["aspects"]["scale"] = "LOCKED (size frozen)"
            # pending-size channel (see the _SIZE_PENDING_OPEN comment). Opens apply
            # unconditionally (investigate rounds are never undone; a stray open is
            # harmless — the next measurement settles it). Clears/re-opens ride on
            # MOVE outcomes and are skipped for undone rounds like every other
            # move-derived fact.
            for g1, g2, g3 in _SIZE_PENDING_OPEN.findall(t):
                obj(g1 or g2 or g3)["size_pending"] = True
            if not undone:
                for name, aspect, _s0, _s1, _i0, _i1 in _MOVE_DELTA.findall(t):
                    if aspect in ("rotation", "scale"):
                        obj(name)["size_pending"] = False
                    if _SIZE_REMEASURE_STILL.search(t):
                        obj(name)["size_pending"] = True
                    if _SIZE_REMEASURE_MATCH.search(t):
                        obj(name)["size_pending"] = False
                for name in _FLIP_APPLIED.findall(t):
                    obj(name)["size_pending"] = False
            if "Physics settle of your edit" in t:
                touched.update(f"{n}#{i}" for n, i in _SETTLE_OID.findall(t))
                touched.update(f"{n}#{i}" for n, i in _SETTLE_OBJ.findall(t))
        if is_exec and touched:
            for name in touched:
                s = obj(name)
                s["exec"] += 1
                s["undone"] += int(undone)
                if not undone:
                    s["keep_r"] = (
                        ridx  # a KEPT edit moved it (undone -> state reverted)
                    )

    if not state:
        return ""

    for s in state.values():
        if s["moves"] is None and s["moves_n"]:
            n = min(s["moves_n"], _MOVE_CAP_DEFAULT)
            s["moves"] = f"{n}/{_MOVE_CAP_DEFAULT}"

    def _budget(v: Optional[str]) -> str:
        if not v:
            return "-"
        counts = budget_counts(v)
        if counts is None:
            return v
        used, cap = counts
        normalized = f"{used}/{cap}"
        return f"{normalized} (CAP)" if used >= cap else normalized

    def _capped(v: Optional[str]) -> bool:
        counts = budget_counts(v)
        return counts is not None and counts[0] >= counts[1]

    done_bar = float(os.environ.get("GRASE_POSE_IOU_DONE", "0.85"))

    def _status(s: dict, stale: bool, pending_active: bool = False) -> str:
        # DONE: sufficiency — good enough for ANY object (magnitude judgments beyond
        # this are deliberately absent: small or even negative deltas are normal
        # optimizer behavior, e.g. a correct rotate_180 lowering silhouette IoU).
        if s["iou"] and not stale and float(s["iou"]) >= done_bar:
            return "DONE"
        if _capped(s["inv"]) and _capped(s["moves"]):
            return "EXHAUSTED"

        def _terminal(o: str) -> bool:
            return (
                not o.startswith("improved")
                and "refunded" not in o
                and not o.startswith("yaw-gate")
            )

        if (
            s["aspects"]
            and not pending_active
            and all(_terminal(o) for o in s["aspects"].values())
        ):
            return "CONVERGED"
        return "workable"

    rows = [  # score BEFORE IoU — uniform-frontend order, same as the emissions
        "| status | object | score | IoU | investigates | moves | aspect outcomes (latest) | exec edits (undone) |"
    ]
    any_stale = False
    any_pending = False
    for name, s in state.items():
        # a pending size mismatch is surfaced ONLY while scale is untried: a tried
        # scale already owns the slot (LOCKED / physics-REJECTED / improved), and
        # re-recommending a terminal aspect would contradict the outcome shown.
        pending_active = s["size_pending"] and "scale" not in s["aspects"]
        disp = dict(s["aspects"])
        if pending_active:
            disp["scale"] = "size-mismatch PENDING (untried)"
            any_pending = True
        aspects = "; ".join(f"{a} {o}" for a, o in disp.items()) or "-"
        iou = s["iou"] or "-"
        # a KEPT exec edit moved the object AFTER its last measurement, and no
        # post-settle IoU line accompanied it -> the number predates the edit
        stale = bool(s["iou"]) and s["keep_r"] > s["iou_r"]
        if stale:
            iou, any_stale = f"{s['iou']} (STALE)", True
        rows.append(
            f"| {_status(s, stale, pending_active)} | {name} | {s['score'] or '-'} | {iou} | {_budget(s['inv'])} "
            f"| {_budget(s['moves'])} | {aspects} | {s['exec']} ({s['undone']}) |"
        )
    return (
        "OBJECT STATE (rolled up from ALL rounds so far; budgets are hard caps).\n"
        "Status: DONE (matches well — leave it), CONVERGED (the optimizer found no "
        "remaining improvement path — more attempts will not help), EXHAUSTED "
        "(budgets spent). None of these need further work; revisiting them wastes "
        "rounds. 'workable' means improvement MAY still be possible — it is not an "
        "obligation: if the scene as a whole already matches the photo well, call "
        "end regardless of remaining workable rows.\n"
        + "\n".join(rows)
        + "\nAn object at (CAP) cannot be investigated/moved again — do not retry it. "
        "A physics-REJECTED aspect will be rejected again; fix it via "
        "execute_and_evaluate (move blocker + object together) or leave it — except "
        "a flip marked 'one-shot refunded', which keeps ONE sanctioned retry after "
        "you change what made it unstable (placement or support). "
        "A 'yaw-gate REFUSED' rotation was vetoed because the measured yaw did not "
        "improve — fix placement with move(obj,'xy') first, then re-investigate. "
        "A BLOCKED pre-physics aspect will re-clamp — and burn a move — until the "
        "neighbor blocking it is moved; move that neighbor first (or both together "
        "in one execute_and_evaluate edit). A SPENT rotate_180 is gone permanently "
        "(one-shot per object: an applied flip — kept or undone — consumes it, and so "
        "does a second physics rejection; a single clean physics rejection refunds it "
        "and shows as physics-REJECTED, not SPENT) — 'rotation' is the only yaw lever "
        "left, and a confirmed FACING reversal must instead go through a direct "
        "execute_and_evaluate yaw about the object's world-space body center. "
        "A LOCKED scale belongs to an upstream same-size "
        "policy and cannot be changed: try depth/placement with 'xy' first, but "
        "treat that as an actionable hypothesis rather than a proven cause. "
        "Undone exec edits mean that hypothesis already failed — try something new."
        + (
            " A (STALE) IoU was measured BEFORE a kept edit moved the object — judge "
            "that object from the renders, not the number."
            if any_stale
            else ""
        )
        + (
            " A 'size-mismatch PENDING' scale entry is a MEASURED size deviation "
            "nobody has acted on yet — the object stays workable until you try "
            "move(obj,'scale') (or test depth with 'y' first when its size is LOCKED)."
            if any_pending
            else ""
        )
    )


def _wall_basis_axis(normal: list[float]) -> Optional[list[float]]:
    """Right-handed local-X basis axis for a measured WALL normal.

    For horizontal ``N`` and world up ``U=(0,0,1)``, ``A=N×U=(ny,-nx,0)`` makes the
    direct-basis columns ``[A,N,U]`` positive-determinant. ``A`` is deliberately only an
    ORIENTATION axis: which half-line a finite corner wall occupies is a separate sign
    selected from its measured anchor. Returns ``None`` for supports and ambiguous tilts.
    """
    from lib.tools.geometry.scene_graph import plumb_plane

    n, kind = plumb_plane(normal)
    if kind != "wall" or n is None:
        return None
    return [n[1], -n[0], 0.0]


class PromptBuilder:
    """Helper class for building system and user prompts for agents.

    Handles prompt construction including image encoding, resource loading, and memory
    views: append-only for every agent, plus the composition generator's windowed ledger.

    Attributes:
        client: OpenAI client instance.
        config: Configuration dictionary with mode, paths, and settings.
    """

    def __init__(self, client: OpenAI, config: dict[str, Any]) -> None:
        """Initialize the prompt builder.

        Args:
            client: OpenAI client for API calls.
            config: Configuration dictionary with mode and path settings.
        """
        self.client = client
        self.config = config

    def build_prompt(
        self,
        agent_type: str,
        prompt_type: str,
        prompts: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Build a prompt for an agent based on type and mode.

        Args:
            agent_type: Either 'generator' or 'verifier'.
            prompt_type: Either 'system' or 'user'.
            prompts: Optional pre-built prompts dict. If None, fetched from the
                lib.prompts SYSTEM_PROMPTS registry via get_system_prompt.

        Returns:
            List of message dictionaries ready for the chat API.

        Raises:
            NotImplementedError: If prompt_type is not 'system' or 'user'.
        """
        if not prompts:
            prompt_agent_type = self.config.get(
                f"{agent_type}_prompt_agent_type", agent_type
            )
            # The initializer registry keys already resolve to the scene-graph prompts.
            # Preprocessing supplies revision zero; initializer tools may transactionally
            # register additional roots in that same active graph.
            prompts = {
                "system": get_system_prompt(
                    prompt_agent_type,
                    harness_profile=self.config.get("harness_profile"),
                    harness_profile_manifest=self.config.get(
                        "harness_profile_manifest"
                    ),
                )
            }

        if prompt_type == "system":
            return self._build_system_prompt(prompts, agent_type)
        elif prompt_type == "user":
            return self._build_user_prompt(prompts)
        else:
            raise NotImplementedError(f"Prompt type '{prompt_type}' not implemented")

    def _build_system_prompt(
        self, prompts: dict[str, Any], agent_type: str
    ) -> list[dict[str, Any]]:
        """Build the system prompt with initial/target images and resources."""
        system_prompt = prompts.get("system", "")
        exclusions = reconstruction_exclusions(self.config.get("ignore_objects"))
        if exclusions:
            system_prompt += "\n\n" + exclusions
        content = []

        def append_images(
            image_path: str,
            label: str,
            preferred_names: tuple[str, ...],
        ) -> None:
            paths = (
                _resolve_prompt_images(image_path, preferred_names)
                if os.path.isdir(image_path)
                else [image_path]
            )
            numbered = len(paths) > 1
            for index, resolved in enumerate(paths, 1):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_base64(resolved, max_edge=AGENT_VLM_EDGE)
                        },
                    }
                )
                prefix = f"{label} image {index}" if numbered else f"{label} image"
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"{prefix} loaded from local path: "
                            f"{display_path(resolved, self.config.get('scene_root'))}"
                        ),
                    }
                )

        if self.config.get("init_code_path") and agent_type == "generator":
            with open(self.config.get("init_code_path"), "r") as f:
                content.append({"type": "text", "text": f"Initial code: {f.read()}"})

        if self.config.get("init_image_path") and agent_type == "generator":
            append_images(self.config["init_image_path"], "Initial", ("render1.png",))

        if self.config.get("target_image_path"):
            append_images(
                self.config["target_image_path"],
                "Target",
                ("visprompt1.png", "style1.png", "render1.png"),
            )

        # CT5: the CURRENT scene render inherited from the previous stage. Attached so
        # the generator orients from evidence it already has instead of spending its
        # opening round(s) on get_scene_info / a bare render_current_scene (the measured
        # 4-6 orientation rounds/scene). Agent-loop cap applies (768).
        if agent_type == "generator" and self.config.get("stage_entry_render"):
            _ser = self.config.get("stage_entry_render")
            if os.path.exists(_ser):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_base64(_ser, max_edge=AGENT_VLM_EDGE)
                        },
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "CURRENT scene render inherited from the previous stage "
                            "(loaded from "
                            f"{display_path(_ser, self.config.get('scene_root'))}). "
                            "This is the scene as it stands NOW — "
                            "orient from it; do not begin by re-rendering the unchanged "
                            "scene."
                        ),
                    }
                )

        if self.config.get("target_description"):
            content.append(
                {
                    "type": "text",
                    "text": f"Task description: {self.config.get('target_description')}",
                }
            )

        if self.config.get("stage_context"):
            content.append(
                {
                    "type": "text",
                    "text": (
                        "Upstream stage context: "
                        + json.dumps(
                            self.config.get("stage_context"),
                            indent=2,
                            ensure_ascii=False,
                        )
                    ),
                }
            )

        moge_block = self._moge_placement_block(agent_type)
        if moge_block:
            content.append(moge_block)
            content.extend(self._main_support_overlay_parts())

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def _main_support_overlay_parts(self) -> list[dict[str, Any]]:
        ""
        moge_dir = self.config.get("moge_dir")
        if not moge_dir:
            return []
        graph_path = os.path.join(moge_dir, "scene_graph.json")
        if not os.path.exists(graph_path):
            return []
        from lib.tools.geometry.surface_relations import surface_build_name

        with open(graph_path) as f:
            graph = json.load(f)
        main = self._resolve_main_support(graph, moge_dir)
        bn = surface_build_name(main["id"])
        overlay = os.path.join(moge_dir, "masks", f"{bn}_overlay.png")
        if not os.path.exists(overlay):
            return []
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": get_image_base64(overlay, max_edge=AGENT_VLM_EDGE)
                },
            },
            {
                "type": "text",
                "text": (
                    "[Main-support grounding] The image above is the TARGET PHOTO "
                    f"with the MAIN support's (`{bn}`) segmentation mask TINTED — an "
                    "approximate but strong hint for WHICH surface is the main "
                    "support and where its visible extent ends. Anchor your "
                    "edge/corner comparisons on THIS tinted surface; edges of OTHER "
                    "furniture (background desks, shelves, counters) are NOT the "
                    "main support's edges. The mask is approximate and may be cut by "
                    "the frame — the photo owns precise edge GEOMETRY (slopes, "
                    "corners, crossings); the tint owns IDENTITY and the OUTER "
                    "extent bound on sides the frame does not cut."
                ),
            },
        ]

    def _main_support_target_anchor_parts(self) -> list[dict[str, Any]]:
        """Initializer-verifier-only target coordinates from immutable mask evidence."""
        moge_dir = self.config.get("moge_dir")
        if not moge_dir:
            return []
        graph_path = os.path.join(moge_dir, "scene_graph.json")
        observation_path = os.path.join(moge_dir, "main_support_yaw_observation.json")
        if not os.path.isfile(graph_path) or not os.path.isfile(observation_path):
            return []
        with open(graph_path) as stream:
            graph = json.load(stream)
        main = self._resolve_main_support(graph, moge_dir)
        try:
            from lib.tools.geometry.main_support_yaw_observation import (
                YawObservationArtifactError,
                load_main_support_yaw_observation,
            )

            observation = load_main_support_yaw_observation(observation_path)
        except (OSError, YawObservationArtifactError) as exc:
            logger.warning(
                "Skipping invalid main-support target-mask anchors from %s: %s",
                observation_path,
                exc,
            )
            return []
        if observation.get("main_surface", {}).get("id") != main.get("id"):
            logger.warning(
                "Skipping main-support target-mask anchors whose source id does not "
                "match the current graph: artifact=%r graph=%r",
                observation.get("main_surface", {}).get("id"),
                main.get("id"),
            )
            return []
        anchors = _main_support_target_mask_anchors(observation)
        if not anchors:
            return []
        return [
            {
                "type": "text",
                "text": (
                    "[AUTHORITATIVE MAIN-SUPPORT TARGET-MASK ANCHORS] "
                    "Backend-extracted coordinates from the source-bound tinted main-"
                    "support top-mask contour follow; x runs left-to-right and y runs "
                    "top-to-bottom in [0,1]: "
                    + json.dumps(anchors, ensure_ascii=False, separators=(",", ":"))
                    + ". These coordinates are authoritative for the TARGET MASK "
                    "anchors they name: cite them instead of inventing or eyeballing a "
                    "different target coordinate. Compare each cited anchor to the "
                    "corresponding visible feature in the BUILT RENDER yourself. This "
                    "entire block is ONE target-mask evidence group and is VETO-ONLY "
                    "grounding for target coordinates: listed points cannot alone "
                    "satisfy TWO INDEPENDENT ANCHORS or prove a pose mismatch, even when "
                    "they are at different locations. The normal tolerance, independent non-mask "
                    "corroboration, visibility, and physical-feasibility rules still "
                    "apply. A frame-crossing label describes where the observed mask "
                    "contour enters or exits a clipped image side, not a finite off-frame "
                    "corner."
                ),
            }
        ]

    def _resolve_main_support(
        self, graph: dict[str, Any], moge_dir: str
    ) -> dict[str, Any]:
        ""
        nodes = [n for n in graph.get("nodes", []) if isinstance(n, dict)]
        by_id = {n.get("id"): n for n in nodes if n.get("id")}
        graph_id = graph.get("main_support_id")
        flagged = [n for n in nodes if n.get("main_support") is True]

        if graph_id is not None or flagged:
            if graph_id is None:
                raise ValueError(
                    "scene_graph main-support contract is incomplete: missing "
                    "graph.main_support_id"
                )
            if len(flagged) != 1:
                raise ValueError(
                    "scene_graph main-support contract requires exactly one "
                    f"main_support=true node; found {len(flagged)}"
                )
            if flagged[0].get("id") != graph_id:
                raise ValueError(
                    "scene_graph main-support id/flag mismatch: "
                    f"graph={graph_id!r}, flagged={flagged[0].get('id')!r}"
                )
            main = by_id.get(graph_id)
            if main is None:
                raise ValueError(
                    f"scene_graph main_support_id {graph_id!r} names no node"
                )
        else:
            masks_path = os.path.join(moge_dir, "masks", "masks.json")
            if not os.path.isfile(masks_path):
                raise ValueError(
                    "legacy scene_graph has no main-support id/flag and is missing "
                    f"{masks_path} required to recompute the support-chain vote"
                )
            from lib.tools.geometry.preprocess import main_support_id

            with open(masks_path) as f:
                legacy_id = main_support_id(json.load(f))
            if legacy_id is None:
                raise ValueError(
                    "legacy masks support-chain vote produced no main support"
                )
            main = by_id.get(legacy_id)
            if main is None:
                raise ValueError(
                    "legacy masks support-chain winner is absent from scene_graph: "
                    f"{legacy_id!r}"
                )

        if main.get("kind") != "root_surface":
            raise ValueError(
                "scene_graph main support must be a root_surface: "
                f"{main.get('id')!r} has kind {main.get('kind')!r}"
            )
        return main

    def _moge_placement_block(self, agent_type: str) -> Optional[dict[str, Any]]:
        """Return the initializer's current registered-root data block.

        The active ``scene_graph.json`` starts in preprocessing and may be revised by
        ``build_root_surface`` between attempts. Reload it for every PromptBuilder so a
        retry never receives the stale pre-addition inventory.
        """
        if agent_type != "generator":
            return None
        if self.config.get("root_stage_name") != "initializer":
            return None
        moge_dir = self.config.get("moge_dir")
        if not moge_dir:
            return None
        graph_path = os.path.join(moge_dir, "scene_graph.json")
        if not os.path.exists(graph_path):
            return None
        with open(graph_path) as f:
            return self._scene_graph_block(json.load(f), moge_dir)

    def _preprocess_pose_history(self, moge_dir: str) -> str:
        """Describe preprocess-time pose outcomes, never current pose verdicts."""
        relative = "physics/preprocess_pose_changes.json"
        pose_path = os.path.join(moge_dir, relative)
        if not os.path.isfile(pose_path):
            return ""
        with open(pose_path) as stream:
            records = json.load(stream)["objects"]
        with open(os.path.join(moge_dir, "placement.json")) as stream:
            current_objects = json.load(stream)["objects"]
        rows = []
        for obj in current_objects:
            name = obj.get("mesh_name")
            record = records.get(name)
            if not record or not (
                record.get("fell")
                or record.get("flip_accepted")
                or record.get("settle_failed")
                or record.get("converged") is False
            ):
                continue
            retained = next(
                (
                    attempt
                    for attempt in record.get("drop_attempts") or []
                    if attempt.get("id") == record.get("retained_drop_attempt")
                ),
                {},
            )
            rows.append(
                {
                    "object": f"{obj['category']}#{obj['instance']}",
                    "mesh_name": name,
                    "fell": record.get("fell"),
                    "flip_accepted": record.get("flip_accepted"),
                    "rollable": record.get("rollable"),
                    "retained_tilt_deg": record.get("tilt_deg"),
                    "retained_drop_disp_xy_mm": retained.get("disp_xy_mm"),
                    "rest_dz_m": record.get("rest_dz"),
                    "converged": record.get("converged"),
                    "settle_failed": record.get("settle_failed"),
                }
            )
        if not rows:
            return ""
        return (
            "[Preprocessing pose history]\n"
            "These flagged preprocessing outcomes were already baked into the imported "
            "geometry. They are history, not a current-state verdict: the current scene "
            "and latest committed repair/replacement or pose-transaction evidence take "
            "precedence, including on later attempts. fell records a retained fallen OR "
            "flipped rest; flip_accepted marks an accepted stable-face flip. A large "
            "rotation alone does not make a rollable object invalid. converged describes "
            "the retained preprocessing drop, not target-pose fidelity. "
            "retained_drop_disp_xy_mm is that drop's XY drift, not cumulative motion; "
            "rest_dz_m is its recorded vertical offset in metres. Compare the current "
            "render with the target and use eligible repair tools only when needed; "
            "do not automatically restore every object upright.\n"
            + json.dumps(rows, ensure_ascii=False, allow_nan=False)
        )

    def _scene_graph_block(
        self, graph: dict[str, Any], moge_dir: str
    ) -> dict[str, Any]:
        """Initializer prompt for the current registered scene-graph revision.

        The initializer builds registered roots from the reference image and may have
        transactionally registered a missing wall in a prior attempt. We pin only the MAIN
        supporting surface to z=0 and expose bounded runtime-root provenance alongside the
        camera, surface geometry, and placed-object hierarchy.
        """
        nodes = {
            n["id"]: n
            for n in graph.get("nodes", [])
            if isinstance(n, dict) and n.get("id")
        }
        surfaces = [n for n in nodes.values() if n.get("kind") == "root_surface"]
        objects = [n for n in nodes.values() if n.get("kind") != "root_surface"]
        main = self._resolve_main_support(graph, moge_dir)
        show_support_ids = initializer_object_repair_enabled(
            self.config.get("harness_profile"),
            self.config.get("harness_profile_manifest"),
        )
        main_id = main["id"]
        main_cat = main["category"]
        main_desc = main.get("description") or ""
        other = [s for s in surfaces if s is not main]

        _screen = (
            "At the reference view, world +X is toward image-LEFT (-X right), -Y goes INTO "
            "the scene (away from the camera) and +Y toward it, +Z up; novel (orbited) views "
            "rotate this, so reason from the world axes, not screen left/right. "
        )
        cam = (
            "The reference-view camera views the scene from the input viewpoint, with +Z up. "
            + _screen
        )
        try:
            g = json.load(open(os.path.join(moge_dir, "moge", "moge.json")))["gravity"]
            R, T = g["R"], g["T"]
            fwd = [round(-R[i][1], 2) for i in range(3)]  # R @ (0,-1,0)
            loc = [round(T[i], 2) for i in range(3)]
            cam = (
                f"The reference-view camera is at ({loc[0]}, {loc[1]}, {loc[2]}) looking toward "
                f"({fwd[0]}, {fwd[1]}, {fwd[2]}) (its view direction), with +Z up. "
                + _screen
            )
        except Exception:  # noqa: BLE001 - missing/old moge.json -> generic convention
            pass

        # --- object extents from placement.json (center + size, metres) ---
        ext: dict[str, tuple] = {}
        try:
            pl = json.load(open(os.path.join(moge_dir, "placement.json")))["objects"]
            for o in pl:
                if o.get("center") and o.get("size"):
                    ext[f"{o.get('category')}#{o.get('instance')}"] = (
                        o["center"],
                        o["size"],
                    )
        except Exception:  # noqa: BLE001
            pass

        # --- support hierarchy (object subtree under each root) ---
        dup = {
            c
            for c in (o.get("category") for o in objects)
            if sum(1 for x in objects if x.get("category") == c) > 1
        }
        child_of: dict[Optional[str], list[str]] = {}
        for o in objects:
            child_of.setdefault(o.get("parent"), []).append(o["id"])
        placed: set[str] = set()

        def _row(oid: str, depth: int) -> list[str]:
            placed.add(oid)
            n = nodes.get(oid, {})
            label = oid if n.get("category") in dup else n.get("category", oid)
            e = ext.get(oid)
            tail = ""
            if e:
                c, s = e
                tail = (
                    f"  center({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f}) "
                    f"size({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})"
                )
            out = ["  " + "    " * depth + f"- {label}{tail}"]
            for ch in child_of.get(oid, []):
                out += _row(ch, depth + 1)
            return out

        hier = [f"  {main_cat} [main support, z=0]"]
        for ch in child_of.get(main_id, []):
            hier += _row(ch, 1)
        for s in other:  # objects parented to a non-main root (rare)
            kids = child_of.get(s["id"], [])
            if kids:
                hier.append(f"  {s['category']}")
                for ch in kids:
                    hier += _row(ch, 1)
        for o in objects:  # orphans (parent not a listed root/object)
            if o["id"] not in placed:
                hier += _row(o["id"], 1)

        # --- overall object span + numeric wall anchor ---
        span = ""
        if ext:
            lo = [min(c[i] - s[i] / 2 for c, s in ext.values()) for i in range(3)]
            hi = [max(c[i] + s[i] / 2 for c, s in ext.values()) for i in range(3)]
            span = (
                f"All objects span X[{lo[0]:.2f}, {hi[0]:.2f}], Y[{lo[1]:.2f}, "
                f"{hi[1]:.2f}], Z[{lo[2]:.2f}, {hi[2]:.2f}] (object world extents; "
                f"lowest point Z ≈ {lo[2]:.2f})."
            )

        # Hybrid: a masked root with a good RANSAC plane gets an explicit point+normal to
        # FOLLOW; an unmasked/low-quality one falls back to a description (visual build).
        # "Good" = planar AND big enough to be measurable (see plane_is_reliable) AND
        # confidently wall-or-support (see plumb_plane): a sliver wall's normal is fit on
        # near-collinear points so its run is noise even at inlier_frac 1.0, and an
        # ambiguous ~45-72deg tilt would be snapped 30-60deg into a fake wall/support —
        # both are better built by eye from the photo. plumb_plane is idempotent, so it
        # is safe on new (pre-snapped at serialization) and old (raw-normal) graphs.
        from lib.tools.geometry.relationship_constraints import (
            load_initializer_constraints,
        )
        from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
        from lib.tools.geometry.surface_relations import (
            relationship_is_active,
            relationship_is_hard,
            surface_build_name,
        )

        # Prompt and rules gate consume the same immutable compiled contract.  Missing
        # or stale artifacts fail here rather than silently reverting to raw rows.
        constraint_artifact = load_initializer_constraints(moge_dir, scene_graph=graph)
        compiled_constraints = constraint_artifact.get("constraints", []) or []

        def _build_name(surface_id: str) -> str:
            """Prefer the graph-registered build name for runtime roots."""
            return root_surface_build_name(nodes.get(surface_id)) or surface_build_name(
                surface_id
            )

        corner_by_id: dict[str, list[tuple[str, tuple[float, float]]]] = {}
        for constraint in compiled_constraints:
            if constraint.get("kind") != "finite_wall_corner":
                continue
            targets = list(constraint.get("targets") or [])
            if len(targets) != 2:
                continue
            aid, bid = targets
            sa, sb = nodes.get(aid), nodes.get(bid)
            if not sa or not sb:
                continue
            corner_xy = (constraint.get("reference") or {}).get("corner_xy_m")
            if not isinstance(corner_xy, list) or len(corner_xy) != 2:
                continue
            corner = (float(corner_xy[0]), float(corner_xy[1]))
            corner_by_id.setdefault(aid, []).append((bid, corner))
            corner_by_id.setdefault(bid, []).append((aid, corner))

        def _surface_line(i: int, s: dict[str, Any]) -> str:
            bn = _build_name(s["id"])  # the agent must name the Blender object this
            runtime_added = s.get("runtime_added") is True or str(
                s.get("source") or ""
            ) in {
                "initializer_build_root_surface",
                "initializer_runtime",
                "build_root_surface",
            }
            origin = (
                "RUNTIME-ADDED CURRENT REGISTERED ROOT"
                if runtime_added
                else "PREPROCESSED CURRENT REGISTERED ROOT"
            )
            label = (
                f"  ({i + 1}) {s['category']} [{origin}] "
                + (
                    f"[scene-graph support ID: `{s['id']}`] "
                    if show_support_ids
                    else ""
                )
                + f"[name the Blender object EXACTLY `{bn}`]: "
                f"{s.get('description') or '(no description)'}"
            )
            wc = s.get("world_center")
            plane = s.get("plane") or {}
            pn, kind = plumb_plane(plane.get("normal"))
            if wc and pn and kind != "ambiguous" and plane_is_reliable(plane):
                p = [round(float(x), 3) for x in wc]
                n = [round(v, 3) for v in pn]
                base = (
                    label + f"\n      plane: point ({p[0]}, {p[1]}, {p[2]}),"
                    f" normal ({n[0]}, {n[1]}, {n[2]})"
                )
                axis = _wall_basis_axis(pn)  # walls only; level supports get None
                if axis:
                    a = [round(v, 3) for v in axis]
                    line = base + (
                        f"; right-handed horizontal BASIS axis A=({a[0]}, {a[1]}, {a[2]})"
                        " and stands vertically. A fixes orientation only, NOT which way the "
                        "finite wall extends."
                    )
                    corners = corner_by_id.get(s["id"], [])
                    if len(corners) == 1:
                        other_id, c = corners[0]
                        cb = _build_name(other_id)
                        line += (
                            f" Shared measured corner with `{cb}`: C=({c[0]:.3f}, "
                            f"{c[1]:.3f}); extend from C toward this wall's measured point P "
                            "using D=A if (P-C)·A>=0 else -A, then center=C+D*width/2. "
                            "Never flip A to choose D."
                        )
                    elif len(corners) > 1:
                        listed = ", ".join(
                            f"`{_build_name(oid)}` at ({c[0]:.3f}, {c[1]:.3f})"
                            for oid, c in corners
                        )
                        line += (
                            f" Measured corners: {listed}; span the finite wall between its "
                            "corners instead of choosing a one-sided extension."
                        )
                    return line
                return base + "."
            return label + "\n      (no reliable plane.)"

        if other:
            others_block = "\n".join(_surface_line(i, s) for i, s in enumerate(other))
        else:
            others_block = "  (none)"

        # --- main-support FORM: thin slab vs full table (legs/base merged in) ---
        # A cached graph may still carry the router's stale ``tabletop`` verdict even
        # though its later surface audit added floor-under-main.  Resolve the effective
        # form here too: staged agent-only reruns intentionally reuse that graph.
        from lib.tools.geometry.surface_relations import relationship_build_line

        raw_main_form = (main or {}).get("form")
        main_form_constraint = next(
            (
                c
                for c in compiled_constraints
                if c.get("kind") == "main_support_form"
                and c.get("stage") == "STRUCTURE"
                and c.get("authority") == "required"
                and list(c.get("targets") or [None, None])[1] == (main or {}).get("id")
            ),
            None,
        )
        floor_under_main = (
            list(main_form_constraint.get("targets") or [None])[0]
            if main_form_constraint
            else None
        )
        main_form = "table" if main_form_constraint else raw_main_form
        if main_form == "table":
            form_note = (
                (
                    " — a floor/ground is explicitly UNDER this support, which requires "
                    "a FULL table; this rule overrides any stale cached tabletop label. "
                    if floor_under_main
                    else " — "
                )
                + "Build it as ONE connected piece: the flat TOP at z=0 PLUS a connected "
                "base/structure (legs / pedestal / cabinet) directly BENEATH it. "
                "MATCH the base TYPE seen in the photo: individual slender legs near the "
                "corners if the table is legged; ONE central column or solid block if it "
                "has a pedestal/block base; a closed full-width box if it is a cabinet. "
                "Do NOT default to four legs when the photo shows a pedestal, or vice versa. "
                "The base must OVERLAP UP INTO the top's underside (its top ends inside the "
                "slab, not merely touching) and stay within the top's footprint. "
                "Extend the base DOWN to the floor if a floor is "
                "present, otherwise to the bottom of the visible structure. Do NOT leave "
                "the base floating or detached from the top."
            )
        elif main_form == "tabletop":
            form_note = (
                " — only its flat TOP is visible; build it as a THIN slab (a PLANE), "
                "top face at z=0, with no base beneath."
            )
        else:
            form_note = " — its top face is at z=0."

        # --- surface relationships -> imperative build lines using the exact Blender names ---
        # Each surface's detailed description already appears in its root-surface row above.
        # Repeating that prose here is both noisy and ambiguous for similar walls; the short
        # build names make each relationship's executable endpoints explicit to the agent.
        relationship_rows: list[tuple[str, str, str]] = []
        for rel in graph.get("relationships", []):
            if (
                rel.get("a") not in nodes
                or rel.get("b") not in nodes
                or not relationship_is_active(rel)
            ):
                continue
            name_a = _build_name(rel["a"])
            name_b = _build_name(rel["b"])
            instruction = relationship_build_line(rel, name_a, name_b)
            if relationship_is_hard(rel):
                relationship_rows.append(
                    ("required", str(rel.get("relationship_id") or ""), instruction)
                )
            else:
                status = str(rel.get("status") or "unverified").strip().lower()
                advisory_kind = {
                    "conflicting": "conflicting with another relationship",
                    "unverified": "unverified",
                }.get(status, status.replace("_", " "))
                reasons = rel.get("adjudication_reasons") or []
                reason = next(
                    (str(item).strip() for item in reasons if str(item).strip()), ""
                )
                if reason:
                    advisory_kind += f"; backend reason: {reason[:240]}"
                if str(rel.get("type") or "").strip().lower() == "corner":
                    # A finite-mask reach miss says only that the photographed fragments do
                    # not prove the shared edge.  Reusing the hard CORNER imperative here
                    # caused weak generators to force the walls together despite ADVISORY.
                    relationship_rows.append(
                        (
                            "advisory",
                            "",
                            f"  - ADVISORY ({advisory_kind}): Possible layout cue: "
                            f"`{name_a}` and `{name_b}` may form a vertical corner, but the "
                            "visible evidence does not establish a shared finite edge. Use "
                            "the photo; do not force the walls to meet.",
                        )
                    )
                else:
                    relationship_rows.append(
                        (
                            "advisory",
                            "",
                            f"  - ADVISORY ({advisory_kind} — use the photo; do not force "
                            "geometry to satisfy it): " + instruction,
                        )
                    )

        # A hard source relationship and its compiled checks are one obligation, not two
        # independent instructions. Group the exact checks under the source imperative while
        # leaving independent source-yaw evidence and advisory rows separate. This preserves
        # every constraint id/tolerance/binding without making the model reconcile duplicate
        # REQUIRED bullets.
        constraint_lines: list[str] = []
        constraints_by_relationship: dict[str, list[str]] = {}

        def _record_constraint(constraint: dict[str, Any], obligation: str) -> None:
            stage = str(constraint.get("stage") or "")
            cid = str(constraint.get("constraint_id") or "")
            detail = f"[{stage}] `{cid}`: {obligation}"
            source_id = str(constraint.get("source_relationship_id") or "")
            if source_id:
                constraints_by_relationship.setdefault(source_id, []).append(detail)
            else:
                constraint_lines.append(f"  - REQUIRED {detail}")

        for constraint in compiled_constraints:
            targets = list(constraint.get("targets") or [])
            kind = str(constraint.get("kind") or "")
            params = constraint.get("parameters") or {}
            reference = constraint.get("reference") or {}
            if kind == "main_support_edge_yaw":
                if len(targets) != 1:
                    continue  # strict loader fails before this defensive branch
                target_name = _build_name(targets[0])
                run = reference.get("run_xy") or [1.0, 0.0]
                angle = math.degrees(math.atan2(float(run[1]), float(run[0]))) % 90.0
                tolerance = float(params.get("max_delta_degrees_mod_90", 12.0))
                resolved = constraint_artifact.get("source_yaw_evidence") or {}
                decision = resolved.get("source_decision") or {}
                evidence_ids = [
                    str(value)[:96] for value in decision.get("candidate_ids", [])[:4]
                ]
                reasons = [
                    str(value)[:96] for value in decision.get("reason_codes", [])[:3]
                ]
                provenance = (
                    "; evidence " + ", ".join(evidence_ids) if evidence_ids else ""
                ) + ("; reason " + ", ".join(reasons) if reasons else "")
                obligation = (
                    f"When `{target_name}` has an edge-bearing top, keep its top edge "
                    f"parallel to the frozen source-photo line run=({float(run[0]):+.4f}, "
                    f"{float(run[1]):+.4f}), angle={angle:.1f}° modulo 90°, within "
                    f"{tolerance:.0f}°. This target is immutable if any wall later "
                    "moves. Because the source evidence confirms an edge-bearing top, "
                    "building a disc/non-edge top is shape_inconsistent and FAILS; it "
                    "cannot waive this yaw obligation" + provenance + "."
                )
                _record_constraint(constraint, obligation)
                continue
            if len(targets) != 2:
                continue  # strict loader normally makes this unreachable
            a, b = (_build_name(targets[0]), _build_name(targets[1]))
            if kind == "main_support_form":
                obligation = (
                    f"`{b}` must be complete connected furniture: no bare/floating top; "
                    f"its integrated legs/pedestal/base extend down to `{a}`."
                )
            elif kind == "edge_parallel_to_surface":
                tol = float(params.get("max_delta_degrees_mod_90", 3.0))
                binding = reference.get("run_binding")
                owner = (
                    " Use the canonical source wall run; do not rotate/move the wall to "
                    "repair table yaw."
                    if binding == "canonical_scene_graph"
                    else " Use the current built root-wall run; later wall rotation changes this target."
                )
                obligation = (
                    f"When `{a}` has an edge-bearing top, keep its edge parallel to "
                    f"`{b}` within {tol:.0f}° modulo 90°.{owner} A disc/non-edge-bearing "
                    "top makes only this yaw obligation not applicable."
                )
            elif kind == "finite_against":
                gap = float(params.get("max_float_gap_m", 0.01))
                pen = float(params.get("max_penetration_m", 0.01))
                run = float(params.get("max_finite_run_gap_m", 0.05))
                vertical = float(params.get("max_vertical_gap_m", 0.05))
                canonical = reference.get("plane_binding") == "canonical_scene_graph"
                obligation = (
                    f"Keep `{a}` flush to `{b}`: gap ≤{gap * 100:.0f}cm, penetration "
                    f"≤{pen * 100:.0f}cm, finite run/vertical miss ≤{run * 100:.0f}/"
                    f"{vertical * 100:.0f}cm."
                    + (
                        " The wall plane is canonical and immutable; repair the support."
                        if canonical
                        else " The plane follows the current built root wall."
                    )
                )
            elif kind == "finite_under":
                gap = float(params.get("max_vertical_gap_m", 0.05))
                obligation = (
                    f"`{a}` must meet the bottom of `{b}` within {gap * 100:.0f}cm and "
                    "their finite XY footprints must overlap."
                )
            elif kind == "finite_wall_corner":
                angle = float(params.get("minimum_normal_angle_degrees", 85.0))
                reach = float(params.get("max_built_reach_error_m", 0.05))
                corner = reference.get("corner_xy_m") or [0.0, 0.0]
                obligation = (
                    f"`{a}` and `{b}` must form a finite wall corner: normals ≥{angle:.0f}°, "
                    f"both root-wall hulls reach C=({float(corner[0]):+.3f}, "
                    f"{float(corner[1]):+.3f}) within {reach * 100:.0f}cm, and their "
                    "vertical spans overlap."
                )
            elif kind == "finite_perpendicular":
                angle = float(params.get("minimum_normal_angle_degrees", 88.0))
                gap = float(params.get("max_finite_contact_gap_m", 0.05))
                obligation = (
                    f"`{a}` and `{b}` must meet finitely within {gap * 100:.0f}cm with "
                    f"normal angle ≥{angle:.0f}°; angle-only infinite-plane agreement is "
                    "insufficient."
                )
            else:
                obligation = (
                    f"Satisfy compiled obligation `{kind}` for `{a}` and `{b}`."
                )
            _record_constraint(constraint, obligation)

        rel_lines: list[str] = []
        for row_kind, relationship_id, instruction in relationship_rows:
            if row_kind == "advisory":
                rel_lines.append(instruction)
                continue
            checks = constraints_by_relationship.pop(relationship_id, [])
            merged = "  - REQUIRED: " + instruction
            if checks:
                merged += "\n    COMPILED CHECKS: " + " | ".join(checks)
            rel_lines.append(merged)
        # The strict artifact loader should make this unreachable, but never silently drop a
        # required check if a future compiler emits a source id absent from the graph rows.
        for relationship_id, checks in constraints_by_relationship.items():
            constraint_lines.append(
                f"  - REQUIRED relationship `{relationship_id}`: " + " | ".join(checks)
            )

        main_geo = ""
        if main:
            try:
                sp, wc = main.get("span"), main.get("world_center")
                if sp and wc:
                    import numpy as np

                    cut = ""
                    mp = os.path.join(
                        moge_dir, "masks", f"{_build_name(main['id'])}.npy"
                    )
                    if os.path.exists(mp):
                        m = np.load(mp) > 0
                        touch = [
                            nm
                            for nm, e in zip(
                                ("top", "bottom", "left", "right"),
                                (
                                    m[0, :].any(),
                                    m[-1, :].any(),
                                    m[:, 0].any(),
                                    m[:, -1].any(),
                                ),
                            )
                            if e
                        ]
                        if touch:
                            cut = (
                                f" — its mask touches the {'/'.join(touch)} frame border(s), "
                                "so the TRUE footprint extends past those cut border(s) "
                                "ONLY — on the un-cut sides the mask/tint boundary is "
                                "authoritative; do NOT build beyond it there"
                            )
                    main_geo = (
                        f"\n  Measured from depth+mask (ADVISORY, approximate): visible-footprint "
                        f"span ≈ {sp[0]:.2f} × {sp[1]:.2f} m around centroid "
                        f"({wc[0]:.2f}, {wc[1]:.2f}){cut}. Read final size and edges from the "
                        "photo; a build far outside these numbers deserves a re-look."
                    )
            except Exception:  # noqa: BLE001 - advisory numbers are best-effort
                main_geo = ""

        main_plane_note = ""
        if main:
            main_plane = main.get("plane") or {}
            main_normal, main_plane_kind = plumb_plane(main_plane.get("normal"))
            main_has_reliable_plane = bool(
                main.get("world_center")
                and main_normal
                and main_plane_kind != "ambiguous"
                and plane_is_reliable(main_plane)
            )
            if not main_has_reliable_plane:
                main_build_name = _build_name(main_id)
                main_plane_note = (
                    f"\n  `{main_build_name}` has no reliable source metric plane. "
                    "It is still a required retained root: build it visually from the "
                    "target photo. Do not omit it or invent a source-plane anchor; "
                    "compiled relationships that use a current-built binding will be "
                    "checked against the geometry you build."
                )

        # DATA ONLY — the build rules live in the initializer system prompt
        # (lib/prompts/static_scene/generators/initializer.py). Keep rule prose out of here
        # so the two sources cannot drift; this block carries only this scene's numbers.
        graph_revision = scene_graph_revision(graph)
        runtime_edits = bounded_runtime_root_surface_edits(graph)
        lines = [
            "[Initialization — per-scene data]",
            f"SCENE GRAPH REVISION: {graph_revision}. The CURRENT REGISTERED ROOT SURFACES "
            "below are authoritative for this attempt.",
            "The build rules are in your system prompt. Below is THIS scene's geometric data: "
            "the camera pose, all current registered root surfaces (with any plane geometry), "
            "and the already-placed objects with their world extents.",
            "",
            "[World Frame] +Z is UP. "
            + cam
            + "The objects are IN FRONT of the camera.",
            "",
            f"CURRENT REGISTERED MAIN SUPPORTING ROOT SURFACE: '{main_cat}'"
            + (f" [scene-graph support ID: `{main_id}`]" if show_support_ids else "")
            + (
                f" [name the Blender object EXACTLY `{_build_name(main_id)}`]"
                if main_id
                else ""
            )
            + (f" ({main_desc})" if main_desc else "")
            + form_note
            + main_geo
            + main_plane_note,
            "",
            "OTHER CURRENT REGISTERED ROOT SURFACES — each with EITHER its plane (a point "
            "on it + its normal) OR only a text description:",
            others_block,
        ]
        if runtime_edits:
            lines += [
                "",
                "RUNTIME-ADDED CURRENT REGISTERED ROOTS — these were transactionally "
                "registered by build_root_surface and are now ordinary required roots:",
            ]
            for edit in runtime_edits:
                detail = (
                    f"  - `{edit.get('build_name')}` ({edit.get('id')}, "
                    f"{edit.get('surface_type') or 'root_surface'})"
                )
                if edit.get("reason"):
                    detail += f"; reason: {edit['reason']}"
                if edit.get("target_region_norm"):
                    detail += (
                        "; motivating target region [x0,y0,x1,y1]: "
                        f"{edit['target_region_norm']}"
                    )
                if edit.get("added_attempt") is not None:
                    detail += f"; added in initializer attempt {edit['added_attempt']}"
                lines.append(detail + ".")
        if rel_lines:
            lines += [
                "",
                "SURFACE RELATIONSHIPS — REQUIRED lines are enforced; ADVISORY lines are "
                "only visual hypotheses and must not override the photo:",
                *rel_lines,
            ]
        if constraint_lines:
            lines += [
                "",
                "INDEPENDENT REQUIRED OBLIGATIONS — these exact generated-scene checks "
                "come from authenticated evidence rather than a surface-relationship row:",
                *constraint_lines,
            ]
        lines += [
            "",
            "OBJECTS ALREADY PLACED — support hierarchy, each with rough "
            "world extent [center(x,y,z); size(w,d,h) m]:",
            *hier,
        ]
        if span:
            lines += ["", span]
        if show_support_ids:
            history = self._preprocess_pose_history(moge_dir)
            if history:
                lines += ["", history]
        return {"type": "text", "text": "\n".join(lines)}

    def _build_user_prompt(self, prompts: dict[str, Any]) -> list[dict[str, Any]]:
        """Build a user prompt from execution results for the verifier.

        The verifier judges the rendered scene against the target image; it is NOT
        given the generator's init_plan (that would bias it toward "what was planned"
        rather than what is correct). The render itself is appended below via the
        execution block."""
        content = []
        if self.config.get("root_stage_name") == "initializer":
            content.extend(self._main_support_overlay_parts())
            argument = prompts.get("argument")
            generator_result = (
                argument.get("generator_result") if isinstance(argument, dict) else None
            )
            structured_gate = (
                generator_result.get("structured_gate_summary")
                if isinstance(generator_result, dict)
                else None
            )
            yaw_advisory = (
                structured_gate.get("yaw_advisory")
                if isinstance(structured_gate, dict)
                else None
            )
            requirement = (
                yaw_advisory.get("requirement")
                if isinstance(yaw_advisory, dict)
                else None
            )
            if (
                isinstance(requirement, dict)
                and requirement.get("state") == "manual_review"
            ):
                content.extend(self._main_support_target_anchor_parts())
        for key, value in prompts["argument"].items():
            content.append({"type": "text", "text": f"{key}: {value}"})
        if "image" in prompts["execution"]:
            exec_texts = prompts["execution"].get("text", [])
            exec_images = prompts["execution"]["image"]
            for idx, image in enumerate(exec_images):
                # Keep the text describing this render (e.g. the root agent's
                # "compare against the target" instruction) instead of dropping it.
                if idx < len(exec_texts):
                    content.append({"type": "text", "text": exec_texts[idx]})
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": get_image_base64(image, max_edge=AGENT_VLM_EDGE)
                        },
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": (
                            "Current scene render image loaded from local path: "
                            f"{display_path(image, self.config.get('scene_root'))}. "
                            "It was taken from the LOCKED REFERENCE camera (0,0) — this IS "
                            "the reference view; do not call render_reference_view to "
                            "re-obtain it."
                        ),
                    }
                )
            # Preserve any trailing texts that have no matching image.
            for text in exec_texts[len(exec_images) :]:
                content.append({"type": "text", "text": text})
        else:
            for text in prompts["execution"]["text"]:
                content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def build_memory_windowed(
        self,
        memory: list[dict[str, Any]],
        window_rounds: int = 5,
        protected_head: int = 2,
    ) -> list[dict[str, Any]]:
        """Sliding-window memory that SUMMARIZES instead of dropping.

        ``memory[:protected_head]`` is always kept in full. It contains the system
        prompt, initial user context, and any server-side session-entry seed. The most
        recent ``window_rounds`` tool rounds (assistant -> tool [-> user] groups) are
        kept verbatim, images included. Every OLDER round collapses into one line —
        which tool was called with which arguments and the first line of its result
        — merged into a single text message. Built for the merged composition stage,
        where dozens of investigate/move rounds each carry images: the agent keeps
        the full recent picture and a faithful ledger of everything before it."""
        head, rest = memory[:protected_head], memory[protected_head:]
        rounds = _group_rounds(rest)
        if len(rounds) <= window_rounds:
            return head + rest

        comp_k = COMP_BLOCK_K
        fresh_rounds = 0  # rounds appended since the last cut (aging leaves them alone)
        if comp_k > 0:
            n_collapsed = ((len(rounds) - window_rounds) // comp_k) * comp_k
            fresh_rounds = (len(rounds) - window_rounds) % comp_k

            # Undo-pair boundary guard: never collapse an edit round while its
            # undo round stays visible (the ledger would assert a move the window
            # reverts). Deterministic from memory content, so byte-stable between
            # cuts. Shrinking only ever WIDENS the window.
            def _is_undo(grp: list[dict]) -> bool:
                return any(
                    m.get("role") == "tool" and m.get("name") == "undo_last_step"
                    for m in grp
                )

            while n_collapsed > 0 and _is_undo(rounds[n_collapsed]):
                n_collapsed -= 1

        # static feedback boilerplate dropped from the ledger so the collapsed line keeps
        # the SIGNAL (every object's IoU + move deltas), not the format explainer / ARMED
        # note. Hints are NOT kept — _trim drops every advisory line (see there); they
        # live only in the recent full-text window. Joining all signal texts (not just
        # texts[0]) means a multi-object investigate retains all objects, not only the first.
        _skip = (
            "These objects are now ARMED", "Two crops follow",
            "Post-move render CROP follows", "IMAGE 1", "IMAGE 2",
            "Image loaded from local path", "The next messages contain",
            "The next message contains", "Investigated ", "Render image ",
        )  # fmt: skip

        def _line(idx: int, grp: list[dict]) -> str:
            a = grp[0]
            call = ""
            fn_name = ""
            tcs = a.get("tool_calls") or []
            if tcs:
                fn = tcs[0].get("function", {})
                fn_name = fn.get("name", "?")
                args = str(fn.get("arguments", ""))
                if len(args) > 120:
                    args = args[:120] + "…"
                call = f"{fn_name}({args})"
            texts: list[str] = []
            for m in grp[1:]:
                # scan BOTH the tool result and the image-paired user message: a move's
                # "moved … IoU a→b" delta rides in the user message with its post-move
                # crop, not the tool message, so a tool-only scan left move rounds blank.
                if m.get("role") not in ("tool", "user"):
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    texts.append(content)
                else:
                    texts.extend(
                        c.get("text", "")
                        for c in (content if isinstance(content, list) else [])
                        if isinstance(c, dict) and c.get("type") == "text"
                    )

            def _trim(t: str) -> Optional[str]:
                low = t.lstrip("- ").lower()
                if (
                    "HINT" in t
                    or low.startswith(
                        (
                            "facing check",
                            "post-flip facing audit",
                            "weak-axis yaw reading",
                            "(size",
                            "(a silhouette-size",
                            "(yaw for",
                            "no measured hint",
                        )
                    )
                    or (low.startswith("(note:") and "strongly rotated" in low)
                ):
                    return (
                        None  # advisory hint / facing check / deferred-size / yaw-oob
                    )
                # per-object settle-note lines carry the P1/P3 evidence pairs
                # inline (no sentence break, so the leading-clause rule below
                # cannot cut them) — strip them; they are that round's keep/undo
                # evidence, not durable state.
                t = _EDIT_EVIDENCE_PAIRS.sub("", t)
                j = t.find("REJECTED it")
                if j != -1:
                    return t[: j + len("REJECTED it")] + "."
                p = t.find(". ")
                return t[: p + 1] if p != -1 else t

            sig = []
            for t in texts:
                t = " ".join(t.split())
                if not t or any(t.lstrip().startswith(p) for p in _skip):
                    continue
                trimmed = _trim(t)
                if trimmed:
                    sig.append(trimmed)
            if fn_name == "get_scene_info":
                # the full inventory JSON is kept verbatim after the ledger (see below);
                # don't dump it here.
                result = "(object inventory — latest get_scene_info kept in full below)"
            else:
                # lines are already condensed (hints dropped, moves -> leading clause),
                # so no per-line truncation.
                result = " | ".join(sig)
            return f"[r{idx}] {call} -> {result}" if call else f"[r{idx}] {result}"

        if comp_k > 0:
            old, recent = rounds[:n_collapsed], rounds[n_collapsed:]
        else:
            old, recent = rounds[:-window_rounds], rounds[-window_rounds:]
        ledger = "\n".join(_line(i + 1, g) for i, g in enumerate(old))

        def _round_tool(g: list[dict]) -> Optional[str]:
            tcs = g[0].get("tool_calls") or []
            return tcs[0].get("function", {}).get("name") if tcs else None

        # If the entry preseed was unavailable and the model called get_scene_info,
        # keep the latest such call verbatim. Its object-id inventory lives only in
        # that tool result; collapsing it to a ledger line would strand the agent.
        # A successful preseed is already safe in the protected head.
        gsi = [g for g in old if _round_tool(g) == "get_scene_info"]
        # per-object rollup over the FULL memory (old + recent): the ledger is
        # chronological, so an object's exhausted budgets / rejected aspects scatter
        # across it and the agent re-visits dead objects; the table makes that a lookup.
        table = build_object_state_table(memory)
        summary = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "EARLIER ROUNDS (trimmed to a ledger — images and full "
                        "outputs removed; your recent rounds follow in full)."
                        + (
                            " The most recent get_scene_info among the COLLAPSED "
                            "rounds is kept in FULL below (object ids + geometry as "
                            "of that call; any newer call appears verbatim in the "
                            "rounds that follow)."
                            if gsi
                            else ""
                        )
                        + "\n"
                        + ledger
                    ),
                }
            ],
        }
        flat: list[dict] = [summary]
        if gsi:
            flat.extend(gsi[-1])  # latest get_scene_info round, verbatim
        for g in recent:
            flat.extend(g)
        frozen_from = None
        if comp_k > 0 and fresh_rounds > 0:
            # W2 cut-aligned: rounds appended since the last cut keep their images
            # (aging them would rewrite bytes mid-segment); they age at the next cut.
            n_fresh_msgs = sum(len(g) for g in recent[-fresh_rounds:])
            frozen_from = len(head) + len(flat) - n_fresh_msgs
        view = age_window_images(head + flat, len(head), frozen_from=frozen_from)
        if table:
            view = view + [
                {"role": "user", "content": [{"type": "text", "text": table}]}
            ]
        return view

    def build_memory_append_only(
        self, memory: list[dict[str, Any]], protected_head: int = 2
    ) -> list[dict[str, Any]]:
        ""
        units = _group_rounds(memory[protected_head:])
        if any(
            m.get("role") == "tool" and m.get("name") == "undo_last_step"
            for u in units
            for m in u
        ):
            kept: list[list[dict[str, Any]]] = []
            for u in units:
                if any(
                    m.get("role") == "tool" and m.get("name") == "undo_last_step"
                    for m in u
                ):
                    if kept:
                        kept.pop()
                    continue
                kept.append(u)
            memory = memory[:protected_head] + [m for u in kept for m in u]
        if _est_view_tokens(memory) <= MAX_VIEW_TOKENS:
            return list(memory)
        head = memory[:protected_head]
        units = _group_rounds(
            memory[protected_head:]
        )  # regroup: undo filter may have run
        while len(units) > 1 and (
            _est_view_tokens(head + [m for u in units for m in u]) > MAX_VIEW_TOKENS
        ):
            units.pop(0)
        return head + [m for u in units for m in u]
