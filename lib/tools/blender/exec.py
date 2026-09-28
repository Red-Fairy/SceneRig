"""Blender Executor MCP Server for executing Blender Python scripts.

This module provides an MCP server that manages Blender script execution,
rendering, and scene manipulation. It supports tool calls from the Generator
agent to execute code, get scene information, and undo operations.
"""

import copy
import datetime as _datetime
import difflib
import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# tool_client puts the repo root on PYTHONPATH for this server subprocess.
# Canonical pseudo-GT view set lives in novel_view.py; the tool-arg enums and note below derive
# from it so the valid (azimuth, elevation) choices are never hardcoded twice.
from lib.prompts.static_scene.scopes import OBJECT_SUPPORT_CONTRACT
from lib.tools.blender.initializer_transaction import validate_object_addition_scope
from lib.utils.durability import durable_fsync_enabled
from lib.tools.blender.procedural_object import format_part_name_map
from lib.tools.geometry.contact_policy import (
    CONTACT_TOLERANCE_M,
    HULL_SIBLING_TOL_M,
    HULL_SUBTOL_M,
    MAIN_SUPPORT_SURFACE_TOL_M,
    PHYSICAL_REPAIR_MAX_M,
    PHYSICAL_REPAIR_MAX_TRANSACTIONS,
    SURFACE_SURFACE,
    contact_class,
    excess_m,
    penetration_improved,
    tolerance_m,
)
from lib.tools.geometry.initializer_pose_policy import committed_initializer_event
from lib.tools.geometry.moge_camera import DEFAULT_SENSOR_MM
from lib.tools.geometry.novel_view import (
    NOVEL_VIEW_AZIMUTHS,
    NOVEL_VIEW_ELEVATIONS,
    NOVEL_VIEW_NOTE,
    NOVEL_VIEW_NOTE_ON_REQUEST,
)
from lib.tools.geometry.pseudo_gt_contract import load_complete_pseudo_gt
from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
from lib.utils.mutation_journal import append_mutation_journal, read_mutation_journal

try:
    from .script_generators import (
        COVERAGE_ID_SLOTS,
        bind_compiled_yaw_evidence,
        compiled_contact_owner_pairs,
        coverage_report,
        generate_coverage_script,
        generate_penetration_script,
        generate_scene_info_script,
        identify_main_support,
        main_support_bottom_report,
        main_support_connected_report,
        main_support_top_report,
        objects_on_direct_supports_report,
        objects_resting_report,
        relationship_contact_constraint_report,
        relationship_pose_constraint_report,
        surface_geometry_report,
    )
except ImportError:
    from script_generators import (
        COVERAGE_ID_SLOTS,
        bind_compiled_yaw_evidence,
        compiled_contact_owner_pairs,
        coverage_report,
        generate_coverage_script,
        generate_penetration_script,
        generate_scene_info_script,
        identify_main_support,
        main_support_bottom_report,
        main_support_connected_report,
        main_support_top_report,
        objects_on_direct_supports_report,
        objects_resting_report,
        relationship_contact_constraint_report,
        relationship_pose_constraint_report,
        surface_geometry_report,
    )

# Image suffixes a Blender render can produce. Matched case-insensitively so a
# render written as ``.PNG`` / ``.JPG`` / ``.jpeg`` is not silently treated as
# "no image generated" and misreported to the agent as a failure.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# Wall-clock cap (seconds) for any Blender subprocess. A hung GPU render (under
# contention) or an LLM-generated infinite loop would otherwise freeze the MCP
# server indefinitely with no recovery. Override via GRASE_BLENDER_TIMEOUT.
BLENDER_TIMEOUT = int(os.getenv("GRASE_BLENDER_TIMEOUT", "600"))

# Per-stream char cap for Blender stderr/stdout relayed in agent-facing error
# texts. tool_client truncates only its non-JSON fallback path, so a chatty
# Blender run (render progress spam, addon warnings) would otherwise flood the
# agent context — cap at the source, keeping head + tail (tracebacks end last).
_PROC_STREAM_CAP = 2000

_TYPED_INITIALIZER_MUTATION_KINDS = frozenset(
    {
        "execute_and_evaluate_objects",
        "add_object",
        "edit_object_mesh",
        "edit_object_pose",
        "remove_runtime_object",
    }
)

_COMPOSITION_MESH_MUTATION_KIND = "composition_edit_object_mesh"

# Tool models occasionally send both advertised rotation representations even
# after choosing one in their reasoning.  Accept that redundancy only when the
# two values describe the same rotation.  Six-decimal model quaternions have produced
# up to 0.187 degree of conversion/rounding separation in live probes; 0.25 degree
# admits that serialization residue while remaining far below a meaningful pose edit.
_TYPED_POSE_DUAL_ROTATION_TOLERANCE_DEG = 0.25

# Match the existing object-integrity pose tolerance.  This is used only as a
# second, GPT-6-initializer-only guard for harmless Blender float round trips;
# it is far below the smallest agent-authorized pose edit.
_TYPED_INITIALIZER_NUMERICAL_JITTER_TOLERANCE = 2e-5


def _profile_capability_enabled(
    harness_profile: str,
    harness_profile_manifest: Optional[dict],
    capability: str,
) -> bool:
    """Return whether one explicitly selected harness capability is active.

    The profile name is an independent fail-closed guard: ambient configuration or a
    malformed manifest can never switch mutation behavior on for the baseline pipeline.
    A missing manifest FAILS CLOSED (2026-09-15 owner decision): a caller that names
    ``gpt6_v1`` without the resolved manifest gets no opt-in capability, so a withdrawn
    tool can never be re-enabled by forgetting to pass the manifest. The normal CLI
    always supplies the resolved manifest.
    """
    if harness_profile != "gpt6_v1":
        return False
    capabilities = (harness_profile_manifest or {}).get("capabilities")
    return bool(capabilities.get(capability, False)) if capabilities else False


def _changed_surface_names(pre: dict, post: dict) -> set[str]:
    """Support surfaces whose evaluated geometry fingerprint differs across an edit."""
    before = pre.get("surface_geometry") or {}
    after = post.get("surface_geometry") or {}
    return {name for name in set(before) | set(after) if before.get(name) != after.get(name)}


_PYTHON_EXCEPTION_LINE = re.compile(
    r"^\s*([A-Za-z_][\w.]*(?:Error|Exception|Warning))\s*:\s*(.*)$"
)
_PROCESS_DEATH_LINES = ("Segmentation fault", "Aborted", "Killed", "Bus error")


def _python_exception_headline(*streams: str) -> Optional[str]:
    """Last Python exception line (``XxxError: message``) or process-death line.

    Blender prints "Error: script failed ... exiting." and "Blender quit" AFTER the
    traceback, so the last line of captured output never names the real failure; the
    2026-09-14 rollback census read 30 identical "Blender quit" lines for 30 distinct
    guard rejections and read-only probes.
    """
    headline = None
    for stream in streams:
        for line in (stream or "").splitlines():
            stripped = line.strip()
            if _PYTHON_EXCEPTION_LINE.match(stripped) or any(
                stripped.startswith(token) for token in _PROCESS_DEATH_LINES
            ):
                headline = stripped
    return headline


def _strict_rejection_hint(reason: str) -> str:
    """One concrete next step for a strict-physics rejection, keyed on the reason text
    that composition_physics produces. The agent used to see only "rejected by physics"
    and gave up (abc_1 croissants, 09-14); the reason plus a direction fixes that."""
    r = str(reason or "").lower()
    if "inside" in r or "penetrat" in r:
        return (
            ". Translate the named body clear of what it ended inside (or lower it onto "
            "its support) and retry with lateral clearance"
        )
    if "required support" in r:
        return (
            ". Keep it on its declared support — place it fully within that support's "
            "extent, or move support and cargo together"
        )
    if "rolled" in r or "slid" in r:
        return (
            ". A rollable neighbour rolled away when its support moved; give it a flat, "
            "level support and clearance, or move it back deliberately in the same edit"
        )
    if "converge" in r or "did not reach rest" in r or "capsiz" in r:
        return (
            ". The requested attitude has no rest state; request a pose the object can "
            "rest in on its support (flat, or its natural upright)"
        )
    if "rotated" in r or "moved" in r:
        return (
            ". Physics carried it away from the request; re-place it near the settled "
            "pose or fix its clearance and support first"
        )
    return ""


def _error_class(error: BaseException | str) -> str:
    """Stable, census-friendly classifier for a rollback: exception name + first clause."""
    text = str(error)
    headline = _python_exception_headline(text)
    if headline:
        name, _, message = headline.partition(":")
        first = message.strip().split(":", 1)[0].strip()
        return f"{name.strip()}: {first[:60]}" if first else name.strip()
    name = type(error).__name__ if isinstance(error, BaseException) else "error"
    return f"{name}: {text.split(':', 1)[0].strip()[:60]}"


def _proc_text(stderr: str, stdout: str) -> str:
    """stderr+stdout for agent-facing error texts, each stream head+tail-capped."""

    def clip(s: str) -> str:
        if len(s) <= _PROC_STREAM_CAP:
            return s
        h = _PROC_STREAM_CAP // 2
        return (
            s[:h]
            + f"\n... [{len(s) - _PROC_STREAM_CAP} chars truncated] ...\n"
            + s[-h:]
        )

    return clip(stderr) + clip(stdout)


# 2026-09-15 owner: relay the script's own print() output to the agent, capped at 1000
# chars. Until then a SUCCESSFUL run returned only the render text, so agents read their
# Blender-side measurements by `raise Exception(str(values))` — 10+ such scripts in the
# v5accept batch, 6 of them initializer rollbacks journaled as "agent code failed".
_SCRIPT_STDOUT_CAP = 1000
_BLENDER_NOISE_PREFIXES = (
    "Blender ",
    "Read blend:",
    "Read prefs:",
    "Read library:",
    "Blender quit",
    "Info: ",
    "Saved: ",
    "Fra:",
    "Time:",
    "Writing:",
    "Color management",
    "[INFO]",
    "[WARN]",
)


def _journal_composition_edit(executor, kind: str, status: str, details: dict) -> None:
    """Audit-only journal row for a composition POSE edit — both harnesses (2026-09-15
    owner). Until now only the initializer transactions and `edit_object_mesh` were
    journaled; the stage with the most scene mutations (18 moves + 6 freeform edits in
    v5accept online6) left no record and the census had to grep tool-result prose.
    Not a write-ahead log: never fails the edit, no capability gate, string ids
    (`composition_edit_N`) so `_next_runtime_mutation_id` keeps ignoring them."""
    try:
        moge_dir = getattr(executor, "moge_dir", None)
        if not moge_dir or getattr(executor, "root_stage_name", None) != "composition":
            return
        seq = int(getattr(executor, "_composition_edit_seq", 0) or 0) + 1
        executor._composition_edit_seq = seq
        path = Path(moge_dir) / "audit" / "scene_mutations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": 1,
            "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "harness_profile": getattr(executor, "harness_profile", None),
            "stage": "composition",
            "attempt_idx": int(getattr(executor, "attempt_idx", 1) or 1),
            "transaction_id": f"composition_edit_{seq}",
            "kind": kind,
            "status": status,
            "details": json.loads(json.dumps(details, default=str)),
        }
        append_mutation_journal(path, row)
    except Exception as exc:  # noqa: BLE001 - an audit row must never fail a pose edit
        logging.getLogger(__name__).warning("composition pose-edit journal skipped: %s", exc)


def _settle_report_summary(reports) -> list:
    """Compact per-body rows of a settle report for the journal (numbers only)."""
    keys = ("name", "members", "disp_mm", "tilt_deg", "lift_mm", "capsized",
            "converged", "penetration", "joint", "requested_to_settled")  # fmt: skip
    return [
        {k: r.get(k) for k in keys if k in r}
        for r in (reports or [])
        if isinstance(r, dict)
    ]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _script_output(stdout: str) -> str:
    """The agent's print() lines from a Blender run as one agent-facing text part:
    Blender's and the wrapper's boilerplate dropped, head+tail clipped to
    _SCRIPT_STDOUT_CAP chars; "" when the script printed nothing."""
    lines = [
        ln
        for ln in (stdout or "").splitlines()
        if ln.strip() and not ln.startswith(_BLENDER_NOISE_PREFIXES)
    ]
    if not lines:
        return ""
    text = "\n".join(lines)
    if len(text) > _SCRIPT_STDOUT_CAP:
        h = _SCRIPT_STDOUT_CAP // 2
        text = (
            text[:h]
            + f"\n... [{len(text) - _SCRIPT_STDOUT_CAP} chars truncated: print less, or only "
            "what you need] ...\n"
            + text[-h:]
        )
    return "Script output (your print() lines, up to 1000 chars):\n" + text


def _capture_part_feedback(captures: list[tuple[str, dict]]) -> str:
    """Preview complete semantic-name mapping rows without flooding tool feedback."""
    blocks = []
    size = 0
    for index, (object_id, capture) in enumerate(captures):
        preview = format_part_name_map(capture, max_parts=8)
        if not preview:
            continue
        block = f"[Object parts: {object_id}]\n{preview}"
        if size + len(block) > 8000:
            blocks.append(
                f"Part-name previews omitted for {len(captures) - index} more objects. "
                "Their full mappings remain in capture records; query each root's "
                "children_recursive and grase_part_label for subsequent edits."
            )
            break
        blocks.append(block)
        size += len(block) + 2
    return "\n\n".join(blocks)


def _yaw_note_from_evidence(evidence: Optional[dict]) -> Optional[str]:
    """Summarize the immutable non-required source-yaw decision."""
    evidence = evidence or {}
    applicability = evidence.get("applicability") or {}
    if applicability.get("status") != "applicable":
        return None  # axisless/room-floor is intentional; unknown is a hard failure
    decision = evidence.get("source_decision") or {}
    authority = str(decision.get("authority") or "unverified")
    verdict = evidence.get("verdict") or {}
    delta = verdict.get("delta_degrees_mod_90")
    tolerance = verdict.get("tolerance_degrees")
    if authority == "advisory":
        measurement = ""
        if delta is not None:
            measurement = f"; estimated delta {float(delta):+.1f}deg"
            if tolerance is not None:
                relation = (
                    "within" if abs(float(delta)) <= float(tolerance) else "beyond"
                )
                measurement += (
                    f" {relation} the {float(tolerance):.0f}deg POSE tolerance"
                )
        reasons = ", ".join(map(str, decision.get("reason_codes") or []))
        return (
            "main-support yaw is ADVISORY from the frozen source-evidence resolver"
            + (f" ({reasons})" if reasons else "")
            + measurement
            + "; follow the machine-verifiable advisory requirement before ending"
        )
    if authority == "conflicting":
        return (
            "main-support yaw source clues CONFLICT; no numeric target or rotation "
            "hint was selected"
        )
    if authority == "unverified":
        return "main-support yaw is UNVERIFIED; no source target was selected"
    return None


def _list_render_images(render_path: str) -> list[str]:
    """Return sorted image paths in ``render_path`` (case-insensitive suffix)."""
    if not os.path.isdir(render_path):
        return []
    return sorted(
        str(p)
        for p in Path(render_path).glob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    )


_SR_BLOCK = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.DOTALL
)
_RESERVED_TOOL_MARKUP = re.compile(
    r"<\s*/?\s*(?:parameter|thought|tool_call)\b", re.IGNORECASE
)
_EXECUTE_FULL_SCRIPT_EXAMPLE = (
    '{"thought":"Run a complete Blender script.","code":"import bpy\\n'
    'print(len(bpy.context.scene.objects))","code_diff":""}'
)
_EXECUTE_PATCH_EXAMPLE = (
    '{"thought":"Name the diagnostic value.","code_diff":"<<<<<<< SEARCH\\n'
    "print(len(bpy.context.scene.objects))\\n=======\\n"
    "object_count = len(bpy.context.scene.objects)\\n"
    'print(object_count)\\n>>>>>>> REPLACE"}'
)
_EXECUTE_ARGUMENT_EXAMPLES = (
    "\n\nExact FULL-SCRIPT arguments:\n" + _EXECUTE_FULL_SCRIPT_EXAMPLE + "\n"
    "Exact PATCH arguments:\n" + _EXECUTE_PATCH_EXAMPLE + "\n"
)


def _has_syntactically_valid_code_diff(code_diff: str) -> bool:
    """Whether ``code_diff`` has at least one usable SEARCH/REPLACE block.

    This deliberately mirrors the syntax accepted by :func:`apply_search_replace`.
    Match applicability cannot be checked until a previous script exists.
    """
    blocks = _SR_BLOCK.findall(code_diff)
    return bool(blocks) and all(search.strip() for search, _replace in blocks)


def _execute_argument_error(
    error_code: str,
    message: str,
    *,
    retryable: bool,
) -> dict[str, object]:
    """Return a non-mutating, machine-readable execute argument failure."""
    return {
        "status": "error",
        "output": {
            "text": [message],
            "error_code": error_code,
            "retryable": retryable,
            "scene_mutation": "not_committed",
            "expected_tool": "execute_and_evaluate",
        },
    }


def apply_search_replace(base: str, diff: str) -> str:
    """EE-2: apply SEARCH/REPLACE blocks to the previous round's script.

    Contract (mirrored in the tool schema): one or more blocks, applied in order;
    each SEARCH text must match EXACTLY ONCE in the current script. Raises
    ValueError with a model-actionable reason on any violation — the caller turns
    that into an error response asking for the full script."""
    blocks = _SR_BLOCK.findall(diff)
    if not blocks:
        raise ValueError(
            "no valid <<<<<<< SEARCH / ======= / >>>>>>> REPLACE block found in "
            "code_diff"
        )
    out = base
    for i, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            raise ValueError(
                f"block {i} has an empty SEARCH section — patches must anchor on "
                "existing lines (send the full script in `code` for a rewrite)"
            )
        n = out.count(search)
        if n == 0:
            preview = search.splitlines()[0][:80] if search.splitlines() else ""
            raise ValueError(
                f"block {i}'s SEARCH text was not found in the previous script "
                f"(first line: {preview!r}) — it must match the script byte-exactly."
                + _closest_current_lines(out, search)
            )
        if n > 1:
            raise ValueError(
                f"block {i}'s SEARCH text matches {n} places in the previous "
                "script — add surrounding lines to make it unique"
            )
        out = out.replace(search, replace, 1)
    return out


_EXPORTER_REMEDIES = (
    (
        "not a closed manifold",
        "every part must be a closed solid with no open shells, holes or unconnected "
        "edges — build parts from closed primitives (cube, cylinder, sphere) or cap every "
        "extruded profile; never copy the reconstructed fragment soup",
    ),
    (
        "disconnected",
        "the parts must overlap with real volume into ONE connected union — move or "
        "enlarge parts so each shares volume with another; touching faces do not count",
    ),
    (
        "duplicate tessellated",
        "the two named parts graze each other along a near-tangent intersection — deepen "
        "their overlap by a few millimetres or shrink the bevel/rim so the union has no "
        "sliver faces",
    ),
    (
        "seam weld",
        "parts that only touch or graze cannot union — make every pair overlap by a few "
        "millimetres of real volume (the message names the pairs when it can)",
    ),
    (
        "pinch a polygon",
        "a razor-thin sliver where parts intersect — make part overlaps a few millimetres "
        "deep and not tangent to each other",
    ),
    (
        "lost or changed a source data layer",
        "the Boolean seam cleanup could not keep the parts' UV/color/normal data — avoid "
        "near-tangent intersections and keep one material per part",
    ),
    (
        "zero/non-finite triangle area",
        "degenerate geometry — no zero-size dimensions, no coincident vertices, finite "
        "coordinates only",
    ),
)


def _exporter_rejection(blender_output: str) -> str:
    """Agent-facing sentence for a failed authored-geometry export: the Python exception
    headline (the watertightness / Boolean check that fired) plus one remedy, instead
    of the raw Blender traceback. Unknown checks pass the headline through."""
    headline = _python_exception_headline(blender_output) or ""
    reason = re.sub(r"^\w+Error: ", "", headline).strip() or "the exporter rejected the geometry"
    reason = reason.replace("authored root-local proposal: ", "")
    remedy = next((fix for key, fix in _EXPORTER_REMEDIES if key in reason), None)
    text = f"authored geometry rejected — {reason}."
    if remedy:
        text += f" Fix: {remedy}."
    return text + " (The full Blender output is kept in the journal.)"


def _closest_current_lines(script: str, search: str, *, cutoff: float = 0.5) -> str:
    """The current script lines that most resemble a SEARCH block that did not match —
    numbered, so the agent's full-script resend edits the script it actually has.
    2026-09-15 census: 76/76 SEARCH mismatches followed a SUCCESSFUL patch, i.e. the
    agent anchored on text its own previous patch had already changed."""
    lines = script.splitlines()
    wanted = [ln for ln in search.splitlines() if ln.strip()]
    if not lines or not wanted:
        return ""
    hit = difflib.get_close_matches(wanted[0], lines, n=1, cutoff=cutoff)
    if not hit:
        return ""
    start = lines.index(hit[0])
    stop = min(len(lines), start + max(len(wanted), 1))
    # repr: a leading space / tab in the SEARCH text is otherwise invisible (abc_3 0916)
    quoted = "\n".join(f"{k + 1}: {lines[k]!r}" for k in range(start, stop))
    return (
        f" Closest CURRENT text (lines {start + 1}-{stop}; your previous patch may have "
        f"changed it):\n{quoted}"
    )


_PATCH_RECEIPT_CAP = 1500


def _patch_receipt(base: str, merged: str) -> str:
    """What a successful code_diff changed, quoted from the MERGED script with current
    line numbers and two lines of context, so the next SEARCH anchors on lines that
    exist. Head+tail capped at _PATCH_RECEIPT_CAP chars."""
    a, b = base.splitlines(), merged.splitlines()
    regions = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        lo, hi = max(0, j1 - 2), min(len(b), (j2 if j2 > j1 else j1 + 1) + 2)
        body = "\n".join(
            f"{k + 1}: {b[k]}" + ("   <- changed" if j1 <= k < j2 else "")
            for k in range(lo, hi)
        )
        regions.append(f"lines {lo + 1}-{hi}:\n{body}")
    text = "\n".join(regions) if regions else "(no line changed)"
    if len(text) > _PATCH_RECEIPT_CAP:
        h = _PATCH_RECEIPT_CAP // 2
        text = text[:h] + f"\n... [{len(text) - _PATCH_RECEIPT_CAP} chars truncated] ...\n" + text[-h:]
    return (
        f"Patch applied. The current script is {len(b)} lines; changed regions "
        f"(current numbering):\n{text}\nAnchor your next SEARCH on these CURRENT lines "
        "— the previous script no longer exists."
    )


# Tool configuration dictionaries for the Generator agent
execute_and_evaluate_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "execute_and_evaluate",
        "description": "Execute Blender Python code, then render the scene from the chosen viewpoint and return that render PAIRED with its reference.\nThe camera is FIXED to the reference view — do NOT add, move, or edit the camera in your code; choose the viewpoint via the (azimuth, elevation) argument instead, which is an orbit OFFSET (delta) from the reference view ((0,0) = the reference view).\nReturns either:\n(1) On error: detailed error information; or\n(2) On success: TWO images — a render of your scene from the chosen view and its reference (the REAL ground-truth target photo at the reference view (0,0), or a pseudo-GT at a novel view) to compare against.\nAfter each successful call, inspect your render against the returned reference and change the scene to close the gap. Verifier feedback is returned between attempts (as the feedback on your next pass), not per call."
        + _EXECUTE_ARGUMENT_EXAMPLES,
        "parameters": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": 'Reason concisely about the current scene and the next edit. This field is reasoning ONLY: never put the runnable Python script, XML tags such as `<parameter name="code">` or `</thought>`, Markdown code fences, or serialized tool arguments here.',
                },
                "code_diff": {
                    "type": "string",
                    "description": 'Choose exactly one mode. FULL-SCRIPT MODE — required on the FIRST execute_and_evaluate call of a stage, after a reported patch failure, or for a rewrite: provide the COMPLETE runnable script in the separate top-level `code` field and set this `code_diff` field to exactly the empty string `""`. PATCH MODE — allowed only after a complete script has been successfully submitted: omit `code` and provide one or more SEARCH/REPLACE blocks in EXACTLY this format:\n\n<<<<<<< SEARCH\n[lines copied byte-exact from your previous script]\n=======\n[replacement lines]\n>>>>>>> REPLACE\n\nIn PATCH MODE, each SEARCH must match the previous script exactly once; copy it byte-exactly, add context when needed for uniqueness, and include no commentary, Markdown, or fences. `thought`, `code`, and `code_diff` are separate top-level arguments; never encode one inside another with XML or text.',
                },
                "code": {
                    "type": "string",
                    "description": 'The COMPLETE runnable Blender Python script in FULL-SCRIPT MODE. It is REQUIRED as an actual top-level `code` argument on the first execute_and_evaluate call of a stage, after a reported patch failure, or for a rewrite; pair it with `code_diff: ""`. Merely saying that code was provided, or placing Python or `<parameter name="code">` text inside `thought`, does NOT populate this field. In PATCH MODE, omit `code` and use a valid non-empty `code_diff` instead.',
                },
            },
            "required": ["thought", "code_diff"],
        },
    },
}

# Pseudo-GT novel-view params: OPTIONAL — if given, az/el must be one of the fixed
# (azimuth, elevation) pairs the pre-compute built; omit to default to (0,0) = the reference
# view. Shared by the two novel-view tools.
_NOVEL_VIEW_PARAMS: dict[str, object] = {
    "azimuth": {
        "type": "number",
        "enum": NOVEL_VIEW_AZIMUTHS,
        "default": 0,
        "description": "Horizontal orbit OFFSET (deg) from the reference view — a delta, not an absolute angle; 0 = the reference view (the default if omitted).",
    },
    "elevation": {
        "type": "number",
        "enum": NOVEL_VIEW_ELEVATIONS,
        "default": 0,
        "description": "Vertical orbit OFFSET (deg) from the reference view — a delta, not an absolute angle; 0 = the reference view (the default if omitted).",
    },
}
_NOVEL_VIEW_NOTE = NOVEL_VIEW_NOTE

# Freeform-relocation prose. COMPOSITION + isaac ONLY — every clause below is false anywhere
# else: move() exists only in composition's tool menu, _settle_after_edit() no-ops unless
# the composition stage, and the freeform budget it mentions is only ever incremented
# inside that same settle. It used to be appended unconditionally, so texture / lighting /
# initializer read "prefer move()" (a tool they lack) and an invitation to RELOCATE objects that
# their own scope forbids. Appended below only to execute_and_evaluate_composition_tool.
_FREEFORM_SETTLE_NOTE = (
    "\n\nPOSE REFINEMENT: move() is the routine physics-gated tool for single-object "
    "position/rotation/scale — prefer it for ordinary fixes. But reach for "
    "execute_and_evaluate whenever you can SEE from the crops which way an object should "
    "move and either (a) move() was physics-REJECTED (boxed in / collision), (b) it is "
    "a relocation move() cannot express (onto a surface, a multi-object layout edit), or "
    "(c) a visually warranted fine-yaw move was rejected or its search was dead. "
    "For a direct-yaw fallback after a rejected or dead move(object, 'rotation'), rotate "
    "about the object's current WORLD-SPACE body center, never the scene/world origin, and "
    "keep the returned crop or call undo_last_step if it regressed. "
    "Choose collision-free XY destinations for every object you relocate. Physics "
    "settlement resolves vertical clearance and the resting height/orientation; it does "
    "NOT plan new XY destinations or intentionally separate objects that laterally "
    "overlap. After a boxed-in result, do not repeat the same single-object translation "
    "into the blocked destination. Instead, place the involved objects at independently "
    "chosen, visually justified final positions with enough lateral clearance; their "
    "relative positions may change. You do NOT need to supply the exact resting pose: "
    "every object your code RELOCATES is physics-settled (lift-to-clear + settle; anything "
    "stacked on it rides along) and the result is reported. KEEP it, or undo_last_step if "
    "it toppled, still overlaps, or drifted from the target. NEVER parent one imported "
    "obj_* object to another or add a parent constraint between them. It is budgeted, so "
    "don't spend it on tiny nudges move() already handles."
)
# World-yaw change (deg) below which a freeform edit does NOT count as a yaw-type
# edit for the settle note's "measured yaw: pre -> post" line (A3/G1, bridge_6):
# settle jitter and lift-to-clear rotations sit under this.
_FREEFORM_YAW_EDIT_DEG = 3.0


def _appearance_pair_qualifier(pre: float, post: float) -> str:
    """Parenthetical qualifier for an appearance-agreement before->after pair —
    the ONE place its direction language lives (freeform settle note + applied
    move('rotation') feedback). Below the instrument's own noise floor
    (HINT_MARGIN: a FULL 180-deg pose difference only separates from noise at
    >= ~0.02, and the bbox-stretch crop normalizes round/texture-poor objects
    to ~0) a sub-margin delta must NOT read as directional evidence — a correct
    yaw fix on a compact texture-poor object can print a flat or slightly
    FALLING pair, recreating the bridge_6 undo trap inside the channel built to
    fix it. Kept parenthetical (no ')' inside) so the ledger's
    _EDIT_EVIDENCE_PAIRS regex strips both variants."""
    from lib.tools.geometry.feature_metric import HINT_MARGIN

    if abs(post - pre) < HINT_MARGIN:
        return (
            "~equal — a difference this small is within this instrument's "
            "noise; judge the crop"
        )
    return "position/scale-independent — higher = this pose looks more like the photo"


# Shared by initializer, texture, and composition; lighting gets the source-view-only
# variant below. Same edit+render as execute_and_evaluate, plus the (azimuth, elevation)
# pseudo-GT view. Same tool NAME, so it dispatches to the same
# execute_and_evaluate function. Carries NO stage-specific prose — anything true of only one
# stage belongs on a derived variant below, not here.
execute_and_evaluate_novel_view_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "execute_and_evaluate",
        "description": execute_and_evaluate_tool["function"]["description"]
        + _NOVEL_VIEW_NOTE,
        "parameters": {
            "type": "object",
            "properties": {
                **execute_and_evaluate_tool["function"]["parameters"]["properties"],
                **_NOVEL_VIEW_PARAMS,
            },
            "required": ["thought", "code_diff"],
        },
    },
}

# Composition + isaac ONLY (see _FREEFORM_SETTLE_NOTE): the shared tool plus the freeform-
# relocation prose (baseline). gpt6_v1 gets the _GPT6_LAYOUT_SETTLE_NOTE variant below. Gate must track _settle_after_edit()'s guard — if that ever settles on
# another stage/backend, widen the branch in initialize() and this comment together.
execute_and_evaluate_composition_tool: dict[str, object] = copy.deepcopy(
    execute_and_evaluate_novel_view_tool
)
execute_and_evaluate_composition_tool["function"]["description"] += (
    _FREEFORM_SETTLE_NOTE  # type: ignore[index]
)

# gpt6_v1 (strict_post_edit_physics): execute_and_evaluate is the GENERAL pose tool, a
# peer of move(), not the budgeted fallback the baseline note describes. Same shared
# tool, different prose; the settle itself is identical.
_GPT6_LAYOUT_SETTLE_NOTE = (
    "\n\nPOSE EDITING: this is your general pose tool, a peer of move(). Write complete "
    "Blender Python that sets world transforms directly — one object or several per call, "
    "a translation toward the target, a yaw or attitude change, a coordinated correction of "
    "several objects toward the photo (never a redesign of the scene) — at any time, with or without a prior investigate_objects call; it does "
    "not clear the armed set. Reason in the WORLD FRAME and choose collision-free XY "
    "destinations; physics resolves vertical clearance, not lateral overlap. Every object "
    "your code moves is physics-settled before the render: the moved objects (with anything "
    "stacked on them) settle TOGETHER in one simulation, each from its requested pose, while "
    "everything you did not move stays fixed — so two objects you moved cannot end inside "
    "each other. Physics may shift or rotate a moved object into a stable rest; that drift "
    "is reported with the result, not rejected. The SETTLED pose is what you see and what is "
    "kept: judge the returned render, then KEEP it or call undo_last_step. Non-convergence, "
    "new or worsened penetration, or a toppled rider or neighbor rejects and rolls back "
    "the whole edit, naming the body and the reason. Code that changes an existing "
    "object's mesh data re-cooks its collider before the settle; edit_object_mesh "
    "replaces a whole object under its contract. Never parent one imported obj_* object "
    "to another."
)
execute_and_evaluate_composition_tool_gpt6: dict[str, object] = copy.deepcopy(
    execute_and_evaluate_novel_view_tool
)
execute_and_evaluate_composition_tool_gpt6["function"]["description"] += (
    _GPT6_LAYOUT_SETTLE_NOTE  # type: ignore[index]
)

# Initializer-only contract.  Registered roots (both preprocessing-provided and
# runtime-added) are ordinary Blender geometry and are still created/refined with
# execute_and_evaluate.  Only *registering a new graph root* belongs to
# build_root_surface below.
execute_and_evaluate_initializer_tool: dict[str, object] = copy.deepcopy(
    execute_and_evaluate_novel_view_tool
)
execute_and_evaluate_initializer_tool["function"]["description"] += (  # type: ignore[index]
    "\n\nINITIALIZER ROOTS: You may create or refine every CURRENT REGISTERED root "
    "surface with this tool, including preprocessing-provided roots and roots returned "
    "by build_root_surface. This tool does not register a new graph root. If the target "
    "clearly requires a distinct missing architectural wall plane, introduce it through "
    "build_root_surface first. Never edit scene_graph.json or the initializer transaction "
    "ledger from generated code."
)

execute_and_evaluate_tool["function"]["description"] += (
    "\nYour script's print() output comes back with the result as 'Script output' (Blender's "
    "own lines removed, up to 1000 chars): print the measurements you need and read them from "
    "a SUCCESSFUL run — never raise an exception to surface values (a failed script is rolled back)."
)

execute_and_evaluate_initializer_gpt6_tool = copy.deepcopy(
    execute_and_evaluate_initializer_tool
)
execute_and_evaluate_initializer_gpt6_tool["function"]["description"] += (
    "\n\nOBJECT TRANSACTIONS: You may freely correct whole-object poses in this "
    "code. To add or remove objects, declare added_objects and removed_objects "
    "explicitly (empty lists when none). To replace a wrong mesh, list its identity "
    "in BOTH, remove the old hierarchy in code, and build its replacement from "
    "primitives. You may also list an object in removed_objects ALONE when it is a "
    "duplicate segmentation of another object or of one of that object's parts (e.g. a "
    "stand's tray base imported a second time as its own object); the reason must quote "
    "the containment evidence — which object contains it, and the footprint/dimension "
    "figures from the scene that show it. "
    "Added/replaced roots must be parentless EMPTY objects with MESH "
    "children. ADDED_OBJECT_NAMES and REMOVED_OBJECT_NAMES are injected Python "
    "dictionaries mapping category#N IDs to exact canonical Blender root names. "
    "All parts must form a connected solid union. The backend exports their union "
    "as one mesh asset, saves graph/placement/inventory and records one undoable "
    "transaction. Object creation/replacement/removal, whole-object pose edits, "
    "and whole-object pose edits run one physics simulation over exactly those "
    "objects' hierarchies (every other object and every root surface static); a "
    "root-surface-only build or edit never simulates. "
    "The saved scene and returned render show the settled poses. Inspect motion, "
    "topple and support reports and repair a toppled object in your next edit. "
    "Converged topples are kept for repair; failed/nonconverged simulations roll "
    "back the transaction. Material-only and unchanged calls skip simulation. "
    "Do not save/export or edit backend files yourself. Authored objects have no "
    "segmentation masks. Existing root-surface registration rules still apply."
)
_initializer_transaction_parameters = execute_and_evaluate_initializer_gpt6_tool[
    "function"
]["parameters"]
_initializer_transaction_parameters["properties"].update(
    {
        "added_objects": {
            "type": "array",
            "description": "New or replacement objects built by this script; [] if none.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "object_id": {
                        "type": "string",
                        "description": "Exact category#N graph ID. Replacements keep the existing ID.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Object appearance and the reference evidence motivating its construction.",
                    },
                    "support": {
                        "type": "string",
                        "description": "Exact final graph support ID; may refer to another declared addition.",
                    },
                    "physical_material_hint": {
                        "type": "string",
                        "description": "Optional physical material label for later physics estimation.",
                    },
                },
                "required": ["object_id", "description", "support"],
            },
        },
        "removed_objects": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Exact existing object IDs whose full hierarchies this script removes; [] if none. A replacement ID appears in both lists.",
        },
    }
)
_initializer_transaction_parameters["required"] = [
    *_initializer_transaction_parameters.get("required", []),
    "added_objects",
    "removed_objects",
]

render_current_scene_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "render_current_scene",
        "description": (
            # NOT _NOVEL_VIEW_NOTE: unlike execute_and_evaluate, this tool pairs a reference
            # only when a viewpoint is asked for (composition aside, which always pairs).
            "Render the CURRENT scene (no edit) from a reference viewpoint to inspect the "
            "layout." + NOVEL_VIEW_NOTE_ON_REQUEST
        ),
        "parameters": {
            "type": "object",
            "properties": dict(_NOVEL_VIEW_PARAMS),
            "required": [],
        },
    },
}

# Lighting ONLY: no (azimuth, elevation) on either render tool.
#
# Illumination is graded photometrically, and a novel view's reference is a GENERATIVE
# COMPLETION of the lifted point cloud — the inpainter invents its lighting. Matching
# exposure, shadow softness or colour temperature against that is chasing a hallucination,
# and only the source view (0,0), whose reference is the REAL photo, can settle them.
#
# Removing the params from the SCHEMA is the whole enforcement: execute_and_evaluate() and
# render_current_scene() default az/el to None -> (0,0), so this stage keeps getting its
# render paired with the real target photo and no dispatch code changes.
#
# The rationale above is for the CODE READER. The agent is deliberately NOT told why it has
# no viewpoint argument — describing a capability a stage does not have is what made every
# stage read composition's move()/physics prose (TL2). Absences here are silent, the way
# texture is never told it has no render_bev.
execute_and_evaluate_source_view_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "execute_and_evaluate",
        "description": (
            "Execute Blender Python code, then render the scene and return that render "
            "PAIRED with the REAL ground-truth target photo.\n"
            "The camera is FIXED to the reference view — do NOT add, move, or edit the "
            "camera in your code.\n"
            "Returns either:\n"
            "(1) On error: detailed error information; or\n"
            "(2) On success: TWO images — a render of your scene and the target photo, to "
            "compare against.\n"
            "After each successful call, inspect your render against the photo and change "
            "the scene to close the gap. Verifier feedback is returned between attempts (as "
            "the feedback on your next pass), not per call."
            + _EXECUTE_ARGUMENT_EXAMPLES
        ),
        "parameters": {
            "type": "object",
            "properties": dict(
                execute_and_evaluate_tool["function"]["parameters"]["properties"]
            ),
            "required": ["thought", "code_diff"],
        },
    },
}

render_current_scene_source_view_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "render_current_scene",
        "description": (
            "Render the CURRENT scene (no edit) from the reference view to inspect the "
            "latest state. Compare it against the target photo at the top of this "
            "conversation. Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

render_bev_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "render_bev",
        "description": (
            "Render a top-down BIRD'S-EYE VIEW (orthographic) of the current scene to check the "
            "layout FROM ABOVE — the clearest way to see which SIDE each wall/surface is on. Each "
            "CURRENT REGISTERED root surface (table_0, wall_0, ...) is flat-tinted + "
            "id-labelled (see the legend): the "
            "table and walls get distinct colours, floors/ceilings a dim tint so a big background "
            "plane doesn't flood the frame. The table and floor also get an axis-aligned bbox "
            "outline; walls are NOT outlined — a wall shows as its true (often diagonal) footprint "
            "band. Objects are dim-gray for context. A yellow dot + arrow marks the REAL camera "
            "position and view direction (clamped to the frame edge, with its distance, when it sits "
            "outside the square); the world +X (red) / +Y (green) axes are bottom-left; the view is CAMERA-ALIGNED (180° vs a raw top-down): image-UP = -Y (deeper into the scene, camera toward the bottom looking up) and image-RIGHT = -X (the camera's right, same as the photo where +X is image-LEFT). READ-ONLY — "
            "it does NOT modify or save the scene. Use it to verify each wall is on the correct side "
            "of the table relative to the camera and matches the photo — corner/perpendicular "
            "relationships only fix relative angles, NOT which side — then correct a "
            "flipped/mis-placed surface with execute_and_evaluate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "size_m": {
                    "type": "number",
                    "description": "Optional side length (metres) of the square BEV region. Omit to "
                    "auto-fit to the scene's surfaces + objects.",
                },
            },
            "required": [],
        },
    },
}

build_root_surface_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "build_root_surface",
        "description": (
            "Initializer-only transactional tool for introducing a genuinely missing "
            "ROOT surface. Version 1 accepts only surface_type='wall'. Use it sparingly, "
            "only when the target clearly contains a distinct architectural wall plane "
            "that is absent from the CURRENT REGISTERED roots. Do not use it to build a "
            "registered wall, extend an existing plane, make a window/opening/frame, or "
            "add wall detail: use execute_and_evaluate for those. The backend allocates "
            "the graph id and exact Blender build name, creates a basic vertical wall, "
            "atomically registers it in the active scene graph and audit ledger, and "
            "returns a render. You may then refine the returned registered root with "
            "execute_and_evaluate. Never assign a new root id or edit scene_graph.json "
            "yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "surface_type": {
                    "type": "string",
                    "enum": ["wall"],
                    "description": "Version 1 supports only a missing architectural wall.",
                },
                "description": {
                    "type": "string",
                    "description": "Short visual identity of the missing wall plane.",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Why the current registered roots cannot represent this distinct "
                        "plane; mention the target evidence and existing wall(s) considered."
                    ),
                },
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Initial WORLD-space [x,y,z] center in metres.",
                },
                "normal_xy": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Horizontal WORLD-space normal [nx,ny] of the wall face.",
                },
                "width_m": {"type": "number", "description": "Initial wall width."},
                "height_m": {"type": "number", "description": "Initial wall height."},
                "thickness_m": {
                    "type": "number",
                    "description": "Initial wall thickness; normally 0.04-0.10m.",
                },
                "target_region_norm": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Normalized target-image [x0,y0,x1,y1] region containing the "
                        "missing plane, with origin at the image top-left, x increasing "
                        "right, and y increasing down (the checker convention). This is "
                        "retained for POSE visibility checking."
                    ),
                },
                "copy_material_from": {
                    "type": "string",
                    "description": (
                        "Optional exact build name of a CURRENT REGISTERED root whose "
                        "first material should seed the new wall."
                    ),
                },
                "floor_support": {
                    "type": "string",
                    "description": (
                        "Optional exact graph id or build name of a registered floor/ground. "
                        "When supplied, UNDER(floor,new wall) is registered as hard."
                    ),
                },
                "relationship_hints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["corner", "perpendicular", "against"],
                            },
                            "other": {
                                "type": "string",
                                "description": "Exact graph id or build name of another root.",
                            },
                        },
                        "required": ["type", "other"],
                    },
                    "description": (
                        "Optional visual hints. They are registered as ADVISORY only and "
                        "do not create hard yaw/contact authority."
                    ),
                },
            },
            "required": [
                "surface_type",
                "description",
                "reason",
                "center",
                "normal_xy",
                "width_m",
                "height_m",
                "thickness_m",
                "target_region_norm",
            ],
        },
    },
}

remove_root_surface_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "remove_root_surface",
        "description": (
            "Initializer-only transactional removal of a root previously introduced by "
            "build_root_surface. Use it when later visual comparison shows that the added "
            "plane was unnecessary or duplicates an existing root. It refuses to remove "
            "any preprocessing-provided root, atomically removes the Blender root and its "
            "registered graph relationships, records the audit event, invalidates rules, "
            "and returns a render."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "surface": {
                    "type": "string",
                    "description": (
                        "Exact graph id or exact build name returned by build_root_surface."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why the runtime-added root is being withdrawn.",
                },
            },
            "required": ["surface", "reason"],
        },
    },
}

nudge_object_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "nudge_object",
        "description": (
            "Initialization-only, transactional exception for translating exactly one "
            "imported object (plus object descendants it supports) by a small WORLD-space "
            "[dx,dy,dz] offset. Use only for a rules-reported penetration/resting defect, "
            "or a clear, visually significant position mismatch in the paired "
            "reference/render. The "
            "backend preserves object identity, mesh, scale, rotation, materials, hierarchy "
            "and visibility; checks resting/support and new or worsened penetration; and "
            "rolls back an unsafe result. Calls and cumulative distance are capped. Use the "
            "backend's small clearance candidates when reported. Numerically clearing "
            "contacts within 0.25 mm are tolerated. Confirmed physical repairs have a "
            "separate budget: 200 mm per call, two calls per target, 16 total; visual "
            "corrections remain limited to two distinct objects/stacks. Use the "
            "smallest correction, inspect the automatically attached current render, then "
            "rerun the rules. Never use this to hide a wrong root surface."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object": {
                    "type": "string",
                    "description": (
                        "One imported object, using either its scene-graph id (mug#0) "
                        "or exact Blender mesh name (obj_mug_0)."
                    ),
                },
                "translation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "Small WORLD-space [dx, dy, dz] translation in metres.",
                },
                "reason": {
                    "type": "string",
                    "enum": [
                        "penetration",
                        "resting_defect",
                        "visible_position_error",
                    ],
                    "description": "The evidence authorizing this exceptional correction.",
                },
            },
            "required": ["object", "translation", "reason"],
        },
    },
}


# GPT-6 harness tools are kept out of every baseline menu. Their direct Blender code is
# executed inside a backend-owned transaction: ``TARGET_OBJECT_NAME`` and
# ``TARGET_OBJECT_ID`` are injected Python globals, while identity and scene artifacts
# remain backend-owned.
_RUNTIME_PHYSICAL_MATERIALS = [
    "aluminum",
    "cardboard",
    "carbon_fiber",
    "ceramic",
    "clay",
    "concrete",
    "cork",
    "fabric",
    "foam",
    "fruit",
    "glass",
    "leather",
    "paper",
    "plastic",
    "rubber",
    "silicone",
    "steel",
    "stone",
    "wood",
]


edit_composition_object_mesh_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "edit_object_mesh",
        "description": (
            "Composition-only transactional replacement of ONE materially "
            "wrong object after a successful investigate_objects call for that "
            "object. Supply a complete Blender Python script that deletes the bound "
            "target tree and creates exactly one parentless EMPTY named "
            "TARGET_OBJECT_NAME with one or more nonempty MESH descendants. The "
            "backend injects TARGET_OBJECT_NAME and TARGET_OBJECT_ID, rejects every "
            "change outside that target, exports and round-trip checks a durable "
            "textured GLB, replaces the maskless runtime revision and material/physics "
            "records, then force-rebuilds its collider and settles the target with its "
            "graph dependents. Use ONLY when the object cannot stand as reconstructed "
            "(physics reports it toppled or unsupported and no pose edit can make it "
            "rest), its reconstruction is missing a critical structural part the "
            "target clearly shows, or its reconstructed proportions are impossible for "
            "what it is (a 5 cm thick bread slice) so no rigid pose or uniform rescale can "
            "match the photo; cite that evidence. The replacement must be WATERTIGHT: every "
            "authored part a closed manifold solid, parts overlapping into ONE connected solid "
            "(exact Boolean union; open shells, non-manifold edges and disconnected solids are "
            "rejected). Do not copy, flatten or remesh the reconstructed mesh in place — it is an "
            "open fragment soup and that fails or times out; rebuild from closed primitives at "
            "the photo's proportions. Never for a pose-only, "
            "size-only, texture-only, or minor-detail mismatch — those are pose edits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "Complete runnable Blender Python. Delete only the bound "
                        "target tree, then create one parentless EMPTY root named "
                        "TARGET_OBJECT_NAME with nonempty MESH descendants. Do not "
                        "save/export, mutate any other object or surface, or write "
                        "scene artifacts."
                    ),
                },
                "object": {
                    "type": "string",
                    "description": (
                        "Exact scene-graph id previously passed successfully to "
                        "investigate_objects (preferred), or its canonical obj_* name."
                    ),
                },
                "physical_material_hint": {
                    "type": "string",
                    "enum": _RUNTIME_PHYSICAL_MATERIALS,
                    "description": (
                        "Optional fallback-only contact material; the backend refreshes "
                        "the normal VLM estimate first."
                    ),
                },
                "expected_resting_mode": {
                    "type": "string",
                    "enum": ["preserve", "side", "free"],
                    "default": "preserve",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Specific target/current evidence from investigate_objects "
                        "showing that the mesh itself is materially wrong."
                    ),
                },
            },
            "required": ["code", "object", "reason"],
            "additionalProperties": False,
        },
    },
}

_POSE_EDIT_SPEC: dict[str, object] = {
    "type": "object",
    "properties": {
        "object": {
            "type": "string",
            "description": "Exact scene-graph id (preferred) or canonical obj_* name.",
        },
        "mode": {"type": "string", "enum": ["delta", "absolute"]},
        "translation_m": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "World-space logical-handle bounds-center translation: offset in "
                "delta mode, target center in absolute mode. Use null for no translation."
            ),
        },
        "rotation_euler_deg": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "XYZ world-frame Euler rotation about the logical-handle bounds "
                "center: offset in delta mode, attitude in absolute mode. "
                "Use null when selecting quaternion or no rotation."
            ),
        },
        "rotation_quaternion_wxyz": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
            "description": (
                "Alternative world-frame rotation about the logical-handle bounds "
                "center; mutually exclusive with Euler. Use null when selecting Euler or no rotation."
            ),
        },
        "expected_resting_mode": {
            "type": "string",
            "enum": ["preserve", "side", "free"],
            "default": "preserve",
        },
    },
    "required": [
        "object",
        "mode",
        "translation_m",
        "rotation_euler_deg",
        "rotation_quaternion_wxyz",
        "expected_resting_mode",
    ],
    "additionalProperties": False,
}


edit_object_poses_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "edit_object_poses",
        "strict": True,
        "description": (
            "Composition direct-edit route for one coordinated batch of explicit "
            "object poses. Use it as a first-class peer of move() whenever the target crops "
            "make the correction clear. Each requested matrix is only a PHYSICS INITIAL "
            "CONDITION: the backend simulates the edited objects and their dependencies, "
            "returns settled matrices/drift and post-simulation visual feedback, and rejects+rolls back "
            "on unavailable/non-converged/unsafe settlement. Committed maskless objects "
            "in the physical scene can be posed by exact scene-graph ID without investigation "
            "or move(). Their Empty root and all Mesh parts move as one rigid object. "
            "Masked-only edits return a crop; any maskless edit returns a full-scene "
            "post-settle render rather than a partial masked-only crop."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 12,
                    "items": _POSE_EDIT_SPEC,
                },
                "reason": {
                    "type": "string",
                    "description": "Concrete target mismatch addressed by this coordinated edit.",
                },
            },
            "required": ["edits", "reason"],
            "additionalProperties": False,
        },
    },
}

get_scene_info_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "get_scene_info",
        "description": 'Read the CURRENT state of the Blender scene: objects (name, transform, visibility, world-space bbox, material_slots), materials (Principled base_color/roughness/metallic, or "textured"; imported materials explicitly non-editable), lights, cameras, world and color_management. Includes actual Blender runtime_capabilities, root node input names, and diagnostic lighting_material_limits. Call when the current state is needed; the stage prompt decides whether its entry preseed is already sufficient.',
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

check_rules_enforced_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "check_rules_enforced",
        "description": (
            "Read-only check of the scene rules, evaluated as an ORDERED LADDER and reported "
            "ONE level at a time: (1) STRUCTURE — imported-object integrity/authorized "
            "nudges + every CURRENT REGISTERED root's exact identity/valid transform + "
            "the main support itself (top at z=0, parts connected, has a bottom); "
            "(2) POSE — each CURRENT REGISTERED surface's "
            "reference-view coverage must match its photo segmentation mask (catches an "
            "oversized/undersized table, a wall out of frame; missing required evidence "
            "is UNVERIFIED and fails this level) + " + OBJECT_SUPPORT_CONTRACT + " "
            "(3) CONTACT — penetration + resting (no object left floating above what is "
            "beneath it) + surface relationships. A FAIL shows only the current "
            "level with the exact fix — apply it, call again, and the next level unlocks; call "
            "end only when ALL levels PASS. A full rules pass reports rules_all_pass "
            "separately from completion_ready: if advisory_requirement.state is required, "
            "resolve its issued candidate IDs before ending. Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

resolve_yaw_advisory_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "resolve_yaw_advisory",
        "description": (
            "Read-only initializer tool for resolving the CURRENT residual main-support "
            "yaw advisory after check_rules_enforced has passed all three levels. Use "
            "only candidate IDs and the opaque observation token issued by that exact "
            "gate response. The backend—not you—calculates edge directions, angular "
            "error, consistency, and pass/mismatch. Never supply coordinates, angles, "
            "tolerances, confidence, or a conclusion. A scene edit makes the token stale."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "advisory_id": {"type": "string"},
                "observation_token": {"type": "string"},
                "method": {
                    "type": "string",
                    "enum": ["two_edge_families"],
                },
                "edge_matches": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "target_edge_id": {"type": "string"},
                            "built_edge_id": {"type": "string"},
                        },
                        "required": ["target_edge_id", "built_edge_id"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "advisory_id",
                "observation_token",
                "method",
                "edge_matches",
            ],
            "additionalProperties": False,
        },
    },
}

bypass_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "bypass",
        "description": (
            'Waive the coverage report (rules level "2/3 POSE") for specific ROOT '
            "SURFACES whose ground-truth segmentation mask is itself wrong, or for a "
            "PREPROCESSED REGISTERED wall that the target genuinely does not depict. "
            "Runtime-added walls cannot be bypassed: fix one or remove it with "
            "remove_root_surface. When a "
            "coverage failure fires you receive per-surface side-by-side images: LEFT "
            "YOUR RENDER with your built surface tinted, RIGHT the PHOTO with that "
            "surface's GT mask tinted the same color. Call this tool ONLY when the PHOTO-side "
            "mask is visibly wrong (e.g. the tint covers a fraction of the real "
            "curtain — segmentation under-shoots large soft surfaces), OR when the "
            "flagged preprocessed registered wall is genuinely absent from the target. "
            "Do NOT use it to skip fixing a "
            "real size/yaw/position error — if your BUILT surface is the wrong one, "
            "fix it instead. A bypass never edits the scene graph and does not waive "
            "the requirement to build every CURRENT REGISTERED wall. USE SPARINGLY: "
            "every bypass permanently removes a safety "
            "check for this stage and is recorded in the run record. Bypassing the "
            "MAIN SUPPORT is allowed but is almost never right — every object "
            "placement is anchored to it; waive it only when its mask is unmistakably "
            "broken AND your build matches the photo by eye. Accepts only "
            "root-surface ids named in the CURRENT coverage failure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "obj_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "root-surface ids exactly as named in the coverage report, "
                        'e.g. ["curtain_0"]'
                    ),
                }
            },
            "required": ["obj_ids"],
        },
    },
}

# Composition variant: two rules, no surface checks — and coverage SHORT-CIRCUITS the
# expensive penetration pass, which the description must say (an agent that read "flat"
# would take a coverage-only FAIL as "penetration is fine").
check_rules_enforced_composition_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "check_rules_enforced",
        "description": (
            "Read-only check of the composition-stage rules, in this order: "
            "(1) investigation coverage — EVERY object must have appeared in at least one "
            "investigate_objects call, and every retained rotate_180 must have a successful "
            "fresh post-flip investigation; (2) no interpenetrating bodies (object<->object and "
            "object<->surface). Coverage is checked FIRST and SHORT-CIRCUITS: while it "
            "fails, the penetration pass is skipped entirely and the report says so "
            "('penetration not checked yet') — so a coverage FAIL is NOT evidence the "
            "geometry is clean. Investigate what it lists, then call again to get the "
            "penetration verdict. A FAIL lists each violation with the fix; call end only "
            "when ALL RULES PASS. Takes no arguments."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

investigate_objects_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "investigate_objects",
        "description": (
            "Zoom in on one or more OBJECTS (never root surfaces): returns TWO images — "
            "IMAGE 1 is the current RENDER crop and IMAGE 2 is the reference PHOTO crop, "
            "with each listed object outlined by the SAME colored box (+id) in both — "
            "from the same camera and crop window (~2x the objects' extent), plus each object's "
            "silhouette IoU. Compare the content inside matching boxes: position, size, "
            "and which way the object points. A de-occluded pair may hide CONFIRMED "
            "cargo to expose its underlying support; lateral/ambiguous or uncertain "
            "neighbours remain visible in the original contextual photo/render so "
            "contact, leaning, spacing, and occlusion can be judged. Investigating ARMS the listed "
            "objects for move(); the armed set is REPLACED by your next investigate call "
            "and cleared by any manual scene edit. Each object may be investigated at most "
            "K times (the tool tells you the count). A retained rotate_180 creates a "
            "mandatory post-flip investigation: the next investigate call must include "
            "that object, and exactly one such visit may exceed K when its ordinary visits "
            "are exhausted. The requirement is satisfied only after both crops and fresh "
            "hints finish successfully; the response then carries a structured "
            "resolved_followup marker. Every object must be investigated at "
            "least once before you may end. AT MOST 2-3 spatially-close objects per "
            "call (the crop spans all of them — a wide group makes each object too "
            "small to judge); small cutlery-scale objects alone or in pairs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "SCENE-GRAPH object ids as category#index, using the "
                    "category's EXACT text INCLUDING SPACES — e.g. ['mug#0', "
                    "'coffee machine#0'] (get_scene_info gives each object's 'id' "
                    "field; copy it verbatim). NOT the Blender object name "
                    "obj_coffee_machine_0, which underscores the spaces.",
                }
            },
            "required": ["objects"],
        },
    },
}

move_object_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "move",
        "description": (
            "Fix ONE aspect of ONE object's pose. You choose only WHAT to fix — the "
            "backend computes the exact amount/direction against the reference photo "
            "and applies it ONLY if it improves the match (it also keeps the object out of "
            "penetration and resting on its support automatically). Aspects: 'xy' (DEFAULT "
            "for placement errors — computes "
            "the planar alignment, including diagonals, and refines around it), 'x'/'y' "
            "(axis-constrained slides; use only when you specifically want one axis, or to "
            "disambiguate depth from scale — pair 'y' with 'scale'), 'rotation' (spin "
            "about vertical; use it for a visually apparent smaller yaw mismatch even when "
            "investigate_objects emitted no YAW HINT), 'rotate_180' (ONE-SHOT 180-deg "
            "YAW about the object's center — "
            "spins in place about the vertical axis, NOT an upside-down flip — applied "
            "REGARDLESS of the match score: a near-symmetric object's silhouette cannot "
            "tell front from back. Use it only when the investigate crops show the object "
            "semantically backwards — a spoon/fork/knife/chopstick whose handle points the "
            "wrong way, a mug handle, a chair, a screen. If you suspect ~180, call it "
            "FIRST, before any other move on that object — refine position/scale only "
            "after the facing is right. At most ONE applied flip per object, ever "
            "— a kept or undone flip consumes the one-shot; a flip that physics "
            "auto-rejects and reverts does NOT (one free retry — the second rejection "
            "consumes it). "
            "Anything stacked on it stays in place. "
            "A retained flip clears the armed set and returns a structured "
            "required_followup: immediately investigate that object again, or undo that "
            "exact flip. Follow with 'rotation' if slightly off), 'scale' (resize — use when a "
            "CENTERED object still over/under-fills its reference region; SAM3D sizes "
            "are often 10-25 percent off, so check size for EVERY object; pair with "
            "'y' to disambiguate too-small from too-far. NOTE: objects in a same-size "
            "category have a LOCKED size — 'scale' is rejected for them; fix such an "
            "object with 'xy'/'rotation' instead). The object "
            "must be ARMED (present "
            "in your latest investigate_objects call); you may chain several moves on "
            "armed objects, but after investigating something else you must re-investigate "
            "before moving it again."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "object": {
                    "type": "string",
                    "description": "SCENE-GRAPH object id as "
                    "category#index, using the category's EXACT text INCLUDING SPACES — e.g. "
                    "'mug#0' or 'coffee machine#0' (get_scene_info's 'id' field, verbatim) — "
                    "NOT the Blender name obj_coffee_machine_0, which underscores the spaces",
                },
                "aspect": {
                    "type": "string",
                    "enum": ["xy", "x", "y", "rotation", "rotate_180", "scale"],
                },
            },
            "required": ["object", "aspect"],
        },
    },
}

undo_last_step_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "undo_last_step",
        "description": (
            "If you believe that your last action did not improve the current state, but "
            "instead moved it further away from the target state, you can call this tool "
            "to undo the last action. Undoing build_root_surface/remove_root_surface "
            "restores the Blender scene, active scene graph, relationships, and audit "
            "authorization state together. When the last action is the exact retained "
            "rotate_180 awaiting mandatory investigation, a successful undo cancels that "
            "requirement and returns a structured resolved_followup marker; it does not "
            "refund the one-shot or move budget."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_POST_FLIP_FOLLOWUP_TYPE = "post_rotate_180_investigation"
_POST_FLIP_ALLOWED_TOOLS = ["investigate_objects", "undo_last_step"]
_INITIALIZER_LEVEL_CHECK_NAMES = (
    (
        "object_integrity",
        "surface_geometry",
        "registered_root_inventory",
        "runtime_root_bindings",
        "main_support_top",
        "main_support_connected",
        "main_support_bottom",
        "relationship_constraints",
    ),
    ("coverage", "objects_inboard", "relationship_constraints"),
    ("penetration", "objects_resting", "relationship_constraints"),
)

mcp = FastMCP("blender-executor")

# Global executor instance
_executor: Optional["Executor"] = None


def _guard_pending_composition_followup(
    tool_name: str, *, object_ids: Optional[list[str]] = None
) -> Optional[dict]:
    """MCP-entry guard mirroring Executor's direct-call boundary."""
    if _executor is None:
        return None
    guard = getattr(_executor, "_guard_post_flip_tool_call", None)
    return guard(tool_name, object_ids=object_ids) if guard is not None else None


def isaac_penetration_gate_pairs(
    pairs: list, support: dict, mesh_pen_keys: set
) -> list:
    """Composition rules-gate filter over the Isaac hull contact pairs (depth in m,
    the min single-direction push to separate — see cmd_contacts). Hulls are FATTER
    than the visual meshes (CoACD convex parts fill concavities + voxel overshoot),
    so every RESTING contact overlaps a few mm:
    - sub-tolerance depths (<= 8mm) drop;
    - siblings RESTING against each other on the same support drop up to contact
      scale (20mm: nestled croissants read 10-20mm); deeper still reports;
    - an object vs its OWN support drops at ANY hull depth (physics settles those
      contacts by design; a mug seated in a shelf compartment overlaps the
      compartment-filling convex parts arbitrarily deep) — UNLESS the pair is also
      in ``mesh_pen_keys`` (the Blender BVH mesh-intersection pairs from the same
      check): an object whose VISUAL mesh intersects its support is a real defect
      the hull exemption must not hide (0720_leanfix_abc1: a spoon buried 28mm
      into its placemat passed the gate)."""
    out = []
    for p in pairs:
        a, b, d = p["a"], p["b"], p["depth"]
        if d <= HULL_SUBTOL_M:
            continue
        if support.get(a) == b or support.get(b) == a:
            if frozenset((a, b)) in mesh_pen_keys:
                out.append(p)
            continue
        if (
            support.get(a) is not None
            and support.get(a) == support.get(b)
            and d <= HULL_SIBLING_TOL_M
        ):
            continue
        out.append(p)
    return out


class Executor:
    """Manages Blender script execution and rendering.

    Handles the lifecycle of Blender script execution including file management,
    subprocess invocation, and result collection.

    Attributes:
        blender_command: Path to the Blender executable.
        blender_file: Path to the .blend file to operate on.
        blender_script: Path to the wrapper script that executes user code.
        script_path: Directory to save generated scripts.
        render_path: Directory to save rendered images.
        blender_save: Optional path to save the Blender state after execution.
        gpu_devices: Comma-separated GPU device IDs (e.g., "0,1").
        count: Counter for executed scripts.
    """

    def __init__(
        self,
        blender_command: str,
        blender_file: str,
        blender_script: str,
        script_save: str,
        render_save: str,
        blender_save: Optional[str] = None,
        target_image_path: Optional[str] = None,
        gpu_devices: Optional[str] = None,
        render_engine: Optional[str] = None,
        moge_dir: Optional[str] = None,
        root_stage_name: Optional[str] = None,
        stage_dir: Optional[str] = None,
        attempt_idx: Optional[int] = None,
        initializer_baseline_blend: Optional[str] = None,
        initializer_ledger_path: Optional[str] = None,
        harness_profile: str = "baseline",
        harness_profile_manifest: Optional[dict] = None,
        model: Optional[str] = None,
        reconstruction_backend: Optional[str] = None,
    ) -> None:
        """Initialize the Blender executor.

        Args:
            blender_command: Path to Blender executable.
            blender_file: Path to .blend file.
            blender_script: Path to wrapper script.
            script_save: Directory to save scripts.
            render_save: Directory to save renders.
            blender_save: Optional path to save Blender state.
            target_image_path: Optional target image path for render aspect ratio.
            gpu_devices: Optional GPU device IDs.
            render_engine: Optional Blender render engine name.
            moge_dir: Scene-artifact directory containing depth/scene-graph data.
            root_stage_name: Stage name selecting tool menus and rule paths.
        """

        # Blender runs with cwd = the stage output dir (see _execute_blender), so a
        # script's bare relative write lands inside the run instead of the repo root
        # (0807: wall_sketch_tex.png). Every path must therefore be absolute — the
        # launcher passes them all relative to the project root. blender_command may
        # be a bare PATH name ("blender"): only absolutize it when it has a slash.
        def _abs(p: Optional[str]) -> Optional[str]:
            return os.path.abspath(p) if p else p

        self.blender_command = (
            _abs(blender_command)
            if os.sep in (blender_command or "")
            else blender_command
        )
        self.blender_file = _abs(blender_file)
        self.blender_script = _abs(blender_script)
        self.script_path = Path(os.path.abspath(script_save))
        self.render_path = Path(os.path.abspath(render_save))
        self.blender_save = _abs(blender_save)
        self.target_image_path = _abs(target_image_path)
        self.gpu_devices = gpu_devices
        self.render_engine = render_engine or "CYCLES"
        self.harness_profile = str(harness_profile or "baseline")
        self.harness_profile_manifest = copy.deepcopy(harness_profile_manifest or {})
        self.physics_model = str(model or "claude-opus-5")
        self.reconstruction_backend = str(reconstruction_backend or "sam3d")
        self.moge_dir = _abs(
            moge_dir  # for the support hierarchy (scene_graph.json/placement.json)
        )
        self._pipeline_object_names: list[str] = []
        if self.moge_dir:
            placement_path = os.path.join(self.moge_dir, "placement.json")
            try:
                with open(placement_path) as placement_file:
                    placement = json.load(placement_file)
                self._pipeline_object_names = [
                    str(record["mesh_name"])
                    for record in placement.get("objects", [])
                    if record.get("mesh_name")
                ]
                if (
                    not self._pipeline_object_names
                    and not self._has_committed_empty_object_inventory()
                ):
                    raise ValueError("placement table contains no mesh names")
                if len(self._pipeline_object_names) != len(
                    set(self._pipeline_object_names)
                ):
                    raise ValueError("placement table contains duplicate mesh names")
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # An interrupted typed removal can leave the new (possibly empty)
                # placement committed while its Blend/graph transaction is still only
                # prepared. Let the opt-in recovery hook below restore the snapshot
                # before enforcing the normal placement invariant. Baseline and every
                # run without a nonterminal marker keep the original fail-fast path.
                pending_runtime_recovery = False
                typed_recovery_enabled = (
                    root_stage_name in {"initializer", "composition"}
                    and _profile_capability_enabled(
                        self.harness_profile,
                        self.harness_profile_manifest,
                        "mutation_journal",
                    )
                    and (
                        (
                            root_stage_name == "composition"
                            and _profile_capability_enabled(
                                self.harness_profile,
                                self.harness_profile_manifest,
                                "composition_mesh_edit",
                            )
                        )
                        or (
                            root_stage_name == "initializer"
                            and _profile_capability_enabled(
                                self.harness_profile,
                                self.harness_profile_manifest,
                                "initializer_code_transactions",
                            )
                        )
                    )
                )
                if typed_recovery_enabled:
                    marker_root = (
                        Path(self.moge_dir) / "runtime_objects" / "transactions"
                    )
                    for marker_path in marker_root.glob("tx_*/recovery.json"):
                        try:
                            marker = json.loads(marker_path.read_text())
                            marker_stage = marker.get("stage")
                            if marker_stage not in {None, root_stage_name}:
                                continue
                            marker_state = str(marker.get("state") or "")
                        except (OSError, ValueError, TypeError, AttributeError):
                            marker_state = "invalid"
                        if marker_state not in {"committed", "rolled_back", "undone"}:
                            pending_runtime_recovery = True
                            break
                if (
                    root_stage_name
                    in {
                        "initializer",
                        "texture",
                        "lighting",
                        "composition",
                    }
                    and not pending_runtime_recovery
                ):
                    raise RuntimeError(
                        f"cannot load pipeline object identities from {placement_path}"
                    ) from exc
                # Some non-static-scene tests/tools have no placement table. The guard is
                # active whenever the canonical object list is available.
                self._pipeline_object_names = []
        self.root_stage_name = (
            root_stage_name  # stage tool menu + stage-only rule paths
        )
        self.attempt_idx = int(attempt_idx or 1)
        self.count = 0
        # Undo support: a stack of state.blend snapshots, one per mutating edit
        # (decoupled from self.count, which is also bumped by read-only/render ops).
        # base_state captures the scene before the first edit so the first edit is undoable.
        self.edit_history: list[str] = []
        self.base_state: Optional[str] = None
        # Ledger snapshots mirror edit_history. They make undo restore both the .blend
        # and initializer object-transaction counters/authorizations atomically.
        self._ledger_history: list[dict] = []
        # The initializer may register a missing root surface at runtime.  Graph
        # snapshots therefore mirror the same chronological edit stack: undoing a
        # later execute keeps the registration, while undoing the build/remove tool
        # restores the previous graph revision together with the blend and ledger.
        self._graph_history: list[Optional[bytes]] = []
        self._graph_base: Optional[bytes] = None
        self._edit_meta: list[dict] = []
        # GPT-6 typed object mutations snapshot every non-Blend artifact they own.
        # Kept as metadata on the chronological edit stack so legacy/default undo
        # behavior and storage remain exactly unchanged.
        # Coverage bypass (initializer): surfaces whose level-2 coverage the agent
        # waived after judging the GT segmentation mask itself wrong (the groot2
        # under-segmented curtain class). Only ids named in the CURRENT failure may
        # be waived; persisted to tmp/coverage_bypass.json for the audit trail.
        self._coverage_bypassed: set = set()
        # main-support yaw: last reported CORRECTION per surface, so the mod-90 fold
        # cannot flip the reported sign between rounds (see pin_yaw_delta).
        self._yaw_target_rep: dict = {}
        self._coverage_last_failing: set = set()
        self._yaw_evidence: dict = {}
        # Residual yaw advisory is a completion requirement, not a relationship or
        # rule constraint. It is minted only after STRUCTURE+POSE+CONTACT all pass and
        # is bound to the exact source/constraint/camera/built-top/scene snapshot.
        self._yaw_advisory_requirement: Optional[dict] = None
        self._yaw_advisory_binding: Optional[dict] = None
        self._yaw_resolution: Optional[dict] = None
        self._relationship_evidence: list[dict] = []
        self._initializer_constraint_artifact: Optional[dict] = None
        self._constraint_results: list[dict] = []

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)
        (self.render_path.parent / "tmp").mkdir(parents=True, exist_ok=True)
        # Pose-refinement session (composition stage): lazy PoseSession + the
        # investigate/move bookkeeping. armed = objects from the MOST RECENT
        # investigate call (moves may chain while armed; a new investigate replaces
        # the set; a manual scene edit clears it and marks the session stale).
        self._pose_session = None
        self._pose_dirty = False
        # Isaac settles every committed composition move on a persistent server
        # (weld-carried subtrees, micro budget).
        self._physics_obj = None
        self._investigated: dict[str, int] = {}
        self._armed: set = set()
        self._flipped: set = set()  # rotate_180 is one-shot per object
        # Objects whose rotate_180 was already physics-rejected once: the first
        # CLEAN rejection (scene auto-reverted) refunds the one-shot, the second
        # consumes it for good — see move_object's rotate_180 branch.
        self._flip_phys_rejected: set = set()
        # A retained semantic flip invalidates every measurement made at its old
        # pose. This record is deliberately separate from ``_flipped``: the latter
        # is the one-shot ledger (undo never refunds it; only a first clean
        # physics rejection or a durably restored strict infrastructure failure
        # does), while this one is a resolvable follow-up tied to the exact
        # undo-stack edit token.
        self._post_flip_investigation_required: Optional[dict] = None
        self._post_flip_followup_events: list[dict] = []
        self._edit_token_seq = 0
        self._last_undo_resolved_followup: Optional[dict] = None
        self._investigate_n = 0
        self.investigate_cap = 3
        # 2026-09-15 final-settle repair round: the coverage gate is scoped to the
        # objects the final free-settle moved (None = every eligible object).
        self.coverage_scope: set | None = None
        # Per-object move budget: every executed move call counts (accepted, dead,
        # or physics-rejected), except a strict infrastructure error whose pre-edit
        # state was durably restored. Usage is surfaced in the OBJECT STATE table;
        # only the cap-refusal reports it directly in move feedback.
        self._moved: dict[str, int] = {}
        self.move_cap = 5
        # Committed edit_object_poses per object id (undo does not refund); the typed
        # tool is the exception to move(), not a substitute (2026-09-14 census: 285 typed
        # calls displaced ~47% of move() calls, 31% were rolled back or undone).
        self._typed_pose: dict[str, int] = {}
        self.typed_pose_cap = 2
        # Freeform-relocation budget: execute_and_evaluate relocations are physics-
        # settled + reported (the escape hatch for placements move() can't express, e.g.
        # lifting an object onto a surface it's boxed out of). Soft-capped per object so
        # it stays a fallback, not a way to churn around the disciplined move tool.
        self._freeform: dict[str, int] = {}
        self.freeform_cap = 3
        # Eager physics warmup: the PoseSession + Isaac boot used to run lazily
        # inside the FIRST investigate, stalling it for minutes while the opening
        # render + LLM rounds had already idled by. Boot in a background thread at
        # stage entry instead; _pose_session_get joins it before first use. All the
        # work is subprocess-side (register server, Isaac, spawn cooks) — nothing
        # here touches the main blender.
        self._warm_thread = None
        # Initializer object transactions use ONE immutable entry artifact and ONE
        # ledger across all generator attempts. The deterministic stage_dir paths let
        # a fresh MCP Executor in attempt 2 reload attempt 1's authorizations while the
        # live scene itself continues to be refined rather than reset.
        self.initializer_baseline_blend: Optional[str] = None
        self.initializer_ledger_path: Optional[str] = None
        self._object_baseline: Optional[dict] = None
        self._initializer_ledger = self._empty_initializer_ledger()
        if root_stage_name == "initializer":
            tx_root = (
                Path(os.path.abspath(stage_dir))
                if stage_dir
                else self.render_path.parent
            )
            tx_dir = tx_root / "initializer_object_transactions"
            tx_dir.mkdir(parents=True, exist_ok=True)
            self.initializer_baseline_blend = _abs(initializer_baseline_blend) or str(
                tx_dir / "entry.blend"
            )
            self.initializer_ledger_path = _abs(initializer_ledger_path) or str(
                tx_dir / "ledger.json"
            )
            baseline = Path(self.initializer_baseline_blend)
            if not baseline.exists():
                source = Path(self.blender_file)
                if not source.exists():
                    raise RuntimeError(
                        f"cannot capture initializer object baseline: {source} is missing"
                    )
                tmp = baseline.with_suffix(".blend.tmp")
                shutil.copy2(source, tmp)
                os.replace(tmp, baseline)
                # The executor only ever opens this as a read-only query input.
                baseline.chmod(0o444)
            self._initializer_ledger = self._load_initializer_ledger()
            if self._typed_initializer_recovery_enabled():
                self._recover_incomplete_runtime_mutations()
                if (
                    not self._pipeline_object_names
                    and not self._has_committed_empty_object_inventory()
                ) or len(self._pipeline_object_names) != len(
                    set(self._pipeline_object_names)
                ):
                    raise RuntimeError(
                        "Object transaction recovery did not restore valid pipeline "
                        "object identities"
                    )
        elif self._composition_mesh_recovery_enabled():
            self._recover_incomplete_composition_mesh_mutations()
            placement_path = Path(self.moge_dir) / "placement.json"
            placement = json.loads(placement_path.read_text())
            names = [
                str(record["mesh_name"])
                for record in placement.get("objects", [])
                if isinstance(record, dict) and record.get("mesh_name")
            ]
            if (not names and not self._has_committed_empty_object_inventory()) or len(
                names
            ) != len(set(names)):
                raise RuntimeError(
                    "Composition recovery did not restore valid pipeline "
                    "object identities"
                )
            self._pipeline_object_names = names
        self._ledger_base = copy.deepcopy(self._initializer_ledger)
        self._graph_base = self._read_scene_graph_bytes()
        if root_stage_name == "composition":
            import threading

            def _warm() -> None:
                try:
                    self._pose_session_get()
                except Exception as exc:  # noqa: BLE001 - warmup is best-effort:
                    # reset so the first real call retries (and surfaces) the error
                    import sys as _sys

                    print(f"[physics-warmup] failed: {exc}", file=_sys.stderr)
                    self._pose_session = None
                    self._physics_obj = None

            self._warm_thread = threading.Thread(target=_warm, daemon=True)
            self._warm_thread.start()

    def _has_capability(self, capability: str) -> bool:
        return _profile_capability_enabled(
            getattr(self, "harness_profile", "baseline"),
            getattr(self, "harness_profile_manifest", None),
            capability,
        )

    def _has_committed_empty_object_inventory(self) -> bool:
        """Accept zero objects only when validated runtime removals account for them.

        An empty placement alone is not proof of an intentional removal. Validate
        its graph, retained source masks, tombstones, and mesh artifacts together;
        an unprofiled scene keeps the original nonempty-placement requirement.
        """
        if not self._has_capability("runtime_object_inventory") or not self.moge_dir:
            return False
        from lib.tools.geometry.inventory_contract import validate_scene_artifacts
        from lib.tools.geometry.runtime_object_repair import load_runtime_inventory

        overlay = load_runtime_inventory(self.moge_dir)
        if not overlay or overlay.get("objects") or not overlay.get("tombstones"):
            return False
        report = validate_scene_artifacts(self.moge_dir, allow_runtime_inventory=True)
        return not report["expected_blender_object_names"]

    def _execute_blender(
        self,
        script_path: str,
        render_path: str = "",
        *,
        persist_scene: bool = True,
        _typed_world_semantics: bool = False,
    ) -> tuple[bool, list[str], str, str]:
        """Execute a Blender script in background mode.

        Args:
            script_path: Path to the Python script to execute.
            render_path: Directory to save rendered images.
            persist_scene: Whether the wrapper may save Blender state. Read-only
                fallback renders pass ``False`` so temporary render settings cannot
                rewrite the live blend and stale a completion token.
            _typed_world_semantics: Private one-call capability for the pre/post
                integrity dumps of a GPT-6 composition mesh transaction. Ordinary
                composition execution must leave this false.

        Returns:
            Tuple of (success, image_paths, stdout, stderr).
        """
        cmd = [
            self.blender_command,
            "--background",
            self.blender_file,
            "--python-exit-code",
            "1",
            "--python-expr",
            f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})",
            "--python",
            self.blender_script,
            "--",
            script_path,
            render_path,
        ]
        wrapper_save = self.blender_save if persist_scene else None
        if wrapper_save or self.target_image_path:
            cmd.append(wrapper_save or "")
        if self.target_image_path:
            cmd.append(self.target_image_path)
        # Set environment variables to control GPU devices
        env = os.environ.copy()
        # Blender ignores PYTHONPATH by default. The explicit repo-only sys.path
        # above enables helpers without importing the main venv's site-packages.
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")

        # Ban blender audio error
        env["AL_LIB_LOGLEVEL"] = "0"
        env["VIGA_RENDER_ENGINE"] = self.render_engine
        env["GRASE_PIPELINE_OBJECT_NAMES"] = json.dumps(
            getattr(self, "_pipeline_object_names", [])
        )
        env["GRASE_ROOT_STAGE_NAME"] = str(self.root_stage_name or "")
        # These are capabilities, not user configuration. Never inherit ambient
        # values into an ordinary execute_and_evaluate subprocess: only the narrow
        # nudge transaction below may mint them for one exact call.
        env.pop("GRASE_INITIALIZER_NUDGE_NAMES", None)
        env.pop("GRASE_INITIALIZER_NUDGE_DELTA", None)
        env.pop("GRASE_RUNTIME_ROOT_BINDINGS", None)
        env.pop("GRASE_INITIALIZER_REMOVE_ROOT_BINDING", None)
        env.pop("GRASE_REGISTERED_ROOT_BUILD_NAMES", None)
        env.pop("GRASE_REGISTERED_ROOT_CATEGORIES", None)
        env.pop("GRASE_INITIALIZER_BUILD_ROOT_RESULT_PATH", None)
        env.pop("GRASE_INITIALIZER_FLOOR_HELPER_ALLOWED", None)
        env.pop("GRASE_TYPED_WORLD_SEMANTICS", None)
        env.pop("GRASE_TYPED_OBJECT_MUTATION", None)
        env.pop("GRASE_INITIALIZER_OBJECT_TRANSACTION", None)
        env.pop("GRASE_COMPOSITION_MESH_TRANSACTION", None)
        if not isinstance(_typed_world_semantics, bool):
            raise RuntimeError("invalid typed world-semantics capability")
        if _typed_world_semantics:
            if not self._composition_mesh_recovery_enabled():
                raise RuntimeError(
                    "typed world-semantics capability is unavailable for this call"
                )
            env["GRASE_TYPED_WORLD_SEMANTICS"] = "1"
        elif (
            getattr(self, "harness_profile", "baseline") == "gpt6_v1"
            and self.root_stage_name == "initializer"
        ):
            # Trusted subprocess capability: baseline snapshots retain their exact
            # historical schema and cost.  GPT-6 initializer dumps additionally
            # carry the canonical world record needed by typed transaction checks.
            env["GRASE_TYPED_WORLD_SEMANTICS"] = "1"
        registered_root_categories = self._registered_root_categories_for_wrapper()
        env["GRASE_REGISTERED_ROOT_BUILD_NAMES"] = json.dumps(
            sorted(registered_root_categories)
        )
        env["GRASE_REGISTERED_ROOT_CATEGORIES"] = json.dumps(registered_root_categories)
        if self._initializer_floor_helper_allowed():
            env["GRASE_INITIALIZER_FLOOR_HELPER_ALLOWED"] = "1"
        runtime_bindings = self._runtime_root_bindings_for_wrapper()
        if runtime_bindings:
            env["GRASE_RUNTIME_ROOT_BINDINGS"] = json.dumps(runtime_bindings)
        remove_binding = getattr(self, "_initializer_remove_root_allow", None)
        if remove_binding:
            env["GRASE_INITIALIZER_REMOVE_ROOT_BINDING"] = json.dumps(remove_binding)
        build_result_path = getattr(self, "_initializer_build_root_result_path", None)
        if build_result_path:
            env["GRASE_INITIALIZER_BUILD_ROOT_RESULT_PATH"] = str(build_result_path)
        allow_names = getattr(self, "_initializer_nudge_allow_names", None)
        allow_delta = getattr(self, "_initializer_nudge_allow_delta", None)
        if allow_names and allow_delta is not None:
            env["GRASE_INITIALIZER_NUDGE_NAMES"] = json.dumps(sorted(allow_names))
            env["GRASE_INITIALIZER_NUDGE_DELTA"] = json.dumps(allow_delta)
        initializer_transaction = getattr(self, "_initializer_object_transaction", None)
        composition_mesh_transaction = getattr(
            self, "_composition_mesh_transaction", None
        )
        if initializer_transaction is not None:
            if (
                getattr(self, "harness_profile", "baseline") != "gpt6_v1"
                or self.root_stage_name != "initializer"
                or composition_mesh_transaction is not None
                or not isinstance(initializer_transaction, dict)
            ):
                raise RuntimeError("invalid initializer object transaction capability")
            env["GRASE_INITIALIZER_OBJECT_TRANSACTION"] = json.dumps(
                initializer_transaction, sort_keys=True
            )
        if composition_mesh_transaction is not None:
            if (
                getattr(self, "harness_profile", "baseline") != "gpt6_v1"
                or self.root_stage_name != "composition"
                or not self._has_capability("composition_mesh_edit")
                or initializer_transaction is not None
                or not isinstance(composition_mesh_transaction, dict)
                or set(composition_mesh_transaction)
                != {"transaction_id", "added_objects", "removed_objects"}
                or not isinstance(
                    composition_mesh_transaction.get("transaction_id"), int
                )
                or int(composition_mesh_transaction["transaction_id"]) <= 0
                or not isinstance(
                    composition_mesh_transaction.get("added_objects"), list
                )
                or not isinstance(
                    composition_mesh_transaction.get("removed_objects"), list
                )
                or len(composition_mesh_transaction["added_objects"]) != 1
                or composition_mesh_transaction["added_objects"]
                != composition_mesh_transaction["removed_objects"]
                or not isinstance(composition_mesh_transaction["added_objects"][0], str)
                or not composition_mesh_transaction["added_objects"][0]
            ):
                raise RuntimeError("invalid composition mesh transaction capability")
            env["GRASE_COMPOSITION_MESH_TRANSACTION"] = json.dumps(
                composition_mesh_transaction, sort_keys=True
            )

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=BLENDER_TIMEOUT,
                cwd=str(self.script_path.parent),
            )
            return True, _list_render_images(render_path), proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            logging.error(f"Blender timed out after {BLENDER_TIMEOUT}s")
            return (
                False,
                [],
                e.stdout or "",
                (f"Blender timed out after {BLENDER_TIMEOUT}s and was killed."),
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, [], e.stdout, e.stderr

    def _execute_raw_blender(
        self,
        script_path: str,
        render_path: str = "",
        *,
        disable_autoexec: bool = False,
    ) -> tuple[bool, list[str], str, str]:
        """Execute a standalone Blender Python script without the normal wrapper."""
        cmd = [
            self.blender_command,
            "--background",
            *(["--disable-autoexec"] if disable_autoexec else []),
            self.blender_file,
            "--python-exit-code",
            "1",
            "--python",
            script_path,
            "--",
            render_path,
        ]
        env = os.environ.copy()
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices
            logging.info(f"Setting CUDA_VISIBLE_DEVICES to: {self.gpu_devices}")
        env["AL_LIB_LOGLEVEL"] = "0"
        env["VIGA_RENDER_ENGINE"] = self.render_engine

        try:
            proc = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=BLENDER_TIMEOUT,
                cwd=str(self.script_path.parent),
            )
            return True, _list_render_images(render_path), proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            logging.error(f"Blender timed out after {BLENDER_TIMEOUT}s")
            return (
                False,
                [],
                e.stdout or "",
                (f"Blender timed out after {BLENDER_TIMEOUT}s and was killed."),
            )
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e}")
            return False, [], e.stdout, e.stderr

    def _parse_code(self, full_code: str) -> str:
        """Strip markdown code fences from code if present."""
        if full_code.startswith("```python") and full_code.endswith("```"):
            return full_code[len("```python") : -len("```")]
        return full_code

    def _generate_scene_info_script(self) -> str:
        """Generate a script to extract scene information."""
        return generate_scene_info_script(
            str(self.render_path.parent / "tmp" / "scene_info.json")
        )

    def _ensure_undo_base(self) -> None:
        """Capture the pre-first-edit scene ONCE so the first edit — execute OR move —
        is undoable. Must run BEFORE the edit mutates blender_save."""
        if (
            self.base_state is None
            and self.blender_save
            and os.path.exists(self.blender_save)
        ):
            base_path = self.render_path.parent / "tmp" / "undo_base.blend"
            base_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.blender_save, base_path)
            self.base_state = str(base_path)
            self._ledger_base = copy.deepcopy(
                getattr(self, "_initializer_ledger", self._empty_initializer_ledger())
            )
            self._graph_base = self._read_scene_graph_bytes()

    def _scene_graph_path(self) -> Optional[Path]:
        if not getattr(self, "moge_dir", None):
            return None
        return Path(self.moge_dir) / "scene_graph.json"

    def _initializer_constraints_path(self) -> Optional[Path]:
        if not getattr(self, "moge_dir", None):
            return None
        from lib.tools.geometry.relationship_constraints import (
            INITIALIZER_CONSTRAINTS_FILENAME,
        )

        return Path(self.moge_dir) / INITIALIZER_CONSTRAINTS_FILENAME

    def _main_support_yaw_observation_path(self) -> Optional[Path]:
        if not getattr(self, "moge_dir", None):
            return None
        from lib.tools.geometry.main_support_yaw_observation import (
            MAIN_SUPPORT_YAW_OBSERVATION_FILENAME,
        )

        return Path(self.moge_dir) / MAIN_SUPPORT_YAW_OBSERVATION_FILENAME

    def _load_initializer_constraint_artifact(
        self, scene_graph: Optional[dict] = None
    ) -> dict:
        """Load and fully validate the current compiled initializer contract.

        There is intentionally no fallback to raw relationship rows.  A missing or
        digest-stale artifact means preprocessing/runtime-root commit was incomplete,
        so every relationship-derived gate must fail closed.
        """
        if self.root_stage_name != "initializer":
            return {"constraints": []}
        if not self.moge_dir:
            raise RuntimeError("initializer scene-artifact directory is unavailable")
        from lib.tools.geometry.relationship_constraints import (
            load_initializer_constraints,
        )

        graph = scene_graph
        if graph is None:
            path = self._scene_graph_path()
            if path is None or not path.is_file():
                raise RuntimeError("active scene_graph.json is unavailable")
            graph = json.loads(path.read_text())
        artifact = load_initializer_constraints(self.moge_dir, scene_graph=graph)
        self._initializer_constraint_artifact = artifact
        return artifact

    @staticmethod
    def _source_relationship_evidence(scene_graph: dict) -> list[dict]:
        """Expose immutable source verdicts without confusing them with build results."""
        return [
            {
                "relationship_id": rel.get("relationship_id"),
                "type": rel.get("type"),
                "a": rel.get("a"),
                "b": rel.get("b"),
                "status": rel.get("status"),
                "enforcement": rel.get("enforcement"),
                "runtime_status": "not_applicable",
                "reason": "source verdict; generated-scene fulfillment is reported by constraint_results",
            }
            for rel in scene_graph.get("relationships", []) or []
        ]

    def _read_scene_graph_bytes(self) -> Optional[bytes]:
        path = self._scene_graph_path()
        if path is None or not path.exists():
            return None
        return path.read_bytes()

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            tmp.write_bytes(payload)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    @classmethod
    def _atomic_write_json(cls, path: Path, payload: dict) -> None:
        cls._atomic_write_bytes(
            path,
            (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8"),
        )

    def _runtime_artifact_paths(self) -> list[Path]:
        """Mutable files owned by a GPT-6 object-repair transaction.

        The append-only journal is intentionally absent: rollback/undo records a new
        terminal event and never erases history.  scene_graph.json has its existing
        chronological snapshot channel and is likewise handled separately.
        """
        if not self.moge_dir:
            return []
        root = Path(self.moge_dir)
        return [
            root / "placement.json",
            root / "runtime_objects" / "inventory.json",
            root / "physics" / "physics_vlm.json",
            root / "physics" / "physics_estimate_manifest.json",
            root / "physics" / "blend_base.json",
            root / "physics" / "pose_changes.json",
        ]

    def _snapshot_runtime_artifacts(self) -> dict[Path, tuple[bool, bytes]]:
        return {
            path: (path.exists(), path.read_bytes() if path.exists() else b"")
            for path in self._runtime_artifact_paths()
        }

    def _restore_runtime_artifacts(
        self,
        snapshots: dict[Path, tuple[bool, bytes]],
        *,
        durable: bool = False,
    ) -> None:
        for path, (existed, payload) in snapshots.items():
            if existed:
                if durable:
                    self._durable_atomic_write_bytes(path, payload)
                else:
                    self._atomic_write_bytes(path, payload)
            else:
                was_present = path.exists()
                path.unlink(missing_ok=True)
                if durable and was_present:
                    self._fsync_directory(path.parent)

    def _mutation_journal_path(self) -> Path:
        if not self.moge_dir:
            raise RuntimeError("scene artifact directory is unavailable")
        return Path(self.moge_dir) / "audit" / "scene_mutations.jsonl"

    def _append_mutation_event(self, event: dict) -> None:
        """Durably append one JSONL event; audit failure is a transaction failure."""
        if not self._has_capability("mutation_journal"):
            raise RuntimeError("mutation_journal capability is not enabled")
        path = self._mutation_journal_path()
        self._durable_mkdir(path.parent)
        row = {
            "schema_version": 1,
            "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
            "harness_profile": self.harness_profile,
            "stage": self.root_stage_name,
            "attempt_idx": int(getattr(self, "attempt_idx", 1)),
            **event,
        }
        existed = path.exists()
        append_mutation_journal(path, row)
        if not existed:
            self._fsync_directory(path.parent)

    def _next_runtime_mutation_id(self) -> int:
        """Allocate one scene-wide numeric id for durable object transactions.

        Initializer and composition object mutations share the same transaction
        directory namespace, so allocation must observe both stages.  UUID-based
        legacy composition journal rows are ignored because they never own a
        ``runtime_objects/transactions/tx_N`` directory.
        """
        next_id = int(self._initializer_ledger.get("next_transaction_id") or 1)
        for row in read_mutation_journal(self._mutation_journal_path()):
            raw_txid = row.get("transaction_id")
            if isinstance(raw_txid, bool):
                continue
            if isinstance(raw_txid, int):
                observed = raw_txid
            elif isinstance(raw_txid, str) and raw_txid.isdigit():
                observed = int(raw_txid)
            else:
                continue
            next_id = max(next_id, observed + 1)
        if self.moge_dir:
            transaction_root = Path(self.moge_dir) / "runtime_objects" / "transactions"
            if transaction_root.is_dir():
                for candidate in transaction_root.glob("tx_*"):
                    match = re.fullmatch(r"tx_([1-9][0-9]*)", candidate.name)
                    if match:
                        next_id = max(next_id, int(match.group(1)) + 1)
        return next_id

    def _next_initializer_mutation_id(self) -> int:
        """Backward-compatible name for the scene-wide runtime id allocator."""
        return self._next_runtime_mutation_id()

    def _begin_mutation(self, kind: str, request: dict) -> int | str:
        if self.root_stage_name == "initializer":
            self._assert_no_incomplete_runtime_mutation()
            txid: int | str = self._next_initializer_mutation_id()
        else:
            txid = f"{self.root_stage_name or 'stage'}-{uuid.uuid4().hex}"
        event = {
            "transaction_id": txid,
            "kind": kind,
            "status": "requested",
            "request": copy.deepcopy(request),
        }
        self._append_mutation_event(event)
        if self.root_stage_name == "initializer":
            ledger_before = copy.deepcopy(self._initializer_ledger)
            ledger_event = {
                "id": int(txid),
                **{k: v for k, v in event.items() if k != "transaction_id"},
            }
            self._initializer_ledger["events"].append(ledger_event)
            self._initializer_ledger["next_transaction_id"] = int(txid) + 1
            try:
                self._persist_initializer_ledger()
            except Exception as persist_error:  # noqa: BLE001
                # The journal request is already durable. Close it directly before
                # doing anything else so a ledger write failure can never leave a
                # permanently pending mutation in the append-only audit stream.
                terminal_event = {
                    "transaction_id": txid,
                    "kind": kind,
                    "status": "error",
                    "details": {
                        "phase": "request_ledger_persist",
                        "error": str(persist_error),
                        "scene_mutation": "not_started",
                        **(
                            {
                                "artifact_manifest": self._safe_typed_initializer_artifact_manifest(
                                    txid=int(txid),
                                    kind=kind,
                                    state="error",
                                )
                            }
                            if kind
                            in {
                                "execute_and_evaluate_objects",
                                "add_object",
                                "edit_object_mesh",
                                "edit_object_pose",
                                "remove_runtime_object",
                            }
                            and self._typed_initializer_recovery_enabled()
                            else {}
                        ),
                    },
                }
                audit_error = None
                try:
                    self._append_mutation_event(terminal_event)
                except Exception as exc:  # noqa: BLE001
                    audit_error = exc

                # A transient first write may still allow us to reconcile the ledger
                # with both journal events. If storage remains unavailable, retain the
                # pre-request in-memory ledger; the journal remains authoritative.
                if audit_error is None:
                    self._initializer_ledger["events"].append(
                        {
                            "id": int(txid),
                            **{
                                key: value
                                for key, value in terminal_event.items()
                                if key != "transaction_id"
                            },
                        }
                    )
                    try:
                        self._persist_initializer_ledger()
                    except Exception:  # noqa: BLE001
                        self._initializer_ledger = ledger_before
                else:
                    self._initializer_ledger = ledger_before

                message = (
                    f"initializer mutation request ledger write failed: {persist_error}"
                )
                if audit_error is not None:
                    message += f"; terminal audit failed: {audit_error}"
                begin_error = RuntimeError(message)
                begin_error.transaction_id = txid  # type: ignore[attr-defined]
                begin_error.audit_recorded = audit_error is None  # type: ignore[attr-defined]
                raise begin_error from persist_error
        return txid

    def _record_mutation_status(
        self,
        txid: int | str,
        kind: str,
        status: str,
        *,
        details: Optional[dict] = None,
    ) -> None:
        if status not in {
            "committed",
            "rejected",
            "rolled_back",
            "error",
            "undone",
        }:
            raise ValueError(f"invalid mutation terminal status {status!r}")
        event = {
            "transaction_id": txid,
            "kind": kind,
            "status": status,
            **({"details": copy.deepcopy(details)} if details else {}),
        }
        self._append_mutation_event(event)
        if self.root_stage_name == "initializer":
            self._initializer_ledger["events"].append(
                {
                    "id": int(txid),
                    **{k: v for k, v in event.items() if k != "transaction_id"},
                }
            )
            self._persist_initializer_ledger()

    def _restore_scene_graph_bytes(
        self, payload: Optional[bytes], *, durable: bool = False
    ) -> None:
        path = self._scene_graph_path()
        if path is None:
            if payload is not None:
                raise RuntimeError("scene-graph path is unavailable")
            return
        if payload is None:
            graph_was_present = path.exists()
            path.unlink(missing_ok=True)
            constraints_path = self._initializer_constraints_path()
            if constraints_path is not None:
                constraints_was_present = constraints_path.exists()
                constraints_path.unlink(missing_ok=True)
            else:
                constraints_was_present = False
            if durable and (graph_was_present or constraints_was_present):
                self._fsync_directory(path.parent)
        else:
            if durable:
                self._durable_atomic_write_bytes(path, payload)
            else:
                self._atomic_write_bytes(path, payload)
            from lib.tools.geometry.relationship_constraints import (
                write_initializer_constraints,
            )

            graph = json.loads(payload.decode("utf-8"))
            constraints_path = write_initializer_constraints(path.parent, graph)
            if durable:
                # The compiler fsyncs its temporary file before replacement. Its
                # parent-directory barrier is owned here because the graph and the
                # compiled constraints form one recovery state.
                self._fsync_file(constraints_path)
                self._fsync_directory(constraints_path.parent)
        self._initializer_constraint_artifact = None

    def _runtime_root_bindings_for_wrapper(self) -> list[dict]:
        if getattr(self, "root_stage_name", None) != "initializer":
            return []
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            path = self._scene_graph_path()
            graph = json.loads(path.read_text()) if path is not None else {}
            return [
                {
                    "graph_id": str(node["id"]),
                    "build_name": str(
                        node.get("build_name") or surface_build_name(node["id"])
                    ),
                    "graph_revision": int(node.get("runtime_graph_revision") or 0),
                    "runtime_transaction_id": int(
                        node.get("runtime_transaction_id") or 0
                    ),
                    "surface_type": "wall",
                }
                for node in graph.get("nodes", [])
                if node.get("kind") == "root_surface"
                and node.get("runtime_added") is True
                and node.get("id")
            ]
        except Exception:  # noqa: BLE001 - STRUCTURE fails closed on unreadable bindings
            return []

    def _registered_root_categories_for_wrapper(self) -> dict[str, str]:
        """Exact current graph-root build name -> normalized category."""
        if getattr(self, "root_stage_name", None) != "initializer":
            return {}
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            path = self._scene_graph_path()
            graph = json.loads(path.read_text()) if path is not None else {}
            return {
                str(node.get("build_name") or surface_build_name(node["id"])): str(
                    node.get("category") or ""
                )
                .strip()
                .lower()
                for node in graph.get("nodes", [])
                if node.get("kind") == "root_surface" and node.get("id")
            }
        except Exception:  # noqa: BLE001 - empty inventory fails closed in the wrapper
            return {}

    def _initializer_floor_helper_allowed(self) -> bool:
        """Whether the graph lacks a registered floor/ground root."""
        if getattr(self, "root_stage_name", None) != "initializer":
            return False
        try:
            path = self._scene_graph_path()
            graph = json.loads(path.read_text()) if path is not None else {}
            return not any(
                node.get("kind") == "root_surface"
                and str(node.get("category") or "").strip().lower()
                in {"floor", "ground"}
                for node in graph.get("nodes", [])
            )
        except Exception:  # noqa: BLE001 - malformed graph must not grant an exception
            return False

    def _protected_artifact_snapshots(self) -> dict[Path, tuple[bool, bytes]]:
        """Bytes generated Blender code is never authorized to mutate.

        This is deliberately content/path based, not a Blender object-name filter:
        execute_and_evaluate remains free to create every graph-registered root.
        """
        paths: list[Path] = []
        graph_path = self._scene_graph_path()
        if graph_path is not None:
            paths.append(graph_path)
        constraints_path = self._initializer_constraints_path()
        if constraints_path is not None:
            paths.append(constraints_path)
        yaw_observation_path = self._main_support_yaw_observation_path()
        if yaw_observation_path is not None:
            paths.append(yaw_observation_path)
        ledger_path = getattr(self, "initializer_ledger_path", None)
        if ledger_path:
            paths.append(Path(ledger_path))
        if self._has_capability("runtime_object_inventory"):
            paths.extend(self._runtime_artifact_paths())
            # The runtime overlay is cryptographically bound to the immutable
            # preprocessing masks.  Protect both the manifest and every mask it
            # names before arbitrary Blender code runs; otherwise a raw initializer
            # edit could alter a source mask immediately before the first typed
            # repair creates (and therefore blesses) the overlay hash.
            masks_path = Path(self.moge_dir) / "masks" / "masks.json"
            paths.append(masks_path)
            if masks_path.is_file():
                try:
                    masks_payload = json.loads(masks_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise RuntimeError(
                        f"cannot protect unreadable source masks: {masks_path} ({exc})"
                    ) from exc

                def add_mask_artifacts(value) -> None:
                    if isinstance(value, dict):
                        for key, child in value.items():
                            if (
                                key == "mask_path"
                                and isinstance(child, str)
                                and child.strip()
                            ):
                                raw = Path(child)
                                candidates = (
                                    [raw.resolve()]
                                    if raw.is_absolute()
                                    else [
                                        (Path(self.moge_dir) / raw).resolve(),
                                        (masks_path.parent / raw.name).resolve(),
                                        (
                                            Path(__file__).resolve().parents[3] / raw
                                        ).resolve(),
                                        raw.resolve(),
                                    ]
                                )
                                paths.append(
                                    next(
                                        (
                                            candidate
                                            for candidate in candidates
                                            if candidate.is_file()
                                        ),
                                        candidates[0],
                                    )
                                )
                            add_mask_artifacts(child)
                    elif isinstance(value, list):
                        for child in value:
                            add_mask_artifacts(child)

                add_mask_artifacts(masks_payload)
        if self._has_capability("mutation_journal") and self.moge_dir:
            paths.append(self._mutation_journal_path())
        # Keep one snapshot per canonical path if a future profile-owned artifact is
        # also promoted into the legacy protected set.
        paths = list(dict.fromkeys(paths))
        return {
            path: (path.exists(), path.read_bytes() if path.exists() else b"")
            for path in paths
        }

    def _restore_protected_artifacts(
        self, snapshots: dict[Path, tuple[bool, bytes]]
    ) -> None:
        for path, (existed, payload) in snapshots.items():
            if existed:
                self._atomic_write_bytes(path, payload)
            else:
                path.unlink(missing_ok=True)

    @staticmethod
    def _changed_protected_artifacts(
        snapshots: dict[Path, tuple[bool, bytes]],
    ) -> list[Path]:
        changed = []
        for path, (existed, payload) in snapshots.items():
            if path.exists() != existed:
                changed.append(path)
            elif existed and path.read_bytes() != payload:
                changed.append(path)
        return changed

    def _restore_unrecorded_scene(self) -> None:
        """Undo a wrapper save before it entered edit_history."""
        if not self.blender_save:
            raise RuntimeError("live Blender save is unavailable")
        restore_from = self.edit_history[-1] if self.edit_history else self.base_state
        if not restore_from or not os.path.isfile(restore_from):
            raise RuntimeError("pre-edit Blender snapshot is unavailable")
        live = Path(self.blender_save)
        staged = live.with_name(
            f".{live.name}.unrecorded-rollback-{os.getpid()}-{getattr(self, 'count', 0)}"
        )
        try:
            shutil.copy2(restore_from, staged)
            os.replace(staged, live)
        finally:
            staged.unlink(missing_ok=True)

    def _atomic_restore_blend(
        self, restore_from: str | Path, *, tag: str, durable: bool = False
    ) -> None:
        """Replace the live Blend from a snapshot without exposing a partial copy."""
        if not self.blender_save:
            raise RuntimeError("live Blender save is unavailable")
        live = Path(self.blender_save)
        if durable:
            self._durable_atomic_copy(restore_from, live)
            return
        staged = live.with_name(
            f".{live.name}.{tag}-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(restore_from, staged)
            os.replace(staged, live)
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Make a recovery-file rename durable on the local filesystem.

        2026-09-15 (owner): OFF unless ``GRASE_DURABLE_FSYNC=1``. The recovery
        snapshots defend against power loss, but the failure this pipeline meets is
        a process crash, which the page cache survives; ``os.replace`` alone keeps
        every published file whole. On the Lustre mount an fsync costs 0.5-2.7 s
        (measured), and one transaction's prepare + commit issue ~20-40 of them —
        9-122 s per transaction in the toast runs, the largest fixed cost after
        the physics itself.
        """
        if not durable_fsync_enabled():
            return
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _fsync_file(path: Path) -> None:
        """Flush an already-written regular file before publishing a WAL decision
        (no-op unless ``GRASE_DURABLE_FSYNC=1``, see ``_fsync_directory``)."""
        if not durable_fsync_enabled():
            return
        with open(path, "rb") as stream:
            os.fsync(stream.fileno())

    @classmethod
    def _durable_mkdir(cls, path: Path) -> None:
        """Create a directory tree and persist every newly published dentry.

        Fsyncing a file inside a new directory is insufficient: after a host crash,
        the child can disappear if the parent directory that names it was never
        flushed. Recovery directories are therefore published one component at a
        time, with a barrier on each parent.
        """
        path = Path(path)
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent
        if cursor.exists() and not cursor.is_dir():
            raise NotADirectoryError(cursor)
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                if not directory.is_dir():
                    raise
            cls._fsync_directory(directory.parent)

    @classmethod
    def _durable_atomic_write_bytes(cls, path: Path, payload: bytes) -> None:
        """Write one GPT-6 recovery artifact atomically and fsync file + parent."""
        cls._durable_mkdir(path.parent)
        staged = path.with_name(
            f".{path.name}.recovery-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(staged, "wb") as stream:
                stream.write(payload)
                stream.flush()
                if durable_fsync_enabled():
                    os.fsync(stream.fileno())
            os.replace(staged, path)
            cls._fsync_directory(path.parent)
        finally:
            staged.unlink(missing_ok=True)

    @classmethod
    def _durable_atomic_write_json(cls, path: Path, payload: dict) -> None:
        cls._durable_atomic_write_bytes(
            path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    @classmethod
    def _durable_atomic_copy(cls, source: str | Path, destination: Path) -> None:
        """Copy a transaction Blend snapshot without exposing partial bytes."""
        cls._durable_mkdir(destination.parent)
        staged = destination.with_name(
            f".{destination.name}.recovery-{os.getpid()}-{uuid.uuid4().hex}.tmp"
        )
        try:
            shutil.copy2(source, staged)
            if durable_fsync_enabled():
                with open(staged, "rb") as stream:
                    os.fsync(stream.fileno())
            os.replace(staged, destination)
            cls._fsync_directory(destination.parent)
        finally:
            staged.unlink(missing_ok=True)

    @staticmethod
    def _empty_initializer_ledger() -> dict:
        return {
            "schema_version": 1,
            "next_transaction_id": 1,
            "events": [],
            "active_transactions": [],
            "runtime_root_surfaces": [],
        }

    def _load_initializer_ledger(self) -> dict:
        path = getattr(self, "initializer_ledger_path", None)
        if not path or not os.path.exists(path):
            return self._empty_initializer_ledger()
        try:
            data = json.load(open(path))
            if data.get("schema_version") != 1:
                raise ValueError("unsupported schema_version")
            if not isinstance(data.get("events"), list) or not isinstance(
                data.get("active_transactions"), list
            ):
                raise ValueError("events/active_transactions must be lists")
            runtime_roots = data.setdefault("runtime_root_surfaces", [])
            if not isinstance(runtime_roots, list):
                raise ValueError("runtime_root_surfaces must be a list")
            data["next_transaction_id"] = int(data.get("next_transaction_id") or 1)
            return data
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"initializer object ledger is invalid: {path}") from exc

    def _persist_initializer_ledger(self) -> None:
        path = getattr(self, "initializer_ledger_path", None)
        if not path:
            return
        dst = Path(path)
        if self._typed_initializer_recovery_enabled():
            self._durable_atomic_write_json(dst, self._initializer_ledger)
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(tmp, "w") as stream:
            json.dump(self._initializer_ledger, stream, indent=2, sort_keys=True)
        os.replace(tmp, dst)

    def _record_edit_snapshot(
        self,
        snapshot: str,
        *,
        kind: str,
        transaction_id: Optional[int | str] = None,
        object_id: Optional[str] = None,
        edit_token: Optional[str] = None,
        artifact_before: Optional[dict[Path, tuple[bool, bytes]]] = None,
    ) -> None:
        if not hasattr(self, "_ledger_history"):
            self._ledger_history = []
        if not hasattr(self, "_graph_history"):
            self._graph_history = []
        if not hasattr(self, "_edit_meta"):
            self._edit_meta = []
        self.edit_history.append(snapshot)
        self._ledger_history.append(
            copy.deepcopy(
                getattr(self, "_initializer_ledger", self._empty_initializer_ledger())
            )
        )
        self._graph_history.append(self._read_scene_graph_bytes())
        self._edit_meta.append(
            {
                "kind": kind,
                "transaction_id": transaction_id,
                "object_id": object_id,
                "edit_token": edit_token,
                # The code_diff base that describes the scene AFTER this edit; undo
                # restores it so scene and script stay in step (2026-09-17: 54 patch
                # failures in 40 scenes came from the base NOT following the undo).
                "code_base": getattr(self, "_last_code", None),
                **(
                    {"artifact_before": artifact_before}
                    if artifact_before is not None
                    else {}
                ),
            }
        )

    def _restore_patch_base_after_undo(self) -> None:
        """Roll the code_diff base back with the scene: the base recorded by the edit now
        on top of the history, or none when the stack is empty."""
        meta = getattr(self, "_edit_meta", None)
        base = meta[-1].get("code_base") if meta else None
        if base:
            self._last_code = base
        else:
            self.__dict__.pop("_last_code", None)

    def _push_edit_snapshot(
        self,
        tag: str,
        *,
        transaction_id: Optional[int | str] = None,
        object_id: Optional[str] = None,
        edit_token: Optional[str] = None,
        artifact_before: Optional[dict[Path, tuple[bool, bytes]]] = None,
    ) -> None:
        """Append the CURRENT blender_save to the undo history. Move-tool edits and
        execute edits share ONE chronological stack, so undo_last_step reverts exactly
        the last edit and never jumps back past a move/flip (0717_e2e6_abc1: undoing a
        toppled freeform edit silently discarded the spoon/fork flip + moves, which had
        never been recorded because only execute() snapshotted). blender_save already
        holds the physics-settled poses here (the move tool saves post-commit)."""
        if not self.blender_save or not os.path.exists(self.blender_save):
            return
        self.count += 1
        snap_dir = self.render_path / f"{self.count}_{tag}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap = snap_dir / "state.blend"
        shutil.copy(self.blender_save, snap)
        self._record_edit_snapshot(
            str(snap),
            kind=tag,
            transaction_id=transaction_id,
            object_id=object_id,
            edit_token=edit_token,
            artifact_before=artifact_before,
        )

    def _new_edit_token(self, kind: str) -> str:
        """Return a stage-local immutable identity for one committed edit.

        ``count`` also advances for renders/read-only checks, so it is not an edit
        identity. A dedicated monotonic sequence keeps follow-up/undo matching exact
        without coupling it to those unrelated calls.
        """
        self._edit_token_seq = int(getattr(self, "_edit_token_seq", 0)) + 1
        attempt = int(getattr(self, "attempt_idx", 1) or 1)
        return f"{kind}:{attempt}:{self._edit_token_seq}"

    def _restore_failed_flip_transaction(
        self,
        session,
        *,
        restore_from: str,
        history_len: int,
        ledger_history_len: int,
        edit_meta_len: int,
    ) -> tuple[bool, str]:
        """Restore the pre-flip blend/session after persistence fails.

        The flip has already consumed its budgets and may already have been saved
        into the live blend. Restore through a sibling staging file, then discard
        any partially-recorded post-flip undo entries. A failed session rebuild is
        handled by discarding both the session and its physics server; the next pose
        call reconstructs them from the restored live blend.
        """
        if not self.blender_save or not os.path.isfile(restore_from):
            return False, "the pre-flip restore snapshot is unavailable"
        live = Path(self.blender_save)
        staged = live.with_name(
            f".{live.name}.flip-rollback-{os.getpid()}-{getattr(self, 'count', 0)}"
        )
        try:
            shutil.copy2(restore_from, staged)
            os.replace(staged, live)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            return False, f"could not restore the pre-flip blend: {exc}"

        del self.edit_history[history_len:]
        if hasattr(self, "_ledger_history"):
            del self._ledger_history[ledger_history_len:]
        if hasattr(self, "_graph_history"):
            del self._graph_history[history_len:]
        if hasattr(self, "_edit_meta"):
            del self._edit_meta[edit_meta_len:]
        self._post_flip_investigation_required = None
        self._armed = set()
        try:
            session.rebuild()
            self._attach_physics(session, resync=True)
            self._pose_session = session
            self._pose_dirty = False
            return True, ""
        except Exception as exc:  # noqa: BLE001 - restored blend remains authoritative
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass
            physics = getattr(self, "_physics_obj", None)
            if physics is not None:
                try:
                    physics.close()
                except Exception:  # noqa: BLE001
                    pass
            self._physics_obj = None
            self._pose_session = None
            self._pose_dirty = False
            return True, f"; pose session will rebuild lazily ({exc})"

    def _discard_composition_pose_runtime(self) -> list[str]:
        """Drop an untrustworthy live pose/physics pair after failed recovery.

        The durable Blend remains the authority. Clearing both references prevents a
        later composition call from reusing a possibly half-applied register or Isaac
        state; :meth:`_pose_session_get` will construct a fresh pair from that Blend.
        This is used only by strict GPT-6 error paths.
        """
        session = getattr(self, "_pose_session", None)
        physics = getattr(self, "_physics_obj", None)
        self._pose_session = None
        self._physics_obj = None
        self._pose_dirty = False
        self._armed = set()
        self._freeform_edit_prepared = False
        errors: list[str] = []
        if session is not None:
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 - fail-closed cleanup
                errors.append(f"pose session close failed: {exc}")
        if physics is not None:
            try:
                physics.close()
            except Exception as exc:  # noqa: BLE001 - fail-closed cleanup
                errors.append(f"physics close failed: {exc}")
        return errors

    def _strict_flip_infrastructure_failure(
        self,
        session,
        *,
        object_id: str,
        failure: str,
        restore_from: str,
        history_len: int,
        ledger_history_len: int,
        edit_meta_len: int,
        moved_before: int,
        physics: Optional[dict] = None,
    ) -> dict:
        """Recover one strict flip infrastructure error and account it atomically.

        A verified restore means no agent edit occurred, so neither the semantic
        one-shot nor the move counter is charged. If the durable restore cannot be
        proven, both consume-before ledgers stay charged and the live runtime is
        discarded; callers receive an explicit unknown-mutation verdict.
        """
        try:
            restored, restore_note = self._restore_failed_flip_transaction(
                session,
                restore_from=restore_from,
                history_len=history_len,
                ledger_history_len=ledger_history_len,
                edit_meta_len=edit_meta_len,
            )
        except Exception as exc:  # noqa: BLE001 - recovery must itself fail closed
            restored = False
            restore_note = f"pre-flip recovery raised: {exc}"

        cleanup_errors: list[str] = []
        if restored:
            self._flipped.discard(object_id)
            if moved_before:
                self._moved[object_id] = moved_before
            else:
                self._moved.pop(object_id, None)
        else:
            cleanup_errors = self._discard_composition_pose_runtime()

        recovery_text = (
            "The pre-flip scene was restored; the one-shot and move budget were "
            "not consumed. Investigate again before retrying."
            if restored
            else (
                "CRITICAL: pre-flip recovery could not be verified; the one-shot "
                "and move budget remain consumed and the pose runtime was discarded."
            )
        )
        diagnostics = [part for part in (restore_note, *cleanup_errors) if part]
        if diagnostics:
            recovery_text += " Recovery diagnostics: " + "; ".join(diagnostics)
        return {
            "status": "error",
            "output": {
                "text": [
                    f"rotate_180 on {object_id} failed because the strict physics "
                    f"infrastructure errored ({failure}). {recovery_text}"
                ],
                "failure_kind": "infrastructure_error",
                "retryable": restored,
                "recovery_succeeded": restored,
                "budget_consumed": not restored,
                "scene_mutation": "not_committed" if restored else "unknown",
                **({"physics": physics} if physics else {}),
            },
        }

    @staticmethod
    def _post_flip_followup_identity(record: dict) -> dict:
        return {
            "type": _POST_FLIP_FOLLOWUP_TYPE,
            "object_ids": list(record["object_ids"]),
            "token": str(record["token"]),
        }

    @classmethod
    def _post_flip_followup_payload(cls, record: dict) -> dict:
        """Required-followup schema shared with GeneratorAgent."""
        return {
            **cls._post_flip_followup_identity(record),
            "allowed_tools": list(_POST_FLIP_ALLOWED_TOOLS),
        }

    def _pending_post_flip_followup(self) -> Optional[dict]:
        record = getattr(self, "_post_flip_investigation_required", None)
        return record if isinstance(record, dict) else None

    def _require_post_flip_investigation(self, object_id: str, token: str) -> dict:
        record = {
            "type": _POST_FLIP_FOLLOWUP_TYPE,
            "object_ids": [object_id],
            "token": token,
            # This is a one-shot emergency allowance, not another general visit.
            "cap_exempt_visit_available": True,
        }
        self._post_flip_investigation_required = record
        if not hasattr(self, "_post_flip_followup_events"):
            self._post_flip_followup_events = []
        self._post_flip_followup_events.append(
            {**self._post_flip_followup_payload(record), "status": "required"}
        )
        return self._post_flip_followup_payload(record)

    def _resolve_post_flip_followup(self, *, token: str, status: str) -> Optional[dict]:
        """Resolve only the exact currently-pending flip identity."""
        record = self._pending_post_flip_followup()
        if record is None or str(record.get("token")) != str(token):
            return None
        resolved = {
            **self._post_flip_followup_identity(record),
            "status": status,
        }
        self._post_flip_investigation_required = None
        if not hasattr(self, "_post_flip_followup_events"):
            self._post_flip_followup_events = []
        self._post_flip_followup_events.append(copy.deepcopy(resolved))
        return resolved

    def _post_flip_followup_evidence(self) -> dict:
        pending = self._pending_post_flip_followup()
        return {
            "pending": (
                [self._post_flip_followup_payload(pending)]
                if pending is not None
                else []
            ),
            "events": copy.deepcopy(getattr(self, "_post_flip_followup_events", [])),
        }

    def _guard_post_flip_tool_call(
        self, tool_name: str, *, object_ids: Optional[list[str]] = None
    ) -> Optional[dict]:
        """Enforce pending recovery/follow-up contracts at the backend boundary.

        GeneratorAgent also narrows the model's next call, but MCP is callable by
        direct clients. While a flip is pending, only a matching investigation or
        undo of the exact flip edit may cross this boundary. Every refusal repeats
        the structured requirement so a stateless client can recover.
        """
        if (
            getattr(self, "root_stage_name", None) == "initializer"
            and self._typed_initializer_recovery_enabled()
        ):
            try:
                self._assert_no_incomplete_runtime_mutation()
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"{tool_name} blocked by pending initializer "
                            f"transaction recovery: {exc}"
                        ],
                        "scene_mutation": "unknown",
                        "recovery_required": True,
                    },
                }
        if (
            getattr(self, "root_stage_name", None) == "composition"
            and self._composition_mesh_recovery_enabled()
        ):
            try:
                self._assert_no_incomplete_composition_mesh_mutation()
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"{tool_name} blocked by pending composition mesh "
                            f"transaction recovery: {exc}"
                        ],
                        "scene_mutation": "unknown",
                        "recovery_required": True,
                    },
                }
        if getattr(self, "root_stage_name", None) != "composition":
            return None
        pending = self._pending_post_flip_followup()
        if pending is None:
            return None
        required_ids = set(pending.get("object_ids") or [])
        if tool_name == "investigate_objects" and required_ids.issubset(
            set(object_ids or [])
        ):
            return None
        if tool_name == "undo_last_step":
            meta = (
                self._edit_meta[-1]
                if getattr(self, "edit_history", None)
                and getattr(self, "_edit_meta", None)
                and len(self.edit_history) == len(self._edit_meta)
                else None
            )
            if (
                meta
                and meta.get("kind") == "flip"
                and meta.get("object_id") in required_ids
                and str(meta.get("edit_token")) == str(pending.get("token"))
            ):
                return None
        required = self._post_flip_followup_payload(pending)
        return {
            "status": "error",
            "output": {
                "text": [
                    f"{tool_name} blocked: a retained rotate_180 has stale "
                    "measurements for "
                    + ", ".join(sorted(required_ids))
                    + ". The next call must be investigate_objects including that "
                    "object, or undo_last_step for the exact flip."
                ],
                "required_followup": required,
                # This guard exists only because an earlier retained flip is live.
                # Preserve that fact through MCP/tool-client error normalization.
                "scene_mutation": "committed",
            },
        }

    def execute(self, code: str, render: bool = True) -> dict[str, object]:
        """Execute Blender code and return results.

        Args:
            code: Python code to execute in Blender.
            render: False = run the code and save the blend but skip the
                wrapper's render loop ("__norender__" sentinel) — used by
                execute_and_evaluate when the pseudo-GT compare path will
                immediately re-render anyway (the wrapper render is discarded).

        Returns:
            Dictionary with status and output (text, images, or errors).
        """
        blocked = self._guard_post_flip_tool_call("execute_and_evaluate")
        if blocked is not None:
            return blocked
        # Snapshot the pre-edit scene once, so the very first edit can be undone.
        self._ensure_undo_base()
        protected_before = self._protected_artifact_snapshots()

        self.count += 1
        code_file = self.script_path / f"{self.count}.py"
        render_file = self.render_path / f"{self.count}"
        code = self._parse_code(code)
        # EE-2 patch base: the last SUBMITTED script (even if it later fails to run —
        # the model patches against what it wrote, not against what succeeded).
        self._last_code = code

        # File operations
        with open(code_file, "w") as f:
            f.write(code)
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))

        # Execute Blender
        success, imgs, stdout, stderr = self._execute_blender(
            str(code_file), str(render_file) if render else "__norender__"
        )
        script_out = _script_output(stdout) if success else ""
        receipt = getattr(self, "_last_patch_receipt", "") if success else ""
        self._last_patch_receipt = ""
        extra_parts = [part for part in (receipt, script_out) if part]
        # Generated code may build/edit any registered Blender root, but graph and
        # authorization artifacts are backend-owned.  Detect this after *every*
        # subprocess outcome (a script can write a file and then raise), restore the
        # artifacts byte-for-byte, and roll back a wrapper save before it can enter
        # the undo history.
        changed_artifacts = self._changed_protected_artifacts(protected_before)
        if changed_artifacts:
            restore_errors = []
            try:
                self._restore_protected_artifacts(protected_before)
            except Exception as exc:  # noqa: BLE001 - report a critical rollback issue
                restore_errors.append(f"artifact restore failed: {exc}")
            try:
                self._restore_unrecorded_scene()
            except Exception as exc:  # noqa: BLE001
                restore_errors.append(f"scene restore failed: {exc}")
            shutil.rmtree(render_file, ignore_errors=True)
            names = ", ".join(path.name for path in changed_artifacts)
            suffix = (
                " CRITICAL rollback issue: " + "; ".join(restore_errors)
                if restore_errors
                else " The scene and protected artifacts were restored."
            )
            return {
                "status": "error",
                "output": {
                    "text": [
                        "execute_and_evaluate rejected: generated code modified "
                        f"backend-owned artifact(s): {names}. Use build_root_surface or "
                        "remove_root_surface for authorized graph mutations; never edit "
                        "scene_graph.json or the initializer ledger directly." + suffix
                    ]
                },
            }
        # Check if render_file is empty or not exist
        if not success:
            # rmtree, not rmdir: a failed Blender run may have left partial
            # output, and rmdir on a non-empty dir would raise and mask the
            # real Blender error captured in stderr/stdout.
            shutil.rmtree(render_file, ignore_errors=True)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Error: " + _proc_text(stderr, stdout),
                        "Reconcile the error with CURRENT SCENE STATE/runtime_capabilities. "
                        "Resolve materials from current object slots; inspect node inputs before assignment. "
                        "Use the supported material helpers or NumPy when SciPy is unavailable. "
                        "After a patch failure, resubmit a complete script with code_diff empty.",
                    ]
                },
            }
        elif not render:
            # Deliberate no-render: the caller re-renders the compare view next.
            # The state.blend snapshot must still land (undo_last_step reads it).
            if self.blender_save:
                shutil.copy(self.blender_save, render_file / "state.blend")
                self._record_edit_snapshot(
                    str(render_file / "state.blend"), kind="execute"
                )
            return {
                "status": "success",
                "output": {
                    "text": [
                        "Edit applied (render deferred to the novel-view compare)."
                    ]
                    + extra_parts
                },
            }
        elif not imgs:
            # Key the "no image" branch on detected images (same predicate used
            # to build `imgs`), not a raw os.listdir count — otherwise a stray
            # non-image file in the dir would route here as success with image=[].
            # copy blender save under render file
            if self.blender_save:
                shutil.copy(self.blender_save, render_file / "state.blend")
                self._record_edit_snapshot(
                    str(render_file / "state.blend"), kind="execute"
                )
            return {
                "status": "success",
                "output": {
                    "text": [
                        "The code ran, but no image was rendered. This is almost always a code "
                        "error — the camera is FIXED to the reference view and managed for you, so do "
                        "NOT add, move, or render a camera in your code (any camera you add is "
                        "discarded). Check the return message below for the actual error and fix it:\n"
                        + _proc_text(stderr, stdout)
                    ]
                },
            }
        else:
            if self.blender_save:
                shutil.copy(self.blender_save, render_file / "state.blend")
                self._record_edit_snapshot(
                    str(render_file / "state.blend"), kind="execute"
                )
            return {
                "status": "success",
                "output": {
                    "image": imgs,
                    "text": [f"Render from camera {x}" for x in range(len(imgs))]
                    + extra_parts,
                },
            }

    def render_current_scene(self) -> dict[str, object]:
        """Render the current Blender scene without changing generated code."""
        blocked = self._guard_post_flip_tool_call("render_current_scene")
        if blocked is not None:
            return blocked
        self.count += 1
        code_file = self.script_path / f"{self.count}_render_current_scene.py"
        render_file = self.render_path / f"{self.count}_render_current_scene"

        with open(code_file, "w") as f:
            f.write("# End-of-round render snapshot. Do not modify the scene.\n")
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))

        success, imgs, stdout, stderr = self._execute_blender(
            str(code_file), str(render_file), persist_scene=False
        )
        if not success:
            shutil.rmtree(render_file, ignore_errors=True)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "End-of-round render failed: " + _proc_text(stderr, stdout)
                    ]
                },
            }
        if not imgs:
            return {
                "status": "success",
                "output": {
                    "text": [
                        "End-of-round render was requested, but no image was rendered (the camera is fixed to the reference view and managed for you — this is usually a transient render error, not something to fix in your code)."
                    ]
                },
            }

        if self.blender_save:
            shutil.copy(self.blender_save, render_file / "state.blend")
        return {
            "status": "success",
            "output": {
                "image": imgs,
                "text": [
                    f"End-of-round normal render from camera {x}"
                    for x in range(len(imgs))
                ],
            },
        }

    def render_bev(self, size_m: Optional[float] = None) -> dict[str, object]:
        """Top-down ORTHOGRAPHIC bird's-eye render of the current scene (read-only, not saved),
        centred on the az/el pivot. Each root surface is flat-tinted + id-labelled (table/walls
        distinct colours, floors/ceilings dim); the table/floor get a bbox outline while a wall
        shows as its true footprint band (not outlined); objects are dim-gray; the source camera is
        a dot+arrow; the world +X/+Y axes are drawn. Lets the agent check which SIDE each wall is on
        vs the camera and catch flips."""
        blocked = self._guard_post_flip_tool_call("render_bev")
        if blocked is not None:
            return blocked
        from lib.tools.blender import bev_overlay
        from lib.tools.blender.script_generators import generate_bev_script

        self.count += 1
        render_dir = self.render_path.parent / "tmp" / "bev" / str(self.count)
        render_dir.mkdir(parents=True, exist_ok=True)
        for f in os.listdir(render_dir):
            os.remove(os.path.join(render_dir, f))
        pivot = self._pseudo_gt_views().get((0.0, 0.0), {}).get("look_at") or [
            0.0,
            0.0,
            0.0,
        ]
        code_file = self.script_path / f"{self.count}_render_bev.py"
        code_file.write_text(generate_bev_script(pivot, size_m=size_m))
        success, imgs, stdout, stderr = self._execute_raw_blender(
            str(code_file), str(render_dir)
        )
        base, sidecar = render_dir / "bev.png", render_dir / "bev.json"
        if not success or not base.exists() or not sidecar.exists():
            return {
                "status": "error",
                "output": {
                    "text": ["BEV render failed: " + _proc_text(stderr, stdout)]
                },
            }
        out = str(render_dir / "bev_overlay.png")
        try:
            bev_overlay.draw_from_sidecar(str(base), str(sidecar), out)
        except Exception as e:  # noqa: BLE001 - overlay is best-effort; fall back to the raw render
            out = str(base)
            logging.warning(f"BEV overlay failed ({e}); returning the raw render")
        return {
            "status": "success",
            "output": {
                "image": [out],
                "text": [
                    "Top-down BIRD'S-EYE VIEW captured from the scene at this moment (read-only; "
                    "the scene was NOT modified). A later scene edit makes this image historical, "
                    "not current. Each root surface is flat-tinted + id-labelled (see the top-left "
                    "legend): the table and walls get distinct colours, floors/ceilings a dim tint; "
                    "the table/floor also get a bbox outline while a wall shows as its true footprint "
                    "band (not outlined); objects are dim-gray. The yellow dot+arrow "
                    "is the REAL camera position + view direction (clamped to the frame edge, with "
                    "its distance, if it sits outside the square); +X (red) / +Y (green) world axes (camera-aligned: image-UP = -Y deeper into the scene, image-RIGHT = -X the camera's right) "
                    "are bottom-left. Check that each wall is on the correct SIDE of the table vs "
                    "the camera and matches the photo, then fix any flipped/mis-placed surface with "
                    "execute_and_evaluate. (corner/perpendicular relations only fix relative angles, "
                    "not which side.)"
                ],
            },
        }

    def _pseudo_gt_views(self) -> dict[tuple[float, float], dict]:
        """Load the pre-computed pseudo-GT camera set (``<task>/pseudo_gt/cameras.json``,
        written by preprocessing), keyed by (azimuth, elevation). Partial bundles are
        unavailable, and copied manifests are resolved against their destination-local
        images rather than stale source-run paths. Cached; empty if absent."""
        if not hasattr(self, "_pgt_views"):
            self._pgt_views = {}
            try:
                bundle = load_complete_pseudo_gt(Path(self.moge_dir) / "pseudo_gt")
                for v in bundle["views"]:
                    self._pgt_views[(float(v["az"]), float(v["el"]))] = v
            except Exception:  # noqa: BLE001 - no pseudo-GT set -> novel views unavailable
                self._pgt_views = {}
        return self._pgt_views

    def _render_novel_view(
        self,
        azimuth: float,
        elevation: float,
        pair_reference: bool = True,
    ) -> dict[str, object]:
        """Render the CURRENT scene from the (azimuth, elevation) pseudo-GT camera and pair
        it with that view's reference only when the reference is marked trustworthy.
        ``(azimuth, elevation)`` MUST be one of the pre-computed views.

        ``pair_reference=False`` returns the render alone — used by a BARE
        render_current_scene() call, which is documented (NOVEL_VIEW_NOTE_ON_REQUEST) to
        pair a reference only when a view is explicitly passed. Routing bare calls here
        (instead of the legacy wrapper render) keeps every agent-visible render on one
        path/resolution (audit §12 workstream A)."""
        views = self._pseudo_gt_views()
        view = views.get((float(azimuth), float(elevation)))
        if view is None:
            opts = (
                ", ".join(f"({a:g}, {e:g})" for a, e in sorted(views)) or "(none built)"
            )
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"No pseudo-GT camera for (azimuth={azimuth}, elevation={elevation}). "
                        f"Choose an (azimuth, elevation) from: {opts}."
                    ]
                },
            }
        # Render at the SOURCE aspect ratio (capped) so the render matches the pseudo-GT.
        w, h = int(view.get("res_x") or 0), int(view.get("res_y") or 0)
        if w and h:
            s = min(1.0, 768.0 / max(w, h))
            w, h = max(1, round(w * s)), max(1, round(h * s))
        else:
            w = h = None
        res = self.render_arbitrary_view(
            camera_location=view["location"],
            look_at=view["look_at"],
            lens=float(view.get("lens", 35.0)),
            width=w,
            height=h,
        )
        if res.get("status") != "success":
            return res
        imgs = list(res.get("output", {}).get("image", []))
        texts = [
            f"Your scene rendered from view (azimuth={azimuth:g}, elevation={elevation:g})."
        ]
        # Reference to compare against: at the SOURCE view (azimuth=elevation=0, where the
        # render camera coincides with the original photo's camera) use the REAL ground-truth
        # photo; at any novel view use that view's pseudo-GT completion (a soft target, since
        # the disoccluded regions are generatively filled). A failed or excessive-drift
        # novel completion is retained for diagnostics but never attached to the agent.
        gt = getattr(self, "target_image_path", None)
        pgt = view.get("pseudo_gt")
        pgt_trusted = view.get("pseudo_gt_trusted", True) is not False
        if not pair_reference:
            gt = pgt = None  # bare call: render only, per NOVEL_VIEW_NOTE_ON_REQUEST
        ref_deduped = False
        if (
            float(azimuth) == 0.0
            and float(elevation) == 0.0
            and gt
            and os.path.isfile(gt)
        ):
            # CT3 dedupe: the target photo is byte-identical on every pairing AND is
            # guaranteed present in the conversation's protected head on every request,
            # so NEVER re-attach it — point at the head copy instead (owner ruling
            # 2026-07-29; the earlier attach-once-per-session anchor was judged
            # redundant too). Pseudo-GT pairings below are NOT deduped: they are
            # view-specific and ride ordinary round messages that slide out of the
            # memory window, so a pointer to them could dangle. The pointer text is
            # byte-stable across calls (cache-friendly). GRASE_DEDUP_REF=0 restores
            # always-attach (the A/B lever). ``ref_deduped`` marks the response so the
            # note composer (_image_feedback_instruction) describes it as a PAIRED
            # comparison, not a lone unpaired render — counting images cannot tell
            # those apart (that mislabel shipped in the first CT3 cut).
            if os.environ.get("GRASE_DEDUP_REF", "1") == "0":
                imgs.append(gt)
                texts.append(
                    "GROUND-TRUTH target photo for the source view (the real reference "
                    "image). Compare your render above against it and adjust so they match."
                )
            else:
                # No pointer text here: the note composer's deduped_pair branch
                # (_image_feedback_instruction) is the single source for "your
                # reference is the head's target photo" — a tool-side pointer
                # duplicated it verbatim in the same tool message and mis-said
                # "your render above" (post-assembly the render is BELOW).
                ref_deduped = True
        elif pgt and pgt_trusted and os.path.exists(pgt):
            imgs.append(pgt)
            texts.append(
                f"PSEUDO-GROUND-TRUTH for view (azimuth={azimuth:g}, elevation={elevation:g}) "
                "(a generative completion of the lifted point cloud — treat as a soft target). "
                "Compare your render above against it and adjust object layout so they match."
            )
        elif pgt and not pgt_trusted:
            reason = view.get("pseudo_gt_untrusted_reason") or "quality check failed"
            texts.append(
                f"The pseudo-ground-truth for this novel view was withheld ({reason}); "
                "judge this render without a generated reference or choose another view."
            )
        out: dict[str, object] = {"image": imgs, "text": texts}
        if ref_deduped:
            out["image_kind"] = "deduped_pair"
        return {"status": "success", "output": out}

    def render_object_focus(
        self, object_name: str, view: str = "current"
    ) -> dict[str, object]:
        """Render one object in isolation from a focused camera."""
        self.count += 1
        safe_name = (
            "".join(
                ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in object_name
            )[:80]
            or "object"
        )
        code_file = self.script_path / f"{self.count}_focus_{safe_name}.py"
        render_file = self.render_path / f"{self.count}_focus_{safe_name}"

        code_file.write_text(self._generate_object_focus_script(object_name, view))
        os.makedirs(render_file, exist_ok=True)
        for img in os.listdir(render_file):
            os.remove(os.path.join(render_file, img))

        success, imgs, stdout, stderr = self._execute_raw_blender(
            str(code_file), str(render_file)
        )
        if not success:
            shutil.rmtree(render_file, ignore_errors=True)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Object focus render failed: " + _proc_text(stderr, stdout)
                    ]
                },
            }
        if not imgs:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"Object focus render did not produce an image for '{object_name}'. "
                        + _proc_text(stderr, stdout)
                    ]
                },
            }

        return {
            "status": "success",
            "output": {
                "image": imgs,
                "text": [
                    f"Focused isolated render for object '{object_name}' from {view} view"
                ],
            },
        }

    def render_arbitrary_view(
        self,
        camera_location: list[float],
        camera_rotation_euler: Optional[list[float]] = None,
        look_at: Optional[list[float]] = None,
        lens: float = 35.0,
        orthographic: bool = False,
        ortho_scale: float = 5.0,
        object_names: Optional[list[str]] = None,
        render_backend: str = "stage_default",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> dict[str, object]:
        """Render the current scene from a temporary camera without saving scene changes.
        ``width``/``height`` force the render resolution (else a 512x512 square)."""
        self.count += 1
        tmp_dir = self.render_path.parent / "tmp"
        script_dir = tmp_dir / "scripts"
        render_dir = tmp_dir / "arbitrary_views" / str(self.count)
        script_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)
        for img in os.listdir(render_dir):
            os.remove(os.path.join(render_dir, img))

        code_file = script_dir / f"{self.count}_render_arbitrary_view.py"
        code_file.write_text(
            self._generate_arbitrary_view_script(
                camera_location=camera_location,
                camera_rotation_euler=camera_rotation_euler,
                look_at=look_at,
                lens=lens,
                orthographic=orthographic,
                ortho_scale=ortho_scale,
                object_names=object_names or [],
                render_backend=render_backend,
                width=width,
                height=height,
            )
        )

        success, imgs, stdout, stderr = self._execute_raw_blender(
            str(code_file), str(render_dir)
        )
        if not success:
            shutil.rmtree(render_dir, ignore_errors=True)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Arbitrary view render failed: " + _proc_text(stderr, stdout)
                    ]
                },
            }
        if not imgs:
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Arbitrary view render did not produce an image. "
                        + _proc_text(stderr, stdout)
                    ]
                },
            }

        return {
            "status": "success",
            "output": {
                "image": imgs,
                "text": [
                    "Temporary arbitrary-view inspection render saved under tmp/. "
                    "This did not modify or save the shared Blender scene."
                ],
            },
        }

    def _generate_arbitrary_view_script(
        self,
        camera_location: list[float],
        camera_rotation_euler: Optional[list[float]],
        look_at: Optional[list[float]],
        lens: float,
        orthographic: bool,
        ortho_scale: float,
        object_names: list[str],
        render_backend: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> str:
        return f"""import os
import sys

import bpy
from mathutils import Vector

render_dir = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv and len(sys.argv) > sys.argv.index("--") + 1 else "/tmp"
camera_location = {repr(camera_location)}
camera_rotation_euler = {repr(camera_rotation_euler)}
look_at = {repr(look_at)}
lens = {float(lens)}
orthographic = {repr(bool(orthographic))}
ortho_scale = {float(ortho_scale)}
object_names = {repr(object_names)}
render_backend = {repr(render_backend or "stage_default")}

def resolve_engine(backend):
    if backend == "workbench":
        return "BLENDER_WORKBENCH"
    if backend == "eevee":
        return "BLENDER_EEVEE_NEXT"
    if backend == "cycles":
        return "CYCLES"
    return os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")

def find_matches(names):
    if not names:
        return []
    matches = []
    lowered = [name.lower() for name in names if name]
    for obj in bpy.data.objects:
        obj_lower = obj.name.lower()
        if any(name == obj_lower or name in obj_lower for name in lowered):
            matches.append(obj)
    return matches

original_camera = bpy.context.scene.camera
original_visibility = {{}}
matches = find_matches(object_names)
if object_names and not matches:
    available = [obj.name for obj in bpy.data.objects if obj.type not in {{"CAMERA", "LIGHT"}}]
    raise ValueError(f"No objects matched {{object_names}}. Available objects: {{available}}")

if matches:
    renderable_types = {{"MESH", "CURVE", "SURFACE", "FONT", "META"}}
    allowed = set(matches)
    for obj in list(matches):
        allowed.update(child for child in obj.children_recursive)
    for obj in bpy.data.objects:
        if obj.type in renderable_types:
            original_visibility[obj.name] = (obj.hide_render, obj.hide_viewport)
            show = obj in allowed
            obj.hide_render = not show
            obj.hide_viewport = not show

camera_data = bpy.data.cameras.new("TemporaryArbitraryViewCamera")
camera = bpy.data.objects.new("TemporaryArbitraryViewCamera", camera_data)
bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera
camera.location = Vector(camera_location)
if camera_rotation_euler:
    camera.rotation_euler = camera_rotation_euler
elif look_at:
    direction = Vector(look_at) - camera.location
    if direction.length < 0.001:
        raise ValueError("look_at must be different from camera_location")
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
else:
    camera.rotation_euler = (1.1, 0.0, 0.78)

if orthographic:
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(float(ortho_scale), 0.01)
else:
    camera.data.type = "PERSP"
    # Match moge_camera.set_blender_camera: `lens` is derived from the source fov_x
    # against a 36 mm sensor, so the sensor MUST be fit HORIZONTALLY. A fresh Blender
    # camera defaults to sensor_fit='AUTO', which fits the LARGER dimension — on a
    # portrait source that applies fov_x vertically and renders 12 deg too narrow
    # against the reference photo and the pseudo-GT. Landscape is unaffected (AUTO ==
    # HORIZONTAL there), which is why this stayed hidden.
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.sensor_width = {float(DEFAULT_SENSOR_MM)}
    camera.data.lens = max(float(lens), 1.0)

engine = resolve_engine(render_backend)
try:
    bpy.context.scene.render.engine = engine
except Exception:
    bpy.context.scene.render.engine = "CYCLES"

if bpy.context.scene.render.engine == "BLENDER_WORKBENCH":
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "SINGLE"
    bpy.context.scene.display.shading.single_color = (0.8, 0.8, 0.8)
elif bpy.context.scene.render.engine == "CYCLES":
    bpy.context.scene.cycles.samples = min(getattr(bpy.context.scene.cycles, "samples", 128), 96)

bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.context.scene.render.resolution_x = {int(width) if width else 512}
bpy.context.scene.render.resolution_y = {int(height) if height else 512}
os.makedirs(render_dir, exist_ok=True)
bpy.context.scene.render.filepath = os.path.join(render_dir, "arbitrary_view.png")
bpy.ops.render.render(write_still=True)

bpy.context.scene.camera = original_camera
bpy.data.objects.remove(camera, do_unlink=True)
for name, visibility in original_visibility.items():
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render, obj.hide_viewport = visibility
"""

    def _generate_object_focus_script(self, object_name: str, view: str) -> str:
        object_name_literal = json.dumps(object_name)
        view_literal = json.dumps(view or "current")
        return f"""import math
import os
import sys

import bpy
from mathutils import Vector

render_dir = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv and len(sys.argv) > sys.argv.index("--") + 1 else "/tmp"
object_name = {object_name_literal}
view = {view_literal}

def find_object(name):
    if name in bpy.data.objects:
        return bpy.data.objects[name]
    lowered = name.lower()
    for obj in bpy.data.objects:
        if obj.name.lower() == lowered:
            return obj
    for obj in bpy.data.objects:
        if lowered in obj.name.lower():
            return obj
    return None

target = find_object(object_name)
if target is None:
    names = [obj.name for obj in bpy.data.objects if obj.type not in {{'CAMERA', 'LIGHT'}}]
    raise ValueError(f"Object '{{object_name}}' not found. Available objects: {{names}}")

renderable_types = {{'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}}
original_visibility = {{}}
for obj in bpy.data.objects:
    if obj.type in renderable_types:
        original_visibility[obj.name] = (obj.hide_render, obj.hide_viewport)
        should_show = obj == target or obj.parent == target
        obj.hide_render = not should_show
        obj.hide_viewport = not should_show

corners = []
if hasattr(target, "bound_box") and target.bound_box:
    corners = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
if not corners:
    corners = [target.matrix_world.translation]

center = sum(corners, Vector((0.0, 0.0, 0.0))) / len(corners)
radius = max((corner - center).length for corner in corners)
radius = max(radius, 0.25)

original_camera = bpy.context.scene.camera
camera_data = bpy.data.cameras.new("FocusRenderCamera")
camera = bpy.data.objects.new("FocusRenderCamera", camera_data)
bpy.context.collection.objects.link(camera)
bpy.context.scene.camera = camera

if view == "front":
    direction = Vector((0.0, -1.0, 0.2)).normalized()
elif view == "side":
    direction = Vector((1.0, 0.0, 0.2)).normalized()
elif view == "top":
    direction = Vector((0.0, 0.0, 1.0)).normalized()
elif view == "isometric" or original_camera is None:
    direction = Vector((1.0, -1.0, 0.65)).normalized()
else:
    direction = (original_camera.matrix_world.translation - center)
    if direction.length < 0.001:
        direction = Vector((1.0, -1.0, 0.65))
    direction.normalize()

camera.location = center + direction * (radius * 4.0)
look_direction = center - camera.location
camera.rotation_euler = look_direction.to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = max(radius * 2.6, 0.5)

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.cycles.samples = min(getattr(bpy.context.scene.cycles, "samples", 128), 128)

width = max(1, bpy.context.scene.render.resolution_x)
height = max(1, bpy.context.scene.render.resolution_y)
if width >= height:
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = max(1, round(512 * height / width))
else:
    bpy.context.scene.render.resolution_x = max(1, round(512 * width / height))
    bpy.context.scene.render.resolution_y = 512

os.makedirs(render_dir, exist_ok=True)
bpy.context.scene.render.filepath = os.path.join(render_dir, f"focus_{{target.name}}.png")
bpy.ops.render.render(write_still=True)

bpy.context.scene.camera = original_camera
bpy.data.objects.remove(camera, do_unlink=True)
for name, visibility in original_visibility.items():
    obj = bpy.data.objects.get(name)
    if obj:
        obj.hide_render, obj.hide_viewport = visibility
"""

    def _name2id(self) -> dict:
        """{Blender obj name -> scene-graph id 'category#index'} from placement.json,
        used to stamp each object's exact id into get_scene_info. Best-effort ({} if the
        placement table is missing/odd) and cached."""
        if getattr(self, "_n2id_cache", None) is None:
            m = {}
            try:
                pl = json.load(open(os.path.join(self.moge_dir, "placement.json")))
                for o in pl.get("objects", []):
                    if o.get("mesh_name"):
                        m[o["mesh_name"]] = f"{o['category']}#{o['instance']}"
            except Exception:  # noqa: BLE001 - id annotation is best-effort
                m = {}
            self._n2id_cache = m
        return self._n2id_cache

    # -- object naming at the agent boundary ----------------------------------------- #
    # Two id spaces meet here: the scene graph the agent reasons about is keyed
    # 'category#instance' (mug#0), while the Blender/Isaac bodies underneath are keyed by
    # the slugified mesh name (obj_mug_0). move/investigate_objects accept ONLY the
    # former (and exist ONLY in composition); code the agent writes in
    # execute_and_evaluate addresses ONLY the latter. Everything below the tool layer
    # speaks mesh names, so every agent-facing string converts HERE — pick the helper by
    # what the agent must DO with the name (0726 dialect unification):
    #   _oid    the fix is a TOOL CALL -> the id move/investigate take
    #   _label  the fix is a CODE EDIT -> both, since the code needs the mesh name
    #   _oids_in  an opaque reason string built downstream in the mesh dialect
    # Surfaces and unplaced bodies have no scene-graph id; their build/mesh name stands.

    def _oid(self, name: str) -> str:
        """Scene-graph id for a Blender mesh name; the name unchanged when it has none."""
        return self._name2id().get(name, name)

    def _label(self, name: str) -> str:
        """Both dialects ('mug#0 (obj_mug_0)') for text whose fix is a code edit. Only in
        composition — no other stage has a tool that consumes the scene-graph id, so
        there the mesh name alone is the actionable handle."""
        oid = self._name2id().get(name)
        if oid is None or self.root_stage_name != "composition":
            return name
        return f"{oid} ({name})"

    def _oids_in(self, text: str) -> str:
        """Rewrite the mesh names embedded in a downstream-built string (register.py's
        physics rejection reasons) into scene-graph ids."""
        n2id = self._name2id()
        return re.sub(r"\bobj_\w+\b", lambda m: n2id.get(m.group(), m.group()), text)

    def get_scene_info(self) -> dict[str, object]:
        """Get scene information by executing a Blender script."""
        blocked = self._guard_post_flip_tool_call("get_scene_info")
        if blocked is not None:
            return blocked
        try:
            # Create tmp directory if it doesn't exist
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Generate and execute scene info script
            scene_info_script = self._generate_scene_info_script()
            self.count += 1
            code_file = self.script_path / f"{self.count}.py"

            with open(code_file, "w") as f:
                f.write(scene_info_script)

            scene_info_path = tmp_dir / "scene_info.json"
            if scene_info_path.exists():
                scene_info_path.unlink()  # don't trust a stale file if extraction fails

            # Execute Blender script
            success, imgs, stdout, stderr = self._execute_blender(str(code_file))

            # The shared wrapper renders the scene AFTER writing scene_info.json, so a
            # trailing render crash fails the subprocess even though the info was
            # extracted. Trust the file: if it's there, the extraction succeeded.
            if scene_info_path.exists():
                with open(scene_info_path, "r") as f:
                    scene_info = json.load(f)
                # Annotate each placed object with its scene-graph id (category#index)
                # so the agent copies it verbatim for investigate/move — the script only
                # records the underscored Blender name (obj_coffee_machine_0), and
                # converting that to "coffee machine#0" by hand drops the spaces.
                n2id = self._name2id()
                for o in scene_info.get("objects", []):
                    oid = n2id.get(o.get("name"))
                    if oid:
                        o["id"] = oid
                return {"status": "success", "output": {"text": [str(scene_info)]}}
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Error: "
                        + (stderr or stdout or "Failed to extract scene information")
                    ]
                },
            }

        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}

    def _prefer_move(
        self, a: str, b: str, centroids=None, main: Optional[str] = None, normals=None
    ) -> Optional[str]:
        """Which body of a penetrating pair to move:
        - OBJECT vs SURFACE -> the OBJECT if the surface was built from PROVIDED geometry (a
          point/normal anchored to a MoGE plane = the fixed frame); else (a purely
          visually-built surface) the SURFACE.
        - two OBJECTS -> the LEAF, deeper in the support hierarchy (a descendant of the
          other); siblings / same level -> None.
        - SURFACE vs SURFACE (only MAIN-SUPPORT pairs reach here) -> the MAIN SUPPORT when the
          other side is a VERTICAL anchored wall (the LEVEL support slides horizontally IN its
          plane to escape, keeping its z=0 anchor; the wall can't move to help — an in-plane
          slide never separates it, an out-of-plane slide breaks its point/normal); the OTHER
          surface when that surface was a free visual guess; None for a HORIZONTAL anchored
          surface (a floor the table's base dips through — no clean translation fix).
        Object vs surface is told apart by the ``obj_`` name prefix; the object hierarchy and
        the surface anchoring are read from the scene graph, not from geometry. ``centroids``
        maps each surface's name -> its world AABB centre (from the penetration script), used
        to match it back to its scene-graph root node by position; ``normals`` maps it -> its
        PCA normal, used to tell a vertical wall (|nz| small) from a horizontal floor."""
        a_obj, b_obj = a.startswith("obj_"), b.startswith("obj_")
        if a_obj != b_obj:  # object vs surface
            obj, surf = (a, b) if a_obj else (b, a)
            return (
                obj
                if self._surface_anchored(surf, (centroids or {}).get(surf))
                else surf
            )
        if not a_obj:  # surface vs surface (main-support-involving)
            if main not in (a, b):
                return None
            other = b if main == a else a
            if not self._surface_anchored(other, (centroids or {}).get(other)):
                return (
                    other  # a free visual guess -> move it, keep the main support put
                )
            n = (normals or {}).get(other)
            other_vertical = n is not None and abs(n[2]) < 0.5
            return (
                main if other_vertical else None
            )  # slide the level support off a wall
        if not hasattr(self, "_hier"):  # lazily load name->id + parent map
            name2id, parent = {}, {}
            try:
                md = self.moge_dir
                pl = json.load(open(os.path.join(md, "placement.json")))
                for o in pl.get("objects", []):
                    if o.get("mesh_name"):
                        name2id[o["mesh_name"]] = f"{o['category']}#{o['instance']}"
                sg = json.load(open(os.path.join(md, "scene_graph.json")))
                for n in sg.get("nodes", []):
                    parent[n["id"]] = n.get("parent")
            except Exception:  # noqa: BLE001 - hint is best-effort
                name2id, parent = {}, {}
            self._hier = (name2id, parent)
        name2id, parent = self._hier
        ia, ib = name2id.get(a), name2id.get(b)
        if not ia or not ib:
            return None

        def _desc(x: str, y: str) -> bool:  # is x a descendant of y?
            seen, cur = set(), parent.get(x)
            while cur and cur not in seen:
                if cur == y:
                    return True
                seen.add(cur)
                cur = parent.get(cur)
            return False

        if _desc(ia, ib):
            return a
        if _desc(ib, ia):
            return b
        return None

    def _anchored_index(self) -> tuple[dict, set]:
        """``({build_name: (world_center, plumbed_normal)}, {every registered root name})``.

        A root is ANCHORED when the initializer prompt handed it a plane to FOLLOW —
        the same predicate ``_surface_line`` uses to decide that (``world_center`` +
        non-ambiguous plumb + ``plane_is_reliable``). Everything else the agent had to
        build by eye. Cached; empty on an unreadable graph."""
        if not hasattr(self, "_anchored_cache"):
            by_name: dict[str, tuple] = {}
            listed: set = set()
            try:
                from lib.tools.geometry.surface_relations import surface_build_name

                sg = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
                for n in sg.get("nodes", []):
                    if n.get("kind") != "root_surface" or not n.get("id"):
                        continue
                    name = surface_build_name(n["id"])
                    listed.add(name)
                    wc, pl = n.get("world_center"), (n.get("plane") or {})
                    nm, kind = plumb_plane(pl.get("normal"))
                    if wc and nm and kind != "ambiguous" and plane_is_reliable(pl):
                        by_name[name] = ([float(x) for x in wc], nm)
            except Exception:  # noqa: BLE001 - hint is best-effort
                by_name, listed = {}, set()
            self._anchored_cache = (by_name, listed)
        return self._anchored_cache

    @staticmethod
    def _on_plane(centroid, wc, nrm, tol: float = 0.15) -> bool:
        """Is ``centroid`` within ``tol`` PERPENDICULAR distance of the plane through
        ``wc`` with normal ``nrm``? Perpendicular distance is deliberate: it is
        invariant to the agent extending or sliding the surface IN-plane (a wall stays
        on its plane even when enlarged). The plane is PLUMBED (2026-08-06) because the
        prompt hands the agent the plumbed plane — against the RAW tilted fit a
        compliant wall drifted |offset|*|nz| and lost its anchor."""
        if not centroid:
            return True  # unmeasurable -> can't tell
        return (
            abs(
                (float(centroid[0]) - wc[0]) * nrm[0]
                + (float(centroid[1]) - wc[1]) * nrm[1]
                + (float(centroid[2]) - wc[2]) * nrm[2]
            )
            <= tol
        )

    def _surface_anchored(self, name: Optional[str], centroid) -> bool:
        """Was this built root surface anchored to a MEASURED plane (so a penetration
        must be fixed by moving the OTHER body), or was it a free visual guess?

        Matched BY NAME since 2026-08-14 (F4). The old form took only a centroid and
        returned True when it fell within 0.15 m of ANY anchored plane. The main
        support's plumbed normal is [0,0,1] at z=0, so that distance collapses to
        |centroid_z| — and the prompt's WALL EXTENT rule centres a wall's height at z=0
        whenever the scene has no floor. A by-eye wall in a tabletop scene therefore
        read as anchored *by the tabletop* and won every penetration fix against the
        main support, which was then told to slide off its photo-matched pose (fighting
        the POSE gate). 175 of 1870 archived scenes carry that geometry. The identity
        match is available now because the initializer contract requires each registered
        root to be built under its exact ``surface_build_name``.

        Defaults to True (anchored -> move the object) only when genuinely
        undeterminable: an unreadable scene graph, or a body with no measurable centroid.
        """
        by_name, listed = self._anchored_index()
        if not listed:
            return True  # no graph -> can't tell (safe default)
        if name in by_name:
            return self._on_plane(centroid, *by_name[name])
        if name in listed:
            return False  # a registered root with NO reliable plane: built by eye
        # Not in the graph at all — the sanctioned floor helper, or a surface the agent
        # renamed. Fall back to the positional test rather than guessing.
        if not centroid:
            return True
        return any(self._on_plane(centroid, wc, nrm) for wc, nrm in by_name.values())

    def _main_support_info(self) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """``(build_name, form, floor_name)`` of the main support from the scene graph — the ONE
        root surface the proposer tagged with a ``form`` (tabletop/table). Authoritative identity for
        the main-support rules, so a mis-placed table isn't mistaken for the floor (and the checks
        can't vacuously pass while the real support exists). The ``form`` feeds the table-bottom rule
        (a full-piece table must extend below its top slab). ``floor_name`` is the build-name of a
        surface the main support RESTS ON via an ``under`` relationship (a floor beneath a full
        table) — used by the top-at-z=0 rule to co-move that floor so the base stays on it; ``None``
        for a plain tabletop or a ground/floor main support (nothing under it). A cached
        ``tabletop`` is treated as a full table only when the compiled contract contains
        a required ``main_support_form`` obligation. ``(None, None, None)`` if no scene
        graph or no form-tagged surface."""
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            sg = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            artifact = (
                self._initializer_constraint_artifact
                or self._load_initializer_constraint_artifact(sg)
            )
            for n in sg.get("nodes", []):
                # main_support is the explicit marker (2026-07-31); ``form`` alone is the
                # pre-redesign fallback for graphs written before it existed.
                if n.get("kind") == "root_surface" and (
                    n.get("main_support") or n.get("form")
                ):
                    mid = n["id"]
                    form_constraint = next(
                        (
                            c
                            for c in artifact.get("constraints", [])
                            if c.get("kind") == "main_support_form"
                            and c.get("stage") == "STRUCTURE"
                            and c.get("authority") == "required"
                            and list(c.get("targets") or [None, None])[1] == mid
                        ),
                        None,
                    )
                    floor_id = (
                        list(form_constraint.get("targets") or [None])[0]
                        if form_constraint
                        else None
                    )
                    floor = surface_build_name(floor_id) if floor_id else None
                    form = str(n.get("form") or "").lower()
                    if form_constraint:
                        form = "table"
                    return surface_build_name(mid), form, floor
        except Exception:  # noqa: BLE001 - best-effort; fall back to the geometric guess
            pass
        return None, None, None

    def _main_support_is_floor(
        self, main_name: Optional[str], scene_graph: Optional[dict] = None
    ) -> bool:
        """Authoritative room-floor yaw routing, with a legacy name fallback.

        Build names are arbitrary slugs (``room_floor_0`` is valid), so their first
        token cannot define semantics. Prefer the matched main node's category/id;
        only cached graphs without that metadata use the historical prefix rule.
        """
        if not main_name:
            return False
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            sg = scene_graph
            if sg is None:
                sg = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            node = next(
                (
                    n
                    for n in sg.get("nodes", [])
                    if n.get("kind") == "root_surface"
                    and n.get("id")
                    and surface_build_name(n["id"]) == main_name
                ),
                None,
            )
            if node is not None:
                category = str(node.get("category") or "").strip().lower()
                if category:
                    return category in ("floor", "ground")
                graph_slug = str(node.get("id") or "").split("#", 1)[0].lower()
                if graph_slug in ("floor", "ground"):
                    return True
        except Exception:  # noqa: BLE001 - legacy fallback below
            pass
        return main_name.split("_")[0].lower() in ("floor", "ground")

    def _rule_surface_geometry(self, data: dict) -> tuple[bool, str]:
        """Root transforms plus finite containment of reliable measured wall points."""
        try:
            sg = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            surfaces = {}
            for node in sg.get("nodes", []):
                if node.get("kind") != "root_surface":
                    continue
                plane = node.get("plane") or {}
                surfaces[node["id"]] = {
                    "category": node.get("category"),
                    "world_center": node.get("world_center"),
                    "normal": plane.get("normal"),
                    "inlier_frac": plane.get("inlier_frac"),
                    "mask_frac": plane.get("mask_frac"),
                }
        except Exception as exc:  # noqa: BLE001 - required geometry evidence
            return False, f"root surface geometry: UNVERIFIED ({exc})"
        return surface_geometry_report(
            surfaces,
            data.get("bodies", []),
            surface_determinants=data.get("surface_determinants", {}),
            surface_hulls=(
                data.get("surface_plane_hulls") or data.get("surface_hulls", {})
            ),
            surface_normals=(
                data.get("surface_plane_normals") or data.get("surface_normals", {})
            ),
            surface_centroids=(
                data.get("surface_plane_centroids") or data.get("surface_centroids", {})
            ),
            surface_plane_bounds=data.get("surface_plane_bounds", {}),
        )

    @staticmethod
    def _runtime_root_binding_report(nodes: list[dict], rows: dict) -> tuple[bool, str]:
        failures = []
        for node in nodes:
            graph_id = str(node["id"])
            build_name = str(node.get("build_name") or "")
            row = rows.get(build_name)
            expected_revision = int(node.get("runtime_graph_revision") or 0)
            if not build_name or row is None:
                failures.append(
                    f"{graph_id}: exact Blender root '{build_name}' is missing"
                )
                continue
            if row.get("type") != "MESH":
                failures.append(f"{build_name}: root must remain a MESH")
            if row.get("parent") is not None:
                failures.append(
                    f"{build_name}: runtime root must remain independent (parent=None)"
                )
            if str(row.get("graph_id") or "") != graph_id:
                failures.append(
                    f"{build_name}: backend graph-id binding was lost or changed"
                )
            if row.get("source") != "initializer_runtime":
                failures.append(
                    f"{build_name}: backend runtime-source binding was lost or changed"
                )
            if row.get("surface_type") != "wall":
                failures.append(
                    f"{build_name}: backend surface-type binding was lost or changed"
                )
            try:
                bound_revision = int(row.get("graph_revision"))
            except (TypeError, ValueError):
                bound_revision = -1
            if bound_revision != expected_revision:
                failures.append(
                    f"{build_name}: backend graph-revision binding was lost or changed"
                )
            try:
                bound_transaction_id = int(row.get("runtime_transaction_id"))
                expected_transaction_id = int(node.get("runtime_transaction_id") or 0)
            except (TypeError, ValueError):
                bound_transaction_id = -1
                expected_transaction_id = 0
            if bound_transaction_id != expected_transaction_id:
                failures.append(
                    f"{build_name}: backend runtime-transaction binding was lost or changed"
                )
            try:
                determinant = float(row.get("determinant"))
            except (TypeError, ValueError):
                determinant = float("nan")
            if not math.isfinite(determinant) or determinant <= 1e-12:
                failures.append(
                    f"{build_name}: transform determinant must remain finite and positive"
                )
            try:
                extents = [float(x) for x in row.get("oriented_extents", [])]
                axes = [[float(x) for x in axis] for axis in row.get("axes_world", [])]
                if (
                    len(extents) != 3
                    or len(axes) != 3
                    or any(len(axis) != 3 for axis in axes)
                    or not all(math.isfinite(x) and x > 1e-6 for x in extents)
                ):
                    raise ValueError("invalid oriented bounds")
                thin_axis = min(range(3), key=lambda i: extents[i])
                broad_axes = [i for i in range(3) if i != thin_axis]
                thin = extents[thin_axis]
                broad_min = min(extents[i] for i in broad_axes)
                normal_verticality = abs(axes[thin_axis][2])
                broad_verticality = max(abs(axes[i][2]) for i in broad_axes)
                if thin > 0.35 or thin > 0.25 * broad_min:
                    failures.append(
                        f"{build_name}: wall must remain broad and thin (oriented "
                        f"extents {extents})"
                    )
                if normal_verticality > 0.25 or broad_verticality < 0.85:
                    failures.append(
                        f"{build_name}: wall must remain plumb with a horizontal face "
                        "normal and a vertical broad axis"
                    )
            except (TypeError, ValueError):
                failures.append(
                    f"{build_name}: wall-like oriented geometry evidence is missing"
                )
        if failures:
            return False, (
                "runtime root binding: FAIL — execute_and_evaluate may refine registered "
                "geometry but must preserve its exact backend identity. Do not create or "
                "forge replacement custom properties; undo/restore the backend-created "
                "original and edit that object in place:\n  - "
                + "\n  - ".join(failures)
            )
        return True, (
            f"runtime root binding: PASS ({len(nodes)} exact graph/custom-property "
            "binding" + ("s" if len(nodes) != 1 else "") + ")"
        )

    def _rule_registered_root_inventory(self) -> tuple[bool, str]:
        """Reject mesh-root identities outside the exact current graph inventory."""
        try:
            graph_path = self._scene_graph_path()
            if graph_path is None:
                raise ValueError("active scene-graph path is unavailable")
            graph = json.loads(graph_path.read_text())
            registered_categories = self._registered_root_categories_for_wrapper()
            registered_names = sorted(registered_categories)
            registered_floor = any(
                node.get("kind") == "root_surface"
                and str(node.get("category") or "").strip().lower()
                in {"floor", "ground"}
                for node in graph.get("nodes", [])
            )
            pipeline_names = sorted(
                set(getattr(self, "_pipeline_object_names", []) or [])
            )
            helper_allowed = not registered_floor
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.count += 1
            out_path = tmp_dir / f"{self.count}_registered_root_inventory.json"
            code_path = self.script_path / f"{self.count}_registered_root_inventory.py"
            code_path.write_text(
                "import bpy, json, numpy as np\n"
                f"registered_names = set({registered_names!r})\n"
                f"registered_categories = {registered_categories!r}\n"
                f"pipeline_names = set({pipeline_names!r})\n"
                f"helper_allowed = {helper_allowed!r}\n"
                "renderable_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'VOLUME', 'CURVES', 'POINTCLOUD', 'GPENCIL', 'GREASEPENCIL'}\n"
                "anchors = {obj for name in registered_names | pipeline_names if (obj := bpy.data.objects.get(name)) is not None}\n"
                "def evaluated_world_points(obj):\n"
                "    depsgraph = bpy.context.evaluated_depsgraph_get()\n"
                "    evaluated = obj.evaluated_get(depsgraph)\n"
                "    mesh = evaluated.to_mesh()\n"
                "    try:\n"
                "        points = np.asarray([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices], dtype=float)\n"
                "    finally:\n"
                "        evaluated.to_mesh_clear()\n"
                "    if len(points) < 3 or not np.isfinite(points).all():\n"
                "        raise ValueError('insufficient finite evaluated world vertices')\n"
                "    return points\n"
                "def wall_finite_frame(wall):\n"
                "    points = evaluated_world_points(wall)\n"
                "    centered = points - points.mean(axis=0)\n"
                "    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)\n"
                "    normal = vh[-1]\n"
                "    normal_norm = np.linalg.norm(normal)\n"
                "    if normal_norm <= 1e-9:\n"
                "        raise ValueError('degenerate parent-wall face-normal fit')\n"
                "    normal = normal / normal_norm\n"
                "    world_up = np.asarray([0.0, 0.0, 1.0])\n"
                "    run = np.cross(world_up, normal)\n"
                "    run_norm = np.linalg.norm(run)\n"
                "    if run_norm <= 1e-9:\n"
                "        raise ValueError('parent-wall face normal is vertical')\n"
                "    run = run / run_norm\n"
                "    in_plane_up = np.cross(normal, run)\n"
                "    in_plane_up = in_plane_up / np.linalg.norm(in_plane_up)\n"
                "    axes = np.asarray([normal, run, in_plane_up])\n"
                "    projected = points @ axes.T\n"
                "    return axes, projected.min(axis=0), projected.max(axis=0)\n"
                "def wall_detail_finite_error(wall, detail, tolerance=0.03):\n"
                "    axes, parent_min, parent_max = wall_finite_frame(wall)\n"
                "    detail_projected = evaluated_world_points(detail) @ axes.T\n"
                "    detail_min = detail_projected.min(axis=0)\n"
                "    detail_max = detail_projected.max(axis=0)\n"
                "    violations = []\n"
                "    worst = 0.0\n"
                "    for axis, label in enumerate(('normal', 'run', 'up')):\n"
                "        overshoot = max(float(parent_min[axis] - detail_min[axis]), float(detail_max[axis] - parent_max[axis]), 0.0)\n"
                "        if overshoot > tolerance + 1e-6:\n"
                "            violations.append(f'{label} overshoot {overshoot:.3f}m')\n"
                "            worst = max(worst, overshoot)\n"
                "    return ', '.join(violations), worst\n"
                "registered_rows = {}\n"
                "descendant_violations = []\n"
                "for name in registered_names:\n"
                "    obj = bpy.data.objects.get(name)\n"
                "    registered_rows[name] = None if obj is None else {\n"
                "        'type': obj.type,\n"
                "        'parent': obj.parent.name if obj.parent else None,\n"
                "        'hide_render': bool(obj.hide_render),\n"
                "        'in_scene': obj.name in bpy.context.scene.objects,\n"
                "        'polygon_count': len(obj.data.polygons) if obj.type == 'MESH' and obj.data is not None else 0,\n"
                "    }\n"
                "    if obj is None:\n"
                "        continue\n"
                "    descendants = list(obj.children_recursive)\n"
                "    category = registered_categories.get(name, '')\n"
                "    if category != 'wall' and descendants:\n"
                "        descendant_violations.append(f\"{name} ({category or 'unknown'}) may not own detail children: \" + ', '.join(sorted(child.name for child in descendants)))\n"
                "    elif category == 'wall':\n"
                "        prefix = name + '_'\n"
                "        invalid = sorted(child.name for child in descendants if child.type != 'MESH' or not child.name.startswith(prefix))\n"
                "        if invalid:\n"
                "            descendant_violations.append(f\"{name} wall details must be MESH objects named {prefix}<part>: \" + ', '.join(invalid))\n"
                "        for detail in descendants:\n"
                "            if detail.type != 'MESH' or not detail.name.startswith(prefix):\n"
                "                continue\n"
                "            try:\n"
                "                finite_error, worst_overshoot = wall_detail_finite_error(obj, detail)\n"
                "            except Exception as exc:\n"
                "                descendant_violations.append(f'{detail.name} wall-detail finite bounds are unverifiable: {exc}')\n"
                "                continue\n"
                "            if finite_error:\n"
                "                stale_hint = worst_overshoot > 0.5 or detail.matrix_world.translation.length < 0.05\n"
                "                descendant_violations.append(f'{detail.name} is outside parent wall finite bounds (0.030m tolerance): {finite_error}' + ('; HINT: this part sits far outside its wall. A freshly created object keeps an identity matrix_world until bpy.context.view_layer.update() runs, so copying it before the update (or parenting first and then assigning WORLD coordinates to .location) drops the part at the world origin. Fix: view_layer.update(), copy matrix_world, set parent and matrix_parent_inverse = wall.matrix_world.inverted(), then restore matrix_world' if stale_hint else ''))\n"
                "helper = bpy.data.objects.get('floor_0')\n"
                "helper_unregistered = helper is not None and 'floor_0' not in registered_names\n"
                "if helper_allowed and helper_unregistered and helper.type == 'MESH' and helper.parent is None:\n"
                "    anchors.add(helper)\n"
                "rogue = set()\n"
                "for mesh in bpy.context.scene.objects:\n"
                "    is_instance_empty = mesh.type == 'EMPTY' and (getattr(mesh, 'instance_type', 'NONE') != 'NONE' or getattr(mesh, 'instance_collection', None) is not None or bool(getattr(mesh, 'is_instancer', False)))\n"
                "    if mesh.type not in renderable_types and not is_instance_empty:\n"
                "        continue\n"
                "    chain = []\n"
                "    cursor = mesh\n"
                "    while cursor is not None:\n"
                "        chain.append(cursor)\n"
                "        cursor = cursor.parent\n"
                "    if not any(item in anchors for item in chain):\n"
                "        rogue.add(chain[-1].name)\n"
                "helper_row = None\n"
                "if helper_unregistered:\n"
                "    helper_row = {'type': helper.type, 'parent': helper.parent.name if helper.parent else None, 'in_scene': helper.name in bpy.context.scene.objects, 'children': sorted(child.name for child in helper.children_recursive)}\n"
                "    if helper.type == 'MESH':\n"
                "        try:\n"
                "            depsgraph = bpy.context.evaluated_depsgraph_get()\n"
                "            evaluated = helper.evaluated_get(depsgraph)\n"
                "            mesh = evaluated.to_mesh()\n"
                "            try:\n"
                "                points = np.asarray([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices], dtype=float)\n"
                "            finally:\n"
                "                evaluated.to_mesh_clear()\n"
                "            if len(points) < 3 or not np.isfinite(points).all():\n"
                "                raise ValueError('insufficient finite world-space vertices')\n"
                "            centered = points - points.mean(axis=0)\n"
                "            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)\n"
                "            extents = [float(np.ptp(points @ axis)) for axis in vh]\n"
                "            helper_row['normal_verticality'] = float(abs(vh[-1][2]))\n"
                "            helper_row['oriented_extents'] = extents\n"
                "        except Exception as exc:\n"
                "            helper_row['geometry_error'] = str(exc)\n"
                f"with open({str(out_path)!r}, 'w') as stream:\n"
                "    json.dump({'rogue_roots': sorted(rogue), 'floor_helper': helper_row, 'registered_roots': registered_rows, 'descendant_violations': descendant_violations}, stream)\n"
            )
            out_path.unlink(missing_ok=True)
            old_file = self.blender_file
            if self.blender_save and os.path.exists(self.blender_save):
                self.blender_file = self.blender_save
            try:
                success, _, stdout, stderr = self._execute_raw_blender(str(code_path))
            finally:
                self.blender_file = old_file
            if not success or not out_path.exists():
                return False, (
                    "registered root inventory: UNVERIFIED — "
                    + _proc_text(stderr, stdout)
                )
            payload = json.loads(out_path.read_text())
            rogue = [str(name) for name in payload.get("rogue_roots", [])]
            helper = payload.get("floor_helper")
            failures = []
            failures.extend(
                str(message)
                for message in payload.get("descendant_violations", [])
                if message
            )
            registered_rows = payload.get("registered_roots") or {}
            missing_roots = sorted(
                name for name in registered_names if registered_rows.get(name) is None
            )
            invalid_roots = []
            for name in registered_names:
                row = registered_rows.get(name)
                if row is None:
                    continue
                if (
                    row.get("type") != "MESH"
                    or row.get("parent") is not None
                    or row.get("hide_render") is True
                    or row.get("in_scene") is not True
                    or int(row.get("polygon_count") or 0) <= 0
                ):
                    invalid_roots.append(name)
            if missing_roots:
                failures.append(
                    "missing current graph root(s): " + ", ".join(missing_roots)
                )
            if invalid_roots:
                failures.append(
                    "current graph root(s) must be independent renderable MESH roots: "
                    + ", ".join(sorted(invalid_roots))
                )
            if rogue:
                failures.append(
                    "unregistered direct renderable root(s): " + ", ".join(rogue)
                )
            if helper is not None:
                if not helper_allowed:
                    failures.append(
                        "unregistered floor_0 helper is forbidden because the graph "
                        "already has a registered floor/ground root"
                    )
                elif (
                    helper.get("type") != "MESH"
                    or helper.get("parent") is not None
                    or helper.get("in_scene") is not True
                ):
                    failures.append(
                        "unregistered floor_0 helper must be one independent MESH root"
                    )
                elif helper.get("children"):
                    failures.append(
                        "unregistered floor_0 helper must not own child objects: "
                        + ", ".join(str(name) for name in helper["children"])
                    )
                else:
                    try:
                        extents = [
                            float(value) for value in helper.get("oriented_extents", [])
                        ]
                        verticality = float(helper.get("normal_verticality"))
                        if len(extents) != 3 or not all(
                            math.isfinite(value) and value >= 0.0 for value in extents
                        ):
                            raise ValueError("invalid oriented extents")
                        thin = extents[-1]
                        broad_min = min(extents[:2])
                        if (
                            verticality < 0.9
                            or broad_min < 0.2
                            or thin > 0.35
                            or thin > 0.25 * broad_min
                        ):
                            failures.append(
                                "unregistered floor_0 helper must remain a level, "
                                f"broad/thin slab (extents={extents}, "
                                f"normal_verticality={verticality:.3f})"
                            )
                    except (TypeError, ValueError):
                        failures.append(
                            "unregistered floor_0 helper lacks valid level-slab geometry"
                        )
            if failures:
                return False, (
                    "registered root inventory: FAIL — "
                    + "; ".join(failures)
                    + ". Undo the unregistered root. A genuinely absent distinct wall "
                    "must be registered with build_root_surface; execute_and_evaluate "
                    "may only build/refine exact current roots, their descendants, "
                    "pipeline-object hierarchies, and the one permitted level floor_0 "
                    "helper."
                )
            return True, (
                "registered root inventory: PASS (all mesh hierarchies belong to exact "
                "current roots/pipeline objects"
                + (" plus one valid floor_0 helper" if helper is not None else "")
                + ")"
            )
        except Exception as exc:  # noqa: BLE001 - required STRUCTURE evidence
            return False, f"registered root inventory: UNVERIFIED ({exc})"

    def _rule_runtime_root_bindings(self) -> tuple[bool, str]:
        """Verify runtime roots survived later free-form refinement as the same roots."""
        try:
            graph = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            nodes = [
                n
                for n in graph.get("nodes", [])
                if n.get("kind") == "root_surface"
                and n.get("runtime_added") is True
                and n.get("id")
            ]
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self.count += 1
            out_path = tmp_dir / f"{self.count}_runtime_root_bindings.json"
            code_path = self.script_path / f"{self.count}_runtime_root_bindings.py"
            build_names = [str(n.get("build_name") or "") for n in nodes]
            code_path.write_text(
                "import bpy, json, numpy as np\n"
                f"names = {build_names!r}\n"
                "def wall_frame(obj):\n"
                "    depsgraph = bpy.context.evaluated_depsgraph_get()\n"
                "    evaluated = obj.evaluated_get(depsgraph)\n"
                "    mesh = evaluated.to_mesh()\n"
                "    try:\n"
                "        points = np.asarray([list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices], dtype=float)\n"
                "    finally:\n"
                "        evaluated.to_mesh_clear()\n"
                "    if len(points) < 4 or not np.isfinite(points).all():\n"
                "        raise ValueError('insufficient finite world-space mesh vertices')\n"
                "    centered = points - points.mean(axis=0)\n"
                "    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)\n"
                "    normal = vh[-1]\n"
                "    normal_norm = np.linalg.norm(normal)\n"
                "    if normal_norm <= 1e-9:\n"
                "        raise ValueError('degenerate face-normal fit')\n"
                "    normal = normal / normal_norm\n"
                "    up = np.asarray([0.0, 0.0, 1.0])\n"
                "    run = np.cross(up, normal)\n"
                "    run_norm = np.linalg.norm(run)\n"
                "    if run_norm <= 1e-9:\n"
                "        raise ValueError('wall face normal is vertical')\n"
                "    run = run / run_norm\n"
                "    axes = [normal, run, up]\n"
                "    extents = [float(np.ptp(points @ axis)) for axis in axes]\n"
                "    return extents, [[float(value) for value in axis] for axis in axes]\n"
                "rows = {}\n"
                "for name in names:\n"
                "    obj = bpy.data.objects.get(name)\n"
                "    if obj is None:\n"
                "        continue\n"
                "    try:\n"
                "        oriented_extents, axes_world = wall_frame(obj)\n"
                "    except Exception as exc:\n"
                "        oriented_extents, axes_world = [], []\n"
                "    rows[name] = {\n"
                "        'type': obj.type,\n"
                "        'parent': obj.parent.name if obj.parent else None,\n"
                "        'graph_id': obj.get('grase_graph_id'),\n"
                "        'source': obj.get('grase_root_source'),\n"
                "        'graph_revision': obj.get('grase_graph_revision'),\n"
                "        'runtime_transaction_id': obj.get('grase_runtime_transaction_id'),\n"
                "        'surface_type': obj.get('grase_surface_type'),\n"
                "        'determinant': obj.matrix_world.determinant(),\n"
                "        'oriented_extents': oriented_extents,\n"
                "        'axes_world': axes_world,\n"
                "    }\n"
                f"with open({str(out_path)!r}, 'w') as stream:\n"
                "    json.dump(rows, stream)\n"
            )
            out_path.unlink(missing_ok=True)
            success, _, stdout, stderr = self._execute_blender(str(code_path))
            if not success:
                return False, (
                    "runtime root binding/root inventory: FAIL — "
                    + _proc_text(stderr, stdout)
                )
            if not out_path.exists():
                return False, (
                    "runtime root binding: UNVERIFIED — binding query failed: "
                    + _proc_text(stderr, stdout)
                )
            rows = json.loads(out_path.read_text())
            if not isinstance(rows, dict):
                raise ValueError("binding query returned a non-object payload")
            return self._runtime_root_binding_report(nodes, rows)
        except Exception as exc:  # noqa: BLE001 - required STRUCTURE evidence
            return False, f"runtime root binding: UNVERIFIED ({exc})"

    def _constraint_context(self) -> tuple[dict, dict, dict]:
        """Return ``(graph, artifact, root surfaces)`` under strict digest validation."""
        graph_path = self._scene_graph_path()
        if graph_path is None or not graph_path.is_file():
            raise RuntimeError("active scene_graph.json is unavailable")
        graph = json.loads(graph_path.read_text())
        artifact = self._load_initializer_constraint_artifact(graph)
        surfaces = {}
        for node in graph.get("nodes", []):
            if node.get("kind") != "root_surface" or not node.get("id"):
                continue
            plane = node.get("plane") or {}
            surfaces[str(node["id"])] = {
                "category": node.get("category"),
                "build_name": node.get("build_name"),
                "world_center": node.get("world_center"),
                "plane": plane,
                # Preserve the legacy flattened shape consumed by relationship_report.
                "normal": plane.get("normal"),
                "inlier_frac": plane.get("inlier_frac"),
                "mask_frac": plane.get("mask_frac"),
            }
        return graph, artifact, surfaces

    def _record_constraint_results(self, results: list[dict]) -> None:
        replacement = {str(row.get("constraint_id")): row for row in results}
        seen: set[str] = set()
        updated = []
        for row in self._constraint_results:
            cid = str(row.get("constraint_id"))
            updated.append(replacement.get(cid, row))
            seen.add(cid)
        updated.extend(row for cid, row in replacement.items() if cid not in seen)
        self._constraint_results = updated

    def _mark_constraint_stage_unverified(self, stage: str, reason_code: str) -> None:
        """Fail closed without dropping per-constraint telemetry on context errors."""
        results = []
        for existing in self._constraint_results:
            if existing.get("stage") != stage:
                continue
            row = dict(existing)
            row["runtime_status"] = "unverified"
            row["measurements"] = dict(row.get("measurements") or {})
            row["reason_codes"] = [reason_code]
            results.append(row)
        self._record_constraint_results(results)

    @staticmethod
    def _deferred_constraint_results(artifact: dict) -> list[dict]:
        return [
            {
                "constraint_id": constraint.get("constraint_id"),
                "source_relationship_id": constraint.get("source_relationship_id"),
                "source_yaw_evidence_id": constraint.get("source_yaw_evidence_id"),
                "kind": constraint.get("kind"),
                "stage": constraint.get("stage"),
                "runtime_status": "deferred",
                "targets": list(constraint.get("targets") or []),
                "measurements": {},
                "reason_codes": ["stage_not_reached"],
            }
            for constraint in artifact.get("constraints", []) or []
        ]

    def _rule_structure_constraints(
        self, data: dict, main_name: Optional[str]
    ) -> tuple[bool, str]:
        """Evaluate compiled relationship-derived STRUCTURE obligations."""
        try:
            _graph, artifact, surfaces = self._constraint_context()
        except Exception as exc:  # noqa: BLE001 - compiled contract is required
            self._mark_constraint_stage_unverified(
                "STRUCTURE", "constraint_context_unavailable"
            )
            return False, f"relationship STRUCTURE constraints: UNVERIFIED ({exc})"
        constraints = [
            c
            for c in artifact.get("constraints", [])
            if c.get("stage") == "STRUCTURE" and c.get("authority") == "required"
        ]
        if not constraints:
            return True, "relationship STRUCTURE constraints: PASS (none to check)"
        from lib.tools.geometry.surface_relations import surface_build_name

        lines, results, ok = [], [], True
        for constraint in constraints:
            cid = str(constraint.get("constraint_id") or "")
            targets = list(constraint.get("targets") or [])
            upper_id = targets[1] if len(targets) == 2 else None
            upper = surfaces.get(upper_id) or {}
            upper_name = str(
                upper.get("build_name")
                or (surface_build_name(upper_id) if upper_id else "")
            )
            result = {
                "constraint_id": cid,
                "source_relationship_id": constraint.get("source_relationship_id"),
                "source_yaw_evidence_id": constraint.get("source_yaw_evidence_id"),
                "kind": constraint.get("kind"),
                "stage": "STRUCTURE",
                "runtime_status": "not_checked",
                "targets": targets,
                "measurements": {"upper_build_name": upper_name},
                "reason_codes": [],
            }
            results.append(result)
            if constraint.get("kind") != "main_support_form" or upper_name != main_name:
                ok = False
                result["runtime_status"] = "unverified"
                result["reason_codes"].append("malformed_main_support_form_constraint")
                lines.append(
                    f"{cid}: UNVERIFIED — constraint does not target the active main support"
                )
                continue
            connected_ok, connected_msg = main_support_connected_report(
                data.get("main_support_islands")
            )
            bottom_ok, bottom_msg = main_support_bottom_report(
                data.get("bodies", []), main_name, "table"
            )
            result["measurements"].update(
                connected=connected_ok,
                has_complete_bottom=bottom_ok,
            )
            if connected_ok and bottom_ok:
                result["runtime_status"] = "pass"
                lines.append(f"{cid}: PASS — complete connected furniture form exists")
            else:
                ok = False
                result["runtime_status"] = "fail"
                if not connected_ok:
                    result["reason_codes"].append("main_support_disconnected")
                if not bottom_ok:
                    result["reason_codes"].append("bare_or_missing_furniture_base")
                lines.append(f"{cid}: FAIL — {connected_msg}; {bottom_msg}")
        for result in results:
            if result.get("runtime_status") == "not_checked":
                ok = False
                result["runtime_status"] = "unverified"
                result["reason_codes"].append("structure_evaluator_no_terminal_status")
        self._record_constraint_results(results)
        return ok, "relationship STRUCTURE constraints: " + (
            "PASS" if ok else "FAIL"
        ) + "\n      - " + "\n      - ".join(lines)

    def _rule_pose_constraints(
        self, data: dict, main_name: Optional[str]
    ) -> tuple[bool, str]:
        """Evaluate the globally feasible set of every compiled required yaw row."""
        try:
            _graph, artifact, surfaces = self._constraint_context()
        except Exception as exc:  # noqa: BLE001 - compiled contract is required
            self._mark_constraint_stage_unverified(
                "POSE", "constraint_context_unavailable"
            )
            return False, f"relationship POSE constraints: UNVERIFIED ({exc})"
        results: list[dict] = []
        outcome = relationship_pose_constraint_report(
            artifact.get("constraints", []),
            surfaces,
            data.get("bodies", []),
            data.get("surface_runs", {}) or {},
            data.get("surface_top_hulls", {}) or {},
            data.get("surface_plane_normals", {}) or {},
            main_name=main_name,
            main_support_run=data.get("main_support_run") or {},
            evidence_out=results,
        )
        self._record_constraint_results(results)
        return outcome

    def _rule_contact_constraints(
        self, data: dict, main_name: Optional[str] = None
    ) -> tuple[bool, str]:
        """Evaluate generated-scene fulfillment from compiled CONTACT constraints."""
        try:
            _graph, artifact, surfaces = self._constraint_context()
        except Exception as exc:  # noqa: BLE001 - hard contracts cannot be certified
            self._mark_constraint_stage_unverified(
                "CONTACT", "constraint_context_unavailable"
            )
            return False, f"relationship CONTACT constraints: UNVERIFIED ({exc})"
        results: list[dict] = []
        outcome = relationship_contact_constraint_report(
            artifact.get("constraints", []),
            surfaces,
            data.get("bodies", []),
            data.get("surface_normals", {}) or {},
            main_name=main_name,
            surface_runs=data.get("surface_runs", {}) or {},
            surface_hulls=data.get("surface_hulls", {}) or {},
            surface_top_hulls=data.get("surface_top_hulls", {}) or {},
            surface_plane_hulls=data.get("surface_plane_hulls", {}) or {},
            surface_plane_bounds=data.get("surface_plane_bounds", {}) or {},
            surface_plane_centroids=data.get("surface_plane_centroids", {}) or {},
            surface_plane_normals=data.get("surface_plane_normals", {}) or {},
            evidence_out=results,
        )
        self._record_constraint_results(results)
        return outcome

    # Private compatibility alias for direct unit callers.  Runtime forcing is still
    # compiled-constraint-only; this never falls back to relationship rows.

    def _related_surface_pairs(self) -> set:
        """Build-name pairs whose compiled constraint GOVERNS their contact. Those
        pairs are excluded from the penetration check: a table flush against a wall always
        overlaps the wall's thickness, and 'against' already reports a table buried too deep
        (with its own tolerance) — double-flagging gave the agent two opposite-direction fixes
        to oscillate between. Angle-only relationships (``perpendicular``, ``corner``) do NOT
        exclude: they do not own tolerated overlap/penetration, so their pairs must stay
        penetration-checked
        (0704_pm_real8210: a perpendicular-only table<->wall pair interpenetrated unflagged).
        The strict artifact loader runs before this method in the initializer gate; no
        raw-relationship or legacy fallback is permitted."""
        pairs: set = set()
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            sg = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            artifact = self._load_initializer_constraint_artifact(sg)
            by_id = {
                str(n["id"]): str(n.get("build_name") or surface_build_name(n["id"]))
                for n in sg.get("nodes", [])
                if n.get("kind") == "root_surface" and n.get("id")
            }
            pairs = compiled_contact_owner_pairs(artifact.get("constraints", []), by_id)
        except Exception:
            # Fail closed at the top-level initializer contract load.  Returning no
            # exemption here is the safe local behavior: it cannot hide penetration.
            return set()
        return pairs

    # ------------------------------------------------------------------ #
    # Pose refinement (composition): investigate_objects / move backend    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _graph_descendant(session, target_mesh: str, occluder_mesh: str) -> bool:
        """Whether ``occluder_mesh`` is cargo under ``target_mesh`` in the graph."""
        parents = getattr(session, "parents", {}) or {}
        seen, cur = set(), occluder_mesh
        while cur in parents and cur not in seen:
            seen.add(cur)
            cur = parents[cur]
            if cur == target_mesh:
                return True
        return False

    def _composition_crop_visibility(self, session, objects: list[str]):
        """Choose a like-for-like composition photo + render hide set.

        Preprocessing's ``removed_of`` is an IMAGE-SPACE occlusion list; it does
        not say whether the blocker is true cargo or a lateral leaner.  Retain the
        useful amodal view only when every removed, unrequested blocker is
        confirmed cargo.  ``ambiguous_same_level`` and uncertain blockers stay
        visible, in which case the ORIGINAL contextual photograph is the only
        matching reference.

        Returns ``(photo_src, hide_ids, context_ids)``.  ``context_ids`` explains
        why a de-occluded edit was not used; it never affects internal amodal
        scoring/hints.
        """
        srcs = {session.ctx[o].get("obj_image") or session.image_path for o in objects}
        if len(srcs) != 1:
            return session.image_path, [], []
        candidate = next(iter(srcs))
        if candidate == session.image_path:
            return session.image_path, [], []

        requested = set(objects)
        removed_of = getattr(session, "_removed_of", {}) or {}
        hide_ids, context_ids = set(), set()
        relation_cache: dict[str, list] = {}
        physics = getattr(session, "physics", None)

        def role(target_id: str, occ_id: str) -> str:
            """``cargo`` or the conservative ``context``."""
            if occ_id in requested:
                return "context"
            target = (session.ctx.get(target_id) or {}).get("mesh_name")
            occ = (session.ctx.get(occ_id) or {}).get("mesh_name")
            if not target or not occ:
                return "context"

            # Live typed geometry is authoritative when it can classify the pair.
            records = None
            if physics is not None:
                try:
                    if occ not in relation_cache:
                        relation_cache[occ] = (
                            physics.client.rpc({"cmd": "supports_of", "name": occ})
                            or {}
                        ).get("supports", [])
                    records = relation_cache[occ]
                except Exception:  # noqa: BLE001 - graph fallback below
                    records = None
            if records is not None:
                pair = next((r for r in records if r.get("name") == target), None)
                if pair is not None:
                    relation = pair.get("relation")
                    if relation in {"strict_below", "clear_below_or_container"}:
                        return "cargo"
                    # The bagel/leaner class explicitly overrides a possibly stale
                    # VLM parent edge: it must remain visible as relationship context.
                    if relation == "ambiguous_same_level":
                        return "context"

            # Scene-graph ancestry is a fallback when live support geometry cannot
            # classify the pair.  Everything else stays visible conservatively.
            return (
                "cargo" if self._graph_descendant(session, target, occ) else "context"
            )

        for target_id in objects:
            for occ_id in removed_of.get(target_id, []):
                if occ_id not in session.ctx:
                    context_ids.add(occ_id)
                elif role(target_id, occ_id) == "cargo":
                    hide_ids.add(occ_id)
                else:
                    context_ids.add(occ_id)

        if context_ids:
            return session.image_path, [], sorted(context_ids)
        return candidate, sorted(hide_ids), []

    def _pose_paths(self) -> Optional[dict]:
        md = self.moge_dir
        if not md:
            return None
        blend = self.blender_save or self.blender_file
        paths = {
            "blend": blend,
            "graph": os.path.join(md, "scene_graph.json"),
            "masks": os.path.join(md, "masks", "masks.json"),
            "placement": os.path.join(md, "placement.json"),
            "image": os.path.join(md, "input.png"),
            "work": os.path.join(md, "register"),
            "moge_json": os.path.join(md, "moge", "moge.json"),
        }
        need = ("blend", "graph", "masks", "placement", "image")
        if not all(paths[k] and os.path.exists(paths[k]) for k in need):
            return None
        return paths

    def _pose_session_get(self):
        import threading

        from lib.tools.geometry.register import PoseSession

        # join a pending warmup first (never self-join: the warm thread lands here)
        t = getattr(self, "_warm_thread", None)
        if t is not None and t is not threading.current_thread():
            t.join()
            self._warm_thread = None
        if self._pose_session is not None and self._pose_dirty:
            # a manual edit changed the scene outside the session: reload + disarm
            self._pose_session.rebuild()
            self._pose_dirty = False
            self._armed = set()
            self._attach_physics(self._pose_session, resync=True)
        if self._pose_session is None:
            paths = self._pose_paths()
            if paths is None:
                raise RuntimeError(
                    "pose refinement unavailable: missing moge_dir artifacts"
                )
            self._pose_session = PoseSession(
                paths["blend"],
                paths["graph"],
                paths["masks"],
                paths["placement"],
                paths["image"],
                paths["work"],
                self.blender_command,
                moge_json=paths["moge_json"],
                harness_profile=getattr(self, "harness_profile", "baseline"),
                harness_profile_manifest=getattr(
                    self, "harness_profile_manifest", None
                ),
            )
            self._pose_dirty = False
            self._attach_physics(self._pose_session, resync=False)
        return self._pose_session

    def _typed_maskless_pose_bindings(self, session) -> dict[str, str]:
        """Bind prepared maskless bodies to fresh, validated canonical object IDs.

        Absent runtime inventory means there are no eligible maskless objects. Present
        malformed or inconsistent artifacts fail closed. Tombstones, tracked-mask
        records, and physical bodies absent from ``session.prepared`` never appear.
        """
        from pathlib import Path

        from lib.tools.geometry.inventory_contract import (
            InventoryContractError,
            validate_runtime_object_bindings,
        )
        from lib.tools.geometry.runtime_object_repair import (
            validate_runtime_inventory_schema,
        )

        directory = getattr(self, "moge_dir", None)
        if not directory:
            raise ValueError("maskless pose binding requires executor.moge_dir")
        scene = Path(directory)
        inventory_path = scene / "runtime_objects/inventory.json"
        if not inventory_path.exists():
            return {}
        inventory = json.loads(inventory_path.read_text())
        records = validate_runtime_inventory_schema(inventory)
        graph = json.loads((scene / "scene_graph.json").read_text())
        placement = json.loads((scene / "placement.json").read_text())
        try:
            validate_runtime_object_bindings(inventory, graph, placement)
        except InventoryContractError as exc:
            raise ValueError(f"invalid maskless canonical pose binding: {exc}") from exc

        # The overlay validator rejects duplicate runtime names; also reject collision
        # with any canonical source row before constructing an inverse name mapping.
        canonical: dict[str, str] = {}
        for row in placement["objects"]:
            mesh_name = row.get("mesh_name")
            object_id = f"{row['category']}#{row['instance']}"
            if (
                not isinstance(mesh_name, str)
                or not mesh_name
                or mesh_name != mesh_name.strip()
            ):
                raise ValueError(
                    f"invalid canonical mesh name for {object_id!r}: {mesh_name!r}"
                )
            if mesh_name in canonical:
                raise ValueError(
                    f"duplicate canonical mesh name {mesh_name!r}: {canonical[mesh_name]!r} and {object_id!r}"
                )
            canonical[mesh_name] = object_id

        prepared = getattr(session, "prepared", None)
        if not isinstance(prepared, (list, tuple, set, frozenset)) or any(
            not isinstance(name, str) or not name or name != name.strip()
            for name in prepared
        ):
            raise ValueError(f"invalid prepared physical mesh names: {prepared!r}")
        if len(prepared) != len(set(prepared)):
            raise ValueError(f"duplicate prepared physical mesh names: {prepared!r}")
        prepared_names = set(prepared)
        bindings: dict[str, str] = {}
        for object_id, record in records.items():
            if (
                record["status"] != "committed"
                or record.get("mask_policy") != "excluded"
            ):
                continue
            mesh_name = record["placement"]["mesh_name"]
            if (
                not mesh_name.startswith("obj_")
                or canonical.get(mesh_name) != object_id
            ):
                raise ValueError(
                    f"maskless object {object_id!r} has noncanonical physical binding {mesh_name!r}"
                )
            if mesh_name in prepared_names:
                bindings[mesh_name] = object_id
        return bindings

    def _normalize_typed_pose_edits(self, edits: list, session) -> list[dict]:
        """Validate agent-facing pose specs and bind each to one exact mesh name."""
        if not isinstance(edits, list) or not 1 <= len(edits) <= 12:
            raise ValueError("edits must contain between 1 and 12 pose specifications")
        by_mesh = self._name2id()
        by_id = {oid: name for name, oid in by_mesh.items()}
        available_ids = set(session.objects())
        maskless_bindings = None
        out: list[dict] = []
        seen: set[str] = set()
        for index, raw in enumerate(edits):
            if not isinstance(raw, dict):
                raise ValueError(f"edits[{index}] must be an object")
            value = str(raw.get("object") or "")
            if value in available_ids:
                object_id = value
                mesh_name = str(session.ctx[value]["mesh_name"])
            elif value in by_mesh and by_mesh[value] in available_ids:
                mesh_name, object_id = value, by_mesh[value]
            elif value in by_id and value in available_ids:
                object_id, mesh_name = value, by_id[value]
            else:
                # Keep the established masked-object resolver above. Supplement it
                # only with fresh validated authored identities, never cached inverse
                # placement entries or invented scoring contexts.
                if maskless_bindings is None:
                    maskless_bindings = (
                        self._typed_maskless_pose_bindings(session)
                        if self._has_capability("runtime_object_inventory")
                        else {}
                    )
                physical_by_id = {oid: name for name, oid in maskless_bindings.items()}
                if value in physical_by_id:
                    object_id, mesh_name = value, physical_by_id[value]
                elif value in maskless_bindings:
                    mesh_name, object_id = value, maskless_bindings[value]
                else:
                    raise ValueError(
                        f"edits[{index}] names unknown pose object {value!r}; use an "
                        "exact scene-graph id"
                    )
            if object_id in seen:
                raise ValueError(f"duplicate pose edit for {object_id}")
            seen.add(object_id)
            mode = str(raw.get("mode") or "")
            if mode not in {"delta", "absolute"}:
                raise ValueError(f"edits[{index}].mode must be delta or absolute")
            euler = raw.get("rotation_euler_deg")
            quat = raw.get("rotation_quaternion_wxyz")
            normalized: dict[str, object] = {
                "object_id": object_id,
                "mesh_name": mesh_name,
                "mode": mode,
                "expected_resting_mode": str(
                    raw.get("expected_resting_mode") or "preserve"
                ),
            }
            if normalized["expected_resting_mode"] not in {
                "preserve",
                "side",
                "free",
            }:
                raise ValueError(f"edits[{index}] has an invalid expected_resting_mode")
            for key, size in (
                ("translation_m", 3),
                ("rotation_euler_deg", 3),
                ("rotation_quaternion_wxyz", 4),
            ):
                if raw.get(key) is not None:
                    normalized[key] = self._finite_vector(raw[key], size, key)
            if quat is not None:
                # hypot avoids the intermediate square overflow that a direct
                # sum-of-squares incurs for otherwise finite model payloads.
                length = math.hypot(
                    *(float(x) for x in normalized["rotation_quaternion_wxyz"])
                )
                if not math.isfinite(length) or length < 1e-8:
                    raise ValueError(f"edits[{index}] quaternion cannot be zero")
                if euler is not None:
                    rx, ry, rz = (
                        math.radians(float(value)) * 0.5
                        for value in normalized["rotation_euler_deg"]
                    )
                    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
                    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
                    euler_quat = (
                        cx * cy * cz + sx * sy * sz,
                        sx * cy * cz - cx * sy * sz,
                        cx * sy * cz + sx * cy * sz,
                        cx * cy * sz - sx * sy * cz,
                    )
                    unit_quat = tuple(
                        float(value) / length
                        for value in normalized["rotation_quaternion_wxyz"]
                    )
                    # q and -q encode the same attitude, hence abs(dot).
                    dot = min(
                        1.0,
                        abs(sum(a * b for a, b in zip(euler_quat, unit_quat))),
                    )
                    separation_deg = math.degrees(2.0 * math.acos(dot))
                    if separation_deg > _TYPED_POSE_DUAL_ROTATION_TOLERANCE_DEG:
                        raise ValueError(
                            f"edits[{index}] supplies conflicting Euler and quaternion "
                            f"rotations ({separation_deg:.3f} degrees apart); use one "
                            "representation"
                        )
                    # Canonicalize an equivalent redundant pair to Euler.  Quaternion-only
                    # requests remain untouched and retain their existing execution path.
                    normalized.pop("rotation_quaternion_wxyz")
            if not any(
                key in normalized
                for key in (
                    "translation_m",
                    "rotation_euler_deg",
                    "rotation_quaternion_wxyz",
                )
            ):
                raise ValueError(f"edits[{index}] contains no transform change")
            out.append(normalized)
        # Physics settles support parents with their descendants as one carried
        # stack. Two independently requested poses inside that same stack would have
        # ambiguous precedence, so V1 asks the caller to split them into two calls.
        graph_path = self._scene_graph_path()
        if graph_path is not None and len(out) > 1:
            graph = json.loads(graph_path.read_text())
            nodes = {
                str(node.get("id")): node
                for node in graph.get("nodes", [])
                if isinstance(node, dict) and node.get("id")
            }

            def ancestors(object_id: str) -> set[str]:
                found: set[str] = set()
                current = nodes.get(object_id)
                while isinstance(current, dict):
                    parent = current.get("parent") or current.get("support")
                    if not isinstance(parent, str) or parent in found:
                        break
                    found.add(parent)
                    current = nodes.get(parent)
                return found

            selected = {str(edit["object_id"]) for edit in out}
            conflicts = sorted(
                (object_id, sorted(ancestors(object_id) & selected))
                for object_id in selected
                if ancestors(object_id) & selected
            )
            if conflicts:
                raise ValueError(
                    "one direct-pose batch cannot independently edit a support ancestor "
                    "and its descendant; split these stack edits into separate calls: "
                    + ", ".join(
                        f"{child} under {'/'.join(parents)}"
                        for child, parents in conflicts
                    )
                )
        return out

    @staticmethod
    def _typed_pose_blender_code(
        edits: list[dict], save_path: str, *, allow_empty_roots: bool = False
    ) -> str:
        payload = json.dumps(edits, sort_keys=True)
        return f"""
import json, math
import bpy
from mathutils import Euler, Matrix, Quaternion, Vector

edits = json.loads({payload!r})

def _logical_parts(mesh_name, object_id):
    root = bpy.data.objects.get(mesh_name)
    if {allow_empty_roots!r} and root is not None and root.type == 'EMPTY':
        if root.parent is not None or root.get('grase_graph_id') != object_id:
            raise RuntimeError('typed pose Empty has invalid canonical binding: ' + mesh_name)
        owners = [
            candidate for candidate in bpy.data.objects
            if candidate.get('grase_graph_id') == object_id
        ]
        if owners != [root]:
            raise RuntimeError('typed pose Empty identity is not uniquely owned: ' + mesh_name)
        parts = sorted(root.children_recursive, key=lambda candidate: candidate.name)
        if not parts or any(part.type != 'MESH' or part.parent != root for part in parts):
            raise RuntimeError('typed pose Empty needs flat nonempty Mesh children: ' + mesh_name)
        if any(part.get('grase_graph_id') not in (None, '') for part in parts):
            raise RuntimeError('typed pose Empty children cannot own graph identities: ' + mesh_name)
        for member in [root] + parts:
            if bpy.context.scene.objects.get(member.name) != member:
                raise RuntimeError('typed pose Empty members must belong to the active scene: ' + mesh_name)
            if member.library is not None or member.override_library is not None:
                raise RuntimeError('typed pose Empty members must be locally editable: ' + mesh_name)
            if member.constraints or member.animation_data is not None:
                raise RuntimeError('typed pose Empty members cannot retain constraints or animation: ' + mesh_name)
            if member.instance_type != 'NONE' or member.instance_collection is not None:
                raise RuntimeError('typed pose Empty members cannot instance geometry: ' + mesh_name)
            if member.hide_render or member.hide_viewport or member.hide_get():
                raise RuntimeError('typed pose Empty members must remain visible: ' + mesh_name)
            if member.type == 'MESH' and (
                member.data is None or member.data.users != 1
                or member.data.library is not None
                or not member.data.vertices or not member.data.polygons
            ):
                raise RuntimeError('typed pose Empty parts require unshared nonempty local meshes: ' + mesh_name)
            matrix = member.matrix_world
            if not all(math.isfinite(value) for row in matrix for value in row):
                raise RuntimeError('typed pose Empty has nonfinite transform: ' + mesh_name)
            if matrix.to_3x3().determinant() <= 1e-12:
                raise RuntimeError('typed pose Empty has invalid transform: ' + mesh_name)
        return parts, [root]
    parts = sorted(
        (
            candidate
            for candidate in bpy.data.objects
            if candidate.type == 'MESH'
            and (candidate.name == mesh_name or candidate.name.startswith(mesh_name + '_'))
        ),
        key=lambda candidate: candidate.name,
    )
    return parts, parts

def _world_bounds_center(parts):
    corners = [
        part.matrix_world @ Vector(corner)
        for part in parts
        for corner in part.bound_box
    ]
    return sum(corners, Vector()) / len(corners)

def _parent_depth(obj):
    depth = 0
    seen = set()
    parent = obj.parent
    while parent is not None:
        if parent.name_full in seen:
            raise RuntimeError('cyclic Blender parent hierarchy at ' + obj.name)
        seen.add(parent.name_full)
        depth += 1
        parent = parent.parent
    return depth

plans = []
authored_checks = []
claimed_parts = {{}}
for edit in edits:
    obj = bpy.data.objects.get(edit['mesh_name'])
    if obj is None:
        raise RuntimeError('typed pose target disappeared: ' + edit['mesh_name'])
    parts, transform_members = _logical_parts(edit['mesh_name'], edit.get('object_id'))
    if not parts:
        raise RuntimeError('typed pose target has no logical mesh parts: ' + edit['mesh_name'])
    collisions = [part.name for part in parts if part.name_full in claimed_parts]
    if collisions:
        raise RuntimeError(
            'typed pose edits overlap on mesh parts: ' + ', '.join(collisions)
        )
    for part in parts:
        claimed_parts[part.name_full] = edit['mesh_name']

    center = _world_bounds_center(parts)
    desired_center = center.copy()
    if 'translation_m' in edit:
        translation = Vector(edit['translation_m'])
        desired_center = (
            translation if edit['mode'] == 'absolute' else center + translation
        )

    turn = Quaternion((1.0, 0.0, 0.0, 0.0))
    requested_rotation = None
    if 'rotation_euler_deg' in edit:
        requested_rotation = Euler(
            tuple(math.radians(v) for v in edit['rotation_euler_deg']), 'XYZ'
        ).to_quaternion()
    elif 'rotation_quaternion_wxyz' in edit:
        requested_rotation = Quaternion(
            edit['rotation_quaternion_wxyz']
        ).normalized()
    if requested_rotation is not None:
        if edit['mode'] == 'absolute':
            current_rotation = obj.matrix_world.decompose()[1]
            turn = requested_rotation @ current_rotation.inverted()
        else:
            turn = requested_rotation

    rigid_delta = (
        Matrix.Translation(desired_center)
        @ turn.to_matrix().to_4x4()
        @ Matrix.Translation(-center)
    )
    if transform_members == [obj] and obj.type == 'EMPTY':
        authored_checks.extend(
            (part, rigid_delta @ part.matrix_world.copy(), part.matrix_basis.copy(), obj)
            for part in parts
        )
    plans.append(
        (
            [(part, part.matrix_world.copy()) for part in transform_members],
            rigid_delta,
        )
    )

# Capture every source matrix before changing the scene, then assign parents before
# their children.  This gives every exact/prefixed mesh part one shared world-space
# rigid delta even when a previously prepared Blend already parents the parts.
# Authored Empty plans assign ONLY the root; all parts inherit the delta once.
assignments = [
    (part, rigid_delta @ old_world)
    for part_worlds, rigid_delta in plans
    for part, old_world in part_worlds
]
for part, desired_world in sorted(
    assignments, key=lambda item: _parent_depth(item[0])
):
    part.matrix_world = desired_world
bpy.context.view_layer.update()
for part, expected_world, old_basis, root in authored_checks:
    if part.parent != root or part.matrix_basis != old_basis:
        raise RuntimeError('typed pose Empty child local transform changed: ' + part.name)
    error = max(abs(part.matrix_world[i][j] - expected_world[i][j]) for i in range(4) for j in range(4))
    if error > 2e-6:
        raise RuntimeError('typed pose Empty child failed rigid-world verification: ' + part.name)
bpy.ops.wm.save_as_mainfile(filepath={os.path.abspath(save_path)!r})
"""

    def _typed_pose_visual_feedback(self, normalized: list[dict], session) -> dict:
        """Never present a masked subset crop as feedback for a mixed pose batch."""
        meshes = [str(edit["mesh_name"]) for edit in normalized]
        no_mask = sorted(
            str(edit["object_id"])
            for edit in normalized
            if edit["object_id"] not in session.ctx
        )
        if not no_mask:
            crop = self._settled_object_crop(meshes)
            return {"image": crop, "image_kind": "relocation_crop"} if crop else {}
        feedback = {
            "no_mask_available": no_mask,
            "visual_feedback_note": "No scoring mask is available for "
            + ", ".join(no_mask)
            + "; showing the full post-settle scene.",
        }
        # The pose transaction is already committed. Presentation failure must never
        # roll it back or claim that a missing/partial crop verifies all targets.
        try:
            rendered = self.render_current_scene()
            images = (rendered.get("output") or {}).get("image")
            if rendered.get("status") == "success" and images:
                feedback.update(image=images, image_kind="post_settle_full_scene")
            else:
                feedback["visual_feedback_note"] = (
                    "No mask available for "
                    + ", ".join(no_mask)
                    + "; full post-settle scene render unavailable. The pose remains committed; "
                    "call render_current_scene to inspect it."
                )
        except Exception as exc:  # noqa: BLE001 - presentation after a durable commit
            feedback["visual_feedback_note"] = (
                "No mask available for "
                + ", ".join(no_mask)
                + f"; full post-settle render failed ({exc}). The pose remains committed; "
                "call render_current_scene to inspect it."
            )
        return feedback

    def edit_composition_object_poses(self, edits: list, reason: str) -> dict:
        """Apply one typed direct-pose batch and accept only its simulated result."""
        if self.root_stage_name != "composition" or not self._has_capability(
            "composition_direct_pose_edit"
        ):
            return {
                "status": "error",
                "output": {"text": ["edit_object_poses is not enabled for this stage"]},
            }
        self._typed_pose = getattr(self, "_typed_pose", {})
        cap = getattr(self, "typed_pose_cap", 2)
        capped = sorted(
            {
                str(edit.get("object"))
                for edit in edits or []
                if isinstance(edit, dict)
                and self._typed_pose.get(str(edit.get("object")), 0) >= cap
            }
        )
        if capped:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"typed pose cap ({cap} committed edits per object) reached "
                        f"for {capped}; use move(object, aspect) for further correction "
                        "- it searches the image against the target, the typed tool "
                        "does not. The scene was not changed."
                    ]
                },
            }
        request = {"edits": copy.deepcopy(edits), "reason": str(reason)}
        try:
            txid = self._begin_mutation("edit_object_poses", request)
        except Exception as exc:  # audit is a prerequisite, never best-effort
            return {
                "status": "error",
                "output": {"text": [f"direct pose edit not started: {exc}"]},
            }
        try:
            session = self._pose_session_get()
            if not str(reason).strip():
                raise ValueError("reason must be nonempty")
            normalized = self._normalize_typed_pose_edits(edits, session)
            prepared = self._prepare_freeform_edit()
            if not prepared or prepared.get("status") != "ready":
                raise RuntimeError(
                    "physics could not initialize before the edit: "
                    + str((prepared or {}).get("reason") or "unavailable")
                )
            if not self.blender_save:
                raise RuntimeError("live Blender save is unavailable")
            old_code = getattr(self, "_last_code", None)
            history_len = len(self.edit_history)
            try:
                result = self.execute(
                    self._typed_pose_blender_code(
                        normalized,
                        self.blender_save,
                        allow_empty_roots=self._has_capability(
                            "runtime_object_inventory"
                        ),
                    ),
                    render=False,
                )
            finally:
                # Typed edits must not become the patch base for a later raw
                # execute_and_evaluate call, even when Blender raises midway through.
                if old_code is None:
                    self.__dict__.pop("_last_code", None)
                else:
                    self._last_code = old_code
                if len(self.edit_history) > history_len and self._edit_meta:
                    self._edit_meta[-1]["code_base"] = old_code
            self._pose_dirty = True
            self._armed = set()
            if result.get("status") != "success":
                raise RuntimeError(
                    "; ".join(result.get("output", {}).get("text", []))
                    or "Blender pose edit failed"
                )
            if not self._edit_meta or len(self._edit_meta) != len(self.edit_history):
                raise RuntimeError("direct pose edit has no durable undo metadata")
            self._edit_meta[-1].update(
                {
                    "kind": "edit_object_poses",
                    "transaction_id": txid,
                    "object_id": ",".join(e["object_id"] for e in normalized),
                }
            )
            intended = self._typed_settle_intents(
                {
                    str(e["mesh_name"]): str(e["expected_resting_mode"])
                    for e in normalized
                }
            )
            freeform_before = copy.deepcopy(self._freeform)
            settled = self._settle_after_edit(intended_resting_modes=intended)
            if not settled or settled.get("status") != "settled":
                physics_result = settled or {
                    "status": "unavailable",
                    "failure_kind": "infrastructure_error",
                    "recovery_succeeded": False,
                    "recovery_errors": ["physics returned no settlement result"],
                }
                failure_kind = str(
                    physics_result.get("failure_kind") or "settlement_rejection"
                )
                infrastructure_error = failure_kind == "infrastructure_error"
                physics_recovered = not infrastructure_error or bool(
                    physics_result.get("recovery_succeeded", False)
                )
                rollback_error = ""
                try:
                    rolled_back = self._rollback_last_edit()
                except Exception as exc:  # noqa: BLE001 - fail closed below
                    rolled_back = False
                    rollback_error = str(exc)
                if rolled_back:
                    # A typed request which did not commit must not consume the
                    # direct/freeform relocation counter either.
                    self._freeform = freeform_before
                recovery_complete = rolled_back and physics_recovered
                cleanup_errors: list[str] = []
                if not recovery_complete:
                    cleanup_errors = self._discard_composition_pose_runtime()
                audit_details = {
                    "physics": physics_result,
                    "failure_kind": failure_kind,
                    "blend_rollback_succeeded": rolled_back,
                    "recovery_complete": recovery_complete,
                    **({"rollback_error": rollback_error} if rollback_error else {}),
                    **({"cleanup_errors": cleanup_errors} if cleanup_errors else {}),
                }
                self._record_mutation_status(
                    txid,
                    "edit_object_poses",
                    "rolled_back" if recovery_complete else "error",
                    details=audit_details,
                )
                if failure_kind == "physics_rejection":
                    failure_text = "Direct pose edit was rejected by physics"
                elif infrastructure_error:
                    failure_text = (
                        "Direct pose edit failed because the strict physics "
                        "infrastructure errored"
                    )
                else:
                    failure_text = "Direct pose edit produced no settled change"
                if recovery_complete:
                    failure_text += "; the pre-edit scene was restored"
                else:
                    failure_text += (
                        "; CRITICAL: full recovery could not be verified, so the "
                        "pose runtime was discarded and scene state is unknown"
                    )
                return {
                    "status": "error",
                    "output": {
                        "text": [failure_text + "."],
                        "transaction_id": txid,
                        "physics": physics_result,
                        "failure_kind": failure_kind,
                        "retryable": recovery_complete,
                        "recovery_succeeded": recovery_complete,
                        "scene_mutation": (
                            "not_committed" if recovery_complete else "unknown"
                        ),
                    },
                }
            reports = settled.get("reports") or []
            self._record_mutation_status(
                txid,
                "edit_object_poses",
                "committed",
                details={"reports": reports},
            )
            for edit in normalized:
                oid = str(edit["object_id"])
                self._typed_pose[oid] = self._typed_pose.get(oid, 0) + 1
            feedback = self._typed_pose_visual_feedback(normalized, session)
            output: dict[str, object] = {
                "text": [
                    "Direct pose request was physics-simulated and committed. The "
                    "requested matrices were initial conditions; judge the settled "
                    "matrices and available post-simulation visual feedback.",
                    json.dumps(reports, sort_keys=True),
                ],
                "transaction_id": txid,
                "requested_edits": normalized,
                "physics_reports": reports,
            }
            output.update(feedback)
            if feedback.get("visual_feedback_note"):
                output["text"].append(feedback["visual_feedback_note"])
            return {"status": "success", "output": output}
        except Exception as exc:  # noqa: BLE001 - typed tool returns structured errors
            # If execute() reached the undo stack, no unsafe half-commit may survive.
            rolled_back = (
                "history_len" in locals() and len(self.edit_history) > history_len
            )
            if not rolled_back:
                rolled_back = (
                    bool(self.edit_history)
                    and bool(self._edit_meta)
                    and (self._edit_meta[-1].get("transaction_id") == txid)
                )
            if rolled_back:
                rolled_back = self._rollback_last_edit()
                if rolled_back and "freeform_before" in locals():
                    # Settlement may have charged direct/freeform counters before
                    # a later persistence, crop, or terminal-audit exception. The
                    # scene transaction was undone, so its counters must be too.
                    self._freeform = freeform_before
            try:
                self._record_mutation_status(
                    txid,
                    "edit_object_poses",
                    "rolled_back" if rolled_back else "rejected",
                    details={"error": str(exc)},
                )
            except Exception as audit_exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"direct pose edit failed: {exc}; terminal audit also "
                            f"failed: {audit_exc}"
                        ]
                    },
                }
            return {
                "status": "error",
                "output": {
                    "text": [f"direct pose edit rejected: {exc}"],
                    "transaction_id": txid,
                },
            }

    def _attach_physics(self, session, resync: bool) -> None:
        """Attach the Isaac physics authority to a (re)built PoseSession. The Isaac
        server survives session rebuilds — only the matrices get resynced (a rebuild
        means a freeform blend edit happened; sync pushes the diffs as exact
        transforms)."""
        # Some focused callers build a read-only Executor via __new__. Real
        # Executors always initialize _physics_obj in __init__.
        if self.root_stage_name != "composition" or not hasattr(self, "_physics_obj"):
            return
        from lib.tools.geometry.composition_physics import CompositionPhysics

        if self._physics_obj is None:
            import atexit

            work = os.path.join(self.moge_dir, "physics", "composition")
            self._physics_obj = CompositionPhysics(session, work)
            atexit.register(self._physics_obj.close)
        session.physics = self._physics_obj
        if resync:
            self._physics_obj.sync(session)

    def _prepare_freeform_edit(self) -> Optional[dict]:
        """Establish the pre-edit physics baseline for a composition code edit.

        Building the PoseSession after the edit would make the raw teleport its
        baseline and ``settle_edited`` would see no delta. Composition therefore
        warms/attaches physics before arbitrary Blender code is allowed to run.
        """
        if self.root_stage_name != "composition":
            return None
        try:
            session = self._pose_session_get()
            if session.physics is None or self._physics_obj is None:
                raise RuntimeError("the composition physics authority is unavailable")
            # Yaw before->after snapshot (bridge_6): the settle path rebuilds the
            # session (wiping the _yaw_hint cache) and the edit itself moves the
            # blend, so BOTH the last cached yaw reading and the pre-edit world
            # yaw must be captured before the code runs. Best-effort — the line
            # is advisory and _world_matrices returns {} on any failure. The FULL
            # matrices are kept beside the derived yaws: the appearance-agreement
            # pair (P1) scores the pre-edit pose AFTER it no longer exists in the
            # blend, and the backend needs the exact pose, not just its yaw.
            self._pre_edit_yaw_reading = dict(getattr(session, "_yaw_hint", None) or {})
            self._pre_edit_world_matrix = self._world_matrices(session)
            self._pre_edit_world_yaw = self._yaws_from_matrices(
                self._pre_edit_world_matrix
            )
            # the rebuild also wipes session.cur, so the post-settle "(was ...)"
            # baseline needs the same snapshot: without it every post-settle
            # readback said "FIRST measurement — no baseline" right after an
            # investigate the agent could see three messages up, and a label the
            # agent can tell is false invites discounting the surrounding
            # caveats too (the bridge_6 trap fired exactly there).
            self._pre_edit_cur = {
                k: dict(v)
                for k, v in (getattr(session, "cur", None) or {}).items()
                if v
            }
            self._freeform_edit_prepared = True
            return {"status": "ready"}
        except Exception as exc:  # noqa: BLE001 - returned as a tool error
            self._freeform_edit_prepared = False
            return {"status": "unavailable", "reason": str(exc)}

    def _rollback_last_edit(self) -> bool:
        """Restore the state immediately preceding the newest committed edit."""
        ok, _ = self._undo_latest_edit(record_undo_event=False)
        return ok

    def _undo_latest_edit(self, *, record_undo_event: bool) -> tuple[bool, str]:
        durable_runtime_undo = (
            getattr(self, "root_stage_name", None) == "initializer"
            and self._typed_initializer_recovery_enabled()
        ) or self._composition_mesh_recovery_enabled()
        if not durable_runtime_undo:
            return self._undo_latest_edit_locked(record_undo_event=record_undo_event)
        lock_fd = self._acquire_initializer_scene_lock()
        try:
            ledger_path = getattr(self, "initializer_ledger_path", None)
            if ledger_path and Path(ledger_path).is_file():
                self._initializer_ledger = self._load_initializer_ledger()
            return self._undo_latest_edit_locked(record_undo_event=record_undo_event)
        finally:
            self._release_initializer_scene_lock(lock_fd)

    def _undo_composition_mesh_locked(self, meta: dict) -> tuple[bool, str]:
        """Crash-safe one-step undo for a committed composition mesh revision."""
        try:
            self._assert_no_incomplete_composition_mesh_mutation()
        except Exception as exc:  # noqa: BLE001
            return False, f"Could not start composition mesh undo: {exc}"
        if not self.edit_history or not self.blender_save:
            return False, "Could not restore the previous scene state."
        restore_from = (
            self.edit_history[-2] if len(self.edit_history) >= 2 else self.base_state
        )
        if not restore_from or not os.path.isfile(restore_from):
            return False, "Could not restore the previous scene state."
        txid = meta.get("transaction_id")
        object_id = str(meta.get("object_id") or "")
        artifact_before = meta.get("artifact_before")
        if (
            isinstance(txid, bool)
            or not isinstance(txid, int)
            or txid <= 0
            or not object_id
            or not isinstance(artifact_before, dict)
        ):
            return False, "Composition mesh undo metadata is incomplete."
        target_mesh = next(
            (name for name, oid in self._name2id().items() if oid == object_id), None
        )
        if not target_mesh:
            return False, "Composition mesh undo target identity is unavailable."

        current_names = list(self._pipeline_object_names)
        prior_graph = (
            self._graph_history[-2]
            if len(self._graph_history) >= 2
            else self._graph_base
        )
        marker_path = self._runtime_recovery_marker_path(txid)
        marker_before = marker_path.read_bytes() if marker_path.is_file() else None
        try:
            self._runtime_transaction_snapshots(
                txid,
                kind=_COMPOSITION_MESH_MUTATION_KIND,
                ledger_after_request={},
                purpose="undo",
            )
        except Exception as exc:  # noqa: BLE001
            if marker_before is not None:
                try:
                    self._durable_atomic_write_bytes(marker_path, marker_before)
                    self._prune_runtime_recovery_payload(txid)
                except Exception as marker_exc:  # noqa: BLE001
                    return False, (
                        f"Could not prepare crash-safe composition mesh undo: {exc}; "
                        f"marker rollback also failed: {marker_exc}"
                    )
            return False, f"Could not prepare crash-safe composition mesh undo: {exc}"

        live = Path(self.blender_save)
        staged = live.with_name(
            f".{live.name}.composition-mesh-undo-{os.getpid()}-{self.count}"
        )
        try:
            shutil.copy2(restore_from, staged)
            self._fsync_file(staged)
            self._restore_scene_graph_bytes(prior_graph, durable=True)
            self._restore_runtime_artifacts(artifact_before, durable=True)
            os.replace(staged, live)
            self._fsync_directory(live.parent)
            restored_placement = self._typed_initializer_json_file(
                Path(self.moge_dir) / "placement.json"
            )
            restored_names = [
                str(row["mesh_name"])
                for row in restored_placement.get("objects", [])
                if isinstance(row, dict) and row.get("mesh_name")
            ]
            if not restored_names or len(restored_names) != len(set(restored_names)):
                raise RuntimeError("undo restored invalid placement identities")
            self._pipeline_object_names = restored_names
            self._n2id_cache = None
            from lib.tools.geometry.inventory_contract import validate_scene_artifacts

            validate_scene_artifacts(
                self.moge_dir,
                allow_runtime_additions=True,
                validate_runtime_physics=True,
            )
            artifact_manifest = self._typed_initializer_artifact_manifest(
                txid=txid,
                kind=_COMPOSITION_MESH_MUTATION_KIND,
                state="undone",
                object_id=object_id,
                target_mesh=target_mesh,
                target_expected_present=True,
                strict=True,
            )
        except Exception as exc:  # noqa: BLE001
            staged.unlink(missing_ok=True)
            rollback_errors: list[str] = []
            try:
                marker = json.loads(marker_path.read_text())
                (
                    committed_blend,
                    committed_graph,
                    committed_artifacts,
                    committed_names,
                    _ledger,
                ) = self._load_runtime_recovery_snapshots(txid, marker)
                self._restore_composition_mesh_snapshot(
                    txid=txid,
                    live_before=committed_blend,
                    graph_before=committed_graph,
                    artifact_before=committed_artifacts,
                    pipeline_names_before=committed_names,
                    cleanup_outputs=False,
                )
                self._set_runtime_recovery_state(
                    txid,
                    "committed",
                    details={"undo_reverted": True, "error": str(exc)},
                )
            except Exception as rollback_exc:  # noqa: BLE001
                rollback_errors.append(str(rollback_exc))
            self._pipeline_object_names = current_names
            self._n2id_cache = None
            message = f"Could not restore composition mesh transaction: {exc}"
            if rollback_errors:
                message += "; recovery rollback failed: " + "; ".join(rollback_errors)
            return False, message

        undo_details = {
            "object_id": object_id,
            "mesh_name": target_mesh,
            "scene_mutation": "restored_pre_transaction_state",
            "artifact_manifest": artifact_manifest,
        }
        undo_event = {
            "transaction_id": txid,
            "kind": _COMPOSITION_MESH_MUTATION_KIND,
            "status": "undone",
            "undo_operation_id": f"composition-mesh-tx-{txid}-undo",
            "details": copy.deepcopy(undo_details),
        }
        decision = {
            "undo_event": copy.deepcopy(undo_event),
            "artifact_manifest": copy.deepcopy(artifact_manifest),
            "post_undo_pipeline_object_names": list(self._pipeline_object_names),
        }
        decision["decision_sha256"] = self._typed_initializer_json_sha256(decision)
        try:
            self._set_runtime_recovery_state(
                txid, "undo_commit_decided", details=decision
            )
        except Exception as exc:  # noqa: BLE001
            return False, (
                "Composition mesh undo reached an uncertain durable decision "
                f"boundary ({exc}); restart the composition stage"
            )

        warnings: list[str] = []
        try:
            self._append_mutation_event(undo_event)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"journal finalization failed: {exc}")
        if not warnings:
            try:
                self._set_runtime_recovery_state(txid, "undone", details=undo_details)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"recovery-marker finalization failed: {exc}")

        self.edit_history.pop()
        self._edit_meta.pop()
        getattr(self, "_replaced_uninvestigated", set()).discard(object_id)
        self._restore_patch_base_after_undo()
        if self._ledger_history:
            self._ledger_history.pop()
        if self._graph_history:
            self._graph_history.pop()
        runtime_errors = self._discard_composition_pose_runtime()
        warnings.extend(runtime_errors)
        return True, (
            "Undo committed; restart composition before another mutation: "
            + "; ".join(warnings)
            if warnings
            else ""
        )

    def _undo_latest_edit_locked(self, *, record_undo_event: bool) -> tuple[bool, str]:
        """Atomically restore the scene, initializer ledger, and undo stacks.

        The prior scene is first copied to a sibling staging file. For initializer
        edits, the post-undo ledger is then persisted atomically. Only after both are
        ready do we replace the live blend and pop the in-memory histories. Thus a
        ledger write failure cannot rewind the scene while leaving authorization and
        counters at the newer state.
        """
        self._last_undo_resolved_followup = None
        if (
            getattr(self, "root_stage_name", None) == "initializer"
            and self._typed_initializer_recovery_enabled()
        ):
            try:
                self._assert_no_incomplete_runtime_mutation()
            except Exception as exc:  # noqa: BLE001
                return False, f"Could not start initializer undo: {exc}"
        if not self.edit_history:
            return False, "No previous edit to undo."
        restore_from = (
            self.edit_history[-2] if len(self.edit_history) >= 2 else self.base_state
        )
        if not restore_from or not os.path.exists(restore_from):
            return False, "Could not restore the previous scene state."
        if not self.blender_save:
            return (
                False,
                "Could not restore the previous scene state: live save is unavailable.",
            )

        meta = (
            self._edit_meta[-1]
            if getattr(self, "_edit_meta", None)
            else {"kind": "unknown", "transaction_id": None}
        )
        if (
            self._composition_mesh_recovery_enabled()
            and meta.get("kind") == _COMPOSITION_MESH_MUTATION_KIND
        ):
            return self._undo_composition_mesh_locked(meta)
        prior_ledger = (
            copy.deepcopy(self._ledger_history[-2])
            if len(getattr(self, "_ledger_history", [])) >= 2
            else copy.deepcopy(
                getattr(self, "_ledger_base", self._empty_initializer_ledger())
            )
        )
        prior_graph = (
            self._graph_history[-2]
            if len(getattr(self, "_graph_history", [])) >= 2
            else getattr(self, "_graph_base", self._read_scene_graph_bytes())
        )
        is_initializer = getattr(self, "root_stage_name", None) == "initializer"
        typed_undo_kinds = {
            "execute_and_evaluate_objects",
            "add_object",
            "edit_object_mesh",
            "edit_object_pose",
            "remove_runtime_object",
        }
        typed_initializer_undo = bool(
            is_initializer
            and record_undo_event
            and meta.get("transaction_id") is not None
            and meta.get("kind") in typed_undo_kinds
            and self._typed_initializer_recovery_enabled()
        )
        current_ledger = copy.deepcopy(
            getattr(self, "_initializer_ledger", self._empty_initializer_ledger())
        )
        current_pipeline_names = list(getattr(self, "_pipeline_object_names", []))
        typed_target = next(
            (
                str(row.get("target"))
                for row in current_ledger.get("active_transactions", [])
                if isinstance(row, dict)
                and str(row.get("id")) == str(meta.get("transaction_id"))
                and row.get("target")
            ),
            None,
        )
        typed_recovery_marker_before = None
        if typed_initializer_undo:
            marker_path = self._runtime_recovery_marker_path(
                int(meta["transaction_id"])
            )
            if marker_path.is_file():
                typed_recovery_marker_before = marker_path.read_bytes()
        current_graph = self._read_scene_graph_bytes()
        artifact_before = meta.get("artifact_before") or None
        current_artifacts = (
            self._snapshot_runtime_artifacts() if artifact_before is not None else None
        )
        restored_ledger = None
        if is_initializer:
            restored_ledger = self._initializer_ledger_after_undo(
                prior_ledger,
                undone_transaction_id=(
                    meta.get("transaction_id")
                    if record_undo_event
                    and not typed_initializer_undo
                    and meta.get("kind")
                    in {
                        "nudge",
                        "execute_and_evaluate_objects",
                        "build_root_surface",
                        "remove_root_surface",
                        "add_object",
                        "edit_object_mesh",
                        "edit_object_pose",
                        "remove_runtime_object",
                    }
                    else None
                ),
                undone_kind=(meta.get("kind") if record_undo_event else None),
            )

        live = Path(self.blender_save)
        staged = live.with_name(
            f".{live.name}.undo-stage-{os.getpid()}-{getattr(self, 'count', 0)}"
        )
        try:
            shutil.copy2(restore_from, staged)
            if typed_initializer_undo:
                self._fsync_file(staged)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            return False, f"Could not stage the previous scene state: {exc}"

        if typed_initializer_undo:
            try:
                self._runtime_transaction_snapshots(
                    int(meta["transaction_id"]),
                    kind=str(meta["kind"]),
                    ledger_after_request=current_ledger,
                    purpose="undo",
                )
            except Exception as exc:  # noqa: BLE001
                staged.unlink(missing_ok=True)
                if typed_recovery_marker_before is not None:
                    try:
                        self._durable_atomic_write_bytes(
                            self._runtime_recovery_marker_path(
                                int(meta["transaction_id"])
                            ),
                            typed_recovery_marker_before,
                        )
                        self._prune_runtime_recovery_payload(
                            int(meta["transaction_id"])
                        )
                    except Exception as marker_exc:  # noqa: BLE001
                        return False, (
                            "Could not prepare crash-safe typed initializer undo: "
                            f"{exc}; recovery-marker rollback also failed: {marker_exc}"
                        )
                return False, (
                    f"Could not prepare crash-safe typed initializer undo: {exc}"
                )

        if is_initializer:
            self._initializer_ledger = restored_ledger
            try:
                self._persist_initializer_ledger()
            except Exception as exc:  # noqa: BLE001 - leave every live state untouched
                self._initializer_ledger = current_ledger
                staged.unlink(missing_ok=True)
                return False, f"Could not persist the previous object ledger: {exc}"
            try:
                self._restore_scene_graph_bytes(
                    prior_graph, durable=typed_initializer_undo
                )
            except Exception as exc:  # noqa: BLE001
                self._initializer_ledger = current_ledger
                try:
                    self._persist_initializer_ledger()
                except Exception as ledger_exc:  # noqa: BLE001
                    staged.unlink(missing_ok=True)
                    return False, (
                        f"Could not restore the previous scene graph: {exc}; ledger "
                        f"rollback also failed: {ledger_exc}"
                    )
                staged.unlink(missing_ok=True)
                return False, f"Could not restore the previous scene graph: {exc}"
            if artifact_before is not None:
                try:
                    self._restore_runtime_artifacts(
                        artifact_before, durable=typed_initializer_undo
                    )
                except Exception as exc:  # noqa: BLE001
                    self._initializer_ledger = current_ledger
                    try:
                        self._persist_initializer_ledger()
                        self._restore_scene_graph_bytes(current_graph)
                        if current_artifacts is not None:
                            self._restore_runtime_artifacts(current_artifacts)
                    except Exception as rollback_exc:  # noqa: BLE001
                        staged.unlink(missing_ok=True)
                        return False, (
                            "Could not restore runtime artifacts and rollback also "
                            f"failed: {exc}; {rollback_exc}"
                        )
                    staged.unlink(missing_ok=True)
                    return False, f"Could not restore runtime artifacts: {exc}"

        try:
            os.replace(staged, live)
        except OSError as exc:
            staged.unlink(missing_ok=True)
            rollback_error = ""
            if is_initializer:
                try:
                    self._restore_scene_graph_bytes(current_graph)
                except Exception as graph_exc:  # noqa: BLE001
                    rollback_error += f"; graph rollback also failed: {graph_exc}"
                self._initializer_ledger = current_ledger
                try:
                    self._persist_initializer_ledger()
                except Exception as ledger_exc:  # noqa: BLE001
                    rollback_error = f"; ledger rollback also failed: {ledger_exc}"
                if current_artifacts is not None:
                    try:
                        self._restore_runtime_artifacts(current_artifacts)
                    except Exception as artifact_exc:  # noqa: BLE001
                        rollback_error += (
                            f"; runtime-artifact rollback also failed: {artifact_exc}"
                        )
            return (
                False,
                f"Could not restore the previous scene state: {exc}{rollback_error}",
            )
        if typed_initializer_undo:
            try:
                # The staged file was fsynced above. Persist the live-name switch
                # before an undo decision can make this U state irreversible.
                self._fsync_directory(live.parent)
            except OSError as exc:
                return False, (
                    "Typed initializer undo reached an uncertain live-scene "
                    f"publication ({exc}); restart the initializer before any "
                    "further scene operation"
                )

        typed_undo_warning = ""
        if typed_initializer_undo:
            txid = int(meta["transaction_id"])
            kind = str(meta["kind"])
            undo_decision_started = False
            undo_commit_decided = False
            undo_details = None
            try:
                is_batch = kind == "execute_and_evaluate_objects"
                if not is_batch and (not meta.get("object_id") or not typed_target):
                    raise RuntimeError("typed undo target identity is unavailable")
                restored_placement = self._typed_initializer_json_file(
                    Path(self.moge_dir) / "placement.json"
                )
                restored_names = [
                    str(row.get("mesh_name"))
                    for row in restored_placement.get("objects", [])
                    if isinstance(row, dict) and row.get("mesh_name")
                ]
                if len(restored_names) != len(set(restored_names)):
                    raise RuntimeError(
                        "typed undo restored duplicate placement identities"
                    )
                self._pipeline_object_names = restored_names
                self._n2id_cache = None
                target_states = None
                if is_batch:
                    batch = next(
                        row
                        for row in current_ledger["active_transactions"]
                        if row["id"] == txid
                    )
                    declarations = batch["declarations"]
                    identities = {
                        **self._name2id(),
                        **{
                            name: oid
                            for oid, name in declarations["added_names"].items()
                        },
                        **{
                            name: oid
                            for oid, name in declarations["removed_names"].items()
                        },
                    }
                    target_states = [
                        {
                            "object_id": identities[name],
                            "mesh_name": name,
                            "present": row["before_signature"] is not None,
                        }
                        for name, row in batch["objects"].items()
                    ]
                artifact_manifest = self._typed_initializer_artifact_manifest(
                    txid=txid,
                    kind=kind,
                    state="undone",
                    object_id=None if is_batch else str(meta["object_id"]),
                    target_mesh=typed_target,
                    target_expected_present=kind != "add_object",
                    target_states=target_states,
                    strict=True,
                )
                undo_details = {
                    "object_id": str(meta["object_id"]),
                    "mesh_name": typed_target,
                    "scene_mutation": "restored_pre_transaction_state",
                    "artifact_manifest": artifact_manifest,
                }
                undo_event = {
                    "transaction_id": txid,
                    "kind": kind,
                    "status": "undone",
                    "undo_operation_id": f"initializer-tx-{txid}-undo",
                    "details": copy.deepcopy(undo_details),
                }
                # Publish one WAL-style decision only after the complete undone
                # state and its strict artifact manifest exist. Before this atomic
                # marker write the committed snapshot wins; at/after it, startup
                # rolls the exact embedded event forward and never compensates.
                undo_decision_started = True
                decision_details = {
                    "undo_event": copy.deepcopy(undo_event),
                    "artifact_manifest": copy.deepcopy(artifact_manifest),
                    "post_undo_ledger": copy.deepcopy(self._initializer_ledger),
                    "post_undo_pipeline_object_names": list(restored_names),
                }
                decision_details["decision_sha256"] = (
                    self._typed_initializer_json_sha256(decision_details)
                )
                try:
                    self._set_runtime_recovery_state(
                        txid,
                        "undo_commit_decided",
                        details=decision_details,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Atomic replacement followed by a failed directory fsync is
                    # deliberately ambiguous. Visibility in this process does not
                    # prove crash durability, so never reread-and-assume or discard
                    # the undo history. Startup will choose C or U from the marker
                    # that actually survived.
                    return False, (
                        "Typed initializer undo reached an uncertain commit-point "
                        f"publication ({exc}); restart the initializer before any "
                        "further scene operation"
                    )
                undo_commit_decided = True
                finalization_errors = []
                try:
                    visible = self._last_mutation_journal_event(txid, kind)
                    if not (
                        visible is not None
                        and visible.get("status") == "undone"
                        and visible.get("undo_operation_id")
                        == undo_event["undo_operation_id"]
                        and (visible.get("details") or {}) == undo_details
                    ):
                        self._append_mutation_event(undo_event)
                    self._reconcile_recovered_terminal_event(undo_event)
                except Exception as finalize_exc:  # noqa: BLE001
                    finalization_errors.append(
                        f"journal/ledger finalization failed: {finalize_exc}"
                    )
                if not finalization_errors:
                    try:
                        self._set_runtime_recovery_state(
                            txid,
                            "undone",
                            details=undo_details,
                        )
                    except Exception as finalize_exc:  # noqa: BLE001
                        finalization_errors.append(
                            f"recovery-marker finalization failed: {finalize_exc}"
                        )
                if finalization_errors:
                    typed_undo_warning = (
                        "Undo is durably committed, but bookkeeping remains recovery-"
                        "pending; restart the initializer before another mutation: "
                        + "; ".join(finalization_errors)
                    )
            except Exception as exc:  # noqa: BLE001
                if undo_decision_started and not undo_commit_decided:
                    return False, (
                        "Typed initializer undo reached an uncertain commit-point "
                        f"publication ({exc}); restart the initializer before any "
                        "further scene operation"
                    )
                if undo_commit_decided:
                    # An unexpected post-decision exception cannot reverse the WAL
                    # decision. Leave it for idempotent startup roll-forward.
                    typed_undo_warning = (
                        "Undo is durably committed, but finalization was interrupted; "
                        f"restart the initializer before another mutation: {exc}"
                    )
                else:
                    rollback_errors = []
                    rollback_stage = live.with_name(
                        f".{live.name}.undo-audit-rollback-{os.getpid()}-"
                        f"{getattr(self, 'count', 0)}"
                    )
                    try:
                        shutil.copy2(self.edit_history[-1], rollback_stage)
                        os.replace(rollback_stage, live)
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_stage.unlink(missing_ok=True)
                        rollback_errors.append(f"Blend rollback failed: {rollback_exc}")
                    try:
                        self._restore_scene_graph_bytes(current_graph)
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_errors.append(f"graph rollback failed: {rollback_exc}")
                    if current_artifacts is not None:
                        try:
                            self._restore_runtime_artifacts(current_artifacts)
                        except Exception as rollback_exc:  # noqa: BLE001
                            rollback_errors.append(
                                f"runtime-artifact rollback failed: {rollback_exc}"
                            )
                    self._initializer_ledger = current_ledger
                    self._pipeline_object_names = current_pipeline_names
                    self._n2id_cache = None
                    try:
                        self._persist_initializer_ledger()
                    except Exception as rollback_exc:  # noqa: BLE001
                        rollback_errors.append(
                            f"ledger rollback failed: {rollback_exc}"
                        )
                    rollback_manifest = self._safe_typed_initializer_artifact_manifest(
                        txid=txid,
                        kind=kind,
                        state="error",
                        object_id=str(meta.get("object_id") or "") or None,
                        target_mesh=typed_target,
                        target_expected_present=kind != "remove_runtime_object",
                    )
                    error_event = {
                        "transaction_id": txid,
                        "kind": kind,
                        "status": "error",
                        "details": {
                            "phase": "undo_audit",
                            "error": str(exc),
                            "scene_mutation": "undo_reverted",
                            "artifact_manifest": rollback_manifest,
                            **(
                                {"restore_errors": rollback_errors}
                                if rollback_errors
                                else {}
                            ),
                        },
                    }
                    audit_complete = False
                    try:
                        self._append_mutation_event(error_event)
                        self._reconcile_recovered_terminal_event(error_event)
                        audit_complete = True
                    except Exception as audit_exc:  # noqa: BLE001
                        rollback_errors.append(f"terminal audit failed: {audit_exc}")
                    if (
                        audit_complete
                        and typed_recovery_marker_before is not None
                        and not rollback_errors
                    ):
                        try:
                            self._durable_atomic_write_bytes(
                                self._runtime_recovery_marker_path(txid),
                                typed_recovery_marker_before,
                            )
                            self._prune_runtime_recovery_payload(txid)
                        except Exception as rollback_exc:  # noqa: BLE001
                            rollback_errors.append(
                                f"recovery-marker rollback failed: {rollback_exc}"
                            )
                    message = f"Could not durably audit typed initializer undo: {exc}"
                    if rollback_errors:
                        message += "; " + "; ".join(rollback_errors)
                    return False, message

        if (
            not is_initializer
            and record_undo_event
            and meta.get("transaction_id") is not None
            and meta.get("kind") == "edit_object_poses"
        ):
            # The scene replacement above is already complete, so an audit-storage
            # failure cannot safely turn this into a failed undo (a retry would undo
            # the preceding edit). Record the terminal status best-effort and return
            # an explicit warning while still consuming this undo-stack entry.
            try:
                self._record_mutation_status(
                    meta["transaction_id"],
                    "edit_object_poses",
                    "undone",
                    details={
                        "scene_mutation": "restored_pre_transaction_state",
                        **(
                            {"object_ids": str(meta["object_id"]).split(",")}
                            if meta.get("object_id")
                            else {}
                        ),
                        **(
                            {"edit_token": str(meta["edit_token"])}
                            if meta.get("edit_token")
                            else {}
                        ),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - scene is already restored
                typed_undo_warning = (
                    "Scene undo succeeded, but the composition mutation journal "
                    f"could not record its undone status: {exc}"
                )
                logging.error(typed_undo_warning)

        # No fallible external operations remain after the live blend replacement.
        self.edit_history.pop()
        if getattr(self, "_edit_meta", None):
            self._edit_meta.pop()
        if getattr(self, "_ledger_history", None):
            self._ledger_history.pop()
        if getattr(self, "_graph_history", None):
            self._graph_history.pop()
        self._restore_patch_base_after_undo()
        if is_initializer:
            if getattr(self, "_ledger_history", None):
                self._ledger_history[-1] = copy.deepcopy(self._initializer_ledger)
            else:
                self._ledger_base = copy.deepcopy(self._initializer_ledger)
            if getattr(self, "_graph_history", None):
                self._graph_history[-1] = self._read_scene_graph_bytes()
            else:
                self._graph_base = self._read_scene_graph_bytes()
            if (
                not typed_initializer_undo
                and record_undo_event
                and meta.get("transaction_id") is not None
                and meta.get("kind")
                in {
                    "execute_and_evaluate_objects",
                    "add_object",
                    "edit_object_mesh",
                    "edit_object_pose",
                    "remove_runtime_object",
                }
            ):
                try:
                    self._append_mutation_event(
                        {
                            "transaction_id": meta["transaction_id"],
                            "kind": meta["kind"],
                            "status": "undone",
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - state is already restored
                    logging.error("failed to append mutation undo audit: %s", exc)
        self._pose_dirty = True
        self._armed = set()
        pending = self._pending_post_flip_followup()
        if (
            pending is not None
            and meta.get("kind") == "flip"
            and meta.get("object_id") in set(pending.get("object_ids") or [])
            and str(meta.get("edit_token")) == str(pending.get("token"))
        ):
            self._last_undo_resolved_followup = self._resolve_post_flip_followup(
                token=str(meta["edit_token"]), status="cancelled_by_undo"
            )
        return True, typed_undo_warning

    def _settle_after_edit(
        self, intended_resting_modes: Optional[dict[str, object]] = None
    ) -> Optional[dict]:
        """Composition only: physics-settle whatever a freeform
        execute_and_evaluate edit RELOCATED (lift-to-clear + settle + carry, via the
        physics authority), persist to the shared blend so the render reflects it, and
        return a keep/undo report. ``None`` means the non-composition path only;
        composition returns an explicit settled/not-needed/error state. A completed
        settle remains reviewable even when its raw diagnostics flag an undesirable
        pose; the agent sees that result and decides whether to keep or undo it."""
        if self.root_stage_name != "composition":
            return None
        strict_physics = self._has_capability("strict_post_edit_physics")
        if not getattr(self, "_freeform_edit_prepared", False):
            unavailable = {
                "status": "unavailable",
                "reports": [],
                "reason": "physics was not initialized before the edit",
            }
            if strict_physics:
                unavailable.update(
                    {
                        "failure_kind": "infrastructure_error",
                        "recovery_succeeded": False,
                        "recovery_errors": [unavailable["reason"]],
                    }
                )
            return unavailable
        if self._physics_obj is None or self._pose_session is None:
            unavailable = {
                "status": "unavailable",
                "reports": [],
                "reason": "composition physics is not attached",
            }
            if strict_physics:
                unavailable.update(
                    {
                        "failure_kind": "infrastructure_error",
                        "recovery_succeeded": False,
                        "recovery_errors": [unavailable["reason"]],
                    }
                )
            return unavailable
        from lib.tools.geometry.composition_physics import (
            PhysicsSettlementRejected,
        )

        try:
            if self._pose_dirty:
                # reflect the edit in the session, reattach physics WITHOUT resyncing:
                # settle_edited diffs the edited blend vs the pre-edit _synced to find
                # what moved, so the pre-edit baseline must survive.
                self._pose_session.rebuild()
                self._pose_session.physics = self._physics_obj
                self._pose_dirty = False
                if not strict_physics:
                    self._armed = set()
            # the settle gate's sibling allowance uses the rules gate's support map
            # (root surfaces included) instead of carry parents alone (audit F-M11)
            self._physics_obj.support_map = self._support_body_map() or {}
            reports = self._physics_obj.settle_edited(
                self._pose_session,
                intended_resting_modes=intended_resting_modes,
            )
            if not reports:
                return {"status": "not_needed", "reports": []}
            self._pose_session.save()  # persist settled poses so the render sees them
            # execute() already snapshotted the PRE-settle blend into edit_history;
            # refresh that entry to the SETTLED state so undo_last_step restores what
            # the agent actually saw (the physics-resolved edit), not the raw teleport.
            if (
                self.edit_history
                and self.blender_save
                and os.path.exists(self.blender_save)
            ):
                shutil.copy(self.blender_save, self.edit_history[-1])
            for r in reports:
                self._freeform[r["name"]] = self._freeform.get(r["name"], 0) + 1
            # Keep the full physics diagnostics internal for audit/debugging. They are
            # not an automatic verdict: the returned render is the keep/undo authority.
            return {"status": "settled", "reports": reports}
        except PhysicsSettlementRejected as exc:
            import sys as _sys

            print(f"[settle-after-edit] rejected: {exc}", file=_sys.stderr)
            return {
                # Keep the historical status so the raw composition edit wrapper
                # still enters its rollback branch; failure_kind carries the new
                # machine-readable distinction from infrastructure errors.
                "status": "error",
                "failure_kind": "physics_rejection",
                "reports": copy.deepcopy(getattr(exc, "reports", []) or []),
                "reason": str(exc),
            }
        except Exception as exc:  # noqa: BLE001 - caller rolls the raw edit back
            import sys as _sys

            print(f"[settle-after-edit] failed: {exc}", file=_sys.stderr)
            result = {
                "status": "error",
                "reports": copy.deepcopy(getattr(exc, "reports", []) or []),
                "reason": str(exc),
            }
            if strict_physics:
                recovery_errors: list[str] = []
                try:
                    recovery_errors.extend(
                        self._physics_obj.recover_after_error(self._pose_session) or []
                    )
                except Exception as recovery_exc:  # noqa: BLE001
                    recovery_errors.append(f"physics recovery failed: {recovery_exc}")
                result.update(
                    {
                        "failure_kind": "infrastructure_error",
                        "recovery_succeeded": not recovery_errors,
                        "recovery_errors": recovery_errors,
                    }
                )
            return result
        finally:
            self._freeform_edit_prepared = False

    def _freeform_settle_note(self, reports: list[dict]) -> str:
        """Report a completed settle without turning raw diagnostics into a verdict.

        The crop is the keep/undo authority, so tilt angles and roll displacement stay
        in the internal ``reports`` rather than biasing the agent-facing text. Objects
        use SCENE-GRAPH ids (croissant#2), matching crops and refinement tools. Soft
        freeform-budget usage and post-settle registration are still reported."""
        to_oid = self._oid
        settled: list[str] = []
        for r in reports:
            for member in [r["name"], *(r.get("members") or [])]:
                name = to_oid(member)
                if name not in settled:
                    settled.append(name)
        parts = [
            "Completed for "
            + ", ".join(settled)
            + (
                " (settled together in one simulation)"
                if any(r.get("joint_settle") for r in reports)
                else ""
            )
            + "; the result is shown above. KEEP it if it matches the target, or "
            "call undo_last_step if it toppled, still overlaps, or drifted from the "
            "target."
        ]
        over = [
            to_oid(r["name"]) for r in reports
            if self._freeform.get(r["name"], 0) >= self.freeform_cap
        ]  # fmt: skip
        if over:
            if self._has_capability("strict_post_edit_physics"):
                parts.append(
                    f"({self.freeform_cap} layout edits on {', '.join(over)} so far — "
                    "re-investigate before editing them again.)"
                )
            else:
                parts.append(
                    f"(Freeform budget reached on {', '.join(over)} — prefer move() from here.)"
                )
        # Post-settle IoU per moved object (riders included): without it the agent's
        # keep/undo call — and the memory state table — inherit the PRE-edit
        # measurement (a kept rescue edit left knife#0 showing IoU 0.00 forever once
        # its investigate cap was spent). Emitter format matches investigate_objects
        # exactly so the ledger/state-table parsers pick it up unchanged. Best-effort:
        # a scoring hiccup must not kill the note.
        try:
            session = getattr(self, "_pose_session", None)
            if session is not None:
                mesh2oid = {v["mesh_name"]: k for k, v in session.ctx.items()}
                oids: list[str] = []
                for r in reports:
                    for m in r.get("members") or [r["name"]]:
                        object_id = m if m in session.ctx else mesh2oid.get(m)
                        if object_id and object_id not in oids:
                            oids.append(object_id)
                lines = []
                post_mats = self._freeform_post_world_matrices(session, oids)
                post_world = self._yaws_from_matrices(post_mats)
                pre_world = getattr(self, "_pre_edit_world_yaw", None) or {}
                emitted_yaw = emitted_app = False
                # the orientation trailer below is suppressed only when the yaw
                # snapshots AFFIRM a pure translation (every settled object
                # measurable and unturned): there the raw numbers ARE the best
                # position evidence and the "yaw re-measure" pointer would
                # dangle. Any unmeasurable object keeps the trailer — dropping
                # it on a yaw edit would un-fix the bridge_6 trap.
                orientation_edit = False
                for object_id in oids:
                    # BASELINE: cur[oid] is the object's last recorded score, i.e. the
                    # pre-edit one — and the re-measure below discards it. Without it the
                    # agent is asked "did your edit worsen the match?" and handed only
                    # absolutes: 0727_tl_e2e_abc4 undid a CORRECT +20deg stand rotation
                    # because cup#4 read IoU 0.00, a permanent condition (the cup is
                    # buried in the stand mesh — 0.00 at every measurement of the run,
                    # before and after) that looked like fresh damage. It re-applied the
                    # same edit 10 rounds later once it had gathered the baseline itself.
                    # No better/worse VERDICT on purpose: that same rotation moved IoU
                    # 0.58 -> 0.57 (the object is ~25% oversized, so containment flattens
                    # IoU's response to yaw), and a "worse" label would have reinforced
                    # the wrong undo. Numbers + baseline; the crops decide.
                    # cur is wiped by the settle path's session rebuild, so fall
                    # back to the pre-edit snapshot (_prepare_freeform_edit) —
                    # either source is "the last measurement before this edit".
                    prev = session.cur.get(object_id) or (
                        getattr(self, "_pre_edit_cur", None) or {}
                    ).get(object_id)
                    session.cur[object_id] = None  # re-measure at the settled pose
                    sc = session.score(object_id)
                    # the CURRENT values must stay leading (score BEFORE IoU —
                    # uniform-frontend order): prompt_builder's _INV_IOU reads the
                    # state table off this line and would capture a trailing
                    # "(was ...)" value as the current one.
                    line = (
                        f"- {object_id}: "
                        + (f"score {sc['score']:.3f}, " if "score" in sc else "")
                        + f"silhouette IoU {sc['iou']:.2f}"
                    )
                    if prev and "iou" in prev:
                        line += (
                            " (was "
                            + (f"{prev['score']:.3f} / " if "score" in prev else "")
                            + f"{prev['iou']:.2f} at its last measurement)"
                        )
                    else:
                        line += (
                            " (FIRST measurement — no baseline, so a low value here is "
                            "not evidence your edit broke it)"
                        )
                    yaw_clause = self._freeform_yaw_line(session, object_id, post_world)
                    app_clause = self._freeform_appearance_line(
                        session, object_id, post_world, post_mats
                    )
                    emitted_yaw = emitted_yaw or bool(yaw_clause)
                    emitted_app = emitted_app or bool(app_clause)
                    line += yaw_clause + app_clause
                    lines.append(line)
                    pw, qw = pre_world.get(object_id), post_world.get(object_id)
                    if (
                        pw is None
                        or qw is None
                        or abs((qw - pw + 180.0) % 360.0 - 180.0)
                        >= _FREEFORM_YAW_EDIT_DEG
                    ):
                        orientation_edit = True
                if lines:
                    # bridge_6 trap: the agent undid a CORRECT +23deg yaw fix on a
                    # raw-IoU drop here — say plainly which evidence owns keep/undo.
                    # The pointer names ONLY the evidence actually present beside
                    # the crop (P1/P2): pointing at a "yaw re-measure" whose pair
                    # was omitted would send the agent hunting for a number that
                    # is not there.
                    pointers = ["the returned crop"]
                    if emitted_yaw:
                        pointers.append("the yaw re-measure")
                    if emitted_app:
                        pointers.append("the appearance agreement")
                    ptr = (
                        pointers[0]
                        if len(pointers) == 1
                        else ", ".join(pointers[:-1]) + " and " + pointers[-1]
                    )
                    # Both pairs silently omitted on an orientation edit (weak
                    # silhouette axis AND appearance unavailable) leaves ONLY
                    # discounted raw numbers on screen — flag the blind spot
                    # instead of letting the message read keep/undo-biased
                    # (a channel-status note, not a weak-number placeholder).
                    no_channel = (
                        " No measured channel could read this edit's "
                        "orientation change — the crop is the ONLY evidence."
                        if not (emitted_yaw or emitted_app)
                        else ""
                    )
                    parts.append(
                        "Post-settle registration (current pose vs target):\n"
                        + "\n".join(lines)
                        + (
                            "\nThese numbers are raw silhouette overlap — NOT an "
                            "authority on orientation edits (a CORRECT yaw fix can "
                            f"LOWER them): judge {ptr} before keep/undo." + no_channel
                            if orientation_edit
                            else ""
                        )
                    )
        except Exception:  # noqa: BLE001 - IoU lines are best-effort
            pass
        return "Physics settle of your edit: " + " ".join(parts)

    @staticmethod
    def _world_matrices(session, oids: Optional[list] = None) -> dict:
        """Full 4x4 world matrix per scene-graph id, read off the live blend via the
        register server's get_matrix. The yaw before->after line derives its angles
        from these (_yaws_from_matrices); the appearance-agreement pair passes them
        to the backend verbatim (it must render the EXACT pre-edit pose, P1).
        Best-effort: {} on any failure, both consumers are advisory."""
        try:
            ids = list(oids) if oids is not None else list(session.ctx)
            names = {
                oid: session.ctx[oid]["mesh_name"] for oid in ids if oid in session.ctx
            }
            ms = session.client.rpc(
                {"cmd": "get_matrix", "names": list(names.values())}
            )["matrices"]
            return {oid: ms[n] for oid, n in names.items() if ms.get(n) is not None}
        except Exception:  # noqa: BLE001 - advisory only
            return {}

    @staticmethod
    def _yaws_from_matrices(mats: dict) -> dict:
        """World-frame yaw (deg) per id off get_matrix output (atan2 of the
        rotation's XY column — enough to DETECT a yaw-type edit; the agent-facing
        angles come from scale_hint)."""
        return {
            oid: math.degrees(math.atan2(m[1][0], m[0][0])) for oid, m in mats.items()
        }

    def _freeform_post_world_matrices(self, session, oids: list) -> dict:
        """Settled-pose world matrices for the yaw/appearance before->after lines —
        only when the pre-edit snapshot exists (captured by _prepare_freeform_edit),
        so the comparison is meaningful; {} disables the lines."""
        if not getattr(self, "_pre_edit_world_yaw", None):
            return {}
        return self._world_matrices(session, oids)

    def _freeform_yaw_line(self, session, object_id: str, post_world: dict) -> str:
        """Yaw before->after for a freeform edit that CHANGED this object's world
        yaw (bridge_6: the raw-IoU drop was the only reading on a CORRECT +23deg
        fix, and the agent undid it). pre = the object's last cached scale_hint
        reading at the pre-edit pose (snapshotted by _prepare_freeform_edit — the
        settle's session rebuild wipes the cache); post = a fresh reading at the
        settled pose. Either side unmeasurable -> no line, never fabricate.
        BOTH-RELIABLE RULE (P2): a weak-axis reading (aniso < _YAW_ANISO_MIN on
        either side) ALIASES — bridge_6 read 28->50 at aniso 1.09 across a CORRECT
        +12deg fix, and the same channel read 55.8->12.7 across a scale-only move
        — so the pair is OMITTED SILENTLY here. No qualifier, no placeholder: this
        is a decision (edit-feedback) surface; the investigate-time F2 LEAD note
        is the sanctioned place a weak reading may be cited."""
        try:
            from lib.tools.geometry.register import _YAW_ANISO_MIN

            pre_world = (getattr(self, "_pre_edit_world_yaw", None) or {}).get(
                object_id
            )
            post_w = post_world.get(object_id)
            pre_read = (getattr(self, "_pre_edit_yaw_reading", None) or {}).get(
                object_id
            )
            if pre_world is None or post_w is None or pre_read is None:
                return ""
            turned = abs((post_w - pre_world + 180.0) % 360.0 - 180.0)
            if turned < _FREEFORM_YAW_EDIT_DEG:
                return ""  # the edit did not change this object's yaw
            scale_fn = getattr(session, "scale_hint", None)
            if scale_fn is None:
                return ""
            sh = scale_fn(object_id, prefix="settle") or {}
            post_yaw = sh.get("yaw_deg")
            if post_yaw is None:
                return ""
            pre_yaw, pre_aniso = pre_read
            if min(pre_aniso or 0.0, sh.get("yaw_aniso") or 0.0) < _YAW_ANISO_MIN:
                return ""  # weak on either side: omit silently (P2)
            return (
                f"; measured yaw: ~{pre_yaw:.0f}deg -> ~{post_yaw:.0f}deg vs the photo"
            )
        except Exception:  # noqa: BLE001 - advisory only
            return ""

    def _freeform_appearance_line(
        self, session, object_id: str, post_world: dict, post_mats: dict
    ) -> str:
        """Appearance-agreement before->after for a freeform yaw-type edit (P1,
        bridge_6): the position/scale-invariant DINO similarity of the isolated
        object crop vs the photo crop, scored by the register backend at the
        pre-edit pose (exec's world-matrix snapshot — the pose no longer exists in
        the blend) and at the settled pose. It is the reliable channel exactly
        where the silhouette yaw pair goes quiet (compact/textured objects), and
        bridge_6's rising sim0 (0.4343 -> 0.4927) was never surfaced while the
        aliased yaw pair was. Gated on the SAME yaw-turn threshold as the yaw line
        so non-yaw edits never spend the two renders — measured on the OUTCOME
        (pre-snapshot vs post-settle world yaw), deliberately: a translation
        whose settle turned the object >= 3deg IS an orientation change the
        agent must judge, so it spends the pair like any yaw edit. Either side
        unavailable (worker dead, crop failure, no backend) -> omit silently."""
        try:
            pre_world = (getattr(self, "_pre_edit_world_yaw", None) or {}).get(
                object_id
            )
            post_w = post_world.get(object_id)
            if pre_world is None or post_w is None:
                return ""
            if abs((post_w - pre_world + 180.0) % 360.0 - 180.0) < (
                _FREEFORM_YAW_EDIT_DEG
            ):
                return ""  # not a yaw-type edit: no appearance cost on relocations
            app_fn = getattr(session, "appearance_agreement", None)
            pre_m = (getattr(self, "_pre_edit_world_matrix", None) or {}).get(object_id)
            post_m = post_mats.get(object_id)
            if app_fn is None or pre_m is None or post_m is None:
                return ""
            pre = app_fn(object_id, pre_m)
            post = app_fn(object_id, post_m)
            if pre is None or post is None:
                return ""
            return (
                f"; appearance agreement vs photo: {pre:.2f} -> {post:.2f} "
                f"({_appearance_pair_qualifier(pre, post)})"
            )
        except Exception:  # noqa: BLE001 - advisory only
            return ""

    def pose_eligible_ids(self) -> list[str]:
        """Objects the coverage gate expects to be investigated: meshed + masked
        instances from the placement table (static — no Blender needed)."""
        self._pose_eligibility_error = None
        paths = self._pose_paths()
        if paths is None:
            self._pose_eligibility_error = "required pose artifact paths are missing"
            return []
        try:
            place = json.load(open(paths["placement"]))["objects"]
            masks = {
                (r["category"], r["instance"])
                for r in json.load(open(paths["masks"]))["instances"]
                if r.get("mask_path")
            }
            excluded_ids = set()
            overlay = None
            if self._has_capability("runtime_object_inventory"):
                from lib.tools.geometry.runtime_object_repair import (
                    load_runtime_inventory,
                )

                overlay = load_runtime_inventory(Path(paths["placement"]).parent)
                for record in (overlay or {}).get("objects", []):
                    if record.get("mask_policy") == "excluded":
                        excluded_ids.add(record["id"])
                        masks.discard((record["category"], int(record["instance"])))
                        continue
                    if record.get("status") == "committed" and record.get("mask_path"):
                        masks.add((record["category"], int(record["instance"])))
            eligible = [
                f"{o['category']}#{o['instance']}"
                for o in place
                if o.get("mesh_glb") and (o["category"], o["instance"]) in masks
            ]
            if not eligible and not (
                self._has_capability("runtime_object_inventory")
                and overlay is not None
                and (
                    (
                        bool(place)
                        and all(
                            f"{row['category']}#{row['instance']}" in excluded_ids
                            for row in place
                        )
                    )
                    or (not place and self._has_committed_empty_object_inventory())
                )
            ):
                self._pose_eligibility_error = (
                    "placement/mask artifacts contain no meshed, masked objects"
                )
            return eligible
        except Exception as exc:  # noqa: BLE001 - surfaced as UNVERIFIED by gate
            self._pose_eligibility_error = (
                f"could not read placement/mask artifacts: {exc}"
            )
            return []

    def _size_locked_ids(self) -> set:
        """Scene-graph ids whose physical size is frozen upstream.

        This includes successfully normalized same-size registry groups and registered
        prescan/catalog assets. Composition may still refine their image-plane pose, but
        it must not distort a measured mesh by changing its calibrated uniform scale.
        """
        cache = getattr(self, "_size_locked_cache", None)
        if cache is not None:
            return cache
        ids: set = set()
        paths = self._pose_paths()
        if paths is not None:
            try:
                place = json.load(open(paths["placement"]))["objects"]
                ids = {
                    f"{o['category']}#{o['instance']}"
                    for o in place
                    if o.get("same_size") or o.get("scale_locked")
                }
            except Exception:  # noqa: BLE001 - advisory when artifacts are odd
                ids = set()
        self._size_locked_cache = ids
        return ids

    def _size_lock_guidance(self, obj: str, next_step: str) -> str:
        """One uncertainty-preserving explanation for every upstream size lock.

        The lock may come from same-size normalization or a calibrated prescanned asset.
        It removes ``scale`` from the composition tool without prescribing which remaining
        pose degree of freedom explains an image mismatch.
        """
        return (
            f"SIZE-LOCK NOTE for {obj}: preprocessing set a LOCKED measured "
            "physical size (from same-size normalization or calibrated asset "
            "registration), so scale edits are unavailable. Refine only eligible pose "
            "degrees of freedom—not because depth is proven, but because scale is no "
            f"longer eligible. {next_step} If a correctly placed object still "
            "mismatches, the asset match/calibration or the classification, shared "
            "measurement, or reconstruction may be wrong."
        )

    def _direct_pose_tool_phrase(self) -> str:
        """Profile-specific explicit pose route used in composition feedback."""
        if self._has_capability("composition_direct_pose_edit"):
            return (
                "edit_object_poses (physics simulation follows the requested edit, "
                "so inspect its settled pose)"
            )
        return "execute_and_evaluate"

    def _object_hint_lines(
        self,
        o: str,
        orient: Optional[dict],
        sh: Optional[dict],
        iou: float,
        *,
        allow_rotate_180: bool = True,
        emit_orientation_check: bool = True,
    ) -> list[str]:
        """Reconcile one object's hints by the precedence chain
        rotate_180 > rotation(yaw) > position > scale, except position outranks
        rotation while the object is displaced (gap >= _POS_HINT_GAP, unless a flip
        is deferred behind the yaw). Emit the FACING guard — a VERIFIED flip
        recommendation (margin >= REVERSED_VERIFIED_MARGIN; suppresses the chain),
        a SUSPECTED-flip note (weaker margin; suppresses nothing), or the
        verified-SAME clearance — then the SINGLE highest-precedence firing
        refinement as the recommended next move, with lower-precedence mismatches
        NAMED but deferred — they are measured at the current (wrong) pose, so
        their magnitudes are trustworthy only after the upstream fix, and are
        re-checked in the move feedback / next investigate. An in-band yaw on a
        weak axis is surfaced as a non-recommending LEAD (never seeded).
        ``allow_rotate_180=False`` (spent one-shot) reroutes every rotate_180
        instruction to the profile's direct-yaw tool."""
        from lib.tools.geometry.feature_metric import (
            REVERSED_VERIFIED_MARGIN,
            VERIFIED_SAME_MARGIN,
            worker_dead,
        )
        from lib.tools.geometry.register import (
            _FLIP_DEFER_ANISO_MIN,
            _POS_HINT_GAP,
            _YAW_ANISO_MIN,
            _YAW_HINT_MAX,
            _YAW_HINT_MIN,
            size_hint_aspect_suppressed,
            yaw_hint_fires,
        )

        lines: list[str] = []
        direct_pose_tool = self._direct_pose_tool_phrase()
        # An EMPTY render silhouette VOIDS every measurement below (area_scale, the
        # principal axis and the shift are all None), so report that instead of falling
        # through to a bare no-hint line: "your object rendered nothing" is different
        # information from "nothing was measurable about it". Usual cause is full occlusion
        # by its own stacking ANCESTOR -- _isolate_iou renders the object visible and its
        # ancestors as occluding HOLDOUTS while hiding every other body, so only an ancestor
        # can do this -- or an off-frame / sunk pose. 40 of 11248 fleet rows, 31 objects;
        # traced on 0802_bulk_misc_online5 pen#0 (support laptop#0). `.get` keeps pre-0803
        # rows, which have no render_px, on the old path.
        if sh is not None and sh.get("render_px") == 0 and sh.get("mask_px"):
            lines.append(
                f"- {o} rendered NO pixels: its silhouette measurements are UNAVAILABLE "
                "(which is NOT the same as 'correct'). It is fully hidden behind whatever "
                "it sits on, off-frame, or sunk into its support — inspect the crops, and "
                "check whether its support assignment is right."
            )
            return lines
        # a same-size object's physical size is FROZEN — move('scale') is hard-rejected
        # for it (see move_object), so its size hints must steer to DEPTH only.
        size_locked = o in self._size_locked_ids()
        flip_flagged = bool(orient and orient.get("flagged"))
        gap = (sh["overlap"] - iou) if sh else 0.0
        shift = sh.get("shift_px") if sh else None
        yaw = sh.get("yaw_deg") if sh else None
        yaw_aniso = sh.get("yaw_aniso") if sh else None
        # a flagged flip on a STRONGLY elongated silhouette whose axis is also
        # yawed vs the photo is unreliable (the stretched-crop DINO compare ran
        # on non-comparable shapes — the 0721 abc2 fork FP); defer it behind the
        # yaw fix instead of recommending it. No _YAW_HINT_MAX cap: the compfix
        # abc2 knife FP measured 44.8 deg. See _FLIP_DEFER_ANISO_MIN.
        # This is sound ONLY because the deferral's aniso floor IS the elongation
        # tier of yaw_hint_fires: every deferred flip therefore has a firing
        # YAW HINT below, so "fix the yaw first" always names a move that
        # exists. Lower that floor below _FLIP_DEFER_ANISO_MIN and the two split
        # apart again — the dead end this fixed (0724 roomval bottle#0 @57deg).
        flip_deferred = (
            flip_flagged
            and yaw is not None
            and yaw_aniso is not None
            and yaw >= _YAW_HINT_MIN
            and yaw_aniso >= _FLIP_DEFER_ANISO_MIN
        )
        # the REVERSED verdict scales with the DINO margin: only a margin at or
        # above REVERSED_VERIFIED_MARGIN earns "(verified)" authority (owns the
        # whole chain, early-return); a flagged-but-weaker margin is "(suspected)"
        # — surfaced (low-margin TRUE reversals exist, see feature_metric.py) but
        # demoted to a judge-the-crops-first note that suppresses nothing below it.
        flip_margin = (
            float(orient["sim180"]) - float(orient["sim0"]) if flip_flagged else 0.0
        )
        flip = (
            flip_flagged
            and not flip_deferred
            and flip_margin >= REVERSED_VERIFIED_MARGIN
        )
        flip_suspected = flip_flagged and not flip_deferred and not flip
        verified_same = bool(
            orient
            and not flip_flagged
            and orient["sim0"] - orient["sim180"] >= VERIFIED_SAME_MARGIN
        )
        yaw_axis_ok = (
            yaw is not None and yaw_aniso is not None and yaw_aniso >= _YAW_ANISO_MIN
        )
        # PRECEDENCE: position outranks rotation while the object is DISPLACED. The rotation
        # search scores on raw silhouette IoU, which is not translation-invariant — rotating
        # about the object's own centre swings its extremities, so on a displaced object the
        # score's preference between the two seeded directions is driven by "which way swings
        # me onto the mask", not "which way aligns my axis". Measured on static_scene_eval:
        # median gap 0.346 among the 6 landed rotations that made yaw WORSE vs 0.032 among the
        # 4 that fixed it, and 4 of the 6 sat above _POS_HINT_GAP — i.e. POSITION had fired
        # internally and was suppressed by rotation's precedence, in exactly the cases where
        # the rotation search cannot work.
        #
        # The two moves are not symmetric: 'xy' works fine on a yawed object (IoU is strongly
        # and monotonically sensitive to translation), while 'rotation' does not work on a
        # displaced one. Order by which tolerates the other's error. `gap` is itself the
        # yaw-robust estimate of how much translation can win, since centroid alignment
        # factors translation out of the first term.
        #
        # The chain's original "a yawed silhouette distorts the position reading" rationale
        # was written when the POSITION hint carried a ~Npx MAGNITUDE; that was removed
        # 2026-07-26 and what survives is a direction (a sign with a 3px noise gate), which a
        # modest yaw does not flip. The ordering was never revisited.
        #
        # Deliberately NOT folded into yaw_hint_fires, despite the one-predicate discipline:
        # that predicate answers "is this a recommendable rotation?" for BOTH the hint and the
        # search SEED, and a displaced object still wants the measured angle as its seed if it
        # rotates for another reason. The search-side guarantee is the yaw-regression gate in
        # optimize_axis, not this. See ROTATION_SIGN_DISPLACEMENT_PROPOSAL_2026_08_03.md.
        # `position_fires` is computed FIRST so the rule below can yield to a hint that will
        # ACTUALLY be emitted. Yielding on `gap` alone would drop the yaw into silence
        # whenever the phase correlation failed (shift None) — suppressed by a position hint
        # that never fires.
        position_fires = not flip and shift is not None and gap >= _POS_HINT_GAP
        # is this a recommendable rotation AT ALL, before precedence? Kept separate from
        # yaw_fires so the out-of-band note below keys off the GENUINE reason (a weak axis
        # past the fine range) and not off "was outranked this round".
        yaw_recommendable = yaw_hint_fires(yaw, yaw_aniso)
        # EXCEPTION to the rule above: never yield when a flip is DEFERRED behind the yaw.
        # flip_deferred's only actionable content is "fix the yaw first" (it exists to stop a
        # rotate_180 firing on a corrupted compare), so if the yaw then yielded to position the
        # agent would get two contradictory FIRSTs — precisely the dead end the 2026-07-26
        # audit removed, and pinned by test_flip_deferred_never_co_recommends_a_position_fix.
        # flip_deferred implies aniso >= 3, which is where the harm concentrates, so this leaves
        # a residual path open — the yaw-regression gate in optimize_axis is what covers it.
        yaw_yields_to_position = position_fires and not flip_deferred
        yaw_fires = not flip and not yaw_yields_to_position and yaw_recommendable
        # a recommendable yaw held back by that rule — named on the POSITION line so the agent
        # knows it is QUEUED, not absent (the courtesy the chain already extends to size)
        yaw_deferred = bool(not flip and yaw_yields_to_position and yaw_recommendable)
        # recorded-but-not-recommended: past the fine range on a WEAK axis, where the
        # fleet says move('rotation') buys ~nothing (median IoU +0.01). A strongly
        # elongated silhouette is recommendable at any yaw, so it never lands here.
        yaw_oob = (
            not flip and yaw_axis_ok and not yaw_recommendable and yaw > _YAW_HINT_MAX
        )
        scale_flag = bool(sh and sh.get("flagged"))
        # A flagged size reading whose EXTENT and AREA ratios disagree is an aspect/pose
        # mismatch, not a uniform size error: ~70% of move('scale') calls in that cell return
        # nothing (47% never land, 43% of the rest gain <0.01 IoU — see
        # size_hint_aspect_suppressed). Withhold the recommendation and let the no-hint
        # fallback below take over. The SEARCH's relaxed gate is cleared off the SAME
        # predicate in scale_hint, so hint and gate cannot drift. Size-locked objects are
        # excluded: their rung recommends DEPTH, and every outcome behind this threshold came
        # from an axis == "scale" move.
        size_aspect_bad = (
            scale_flag
            and not size_locked
            and size_hint_aspect_suppressed(
                sh.get("scale_est"), sh.get("area_scale"), yaw_aniso
            )
        )
        # Does any line below NAME A TOOL? The two rungs above (rotate_180, rotation) return
        # early and always do, so only the lower rungs need tracking.
        recommended = False

        # --- FACING guard: a verified rotate_180 dominates the whole chain ---
        if flip:
            if allow_rotate_180:
                lines.append(
                    f"- FACING check for {o}: REVERSED (verified) — the 180-degree "
                    "flip matches the photo "
                    f"as well or better (sim 0deg {orient['sim0']:.3f} vs 180deg "
                    f"{orient['sim180']:.3f}) — inspect the crops and consider "
                    f"move({o},'rotate_180') FIRST; any position/size/yaw mismatch is "
                    "re-measured after the flip — a reversed silhouette lands in the "
                    "wrong PLACE, so the position reading in particular is not usable "
                    "yet."
                )
            else:
                # the one-shot is spent: never instruct rotate_180 again — route the
                # correction through the direct-yaw escape hatch instead of leaving
                # the object stranded facing the wrong way.
                lines.append(
                    f"- FACING check for {o}: REVERSED (verified) — the 180-degree "
                    "flip matches the photo "
                    f"as well or better (sim 0deg {orient['sim0']:.3f} vs 180deg "
                    f"{orient['sim180']:.3f}). The rotate_180 one-shot for this object "
                    "is already spent — do NOT call it again. If the crops confirm it "
                    "faces backwards, apply a direct 180-degree yaw about its "
                    f"world-space body center with {direct_pose_tool}, judge the "
                    "returned crop, and keep it or call undo_last_step. Any "
                    "position/size/yaw mismatch is re-measured after the correction."
                )
            return lines
        if flip_deferred:
            lines.append(
                f"- FACING check for {o}: UNKNOWN (ambiguous) — a 180-flip "
                "scored marginally better "
                f"(sim 0deg {orient['sim0']:.3f} vs 180deg {orient['sim180']:.3f}), "
                f"but the verdict is UNRELIABLE here — the silhouette is also yawed "
                f"~{yaw:.0f}deg vs the photo, which corrupts the flip compare on an "
                "elongated object. Do NOT rotate_180 on this signal alone; fix the "
                "yaw first — the flip is re-checked once a rotation lands."
            )
        if verified_same:
            lines.append(
                f"- FACING check for {o}: SAME (verified) — matches the photo "
                f"(sim 0deg "
                f"{orient['sim0']:.3f} vs 180deg {orient['sim180']:.3f}); do NOT "
                f"rotate_180 this object."
            )
        if flip_suspected:
            fix = (
                f"flip it with move({o},'rotate_180')"
                if allow_rotate_180
                else (
                    "apply a direct 180-degree yaw about its world-space body "
                    f"center with {direct_pose_tool} (the rotate_180 one-shot "
                    "for this object is already spent), judge the returned crop, "
                    "and keep it or call undo_last_step"
                )
            )
            lines.append(
                f"- FACING check for {o}: REVERSED (suspected) — the "
                f"180-degree flip scored better (sim 0deg {orient['sim0']:.3f} vs "
                f"180deg {orient['sim180']:.3f}) but only by a weak margin "
                f"(+{flip_margin:.3f}, below the {REVERSED_VERIFIED_MARGIN:.2f} "
                "verification bar). Judge which way the object faces from the "
                "paired crops FIRST; only if they visually confirm a reversal, "
                f"{fix}. Any position/size/yaw hint below is measured at the "
                "CURRENT pose and stands on its own."
            )
            recommended = True

        # --- ONE recommended refinement: yaw > position > scale ---
        deferred = (["position"] if position_fires else []) + (
            [("depth" if size_locked else "size")] if scale_flag else []
        )
        if yaw_fires:
            lines.append(
                f"- YAW HINT for {o}: it appears yawed ~{yaw:.0f}deg vs the "
                f"photo (silhouette axes not aligned) — correct the yaw with "
                f"move({o},'rotation') FIRST"
                + (
                    f"; the {' and '.join(deferred)} mismatch also measured is "
                    "re-checked after (a yawed silhouette distorts those readings)."
                    if deferred
                    else "."
                )
                # a LARGE yaw is still a 'rotation' move (the search seeds the angle
                # directly), but it is the one case where the ladder can fall short —
                # so name the fallback in the same breath rather than leaving a dead
                # end if it does.
                + (
                    " This is a LARGE yaw, past the fine-'rotation' range, but the "
                    "search seeds it directly — try the move first. If it does not "
                    "land, judge the correction from the CROPS and place the object "
                    f"with {direct_pose_tool}; do NOT type the number above as a "
                    "rotation_euler, it is measured in the image plane and the "
                    "equivalent world rotation is a different value."
                    if yaw > _YAW_HINT_MAX
                    else ""
                )
            )
            return lines
        if position_fires:
            # NOTE: the size MAGNITUDE is withheld from _POS_HINT_GAP (0.12) upward here,
            # NOT from _SCALE_DEFER_GAP (0.25) — this site deliberately does not use that
            # constant (see its comment in register.py). The object has not moved yet, so
            # a number would steer to move('scale') when move('xy') is the right next call.
            du, dv = shift
            # DIRECTION ONLY — the ~Npx magnitude was removed 2026-07-26. It was
            # display-only (shift_px has no other consumer; move('xy') re-derives its
            # own offset), and the 0715-0726 composition logs show it doing two kinds
            # of harm. (1) Fabricated conversions: writing world-space code the agent
            # invented a factor with no intrinsics or depth — "~9px is small, roughly
            # 0.02-0.03m in world" then `location.x += 0.025`, ~3x the true shift on a
            # 1500px frame. (2) Wrong triage: px is ABSOLUTE while significance is
            # relative to object size (4px is 1% of a laptop, 13% of a fork), so the
            # agent read gate-flagged objects as negligible — "the pear#0 is only ~18px
            # off, small" — and moved on. `gap` already encodes relative significance
            # (IoU is normalised by the object's own silhouette), and the hint only
            # fires once it clears _POS_HINT_GAP, so every hint shown is significant by
            # construction and needs no severity number. Direction survives: it is a
            # sign, resolution-independent, and the agent maps it correctly to world
            # axes. The 3px floor below is only a noise gate, not a tuned constant.
            dirs = [d for d, v in (("image-right", du), ("image-left", -du),
                                   ("down", dv), ("up", -dv)) if v >= 3]  # fmt: skip
            lines.append(
                f"- POSITION HINT for {o}: the silhouette might fit the photo better "
                "once shifted"
                + (f" toward {' and '.join(dirs)}" if dirs else "")
                + f" — fix placement with move({o},'xy') first."
            )
            recommended = True
            if yaw_deferred:
                lines.append(
                    f"- (yaw for {o}: a ~{yaw:.0f}deg rotation mismatch was ALSO measured "
                    "and is deliberately held back, not missed — the rotation search scores "
                    "on silhouette overlap, which a displacement this large outweighs, so it "
                    "would pick the wrong direction. Fix placement first; the yaw is "
                    "re-measured at the new pose and recommended then.)"
                )
            if scale_flag and size_locked:
                lines.append(
                    "- (a silhouette-size mismatch also measured; the number is "
                    "unreliable until placement is corrected and will be re-measured.) "
                    + self._size_lock_guidance(
                        o,
                        f"Fix placement with move({o},'xy') first, then re-measure.",
                    )
                )
            elif scale_flag:
                lines.append(
                    f"- (size for {o}: a mismatch also measured, but the number is "
                    "unreliable until the object is placed — re-measured in the "
                    "move feedback.)"
                )
        elif scale_flag and size_locked:
            area_s = sh.get("area_scale") or sh.get("scale_est") or 1.0
            pct = abs(1 / area_s - 1) * 100
            small = area_s > 1
            lines.append(
                f"- Its silhouette is ~{pct:.0f}% too "
                f"{'small' if small else 'large'} vs the photo. "
                + self._size_lock_guidance(
                    o,
                    f"For this reading, try move({o},'y') "
                    f"{'nearer' if small else 'farther'} and re-measure.",
                )
            )
            recommended = True
        elif scale_flag and not size_aspect_bad:
            area_s = sh.get("area_scale")  # AREA factor: what a uniform scale achieves
            if area_s is None:  # compat: pre-area hints fall back to extent
                area_s = sh.get("scale_est") or 1.0
            # area_s is the resize FACTOR (target/render); report the deviation
            # relative to the photo target, not |factor-1| (factor 1.5 = 33%
            # smaller, not 50%).
            pct = abs(1 / area_s - 1) * 100
            small = area_s > 1
            # 2D-projected size confounds true scale with DEPTH. The agent judges which
            # from the crops (2026-07-25, replacing the unconditional depth-first
            # steering); depth stays the tiebreak so move('scale') doesn't distort the
            # (trusted) real-scale mesh. The mesh is authoritative: any residual after a
            # correct placement+scale is accepted, never flagged as wrong (the extent
            # metric is logged, never surfaced).
            lines.append(
                f"- SIZE HINT for {o}: ~{pct:.0f}% too {'small' if small else 'large'} "
                f"by 2D-projected silhouette — this confounds true size with depth. "
                f"Judge which from the crops: if it sits too "
                f"{'far' if small else 'close'} (base {'higher' if small else 'lower'} "
                f"in frame than the photo), fix depth with move({o},'y') "
                f"{'nearer' if small else 'farther'}; if placement matches but it still "
                f"looks too {'small' if small else 'large'} next to its neighbours, fix "
                f"size with move({o},'scale'). Unsure? Try depth first (a resize "
                f"distorts the trusted real-scale mesh)."
            )
            recommended = True
        if yaw_oob:
            lines.append(
                f"- (note: {o} looks strongly rotated (~{yaw:.0f}deg) vs the photo, "
                "but its silhouette is too round for that angle to be reliable, so "
                "move('rotation') is unlikely to help. If the crops show a real "
                "yaw error, judge the correction from the crops and place it "
                f"with {direct_pose_tool} — the number above is an image-plane "
                "measurement, not a rotation_euler value.)"
            )
            recommended = True
        # an in-band yaw on a WEAK axis (aniso < _YAW_ANISO_MIN) used to die in
        # total silence — no rotation hint and no yaw_oob note (bridge_6 box#0:
        # 27.9-33.8deg at aniso 1.11-1.19, three visits, shipped ~28deg wrong).
        # The measurement is often still correct for exactly that class (square
        # footprint, visually distinct faces), so surface it as a LEAD handed to
        # the agent's eyes; the search is never seeded from it (reliability is
        # the search's own gate). Mutually exclusive with yaw_oob (yaw_axis_ok).
        yaw_weak = (
            not flip
            and yaw is not None
            and yaw_aniso is not None
            and yaw >= _YAW_HINT_MIN
            and yaw_aniso < _YAW_ANISO_MIN
        )
        if yaw_weak:
            fix = (
                f"fix it with move({o},'rotation') (for a clearly large error use "
                f"{direct_pose_tool} to yaw about the object's world-space body "
                "center)"
                if yaw <= _YAW_HINT_MAX
                else (
                    f"use {direct_pose_tool} to yaw about the object's "
                    "world-space body center (for a smaller visible error "
                    f"move({o},'rotation') also works)"
                )
            )
            lines.append(
                f"- weak-axis YAW reading for {o}: ~{yaw:.0f}deg vs the photo, but "
                "this compact silhouette has no reliable axis (aniso "
                f"{yaw_aniso:.2f}) so the number is a LEAD, not a measurement. "
                "Judge which way the object points from the crops yourself; if a "
                f"yaw error is visible, {fix}, then re-investigate."
            )
            recommended = True
        if not recommended:
            lines.append(
                f"- no measured hint for {o} (IoU {iou:.2f}) — verify from the crops "
                "yourself."
            )
        if emit_orientation_check and not (
            flip or flip_suspected or flip_deferred or verified_same
        ):
            if orient is None and worker_dead():
                # infrastructure failure, not ambiguity: one DINO worker death used
                # to silently downgrade every check this session to UNKNOWN.
                lines.append(
                    f"- FACING check for {o}: UNAVAILABLE — the appearance "
                    "backend is offline for this session (infrastructure failure, "
                    "not ambiguity). Judge facing from the paired crops yourself."
                )
                return lines
            if yaw_oob:
                lines.append(
                    f"- FACING check for {o}: UNKNOWN "
                    "— this applies ONLY to the ~180-degree point-reversal check; follow "
                    "the strong-yaw guidance above for the separate fine-yaw discrepancy."
                )
                return lines
            visual_check = (
                "Complete the POSITION HINT above first. Then compare the object's "
                "directional features in the crops."
                if position_fires
                else "Compare the object's directional features in the crops."
            )
            post_facing_route = (
                "Use rotate_180 only for a clear point reversal."
                if allow_rotate_180
                else (
                    "The rotate_180 one-shot for this object is already spent; do "
                    "NOT call it again — if the crops show a clear point reversal, "
                    "apply a direct 180-degree yaw about the object's world-space "
                    f"body center with {direct_pose_tool} instead, judge the "
                    "returned crop, and keep it or call undo_last_step."
                )
            )
            lines.append(
                f"- FACING check for {o}: UNKNOWN — this applies ONLY to the ~180-degree "
                "point-reversal check; it does not verify that fine yaw is correct. "
                f"{visual_check} If a non-180-degree yaw mismatch remains and you can judge "
                f"the correction, use {direct_pose_tool} to yaw the object about its world-space "
                "body center even when no YAW HINT was available, "
                "then keep the returned result or call undo_last_step. "
                + post_facing_route
            )
        return lines

    @staticmethod
    def _post_flip_orientation_line(o: str, orient: Optional[dict]) -> str:
        """Audit a mandatory post-flip semantic check without reopening the flip.

        Code name kept (``orientation``); the agent-facing surface is the
        "post-flip FACING audit", carrying the SAME qualified token set as the
        investigate-time FACING check — one token set, every surface."""
        from lib.tools.geometry.feature_metric import (
            REVERSED_VERIFIED_MARGIN,
            VERIFIED_SAME_MARGIN,
            worker_dead,
        )

        if orient is None and worker_dead():
            # same distinct state as the investigate-time FACING check (F6d): an
            # offline backend is an infrastructure failure, not ambiguity — do not
            # let it masquerade as UNKNOWN on the one surface where the agent must
            # decide keep-vs-undo.
            return (
                f"- post-flip FACING audit for {o}: UNAVAILABLE — the appearance "
                "backend is offline for this session (infrastructure failure, not "
                "ambiguity). Judge facing from the paired crops; use undo_last_step "
                "only if the retained flip is clearly worse. The rotate_180 "
                "one-shot remains spent; do NOT call rotate_180 again."
            )
        sim0 = orient.get("sim0") if orient else None
        sim180 = orient.get("sim180") if orient else None
        scores = (
            f" (sim 0deg {float(sim0):.3f} vs 180deg {float(sim180):.3f})"
            if sim0 is not None and sim180 is not None
            else ""
        )
        if orient and orient.get("flagged"):
            verified = (
                sim0 is not None
                and sim180 is not None
                and float(sim180) - float(sim0) >= REVERSED_VERIFIED_MARGIN
            )
            verdict = "REVERSED (verified)" if verified else "REVERSED (suspected)"
            route = (
                "Inspect the paired crops and use undo_last_step if the retained flip "
                "is clearly worse."
            )
        elif (
            sim0 is not None
            and sim180 is not None
            and float(sim0) - float(sim180) >= VERIFIED_SAME_MARGIN
        ):
            verdict = "SAME (verified)"
            route = "The current semantic direction appears to match the photo."
        else:
            verdict = "UNKNOWN"
            route = (
                "Inspect the paired crops; use undo_last_step only if the retained "
                "flip is clearly worse."
            )
        return (
            f"- post-flip FACING audit for {o}: {verdict}{scores}. {route} "
            "The rotate_180 one-shot remains spent; do NOT call rotate_180 again."
        )

    def investigate_objects(self, objects: list) -> dict:
        """Comparable RENDER|PHOTO crops around the listed objects + their scores;
        arms them for `move`. Counters: each object <= investigate_cap visits."""
        if self.root_stage_name != "composition":
            return {
                "status": "error",
                "output": {"text": ["investigate_objects is a composition-stage tool"]},
            }
        if not objects or not isinstance(objects, list):
            return {
                "status": "error",
                "output": {"text": ["pass a non-empty list of object ids"]},
            }
        objects = [str(o) for o in objects]
        blocked = self._guard_post_flip_tool_call(
            "investigate_objects", object_ids=objects
        )
        if blocked is not None:
            return blocked
        if self._has_capability("runtime_object_inventory"):
            from lib.tools.geometry.runtime_object_repair import load_runtime_inventory

            overlay = load_runtime_inventory(self.moge_dir)
            excluded = {
                identity
                for row in (overlay or {}).get("objects", [])
                if row.get("mask_policy") == "excluded"
                for identity in (row["id"], row["placement"]["mesh_name"])
            }
            unavailable = sorted(set(objects) & excluded)
            if unavailable:
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            "No mask available for GPT-authored objects (added at "
                            "initialization): "
                            + ", ".join(unavailable)
                            + ". They are excluded from investigation coverage; "
                            "inspect them with render_current_scene instead."
                        ],
                        "no_mask_available": unavailable,
                        "scene_mutation": "not_started",
                    },
                }
        pending = self._pending_post_flip_followup()
        required_ids = set(pending.get("object_ids") or []) if pending else set()
        missing_required = sorted(required_ids.difference(objects))
        if missing_required:
            required = self._post_flip_followup_payload(pending)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "mandatory post-rotate_180 investigation rejected: include "
                        + ", ".join(missing_required)
                        + " in this investigate_objects call. The retained flip's "
                        "measurements remain stale; investigate it now or undo that "
                        "exact flip."
                    ],
                    "required_followup": required,
                },
            }
        session = self._pose_session_get()
        known = session.objects()
        bad = [o for o in objects if o not in known]
        if bad:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"unknown object id(s): {', '.join(bad)} — valid ids: "
                        f"{', '.join(sorted(known))} (root surfaces cannot be "
                        "investigated/moved)"
                    ]
                },
            }
        # The pending flip receives one mandatory cap exception. It is scoped to
        # this exact follow-up/object; other capped objects in a grouped call still
        # reject normally, and a failed crop does not consume the exception.
        cap_exempt = {
            o
            for o in required_ids
            if self._investigated.get(o, 0) >= self.investigate_cap
        }
        over = [
            o
            for o in objects
            if self._investigated.get(o, 0) >= self.investigate_cap
            and o not in cap_exempt
        ]
        if over:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"{', '.join(over)} already investigated "
                        f"{self.investigate_cap} time(s) (the cap) — drop them from "
                        "the list and investigate other objects, or end if all are "
                        "aligned"
                    ]
                },
            }
        # Transactional proposal only: counters/arming and crop state are committed
        # after rendering, crop creation, scoring, and hint generation all finish.
        next_investigated = dict(self._investigated)
        for o in objects:
            if o not in cap_exempt:
                next_investigated[o] = next_investigated.get(o, 0) + 1
        n = self._investigate_n + 1

        import numpy as np
        from PIL import Image

        from lib.tools.geometry.agentic_mask import clamp_crop_aspect

        work = Path(session.work)
        # Photo source decided BEFORE the render.  A de-occluded edit is used only
        # when every removed blocker is CONFIRMED cargo; lateral/uncertain contacts
        # remain visible on both sides via the original contextual photograph.
        photo_src, hide_ids, context_ids = self._composition_crop_visibility(
            session, objects
        )
        rd_objs = [o for o in objects if session.ctx[o].get("redetect")]
        hide_meshes = tuple(session.ctx[rid]["mesh_name"] for rid in hide_ids)
        # move()'s post-move crop re-renders with the SAME hide set, so its feedback
        # matches the investigate baseline the agent is comparing against.
        next_crop_hide = hide_meshes
        render_png = str(work / f"investigate_{n}_render.png")
        session.full_render(render_png, hide=hide_meshes)
        # per-object bboxes: current rendered silhouette (one isolated render each)
        # and GT mask — drawn on both sides below, and unioned for the crop window
        gw, gh = session.gt_w, session.gt_h
        boxes = []
        gt_box: dict = {}
        rd_box: dict = {}
        for o in objects:
            rb = session.rendered_bbox([o])
            if rb:
                rd_box[o] = tuple(int(v) for v in rb)
                boxes.append(rd_box[o])
            # Agent-facing boxes correspond to the ORIGINAL contextual photo.  The
            # amodal/redetected mask remains ``mask_path`` for internal scoring.
            box_mask = (
                session.ctx[o].get("visible_mask_path")
                if photo_src == session.image_path
                else None
            )
            m = np.load(box_mask or session.ctx[o]["mask_path"])
            if m.shape != (gh, gw):
                m = (
                    np.asarray(
                        Image.fromarray((m > 0).astype("uint8") * 255).resize((gw, gh))
                    )
                    > 127
                )
            ys, xs = np.nonzero(m)
            if xs.size:
                gt_box[o] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
                boxes.append(gt_box[o])
        if not boxes:
            return {
                "status": "error",
                "output": {"text": ["objects have no visible extent to crop around"]},
            }
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[2] for b in boxes)
        y1 = max(b[3] for b in boxes)
        # crop = 2x the union bbox (50% context each side), aspect-clamped
        cw, ch = x1 - x0 + 1, y1 - y0 + 1
        cx0 = int(x0 - cw / 2)
        cy0 = int(y0 - ch / 2)
        cx1, cy1 = cx0 + 2 * cw, cy0 + 2 * ch
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(gw, cx1), min(gh, cy1)
        cx0, cy0, cx1, cy1 = clamp_crop_aspect(cx0, cy0, cx1, cy1, gw, gh)
        # move() attaches a post-move render crop of this window (cheap feedback)
        next_crop_window = (cx0, cy0, cx1, cy1)
        rimg = Image.open(render_png).convert("RGB").resize((gw, gh))
        # photo_src / rd_objs / hide_ids were decided above, BEFORE the render.
        photo = Image.open(photo_src).convert("RGB").resize((gw, gh))
        # mark each listed object with the SAME colored bbox (+id) on BOTH sides —
        # render box from its isolated silhouette, photo box from its GT mask — so
        # the render<->photo correspondence is explicit and the pixels stay
        # untinted (a tint hid appearance cues like which way a fork points).
        from PIL import ImageDraw

        palette = [
            (255, 40, 40), (255, 160, 0), (60, 120, 255),
            (200, 40, 200), (0, 180, 180), (160, 220, 0),
        ]  # fmt: skip
        draw_r, draw_p = ImageDraw.Draw(rimg), ImageDraw.Draw(photo)
        pad = 4
        for i, o in enumerate(objects):
            col = (255, 40, 40) if len(objects) == 1 else palette[i % len(palette)]
            for draw, b in ((draw_p, gt_box.get(o)), (draw_r, rd_box.get(o))):
                if b:
                    draw.rectangle(
                        [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad],
                        outline=col, width=3,
                    )  # fmt: skip
                    draw.text((b[0] - pad, max(0, b[1] - pad - 12)), o, fill=col)
        rc = rimg.crop((cx0, cy0, cx1, cy1))
        pc = photo.crop((cx0, cy0, cx1, cy1))
        # the agent gets TWO SEPARATE images (cleaner for the VLM than one
        # concatenation, which also tripped the lone-image feedback wording);
        # the side-by-side composite is still written for the demo panel.
        rc_png = str(work / f"investigate_{n}_scene.png")
        pc_png = str(work / f"investigate_{n}_photo.png")
        rc.save(rc_png)
        pc.save(pc_png)
        div = 6
        comp = Image.new(
            "RGB",
            (rc.width + pc.width + div, max(rc.height, pc.height)),
            (255, 255, 255),
        )
        comp.paste(rc, (0, 0))
        comp.paste(pc, (rc.width + div, 0))
        comp_png = str(work / f"investigate_{n}.png")
        comp.save(comp_png)
        # index 0 pairs with the render crop, index 1 with the photo crop (see the
        # generator's image packaging). The static "how to read the two crops"
        # explanation lives ONCE in the composition system prompt; only the DYNAMIC
        # de-occlusion note (scene-specific) stays here on the photo-crop line.
        deocc_note = (
            " NOTE: IMAGE 2 is the DE-OCCLUDED edited photo, and IMAGE 1 hides only "
            f"confirmed cargo ({', '.join(hide_ids) if hide_ids else 'none'}) — "
            "both sides expose the underlying object for amodal inspection."
            if photo_src != session.image_path
            else (
                " NOTE: the ORIGINAL contextual photo is used and no imported "
                "object is hidden because the removal set contains a lateral or "
                f"unconfirmed neighbour ({', '.join(context_ids)}). Judge their "
                "contact, spacing, and visible occlusion together."
                if context_ids
                else (
                    f" NOTE: {', '.join(rd_objs)} have an internal de-occluded mask, "
                    "but this batch uses the ORIGINAL contextual photo; all imported "
                    "objects remain visible."
                    if rd_objs
                    else ""
                )
            )
        )
        lines = [
            f"Investigated {', '.join(objects)} (visit "
            + ", ".join(
                (
                    f"{o}: mandatory post-flip verification; normal visits remain "
                    f"{self._investigated.get(o, 0)}/{self.investigate_cap}"
                    if o in cap_exempt
                    else f"{o}: {next_investigated[o]}/{self.investigate_cap}"
                )
                for o in objects
            )
            + ").",
            "IMAGE 2 = reference photo crop." + deocc_note,
        ]
        ious: dict = {}
        for o in objects:
            sc = session.score(o)
            ious[o] = sc["iou"]
            lines.append(  # score BEFORE IoU — uniform-frontend order (_INV_IOU)
                f"- {o}: "
                + (f"score {sc['score']:.3f}, " if "score" in sc else "")
                + f"silhouette IoU {sc['iou']:.2f}"
            )
        # Per-object hints, RECONCILED by the precedence chain
        #   rotate_180 -> rotation(yaw) -> position -> scale, except position outranks
        #   rotation while displaced (gap >= _POS_HINT_GAP, unless flip is deferred).
        # Each transform corrupts the measurement the next one depends on (a
        # reversed/yawed silhouette is the wrong shape, so scale/position readings
        # are meaningless), so the backend recommends only the TOP firing hint as
        # the next move and NAMES the rest as deferred (re-measured after the
        # upstream fix). See _object_hint_lines. Best-effort per object.
        orient_fn = getattr(session, "orientation_hint", None)
        scale_fn = getattr(session, "scale_hint", None)
        flipped = getattr(self, "_flipped", set())
        from lib.tools.geometry.feature_metric import worker_dead

        for o in objects:
            mandatory_post_flip = pending is not None and o in required_ids
            orient = None
            # Always attempted, spent one-shot or not — a consumed flip used to
            # freeze the check at UNKNOWN forever; the per-pose freeze + landed-move
            # invalidation in register.py already prevent recompute thrash.
            if orient_fn is not None:
                try:
                    orient = orient_fn(o, prefix=f"inv{n}")
                except Exception:  # noqa: BLE001 - hints are advisory
                    orient = None
            if (
                orient is None
                and worker_dead()
                and not getattr(self, "_orient_backend_dead_logged", False)
            ):
                # once per session: the hint lines say UNAVAILABLE per object, the
                # log pins WHEN the backend died for the artifact trail.
                self._orient_backend_dead_logged = True
                import sys as _sys

                print(
                    "[orientation-hint] appearance backend offline for this "
                    "session — orientation checks are UNAVAILABLE (infrastructure "
                    "failure, not ambiguity)",
                    file=_sys.stderr,
                )
            sh = None
            if scale_fn is not None:
                try:
                    sh = scale_fn(o, prefix=f"inv{n}")
                except Exception:  # noqa: BLE001 - hints are advisory
                    sh = None
            if mandatory_post_flip:
                # Recompute semantic direction for the audit, but never feed a
                # flagged result into the normal precedence ladder: that would
                # recommend an impossible second flip and suppress current-pose
                # position/fine-yaw/size measurements.
                lines.append(self._post_flip_orientation_line(o, orient))
                lines.extend(
                    self._object_hint_lines(
                        o,
                        None,
                        sh,
                        ious.get(o, 1.0),
                        allow_rotate_180=False,
                        emit_orientation_check=False,
                    )
                )
            else:
                lines.extend(
                    self._object_hint_lines(
                        o,
                        orient,
                        sh,
                        ious.get(o, 1.0),
                        # a spent one-shot routes any REVERSED/UNKNOWN advice to the
                        # execute_and_evaluate direct-yaw variant, never rotate_180
                        allow_rotate_180=o not in flipped,
                    )
                )
        if self._has_capability("composition_direct_pose_edit"):
            lines.append(
                "These objects are ARMED for move: use move(object, aspect) first, it "
                "searches the image against the target. Use edit_object_poses only for "
                "an attitude correction move cannot express, or after move failed on "
                "that object (two committed typed edits per object at most)."
            )
        elif self._has_capability("strict_post_edit_physics"):
            # 2026-09-15 owner: the agent chooses the tool, but is reminded of both
            # routes after every investigate so the code route is not forgotten.
            lines.append(
                "Both pose routes are open to you now: execute_and_evaluate applies a "
                "correction you can state yourself (world-space translation / yaw, one or "
                "several of these objects in one call, physics-settled before the render); "
                "move(object, aspect) measures ONE aspect against the photo. These objects "
                "are ARMED for move until your next investigate_objects call."
            )
        else:
            lines.append(
                "These objects are now ARMED: you may call move(object, aspect) on any "
                "of them (repeatedly) until your next investigate_objects call."
            )
        # Commit after every fallible operation above. In particular, an exception
        # from full_render/Image/scoring leaves visits, arming, and the pending flip
        # exactly as they were before this call.
        self._investigated = next_investigated
        self._replaced_uninvestigated = getattr(
            self, "_replaced_uninvestigated", set()
        ) - set(objects)
        self._armed = set(objects)
        self._investigate_n = n
        self._last_crop_hide = next_crop_hide
        self._last_crop_window = next_crop_window
        resolved = None
        if pending is not None:
            resolved = self._resolve_post_flip_followup(
                token=str(pending["token"]), status="satisfied"
            )
            lines.append(
                "Mandatory post-rotate_180 investigation satisfied for "
                + ", ".join(pending["object_ids"])
                + "; its current-pose crops and measurements now replace the stale "
                "pre-flip evidence."
            )
        output: dict = {"text": lines, "image": [rc_png, pc_png]}
        if resolved is not None:
            output["resolved_followup"] = resolved
        return {
            "status": "success",
            "output": output,
        }

    def _settled_object_crop(self, names: list) -> Optional[list]:
        """TWO cheap crops windowed on the object(s) a freeform edit RELOCATED — the
        post-``execute_and_evaluate`` visual check (mirrors investigate_objects' two-crop
        return): IMAGE 1 the EDITED render, IMAGE 2 the REFERENCE target photo at the
        SAME window, so the agent compares its relocation against the target. Builds the
        window from each object's rendered silhouette bbox unioned with its GT-mask bbox
        (same 2x-context logic as investigate), records it as ``_last_crop_window`` (so a
        follow-up move() reuses it), renders+crops the edited scene via
        ``_move_region_crop``, then crops the reference photo (the shared de-occluded edit
        when the moved batch shares one, else the source photo — mirrors investigate).
        Each moved object is boxed (same colour + id) on BOTH crops -- RENDER box from its
        settled silhouette, PHOTO box from its GT mask. Sets ``_settle_photo_deocc`` for
        the caller's caption. Best-effort: None when no
        object has a visible extent (caller then falls back to the normal render)."""
        import numpy as np
        from PIL import Image, ImageDraw

        from lib.tools.geometry.agentic_mask import clamp_crop_aspect

        try:
            session = self._pose_session_get()
            # settle_edited reports MESH names (obj_croissant_2); ctx is keyed by obj_id
            # (croissant#2). Accept either (mirrors register.py's mesh2oid mapping).
            mesh2oid = {v["mesh_name"]: k for k, v in session.ctx.items()}
            gw, gh = session.gt_w, session.gt_h
            oids = [
                oid
                for name in names
                if (oid := (name if name in session.ctx else mesh2oid.get(name)))
                is not None
            ]
            # Decide the photo before loading target boxes: contextual/original
            # images need modal boxes, while de-occluded images use amodal boxes.
            photo_src, hide_ids, _context_ids = self._composition_crop_visibility(
                session, oids
            )
            rd_box, gt_box = {}, {}  # render (settled pose) + photo (GT mask) boxes
            for oid in oids:
                rb = session.rendered_bbox([oid])
                if rb:
                    rd_box[oid] = tuple(int(v) for v in rb)
                box_mask = (
                    session.ctx[oid].get("visible_mask_path")
                    if photo_src == session.image_path
                    else None
                )
                m = np.load(box_mask or session.ctx[oid]["mask_path"])
                if m.shape != (gh, gw):
                    m = (
                        np.asarray(
                            Image.fromarray((m > 0).astype("uint8") * 255).resize(
                                (gw, gh)
                            )
                        )
                        > 127
                    )
                ys, xs = np.nonzero(m)
                if xs.size:
                    gt_box[oid] = (
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    )
            boxes = list(rd_box.values()) + list(gt_box.values())
            if not boxes:
                return None
            x0 = min(b[0] for b in boxes)
            y0 = min(b[1] for b in boxes)
            x1 = max(b[2] for b in boxes)
            y1 = max(b[3] for b in boxes)
            cw, ch = (
                x1 - x0 + 1,
                y1 - y0 + 1,
            )  # 2x the union bbox (50% context each side)
            cx0, cy0 = max(0, int(x0 - cw / 2)), max(0, int(y0 - ch / 2))
            cx1, cy1 = (
                min(gw, int(x0 - cw / 2) + 2 * cw),
                min(gh, int(y0 - ch / 2) + 2 * ch),
            )
            self._last_crop_window = clamp_crop_aspect(cx0, cy0, cx1, cy1, gw, gh)
            self._settle_photo_deocc = photo_src != session.image_path
            self._last_crop_hide = tuple(
                session.ctx[rid]["mesh_name"] for rid in hide_ids
            )
            edited = self._move_region_crop(session)  # IMAGE 1: [edited_png]
            if not edited:
                return None
            photo_png = str(Path(session.work) / "settle_check_photo.png")
            (
                Image.open(photo_src)
                .convert("RGB")
                .resize((gw, gh))
                .crop(self._last_crop_window)
                .save(photo_png)
            )
            # box the moved object(s) on BOTH crops (window-relative), same palette/style
            # as investigate: RENDER box from the settled silhouette, PHOTO box from the
            # GT mask -- makes the where-it-sits vs where-the-target-shows-it explicit.
            palette = [
                (255, 40, 40), (255, 160, 0), (60, 120, 255),
                (200, 40, 200), (0, 180, 180), (160, 220, 0),
            ]  # fmt: skip
            wx0, wy0 = self._last_crop_window[0], self._last_crop_window[1]
            pad = 4
            ei = Image.open(edited[0]).convert("RGB")
            pi = Image.open(photo_png).convert("RGB")
            de, dp = ImageDraw.Draw(ei), ImageDraw.Draw(pi)
            for i, oid in enumerate(oids):
                col = (255, 40, 40) if len(oids) == 1 else palette[i % len(palette)]
                for draw, b in ((de, rd_box.get(oid)), (dp, gt_box.get(oid))):
                    if not b:
                        continue
                    bx0, by0, bx1, by1 = b[0] - wx0, b[1] - wy0, b[2] - wx0, b[3] - wy0
                    draw.rectangle(
                        [bx0 - pad, by0 - pad, bx1 + pad, by1 + pad],
                        outline=col,
                        width=3,
                    )
                    draw.text((bx0 - pad, max(0, by0 - pad - 12)), oid, fill=col)
            ei.save(edited[0])
            pi.save(photo_png)
            return edited + [photo_png]
        except Exception:  # noqa: BLE001 - feedback is best-effort
            return None

    def _move_region_crop(self, session) -> Optional[list]:
        """One cheap EEVEE render cropped to the LAST INVESTIGATE's window — the
        post-move visual check that replaced the full Cycles render+reference pair
        (minutes per scene). Best-effort: None when no investigate happened yet."""
        win = getattr(self, "_last_crop_window", None)
        if not win:
            return None
        try:
            from PIL import Image

            full = str(Path(session.work) / "move_check_full.png")
            session.full_render(full, hide=getattr(self, "_last_crop_hide", ()))
            crop_png = str(Path(session.work) / "move_check.png")
            Image.open(full).convert("RGB").crop(win).save(crop_png)
            return [crop_png]
        except Exception:  # noqa: BLE001 - feedback is best-effort
            return None

    def move_object(self, object: str, aspect: str) -> dict:  # noqa: A002
        """One backend-optimized correction of `object` along `aspect`."""
        blocked = self._guard_post_flip_tool_call("move")
        if blocked is not None:
            return blocked
        if self.root_stage_name != "composition":
            return {
                "status": "error",
                "output": {"text": ["move is a composition-stage tool"]},
            }
        direct_pose_tool = self._direct_pose_tool_phrase()
        typed_direct_pose = self._has_capability("composition_direct_pose_edit")
        blocked_batch_route = (
            f"both together with {direct_pose_tool}"
            if typed_direct_pose
            else "both together in one execute_and_evaluate edit"
        )
        explicit_layout_route = (
            f"use {direct_pose_tool} to place"
            if typed_direct_pose
            else "use one execute_and_evaluate code edit to place"
        )
        aspect_map = {
            "xy": "xy",
            "x": "x-axis",
            "y": "y-axis",
            "rotation": "rotation",
            "rotate_180": "rotate_180",
            "scale": "scale",
        }
        if aspect not in aspect_map:
            return {
                "status": "error",
                "output": {"text": [f"aspect must be one of {sorted(aspect_map)}"]},
            }
        if aspect == "scale" and object in self._size_locked_ids():
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"move rejected: '{object}' has a LOCKED size. "
                        + self._size_lock_guidance(
                            object,
                            f"Re-check placement and try move('{object}','xy') or "
                            f"move('{object}','y'), then record any unresolved mismatch.",
                        )
                    ]
                },  # fmt: skip
            }
        if self._pose_session is None or object not in self._armed:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"move rejected: '{object}' is not armed — a move must "
                        "follow an investigate_objects call that includes it "
                        f"(currently armed: {', '.join(sorted(self._armed)) or 'none'})."
                        " Investigate it first."
                    ]
                },
            }
        session = self._pose_session_get()
        self._ensure_undo_base()  # the first edit of the stage may be a move — capture
        # the pre-edit base BEFORE it, so undo_last_step can restore to before it too.
        if object not in session.objects():
            return {
                "status": "error",
                "output": {"text": [f"unknown object id '{object}'"]},
            }
        if self._moved.get(object, 0) >= self.move_cap:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"move rejected: '{object}' has used its full move budget "
                        f"({self.move_cap}/{self.move_cap}). Leave it as it is — "
                        "further correction must come from other objects or a "
                        "layout edit, not more moves on this one."
                    ]
                },  # fmt: skip
            }
        if aspect == "rotate_180":
            # one free retry after a CLEAN physics rejection: the set below records
            # objects that already used theirs, so a physically-impossible flip
            # cannot ping-pong (move_cap bounds the calls either way).
            rejected_once = getattr(self, "_flip_phys_rejected", None)
            if rejected_once is None:
                rejected_once = self._flip_phys_rejected = set()
            if object in self._flipped:
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"rotate_180 already used on {object} — it is ONE-SHOT "
                            "per object (consumed by an applied flip, kept or "
                            "undone, and by a second physics-rejected attempt); "
                            "use 'rotation' for fine adjustment instead."
                        ]
                    },  # fmt: skip
                }
            restore_from = (
                self.edit_history[-1] if self.edit_history else self.base_state
            )
            if not restore_from or not os.path.isfile(restore_from):
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"rotate_180 on {object} could not start: no durable "
                            "pre-flip scene snapshot is available. Nothing changed."
                        ]
                    },  # fmt: skip
                }
            history_len = len(self.edit_history)
            ledger_history_len = len(getattr(self, "_ledger_history", []))
            edit_meta_len = len(getattr(self, "_edit_meta", []))
            strict_physics = self._has_capability("strict_post_edit_physics")
            moved_before = self._moved.get(object, 0)
            self._flipped.add(object)
            self._moved[object] = moved_before + 1
            try:
                res = session.flip_180(object)
            except Exception as exc:  # noqa: BLE001 - strict path restores durably
                if not strict_physics:
                    raise
                return self._strict_flip_infrastructure_failure(
                    session,
                    object_id=object,
                    failure=str(exc),
                    restore_from=restore_from,
                    history_len=history_len,
                    ledger_history_len=ledger_history_len,
                    edit_meta_len=edit_meta_len,
                    moved_before=moved_before,
                )
            required_followup = None
            refunded = phys_rejected = False
            ph = res.get("physics") or {}
            if strict_physics and ph.get("infrastructure_error"):
                return self._strict_flip_infrastructure_failure(
                    session,
                    object_id=object,
                    failure=str(
                        ph.get("reason")
                        or res.get("reason")
                        or "physics authority returned an infrastructure failure"
                    ),
                    restore_from=restore_from,
                    history_len=history_len,
                    ledger_history_len=ledger_history_len,
                    edit_meta_len=edit_meta_len,
                    moved_before=moved_before,
                    physics=ph,
                )
            if res.get("applied"):
                try:
                    edit_token = self._new_edit_token("flip")
                except Exception as exc:  # noqa: BLE001
                    if not strict_physics:
                        raise
                    return self._strict_flip_infrastructure_failure(
                        session,
                        object_id=object,
                        failure=f"edit-token creation failed: {exc}",
                        restore_from=restore_from,
                        history_len=history_len,
                        ledger_history_len=ledger_history_len,
                        edit_meta_len=edit_meta_len,
                        moved_before=moved_before,
                        physics=ph or None,
                    )
                try:
                    session.save()
                    self._push_edit_snapshot(
                        "flip", object_id=object, edit_token=edit_token
                    )  # record so a later undo can't wipe it
                    meta = self._edit_meta[-1] if self._edit_meta else None
                    if (
                        len(self.edit_history) != history_len + 1
                        or len(self._ledger_history) != ledger_history_len + 1
                        or len(self._edit_meta) != edit_meta_len + 1
                        or not os.path.isfile(self.edit_history[-1])
                        or not meta
                        or meta.get("kind") != "flip"
                        or meta.get("object_id") != object
                        or str(meta.get("edit_token")) != edit_token
                    ):
                        raise RuntimeError(
                            "durable flip snapshot/edit metadata was not recorded"
                        )
                except Exception as exc:  # noqa: BLE001 - transaction must roll back
                    if strict_physics:
                        return self._strict_flip_infrastructure_failure(
                            session,
                            object_id=object,
                            failure=f"durable snapshot failed: {exc}",
                            restore_from=restore_from,
                            history_len=history_len,
                            ledger_history_len=ledger_history_len,
                            edit_meta_len=edit_meta_len,
                            moved_before=moved_before,
                            physics=ph or None,
                        )
                    restored, restore_note = self._restore_failed_flip_transaction(
                        session,
                        restore_from=restore_from,
                        history_len=history_len,
                        ledger_history_len=ledger_history_len,
                        edit_meta_len=edit_meta_len,
                    )
                    return {
                        "status": "error",
                        "output": {
                            "text": [
                                f"rotate_180 on {object} was rolled back because its "
                                f"durable snapshot failed ({exc}). "
                                + (
                                    "The pre-flip scene/session was restored."
                                    + restore_note
                                    if restored
                                    else "CRITICAL: " + restore_note
                                )
                                + " The one-shot and move budget remain consumed; no "
                                "post-flip investigation is pending."
                            ],
                            # explicit, not the wrapper's setdefault: a FAILED
                            # restore leaves the live blend possibly still flipped
                            # — "unknown" makes the generator fail closed instead
                            # of trusting a pre-flip ALL RULES PASS.
                            "scene_mutation": (
                                "not_committed" if restored else "unknown"
                            ),
                        },  # fmt: skip
                    }
                if strict_physics:
                    try:
                        required_followup = self._require_post_flip_investigation(
                            object, edit_token
                        )
                    except Exception as exc:  # noqa: BLE001
                        return self._strict_flip_infrastructure_failure(
                            session,
                            object_id=object,
                            failure=f"post-flip follow-up creation failed: {exc}",
                            restore_from=restore_from,
                            history_len=history_len,
                            ledger_history_len=ledger_history_len,
                            edit_meta_len=edit_meta_len,
                            moved_before=moved_before,
                            physics=ph or None,
                        )
                else:
                    required_followup = self._require_post_flip_investigation(
                        object, edit_token
                    )
                # Every pre-flip measurement/authorization is stale. The only
                # permitted paths are a fresh investigate (which re-arms) or exact
                # undo; do not let direct Executor callers chain another move.
                self._armed = set()
            else:
                # flip_180 has already restored its in-session pre-flip pose.
                # Persist that unchanged pose, but never create a follow-up.
                try:
                    session.save()
                except Exception as exc:  # noqa: BLE001
                    if not strict_physics:
                        raise
                    return self._strict_flip_infrastructure_failure(
                        session,
                        object_id=object,
                        failure=f"rejected-pose persistence failed: {exc}",
                        restore_from=restore_from,
                        history_len=history_len,
                        ledger_history_len=ledger_history_len,
                        edit_meta_len=edit_meta_len,
                        moved_before=moved_before,
                        physics=ph or None,
                    )
                # A CLEAN physics rejection (tilt/capsize/penetration gate, scene
                # auto-reverted and re-persisted above) says nothing about which
                # way the object FACES, so it refunds the consumed-above one-shot
                # ONCE per object; the second rejection consumes it for good.
                # Every committed/ambiguous path keeps the 08-20 fail-closed
                # consume-before handshake — a missing physics verdict never
                # refunds.
                phys_rejected = ph.get("accepted") is False and not ph.get(
                    "infrastructure_error", False
                )
                refunded = phys_rejected and object not in rejected_once
                if refunded:
                    rejected_once.add(object)
                    self._flipped.discard(object)
            b, a = res["before"], res["after"]
            ph = res.get("physics")
            if res["applied"]:
                # deliberately NO IoU numbers: 0714_hr_abc1's agent flipped the
                # spoon CORRECTLY, saw "0.26 -> 0.20", talked itself out of its
                # own right judgment and tried to undo. The score is uninformative
                # here by construction.
                # The anti-undo steering states the real ASYMMETRY (undo restores
                # the scene but never refunds the one-shot) instead of the old
                # "there is no undo", which was simply false — the flip IS on the
                # undo stack (_push_edit_snapshot above), so an agent that saw a
                # genuinely bad flip was told it could not revert something it
                # could (HARNESS_AUDIT_2026_07_26 T4).
                txt = (
                    f"yaw-rotated {object} 180 deg about its center. Do NOT judge "
                    "this by IoU: a near-symmetric silhouette often scores the "
                    "same or LOWER after a CORRECT flip — a drop is NOT evidence "
                    "the flip was wrong. The one-shot is spent either way: "
                    "undo_last_step would roll the SCENE back but NOT refund it, "
                    "so undoing on a low IoU alone leaves this object stuck facing "
                    "the wrong way with no flip left. REQUIRED NEXT CALL: call "
                    f"investigate_objects including {object} for fresh paired crops "
                    "and current-pose measurements. The only alternative is "
                    "undo_last_step for this exact flip; do not call another tool "
                    "first."
                )
                if ph:
                    txt += (
                        f" Physics-settled (tilt {ph['tilt_deg']} deg); anything "
                        "stacked on it stayed in place."
                    )
            else:
                reason_txt = self._oids_in(
                    res.get("reason") or "the object cannot rest after the 180-yaw"
                )
                if refunded:
                    txt = (
                        f"rotate_180 on {object} was REJECTED by physics: "
                        + reason_txt
                        + " — everything reverted; the scene is unchanged. The "
                        "one-shot was NOT consumed: a physics rejection says "
                        "nothing about which way the object faces, so ONE retry "
                        "remains (a second physics rejection consumes it; this "
                        "call still spent move budget). Change what made the flip "
                        "unstable — placement or support — before retrying."
                    )
                else:
                    txt = (
                        f"rotate_180 on {object} was REJECTED: "
                        + reason_txt
                        + " — everything reverted. The one-shot budget is consumed"
                        + (
                            " — this was the free retry; a flip physics rejected "
                            "twice cannot stand here."
                            if phys_rejected
                            else "."
                        )
                    )
                txt += self._post_move_pending_note(session, object, "rotate_180")
            # No per-move budget sentence (owner 2026-07-31): the OBJECT STATE table
            # already carries moves-used per object; the table now COUNTS move rounds
            # itself (build_object_state_table), so the sentence was pure repetition.
            fout: dict = {"text": [txt]}
            if required_followup is not None:
                fout["required_followup"] = required_followup
            fimg = self._move_region_crop(session)
            if fimg:
                fout["image"] = fimg
                fout["text"][0] += (
                    " (Attached: a post-move render CROP of the "
                    "last-investigated region, reference camera.)"
                )
            return {"status": "success", "output": fout}
        self._moved[object] = self._moved.get(object, 0) + 1
        res = session.optimize_axis(object, aspect_map[aspect])
        session.save()  # persist to the shared blend so scene renders see it
        if res.get("applied"):
            self._push_edit_snapshot("move")  # one undo stack across move + execute
        b, a = res["before"], res["after"]
        ph = res.get("physics")
        if res["applied"]:
            txt = (  # score BEFORE IoU — uniform-frontend order (_MOVE_DELTA)
                f"moved {object} along {aspect}: score {b['score']:.3f} -> "
                f"{a['score']:.3f}, IoU {b['iou']:.2f} -> {a['iou']:.2f}."
            )
            if aspect == "rotation":
                # raw IoU is yaw-blind (translation-variant), so on a yaw change it
                # must not be read as the verdict — same framing as the flip text.
                # P1/P3 (misc_online5 round 7): the backend stashes the yaw pair its
                # improvement gate measured and the appearance-agreement pair into
                # the move result; surface them HERE, beside the misleading raw
                # numbers. The next-investigate promise stays ONLY when the yaw
                # pair is absent — promising a re-measure that is already printed
                # right here reads as "the number you see is not it".
                pair_txt, yaw_pair_shown = self._rotation_pair_notes(res)
                txt += (
                    " (Raw IoU is not the authority on a yaw change — judge the "
                    "post-move crop"
                    + (
                        " and the yaw re-measure.)"
                        if yaw_pair_shown
                        else "; the yaw is re-measured at your next investigate.)"
                    )
                    + pair_txt
                )
            elif aspect == "scale" and a["iou"] < b["iou"]:
                # the scale backend selects/accepts on a displacement-tolerant
                # metric (register.optimize_axis), so a CORRECT resize on a
                # displaced object can land while the reported raw pair DECLINES
                # — same class as the rotation case, and undo_last_step works on
                # moves. Conditional on the dip: the caveat is not spammed on
                # ordinary scale lines (the spec forbade that).
                txt += (
                    " (Raw IoU is not the authority on a resize — a correct "
                    "resize on a displaced object can lower it; judge the size "
                    "re-measure and the post-move crop before any undo.)"
                )
            # Productive-repeat nudges (owner 07-29: after a LARGE gain the agent never
            # re-tried the same aspect — placemat#0 stopped at 0.59 with budget left —
            # while the failure paths' "do NOT retry" bled into a blanket norm). Only
            # emitted when the BACKEND knows a repeat can reach further, and only with
            # budget remaining.
            if self._moved.get(object, 0) < self.move_cap:
                if res.get("clamped_frac") is not None:
                    txt += (
                        f" Only ~{res['clamped_frac'] * 100:.0f}% of the computed move was "
                        "applied before a neighbor blocked it — move that neighbor (or "
                        f"{blocked_batch_route}), then repeat "
                        "THIS same aspect to recover the rest."
                    )
                elif res.get("range_limited"):
                    txt += (
                        " The winning step was at this call's search-range limit — the "
                        "optimum may lie further. Repeating this SAME aspect searches "
                        "onward from the new pose and is often worth a follow-up call."
                    )
            if ph and ph.get("accepted"):
                # members/topple come from the physics layer as mesh names; the fix for
                # anything named here is another move() call, so emit scene-graph ids
                carried = [self._oid(m) for m in ph["members"][1:]]
                if ph.get("followed") is False:
                    cd = ph.get("carry_decision") or {}
                    txt += (
                        f" Physics-settled (cum tilt {ph['cum_tilt_deg']} deg). "
                        "Its children STAYED IN PLACE — their current positions "
                        "match the photo better than riding along"
                        + (
                            f" (mean IoU staying {cd['avg_iou_stay']:.2f} vs "
                            f"following {cd['avg_iou_follow']:.2f})."
                            if cd.get("avg_iou_stay") is not None
                            and cd.get("avg_iou_follow") is not None
                            else "."
                        )
                    )
                elif ph.get("followed") and (ph.get("carry_decision") or {}).get(
                    "topple"
                ):
                    txt += (
                        f" Physics-settled (cum tilt {ph['cum_tilt_deg']} deg; "
                        f"carried along: {', '.join(carried)} — leaving them "
                        "would topple "
                        + ", ".join(
                            self._oid(t) for t in ph["carry_decision"]["topple"]
                        )
                        + ")."
                    )
                else:
                    txt += (
                        f" Physics-settled (cum tilt {ph['cum_tilt_deg']} deg"
                        + (f"; carried along: {', '.join(carried)}" if carried else "")
                        + ")."
                    )
            if ph and ph.get("stability_clamped_frac"):
                txt += (
                    f" NOTE: settled at ~{ph['stability_clamped_frac'] * 100:.0f}% "
                    "of the move — the full move tips over; this is as far as it "
                    "can stand."
                )
            off = a.get("resolve_offset") or [0, 0, 0]
            if any(abs(v) > 1e-4 for v in off):
                txt += (
                    f" Contact resolved: pushed out of its support by "
                    f"({off[0]:+.3f}, {off[1]:+.3f}, {off[2]:+.3f})m."
                )
            # post-move size re-measure: the scale reading becomes trustworthy
            # once the object is placed, and delivering it HERE saves the full
            # investigate (render + hints + an LLM round) that abc2's spoon spent
            # just to learn its corrected deficit.
            txt += self._post_move_size_note(session, object, aspect)
            # G2 (misc_online5 stapler): the same fresh reading carries yaw —
            # surface it instead of leaving it silent in the session cache.
            txt += self._post_move_yaw_note(session, object, aspect)
        elif ph and not ph.get("accepted"):
            if aspect == "rotation":
                txt = (
                    f"{object} rotation: the best fine-yaw candidate improved the render "
                    f"match but physics REJECTED it ({self._oids_in(str(ph.get('reason')))}) "
                    "and everything was reverted. Do NOT retry move('rotation') from the "
                    "unchanged pose. If position already matches and the crops still show a "
                    f"smaller, non-180-degree yaw error, use {direct_pose_tool} to yaw the "
                    "object about its world-space body center, then keep the returned result "
                    "or call undo_last_step. Otherwise keep the current pose."
                )
            else:
                txt = (
                    f"{object} {aspect}: the best candidate improved the render match "
                    f"but physics REJECTED it ({self._oids_in(str(ph.get('reason')))}) "
                    "and everything was reverted. "
                    "The backend already tried the best available magnitude "
                    "— do NOT retry this same aspect. If you can SEE a collision-free final "
                    f"layout from the crops, use {direct_pose_tool} to place the involved "
                    "objects at independently chosen, visually justified XY positions with "
                    "enough lateral clearance; their relative positions may change. Physics "
                    "then resolves vertical clearance and the resting pose, but it does NOT "
                    "plan collision-free XY positions. Otherwise try a different aspect or "
                    "keep the current pose."
                )
        elif res.get("yaw_regressed"):
            # The rotation search FOUND gain and was vetoed because the re-measured yaw
            # failed the improvement demand (see the yaw gate in optimize_axis; since
            # phase C1 it gates seeded AND unseeded rotations). Never report this as "no
            # candidate improved" — the agent would reasonably retry, and the search would
            # pick the same wrong candidate again. Name the real obstacle instead: the
            # current pose (usually a displacement) corrupts the axis reading.
            pre, post = res["yaw_regressed"]
            txt = (
                f"{object} {aspect}: the search found a better silhouette match but it "
                f"FAILED THE YAW CHECK — the measured yaw went {pre:.0f}deg -> {post:.0f}deg "
                "(the gate demands a real improvement), so the move was refused and nothing "
                "changed. This happens when the current pose corrupts the axis reading — "
                "most often the object is also out of POSITION, which even the "
                "displacement-tolerant candidate scoring cannot fully factor out. Do NOT "
                "retry 'rotation' from this pose. Fix placement first with "
                f"move({object},'xy'), then re-investigate — the yaw is re-measured at the "
                "new pose. If placement already matches and the crops still show a real "
                f"smaller, non-180-degree yaw error, use {direct_pose_tool} to yaw it about "
                "its world-space body center, then keep the returned result or call "
                "undo_last_step."
            )
        elif res.get("clamp_reason"):
            # T3b: the feasibility clamp's verdict used to be swallowed — the
            # agent got "looks already right" for a move that was BLOCKED by a
            # neighbor, the one case where the right next action is knowable.
            if aspect == "rotation":
                txt = (
                    f"{object} rotation: the search found a fine-yaw improvement but the "
                    f"move was rejected pre-physics: {res['clamp_reason']}. Do NOT retry "
                    "move('rotation') while the scene is unchanged. If position already "
                    "matches and the crops still show a smaller, non-180-degree yaw error, "
                    f"use {direct_pose_tool} to yaw the object about its world-space body "
                    "center, then keep the returned result or call undo_last_step. Otherwise "
                    "keep the current pose."
                )
            else:
                txt = (
                    f"{object} {aspect}: the search found an improvement but the "
                    f"move was rejected pre-physics: {res['clamp_reason']}. "
                    "Do NOT retry this same aspect while the scene is unchanged. "
                    "If you can SEE a collision-free final layout, "
                    f"{explicit_layout_route} the involved objects at "
                    "independently chosen, visually justified XY positions with enough "
                    "lateral clearance; their relative positions may change. Do not repeat "
                    "the same single-object translation; otherwise keep the current pose."
                )
        else:
            # With a pending size mismatch, "looks already right" is affirmatively
            # misleading (it told cluttershelf's agent pomegranate#1 was fine while a
            # 25% deficit sat deferred) — soften to a per-aspect statement.
            pending = bool(self._post_move_pending_note(session, object, aspect))
            if aspect == "rotation":
                # G4 wording: "no candidate improved" understated reality when a
                # candidate had gain below the bar — say the bar was not cleared.
                txt = (
                    f"{object} rotation: the best fine-yaw candidate did not clear the "
                    f"acceptance bar (IoU stays {b['iou']:.2f}) — this rotation search is "
                    "dead at the current pose. If position already matches and the crops "
                    "still show a smaller, non-180-degree yaw error, use "
                    f"{direct_pose_tool} to yaw the object about its world-space body "
                    "center, then keep the returned result or call undo_last_step. "
                    "Otherwise keep the current pose."
                )
            else:
                txt = (
                    f"{object} {aspect}: no candidate improved the match "
                    f"(IoU stays {b['iou']:.2f}) — "
                    + (
                        "this aspect found nothing further."
                        if pending
                        else "this aspect looks already right; try another aspect or "
                        "another object."
                    )
                )
        if not res.get("applied"):
            txt += self._post_move_pending_note(session, object, aspect)
        # No per-move budget sentence (owner 2026-07-31) — see the rotate_180 branch.
        out: dict = {"text": [txt]}
        img = self._move_region_crop(session)
        if img:
            out["image"] = img
            out["text"][0] += (
                " (Attached: a post-move render CROP of the last-investigated "
                "region, reference camera — re-investigate for a fresh "
                "side-by-side against the photo.)"
            )
        else:
            out["text"][0] += (
                " (No auto-render after moves — call investigate_objects or "
                "render_current_scene when you want visual confirmation.)"
            )
        return {"status": "success", "output": out}

    def _post_move_pending_note(self, session, obj: str, failed_aspect: str) -> str:
        """After a NON-applied move: re-surface the deferred SIZE mismatch from the
        session cache (owner-approved 2026-07-31). The precedence chain defers size
        behind position with "re-measured in the move feedback" — but that promise
        was only redeemed by an ACCEPTED move, so a rejected/no-gain attempt stranded
        the deferral (fleet 07-30: 15 of 60 deferrals never got a size-relevant move;
        0730_fix_cluttershelf pomegranate#1 shipped 25% undersized after ONE no-gain
        'xy'). The cache is current by construction here: a failed move reverted the
        scene, an accepted rotation cleared the cache (_invalidate_hints), an
        accepted placement re-measured it (_post_move_size_note). Wording carries NO
        'HINT' substring: the whole move text collapses to a ledger line later, and
        _trim drops any line containing HINT — that would erase the move outcome
        itself. The OBJECT STATE table parses "SIZE PENDING for" from this note."""
        if failed_aspect == "scale":
            return ""  # no self-loop: the failed aspect must not be the remedy
        if self._moved.get(obj, 0) >= self.move_cap:
            return ""  # budget gone — the cap refusal already says "leave it"
        s = (getattr(session, "_scale_hint", None) or {}).get(obj)
        if s is None:
            return ""
        pct = abs(1 / s - 1) * 100  # deviation vs target (same as _post_move_size_note)
        small = s > 1
        if obj in self._size_locked_ids():
            return (
                f" SIZE PENDING for {obj}: it also measured ~{pct:.0f}% too "
                f"{'small' if small else 'large'} (2D projected) at this SAME pose — "
                "the failed move changed nothing, so that reading is still current. "
                + self._size_lock_guidance(
                    obj,
                    f"For this reading, try move({obj},'y') "
                    f"({'nearer' if small else 'farther'}) and re-measure.",
                )
            )
        return (
            f" SIZE PENDING for {obj}: it also measured ~{pct:.0f}% too "
            f"{'small' if small else 'large'} (2D projected) at this SAME pose — "
            "the failed move changed nothing, so that reading is still current. If "
            "its 3D size looks wrong next to neighbours, try "
            f"move({obj},'scale'); if it looks right, it is a DEPTH error — "
            f"move({obj},'y') ({'nearer' if small else 'farther'})."
        )

    def _post_move_size_note(self, session, obj: str, aspect: str) -> str:
        """C2: after an accepted placement/scale move, re-run the scale check and
        report the now-trustworthy reading inline. Empty string when the aspect
        isn't size-relevant, the object is still displaced (gap high — the number
        would mislead again), or anything fails (advisory only)."""
        if aspect not in ("xy", "x", "y", "scale"):
            return ""
        # freshness handshake for _post_move_yaw_note (called right after this):
        # True only when the scale_hint re-measure below actually RAN — its sh
        # dict is what refreshes session._yaw_hint at the new pose. On failure
        # the cached yaw ANGLE is still current (these aspects don't change yaw)
        # but the yaw note must not claim a re-measure that did not happen.
        self._post_move_hint_fresh = False
        scale_fn = getattr(session, "scale_hint", None)
        if scale_fn is None:
            return ""
        try:
            from lib.tools.geometry.register import _POS_HINT_GAP, _SCALE_DEFER_GAP

            sh = scale_fn(obj, prefix="move")
            if not sh:
                return ""
            self._post_move_hint_fresh = True
            gap = sh["overlap"] - session.score(obj)["iou"]
            if gap >= _SCALE_DEFER_GAP:
                return (
                    " Placement still looks off (the silhouette might fit the photo "
                    "better once shifted) — size not re-measured; "
                    "consider another 'xy'."
                )
            s = sh.get("area_scale")  # AREA factor, consistent with the hint's firing
            if s is None:  # compat: pre-area hints fall back to extent
                s = sh.get("scale_est")
            if s is None:
                return ""
            if sh.get("flagged"):
                pct = abs(1 / s - 1) * 100  # deviation vs target, not |factor-1|
                small = s > 1
                if obj in self._size_locked_ids():
                    return (
                        f" Size re-measured at the new pose: still ~{pct:.0f}% too "
                        f"{'small' if small else 'large'} (2D projected). "
                        + self._size_lock_guidance(
                            obj,
                            f"For this reading, try move({obj},'y') "
                            f"({'nearer' if small else 'farther'}) and re-measure.",
                        )
                    )
                # placed but still mismatched: more likely a true-size error now,
                # but the reading is still 2D — a residual depth error reads the
                # same, so keep depth as the cheaper first try before distorting.
                return (
                    f" Size re-measured at the new pose: still ~{pct:.0f}% too "
                    f"{'small' if small else 'large'} (2D projected). If its 3D "
                    "size looks right next to neighbours it is a DEPTH error — try "
                    f"move({obj},'y') ({'nearer' if small else 'farther'}); else "
                    f"move({obj},'scale')."
                    + (
                        " (Placement may still be slightly off — re-check after "
                        "another 'xy'.)"
                        if gap >= _POS_HINT_GAP
                        else ""
                    )
                )
            return " Size re-measured at the new pose: matches the photo mask."
        except Exception:  # noqa: BLE001 - feedback is best-effort
            return ""

    def _post_move_yaw_note(self, session, obj: str, aspect: str) -> str:
        """G2 (misc_online5 stapler): after an accepted placement/size move, surface
        the yaw reading the size re-measure just cached — scale_hint's sh dict
        carries yaw_deg/yaw_aniso and writes them into session._yaw_hint, the SAME
        channel that seeds the rotation search, so this costs no extra render.
        Only for the aspects _post_move_size_note re-measured: an applied rotation
        deliberately does not re-measure (its own yaw gate already did, and the
        landed move cleared the cache via _invalidate_hints), and translations/
        resizes don't change the yaw, so a cached reading is current here by
        construction. Gated on yaw_hint_fires (in-band angle on a reliable axis) —
        the one predicate behind hint and seed — so the seeding promise below is
        true by construction. When the size note's re-measure FAILED
        (_post_move_hint_fresh False) the reading is the pre-move cache: still
        current, but worded so it never claims a re-measure ran. Best-effort."""
        if aspect not in ("xy", "x", "y", "scale"):
            return ""
        try:
            from lib.tools.geometry.register import yaw_hint_fires

            yaw, aniso = (getattr(session, "_yaw_hint", None) or {}).get(
                obj, (None, None)
            )
            if not yaw_hint_fires(yaw, aniso):
                return ""
            if getattr(self, "_post_move_hint_fresh", False):
                return (
                    f" Yaw re-measured at the new pose: ~{yaw:.0f}deg remaining "
                    f"(axis reliable) — another move({obj},'rotation') will be "
                    "hint-seeded."
                )
            # the fresh re-measure failed, so this is the CACHED reading — still
            # numerically current (these aspects don't change yaw, the docstring
            # invariant) but not a re-measure; say so honestly.
            return (
                f" Yaw reading (still current at this pose): ~{yaw:.0f}deg "
                f"remaining (axis reliable) — another move({obj},'rotation') "
                "will be hint-seeded."
            )
        except Exception:  # noqa: BLE001 - feedback is best-effort
            return ""

    def _rotation_pair_notes(self, res: dict) -> tuple[str, bool]:
        """P3 + P1 evidence pairs for an APPLIED move('rotation'), read off the
        move result dict (register.optimize_axis stashes what it already computed
        — no extra renders): yaw_pre/yaw_post from the improvement gate with their
        yaw_pre_aniso/yaw_post_aniso, and app_pre/app_post appearance-agreement
        scores at the keep pose and the rested winner. The yaw pair emits ONLY
        when both readings are reliable (aniso >= _YAW_ANISO_MIN on both sides) —
        a weak-axis reading aliases (bridge_6) and is omitted SILENTLY (P2). The
        appearance pair emits whenever both values exist (worker-dead/crop failure
        arrives as None -> omitted). misc_online5 round 7: a CORRECT second
        -42.5deg printed only a falling raw pair (0.535->0.471 / 0.57->0.51) with
        no positive evidence beside it. Defensive .get(): a backend without the
        fields emits nothing. Returns (text, yaw_pair_shown)."""
        from lib.tools.geometry.register import _YAW_ANISO_MIN

        txt = ""
        yaw_pair = False
        try:
            yp, yq = res.get("yaw_pre"), res.get("yaw_post")
            if (
                yp is not None
                and yq is not None
                and min(
                    res.get("yaw_pre_aniso") or 0.0,
                    res.get("yaw_post_aniso") or 0.0,
                )
                >= _YAW_ANISO_MIN
            ):
                txt += f" Measured yaw: ~{yp:.0f}deg -> ~{yq:.0f}deg vs the photo."
                yaw_pair = True
            ap, aq = res.get("app_pre"), res.get("app_post")
            if ap is not None and aq is not None:
                txt += (
                    f" Appearance agreement vs photo: {ap:.2f} -> {aq:.2f} "
                    f"({_appearance_pair_qualifier(ap, aq)})."
                )
        except Exception:  # noqa: BLE001 - feedback is best-effort
            return "", False
        return txt, yaw_pair

    def _rule_investigation_coverage(self) -> tuple[bool, str]:
        """Composition end-gate: initial coverage plus no stale retained flip."""
        eligible = self.pose_eligible_ids()
        evidence_error = getattr(self, "_pose_eligibility_error", None)
        if evidence_error:
            return (
                False,
                "investigation coverage: UNVERIFIED — " + str(evidence_error),
            )
        scope = getattr(self, "coverage_scope", None)
        if scope is not None:  # final-settle repair round
            eligible = [o for o in eligible if o in scope]
        todo = [o for o in eligible if self._investigated.get(o, 0) == 0]
        replaced = sorted(
            o for o in eligible if o in getattr(self, "_replaced_uninvestigated", set())
        )
        pending = self._pending_post_flip_followup()
        if not todo and not replaced and pending is None:
            return True, (
                "investigation coverage: PASS (every listed object investigated)"
                if scope is not None
                else "investigation coverage: PASS (every object investigated)"
            )
        failures = []
        if todo:
            failures.append(
                "not yet investigated: "
                + ", ".join(sorted(todo))
                + " (every object must appear in at least one investigate_objects call)"
            )
        if replaced:
            failures.append(
                "replaced since your last investigate: "
                + ", ".join(replaced)
                + " (an object whose mesh you replaced with edit_object_mesh is a new "
                "reconstruction; investigate it once more before you end)"
            )
        if pending is not None:
            failures.append(
                "retained rotate_180 has stale measurements for "
                + ", ".join(pending["object_ids"])
                + f" (follow-up token {pending['token']}); call investigate_objects "
                "including that object, or undo that exact flip"
            )
        return (
            False,
            "investigation coverage: FAIL — " + "; ".join(failures),
        )

    def _support_body_map(self) -> dict:
        """Scene-graph support edges as penetration-dump body names:
        ``obj_<slug>`` -> its support's body name (``obj_<slug>`` for a parent object,
        the surface build-name for a root surface). Best-effort ({} on any failure)."""
        try:
            from lib.tools.geometry.agentic_mask import slugify
            from lib.tools.geometry.surface_relations import surface_build_name

            sgj = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            nodes = {n["id"]: n for n in sgj.get("nodes", [])}
            out = {}
            for nid, n in nodes.items():
                if n.get("kind") == "root_surface" or not n.get("support"):
                    continue
                sup = n["support"]
                sup_node = nodes.get(sup)
                sup_body = (
                    surface_build_name(sup)
                    if sup_node is None or sup_node.get("kind") == "root_surface"
                    else "obj_" + slugify(sup.replace("#", "_"))
                )
                out["obj_" + slugify(nid.replace("#", "_"))] = sup_body
            return out
        except Exception:  # noqa: BLE001 - resting rule is best-effort
            return {}

    def _rule_objects_inboard(
        self, data: dict, main_name: Optional[str]
    ) -> tuple[bool, str]:
        """POSE sub-rule: every object covered by its exact declared direct support.

        One contract for both harnesses since 2026-09-16 (owner): the scene-graph
        support edge decides which body an object is checked against; a missing or
        unbuilt support fails closed with the exact binding to restore.
        """
        return objects_on_direct_supports_report(
            data.get("bodies", []),
            self._support_body_map(),
            support_hulls=data.get("surface_top_hulls") or data.get("surface_hulls"),
            object_hulls=data.get("object_hulls"),
        )

    def _typed_settle_intents(
        self, resting_modes: dict[str, str]
    ) -> dict[str, dict[str, str]]:
        """Bind typed-tool resting modes to their exact declared support bodies.

        Support resolution is deliberately best-effort, matching
        :meth:`_support_body_map`. A missing graph edge keeps the established
        resting-mode behavior through a mode-only intent instead of inventing a
        support constraint.
        """
        support_bodies = self._support_body_map()
        intents: dict[str, dict[str, str]] = {}
        for raw_mesh_name, raw_mode in resting_modes.items():
            mesh_name = str(raw_mesh_name)
            intent = {"mode": str(raw_mode)}
            required_support = support_bodies.get(mesh_name)
            if isinstance(required_support, str) and required_support:
                intent["required_support"] = required_support
            intents[mesh_name] = intent
        return intents

    def _read_penetration_data(
        self,
        *,
        blend_path: Optional[str] = None,
        tag: str = "object_state",
        _typed_world_semantics: bool = False,
    ) -> dict:
        """Run the read-only geometry dump against ``blend_path`` or the live scene.

        Opening an explicit immutable path is the important distinction from the old
        lazy freeze snapshot, which merely disabled saving while still opening the
        mutable live blend and consequently blessed attempt-1 edits in attempt 2.
        """
        if not isinstance(_typed_world_semantics, bool):
            raise RuntimeError("invalid typed world-semantics snapshot request")
        if _typed_world_semantics:
            match = re.fullmatch(
                r"composition_mesh_([1-9][0-9]*)_(before|after)", str(tag)
            )
            if (
                match is None
                or not self._composition_mesh_recovery_enabled()
                or (match.group(2) == "before") != bool(blend_path)
            ):
                raise RuntimeError(
                    "typed world-semantics snapshot is not an exact composition "
                    "mesh pre/post query"
                )
        tmp_dir = self.render_path.parent / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.count += 1
        out_path = tmp_dir / f"{self.count}_{tag}.json"
        code_file = self.script_path / f"{self.count}_{tag}.py"
        target = os.path.abspath(blend_path) if blend_path else self.blender_file
        # 2026-09-15 owner: ONE remembered dump, reused when the blend bytes it described
        # are exactly the bytes about to be read (0.02 s sha256). The dump is a pure
        # function of those bytes plus the two env values the script reads. Every
        # transaction ran three of these 15-27 s passes and the after-dump of tx N is
        # the before-dump of tx N+1 unless an undo / rollback / bake changed the file —
        # which the hash catches (blind reuse would have fed the change detector a
        # rejected candidate: 10 rollbacks and 25 undos in today's 19 initializers).
        key = None
        if not _typed_world_semantics:
            key = (
                _file_sha256(target),
                json.dumps(getattr(self, "_pipeline_object_names", [])),
                str(self.root_stage_name or ""),
            )
            cached = getattr(self, "_last_geometry_dump", None)
            if cached and cached[0] == key:
                with open(code_file, "w") as stream:
                    stream.write(
                        f"# reused geometry dump {cached[1]} (blend sha256 "
                        f"{key[0][:12]} unchanged); no Blender run\n"
                    )
                with open(out_path, "w") as stream:
                    json.dump(cached[2], stream)
                print(
                    f"[dump] {tag}: reused {cached[1]} (blend {key[0][:12]})",
                    file=sys.stderr,
                    flush=True,
                )
                return copy.deepcopy(cached[2])
        with open(code_file, "w") as stream:
            # transaction dumps read pairs / integrity / bodies / surface digests only
            stream.write(generate_penetration_script(str(out_path), sections="transaction"))
        if out_path.exists():
            out_path.unlink()
        old_file, old_save = self.blender_file, self.blender_save
        if blend_path:
            self.blender_file = os.path.abspath(blend_path)
        self.blender_save = None
        started = time.monotonic()
        try:
            execute_kwargs = (
                {"_typed_world_semantics": True} if _typed_world_semantics else {}
            )
            success, _, stdout, stderr = self._execute_blender(
                str(code_file), **execute_kwargs
            )
        finally:
            self.blender_file, self.blender_save = old_file, old_save
        if not success or not out_path.exists():
            raise RuntimeError(
                f"{tag} geometry query failed: "
                + _proc_text(stderr, stdout or "no output artifact was produced")
            )
        try:
            with open(out_path) as stream:
                data = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"{tag} geometry query returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("bodies"), list):
            raise RuntimeError(f"{tag} geometry query returned an invalid schema")
        # remember it only if the (read-only) run really left the file untouched
        if key is not None and _file_sha256(target) == key[0]:
            self._last_geometry_dump = (key, f"{self.count}_{tag}", copy.deepcopy(data))
        print(
            f"[dump] {tag}: fresh {time.monotonic() - started:.1f}s"
            + (f" (blend {key[0][:12]})" if key else ""),
            file=sys.stderr,
            flush=True,
        )
        return data

    def _objects_baseline_snapshot(self) -> dict:
        if self._object_baseline is None:
            baseline = getattr(self, "initializer_baseline_blend", None)
            if not baseline or not os.path.exists(baseline):
                raise RuntimeError(
                    "immutable initializer-entry baseline is unavailable"
                )
            self._object_baseline = self._read_penetration_data(
                blend_path=baseline, tag="initializer_entry_objects"
            )
        return self._object_baseline

    @staticmethod
    def _matrix_error(a, b) -> float:
        try:
            return max(
                abs(float(a[i][j]) - float(b[i][j])) for i in range(4) for j in range(4)
            )
        except (TypeError, ValueError, IndexError):
            return float("inf")

    def _authorized_object_matrices(self) -> dict[str, list]:
        out: dict[str, list] = {}
        for tx in (getattr(self, "_initializer_ledger", {}) or {}).get(
            "active_transactions", []
        ):
            for name, record in (tx.get("objects") or {}).items():
                if record.get("after_matrix") is not None:
                    out[name] = record["after_matrix"]
        return out

    def _authorized_object_signatures(self) -> dict[str, dict]:
        """Latest complete signature authorized by a typed GPT-6 transaction."""
        if not self._has_capability("runtime_object_inventory"):
            return {}
        out: dict[str, dict] = {}
        for tx in (getattr(self, "_initializer_ledger", {}) or {}).get(
            "active_transactions", []
        ):
            for name, record in (tx.get("objects") or {}).items():
                signature = record.get("after_signature")
                if isinstance(signature, dict):
                    out[name] = signature
        return out

    def _rule_object_integrity(
        self, data: dict, pose_tol: float = 2e-5
    ) -> tuple[bool, str]:
        """Preserve imported-object content while allowing ledger-authorized poses."""
        ref = self._objects_baseline_snapshot()
        ref_sig = ref.get("object_integrity", {}) or {}
        cur_sig = data.get("object_integrity", {}) or {}
        committed_empty = (
            getattr(self, "_pipeline_object_names", None) == []
            and self._has_committed_empty_object_inventory()
        )
        expected_names = (
            set()
            if committed_empty
            else set(getattr(self, "_pipeline_object_names", []) or ref_sig)
        )
        if not expected_names and not committed_empty:
            expected_names = {
                b["name"] for b in ref.get("bodies", []) if b.get("is_obj")
            }
        missing = sorted(expected_names - set(cur_sig))
        extra = sorted(set(cur_sig) - expected_names)
        altered, unauthorized, stale_authorization = [], [], []
        authorized = self._authorized_object_matrices()
        authorized_signatures = self._authorized_object_signatures()

        ref_box = {
            b["name"]: (b["lo"], b["hi"])
            for b in ref.get("bodies", [])
            if b.get("is_obj")
        }
        cur_box = {
            b["name"]: (b["lo"], b["hi"])
            for b in data.get("bodies", [])
            if b.get("is_obj")
        }
        for name in sorted(expected_names & set(cur_sig)):
            r = authorized_signatures.get(name, ref_sig.get(name, {}))
            c = cur_sig[name]
            immutable_keys = (
                "integrity_sha256",
                "scale",
                "parent",
                "hide_render",
                "hide_viewport",
            )
            if any(r.get(k) != c.get(k) for k in immutable_keys):
                altered.append(name)
                continue
            want = authorized.get(name, r.get("matrix"))
            if (
                want is not None
                and self._matrix_error(c.get("matrix"), want) > pose_tol
            ):
                if name in authorized:
                    stale_authorization.append(name)
                else:
                    unauthorized.append(name)
            elif want is None and name in ref_box and name in cur_box:
                # Compatibility fallback for archived dumps without matrix records.
                lo0, hi0 = ref_box[name]
                lo1, hi1 = cur_box[name]
                drift = max(
                    max(abs(x - y) for x, y in zip(lo0, lo1)),
                    max(abs(x - y) for x, y in zip(hi0, hi1)),
                )
                if drift > 0.002 and name not in authorized:
                    unauthorized.append(name)

        if not (missing or extra or altered or unauthorized or stale_authorization):
            n = len(authorized)
            if not self._has_capability("runtime_object_inventory"):
                return True, (
                    "object integrity: PASS (identity/mesh/scale/rotation/material/"
                    f"hierarchy/visibility preserved; {n} authorized pose"
                    + ("s" if n != 1 else "")
                    + ")"
                )
            return True, (
                "object integrity: PASS (identity and backend-authorized mesh/pose/"
                f"material/hierarchy/visibility state preserved; {n} authorized pose"
                + ("s" if n != 1 else "")
                + ")"
            )
        parts = []
        if missing:
            parts.append("missing/deleted/renamed: " + ", ".join(missing))
        if extra:
            parts.append("unexpected imported identities: " + ", ".join(extra))
        if altered:
            parts.append(
                "mesh/material/hierarchy/visibility/scale changed: "
                + ", ".join(altered)
            )
        if unauthorized:
            parts.append("unauthorized pose: " + ", ".join(unauthorized))
        if stale_authorization:
            parts.append(
                "changed after its authorized nudge: " + ", ".join(stale_authorization)
            )
        policy = (
            "use only the typed object mutation tools; ordinary Blender code "
            "cannot mutate imported objects"
            if self._has_capability("runtime_object_inventory")
            else "imported objects may be translated only through nudge_object"
        )
        return False, (
            "object integrity: FAIL — "
            + policy
            + "; undo the raw edit or restore the object exactly: "
            + "; ".join(parts)
        )

    def _initializer_nudge_budget(
        self, target: str, *, physical_repair: bool = False
    ) -> tuple[bool, str]:
        active = (getattr(self, "_initializer_ledger", {}) or {}).get(
            "active_transactions", []
        )
        active = [
            tx
            for tx in active
            if tx.get("kind") in {None, "nudge"}
            and bool(tx.get("physical_repair")) == physical_repair
        ]
        on_target = [tx for tx in active if tx.get("target") == target]
        targets = {tx.get("target") for tx in active}
        if len(on_target) >= 2:
            return False, f"'{target}' already has its full 2 active nudge corrections"
        if physical_repair:
            if len(active) >= PHYSICAL_REPAIR_MAX_TRANSACTIONS:
                return (
                    False,
                    "the initializer has exhausted its physical-repair allowance",
                )
            return True, ""
        if target not in targets and len(targets) >= 2:
            return False, "two distinct objects/stacks already have active corrections"
        if len(active) >= 4:
            return False, "the initializer has its full 4 active nudge corrections"
        return True, ""

    def _initializer_ledger_after_undo(
        self,
        prior: dict,
        *,
        undone_transaction_id: Optional[int] = None,
        undone_kind: Optional[str] = None,
    ) -> dict:
        """Build the post-undo ledger without mutating memory or disk.

        Keeping construction separate lets undo stage the scene and persist this
        ledger before committing either history stack.
        """
        current = copy.deepcopy(
            getattr(self, "_initializer_ledger", self._empty_initializer_ledger())
        )
        restored = copy.deepcopy(prior)
        # Audit events are append-only even though active authorizations/counters roll
        # back with the scene. A rejected transaction uses the same event stream but
        # never enters edit_history, so it cannot corrupt undo state.
        restored["events"] = list(current.get("events", []))
        if undone_transaction_id is not None:
            restored["events"].append(
                {
                    "id": int(undone_transaction_id),
                    "status": "undone",
                    **({"kind": undone_kind} if undone_kind else {}),
                    "attempt_idx": int(getattr(self, "attempt_idx", 1)),
                }
            )
        restored["next_transaction_id"] = max(
            int(current.get("next_transaction_id") or 1),
            int(restored.get("next_transaction_id") or 1),
        )
        return restored

    def _resolve_initializer_object(self, value: str) -> tuple[Optional[str], str]:
        names = set(getattr(self, "_pipeline_object_names", []) or [])
        if value in names:
            return value, ""
        inv = {oid: name for name, oid in self._name2id().items()}
        name = inv.get(value)
        if name in names:
            return name, ""
        return None, (
            f"unknown imported object '{value}'; use a scene-graph id or exact obj_* "
            "mesh name from get_scene_info"
        )

    def _initializer_nudge_members(self, target: str) -> list[str]:
        """Target plus object descendants it supports, in parent-first order."""
        name2id = self._name2id()
        target_id = name2id.get(target)
        if not target_id:
            raise RuntimeError(
                f"cannot resolve {target}'s scene-graph identity; refusing to move a "
                "possibly loaded support without its dependents"
            )
        if not self.moge_dir:
            raise RuntimeError(
                "scene-graph artifacts are unavailable; support-stack membership "
                "cannot be validated"
            )
        try:
            graph_path = os.path.join(self.moge_dir, "scene_graph.json")
            with open(graph_path) as graph_file:
                graph = json.load(graph_file)
            node_rows = graph.get("nodes")
            if not isinstance(node_rows, list):
                raise ValueError("nodes is not a list")
            nodes = {
                n.get("id"): n for n in node_rows if isinstance(n, dict) and n.get("id")
            }
            if target_id not in nodes:
                raise ValueError(f"target id {target_id!r} is absent")
            id2name = {oid: name for name, oid in name2id.items()}
            out, queue = [target], [target_id]
            seen = {target_id}
            while queue:
                parent = queue.pop(0)
                for oid, node in nodes.items():
                    if oid in seen or node.get("kind") == "root_surface":
                        continue
                    if (node.get("parent") or node.get("support")) != parent:
                        continue
                    seen.add(oid)
                    queue.append(oid)
                    mesh = id2name.get(oid)
                    if (
                        mesh in getattr(self, "_pipeline_object_names", [])
                        and mesh not in out
                    ):
                        out.append(mesh)
            return out
        except Exception as exc:  # noqa: BLE001 - ambiguous carrying must fail closed
            raise RuntimeError(
                f"could not validate support-stack membership for {target}: {exc}"
            ) from exc

    @staticmethod
    def _body_map(data: dict) -> dict[str, dict]:
        return {b["name"]: b for b in data.get("bodies", [])}

    @staticmethod
    def _xy_overlap_fraction(obj: dict, support: dict) -> float:
        ox = max(
            0.0,
            min(float(obj["hi"][0]), float(support["hi"][0]))
            - max(float(obj["lo"][0]), float(support["lo"][0])),
        )
        oy = max(
            0.0,
            min(float(obj["hi"][1]), float(support["hi"][1]))
            - max(float(obj["lo"][1]), float(support["lo"][1])),
        )
        area = max(
            1e-12,
            (float(obj["hi"][0]) - float(obj["lo"][0]))
            * (float(obj["hi"][1]) - float(obj["lo"][1])),
        )
        return ox * oy / area

    def _object_resting_state(self, data: dict, names: set[str]) -> dict[str, dict]:
        boxes = self._body_map(data)
        supports = self._support_body_map()
        out = {}
        for name in names:
            obj = boxes.get(name)
            if obj is None:
                continue
            candidates: list[tuple[str, float, float]] = []
            for other_name, other in boxes.items():
                if other_name == name:
                    continue
                frac = self._xy_overlap_fraction(obj, other)
                if frac <= 1e-6:
                    continue
                top = float(other["hi"][2])
                if top <= float(obj["lo"][2]) + 0.015:
                    candidates.append((other_name, top, frac))
            declared = supports.get(name)
            support = boxes.get(declared) if declared else None
            declared_frac = self._xy_overlap_fraction(obj, support) if support else None
            if support is not None and (declared_frac or 0.0) > 1e-6:
                candidates.append((declared, float(support["hi"][2]), declared_frac))
            if candidates:
                below, top, frac = max(candidates, key=lambda row: row[1])
                gap = float(obj["lo"][2]) - top
            else:
                below, gap, frac = None, None, 0.0
            out[name] = {
                "declared_support": declared,
                "declared_overlap": declared_frac,
                "below": below,
                "gap": gap,
                "below_overlap": frac,
            }
        return out

    @staticmethod
    def _member_penetrations(data: dict, members: set[str]) -> dict[frozenset, float]:
        return {
            frozenset((p["a"], p["b"])): float(p.get("depth", 0.0))
            for p in data.get("penetrating_pairs", [])
            if p.get("a") in members or p.get("b") in members
        }

    def _nudge_safety_violations(
        self, pre: dict, post: dict, target: str, members: list[str], reason: str
    ) -> list[str]:
        member_set = set(members)
        problems = []
        pre_pen = self._member_penetrations(pre, member_set)
        post_pen = self._member_penetrations(post, member_set)
        support = self._support_body_map()
        dirs = {
            frozenset((p["a"], p["b"])): p.get("separation_direction")
            for src in (pre, post)
            for p in src.get("penetrating_pairs", [])
        }

        def _cls(pair):
            a, b = sorted(pair)
            return contact_class(a, b, support, dirs.get(pair))

        # A pair only counts as worsened when it grows AND ends beyond its class
        # allowance: two siblings settling into a 1 mm lateral kiss is not a defect.
        worsened = [
            (pair, depth, pre_pen.get(pair, 0.0))
            for pair, depth in post_pen.items()
            if depth > pre_pen.get(pair, 0.0) + CONTACT_TOLERANCE_M
            and excess_m(depth, _cls(pair)) > 0.0
        ]
        if worsened:
            problems.append(
                "new/worsened penetration: "
                + ", ".join(
                    f"{' <-> '.join(sorted(pair))} {old:.3f}m -> {depth:.3f}m"
                    for pair, depth, old in worsened
                )
            )
        if reason == "penetration":
            # CLUSTER objective: total depth BEYOND each pair's allowance over every
            # pair touching the target. A 3-body fix that clears one pair while
            # nudging another within its allowance is progress, not a violation.
            pre_target = sum(
                excess_m(d, _cls(pair)) for pair, d in pre_pen.items() if target in pair
            )
            post_target = sum(
                excess_m(d, _cls(pair))
                for pair, d in post_pen.items()
                if target in pair
            )
            if pre_target <= 0.0:
                problems.append(
                    f"no measured penetration involving {target} authorizes this reason"
                )
            elif not penetration_improved(pre_target, post_target, tol=0.0):
                problems.append(
                    f"over-allowance penetration involving {target} did not materially "
                    f"improve ({pre_target:.3f}m -> {post_target:.3f}m)"
                )

        pre_rest = self._object_resting_state(pre, member_set)
        post_rest = self._object_resting_state(post, member_set)
        for name, state in post_rest.items():
            before = pre_rest.get(name, {})
            gap = state.get("gap")
            old_gap = before.get("gap")
            old_declared_overlap = before.get("declared_overlap")
            declared = before.get("declared_support")
            # Preserve the same physical support, not merely "something below".
            # Without a scene-graph support body, freeze the measured pre-transaction
            # below-body identity and overlap instead of allowing a slide onto a floor or
            # unrelated neighbour to masquerade as a valid rest.
            if (
                declared
                and old_declared_overlap is not None
                and old_declared_overlap > 1e-6
            ):
                if state.get("declared_support") != declared:
                    problems.append(f"{name} changed declared support from {declared}")
                if state.get("declared_overlap") is None:
                    problems.append(f"{name} lost declared support {declared}")
            else:
                old_below = before.get("below")
                if old_below is None:
                    problems.append(
                        f"{name} had no declared or measurable pre-transaction support; "
                        "support preservation cannot be validated"
                    )
                elif state.get("below") != old_below:
                    problems.append(
                        f"{name} changed measured support from {old_below} "
                        f"to {state.get('below') or 'none'}"
                    )
                elif float(state.get("below_overlap") or 0.0) + 0.05 < float(
                    before.get("below_overlap") or 0.0
                ):
                    problems.append(
                        f"{name} materially reduced overlap with measured support {old_below}"
                    )
            if gap is None:
                problems.append(f"{name} lost all measurable support beneath it")
            elif gap > 0.015 and (old_gap is None or gap > old_gap + 0.002):
                problems.append(f"{name} now floats {gap * 100:.1f}cm")
            old_overlap = old_declared_overlap
            overlap = state.get("declared_overlap")
            if old_overlap is not None and overlap is not None:
                if old_overlap >= 0.25 and overlap < 0.25:
                    problems.append(
                        f"{name} lost declared-support footprint "
                        f"({old_overlap:.0%} -> {overlap:.0%})"
                    )
                elif overlap + 0.05 < old_overlap:
                    problems.append(
                        f"{name} materially reduced declared-support overlap "
                        f"({old_overlap:.0%} -> {overlap:.0%})"
                    )
        if reason == "resting_defect":
            before = pre_rest.get(target, {}).get("gap")
            after = post_rest.get(target, {}).get("gap")
            if before is None or before <= 0.015:
                problems.append(
                    f"no measured resting defect involving {target} authorizes this reason"
                )
            elif after is None or after > 0.015:
                problems.append(f"{target} is still floating after the correction")
        return problems

    def _record_nudge_rejection(
        self,
        txid: int,
        target: str,
        delta: list[float],
        reason: str,
        failure: str,
    ) -> None:
        self._initializer_ledger["events"].append(
            {
                "id": txid,
                "status": "rejected",
                "attempt_idx": int(getattr(self, "attempt_idx", 1)),
                "target": target,
                "translation": delta,
                "reason": reason,
                "failure": failure,
            }
        )
        self._initializer_ledger["next_transaction_id"] = max(
            int(self._initializer_ledger.get("next_transaction_id") or 1), txid + 1
        )
        self._persist_initializer_ledger()

    def nudge_initializer_object(
        self,
        object: str,
        translation: list,
        reason: str,  # noqa: A002
    ) -> dict:
        if self.root_stage_name != "initializer":
            return {
                "status": "error",
                "output": {"text": ["nudge_object is an initializer-stage tool"]},
            }
        try:
            self._assert_no_incomplete_runtime_mutation()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "output": {"text": [f"nudge_object blocked by recovery: {exc}"]},
            }
        valid_reasons = {
            "penetration",
            "resting_defect",
            "visible_position_error",
        }
        target, name_error = self._resolve_initializer_object(str(object))
        if target is None:
            return {"status": "error", "output": {"text": [name_error]}}
        txid = int(self._initializer_ledger.get("next_transaction_id") or 1)
        try:
            delta = [float(x) for x in translation]
            if len(delta) != 3 or not all(math.isfinite(x) for x in delta):
                raise ValueError(
                    "translation must contain exactly three finite numbers"
                )
        except (TypeError, ValueError) as exc:
            return {"status": "error", "output": {"text": [str(exc)]}}
        if reason not in valid_reasons:
            return {
                "status": "error",
                "output": {"text": [f"reason must be one of {sorted(valid_reasons)}"]},
            }

        def reject(message: str, *, restore: Optional[str] = None) -> dict:
            if restore and self.blender_save and os.path.exists(restore):
                shutil.copy2(restore, self.blender_save)
            self._record_nudge_rejection(txid, target, delta, reason, message)
            return {
                "status": "error",
                "output": {
                    "text": [
                        "nudge_object rejected and the live scene was left unchanged: "
                        + message
                    ]
                },
            }

        magnitude = math.sqrt(sum(x * x for x in delta))
        if reason != "penetration" and magnitude < 0.002:
            return reject("translation is below the 2mm anti-churn minimum")
        if reason == "penetration" and magnitude < CONTACT_TOLERANCE_M:
            return reject("translation is below the 0.25mm numerical contact tolerance")
        if reason == "resting_defect" and (
            abs(delta[0]) > 1e-6 or abs(delta[1]) > 1e-6 or delta[2] >= -0.002
        ):
            return reject(
                "resting_defect permits only a downward Z correction of at least 2mm"
            )
        if reason == "visible_position_error" and abs(delta[2]) > 0.002:
            return reject(
                "visible_position_error is XY-only; use resting_defect for a measured Z gap"
            )

        try:
            pre = self._read_penetration_data(tag=f"nudge_{txid}_before")
        except Exception as exc:  # noqa: BLE001
            return reject(f"could not inspect the pre-transaction scene: {exc}")
        body = self._body_map(pre).get(target)
        if body is None:
            return reject(f"{target} is missing from the live scene")
        physical_repair = reason in {"penetration", "resting_defect"}
        if reason == "penetration" and not any(
            target in (p.get("a"), p.get("b"))
            and excess_m(
                float(p.get("depth", 0)),
                contact_class(
                    p.get("a"),
                    p.get("b"),
                    self._support_body_map(),
                    p.get("separation_direction"),
                ),
            )
            > 0.0
            for p in pre.get("penetrating_pairs", [])
        ):
            return reject(
                "no measured penetration authorizes a physical-repair allowance"
            )
        if reason == "resting_defect":
            gap = self._object_resting_state(pre, {target}).get(target, {}).get("gap")
            if gap is None or gap <= 0.015:
                return reject(
                    "no measured resting gap authorizes a physical-repair allowance"
                )
        ok, budget_reason = self._initializer_nudge_budget(
            target, physical_repair=physical_repair
        )
        if not ok:
            return reject(budget_reason)
        ext = [float(body["hi"][i]) - float(body["lo"][i]) for i in range(3)]
        diag = math.sqrt(sum(x * x for x in ext))
        per_call_cap = max(0.10, min(0.75, 2.0 * diag))
        if physical_repair:
            per_call_cap = PHYSICAL_REPAIR_MAX_M
        if magnitude > per_call_cap + 1e-9:
            return reject(
                f"requested {magnitude:.3f}m exceeds this object's per-call cap "
                f"of {per_call_cap:.3f}m"
            )
        active = [
            tx
            for tx in self._initializer_ledger.get("active_transactions", [])
            if tx.get("kind") in {None, "nudge"}
        ]
        target_distance = sum(
            float(tx.get("distance", 0.0))
            for tx in active
            if tx.get("target") == target
        )
        target_cap = max(0.20, min(1.0, 4.0 * diag))
        if target_distance + magnitude > target_cap + 1e-9:
            return reject(
                f"cumulative correction on {target} would be "
                f"{target_distance + magnitude:.3f}m, above its {target_cap:.3f}m cap"
            )
        if sum(float(tx.get("distance", 0.0)) for tx in active) + magnitude > 2.0:
            return reject("initializer-wide cumulative translation would exceed 2.0m")

        try:
            members = self._initializer_nudge_members(target)
        except Exception as exc:  # noqa: BLE001 - a target-only fallback can strand a stack
            return reject(str(exc))
        before_sig = pre.get("object_integrity", {}) or {}
        before_matrices = {
            name: before_sig[name]["matrix"]
            for name in members
            if name in before_sig and before_sig[name].get("matrix") is not None
        }
        if set(before_matrices) != set(members):
            return reject("could not capture every support-stack member's pose")

        self._ensure_undo_base()
        backup = self.render_path.parent / "tmp" / f"nudge_{txid}_pre.blend"
        if not self.blender_save or not os.path.exists(self.blender_save):
            return reject("live Blender save is unavailable")
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.blender_save, backup)
        self.count += 1
        script_path = self.script_path / f"{self.count}_trusted_nudge.py"
        script = (
            "import bpy\nfrom mathutils import Vector\n"
            f"names = {members!r}\n"
            f"delta = Vector({tuple(delta)!r})\n"
            "for name in names:\n"
            "    obj = bpy.data.objects.get(name)\n"
            "    if obj is None:\n"
            "        raise RuntimeError(f'missing nudge member {name}')\n"
            "    obj.location += delta\n"
        )
        with open(script_path, "w") as stream:
            stream.write(script)
        self._initializer_nudge_allow_names = set(members)
        self._initializer_nudge_allow_delta = delta
        try:
            success, _, stdout, stderr = self._execute_blender(
                str(script_path), "__norender__"
            )
        finally:
            self._initializer_nudge_allow_names = None
            self._initializer_nudge_allow_delta = None
        if not success:
            return reject(
                "trusted translation failed its integrity guard: "
                + _proc_text(stderr, stdout),
                restore=str(backup),
            )
        try:
            post = self._read_penetration_data(tag=f"nudge_{txid}_after")
        except Exception as exc:  # noqa: BLE001
            return reject(
                f"could not validate the translated scene: {exc}", restore=str(backup)
            )
        post_sig = post.get("object_integrity", {}) or {}
        after_matrices = {
            name: post_sig[name]["matrix"]
            for name in members
            if name in post_sig and post_sig[name].get("matrix") is not None
        }
        provisional = {
            "id": txid,
            "kind": "nudge",
            "target": target,
            "objects": {
                name: {
                    "before_matrix": before_matrices[name],
                    "after_matrix": after_matrices.get(name),
                }
                for name in members
            },
        }
        old_active = self._initializer_ledger["active_transactions"]
        self._initializer_ledger["active_transactions"] = [*old_active, provisional]
        validation_error = None
        try:
            integrity_ok, integrity_msg = self._rule_object_integrity(post)
        except Exception as exc:  # noqa: BLE001 - every failed validation rolls back
            validation_error = f"object-integrity validation could not complete: {exc}"
        finally:
            self._initializer_ledger["active_transactions"] = old_active
        if validation_error is not None:
            return reject(validation_error, restore=str(backup))
        problems = [] if integrity_ok else [integrity_msg]
        try:
            problems.extend(
                self._nudge_safety_violations(pre, post, target, members, reason)
            )
        except Exception as exc:  # noqa: BLE001 - unsafe uncertainty fails closed
            return reject(
                f"support/contact validation could not complete: {exc}",
                restore=str(backup),
            )
        if problems:
            return reject("; ".join(problems), restore=str(backup))

        tx = {
            **provisional,
            "translation": delta,
            "distance": magnitude,
            "reason": reason,
            "physical_repair": physical_repair,
            "attempt_idx": int(getattr(self, "attempt_idx", 1)),
        }
        pre_commit_ledger = copy.deepcopy(self._initializer_ledger)
        history_n = len(self.edit_history)
        ledger_history_n = len(self._ledger_history)
        edit_meta_n = len(self._edit_meta)
        try:
            self._initializer_ledger["active_transactions"].append(tx)
            self._initializer_ledger["events"].append({**tx, "status": "accepted"})
            self._initializer_ledger["next_transaction_id"] = txid + 1
            self._persist_initializer_ledger()
            self._push_edit_snapshot("nudge", transaction_id=txid)
        except Exception as exc:  # noqa: BLE001 - commit must be all-or-nothing
            self._atomic_restore_blend(backup, tag=f"nudge-{txid}-rollback")
            self._initializer_ledger = pre_commit_ledger
            del self.edit_history[history_n:]
            del self._ledger_history[ledger_history_n:]
            if hasattr(self, "_graph_history"):
                del self._graph_history[history_n:]
            del self._edit_meta[edit_meta_n:]
            failure = (
                f"could not commit the scene+ledger+undo transaction atomically: {exc}"
            )
            try:
                self._record_nudge_rejection(txid, target, delta, reason, failure)
            except (
                Exception
            ):  # keep the prior ledger rather than an accepted half-commit
                self._initializer_ledger = pre_commit_ledger
                try:
                    self._persist_initializer_ledger()
                except Exception:
                    pass
            return {
                "status": "error",
                "output": {
                    "text": ["nudge_object rejected and rolled back: " + failure]
                },
            }
        used_target = sum(
            float(row.get("distance", 0.0))
            for row in self._initializer_ledger["active_transactions"]
            if row.get("kind") in {None, "nudge"} and row.get("target") == target
        )
        return {
            "status": "success",
            "output": {
                "text": [
                    f"nudge_object committed transaction {txid}: translated {target} "
                    f"and {len(members) - 1} supported descendant(s) by "
                    f"({delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f})m. "
                    "Integrity, resting/support, and new/worsened penetration checks "
                    "passed. The result is undoable with undo_last_step. "
                    f"Target cumulative distance: {used_target:.3f}/{target_cap:.3f}m; "
                    "inspect the automatically attached current render, then rerun "
                    "check_rules_enforced now."
                ],
                "transaction": tx,
            },
        }

    # -- GPT-6 initializer object repair --------------------------------------- #

    def _require_initializer_capability(self, capability: str, tool: str) -> None:
        if self.root_stage_name != "initializer" or not self._has_capability(
            capability
        ):
            raise RuntimeError(f"{tool} is not enabled for this stage")
        if not self.blender_save or not os.path.isfile(self.blender_save):
            raise RuntimeError("live Blender scene is unavailable")
        if not self.moge_dir:
            raise RuntimeError("scene artifact directory is unavailable")

    def _load_runtime_scene_state(self) -> tuple[dict, dict, Optional[dict]]:
        from lib.tools.geometry.runtime_object_repair import load_runtime_inventory

        scene = Path(self.moge_dir)
        graph_path = scene / "scene_graph.json"
        placement_path = scene / "placement.json"
        graph = json.loads(graph_path.read_text())
        placement = json.loads(placement_path.read_text())
        if not isinstance(graph.get("nodes"), list):
            raise RuntimeError("scene_graph.json has no node list")
        if not isinstance(placement.get("objects"), list):
            raise RuntimeError("placement.json has no object list")
        return graph, placement, load_runtime_inventory(scene)

    def _resolve_scene_artifact(self, value: object, label: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"{label} path is unavailable")
        path = Path(value)
        candidates = (
            [path.resolve()]
            if path.is_absolute()
            else [
                (Path(self.moge_dir) / path).resolve(),
                (Path(__file__).resolve().parents[3] / path).resolve(),
                path.resolve(),
            ]
        )
        resolved = next(
            (candidate for candidate in candidates if candidate.is_file()), None
        )
        if resolved is None:
            raise RuntimeError(f"{label} is unavailable: {candidates[0]}")
        return resolved

    @staticmethod
    def _fallback_material_for(category: str) -> str:
        value = category.lower()
        if any(word in value for word in ("fruit", "apple", "banana", "orange")):
            return "fruit"
        if any(word in value for word in ("fork", "knife", "spoon", "scissor")):
            return "steel"
        if any(word in value for word in ("cloth", "towel", "napkin", "fabric")):
            return "fabric"
        if any(word in value for word in ("book", "paper")):
            return "paper"
        if any(word in value for word in ("mug", "cup", "plate", "bowl")):
            return "ceramic"
        if "glass" in value:
            return "glass"
        return "plastic"

    def _estimate_runtime_physics_materials(
        self,
        rows: list[dict],
        *,
        hints: dict[str, Optional[str]],
        transaction_id: int,
        force_refresh: bool,
        final_expected_names: Optional[list[str]] = None,
    ) -> dict[str, dict]:
        """Physics material/mass for EVERY object authored in one transaction.

        2026-09-15 (owner): with several new objects the estimate ran per object —
        one Blender render pass and one or two VLM calls each, ~35 s apiece (toast
        random tx6: four slices, ~2.5 min). Now one ``estimate_scene`` prefetch
        renders all of them in a single Blender pass and asks the VLM concurrently;
        the per-object call below then finds its entry cached in physics_vlm.json
        (``estimate_scene`` skips cached names), so every downstream record, manifest
        and fallback path is exactly the single-object one.
        """
        names = [str(row["mesh_name"]) for row in rows]
        if len(rows) > 1:
            self._prefetch_runtime_physics_estimates(
                rows, force_refresh=force_refresh, final_expected_names=final_expected_names
            )
            force_refresh = False
        return {
            name: self._estimate_runtime_physics_material(
                row,
                physical_material_hint=hints.get(name),
                transaction_id=transaction_id,
                force_refresh=force_refresh,
                final_expected_names=final_expected_names,
            )
            for name, row in zip(names, rows)
        }

    def _prefetch_runtime_physics_estimates(
        self,
        rows: list[dict],
        *,
        force_refresh: bool,
        final_expected_names: Optional[list[str]] = None,
    ) -> None:
        from lib.tools.geometry.physics_estimate import estimate_scene, objects_from_table

        scene = Path(self.moge_dir)
        out_json = scene / "physics" / "physics_vlm.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        names = [str(row["mesh_name"]) for row in rows]
        if force_refresh and out_json.is_file():
            try:
                existing = json.loads(out_json.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot refresh runtime materials from unreadable physics estimates: {out_json}"
                ) from exc
            if not isinstance(existing, dict):
                raise RuntimeError(
                    f"cannot refresh runtime materials because physics estimates are not a JSON object: {out_json}"
                )
            for name in names:
                existing.pop(name, None)
            self._atomic_write_json(out_json, existing)
        if final_expected_names is not None:
            expected_names = list(final_expected_names)
        else:
            placement = json.loads((scene / "placement.json").read_text())
            expected_names = [
                str(item["mesh_name"]) for item in placement.get("objects", []) if item.get("mesh_name")
            ]
        for name in names:
            if name not in expected_names:
                expected_names.append(name)
        print(
            f"[physics-estimate] one pass for {len(rows)} authored objects: {', '.join(names)}",
            file=sys.stderr,
            flush=True,
        )
        estimate_scene(
            objects_from_table(rows, scene),
            out_json,
            model=self.physics_model,
            expected_names=expected_names,
            log_stream=sys.stderr,
        )

    def _estimate_runtime_physics_material(
        self,
        row: dict,
        *,
        physical_material_hint: Optional[str],
        transaction_id: int,
        force_refresh: bool,
        final_expected_names: Optional[list[str]] = None,
    ) -> dict:
        from lib.tools.geometry.physics_estimate import (
            FALLBACK_DENSITY_KGM3,
            MATERIALS,
            estimate_scene,
            glb_anchors,
            objects_from_table,
        )

        scene = Path(self.moge_dir)
        out_json = scene / "physics" / "physics_vlm.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        name = str(row["mesh_name"])
        if force_refresh and out_json.is_file():
            try:
                existing = json.loads(out_json.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    "cannot refresh runtime material from unreadable "
                    f"physics estimates: {out_json}"
                ) from exc
            if not isinstance(existing, dict):
                raise RuntimeError(
                    "cannot refresh runtime material because physics estimates "
                    f"are not a JSON object: {out_json}"
                )
            existing.pop(name, None)
            self._atomic_write_json(out_json, existing)
        placement = json.loads((scene / "placement.json").read_text())
        expected_names = [
            str(item["mesh_name"])
            for item in placement.get("objects", [])
            if item.get("mesh_name")
        ]
        if name not in expected_names:
            expected_names.append(name)
        if final_expected_names is not None:
            expected_names = list(final_expected_names)
            if name not in expected_names:
                raise ValueError(
                    f"estimated object {name!r} is absent from final inventory"
                )
        results = estimate_scene(
            objects_from_table([row], scene),
            out_json,
            model=self.physics_model,
            expected_names=expected_names,
            # This method runs inside the stdio MCP server.  Estimator progress is
            # useful backend evidence, but stdout is reserved for JSON-RPC frames.
            log_stream=sys.stderr,
        )
        entry = results.get(name)
        manifest_path = out_json.with_name("physics_estimate_manifest.json")
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        )
        manifest_record = (manifest.get("objects") or {}).get(name, {})
        if isinstance(entry, dict) and entry.get("material") in MATERIALS:
            return {
                "status": "estimated",
                "material": entry.get("material"),
                "mass_kg": entry.get("mass_kg"),
                "mass_range_kg": entry.get("mass_range_kg"),
                "friction": entry.get("friction"),
                "density_kgm3": entry.get("density_kgm3"),
                "density_gate": entry.get("density_gate"),
                "provenance": {
                    "source": "physics_estimate.estimate_scene",
                    "model": self.physics_model,
                    "manifest_status": manifest_record.get("status"),
                    "transaction_id": transaction_id,
                },
            }

        material = str(
            physical_material_hint or self._fallback_material_for(str(row["category"]))
        )
        if material not in MATERIALS:
            raise ValueError(f"unknown physical material fallback {material!r}")
        volume, extents = glb_anchors(row["mesh_glb"])
        mass = max(float(volume) * FALLBACK_DENSITY_KGM3, 0.001)
        fallback = {
            "material": material,
            "mass_kg": round(mass, 4),
            "mass_range_kg": [round(mass * 0.5, 4), round(mass * 2.0, 4)],
            "mass_source": "runtime_explicit_fallback",
            "friction": float(MATERIALS[material]["friction"]),
            "density_kgm3": FALLBACK_DENSITY_KGM3,
            "density_gate": "fallback",
            "est_volume_m3": round(float(volume), 7),
            "est_obb_extents_m": [round(float(x), 5) for x in extents],
            "_note": (
                "runtime object estimate unavailable; explicit harness fallback "
                f"for transaction {transaction_id}"
            ),
        }
        all_results = {}
        if out_json.is_file():
            try:
                all_results = json.loads(out_json.read_text())
            except (OSError, json.JSONDecodeError):
                all_results = {}
        all_results[name] = fallback
        self._atomic_write_json(out_json, all_results)
        records = manifest.setdefault("objects", {})
        records[name] = {
            "status": "failed_fallback",
            "model": self.physics_model,
            "material": material,
            "mass_kg": fallback["mass_kg"],
            "mass_range_kg": fallback["mass_range_kg"],
            "mass_source": fallback["mass_source"],
            "friction": fallback["friction"],
            "density_kgm3": fallback["density_kgm3"],
            "density_gate": "fallback",
            "fallback": {
                "density_kgm3": FALLBACK_DENSITY_KGM3,
                "friction": fallback["friction"],
                "material": material,
            },
            "failure": manifest_record.get("failure", "estimate unavailable"),
            "runtime_transaction_id": transaction_id,
        }
        manifest["schema_version"] = int(manifest.get("schema_version") or 1)
        manifest["model"] = self.physics_model
        manifest["expected_count"] = len(records)
        manifest["estimated_count"] = sum(
            record.get("status") in {"estimated", "cached"}
            for record in records.values()
            if isinstance(record, dict)
        )
        manifest["fallback_count"] = len(records) - manifest["estimated_count"]
        manifest["coverage_status"] = (
            "full" if manifest["fallback_count"] == 0 else "partial"
        )
        manifest["physics_vlm"] = str(out_json)
        self._atomic_write_json(manifest_path, manifest)
        return {
            "status": "fallback",
            **{
                key: fallback.get(key)
                for key in (
                    "material",
                    "mass_kg",
                    "mass_range_kg",
                    "friction",
                    "density_kgm3",
                    "density_gate",
                )
            },
            "provenance": {
                "source": "runtime_explicit_fallback",
                "model": self.physics_model,
                "requested_material_hint": physical_material_hint,
                "transaction_id": transaction_id,
            },
        }

    @staticmethod
    def _visual_material_record(
        mesh_sha256: str,
        backend: str,
        transaction_id: int,
        *,
        code_sha256: Optional[str] = None,
        capture: Optional[dict] = None,
    ) -> dict:
        procedural = backend == "gpt6_blender_code"
        return {
            "status": "embedded",
            "sha256": mesh_sha256,
            "provenance": {
                "source": ("gpt6_blender_code" if procedural else "reconstruction_glb"),
                **({"backend": backend} if not procedural else {}),
                "texture_mode": "embedded_pbr_or_baked",
                "transaction_id": transaction_id,
                **({"code_sha256": code_sha256} if code_sha256 else {}),
                **(
                    {
                        "material_sha256": capture.get("material_sha256"),
                        "roundtrip_verified": bool(
                            capture.get("roundtrip_verified", False)
                        ),
                    }
                    if procedural and isinstance(capture, dict)
                    else {}
                ),
            },
        }

    def _write_runtime_bundle(self, bundle: dict) -> None:
        from lib.tools.geometry.runtime_object_repair import write_runtime_inventory

        scene = Path(self.moge_dir)
        self._restore_scene_graph_bytes(
            (json.dumps(bundle["scene_graph"], indent=2) + "\n").encode("utf-8")
        )
        self._atomic_write_json(scene / "placement.json", bundle["placement"])
        write_runtime_inventory(scene, bundle["runtime_inventory"])
        self._n2id_cache = None

    def _run_initializer_authored_code(
        self,
        *,
        code: str,
        kind: str,
        transaction_id: int,
        object_id: str,
        mesh_name: str,
        live_before: Path,
        declarations: Optional[dict] = None,
    ) -> dict:
        """Execute GPT-authored object code through the normal guarded wrapper.

        The model code runs against a transaction-local candidate Blend.  The live
        scene is promoted only after the Blender-side, capability-bound mutation
        guard accepts the exact add/mesh/pose delta and all backend-owned files are
        proven unchanged.  This deliberately does not create a generic
        execute-and-evaluate undo entry; the enclosing typed transaction owns the
        single chronological snapshot and commit/rollback decision.
        """
        import hashlib

        from lib.tools.blender.procedural_object import (
            build_target_prelude,
            validate_authored_code,
        )

        if kind not in {
            "execute_and_evaluate_objects",
            _COMPOSITION_MESH_MUTATION_KIND,
        }:
            raise ValueError(f"unsupported authored object mutation {kind!r}")
        parsed = validate_authored_code(self._parse_code(str(code)))
        is_batch = kind == "execute_and_evaluate_objects"
        if is_batch and not isinstance(declarations, dict):
            raise ValueError("initializer transaction has no resolved declarations")
        if not is_batch and (not object_id or not mesh_name):
            raise ValueError("backend object identity is unavailable")
        live_before = Path(live_before).resolve()
        if not live_before.is_file() or live_before.stat().st_size <= 0:
            raise RuntimeError("typed mutation recovery Blend is unavailable")
        if not self.blender_save:
            raise RuntimeError("live Blender save is unavailable")

        tx_dir = self._runtime_transaction_dir(int(transaction_id))
        candidate = tx_dir / "authored_candidate.blend"
        code_path = tx_dir / "authored_code.py"
        prelude = (
            f"ADDED_OBJECT_NAMES = {declarations['added_names']!r}\n"
            f"REMOVED_OBJECT_NAMES = {declarations['removed_names']!r}\n"
            if is_batch
            else build_target_prelude(mesh_name, object_id)
        ) + f"TRANSACTION_ID = {int(transaction_id)!r}\n\n"
        # Compile the unchanged submitted script separately from trusted bindings:
        # traceback line N must mean line N of the model's code, not N minus the
        # initializer/typed prelude. The outer wrapper still performs its complete
        # before/after integrity checks around this execution in module globals.
        executable = prelude + (
            f"exec(compile({parsed!r}, '<authored-code>', 'exec'), globals())\n"
        )
        self._durable_atomic_write_bytes(code_path, executable.encode("utf-8"))
        code_sha256 = hashlib.sha256(parsed.encode("utf-8")).hexdigest()

        protected = self._protected_artifact_snapshots()
        for path in (
            tx_dir / "recovery.json",
            tx_dir / "snapshot_manifest.json",
            code_path,
        ):
            protected.setdefault(
                path, (path.exists(), path.read_bytes() if path.exists() else b"")
            )

        live_path = Path(self.blender_save).resolve()
        old_file, old_save = self.blender_file, self.blender_save
        old_batch = getattr(self, "_initializer_object_transaction", None)
        old_composition = getattr(self, "_composition_mesh_transaction", None)
        had_composition = hasattr(self, "_composition_mesh_transaction")
        with tempfile.TemporaryDirectory(prefix="grase_authored_object_") as tmp:
            pristine = Path(tmp) / "before.blend"
            shutil.copy2(live_before, pristine)
            pristine_sha256 = self._typed_initializer_file_sha256(pristine)
            if self._typed_initializer_file_sha256(live_before) != pristine_sha256:
                raise RuntimeError("typed mutation recovery Blend changed while copied")
            self._durable_atomic_copy(pristine, candidate)
            success = False
            stdout = stderr = ""
            try:
                self.blender_file = str(candidate)
                self.blender_save = str(candidate)
                if is_batch:
                    self._initializer_object_transaction = {
                        "transaction_id": int(transaction_id),
                        "added_objects": list(declarations["added_names"].values()),
                        "removed_objects": list(declarations["removed_names"].values()),
                        # material-only edits on authored objects pass the guard
                        "authored_material_edit": True,
                    }
                else:
                    self._composition_mesh_transaction = {
                        "transaction_id": int(transaction_id),
                        "added_objects": [mesh_name],
                        "removed_objects": [mesh_name],
                    }
                success, _, stdout, stderr = self._execute_blender(
                    str(code_path), "__norender__"
                )
            finally:
                self.blender_file, self.blender_save = old_file, old_save
                self._initializer_object_transaction = old_batch
                if had_composition:
                    self._composition_mesh_transaction = old_composition
                else:
                    try:
                        del self._composition_mesh_transaction
                    except AttributeError:
                        pass

            changed = self._changed_protected_artifacts(protected)
            live_changed = (
                not live_path.is_file()
                or self._typed_initializer_file_sha256(live_path) != pristine_sha256
            )
            recovery_changed = (
                not live_before.is_file()
                or self._typed_initializer_file_sha256(live_before) != pristine_sha256
            )
            if changed or live_changed or recovery_changed:
                restore_errors = []
                try:
                    self._restore_protected_artifacts(protected)
                except Exception as exc:  # noqa: BLE001
                    restore_errors.append(f"protected artifact restore failed: {exc}")
                for label, path in (
                    ("live Blend", live_path),
                    ("recovery Blend", live_before),
                ):
                    try:
                        self._durable_atomic_copy(pristine, path)
                    except Exception as exc:  # noqa: BLE001
                        restore_errors.append(f"{label} restore failed: {exc}")
                detail = ", ".join(str(path) for path in changed)
                if live_changed:
                    detail = (detail + ", " if detail else "") + str(live_path)
                if recovery_changed:
                    detail = (detail + ", " if detail else "") + str(live_before)
                suffix = "; " + "; ".join(restore_errors) if restore_errors else ""
                raise RuntimeError(
                    "authored object code modified backend-owned transaction state: "
                    + detail
                    + suffix
                )
            if not success:
                headline = _python_exception_headline(stderr, stdout)
                raise RuntimeError(
                    "authored Blender code was rejected"
                    + (f" ({headline})" if headline else "")
                    + ": "
                    + _proc_text(stderr, stdout)
                )
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                raise RuntimeError("authored Blender code produced no candidate scene")
            candidate_sha256 = self._typed_initializer_file_sha256(candidate)
            self._durable_atomic_copy(candidate, live_path)
        candidate.unlink(missing_ok=True)
        self._last_authored_stdout = stdout
        return {
            "source": "gpt6_blender_code",
            "code_sha256": code_sha256,
            "code_path": str(code_path),
            "candidate_blend_sha256": candidate_sha256,
            "transaction_id": int(transaction_id),
            "object_id": object_id,
            "mesh_name": mesh_name,
            **({"declarations": copy.deepcopy(declarations)} if is_batch else {}),
        }

    def _take_authored_stdout(self) -> str:
        """Agent-facing print() output of the last authored transaction script (consumed
        once, so a commit reached without a fresh run cannot show a stale block)."""
        out = _script_output(getattr(self, "_last_authored_stdout", ""))
        self._last_authored_stdout = ""
        return out

    def _run_trusted_blender_code(self, code: str, tag: str) -> None:
        if not self.blender_save:
            raise RuntimeError("live Blender save is unavailable")
        self.count += 1
        script = self.script_path / f"{self.count}_{tag}.py"
        script.write_text(code)
        old_file = self.blender_file
        self.blender_file = self.blender_save
        try:
            success, _, stdout, stderr = self._execute_raw_blender(str(script))
        finally:
            self.blender_file = old_file
        if not success:
            raise RuntimeError(
                f"{tag} Blender operation failed: " + _proc_text(stderr, stdout)
            )

    def _export_initializer_authored_object(
        self,
        *,
        mesh_name: str,
        object_id: str,
        transaction_id: int,
    ) -> tuple[Path, dict]:
        """Canonicalize and durably export one code-authored target hierarchy."""
        from lib.tools.blender.procedural_object import (
            build_procedural_capture_script,
        )
        from lib.tools.geometry.runtime_object_repair import sha256_file

        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", mesh_name).strip("_")
        asset_dir = (
            Path(self.moge_dir)
            / "runtime_objects"
            / "assets"
            / f"tx_{int(transaction_id)}_{slug}"
        ).resolve()
        self._durable_mkdir(asset_dir)
        mesh_path = asset_dir / f"{mesh_name}.glb"
        capture_path = asset_dir / "capture.json"
        mesh_path.unlink(missing_ok=True)
        capture_path.unlink(missing_ok=True)
        script = build_procedural_capture_script(
            target_object_name=mesh_name,
            target_object_id=object_id,
            mesh_glb_path=mesh_path,
            capture_json_path=capture_path,
            diagnostics_json_path=self._runtime_transaction_dir(transaction_id)
            / f"{slug}_capture_validation.json",
            blend_output_path=Path(self.blender_save).resolve(),
        )
        try:
            self._run_trusted_blender_code(
                script, f"procedural_export_{int(transaction_id)}"
            )
        except RuntimeError as exc:
            # 2026-09-15 (owner): the agent used to receive the whole Blender output
            # (a 14-line traceback ending in "Blender quit") — the one line that names
            # the failed watertightness check was buried; 32 rollbacks across the gpt6
            # roots. Lead with that line and its remedy; keep the output for the journal.
            rejected = RuntimeError(_exporter_rejection(str(exc)))
            rejected.blender_output = str(exc)  # type: ignore[attr-defined]
            raise rejected from exc
        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            raise RuntimeError("procedural object export produced no nonempty GLB")
        try:
            capture = json.loads(capture_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "procedural object export produced no valid capture"
            ) from exc
        if not isinstance(capture, dict):
            raise RuntimeError("procedural object capture must be a JSON object")
        actual_sha256 = sha256_file(mesh_path)
        recorded_sha256 = capture.get("glb_sha256")
        if recorded_sha256 is not None and recorded_sha256 != actual_sha256:
            raise RuntimeError("procedural object capture has a stale GLB digest")
        capture["glb_sha256"] = actual_sha256
        self._fsync_file(mesh_path)
        self._fsync_file(capture_path)
        self._fsync_directory(asset_dir)
        return mesh_path.resolve(), capture

    def _initializer_bake_scope(
        self,
        touched: dict,
        declarations: dict,
        post: dict,
    ) -> dict:
        """Which bodies the joint settle SIMULATES and writes back (2026-09-15 rule).

        Owner rule: an execute_and_evaluate call that only builds or edits root surfaces
        does not simulate at all; a call that adds, replaces, removes or moves objects
        simulates exactly those objects' hierarchies — the authored objects plus every
        body whose support chain (scene-graph edges in Blender body names) reaches an
        authored or removed object — and every other object enters the simulation as a
        STATIC obstacle (see initializer_physics.settle_initializer_candidate
        dynamic_names). Changed support surfaces are deliberately NOT roots any more:
        the 09-14 rule made a desk/wall rebuild simulate everything on it, and on online7
        the shove out of an overlapping wall toppled a vase that stood on its own. An
        unavailable support map falls back to simulating every body when anything was
        authored, rather than risk leaving a dependent floating.
        """
        removed = set((declarations.get("removed_names") or {}).values())
        added = set((declarations.get("added_names") or {}).values())
        # A same-name REPLACEMENT is in both maps: it is new geometry that must settle
        # (audit F-H1: it used to be admitted static at its authored pose, never settled,
        # and its cargo dropped onto it).
        authored = {name for name in touched if name not in removed or name in added}
        roots = authored | removed
        bodies = set(post.get("object_integrity") or {})
        support = self._support_body_map()
        if not support:
            return {
                "roots": sorted(roots),
                "bake": sorted(bodies) if roots else [],
                "fallback": "support map unavailable: every body bakes",
            }

        def depends(body: str) -> bool:
            seen: set[str] = set()
            current: Optional[str] = body
            while current and current not in seen:
                seen.add(current)
                current = support.get(current)
                if current in roots:
                    return True
            return False

        bake = {body for body in bodies if body in authored or depends(body)}
        return {"roots": sorted(roots), "bake": sorted(bake)}

    def _surface_overlap_notices(
        self, post: dict, changed_surfaces: set[str], authored: set[str]
    ) -> list[str]:
        """Informational overlap notices for an initializer transaction (never a gate).

        A penetrating pair is reported when one side is a surface whose geometry this
        call added/changed, or an object this call authored, the pair is not
        surface<->surface (owned by the relationships / main-support rules), and the
        depth exceeds the pair's contact-class tolerance (the same ladder the
        no-penetration rule applies). Each notice names both bodies and the depth; the
        transaction commits regardless (owner rule: a wall build is neither simulated
        nor rejected) and the no-penetration rule remains the only enforcement.
        """
        interesting = set(changed_surfaces) | set(authored)
        if not interesting:
            return []
        surfaces = set(post.get("surface_geometry") or {})
        support = self._support_body_map()
        findings = []
        for p in post.get("penetrating_pairs") or []:
            a, b = str(p.get("a")), str(p.get("b"))
            if not ({a, b} & interesting):
                continue
            if a in surfaces and b in surfaces:
                continue
            depth = float(p.get("depth", 0.0))
            cls = contact_class(a, b, support, p.get("separation_direction"))
            if cls == SURFACE_SURFACE or depth <= tolerance_m(cls):
                continue
            mover = a if a in interesting else b
            other = b if mover == a else a
            role = "root surface" if mover in surfaces else "authored object"
            findings.append(
                f"{mover} ({role} in this call) overlaps {other} by {depth * 1000:.0f} mm "
                f"(contact tolerance {tolerance_m(cls) * 1000:.0f} mm)"
            )
        return findings

    def _settle_initializer_candidate(self, transaction_id: int) -> dict:
        from lib.tools.geometry.initializer_physics import (
            build_initializer_physics_bake_script,
            settle_initializer_candidate,
        )

        # execute_initializer_transaction publishes the bake scope through executor
        # state so the call signature (and its test stubs) stay unchanged.
        scope = getattr(self, "_initializer_bake_scope_state", None) or {}
        bake_names = scope.get("bake")
        result = settle_initializer_candidate(
            self.moge_dir,
            self.blender_save,
            self.blender_command,
            transaction_id,
            **({"dynamic_names": list(bake_names)} if bake_names is not None else {}),
        )
        bodies = sorted(result["body_parts"])
        baked = set(bodies) if bake_names is None else set(bake_names) & set(bodies)
        if baked:
            self._run_trusted_blender_code(
                build_initializer_physics_bake_script(
                    {name: result["total"][name] for name in bodies if name in baked},
                    {name: result["body_parts"][name] for name in bodies if name in baked},
                    self.blender_save,
                ),
                f"initializer_physics_bake_{transaction_id}",
            )
        for report in result.get("reports") or []:
            report["pose_applied"] = "baked" if report.get("name") in baked else "static"
        result["baked"] = True
        result["baked_bodies"] = sorted(baked)
        # Out-of-scope bodies were static obstacles in the sim (not simulated-and-
        # restored as before 2026-09-15); the key name is kept for readers.
        result["restored_bodies"] = sorted(set(bodies) - baked)
        result["static_bodies"] = result["restored_bodies"]
        self._atomic_write_json(
            Path(self.moge_dir)
            / "physics"
            / f"initializer_tx_{transaction_id}"
            / "result.json",
            result,
        )
        return result

    @staticmethod
    def _physics_report_names(reports: list[dict]) -> set[str]:
        names: set[str] = set()
        for report in reports:
            for value in (report.get("name"), *(report.get("members") or [])):
                if value:
                    names.add(str(value))
        return names

    @staticmethod
    def _typed_world_semantics_record_valid(
        value: object, *, require_hierarchy: bool = False
    ) -> bool:
        """Authenticate one generated world-semantics record."""
        if not isinstance(value, dict):
            return False
        required_ints = (
            "part_count",
            "mesh_count",
            "vertex_count",
            "edge_count",
            "polygon_count",
        )
        required_digests = (
            "geometry_sha256",
            "topology_sha256",
            "material_sha256",
            "modifier_sha256",
            "visibility_sha256",
            "sha256",
        )
        if require_hierarchy:
            required_digests = (*required_digests[:-1], "hierarchy_sha256", "sha256")
        if (
            value.get("schema_version") != 1
            or value.get("world_coordinate_decimals") != 5
        ):
            return False
        if any(
            not isinstance(value.get(key), int)
            or isinstance(value.get(key), bool)
            or value[key] < 0
            for key in required_ints
        ):
            return False
        if any(
            not isinstance(value.get(key), str)
            or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
            for key in required_digests
        ):
            return False
        unsigned = copy.deepcopy(value)
        recorded_digest = unsigned.pop("sha256")
        return recorded_digest == Executor._typed_initializer_json_sha256(unsigned)

    @staticmethod
    def _typed_world_semantics_match(before: dict, after: dict) -> bool:
        """Accept only a complete, matching canonical world-semantics record.

        The ordinary integrity signature intentionally notices local mesh/origin and
        hierarchy rewrites.  Typed GPT-6 physics, however, runs PoseSession.prepare,
        which may recenter those representations while leaving rendered world geometry
        exactly unchanged.  Missing or malformed records fail closed.
        """
        left = before.get("world_semantics")
        right = after.get("world_semantics")
        return Executor._typed_world_semantics_record_valid(left) and left == right

    @staticmethod
    def _typed_world_vertex_fallback_eligible(before: dict, after: dict) -> bool:
        """Require authenticated equality of every non-geometric world component."""
        if not isinstance(before, dict) or not isinstance(after, dict):
            return False
        left = before.get("world_semantics")
        right = after.get("world_semantics")
        if not Executor._typed_world_semantics_record_valid(
            left, require_hierarchy=True
        ) or not Executor._typed_world_semantics_record_valid(
            right, require_hierarchy=True
        ):
            return False
        left_unsigned = copy.deepcopy(left)
        right_unsigned = copy.deepcopy(right)
        for value in (left_unsigned, right_unsigned):
            value.pop("geometry_sha256", None)
            value.pop("sha256", None)
        return left_unsigned == right_unsigned

    @staticmethod
    def _typed_world_vertex_dump_code(names: list[str], output_path: Path) -> str:
        """Generate a read-only Blender dump of corresponding evaluated vertices."""
        if (
            not names
            or len(names) != len(set(names))
            or any(not isinstance(name, str) or not name for name in names)
        ):
            raise ValueError("world-vertex proof requires unique nonempty object names")
        return f"""
import bpy
import numpy as np

names = {names!r}
output_path = {str(output_path)!r}
depsgraph = bpy.context.evaluated_depsgraph_get()
arrays = {{
    'schema_version': np.asarray([1], dtype=np.int64),
    'names': np.asarray(names, dtype=np.str_),
}}
for index, name in enumerate(names):
    root = bpy.data.objects.get(name)
    if root is None:
        raise RuntimeError('world-vertex proof root is absent: ' + name)
    descendants = [root] + sorted(list(root.children_recursive), key=lambda o: o.name)
    part_names = []
    vertex_counts = []
    chunks = []
    for item in descendants:
        if item.type != 'MESH' or item.data is None:
            continue
        evaluated = item.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        try:
            local = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get('co', local)
            local = local.reshape((-1, 3))
            matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
            world = local @ matrix[:3, :3].T + matrix[:3, 3]
        finally:
            evaluated.to_mesh_clear()
        part_names.append(item.name)
        vertex_counts.append(len(world))
        chunks.append(world)
    if not chunks or sum(vertex_counts) <= 0:
        raise RuntimeError('world-vertex proof root has no vertices: ' + name)
    arrays['part_names_' + str(index)] = np.asarray(part_names, dtype=np.str_)
    arrays['vertex_counts_' + str(index)] = np.asarray(vertex_counts, dtype=np.int64)
    arrays['vertices_' + str(index)] = np.vstack(chunks).astype(np.float64, copy=False)
np.savez(output_path, **arrays)
"""

    def _dump_typed_world_vertices(
        self, blend_path: Path, names: list[str], output_path: Path
    ) -> None:
        """Run one trusted, no-autoexec, read-only vertex dump."""
        blend_path = Path(blend_path).resolve()
        if not blend_path.is_file() or blend_path.stat().st_size <= 0:
            raise RuntimeError(f"world-vertex proof Blend is unavailable: {blend_path}")
        script_path = output_path.with_suffix(".py")
        script_path.write_text(
            self._typed_world_vertex_dump_code(names, output_path), encoding="utf-8"
        )
        output_path.unlink(missing_ok=True)
        before_sha = self._typed_initializer_file_sha256(blend_path)
        old_file = self.blender_file
        self.blender_file = str(blend_path)
        try:
            success, _, stdout, stderr = self._execute_raw_blender(
                str(script_path), disable_autoexec=True
            )
        finally:
            self.blender_file = old_file
        after_sha = self._typed_initializer_file_sha256(blend_path)
        if before_sha != after_sha:
            raise RuntimeError("read-only world-vertex proof changed its input Blend")
        if not success or not output_path.is_file():
            raise RuntimeError(
                "world-vertex proof Blender query failed: "
                + _proc_text(stderr, stdout or "no NPZ artifact was produced")
            )

    @staticmethod
    def _read_typed_world_vertex_dump(path: Path, names: list[str]) -> dict[str, dict]:
        import numpy as np

        try:
            with np.load(path, allow_pickle=False) as dump:
                schema = np.asarray(dump["schema_version"])
                recorded_names = [str(value) for value in np.asarray(dump["names"])]
                if (
                    schema.shape != (1,)
                    or int(schema[0]) != 1
                    or recorded_names != names
                ):
                    raise RuntimeError("world-vertex proof NPZ identity is invalid")
                records = {}
                for index, name in enumerate(names):
                    part_names = [
                        str(value) for value in np.asarray(dump[f"part_names_{index}"])
                    ]
                    counts = np.asarray(dump[f"vertex_counts_{index}"], dtype=np.int64)
                    vertices = np.asarray(dump[f"vertices_{index}"], dtype=np.float64)
                    if (
                        not part_names
                        or len(part_names) != len(set(part_names))
                        or counts.shape != (len(part_names),)
                        or np.any(counts < 0)
                        or vertices.ndim != 2
                        or vertices.shape[1:] != (3,)
                        or int(counts.sum()) != len(vertices)
                        or len(vertices) <= 0
                        or not np.isfinite(vertices).all()
                    ):
                        raise RuntimeError(
                            f"world-vertex proof NPZ payload is invalid for {name}"
                        )
                    records[name] = {
                        "part_names": part_names,
                        "vertex_counts": counts,
                        "vertices": vertices,
                    }
        except (OSError, KeyError, ValueError) as exc:
            raise RuntimeError("world-vertex proof returned an invalid NPZ") from exc
        return records

    @staticmethod
    def _compare_typed_world_vertex_records(
        before: dict[str, dict],
        after: dict[str, dict],
        names: list[str],
        *,
        tolerance_m: float = _TYPED_INITIALIZER_NUMERICAL_JITTER_TOLERANCE,
    ) -> dict[str, dict]:
        import numpy as np

        if (
            not isinstance(tolerance_m, (int, float))
            or isinstance(tolerance_m, bool)
            or not math.isfinite(float(tolerance_m))
            or tolerance_m < 0
        ):
            raise ValueError(
                "world-vertex proof tolerance must be finite and nonnegative"
            )
        results = {}
        for name in names:
            left, right = before.get(name), after.get(name)
            if not isinstance(left, dict) or not isinstance(right, dict):
                raise RuntimeError(f"world-vertex proof is missing {name}")
            if left.get("part_names") != right.get("part_names") or not np.array_equal(
                left.get("vertex_counts"), right.get("vertex_counts")
            ):
                raise RuntimeError(
                    f"typed repair unexpectedly changed {name} world-vertex correspondence"
                )
            left_vertices = np.asarray(left.get("vertices"), dtype=np.float64)
            right_vertices = np.asarray(right.get("vertices"), dtype=np.float64)
            if (
                left_vertices.ndim != 2
                or left_vertices.shape[1:] != (3,)
                or left_vertices.shape != right_vertices.shape
                or len(left_vertices) <= 0
                or not np.isfinite(left_vertices).all()
                or not np.isfinite(right_vertices).all()
            ):
                raise RuntimeError(f"world-vertex proof arrays are invalid for {name}")
            max_error = float(np.max(np.abs(left_vertices - right_vertices)))
            if max_error > float(tolerance_m):
                raise RuntimeError(
                    f"typed repair unexpectedly changed {name}: world-vertex error "
                    f"{max_error:.9g}m exceeds {float(tolerance_m):.9g}m"
                )
            results[name] = {
                "schema_version": 1,
                "method": "trusted_corresponding_world_vertices",
                "max_abs_error_m": max_error,
                "tolerance_m": float(tolerance_m),
                "vertex_count": int(len(left_vertices)),
                "part_count": len(left["part_names"]),
            }
        return results

    def _typed_world_vertex_equivalence(
        self,
        pre_blend_path: Path,
        post_blend_path: Path,
        names: list[str],
    ) -> dict[str, dict]:
        """Prove near-identical world geometry from immutable pre/post Blend files."""
        if getattr(self, "harness_profile", "baseline") != "gpt6_v1":
            raise RuntimeError("world-vertex proof is unavailable outside gpt6_v1")
        pre_path = Path(pre_blend_path).resolve()
        post_path = Path(post_blend_path).resolve()
        if pre_path == post_path:
            raise RuntimeError("world-vertex proof requires distinct pre/post Blends")
        ordered_names = sorted(set(names))
        if ordered_names != sorted(names) or not ordered_names:
            raise RuntimeError("world-vertex proof object set is invalid")
        with tempfile.TemporaryDirectory(prefix="grase_typed_world_vertices_") as tmp:
            root = Path(tmp)
            pre_dump = root / "pre.npz"
            post_dump = root / "post.npz"
            self._dump_typed_world_vertices(pre_path, ordered_names, pre_dump)
            self._dump_typed_world_vertices(post_path, ordered_names, post_dump)
            before = self._read_typed_world_vertex_dump(pre_dump, ordered_names)
            after = self._read_typed_world_vertex_dump(post_dump, ordered_names)
            return self._compare_typed_world_vertex_records(
                before, after, ordered_names
            )

    def _validate_typed_initializer_result(
        self,
        pre: dict,
        post: dict,
        *,
        target: str,
        physics_reports: Optional[list[dict]] = None,
        pre_blend_path: Optional[Path] = None,
        post_blend_path: Optional[Path] = None,
    ) -> dict[str, dict]:
        before = pre.get("object_integrity", {}) or {}
        after = post.get("object_integrity", {}) or {}
        before_names, after_names = set(before), set(after)
        if before_names != after_names or target not in after_names:
            raise RuntimeError("typed repair changed the imported-object identity set")
        physics_names = self._physics_report_names(physics_reports or [])
        allowed_changes = {target, *physics_names}
        representation_only = set()
        numerical_candidates = []
        for name in sorted(before_names & after_names):
            if name in allowed_changes:
                continue
            if before[name] != after[name]:
                if getattr(
                    self, "harness_profile", "baseline"
                ) == "gpt6_v1" and self._typed_world_semantics_match(
                    before[name], after[name]
                ):
                    representation_only.add(name)
                    continue
                if getattr(
                    self, "harness_profile", "baseline"
                ) == "gpt6_v1" and self._typed_world_vertex_fallback_eligible(
                    before[name], after[name]
                ):
                    numerical_candidates.append(name)
                    continue
                raise RuntimeError(f"typed repair unexpectedly changed {name}")
        numerical_proof = {}
        if numerical_candidates:
            if pre_blend_path is None or post_blend_path is None:
                raise RuntimeError(
                    f"typed repair unexpectedly changed {numerical_candidates[0]}"
                )
            numerical_proof = self._typed_world_vertex_equivalence(
                Path(pre_blend_path),
                Path(post_blend_path),
                numerical_candidates,
            )
        numerical_jitter = set(numerical_proof)
        touched = {}
        touched_names = (
            ((before_names | after_names) & allowed_changes)
            | representation_only
            | numerical_jitter
        )
        for name in sorted(touched_names):
            touched[name] = {
                "before_signature": copy.deepcopy(before.get(name)),
                "after_signature": copy.deepcopy(after.get(name)),
                "before_matrix": (before.get(name) or {}).get("matrix"),
                "after_matrix": (after.get(name) or {}).get("matrix"),
                **(
                    {"representation_only": True} if name in representation_only else {}
                ),
                **({"numerical_jitter": True} if name in numerical_jitter else {}),
                **(
                    {"world_vertex_equivalence": numerical_proof[name]}
                    if name in numerical_jitter
                    else {}
                ),
            }
        return touched

    @staticmethod
    def _typed_initializer_json_sha256(value: object) -> str:
        """Digest a semantic JSON snapshot with one deterministic encoding."""
        import hashlib

        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _typed_initializer_file_sha256(path: Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _typed_initializer_bytes_sha256(payload: bytes) -> str:
        import hashlib

        return hashlib.sha256(payload).hexdigest()

    def _typed_initializer_artifact_binding(
        self,
        path: Path,
        *,
        display_path: Optional[str] = None,
        required: bool,
    ) -> dict:
        """Bind one canonical artifact path to its exact current bytes."""
        resolved = path.resolve()
        row = {
            "path": str(display_path if display_path is not None else resolved),
            "resolved_path": str(resolved),
            "exists": resolved.is_file(),
        }
        if not row["exists"]:
            row.update(size_bytes=None, sha256=None)
            if required:
                raise RuntimeError(
                    f"required transaction artifact is missing: {resolved}"
                )
            return row
        size = resolved.stat().st_size
        if size <= 0 and required:
            raise RuntimeError(f"required transaction artifact is empty: {resolved}")
        row.update(
            size_bytes=int(size),
            sha256=self._typed_initializer_file_sha256(resolved),
        )
        return row

    def _typed_initializer_recorded_artifact_binding(
        self,
        value: object,
        *,
        label: str,
        required: bool,
    ) -> dict:
        """Resolve a placement/inventory path and retain its recorded spelling."""
        if not isinstance(value, str) or not value.strip():
            if required:
                raise RuntimeError(f"{label} has no nonempty artifact path")
            return {
                "path": value,
                "resolved_path": None,
                "exists": False,
                "size_bytes": None,
                "sha256": None,
            }
        try:
            resolved = self._resolve_scene_artifact(value, label)
        except Exception:
            if required:
                raise
            raw = Path(value)
            resolved = (
                raw.resolve()
                if raw.is_absolute()
                else (Path(self.moge_dir) / raw).resolve()
            )
        return self._typed_initializer_artifact_binding(
            resolved,
            display_path=value,
            required=required,
        )

    @staticmethod
    def _typed_initializer_json_file(path: Path) -> dict:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"cannot read transaction artifact JSON at {path}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"transaction artifact JSON at {path} must be an object")
        return value

    def _typed_initializer_target_manifest(
        self,
        *,
        object_id: str,
        target_mesh: str,
        graph: Optional[dict] = None,
        placement: Optional[dict] = None,
        runtime_inventory: Optional[dict] = None,
        inventory_record: Optional[dict] = None,
        expected_present: bool,
    ) -> dict:
        """Bind one target's canonical identity, geometry, masks, pose, and material.

        ``inventory_record`` is used by removal to preserve the exact withdrawn
        overlay record after the active inventory no longer contains the object.
        """
        scene = Path(self.moge_dir)
        if graph is None:
            graph = self._typed_initializer_json_file(scene / "scene_graph.json")
        if placement is None:
            placement = self._typed_initializer_json_file(scene / "placement.json")
        if runtime_inventory is None:
            inventory_path = scene / "runtime_objects" / "inventory.json"
            runtime_inventory = (
                self._typed_initializer_json_file(inventory_path)
                if inventory_path.is_file()
                else None
            )

        nodes = [
            node
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id") == object_id
        ]
        rows = [
            row
            for row in placement.get("objects", [])
            if isinstance(row, dict)
            and (
                row.get("mesh_name") == target_mesh
                or f"{row.get('category')}#{row.get('instance')}" == object_id
            )
        ]
        active_records = [
            row
            for row in (runtime_inventory or {}).get("objects", [])
            if isinstance(row, dict) and row.get("id") == object_id
        ]
        if len(nodes) > 1 or len(rows) > 1 or len(active_records) > 1:
            raise RuntimeError(f"target {object_id} has duplicate canonical bindings")
        present = len(nodes) == 1 and len(rows) == 1
        if expected_present and not present:
            raise RuntimeError(f"target {object_id} is absent from graph/placement")
        if not expected_present and (nodes or rows or active_records):
            raise RuntimeError(f"withdrawn target {object_id} remains active")

        record = copy.deepcopy(
            inventory_record or (active_records[0] if active_records else None)
        )
        node = copy.deepcopy(nodes[0] if nodes else (record or {}).get("graph_node"))
        placement_row = copy.deepcopy(
            rows[0] if rows else (record or {}).get("placement")
        )
        if not expected_present and inventory_record is None:
            result = {
                "object_id": object_id,
                "mesh_name": target_mesh,
                "present": False,
            }
            tombstones = [
                row
                for row in (runtime_inventory or {}).get("tombstones", [])
                if row.get("id") == object_id
            ]
            if len(tombstones) > 1:
                raise RuntimeError(
                    f"withdrawn target {object_id} has duplicate tombstones"
                )
            if tombstones:
                tombstone = copy.deepcopy(tombstones[0])
                retired_mesh = self._typed_initializer_recorded_artifact_binding(
                    tombstone["placement"]["mesh_glb"],
                    label=f"withdrawn target {object_id} mesh",
                    required=True,
                )
                if retired_mesh["sha256"] != tombstone["mesh_sha256"]:
                    raise RuntimeError(
                        f"withdrawn target {object_id} mesh digest changed"
                    )
                result.update(tombstone=tombstone, retired_mesh=retired_mesh)
            result["semantic_sha256"] = self._typed_initializer_json_sha256(result)
            return result
        if not isinstance(node, dict) or not isinstance(placement_row, dict):
            raise RuntimeError(
                f"target {object_id} has no manifestable graph/placement state"
            )

        current_revision = None
        if isinstance(record, dict):
            revision_number = record.get("current_mesh_revision")
            revisions = [
                revision
                for revision in record.get("mesh_revisions", [])
                if isinstance(revision, dict)
                and revision.get("revision") == revision_number
            ]
            if len(revisions) != 1:
                raise RuntimeError(
                    f"target {object_id} has no unique current mesh revision"
                )
            current_revision = copy.deepcopy(revisions[0])
            mesh_path = current_revision.get("mesh_glb")
            mask_path = record.get("mask_path")
            scoring_mask_path = record.get("scoring_mask_path", mask_path)
        else:
            mesh_path = placement_row.get("mesh_glb")
            mask_path = node.get("mask_path")
            scoring_mask_path = placement_row.get("mask_path") or mask_path

        mesh = self._typed_initializer_recorded_artifact_binding(
            mesh_path, label=f"target {object_id} current mesh", required=True
        )
        mask = self._typed_initializer_recorded_artifact_binding(
            mask_path,
            label=f"target {object_id} source mask",
            required=(record or {}).get("mask_policy") != "excluded",
        )
        scoring_mask = self._typed_initializer_recorded_artifact_binding(
            scoring_mask_path,
            label=f"target {object_id} scoring mask",
            required=(record or {}).get("mask_policy") != "excluded",
        )
        source_mask = None
        if (record or {}).get("mask_policy") == "excluded" and (record or {}).get(
            "origin"
        ) == "source_revised":
            source_mask = self._typed_initializer_recorded_artifact_binding(
                record.get("source_mask_path"),
                label=f"target {object_id} immutable source mask",
                required=True,
            )
            if source_mask["sha256"] != record.get("source_mask_sha256"):
                raise RuntimeError(
                    f"target {object_id} immutable source mask digest changed"
                )
        if isinstance(record, dict):
            expected_hashes = {
                "current mesh": (current_revision or {}).get("sha256"),
                "source mask": record.get("mask_sha256"),
                "scoring mask": record.get("scoring_mask_sha256"),
            }
            actual_hashes = {
                "current mesh": mesh["sha256"],
                "source mask": mask["sha256"],
                "scoring mask": scoring_mask["sha256"],
            }
            for label, expected in expected_hashes.items():
                if expected != actual_hashes[label]:
                    raise RuntimeError(
                        f"target {object_id} {label} digest differs from runtime inventory"
                    )

        def object_value(path: Path, *, container: Optional[str] = None):
            if not path.is_file():
                return None
            payload = self._typed_initializer_json_file(path)
            values = payload.get(container) if container else payload
            return (
                copy.deepcopy((values or {}).get(target_mesh))
                if isinstance(values, dict)
                else None
            )

        physics_values = {
            "physics_vlm": object_value(scene / "physics" / "physics_vlm.json"),
            "physics_estimate_manifest": object_value(
                scene / "physics" / "physics_estimate_manifest.json",
                container="objects",
            ),
            "pose_changes": object_value(
                scene / "physics" / "pose_changes.json", container="objects"
            ),
            "blend_base": object_value(scene / "physics" / "blend_base.json"),
        }
        result = {
            "object_id": object_id,
            "mesh_name": target_mesh,
            "present": bool(expected_present),
            "origin": (record or {}).get("origin", "source"),
            "graph_node": node,
            "placement": placement_row,
            "inventory_record": record,
            "inventory_record_sha256": (
                self._typed_initializer_json_sha256(record)
                if isinstance(record, dict)
                else None
            ),
            "current_mesh_revision": current_revision,
            "current_mesh": mesh,
            "mask": mask,
            "scoring_mask": scoring_mask,
            **({"source_mask": source_mask} if source_mask is not None else {}),
            "physical_material": copy.deepcopy(
                (record or {}).get("physical_material") or physics_values["physics_vlm"]
            ),
            "visual_material": copy.deepcopy((record or {}).get("visual_material")),
            "physics_records": physics_values,
            "physics_records_sha256": self._typed_initializer_json_sha256(
                physics_values
            ),
        }
        result["semantic_sha256"] = self._typed_initializer_json_sha256(result)
        return result

    def _typed_initializer_artifact_manifest(
        self,
        *,
        txid: int,
        kind: str,
        state: str,
        object_id: Optional[str] = None,
        target_mesh: Optional[str] = None,
        target_expected_present: bool = True,
        withdrawn_target: Optional[dict] = None,
        touched_objects: Optional[dict] = None,
        authored_objects: Optional[dict] = None,
        target_states: Optional[list[dict]] = None,
        strict: bool,
    ) -> dict:
        """Hash-bind the canonical scene/USD inputs for one terminal transaction."""
        scene = Path(self.moge_dir)
        paths = {
            "live_blend": (
                Path(self.blender_save),
                str(Path(self.blender_save).resolve()),
            ),
            "scene_graph": (scene / "scene_graph.json", "scene_graph.json"),
            "placement": (scene / "placement.json", "placement.json"),
            "runtime_inventory": (
                scene / "runtime_objects" / "inventory.json",
                "runtime_objects/inventory.json",
            ),
            "physics_vlm": (
                scene / "physics" / "physics_vlm.json",
                "physics/physics_vlm.json",
            ),
            "physics_estimate_manifest": (
                scene / "physics" / "physics_estimate_manifest.json",
                "physics/physics_estimate_manifest.json",
            ),
            "pose_changes": (
                scene / "physics" / "pose_changes.json",
                "physics/pose_changes.json",
            ),
            "blend_base": (
                scene / "physics" / "blend_base.json",
                "physics/blend_base.json",
            ),
        }
        artifacts = {
            role: self._typed_initializer_artifact_binding(
                path,
                display_path=display,
                # An absent runtime overlay is a valid source-only state. Every
                # other artifact is a downstream scene/USD input and must exist at
                # a successful typed commit.
                required=strict and role != "runtime_inventory",
            )
            for role, (path, display) in paths.items()
        }
        manifest = {
            "schema_version": 1,
            "harness_profile": self.harness_profile,
            "transaction_id": int(txid),
            "kind": kind,
            "state": state,
            "artifacts": artifacts,
        }
        if object_id and target_mesh:
            manifest["target"] = self._typed_initializer_target_manifest(
                object_id=object_id,
                target_mesh=target_mesh,
                expected_present=target_expected_present,
            )
        if target_states is not None:
            manifest["targets"] = [
                self._typed_initializer_target_manifest(
                    object_id=row["object_id"],
                    target_mesh=row["mesh_name"],
                    expected_present=row["present"],
                )
                for row in target_states
            ]
        if withdrawn_target is not None:
            manifest["withdrawn_target"] = copy.deepcopy(withdrawn_target)
        if touched_objects is not None:
            manifest["touched_objects"] = copy.deepcopy(touched_objects)
        if authored_objects is not None:
            manifest["authored_objects"] = copy.deepcopy(authored_objects)
        manifest["manifest_sha256"] = self._typed_initializer_json_sha256(manifest)
        return manifest

    def _safe_typed_initializer_artifact_manifest(self, **kwargs) -> dict:
        """Best-effort state binding for a rollback/error terminal record."""
        kwargs["strict"] = False
        try:
            return self._typed_initializer_artifact_manifest(**kwargs)
        except Exception as exc:  # noqa: BLE001
            scene = Path(str(getattr(self, "moge_dir", None) or "."))
            partial = {
                "live_blend": Path(
                    str(getattr(self, "blender_save", None) or "missing.blend")
                ),
                "scene_graph": scene / "scene_graph.json",
                "placement": scene / "placement.json",
                "runtime_inventory": scene / "runtime_objects" / "inventory.json",
                "physics_vlm": scene / "physics" / "physics_vlm.json",
                "physics_estimate_manifest": scene
                / "physics"
                / "physics_estimate_manifest.json",
                "pose_changes": scene / "physics" / "pose_changes.json",
                "blend_base": scene / "physics" / "blend_base.json",
            }
            artifact_rows = {}
            for role, path in partial.items():
                try:
                    artifact_rows[role] = self._typed_initializer_artifact_binding(
                        path, required=False
                    )
                except Exception as artifact_error:  # noqa: BLE001
                    artifact_rows[role] = {
                        "path": str(path),
                        "resolved_path": os.path.abspath(str(path)),
                        "exists": False,
                        "size_bytes": None,
                        "sha256": None,
                        "error": str(artifact_error),
                    }
            result = {
                "schema_version": 1,
                "harness_profile": self.harness_profile,
                "transaction_id": int(kwargs["txid"]),
                "kind": kwargs["kind"],
                "state": kwargs["state"],
                "status": "partial",
                "error": str(exc),
                "artifacts": artifact_rows,
            }
            result["manifest_sha256"] = self._typed_initializer_json_sha256(result)
            return result

    def _verify_typed_initializer_artifact_manifest(
        self,
        manifest: object,
        *,
        txid: int,
        kind: str,
        state: str,
    ) -> dict:
        """Verify one strict terminal manifest against the current scene bytes.

        The narrow startup-commit reconciliation path is reachable after the journal
        and active ledger were persisted but before the recovery marker became
        terminal.  Neither durable record alone is permission to bless whatever files
        happen to be present after a crash: both copies must agree, their semantic
        digest must be intact, and rebuilding the manifest from the current canonical
        artifacts must reproduce the exact payload.
        """
        if not isinstance(manifest, dict):
            raise RuntimeError(
                f"transaction {txid} has no strict committed artifact manifest"
            )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("harness_profile") != self.harness_profile
            or int(manifest.get("transaction_id") or 0) != int(txid)
            or manifest.get("kind") != kind
            or manifest.get("state") != state
        ):
            raise RuntimeError(
                f"transaction {txid} committed artifact manifest identity is invalid"
            )
        unsigned = copy.deepcopy(manifest)
        recorded_digest = unsigned.pop("manifest_sha256", None)
        if not isinstance(
            recorded_digest, str
        ) or recorded_digest != self._typed_initializer_json_sha256(unsigned):
            raise RuntimeError(
                f"transaction {txid} committed artifact manifest digest is invalid"
            )
        if kind == "execute_and_evaluate_objects":
            targets = manifest.get("targets")
            if not isinstance(targets, list) or any(
                not isinstance(row, dict)
                or not isinstance(row.get("object_id"), str)
                or not isinstance(row.get("mesh_name"), str)
                or not isinstance(row.get("present"), bool)
                for row in targets
            ):
                raise RuntimeError(f"transaction {txid} has invalid batch targets")
            rebuilt = self._typed_initializer_artifact_manifest(
                txid=int(txid),
                kind=kind,
                state=state,
                target_states=targets,
                touched_objects=manifest.get("touched_objects"),
                authored_objects=manifest.get("authored_objects"),
                strict=True,
            )
            if rebuilt != manifest:
                raise RuntimeError(
                    f"transaction {txid} committed batch artifacts no longer match"
                )
            return manifest
        target = manifest.get("target")
        if not isinstance(target, dict) or not isinstance(target.get("present"), bool):
            raise RuntimeError(
                f"transaction {txid} committed artifact manifest has no target state"
            )
        object_id = target.get("object_id")
        target_mesh = target.get("mesh_name")
        if not isinstance(object_id, str) or not isinstance(target_mesh, str):
            raise RuntimeError(
                f"transaction {txid} committed artifact target identity is invalid"
            )
        rebuilt = self._typed_initializer_artifact_manifest(
            txid=int(txid),
            kind=kind,
            state=state,
            object_id=object_id,
            target_mesh=target_mesh,
            target_expected_present=target["present"],
            withdrawn_target=manifest.get("withdrawn_target"),
            touched_objects=manifest.get("touched_objects"),
            strict=True,
        )
        if rebuilt != manifest:
            raise RuntimeError(
                f"transaction {txid} committed artifacts no longer match their manifest"
            )
        return manifest

    def _restore_failed_runtime_mutation(
        self,
        *,
        txid: int,
        live_before: Path,
        graph_before: Optional[bytes],
        artifact_before: dict[Path, tuple[bool, bytes]],
        pipeline_names_before: list[str],
        ledger_after_request: dict,
    ) -> list[str]:
        errors: list[str] = []
        try:
            self._atomic_restore_blend(
                live_before, tag="runtime-mutation-rollback", durable=True
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Blend restore failed: {exc}")
        try:
            self._restore_runtime_artifacts(artifact_before, durable=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"artifact restore failed: {exc}")
        try:
            self._restore_scene_graph_bytes(graph_before, durable=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"graph restore failed: {exc}")
        self._pipeline_object_names = list(pipeline_names_before)
        self._n2id_cache = None
        self._initializer_ledger = copy.deepcopy(ledger_after_request)
        try:
            self._persist_initializer_ledger()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ledger restore failed: {exc}")
        try:
            self._cleanup_runtime_transaction_outputs(txid)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"transaction artifact cleanup failed: {exc}")
        return errors

    @staticmethod
    def _initializer_physics_feedback(
        reports: list[dict], rest: dict, authored_objects: dict
    ) -> str:
        """Present the validated initializer outcome in ordinary agent tool text."""
        summary = {
            key: rest[key]
            for key in (
                "status",
                "physics_hz",
                "steps_used",
                "quiet_steps",
                "thresholds",
            )
        }
        lines = [
            "Initializer simulation report for the saved, baked scene. "
            "Motion is relative to the scene just before simulation: dxy/dz are "
            "meters and tilt_deg is degrees. Support observations describe geometric "
            "relationships, not confirmed contact; top_z is meters. Raw USD speeds "
            "are diagnostic; the validated rest speeds account for sleeping bodies. "
            "Use the image and these observations to decide the next repair.",
            "Joint rest: " + json.dumps(summary, sort_keys=True),
        ]
        for report in reports:
            public = {
                "name": report["name"],
                "change": (
                    "authored"
                    if report["name"] in authored_objects
                    else "physics_bystander"
                ),
                "drift": report["drift"],
                "toppled": report["toppled"],
                "repair_needed": report["repair_needed"],
                "supports": report["supports"],
                "rest": report["velocity"],
                # "restored": simulated as an obstacle, pre-simulation pose kept.
                "pose_applied": report.get("pose_applied", "baked"),
            }
            lines.append(json.dumps(public, sort_keys=True))
        return "\n".join(lines)

    def _commit_typed_initializer_mutation(
        self,
        *,
        txid: int,
        reason: str,
        pre: dict,
        reports: list[dict],
        artifact_before: dict[Path, tuple[bool, bytes]],
        physics_state_change: Optional[dict] = None,
        authored_code: Optional[dict] = None,
        declarations: dict,
        render_view: tuple[float, float] = (0.0, 0.0),
        part_name_feedback: str = "",
        authored_objects: Optional[dict] = None,
        physics_simulated: bool = False,
        overlap_notice: Optional[list[str]] = None,
        script_output: str = "",
    ) -> dict:
        from lib.tools.geometry.inventory_contract import validate_scene_artifacts

        kind = "execute_and_evaluate_objects"
        object_id = target_mesh = ""

        # This is the same canonical graph/placement/material boundary consumed by
        # final Blender validation and later Isaac/USD conversion. Do not grant an
        # initializer ledger authorization until that downstream contract is complete.
        validate_scene_artifacts(
            self.moge_dir,
            allow_runtime_additions=True,
            validate_runtime_physics=True,
        )
        post = self._read_penetration_data(tag=f"{kind}_{txid}_after")
        from lib.tools.blender.initializer_transaction import (
            observed_object_changes,
        )

        touched = observed_object_changes(
            pre,
            post,
            added_names=list(declarations["added_names"].values()),
            removed_names=list(declarations["removed_names"].values()),
        )
        # An authored pose can settle back to its original position. Keep its
        # direct authoring evidence even when the full transaction has no net delta.
        for name in authored_objects or {}:
            touched.setdefault(
                name,
                {
                    "before_signature": copy.deepcopy(
                        pre["object_integrity"].get(name)
                    ),
                    "after_signature": copy.deepcopy(
                        post["object_integrity"].get(name)
                    ),
                    "before_matrix": copy.deepcopy(
                        (pre["object_integrity"].get(name) or {}).get("matrix")
                    ),
                    "after_matrix": copy.deepcopy(
                        (post["object_integrity"].get(name) or {}).get("matrix")
                    ),
                },
            )
        name_to_id = {
            **self._name2id(),
            **{name: oid for oid, name in declarations["removed_names"].items()},
            **{name: oid for oid, name in declarations["added_names"].items()},
        }
        target_states = [
            {
                "object_id": name_to_id[name],
                "mesh_name": name,
                "present": row["after_signature"] is not None,
            }
            for name, row in touched.items()
        ]
        artifact_manifest = self._typed_initializer_artifact_manifest(
            txid=txid,
            kind=kind,
            state="committed",
            object_id=object_id,
            target_mesh=target_mesh,
            touched_objects=touched,
            authored_objects=authored_objects,
            target_states=target_states,
            strict=True,
        )
        physics_feedback = (
            self._initializer_physics_feedback(
                reports, physics_state_change["rest"], authored_objects or {}
            )
            if physics_simulated
            else ""
        )
        self._set_runtime_recovery_state(txid, "committing")
        transaction = {
            "id": txid,
            "kind": kind,
            "status": "committed",
            "attempt_idx": int(self.attempt_idx),
            "target": target_mesh,
            "object_id": object_id,
            "reason": reason,
            "objects": touched,
            "physics_reports": copy.deepcopy(reports),
            **(
                {
                    "declarations": copy.deepcopy(declarations),
                    "physics_simulated": physics_simulated,
                    "authored_objects": copy.deepcopy(authored_objects),
                }
            ),
            "overlap_notice": list(overlap_notice or []),
            "artifact_manifest": copy.deepcopy(artifact_manifest),
            **(
                {"physics_state_change": copy.deepcopy(physics_state_change)}
                if physics_state_change
                else {}
            ),
            **(
                {"authored_code": copy.deepcopy(authored_code)} if authored_code else {}
            ),
        }
        self._initializer_ledger["active_transactions"].append(transaction)
        self._persist_initializer_ledger()
        self._push_edit_snapshot(
            kind,
            transaction_id=txid,
            object_id=object_id,
            artifact_before=artifact_before,
        )
        self._durably_sync_typed_initializer_state(txid)
        commit_details = {
            "object_id": object_id,
            "mesh_name": target_mesh,
            "physical_material": None,
            "physics_reports": reports,
            "objects": touched,
            "artifact_manifest": artifact_manifest,
            **(
                {
                    "declarations": copy.deepcopy(declarations),
                    "physics_simulated": physics_simulated,
                    "authored_objects": copy.deepcopy(authored_objects),
                }
            ),
            "overlap_notice": list(overlap_notice or []),
            **(
                {"physics_state_change": physics_state_change}
                if physics_state_change
                else {}
            ),
            **({"authored_code": authored_code} if authored_code else {}),
        }
        try:
            # This append is the irreversible commit decision. An exception from the
            # combined journal+ledger helper is deliberately treated as ambiguous: the
            # JSONL frame may already be durable even if the following ledger write
            # failed. From this call onward the outer tool wrapper must never restore
            # the pre-transaction snapshot or publish a contradictory rolled_back
            # event. Startup recovery decides from the durable journal and either
            # rolls the commit forward or, when no complete frame exists, restores the
            # still-retained recovery snapshot.
            self._record_mutation_status(
                txid,
                kind,
                "committed",
                details=commit_details,
            )
            self._set_runtime_recovery_state(
                txid,
                "committed",
                details={
                    "object_id": object_id,
                    "mesh_name": target_mesh,
                    "objects": touched,
                    "artifact_manifest": artifact_manifest,
                    **(
                        {"physics_state_change": physics_state_change}
                        if physics_state_change
                        else {}
                    ),
                    **({"authored_code": authored_code} if authored_code else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001 - preserve an ambiguous durable commit
            marker_error = None
            try:
                self._set_runtime_recovery_state(
                    txid,
                    "recovery_needed",
                    details={
                        "phase": "commit_decision_reconciliation",
                        "error": str(exc),
                        "artifact_manifest": artifact_manifest,
                    },
                )
            except Exception as marker_exc:  # noqa: BLE001
                marker_error = str(marker_exc)
            detail = (
                f"{kind} transaction {txid} reached an ambiguous durable commit "
                "boundary. Its edited scene was preserved and was NOT "
                "rolled back. Restart the initializer to reconcile this transaction "
                "before making another edit or finalizing the scene."
            )
            if marker_error:
                detail += f" Recovery-marker update also failed: {marker_error}"
            return {
                "status": "error",
                "output": {
                    "text": [detail],
                    "transaction_id": txid,
                    "scene_mutation": "commit_recovery_pending",
                    "commit_recovery_pending": True,
                    "retryable": False,
                    "artifact_manifest": artifact_manifest,
                },
            }

        commit_text = (
            f"Initializer code transaction {txid} committed. "
            f"Added/replaced: {list(declarations['added_names'])!r}; "
            f"removed/replaced: {declarations['removed_objects']!r}. "
            "Object meshes, scene graph, placement and provenance are saved. "
            "Pose changes are recorded and the whole transaction is undoable. "
            + (
                "The returned scene and render are post-simulation, with settled poses baked. "
                f"Repair-needed objects: {[row['name'] for row in reports if row.get('repair_needed')]!r}. "
                "Inspect the reports and image; repair toppled or unsupported objects in your next edit."
                if physics_simulated
                else "No object creation/replacement or object pose change triggered initializer simulation in this call "
                "(a root-surface-only build or edit is checked geometrically for overlap and never simulates)."
            )
        )
        if overlap_notice:
            commit_text += (
                "\n\nOverlap notice (informational; nothing was rejected or simulated for "
                "it): " + "; ".join(overlap_notice) + ". The no-penetration rule lists any "
                "of these that persist; decide from the photo which side to move, if any."
            )
        if physics_feedback:
            commit_text += "\n\n" + physics_feedback
        if part_name_feedback:
            commit_text += "\n\n" + part_name_feedback
        if script_output:
            commit_text += "\n\n" + script_output
        # Rendering is presentation-only and happens after the terminal commit marker.
        # Keep both the call and response normalization inside this boundary: malformed
        # presentation output must not escape to a typed-tool wrapper that is still
        # responsible for rolling back genuine pre-decision failures.
        try:
            rendered = self._render_novel_view(*render_view)
            if not isinstance(rendered, dict) or rendered.get("status") != "success":
                raise RuntimeError("post-commit render was unavailable")
            output = rendered.get("output")
            if not isinstance(output, dict):
                raise TypeError("post-commit render output is malformed")
            text = output.get("text")
            if not isinstance(text, list):
                raise TypeError("post-commit render text is malformed")
            text.insert(0, commit_text)
            output["transaction"] = transaction
        except Exception as exc:  # noqa: BLE001 - committed state remains authoritative
            rendered = {
                "status": "success",
                "output": {"text": [commit_text], "transaction": transaction},
            }
            rendered["output"]["text"].append(
                f"The committed scene is preserved, but its requested view "
                f"(azimuth={render_view[0]:g}, elevation={render_view[1]:g}) "
                "could not be rendered. Use render_current_scene with the same "
                "view to inspect it; do not repeat the committed edit."
            )
            rendered["output"]["render_warning"] = {
                "azimuth": render_view[0],
                "elevation": render_view[1],
                "error": str(exc),
            }
        return rendered

    def _typed_initializer_failure(
        self,
        *,
        txid: int,
        kind: str,
        error: Exception,
        live_before: Path,
        graph_before: Optional[bytes],
        artifact_before: dict[Path, tuple[bool, bytes]],
        pipeline_names_before: list[str],
        ledger_after_request: dict,
        object_id: Optional[str] = None,
        target_mesh: Optional[str] = None,
    ) -> dict:
        # A commit-status audit failure can occur after the chronological snapshot
        # was pushed. Remove only that exact in-memory entry before restoring the
        # durable pre-transaction state below.
        if self._edit_meta and self._edit_meta[-1].get("transaction_id") == txid:
            self._edit_meta.pop()
            self.edit_history.pop()
            if self._ledger_history:
                self._ledger_history.pop()
            if self._graph_history:
                self._graph_history.pop()
        physics_reports = copy.deepcopy(getattr(error, "reports", []) or [])
        restore_errors = self._restore_failed_runtime_mutation(
            txid=txid,
            live_before=live_before,
            graph_before=graph_before,
            artifact_before=artifact_before,
            pipeline_names_before=pipeline_names_before,
            ledger_after_request=ledger_after_request,
        )
        status = "rolled_back" if not restore_errors else "error"
        artifact_manifest = self._safe_typed_initializer_artifact_manifest(
            txid=txid,
            kind=kind,
            state=status,
            object_id=object_id,
            target_mesh=target_mesh,
            target_expected_present=kind != "add_object",
        )
        audit_error = None
        try:
            self._record_mutation_status(
                txid,
                kind,
                status,
                details={
                    "error": str(error),
                    "error_class": _error_class(error),
                    **({"physics_reports": physics_reports} if physics_reports else {}),
                    **({"restore_errors": restore_errors} if restore_errors else {}),
                    "artifact_manifest": artifact_manifest,
                },
            )
        except Exception as exc:  # noqa: BLE001
            audit_error = str(exc)
        marker_error = None
        try:
            self._set_runtime_recovery_state(
                txid,
                "rolled_back"
                if not restore_errors and audit_error is None
                else "recovery_needed",
                details={
                    "error": str(error),
                    "error_class": _error_class(error),
                    **({"physics_reports": physics_reports} if physics_reports else {}),
                    **({"restore_errors": restore_errors} if restore_errors else {}),
                    **({"audit_error": audit_error} if audit_error else {}),
                    "artifact_manifest": artifact_manifest,
                },
            )
        except Exception as exc:  # noqa: BLE001
            marker_error = str(exc)
        detail = f"{kind} rejected and {'rolled back' if not restore_errors else 'left in a recovery-error state'}: {error}"
        if restore_errors:
            detail += "; " + "; ".join(restore_errors)
        if audit_error:
            detail += f"; terminal audit failed: {audit_error}"
        if marker_error:
            detail += f"; recovery marker update failed: {marker_error}"
        return {
            "status": "error",
            "output": {
                "text": [detail],
                "transaction_id": txid,
                "scene_mutation": "not_committed" if not restore_errors else "unknown",
                "artifact_manifest": artifact_manifest,
                **({"physics_reports": physics_reports} if physics_reports else {}),
            },
        }

    def _typed_initializer_recovery_enabled(self) -> bool:
        return (
            getattr(self, "root_stage_name", None) == "initializer"
            and self._has_capability("mutation_journal")
            and self._has_capability("initializer_code_transactions")
        )

    def _composition_mesh_recovery_enabled(self) -> bool:
        """Whether crash-safe procedural mesh replacement is active here."""
        return (
            getattr(self, "root_stage_name", None) == "composition"
            and getattr(self, "harness_profile", "baseline") == "gpt6_v1"
            and self._has_capability("mutation_journal")
            and self._has_capability("composition_mesh_edit")
        )

    def _composition_mesh_mutation_journal_events(self) -> list[dict]:
        """Load only authenticated numeric composition-mesh journal rows."""
        events: list[dict] = []
        for line_number, row in enumerate(
            read_mutation_journal(self._mutation_journal_path()), start=1
        ):
            if (
                row.get("stage") != "composition"
                or row.get("harness_profile") != "gpt6_v1"
                or row.get("kind") != _COMPOSITION_MESH_MUTATION_KIND
            ):
                continue
            raw_txid = row.get("transaction_id")
            if isinstance(raw_txid, bool):
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                )
            try:
                txid = int(raw_txid)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                ) from exc
            if txid < 1 or str(raw_txid) != str(txid):
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                )
            events.append({**row, "transaction_id": txid})
        return events

    def _composition_mesh_recovery_markers(self) -> list[tuple[Path, dict]]:
        if not self.moge_dir:
            return []
        root = Path(self.moge_dir) / "runtime_objects" / "transactions"
        if not root.is_dir():
            return []
        markers: list[tuple[Path, dict]] = []
        for marker_path in root.glob("tx_*/recovery.json"):
            try:
                marker = json.loads(marker_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot read runtime recovery marker {marker_path}"
                ) from exc
            if marker.get("kind") != _COMPOSITION_MESH_MUTATION_KIND:
                continue
            txid = marker.get("transaction_id")
            if (
                marker.get("schema_version") != 1
                or marker.get("stage") != "composition"
                or isinstance(txid, bool)
                or not isinstance(txid, int)
                or txid < 1
                or marker_path.parent != self._runtime_transaction_dir(txid)
            ):
                raise RuntimeError(f"invalid composition recovery marker {marker_path}")
            markers.append((marker_path, marker))
        return sorted(markers, key=lambda item: int(item[1]["transaction_id"]))

    def _assert_no_incomplete_composition_mesh_mutation(self) -> None:
        if not self._composition_mesh_recovery_enabled():
            return
        events = self._composition_mesh_mutation_journal_events()
        markers = self._composition_mesh_recovery_markers()
        marker_ids = {int(marker["transaction_id"]) for _, marker in markers}
        terminal = {"committed", "rolled_back", "undone"}
        for _path, marker in markers:
            if str(marker.get("state") or "") not in terminal:
                raise RuntimeError(
                    "composition mesh transaction "
                    f"{marker['transaction_id']} still requires recovery; restart "
                    "the composition stage before another scene edit"
                )
        grouped: dict[int, list[dict]] = {}
        for event in events:
            grouped.setdefault(int(event["transaction_id"]), []).append(event)
        for txid, rows in grouped.items():
            if txid in marker_ids:
                continue
            statuses = {str(row.get("status") or "") for row in rows}
            if (
                statuses & {"committed", "undone"}
                or str(rows[-1].get("status") or "") == "requested"
            ):
                raise RuntimeError(
                    f"composition mesh transaction {txid} has no recovery marker"
                )

    def _restore_composition_mesh_snapshot(
        self,
        *,
        txid: int,
        live_before: Path,
        graph_before: Optional[bytes],
        artifact_before: dict[Path, tuple[bool, bytes]],
        pipeline_names_before: list[str],
        cleanup_outputs: bool,
    ) -> None:
        errors: list[str] = []
        try:
            self._atomic_restore_blend(
                live_before, tag="composition-mesh-rollback", durable=True
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Blend restore failed: {exc}")
        try:
            self._restore_runtime_artifacts(artifact_before, durable=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"artifact restore failed: {exc}")
        try:
            self._restore_scene_graph_bytes(graph_before, durable=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"graph restore failed: {exc}")
        self._pipeline_object_names = list(pipeline_names_before)
        self._n2id_cache = None
        if cleanup_outputs:
            try:
                self._cleanup_runtime_transaction_outputs(txid)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"transaction output cleanup failed: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    def _recover_incomplete_composition_mesh_mutations(self) -> None:
        """Resolve composition mesh WAL records before physics warmup."""
        if not self._composition_mesh_recovery_enabled() or not self.moge_dir:
            return
        lock_fd = self._acquire_initializer_scene_lock()
        try:
            events = self._composition_mesh_mutation_journal_events()
            markers = self._composition_mesh_recovery_markers()
            marker_ids = {int(marker["transaction_id"]) for _, marker in markers}
            grouped: dict[int, list[dict]] = {}
            for event in events:
                grouped.setdefault(int(event["transaction_id"]), []).append(event)
            for txid, rows in sorted(grouped.items()):
                if txid in marker_ids:
                    continue
                if str(rows[-1].get("status") or "") == "requested":
                    self._record_mutation_status(
                        txid,
                        _COMPOSITION_MESH_MUTATION_KIND,
                        "rolled_back",
                        details={
                            "phase": "startup_request_recovery",
                            "scene_mutation": "not_started",
                        },
                    )
                elif any(
                    str(row.get("status") or "") in {"committed", "undone"}
                    for row in rows
                ):
                    raise RuntimeError(
                        f"composition mesh transaction {txid} lost its recovery marker"
                    )

            for marker_path, marker in markers:
                txid = int(marker["transaction_id"])
                state = str(marker.get("state") or "")
                rows = grouped.get(txid, [])
                last_committed = next(
                    (row for row in reversed(rows) if row.get("status") == "committed"),
                    None,
                )
                last_undone = next(
                    (row for row in reversed(rows) if row.get("status") == "undone"),
                    None,
                )
                if state in {"committed", "rolled_back", "undone"}:
                    expected_status = state
                    terminal_event = next(
                        (
                            row
                            for row in reversed(rows)
                            if row.get("status") == expected_status
                        ),
                        None,
                    )
                    if terminal_event is None:
                        raise RuntimeError(
                            f"composition mesh transaction {txid} has terminal marker "
                            f"{state!r} without its journal decision"
                        )
                    # A terminal marker with its journal decision is reconciled; the
                    # committed artifact manifest is NOT re-verified here: later layout
                    # edits and the final certify legitimately rewrite placement /
                    # scene_graph, so a second composition session (the final-settle
                    # repair round, 2026-09-15) could never start after a committed mesh
                    # edit ("cannot recover composition mesh transaction",
                    # v5accept robolab_clutter_shelf). Interrupted states below still verify.
                    tx_dir = self._runtime_transaction_dir(txid)
                    if (
                        marker.get("payloads_pruned") is not True
                        or (tx_dir / "before.blend").exists()
                        or (tx_dir / "snapshots").exists()
                    ):
                        self._prune_runtime_recovery_payload(txid)
                    continue
                if state == "undo_commit_decided" or last_undone is not None:
                    decision = marker.get("terminal_details") or {}
                    undo_event = decision.get("undo_event")
                    if last_undone is None:
                        unsigned = {
                            key: copy.deepcopy(value)
                            for key, value in decision.items()
                            if key != "decision_sha256"
                        }
                        if (
                            not isinstance(undo_event, dict)
                            or undo_event.get("status") != "undone"
                            or int(undo_event.get("transaction_id") or 0) != txid
                            or decision.get("decision_sha256")
                            != self._typed_initializer_json_sha256(unsigned)
                        ):
                            raise RuntimeError(
                                f"composition mesh transaction {txid} has an invalid "
                                "durable undo decision"
                            )
                        self._append_mutation_event(undo_event)
                        last_undone = undo_event
                    self._verify_typed_initializer_artifact_manifest(
                        (last_undone.get("details") or {}).get("artifact_manifest"),
                        txid=txid,
                        kind=_COMPOSITION_MESH_MUTATION_KIND,
                        state="undone",
                    )
                    self._set_runtime_recovery_state(
                        txid,
                        "undone",
                        details={"startup_reconciled": True},
                    )
                    continue
                if last_committed is not None and state != "undo_prepared":
                    manifest = (last_committed.get("details") or {}).get(
                        "artifact_manifest"
                    )
                    self._verify_typed_initializer_artifact_manifest(
                        manifest,
                        txid=txid,
                        kind=_COMPOSITION_MESH_MUTATION_KIND,
                        state="committed",
                    )
                    self._set_runtime_recovery_state(
                        txid,
                        "committed",
                        details={"startup_reconciled": True},
                    )
                    continue
                if state == "preparing":
                    self._record_mutation_status(
                        txid,
                        _COMPOSITION_MESH_MUTATION_KIND,
                        "rolled_back",
                        details={
                            "phase": "startup_recovery",
                            "interrupted_state": state,
                            "scene_mutation": "not_started",
                        },
                    )
                    self._set_runtime_recovery_state(
                        txid, "rolled_back", details={"startup_recovered": True}
                    )
                    continue
                (
                    live_before,
                    graph_before,
                    artifact_before,
                    names_before,
                    _ledger,
                ) = self._load_runtime_recovery_snapshots(txid, marker)
                undo_abandoned = state == "undo_prepared"
                self._restore_composition_mesh_snapshot(
                    txid=txid,
                    live_before=live_before,
                    graph_before=graph_before,
                    artifact_before=artifact_before,
                    pipeline_names_before=names_before,
                    cleanup_outputs=not undo_abandoned,
                )
                if undo_abandoned:
                    if last_committed is None:
                        raise RuntimeError(
                            f"composition mesh transaction {txid} has no committed "
                            "baseline for interrupted undo recovery"
                        )
                    self._verify_typed_initializer_artifact_manifest(
                        (last_committed.get("details") or {}).get("artifact_manifest"),
                        txid=txid,
                        kind=_COMPOSITION_MESH_MUTATION_KIND,
                        state="committed",
                    )
                    self._set_runtime_recovery_state(
                        txid,
                        "committed",
                        details={"startup_recovered_undo": True},
                    )
                else:
                    self._record_mutation_status(
                        txid,
                        _COMPOSITION_MESH_MUTATION_KIND,
                        "rolled_back",
                        details={
                            "phase": "startup_recovery",
                            "interrupted_state": state,
                            "scene_mutation": "restored",
                        },
                    )
                    self._set_runtime_recovery_state(
                        txid, "rolled_back", details={"startup_recovered": True}
                    )
        except Exception as exc:
            raise RuntimeError("cannot recover composition mesh transaction") from exc
        finally:
            self._release_initializer_scene_lock(lock_fd)

    def _acquire_initializer_scene_lock(self) -> int:
        """Serialize recovery and typed mutations across initializer MCP processes."""
        if not self.moge_dir:
            raise RuntimeError("scene artifact directory is unavailable")
        import fcntl

        lock_path = Path(self.moge_dir) / "audit" / "initializer_mutation.lock"
        self._durable_mkdir(lock_path.parent)
        existed = lock_path.exists()
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if not existed:
                self._fsync_directory(lock_path.parent)
            fcntl.flock(fd, fcntl.LOCK_EX)
            return fd
        except Exception:
            os.close(fd)
            raise

    @staticmethod
    def _release_initializer_scene_lock(fd: int) -> None:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _release_active_initializer_scene_lock(self) -> None:
        fd = getattr(self, "_active_initializer_scene_lock_fd", None)
        if fd is None:
            return
        self._active_initializer_scene_lock_fd = None
        self._release_initializer_scene_lock(fd)

    def _runtime_transaction_dir(self, txid: int) -> Path:
        if not self.moge_dir:
            raise RuntimeError("scene artifact directory is unavailable")
        if int(txid) < 1:
            raise ValueError("runtime transaction id must be positive")
        return (
            Path(self.moge_dir) / "runtime_objects" / "transactions" / f"tx_{int(txid)}"
        )

    def _runtime_recovery_marker_path(self, txid: int) -> Path:
        return self._runtime_transaction_dir(txid) / "recovery.json"

    def _set_runtime_recovery_state(
        self,
        txid: int,
        state: str,
        *,
        details: Optional[dict] = None,
    ) -> None:
        marker_path = self._runtime_recovery_marker_path(txid)
        if not marker_path.is_file():
            raise RuntimeError(
                f"runtime recovery marker is missing for transaction {txid}"
            )
        marker = json.loads(marker_path.read_text())
        if marker.get("schema_version") != 1 or int(
            marker.get("transaction_id") or 0
        ) != int(txid):
            raise RuntimeError(
                f"runtime recovery marker is invalid for transaction {txid}"
            )
        marker["state"] = state
        marker["updated_at_utc"] = _datetime.datetime.now(
            _datetime.timezone.utc
        ).isoformat()
        if details is not None:
            marker["terminal_details"] = copy.deepcopy(details)
        if state in {"committed", "rolled_back", "undone"}:
            marker.pop("recovery_error", None)
        self._durable_atomic_write_json(marker_path, marker)
        if state in {"committed", "rolled_back", "undone"}:
            self._prune_runtime_recovery_payload(txid)

    def _prune_runtime_recovery_payload(self, txid: int) -> None:
        """Drop large terminal snapshots while retaining marker + manifest metadata."""
        tx_dir = self._runtime_transaction_dir(txid)
        errors = []
        blend_snapshot = tx_dir / "before.blend"
        snapshot_dir = tx_dir / "snapshots"
        try:
            blend_snapshot.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"before.blend: {exc}")
        try:
            if snapshot_dir.is_dir():
                shutil.rmtree(snapshot_dir)
            elif snapshot_dir.exists():
                snapshot_dir.unlink()
        except OSError as exc:
            errors.append(f"snapshots: {exc}")
        try:
            self._fsync_directory(tx_dir)
        except OSError as exc:
            errors.append(f"directory fsync: {exc}")

        marker_path = tx_dir / "recovery.json"
        try:
            marker = json.loads(marker_path.read_text())
            marker["payloads_pruned"] = not errors
            marker["payloads_pruned_at_utc"] = _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat()
            if errors:
                marker["payload_prune_errors"] = errors
            else:
                marker.pop("payload_prune_errors", None)
            self._durable_atomic_write_json(marker_path, marker)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"marker annotation: {exc}")
        if errors:
            logging.warning(
                "runtime recovery payload pruning for transaction %s was incomplete: %s",
                txid,
                "; ".join(errors),
            )

    def _cleanup_runtime_transaction_outputs(self, txid: int) -> None:
        """Remove only uncommitted assets whose names are bound to this tx id."""
        scene = Path(self.moge_dir)
        for root_name in ("assets", "masks"):
            root = scene / "runtime_objects" / root_name
            if not root.is_dir():
                continue
            for candidate in root.glob(f"tx_{int(txid)}_*"):
                if candidate.is_symlink() or candidate.is_file():
                    candidate.unlink(missing_ok=True)
                elif candidate.is_dir():
                    shutil.rmtree(candidate)

    def _durably_sync_typed_initializer_state(self, txid: int) -> None:
        """Flush every canonical C-state byte before its committed journal event.

        The recovery marker is not the first commit signal: the append-only journal
        event is. Therefore Blender, graph, placement, physics/material records, and
        transaction-owned mesh/mask outputs must all cross a durability barrier before
        that event can be appended.
        """
        scene = Path(self.moge_dir)
        paths: set[Path] = {
            path
            for path in (
                Path(self.blender_save) if self.blender_save else None,
                self._scene_graph_path(),
                self._initializer_constraints_path(),
                Path(self.initializer_ledger_path)
                if getattr(self, "initializer_ledger_path", None)
                else None,
                *self._runtime_artifact_paths(),
            )
            if path is not None
        }
        runtime_root = scene / "runtime_objects"
        if runtime_root.is_dir():
            for owner in runtime_root.rglob(f"tx_{int(txid)}_*"):
                if owner.is_file():
                    paths.add(owner)
                elif owner.is_dir():
                    paths.update(
                        candidate
                        for candidate in owner.rglob("*")
                        if candidate.is_file()
                    )

        directories: set[Path] = set()
        for path in sorted(paths, key=str):
            if not path.is_file():
                continue
            self._fsync_file(path)
            directories.add(path.parent)
            if path.is_relative_to(scene):
                cursor = path.parent
                while cursor != scene:
                    directories.add(cursor)
                    cursor = cursor.parent
                directories.add(scene)
        for directory in sorted(
            directories, key=lambda value: len(value.parts), reverse=True
        ):
            self._fsync_directory(directory)

    @staticmethod
    def _runtime_recovery_error(
        kind: str,
        error: Exception,
        *,
        txid: Optional[int] = None,
        audited: bool,
    ) -> dict:
        audit_note = (
            "terminal audit recorded"
            if audited
            else "terminal audit could not be confirmed"
        )
        return {
            "status": "error",
            "output": {
                "text": [f"{kind} was not started; {audit_note}: {error}"],
                **({"transaction_id": txid} if txid is not None else {}),
                "scene_mutation": "not_started",
                "terminal_audit_recorded": audited,
            },
        }

    def _prepare_typed_initializer_mutation(self, kind: str, request: dict) -> dict:
        """Audit and durably snapshot a typed mutation before it can touch the scene."""
        if getattr(self, "_active_initializer_scene_lock_fd", None) is not None:
            return self._runtime_recovery_error(
                kind,
                RuntimeError("this executor already owns a typed initializer mutation"),
                audited=False,
            )
        try:
            lock_fd = self._acquire_initializer_scene_lock()
            self._active_initializer_scene_lock_fd = lock_fd
            ledger_path = getattr(self, "initializer_ledger_path", None)
            if ledger_path and Path(ledger_path).is_file():
                # Another MCP process may have committed since this executor was
                # constructed. Admission and ID allocation must use the durable
                # ledger while the per-scene lock is held.
                self._initializer_ledger = self._load_initializer_ledger()
            txid = int(self._begin_mutation(kind, request))
        except Exception as exc:  # noqa: BLE001
            self._release_active_initializer_scene_lock()
            raw_txid = getattr(exc, "transaction_id", None)
            return self._runtime_recovery_error(
                kind,
                exc,
                txid=int(raw_txid) if raw_txid is not None else None,
                audited=bool(getattr(exc, "audit_recorded", False)),
            )

        ledger_after_request = copy.deepcopy(self._initializer_ledger)
        try:
            live_before, graph_before, artifact_before, names_before = (
                self._runtime_transaction_snapshots(
                    txid,
                    kind=kind,
                    ledger_after_request=ledger_after_request,
                )
            )
        except Exception as exc:  # noqa: BLE001
            audit_error = None
            artifact_manifest = self._safe_typed_initializer_artifact_manifest(
                txid=txid,
                kind=kind,
                state="rolled_back",
            )
            try:
                self._record_mutation_status(
                    txid,
                    kind,
                    "rolled_back",
                    details={
                        "phase": "recovery_snapshot",
                        "error": str(exc),
                        "scene_mutation": "not_started",
                        "artifact_manifest": artifact_manifest,
                    },
                )
            except Exception as terminal_error:  # noqa: BLE001
                audit_error = terminal_error
            try:
                marker = self._runtime_recovery_marker_path(txid)
                if marker.is_file():
                    self._set_runtime_recovery_state(
                        txid,
                        "rolled_back" if audit_error is None else "recovery_needed",
                        details={
                            "phase": "recovery_snapshot",
                            "error": str(exc),
                            "artifact_manifest": artifact_manifest,
                            **(
                                {"audit_error": str(audit_error)}
                                if audit_error is not None
                                else {}
                            ),
                        },
                    )
            except Exception:  # noqa: BLE001
                # The terminal journal result above remains authoritative. Leaving a
                # nonterminal marker makes startup reconciliation retry its finalization.
                pass
            self._release_active_initializer_scene_lock()
            return self._runtime_recovery_error(
                kind,
                exc,
                txid=txid,
                audited=audit_error is None,
            )
        return {
            "status": "ready",
            "txid": txid,
            "ledger_after_request": ledger_after_request,
            "live_before": live_before,
            "graph_before": graph_before,
            "artifact_before": artifact_before,
            "pipeline_names_before": names_before,
        }

    def _prepare_composition_mesh_mutation(self, request: dict) -> dict:
        """Publish request + recovery snapshot before authored composition code."""
        kind = _COMPOSITION_MESH_MUTATION_KIND
        if not self._composition_mesh_recovery_enabled():
            return self._runtime_recovery_error(
                kind,
                RuntimeError("edit_object_mesh is not enabled for this stage"),
                audited=False,
            )
        if getattr(self, "_active_initializer_scene_lock_fd", None) is not None:
            return self._runtime_recovery_error(
                kind,
                RuntimeError("this executor already owns a runtime object mutation"),
                audited=False,
            )
        txid: Optional[int] = None
        try:
            lock_fd = self._acquire_initializer_scene_lock()
            self._active_initializer_scene_lock_fd = lock_fd
            self._assert_no_incomplete_composition_mesh_mutation()
            txid = self._next_runtime_mutation_id()
            self._append_mutation_event(
                {
                    "transaction_id": txid,
                    "kind": kind,
                    "status": "requested",
                    "request": copy.deepcopy(request),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self._release_active_initializer_scene_lock()
            return self._runtime_recovery_error(kind, exc, txid=txid, audited=False)

        try:
            live_before, graph_before, artifact_before, names_before = (
                self._runtime_transaction_snapshots(
                    txid,
                    kind=kind,
                    ledger_after_request={},
                )
            )
        except Exception as exc:  # noqa: BLE001
            audit_error = None
            try:
                self._record_mutation_status(
                    txid,
                    kind,
                    "rolled_back",
                    details={
                        "phase": "recovery_snapshot",
                        "error": str(exc),
                        "scene_mutation": "not_started",
                    },
                )
            except Exception as terminal_error:  # noqa: BLE001
                audit_error = terminal_error
            try:
                marker = self._runtime_recovery_marker_path(txid)
                if marker.is_file():
                    self._set_runtime_recovery_state(
                        txid,
                        "rolled_back" if audit_error is None else "recovery_needed",
                        details={
                            "phase": "recovery_snapshot",
                            "error": str(exc),
                            **(
                                {"audit_error": str(audit_error)}
                                if audit_error is not None
                                else {}
                            ),
                        },
                    )
            except Exception:  # noqa: BLE001
                pass
            self._release_active_initializer_scene_lock()
            return self._runtime_recovery_error(
                kind, exc, txid=txid, audited=audit_error is None
            )
        return {
            "status": "ready",
            "txid": txid,
            "live_before": live_before,
            "graph_before": graph_before,
            "artifact_before": artifact_before,
            "pipeline_names_before": names_before,
        }

    def _runtime_transaction_snapshots(
        self,
        txid: int,
        *,
        kind: str,
        ledger_after_request: dict,
        purpose: str = "mutation",
    ) -> tuple[Path, Optional[bytes], dict[Path, tuple[bool, bytes]], list[str]]:
        if purpose not in {"mutation", "undo"}:
            raise ValueError(f"invalid runtime recovery snapshot purpose {purpose!r}")
        self._ensure_undo_base()
        if (
            not self.blender_save
            or not Path(self.blender_save).is_file()
            or Path(self.blender_save).stat().st_size <= 0
        ):
            raise RuntimeError("live Blender save is unavailable for recovery snapshot")
        tx_dir = self._runtime_transaction_dir(txid)
        # This barrier must precede the first marker. Otherwise a durable request
        # record could survive a power loss while the entire tx_N directory dentry
        # disappears, making markerless recovery incorrectly classify a mutation as
        # never started.
        self._durable_mkdir(tx_dir)
        names_before = list(self._pipeline_object_names)
        marker = {
            "schema_version": 1,
            "transaction_id": int(txid),
            "kind": kind,
            "stage": str(self.root_stage_name or ""),
            # Undo snapshots preserve the already-committed state so a process
            # death midway through the rewind can abandon the undo atomically.
            "purpose": purpose,
            "state": ("undo_snapshot_preparing" if purpose == "undo" else "preparing"),
            "created_at_utc": _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat(),
            "ledger_after_request": copy.deepcopy(ledger_after_request),
            "pipeline_object_names_before": names_before,
            "snapshot_manifest": "snapshot_manifest.json",
        }
        if purpose == "mutation":
            self._durable_atomic_write_json(tx_dir / "recovery.json", marker)

        live_before = tx_dir / "before.blend"
        self._durable_atomic_copy(self.blender_save, live_before)
        graph_before = self._read_scene_graph_bytes()
        artifact_before = self._snapshot_runtime_artifacts()
        artifact_rows = []
        scene_root = Path(self.moge_dir)
        for index, (path, (existed, payload)) in enumerate(
            sorted(artifact_before.items(), key=lambda item: str(item[0]))
        ):
            relative = str(path.relative_to(scene_root))
            snapshot_name = f"snapshots/artifact_{index:02d}.snapshot"
            if existed:
                self._durable_atomic_write_bytes(tx_dir / snapshot_name, payload)
            artifact_rows.append(
                {
                    "relative_path": relative,
                    "existed": bool(existed),
                    "snapshot_path": snapshot_name if existed else None,
                    "snapshot_size_bytes": len(payload) if existed else None,
                    "snapshot_sha256": (
                        self._typed_initializer_bytes_sha256(payload)
                        if existed
                        else None
                    ),
                }
            )
        graph_snapshot = None
        if graph_before is not None:
            graph_snapshot = "snapshots/scene_graph.json.snapshot"
            self._durable_atomic_write_bytes(tx_dir / graph_snapshot, graph_before)
        manifest = {
            "schema_version": 1,
            "transaction_id": int(txid),
            "kind": kind,
            "stage": str(self.root_stage_name or ""),
            "purpose": purpose,
            "blend_snapshot": "before.blend",
            "blend_snapshot_size_bytes": live_before.stat().st_size,
            "blend_snapshot_sha256": self._typed_initializer_file_sha256(live_before),
            "scene_graph": {
                "existed": graph_before is not None,
                "snapshot_path": graph_snapshot,
                "snapshot_size_bytes": (
                    len(graph_before) if graph_before is not None else None
                ),
                "snapshot_sha256": (
                    self._typed_initializer_bytes_sha256(graph_before)
                    if graph_before is not None
                    else None
                ),
            },
            "runtime_artifacts": artifact_rows,
            "ledger_after_request": copy.deepcopy(ledger_after_request),
            "pipeline_object_names_before": names_before,
        }
        manifest_path = tx_dir / "snapshot_manifest.json"
        self._durable_atomic_write_json(manifest_path, manifest)
        marker["snapshot_manifest_size_bytes"] = manifest_path.stat().st_size
        marker["snapshot_manifest_sha256"] = self._typed_initializer_file_sha256(
            manifest_path
        )
        if purpose == "undo":
            # The existing committed marker remains authoritative while the undo
            # snapshot is assembled. Only a complete, hash-bound snapshot is exposed
            # to startup recovery, in one atomic marker publication.
            marker["state"] = "undo_prepared"
            marker["updated_at_utc"] = _datetime.datetime.now(
                _datetime.timezone.utc
            ).isoformat()
            self._durable_atomic_write_json(tx_dir / "recovery.json", marker)
        else:
            self._durable_atomic_write_json(tx_dir / "recovery.json", marker)
            self._set_runtime_recovery_state(txid, "prepared")
        return (
            live_before,
            graph_before,
            artifact_before,
            names_before,
        )

    def _load_runtime_recovery_snapshots(
        self, txid: int, marker: dict
    ) -> tuple[Path, Optional[bytes], dict[Path, tuple[bool, bytes]], list[str], dict]:
        tx_dir = self._runtime_transaction_dir(txid)
        manifest_path = tx_dir / "snapshot_manifest.json"
        if (
            marker.get("snapshot_manifest") != "snapshot_manifest.json"
            or not manifest_path.is_file()
            or manifest_path.stat().st_size
            != marker.get("snapshot_manifest_size_bytes")
            or self._typed_initializer_file_sha256(manifest_path)
            != marker.get("snapshot_manifest_sha256")
        ):
            raise RuntimeError(
                f"snapshot manifest integrity check failed for transaction {txid}"
            )
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != 1
            or int(manifest.get("transaction_id") or 0) != int(txid)
            or manifest.get("kind") != marker.get("kind")
            or (
                marker.get("stage") is not None
                and manifest.get("stage") != marker.get("stage")
            )
            or manifest.get("purpose", "mutation") != marker.get("purpose", "mutation")
            or manifest.get("blend_snapshot") != "before.blend"
        ):
            raise RuntimeError(f"snapshot manifest is invalid for transaction {txid}")
        live_before = tx_dir / "before.blend"
        try:
            live_before.resolve().relative_to(tx_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"Blend snapshot escapes transaction {txid}") from exc
        if not live_before.is_file() or live_before.stat().st_size <= 0:
            raise RuntimeError(f"Blend snapshot is missing for transaction {txid}")
        if live_before.stat().st_size != manifest.get(
            "blend_snapshot_size_bytes"
        ) or self._typed_initializer_file_sha256(live_before) != manifest.get(
            "blend_snapshot_sha256"
        ):
            raise RuntimeError(
                f"Blend snapshot integrity check failed for transaction {txid}"
            )

        graph_row = manifest.get("scene_graph") or {}
        graph_before = None
        if graph_row.get("existed") is True:
            graph_path = tx_dir / str(graph_row.get("snapshot_path") or "")
            try:
                graph_path.resolve().relative_to(tx_dir.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    f"scene-graph snapshot escapes transaction {txid}"
                ) from exc
            if not graph_path.is_file():
                raise RuntimeError(
                    f"scene-graph snapshot is missing for transaction {txid}"
                )
            graph_before = graph_path.read_bytes()
            if len(graph_before) != graph_row.get(
                "snapshot_size_bytes"
            ) or self._typed_initializer_bytes_sha256(graph_before) != graph_row.get(
                "snapshot_sha256"
            ):
                raise RuntimeError(
                    f"scene-graph snapshot integrity check failed for transaction {txid}"
                )

        expected_paths = {
            str(path.relative_to(Path(self.moge_dir))): path
            for path in self._runtime_artifact_paths()
        }
        artifact_before: dict[Path, tuple[bool, bytes]] = {}
        seen_paths: set[str] = set()
        for row in manifest.get("runtime_artifacts") or []:
            relative = str(row.get("relative_path") or "")
            if relative not in expected_paths or relative in seen_paths:
                raise RuntimeError(
                    f"runtime artifact snapshot path is invalid for transaction {txid}"
                )
            seen_paths.add(relative)
            existed = row.get("existed") is True
            payload = b""
            if existed:
                snapshot_path = tx_dir / str(row.get("snapshot_path") or "")
                try:
                    snapshot_path.resolve().relative_to(tx_dir.resolve())
                except ValueError as exc:
                    raise RuntimeError(
                        f"runtime artifact snapshot escapes transaction {txid}"
                    ) from exc
                if not snapshot_path.is_file():
                    raise RuntimeError(
                        f"runtime artifact snapshot is missing for transaction {txid}"
                    )
                payload = snapshot_path.read_bytes()
                if len(payload) != row.get(
                    "snapshot_size_bytes"
                ) or self._typed_initializer_bytes_sha256(payload) != row.get(
                    "snapshot_sha256"
                ):
                    raise RuntimeError(
                        "runtime artifact snapshot integrity check failed for "
                        f"transaction {txid}: {relative}"
                    )
            artifact_before[expected_paths[relative]] = (existed, payload)
        if seen_paths != set(expected_paths):
            raise RuntimeError(
                f"runtime artifact snapshot set is incomplete for transaction {txid}"
            )

        names_before = manifest.get("pipeline_object_names_before")
        ledger_after_request = manifest.get("ledger_after_request")
        if not isinstance(names_before, list) or not all(
            isinstance(name, str) for name in names_before
        ):
            raise RuntimeError(
                f"pipeline-name snapshot is invalid for transaction {txid}"
            )
        if not isinstance(ledger_after_request, dict):
            raise RuntimeError(f"ledger snapshot is invalid for transaction {txid}")
        return (
            live_before,
            graph_before,
            artifact_before,
            names_before,
            ledger_after_request,
        )

    def _last_mutation_journal_event(self, txid: int, kind: str) -> Optional[dict]:
        last = None
        for row in self._initializer_mutation_journal_events():
            if int(row["transaction_id"]) == int(txid) and row.get("kind") == kind:
                last = row
        return last

    def _initializer_mutation_journal_events(self) -> list[dict]:
        """Load durable GPT-6 initializer events with strict JSON validation."""
        events: list[dict] = []
        for line_number, row in enumerate(
            read_mutation_journal(self._mutation_journal_path()), start=1
        ):
            if (
                row.get("stage") != "initializer"
                or row.get("harness_profile") != "gpt6_v1"
            ):
                continue
            raw_txid = row.get("transaction_id")
            if isinstance(raw_txid, bool):
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                )
            try:
                txid = int(raw_txid)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                ) from exc
            if txid < 1 or str(raw_txid) != str(txid):
                raise RuntimeError(
                    f"mutation journal line {line_number} has an invalid transaction id"
                )
            events.append({**row, "transaction_id": txid})
        return events

    def _initializer_recovery_markers(self) -> list[tuple[Path, dict]]:
        """Select initializer markers from the scene-wide transaction namespace."""
        root = Path(self.moge_dir) / "runtime_objects" / "transactions"
        markers = []
        for path in root.glob("tx_*/recovery.json"):
            try:
                marker = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"cannot read initializer recovery marker {path}"
                ) from exc
            txid = marker.get("transaction_id") if isinstance(marker, dict) else None
            if (
                not isinstance(marker, dict)
                or marker.get("schema_version") != 1
                or type(txid) is not int
                or txid < 1
                or path.parent != self._runtime_transaction_dir(txid)
            ):
                raise RuntimeError(f"invalid initializer recovery marker {path}")
            stage = marker.get("stage", "initializer")
            kind = marker.get("kind")
            if stage == "composition" and kind == _COMPOSITION_MESH_MUTATION_KIND:
                continue
            if stage != "initializer" or kind not in _TYPED_INITIALIZER_MUTATION_KINDS:
                raise RuntimeError(f"invalid initializer recovery marker {path}")
            markers.append((path, marker))
        return sorted(markers, key=lambda pair: pair[1]["transaction_id"])

    def _assert_no_incomplete_runtime_mutation(self) -> None:
        """Prevent a later edit from crossing an unresolved recovery boundary."""
        if not self._typed_initializer_recovery_enabled() or not self.moge_dir:
            return
        terminal_states = {"committed", "rolled_back", "undone"}
        marker_keys: set[tuple[int, str]] = set()
        for _path, marker in self._initializer_recovery_markers():
            txid, kind = marker["transaction_id"], marker["kind"]
            marker_keys.add((txid, kind))
            if str(marker.get("state") or "") not in terminal_states:
                raise RuntimeError(
                    f"runtime transaction {txid} still requires recovery; "
                    "restart the initializer before making another scene edit"
                )

        grouped: dict[tuple[int, str], list[dict]] = {}
        for event in self._initializer_mutation_journal_events():
            kind = str(event.get("kind") or "")
            if kind in _TYPED_INITIALIZER_MUTATION_KINDS:
                grouped.setdefault((int(event["transaction_id"]), kind), []).append(
                    event
                )
        for key, events in grouped.items():
            txid, _kind = key
            if key in marker_keys:
                continue
            durable_statuses = {str(event.get("status") or "") for event in events}
            if durable_statuses.intersection({"committed", "undone"}):
                raise RuntimeError(
                    f"runtime transaction {txid} contains a committed/undone event "
                    "without its exact recovery marker; restart the initializer to "
                    "reconcile it"
                )
            if str(events[-1].get("status") or "") == "requested":
                raise RuntimeError(
                    f"runtime transaction {txid} has an unterminated request without "
                    "its exact recovery marker; restart the initializer to reconcile it"
                )

    def _reconcile_markerless_runtime_requests(
        self,
        journal_events: list[dict],
        marker_keys: set[tuple[int, str]],
    ) -> None:
        """Close a crash window between the request audit and marker publication."""
        grouped: dict[tuple[int, str], list[dict]] = {}
        for event in journal_events:
            kind = str(event.get("kind") or "")
            if kind not in _TYPED_INITIALIZER_MUTATION_KINDS:
                continue
            key = (int(event["transaction_id"]), kind)
            grouped.setdefault(key, []).append(event)

        for (txid, kind), events in sorted(grouped.items()):
            if (txid, kind) in marker_keys:
                continue
            last = events[-1]
            status = str(last.get("status") or "")
            durable_statuses = {str(event.get("status") or "") for event in events}
            if durable_statuses.intersection({"committed", "undone"}):
                raise RuntimeError(
                    f"runtime transaction {txid}/{kind} contains a committed/undone "
                    "event but its exact recovery marker is missing"
                )
            for event in events:
                self._reconcile_recovered_terminal_event(event)
            if status != "requested":
                continue
            artifact_manifest = self._safe_typed_initializer_artifact_manifest(
                txid=txid,
                kind=kind,
                state="error",
            )
            terminal_event = {
                "transaction_id": txid,
                "kind": kind,
                "status": "error",
                "details": {
                    "phase": "startup_request_recovery",
                    "scene_mutation": "not_started",
                    "artifact_manifest": artifact_manifest,
                },
            }
            self._append_mutation_event(terminal_event)
            self._reconcile_recovered_terminal_event(terminal_event)

    def _reconcile_recovered_terminal_event(self, event: dict) -> None:
        txid = int(event["transaction_id"])
        status = str(event.get("status") or "")
        ledger_event = {
            "id": txid,
            **{
                key: copy.deepcopy(value)
                for key, value in event.items()
                if key
                not in {
                    "transaction_id",
                    "schema_version",
                    "timestamp_utc",
                    "harness_profile",
                    "stage",
                    "attempt_idx",
                }
            },
        }
        same_status = [
            row
            for row in self._initializer_ledger.get("events", [])
            if isinstance(row, dict)
            and int(row.get("id") or -1) == txid
            and row.get("status") == status
        ]
        exact = [row for row in same_status if row == ledger_event]
        if len(same_status) > 1:
            raise RuntimeError(
                f"transaction {txid} has duplicate/conflicting {status} ledger events"
            )
        if status == "undone" and same_status:
            if exact:
                return
            raise RuntimeError(
                f"transaction {txid} has a conflicting undone ledger event"
            )
        if exact:
            return
        ledger_before = copy.deepcopy(self._initializer_ledger)
        self._initializer_ledger["events"].append(ledger_event)
        self._initializer_ledger["next_transaction_id"] = max(
            int(self._initializer_ledger.get("next_transaction_id") or 1), txid + 1
        )
        try:
            self._persist_initializer_ledger()
        except Exception:
            # A durable journal event remains authoritative. Keep memory aligned
            # with the last durable ledger so retry/startup can reconcile it again.
            self._initializer_ledger = ledger_before
            raise

    def _validate_terminal_runtime_transaction(
        self, *, txid: int, kind: str, state: str, marker: dict
    ) -> None:
        """Fail closed unless a terminal marker agrees with journal and ledger."""
        events = [
            event
            for event in self._initializer_mutation_journal_events()
            if int(event["transaction_id"]) == txid and event.get("kind") == kind
        ]
        matching = [event for event in events if event.get("status") == state]
        if not matching:
            raise RuntimeError(
                f"transaction {txid} terminal {state} marker has no matching journal event"
            )
        if state in {"rolled_back", "undone"} and len(matching) != 1:
            raise RuntimeError(
                f"transaction {txid} has duplicate/conflicting {state} journal events"
            )
        if state == "committed" and any(
            event.get("status") == "undone" for event in events
        ):
            raise RuntimeError(
                f"transaction {txid} has a committed marker after an undone event"
            )
        if state == "committed":
            committed_initializer_event(
                txid,
                kind,
                (marker.get("terminal_details") or {}).get("artifact_manifest"),
                self._initializer_ledger.get("events", []),
                events,
            )
        authoritative = matching[-1]
        ledger_event = {
            "id": txid,
            **{
                key: copy.deepcopy(value)
                for key, value in authoritative.items()
                if key
                not in {
                    "transaction_id",
                    "schema_version",
                    "timestamp_utc",
                    "harness_profile",
                    "stage",
                    "attempt_idx",
                }
            },
        }
        exact_ledger = [
            row
            for row in self._initializer_ledger.get("events", [])
            if isinstance(row, dict) and row == ledger_event
        ]
        same_status = [
            row
            for row in self._initializer_ledger.get("events", [])
            if isinstance(row, dict)
            and int(row.get("id") or -1) == txid
            and row.get("status") == state
        ]
        if len(exact_ledger) != 1:
            raise RuntimeError(
                f"transaction {txid} terminal {state} journal event has no unique "
                "ledger peer"
            )
        if state in {"rolled_back", "undone"} and len(same_status) != 1:
            raise RuntimeError(
                f"transaction {txid} has duplicate/conflicting {state} ledger events"
            )

        active_rows = [
            row
            for row in self._initializer_ledger.get("active_transactions", [])
            if isinstance(row, dict) and int(row.get("id") or -1) == txid
        ]
        if state == "committed" and len(active_rows) != 1:
            raise RuntimeError(
                f"transaction {txid} committed marker has no unique active ledger row"
            )
        if state != "committed" and active_rows:
            raise RuntimeError(
                f"transaction {txid} terminal {state} marker remains active in ledger"
            )

        journal_manifest = (authoritative.get("details") or {}).get("artifact_manifest")
        marker_manifest = (marker.get("terminal_details") or {}).get(
            "artifact_manifest"
        )
        if journal_manifest is None or marker_manifest != journal_manifest:
            raise RuntimeError(
                f"transaction {txid} terminal {state} artifact manifests differ"
            )
        if (
            state == "committed"
            and active_rows[0].get("artifact_manifest") != journal_manifest
        ):
            raise RuntimeError(
                f"transaction {txid} committed active/journal manifests differ"
            )

    def _recover_incomplete_runtime_mutations(self) -> None:
        """Recover GPT-6 typed mutations interrupted by an MCP/process crash."""
        if not self._typed_initializer_recovery_enabled() or not self.moge_dir:
            return
        lock_fd = self._acquire_initializer_scene_lock()
        try:
            ledger_path = getattr(self, "initializer_ledger_path", None)
            if ledger_path and Path(ledger_path).is_file():
                self._initializer_ledger = self._load_initializer_ledger()
            self._recover_incomplete_runtime_mutations_locked()
        finally:
            self._release_initializer_scene_lock(lock_fd)

    def _recover_incomplete_runtime_mutations_locked(self) -> None:
        """Recovery implementation; caller holds the per-scene initializer lock."""
        journal_events = self._initializer_mutation_journal_events()
        selected = self._initializer_recovery_markers()
        markers = [path for path, _marker in selected]
        marker_keys = {
            (marker["transaction_id"], marker["kind"]) for _path, marker in selected
        }
        self._reconcile_markerless_runtime_requests(journal_events, marker_keys)
        if not markers:
            return
        terminal_states = {"committed", "rolled_back", "undone"}
        for marker_path in markers:
            txid = None
            kind = "unknown"
            state = ""
            last_event = None
            try:
                marker = json.loads(marker_path.read_text())
                txid = int(marker.get("transaction_id") or 0)
                kind = str(marker.get("kind") or "")
                state = str(marker.get("state") or "")
                if marker.get("schema_version") != 1 or txid < 1 or not kind:
                    raise RuntimeError(f"invalid recovery marker: {marker_path}")
                if marker_path.parent != self._runtime_transaction_dir(txid):
                    raise RuntimeError(f"misbound recovery marker: {marker_path}")
                if state in terminal_states:
                    self._validate_terminal_runtime_transaction(
                        txid=txid, kind=kind, state=state, marker=marker
                    )
                    tx_dir = self._runtime_transaction_dir(txid)
                    if (
                        marker.get("payloads_pruned") is not True
                        or (tx_dir / "before.blend").exists()
                        or (tx_dir / "snapshots").exists()
                    ):
                        self._prune_runtime_recovery_payload(txid)
                    continue

                newer_ids = sorted(
                    {
                        int(event["transaction_id"])
                        for event in journal_events
                        if int(event["transaction_id"]) > txid
                    }
                )
                if newer_ids:
                    raise RuntimeError(
                        f"transaction {txid} cannot be recovered across newer "
                        f"initializer transaction(s): {newer_ids}"
                    )

                last_event = self._last_mutation_journal_event(txid, kind)
                active_rows = [
                    row
                    for row in self._initializer_ledger.get("active_transactions", [])
                    if isinstance(row, dict) and int(row.get("id") or -1) == txid
                ]
                active = bool(active_rows)
                if state == "undo_commit_decided":
                    decision = marker.get("terminal_details")
                    if not isinstance(decision, dict):
                        raise RuntimeError(
                            f"transaction {txid} has no durable undo decision payload"
                        )
                    undo_event = decision.get("undo_event")
                    post_undo_ledger = decision.get("post_undo_ledger")
                    post_undo_names = decision.get("post_undo_pipeline_object_names")
                    decision_digest = decision.get("decision_sha256")
                    decision_body = {
                        key: copy.deepcopy(value)
                        for key, value in decision.items()
                        if key != "decision_sha256"
                    }
                    if (
                        not isinstance(decision_digest, str)
                        or decision_digest
                        != self._typed_initializer_json_sha256(decision_body)
                        or not isinstance(undo_event, dict)
                        or int(undo_event.get("transaction_id") or 0) != txid
                        or undo_event.get("kind") != kind
                        or undo_event.get("status") != "undone"
                        or undo_event.get("undo_operation_id")
                        != f"initializer-tx-{txid}-undo"
                        or not isinstance(undo_event.get("details"), dict)
                        or not isinstance(post_undo_ledger, dict)
                        or post_undo_ledger.get("schema_version") != 1
                        or not isinstance(post_undo_ledger.get("events"), list)
                        or not isinstance(
                            post_undo_ledger.get("active_transactions"), list
                        )
                        or not isinstance(
                            post_undo_ledger.get("runtime_root_surfaces"), list
                        )
                        or not isinstance(
                            post_undo_ledger.get("next_transaction_id"), int
                        )
                        or int(post_undo_ledger["next_transaction_id"]) <= txid
                        or not isinstance(post_undo_names, list)
                        or not all(isinstance(name, str) for name in post_undo_names)
                    ):
                        raise RuntimeError(
                            f"transaction {txid} durable undo decision is invalid"
                        )
                    if any(
                        isinstance(row, dict) and int(row.get("id") or -1) == txid
                        for row in post_undo_ledger["active_transactions"]
                    ):
                        raise RuntimeError(
                            f"transaction {txid} post-undo ledger remains active"
                        )
                    journal_manifest = undo_event["details"].get("artifact_manifest")
                    if journal_manifest != decision.get("artifact_manifest"):
                        raise RuntimeError(
                            f"transaction {txid} durable undo manifests differ"
                        )
                    artifact_manifest = (
                        self._verify_typed_initializer_artifact_manifest(
                            journal_manifest,
                            txid=txid,
                            kind=kind,
                            state="undone",
                        )
                    )
                    matching_undone = [
                        event
                        for event in self._initializer_mutation_journal_events()
                        if int(event["transaction_id"]) == txid
                        and event.get("kind") == kind
                        and event.get("status") == "undone"
                    ]
                    if len(matching_undone) > 1:
                        raise RuntimeError(
                            f"transaction {txid} has duplicate durable undone events"
                        )
                    if any(
                        event.get("undo_operation_id")
                        != undo_event.get("undo_operation_id")
                        or (event.get("details") or {})
                        != (undo_event.get("details") or {})
                        for event in matching_undone
                    ):
                        raise RuntimeError(
                            f"transaction {txid} has a conflicting durable undone event"
                        )

                    placement = self._typed_initializer_json_file(
                        Path(self.moge_dir) / "placement.json"
                    )
                    current_names = [
                        str(row.get("mesh_name"))
                        for row in placement.get("objects", [])
                        if isinstance(row, dict) and row.get("mesh_name")
                    ]
                    if (
                        len(current_names) != len(set(current_names))
                        or current_names != post_undo_names
                    ):
                        raise RuntimeError(
                            f"transaction {txid} durable undo names do not match "
                            "current placement"
                        )

                    ledger_event = {
                        "id": txid,
                        **{
                            key: copy.deepcopy(value)
                            for key, value in undo_event.items()
                            if key != "transaction_id"
                        },
                    }
                    final_ledger = copy.deepcopy(post_undo_ledger)
                    final_ledger["events"].append(ledger_event)
                    final_ledger["next_transaction_id"] = max(
                        int(final_ledger.get("next_transaction_id") or 1), txid + 1
                    )
                    if not (
                        self._initializer_ledger == post_undo_ledger
                        or self._initializer_ledger == final_ledger
                    ):
                        raise RuntimeError(
                            f"transaction {txid} current ledger does not match either "
                            "durable undo decision state"
                        )

                    self._pipeline_object_names = list(post_undo_names)
                    self._n2id_cache = None
                    if not matching_undone:
                        self._append_mutation_event(undo_event)
                    self._reconcile_recovered_terminal_event(undo_event)
                    self._set_runtime_recovery_state(
                        txid,
                        "undone",
                        details={
                            "startup_rollforward": True,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    continue
                if state == "undo_prepared":
                    if last_event is not None and last_event.get("status") == "undone":
                        journal_manifest = (last_event.get("details") or {}).get(
                            "artifact_manifest"
                        )
                        undone_ledger_rows = [
                            row
                            for row in self._initializer_ledger.get("events", [])
                            if isinstance(row, dict)
                            and int(row.get("id") or -1) == txid
                            and row.get("status") == "undone"
                        ]
                        if len(undone_ledger_rows) > 1:
                            raise RuntimeError(
                                f"transaction {txid} has duplicate undone ledger records"
                            )
                        if (
                            undone_ledger_rows
                            and (undone_ledger_rows[0].get("details") or {}).get(
                                "artifact_manifest"
                            )
                            != journal_manifest
                        ):
                            raise RuntimeError(
                                f"transaction {txid} ledger/journal undo manifests differ"
                            )
                        artifact_manifest = (
                            self._verify_typed_initializer_artifact_manifest(
                                journal_manifest,
                                txid=txid,
                                kind=kind,
                                state="undone",
                            )
                        )
                        self._initializer_ledger["active_transactions"] = [
                            row
                            for row in self._initializer_ledger.get(
                                "active_transactions", []
                            )
                            if not (
                                isinstance(row, dict)
                                and int(row.get("id") or -1) == txid
                            )
                        ]
                        self._persist_initializer_ledger()
                        self._reconcile_recovered_terminal_event(last_event)
                        self._set_runtime_recovery_state(
                            txid,
                            "undone",
                            details={
                                "startup_reconciled": True,
                                "artifact_manifest": artifact_manifest,
                            },
                        )
                        continue

                    # Without a durable `undone` journal event, the rewind never
                    # committed. Restore its last committed scene snapshot. Keep
                    # transaction-owned mesh/mask files: they belong to that state.
                    (
                        live_before,
                        graph_before,
                        artifact_before,
                        names_before,
                        committed_ledger,
                    ) = self._load_runtime_recovery_snapshots(txid, marker)
                    restore_errors: list[str] = []
                    try:
                        self._atomic_restore_blend(
                            live_before,
                            tag="runtime-undo-startup-rollback",
                            durable=True,
                        )
                    except Exception as restore_exc:  # noqa: BLE001
                        restore_errors.append(f"Blend restore failed: {restore_exc}")
                    try:
                        self._restore_runtime_artifacts(artifact_before, durable=True)
                    except Exception as restore_exc:  # noqa: BLE001
                        restore_errors.append(f"artifact restore failed: {restore_exc}")
                    try:
                        self._restore_scene_graph_bytes(graph_before, durable=True)
                    except Exception as restore_exc:  # noqa: BLE001
                        restore_errors.append(f"graph restore failed: {restore_exc}")
                    self._pipeline_object_names = list(names_before)
                    self._n2id_cache = None
                    self._initializer_ledger = copy.deepcopy(committed_ledger)
                    try:
                        self._persist_initializer_ledger()
                    except Exception as restore_exc:  # noqa: BLE001
                        restore_errors.append(f"ledger restore failed: {restore_exc}")
                    if restore_errors:
                        raise RuntimeError(
                            f"transaction {txid} interrupted undo restore failed: "
                            + "; ".join(restore_errors)
                        )

                    committed_rows = [
                        row
                        for row in self._initializer_ledger.get(
                            "active_transactions", []
                        )
                        if isinstance(row, dict) and int(row.get("id") or -1) == txid
                    ]
                    if len(committed_rows) != 1:
                        raise RuntimeError(
                            f"transaction {txid} has no unique committed undo baseline"
                        )
                    if last_event is not None and last_event.get("status") == "error":
                        self._reconcile_recovered_terminal_event(last_event)
                    committed_event = committed_initializer_event(
                        txid,
                        kind,
                        committed_rows[0].get("artifact_manifest"),
                        self._initializer_ledger.get("events", []),
                        journal_events,
                    )
                    journal_manifest = committed_event["details"]["artifact_manifest"]
                    artifact_manifest = (
                        self._verify_typed_initializer_artifact_manifest(
                            journal_manifest,
                            txid=txid,
                            kind=kind,
                            state="committed",
                        )
                    )
                    # The existing commit and its complete authored provenance
                    # remain authoritative; abandoning an undo adds no new commit.
                    self._set_runtime_recovery_state(
                        txid,
                        "committed",
                        details={
                            "startup_recovered_undo": True,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    continue
                if (
                    last_event is not None
                    and last_event.get("status") == "committed"
                    and active
                ):
                    if len(active_rows) != 1:
                        raise RuntimeError(
                            f"transaction {txid} has duplicate active ledger records"
                        )
                    journal_manifest = (last_event.get("details") or {}).get(
                        "artifact_manifest"
                    )
                    ledger_manifest = active_rows[0].get("artifact_manifest")
                    if journal_manifest != ledger_manifest:
                        raise RuntimeError(
                            f"transaction {txid} ledger/journal artifact manifests differ"
                        )
                    artifact_manifest = (
                        self._verify_typed_initializer_artifact_manifest(
                            journal_manifest,
                            txid=txid,
                            kind=kind,
                            state="committed",
                        )
                    )
                    self._reconcile_recovered_terminal_event(last_event)
                    self._set_runtime_recovery_state(
                        txid,
                        "committed",
                        details={
                            "startup_reconciled": True,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    continue
                if (
                    last_event is not None
                    and last_event.get("status") == "rolled_back"
                    and not active
                ):
                    self._reconcile_recovered_terminal_event(last_event)
                    artifact_manifest = (last_event.get("details") or {}).get(
                        "artifact_manifest"
                    ) or self._safe_typed_initializer_artifact_manifest(
                        txid=txid,
                        kind=kind,
                        state="rolled_back",
                    )
                    self._set_runtime_recovery_state(
                        txid,
                        "rolled_back",
                        details={
                            "startup_reconciled": True,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    continue

                if state == "preparing":
                    ledger = marker.get("ledger_after_request")
                    names = marker.get("pipeline_object_names_before")
                    if not isinstance(ledger, dict) or not isinstance(names, list):
                        raise RuntimeError(
                            f"preparing recovery marker lacks state for transaction {txid}"
                        )
                    self._initializer_ledger = copy.deepcopy(ledger)
                    self._pipeline_object_names = [str(name) for name in names]
                    self._n2id_cache = None
                    self._persist_initializer_ledger()
                    restore_errors: list[str] = []
                else:
                    (
                        live_before,
                        graph_before,
                        artifact_before,
                        names_before,
                        ledger_after_request,
                    ) = self._load_runtime_recovery_snapshots(txid, marker)
                    restore_errors = self._restore_failed_runtime_mutation(
                        txid=txid,
                        live_before=live_before,
                        graph_before=graph_before,
                        artifact_before=artifact_before,
                        pipeline_names_before=names_before,
                        ledger_after_request=ledger_after_request,
                    )
                if restore_errors:
                    artifact_manifest = self._safe_typed_initializer_artifact_manifest(
                        txid=txid,
                        kind=kind,
                        state="error",
                    )
                    self._record_mutation_status(
                        txid,
                        kind,
                        "error",
                        details={
                            "phase": "startup_recovery",
                            "restore_errors": restore_errors,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    self._set_runtime_recovery_state(
                        txid,
                        "recovery_needed",
                        details={
                            "restore_errors": restore_errors,
                            "artifact_manifest": artifact_manifest,
                        },
                    )
                    raise RuntimeError(
                        f"runtime transaction {txid} recovery failed: "
                        + "; ".join(restore_errors)
                    )
                artifact_manifest = self._safe_typed_initializer_artifact_manifest(
                    txid=txid,
                    kind=kind,
                    state="rolled_back",
                )
                self._record_mutation_status(
                    txid,
                    kind,
                    "rolled_back",
                    details={
                        "phase": "startup_recovery",
                        "interrupted_state": state,
                        "scene_mutation": "restored",
                        "artifact_manifest": artifact_manifest,
                    },
                )
                self._set_runtime_recovery_state(
                    txid,
                    "rolled_back",
                    details={
                        "startup_recovered": True,
                        "interrupted_state": state,
                        "artifact_manifest": artifact_manifest,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                if (
                    txid is not None
                    and txid > 0
                    and marker_path.parent == self._runtime_transaction_dir(txid)
                ):
                    try:
                        # A durable undo decision is irreversible. Preserve both
                        # its state and embedded roll-forward payload when current
                        # U-state verification or bookkeeping repair fails. The
                        # next startup must retry U; it must never reinterpret this
                        # as a generic recovery state and restore committed state C.
                        durable_undo = state == "undo_commit_decided" or (
                            state == "undo_prepared"
                            and last_event is not None
                            and last_event.get("status") == "undone"
                        )
                        if durable_undo:
                            current_marker = json.loads(marker_path.read_text())
                            current_marker["recovery_error"] = {
                                "phase": "startup_recovery",
                                "error": str(exc),
                            }
                            current_marker["updated_at_utc"] = _datetime.datetime.now(
                                _datetime.timezone.utc
                            ).isoformat()
                            self._durable_atomic_write_json(marker_path, current_marker)
                        else:
                            artifact_manifest = (
                                self._safe_typed_initializer_artifact_manifest(
                                    txid=txid,
                                    kind=kind,
                                    state="error",
                                )
                            )
                            self._set_runtime_recovery_state(
                                txid,
                                "recovery_needed",
                                details={
                                    "phase": "startup_recovery",
                                    "error": str(exc),
                                    "artifact_manifest": artifact_manifest,
                                },
                            )
                    except Exception:  # noqa: BLE001
                        pass
                raise RuntimeError(
                    f"cannot recover initializer transaction at {marker_path}"
                ) from exc

    def _update_blend_base_entry(self, mesh_name: str, matrix: list) -> None:
        path = Path(self.moge_dir) / "physics" / "blend_base.json"
        try:
            payload = json.loads(path.read_text()) if path.is_file() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload[mesh_name] = matrix
        self._atomic_write_json(path, payload)

    def _remove_physics_identity(self, mesh_name: str, *, transaction_id: int) -> dict:
        """Withdraw one runtime identity from every downstream physics input.

        Removal also invalidates whole-scene certification: those measurements were
        made with a different identity set and cannot remain authoritative.
        """
        scene = Path(self.moge_dir)
        estimates_path = scene / "physics" / "physics_vlm.json"
        manifest_path = scene / "physics" / "physics_estimate_manifest.json"
        base_path = scene / "physics" / "blend_base.json"
        pose_path = scene / "physics" / "pose_changes.json"
        change = {
            "transaction_id": int(transaction_id),
            "mesh_name": mesh_name,
            "removed_records": {},
            "invalidated_late_blocks": [],
        }

        def remember(label: str, value: object) -> None:
            if value is not None:
                change["removed_records"][label] = {
                    "value": copy.deepcopy(value),
                    "sha256": self._typed_initializer_json_sha256(value),
                }

        if estimates_path.is_file():
            estimates = json.loads(estimates_path.read_text())
            if not isinstance(estimates, dict):
                raise RuntimeError(f"physics estimates at {estimates_path} are invalid")
            remember("physics_vlm", estimates.pop(mesh_name, None))
            self._atomic_write_json(estimates_path, estimates)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise RuntimeError(f"physics manifest at {manifest_path} is invalid")
            records = manifest.get("objects") or {}
            if not isinstance(records, dict):
                raise RuntimeError(f"physics manifest at {manifest_path} is invalid")
            remember("physics_estimate_manifest", records.pop(mesh_name, None))
            manifest["expected_count"] = len(records)
            manifest["estimated_count"] = sum(
                record.get("status") in {"estimated", "cached"}
                for record in records.values()
                if isinstance(record, dict)
            )
            manifest["fallback_count"] = len(records) - manifest["estimated_count"]
            manifest["coverage_status"] = (
                "full" if manifest["fallback_count"] == 0 else "partial"
            )
            self._atomic_write_json(manifest_path, manifest)
        if base_path.is_file():
            base = json.loads(base_path.read_text())
            if not isinstance(base, dict):
                raise RuntimeError(f"Blend base at {base_path} is invalid")
            remember("blend_base", base.pop(mesh_name, None))
            self._atomic_write_json(base_path, base)
        if pose_path.is_file():
            pose = json.loads(pose_path.read_text())
            objects = pose.get("objects") if isinstance(pose, dict) else None
            if not isinstance(objects, dict):
                raise RuntimeError(f"physics pose state at {pose_path} is invalid")
            remember("pose_changes", objects.pop(mesh_name, None))
            for key in (
                "composition_certify",
                "composition_certify_converged",
                "composition_certify_coverage",
                "actual_surface_validation",
                "delivered",
                "certify_run_id",
            ):
                if key in pose:
                    pose.pop(key)
                    change["invalidated_late_blocks"].append(key)
            history = pose.get("runtime_object_removals") or []
            if not isinstance(history, list):
                raise RuntimeError(
                    f"physics pose state at {pose_path} has invalid removal history"
                )
            change["semantic_sha256"] = self._typed_initializer_json_sha256(change)
            history.append(copy.deepcopy(change))
            pose["runtime_object_removals"] = history
            self._atomic_write_json(pose_path, pose)
        if "semantic_sha256" not in change:
            change["semantic_sha256"] = self._typed_initializer_json_sha256(change)
        return change

    def _invalidate_runtime_mesh_pose_state(
        self,
        mesh_name: str,
        *,
        transaction_id: int,
        mesh_sha256: str,
    ) -> dict:
        """Reset preprocess physics state that cannot survive a mesh revision.

        ``pose_changes.objects[mesh_name]`` selects a preprocess collider rung and
        carries its stabilization bundle into every later composition boot.  Neither
        that selection nor any of its measurements/overrides describe replacement
        geometry.  Keep only a new identity settlement baseline (the replacement is
        imported at the current placed pose); the runtime settle is subsequently
        represented by the live Blend delta from ``blend_base.json``.

        A mesh change also invalidates the whole-scene late certification blocks.
        Other objects' preprocess records and the file's run/mode identity remain
        valid and are preserved.  The compact history records names and a digest, not
        stale values that a downstream consumer could accidentally reactivate.
        """
        import hashlib

        path = Path(self.moge_dir) / "physics" / "pose_changes.json"
        if not path.is_file():
            return {
                "pose_changes_present": False,
                "mesh_name": mesh_name,
                "transaction_id": int(transaction_id),
            }
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read physics pose state at {path}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"physics pose state at {path} must be a JSON object")
        objects = payload.get("objects")
        if not isinstance(objects, dict):
            raise RuntimeError(
                f"physics pose state at {path} has no valid objects mapping"
            )

        previous = objects.get(mesh_name)
        previous_fields = (
            sorted(str(key) for key in previous)
            if isinstance(previous, dict)
            else ([] if previous is None else ["<invalid-record>"])
        )
        previous_sha256 = None
        if previous is not None:
            encoded = json.dumps(
                previous,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            previous_sha256 = hashlib.sha256(encoded).hexdigest()

        objects[mesh_name] = {
            "settle_total": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "runtime_mesh_revision": {
                "transaction_id": int(transaction_id),
                "mesh_sha256": str(mesh_sha256),
                "collider_state": "fresh_rebuild_required",
                "physics_override_state": "invalidated",
            },
        }

        invalidated_blocks = []
        for key in (
            "composition_certify",
            "composition_certify_converged",
            "composition_certify_coverage",
            "actual_surface_validation",
            "delivered",
            "certify_run_id",
        ):
            if key in payload:
                payload.pop(key)
                invalidated_blocks.append(key)

        history = payload.get("runtime_mesh_invalidations")
        if history is None:
            history = []
        if not isinstance(history, list):
            raise RuntimeError(
                f"physics pose state at {path} has invalid runtime mesh history"
            )
        event = {
            "transaction_id": int(transaction_id),
            "mesh_name": mesh_name,
            "replacement_mesh_sha256": str(mesh_sha256),
            "invalidated_object_fields": previous_fields,
            "invalidated_late_blocks": invalidated_blocks,
            "superseded_record_sha256": previous_sha256,
        }
        history = [
            item
            for item in history
            if not (
                isinstance(item, dict)
                and item.get("transaction_id") == int(transaction_id)
                and item.get("mesh_name") == mesh_name
            )
        ]
        history.append(event)
        payload["runtime_mesh_invalidations"] = history
        self._atomic_write_json(path, payload)
        return {"pose_changes_present": True, **event}

    def execute_initializer_transaction(
        self,
        *,
        code: str,
        added_objects: Optional[list[dict]] = None,
        removed_objects: Optional[list[str]] = None,
        reason: str = "",
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
    ) -> dict:
        """Execute one declared initializer edit and bake its joint physical outcome.

        All inventory edits and freely authored whole-object poses share one
        recovery snapshot and one undo step. Surface edits remain in the same
        Blender script, under the existing root-registration checks. An explicit
        feedback view must exist before mutation; its paired render is generated
        only after the transaction commits. Omitted angles retain the source view.
        """
        self._require_initializer_capability(
            "initializer_code_transactions", "execute_and_evaluate"
        )
        try:
            render_view = (
                0.0 if azimuth is None else float(azimuth),
                0.0 if elevation is None else float(elevation),
            )
        except (TypeError, ValueError) as exc:
            return {
                "status": "error",
                "output": {
                    "text": [f"Invalid initializer feedback view: {exc}"],
                    "scene_mutation": "not_started",
                },
            }
        if azimuth is not None or elevation is not None:
            views = self._pseudo_gt_views()
            if render_view not in views:
                options = ", ".join(f"({a:g}, {e:g})" for a, e in sorted(views))
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"No pseudo-GT camera for (azimuth={render_view[0]}, "
                            f"elevation={render_view[1]}). Choose an (azimuth, "
                            f"elevation) from: {options or '(none built)'}. "
                            "The scene was not changed."
                        ],
                        "scene_mutation": "not_started",
                    },
                }
        kind = "execute_and_evaluate_objects"
        request = {
            "code": code,
            "added_objects": added_objects or [],
            "removed_objects": removed_objects or [],
            "reason": reason,
            "azimuth": render_view[0],
            "elevation": render_view[1],
        }
        prepared = self._prepare_typed_initializer_mutation(kind, request)
        if prepared.get("status") != "ready":
            return prepared
        txid = int(prepared["txid"])
        try:
            from lib.tools.blender.initializer_transaction import (
                normalize_object_declarations,
                observed_object_changes,
            )
            from lib.tools.blender.procedural_object import (
                build_procedural_placement_row,
            )
            from lib.tools.geometry.runtime_object_repair import (
                build_runtime_object_record,
                stage_runtime_object_batch,
            )

            graph, placement, inventory = self._load_runtime_scene_state()
            declarations = normalize_object_declarations(
                added_objects, removed_objects, graph=graph, placement=placement
            )
            validate_object_addition_scope(
                graph=graph,
                added_objects=declarations["added_objects"],
                removed_objects=declarations["removed_objects"],
            )
            pre = self._read_penetration_data(
                blend_path=str(prepared["live_before"]),
                tag=f"initializer_code_{txid}_before",
            )
            self._last_code = self._parse_code(code)
            authored = self._run_initializer_authored_code(
                code=code,
                kind=kind,
                transaction_id=txid,
                object_id="",
                mesh_name="",
                live_before=prepared["live_before"],
                declarations=declarations,
            )
            old_nodes = {node["id"]: node for node in graph["nodes"]}
            old_rows = {row["mesh_name"]: row for row in placement["objects"]}
            final_names = sorted(
                (set(old_rows) - set(declarations["removed_names"].values()))
                | set(declarations["added_names"].values())
            )
            for name in declarations["removed_names"].values():
                self._remove_physics_identity(name, transaction_id=txid)
            records = []
            part_captures = []
            staged = []
            for addition in declarations["added_objects"]:
                object_id, name = addition["object_id"], addition["mesh_name"]
                mesh_path, capture = self._export_initializer_authored_object(
                    mesh_name=name, object_id=object_id, transaction_id=txid
                )
                part_captures.append((object_id, capture))
                row = build_procedural_placement_row(
                    category=addition["category"],
                    instance=addition["instance"],
                    support=addition["support"],
                    mask=None,
                    capture=capture,
                    mesh_glb=mesh_path,
                    old_row=old_rows.get(name),
                    transaction_id=txid,
                )
                row.update(
                    description=addition["description"],
                    mask_path=None,
                    runtime_added=True,
                    runtime_transaction_id=txid,
                )
                staged.append((addition, object_id, name, mesh_path, capture, row))
            # ONE physics estimate pass for every object authored in this call (2026-09-15
            # owner): exports first, then a single render pass + concurrent VLM calls.
            physicals = (
                self._estimate_runtime_physics_materials(
                    [item[5] for item in staged],
                    hints={item[2]: item[0].get("physical_material_hint") for item in staged},
                    transaction_id=txid,
                    force_refresh=True,
                    final_expected_names=final_names,
                )
                if staged
                else {}
            )
            for addition, object_id, name, mesh_path, capture, row in staged:
                physical = physicals[name]
                mesh_sha = capture["glb_sha256"]
                visual = self._visual_material_record(
                    mesh_sha,
                    "gpt6_blender_code",
                    txid,
                    code_sha256=authored["code_sha256"],
                    capture=capture,
                )
                node = {
                    **copy.deepcopy(old_nodes.get(object_id, {})),
                    "id": object_id,
                    "category": addition["category"],
                    "kind": "object",
                    "description": addition["description"],
                    "mask_path": None,
                    "support": addition["support"],
                    "parent": addition["support"],
                    "children": copy.deepcopy(
                        old_nodes.get(object_id, {}).get("children", [])
                    ),
                    "world_center": copy.deepcopy(row.get("center")),
                    "runtime_added": True,
                    "runtime_transaction_id": txid,
                }
                record = build_runtime_object_record(
                    transaction_id=txid,
                    category=addition["category"],
                    instance=addition["instance"],
                    mask_path=None,
                    mask_sha256=None,
                    mask_policy="excluded",
                    graph_node=node,
                    placement=row,
                    mesh_sha256=mesh_sha,
                    physical_material=physical,
                    visual_material=visual,
                    evidence={
                        "reason": reason or addition["description"],
                        "procedural_capture": capture,
                    },
                    revision_provenance={
                        "source": "gpt6_blender_code",
                        "transaction_id": txid,
                        "code_sha256": authored["code_sha256"],
                        "mesh_sha256": mesh_sha,
                    },
                )
                records.append(record)
                self._invalidate_runtime_mesh_pose_state(
                    name, transaction_id=txid, mesh_sha256=mesh_sha
                )
            if records or declarations["removed_objects"]:
                bundle = stage_runtime_object_batch(
                    self.moge_dir,
                    graph=graph,
                    placement=placement,
                    runtime_inventory=inventory,
                    transaction_id=txid,
                    additions=records,
                    removed_object_ids=declarations["removed_objects"],
                )
                self._write_runtime_bundle(bundle)
                self._pipeline_object_names = [
                    row["mesh_name"] for row in bundle["placement"]["objects"]
                ]
                self._n2id_cache = None
            post = self._read_penetration_data(tag=f"initializer_code_{txid}_authored")
            touched = observed_object_changes(
                pre,
                post,
                added_names=list(declarations["added_names"].values()),
                removed_names=list(declarations["removed_names"].values()),
            )
            # Captures bind the authored candidate frame, before simulation. Keep
            # that reference after baking so later export measures the delta once.
            for name in declarations["added_names"].values():
                self._update_blend_base_entry(name, touched[name]["after_matrix"])
            self._pose_dirty = True
            self._armed = set()
            changed_surfaces = _changed_surface_names(pre, post)
            # Overlap NOTICE (2026-09-15, owner: a wall build is neither simulated nor
            # rejected). A surface changed in this call, or an object authored in it,
            # that intersects an existing object is REPORTED in the result and nothing
            # more; the no-penetration rule in check_rules_enforced stays the only
            # enforcement and the agent decides from the photo which side to move
            # (online7: the wall the photo dictates passes through the vase's
            # reconstruction; rejecting the wall sent the agent hunting vase poses).
            overlap_notice = self._surface_overlap_notices(
                post, changed_surfaces, set(touched) - set(declarations["removed_names"].values())
            )
            scope = self._initializer_bake_scope(touched, declarations, post)
            # Simulate only when an OBJECT was authored/removed/moved: the scope is those
            # objects' hierarchies; a surface-only call has an empty scope and skips physics.
            simulate = bool(post.get("object_integrity")) and bool(scope["bake"])
            self._initializer_bake_scope_state = scope
            try:
                physics_result = (
                    self._settle_initializer_candidate(txid) if simulate else None
                )
            finally:
                self._initializer_bake_scope_state = None
            return self._commit_typed_initializer_mutation(
                txid=txid,
                reason=reason,
                pre=pre,
                reports=physics_result["reports"] if physics_result else [],
                physics_simulated=simulate,
                authored_objects=touched,
                physics_state_change=(
                    {
                        "kind": "initializer_joint_settle",
                        "rest_policy": physics_result["rest_policy"],
                        "rest": physics_result["rest"],
                        "baked": physics_result["baked"],
                    }
                    if physics_result
                    else None
                ),
                artifact_before=prepared["artifact_before"],
                authored_code=authored,
                script_output=self._take_authored_stdout(),
                declarations=declarations,
                render_view=render_view,
                part_name_feedback=_capture_part_feedback(part_captures),
                overlap_notice=overlap_notice,
            )
        except Exception as exc:  # noqa: BLE001 - transaction rollback boundary
            return self._typed_initializer_failure(
                txid=txid,
                kind=kind,
                error=exc,
                live_before=prepared["live_before"],
                graph_before=prepared["graph_before"],
                artifact_before=prepared["artifact_before"],
                pipeline_names_before=prepared["pipeline_names_before"],
                ledger_after_request=prepared["ledger_after_request"],
            )
        finally:
            self._release_active_initializer_scene_lock()

    def _composition_mesh_failure(
        self,
        *,
        txid: int,
        error: Exception,
        live_before: Path,
        graph_before: Optional[bytes],
        artifact_before: dict[Path, tuple[bool, bytes]],
        pipeline_names_before: list[str],
        object_id: str,
        target_mesh: str,
    ) -> dict:
        """Restore every authoritative state after a rejected mesh replacement."""
        kind = _COMPOSITION_MESH_MUTATION_KIND
        if self._edit_meta and self._edit_meta[-1].get("transaction_id") == txid:
            self._edit_meta.pop()
            self.edit_history.pop()
            if self._ledger_history:
                self._ledger_history.pop()
            if self._graph_history:
                self._graph_history.pop()
        restore_errors: list[str] = []
        try:
            self._restore_composition_mesh_snapshot(
                txid=txid,
                live_before=live_before,
                graph_before=graph_before,
                artifact_before=artifact_before,
                pipeline_names_before=pipeline_names_before,
                cleanup_outputs=True,
            )
        except Exception as exc:  # noqa: BLE001
            restore_errors.append(str(exc))
        runtime_errors = self._discard_composition_pose_runtime()
        physics_reports = copy.deepcopy(getattr(error, "reports", []) or [])
        state = "rolled_back" if not restore_errors else "error"
        artifact_manifest = self._safe_typed_initializer_artifact_manifest(
            txid=txid,
            kind=kind,
            state=state,
            object_id=object_id,
            target_mesh=target_mesh,
            target_expected_present=True,
        )
        audit_error = None
        try:
            self._record_mutation_status(
                txid,
                kind,
                state,
                details={
                    "error": str(error),
                    "error_class": _error_class(error),
                    "object_id": object_id,
                    "artifact_manifest": artifact_manifest,
                    **(
                        {"blender_output": getattr(error, "blender_output")}
                        if getattr(error, "blender_output", None)
                        else {}
                    ),
                    **({"physics_reports": physics_reports} if physics_reports else {}),
                    **({"restore_errors": restore_errors} if restore_errors else {}),
                    **(
                        {"physics_runtime_cleanup_errors": runtime_errors}
                        if runtime_errors
                        else {}
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            audit_error = str(exc)
        marker_error = None
        try:
            self._set_runtime_recovery_state(
                txid,
                "rolled_back"
                if not restore_errors and audit_error is None
                else "recovery_needed",
                details={
                    "error": str(error),
                    "error_class": _error_class(error),
                    "artifact_manifest": artifact_manifest,
                    **({"restore_errors": restore_errors} if restore_errors else {}),
                    **({"audit_error": audit_error} if audit_error else {}),
                },
            )
        except Exception as exc:  # noqa: BLE001
            marker_error = str(exc)
        detail = (
            "edit_object_mesh rejected and "
            + ("rolled back" if not restore_errors else "requires recovery")
            + f": {error}"
        )
        if restore_errors:
            detail += "; " + "; ".join(restore_errors)
        if audit_error:
            detail += f"; terminal audit failed: {audit_error}"
        if marker_error:
            detail += f"; recovery marker update failed: {marker_error}"
        return {
            "status": "error",
            "output": {
                "text": [detail],
                "transaction_id": txid,
                "scene_mutation": (
                    "not_committed" if not restore_errors else "unknown"
                ),
                "artifact_manifest": artifact_manifest,
                **({"physics_reports": physics_reports} if physics_reports else {}),
            },
        }

    def _commit_composition_mesh_mutation(
        self,
        *,
        txid: int,
        object_id: str,
        target_mesh: str,
        reason: str,
        pre: dict,
        pre_blend_path: Path,
        reports: list[dict],
        artifact_before: dict[Path, tuple[bool, bytes]],
        material: dict,
        physics_state_change: dict,
        authored_code: dict,
        part_name_feedback: str = "",
    ) -> dict:
        from lib.tools.geometry.inventory_contract import validate_scene_artifacts

        kind = _COMPOSITION_MESH_MUTATION_KIND
        validate_scene_artifacts(
            self.moge_dir,
            allow_runtime_additions=True,
            validate_runtime_physics=True,
        )
        post = self._read_penetration_data(
            tag=f"composition_mesh_{txid}_after",
            _typed_world_semantics=True,
        )
        touched = self._validate_typed_initializer_result(
            pre,
            post,
            target=target_mesh,
            physics_reports=reports,
            pre_blend_path=pre_blend_path,
            post_blend_path=Path(self.blender_save),
        )
        artifact_manifest = self._typed_initializer_artifact_manifest(
            txid=txid,
            kind=kind,
            state="committed",
            object_id=object_id,
            target_mesh=target_mesh,
            target_expected_present=True,
            touched_objects=touched,
            strict=True,
        )
        transaction = {
            "id": txid,
            "kind": kind,
            "status": "committed",
            "attempt_idx": int(self.attempt_idx),
            "target": target_mesh,
            "object_id": object_id,
            "reason": reason,
            "objects": touched,
            "physics_reports": copy.deepcopy(reports),
            "physical_material": copy.deepcopy(material),
            "physics_state_change": copy.deepcopy(physics_state_change),
            "authored_code": copy.deepcopy(authored_code),
            "artifact_manifest": copy.deepcopy(artifact_manifest),
        }
        self._set_runtime_recovery_state(txid, "committing")
        history_len = len(self.edit_history)
        self._push_edit_snapshot(
            kind,
            transaction_id=txid,
            object_id=object_id,
            artifact_before=artifact_before,
        )
        undo_snapshot = (
            Path(self.edit_history[-1])
            if len(self.edit_history) > history_len
            else None
        )
        if (
            len(self.edit_history) != history_len + 1
            or not self._edit_meta
            or self._edit_meta[-1].get("transaction_id") != txid
            or self._edit_meta[-1].get("kind") != kind
            or undo_snapshot is None
            or not undo_snapshot.is_file()
        ):
            raise RuntimeError("composition mesh undo snapshot was not recorded")
        # The post-edit undo image is part of C even though it lives outside the
        # canonical scene directory. Flush it before the journal commit decision;
        # otherwise a successful transaction could advertise an undo step whose
        # only Blender payload vanished on power loss.
        self._fsync_file(undo_snapshot)
        self._fsync_directory(undo_snapshot.parent)
        self._durably_sync_typed_initializer_state(txid)
        details = {
            "object_id": object_id,
            "mesh_name": target_mesh,
            "physical_material": material,
            "physics_reports": reports,
            "objects": touched,
            "physics_state_change": physics_state_change,
            "authored_code": authored_code,
            "artifact_manifest": artifact_manifest,
        }
        try:
            # The complete journal frame is the irreversible commit decision.  If
            # this call is ambiguous, startup recovery inspects the durable frame;
            # the live replacement must not be compensated in-process.
            self._record_mutation_status(txid, kind, "committed", details=details)
            # 2026-09-15 (owner): a replaced mesh is a new reconstruction — the coverage
            # end-gate requires one more investigate of it (v5accept toast standard ended
            # with toast#1's last reading taken before its replacement).
            self._replaced_uninvestigated = getattr(
                self, "_replaced_uninvestigated", set()
            ) | {object_id}
            self._set_runtime_recovery_state(
                txid,
                "committed",
                details={
                    "object_id": object_id,
                    "mesh_name": target_mesh,
                    "artifact_manifest": artifact_manifest,
                },
            )
        except Exception as exc:  # noqa: BLE001
            marker_error = None
            try:
                self._set_runtime_recovery_state(
                    txid,
                    "recovery_needed",
                    details={
                        "phase": "commit_decision_reconciliation",
                        "error": str(exc),
                        "artifact_manifest": artifact_manifest,
                    },
                )
            except Exception as marker_exc:  # noqa: BLE001
                marker_error = str(marker_exc)
            self._discard_composition_pose_runtime()
            message = (
                f"edit_object_mesh transaction {txid} reached an ambiguous durable "
                "commit boundary. Its edited scene was preserved; restart the "
                "composition stage before another mutation."
            )
            if marker_error:
                message += f" Recovery-marker update also failed: {marker_error}"
            return {
                "status": "error",
                "output": {
                    "text": [message],
                    "transaction_id": txid,
                    "scene_mutation": "commit_recovery_pending",
                    "commit_recovery_pending": True,
                    "retryable": False,
                    "artifact_manifest": artifact_manifest,
                },
            }

        commit_text = (
            f"edit_object_mesh committed transaction {txid} for {object_id}. "
            "The replacement is a durable maskless runtime revision; its material "
            "record and collider were rebuilt, and the target/support-dependent "
            "stack was physics-settled. The transaction is one undo step."
        )
        if part_name_feedback:
            commit_text += "\n\n" + part_name_feedback
        script_output = self._take_authored_stdout()
        if script_output:
            commit_text += "\n\n" + script_output
        try:
            rendered = self._render_novel_view(0.0, 0.0)
            if not isinstance(rendered, dict) or rendered.get("status") != "success":
                raise RuntimeError("post-commit render was unavailable")
            output = rendered.get("output")
            if not isinstance(output, dict) or not isinstance(output.get("text"), list):
                raise RuntimeError("post-commit render output is malformed")
            output["text"].insert(0, commit_text)
            output["transaction"] = transaction
            return rendered
        except Exception:  # noqa: BLE001 - presentation cannot reverse commit
            return {
                "status": "success",
                "output": {"text": [commit_text], "transaction": transaction},
            }

    def edit_composition_object_mesh(
        self,
        *,
        code: str,
        object: str,  # noqa: A002
        physical_material_hint: Optional[str],
        expected_resting_mode: str,
        reason: str,
    ) -> dict:
        """Replace one investigated composition target through authored Blender code."""
        if not self._composition_mesh_recovery_enabled():
            return {
                "status": "error",
                "output": {
                    "text": [
                        "edit_object_mesh is available only in the "
                        "composition stage with mesh editing enabled"
                    ]
                },
            }
        if not str(reason).strip():
            return {
                "status": "error",
                "output": {"text": ["edit_object_mesh reason must be nonempty"]},
            }
        resting_mode = str(expected_resting_mode or "preserve")
        if resting_mode not in {"preserve", "side", "free"}:
            return {
                "status": "error",
                "output": {"text": ["invalid expected_resting_mode"]},
            }
        target, name_error = self._resolve_initializer_object(str(object))
        if target is None:
            return {"status": "error", "output": {"text": [name_error]}}
        object_id = self._name2id().get(target)
        if not object_id:
            return {
                "status": "error",
                "output": {"text": [f"cannot resolve graph identity for {target}"]},
            }
        if int(self._investigated.get(object_id, 0)) <= 0:
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"edit_object_mesh rejected: {object_id} has not been "
                        "successfully investigated in this composition stage. Call "
                        f"investigate_objects(objects=['{object_id}']) first."
                    ]
                },
            }
        try:
            session = self._pose_session_get()
            if object_id not in session.objects():
                raise RuntimeError(f"pose session has no object {object_id}")
            session_record = (getattr(session, "ctx", None) or {}).get(object_id)
            if (
                not isinstance(session_record, dict)
                or session_record.get("mesh_name") != target
            ):
                raise RuntimeError(
                    f"pose session binding for {object_id} is not {target}"
                )
            if self._physics_obj is None or session.physics is not self._physics_obj:
                raise RuntimeError("composition physics authority is unavailable")
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"edit_object_mesh could not establish its physics baseline: {exc}"
                    ],
                    "scene_mutation": "not_started",
                },
            }

        request = {
            "code": code,
            "object": object,
            "physical_material_hint": physical_material_hint,
            "expected_resting_mode": resting_mode,
            "reason": reason,
        }
        prepared = self._prepare_composition_mesh_mutation(request)
        if prepared.get("status") != "ready":
            return prepared
        txid = int(prepared["txid"])
        live_before = prepared["live_before"]
        graph_before = prepared["graph_before"]
        artifact_before = prepared["artifact_before"]
        names_before = prepared["pipeline_names_before"]
        authored_code = None
        try:
            from lib.tools.blender.procedural_object import (
                build_procedural_placement_row,
            )
            from lib.tools.geometry.runtime_object_repair import (
                sha256_file,
                stage_runtime_mesh_revision,
            )

            graph, placement, inventory = self._load_runtime_scene_state()
            node = next(
                (
                    row
                    for row in graph["nodes"]
                    if isinstance(row, dict) and row.get("id") == object_id
                ),
                None,
            )
            old_row = next(
                (
                    row
                    for row in placement["objects"]
                    if isinstance(row, dict) and row.get("mesh_name") == target
                ),
                None,
            )
            if node is None or old_row is None:
                raise RuntimeError(
                    "graph/placement binding for composition mesh target is incomplete"
                )
            if self._name2id().get(target) != object_id:
                raise RuntimeError(
                    "composition target identity changed before execution"
                )
            pre = self._read_penetration_data(
                blend_path=str(live_before),
                tag=f"composition_mesh_{txid}_before",
                _typed_world_semantics=True,
            )
            authored_code = self._run_initializer_authored_code(
                code=code,
                kind=_COMPOSITION_MESH_MUTATION_KIND,
                transaction_id=txid,
                object_id=object_id,
                mesh_name=target,
                live_before=live_before,
            )
            mesh_path, capture = self._export_initializer_authored_object(
                mesh_name=target,
                object_id=object_id,
                transaction_id=txid,
            )
            category = str(old_row["category"])
            instance = int(old_row["instance"])
            support = str(node.get("support") or node.get("parent") or "")
            new_row = build_procedural_placement_row(
                category=category,
                instance=instance,
                support=support,
                mask=None,
                capture=capture,
                mesh_glb=mesh_path,
                old_row=old_row,
                transaction_id=txid,
            )
            new_row.pop("mask_path", None)
            new_row.update(
                {
                    "mesh_name": target,
                    "support": support,
                    "runtime_mesh_revision_transaction_id": txid,
                }
            )
            mesh_sha = sha256_file(mesh_path)
            physical = self._estimate_runtime_physics_material(
                new_row,
                physical_material_hint=physical_material_hint,
                transaction_id=txid,
                force_refresh=True,
            )
            visual = self._visual_material_record(
                mesh_sha,
                "gpt6_blender_code",
                txid,
                code_sha256=authored_code["code_sha256"],
                capture=capture,
            )
            evidence = {
                "reason": str(reason).strip(),
                "investigated_count": int(self._investigated[object_id]),
                "authored_code_sha256": authored_code["code_sha256"],
                "procedural_capture": copy.deepcopy(capture),
            }
            bundle = stage_runtime_mesh_revision(
                self.moge_dir,
                graph=graph,
                placement=placement,
                object_id=object_id,
                transaction_id=txid,
                mesh_glb=str(mesh_path),
                mesh_sha256=mesh_sha,
                mesh_name=target,
                replacement_placement=new_row,
                evidence=evidence,
                revision_provenance={
                    "source": "gpt6_composition_blender_code",
                    "transaction_id": txid,
                    "code_sha256": authored_code["code_sha256"],
                    "mesh_sha256": mesh_sha,
                },
                physical_material=physical,
                visual_material=visual,
                # 2026-09-16 (owner): a replacement keeps the SAME logical identity, so it
                # inherits the original object's photo mask (tracked) — investigate_objects
                # measures the new mesh against the photo and the coverage gate can list
                # it until it is investigated once more. Initializer ADDITIONS stay
                # maskless (excluded): they are new objects with no photo mask.
                mask_policy="tracked",
                runtime_inventory=inventory,
            )
            self._write_runtime_bundle(bundle)
            self._update_blend_base_entry(target, capture["root_matrix_world"])
            physics_state_change = self._invalidate_runtime_mesh_pose_state(
                target,
                transaction_id=txid,
                mesh_sha256=mesh_sha,
            )

            # Keep the old physics authority and its geometry signatures: rebuilding
            # only PoseSession lets settle_edited detect the changed target, force a
            # fresh collider cook, and use its normal parent-first/dependent closure.
            session.rebuild()
            session.physics = self._physics_obj
            self._pose_dirty = False
            self._physics_obj.support_map = self._support_body_map() or {}
            reports = self._physics_obj.settle_edited(
                session,
                intended_resting_modes=self._typed_settle_intents(
                    {target: resting_mode}
                ),
                force_rebuild_names=[target],
            )
            if target not in self._physics_report_names(reports):
                raise RuntimeError(
                    "physics did not force-rebuild and settle the replaced target"
                )
            session.save()
            self._armed = set()
            return self._commit_composition_mesh_mutation(
                txid=txid,
                object_id=object_id,
                target_mesh=target,
                reason=str(reason).strip(),
                pre=pre,
                pre_blend_path=live_before,
                reports=reports,
                artifact_before=artifact_before,
                material=physical,
                physics_state_change=physics_state_change,
                authored_code=authored_code,
                part_name_feedback=_capture_part_feedback([(object_id, capture)]),
            )
        except Exception as exc:  # noqa: BLE001
            return self._composition_mesh_failure(
                txid=txid,
                error=exc,
                live_before=live_before,
                graph_before=graph_before,
                artifact_before=artifact_before,
                pipeline_names_before=names_before,
                object_id=object_id,
                target_mesh=target,
            )
        finally:
            self._release_active_initializer_scene_lock()

    # -- initializer runtime root surfaces --------------------------------------- #

    @staticmethod
    def _finite_vector(value, size: int, label: str) -> list[float]:
        try:
            out = [float(x) for x in value]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} must contain exactly {size} finite numbers"
            ) from exc
        if len(out) != size or not all(math.isfinite(x) for x in out):
            raise ValueError(f"{label} must contain exactly {size} finite numbers")
        return out

    @staticmethod
    def _walk_json_values(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from Executor._walk_json_values(child)
        elif isinstance(value, list):
            for child in value:
                yield from Executor._walk_json_values(child)
        else:
            yield value

    def _reserved_wall_indices(self, graph: dict) -> set[int]:
        """All historically mentioned wall indices, not just retained nodes.

        Pruned/merged masks and removed runtime roots keep their identities reserved,
        so a newly added wall never impersonates an earlier wall#N.  Existing Blender
        names/custom properties are checked again inside the trusted creation script.
        This allocator is not an execute_and_evaluate name-policy filter.
        """
        indices: set[int] = set()

        def scan(payload) -> None:
            for value in self._walk_json_values(payload):
                if not isinstance(value, str):
                    continue
                for pattern in (r"\bwall#(\d+)\b", r"\bwall_(\d+)\b"):
                    indices.update(int(x) for x in re.findall(pattern, value.lower()))
            if isinstance(payload, dict):
                for row in payload.get("instances", []) or []:
                    if (
                        isinstance(row, dict)
                        and str(row.get("category", "")).strip().lower() == "wall"
                        and isinstance(row.get("instance"), int)
                    ):
                        indices.add(int(row["instance"]))
                for row in payload.get("unmasked_root_surfaces", []) or []:
                    if (
                        isinstance(row, dict)
                        and str(row.get("category", "")).strip().lower() == "wall"
                        and isinstance(row.get("instance"), int)
                    ):
                        indices.add(int(row["instance"]))

        scan(graph)
        scan(getattr(self, "_initializer_ledger", {}) or {})
        if self.moge_dir:
            masks_path = Path(self.moge_dir) / "masks" / "masks.json"
            try:
                scan(json.loads(masks_path.read_text()))
            except (OSError, ValueError, TypeError):
                pass
            masks_dir = Path(self.moge_dir) / "masks"
            if masks_dir.is_dir():
                for path in masks_dir.iterdir():
                    match = re.match(r"wall_(\d+)(?:\D|$)", path.name.lower())
                    if match:
                        indices.add(int(match.group(1)))
        return indices

    def _next_relationship_id(self, graph: dict) -> str:
        used = set()
        for key in ("relationships", "relationship_audit"):
            for rel in graph.get(key, []) or []:
                match = re.fullmatch(r"rel-(\d+)", str(rel.get("relationship_id", "")))
                if match:
                    used.add(int(match.group(1)))
        for value in self._walk_json_values(
            getattr(self, "_initializer_ledger", {}) or {}
        ):
            match = re.fullmatch(r"rel-(\d+)", str(value))
            if match:
                used.add(int(match.group(1)))
        i = 0
        while i in used:
            i += 1
        return f"rel-{i:03d}"

    @staticmethod
    def _refresh_relationship_counts(graph: dict) -> None:
        summary = graph.get("relationship_adjudication")
        if not isinstance(summary, dict):
            return
        rels = graph.get("relationships", []) or []
        summary["counts_by_enforcement"] = {
            key: sum(1 for r in rels if r.get("enforcement") == key)
            for key in ("hard", "advisory", "none")
        }
        summary["counts_by_status"] = {
            key: sum(1 for r in rels if r.get("status") == key)
            for key in ("confirmed", "conflicting", "rejected", "unverified")
        }

    @staticmethod
    def _runtime_root_record(node: dict) -> dict:
        return {
            "id": node["id"],
            "build_name": node["build_name"],
            "category": node["category"],
            "description": node["description"],
            "reason": node["runtime_reason"],
            "target_region_norm": list(node["target_region_norm"]),
            "added_attempt": node["added_attempt"],
            "transaction_id": node["runtime_transaction_id"],
            "scene_graph_revision": node["runtime_graph_revision"],
        }

    def _runtime_root_budget(self, graph: dict) -> tuple[bool, str]:
        active_ids = {
            str(n.get("id"))
            for n in graph.get("nodes", []) or []
            if n.get("kind") == "root_surface" and n.get("runtime_added") is True
        }
        ledger_ids = {
            str(row.get("id"))
            for row in (getattr(self, "_initializer_ledger", {}) or {}).get(
                "runtime_root_surfaces", []
            )
            if isinstance(row, dict) and row.get("id")
        }
        count = len(active_ids | ledger_ids)
        if count >= 2:
            return (
                False,
                "the initializer already has its full 2 runtime-added root surfaces",
            )
        return True, ""

    @staticmethod
    def _resolve_graph_root(graph: dict, value: str) -> tuple[Optional[dict], str]:
        from lib.tools.geometry.surface_relations import surface_build_name

        matches = []
        for node in graph.get("nodes", []) or []:
            if node.get("kind") != "root_surface" or not node.get("id"):
                continue
            build = str(node.get("build_name") or surface_build_name(node["id"]))
            if value in {str(node["id"]), build}:
                matches.append(node)
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return None, f"root identity '{value}' is ambiguous"
        return None, f"'{value}' is not a current registered root id/build name"

    def _record_runtime_root_rejection(
        self, txid: int, action: str, reason: str, failure: str
    ) -> None:
        ledger = getattr(self, "_initializer_ledger", self._empty_initializer_ledger())
        ledger.setdefault("events", []).append(
            {
                "id": int(txid),
                "kind": "root_surface",
                "action": action,
                "status": "rejected",
                "attempt_idx": int(getattr(self, "attempt_idx", 1)),
                "reason": str(reason),
                "failure": str(failure),
            }
        )
        ledger["next_transaction_id"] = max(
            int(ledger.get("next_transaction_id") or 1), int(txid) + 1
        )
        self._persist_initializer_ledger()

    def _rollback_runtime_root_edit(
        self,
        *,
        history_n: int,
        ledger_history_n: int,
        graph_history_n: int,
        edit_meta_n: int,
        ledger_before: dict,
        graph_before: Optional[bytes],
    ) -> None:
        restore_from = (
            self.edit_history[history_n - 1] if history_n else self.base_state
        )
        if (
            not restore_from
            or not self.blender_save
            or not os.path.isfile(restore_from)
        ):
            raise RuntimeError("pre-transaction Blender snapshot is unavailable")
        live = Path(self.blender_save)
        staged = live.with_name(
            f".{live.name}.runtime-root-rollback-{os.getpid()}-{getattr(self, 'count', 0)}"
        )
        graph_current = self._read_scene_graph_bytes()
        ledger_current = copy.deepcopy(self._initializer_ledger)
        try:
            shutil.copy2(restore_from, staged)
            try:
                self._restore_scene_graph_bytes(graph_before)
                self._initializer_ledger = copy.deepcopy(ledger_before)
                self._persist_initializer_ledger()
                os.replace(staged, live)
            except Exception as exc:
                # Blend replacement is the final commit point.  If it (or an earlier
                # staged restore) fails, compensate graph+ledger back to the current
                # transaction state so we never expose an old graph with a new blend.
                compensation = []
                try:
                    self._restore_scene_graph_bytes(graph_current)
                except Exception as graph_exc:  # noqa: BLE001
                    compensation.append(f"graph compensation failed: {graph_exc}")
                self._initializer_ledger = ledger_current
                try:
                    self._persist_initializer_ledger()
                except Exception as ledger_exc:  # noqa: BLE001
                    compensation.append(f"ledger compensation failed: {ledger_exc}")
                if not compensation and len(self.edit_history) > history_n:
                    # Keep the chronological snapshots coherent with the compensated
                    # NEW state.  A subsequent edit+undo must not restore the old ledger
                    # merely because this failed rollback left the new blend committed.
                    if len(self._ledger_history) > ledger_history_n:
                        self._ledger_history[-1] = copy.deepcopy(ledger_current)
                    if len(getattr(self, "_graph_history", [])) > graph_history_n:
                        self._graph_history[-1] = graph_current
                suffix = "; CRITICAL " + "; ".join(compensation) if compensation else ""
                raise RuntimeError(
                    f"runtime-root rollback could not commit: {exc}{suffix}"
                ) from exc
        finally:
            staged.unlink(missing_ok=True)
        del self.edit_history[history_n:]
        del self._ledger_history[ledger_history_n:]
        if hasattr(self, "_graph_history"):
            del self._graph_history[graph_history_n:]
        del self._edit_meta[edit_meta_n:]
        self._pose_dirty = True
        self._armed = set()

    def _commit_runtime_root_edit(
        self,
        *,
        graph: dict,
        ledger: dict,
        kind: str,
        txid: int,
        graph_id: str,
    ) -> None:
        path = self._scene_graph_path()
        if path is None:
            raise RuntimeError("active scene_graph.json is unavailable")
        self._atomic_write_json(path, graph)
        from lib.tools.geometry.relationship_constraints import (
            write_initializer_constraints,
        )

        write_initializer_constraints(path.parent, graph)
        self._initializer_constraint_artifact = None
        self._initializer_ledger = ledger
        self._persist_initializer_ledger()
        if not self.edit_history:
            raise RuntimeError(
                "the Blender edit did not create a durable undo snapshot"
            )
        self._ledger_history[-1] = copy.deepcopy(ledger)
        if not hasattr(self, "_graph_history") or not self._graph_history:
            raise RuntimeError("the Blender edit did not create a graph undo snapshot")
        self._graph_history[-1] = self._read_scene_graph_bytes()
        self._edit_meta[-1] = {
            "kind": kind,
            "transaction_id": txid,
            "object_id": graph_id,
            "edit_token": None,
        }
        self._pose_dirty = True
        self._armed = set()

    def _runtime_root_render(self) -> dict[str, object]:
        if self._pseudo_gt_views():
            return self._render_novel_view(0.0, 0.0)
        return self.render_current_scene()

    def _runtime_root_safety_violations(self, data: dict, build_name: str) -> list[str]:
        body = self._body_map(data).get(build_name)
        if body is None:
            return [
                f"created Blender root '{build_name}' is missing from geometry output"
            ]
        ext = [float(body["hi"][i]) - float(body["lo"][i]) for i in range(3)]
        # Do not infer thickness from the WORLD AABB: a valid thin wall at 45deg has
        # large X and Y AABB spans.  The trusted construction script owns the validated
        # local thickness; here we only fail closed on a collapsed vertical extent.
        if ext[2] < 0.18:
            return [
                f"created geometry has a collapsed vertical wall extent (world extents {ext})"
            ]
        main, _ = identify_main_support(data.get("bodies", []))
        problems = []
        for pair in data.get("penetrating_pairs", []) or []:
            if build_name not in {pair.get("a"), pair.get("b")}:
                continue
            other = pair.get("b") if pair.get("a") == build_name else pair.get("a")
            depth = float(pair.get("depth", 0.0))
            if depth > 0.10 and (str(other).startswith("obj_") or other == main):
                problems.append(
                    f"gross penetration with {other} ({depth:.3f}m > 0.100m)"
                )
        return problems

    def build_root_surface(
        self,
        *,
        surface_type: str,
        description: str,
        reason: str,
        center: list,
        normal_xy: list,
        width_m: float,
        height_m: float,
        thickness_m: float,
        target_region_norm: list,
        copy_material_from: Optional[str] = None,
        floor_support: Optional[str] = None,
        relationship_hints: Optional[list] = None,
    ) -> dict[str, object]:
        if self.root_stage_name != "initializer":
            return {
                "status": "error",
                "output": {"text": ["build_root_surface is an initializer-stage tool"]},
            }
        try:
            self._assert_no_incomplete_runtime_mutation()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "output": {"text": [f"build_root_surface blocked by recovery: {exc}"]},
            }
        txid = int(self._initializer_ledger.get("next_transaction_id") or 1)

        def fail(message: str, *, audit: bool = True) -> dict[str, object]:
            if audit:
                try:
                    self._record_runtime_root_rejection(txid, "add", reason, message)
                except Exception as exc:  # noqa: BLE001
                    message += f"; rejection audit could not be persisted: {exc}"
            return {
                "status": "error",
                "output": {
                    "text": [
                        "build_root_surface rejected and left the scene graph/blend "
                        "unchanged: " + message
                    ]
                },
            }

        if surface_type != "wall":
            return fail("version 1 accepts only surface_type='wall'")
        description = str(description or "").strip()
        reason = str(reason or "").strip()
        if not description or not reason:
            return fail("description and reason must both be non-empty")
        try:
            center_v = self._finite_vector(center, 3, "center")
            normal_v = self._finite_vector(normal_xy, 2, "normal_xy")
            region = self._finite_vector(target_region_norm, 4, "target_region_norm")
            width, height, thickness = map(float, (width_m, height_m, thickness_m))
            if not all(math.isfinite(x) for x in (width, height, thickness)):
                raise ValueError("wall dimensions must be finite")
        except (TypeError, ValueError) as exc:
            return fail(str(exc))
        nlen = math.hypot(*normal_v)
        if nlen < 1e-6:
            return fail("normal_xy must be non-zero")
        normal_v = [normal_v[0] / nlen, normal_v[1] / nlen]
        if not (0.10 <= width <= 20.0 and 0.20 <= height <= 10.0):
            return fail("width_m must be 0.10-20m and height_m must be 0.20-10m")
        if not (0.01 <= thickness <= 0.25):
            return fail("thickness_m must be 0.01-0.25m")
        if thickness > 0.25 * min(width, height):
            return fail(
                "thickness_m must be at most 25% of both width_m and height_m "
                "so the registered wall is broad and thin"
            )
        if max(abs(x) for x in center_v) > 100.0:
            return fail("center is outside the 100m scene-safety bound")
        x0, y0, x1, y1 = region
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            return fail("target_region_norm must satisfy 0<=x0<x1<=1 and 0<=y0<y1<=1")
        graph_path = self._scene_graph_path()
        if graph_path is None or not graph_path.exists():
            return fail("active scene_graph.json is unavailable")
        try:
            graph = json.loads(graph_path.read_text())
            if not isinstance(graph.get("nodes"), list):
                raise ValueError("nodes must be a list")
        except (OSError, ValueError, TypeError) as exc:
            return fail(f"active scene graph is invalid: {exc}")
        budget_ok, budget_reason = self._runtime_root_budget(graph)
        if not budget_ok:
            return fail(budget_reason)

        floor_node = None
        if floor_support:
            floor_node, error = self._resolve_graph_root(graph, str(floor_support))
            if floor_node is None:
                return fail("floor_support " + error)
            if str(floor_node.get("category", "")).strip().lower() not in {
                "floor",
                "ground",
            }:
                return fail(
                    "floor_support must identify a registered floor/ground root"
                )
        hint_rows = []
        seen_hint = set()
        for hint in relationship_hints or []:
            if not isinstance(hint, dict):
                return fail("each relationship_hints entry must be an object")
            rel_type = str(hint.get("type", "")).strip().lower()
            if rel_type not in {"corner", "perpendicular", "against"}:
                return fail(
                    "relationship hint type must be corner/perpendicular/against"
                )
            other, error = self._resolve_graph_root(graph, str(hint.get("other", "")))
            if other is None:
                return fail("relationship hint other " + error)
            if (
                rel_type == "corner"
                and str(other.get("category", "")).strip().lower() != "wall"
            ):
                return fail(
                    "CORNER hint 'other' must identify another registered wall root"
                )
            if rel_type == "against":
                category = str(other.get("category", "")).strip().lower()
                _normal, plane_kind = plumb_plane(
                    (other.get("plane") or {}).get("normal")
                )
                horizontal = (
                    plane_kind == "support"
                    or other.get("main_support") is True
                    or other.get("id") == graph.get("main_support_id")
                )
                if category in {"wall", "floor", "ground"} or not horizontal:
                    return fail(
                        "AGAINST hint 'other' must be the non-floor horizontal support "
                        "that is against the new wall; wall-to-wall/floor AGAINST is invalid"
                    )
            key = (rel_type, other["id"])
            if key not in seen_hint:
                hint_rows.append((rel_type, other))
                seen_hint.add(key)
        if copy_material_from:
            material_node, error = self._resolve_graph_root(
                graph, str(copy_material_from)
            )
            if material_node is None:
                return fail("copy_material_from " + error)
            material_build_name = str(material_node.get("build_name") or "")
            if not material_build_name:
                from lib.tools.geometry.surface_relations import surface_build_name

                material_build_name = surface_build_name(material_node["id"])
        else:
            material_build_name = None

        reserved = sorted(self._reserved_wall_indices(graph))
        revision = int(graph.get("scene_graph_revision") or 0) + 1
        result_path = self.render_path.parent / "tmp" / f"runtime_root_{txid}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.unlink(missing_ok=True)
        yaw = math.atan2(normal_v[1], normal_v[0])
        code = f"""import bpy, json, math, re
reserved = set({reserved!r})
occupied = set()
for existing in bpy.data.objects:
    match = re.fullmatch(r'wall_(\\d+)(?:\\.\\d+)*', existing.name)
    if match:
        occupied.add(int(match.group(1)))
i = 0
while (i in reserved or i in occupied or bpy.data.objects.get(f'wall_{{i}}') is not None or
       any(str(obj.get('grase_graph_id', '')) == f'wall#{{i}}' for obj in bpy.data.objects)):
    i += 1
name = f'wall_{{i}}'
graph_id = f'wall#{{i}}'
bpy.ops.mesh.primitive_cube_add(location={tuple(center_v)!r})
wall = bpy.context.active_object
wall.name = name
wall.dimensions = ({thickness!r}, {width!r}, {height!r})
wall.rotation_euler = (0.0, 0.0, {yaw!r})
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
wall['grase_graph_id'] = graph_id
wall['grase_root_source'] = 'initializer_runtime'
wall['grase_graph_revision'] = {revision!r}
wall['grase_surface_type'] = 'wall'
wall['grase_runtime_transaction_id'] = {txid!r}
source_name = {material_build_name!r}
if source_name:
    source = bpy.data.objects.get(source_name)
    if source is None:
        raise RuntimeError(f"copy_material_from root {{source_name!r}} is absent in Blender")
    if source.data and getattr(source.data, 'materials', None) and len(source.data.materials):
        seeded = source.data.materials[0].copy()
        seeded.name = f'RuntimeWallMaterial_{{i}}'
        wall.data.materials.append(seeded)
if not wall.data.materials:
    mat = bpy.data.materials.new(name=f'RuntimeWallMaterial_{{i}}')
    mat.diffuse_color = (0.55, 0.55, 0.55, 1.0)
    wall.data.materials.append(mat)
with open({str(result_path)!r}, 'w') as stream:
    json.dump({{
        'index': i,
        'graph_id': graph_id,
        'build_name': name,
        'graph_revision': {revision!r},
        'runtime_transaction_id': {txid!r},
        'surface_type': 'wall',
    }}, stream)
"""
        self._ensure_undo_base()
        ledger_before = copy.deepcopy(self._initializer_ledger)
        graph_before = self._read_scene_graph_bytes()
        history_n = len(self.edit_history)
        ledger_history_n = len(self._ledger_history)
        graph_history_n = len(getattr(self, "_graph_history", []))
        edit_meta_n = len(self._edit_meta)
        had_last_code = hasattr(self, "_last_code")
        last_code = getattr(self, "_last_code", None)
        had_build_result = hasattr(self, "_initializer_build_root_result_path")
        old_build_result = getattr(self, "_initializer_build_root_result_path", None)
        self._initializer_build_root_result_path = str(result_path)
        edit = None
        execute_error = None
        try:
            edit = self.execute(code, render=False)
        except Exception as exc:  # noqa: BLE001 - trusted edit is transactional
            execute_error = exc
        finally:
            if had_build_result:
                self._initializer_build_root_result_path = old_build_result
            else:
                self.__dict__.pop("_initializer_build_root_result_path", None)
            if had_last_code:
                self._last_code = last_code
            else:
                self.__dict__.pop("_last_code", None)
            if len(self._edit_meta) > edit_meta_n:
                self._edit_meta[-1]["code_base"] = last_code if had_last_code else None
        if (
            execute_error is not None
            or not isinstance(edit, dict)
            or edit.get("status") != "success"
        ):
            message = (
                str(execute_error)
                if execute_error is not None
                else (edit or {})
                .get("output", {})
                .get("text", ["Blender edit failed"])[0]
            )
            try:
                self._rollback_runtime_root_edit(
                    history_n=history_n,
                    ledger_history_n=ledger_history_n,
                    graph_history_n=graph_history_n,
                    edit_meta_n=edit_meta_n,
                    ledger_before=ledger_before,
                    graph_before=graph_before,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                return fail(
                    f"{message}; CRITICAL transaction rollback failed: {rollback_exc}",
                    audit=False,
                )
            return fail(str(message))
        try:
            selected = json.loads(result_path.read_text())
            graph_id = str(selected["graph_id"])
            build_name = str(selected["build_name"])
            if graph_id in {str(n.get("id")) for n in graph["nodes"]}:
                raise RuntimeError(f"allocator produced existing graph id {graph_id}")
            post = self._read_penetration_data(tag=f"runtime_root_{txid}_after")
            problems = self._runtime_root_safety_violations(post, build_name)
            if problems:
                raise RuntimeError("; ".join(problems))

            node = {
                "id": graph_id,
                "category": "wall",
                "kind": "root_surface",
                "support": None,
                "rollable": None,
                "description": description,
                "mask_path": None,
                "plane": None,
                "extent": None,
                "world_center": list(center_v),
                "span": [width, height],
                "children": [],
                "source": "initializer_build_root_surface",
                "runtime_added": True,
                "build_name": build_name,
                "target_region_norm": list(region),
                "runtime_reason": reason,
                "added_attempt": int(getattr(self, "attempt_idx", 1)),
                "runtime_transaction_id": txid,
                "runtime_graph_revision": revision,
            }
            graph["nodes"].append(node)
            roots = graph.setdefault("roots", [])
            if graph_id not in roots:
                roots.append(graph_id)
            graph["scene_graph_revision"] = revision
            added_relationships = []

            def add_rel(rel_type: str, a: str, b: str, hard: bool) -> None:
                rel = {
                    "type": rel_type,
                    "a": a,
                    "b": b,
                    "provenance": [
                        {
                            "source": "initializer_build_root_surface",
                            "action": "registered",
                            "transaction_id": txid,
                        }
                    ],
                    "status": "confirmed" if hard else "unverified",
                    "enforcement": "hard" if hard else "advisory",
                    "relationship_id": self._next_relationship_id(graph),
                    "adjudication_reasons": [
                        "initializer-declared floor support is policy-authoritative"
                        if hard
                        else "initializer visual hint; advisory only"
                    ],
                    "measurements": {},
                    "runtime_added": True,
                }
                graph.setdefault("relationships", []).append(rel)
                graph.setdefault("relationship_audit", []).append(copy.deepcopy(rel))
                added_relationships.append(rel["relationship_id"])

            if floor_node is not None:
                add_rel("under", floor_node["id"], graph_id, True)
            for rel_type, other in hint_rows:
                # Canonical AGAINST semantics are support -> wall.  Symmetric
                # CORNER/PERPENDICULAR hints keep the new wall first for readability.
                if rel_type == "against":
                    add_rel(rel_type, other["id"], graph_id, False)
                else:
                    add_rel(rel_type, graph_id, other["id"], False)
            self._refresh_relationship_counts(graph)

            ledger = copy.deepcopy(ledger_before)
            root_record = self._runtime_root_record(node)
            ledger.setdefault("runtime_root_surfaces", []).append(root_record)
            event = {
                "id": txid,
                "transaction_id": txid,
                "kind": "root_surface",
                "action": "add",
                "status": "accepted",
                "surface_id": graph_id,
                "build_name": build_name,
                "category": "wall",
                "description": description,
                "reason": reason,
                "target_region_norm": list(region),
                "attempt_idx": int(getattr(self, "attempt_idx", 1)),
                "scene_graph_revision": revision,
                "added_relationships": [
                    {
                        "relationship_id": str(rel.get("relationship_id")),
                        "type": rel.get("type"),
                        "a": rel.get("a"),
                        "b": rel.get("b"),
                        "status": rel.get("status"),
                        "enforcement": rel.get("enforcement"),
                    }
                    for rel in graph.get("relationships", [])
                    if rel.get("relationship_id") in set(added_relationships)
                ],
            }
            ledger.setdefault("events", []).append(event)
            ledger["next_transaction_id"] = txid + 1
            self._commit_runtime_root_edit(
                graph=graph,
                ledger=ledger,
                kind="build_root_surface",
                txid=txid,
                graph_id=graph_id,
            )
            rendered = self._runtime_root_render()
            if rendered.get("status") != "success" or not rendered.get(
                "output", {}
            ).get("image"):
                raise RuntimeError("post-commit render failed or returned no image")
        except Exception as exc:  # noqa: BLE001 - every failure rolls all artifacts back
            try:
                self._rollback_runtime_root_edit(
                    history_n=history_n,
                    ledger_history_n=ledger_history_n,
                    graph_history_n=graph_history_n,
                    edit_meta_n=edit_meta_n,
                    ledger_before=ledger_before,
                    graph_before=graph_before,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                return fail(
                    f"{exc}; CRITICAL transaction rollback failed: {rollback_exc}",
                    audit=False,
                )
            return fail(str(exc))

        runtime_edit = {
            "action": "add",
            **self._runtime_root_record(node),
        }
        output = rendered.setdefault("output", {})
        output.setdefault("text", []).insert(
            0,
            f"build_root_surface committed transaction {txid}: registered and built "
            f"{graph_id} as exact Blender root '{build_name}'. Refine this CURRENT "
            "REGISTERED root with execute_and_evaluate, then rerun check_rules_enforced.",
        )
        output.update(
            transaction=event,
            runtime_root_surface_edit=runtime_edit,
            scene_graph_revision=revision,
            graph_delta={
                "added_nodes": [graph_id],
                "removed_nodes": [],
                "added_relationships": added_relationships,
                "removed_relationships": [],
            },
        )
        return rendered

    def remove_root_surface(self, *, surface: str, reason: str) -> dict[str, object]:
        if self.root_stage_name != "initializer":
            return {
                "status": "error",
                "output": {
                    "text": ["remove_root_surface is an initializer-stage tool"]
                },
            }
        try:
            self._assert_no_incomplete_runtime_mutation()
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "output": {"text": [f"remove_root_surface blocked by recovery: {exc}"]},
            }
        reason = str(reason or "").strip()
        if not reason:
            return {
                "status": "error",
                "output": {"text": ["remove_root_surface reason must be non-empty"]},
            }
        txid = int(self._initializer_ledger.get("next_transaction_id") or 1)
        graph_path = self._scene_graph_path()
        if graph_path is None or not graph_path.exists():
            return {
                "status": "error",
                "output": {"text": ["active scene_graph.json is unavailable"]},
            }
        try:
            graph = json.loads(graph_path.read_text())
        except (OSError, ValueError, TypeError) as exc:
            return {
                "status": "error",
                "output": {"text": [f"active scene graph is invalid: {exc}"]},
            }
        node, error = self._resolve_graph_root(graph, str(surface))
        if node is None:
            return {"status": "error", "output": {"text": [error]}}
        if node.get("runtime_added") is not True or node.get("source") != (
            "initializer_build_root_surface"
        ):
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"remove_root_surface refused: {node['id']} is a preprocessing-"
                        "provided root. Repair it with execute_and_evaluate; only roots "
                        "introduced by build_root_surface may be removed."
                    ]
                },
            }
        graph_id = str(node["id"])
        build_name = str(node.get("build_name") or "")
        if not build_name:
            return {
                "status": "error",
                "output": {
                    "text": [f"runtime root {graph_id} has no exact build_name"]
                },
            }
        from lib.tools.geometry.surface_relations import surface_build_name

        protected_root_bindings = [
            {
                "id": str(other["id"]),
                "build_name": str(
                    other.get("build_name") or surface_build_name(other["id"])
                ),
            }
            for other in graph.get("nodes", [])
            if other.get("kind") == "root_surface"
            and other.get("id")
            and other.get("id") != graph_id
        ]
        pipeline_names = sorted(set(getattr(self, "_pipeline_object_names", []) or []))
        code = f"""import bpy
root = bpy.data.objects.get({build_name!r})
if root is None:
    raise RuntimeError('runtime root {build_name} is absent from Blender')
if str(root.get('grase_graph_id', '')) != {graph_id!r} or root.get('grase_root_source') != 'initializer_runtime':
    raise RuntimeError('runtime root binding/custom properties do not match the graph')
todo = []
stack = [root]
while stack:
    current = stack.pop()
    stack.extend(list(current.children))
    todo.append(current)
pipeline_names = set({pipeline_names!r})
protected = sorted(obj.name for obj in todo if obj.name in pipeline_names)
if protected:
    raise RuntimeError('refusing to remove runtime wall with imported-object descendants: ' + ', '.join(protected))
root_bindings = {protected_root_bindings!r}
root_names = {{row['build_name'] for row in root_bindings}}
root_ids = {{row['id'] for row in root_bindings}}
nested_roots = sorted(
    obj.name for obj in todo[1:]
    if obj.name in root_names or str(obj.get('grase_graph_id', '')) in root_ids
)
if nested_roots:
    raise RuntimeError('refusing to remove runtime wall with other registered root descendants: ' + ', '.join(nested_roots))
for current in reversed(todo):
    bpy.data.objects.remove(current, do_unlink=True)
"""
        self._ensure_undo_base()
        ledger_before = copy.deepcopy(self._initializer_ledger)
        graph_before = self._read_scene_graph_bytes()
        history_n = len(self.edit_history)
        ledger_history_n = len(self._ledger_history)
        graph_history_n = len(getattr(self, "_graph_history", []))
        edit_meta_n = len(self._edit_meta)
        had_last_code = hasattr(self, "_last_code")
        last_code = getattr(self, "_last_code", None)
        had_remove_allow = hasattr(self, "_initializer_remove_root_allow")
        old_remove_allow = getattr(self, "_initializer_remove_root_allow", None)
        self._initializer_remove_root_allow = {
            "graph_id": graph_id,
            "build_name": build_name,
        }
        edit = None
        execute_error = None
        try:
            edit = self.execute(code, render=False)
        except Exception as exc:  # noqa: BLE001 - trusted edit is transactional
            execute_error = exc
        finally:
            if had_remove_allow:
                self._initializer_remove_root_allow = old_remove_allow
            else:
                self.__dict__.pop("_initializer_remove_root_allow", None)
            if had_last_code:
                self._last_code = last_code
            else:
                self.__dict__.pop("_last_code", None)
            if len(self._edit_meta) > edit_meta_n:
                self._edit_meta[-1]["code_base"] = last_code if had_last_code else None
        if (
            execute_error is not None
            or not isinstance(edit, dict)
            or edit.get("status") != "success"
        ):
            message = (
                str(execute_error)
                if execute_error is not None
                else (edit or {})
                .get("output", {})
                .get("text", ["Blender edit failed"])[0]
            )
            try:
                self._rollback_runtime_root_edit(
                    history_n=history_n,
                    ledger_history_n=ledger_history_n,
                    graph_history_n=graph_history_n,
                    edit_meta_n=edit_meta_n,
                    ledger_before=ledger_before,
                    graph_before=graph_before,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            "remove_root_surface failed ("
                            f"{message}); CRITICAL transaction rollback failed: "
                            f"{rollback_exc}"
                        ]
                    },
                }
            try:
                self._record_runtime_root_rejection(txid, "remove", reason, message)
            except Exception:
                pass
            return {
                "status": "error",
                "output": {
                    "text": [
                        "remove_root_surface rejected; blend and scene graph were left "
                        "unchanged: " + message
                    ]
                },
            }
        try:
            removed_rel_rows = [
                {
                    "relationship_id": str(r.get("relationship_id")),
                    "type": r.get("type"),
                    "a": r.get("a"),
                    "b": r.get("b"),
                    "status": r.get("status"),
                    "enforcement": r.get("enforcement"),
                }
                for r in graph.get("relationships", []) or []
                if graph_id in {r.get("a"), r.get("b")}
            ]
            removed_rels = [row["relationship_id"] for row in removed_rel_rows]
            graph["nodes"] = [n for n in graph.get("nodes", []) if n is not node]
            graph["roots"] = [rid for rid in graph.get("roots", []) if rid != graph_id]
            graph["relationships"] = [
                r
                for r in graph.get("relationships", []) or []
                if graph_id not in {r.get("a"), r.get("b")}
            ]
            if isinstance(graph.get("relationship_audit"), list):
                # Audit is historical: never erase the relationship that existed.
                # Mark a separate lifecycle (leaving semantic status/enforcement intact).
                for audit_rel in graph["relationship_audit"]:
                    if graph_id not in {audit_rel.get("a"), audit_rel.get("b")}:
                        continue
                    audit_rel["runtime_lifecycle_status"] = "removed"
                    audit_rel["runtime_removed_transaction_id"] = txid
                    audit_rel.setdefault("lifecycle_provenance", []).append(
                        {
                            "source": "initializer_remove_root_surface",
                            "action": "removed",
                            "transaction_id": txid,
                            "reason": reason,
                        }
                    )
            revision = int(graph.get("scene_graph_revision") or 0) + 1
            graph["scene_graph_revision"] = revision
            self._refresh_relationship_counts(graph)
            ledger = copy.deepcopy(ledger_before)
            ledger["runtime_root_surfaces"] = [
                row
                for row in ledger.get("runtime_root_surfaces", [])
                if str(row.get("id")) != graph_id
            ]
            event = {
                "id": txid,
                "transaction_id": txid,
                "kind": "root_surface",
                "action": "remove",
                "status": "accepted",
                "surface_id": graph_id,
                "build_name": build_name,
                "reason": reason,
                "attempt_idx": int(getattr(self, "attempt_idx", 1)),
                "scene_graph_revision": revision,
                "removed_relationships": removed_rel_rows,
            }
            ledger.setdefault("events", []).append(event)
            ledger["next_transaction_id"] = txid + 1
            self._commit_runtime_root_edit(
                graph=graph,
                ledger=ledger,
                kind="remove_root_surface",
                txid=txid,
                graph_id=graph_id,
            )
            rendered = self._runtime_root_render()
            if rendered.get("status") != "success" or not rendered.get(
                "output", {}
            ).get("image"):
                raise RuntimeError("post-removal render failed or returned no image")
        except Exception as exc:  # noqa: BLE001
            try:
                self._rollback_runtime_root_edit(
                    history_n=history_n,
                    ledger_history_n=ledger_history_n,
                    graph_history_n=graph_history_n,
                    edit_meta_n=edit_meta_n,
                    ledger_before=ledger_before,
                    graph_before=graph_before,
                )
            except Exception as rollback_exc:  # noqa: BLE001
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"remove_root_surface failed ({exc}); CRITICAL transaction "
                            f"rollback failed: {rollback_exc}"
                        ]
                    },
                }
            try:
                self._record_runtime_root_rejection(txid, "remove", reason, str(exc))
            except Exception:
                pass
            return {
                "status": "error",
                "output": {
                    "text": [
                        "remove_root_surface rejected and rolled back: " + str(exc)
                    ]
                },
            }

        self._coverage_bypassed.discard(build_name)
        try:
            (self.render_path.parent / "tmp" / "coverage_bypass.json").write_text(
                json.dumps(sorted(self._coverage_bypassed))
            )
        except Exception:  # noqa: BLE001 - best-effort mirror of in-memory state
            pass
        runtime_edit = {
            "action": "remove",
            "id": graph_id,
            "build_name": build_name,
            "reason": reason,
            "transaction_id": txid,
            "scene_graph_revision": revision,
        }
        output = rendered.setdefault("output", {})
        output.setdefault("text", []).insert(
            0,
            f"remove_root_surface committed transaction {txid}: removed runtime root "
            f"{graph_id} ('{build_name}'). Rerun check_rules_enforced.",
        )
        output.update(
            transaction=event,
            runtime_root_surface_edit=runtime_edit,
            scene_graph_revision=revision,
            graph_delta={
                "added_nodes": [],
                "removed_nodes": [graph_id],
                "added_relationships": [],
                "removed_relationships": removed_rels,
            },
        )
        return rendered

    def _rule_no_penetration(self, data: dict) -> tuple[bool, str]:
        """Rule (1): no interpenetrating bodies. Object-involving pairs are always reported; a
        surface<->surface pair is reported ONLY when one side is the MAIN SUPPORT penetrated by
        more than 2cm, ONLY in the initializer stage (surfaces are frozen after it), and ONLY when
        the two surfaces have NO listed relationship (a related pair's contact is owned by the
        RELATIONSHIPS rule -- see _related_surface_pairs)."""
        grouped_centroids = data.get("surface_centroids", {}) or {}
        grouped_normals = data.get("surface_normals", {}) or {}
        # Collision detection deliberately uses each complete grouped body, but repair
        # ownership and push direction belong to the exact root surface.  A permitted
        # frame/backdrop child must not move a canonical wall's centroid/PCA enough to
        # make it look free or reverse the separating direction.
        centroids = dict(grouped_centroids)
        centroids.update(data.get("surface_plane_centroids", {}) or {})
        normals = dict(grouped_normals)
        normals.update(data.get("surface_plane_normals", {}) or {})
        main, _ = identify_main_support(data.get("bodies", []))
        related = (
            self._related_surface_pairs()
            if self.root_stage_name == "initializer"
            else set()
        )
        # Class-aware allowance from the ONE contact ladder (contact_policy): the
        # probe's real separating depth vs the pair's class (resting on its support,
        # lateral same-support sibling, unrelated object/surface). No support map ->
        # strictest classes, never looser.
        support = self._support_body_map()
        reported, classes = [], {}
        for p in data.get("penetrating_pairs", []):
            a, b = p["a"], p["b"]
            depth = float(p.get("depth", 0.0))
            cls = contact_class(a, b, support, p.get("separation_direction"))
            classes[frozenset((a, b))] = cls
            if cls != SURFACE_SURFACE:
                if depth > tolerance_m(cls):
                    reported.append(p)
            elif (
                self.root_stage_name == "initializer"
                and main in (a, b)
                and depth > MAIN_SUPPORT_SURFACE_TOL_M
                and frozenset((a, b))
                not in related  # not governed by the relationships rule
            ):
                reported.append(p)  # main support crossing an UNRELATED surface > 2cm
        if not reported:
            return (
                True,
                "no penetration: PASS (no interpenetrating objects or surfaces)",
            )
        lines = [
            "no penetration: FAIL — interpenetrating pairs (fix unless a genuine "
            "resting contact):"
        ]
        from lib.tools.blender.script_generators import _out_dir, _yaw_delta_deg

        runs = data.get("surface_runs", {}) or {}
        for p in reported:
            a, b = p["a"], p["b"]
            # dual label: the fix here is a CODE edit (which needs the mesh name), but in
            # composition the agent may equally reach for move() (which needs the id)
            cls = classes[frozenset((a, b))]
            line = (
                f"  - {self._label(a)} <-> {self._label(b)} "
                f"[{cls.replace('_', ' ')}: {float(p.get('depth', 0.0)) * 1000:.1f}mm"
                f" > {tolerance_m(cls) * 1000:.0f}mm allowed]"
            )
            code_route = self.root_stage_name == "initializer" and self._has_capability(
                "initializer_code_transactions"
            )  # gpt6_v1: no nudge_object; object translations are execute_and_evaluate transactions
            need_mm = (float(p.get("depth", 0.0)) + tolerance_m(cls)) * 1000.0
            if p.get("suggested_repairs"):
                line += (
                    f"; measured small-clearance candidates: {p['suggested_repairs']} "
                    + ("(revalidate support/contact after the correction)" if code_route
                       else "(revalidate support/contact after nudging)")
                )
            leaf = self._prefer_move(a, b, centroids, main=main, normals=normals)
            mixed = a.startswith("obj_") != b.startswith("obj_")
            if p.get("depth_source") == "inside_depth":
                # Nested pair (object inside a tray / drawer / bowl, or sunk deep into
                # a slab): the depth is the mesh intrusion into the container's solid,
                # not a separating translation. A lift-out is never the fix here.
                container = p.get("nested_in") or (b if leaf == a else a)
                line += (
                    f" — nested inside {self._label(container)}: {float(p.get('depth', 0.0)) * 1000:.1f} mm "
                    "of mesh intrusion. Do NOT lift it out of its container; re-settle it, "
                    "or raise/lower it by at most that amount, keeping it inside"
                )
                lines.append(line)
                continue
            if (
                self.root_stage_name == "initializer"
                and a.startswith("obj_")
                and b.startswith("obj_")
            ):
                target = leaf or b
                if code_route:
                    line += (
                        f" — translate {target} clear with an execute_and_evaluate object "
                        f"transaction: a rigid translation of at least {need_mm:.0f} mm on ONE "
                        "affected object/support stack along the smallest separating direction, "
                        "resting preserved (the transaction simulates that object's hierarchy); "
                        "do not rotate or scale either object"
                    )
                else:
                    line += (
                        f" — use nudge_object(object='{target}', reason='penetration', "
                        "translation=[dx,dy,dz]) on ONE affected object/support stack, "
                        "with the smallest separating translation that preserves resting; "
                        "do not rotate or scale either object"
                    )
                lines.append(line)
                continue
            if self.root_stage_name == "initializer" and mixed:
                surf = b if a.startswith("obj_") else a
                obj = a if a.startswith("obj_") else b
                out = None
                n = normals.get(surf)
                box = next(
                    (bd for bd in data.get("bodies", []) if bd["name"] == obj), None
                )
                if n is not None and box is not None and centroids.get(surf):
                    c_obj = [(box["lo"][i] + box["hi"][i]) / 2.0 for i in range(3)]
                    c_s = centroids[surf]
                    nx, ny = float(n[0]), float(n[1])
                    h = math.hypot(nx, ny)
                    if h > 1e-6:
                        nx, ny = nx / h, ny / h
                        if (c_obj[0] - c_s[0]) * nx + (c_obj[1] - c_s[1]) * ny < 0:
                            nx, ny = -nx, -ny
                        out = (nx, ny)
                direction = (
                    f" along world ({out[0]:+.2f}, {out[1]:+.2f}, 0)"
                    if out is not None
                    else " away from the surface"
                )
                if code_route:
                    line += (
                        f" — if '{surf}' is visibly the wrong surface, repair it. Otherwise "
                        f"translate {obj} with an execute_and_evaluate object transaction by at "
                        f"least {need_mm:.0f} mm horizontally{direction} (rigid translation only, "
                        "keep it resting; the transaction simulates its hierarchy); never move an "
                        "object to hide a support/wall error"
                    )
                else:
                    line += (
                        f" — if '{surf}' is visibly the wrong surface, repair it. Otherwise "
                        f"use nudge_object(object='{obj}', reason='penetration', "
                        f"translation=[dx,dy,0]) with the smallest horizontal translation"
                        f"{direction}; keep the object resting and do not use an object nudge "
                        "to hide a support/wall error"
                    )
                lines.append(line)
                continue
            if leaf:
                line += f" — prefer moving {self._label(leaf)}"
                # OBJECT penetrating a SURFACE: spell out a HORIZONTAL slide. The bare
                # "prefer moving obj_x" let an agent clear a fork<->wall overlap by
                # TILTING the fork into mid-air (0709_eval3_raw1) — floating passes
                # every other check, so the advice must forbid lifting outright.
                if leaf.startswith("obj_") and (
                    not a.startswith("obj_") or not b.startswith("obj_")
                ):
                    other = b if leaf == a else a
                    out = None
                    n = normals.get(other)
                    box = next(
                        (bd for bd in data.get("bodies", []) if bd["name"] == leaf),
                        None,
                    )
                    if n is not None and box is not None and centroids.get(other):
                        c_obj = [(box["lo"][i] + box["hi"][i]) / 2.0 for i in range(3)]
                        c_s = centroids[other]
                        nx, ny = float(n[0]), float(n[1])
                        h = math.hypot(nx, ny)
                        if h > 1e-6:  # horizontal component of the surface normal
                            nx, ny = nx / h, ny / h
                            d = (c_obj[0] - c_s[0]) * nx + (c_obj[1] - c_s[1]) * ny
                            if d < 0:
                                nx, ny = -nx, -ny  # push AWAY from the surface
                            out = (nx, ny)
                    depth = max(float(p.get("depth", 0.0)), 0.01)
                    if out is not None:
                        line += (
                            f": SLIDE it horizontally out of '{other}' by "
                            f">={depth:.2f}m along world ({out[0]:+.2f}, {out[1]:+.2f}, 0) "
                            f"— keep its z (do NOT lift or tilt it; it must stay resting "
                            f"on its support)"
                        )
                    else:
                        line += (
                            f": SLIDE it horizontally out of '{other}' — keep its z "
                            f"(do NOT lift or tilt it; it must stay resting on its support)"
                        )
                # Surface pair where the movable body is the MAIN SUPPORT: spell out the fix with
                # the sign-resolved direction (the PCA normal's sign is a coin flip) + exact yaw.
                if (
                    leaf == main
                    and not a.startswith("obj_")
                    and not b.startswith("obj_")
                ):
                    other = b if a == main else a
                    out = None
                    if centroids.get(main) and centroids.get(other):
                        out = _out_dir(
                            centroids[main], centroids[other], normals.get(other)
                        )
                    depth = float(p.get("depth", 0.0))
                    # exact directional distance (projection spans along the other
                    # surface's normal) beats the AABB min-axis depth for yawed bodies
                    from lib.tools.blender.script_generators import (
                        directional_pull_depth,
                    )

                    d_dir = directional_pull_depth(
                        (p.get("spans") or {}).get(other), main, other, out
                    )
                    if d_dir is not None:
                        depth = max(d_dir, 0.01)
                    delta = _yaw_delta_deg(runs.get(main), runs.get(other))
                    rot = (
                        f", or rotate it by {delta:+.0f}° around the world Z (up) axis "
                        f"pivoting about its own centre (objects on it do NOT rotate with "
                        f"it — re-translate afterwards instead of undoing)"
                        if delta is not None and abs(delta) >= 3.0
                        else ", or rotate it around the world Z (up) axis"
                    )
                    if out is not None:
                        line += (
                            f": PULL it out of '{other}' by >={depth:.2f}m along world "
                            f"({out[0]:+.2f}, {out[1]:+.2f}, 0){rot} "
                            f"(its top stays at z=0; do not lift or tilt it)"
                        )
                    else:
                        line += (
                            f": PULL it out of '{other}' horizontally{rot} "
                            f"(its top stays at z=0; do not lift or tilt it)"
                        )
            lines.append(line)
        return False, "\n".join(lines)

    def check_rules_enforced(self) -> dict[str, object]:
        """Check the stage-specific scene rules before finishing (read-only).

        Composition checks investigation coverage first (short-circuiting) and then
        penetration. Initializer runs the three-level structure/pose/contact ladder.
        Other stages, when called directly, check penetration only.
        """
        blocked = self._guard_post_flip_tool_call("check_rules_enforced")
        if blocked is not None:
            if self.root_stage_name == "initializer":
                constraints: list[dict] = []
                relationships: list[dict] = []
                try:
                    graph_path = self._scene_graph_path()
                    if graph_path is None or not graph_path.is_file():
                        raise RuntimeError("active scene_graph.json is unavailable")
                    graph = json.loads(graph_path.read_text())
                    artifact = self._load_initializer_constraint_artifact(graph)
                    constraints = self._deferred_constraint_results(artifact)
                    relationships = self._source_relationship_evidence(graph)
                except Exception:  # noqa: BLE001 - guard response still fails closed
                    pass
                output = blocked.setdefault("output", {})
                output["rules_all_pass"] = False
                output["relationship_evidence"] = relationships
                output["constraint_results"] = constraints
                output["rule_evidence"] = {
                    "schema_version": 2,
                    "overall_status": "deferred",
                    "evaluated_levels": [],
                    "deferred_levels": [
                        "1/3 STRUCTURE",
                        "2/3 POSE",
                        "3/3 CONTACT",
                    ],
                    "yaw_evidence": {},
                    "relationship_evidence": relationships,
                    "constraint_results": constraints,
                }
            return blocked
        try:
            # Fast-fail: the composition end-gate also requires every object to have been
            # investigated >= once. That check is in-memory, whereas the penetration check
            # costs a Blender pass + an Isaac sync/contacts probe. If coverage already
            # fails there is nothing to fix in the geometry yet, so report it now and skip
            # the expensive penetration check (the agent must investigate first; penetration
            # is re-checked once coverage passes).
            if self.root_stage_name == "composition":
                cov_ok, cov_msg = self._rule_investigation_coverage()
                if not cov_ok:
                    return {
                        "status": "success",
                        "output": {
                            "text": [
                                "check_rules_enforced: FAIL — fix the rule(s) below, "
                                f"then call again:\n[1] {cov_msg}\n(penetration not "
                                "checked yet — it runs once every object has been "
                                "investigated)"
                            ],
                            "rules_all_pass": False,
                            "post_flip_followup": self._post_flip_followup_evidence(),
                        },
                    }
            if self.root_stage_name == "initializer":
                # A new ordered-ladder evaluation supersedes every prior residual-yaw
                # request. Only a fresh full pass below may mint another token.
                self._yaw_advisory_requirement = None
                self._yaw_advisory_binding = None
                self._yaw_resolution = None
                self._constraint_results = []
                try:
                    graph_path = self._scene_graph_path()
                    if graph_path is None or not graph_path.is_file():
                        raise RuntimeError("active scene_graph.json is unavailable")
                    active_graph = json.loads(graph_path.read_text())
                    active_artifact = self._load_initializer_constraint_artifact(
                        active_graph
                    )
                    self._constraint_results = self._deferred_constraint_results(
                        active_artifact
                    )
                    self._relationship_evidence = self._source_relationship_evidence(
                        active_graph
                    )
                except Exception as exc:  # noqa: BLE001 - no legacy fallback
                    message = (
                        "initializer relationship constraints: UNVERIFIED — "
                        f"{exc}. Rerun preprocessing; raw relationship rows are not "
                        "accepted as a runtime fallback."
                    )
                    rule_evidence = {
                        "schema_version": 2,
                        "overall_status": "error",
                        "evaluated_levels": [],
                        "deferred_levels": [
                            "1/3 STRUCTURE",
                            "2/3 POSE",
                            "3/3 CONTACT",
                        ],
                        "yaw_evidence": {},
                        "relationship_evidence": [],
                        "constraint_results": [],
                    }
                    return {
                        "status": "error",
                        "output": {
                            "text": [message],
                            "rules_all_pass": False,
                            "relationship_evidence": [],
                            "constraint_results": [],
                            "rule_evidence": rule_evidence,
                        },
                    }
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            out_path = tmp_dir / "penetration.json"
            main_name, main_form, main_floor = (
                self._main_support_info()
                if self.root_stage_name == "initializer"
                else (None, None, None)
            )
            script = generate_penetration_script(str(out_path), main_name=main_name)
            self.count += 1
            code_file = self.script_path / f"{self.count}.py"
            with open(code_file, "w") as f:
                f.write(script)
            if out_path.exists():
                out_path.unlink()  # don't trust a stale result if the check fails
            # persist_scene=False (2026-09-15): the check is read-only; the wrapper's
            # trailing save rewrote the 26 MB blend with different bytes every time and
            # spoiled the geometry-dump reuse for the next transaction.
            success, imgs, stdout, stderr = self._execute_blender(
                str(code_file), persist_scene=False
            )
            # Trust the file (like get_scene_info): the read-only script writes it and
            # exits before the wrapper's trailing render, so a render crash is irrelevant.
            if not out_path.exists():
                structured = {}
                if self.root_stage_name == "initializer":
                    yaw = bind_compiled_yaw_evidence(
                        main_name,
                        {
                            "status": "unknown",
                            "reason": "penetration_geometry_dump_failed",
                            "geometry_source": "main_top_hull",
                            "rectangularity": None,
                        },
                        None,
                        active_artifact.get("source_yaw_evidence") or {},
                    )
                    structured = {
                        "yaw_evidence": yaw,
                        "relationship_evidence": self._relationship_evidence,
                        "constraint_results": self._constraint_results,
                        "rule_evidence": {
                            "schema_version": 2,
                            "overall_status": "error",
                            "evaluated_levels": [],
                            "deferred_levels": [
                                "1/3 STRUCTURE",
                                "2/3 POSE",
                                "3/3 CONTACT",
                            ],
                            "yaw_evidence": yaw,
                            "relationship_evidence": self._relationship_evidence,
                            "constraint_results": self._constraint_results,
                        },
                    }
                return {
                    "status": "error",
                    "output": {
                        "text": ["Error: " + (stderr or stdout or "Rule check failed")],
                        **structured,
                    },
                }
            with open(out_path) as f:
                data = json.load(f)
            if self.root_stage_name != "initializer":
                # Non-initializer stages keep a flat check: penetration always; the
                # COMPOSITION stage additionally gates on investigation coverage
                # (every object must be investigated at least once before end).
                # Composition gates on SIMULATOR geometry: convex-hull
                # overlap pairs from the Isaac session (verify_sync'd first),
                # filtered by isaac_penetration_gate_pairs. The Blender-side BVH
                # pairs already in data are the MESH-intersection cross-check that
                # bounds the support-edge exemption (captured before the
                # overwrite). Any failure falls back to those Blender pairs.
                gate_notes: list[str] = []
                if self.root_stage_name == "composition":
                    try:
                        session = self._pose_session_get()
                        if session.physics is not None:  # None = disabled after error
                            mesh_keys = {
                                frozenset((p["a"], p["b"]))
                                for p in data.get("penetrating_pairs", [])
                                if p["a"].startswith("obj_")
                                or p["b"].startswith("obj_")
                            }
                            # Isaac<->blend invariant AT THE GATE: heal any hull
                            # drift before trusting the session's contact pairs —
                            # 0720_orinit3_abc1's desync made this gate PASS a
                            # croissant buried 30mm in the tray.
                            try:
                                session.physics.verify_sync(
                                    session, list(session.physics._names)
                                )
                            except Exception:  # noqa: BLE001 - advisory
                                pass
                            raw_pairs = session.physics.contact_pairs()
                            data["penetrating_pairs"] = isaac_penetration_gate_pairs(
                                raw_pairs,
                                self._support_body_map(),
                                mesh_keys,
                            )
                            # mesh-vs-hull discrepancy: a BVH mesh crossing with NO
                            # hull overlap at verified-synced poses. Warn, never
                            # fail: the BVH 'depth' is a thin-axis AABB proxy that
                            # reads rim-height values for correctly-seated objects
                            # (donut-in-tray 20.8mm), so it cannot gate by itself.
                            hull_keys = {frozenset((p["a"], p["b"])) for p in raw_pairs}
                            for k in sorted(
                                mesh_keys - hull_keys,
                                key=lambda s: tuple(sorted(s)),
                            ):
                                a, b = sorted(k)
                                gate_notes.append(
                                    f"note: the visual meshes of {self._label(a)} "
                                    f"<-> {self._label(b)} "
                                    "intersect while their collision hulls do not "
                                    "— eyeball that pair in the render."
                                )
                    except Exception:  # noqa: BLE001
                        pass
                checks = [self._rule_no_penetration(data)]
                if self.root_stage_name == "composition":
                    checks.append(self._rule_investigation_coverage())
                ok = all(c[0] for c in checks)
                header = (
                    "check_rules_enforced: ALL RULES PASS — you may call end."
                    if ok
                    else "check_rules_enforced: FAIL — fix the rule(s) below, then call again:"
                )
                body = "\n".join(
                    [f"[{i + 1}] {m}" for i, (_, m) in enumerate(checks)] + gate_notes
                )
                output = {"text": [f"{header}\n{body}"], "rules_all_pass": ok}
                if self.root_stage_name == "composition":
                    output["post_flip_followup"] = self._post_flip_followup_evidence()
                return {"status": "success", "output": output}
            # Initialize structured evidence before the ordered ladder.  A STRUCTURE
            # failure therefore still reports what is known about yaw geometry and
            # explicitly records that POSE was deferred.
            run_info = dict(data.get("main_support_run") or {})
            main_is_floor = self._main_support_is_floor(main_name)
            applicability = {
                "status": (
                    "not_applicable"
                    if main_is_floor
                    else run_info.get("status") or "unknown"
                ),
                "reason": (
                    "room_floor_policy"
                    if main_is_floor
                    else run_info.get("reason") or "main_top_run_evidence_missing"
                ),
                "geometry_source": (
                    "room_floor_policy"
                    if main_is_floor
                    else run_info.get("geometry_source") or "main_top_hull"
                ),
                "rectangularity": run_info.get("rectangularity"),
            }
            self._yaw_evidence = bind_compiled_yaw_evidence(
                main_name,
                applicability,
                (data.get("main_support_run") or {}).get("run"),
                active_artifact.get("source_yaw_evidence") or {},
            )
            if applicability["status"] != "not_applicable":
                self._yaw_evidence["verdict"].update(
                    status="not_evaluated", reason="pose_level_not_reached"
                )
            evaluated_levels: list[dict] = []

            def _structured_gate(overall_status: str, deferred: list[str]) -> dict:
                return {
                    "schema_version": 2,
                    "overall_status": overall_status,
                    "evaluated_levels": evaluated_levels,
                    "deferred_levels": deferred,
                    "yaw_evidence": self._yaw_evidence,
                    "relationship_evidence": self._relationship_evidence,
                    "constraint_results": self._constraint_results,
                }

            # Initializer: an ORDERED LADDER, reported one level at a time. Dumping every
            # rule at once made the agent juggle 3-8 simultaneous directives (compound edits
            # built on stale increments, cross-rule tug-of-war); each level's fixes also
            # only make sense once the previous level holds (structure -> pose -> contact).
            levels = [
                (
                    "1/3 STRUCTURE (object integrity + authorized object corrections + valid root transforms + the main support itself)",
                    lambda: [
                        self._rule_object_integrity(data),
                        self._rule_surface_geometry(data),
                        self._rule_registered_root_inventory(),
                        self._rule_runtime_root_bindings(),
                        main_support_top_report(
                            data.get("bodies", []),
                            main_name=main_name,
                            form=main_form,
                            floor_name=main_floor,
                        ),
                        main_support_connected_report(data.get("main_support_islands")),
                        main_support_bottom_report(
                            data.get("bodies", []), main_name, main_form
                        ),
                        self._rule_structure_constraints(data, main_name),
                    ],
                ),
                (
                    "2/3 POSE (reference-view coverage vs the photo)",
                    lambda: [
                        self._rule_coverage(
                            tmp_dir, main_name, data, main_form=main_form
                        ),
                        self._rule_objects_inboard(data, main_name),
                        self._rule_pose_constraints(data, main_name),
                    ],
                ),
                (
                    "3/3 CONTACT (penetration + resting + relationships)",
                    lambda: [
                        self._rule_no_penetration(data),
                        objects_resting_report(
                            data.get("bodies", []), self._support_body_map()
                        ),
                        self._rule_contact_constraints(data, main_name=main_name),
                    ],
                ),
            ]
            for idx, (title, run) in enumerate(levels):
                checks = run()
                if len(checks) != len(_INITIALIZER_LEVEL_CHECK_NAMES[idx]):
                    raise RuntimeError(
                        f"initializer rule evidence schema mismatch at level {idx + 1}: "
                        f"{len(checks)} checks vs "
                        f"{len(_INITIALIZER_LEVEL_CHECK_NAMES[idx])} names"
                    )
                level_ok = all(check_ok for check_ok, _ in checks)
                evaluated_levels.append(
                    {
                        "level": idx + 1,
                        "name": title.split(" (")[0].split(" ", 1)[-1].lower(),
                        "status": "pass" if level_ok else "fail",
                        "checks": [
                            {
                                "name": _INITIALIZER_LEVEL_CHECK_NAMES[idx][check_idx],
                                "status": "pass" if check_ok else "fail",
                                "message": message,
                            }
                            for check_idx, (check_ok, message) in enumerate(checks)
                        ],
                        "constraint_results": [
                            row
                            for row in self._constraint_results
                            if row.get("stage") == ("STRUCTURE", "POSE", "CONTACT")[idx]
                        ],
                    }
                )
                if level_ok:
                    continue
                deferred_names = [t.split(" (")[0] for t, _ in levels[idx + 1 :]]
                later = ", ".join(deferred_names)
                tail = (
                    f"\n(level(s) {later} not checked yet — they only make sense once "
                    f"this level holds; fix ONLY the above, then call again.)"
                    if later
                    else "\n(fix the above, then call again.)"
                )
                msg = (
                    "\n".join(
                        [f"check_rules_enforced: FAIL at level {title}:"]
                        + [f"[{i + 1}] {m}" for i, (_, m) in enumerate(checks)]
                    )
                    + tail
                )
                imgs: list = []
                # POSE-level failure: attach the tinted GT-vs-render pair per
                # failing surface + advertise the bypass tool — coverage's ground
                # truth can itself be wrong (groot2's under-segmented curtain)
                if title.startswith("2/3 POSE") and self._coverage_last_failing:
                    failing = sorted(self._coverage_last_failing)
                    imgs = self._coverage_mismatch_visual(failing)
                    if imgs:
                        runtime_builds = set()
                        try:
                            from lib.tools.geometry.surface_relations import (
                                surface_build_name,
                            )

                            graph = json.load(
                                open(os.path.join(self.moge_dir, "scene_graph.json"))
                            )
                            runtime_builds = {
                                str(n.get("build_name") or surface_build_name(n["id"]))
                                for n in graph.get("nodes", [])
                                if n.get("kind") == "root_surface"
                                and n.get("runtime_added") is True
                                and n.get("id")
                            }
                        except Exception:  # noqa: BLE001 - bypass itself fails closed
                            runtime_builds = set()
                        bypassable = [n for n in failing if n not in runtime_builds]
                        msg += (
                            "\nThe attached images show each failing surface "
                            "twice, tinted the same color: IMAGE LEFT = your "
                            "render with your built surface; IMAGE RIGHT = the "
                            "target photo with the surface's GT segmentation "
                            "mask. First assume the report is right and fix "
                            "your build. If instead the PHOTO-side tint is "
                            "visibly wrong (it misses part of the real surface, "
                            "or covers something else), the rule is grading you "
                            "against a bad mask. For a PREPROCESSED REGISTERED wall, "
                            "bypass is also "
                            "appropriate when the target genuinely does not depict "
                            "that wall; it does not edit the scene graph or waive the "
                            "requirement to build it. Valid bypass IDs for this report: "
                            + (", ".join(f"'{name}'" for name in bypassable) or "none")
                            + ". Call "
                            + 'bypass(["<specific surface>"]) only for each PREPROCESSED '
                            "surface whose PHOTO-side mask is visibly wrong, or whose "
                            "preprocessed wall is genuinely absent from the target; do not bypass "
                            "the other failing surfaces. Use it sparingly: a bypass "
                            "permanently removes this check for the stage and "
                            "is recorded."
                        )
                        runtime_failing = sorted(set(failing) & runtime_builds)
                        if runtime_failing:
                            msg += (
                                "\nRuntime-added roots cannot be bypassed: "
                                + ", ".join(runtime_failing)
                                + ". Fix their visibility/placement, or call "
                                "remove_root_surface if the added plane was unnecessary."
                            )
                        if main_name in failing:
                            msg += (
                                f"\nNOTE: '{main_name}' is the MAIN SUPPORT — "
                                "everything in the scene is anchored to it. "
                                "Bypass it only as a last resort: its mask must "
                                "be unmistakably broken AND your built support "
                                "must match the photo by eye. A wrongly-bypassed "
                                "main support corrupts every later object "
                                "placement."
                            )
                return {
                    "status": "success",
                    "output": {
                        "text": [msg],
                        **({"image": imgs} if imgs else {}),
                        "rules_all_pass": False,
                        "yaw_evidence": self._yaw_evidence,
                        "relationship_evidence": self._relationship_evidence,
                        "constraint_results": self._constraint_results,
                        "rule_evidence": _structured_gate("fail", deferred_names),
                    },
                }
            msg = (
                "check_rules_enforced: ALL RULES PASS (structure, pose, contact). "
                "PASS is a FLOOR, not a match — the size/pose tolerances are wide, so a "
                "passing scene can still be visibly wrong. Do the MATCH PASS from your "
                "instructions before ending: compare the main support's edges/corners and "
                "the object-to-edge gaps against the photo at (0,0), check one novel view "
                "for depth-axis size, and fix what is visibly off. Call end only when you "
                "cannot articulate a remaining mismatch."
            )
            yaw_note = getattr(self, "_yaw_advisory", None)
            if yaw_note:
                msg += f" NOTE (not a failure): {yaw_note}."
            advisory_requirement = self._build_yaw_advisory_requirement(data, main_name)
            advisory_state = advisory_requirement.get("state")
            completion_ready = advisory_state != "required"
            if advisory_state == "required":
                msg += (
                    " COMPLETION PENDING: residual yaw has a machine-checkable "
                    "candidate basis. Call resolve_yaw_advisory with only the issued "
                    "candidate IDs/token (or include the same ID-only submission in "
                    "end); end is refused until the backend returns VERIFIED_MATCH."
                )
            elif advisory_state == "manual_review":
                msg += (
                    " Residual yaw is not machine-observable from a complete candidate "
                    "basis. You may end after your visual match pass; the initializer "
                    "verifier must record a structured yaw-advisory review."
                )
            # Expose the boolean so the generator can gate `end` on it (the text header alone
            # is not machine-readable). Root uses this to reject a scene whose rules never passed.
            return {
                "status": "success",
                "output": {
                    "text": [msg],
                    "rules_all_pass": True,
                    "completion_ready": completion_ready,
                    "advisory_requirement": advisory_requirement,
                    "yaw_evidence": self._yaw_evidence,
                    "relationship_evidence": self._relationship_evidence,
                    "constraint_results": self._constraint_results,
                    "rule_evidence": _structured_gate("pass", []),
                    **({"yaw_note": yaw_note} if yaw_note else {}),
                },
            }
        except Exception as e:
            output = {"text": [str(e)]}
            if self.root_stage_name == "initializer":
                yaw = getattr(self, "_yaw_evidence", None)
                if not yaw:
                    source = (
                        getattr(self, "_initializer_constraint_artifact", None) or {}
                    ).get("source_yaw_evidence") or {
                        "source_candidates": [],
                        "source_decision": {
                            "authority": "unverified",
                            "status": "unverified",
                            "target_run_xy": None,
                            "candidate_ids": [],
                            "reason_codes": ["rule_evaluation_exception"],
                        },
                    }
                    yaw = bind_compiled_yaw_evidence(
                        None,
                        {
                            "status": "unknown",
                            "reason": "rule_evaluation_exception",
                            "geometry_source": "main_top_hull",
                            "rectangularity": None,
                        },
                        None,
                        source,
                    )
                relationships = getattr(self, "_relationship_evidence", [])
                constraints = getattr(self, "_constraint_results", [])
                output.update(
                    yaw_evidence=yaw,
                    relationship_evidence=relationships,
                    constraint_results=constraints,
                    rule_evidence={
                        "schema_version": 2,
                        "overall_status": "error",
                        "evaluated_levels": [],
                        "deferred_levels": [],
                        "yaw_evidence": yaw,
                        "relationship_evidence": relationships,
                        "constraint_results": constraints,
                    },
                )
            return {"status": "error", "output": output}

    @staticmethod
    def _sha256_file(path: Optional[str]) -> Optional[str]:
        """Content digest for an advisory snapshot; missing input is explicit."""
        if not path or not os.path.isfile(path):
            return None
        import hashlib

        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _yaw_advisory_json_digest(value: object) -> str:
        import hashlib

        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _yaw_advisory_scene_binding(
        self,
        *,
        main_name: Optional[str],
        top_hull: object,
        top_z: Optional[float],
        projection_camera: object,
        source_candidates: list[dict],
        built_candidates: list[dict],
    ) -> dict:
        """Inputs whose equality makes an observation token current.

        The blend content catches every committed scene edit. Graph/constraint/bypass
        digests also cover rule-input changes that need not alter the blend. Camera and
        source-evidence digests make a cached token unusable after preprocessing changes.
        """
        graph_path = self._scene_graph_path()
        constraints_path = self._initializer_constraints_path()
        live_blend = self.blender_save or self.blender_file
        return {
            "main_surface": main_name,
            "scene_blend_sha256": self._sha256_file(live_blend),
            "scene_graph_sha256": self._sha256_file(
                str(graph_path) if graph_path is not None else None
            ),
            "constraint_artifact_sha256": self._sha256_file(
                str(constraints_path) if constraints_path is not None else None
            ),
            "source_observation_sha256": self._sha256_file(
                os.path.join(self.moge_dir, "main_support_yaw_observation.json")
                if self.moge_dir
                else None
            ),
            # Bind the exact derived projection config as well as its source file.
            # This is deliberately not the simplified legacy pseudo-view record.
            "camera_config_sha256": self._yaw_advisory_json_digest(projection_camera),
            "camera_artifact_sha256": self._sha256_file(
                os.path.join(self.moge_dir, "moge", "moge.json")
                if self.moge_dir
                else None
            ),
            "source_yaw_sha256": self._yaw_advisory_json_digest(
                {
                    "source_decision": (self._yaw_evidence or {}).get(
                        "source_decision"
                    ),
                    "source_candidates": source_candidates,
                }
            ),
            "built_top_sha256": self._yaw_advisory_json_digest(
                {
                    "top_hull": top_hull,
                    "top_z": top_z,
                    "candidates": built_candidates,
                }
            ),
            "constraint_results_sha256": self._yaw_advisory_json_digest(
                self._constraint_results
            ),
            "coverage_bypass": sorted(str(name) for name in self._coverage_bypassed),
        }

    def _yaw_advisory_projection_camera(self) -> Optional[dict]:
        """Return the exact calibrated source camera used for candidate pixels."""
        if not self.moge_dir:
            return None
        moge_path = os.path.join(self.moge_dir, "moge", "moge.json")
        if not os.path.isfile(moge_path):
            return None
        import numpy as np

        from lib.tools.geometry.moge_camera import camera_config_from_moge

        with open(moge_path, encoding="utf-8") as stream:
            moge = json.load(stream)
        gravity = moge.get("gravity") or {}
        raw_r, raw_t = gravity.get("R"), gravity.get("T")
        if raw_r is None or raw_t is None:
            return None
        return camera_config_from_moge(
            moge,
            R=np.asarray(raw_r, dtype=np.float64),
            T=np.asarray(raw_t, dtype=np.float64),
        )

    def _required_yaw_already_verified(self, main_name: Optional[str]) -> bool:
        """Whether current POSE results passed a required yaw target for *the main*.

        Constraint results can also contain yaw rows for secondary supports.  Bind the
        result back to the active graph main id and the exact compiled constraint so a
        secondary PASS/N/A cannot waive unresolved main-support completion evidence.
        Main yaw applicability N/A is handled separately by ``_yaw_evidence``.
        """
        if not main_name:
            return False
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            graph_path = self._scene_graph_path()
            if graph_path is None or not graph_path.is_file():
                return False
            graph = json.loads(graph_path.read_text())
            main_id = str(graph.get("main_support_id") or "")
            main_node = next(
                (
                    node
                    for node in graph.get("nodes", []) or []
                    if isinstance(node, dict) and str(node.get("id") or "") == main_id
                ),
                None,
            )
            if not main_id or not isinstance(main_node, dict):
                return False
            expected_build_name = str(
                main_node.get("build_name") or surface_build_name(main_id)
            )
            if expected_build_name != main_name:
                return False
            artifact = self._initializer_constraint_artifact or {}
            compiled = {
                str(constraint.get("constraint_id")): constraint
                for constraint in artifact.get("constraints", []) or []
                if isinstance(constraint, dict)
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

        for row in getattr(self, "_constraint_results", []):
            if (
                not isinstance(row, dict)
                or row.get("stage") != "POSE"
                or row.get("runtime_status") != "pass"
            ):
                continue
            constraint = compiled.get(str(row.get("constraint_id") or ""))
            if not isinstance(constraint, dict):
                continue
            kind = str(constraint.get("kind") or "")
            if (
                kind not in {"edge_parallel_to_surface", "main_support_edge_yaw"}
                or constraint.get("authority") != "required"
                or constraint.get("stage") != "POSE"
                or row.get("kind") != kind
            ):
                continue
            compiled_targets = list(constraint.get("targets") or [])
            if list(row.get("targets") or []) != compiled_targets:
                continue
            applicability = constraint.get("applicability") or {}
            target_id = str(
                applicability.get("target_surface_id")
                or (compiled_targets[0] if compiled_targets else "")
            )
            if target_id == main_id:
                return True
        return False

    def _build_yaw_advisory_requirement(
        self, data: dict, main_name: Optional[str]
    ) -> dict:
        """Mint a residual completion requirement after the full ladder passes.

        This method never promotes source evidence and never writes a constraint. A
        complete candidate basis makes selection machine-required; genuinely
        unobservable evidence is routed to a structured verifier review instead.
        """
        import hashlib
        import secrets

        from lib.tools.geometry.main_support_yaw_observation import (
            load_main_support_yaw_observation,
        )
        from lib.tools.geometry.yaw_advisory import (
            built_candidates_from_top_hull,
            machine_methods,
            normalize_advisory_requirement,
            project_built_candidates,
            source_candidates_from_observation,
        )

        yaw = self._yaw_evidence if isinstance(self._yaw_evidence, dict) else {}
        applicability = yaw.get("applicability") or {}
        decision = yaw.get("source_decision") or {}
        source_authority = str(decision.get("authority") or "").lower()
        app_status = str(applicability.get("status") or "").lower()

        not_required_reasons: list[str] = []
        if app_status in {"not_applicable", "not-applicable"}:
            not_required_reasons.append("yaw_not_applicable")
        if source_authority == "not_applicable":
            not_required_reasons.append("source_yaw_not_applicable")
        if source_authority == "required":
            not_required_reasons.append("source_yaw_compiled_required")
        if self._required_yaw_already_verified(main_name):
            not_required_reasons.append("required_yaw_constraint_verified")
        if not_required_reasons:
            requirement = {
                "schema_version": 1,
                "state": "not_required",
                "reason_codes": not_required_reasons,
            }
            self._yaw_advisory_binding = None
            self._yaw_advisory_requirement = normalize_advisory_requirement(requirement)
            self._yaw_resolution = {
                "schema_version": 1,
                "advisory_id": f"yaw:{main_name or 'main'}:not-applicable",
                "status": "not_applicable",
                "reason_codes": not_required_reasons,
            }
            return self._yaw_advisory_requirement

        # Strict schema-v1 load: a folded resolver candidate is not two physical
        # edges. Issue the immutable artifact's individual eligible segments instead.
        observation = load_main_support_yaw_observation(self.moge_dir)
        if (observation.get("main_surface") or {}).get("build_name") != main_name:
            raise RuntimeError(
                "main_support_yaw_observation identity does not match the active main "
                "support; rerun preprocessing"
            )
        source_candidates = source_candidates_from_observation(observation)
        top_hull = (data.get("surface_top_hulls") or {}).get(main_name)
        built_candidates = built_candidates_from_top_hull(top_hull)
        main_body = next(
            (
                body
                for body in data.get("bodies", []) or []
                if isinstance(body, dict) and body.get("name") == main_name
            ),
            None,
        )
        top_z = None
        try:
            top_z = float((main_body or {}).get("hi", [None, None, None])[2])
        except (TypeError, ValueError, IndexError):
            top_z = None
        camera = self._yaw_advisory_projection_camera()
        built_candidates = (
            project_built_candidates(built_candidates, camera, top_z=top_z)
            if top_z is not None
            else []
        )
        methods = machine_methods(source_candidates, built_candidates)
        # A recorded source conflict cannot be resolved by letting the model select
        # whichever two raw edges happen to pass. Preserve that conflict and route it
        # to bounded manual review. Advisory/unverified evidence becomes required only
        # when the backend can check a complete independent physical basis.
        source_conflicting = source_authority == "conflicting"
        if source_conflicting:
            methods = []
        state = "required" if methods else "manual_review"
        reasons = list(decision.get("reason_codes") or [])[:8]
        reasons.append(
            "source_yaw_conflicting_requires_manual_review"
            if source_conflicting
            else "machine_candidate_basis_available"
            if methods
            else "machine_candidate_basis_incomplete"
        )
        binding = self._yaw_advisory_scene_binding(
            main_name=main_name,
            top_hull=top_hull,
            top_z=top_z,
            projection_camera=camera,
            source_candidates=source_candidates,
            built_candidates=built_candidates,
        )
        binding_digest = self._yaw_advisory_json_digest(binding)
        advisory_id = (
            "yaw:"
            + hashlib.sha256(
                f"{main_name}|{binding['source_yaw_sha256']}".encode("utf-8")
            ).hexdigest()[:20]
        )
        token = hashlib.sha256(
            (secrets.token_hex(32) + binding_digest).encode("ascii")
        ).hexdigest()
        requirement = {
            "schema_version": 1,
            "state": state,
            "advisory_id": advisory_id,
            "observation_token": token,
            "allowed_methods": methods,
            "target_candidates": source_candidates,
            "built_candidates": built_candidates,
            "reason_codes": reasons,
        }
        self._yaw_advisory_binding = binding
        self._yaw_advisory_requirement = normalize_advisory_requirement(requirement)
        self._yaw_resolution = None
        return copy.deepcopy(self._yaw_advisory_requirement)

    def resolve_yaw_advisory(
        self,
        advisory_id: str,
        observation_token: str,
        method: str,
        edge_matches: list,
    ) -> dict[str, object]:
        """Validate an ID-only residual-yaw correspondence against current state."""
        from lib.tools.geometry.yaw_advisory import (
            evaluate_submission,
            normalize_resolution_submission,
            normalize_yaw_resolution,
        )

        requirement = getattr(self, "_yaw_advisory_requirement", None)
        binding = getattr(self, "_yaw_advisory_binding", None)
        submission_raw = {
            "advisory_id": advisory_id,
            "observation_token": observation_token,
            "method": method,
            "edge_matches": edge_matches,
        }
        try:
            submission = normalize_resolution_submission(submission_raw)
        except ValueError as exc:
            result = {
                "schema_version": 1,
                "advisory_id": str(advisory_id or "invalid")[:160] or "invalid",
                "status": "unverified",
                "reason_codes": ["invalid_id_only_submission"],
            }
            return {
                "status": "success",
                "output": {
                    "text": [f"yaw advisory: UNVERIFIED — {exc}"],
                    "yaw_resolution": result,
                    "completion_ready": False,
                },
            }
        if not isinstance(requirement, dict):
            result = {
                "schema_version": 1,
                "advisory_id": submission["advisory_id"],
                "status": "stale",
                "reason_codes": ["no_current_full_pass_request"],
            }
        elif requirement.get("state") == "not_required":
            result = {
                "schema_version": 1,
                "advisory_id": submission["advisory_id"],
                "status": "not_applicable",
                "reason_codes": ["advisory_not_required"],
            }
        else:
            # Recompute the non-random half of the binding. Any edit, graph/constraint
            # rewrite, camera/source change, or bypass makes the old token stale.
            current = self._yaw_advisory_scene_binding(
                main_name=binding.get("main_surface")
                if isinstance(binding, dict)
                else None,
                top_hull=None,
                top_z=None,
                projection_camera=self._yaw_advisory_projection_camera(),
                source_candidates=requirement.get("target_candidates") or [],
                built_candidates=requirement.get("built_candidates") or [],
            )
            # built_top_sha256 was derived from the exact hull plus candidates. The
            # exact hull is not retained in the public request, so re-reading geometry
            # would require a Blender subprocess. The live blend digest already catches
            # every geometry edit; retain the original geometric digest for comparison.
            if isinstance(binding, dict):
                current["built_top_sha256"] = binding.get("built_top_sha256")
            stale = not isinstance(binding, dict) or any(
                current.get(key) != binding.get(key)
                for key in (
                    "scene_blend_sha256",
                    "scene_graph_sha256",
                    "constraint_artifact_sha256",
                    "source_observation_sha256",
                    "camera_config_sha256",
                    "camera_artifact_sha256",
                    "source_yaw_sha256",
                    "coverage_bypass",
                )
            )
            if stale:
                result = {
                    "schema_version": 1,
                    "advisory_id": submission["advisory_id"],
                    "status": "stale",
                    "reason_codes": ["scene_or_rule_inputs_changed"],
                }
            else:
                result = evaluate_submission(requirement, submission)
        result = normalize_yaw_resolution(result)
        self._yaw_resolution = copy.deepcopy(result)
        status = result["status"]
        completion_ready = requirement is not None and (
            requirement.get("state") in {"manual_review", "not_required"}
            or status in {"verified_match", "not_applicable"}
        )
        measurement = (result.get("measurements") or {}).get("max_delta_degrees_mod_90")
        text = f"yaw advisory: {status.upper()}"
        if measurement is not None:
            text += f" — backend max edge delta {float(measurement):.1f}deg"
        if status == "verified_mismatch":
            text += "; correct the main-support yaw, then rerun the full rules gate"
        elif status == "stale":
            text += "; rerun check_rules_enforced to obtain current candidate ids"
        elif status == "unverified":
            text += "; select only a complete, independent issued candidate basis"
        return {
            "status": "success",
            "output": {
                "text": [text],
                "yaw_resolution": result,
                "advisory_requirement": copy.deepcopy(requirement),
                "completion_ready": completion_ready,
            },
        }

    def _surface_ids(self) -> dict:
        """{graph id (e.g. 'curtain#0'): build name (e.g. 'curtain_0')} for every
        root surface in the scene graph. Best-effort ({} on missing artifacts)."""
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            sgj = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            return {
                n["id"]: surface_build_name(n["id"])
                for n in sgj.get("nodes", [])
                if n.get("kind") == "root_surface"
            }
        except Exception:  # noqa: BLE001
            return {}

    def bypass(self, obj_ids: list) -> dict:
        """Waive level-2 coverage for a bad preprocessed mask/absent preprocessed wall.

        Runtime-added roots are never bypassable: remove an unnecessary one through
        remove_root_surface.  All CURRENT REGISTERED roots remain structurally required.
        """
        surfaces = self._surface_ids()
        build_of = {**surfaces, **{v: v for v in surfaces.values()}}  # both dialects
        runtime_builds = set()
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            graph = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            runtime_builds = {
                str(n.get("build_name") or surface_build_name(n["id"]))
                for n in graph.get("nodes", [])
                if n.get("kind") == "root_surface"
                and n.get("runtime_added") is True
                and n.get("id")
            }
        except Exception:  # noqa: BLE001 - normal identity checks below still apply
            runtime_builds = set()
        waived, main_waived = [], False
        main_name, _, _ = self._main_support_info()
        for oid in obj_ids:
            build = build_of.get(oid)
            if build is None:
                if str(oid).startswith("obj_") or "#" in str(oid):
                    known = ", ".join(sorted(surfaces)) or "none"
                    return {
                        "status": "error",
                        "output": {"text": [
                            f"bypass only accepts ROOT SURFACES ({known}); "
                            f"'{oid}' is not one — objects are fixed with "
                            "move/execute_and_evaluate, not bypassed."
                            if str(oid).startswith("obj_")
                            else f"unknown root surface '{oid}' — use the exact id "
                            "shown in the coverage report."
                        ]},
                    }  # fmt: skip
                return {
                    "status": "error",
                    "output": {"text": [
                        f"unknown root surface '{oid}' — use the exact id shown "
                        "in the coverage report."
                    ]},
                }  # fmt: skip
            if build in runtime_builds:
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            f"Coverage bypass is not allowed for runtime-added root '{oid}'. "
                            "If the wall was unnecessary, remove it with "
                            "remove_root_surface; otherwise fix its visibility/placement."
                        ]
                    },
                }
            if build not in self._coverage_last_failing:
                return {
                    "status": "error",
                    "output": {"text": [
                        f"'{oid}' is not named in the CURRENT coverage failure — "
                        "you can only bypass surfaces the report is currently "
                        "flagging. Call check_rules_enforced first."
                    ]},
                }  # fmt: skip
            waived.append(build)
            if build == main_name:
                main_waived = True
        self._coverage_bypassed.update(waived)
        try:
            tmp_dir = self.render_path.parent / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            (tmp_dir / "coverage_bypass.json").write_text(
                json.dumps(sorted(self._coverage_bypassed))
            )
        except Exception:  # noqa: BLE001 - the audit record is best-effort
            pass
        txt = (
            f"Coverage waived for {', '.join(waived)} (recorded in the run "
            "record). Call check_rules_enforced again to re-evaluate the "
            "remaining rules."
        )
        if main_waived:
            txt += (
                " WARNING: you waived the MAIN SUPPORT's coverage — its size/yaw "
                "will not be checked again this stage. Confirm from your next "
                "full render that the support visually matches the photo before "
                "proceeding."
            )
        return {"status": "success", "output": {"text": [txt]}}

    _TINT = (255, 64, 200)  # magenta: distinct from scene hues in photos/renders

    def _runtime_root_pose_report(
        self, rendered: dict, tmp_dir: Path
    ) -> tuple[bool, str]:
        """Visibility/authorized-region evidence for mask-less runtime roots.

        The submitted region is visual provenance, not a substitute segmentation
        mask.  Use a generous 12%-of-image expansion and accept either centroid
        inclusion or 10% silhouette overlap, while still requiring a non-trivial
        visible footprint.  This catches a claimed left wall built wholly on the
        right without pretending the rough box is pixel-accurate ground truth.
        """
        try:
            import numpy as np

            from lib.tools.geometry.surface_relations import surface_build_name

            graph = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            nodes = [
                n
                for n in graph.get("nodes", [])
                if n.get("kind") == "root_surface"
                and n.get("runtime_added") is True
                and n.get("id")
            ]
            if not nodes:
                return True, "runtime root visibility: PASS (none registered)"
            ids_path = tmp_dir / "coverage.json_ids.npz"
            if not ids_path.exists():
                return False, "runtime root visibility: UNVERIFIED (id render missing)"
            packed = np.load(ids_path, allow_pickle=True)
            ids = packed["ids"]
            names = [str(x) for x in packed["names"]]
            h, w = ids.shape[:2]
            lines, ok = [], True
            for node in nodes:
                name = str(node.get("build_name") or surface_build_name(node["id"]))
                row = rendered.get(name) or {}
                frac = float(row.get("frac") or 0.0)
                region = node.get("target_region_norm")
                if (
                    not isinstance(region, list)
                    or len(region) != 4
                    or not all(isinstance(x, (int, float)) for x in region)
                ):
                    ok = False
                    self._coverage_last_failing.add(name)
                    lines.append(
                        f"runtime wall '{name}': UNVERIFIED — authorized target region missing"
                    )
                    continue
                if name not in names:
                    ok = False
                    self._coverage_last_failing.add(name)
                    lines.append(
                        f"runtime wall '{name}': FAIL — registered wall is absent from id render"
                    )
                    continue
                mask = ids == names.index(name)
                visible = int(mask.sum())
                if frac < 0.001 or visible == 0:
                    ok = False
                    self._coverage_last_failing.add(name)
                    lines.append(
                        f"runtime wall '{name}': FAIL — not meaningfully visible from the "
                        "reference camera; reposition/resize it or remove it if unnecessary"
                    )
                    continue
                x0, y0, x1, y1 = [float(x) for x in region]
                margin = 0.12
                ax0, ay0 = max(0.0, x0 - margin), max(0.0, y0 - margin)
                ax1, ay1 = min(1.0, x1 + margin), min(1.0, y1 + margin)
                ix0, iy0 = int(math.floor(ax0 * w)), int(math.floor(ay0 * h))
                ix1, iy1 = int(math.ceil(ax1 * w)), int(math.ceil(ay1 * h))
                inside = int(mask[iy0:iy1, ix0:ix1].sum()) / max(1, visible)
                cx, cy = float(row.get("cx", 0.5)), float(row.get("cy", 0.5))
                centroid_inside = ax0 <= cx <= ax1 and ay0 <= cy <= ay1
                if not centroid_inside and inside < 0.10:
                    ok = False
                    self._coverage_last_failing.add(name)
                    lines.append(
                        f"runtime wall '{name}': FAIL — only {inside:.0%} of its visible "
                        "silhouette overlaps the generously expanded authorized target "
                        "region; fix placement or remove the mistaken wall"
                    )
                else:
                    lines.append(
                        f"runtime wall '{name}': PASS ({frac:.2%} image coverage; "
                        f"{inside:.0%} inside expanded authorized region)"
                    )
            return ok, "\n".join(lines)
        except Exception as exc:  # noqa: BLE001 - runtime provenance must fail closed
            return False, f"runtime root visibility: UNVERIFIED ({exc})"

    def _source_surface_mask_paths(self) -> dict[str, Optional[Path]]:
        """Return source-inventory mask bindings keyed by Blender build name.

        File existence is deliberately *not* inferred from ``masks/<name>.npy``.
        A retained row in ``unmasked_root_surfaces`` is categorically maskless even
        if a stale same-name file remains on disk, while a declared masked instance
        keeps its exact artifact binding and fails clearly if that artifact is bad.
        Runtime-added roots are absent from this immutable source mapping.
        """

        from lib.tools.geometry.surface_relations import surface_build_name

        masks_file = Path(self.moge_dir) / "masks" / "masks.json"
        payload = json.loads(masks_file.read_text())
        result: dict[str, Optional[Path]] = {}
        for section in ("instances", "unmasked_root_surfaces"):
            rows = payload.get(section, []) or []
            if not isinstance(rows, list):
                raise ValueError(f"masks.{section} must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError(f"masks.{section} contains a non-object row")
                category, instance = row.get("category"), row.get("instance")
                if not isinstance(category, str) or not isinstance(instance, int):
                    raise ValueError(f"masks.{section} contains an invalid identity")
                name = surface_build_name(f"{category}#{instance}")
                if name in result:
                    raise ValueError(f"duplicate retained source identity for {name}")
                if section == "unmasked_root_surfaces":
                    result[name] = None
                    continue
                raw_path = row.get("mask_path")
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise ValueError(
                        f"retained masked instance {category}#{instance} has no mask_path"
                    )
                path = Path(raw_path)
                if not path.is_absolute():
                    scene_relative = Path(self.moge_dir) / path
                    local_copy = Path(self.moge_dir) / "masks" / path.name
                    project_relative = Path(__file__).resolve().parents[3] / path
                    path = next(
                        (
                            candidate
                            for candidate in (
                                scene_relative,
                                local_copy,
                                project_relative,
                            )
                            if candidate.is_file()
                        ),
                        scene_relative,
                    )
                result[name] = path
        return result

    def _coverage_mismatch_visual(self, failing: list) -> list:
        """Per failing root surface (cap 3): one side-by-side composite — LEFT the
        coverage BEAUTY render (same camera/state/resolution as the id pass) with
        the BUILT surface's silhouette tinted, RIGHT the target photo with the
        surface's GT segmentation mask tinted the SAME color (render-left/photo-right,
        matching investigate_objects); falls back to a flat id schematic when the
        beauty png is missing. The pair lets the agent judge whether the GT mask
        itself is wrong (the bypass-tool flow). Best-effort: [] on any missing
        artifact."""
        out = []
        try:
            import numpy as np
            from PIL import Image

            tmp_dir = self.render_path.parent / "tmp"
            ids_npz = tmp_dir / "coverage.json_ids.npz"
            d = np.load(ids_npz, allow_pickle=True) if ids_npz.exists() else None
            beauty_p = tmp_dir / "coverage.json_beauty.png"
            beauty = (
                np.asarray(Image.open(beauty_p).convert("RGB"), dtype=np.float32)
                if beauty_p.exists()
                else None
            )
            photo = Image.open(os.path.join(self.moge_dir, "input.png")).convert("RGB")
            tint = np.array(self._TINT, dtype=np.float32)
            source_masks = self._source_surface_mask_paths()
            for name in failing[:3]:
                # RIGHT: photo + GT mask tint (or untinted with a caption when the
                # surface has no mask — description-only helper)
                photo_side = np.asarray(photo, dtype=np.float32).copy()
                mp = source_masks.get(name)
                has_mask = mp is not None
                if has_mask:
                    m = np.load(mp, allow_pickle=False) > 0
                    if m.shape[:2] != photo_side.shape[:2]:
                        m = (
                            np.asarray(
                                Image.fromarray(m.astype(np.uint8) * 255).resize(
                                    (photo_side.shape[1], photo_side.shape[0]),
                                    Image.NEAREST,
                                )
                            )
                            > 127
                        )
                    photo_side[m] = 0.55 * photo_side[m] + 0.45 * tint
                # LEFT: the build as the agent sees it — beauty render + tint on
                # the failing surface (schematic fallback for pre-beauty artifacts)
                render_side = None
                if d is not None:
                    ids = d["ids"]
                    names = list(d["names"])
                    objects = d["objects"]
                    if beauty is not None and beauty.shape[:2] == ids.shape:
                        canvas = beauty.copy()
                        if name in names:
                            sm = ids == names.index(name)
                            canvas[sm] = 0.55 * canvas[sm] + 0.45 * tint
                    else:
                        canvas = np.full((*ids.shape, 3), 245.0, dtype=np.float32)
                        canvas[ids > 0] = 210.0  # other surfaces: light gray
                        canvas[objects] = 150.0  # objects: mid gray
                        if name in names:
                            canvas[ids == names.index(name)] = tint
                    render_side = canvas
                if render_side is None and not has_mask:
                    continue  # nothing to show on either side
                ph = photo_side.shape[0]
                if render_side is not None:
                    r_img = Image.fromarray(render_side.astype(np.uint8)).resize(
                        (int(render_side.shape[1] * ph / render_side.shape[0]), ph),
                        Image.NEAREST,
                    )
                    render_side = np.asarray(r_img, dtype=np.float32)
                    gap = np.full((ph, 8, 3), 255.0, dtype=np.float32)
                    combo = np.concatenate([render_side, gap, photo_side], axis=1)
                else:
                    combo = photo_side
                p = str(tmp_dir / f"coverage_mismatch_{name}.png")
                Image.fromarray(combo.astype(np.uint8)).save(p)
                out.append(p)
        except Exception:  # noqa: BLE001 - visuals are advisory, never fail the rule
            return []
        return out

    def _rule_coverage(
        self,
        tmp_dir: "Path",
        main_name: Optional[str],
        data: Optional[dict] = None,
        main_form: Optional[str] = None,
    ) -> tuple[bool, str]:
        """POSE rule: run the flat-id coverage render from the reference camera and compare
        each root surface's visible fraction to the exact mask declared by the retained
        source inventory. Also
        binds the exact current main-top run to the immutable source-yaw decision in
        ``initializer_constraints.json``. Required yaw is checked once by the compiled
        POSE evaluator; this coverage check only carries residual advisory telemetry.
        A missing render or malformed declared mask is explicit UNVERIFIED evidence.
        A registered surface that intentionally has no mask continues through the
        description-only visibility path in ``coverage_report``."""
        self._coverage_last_failing = set()
        self._yaw_advisory = None
        cov_path = tmp_dir / "coverage.json"
        script = generate_coverage_script(str(cov_path))
        self.count += 1
        code_file = self.script_path / f"{self.count}_coverage.py"
        with open(code_file, "w") as f:
            f.write(script)
        if cov_path.exists():
            cov_path.unlink()
        self._execute_blender(str(code_file), persist_scene=False)  # read-only mask render
        if not cov_path.exists():
            return False, "coverage: UNVERIFIED (coverage render failed)"
        with open(cov_path) as f:
            rendered = json.load(f)
        # F5: surfaces past the 9 id slots are painted object-dark rather than aliasing
        # onto slot 9, so the graded numbers stay honest — but they are UNGRADED, and
        # the agent must be told rather than reading silence as a pass. Never a hard
        # failure: the root count is the scene graph's business, not the agent's.
        id_overflow = rendered.pop("_id_overflow", [])
        photo: dict = {}
        try:
            import numpy as np

            source_masks = self._source_surface_mask_paths()
            for name in rendered:
                p = source_masks.get(name)
                if p is None:
                    continue
                m = np.load(p, allow_pickle=False) > 0
                if not m.any():
                    continue
                ys, xs = np.nonzero(m)
                h, w = m.shape[:2]
                photo[name] = {
                    "frac": float(m.mean()),
                    "cx": float(xs.mean()) / w,
                    "cy": float(ys.mean()) / h,
                }
        except Exception as exc:  # noqa: BLE001 - required evidence, not a pass
            return False, f"coverage: UNVERIFIED (mask evidence failed: {exc})"
        main_yaw = None
        main_yaw_src = None
        yaw_evidence_error = None
        attached_walls: list = []
        # Room mode: the scene-level router makes the FLOOR the anchor — its mask edge
        # direction carries no yaw information (and the work surface's rotation is
        # owned by SAM3D pose + register ICP), so the whole yaw ladder is skipped
        # and the report must not flag yaw as unverified.
        main_is_floor = self._main_support_is_floor(main_name)
        try:
            from lib.tools.geometry.surface_relations import surface_build_name

            run_info = dict((data or {}).get("main_support_run") or {})
            built_run = run_info.get("run")
            if main_is_floor:
                applicability = {
                    "status": "not_applicable",
                    "reason": "room_floor_policy",
                    "geometry_source": "room_floor_policy",
                    "rectangularity": run_info.get("rectangularity"),
                }
            elif run_info:
                applicability = {
                    "status": run_info.get("status") or "unknown",
                    "reason": run_info.get("reason") or "run_applicability_missing",
                    "geometry_source": run_info.get("geometry_source")
                    or "main_top_hull",
                    "rectangularity": run_info.get("rectangularity"),
                }
            else:
                applicability = {
                    "status": "unknown",
                    "reason": "main_top_run_evidence_missing",
                    "geometry_source": "main_top_hull",
                    "rectangularity": None,
                }

            sgj = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            main_is_floor = self._main_support_is_floor(main_name, sgj)
            main_id = next(
                (
                    n["id"]
                    for n in sgj.get("nodes", [])
                    if n.get("kind") == "root_surface"
                    and surface_build_name(n["id"]) == main_name
                ),
                None,
            )
            compiled = (
                self._initializer_constraint_artifact
                or self._load_initializer_constraint_artifact(sgj)
            )
            compiled_constraints = compiled.get("constraints", []) or []
            attached_walls = [
                surface_build_name(
                    (c.get("targets") or [None, None])[1]
                    if (c.get("targets") or [None])[0] == main_id
                    else (c.get("targets") or [None, None])[0]
                )
                for c in compiled_constraints
                if c.get("kind") in {"finite_against", "finite_wall_corner"}
                and main_id in (c.get("targets") or [])
            ]
            # Source selection and corroboration were frozen during preprocessing.
            # Runtime only measures the exact current top against that compiled target;
            # it never recomputes mask PCA or promotes a mutable built wall.
            self._yaw_evidence = bind_compiled_yaw_evidence(
                main_name,
                applicability,
                built_run,
                compiled.get("source_yaw_evidence") or {},
            )
            selection = self._yaw_evidence.get("selection") or {}
            verdict = self._yaw_evidence.get("verdict") or {}
            source_decision = self._yaw_evidence.get("source_decision") or {}
            source_authority = source_decision.get("authority")
            main_yaw_src = "source-advisory" if source_authority == "advisory" else None
            # Required yaw is evaluated exactly once by the compiled POSE set.
            # Coverage carries only residual advisory telemetry.
            main_yaw = (
                verdict.get("delta_degrees_mod_90")
                if source_authority == "advisory"
                else None
            )
            if main_yaw is not None:
                # Preserve one side of the mod-90 fold between rounds.
                from lib.tools.blender.script_generators import pin_yaw_delta

                main_yaw = pin_yaw_delta(main_yaw, self._yaw_target_rep.get(main_name))
                self._yaw_target_rep[main_name] = main_yaw
                verdict["delta_degrees_mod_90"] = main_yaw
                tol = float(verdict.get("tolerance_degrees") or 12.0)
                if selection.get("enforcement") == "advisory" and abs(main_yaw) > tol:
                    verdict["status"] = "advisory"
                else:
                    verdict["status"] = "fail" if abs(main_yaw) > tol else "pass"
            if not main_id and not main_is_floor:
                self._yaw_evidence["applicability"].update(
                    status="unknown", reason="main_support_scene_graph_identity_missing"
                )
                verdict.update(
                    status="unverified",
                    reason="main_support_scene_graph_identity_missing",
                )
            if self._yaw_evidence["applicability"].get("status") == "unknown":
                unknown_reason = self._yaw_evidence["applicability"].get("reason")
                yaw_evidence_error = {
                    "top_hull_missing": (
                        "main-support top hull is missing; rebuild the support with a "
                        "distinct, level top face"
                    ),
                    "top_hull_degenerate": (
                        "main-support top hull is degenerate; rebuild a non-zero-area top"
                    ),
                    "top_run_fit_failed": (
                        "main-support top edge fit failed; rebuild a clean top footprint"
                    ),
                    "non_edge_bearing_top_shape": (
                        "main-support top is neither a rectangular edge-bearing footprint "
                        "nor a verified axisymmetric disc; its yaw applicability is unknown"
                    ),
                    "main_top_run_evidence_missing": (
                        "structured main-top yaw evidence is missing"
                    ),
                    "main_support_scene_graph_identity_missing": (
                        "main-support scene-graph identity is missing"
                    ),
                }.get(str(unknown_reason), str(unknown_reason))
        except Exception as exc:  # noqa: BLE001 - surfaced as required evidence failure
            yaw_evidence_error = str(exc)
            current = getattr(self, "_yaw_evidence", {}) or {}
            current.setdefault("schema_version", 1)
            current.setdefault("main_surface", main_name)
            current.setdefault(
                "applicability",
                {
                    "status": "unknown",
                    "reason": "yaw_evidence_collection_failed",
                    "geometry_source": "main_top_hull",
                    "rectangularity": None,
                },
            )
            current.setdefault("built", {"run": None, "source": "main_top_hull"})
            current.setdefault("anchor_candidates", [])
            current["selection"] = None
            current["verdict"] = {
                "status": "unverified",
                "delta_degrees_mod_90": None,
                "tolerance_degrees": 12.0,
                "reason": "yaw_evidence_collection_failed",
                "error": str(exc),
            }
            self._yaw_evidence = current
        # Scene-graph root-surface names: a rendered mask-less surface NOT in this set is a
        # pipeline-mandated helper (the invented floor) and is exempt from the visibility floor.
        known = None
        required_walls: set = set()
        anchored_walls: set = set()
        try:
            from lib.tools.geometry.scene_graph import plane_is_reliable, plumb_plane
            from lib.tools.geometry.surface_relations import surface_build_name

            sgj = json.load(open(os.path.join(self.moge_dir, "scene_graph.json")))
            names = {
                surface_build_name(n["id"])
                for n in sgj.get("nodes", [])
                if n.get("kind") == "root_surface"
            }
            known = (
                names or None
            )  # empty/unreadable graph -> keep the old strict behavior
            required_walls = {
                surface_build_name(n["id"])
                for n in sgj.get("nodes", [])
                if n.get("kind") == "root_surface"
                and str(n.get("category", "")).strip().lower() == "wall"
            }
            for node in sgj.get("nodes", []):
                if node.get("kind") != "root_surface" or not node.get("id"):
                    continue
                plane = node.get("plane") or {}
                _normal, kind = plumb_plane(plane.get("normal"))
                if kind == "wall" and plane_is_reliable(plane):
                    anchored_walls.add(surface_build_name(node["id"]))
        except Exception:  # noqa: BLE001
            pass
        ok, msg = coverage_report(
            rendered,
            photo,
            main_name,
            main_yaw_delta=main_yaw,
            main_yaw_src=main_yaw_src,
            main_yaw_anchor=(self._yaw_evidence.get("selection") or None),
            known_names=known,
            required_walls=required_walls,
            attached_walls=attached_walls or None,
            anchored_walls=anchored_walls or None,
            # main_yaw None here means the anchor ladder was starved (weak PCA, no
            # wall rel) — the report must flag the yaw axis as unverified. A FLOOR
            # anchor (room mode) has no yaw to verify: not flagged.
            yaw_unverified=(
                (self._yaw_evidence.get("applicability") or {}).get("status")
                == "applicable"
                and (self._yaw_evidence.get("verdict") or {}).get("status")
                == "unverified"
            ),
            main_form=main_form,
            bypassed=self._coverage_bypassed or None,
        )
        runtime_ok, runtime_msg = self._runtime_root_pose_report(rendered, tmp_dir)
        ok = ok and runtime_ok
        msg += "\n" + runtime_msg
        if yaw_evidence_error:
            ok = False
            msg += (
                "\nmain-support yaw: UNVERIFIED — required evidence failed: "
                + yaw_evidence_error
            )
        elif (self._yaw_evidence.get("applicability") or {}).get(
            "status"
        ) == "not_applicable":
            reason = (self._yaw_evidence.get("applicability") or {}).get("reason")
            msg += "\nmain-support yaw: NOT APPLICABLE — " + (
                "the main top is axisymmetric and has no meaningful edge yaw"
                if reason == "axisymmetric_main_top"
                else (
                    "the main top has a valid non-rectangular footprint with no "
                    "rectangular edge family to enforce"
                )
                if reason == "non_edge_bearing_main_top"
                else "room-floor policy owns no support yaw"
                if reason == "room_floor_policy"
                else str(reason)
            )
        if id_overflow:
            msg += (
                "\ncoverage: CAUTION — the id render grades at most "
                f"{COVERAGE_ID_SLOTS} surfaces, so these were NOT graded: "
                + ", ".join(sorted(id_overflow))
                + ". They still occlude the graded ones. Remove any surface you built "
                "that the CURRENT scene graph does not register."
            )
        # T3b follow-up (2026-07-24): the yaw CAUTION / UNVERIFIED advisory is
        # composed INSIDE the coverage report, but on an overall pass the gate
        # returns a one-line ALL-RULES-PASS summary — the clutter_fruit table
        # shipped 90-degrees off with the advisory swallowed. Keep it.
        m = re.search(
            r"(yaw UNVERIFIED[^\n]*|CAUTION — weak photo evidence[^\n]*)", msg
        )
        self._yaw_advisory = _yaw_note_from_evidence(self._yaw_evidence)
        if self._yaw_advisory is None and m:
            self._yaw_advisory = m.group(1).rstrip(")").strip()
        # currently-failing surfaces, parsed from the report's own lines: the
        # bypass tool may only waive what the CURRENT report names
        self._coverage_last_failing = {
            n for n in rendered
            if any(line.startswith(f"coverage '{n}': FAIL")
                   for line in msg.splitlines())
        }  # fmt: skip
        return ok, msg


@mcp.tool()
def initialize(args: dict[str, object]) -> dict[str, object]:
    """Initialize Blender executor and set all necessary parameters.

    Args:
        args: Dictionary containing configuration keys including blender_command,
              blender_file, blender_script, output_dir, blender_save, gpu_devices.
    """
    global _executor
    try:
        _executor = Executor(
            blender_command=args.get("blender_command"),
            blender_file=args.get("blender_file"),
            blender_script=args.get("blender_script"),
            script_save=args.get("output_dir") + "/scripts",
            render_save=args.get("output_dir") + "/renders",
            # Fall back to the input blend (as investigator.initialize already does):
            # with no save path every agent edit renders and is then discarded, which
            # silently produced 80 unusable runs on 2026-08-09.
            blender_save=args.get("blender_save") or args.get("blender_file"),
            target_image_path=args.get("target_image_path"),
            gpu_devices=args.get("gpu_devices"),
            render_engine=args.get("render_engine"),
            moge_dir=args.get("moge_dir"),
            root_stage_name=args.get("root_stage_name"),
            stage_dir=args.get("stage_dir"),
            attempt_idx=args.get("attempt_idx"),
            initializer_baseline_blend=args.get("initializer_baseline_blend"),
            initializer_ledger_path=args.get("initializer_ledger_path"),
            harness_profile=args.get("harness_profile") or "baseline",
            harness_profile_manifest=args.get("harness_profile_manifest"),
            model=args.get("preprocess_model") or args.get("model"),
            reconstruction_backend=args.get("backend"),
        )

        def has_capability(capability: str) -> bool:
            return _profile_capability_enabled(
                getattr(_executor, "harness_profile", "baseline"),
                getattr(_executor, "harness_profile_manifest", None),
                capability,
            )

        # Every stage except lighting uses the generic executor + fixed (azimuth, elevation)
        # pseudo-GT views: execute_and_evaluate edits then renders the chosen view, while
        # render_current_scene inspects a view without editing. Stage scope is enforced by the prompt, not a code
        # filter — so the SHARED execute_and_evaluate_novel_view_tool must not carry prose
        # about tools or physics only one stage has (see _FREEFORM_SETTLE_NOTE).
        if args.get("root_stage_name") == "texture":
            tool_configs = [
                execute_and_evaluate_novel_view_tool,
                render_current_scene_tool,
                get_scene_info_tool,
                undo_last_step_tool,
            ]
        elif args.get("root_stage_name") == "composition":
            mo_tool = copy.deepcopy(move_object_tool)
            mo_tool["function"]["description"] += (
                " Every committed move is physics-simulated to rest "
                "(rejected + auto-reverted if it topples or loses the "
                "match); anything stacked on the moved object rides along "
                "(except rotate_180, whose stack stays put; a 180-yaw "
                "after which the object cannot rest upright is rejected)."
            )
            tool_configs = [
                investigate_objects_tool,
                mo_tool,
                (
                    execute_and_evaluate_composition_tool_gpt6
                    if has_capability("strict_post_edit_physics")
                    else execute_and_evaluate_composition_tool
                ),
                render_current_scene_tool,
                render_bev_tool,
                get_scene_info_tool,
                check_rules_enforced_composition_tool,
                undo_last_step_tool,
            ]
            if has_capability("composition_direct_pose_edit"):
                # Directly AFTER move(): the measured optimizer stays the primary
                # pose route; the typed tool is the exception for attitude corrections.
                tool_configs.insert(tool_configs.index(mo_tool) + 1, edit_object_poses_tool)
            if has_capability("composition_mesh_edit") and has_capability(
                "mutation_journal"
            ):
                # A mesh replacement must follow investigate_objects, so keep it
                # immediately beside that evidence-producing tool and before pose/move.
                tool_configs.insert(1, edit_composition_object_mesh_tool)
            if has_capability("strict_post_edit_physics"):
                # execute_and_evaluate is the general pose tool here: a wider soft cap
                # with neutral wording (see _typed feedback in _settle_after_edit note).
                _executor.freeform_cap = 6
                # investigate description: under strict physics execute_and_evaluate
                # leaves the armed set alone (only undo_last_step clears it) — audit F-L1.
                strict_investigate = copy.deepcopy(investigate_objects_tool)
                strict_investigate["function"]["description"] = strict_investigate[
                    "function"
                ]["description"].replace(
                    "and cleared by any manual scene edit.",
                    "and cleared by undo_last_step; execute_and_evaluate leaves it armed.",
                )
                tool_configs[tool_configs.index(investigate_objects_tool)] = strict_investigate
            if args.get("investigate_cap"):
                _executor.investigate_cap = int(args["investigate_cap"])
            if args.get("final_settle_repair"):
                _executor.coverage_scope = {
                    str(row["object"])
                    for row in args["final_settle_repair"]["objects"]
                }
        elif args.get("root_stage_name") == "lighting":
            # Source view only — see execute_and_evaluate_source_view_tool for why.
            tool_configs = [
                execute_and_evaluate_source_view_tool,
                render_current_scene_source_view_tool,
                get_scene_info_tool,
                undo_last_step_tool,
            ]
        else:  # initializer
            tool_configs = [
                execute_and_evaluate_initializer_tool,
                render_current_scene_tool,
                render_bev_tool,
                get_scene_info_tool,
                check_rules_enforced_tool,
                resolve_yaw_advisory_tool,
                build_root_surface_tool,
                remove_root_surface_tool,
                nudge_object_tool,
                bypass_tool,
                undo_last_step_tool,
            ]
            if has_capability("initializer_code_transactions"):
                tool_configs[0] = execute_and_evaluate_initializer_gpt6_tool
                # 2026-09-15 owner: no nudge_object under gpt6_v1 — every object pose
                # change is an execute_and_evaluate object transaction.
                tool_configs.remove(nudge_object_tool)
        tool_effects = {
            "execute_and_evaluate": {"mutates_scene": True},
            "move": {"mutates_scene": True},
            "nudge_object": {"mutates_scene": True},
            "build_root_surface": {
                "mutates_scene": True,
                "mutates_scene_graph": True,
            },
            "remove_root_surface": {
                "mutates_scene": True,
                "mutates_scene_graph": True,
            },
            "undo_last_step": {"mutates_scene": True},
            # A bypass changes the inputs to the rules decision even though
            # it does not alter Blender geometry.
            "bypass": {"invalidates_rules": True},
            # Candidate validation is read-only and preserves the current
            # rules pass; GeneratorAgent binds its result to that revision.
            "resolve_yaw_advisory": {
                "mutates_scene": False,
                "invalidates_rules": False,
            },
        }
        if has_capability("composition_direct_pose_edit"):
            tool_effects["edit_object_poses"] = {"mutates_scene": True}
        if args.get("root_stage_name") == "initializer" and has_capability(
            "initializer_code_transactions"
        ):
            tool_effects["execute_and_evaluate"].update(
                mutates_scene_graph=True, mutates_inventory=True
            )
        if (
            args.get("root_stage_name") == "composition"
            and has_capability("composition_mesh_edit")
            and has_capability("mutation_journal")
        ):
            tool_effects["edit_object_mesh"] = {
                "mutates_scene": True,
                "mutates_inventory": True,
                "mutates_scene_graph": True,
            }
        return {
            "status": "success",
            "output": {
                "text": ["Executor initialized successfully"],
                "tool_configs": tool_configs,
                # Machine-readable gate invalidation metadata consumed by
                # GeneratorAgent. Keep this beside the model-visible schemas so a
                # future mutator cannot silently inherit a stale rules pass.
                "tool_effects": tool_effects,
            },
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def execute_and_evaluate(
    thought: str = "",
    code_diff: str = "",
    code: str = "",
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
    added_objects: Optional[list[dict]] = None,
    removed_objects: Optional[list[str]] = None,
) -> dict[str, object]:
    """Execute Blender Python script and return the rendered image paired with a reference.
    Whenever a pseudo-GT camera set exists, the edited scene is rendered from the
    (``azimuth``, ``elevation``) view (default the source view (0,0)) and a reference image is
    returned alongside: the REAL ground-truth target photo at the source view (0,0), or that
    view's pseudo-GT completion at a novel view."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("execute_and_evaluate")
    if blocked is not None:
        return blocked
    initializer_transaction = getattr(
        _executor, "root_stage_name", None
    ) == "initializer" and _executor._has_capability("initializer_code_transactions")
    if not initializer_transaction and (
        added_objects is not None or removed_objects is not None
    ):
        return {
            "status": "error",
            "output": {
                "text": [
                    "Object addition/removal declarations are available only in "
                    "an initializer with object transactions enabled. The scene was not changed."
                ],
                "scene_mutation": "not_started",
            },
        }
    if not code.strip():
        # EE-2: no full script sent — apply the SEARCH/REPLACE blocks server-side
        # to the previous round's script. Argument-shape failures are classified
        # before any undo snapshot, script execution, render, or scene mutation.
        if _RESERVED_TOOL_MARKUP.search(str(thought or "")):
            return _execute_argument_error(
                "tool_argument_protocol_error",
                "no actual top-level `code` argument was received, while reserved "
                "tool-call markup was placed inside `thought`. Retry in FULL-SCRIPT "
                "MODE: send the COMPLETE runnable script in the separate `code` field "
                "and set `code_diff` to the empty string. Do not put Python or "
                '`<parameter name="code">` text inside `thought`; that is plain '
                "reasoning text and is not executable code. Exact arguments: "
                + _EXECUTE_FULL_SCRIPT_EXAMPLE,
                retryable=True,
            )
        if not _has_syntactically_valid_code_diff(code_diff):
            return _execute_argument_error(
                "tool_argument_protocol_error",
                "no actual top-level `code` argument and no valid SEARCH/REPLACE "
                "`code_diff` were received. Retry in FULL-SCRIPT MODE: send the "
                "COMPLETE runnable script in the separate `code` field and set "
                "`code_diff` to the empty string. Exact arguments: "
                + _EXECUTE_FULL_SCRIPT_EXAMPLE,
                retryable=True,
            )
        base = getattr(_executor, "_last_code", None)
        if not base:
            return _execute_argument_error(
                "patch_base_missing",
                "a syntactically valid `code_diff` was received, but there is no "
                "previous execute_and_evaluate script to patch. Retry in FULL-SCRIPT "
                "MODE: send the COMPLETE runnable script in the separate `code` field "
                "and set `code_diff` to the empty string.",
                retryable=False,
            )
        try:
            code = apply_search_replace(base, code_diff)
        except ValueError as e:
            return _execute_argument_error(
                "code_diff_apply_error",
                f"code_diff could not be applied to your previous script: {e}\n"
                "After a patch failure, resend the COMPLETE script in `code` with "
                "`code_diff` set to the empty string.",
                retryable=False,
            )
        _executor._last_patch_receipt = _patch_receipt(base, code)
    try:
        if initializer_transaction:
            return _executor.execute_initializer_transaction(
                code=code,
                added_objects=added_objects,
                removed_objects=removed_objects,
                reason=thought,
                azimuth=azimuth,
                elevation=elevation,
            )
        if getattr(_executor, "root_stage_name", None) == "composition":
            prepared = _executor._prepare_freeform_edit()
            if prepared and prepared.get("status") != "ready":
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            "Composition edit rejected before execution: its physical "
                            "outcome could not be verified ("
                            + str(prepared.get("reason") or "physics unavailable")
                            + "). The scene was not changed; use move() or retry after "
                            "the physics service recovers."
                        ]
                    },
                }
        # Skip the wrapper's samples=512 render when a pseudo-GT camera set
        # exists: BOTH branches below (relocation crops / novel-view compare)
        # discard it and re-render themselves. Without pseudo-GT the wrapper
        # render IS the round's image — keep it.
        # composition+isaac: physics-settle any objects this freeform edit RELOCATED
        # (lift-to-clear + settle + carry) BEFORE rendering, so the agent sees the
        # physical result and can keep it or undo_last_step. No-op otherwise.
        strict_physics = getattr(
            _executor, "root_stage_name", None
        ) == "composition" and _executor._has_capability("strict_post_edit_physics")
        result = _executor.execute(code, render=not _executor._pseudo_gt_views())
        # a manual scene edit stales the pose-refinement session: rebuild lazily. The
        # baseline also DISARMS (a fresh investigate before the next move); under
        # gpt6_v1 execute_and_evaluate is the general pose tool and leaves the armed
        # set alone — the settle report carries the post-edit poses.
        _executor._pose_dirty = True
        if not strict_physics:
            _executor._armed = set()
        if result.get("status") != "success":
            return result
        freeform_before = copy.deepcopy(_executor._freeform) if strict_physics else None
        phys = _executor._settle_after_edit()
        if phys and phys.get("status") in {"unavailable", "error"}:
            phys_status = str(phys.get("status"))
            phys_reason = str(phys.get("reason") or "no reviewable settled result")
            if not strict_physics:
                # Preserve the baseline response and rollback behavior exactly.
                rolled_back = _executor._rollback_last_edit()
                _journal_composition_edit(
                    _executor, "composition_execute", "rolled_back",
                    {"tool": "execute_and_evaluate",
                     "code_sha256": hashlib.sha256(str(code).encode("utf-8")).hexdigest(),
                     "strict": False, "phys_status": phys_status, "reason": phys_reason,
                     "rolled_back": bool(rolled_back)},
                )  # fmt: skip
                return {
                    "status": "error",
                    "output": {
                        "text": [
                            "Composition edit "
                            + (
                                "rolled back"
                                if rolled_back
                                else "could not be rolled back"
                            )
                            + " because physics settlement could not produce a reviewable "
                            "result: "
                            + phys_status
                            + " ("
                            + phys_reason
                            + "). Retry after the physics service recovers."
                        ]
                    },
                }

            failure_kind = str(phys.get("failure_kind") or "settlement_rejection")
            infrastructure_error = failure_kind == "infrastructure_error"
            physics_recovery_errors = [
                str(error) for error in (phys.get("recovery_errors") or [])
            ]
            physics_recovered = not infrastructure_error or (
                phys.get("recovery_succeeded") is True and not physics_recovery_errors
            )
            rollback_error = ""
            try:
                rolled_back = _executor._rollback_last_edit()
            except Exception as exc:  # noqa: BLE001 - strict path fails closed
                rolled_back = False
                rollback_error = str(exc)
            if rolled_back and freeform_before is not None:
                _executor._freeform = freeform_before
            recovery_complete = rolled_back and physics_recovered
            cleanup_errors: list[str] = []
            if not recovery_complete:
                cleanup_errors = _executor._discard_composition_pose_runtime()

            if failure_kind == "physics_rejection":
                failure_text = (
                    "Composition edit was rejected by strict physics: "
                    + phys_reason
                    + _strict_rejection_hint(phys_reason)
                )
            elif infrastructure_error:
                failure_text = (
                    "Composition edit failed because the strict physics "
                    "infrastructure errored"
                )
            else:
                failure_text = (
                    "Composition edit did not produce a strict settled result"
                )
            _journal_composition_edit(
                _executor, "composition_execute",
                "rejected" if failure_kind == "physics_rejection" else "error",
                {"tool": "execute_and_evaluate",
                 "code_sha256": hashlib.sha256(str(code).encode("utf-8")).hexdigest(),
                 "strict": True, "failure_kind": failure_kind, "reason": phys_reason,
                 "hint": _strict_rejection_hint(phys_reason) if failure_kind == "physics_rejection" else "",
                 "recovery_complete": bool(recovery_complete)},
            )  # fmt: skip
            if recovery_complete:
                failure_text += "; the pre-edit scene was restored"
            else:
                failure_text += (
                    "; CRITICAL: full recovery could not be verified, so the pose "
                    "runtime was discarded and scene state is unknown"
                )
            diagnostics = [
                *physics_recovery_errors,
                *(
                    [f"Blend rollback failed: {rollback_error}"]
                    if rollback_error
                    else []
                ),
                *cleanup_errors,
            ]
            if diagnostics:
                failure_text += ". Recovery diagnostics: " + "; ".join(diagnostics)
            return {
                "status": "error",
                "output": {
                    "text": [failure_text + "."],
                    "physics": phys,
                    "failure_kind": failure_kind,
                    "retryable": recovery_complete,
                    "recovery_succeeded": recovery_complete,
                    "recovery_errors": copy.deepcopy(physics_recovery_errors),
                    "scene_mutation": (
                        "not_committed" if recovery_complete else "unknown"
                    ),
                },
            }
        # RELOCATION edit: physics settled object(s). Give TWO render CROPs windowed on
        # the moved object(s) (mirrors investigate_objects) + the soft settle note — NOT
        # the costly full-scene render+reference pair. The model compares the crops and
        # decides keep/undo (the note is a hint, not an undo command). Returning `image`
        # makes _ensure_end_of_round_render skip the full-scene pair.
        crop = (
            _executor._settled_object_crop([r["name"] for r in phys["reports"]])
            if phys and phys.get("status") == "settled" and phys.get("reports")
            else None
        )
        if crop:
            cap = (
                "IMAGE 1 = your edited render (cropped to the moved region); "
                "IMAGE 2 = the reference target photo, same crop. Each moved object is "
                "boxed (same color + id) in both — the render box is where it sits now, "
                "the photo box where the target shows it."
            )
            if getattr(_executor, "_settle_photo_deocc", False):
                cap += (
                    " NOTE: IMAGE 2 is a de-occluded edit (occluders removed by "
                    "re-segmentation), so the full object is visible there."
                )
            # image_kind tags the payload shape for the generator's image-feedback
            # note (crop pair vs full render+reference pair — both are 2 images, so
            # a count can't tell them apart).
            out = {
                "status": "success",
                "output": {
                    "text": [cap],
                    "image": crop,
                    "image_kind": "relocation_crop",
                },
            }
        # Otherwise (materials/lighting/geometry — no relocation) pair the render with a
        # reference whenever a pseudo-GT camera set exists: the REAL ground-truth photo at
        # the source view (azimuth=elevation=0), or the matching pseudo-GT at a novel view.
        elif _executor._pseudo_gt_views():
            out = _executor._render_novel_view(
                0.0 if azimuth is None else azimuth,
                0.0 if elevation is None else elevation,
            )
        else:
            out = result
        if out is not result:
            # the crop / novel-view payloads replace execute()'s text: carry the script's
            # own print() output over (owner 09-15: the agent reads its measurements here)
            for part in (result.get("output") or {}).get("text") or []:
                if isinstance(part, str) and part.startswith(("Script output (", "Patch applied.")):
                    out.setdefault("output", {}).setdefault("text", []).append(part)
        _journal_composition_edit(
            _executor, "composition_execute", "committed",
            {"tool": "execute_and_evaluate",
             "code_sha256": hashlib.sha256(str(code).encode("utf-8")).hexdigest(),
             "strict": bool(strict_physics),
             "settle_status": (phys or {}).get("status") if isinstance(phys, dict) else None,
             "joint": bool((phys or {}).get("joint_settle")) if isinstance(phys, dict) else False,
             "objects": _settle_report_summary((phys or {}).get("reports") if isinstance(phys, dict) else [])},
        )  # fmt: skip
        if phys and phys.get("status") == "settled":
            note = _executor._freeform_settle_note(phys["reports"])
            # ALWAYS its own text part, never glued onto text[0]: on the crop
            # path text[0] is the "IMAGE 1 = ..." caption, which the memory
            # ledger drops wholesale (prompt_builder _skip) — a concatenated
            # note vanished with it, leaving collapsed exec rounds with no
            # settle signal at all (bridge_6 r10/r12/r14 read "Render image 1").
            out.setdefault("output", {}).setdefault("text", []).append(note)
        return out
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def render_object_focus(object_name: str, view: str = "current") -> dict[str, object]:
    """Render a single object in isolation for geometry inspection."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("render_object_focus")
    if blocked is not None:
        return blocked
    try:
        return _executor.render_object_focus(object_name, view)
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def render_arbitrary_view(
    camera_location: list[float],
    camera_rotation_euler: Optional[list[float]] = None,
    look_at: Optional[list[float]] = None,
    lens: float = 35.0,
    orthographic: bool = False,
    ortho_scale: float = 5.0,
    object_names: Optional[list[str]] = None,
    render_backend: str = "stage_default",
) -> dict[str, object]:
    """Render a temporary arbitrary camera view for generator inspection."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("render_arbitrary_view")
    if blocked is not None:
        return blocked
    try:
        return _executor.render_arbitrary_view(
            camera_location=camera_location,
            camera_rotation_euler=camera_rotation_euler,
            look_at=look_at,
            lens=lens,
            orthographic=orthographic,
            ortho_scale=ortho_scale,
            object_names=object_names or [],
            render_backend=render_backend,
        )
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def investigate_objects(objects: list) -> dict[str, object]:
    """Side-by-side render/photo crops of the listed objects; arms them for move()."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup(
        "investigate_objects",
        object_ids=[str(o) for o in objects] if isinstance(objects, list) else [],
    )
    if blocked is not None:
        return blocked
    try:
        return _executor.investigate_objects(objects)
    except Exception as e:  # noqa: BLE001 - tool errors go back to the agent
        return {"status": "error", "output": {"text": [f"investigate failed: {e}"]}}


@mcp.tool()
def move(object: str, aspect: str) -> dict[str, object]:  # noqa: A002
    """Backend-optimized single-aspect pose fix of an ARMED object. Each object
    has a budget of 5 move calls total (any aspect; rejected/dead calls count).
    Usage is shown in the OBJECT STATE table; a direct count appears on cap refusal."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup("move")
    if blocked is not None:
        return blocked
    try:
        result = _executor.move_object(object, aspect)
        _journal_composition_edit(
            _executor, "composition_move",
            "committed" if isinstance(result, dict) and result.get("status") == "success" else "rejected",
            {"tool": "move", "object": str(object), "aspect": str(aspect),
             "text": str(((result.get("output") or {}).get("text") or [""])[0])[:400]
             if isinstance(result, dict) else str(result)[:400]},
        )  # fmt: skip
        if (
            aspect == "rotate_180"
            and isinstance(result, dict)
            and result.get("status") == "error"
            and isinstance(result.get("output"), dict)
        ):
            # A returned error (as opposed to an escaping exception) completed the
            # Executor's transactional branch.  Required metadata proves a retained
            # commit; otherwise the flip was rejected, never started, or rolled back.
            result["output"].setdefault(
                "scene_mutation",
                (
                    "committed"
                    if "required_followup" in result["output"]
                    else "not_committed"
                ),
            )
        return result
    except Exception as e:  # noqa: BLE001
        output: dict[str, object] = {
            "text": [f"move failed: {e}"],
        }
        if aspect == "rotate_180":
            # Response construction can fail after the flip, undo snapshot, and
            # pending token are durable.  Surface that token on the error path.  If
            # no token exists, the commit state is genuinely ambiguous (e.g. a
            # PoseSession failure after changing its live Blender pose), which the
            # generator must also treat fail-closed.
            pending = _executor._pending_post_flip_followup()
            if pending is not None:
                output["scene_mutation"] = "committed"
                try:
                    output["required_followup"] = _executor._post_flip_followup_payload(
                        pending
                    )
                except Exception:  # noqa: BLE001 - retain committed verdict
                    pass
            else:
                output["scene_mutation"] = "unknown"
        return {"status": "error", "output": output}


@mcp.tool()
def edit_object_poses(edits: list, reason: str) -> dict[str, object]:
    """Typed composition pose batch followed by strict physics."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup("edit_object_poses")
    if blocked is not None:
        return blocked
    try:
        return _executor.edit_composition_object_poses(edits, reason)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "output": {"text": [f"edit_object_poses failed: {exc}"]},
        }


@mcp.tool()
def nudge_object(
    object: str,
    translation: list,
    reason: str,  # noqa: A002
) -> dict[str, object]:
    """Transactional initializer-only translation of one object/support stack."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup("nudge_object")
    if blocked is not None:
        return blocked
    if _executor._has_capability("initializer_code_transactions"):
        return {
            "status": "error",
            "output": {
                "text": [
                    "nudge_object is not part of this harness: translate the object with an "
                    "execute_and_evaluate object transaction (rigid translation of exactly "
                    "that object; it is simulated within its hierarchy)."
                ]
            },
        }
    try:
        return _executor.nudge_initializer_object(object, translation, reason)
    except Exception as exc:  # noqa: BLE001 - tool errors return to the generator
        return {"status": "error", "output": {"text": [f"nudge_object failed: {exc}"]}}


@mcp.tool()
def edit_object_mesh(
    code: str,
    object: str,  # noqa: A002
    reason: str,
    points: Optional[list] = None,
    labels: Optional[list] = None,
    physical_material_hint: Optional[str] = None,
    expected_resting_mode: str = "preserve",
) -> dict[str, object]:
    """Replace one materially wrong object mesh and physics-settle it."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    if getattr(
        _executor, "root_stage_name", None
    ) == "composition" and _executor._has_capability("composition_mesh_edit"):
        blocked = _guard_pending_composition_followup("edit_object_mesh")
        if blocked is not None:
            return blocked
        if points is not None or labels is not None:
            return {
                "status": "error",
                "output": {
                    "text": [
                        "Composition edit_object_mesh does not accept segmentation "
                        "points/labels: authored and replaced objects are maskless."
                    ],
                    "scene_mutation": "not_started",
                },
            }
        try:
            return _executor.edit_composition_object_mesh(
                code=code,
                object=object,
                physical_material_hint=physical_material_hint,
                expected_resting_mode=expected_resting_mode,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "output": {"text": [f"edit_object_mesh failed: {exc}"]},
            }
    return {
        "status": "error",
        "output": {
            "text": ["edit_object_mesh is enabled only for composition"],
            "scene_mutation": "not_started",
        },
    }


@mcp.tool()
def build_root_surface(
    surface_type: str,
    description: str,
    reason: str,
    center: list,
    normal_xy: list,
    width_m: float,
    height_m: float,
    thickness_m: float,
    target_region_norm: list,
    copy_material_from: Optional[str] = None,
    floor_support: Optional[str] = None,
    relationship_hints: Optional[list] = None,
) -> dict[str, object]:
    """Register and create one missing initializer wall as an atomic transaction."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup("build_root_surface")
    if blocked is not None:
        return blocked
    try:
        return _executor.build_root_surface(
            surface_type=surface_type,
            description=description,
            reason=reason,
            center=center,
            normal_xy=normal_xy,
            width_m=width_m,
            height_m=height_m,
            thickness_m=thickness_m,
            target_region_norm=target_region_norm,
            copy_material_from=copy_material_from,
            floor_support=floor_support,
            relationship_hints=relationship_hints,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "output": {"text": [f"build_root_surface failed: {exc}"]},
        }


@mcp.tool()
def remove_root_surface(surface: str, reason: str) -> dict[str, object]:
    """Remove one initializer-runtime root and its graph relationships transactionally."""
    if _executor is None:
        return {"status": "error", "output": {"text": ["Executor not initialized"]}}
    blocked = _guard_pending_composition_followup("remove_root_surface")
    if blocked is not None:
        return blocked
    try:
        return _executor.remove_root_surface(surface=surface, reason=reason)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "output": {"text": [f"remove_root_surface failed: {exc}"]},
        }


@mcp.tool()
def undo_last_step() -> dict[str, object]:
    """Undo the last executed step by reverting to previous state."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("undo_last_step")
    if blocked is not None:
        return blocked
    if not _executor.edit_history:
        return {
            "status": "error",
            "output": {"text": ["No previous edit to undo."]},
        }
    undone_kind = (
        _executor._edit_meta[-1].get("kind")
        if getattr(_executor, "_edit_meta", None)
        else None
    )
    ok, error = _executor._undo_latest_edit(record_undo_event=True)
    _journal_composition_edit(
        _executor, "composition_undo", "undone" if ok else "error",
        {"tool": "undo_last_step", "undone_kind": undone_kind, "message": str(error or "")[:400]},
    )  # fmt: skip
    if not ok:
        return {"status": "error", "output": {"text": [error]}}
    if undone_kind in {"build_root_surface", "remove_root_surface"}:
        message = (
            "Last root-surface transaction undone; Blender scene, active scene graph, "
            "relationships, and authorization ledger reverted together. Your previous "
            "agent-submitted execute_and_evaluate script remains the code_diff base."
        )
    else:
        base = getattr(_executor, "_last_code", None)
        if base:
            message = (
                "Last edit undone; scene AND script reverted to before your last "
                f"execute_and_evaluate. A code_diff now patches that earlier script "
                f"({len(base.splitlines())} lines) — anchor SEARCH text on it, not on "
                "the undone script; or send the COMPLETE script in `code`."
            )
        else:
            message = (
                "Last edit undone; scene reverted to the previous state. No earlier "
                "agent script remains as a code_diff base, so your next "
                "execute_and_evaluate must send the COMPLETE script in `code`."
            )
    output: dict = {"text": [message]}
    if error:
        output["text"].append(error)
    resolved = getattr(_executor, "_last_undo_resolved_followup", None)
    if resolved is not None:
        output["resolved_followup"] = resolved
        output["text"][0] += (
            " The exact retained rotate_180 was undone; its mandatory post-flip "
            "investigation is cancelled without refunding the one-shot or move budget."
        )
    return {"status": "success", "output": output}


@mcp.tool()
def get_scene_info() -> dict[str, object]:
    """Get scene information including objects, materials, lights, and cameras."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("get_scene_info")
    if blocked is not None:
        return blocked
    try:
        result = _executor.get_scene_info()
        return result
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def check_rules_enforced() -> dict[str, object]:
    """Check stage rules: composition coverage+penetration or initializer's 3-level ladder."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("check_rules_enforced")
    if blocked is not None:
        return blocked
    try:
        return _executor.check_rules_enforced()
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def resolve_yaw_advisory(
    advisory_id: str,
    observation_token: str,
    method: str,
    edge_matches: list,
) -> dict[str, object]:
    """Resolve the current initializer yaw advisory from backend-issued IDs only."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    if _executor.root_stage_name != "initializer":
        return {
            "status": "error",
            "output": {"text": ["resolve_yaw_advisory is initializer-only."]},
        }
    try:
        return _executor.resolve_yaw_advisory(
            advisory_id,
            observation_token,
            method,
            edge_matches,
        )
    except Exception as exc:  # noqa: BLE001 - malformed requests fail closed
        return {
            "status": "error",
            "output": {
                "text": [f"yaw advisory resolution failed: {exc}"],
                "completion_ready": False,
            },
        }


@mcp.tool()
def bypass(obj_ids: list) -> dict[str, object]:
    """Waive a bad preprocessed mask/absent preprocessed wall, never a runtime root."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("bypass")
    if blocked is not None:
        return blocked
    try:
        return _executor.bypass(obj_ids)
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def render_current_scene(
    azimuth: Optional[float] = None, elevation: Optional[float] = None
) -> dict[str, object]:
    """Render the current scene so the generator can inspect the latest visual state. When
    ``azimuth`` or ``elevation`` is given (the other defaults to 0), or for composition, which
    always uses a view, render from that camera and return the matching reference alongside —
    the REAL ground-truth target photo at the source view (0,0), or that view's pseudo-GT at a
    novel view; with BOTH omitted, render the locked source view."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("render_current_scene")
    if blocked is not None:
        return blocked
    try:
        # ONE render path whenever the pseudo-GT camera set exists (audit §12 A): a bare
        # call renders the locked source view (0,0) UNPAIRED — NOVEL_VIEW_NOTE_ON_REQUEST
        # promises a reference only when a view is explicitly passed; composition keeps
        # its always-paired behavior. A lone azimuth=30 still means (30, 0) — the schema
        # advertises the args independently, so a partial spec must never silently fall
        # back to a different viewpoint. The legacy wrapper render below survives ONLY
        # for the no-pseudo-GT fallback (cameras.json missing: nothing to route to).
        view_requested = azimuth is not None or elevation is not None
        if _executor._pseudo_gt_views():
            return _executor._render_novel_view(
                0.0 if azimuth is None else azimuth,
                0.0 if elevation is None else elevation,
                pair_reference=(
                    view_requested or _executor.root_stage_name == "composition"
                ),
            )
        return _executor.render_current_scene()
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def render_bev(size_m: Optional[float] = None) -> dict[str, object]:
    """Render a top-down orthographic bird's-eye view of the current scene (read-only) to check the
    layout from above — which SIDE each wall/surface is on relative to the camera, so a flipped or
    mis-placed surface is obvious. ``size_m`` optionally sets the square's side (metres); omit to
    auto-fit."""
    global _executor
    if _executor is None:
        return {
            "status": "error",
            "output": {
                "text": ["Executor not initialized. Call initialize_executor first."]
            },
        }
    blocked = _guard_pending_composition_followup("render_bev")
    if blocked is not None:
        return blocked
    try:
        return _executor.render_bev(size_m)
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


def main() -> None:
    """Run MCP server or execute test mode."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running blender-executor tools test...")
        # Read args from environment for convenience
        args = {
            "blender_command": os.getenv(
                "BLENDER_COMMAND", "lib/utils/third_party/blender-4.5/blender"
            ),
            "blender_file": os.getenv(
                "BLENDER_FILE",
                "data/static_scene/christmas1/reasonable_init/christmas1_gt.blend",
            ),
            "blender_script": os.getenv(
                "BLENDER_SCRIPT", "data/static_scene/generator_script.py"
            ),
            "output_dir": os.getenv("OUTPUT_DIR", "output/test/exec_blender"),
            "blender_save": os.getenv("BLENDER_SAVE", None),
            "gpu_devices": os.getenv("GPU_DEVICES", None),
        }

        print(
            "[test] initialize(...) with:",
            json.dumps(
                {k: v for k, v in args.items() if k != "gpu_devices"},
                ensure_ascii=False,
            ),
        )
        init_res = initialize(args)
        print("[test:init]", init_res)

        # Test get_scene_info
        scene_info_res = get_scene_info()
        print("[test:get_scene_info]", json.dumps(scene_info_res, ensure_ascii=False))
        raise NotImplementedError

    else:
        # Run MCP service normally
        mcp.run()


if __name__ == "__main__":
    main()
