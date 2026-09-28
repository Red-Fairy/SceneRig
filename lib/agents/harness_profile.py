"""Versioned, fail-closed policy profiles for the static-scene harness.

This module is intentionally dependency-free so both the lightweight CLI runner and
the agent process can resolve exactly the same profile payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from lib.utils.mutation_journal import read_mutation_journal

HARNESS_PROFILE_SCHEMA_VERSION = 1
DEFAULT_HARNESS_PROFILE = "baseline"
HARNESS_PROFILE_NAMES = (DEFAULT_HARNESS_PROFILE, "gpt6_v1")
HARNESS_CAPABILITY_NAMES = (
    "composition_direct_pose_edit",
    "runtime_object_inventory",
    "mutation_journal",
    "strict_post_edit_physics",
    "initializer_code_transactions",
    "composition_mesh_edit",
)

GPT6_SCENE_ARTIFACTS = {
    "scene_graph": "scene_graph.json",
    "placement": "placement.json",
    "physics_vlm": "physics/physics_vlm.json",
    "physics_estimate_manifest": "physics/physics_estimate_manifest.json",
    "physics_pose_changes": "physics/pose_changes.json",
    "physics_blend_base": "physics/blend_base.json",
    "runtime_object_inventory": "runtime_objects/inventory.json",
    "scene_mutation_journal": "audit/scene_mutations.jsonl",
}
_RUNTIME_RECOVERY_STATES = {
    "preparing",
    "prepared",
    "committing",
    "undo_snapshot_preparing",
    "undo_prepared",
    "undo_commit_decided",
    "recovery_needed",
    "committed",
    "rolled_back",
    "undone",
}
_RUNTIME_RECOVERY_TERMINAL_STATES = {"committed", "rolled_back", "undone"}


def resolve_harness_profile(name: Optional[str] = None) -> dict[str, Any]:
    """Return a fresh canonical, JSON-native profile manifest."""

    selected = str(name or DEFAULT_HARNESS_PROFILE)
    if selected not in HARNESS_PROFILE_NAMES:
        supported = ", ".join(HARNESS_PROFILE_NAMES)
        raise ValueError(
            f"unsupported harness profile {selected!r}; expected one of: {supported}"
        )
    if selected == DEFAULT_HARNESS_PROFILE:
        capabilities = {
            "initializer_object_add": False,
            "initializer_mesh_edit": False,
            "initializer_pose_edit": False,
            "composition_direct_pose_edit": False,
            "runtime_object_inventory": False,
            "mutation_journal": False,
            "strict_post_edit_physics": False,
        }
    else:
        capabilities = {capability: True for capability in HARNESS_CAPABILITY_NAMES}
        capabilities["composition_direct_pose_edit"] = False
    return {
        "schema_version": HARNESS_PROFILE_SCHEMA_VERSION,
        "name": selected,
        "version": 4 if selected == "gpt6_v1" else 1,
        "capabilities": capabilities,
    }


def base_manifest_profile_matches(
    payload: Mapping[str, Any], selected_profile: str
) -> bool:
    ""

    expected = resolve_harness_profile(selected_profile)
    declared = payload.get("harness_profile")
    if declared is None:
        return expected["name"] == DEFAULT_HARNESS_PROFILE
    if not isinstance(declared, Mapping):
        return False
    keys = (set(declared) | set(expected)) - {"version"}
    return all(declared.get(k) == expected.get(k) for k in keys)


def nonterminal_runtime_recovery_markers(scene_dir: str | Path) -> list[Path]:
    ""

    scene = Path(scene_dir)
    marker_root = scene / "runtime_objects" / "transactions"
    marker_paths = sorted(marker_root.glob("tx_*/recovery.json"), key=str)
    journal_path = scene / "audit" / "scene_mutations.jsonl"
    journal_rows = read_mutation_journal(journal_path)
    if not marker_paths:
        return []
    if not journal_path.is_file():
        raise RuntimeError(
            "runtime recovery marker exists without its mutation journal"
        )

    pending = []
    for marker_path in marker_paths:
        try:
            marker = json.loads(marker_path.read_text())
            if not isinstance(marker, Mapping):
                raise ValueError("marker is not an object")
            txid = int(marker.get("transaction_id") or 0)
            kind = str(marker.get("kind") or "")
            state = str(marker.get("state") or "")
            marker_stage = marker.get("stage", "initializer")
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"runtime recovery marker is invalid: {marker_path}"
            ) from exc
        if (
            marker.get("schema_version") != 1
            or txid < 1
            or marker_path.parent.name != f"tx_{txid}"
            or not kind
            or state not in _RUNTIME_RECOVERY_STATES
            or marker_stage not in {"initializer", "composition"}
        ):
            raise RuntimeError(f"runtime recovery marker is invalid: {marker_path}")
        bound = [
            row
            for row in journal_rows
            if str(row.get("transaction_id")) == str(txid)
            and row.get("kind") == kind
            and row.get("stage") == marker_stage
            and row.get("harness_profile") == "gpt6_v1"
        ]
        if not bound:
            raise RuntimeError(
                f"runtime recovery marker is not bound to a GPT-6 runtime audit: "
                f"{marker_path}"
            )
        statuses = {row.get("status") for row in bound}
        if (
            (
                state not in _RUNTIME_RECOVERY_TERMINAL_STATES
                and marker.get("payloads_pruned") is True
            )
            or (state.startswith("undo_") and "committed" not in statuses)
            or (
                state == "committed"
                and ("committed" not in statuses or "undone" in statuses)
            )
            or (state == "undone" and "undone" not in statuses)
            or (
                state == "rolled_back"
                and not statuses & {"rolled_back", "error", "rejected"}
            )
        ):
            raise RuntimeError(
                f"runtime recovery marker state {state!r} conflicts with journal: {marker_path}"
            )
        if state not in _RUNTIME_RECOVERY_TERMINAL_STATES:
            pending.append(marker_path)
    return pending


__all__ = [
    "DEFAULT_HARNESS_PROFILE",
    "HARNESS_CAPABILITY_NAMES",
    "HARNESS_PROFILE_NAMES",
    "HARNESS_PROFILE_SCHEMA_VERSION",
    "GPT6_SCENE_ARTIFACTS",
    "base_manifest_profile_matches",
    "nonterminal_runtime_recovery_markers",
    "resolve_harness_profile",
]
