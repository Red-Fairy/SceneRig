"""Reporting-only exemptions for durably committed initializer object edits.

The initializer ledger owns active authorization (and removes it on undo); the
stage-tagged mutation journal owns the irreversible commit decision. Both must
agree. Historical manifest digests bind the recorded evidence, not today's mesh
or placement bytes, which later composition edits can legitimately change.
Nothing here exempts an object from simulation, collision, or surface checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from lib.utils.mutation_journal import read_mutation_journal

INITIALIZER_POSE_EXCLUSION_POLICY = "exclude_initializer_edits_v1"
_LEDGER = "stages/0/initializer_object_transactions/ledger.json"
_EDIT_KINDS = {
    "execute_and_evaluate_objects",
    "add_object",
    "edit_object_mesh",
    "edit_object_pose",
}
_JOURNAL_ENVELOPE = {
    "transaction_id",
    "schema_version",
    "timestamp_utc",
    "harness_profile",
    "stage",
    "attempt_idx",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid initializer pose-exclusion provenance: {message}")


def _digest_matches(value: dict, key: str) -> bool:
    unsigned = {name: item for name, item in value.items() if name != key}
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return value.get(key) == hashlib.sha256(encoded).hexdigest()


def _transaction_id(value: object) -> int:
    _require(type(value) is int and value > 0, "invalid transaction id")
    return value


def _journal(scene: Path) -> list[dict]:
    events = []
    for row in read_mutation_journal(scene / "audit/scene_mutations.jsonl"):
        if (
            row.get("stage") == "initializer"
            and row.get("harness_profile") == "gpt6_v1"
        ):
            _transaction_id(row.get("transaction_id"))
            events.append(row)
    return events


def committed_initializer_event(
    txid: int, kind: str, manifest: dict, ledger_events: list, journal: list[dict]
) -> dict:
    """Select one commit; known abandoned-undo records do not grant authorization."""
    events = [event for event in journal if event["transaction_id"] == txid]
    terminal = [event for event in events if event.get("status") != "requested"]
    _require(
        isinstance(manifest, dict)
        and all(isinstance(event.get("details"), dict) for event in terminal),
        f"transaction {txid} has malformed commit evidence",
    )
    commits = [
        event
        for event in terminal
        if event.get("status") == "committed"
        and (event.get("details") or {}).get("phase") != "startup_undo_recovery"
    ]
    _require(
        len(commits) == 1,
        f"transaction {txid} has no unique durable initializer commit",
    )
    commit = commits[0]
    _require(
        terminal[0] == commit
        and (commit.get("details") or {}).get("artifact_manifest") == manifest,
        f"transaction {txid} commit order/manifest differs",
    )
    for event in terminal:
        peer = {
            "id": txid,
            **{
                key: value
                for key, value in event.items()
                if key not in _JOURNAL_ENVELOPE
            },
        }
        _require(
            event.get("kind") == kind and ledger_events.count(peer) == 1,
            f"transaction {txid} has no unique ledger peer",
        )
        if event is commit:
            continue
        details = event.get("details") or {}
        historical_recovery = event.get("status") == "committed" and details == {
            "phase": "startup_undo_recovery",
            "scene_mutation": "interrupted_undo_abandoned",
            "artifact_manifest": manifest,
        }
        restored = details.get("artifact_manifest")
        failed_undo = (
            event.get("status") == "error"
            and details.get("phase") == "undo_audit"
            and details.get("scene_mutation") == "undo_reverted"
            and not details.get("restore_errors")
            and isinstance(restored, dict)
            and restored.get("schema_version") == manifest.get("schema_version")
            and restored.get("harness_profile") == "gpt6_v1"
            and restored.get("transaction_id") == txid
            and restored.get("kind") == kind
            and restored.get("state") == "error"
            and restored.get("status") != "partial"
            and restored.get("artifacts") == manifest.get("artifacts")
            and _digest_matches(restored, "manifest_sha256")
        )
        _require(
            historical_recovery or failed_undo,
            f"transaction {txid} has a conflicting terminal event",
        )
    return commit


def _committed_manifest(tx: dict, ledger_events: list, journal: list[dict]) -> dict:
    txid = _transaction_id(tx.get("id"))
    kind = tx["kind"]
    manifest = tx.get("artifact_manifest")
    _require(isinstance(manifest, dict), f"transaction {txid} has no manifest")
    event = committed_initializer_event(txid, kind, manifest, ledger_events, journal)
    details = event.get("details")
    _require(
        isinstance(details, dict)
        and isinstance(manifest, dict)
        and details.get("artifact_manifest") == manifest,
        f"transaction {txid} active/journal manifests differ",
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("harness_profile") == "gpt6_v1"
        and type(manifest.get("transaction_id")) is int
        and manifest.get("transaction_id") == txid
        and manifest.get("kind") == kind
        and manifest.get("state") == "committed"
        and manifest.get("status") != "partial"
        and isinstance(manifest.get("artifacts"), dict)
        and _digest_matches(manifest, "manifest_sha256"),
        f"transaction {txid} manifest identity/digest is invalid",
    )
    _require(
        isinstance(tx.get("objects"), dict)
        and tx["objects"] == manifest.get("touched_objects") == details.get("objects")
        and tx.get("declarations") == details.get("declarations"),
        f"transaction {txid} observed changes differ",
    )
    if "authored_objects" in manifest:
        _require(
            isinstance(manifest["authored_objects"], dict)
            and tx.get("authored_objects")
            == manifest["authored_objects"]
            == details.get("authored_objects")
            and set(manifest["authored_objects"]) <= set(tx["objects"]),
            f"transaction {txid} authored changes differ",
        )
    return manifest


def _targets(tx: dict, manifest: dict) -> dict[str, dict]:
    if tx["kind"] == "execute_and_evaluate_objects":
        rows = manifest.get("targets")
    else:
        rows = [manifest.get("target")]
    _require(isinstance(rows, list), "missing committed targets")
    targets = {}
    for row in rows:
        _require(isinstance(row, dict), "invalid target")
        name, object_id = row.get("mesh_name"), row.get("object_id")
        _require(
            isinstance(name, str)
            and bool(name)
            and isinstance(object_id, str)
            and bool(object_id)
            and type(row.get("present")) is bool
            and _digest_matches(row, "semantic_sha256")
            and name not in targets,
            "target identity/digest is invalid",
        )
        if row["present"]:
            node, placement = row.get("graph_node"), row.get("placement")
            _require(
                isinstance(node, dict)
                and node.get("id") == object_id
                and node.get("kind") == "object"
                and isinstance(placement, dict)
                and placement.get("mesh_name") == name
                and f"{placement.get('category')}#{placement.get('instance')}"
                == object_id,
                "target is not a consistently bound object",
            )
        targets[name] = row
    if tx["kind"] != "execute_and_evaluate_objects":
        _require(
            len(targets) == 1
            and tx.get("target") in targets
            and targets[tx["target"]]["object_id"] == tx.get("object_id"),
            "typed target differs from its committed manifest",
        )
    return targets


def _matrix(signature: dict, name: str) -> list:
    matrix = signature.get("matrix")
    _require(
        signature.get("name") == name
        and isinstance(matrix, list)
        and len(matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in matrix)
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for row in matrix
            for value in row
        ),
        f"invalid observed signature for {name}",
    )
    return matrix


def _geometry_or_pose_changed(name: str, row: dict) -> bool:
    """Use observed pose/world geometry, not material/integrity-only changes."""
    _require(isinstance(row, dict), "invalid observed object change")
    before, after = row.get("before_signature"), row.get("after_signature")
    if after is None:
        return False
    _require(isinstance(after, dict), "invalid after signature")
    after_matrix = _matrix(after, name)
    if before is None:
        return True
    _require(isinstance(before, dict), "invalid before signature")
    before_matrix = _matrix(before, name)
    if row.get("representation_only") or row.get("numerical_jitter"):
        return False
    if before_matrix != after_matrix:
        return True
    left, right = before.get("world_semantics"), after.get("world_semantics")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False  # Old opaque integrity hashes cannot prove a geometry edit.
    return any(
        isinstance(left.get(key), str)
        and isinstance(right.get(key), str)
        and left[key] != right[key]
        for key in ("geometry_sha256", "topology_sha256")
    )


def initializer_pose_exclusions(
    scene: Path, active_mesh_names: list[str]
) -> dict[str, dict]:
    """Return explicit reporting exemptions for active initializer-edited objects.

    ``active_mesh_names`` must come from the caller's validated object inventory.
    A missing ledger is a legacy/no-edit scene and grants no exemptions. Present
    malformed or contradictory evidence raises ``ValueError``; the caller must
    report that provenance failure, never silently exempt all authored objects.
    Only two fixed JSON/JSONL paths are read, without native processes or scans.
    """
    _require(
        isinstance(active_mesh_names, list)
        and all(isinstance(name, str) and name for name in active_mesh_names)
        and len(active_mesh_names) == len(set(active_mesh_names)),
        "active mesh names are invalid",
    )
    scene = Path(scene)
    try:
        ledger = json.loads((scene / _LEDGER).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        raise ValueError("invalid initializer pose-exclusion ledger") from exc
    _require(
        isinstance(ledger, dict)
        and ledger.get("schema_version") == 1
        and isinstance(ledger.get("active_transactions"), list)
        and isinstance(ledger.get("events"), list)
        and all(isinstance(row, dict) for row in ledger["active_transactions"]),
        "invalid initializer ledger structure",
    )
    _require(
        all(isinstance(row, dict) for row in ledger["events"])
        and all(
            (row.get("kind") is None or isinstance(row["kind"], str))
            and (
                isinstance(row.get("status"), str)
                or (row.get("status") is None and row.get("kind") not in _EDIT_KINDS)
            )
            for row in ledger["active_transactions"]
        ),
        "invalid initializer ledger event/transaction fields",
    )
    transactions = [
        tx
        for tx in ledger["active_transactions"]
        if tx.get("kind") in _EDIT_KINDS and tx.get("status") == "committed"
    ]
    if not transactions:
        return {}
    ids = [_transaction_id(tx.get("id")) for tx in transactions]
    _require(len(ids) == len(set(ids)), "duplicate active transaction ids")
    try:
        journal = _journal(scene)
        exclusions = {}
        for tx in transactions:
            manifest = _committed_manifest(tx, ledger["events"], journal)
            targets = _targets(tx, manifest)
            names = set()
            # Batch touched_objects includes physics bystanders. Only the signed
            # pre-simulation authored set can authorize reporting exclusions.
            _require(
                tx["kind"] != "execute_and_evaluate_objects"
                or "authored_objects" in manifest,
                f"transaction {tx['id']} has unverified historical authorship: "
                "batch commit lacks authored_objects",
            )
            authored = manifest.get("authored_objects", tx["objects"])
            for name, change in authored.items():
                # Typed tools may record simulated collateral and world-equivalent
                # rewrites. Those are not direct authoring of the other objects.
                if (
                    tx["kind"] != "execute_and_evaluate_objects"
                    and name != tx["target"]
                ):
                    continue
                if _geometry_or_pose_changed(name, change):
                    _require(
                        name in targets and targets[name]["present"],
                        "unbound edited object",
                    )
                    names.add(name)
            declarations = tx.get("declarations") or {}
            _require(isinstance(declarations, dict), "invalid declarations")
            additions = declarations.get("added_names", {})
            _require(isinstance(additions, dict), "invalid addition identities")
            for object_id, name in additions.items():
                _require(
                    name in targets
                    and targets[name]["present"]
                    and targets[name]["object_id"] == object_id,
                    "addition has no committed present target",
                )
                names.add(name)
            if tx["kind"] == "add_object" and targets[tx["target"]]["present"]:
                names.add(tx["target"])
            for name in names.intersection(active_mesh_names):
                item = exclusions.setdefault(
                    name,
                    {
                        "reason_code": "initializer_edited",
                        "stage": "initializer",
                        "policy": INITIALIZER_POSE_EXCLUSION_POLICY,
                        "transaction_ids": [],
                        "mutation_kinds": [],
                        "provenance": _LEDGER,
                    },
                )
                item["transaction_ids"].append(tx["id"])
                item["mutation_kinds"].append(tx["kind"])
        for item in exclusions.values():
            item["transaction_ids"] = sorted(set(item["transaction_ids"]))
            item["mutation_kinds"] = sorted(set(item["mutation_kinds"]))
        return dict(sorted(exclusions.items()))
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        raise ValueError(
            f"invalid initializer pose-exclusion provenance: {exc}"
        ) from exc
