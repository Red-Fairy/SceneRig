"""Fail-closed same-size group resolution for geometry preprocessing.

The initial scene proposer deliberately has no responsibility for same-size
judgments.  This module consumes its *retained* inventory later and applies one
of two policies:

* an explicit user allow-list is authoritative, allows all semantic categories,
  and needs two surviving members; or
* automatic discovery is a high-precision visual audit, needs three members,
  and excludes food/natural/deformable categories.

All VLM calls are injected.  The functions here only construct requests,
validate replies, and return a versioned registry that callers may persist.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REGISTRY_KEY = "same_size_resolution"
REGISTRY_SCHEMA_VERSION = 1
GROUP_SCHEMA_VERSION = 1
MANUAL_MIN_MEMBERS = 2
AUTOMATIC_MIN_MEMBERS = 3
DEFAULT_MODEL = "claude-opus-5"

MANUAL_RESOLVER_PROMPT_VERSION = "manual-category-resolver-v1"
AUTOMATIC_AUDIT_PROMPT_VERSION = "automatic-size-auditor-v6"
# Bump this whenever deterministic matching, strict reply validation, survivor
# revalidation, or fingerprint semantics change. Prompt hashes alone cannot detect such
# policy drift in cached/resettled scenes.
POLICY_VERSION = "same-size-policy-v5-pcand-frozen-frame"
NORMALIZATION_VERSION = "pcand-frozen-frame-v1"

MANUAL_RESOLVER_SYSTEM_PROMPT = """You are a HIGH-PRECISION category-reference resolver. You receive a completed scene inventory and explicit user-written object terms that the caller could not resolve deterministically.

Your only job is semantic reference resolution: determine which EXISTING object category, if any, each user term denotes in this particular inventory. Do NOT judge whether any instances are physically the same size. Do NOT create, rename, merge, split, or remove categories or instances. Do NOT infer a size group. The caller will separately apply the user's authoritative same-size instruction after resolution.

Use only the supplied category names, synonyms, and retained-instance ids, kinds, supports, and descriptions. A term may be a singular/plural variant, an ordinary synonym, a broader common label, or an attribute-rich description. For example, "seats" can identify "chair" when chair is the only seat category present; "wooden discs" can identify "coaster" when the retained coaster descriptions establish that they are wooden round discs.

All ordinary object identities are valid referents here, including fruit, produce, food, plants, natural, handmade, soft, and deformable objects. Those semantics must never prevent an explicit user term from resolving. This is identity resolution only, not automatic visual approval.

Resolve a term as "matched" only when exactly ONE supplied category is a plausible referent in this scene. If TWO OR MORE supplied categories are plausible referents, return "ambiguous" and list all plausible categories; never choose one merely because it has more instances, has at least three instances, appears first, or is a closer lexical match. For example, "utensils" is ambiguous when spoon, fork, and knife are present, and "writing instruments" is ambiguous when pen and pencil are present. If no supplied category is a plausible referent, return "unmatched". Root surfaces are not selectable categories; their ids may still appear as support context.

Treat every input term independently and copy it verbatim. Category strings in the output must be copied exactly from the input. Instance count is NOT a resolution signal; the caller validates the minimum member count only after semantic resolution. Keep reasons short and based on the supplied inventory.

Return JSON only, with exactly this schema:
{"resolutions":[{"term":str,"verdict":"matched"|"ambiguous"|"unmatched","selected_category":str|null,"candidate_categories":[str,...],"reason":str}]}

Output exactly one resolution for every input term, in the same order, and no others. For "matched", selected_category must be the one exact category and candidate_categories must contain exactly that same category. For "ambiguous", selected_category must be null and candidate_categories must contain at least two distinct exact categories. For "unmatched", selected_category must be null and candidate_categories must be empty."""

AUTOMATIC_AUDIT_SYSTEM_PROMPT = """You are a HIGH-PRECISION physical-size-group auditor for single-image 3D reconstruction. You receive one photo and one JSON object containing "automatic_candidates".

The caller has already resolved any explicit user-declared same-size terms outside this audit. Categories covered by an accepted user declaration are completely omitted from automatic_candidates. You must neither assess nor reproduce user declarations; make decisions only for automatic_candidates. If automatic_candidates is empty, the caller will skip this audit rather than call you.

Find only automatic groups that are safe to force to one shared real-world 3D size. FALSE POSITIVES ARE MUCH MORE HARMFUL THAN MISSES. A group is CONFIRMED only if all members are rigid manufactured duplicate products with the same canonical dimensions, supported by at least TWO matching structural traits visible in the photo (for example the same silhouette/profile plus the same part layout/proportions). Correct for perspective, depth, pose, foreshortening, and occlusion; raw pixel size alone is not evidence.

HARD EXCLUSIONS FOR AUTOMATIC CONFIRMATION:
- Never confirm natural, food, produce, plant, handmade, crumpled, soft/deformable, or otherwise organically variable objects, even when they look similar. Those require an explicit external override.
- Never confirm a singleton, root surface, detached components (body versus cap), graded/nesting sets, different glyphs/digits, or a broad/mixed category merely because its noun matches.
- Visible small/large/tiny wording, different profiles, proportions, part layouts, or model designs are counterevidence.
- A category may contain a confirmed subset and excluded members. Use exact ids; never group across categories.
- If duplicate-size identity is plausible but not directly established, output unresolved. Do not guess.

CATEGORY AND DESCRIPTIONS ARE AUTHORITATIVE SEMANTIC IDENTITY. Never reinterpret food or natural objects as toy, replica, manufactured, molded, or plastic from smooth, stylized, synthetic, rendered, or CGI appearance unless the supplied category or description explicitly says toy, model, replica, manufactured, molded, or plastic. A category or description that says doughnut, bagel, fruit, produce, or another food identity remains food and must receive the food exclusion.

Explicit comparative or opposed size descriptions within one category (for example small versus large, larger, largest, tiny, tiniest, medium, or tall in a size-grading context) are binding counterevidence. Members carrying those opposed/comparative descriptions cannot share a same-size group. One generic size adjective on one member, without an opposed or comparative size description on another member, is not decisive by itself.

ELIGIBILITY REQUIRES THREE OR MORE INSTANCES. Every input category and every confirmed automatic group must each contain at least THREE distinct retained ids. One- or two-member categories and one- or two-member groups are ineligible for automatic confirmation. Never emit a confirmed pair, including as a subset of a larger category.

Color, material, texture, finish, lighting, and render style are not independent structural traits and cannot satisfy either of the two required structural-evidence slots. Two evidence strings that merely restate such appearance properties do not establish duplicate physical dimensions. A generic featureless primitive shape alone, such as cube, sphere, cylinder, plain disc, or plain rectangular box, is insufficient: without additional distinctive manufactured geometry or part-layout evidence, output unresolved or rejected rather than confirming.

Before answering, perform a counterexample check for each proposed group: state why perspective/pose explains apparent differences and why no member is a different design or size.

Return JSON only with exactly this schema:
{"decisions":[{"category":str,"verdict":"confirmed_all"|"confirmed_subset"|"unresolved"|"rejected","groups":[{"members":[str,...],"confidence":"high","structural_evidence":[str,str],"counterexample_check":str}],"excluded_members":[{"id":str,"reason":str}],"reason":str}]}

Output exactly ONE decision for EVERY category in automatic_candidates and no decision for any other category. Each category must occur exactly once. For confirmed_all, groups must contain exactly one group covering every input id in that category and excluded_members must be empty. For confirmed_subset, every group must contain at least three unique exact input ids, groups must be disjoint, and excluded_members must list every ungrouped input id exactly once. For unresolved or rejected, groups and excluded_members must both be empty. Use rejected when the evidence or a hard exclusion rules grouping out; use unresolved only when duplicate size remains plausible but is not directly established. structural_evidence must be an array of exactly two non-empty strings. Do not place one category in multiple verdicts."""

_ACTIVE_STATUSES = frozenset({"active", "applied"})
_TERMINAL_STATUSES = frozenset(
    {"active", "applied", "ineligible_count", "normalization_failed", "detected_unlocked"}
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def normalize_user_terms(user_terms: Sequence[str] | str | None) -> list[str]:
    """Strip surrounding whitespace while preserving user spelling, case, and order."""
    if user_terms is None:
        return []
    if not isinstance(user_terms, (str, Sequence)):
        return []
    values: Sequence[Any] = [user_terms] if isinstance(user_terms, str) else user_terms
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        term = value.strip()
        key = _normal_text(term)
        if key not in seen:
            seen.add(key)
            result.append(term)
    return result


def _normal_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = re.sub(r"[_-]+", " ", text)
    return " ".join(text.split())


def _singular_candidates(word: str) -> frozenset[str]:
    """Every plausible singular of ``word``, plus the word itself.

    Single-output suffix rules mis-singularize irregular plurals (cookies->cooky,
    tomatoes->tomatoe, knives->knive), so matching intersects candidate SETS from both
    sides instead: a rule that is wrong for one word only adds an inert candidate."""
    candidates = {word}
    if len(word) > 4 and word.endswith("ies"):
        candidates.add(word[:-3] + "y")
    if len(word) > 4 and word.endswith("ves"):
        candidates.add(word[:-3] + "f")
        candidates.add(word[:-3] + "fe")
    if len(word) > 4 and word.endswith(("oes", "ches", "shes", "sses", "xes", "zes")):
        candidates.add(word[:-2])
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        candidates.add(word[:-1])
    return frozenset(candidates)


def _singular_phrase_keys(value: str) -> frozenset[str]:
    words = _normal_text(value).split()
    if not words:
        return frozenset()
    prefix = " ".join(words[:-1])
    return frozenset(
        f"{prefix} {candidate}".strip() for candidate in _singular_candidates(words[-1])
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "_", value.casefold()).strip("_") or "object"


def _strict_inventory(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise ValueError("inventory must be a sequence of category objects")
    result: list[dict[str, Any]] = []
    category_keys: set[str] = set()
    all_ids: set[str] = set()
    for raw in inventory:
        if not isinstance(raw, Mapping):
            raise ValueError("every inventory category must be an object")
        category = raw.get("category")
        if not isinstance(category, str) or not category.strip() or "#" in category:
            raise ValueError(f"invalid inventory category: {category!r}")
        category = category.strip()
        category_key = _normal_text(category)
        if category_key in category_keys:
            raise ValueError(f"duplicate normalized inventory category: {category!r}")
        category_keys.add(category_key)
        raw_synonyms = raw.get("synonyms", [])
        if not isinstance(raw_synonyms, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_synonyms
        ):
            raise ValueError(f"invalid synonyms for {category!r}")
        synonyms = list(dict.fromkeys(item.strip() for item in raw_synonyms))
        raw_instances = raw.get("instances", [])
        if not isinstance(raw_instances, list):
            raise ValueError(f"instances for {category!r} must be a list")
        instances: list[dict[str, Any]] = []
        for index, instance in enumerate(raw_instances):
            if not isinstance(instance, Mapping):
                raise ValueError(f"invalid instance in {category!r}")
            kind = instance.get("kind", "object")
            if kind == "root_surface":
                continue
            if not isinstance(kind, str) or kind != "object":
                raise ValueError(f"invalid kind in {category!r}: {kind!r}")
            member_id = instance.get("id")
            if member_id is None:
                member_index = instance.get("instance", index)
                if not isinstance(member_index, int) or member_index < 0:
                    raise ValueError(f"invalid instance index in {category!r}")
                member_id = f"{category}#{member_index}"
            if (
                not isinstance(member_id, str)
                or not member_id.startswith(f"{category}#")
                or member_id in all_ids
            ):
                raise ValueError(f"invalid or duplicate member id: {member_id!r}")
            all_ids.add(member_id)
            description = instance.get("description", "")
            support = instance.get("support")
            if not isinstance(description, str):
                raise ValueError(f"invalid description for {member_id!r}")
            if support is not None and not isinstance(support, str):
                raise ValueError(f"invalid support for {member_id!r}")
            instances.append(
                {
                    "id": member_id,
                    "kind": "object",
                    "support": support,
                    "description": description.strip(),
                }
            )
        if instances:
            result.append(
                {"category": category, "synonyms": synonyms, "instances": instances}
            )
    return result


def build_retained_inventory(
    objects: Sequence[Mapping[str, Any]],
    retained_instances: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the canonical resolver inventory from ``masks.json`` structures."""
    synonym_map: dict[str, list[str]] = {}
    for obj in objects:
        if isinstance(obj, Mapping) and isinstance(obj.get("category"), str):
            raw_synonyms = obj.get("synonyms") or []
            synonym_map[obj["category"]] = [
                value
                for value in raw_synonyms
                if isinstance(value, str) and value.strip()
            ]
    grouped: dict[str, dict[str, Any]] = {}
    for row in retained_instances:
        if not isinstance(row, Mapping) or row.get("kind") == "root_surface":
            continue
        category = row.get("category")
        instance = row.get("instance")
        if not isinstance(category, str) or not isinstance(instance, int):
            raise ValueError(f"invalid retained instance identity: {row!r}")
        entry = grouped.setdefault(
            category,
            {
                "category": category,
                "synonyms": synonym_map.get(category, []),
                "instances": [],
            },
        )
        entry["instances"].append(
            {
                "id": f"{category}#{instance}",
                "kind": "object",
                "support": row.get("support"),
                "description": row.get("description", ""),
            }
        )
    return _strict_inventory(list(grouped.values()))


def _inventory_identity_projection(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Fields that remain stable after same-size resolution.

    Support is useful context for the one-time semantic resolver, but build_scene_graph
    can adjudicate and write a different support back to masks.json later in the same
    preprocess. It is therefore deliberately excluded from cache/reuse fingerprints.
    """
    return [
        {
            "category": entry["category"],
            "synonyms": entry["synonyms"],
            "instances": [
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    "description": item["description"],
                }
                for item in entry["instances"]
            ],
        }
        for entry in _strict_inventory(inventory)
    ]


def inventory_fingerprint(inventory: Sequence[Mapping[str, Any]]) -> str:
    return _fingerprint(_inventory_identity_projection(inventory))


def _manual_candidates(
    inventory: list[dict[str, Any]], terms: list[str]
) -> dict[str, Any]:
    return {"terms": terms, "categories": inventory}


def _manual_candidate_fingerprint_payload(
    inventory: Sequence[Mapping[str, Any]], terms: Sequence[str]
) -> dict[str, Any]:
    normalized_terms = sorted(
        {_normal_text(term) for term in terms if _normal_text(term)}
    )
    return {
        "terms": normalized_terms,
        "categories": _inventory_identity_projection(inventory),
    }


def _automatic_candidates(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "category": entry["category"],
            "instances": [
                {"id": item["id"], "description": item["description"]}
                for item in entry["instances"]
            ],
        }
        for entry in inventory
        if len(entry["instances"]) >= AUTOMATIC_MIN_MEMBERS
    ]


def same_size_policy_fingerprint(mode: str) -> str:
    if mode == "manual_allowlist":
        payload = {
            "policy_version": POLICY_VERSION,
            "registry_schema": REGISTRY_SCHEMA_VERSION,
            "group_schema": GROUP_SCHEMA_VERSION,
            "mode": mode,
            "minimum_members": MANUAL_MIN_MEMBERS,
            "prompt_version": MANUAL_RESOLVER_PROMPT_VERSION,
            "prompt": MANUAL_RESOLVER_SYSTEM_PROMPT,
            "normalizer": "nfkc-casefold-hyphen-space-conservative-last-token-v2",
            "resolver_validator_policy": POLICY_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "semantic_exclusions": [],
        }
    elif mode == "automatic":
        payload = {
            "policy_version": POLICY_VERSION,
            "registry_schema": REGISTRY_SCHEMA_VERSION,
            "group_schema": GROUP_SCHEMA_VERSION,
            "mode": mode,
            "minimum_members": AUTOMATIC_MIN_MEMBERS,
            "prompt_version": AUTOMATIC_AUDIT_PROMPT_VERSION,
            "prompt": AUTOMATIC_AUDIT_SYSTEM_PROMPT,
            "resolver_validator_policy": POLICY_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "semantic_exclusions": [],
        }
    else:
        raise ValueError(f"unknown same-size mode: {mode!r}")
    return _fingerprint(payload)


def _image_part(image_path: str | Path) -> dict[str, Any]:
    from lib.utils.common import get_image_base64

    return {
        "type": "image_url",
        "image_url": {"url": get_image_base64(str(image_path))},
    }


def _text_part(value: Any) -> dict[str, Any]:
    return {
        "type": "text",
        "text": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
    }


def _parse_reply(raw: Any) -> dict[str, Any]:
    """Parse a literal JSON object, tolerating only one surrounding code fence."""
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        raise ValueError("VLM reply is not a JSON object or string")
    text = raw.strip()
    match = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if match:
        text = match.group(1).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("VLM reply must be one JSON object")
    return value


def _call_vlm(
    call: Callable[..., str],
    system: str,
    parts: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    return _parse_reply(call(system, parts, max_tokens=max_tokens))


def _group_id(source: str, category: str, members: Sequence[str]) -> str:
    digest = _fingerprint(
        {"source": source, "category": category, "members": sorted(members)}
    ).split(":", 1)[1][:12]
    return f"same_size:{_slug(category)}:{digest}"


_IMMUTABLE_GROUP_FIELDS = (
    "schema_version",
    "group_id",
    "canonical_category",
    "display_name",
    "requested_terms",
    "members",
    "source",
    "minimum_members",
    "evidence",
)


def _resolution_fingerprint(term_resolutions: Any, decisions: Any, groups: Any) -> str:
    immutable_groups = []
    for group in groups if isinstance(groups, list) else []:
        if isinstance(group, Mapping):
            immutable_groups.append(
                {
                    field: copy.deepcopy(group.get(field))
                    for field in _IMMUTABLE_GROUP_FIELDS
                }
            )
        else:
            immutable_groups.append(group)
    return _fingerprint(
        {
            "term_resolutions": term_resolutions,
            "decisions": decisions,
            "groups": immutable_groups,
        }
    )


def _manual_resolution_row(
    term: str,
    *,
    verdict: str,
    category: str | None,
    candidates: Sequence[str],
    method: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "term": term,
        "verdict": verdict,
        "selected_category": category,
        "candidate_categories": list(candidates),
        "method": method,
        "reason": reason,
    }


def _manual_deterministic_resolution(
    inventory: list[dict[str, Any]], term: str
) -> dict[str, Any]:
    term_keys = _singular_phrase_keys(term)

    exact = [
        entry["category"]
        for entry in inventory
        if term_keys & _singular_phrase_keys(entry["category"])
    ]
    if len(exact) == 1:
        return _manual_resolution_row(
            term,
            verdict="matched",
            category=exact[0],
            candidates=exact,
            method="deterministic_category",
            reason="unique exact or conservative singular/plural category match",
        )
    if len(exact) > 1:
        return _manual_resolution_row(
            term,
            verdict="ambiguous",
            category=None,
            candidates=exact,
            method="deterministic_ambiguous",
            reason="more than one category matches the normalized term",
        )

    synonym_matches: list[str] = []
    for entry in inventory:
        if any(
            term_keys & _singular_phrase_keys(synonym) for synonym in entry["synonyms"]
        ):
            synonym_matches.append(entry["category"])
    synonym_matches = list(dict.fromkeys(synonym_matches))
    if len(synonym_matches) == 1:
        return _manual_resolution_row(
            term,
            verdict="matched",
            category=synonym_matches[0],
            candidates=synonym_matches,
            method="deterministic_synonym",
            reason="unique exact or conservative singular/plural synonym match",
        )
    if len(synonym_matches) > 1:
        return _manual_resolution_row(
            term,
            verdict="ambiguous",
            category=None,
            candidates=synonym_matches,
            method="deterministic_ambiguous",
            reason="the term is a synonym for more than one retained category",
        )
    return _manual_resolution_row(
        term,
        verdict="unmatched",
        category=None,
        candidates=[],
        method="deterministic_unmatched",
        reason="no exact category or unique synonym match",
    )


def _validate_manual_reply(
    reply: Mapping[str, Any],
    unresolved_terms: Sequence[str],
    inventory: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    errors: list[str] = []
    if set(reply) != {"resolutions"} or not isinstance(reply.get("resolutions"), list):
        return {}, ["manual resolver reply must contain only a resolutions list"]
    categories = {entry["category"] for entry in inventory}
    expected = set(unresolved_terms)
    rows_by_term: dict[str, list[Any]] = defaultdict(list)
    for row in reply["resolutions"]:
        if isinstance(row, Mapping) and isinstance(row.get("term"), str):
            rows_by_term[row["term"]].append(row)
        else:
            errors.append("manual resolver emitted a row without an exact string term")
    extras = sorted(set(rows_by_term) - expected)
    if extras:
        errors.append(f"manual resolver emitted unknown terms: {extras}")

    accepted: dict[str, dict[str, Any]] = {}
    required_keys = {
        "term",
        "verdict",
        "selected_category",
        "candidate_categories",
        "reason",
    }
    for term in unresolved_terms:
        rows = rows_by_term.get(term, [])
        if len(rows) != 1:
            errors.append(
                f"manual resolver must emit exactly one row for {term!r}; got {len(rows)}"
            )
            continue
        row = rows[0]
        if set(row) != required_keys:
            errors.append(f"manual resolver row for {term!r} has the wrong keys")
            continue
        verdict = row.get("verdict")
        selected = row.get("selected_category")
        candidates = row.get("candidate_categories")
        reason = row.get("reason")
        if (
            verdict not in {"matched", "ambiguous", "unmatched"}
            or not isinstance(candidates, list)
            or not all(isinstance(item, str) for item in candidates)
            or len(candidates) != len(set(candidates))
            or any(item not in categories for item in candidates)
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            errors.append(f"manual resolver row for {term!r} has invalid values")
            continue
        if verdict == "matched":
            valid = (
                isinstance(selected, str)
                and selected in categories
                and candidates == [selected]
            )
        elif verdict == "ambiguous":
            valid = selected is None and len(candidates) >= 2
        else:
            valid = selected is None and candidates == []
        if not valid:
            errors.append(
                f"manual resolver row for {term!r} violates {verdict!r} schema"
            )
            continue
        accepted[term] = _manual_resolution_row(
            term,
            verdict=verdict,
            category=selected if isinstance(selected, str) else None,
            candidates=candidates,
            method="vlm_semantic",
            reason=reason.strip(),
        )
    return accepted, errors


def _groups_from_manual_resolutions(
    resolutions: Sequence[Mapping[str, Any]], inventory: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for resolution in resolutions:
        category = resolution.get("selected_category")
        if resolution.get("verdict") == "matched" and isinstance(category, str):
            by_category[category].append(str(resolution["term"]))
    inventory_by_category = {entry["category"]: entry for entry in inventory}
    groups: list[dict[str, Any]] = []
    for category, terms in by_category.items():
        members = [item["id"] for item in inventory_by_category[category]["instances"]]
        groups.append(
            {
                "schema_version": GROUP_SCHEMA_VERSION,
                "group_id": _group_id("user_explicit", category, members),
                "canonical_category": category,
                "display_name": terms[0],
                "requested_terms": terms,
                "members": members,
                "active_members": members.copy(),
                "source": "user_explicit",
                "minimum_members": MANUAL_MIN_MEMBERS,
                "status": "resolved",
                "evidence": {
                    "authority": "user_explicit",
                    "semantic_exclusions_applied": False,
                },
            }
        )
    return groups


def _resolve_manual(
    inventory: list[dict[str, Any]],
    terms: list[str],
    resolver: Callable[..., str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    resolutions = [_manual_deterministic_resolution(inventory, term) for term in terms]
    unresolved = [row["term"] for row in resolutions if row.get("verdict") != "matched"]
    errors: list[str] = []
    if unresolved:
        if resolver is None:
            errors.append(
                "manual category resolver is unavailable for unresolved user terms"
            )
        else:
            try:
                reply = _call_vlm(
                    resolver,
                    MANUAL_RESOLVER_SYSTEM_PROMPT,
                    [_text_part(_manual_candidates(inventory, unresolved))],
                    max_tokens=1200,
                )
                accepted, reply_errors = _validate_manual_reply(
                    reply, unresolved, inventory
                )
                errors.extend(reply_errors)
                resolutions = [accepted.get(row["term"], row) for row in resolutions]
            except Exception as exc:  # noqa: BLE001 - fail closed on model/JSON failure
                errors.append(f"manual resolver failed: {type(exc).__name__}: {exc}")
    return resolutions, _groups_from_manual_resolutions(resolutions, inventory), errors


def _validate_auto_group(
    raw: Any,
    *,
    category: str,
    known_ids: set[str],
) -> tuple[dict[str, Any] | None, str | None]:
    required_keys = {
        "members",
        "confidence",
        "structural_evidence",
        "counterexample_check",
    }
    if not isinstance(raw, Mapping) or set(raw) != required_keys:
        return None, "group has the wrong keys"
    members = raw.get("members")
    evidence = raw.get("structural_evidence")
    if (
        not isinstance(members, list)
        or len(members) < AUTOMATIC_MIN_MEMBERS
        or not all(isinstance(item, str) for item in members)
        or len(members) != len(set(members))
        or any(item not in known_ids for item in members)
        or raw.get("confidence") != "high"
        or not isinstance(evidence, list)
        or len(evidence) != 2
        or not all(isinstance(item, str) and item.strip() for item in evidence)
        or (
            len(evidence) == 2
            and isinstance(evidence[0], str)
            and isinstance(evidence[1], str)
            and _normal_text(evidence[0]) == _normal_text(evidence[1])
        )
        or not isinstance(raw.get("counterexample_check"), str)
        or not raw["counterexample_check"].strip()
    ):
        return None, "group has invalid membership or evidence"
    return (
        {
            "members": members,
            "confidence": "high",
            "structural_evidence": [item.strip() for item in evidence],
            "counterexample_check": raw["counterexample_check"].strip(),
            "category": category,
        },
        None,
    )


def _validate_auto_decision(
    row: Any, candidate: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
    category = candidate["category"]
    required_keys = {"category", "verdict", "groups", "excluded_members", "reason"}
    if not isinstance(row, Mapping) or set(row) != required_keys:
        return None, [], "decision has the wrong keys"
    verdict = row.get("verdict")
    groups = row.get("groups")
    excluded = row.get("excluded_members")
    reason = row.get("reason")
    if (
        row.get("category") != category
        or verdict
        not in {"confirmed_all", "confirmed_subset", "unresolved", "rejected"}
        or not isinstance(groups, list)
        or not isinstance(excluded, list)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None, [], "decision has invalid values"
    if verdict in {"unresolved", "rejected"}:
        if groups or excluded:
            return None, [], "negative decision must have empty groups and exclusions"
        return (
            {
                "category": category,
                "verdict": verdict,
                "groups": [],
                "excluded_members": [],
                "reason": reason.strip(),
            },
            [],
            None,
        )

    known_ids = {item["id"] for item in candidate["instances"]}
    accepted_groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_group in groups:
        group, error = _validate_auto_group(
            raw_group, category=category, known_ids=known_ids
        )
        if error or group is None:
            return None, [], error or "invalid group"
        members = set(group["members"])
        if seen & members:
            return None, [], "confirmed groups overlap"
        seen |= members
        accepted_groups.append(group)
    excluded_ids: list[str] = []
    for item in excluded:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"id", "reason"}
            or not isinstance(item.get("id"), str)
            or item["id"] not in known_ids
            or item["id"] in excluded_ids
            or not isinstance(item.get("reason"), str)
            or not item["reason"].strip()
        ):
            return None, [], "excluded_members is invalid"
        excluded_ids.append(item["id"])
    if verdict == "confirmed_all":
        if len(accepted_groups) != 1 or seen != known_ids or excluded_ids:
            return None, [], "confirmed_all must cover every id in one group"
    elif (
        not accepted_groups
        or seen & set(excluded_ids)
        or seen | set(excluded_ids) != known_ids
    ):
        return None, [], "confirmed_subset must partition every candidate id"
    decision = {
        "category": category,
        "verdict": verdict,
        "groups": [
            {key: value for key, value in group.items() if key != "category"}
            for group in accepted_groups
        ],
        "excluded_members": [dict(item) for item in excluded],
        "reason": reason.strip(),
    }
    return decision, accepted_groups, None


def _validate_automatic_reply(
    reply: Mapping[str, Any], candidates: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if set(reply) != {"decisions"} or not isinstance(reply.get("decisions"), list):
        return [], [], ["automatic auditor reply must contain only a decisions list"]
    expected = {candidate["category"]: candidate for candidate in candidates}
    rows: dict[str, list[Any]] = defaultdict(list)
    errors: list[str] = []
    for row in reply["decisions"]:
        if isinstance(row, Mapping) and isinstance(row.get("category"), str):
            rows[row["category"]].append(row)
        else:
            errors.append("automatic auditor emitted a row without a category")
    extras = sorted(set(rows) - set(expected))
    if extras:
        errors.append(f"automatic auditor emitted unknown categories: {extras}")

    decisions: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for category, candidate in expected.items():
        category_rows = rows.get(category, [])
        if len(category_rows) != 1:
            errors.append(
                f"automatic auditor must emit exactly one {category!r} decision; "
                f"got {len(category_rows)}"
            )
            continue
        decision, accepted, error = _validate_auto_decision(category_rows[0], candidate)
        if error or decision is None:
            errors.append(f"automatic {category!r} decision rejected: {error}")
            continue
        decisions.append(decision)
        for accepted_group in accepted:
            members = accepted_group["members"]
            groups.append(
                {
                    "schema_version": GROUP_SCHEMA_VERSION,
                    "group_id": _group_id("vlm_auto", category, members),
                    "canonical_category": category,
                    "display_name": category,
                    "requested_terms": [],
                    "members": members,
                    "active_members": members.copy(),
                    "source": "vlm_auto",
                    "minimum_members": AUTOMATIC_MIN_MEMBERS,
                    "status": "resolved",
                    "evidence": {
                        "confidence": accepted_group["confidence"],
                        "structural_evidence": accepted_group["structural_evidence"],
                        "counterexample_check": accepted_group["counterexample_check"],
                    },
                }
            )
    return decisions, groups, errors


def _resolve_automatic(
    inventory: list[dict[str, Any]],
    image_path: str | Path | None,
    auditor: Callable[..., str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    candidates = _automatic_candidates(inventory)
    if not candidates:
        return [], [], [], candidates
    if auditor is None:
        return [], [], ["automatic same-size auditor is unavailable"], candidates
    if image_path is None or not Path(image_path).is_file():
        return [], [], ["automatic same-size audit image is unavailable"], candidates
    try:
        reply = _call_vlm(
            auditor,
            AUTOMATIC_AUDIT_SYSTEM_PROMPT,
            [_text_part({"automatic_candidates": candidates}), _image_part(image_path)],
            max_tokens=2400,
        )
        decisions, groups, errors = _validate_automatic_reply(reply, candidates)
        # The automatic contract is deliberately whole-response strict. A partially
        # malformed/missing category means the auditor did not follow the exhaustive
        # counterexample protocol, so no positive from that call may resize geometry.
        if errors:
            groups = []
        return decisions, groups, errors, candidates
    except Exception as exc:  # noqa: BLE001 - fail closed on model/JSON failure
        return (
            [],
            [],
            [f"automatic auditor failed: {type(exc).__name__}: {exc}"],
            candidates,
        )


def resolve_same_size_registry(
    inventory: Sequence[Mapping[str, Any]],
    *,
    user_terms: Sequence[str] | str | None = None,
    image_path: str | Path | None = None,
    manual_resolver_vlm: Callable[..., str] | None = None,
    automatic_auditor_vlm: Callable[..., str] | None = None,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Resolve one versioned registry over the final retained object inventory.

    A nonempty user list is a manual allow-list and completely disables automatic
    discovery. Manual groups accept two members. With no user terms, every retained
    category with at least three instances is sent once to the visual auditor, whose
    calibrated prompt owns the food/natural/soft-object policy; every accepted group
    must contain at least three members.
    """
    strict_inventory = _strict_inventory(inventory)
    terms = normalize_user_terms(user_terms)
    mode = "manual_allowlist" if terms else "automatic"
    if mode == "manual_allowlist":
        resolutions, groups, errors = _resolve_manual(
            strict_inventory, terms, manual_resolver_vlm
        )
        decisions: list[dict[str, Any]] = []
        candidates: Any = _manual_candidate_fingerprint_payload(strict_inventory, terms)
        prompt = MANUAL_RESOLVER_SYSTEM_PROMPT
    else:
        decisions, groups, errors, candidates = _resolve_automatic(
            strict_inventory, image_path, automatic_auditor_vlm
        )
        resolutions = []
        prompt = AUTOMATIC_AUDIT_SYSTEM_PROMPT
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "mode": mode,
        "requested_terms": terms,
        "model": model,
        "inventory_fingerprint": inventory_fingerprint(strict_inventory),
        "candidate_fingerprint": _fingerprint(candidates),
        "policy_fingerprint": same_size_policy_fingerprint(mode),
        "prompt_fingerprint": _fingerprint(prompt),
        "term_resolutions": resolutions,
        "decisions": decisions,
        "groups": groups,
        "errors": errors,
        "validations": [],
    }
    registry["resolution_fingerprint"] = _resolution_fingerprint(
        registry["term_resolutions"], registry["decisions"], registry["groups"]
    )
    return registry


def invalid_inventory_registry(
    user_terms: Sequence[str] | str | None, model: str, error: str
) -> dict[str, Any]:
    """Schema-complete zero-group registry for an inventory that failed validation.

    Fingerprints stay None so skip-preprocess/resettle reuse fails closed, while the
    error is persisted where audit consumers already look (``errors``)."""
    terms = normalize_user_terms(user_terms)
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "mode": "manual_allowlist" if terms else "automatic",
        "requested_terms": terms,
        "model": model,
        "inventory_fingerprint": None,
        "candidate_fingerprint": None,
        "policy_fingerprint": None,
        "prompt_fingerprint": None,
        "term_resolutions": [],
        "decisions": [],
        "groups": [],
        "errors": [error],
        "validations": [],
        "resolution_fingerprint": None,
    }


def revalidate_same_size_registry(
    registry: Mapping[str, Any],
    surviving_ids: Iterable[str],
    *,
    stage: str,
) -> dict[str, Any]:
    """Reapply source-specific cardinality after a destructive pipeline stage."""
    result = copy.deepcopy(dict(registry))
    survivors = {str(value) for value in surviving_ids}
    stage_rows: list[dict[str, Any]] = []
    for group in result.get("groups", []):
        members = [str(value) for value in group.get("members", [])]
        eligible = [member for member in members if member in survivors]
        active = eligible.copy()
        minimum = int(group.get("minimum_members") or 0)
        old_status = group.get("status")
        old_active = [str(value) for value in group.get("active_members", [])]
        if old_status == "normalization_failed":
            active = []
            status = "normalization_failed"
        elif old_status == "detected_unlocked":
            # an automatic group the run chose not to lock stays a record, never re-armed
            active = []
            status = "detected_unlocked"
        elif len(eligible) < minimum:
            status = "ineligible_count"
            active = []
        elif old_status == "applied" and active == old_active:
            status = "applied"
        elif old_status == "applied":
            status = "normalization_failed"
            active = []
        else:
            status = "active"
        group["active_members"] = active
        group["status"] = status
        group["last_validation_stage"] = stage
        stage_rows.append(
            {
                "group_id": group.get("group_id"),
                "stage": stage,
                "surviving_members": eligible,
                "surviving_member_count": len(eligible),
                "minimum_members": minimum,
                "status": status,
            }
        )
    result.setdefault("validations", []).append(
        {
            "stage": stage,
            "surviving_ids_fingerprint": _fingerprint(sorted(survivors)),
            "groups": stage_rows,
        }
    )
    return result


def auto_lock_policy_conflict(
    registry: Mapping[str, Any], requested_auto_lock: bool
) -> str | None:
    """Explain why a staged registry cannot be reused under ``requested_auto_lock``.

    Only conflicts that would change meshes count: an APPLIED ``vlm_auto`` group whose
    lock is no longer requested (its meshes were already resized), or ``detected_unlocked``
    automatic groups when the lock IS requested (normalization only happens in a full
    preprocess). Registries written before the flag existed carry no ``auto_lock_enabled``;
    their applied automatic groups are judged by status, so an old registry with no
    automatic group at all stays reusable either way.
    """
    groups = [g for g in registry.get("groups", []) if isinstance(g, Mapping) and g.get("source") == "vlm_auto"]
    applied = [str(g.get("group_id")) for g in groups if g.get("status") == "applied"]
    unlocked = [str(g.get("group_id")) for g in groups if g.get("status") == "detected_unlocked"]
    if applied and not requested_auto_lock:
        return (f"staged run LOCKED automatic same-size groups {applied} but --same-size-auto-lock "
                "is not requested; their meshes were already resized")
    if unlocked and requested_auto_lock:
        return (f"staged run left automatic same-size groups {unlocked} unlocked but "
                "--same-size-auto-lock is requested; normalization needs a full preprocess")
    return None


def active_same_size_members(registry: Mapping[str, Any]) -> dict[str, str]:
    """Return exact active member id -> stable group id, rejecting overlaps."""
    result: dict[str, str] = {}
    for group in registry.get("groups", []):
        if (
            not isinstance(group, Mapping)
            or group.get("status") not in _ACTIVE_STATUSES
        ):
            continue
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            continue
        for member in group.get("active_members", []):
            member_id = str(member)
            previous = result.get(member_id)
            if previous is not None and previous != group_id:
                raise ValueError(
                    f"same-size member {member_id!r} belongs to overlapping groups"
                )
            result[member_id] = group_id
    return result


def validate_registry_for_inventory(
    registry: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    *,
    user_terms: Sequence[str] | str | None = None,
    model: str = DEFAULT_MODEL,
) -> list[str]:
    """Validate a persisted registry against current code policy and final inventory."""
    errors: list[str] = []
    try:
        strict_inventory = _strict_inventory(inventory)
    except ValueError as exc:
        return [f"invalid current inventory: {exc}"]
    terms = normalize_user_terms(user_terms)
    mode = "manual_allowlist" if terms else "automatic"
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        errors.append("registry schema_version differs")
    if registry.get("mode") != mode:
        errors.append("registry mode differs")
    staged_term_keys = sorted(
        {
            _normal_text(term)
            for term in normalize_user_terms(registry.get("requested_terms"))
        }
    )
    requested_term_keys = sorted({_normal_text(term) for term in terms})
    if staged_term_keys != requested_term_keys:
        errors.append("registry requested_terms differ")
    if registry.get("model") != model:
        errors.append("registry model differs from the configured preprocess model")
    if registry.get("inventory_fingerprint") != inventory_fingerprint(strict_inventory):
        errors.append("registry inventory fingerprint differs")
    expected_candidates: Any
    if mode == "manual_allowlist":
        expected_candidates = _manual_candidate_fingerprint_payload(
            strict_inventory, terms
        )
        expected_prompt = MANUAL_RESOLVER_SYSTEM_PROMPT
    else:
        expected_candidates = _automatic_candidates(strict_inventory)
        expected_prompt = AUTOMATIC_AUDIT_SYSTEM_PROMPT
    if registry.get("candidate_fingerprint") != _fingerprint(expected_candidates):
        errors.append("registry candidate fingerprint differs")
    if registry.get("policy_fingerprint") != same_size_policy_fingerprint(mode):
        errors.append("registry policy fingerprint differs")
    if registry.get("prompt_fingerprint") != _fingerprint(expected_prompt):
        errors.append("registry prompt fingerprint differs")
    if registry.get("resolution_fingerprint") != _resolution_fingerprint(
        registry.get("term_resolutions"),
        registry.get("decisions"),
        registry.get("groups"),
    ):
        errors.append("registry immutable resolution fingerprint differs")

    inventory_by_category = {entry["category"]: entry for entry in strict_inventory}
    known_ids = {
        item["id"]: entry["category"]
        for entry in strict_inventory
        for item in entry["instances"]
    }
    group_ids: set[str] = set()
    claimed_members: set[str] = set()
    groups = registry.get("groups")
    if not isinstance(groups, list):
        return errors + ["registry groups is not a list"]
    registry_errors = registry.get("errors")
    if not isinstance(registry_errors, list):
        errors.append("registry errors is not a list")
    elif mode == "manual_allowlist" and registry_errors:
        errors.append("manual registry contains unresolved resolver errors")
    elif registry_errors and groups:
        errors.append("automatic registry has groups despite strict audit errors")

    if mode == "manual_allowlist":
        resolutions = registry.get("term_resolutions")
        if not isinstance(resolutions, list):
            errors.append("manual registry term_resolutions is not a list")
        else:
            requested_by_key = {
                _normal_text(term): term
                for term in normalize_user_terms(registry.get("requested_terms"))
            }
            rows_by_key: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for row in resolutions:
                if isinstance(row, Mapping) and isinstance(row.get("term"), str):
                    rows_by_key[_normal_text(row["term"])].append(row)
                else:
                    errors.append("manual registry has an invalid term resolution row")
            if set(rows_by_key) != set(requested_by_key):
                errors.append("manual registry term resolution coverage differs")
            for key, rows in rows_by_key.items():
                if len(rows) != 1:
                    errors.append(f"manual term resolution {key!r} is not unique")
                    continue
                row = rows[0]
                if (
                    row.get("verdict") != "matched"
                    or row.get("selected_category") not in inventory_by_category
                    or row.get("candidate_categories") != [row.get("selected_category")]
                ):
                    errors.append(
                        f"manual term resolution {key!r} is not a valid match"
                    )
    elif registry.get("term_resolutions") not in ([], None):
        errors.append("automatic registry unexpectedly contains manual resolutions")
    for index, group in enumerate(groups):
        label = f"group[{index}]"
        if not isinstance(group, Mapping):
            errors.append(f"{label} is not an object")
            continue
        group_id = group.get("group_id")
        category = group.get("canonical_category")
        source = group.get("source")
        members = group.get("members")
        active = group.get("active_members")
        minimum = group.get("minimum_members")
        status = group.get("status")
        expected_source = "user_explicit" if mode == "manual_allowlist" else "vlm_auto"
        expected_minimum = (
            MANUAL_MIN_MEMBERS if mode == "manual_allowlist" else AUTOMATIC_MIN_MEMBERS
        )
        if not isinstance(group_id, str) or not group_id or group_id in group_ids:
            errors.append(f"{label} has an invalid or duplicate group_id")
        else:
            group_ids.add(group_id)
        if category not in inventory_by_category:
            errors.append(f"{label} has an unknown canonical category")
        if group.get("schema_version") != GROUP_SCHEMA_VERSION:
            errors.append(f"{label} has the wrong group schema_version")
        if source != expected_source or minimum != expected_minimum:
            errors.append(f"{label} has the wrong source or minimum")
        if status not in _TERMINAL_STATUSES:
            errors.append(f"{label} has an invalid status")
        if (
            not isinstance(members, list)
            or not all(isinstance(member, str) for member in members)
            or len(members) != len(set(members))
        ):
            errors.append(f"{label} has invalid members")
            members = []
        if (
            not isinstance(active, list)
            or not all(isinstance(member, str) for member in active)
            or len(active) != len(set(active))
        ):
            errors.append(f"{label} has invalid active_members")
            active = []
        if any(known_ids.get(member) != category for member in members):
            errors.append(f"{label} contains a foreign or unknown member")
        if isinstance(category, str) and category in inventory_by_category:
            category_ids = {
                item["id"] for item in inventory_by_category[category]["instances"]
            }
            if source == "user_explicit" and set(members) != category_ids:
                errors.append(
                    f"{label} manual membership does not cover every retained category id"
                )
        if (
            isinstance(group_id, str)
            and isinstance(category, str)
            and source in {"user_explicit", "vlm_auto"}
            and group_id != _group_id(source, category, members)
        ):
            errors.append(f"{label} has a non-deterministic group_id")
        if any(member not in members for member in active):
            errors.append(f"{label} active_members are not a member subset")
        overlap = claimed_members & set(members)
        if overlap:
            errors.append(f"{label} overlaps another group: {sorted(overlap)}")
        claimed_members.update(members)
        if source == "vlm_auto" and len(members) < AUTOMATIC_MIN_MEMBERS:
            errors.append(f"{label} is an automatic group below three members")
        if status in _ACTIVE_STATUSES and len(active) < expected_minimum:
            errors.append(f"{label} is active below its minimum")
        if status in {"ineligible_count", "normalization_failed"} and active:
            errors.append(f"{label} is inactive but retains active members")
    return errors
