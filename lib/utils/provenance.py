"""Bounded, secret-safe provenance for delivered Blender and Isaac artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

from lib.agents.tool_defaults import split_tool_scripts

PROMPT_POLICY_SCHEMA_VERSION = "static-scene-v3-2026-08-13"
_SECRET_MARKERS = ("api_key", "token", "password", "secret", "credential")


def sha256_file(path: str | Path | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sanitize(value: Any, key: str = "") -> Any:
    """Recursively redact keys that can carry credentials before serialization."""
    lowered = key.lower().replace("-", "_")
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "<redacted>"
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return result.stdout if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=4)
def _git_state_cached(cwd_text: str) -> dict[str, Any]:
    cwd = Path(cwd_text)
    top_text = _git(["rev-parse", "--show-toplevel"], cwd)
    if not top_text:
        return {"commit": None, "dirty": None, "changed_paths": [], "diff_sha256": None}
    top = Path(top_text.strip())
    commit = (_git(["rev-parse", "HEAD"], top) or "").strip() or None
    porcelain = _git(["status", "--porcelain=v1", "--untracked-files=all"], top) or ""
    changed_paths = sorted(
        {
            line[3:].split(" -> ")[-1]
            for line in porcelain.splitlines()
            if len(line) > 3
        }
    )
    diff = _git(["diff", "--binary", "HEAD", "--"], top) or ""
    # Include status/path identity so untracked paths affect the fingerprint without
    # embedding their potentially sensitive contents.
    diff_hash = stable_hash({"diff": diff, "status": porcelain})
    return {
        "repo_root": str(top),
        "commit": commit,
        "dirty": bool(porcelain),
        "changed_paths": changed_paths,
        "diff_sha256": diff_hash,
    }


def git_state(cwd: str | Path) -> dict[str, Any]:
    """Process-stable source snapshot; copied so callers cannot mutate the cache."""
    return json.loads(
        json.dumps(_git_state_cached(str(Path(cwd).resolve())))
    )


def _args_json_path(args: dict[str, Any]) -> Path | None:
    candidates = []
    for key in ("args_json", "moge_dir", "output_dir"):
        value = args.get(key)
        if not value:
            continue
        path = Path(value)
        candidates.extend([path, path.parent])
    for directory in candidates:
        candidate = directory if directory.name == "args.json" else directory / "args.json"
        if candidate.is_file():
            return candidate
    return None


def blender_provenance(args: dict[str, Any], cwd: str | Path) -> dict[str, Any]:
    safe_args = sanitize(args)
    tools: dict[str, list[dict[str, Any]]] = {}
    for role in ("generator", "verifier"):
        value = str(args.get(f"{role}_tools") or "")
        try:
            scripts = split_tool_scripts(value) if value else []
        except ValueError:
            scripts = []
        tools[role] = [
            {
                "path": script,
                "resolved_path": str(Path(script).resolve()),
                "sha256": sha256_file(script),
            }
            for script in scripts
        ]
    args_path = _args_json_path(args)
    return {
        "code": git_state(cwd),
        "effective_args": safe_args,
        "effective_args_sha256": stable_hash(safe_args),
        "models": {
            "agent": args.get("model"),
            "preprocess": args.get("preprocess_model") or args.get("model"),
        },
        "tool_scripts": tools,
        "prompt_policy_schema_version": PROMPT_POLICY_SCHEMA_VERSION,
        "args_json": str(args_path) if args_path else None,
        "args_json_sha256": sha256_file(args_path),
    }


def directory_manifest(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = {
        str(item.relative_to(root)): {
            "sha256": sha256_file(item),
            "size_bytes": item.stat().st_size,
        }
        for item in sorted(root.rglob("*"))
        if item.is_file()
    } if root.is_dir() else {}
    return {"files": files, "sha256": stable_hash(files)}
