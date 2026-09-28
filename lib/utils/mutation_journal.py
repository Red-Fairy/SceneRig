"""Durable scene journal frames shared by startup, transactions, and provenance."""

from __future__ import annotations

import fcntl
import json
import logging
import os

from lib.utils.durability import durable_fsync_enabled
from pathlib import Path


def _complete_payload(fd: int) -> bytes:
    """Caller holds the append lock; only a final unterminated frame is repairable."""
    end = os.lseek(fd, 0, os.SEEK_END)
    payload = os.pread(fd, end, 0)
    if payload and not payload.endswith(b"\n"):
        complete_end = payload.rfind(b"\n") + 1
        os.ftruncate(fd, complete_end)
        os.fsync(fd)
        logging.warning(
            "discarded %s-byte torn mutation-journal tail", end - complete_end
        )
        payload = payload[:complete_end]
    return payload


def _rows(payload: bytes) -> list[dict]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("mutation journal is not UTF-8") from exc
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"mutation journal line {line_number} is invalid"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"mutation journal line {line_number} is not an object")
        rows.append(row)
    return rows


def _read_snapshot(path: str | Path) -> tuple[bytes, list[dict]]:
    """Read and validate complete frames once while holding the writer lock."""
    try:
        stream = open(path, "r+b")
    except FileNotFoundError:
        return b"", []
    with stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        payload = _complete_payload(stream.fileno())
        return payload, _rows(payload)


def read_mutation_journal_payload(path: str | Path) -> bytes:
    """Return complete validated frames; a missing journal is empty."""
    return _read_snapshot(path)[0]


def read_mutation_journal(path: str | Path) -> list[dict]:
    """Return validated object rows from the same complete-frame byte snapshot."""
    return _read_snapshot(path)[1]


def append_mutation_journal(path: str | Path, row: dict) -> None:
    """Append and fsync one frame; caller durably creates the parent directory."""
    encoded = (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    with open(path, "a+b") as stream:
        fd = stream.fileno()
        fcntl.flock(fd, fcntl.LOCK_EX)
        _rows(_complete_payload(fd))
        remaining = memoryview(encoded)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("mutation journal append made no progress")
            remaining = remaining[written:]
        if durable_fsync_enabled():
            os.fsync(fd)
