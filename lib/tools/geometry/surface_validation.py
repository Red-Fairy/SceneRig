"""Read-only final-scene diagnostics, independent of proxy-assisted certification."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh

from lib.tools.geometry.physics import capsized


def audit_surface_meshes(paths: dict[str, str], surfaces: list[str]) -> dict:
    """Validate static collider inputs and record topology as diagnostic metadata."""
    result = {}
    for name in surfaces:
        if name not in paths:
            result[name] = {"valid": False, "reason": "missing mesh dump"}
            continue
        try:
            with np.load(Path(paths[name])) as data:
                parts = []
                for i in range(int(data["n"])):
                    vertices, faces = data[f"v{i}"], data[f"f{i}"]
                    if (
                        not len(vertices)
                        or not len(faces)
                        or not np.isfinite(vertices).all()
                    ):
                        raise ValueError("empty/nonfinite collider")
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                    parts.append(
                        {
                            "watertight": bool(mesh.is_watertight),
                            "z_extent_m": float(mesh.extents[2]),
                            "triangles": len(mesh.faces),
                        }
                    )
                result[name] = {"valid": bool(parts), "parts": parts}
        except (OSError, ValueError, KeyError, IndexError) as exc:
            result[name] = {"valid": False, "reason": str(exc)}
    return result


def summarize_surface_validation(
    response: dict, expected: list[str], rollable: dict, colliders: dict
) -> dict:
    """Require complete, finite evidence before calling the diagnostic a pass."""
    result = {**response, "colliders": colliders, "baked": False}
    if (
        response.get("status") != "measured"
        or response.get("support_proxy") is not False
    ):
        result["status"] = "unavailable"
        return result
    drift = response.get("drift", {})
    problems = []
    if not expected or set(drift) != set(expected):
        problems.append("incomplete dynamic-object measurements")
    if not colliders or any(not r.get("valid") for r in colliders.values()):
        problems.append("missing/invalid actual surface colliders")
    if set(response.get("surfaces", [])) != set(colliders):
        problems.append("server/client collider inventory mismatch")
    if response.get("converged") is not True:
        problems.append("actual-surface settle did not converge")
    for name, d in drift.items():
        values = [d.get(k) for k in ("dxy", "dz", "tilt_deg")]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            problems.append(f"{name}: missing/nonfinite drift")
            continue
        d["toppled"] = capsized(
            d["tilt_deg"], d["dxy"] * 1000, bool(rollable.get(name))
        )
        # Vertical fall is independent of the class-aware capsize predicate.
        d["vertical_drift_warning"] = abs(d["dz"]) > 0.02
        d["horizontal_drift_warning"] = d["dxy"] > 0.05
        if d["toppled"] or d["vertical_drift_warning"] or d["horizontal_drift_warning"]:
            problems.append(f"{name}: unstable on actual surfaces")
    result.update(status="warning" if problems else "passed", warnings=problems)
    return result
