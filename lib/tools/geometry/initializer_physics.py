"""One joint physical outcome inside an existing initializer transaction.

The caller owns rollback, reference frames, provenance, commit and rendering. This
operation returns world deltas; the caller bakes onto the original candidate, retaining
its mesh frames, hierarchy and materials. It never writes final-certification files.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from isaac.physics_config import EXPORT_VELOCITY_REST_POLICY

from lib.tools.geometry.composition_physics import (
    PhysicsSettlementRejected,
    _add_overrides,
    _boot_colliders,
    _gpt6_certification_contract,
    _log_boot_overrides,
    _merge_vlm_physics,
    _preprocess_records,
    _protected_support_names,
    _require_exact_names,
    _rollable_flags,
    _validate_joint_certification,
    _validated_support_records,
)
from lib.tools.geometry.physics import DEFAULT_ISAAC_PYTHON, SettleClient, capsized
from lib.tools.geometry.register import RenderClient
from lib.tools.geometry.surface_validation import audit_surface_meshes


def fresh_transaction_workdir(scene, transaction_id) -> Path:
    ""
    scene = Path(scene)
    work = scene / "physics" / f"initializer_tx_{int(transaction_id)}"
    if work.exists():
        stale = work.with_name(f"{work.name}.stale_{time.strftime('%Y%m%dT%H%M%S')}")
        n = 0
        while stale.exists():
            n += 1
            stale = work.with_name(
                f"{work.name}.stale_{time.strftime('%Y%m%dT%H%M%S')}_{n}"
            )
        os.replace(work, stale)
        print(f"[initializer-physics] stale {work.name} moved aside to {stale.name}")
    work.mkdir(parents=True, exist_ok=False)
    return work


def settle_initializer_candidate(
    scene: str,
    blend: str,
    blender_cmd: str,
    transaction_id: int,
    *,
    isaac_python: str = DEFAULT_ISAAC_PYTHON,
    dynamic_names: list[str] | None = None,
) -> dict:
    ""
    scene = Path(scene)
    placement = json.loads((scene / "placement.json").read_text())
    names = sorted(row["mesh_name"] for row in placement["objects"])
    dynamic = (
        names if dynamic_names is None else sorted(set(names) & set(dynamic_names))
    )
    static_objects = [name for name in names if name not in set(dynamic)]
    _, authored_ids, _ = _gpt6_certification_contract(scene, names)
    # Existing collider/property helpers resolve scene inputs two levels above work.
    work = fresh_transaction_workdir(scene, transaction_id)
    register_server = Path(__file__).with_name("register_blender_server.py")
    rc = RenderClient(
        blender_cmd, str(register_server), blend, log_path=str(work / "blender.log")
    )
    isaac = None
    try:
        prep = rc.rpc(
            {
                "cmd": "prepare",
                "objects": names,
                "parents": {},
                "allow_empty_roots": True,
                "authored_empty_ids": authored_ids,
            }
        )
        _require_exact_names(
            "initializer prepared dynamic", prep.get("prepared"), names
        )
        if prep.get("authored_empty_roots") != authored_ids:
            raise RuntimeError("initializer prepared authored Empty bindings changed")
        body_parts = prep.get("body_parts")
        if not isinstance(body_parts, dict):
            raise RuntimeError("initializer prepared body membership is not a mapping")
        _require_exact_names("initializer logical body membership", body_parts, names)
        all_parts = []
        for name, children in body_parts.items():
            if (
                not isinstance(children, list)
                or not children
                or any(not isinstance(c, str) or not c for c in children)
            ):
                raise RuntimeError(f"initializer body parts are invalid for {name!r}")
            all_parts.extend(children)
        if len(all_parts) != len(set(all_parts)):
            raise RuntimeError("initializer body parts are multiply owned")
        parts = prep.get("logical_parts")
        _require_exact_names("initializer authored logical parts", parts, authored_ids)
        descendants = []
        for name, children in parts.items():
            if (
                not isinstance(children, list)
                or not children
                or any(not isinstance(c, str) or not c for c in children)
            ):
                raise RuntimeError(
                    f"initializer authored parts are invalid for {name!r}"
                )
            descendants.extend(children)
        surfaces = prep.get("surfaces")
        if (
            not isinstance(surfaces, list)
            or not surfaces
            or len(surfaces) != len(set(surfaces))
            or len(descendants) != len(set(descendants))
            or set(surfaces) & (set(names) | set(descendants))
        ):
            raise RuntimeError(
                "initializer requires actual static surfaces disjoint from dynamic bodies"
            )
        paths = rc.rpc(
            {"cmd": "dump_npz", "names": names + surfaces, "out": str(work)}
        )["paths"]
        _require_exact_names("initializer geometry dump", paths, names + surfaces)
        if any(not Path(path).is_file() for path in paths.values()):
            raise RuntimeError("initializer geometry dump is missing a file")
        surface_audit = audit_surface_meshes(paths, surfaces)
        _require_exact_names(
            "initializer static collider audit", surface_audit, surfaces
        )
        if any(row.get("valid") is not True for row in surface_audit.values()):
            raise RuntimeError("initializer actual static colliders are invalid")
        records, rollable = _preprocess_records(work), _rollable_flags(work)
        colliders, bindings = _boot_colliders(work, paths, names, records)
        _require_exact_names("initializer dynamic colliders", colliders, names)
        isaac = SettleClient(work, isaac_python, shared_dir=work.parent)
        applied = {}
        for name in dynamic:
            overrides = _merge_vlm_physics(
                work,
                name,
                paths[name],
                _add_overrides(
                    records,
                    name,
                    colliders[name],
                    binding=bindings.get(name),
                    strict=True,
                ),
            )
            applied[name] = sorted(overrides)
            isaac.rpc(
                {
                    "cmd": "add",
                    "name": name,
                    "npz": colliders[name],
                    **({"rollable": rollable[name]} if name in rollable else {}),
                    **overrides,
                }
            )
        _log_boot_overrides(work, records, applied, bindings)
        for name in surfaces:
            isaac.rpc({"cmd": "add_static", "name": name, "npz": paths[name]})
        for name in static_objects:
            # Out-of-scope object: a static obstacle at its current pose, admitted with
            # the SAME CoACD collider a dynamic body would use (audit F-L8: the visual
            # mesh dump is ~1 mm thinner than the hulls the movers are judged against).
            # It gets no drift row from the sim.
            isaac.rpc({"cmd": "add_static", "name": name, "npz": colliders[name]})
        result = isaac.rpc(
            {
                "cmd": "certify",
                "support_policy": "actual_surfaces_v1",
                "rest_policy": EXPORT_VELOCITY_REST_POLICY,
            }
        )
        rest = _validate_joint_certification(
            result, dynamic, sorted(surfaces + static_objects)
        )
        reports = []
        for name in static_objects:
            reports.append(
                {
                    "name": name,
                    "members": [name],
                    "converged": result["converged"],
                    "accepted": True,
                    "toppled": False,
                    "repair_needed": False,
                    "static": True,
                    "drift": {"dxy": 0.0, "dz": 0.0, "tilt_deg": 0.0},
                    "delta_matrix": np.eye(4).tolist(),
                    "supports": [],
                    "velocity": None,
                }
            )
        for name in dynamic:
            drift = result["drift"][name]
            toppled = capsized(
                drift["tilt_deg"], drift["dxy"] * 1000, bool(rollable.get(name))
            )
            supports = (
                _validated_support_records(
                    isaac.rpc({"cmd": "supports_of", "name": name}), subject=name
                )
                if result["converged"]
                else []
            )
            reports.append(
                {
                    "name": name,
                    "members": [name],
                    "converged": result["converged"],
                    "accepted": result["converged"],
                    "toppled": toppled,
                    "repair_needed": bool(
                        toppled or not _protected_support_names(supports)
                    ),
                    "drift": drift,
                    "delta_matrix": result["total"][name],
                    "supports": supports,
                    "velocity": rest["objects"][name],
                }
            )
        result.update(
            reports=reports,
            surface_audit=surface_audit,
            body_parts=body_parts,
            baked=False,
            dynamic_bodies=list(dynamic),
            static_bodies=list(static_objects),
        )
        (work / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
        if not result["converged"]:
            raise PhysicsSettlementRejected(
                "Initializer joint simulation did not reach rest within its step budget; "
                f"moving/invalid bodies: {rest['offending_objects']!r}",
                reports=reports,
            )
        return result
    finally:
        if isaac is not None:
            isaac.disconnect()
        rc.close()


def build_initializer_physics_bake_script(
    deltas: dict, body_parts: dict, blend: str
) -> str:
    """Apply world deltas to original logical bodies without preparing/recentering them."""
    _require_exact_names("initializer bake bodies", deltas, body_parts)
    for matrix in deltas.values():
        value = np.asarray(matrix, dtype=float)
        if value.shape != (4, 4) or not np.isfinite(value).all():
            raise ValueError("initializer bake requires finite 4x4 deltas")
    return f"""
import bpy
from mathutils import Matrix

deltas = {deltas!r}
body_parts = {body_parts!r}
bpy.context.view_layer.update()
groups = {{name: set(parts) | {{name}} for name, parts in body_parts.items()}}
owner = {{part: name for name, parts in groups.items() for part in parts}}
if sum(len(parts) for parts in groups.values()) != len(owner):
    raise RuntimeError('initializer bake logical bodies overlap')
world = {{name: bpy.data.objects[name].matrix_world.copy() for name in owner}}
updates = []
for name, group in owner.items():
    obj = bpy.data.objects[name]
    parent = obj.parent
    inherited = False
    depth = 0
    nearest_owner = None
    while parent is not None:
        depth += 1
        if nearest_owner is None and parent.name in owner:
            nearest_owner = owner[parent.name]
        parent = parent.parent
    inherited = nearest_owner == group
    if not inherited:
        updates.append((depth, name, Matrix(deltas[group]) @ world[name]))
for _, name, matrix in sorted(updates, key=lambda row: (row[0], row[1])):
    current = bpy.data.objects[name].matrix_world
    if any(matrix[r][c] != current[r][c] for r in range(4) for c in range(4)):
        bpy.data.objects[name].matrix_world = matrix
        bpy.context.view_layer.update()
bpy.ops.wm.save_as_mainfile(filepath={str(blend)!r})
"""
