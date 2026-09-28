"""Blender script for static scene generator with all-camera rendering."""

import hashlib
import json
import math
import os
import re
import sys
import traceback  # noqa: F401 - captured by name inside the untrusted-code guard

import bpy
import numpy as np


def _resolve_target_image_path(target_image_path):
    if not target_image_path:
        return None
    if os.path.isfile(target_image_path):
        return target_image_path
    if not os.path.isdir(target_image_path):
        return None

    preferred_names = (
        "target.png",
        "target.jpg",
        "target.jpeg",
        "visprompt1.png",
        "style1.png",
        "render1.png",
    )
    for name in preferred_names:
        candidate = os.path.join(target_image_path, name)
        if os.path.isfile(candidate):
            return candidate

    for name in sorted(os.listdir(target_image_path)):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            return os.path.join(target_image_path, name)
    return None


def _set_render_resolution_from_target(target_image_path, long_side=512):
    target_image_path = _resolve_target_image_path(target_image_path)
    width = height = None
    if target_image_path:
        try:
            image = bpy.data.images.load(target_image_path, check_existing=True)
            width, height = image.size
        except Exception as exc:
            print(
                f"[WARN] Failed to read target image size from {target_image_path}: {exc}"
            )

    if not width or not height:
        width = height = long_side

    if width >= height:
        resolution_x = long_side
        resolution_y = max(1, round(long_side * height / width))
    else:
        resolution_x = max(1, round(long_side * width / height))
        resolution_y = long_side

    bpy.context.scene.render.resolution_x = int(resolution_x)
    bpy.context.scene.render.resolution_y = int(resolution_y)
    print(f"[INFO] Render resolution set to {resolution_x}x{resolution_y}")


if __name__ == "__main__":
    # Blender options can grow (for example ``--python-exit-code 1``); resolve our
    # arguments relative to the explicit separator instead of relying on fragile
    # absolute argv positions.
    _arg_start = sys.argv.index("--") + 1
    _tool_args = sys.argv[_arg_start:]
    code_fpath = _tool_args[0]  # Path to the code file
    if len(_tool_args) > 1:
        rendering_dir = _tool_args[1]  # Path to save the rendering from camera1
    else:
        rendering_dir = None
    if len(_tool_args) > 2:
        save_blend = _tool_args[2]  # Path to save the blend file
    else:
        save_blend = None
    if len(_tool_args) > 3:
        target_image_path = _tool_args[3]  # Path to target image for aspect ratio
    else:
        target_image_path = None

    # "__norender__": run the agent's code and SAVE, but skip the render loop —
    # execute() passes this when execute_and_evaluate will immediately re-render
    # via the pseudo-GT compare path (the wrapper render would be discarded).
    norender = rendering_dir == "__norender__"
    if norender:
        rendering_dir = None

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    # A factory-empty scene has no world, so agent code that touches
    # scene.world.use_nodes / scene.world.node_tree would crash with AttributeError.
    # Ensure a node-enabled world exists before running the agent's code.
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("World")
    if not bpy.context.scene.world.use_nodes:
        bpy.context.scene.world.use_nodes = True
    # The source-view camera is LOCKED (set in preprocessing); no static_scene stage may
    # move/replace it. Snapshot its pose/lens (as values) so we can restore it after the
    # agent's code runs — otherwise an accidental camera edit persists (saved to the shared
    # blend) and every later stage renders from the wrong viewpoint.
    _c = bpy.context.scene.camera
    _cam_snap = None
    if _c is not None and _c.type == "CAMERA":
        _cam_snap = {
            "name": _c.name,
            "mw": [list(row) for row in _c.matrix_world],
            "lens": _c.data.lens,
            "sensor": _c.data.sensor_width,
            "sensor_height": _c.data.sensor_height,
            "fit": _c.data.sensor_fit,
            "shift": (_c.data.shift_x, _c.data.shift_y),
            "clip": (_c.data.clip_start, _c.data.clip_end),
            "type": _c.data.type,
        }
    def _plain(v):
        if isinstance(v, (str, bool, int)) or v is None:
            return v
        if isinstance(v, float):
            return round(v, 9)
        try:
            return [round(float(x), 9) for x in v]
        except Exception:
            return str(v)

    def _material_payload(mat, _plain_fn=_plain):
        if mat is None:
            return None
        out = {
            "name": mat.name,
            "diffuse_color": _plain_fn(mat.diffuse_color),
            "use_nodes": bool(mat.use_nodes),
        }
        for key in ("metallic", "roughness", "blend_method"):
            if hasattr(mat, key):
                out[key] = _plain_fn(getattr(mat, key))
        if mat.use_nodes and mat.node_tree:
            out["nodes"] = []
            for node in sorted(mat.node_tree.nodes, key=lambda n: n.name):
                inputs = []
                for socket in node.inputs:
                    if hasattr(socket, "default_value"):
                        inputs.append([socket.name, _plain_fn(socket.default_value)])
                image = getattr(node, "image", None)
                image_ref = None
                if image is not None:
                    image_ref = [
                        image.name,
                        image.filepath,
                        image.source,
                        image.colorspace_settings.name,
                    ]
                out["nodes"].append([node.name, node.bl_idname, inputs, image_ref])
            out["links"] = sorted(
                [link.from_node.name, link.from_socket.name,
                 link.to_node.name, link.to_socket.name]
                for link in mat.node_tree.links
            )
        return out

    def _object_integrity(
        root,
        _material_fn=_material_payload,
        _plain_fn=_plain,
        _json_dumps=json.dumps,
        _sha256=hashlib.sha256,
        _translation_only=False,
    ):
        descendants = [root] + sorted(list(root.children_recursive), key=lambda o: o.name)
        payload = []
        for obj in descendants:
            row = {
                "name": obj.name,
                "type": obj.type,
                "parent": obj.parent.name if obj.parent else None,
                "matrix_local": (
                    [[round(float(x), 9) for x in r]
                     for r in obj.matrix_local.to_3x3()]
                    if obj is root
                    else [[round(float(x), 9) for x in r] for r in obj.matrix_local]
                ),
                "scale": _plain_fn(obj.scale),
                "rotation_mode": obj.rotation_mode,
                "hide_render": bool(obj.hide_render),
                "hide_viewport": bool(obj.hide_viewport),
                "hide_get": bool(obj.hide_get()),
                "materials": [_material_fn(s.material) for s in obj.material_slots],
                "modifiers": sorted(
                    [m.name, m.type, bool(m.show_render)] for m in obj.modifiers
                ),
            }
            if _translation_only and obj is not root:
                # A trusted root translation can perturb derived matrix_local by
                # 1 nm even when the child's authored channels did not change.
                # Hash those channels exactly; world motion is checked separately.
                row["matrix_local"] = [list(r) for r in obj.matrix_basis]
                row["matrix_parent_inverse"] = [
                    list(r) for r in obj.matrix_parent_inverse
                ]
                row["parent_type"] = obj.parent_type
                row["parent_bone"] = obj.parent_bone
                row["transform_channels"] = {
                    name: list(getattr(obj, name))
                    for name in (
                        "location", "rotation_euler", "rotation_quaternion",
                        "rotation_axis_angle", "scale", "delta_location",
                        "delta_rotation_euler", "delta_rotation_quaternion",
                        "delta_scale",
                    )
                }
            if obj.type == "MESH" and obj.data is not None:
                me = obj.data
                co = np.empty(len(me.vertices) * 3, dtype=np.float32)
                me.vertices.foreach_get("co", co)
                ev = np.empty(len(me.edges) * 2, dtype=np.int64); me.edges.foreach_get("vertices", ev)
                lv = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get("vertex_index", lv)
                ls = np.empty(len(me.polygons), dtype=np.int64); me.polygons.foreach_get("loop_start", ls)
                lt = np.empty(len(me.polygons), dtype=np.int64); me.polygons.foreach_get("loop_total", lt)
                mi = np.empty(len(me.polygons), dtype=np.int64); me.polygons.foreach_get("material_index", mi)
                topo = _sha256()
                for a in (ev, lv, ls, lt, mi):
                    topo.update(a.tobytes())
                uv_layers = []
                for uv in me.uv_layers:
                    a = np.empty(len(uv.data) * 2, dtype=np.float64); uv.data.foreach_get("uv", a)
                    uv_layers.append([uv.name, _sha256(np.round(a, 9).tobytes()).hexdigest()])
                row["mesh"] = {
                    "name": me.name,
                    "vertices_sha256": _sha256(co.tobytes()).hexdigest(),
                    "topology_sha256": topo.hexdigest(),
                    "uv_layers": uv_layers,
                    "counts": [len(me.vertices), len(me.edges), len(me.polygons)],
                }
            payload.append(row)
        raw = _json_dumps(payload, sort_keys=True, separators=(",", ":"))
        result = {
            "matrix": [[float(x) for x in row] for row in root.matrix_world],
            "sha256": _sha256(raw.encode("utf-8")).hexdigest(),
        }
        if _translation_only:
            result["member_world_matrices"] = {
                obj.name: [list(row) for row in obj.matrix_world]
                for obj in descendants
            }
        return result

    def _typed_rna_payload(value, _plain_fn=_plain):
        """Stable, bounded settings for mutation-sensitive Blender RNA blocks."""
        if value is None or not hasattr(value, "bl_rna"):
            return None
        payload = []
        for prop in sorted(value.bl_rna.properties, key=lambda item: item.identifier):
            key = prop.identifier
            if key == "rna_type" or prop.type == "COLLECTION":
                continue
            # Blender's conversion operators toggle ID.tag on unrelated camera/light
            # datablocks. It is scratch bookkeeping, not authored scene content. Keep
            # every other RNA field checked, including real camera/light settings.
            if key == "tag" and isinstance(value, bpy.types.ID):
                continue
            try:
                current = getattr(value, key)
                if prop.type == "POINTER":
                    encoded = (
                        None
                        if current is None
                        else [
                            getattr(current.bl_rna, "identifier", type(current).__name__),
                            getattr(current, "name", None),
                        ]
                    )
                else:
                    encoded = _plain_fn(current)
            except Exception:
                # Some context-dependent/read-only RNA properties raise when queried in
                # background mode. Explicit geometry/material fields below remain the
                # authoritative payload; omit only that unavailable auxiliary setting.
                continue
            payload.append([key, encoded])
        return payload

    def _typed_object_snapshot(
        obj,
        _material_fn=_material_payload,
        _plain_fn=_plain,
        _rna_fn=_typed_rna_payload,
        _json_dumps=json.dumps,
        _sha256=hashlib.sha256,
    ):
        ""

        def matrix_payload(matrix):
            return [[float(component) for component in row] for row in matrix]

        visibility = {
            "hide_render": bool(obj.hide_render),
            "hide_viewport": bool(obj.hide_viewport),
            "hide_get": bool(obj.hide_get()),
            "display_type": str(getattr(obj, "display_type", "")),
            "visible_camera": bool(getattr(obj, "visible_camera", True)),
            "visible_diffuse": bool(getattr(obj, "visible_diffuse", True)),
            "visible_glossy": bool(getattr(obj, "visible_glossy", True)),
            "visible_shadow": bool(getattr(obj, "visible_shadow", True)),
            "visible_transmission": bool(
                getattr(obj, "visible_transmission", True)
            ),
            "visible_volume_scatter": bool(
                getattr(obj, "visible_volume_scatter", True)
            ),
        }
        constraints = [
            [constraint.name, constraint.type, _rna_fn(constraint)]
            for constraint in obj.constraints
        ]
        custom_properties = sorted(
            [str(key), _plain_fn(value)]
            for key, value in obj.items()
            if str(key) != "_RNA_UI"
        )
        protected = {
            "name": obj.name,
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
            "collections": sorted(collection.name for collection in obj.users_collection),
            "visibility": visibility,
            "constraints": constraints,
            "custom_properties": custom_properties,
        }

        content = {
            "data_type": (
                getattr(obj.data.bl_rna, "identifier", type(obj.data).__name__)
                if obj.data is not None and hasattr(obj.data, "bl_rna")
                else None
            ),
            "data_name": getattr(obj.data, "name", None),
            "materials": [_material_fn(slot.material) for slot in obj.material_slots],
            "modifiers": [
                [modifier.name, modifier.type, _rna_fn(modifier)]
                for modifier in obj.modifiers
            ],
        }
        if obj.type == "MESH" and obj.data is not None:
            mesh = obj.data
            content["mesh"] = {
                "vertices": [
                    [round(float(component), 9) for component in vertex.co]
                    for vertex in mesh.vertices
                ],
                "edges": [list(edge.vertices) for edge in mesh.edges],
                "polygons": [
                    [list(poly.vertices), int(poly.material_index), bool(poly.use_smooth)]
                    for poly in mesh.polygons
                ],
                "uv_layers": [
                    [
                        uv.name,
                        [
                            [round(float(loop.uv[0]), 9), round(float(loop.uv[1]), 9)]
                            for loop in uv.data
                        ],
                    ]
                    for uv in mesh.uv_layers
                ],
                "shape_keys": (
                    [
                        [
                            block.name,
                            [
                                [round(float(component), 9) for component in point.co]
                                for point in block.data
                            ],
                        ]
                        for block in mesh.shape_keys.key_blocks
                    ]
                    if mesh.shape_keys is not None
                    else []
                ),
            }
        elif obj.data is not None:
            content["data_settings"] = _rna_fn(obj.data)

        def digest(payload):
            raw = _json_dumps(payload, sort_keys=True, separators=(",", ":"))
            return _sha256(raw.encode("utf-8")).hexdigest()

        return {
            "matrix_world": matrix_payload(obj.matrix_world),
            "matrix_local": matrix_payload(obj.matrix_local),
            "protected_sha256": digest(protected),
            "content_sha256": digest(content),
            # content split so an authored object's material-only edit is separable
            "geometry_sha256": digest(
                {key: value for key, value in content.items() if key != "materials"}
            ),
            "full_sha256": digest({"protected": protected, "content": content}),
        }

    def _execute_with_integrity_snapshot(
        agent_code, integrity_fn, material_fn, plain_fn, typed_snapshot_fn
    ):
        """Run untrusted code while authorization and baselines remain function locals.

        ``exec(..., globals())`` lets ordinary Blender scripts keep their historical
        module-like behavior, but cannot rebind these locals by mutating ``os.environ``
        or assigning similarly named globals.
        """
        # Agent code executes in module globals for historical compatibility. Keep
        # every dependency used by the post-exec guard in locals and restore its
        # global binding before inspecting the result, so rebinding ``bpy``, ``json``,
        # ``hashlib`` or a helper name cannot disable the check.
        dependency_globals = {
            name: globals()[name]
            for name in (
                "bpy",
                "hashlib",
                "json",
                "math",
                "np",
                "os",
                "re",
                "sys",
                "traceback",
                "_plain",
                "_material_payload",
                "_object_integrity",
                "_typed_rna_payload",
                "_typed_object_snapshot",
            )
        }
        bpy_module = dependency_globals["bpy"]
        json_module = dependency_globals["json"]
        os_module = dependency_globals["os"]
        traceback_module = dependency_globals["traceback"]
        sys_module = dependency_globals["sys"]
        open_file = open
        pipeline_names = set(
            json_module.loads(
                os_module.environ.get("GRASE_PIPELINE_OBJECT_NAMES", "[]")
            )
        )
        pipeline_objects = {
            obj
            for name in pipeline_names
            if (obj := bpy_module.data.objects.get(name)) is not None
        }
        # Keep immutable names beside the exact pre-exec Blender references.  Once
        # untrusted code removes an object, *any* RNA property access on that saved
        # reference (including ``obj.name`` in a sort key) raises ReferenceError.
        # The stable pair lets the post-exec guard report the intended identity error
        # before dereferencing a possibly dead StructRNA.
        pipeline_object_records = tuple(
            sorted(((obj.name, obj) for obj in pipeline_objects), key=lambda row: row[0])
        )
        all_obj_objects = {
            obj.name: obj
            for obj in bpy_module.data.objects
            if obj.name.startswith("obj_")
        }
        object_snap = {obj: integrity_fn(obj) for obj in pipeline_objects}
        root_stage_name = os_module.environ.get("GRASE_ROOT_STAGE_NAME", "")
        initializer_object_transaction = None
        composition_mesh_transaction = None
        object_transaction = None
        object_transaction_kind = None
        object_transaction_root_records = {}
        object_transaction_all_records = {}
        object_transaction_before_refs = tuple(bpy_module.data.objects)

        def canonical_object_root_name(value):
            return (
                isinstance(value, str)
                and re.fullmatch(r"obj_[A-Za-z0-9_-]+_[0-9]+", value) is not None
            )

        def parse_object_transaction(raw, env_name, expected_stage):
            try:
                transaction = json_module.loads(raw)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{env_name} must be valid JSON") from exc
            required_keys = {
                "transaction_id",
                "added_objects",
                "removed_objects",
            }
            optional_keys = (
                {"authored_material_edit"} if expected_stage == "initializer" else set()
            )
            valid = (
                root_stage_name == expected_stage
                and isinstance(transaction, dict)
                and required_keys <= set(transaction) <= required_keys | optional_keys
                and isinstance(transaction.get("authored_material_edit", False), bool)
                and isinstance(transaction.get("transaction_id"), int)
                and not isinstance(transaction.get("transaction_id"), bool)
                and transaction["transaction_id"] > 0
            )
            for key in ("added_objects", "removed_objects"):
                values = transaction.get(key) if isinstance(transaction, dict) else None
                valid = valid and (
                    isinstance(values, list)
                    and all(canonical_object_root_name(value) for value in values)
                    and len(values) == len(set(values))
                )
            if not valid:
                label = env_name.removeprefix("GRASE_").lower().replace("_", " ")
                raise RuntimeError(f"invalid {label} capability")
            return transaction

        # These are trusted, one-shot root-tree transaction capabilities. Consume
        # both environment variables and retain their declaration/baselines in
        # function locals before agent code can mutate os.environ or rebind module
        # globals. The initializer form permits a broad root-set transaction and
        # whole-root pose edits; the composition form permits exactly one same-name
        # mesh replacement and freezes every surviving root tree.
        initializer_transaction_raw = os_module.environ.pop(
            "GRASE_INITIALIZER_OBJECT_TRANSACTION", None
        )
        composition_transaction_raw = os_module.environ.pop(
            "GRASE_COMPOSITION_MESH_TRANSACTION", None
        )
        if (
            initializer_transaction_raw is not None
            and composition_transaction_raw is not None
        ):
            raise RuntimeError(
                "initializer and composition object transaction capabilities are "
                "mutually exclusive"
            )
        if initializer_transaction_raw is not None:
            object_transaction = parse_object_transaction(
                initializer_transaction_raw,
                "GRASE_INITIALIZER_OBJECT_TRANSACTION",
                "initializer",
            )
            initializer_object_transaction = object_transaction
            object_transaction_kind = "initializer"
        elif composition_transaction_raw is not None:
            object_transaction = parse_object_transaction(
                composition_transaction_raw,
                "GRASE_COMPOSITION_MESH_TRANSACTION",
                "composition",
            )
            composition_mesh_transaction = object_transaction
            object_transaction_kind = "composition_mesh"
            added_values = composition_mesh_transaction["added_objects"]
            removed_values = composition_mesh_transaction["removed_objects"]
            if len(added_values) != 1 or added_values != removed_values:
                raise RuntimeError(
                    "invalid composition mesh transaction capability: added_objects "
                    "and removed_objects must contain the same one canonical root"
                )

        if object_transaction is not None:
            before_by_name = {obj.name: obj for obj in bpy_module.data.objects}
            before_roots = {
                obj.name: obj
                for obj in bpy_module.data.objects
                if obj.parent is None and obj.name.startswith("obj_")
            }
            added = set(object_transaction["added_objects"])
            removed = set(object_transaction["removed_objects"])
            missing_removed = removed - set(before_roots)
            occupied_additions = {
                name
                for name in added - removed
                if before_by_name.get(name) is not None
            }
            if missing_removed or occupied_additions:
                details = []
                if missing_removed:
                    details.append(
                        "removed_objects are not existing independent roots: "
                        + ", ".join(sorted(missing_removed))
                    )
                if occupied_additions:
                    details.append(
                        "added_objects already exist without replacement declaration: "
                        + ", ".join(sorted(occupied_additions))
                    )
                raise RuntimeError(
                    f"invalid {object_transaction_kind} object transaction declaration: "
                    + "; ".join(details)
                )

            if object_transaction_kind == "composition_mesh":
                object_transaction_all_records = {
                    obj.name: {
                        "ref": obj,
                        "parent": obj.parent,
                        "snapshot": typed_snapshot_fn(obj),
                    }
                    for obj in bpy_module.data.objects
                }
            for name, root in before_roots.items():
                members = [root] + sorted(
                    list(root.children_recursive), key=lambda item: item.name
                )
                object_transaction_root_records[name] = {
                    "ref": root,
                    "members": {
                        member.name: {
                            "ref": member,
                            "parent": member.parent,
                            "snapshot": (
                                object_transaction_all_records[member.name]["snapshot"]
                                if object_transaction_all_records
                                else typed_snapshot_fn(member)
                            ),
                        }
                        for member in members
                    },
                }
        registered_root_names = frozenset(
            json_module.loads(
                os_module.environ.get("GRASE_REGISTERED_ROOT_BUILD_NAMES", "[]")
            )
        )
        registered_root_categories = json_module.loads(
            os_module.environ.get("GRASE_REGISTERED_ROOT_CATEGORIES", "{}")
        )
        build_result_path = os_module.environ.get(
            "GRASE_INITIALIZER_BUILD_ROOT_RESULT_PATH", ""
        )
        floor_helper_allowed = (
            os_module.environ.get("GRASE_INITIALIZER_FLOOR_HELPER_ALLOWED") == "1"
        )
        runtime_bindings = json_module.loads(
            os_module.environ.get("GRASE_RUNTIME_ROOT_BINDINGS", "[]")
        )
        remove_root_binding = json_module.loads(
            os_module.environ.get("GRASE_INITIALIZER_REMOVE_ROOT_BINDING", "null")
        )
        if not isinstance(runtime_bindings, list):
            raise RuntimeError("GRASE_RUNTIME_ROOT_BINDINGS must be a list")
        if not all(isinstance(name, str) and name for name in registered_root_names):
            raise RuntimeError("registered root build names must be non-empty strings")
        if (
            not isinstance(registered_root_categories, dict)
            or set(registered_root_categories) != set(registered_root_names)
        ):
            raise RuntimeError("registered root category inventory does not match names")
        if remove_root_binding is not None and not isinstance(remove_root_binding, dict):
            raise RuntimeError("GRASE_INITIALIZER_REMOVE_ROOT_BINDING must be an object")
        # Hold exact Blender object references.  Geometry/material/transform refinement
        # remains allowed, but a delete+counterfeit-recreate under the same name cannot
        # satisfy object identity.  Missing/broken entry bindings are left to the
        # STRUCTURE rule; this guard protects only valid active runtime roots found now.
        runtime_objects = {}
        for binding in runtime_bindings:
            if not isinstance(binding, dict):
                raise RuntimeError("runtime root binding rows must be objects")
            graph_id = str(binding.get("graph_id") or "")
            build_name = str(binding.get("build_name") or "")
            try:
                graph_revision = int(binding.get("graph_revision"))
                runtime_transaction_id = int(binding.get("runtime_transaction_id"))
            except (TypeError, ValueError):
                continue
            surface_type = str(binding.get("surface_type") or "")
            obj = bpy_module.data.objects.get(build_name)
            if graph_id and build_name and surface_type:
                runtime_objects[graph_id] = {
                    "binding": {
                        "graph_id": graph_id,
                        "build_name": build_name,
                        "graph_revision": graph_revision,
                        "runtime_transaction_id": runtime_transaction_id,
                        "surface_type": surface_type,
                    },
                    "object": obj,
                    "parent": obj.parent if obj is not None else None,
                }
        allow_names = frozenset(
            json_module.loads(
                os_module.environ.get("GRASE_INITIALIZER_NUDGE_NAMES", "[]")
            )
        )
        allow_delta = tuple(
            float(x)
            for x in json_module.loads(
                os_module.environ.get(
                    "GRASE_INITIALIZER_NUDGE_DELTA", "[0,0,0]"
                )
            )
        )
        if allow_names:
            bpy_module.context.view_layer.update()
            for obj in pipeline_objects:
                if obj.name in allow_names:
                    object_snap[obj] = integrity_fn(obj, _translation_only=True)

        agent_error = None
        try:
            exec(agent_code, globals())
        except Exception as exc:
            agent_error = exc
        finally:
            globals().update(dependency_globals)
        if agent_error is not None:
            trace = "".join(
                traceback_module.format_exception(
                    type(agent_error),
                    agent_error,
                    agent_error.__traceback__,
                    limit=8,
                )
            )
            print(trace[-8000:], file=sys_module.stderr)
            raise RuntimeError(
                f"agent code failed: {type(agent_error).__name__}: {agent_error}"
            ) from agent_error
        build_root_binding = None
        if build_result_path:
            with open_file(build_result_path) as stream:
                build_root_binding = json_module.load(stream)
            if not isinstance(build_root_binding, dict):
                raise RuntimeError("trusted build-root result must be an object")
        bpy_module.context.view_layer.update()

        if object_transaction is not None:
            transaction_errors = []
            material_edited_members = []
            authored_material_edit = (
                object_transaction_kind == "initializer"
                and bool(object_transaction.get("authored_material_edit", False))
            )

            def is_authored_hierarchy(obj):
                # An authored object carries grase_authored_name on its parts
                # (procedural_object stamps them at commit); imported SAM3D objects never do.
                root = obj
                while root.parent is not None:
                    root = root.parent
                return any(
                    member.get("grase_authored_name") is not None
                    for member in [root] + list(root.children_recursive)
                )

            added = set(object_transaction["added_objects"])
            removed = set(object_transaction["removed_objects"])
            before_root_names = set(object_transaction_root_records)
            after_objects = {obj.name: obj for obj in bpy_module.data.objects}
            after_refs = tuple(after_objects.values())
            after_roots = {
                obj.name: obj
                for obj in bpy_module.data.objects
                if obj.parent is None and obj.name.startswith("obj_")
            }
            expected_root_names = (before_root_names - removed) | added
            if set(after_roots) != expected_root_names:
                missing = expected_root_names - set(after_roots)
                undeclared = set(after_roots) - expected_root_names
                if missing:
                    transaction_errors.append(
                        "declared/surviving object roots are absent: "
                        + ", ".join(sorted(missing))
                    )
                if undeclared:
                    transaction_errors.append(
                        "undeclared object roots were created or renamed: "
                        + ", ".join(sorted(undeclared))
                    )

            def transaction_matrix_error(left, right):
                return max(
                    abs(float(left[i][j]) - float(right[i][j]))
                    for i in range(4)
                    for j in range(4)
                )

            def transaction_matrix_valid(obj, require_positive=True):
                try:
                    matrix = obj.matrix_world
                    values = [float(value) for row in matrix for value in row]
                    determinant = float(matrix.to_3x3().determinant())
                    affine_error = max(
                        abs(values[12 + index] - expected)
                        for index, expected in enumerate((0.0, 0.0, 0.0, 1.0))
                    )
                    return (
                        all(math.isfinite(value) for value in values)
                        and math.isfinite(determinant)
                        and (
                            determinant > 1e-12
                            if require_positive
                            else abs(determinant) > 1e-12
                        )
                        and affine_error <= 1e-6
                    )
                except Exception:
                    return False

            def validate_unchanged_transaction_member(
                member_name, member_record, *, allow_root_pose
            ):
                current = after_objects.get(member_name)
                if current is not member_record["ref"]:
                    transaction_errors.append(
                        f"unchanged object hierarchy member {member_name} was "
                        "deleted, replaced, or renamed"
                    )
                    return
                if current.parent is not member_record["parent"]:
                    transaction_errors.append(
                        f"unchanged object hierarchy member {member_name} was reparented"
                    )
                snapshot = typed_snapshot_fn(current)
                before = member_record["snapshot"]
                if snapshot["protected_sha256"] != before["protected_sha256"]:
                    transaction_errors.append(
                        f"unchanged object hierarchy/visibility changed for {member_name}"
                    )
                if snapshot["content_sha256"] != before["content_sha256"]:
                    if (
                        authored_material_edit
                        and snapshot["geometry_sha256"] == before["geometry_sha256"]
                        and is_authored_hierarchy(current)
                    ):
                        material_edited_members.append(member_name)
                    else:
                        transaction_errors.append(
                            f"unchanged object content/material changed for {member_name}"
                            + (
                                " (material-only edits are allowed on authored objects only)"
                                if authored_material_edit
                                else ""
                            )
                        )
                if object_transaction_kind == "composition_mesh":
                    if transaction_matrix_error(
                        snapshot["matrix_world"], before["matrix_world"]
                    ) > 1e-6 or transaction_matrix_error(
                        snapshot["matrix_local"], before["matrix_local"]
                    ) > 1e-6:
                        transaction_errors.append(
                            f"surviving non-target object pose changed for {member_name}"
                        )
                elif not allow_root_pose and transaction_matrix_error(
                    snapshot["matrix_local"], before["matrix_local"]
                ) > 1e-6:
                    transaction_errors.append(
                        f"unchanged object child pose changed independently for {member_name}"
                    )
                if not transaction_matrix_valid(current, require_positive=False):
                    transaction_errors.append(
                        f"unchanged object pose is not finite positive-affine for {member_name}"
                    )

            for name in sorted(removed):
                record = object_transaction_root_records[name]
                for member_name, member_record in record["members"].items():
                    ref = member_record["ref"]
                    if any(current is ref for current in after_refs):
                        current_name = getattr(ref, "name", "<renamed>")
                        transaction_errors.append(
                            f"removed hierarchy {name} retains {member_name} as "
                            f"{current_name}"
                        )

            if object_transaction_kind == "composition_mesh":
                target_name = next(iter(removed))
                before_target_names = set(
                    object_transaction_root_records[target_name]["members"]
                )
                replacement_root = after_roots.get(target_name)
                replacement_names = (
                    {
                        member.name
                        for member in [replacement_root]
                        + list(replacement_root.children_recursive)
                    }
                    if replacement_root is not None
                    else set()
                )
                survivor_names = (
                    set(object_transaction_all_records) - before_target_names
                )
                final_non_target_names = set(after_objects) - replacement_names
                if final_non_target_names != survivor_names:
                    transaction_errors.append(
                        "composition mesh edit changed object set outside the target "
                        "hierarchy: "
                        + ", ".join(
                            sorted(final_non_target_names ^ survivor_names)
                        )
                    )

            # Every undeleted root retains its exact object tree and content. The
            # initializer transaction may transform the root handle as a whole. A
            # composition mesh transaction freezes every surviving member's world
            # and local transforms; later trusted physics code moves dependents after
            # this wrapper returns.
            for name in sorted(before_root_names - removed):
                record = object_transaction_root_records[name]
                root = after_roots.get(name)
                if root is not record["ref"]:
                    transaction_errors.append(
                        f"undeclared replacement, deletion, or rename of object root {name}"
                    )
                    continue
                live_members = [root] + sorted(
                    list(root.children_recursive), key=lambda item: item.name
                )
                live_names = {member.name for member in live_members}
                expected_names = set(record["members"])
                if live_names != expected_names:
                    transaction_errors.append(
                        f"unchanged object hierarchy {name} changed members: "
                        + ", ".join(sorted(live_names ^ expected_names))
                    )
                for member_name, member_record in record["members"].items():
                    validate_unchanged_transaction_member(
                        member_name,
                        member_record,
                        allow_root_pose=(
                            object_transaction_kind == "initializer"
                            and member_name == name
                        ),
                    )

            if material_edited_members:
                print(
                    "GRASE_AUTHORED_MATERIAL_EDIT "
                    + json_module.dumps(sorted(material_edited_members))
                )
            if object_transaction_kind == "composition_mesh":
                canonical_member_names = {
                    member_name
                    for record in object_transaction_root_records.values()
                    for member_name in record["members"]
                }
                before_target_names = set(
                    object_transaction_root_records[next(iter(removed))]["members"]
                )
                for member_name in sorted(
                    set(object_transaction_all_records)
                    - canonical_member_names
                    - before_target_names
                ):
                    validate_unchanged_transaction_member(
                        member_name,
                        object_transaction_all_records[member_name],
                        allow_root_pose=False,
                    )

            # Additions (including the new side of a declared replacement) are
            # canonical parentless Empty handles with owned, concrete mesh parts.
            for name in sorted(added):
                root = after_roots.get(name)
                if root is None:
                    continue
                members = [root] + sorted(
                    list(root.children_recursive), key=lambda item: item.name
                )
                mesh_parts = members[1:]
                if any(
                    any(member is before_ref for before_ref in object_transaction_before_refs)
                    for member in members
                ):
                    transaction_errors.append(
                        f"added object hierarchy {name} reuses a pre-existing Blender object"
                    )
                if root.type != "EMPTY" or root.parent is not None:
                    transaction_errors.append(
                        f"added object {name} must be one independent EMPTY root"
                    )
                if root.library is not None:
                    transaction_errors.append(
                        f"added object root {name} may not be externally linked"
                    )
                if (
                    getattr(root, "instance_type", "NONE") != "NONE"
                    or getattr(root, "instance_collection", None) is not None
                    or bool(getattr(root, "is_instancer", False))
                ):
                    transaction_errors.append(
                        f"added object {name} may not instance external geometry"
                    )
                if root.constraints:
                    transaction_errors.append(
                        f"added object root {name} may not retain constraints"
                    )
                if not mesh_parts or any(part.type != "MESH" for part in mesh_parts):
                    transaction_errors.append(
                        f"added object {name} must have a nonempty MESH-only child tree"
                    )
                if not transaction_matrix_valid(root):
                    transaction_errors.append(
                        f"added object root {name} requires a finite positive transform"
                    )
                for part in mesh_parts:
                    if part.type != "MESH":
                        continue
                    if (
                        part.data is None
                        or len(part.data.vertices) < 3
                        or len(part.data.polygons) < 1
                    ):
                        transaction_errors.append(
                            f"added object mesh part {part.name} is empty"
                        )
                    if (
                        part.library is not None
                        or part.data is None
                        or part.data.library is not None
                        or part.data.users != 1
                    ):
                        transaction_errors.append(
                            f"added object mesh part {part.name} shares or links external data"
                        )
                    if part.constraints:
                        transaction_errors.append(
                            f"added object mesh part {part.name} may not retain constraints"
                        )
                    if not transaction_matrix_valid(part):
                        transaction_errors.append(
                            f"added object mesh part {part.name} requires a finite positive transform"
                        )
                for member in members:
                    if (
                        bpy_module.context.scene.objects.get(member.name) is not member
                    ):
                        transaction_errors.append(
                            f"added object hierarchy member {member.name} is not linked "
                            "to the active scene"
                        )

            if transaction_errors:
                transaction_label = (
                    "Initializer object"
                    if object_transaction_kind == "initializer"
                    else "Composition mesh"
                )
                raise RuntimeError(
                    f"{transaction_label} transaction violation: "
                    + "; ".join(dict.fromkeys(transaction_errors))
                    + ". The declared object transaction was rejected before save."
                )

            # Every later identity/coverage guard must reason about the committed
            # post-transaction roots, not dead pre-exec references.
            pipeline_names.difference_update(removed)
            pipeline_names.update(added)
            pipeline_objects = {
                obj
                for name in pipeline_names
                if (obj := bpy_module.data.objects.get(name)) is not None
            }
            pipeline_object_records = tuple(
                sorted(
                    ((obj.name, obj) for obj in pipeline_objects),
                    key=lambda row: row[0],
                )
            )

        return (
            pipeline_names,
            pipeline_objects,
            pipeline_object_records,
            all_obj_objects,
            object_snap,
            root_stage_name,
            allow_names,
            allow_delta,
            integrity_fn,
            initializer_object_transaction,
            object_transaction,
            runtime_objects,
            remove_root_binding,
            registered_root_names,
            registered_root_categories,
            build_root_binding,
            floor_helper_allowed,
        )

    (
        _pipeline_names,
        _pipeline_objects,
        _pipeline_object_records,
        _all_obj_objects,
        _object_snap,
        _root_stage_name,
        _allow_names,
        _allow_delta,
        _integrity_fn,
        _initializer_object_transaction,
        _object_transaction,
        _runtime_objects,
        _remove_root_binding,
        _registered_root_names,
        _registered_root_categories,
        _build_root_binding,
        _floor_helper_allowed,
    ) = _execute_with_integrity_snapshot(
        code, _object_integrity, _material_payload, _plain, _typed_object_snapshot
    )
    # INITIALIZER object mutation is code-enforced, not merely a prompt convention.
    # Arbitrary execute_and_evaluate code may build/edit root surfaces, but imported
    # objects can move only through exec.py's trusted nudge transaction. The trusted
    # path supplies an exact allow-list + one world-space delta; every non-pose field
    # remains protected by the same signature.
    if (
        _root_stage_name == "initializer"
        and _initializer_object_transaction is None
    ):
        _errors = []
        _after_obj_names = {
            obj.name: obj for obj in bpy.data.objects if obj.name.startswith("obj_")
        }
        _expected_obj_names = set(_all_obj_objects)
        if set(_after_obj_names) != _expected_obj_names or any(
            _after_obj_names.get(name) is not ref
            for name, ref in _all_obj_objects.items()
        ):
            _errors.append(
                "obj_* identity set changed (created/deleted/renamed imported object)"
            )
        for _root_name, _root in _pipeline_object_records:
            if bpy.data.objects.get(_root_name) is not _root:
                _errors.append(f"{_root_name} was deleted, replaced, or renamed")
                continue
            _before = _object_snap[_root]
            _after = _integrity_fn(
                _root, _translation_only=_root_name in _allow_names
            )
            if _before["sha256"] != _after["sha256"]:
                _errors.append(
                    f"{_root_name} changed mesh/material/scale/rotation/hierarchy/visibility"
                )
            _expected = [list(row) for row in _before["matrix"]]
            if _root_name in _allow_names:
                for _axis in range(3):
                    _expected[_axis][3] += float(_allow_delta[_axis])
            _matrix_error = max(
                abs(float(_after["matrix"][i][j]) - float(_expected[i][j]))
                for i in range(4)
                for j in range(4)
            )
            _matrix_values = [
                float(_after["matrix"][i][j])
                for i in range(4)
                for j in range(4)
            ] + [
                float(_expected[i][j]) for i in range(4) for j in range(4)
            ]
            if not all(math.isfinite(value) for value in _matrix_values):
                _errors.append(f"{_root_name} pose contains a non-finite value")
            elif _matrix_error > 1e-6:
                _errors.append(
                    f"{_root_name} pose changed outside the authorized nudge translation"
                )
            if _root_name in _allow_names:
                for _member_name, _member_before in _before[
                    "member_world_matrices"
                ].items():
                    _member_after = _after["member_world_matrices"].get(_member_name)
                    if _member_after is None:
                        _errors.append(f"{_member_name} nudge hierarchy member is missing")
                        continue
                    _member_expected = [list(row) for row in _member_before]
                    for _axis in range(3):
                        _member_expected[_axis][3] += float(_allow_delta[_axis])
                    if any(
                        not math.isfinite(float(_member_after[i][j]))
                        or abs(
                            float(_member_after[i][j])
                            - float(_member_expected[i][j])
                        ) > 1e-6
                        for i in range(4)
                        for j in range(4)
                    ):
                        _errors.append(
                            f"{_member_name} moved outside the authorized nudge translation"
                        )
        if _errors:
            raise RuntimeError(
                "Initializer object integrity violation: "
                + "; ".join(_errors)
                + ". Imported objects may be translated only with nudge_object; "
                "this edit was rejected before save."
            )
    _pipeline_identity_errors = []
    _hierarchy_errors = []
    for _child_name, _child in _pipeline_object_records:
        if bpy.data.objects.get(_child_name) is not _child:
            _pipeline_identity_errors.append(
                f"{_child_name} was deleted, replaced, or renamed"
            )
            # A rename leaves the exact reference live, so continue collecting
            # hierarchy errors from that object.  A deletion/replacement leaves a
            # dead StructRNA; identity comparison against live collection entries is
            # safe, but no RNA property may be read from the saved reference.
            if not any(obj is _child for obj in bpy.data.objects):
                continue
        if _child.parent is not None:
            _hierarchy_errors.append(
                f"{_child_name} is parented under {_child.parent.name}"
            )
        for _constraint in _child.constraints:
            if _constraint.type != "CHILD_OF":
                continue
            _target = getattr(_constraint, "target", None)
            _hierarchy_errors.append(
                f"{_child_name} has a CHILD_OF constraint targeting "
                f"{getattr(_target, 'name', '<none>')}"
            )
    if _hierarchy_errors:
        raise RuntimeError(
            "Pipeline-object parenting is forbidden: "
            + "; ".join(_hierarchy_errors + _pipeline_identity_errors)
            + ". Apply the same world-space transform delta to each object independently; "
            "the physics backend carries support stacks."
        )
    if _pipeline_identity_errors:
        raise RuntimeError(
            "Pipeline-object integrity violation: "
            + "; ".join(_pipeline_identity_errors)
            + ". Pipeline objects may not be deleted, replaced, or renamed; "
            "this edit was rejected before save."
        )
    # Restore the locked camera: discard ANY camera the agent added/moved and re-create the
    # single source-view camera in its snapshotted pose. Keeps every render in the input view.
    if _cam_snap is not None:
        from mathutils import Matrix

        for o in [o for o in bpy.data.objects if o.type == "CAMERA"]:
            bpy.data.objects.remove(o, do_unlink=True)
        cdata = bpy.data.cameras.new(_cam_snap["name"])
        cdata.type = _cam_snap["type"]
        # sensor_fit BEFORE lens: the fit decides which image dimension the sensor
        # spans, so lens->FOV is only meaningful once it is set (same ordering the
        # arbitrary-view camera uses; see lib/tools/blender/arbitrary_view_sensor_fit_test.py).
        # PERSP only -- an ORTHO camera frames by ortho_scale and has no sensor fit.
        if _cam_snap["type"] == "PERSP":
            cdata.sensor_fit = _cam_snap["fit"]
        cdata.lens = _cam_snap["lens"]
        cdata.sensor_width = _cam_snap["sensor"]
        cdata.sensor_height = _cam_snap["sensor_height"]
        cdata.shift_x, cdata.shift_y = _cam_snap["shift"]
        cdata.clip_start, cdata.clip_end = _cam_snap["clip"]
        cobj = bpy.data.objects.new(_cam_snap["name"], cdata)
        bpy.context.scene.collection.objects.link(cobj)
        cobj.matrix_world = Matrix(_cam_snap["mw"])
        bpy.context.scene.camera = cobj

    # Guardrail against duplicate root surfaces / lights. The agent may re-run full scene
    # construction each iteration; `obj.name = "Table"` then collides with the existing one and
    # Blender appends ".001"/".002", so a duplicate accumulates (coincident copies self-shadow
    # black in Cycles; a *moved* rebuild leaves two separate surfaces, which check_penetrate
    # then flags). A ".NNN" suffix only ever appears on an exact name collision -> always an
    # accidental duplicate. Collapse each "<base>.NNN" group to ONE survivor regardless of
    # position: keep the NEWEST (the agent's latest build = highest suffix) and rename it back
    # to "<base>". obj_* object meshes (imported SAM3D) are never touched. This mirrors
    # lib.tools.blender.script_generators.dedup_surface_plan (unit-tested there); kept inline
    # because this script runs inside Blender via a bare --python invocation.
    import re as _re

    _SUF = _re.compile(r"^(.+)\.(\d{3,})$")
    _groups = {}
    # The initializer root-tree transaction permits root-surface editing: retain
    # dedup there, but exclude
    # every canonical object hierarchy so non-obj_* mesh-part names remain untouched.
    # A composition mesh transaction is likewise target-only and skips cleanup so the
    # wrapper cannot mutate a protected non-target surface after its snapshot check.
    if _object_transaction is None or _initializer_object_transaction is not None:
        _dedup_object_anchors = (
            set(_pipeline_objects)
            if _object_transaction is not None
            else set()
        )
        for _o in bpy.data.objects:
            if _o.type not in {"MESH", "LIGHT"} or _o.name.startswith("obj_"):
                continue
            _cursor = _o
            _inside_object_tree = False
            while _cursor is not None:
                if _cursor in _dedup_object_anchors:
                    _inside_object_tree = True
                    break
                _cursor = _cursor.parent
            if _inside_object_tree:
                continue
            _m = _SUF.match(_o.name)
            _groups.setdefault(_m.group(1) if _m else _o.name, []).append(_o)

    def _suf(_o):
        _m = _SUF.match(_o.name)
        return int(_m.group(2)) if _m else -1

    for _base, _members in _groups.items():
        if len(_members) < 2:
            continue
        _members.sort(key=_suf)
        _keep = _members[-1]  # newest = the agent's latest build
        for _o in _members[:-1]:
            bpy.data.objects.remove(_o, do_unlink=True)
        if _keep.name != _base:
            _keep.name = _base  # canonical name; later refs stay stable

    if _root_stage_name == "initializer":
        _build_root_ref = None
        if _build_root_binding is not None:
            _build_errors = []
            _build_graph_id = str(_build_root_binding.get("graph_id") or "")
            _build_name = str(_build_root_binding.get("build_name") or "")
            _build_surface_type = str(
                _build_root_binding.get("surface_type") or ""
            )
            try:
                _build_revision = int(_build_root_binding.get("graph_revision"))
                _build_transaction_id = int(
                    _build_root_binding.get("runtime_transaction_id")
                )
            except (TypeError, ValueError):
                _build_revision = _build_transaction_id = None
            _build_root_ref = bpy.data.objects.get(_build_name)
            if not _build_graph_id or not _build_name:
                _build_errors.append("trusted build result omitted its exact identity")
            elif _build_root_ref is None:
                _build_errors.append(
                    f"{_build_graph_id} ({_build_name}) is absent after wrapper cleanup"
                )
            else:
                if _build_root_ref.type != "MESH" or _build_root_ref.parent is not None:
                    _build_errors.append(
                        f"{_build_graph_id} must be an independent MESH root"
                    )
                if str(_build_root_ref.get("grase_graph_id", "")) != _build_graph_id:
                    _build_errors.append("backend graph-id binding did not survive")
                if (
                    _build_root_ref.get("grase_root_source")
                    != "initializer_runtime"
                ):
                    _build_errors.append("backend runtime-source binding did not survive")
                try:
                    _actual_revision = int(
                        _build_root_ref.get("grase_graph_revision", -1)
                    )
                    _actual_transaction_id = int(
                        _build_root_ref.get("grase_runtime_transaction_id", -1)
                    )
                except (TypeError, ValueError):
                    _actual_revision = _actual_transaction_id = None
                if _actual_revision != _build_revision:
                    _build_errors.append("backend graph-revision binding did not survive")
                if _actual_transaction_id != _build_transaction_id:
                    _build_errors.append(
                        "backend runtime-transaction binding did not survive"
                    )
                if (
                    str(_build_root_ref.get("grase_surface_type", ""))
                    != _build_surface_type
                    or _build_surface_type != "wall"
                ):
                    _build_errors.append("backend wall surface-type binding did not survive")
                _build_same_id = [
                    obj
                    for obj in bpy.data.objects
                    if str(obj.get("grase_graph_id", "")) == _build_graph_id
                ]
                if len(_build_same_id) != 1 or _build_same_id[0] is not _build_root_ref:
                    _build_errors.append("duplicate/counterfeit graph-id binding exists")
            if _build_errors:
                raise RuntimeError(
                    "Initializer trusted root-build validation failed: "
                    + "; ".join(_build_errors)
                    + ". The build was rejected before save."
                )

        # Only current graph roots, their descendants, canonical imported-object
        # hierarchies, the exact trusted root being added, and one exact ``floor_0``
        # helper hierarchy may own mesh geometry.  This is an exact graph inventory,
        # not a name-prefix classifier: an arbitrary direct ``rogue_wall`` cannot hide
        # from coverage as a generic helper.
        _allowed_root_anchors = {
            obj
            for name in _registered_root_names
            if (obj := bpy.data.objects.get(name)) is not None
        }
        _allowed_root_anchors.update(_pipeline_objects)
        if _build_root_ref is not None:
            _allowed_root_anchors.add(_build_root_ref)
        if _floor_helper_allowed and "floor_0" not in _registered_root_names:
            _floor_helper = bpy.data.objects.get("floor_0")
            if (
                _floor_helper is not None
                and _floor_helper.type == "MESH"
                and _floor_helper.parent is None
            ):
                _allowed_root_anchors.add(_floor_helper)

        bpy.context.view_layer.update()

        def _evaluated_world_points(_obj):
            _depsgraph = bpy.context.evaluated_depsgraph_get()
            _evaluated = _obj.evaluated_get(_depsgraph)
            _mesh = _evaluated.to_mesh()
            try:
                _points = np.asarray(
                    [
                        list(_evaluated.matrix_world @ _vertex.co)
                        for _vertex in _mesh.vertices
                    ],
                    dtype=float,
                )
            finally:
                _evaluated.to_mesh_clear()
            if len(_points) < 3 or not np.isfinite(_points).all():
                raise ValueError("insufficient finite evaluated world vertices")
            return _points

        def _wall_finite_frame(_wall):
            _points = _evaluated_world_points(_wall)
            _centered = _points - _points.mean(axis=0)
            _u, _s, _vh = np.linalg.svd(_centered, full_matrices=False)
            _normal = _vh[-1]
            _normal_norm = np.linalg.norm(_normal)
            if _normal_norm <= 1e-9:
                raise ValueError("degenerate parent-wall face-normal fit")
            _normal = _normal / _normal_norm
            _world_up = np.asarray([0.0, 0.0, 1.0])
            _run = np.cross(_world_up, _normal)
            _run_norm = np.linalg.norm(_run)
            if _run_norm <= 1e-9:
                raise ValueError("parent-wall face normal is vertical")
            _run = _run / _run_norm
            _in_plane_up = np.cross(_normal, _run)
            _in_plane_up = _in_plane_up / np.linalg.norm(_in_plane_up)
            _axes = np.asarray([_normal, _run, _in_plane_up])
            _projected = _points @ _axes.T
            return _axes, _projected.min(axis=0), _projected.max(axis=0)

        def _wall_detail_finite_error(_wall, _detail, _tolerance=0.03):
            _axes, _parent_min, _parent_max = _wall_finite_frame(_wall)
            _detail_projected = _evaluated_world_points(_detail) @ _axes.T
            _detail_min = _detail_projected.min(axis=0)
            _detail_max = _detail_projected.max(axis=0)
            _labels = ("normal", "run", "up")
            _violations = []
            _worst = 0.0
            for _axis, _label in enumerate(_labels):
                _overshoot = max(
                    float(_parent_min[_axis] - _detail_min[_axis]),
                    float(_detail_max[_axis] - _parent_max[_axis]),
                    0.0,
                )
                if _overshoot > _tolerance + 1e-6:
                    _violations.append(f"{_label} overshoot {_overshoot:.3f}m")
                    _worst = max(_worst, _overshoot)
            return ", ".join(_violations), _worst

        _descendant_contract_errors = []
        for _registered_name, _registered_category in sorted(
            _registered_root_categories.items()
        ):
            _registered_root = bpy.data.objects.get(_registered_name)
            if _registered_root is None:
                continue
            _descendants = list(_registered_root.children_recursive)
            if _registered_category != "wall" and _descendants:
                _descendant_contract_errors.append(
                    f"{_registered_name} ({_registered_category or 'unknown'}) may not "
                    "own detail children: "
                    + ", ".join(sorted(obj.name for obj in _descendants))
                )
                continue
            if _registered_category == "wall":
                _prefix = _registered_name + "_"
                _invalid_details = sorted(
                    obj.name
                    for obj in _descendants
                    if obj.type != "MESH" or not obj.name.startswith(_prefix)
                )
                if _invalid_details:
                    _descendant_contract_errors.append(
                        f"{_registered_name} wall details must be MESH objects named "
                        f"{_prefix}<part>: " + ", ".join(_invalid_details)
                    )
                for _detail in _descendants:
                    if _detail.type != "MESH" or not _detail.name.startswith(_prefix):
                        continue
                    try:
                        _finite_error, _worst_overshoot = _wall_detail_finite_error(
                            _registered_root, _detail
                        )
                    except Exception as _detail_exc:
                        _descendant_contract_errors.append(
                            f"{_detail.name} wall-detail finite bounds are unverifiable: "
                            f"{_detail_exc}"
                        )
                        continue
                    if _finite_error:
                        _stale_hint = (
                            _worst_overshoot > 0.5
                            or _detail.matrix_world.translation.length < 0.05
                        )
                        _descendant_contract_errors.append(
                            f"{_detail.name} is outside parent wall finite bounds "
                            f"(0.030m tolerance): {_finite_error}"
                            + ("; HINT: this part sits far outside its wall. A freshly created object keeps an identity matrix_world until bpy.context.view_layer.update() runs, so copying it before the update (or parenting first and then assigning WORLD coordinates to .location) drops the part at the world origin. Fix: view_layer.update(), copy matrix_world, set parent and matrix_parent_inverse = wall.matrix_world.inverted(), then restore matrix_world" if _stale_hint else "")
                        )
        if _floor_helper_allowed and "floor_0" not in _registered_root_names:
            _floor_helper = bpy.data.objects.get("floor_0")
            if _floor_helper is not None and list(_floor_helper.children_recursive):
                _descendant_contract_errors.append(
                    "the floor_0 helper must not own child objects"
                )
        if _descendant_contract_errors:
            raise RuntimeError(
                "Initializer registered-root descendant violation: "
                + "; ".join(_descendant_contract_errors)
                + ". Join non-wall structure into its exact root mesh; wall details "
                "alone may be child meshes and must use the exact <wall>_<part> name."
            )
        _renderable_root_types = {
            "MESH",
            "CURVE",
            "SURFACE",
            "META",
            "FONT",
            "VOLUME",
            "CURVES",
            "POINTCLOUD",
            "GPENCIL",
            "GREASEPENCIL",
        }
        _rogue_mesh_roots = set()
        for _mesh in bpy.context.scene.objects:
            _is_instance_empty = _mesh.type == "EMPTY" and (
                getattr(_mesh, "instance_type", "NONE") != "NONE"
                or getattr(_mesh, "instance_collection", None) is not None
                or bool(getattr(_mesh, "is_instancer", False))
            )
            if _mesh.type not in _renderable_root_types and not _is_instance_empty:
                continue
            _cursor = _mesh
            _chain = []
            while _cursor is not None:
                _chain.append(_cursor)
                _cursor = _cursor.parent
            if any(anchor in _allowed_root_anchors for anchor in _chain):
                continue
            _rogue_mesh_roots.add(_chain[-1].name)
        if _rogue_mesh_roots:
            raise RuntimeError(
                "Initializer registered-root inventory violation: unregistered direct "
                "renderable root(s): "
                + ", ".join(sorted(_rogue_mesh_roots))
                + ". Undo this edit. If a genuinely absent distinct wall is required, "
                "register it with build_root_surface; execute_and_evaluate may only "
                "build/refine current exact graph roots, their detail children, and "
                "the single level floor_0 helper."
            )

        _runtime_errors = []
        _remove_graph_id = str((_remove_root_binding or {}).get("graph_id") or "")
        _remove_build_name = str((_remove_root_binding or {}).get("build_name") or "")
        for _graph_id, _record in sorted(_runtime_objects.items()):
            _binding = _record["binding"]
            _name = _binding["build_name"]
            _ref = _record["object"]
            _removal_authorized = (
                _graph_id == _remove_graph_id and _name == _remove_build_name
            )
            _current = bpy.data.objects.get(_name)
            _same_id_objects = [
                obj
                for obj in bpy.data.objects
                if str(obj.get("grase_graph_id", "")) == _graph_id
            ]
            if _removal_authorized:
                if _current is not None or _same_id_objects:
                    _runtime_errors.append(
                        f"trusted removal of {_graph_id} did not remove its exact root"
                    )
                continue
            if _ref is None:
                _runtime_errors.append(
                    f"{_graph_id} ({_name}) was already missing before this edit"
                )
                continue
            if _current is not _ref:
                _runtime_errors.append(
                    f"{_graph_id} ({_name}) was deleted, replaced, or renamed"
                )
                continue
            if _ref.parent is not _record["parent"]:
                _runtime_errors.append(f"{_graph_id} ({_name}) was reparented")
            if _ref.parent is not None:
                _runtime_errors.append(
                    f"{_graph_id} ({_name}) is no longer an independent Blender root"
                )
            if str(_ref.get("grase_graph_id", "")) != _graph_id:
                _runtime_errors.append(f"{_graph_id} graph binding changed")
            if _ref.get("grase_root_source") != "initializer_runtime":
                _runtime_errors.append(f"{_graph_id} runtime source binding changed")
            try:
                _current_graph_revision = int(
                    _ref.get("grase_graph_revision", -1)
                )
            except (TypeError, ValueError):
                _current_graph_revision = None
            if _current_graph_revision != int(_binding["graph_revision"]):
                _runtime_errors.append(f"{_graph_id} graph revision binding changed")
            try:
                _current_transaction_id = int(
                    _ref.get("grase_runtime_transaction_id", -1)
                )
            except (TypeError, ValueError):
                _current_transaction_id = None
            if _current_transaction_id != int(_binding["runtime_transaction_id"]):
                _runtime_errors.append(
                    f"{_graph_id} runtime transaction binding changed"
                )
            if str(_ref.get("grase_surface_type", "")) != _binding["surface_type"]:
                _runtime_errors.append(f"{_graph_id} surface-type binding changed")
            if len(_same_id_objects) != 1 or _same_id_objects[0] is not _ref:
                _runtime_errors.append(
                    f"{_graph_id} has a counterfeit or duplicate runtime binding"
                )
            if any(c.type == "CHILD_OF" for c in _ref.constraints):
                _runtime_errors.append(f"{_graph_id} has a forbidden CHILD_OF constraint")
        if _runtime_errors:
            raise RuntimeError(
                "Initializer runtime-root identity violation: "
                + "; ".join(_runtime_errors)
                + ". Runtime roots returned by build_root_surface may be refined in "
                "place, but never deleted/replaced/renamed/reparented by "
                "execute_and_evaluate. Use remove_root_surface for a deliberate removal."
            )

    if not rendering_dir and not norender:
        # Info-only calls (get_scene_info): exit BEFORE the save on purpose.
        print("[INFO] No rendering directory provided, skipping rendering.")
        exit(0)

    if rendering_dir:
        render_engine = os.environ.get("VIGA_RENDER_ENGINE", "CYCLES")
        try:
            bpy.context.scene.render.engine = render_engine
        except Exception:
            bpy.context.scene.render.engine = "CYCLES"

        if bpy.context.scene.render.engine == "CYCLES":
            bpy.context.preferences.addons[
                "cycles"
            ].preferences.compute_device_type = "CUDA"  # or 'OPTIX' if your GPU supports it
            bpy.context.preferences.addons["cycles"].preferences.get_devices()

            for device in bpy.context.preferences.addons["cycles"].preferences.devices:
                device.use = device.type in ("CUDA", "OPTIX")

            # Set the rendering device to GPU
            bpy.context.scene.cycles.device = "GPU"

        # Match target image aspect ratio with the longer side normalized to 512 px.
        _set_render_resolution_from_target(target_image_path)

        if bpy.context.scene.render.engine == "CYCLES":
            bpy.context.scene.cycles.samples = 512

        # Set color mode to RGB
        bpy.context.scene.render.image_settings.color_mode = "RGB"

        # render from all the camera, save the rendering to the rendering_dir
        for camera in bpy.data.objects:
            if camera.type == "CAMERA":
                bpy.context.scene.camera = camera
                bpy.context.scene.render.image_settings.file_format = "PNG"
                bpy.context.scene.render.filepath = os.path.join(
                    rendering_dir, f"{camera.name}.png"
                )
                bpy.ops.render.render(write_still=True)

    # Save the blend file (also reached by "__norender__" — the edit must persist)
    if save_blend:
        # Set the save version to 0
        bpy.context.preferences.filepaths.save_version = 0
        # Save the blend file
        bpy.ops.wm.save_as_mainfile(filepath=save_blend)
