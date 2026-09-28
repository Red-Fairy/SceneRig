"""Stamp PhysX-ready physics onto a GRASE scene USD (run with the Isaac venv python).

    $SCENERIG_ISAAC_PYTHON isaac/isaac_add_physics.py \
        <scene_visual.usdc> <scene.usd> [collision_dir]

Conventions (mirrors the isaac settle backend in lib/tools/geometry/physics.py):
  - ``object_identity.json`` next to <scene.usd> is required. Each mapped prim path is one
    independent dynamic rigid-body root. Colliders are the object's CoACD convex parts from
    <collision_dir>/<usd_name>.npz (world-frame, written by isaac/build_collision.py),
    authored as invisible child meshes with exact convexHull approximation — the SAME
    collision geometry the pipeline settle used, so settled poses are at rest here by
    construction. If collision_dir is omitted, each mapped root instead gets runtime
    convexDecomposition + bbox mass.
  - meshes outside every mapped root -> static collider (triangle mesh), friction 0.8
World is Z-up meters (asserted). No PhysicsScene prim is baked in; Isaac Lab or
the verify script provides one.

Per-object overrides: if a physics_overrides.json sits next to <scene.usd>, its
entries are applied on top of the defaults, keyed by the manifest's ``usd_name``:

    {"obj_plush_toy_0": {"com_world": [x, y, z],   // authored COM (world coords, m)
                         "mass": 0.15,             // kg, replaces density-derived mass
                         "angular_damping": 2.0, "linear_damping": 0.0,
                         "friction": 0.9,          // dedicated physics material
                         "flatten_base_mm": 8,     // flat collider base facet
                         "kinematic": true}}       // hold pose exactly (last resort)

isaac/isaac_auto_stabilize.py generates this file automatically for unstable objects.

VLM-estimated physics: if a physics_vlm.json sits next to <scene.usd> (written by
preprocessing and translated by ``translate_vlm_physics.py``; keyed by the same
object names), its per-object mass_kg and
friction are used below overrides but above the defaults — per-key precedence:
manual/auto-stabilize overrides > VLM > heuristic (250 kg/m^3 density, friction 0.6).
material / mass_range_kg / mass_source are stamped as custom attrs (grase:material,
grase:massRange, grase:massSource) on the body Xform for downstream randomization.

Use case: marginally stable objects (e.g. a plush toy whose real stability comes
from a soft squishy base) get a low authored COM — the rigid-body "weeble" stand-in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

try:
    from .export_identity import load_identity_manifest
    from .physics_config import (
        CONTACT_OFFSET,
        DENSITY,
        OBJ_FRICTION,
        REST_OFFSET,
        SURF_FRICTION,
    )
    from .physics_values import pick_mass_friction
except ImportError:  # direct ``python isaac/isaac_add_physics.py`` execution
    from export_identity import load_identity_manifest
    from physics_config import (
        CONTACT_OFFSET,
        DENSITY,
        OBJ_FRICTION,
        REST_OFFSET,
        SURF_FRICTION,
    )
    from physics_values import pick_mass_friction

MIN_MASS = 0.02  # kg (legacy bbox-mass fallback only)
# DENSITY / OBJ_FRICTION / SURF_FRICTION / CONTACT_OFFSET / REST_OFFSET live in
# physics_config so the settle stage and this export cannot drift apart.


def bbox_volume(prim, cache):
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        return 0.0
    s = r.GetSize()
    return s[0] * s[1] * s[2]


def signed_volume(points, counts, indices):
    """Volume enclosed by a closed mesh; negative when it is wound inside-out."""
    total, offset = 0.0, 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        for k in range(1, count - 1):
            a, b, c = points[face[0]], points[face[k]], points[face[k + 1]]
            total += float(np.dot(a, np.cross(b, c))) / 6.0
    return total


def orient_faces_outward(mesh):
    ""
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    points = mesh.GetPointsAttr().Get()
    if not counts or not indices or not points:
        return False
    if signed_volume(np.array(points), counts, indices) >= 0.0:
        return False

    def reverse_per_face(values):
        out, offset = [], 0
        for count in counts:
            out.extend(list(values[offset : offset + count])[::-1])
            offset += count
        return out

    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(reverse_per_face(indices)))
    attrs = (
        [mesh.GetNormalsAttr()]
        if mesh.GetNormalsInterpolation() == "faceVarying"
        else []
    )
    attrs += [
        p.GetAttr()
        for p in UsdGeom.PrimvarsAPI(mesh).GetPrimvars()
        if p.GetInterpolation() == "faceVarying" and p.Get() is not None
    ]
    for attr in attrs:
        values = attr.Get()
        if values is not None and len(values) == len(indices):
            attr.Set(type(values)(reverse_per_face(values)))
    return True


def make_physics_material(stage, path, friction):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(friction)
    api.CreateDynamicFrictionAttr(friction)
    return mat


def _world_matrix(prim: Usd.Prim) -> np.ndarray:
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
        or np.linalg.det(matrix[:3, :3]) <= 1e-12
    ):
        raise ValueError(
            f"rigid-body frame must be finite, nonsingular and right-handed: {prim.GetPath()}"
        )
    return matrix


def _is_rigid(matrix: np.ndarray) -> bool:
    linear = matrix[:3, :3]
    return np.allclose(linear.T @ linear, np.eye(3), atol=1e-8, rtol=0)


def _set_matrix(
    xform: UsdGeom.Xformable, matrix: np.ndarray, *, reset: bool = False
) -> None:
    xform.MakeMatrixXform().Set(Gf.Matrix4d(matrix.T.tolist()))
    xform.SetResetXformStack(reset)


def normalize_rigid_body_frame(body: Usd.Prim) -> bool:
    """Move root scale/shear into visual geometry before adding physics.

    USD COM and principal axes are body-local, while our mass/inertia inputs are
    already in physical world units. A unit-scale, proper rigid frame avoids
    anisotropic quaternion conversion and a second scale of explicit inertia.
    First transformable descendants retain their exact world transforms; a Mesh
    root instead bakes the residual affine transform into its points/normals.
    No prim is renamed, reparented, or assigned a second rigid body.

    Static exports with rigid ancestors are supported. Reflection, singularity,
    animated affected transforms, and already-authored colliders fail closed.
    https://openusd.org/release/api/class_usd_physics_mass_a_p_i.html
    """
    root = UsdGeom.Xformable(body)
    if not root or not (body.IsA(UsdGeom.Xform) or body.IsA(UsdGeom.Mesh)):
        raise ValueError(f"unsupported rigid-body root: {body.GetPath()}")
    world = _world_matrix(body)
    parent_world = np.eye(4)
    if not root.GetResetXformStack() and not body.GetParent().IsPseudoRoot():
        parent_world = _world_matrix(body.GetParent())
        if not _is_rigid(parent_world):
            raise ValueError(
                f"rigid-body ancestors must have unit scale and no shear: {body.GetPath()}"
            )
    if _is_rigid(world):
        return False
    if root.TransformMightBeTimeVarying():
        raise ValueError(f"animated rigid-body frame is unsupported: {body.GetPath()}")
    if any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(body)):
        raise ValueError("normalize the rigid-body frame before authoring colliders")
    left, _, right = np.linalg.svd(world[:3, :3])
    rigid = world.copy()
    rigid[:3, :3] = left @ right
    descendants = []

    def visit(prim: Usd.Prim) -> None:
        child = UsdGeom.Xformable(prim)
        if child:
            if child.TransformMightBeTimeVarying() or child.GetResetXformStack():
                raise ValueError(
                    f"animated/reset visual transform is unsupported: {prim.GetPath()}"
                )
            descendants.append((child, _world_matrix(prim)))
        else:
            for nested in prim.GetChildren():
                visit(nested)

    for child in body.GetChildren():
        visit(child)
    mesh_points, normal_values = None, []
    if body.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(body)
        points_attr = mesh.GetPointsAttr()
        points = np.asarray(points_attr.Get(), dtype=float)
        if (
            points_attr.GetNumTimeSamples()
            or points.ndim != 2
            or points.shape[1:] != (3,)
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"invalid/animated Mesh-root points: {body.GetPath()}")
        residual = np.linalg.inv(rigid) @ world
        mesh_points = points @ residual[:3, :3].T + residual[:3, 3]
        normals = [mesh.GetNormalsAttr()]
        normals.extend(
            primvar.GetAttr()
            for primvar in UsdGeom.PrimvarsAPI(body).GetPrimvars()
            if primvar.GetTypeName().role == Sdf.ValueRoleNames.Normal
        )
        for attr in normals:
            values = attr.Get()
            if values is None:
                continue
            array_type = type(values)
            values = np.asarray(values, dtype=float)
            if (
                attr.GetNumTimeSamples()
                or values.ndim != 2
                or values.shape[1:] != (3,)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"invalid/animated Mesh-root normals: {attr.GetPath()}"
                )
            transformed = values @ np.linalg.inv(residual[:3, :3])
            lengths = np.linalg.norm(transformed, axis=1)
            if np.any(lengths <= 1e-12):
                raise ValueError(f"zero Mesh-root normal: {attr.GetPath()}")
            normal_values.append((attr, array_type, transformed / lengths[:, None]))

    _set_matrix(
        root, np.linalg.inv(parent_world) @ rigid, reset=root.GetResetXformStack()
    )
    for child, old_world in descendants:
        _set_matrix(child, np.linalg.inv(rigid) @ old_world)
    if mesh_points is not None:
        mesh.GetPointsAttr().Set(Vt.Vec3fArray(mesh_points.tolist()))
        mesh.GetExtentAttr().Set(
            Vt.Vec3fArray(
                [
                    mesh_points.min(axis=0).tolist(),
                    mesh_points.max(axis=0).tolist(),
                ]
            )
        )
        for attr, array_type, values in normal_values:
            attr.Set(array_type(values.tolist()))
    body.CreateAttribute("grase:rigidFrameNormalized", Sdf.ValueTypeNames.Bool).Set(
        True
    )
    body.CreateAttribute("grase:sourceWorldTransform", Sdf.ValueTypeNames.Matrix4d).Set(
        Gf.Matrix4d(world.T.tolist())
    )
    return True


def _rigid_inverse(body: Usd.Prim) -> np.ndarray:
    world = _world_matrix(body)
    if not _is_rigid(world):
        raise ValueError(
            f"physics authoring requires a unit-scale rigid frame: {body.GetPath()}"
        )
    return np.linalg.inv(world)


def apply_body_overrides(body, ov):
    normalize_rigid_body_frame(body)
    if ov.get("kinematic"):
        # pose is deliberately non-physical (e.g. settle-capped topplers kept
        # photo-faithful): body holds its pose exactly, still collidable/pushable-from
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body)
    if "mass" in ov:
        mass_api.CreateMassAttr(float(ov["mass"]))
    if "com_world" in ov:
        inv = _rigid_inverse(body)
        p = np.asarray(ov["com_world"], dtype=float)
        local = p @ inv[:3, :3].T + inv[:3, 3]
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*local))
        if "diagonal_inertia" in ov:
            mass_api.CreateDiagonalInertiaAttr(
                Gf.Vec3f(*[float(v) for v in ov["diagonal_inertia"]])
            )
            if "principal_axes" in ov:
                q = np.asarray(ov["principal_axes"], dtype=float)
                if (
                    q.shape != (4,)
                    or not np.isfinite(q).all()
                    or not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0)
                ):
                    raise ValueError(
                        "principal_axes must be a finite unit quaternion [x,y,z,w]"
                    )
                q = q / np.linalg.norm(q)
                world_axes = np.asarray(
                    Gf.Matrix3d(Gf.Rotation(Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3]))))
                ).T
                local_axes = inv[:3, :3] @ world_axes
                local_q = (
                    Gf.Matrix3d(local_axes.T.tolist())
                    .ExtractRotation()
                    .GetQuat()
                    .GetNormalized()
                )
                mass_api.CreatePrincipalAxesAttr(
                    Gf.Quatf(
                        float(local_q.GetReal()), Gf.Vec3f(*local_q.GetImaginary())
                    )
                )
    if "angular_damping" in ov or "linear_damping" in ov:
        body.AddAppliedSchema("PhysxRigidBodyAPI")
        for key, attr in [
            ("angular_damping", "physxRigidBody:angularDamping"),
            ("linear_damping", "physxRigidBody:linearDamping"),
        ]:
            if key in ov:
                body.CreateAttribute(attr, Sdf.ValueTypeNames.Float).Set(float(ov[key]))


def author_coacd_parts(stage, body, npz, mat, flatten_mm=0.0):
    """Author world-frame CoACD parts as invisible convex colliders under ``body``.

    flatten_mm > 0 clamps every part vertex within that distance of the object's
    lowest point down onto it, giving the convex hulls a flat base facet spanning
    the real contact footprint (counters SAM3D's hallucinated rounded undersides,
    which rest on a point and rock). zmin itself is unchanged, so the pose holds.
    """
    d = np.load(npz)
    parts = [d[f"v{i}"] for i in range(int(d["n"]))]
    zmin = min(v[:, 2].min() for v in parts)
    inv = _rigid_inverse(body)  # column-vector convention: p_local = inv @ p_world
    for i in range(int(d["n"])):
        v = parts[i]
        f = d[f"f{i}"]
        if flatten_mm > 0:
            v = v.copy()
            v[v[:, 2] < zmin + flatten_mm / 1000.0, 2] = zmin
        local = v @ inv[:3, :3].T + inv[:3, 3]
        mesh = UsdGeom.Mesh.Define(
            stage, body.GetPath().AppendChild(f"collision_{i:03d}")
        )
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in local])
        mesh.CreateFaceVertexCountsAttr([3] * len(f))
        mesh.CreateFaceVertexIndicesAttr([int(x) for x in f.reshape(-1)])
        mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision-only, never rendered
        prim = mesh.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "convexHull"  # exact: parts are already convex
        )
        # PhysxCollisionAPI attrs authored raw: this script runs on usd-core (no kit),
        # which ships UsdPhysics but not the PhysxSchema module.
        prim.AddAppliedSchema("PhysxCollisionAPI")
        prim.CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        ).Set(CONTACT_OFFSET)
        prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
            REST_OFFSET
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose="physics")
    return int(d["n"])


def main(src, dst, collision_dir=None):
    stage = Usd.Stage.Open(str(src))
    assert UsdGeom.GetStageUpAxis(stage) == "Z", "expected Z-up stage"
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0, "expected meters"

    root = stage.GetDefaultPrim() or stage.GetPseudoRoot().GetChildren()[0]
    stage.SetDefaultPrim(root)

    ov_path = Path(dst).parent / "physics_overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else {}
    vlm_path = Path(dst).parent / "physics_vlm.json"
    vlm_data = json.loads(vlm_path.read_text()) if vlm_path.exists() else {}
    identity_path = Path(dst).parent / "object_identity.json"
    manifest = load_identity_manifest(identity_path)

    records = manifest["objects"]
    usd_names = {record["usd_name"] for record in records}
    for label, data in (
        ("physics_overrides.json", overrides),
        ("physics_vlm.json", vlm_data),
    ):
        unknown = sorted(set(data) - usd_names)
        if unknown:
            raise RuntimeError(
                f"{label} contains names absent from object_identity.json: {unknown}"
            )
    body_paths = [str(record["prim_path"]).rstrip("/") for record in records]
    original_meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]

    def under_dynamic_root(prim):
        path = str(prim.GetPath())
        return any(path == root_path or path.startswith(root_path + "/")
                   for root_path in body_paths)  # fmt: skip

    obj_mat = make_physics_material(
        stage, root.GetPath().AppendChild("PhysMatObj"), OBJ_FRICTION
    )
    surf_mat = make_physics_material(
        stage, root.GetPath().AppendChild("PhysMatSurf"), SURF_FRICTION
    )

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    n_dyn = n_stat = n_parts = n_legacy = n_vlm = n_normalized = 0
    for record in records:
        name = record["usd_name"]
        body = stage.GetPrimAtPath(record["prim_path"])
        if not body.IsValid():
            raise RuntimeError(
                f"identity manifest object root is absent from the stage: "
                f"{record['prim_path']}"
            )
        visual_meshes = [prim for prim in Usd.PrimRange(body) if prim.IsA(UsdGeom.Mesh)]
        if not visual_meshes:
            raise RuntimeError(
                f"identity manifest object root has no visual meshes: {record['prim_path']}"
            )

        n_normalized += int(normalize_rigid_body_frame(body))
        UsdPhysics.RigidBodyAPI.Apply(body)
        ov = overrides.get(name, {})
        npz = Path(collision_dir) / f"{name}.npz" if collision_dir else None
        if collision_dir and not npz.exists():
            raise RuntimeError(
                f"missing collision file for manifest object {name!r}: {npz}"
            )
        # All semantic files use the recorded exported USD name. Pipeline-name lookup
        # happened once, when object_identity.json was built.
        vlm = vlm_data.get(name) or {}
        mass, friction = pick_mass_friction(ov, vlm)
        mat = obj_mat
        if friction is not None:
            mat = make_physics_material(
                stage, root.GetPath().AppendChild(f"PhysMat_{name}"), friction
            )
        if npz:
            n_parts += author_coacd_parts(
                stage, body, npz, mat, float(ov.get("flatten_base_mm", 0.0))
            )
            mass_api = UsdPhysics.MassAPI.Apply(body)
            if mass is not None:  # VLM or override mass; PhysX derives CoM/inertia
                mass_api.CreateMassAttr(float(mass))
            else:
                mass_api.CreateDensityAttr(DENSITY)
        else:  # runtime decomposition of every visual mesh under the explicit body root
            for prim in visual_meshes:
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                    "convexDecomposition"
                )
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, materialPurpose="physics"
                )
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr(
                float(mass)
                if mass is not None
                else max(MIN_MASS, DENSITY * bbox_volume(body, cache))
            )
            n_legacy += 1
        if vlm:
            body.CreateAttribute("grase:material", Sdf.ValueTypeNames.String).Set(
                vlm["material"]
            )
            body.CreateAttribute("grase:massRange", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(*[float(x) for x in vlm["mass_range_kg"]])
            )
            body.CreateAttribute("grase:massSource", Sdf.ValueTypeNames.String).Set(
                vlm["mass_source"]
            )
            n_vlm += 1
        if ov:
            ov = dict(ov)
            ref = ov.pop("solver_mass_kg", None)
            # the override inertia was solved at density 250; when a VLM or
            # override mass is authored instead, rescale it (inertia is
            # linear in mass) — mass from one source + inertia from another
            # is the same self-inconsistent triple the inertia companion
            # exists to prevent
            if ref and mass is not None and ov.get("diagonal_inertia"):
                ov["diagonal_inertia"] = [
                    float(x) * float(mass) / float(ref) for x in ov["diagonal_inertia"]
                ]
            apply_body_overrides(body, ov)
        n_dyn += 1

    n_flipped = 0
    for prim in original_meshes:
        if under_dynamic_root(prim):
            continue
        if orient_faces_outward(UsdGeom.Mesh(prim)):
            n_flipped += 1
            print(f"  flipped inside-out static collider: {prim.GetPath()}")
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "none"  # static triangle mesh
        )
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            surf_mat, materialPurpose="physics"
        )
        n_stat += 1

    # SAM3D GLBs carry no metallicFactor, so glTF defaults it to 1.0 (mirror) and the
    # objects render black; their textures are baked albedo, so force dielectric.
    n_demetal = 0
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if shader and shader.GetIdAttr().Get() == "UsdPreviewSurface":
            m = shader.GetInput("metallic")
            if m and m.Get() == 1.0:
                m.Set(0.0)
                r = shader.GetInput("roughness")
                if r and r.Get() == 1.0:
                    r.Set(0.6)
                n_demetal += 1

    stage.Export(str(dst))
    print(
        f"ISAAC_PHYSICS_OK {dst} dynamic={n_dyn} static={n_stat} "
        f"coacd_parts={n_parts} legacy={n_legacy} demetal={n_demetal} "
        f"overrides={len(overrides)} vlm={n_vlm} flipped_statics={n_flipped} "
        f"normalized_frames={n_normalized}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
