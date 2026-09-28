"""Physically-consistent rigid-body mass properties for the composition/export CoM
stabilization override.

Overriding only ``centerOfMass`` on a PhysX rigid body (as the settle stage's
stabilization does) leaves the auto-computed inertia tensor describing the object's
NATURAL (uniform-density, geometric-centroid) mass distribution — a self-inconsistent
(mass, CoM, inertia) triple. Under a plain vertical drop the inconsistency is
invisible (little torque is needed); under real contact torque it can produce
runaway rotational dynamics.

This module computes a mathematically valid (mass, CoM, diagonal-inertia,
principal-axes) tuple for an ARBITRARY target CoM: the object's natural inertia
(uniform density, standard signed-tetrahedra-from-origin mass integral over each
part's OWN convex hull — matching what PhysX's convexHull approximation will
actually simulate, not the raw input faces) is computed once, then shifted to the
target CoM via the parallel-axis theorem and re-diagonalized. The result describes
a physically valid virtual mass distribution (not necessarily the real object's
true distribution — just self-consistent, which is what the solver needs).

Pure numpy/scipy, no Isaac/USD dependency — fully unit-testable standalone.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

# Reference-tetrahedron (0, e1, e2, e3, volume=1/6) barycentric moment constants:
# int s^2 dV = int t^2 = int u^2 = 1/60; int s*t = int s*u = int t*u = 1/120.
_REF_D = np.array([[1 / 60, 1 / 120, 1 / 120],
                    [1 / 120, 1 / 60, 1 / 120],
                    [1 / 120, 1 / 120, 1 / 60]])  # fmt: skip


def _oriented_hull_triangles(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convex hull of ``v`` (own points, NOT the input face connectivity — matches
    PhysX's convexHull approximation, which rebuilds a fresh hull from points only)
    as (hull_vertices, triangles) with every triangle re-oriented so its
    vertex-order normal points OUTWARD (matched against the hull's own facet
    equations) — required for the signed-tetrahedra mass integral below."""
    hull = ConvexHull(v)
    pts = hull.points
    tris = hull.simplices.copy()
    for i, tri in enumerate(tris):
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        normal = np.cross(b - a, c - a)
        if np.dot(normal, hull.equations[i, :3]) < 0:
            tris[i, [1, 2]] = tris[i, [2, 1]]
    return pts, tris


def _part_moments(v: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """One convex part's (volume, first_moment, second_moment) at density=1, all
    relative to the GIVEN coordinate origin (not the part's own centroid) — signed
    tetrahedra from the origin to each outward-oriented hull triangle. The signed
    sum is correct regardless of where the origin sits relative to the part (the
    standard divergence-theorem mass-integral technique), so parts can be combined
    by simple summation about one common reference point (world origin here)."""
    pts, tris = _oriented_hull_triangles(v)
    if len(pts) < 4:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    # signed volume of each tetra (origin, a, b, c): det([a,b,c]) / 6
    det = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = det / 6.0
    volume = float(vol.sum())
    first = (vol[:, None] * (a + b + c) / 4.0).sum(axis=0)
    # second moment: sum_tet det(M) * (M @ D @ M^T), M = [a|b|c] as columns
    M = np.stack([a, b, c], axis=-1)  # (n, 3, 3)
    MD = np.einsum("nij,jk->nik", M, _REF_D)
    S = np.einsum("nij,nkj->nik", MD, M)  # (M@D) @ M^T per tet
    second = (det[:, None, None] * S).sum(axis=0)
    return volume, first, second


def assembly_natural_properties(
    parts: list[tuple[np.ndarray, np.ndarray]], density: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine multiple convex parts (each ``(v, f)``; ``f`` unused, see
    :func:`_oriented_hull_triangles`) into whole-assembly (mass, natural_centroid,
    inertia_about_natural_centroid) at uniform ``density``. Parts are summed about a
    common reference point (the world origin these vertices are already expressed
    in) before shifting once to the assembly's own centroid — correct for
    disjoint parts at arbitrary relative positions."""
    volume = 0.0
    first = np.zeros(3)
    second = np.zeros((3, 3))
    for v, _f in parts:
        pv, pf, ps = _part_moments(np.asarray(v, dtype=float))
        volume += pv
        first += pf
        second += ps
    if volume <= 0.0:
        raise ValueError("assembly has non-positive volume — degenerate geometry")
    centroid = first / volume
    mass = density * volume
    # second-moment -> inertia about origin, then parallel-axis to the centroid
    inertia_origin = np.trace(second) * np.eye(3) - second
    inertia_origin *= density
    r = -centroid  # vector from natural centroid to the origin (P=origin)
    inertia_centroid = inertia_origin - mass * (
        np.dot(r, r) * np.eye(3) - np.outer(r, r)
    )
    return mass, centroid, inertia_centroid


def shift_inertia(
    inertia_about_p: np.ndarray, mass: float, p: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Parallel-axis shift: given the inertia tensor of a mass ``mass`` about the
    point ``p`` (must be the body's TRUE center of mass — the theorem only relates
    inertia-about-COM to inertia-about-any-other-point, not two arbitrary points),
    return the inertia tensor about a different point ``q``. The result together
    with (mass, q) is a mathematically valid, self-consistent rigid-body triple —
    the mass distribution it describes generally differs from the real object's
    (unless q also happens to be the real COM), but it is internally consistent,
    which is what the physics solver requires."""
    r = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
    return inertia_about_p + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))


def principal_axes(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a symmetric inertia tensor: (diagonal_inertia [3], quaternion
    [x,y,z,w]) — the rotation taking body/world axes to the principal frame. Flips
    one eigenvector's sign if needed so the eigenbasis is a proper rotation
    (det=+1), never a reflection (``Rotation.from_matrix`` requires this)."""
    eigvals, eigvecs = np.linalg.eigh(inertia)
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 0] *= -1.0
    quat = Rotation.from_matrix(eigvecs).as_quat()  # scipy: [x, y, z, w]
    return eigvals, quat


def stabilized_mass_properties(
    parts: list[tuple[np.ndarray, np.ndarray]], density: float, target_com
) -> dict:
    """The composition/export override's full recipe: natural mass properties from
    the (flattened, if applicable) collision parts, shifted to ``target_com`` via
    the parallel-axis theorem, diagonalized. Returns
    ``{"mass": float, "diagonal_inertia": [3 floats], "principal_axes": [x,y,z,w]}``
    — pair with the caller's own ``com_world`` (= ``target_com``) when authoring."""
    mass, natural_centroid, inertia_natural = assembly_natural_properties(
        parts, density
    )
    inertia_target = shift_inertia(
        inertia_natural, mass, natural_centroid, np.asarray(target_com, dtype=float)
    )
    diag, quat = principal_axes(inertia_target)
    return {
        "mass": float(mass),
        "diagonal_inertia": [float(x) for x in diag],
        "principal_axes": [float(x) for x in quat],
    }
