"""The ONE contact ladder: numerical tolerance, contact classes and their allowances.

Every penetration consumer (the Blender rules gate, ``nudge_object`` acceptance, the
composition hull gate) reads its numbers from here — never a second table elsewhere,
the same way provider branching lives only in ``common.provider_of``.

PhysX cooks each part as a 64-vertex convex hull that is slightly fatter than the visual mesh, so resolved contact
shows ~1 mm of visual overlap per body; a jammed same-support cluster (eggs in a bowl)
is delivered with 3-5 mm of residual solver penetration that composition's hull gate
forgives by contact class.
"""

from __future__ import annotations

from typing import Optional, Sequence

CONTACT_TOLERANCE_M = 0.00025
PHYSICAL_REPAIR_MAX_M = 0.20
PHYSICAL_REPAIR_MAX_TRANSACTIONS = 16

# Visual-mesh allowances (the Blender gate), by contact class.
OBJECT_OBJECT_TOL_M = 0.001  # unrelated objects: ~cooking-inflation floor per body
LATERAL_SIBLING_TOL_M = 0.006  # side-by-side on the same support (leaning eggs, bottles)
RESTING_ON_SUPPORT_TOL_M = 0.005
# was below the PhysX rest depth — v5accept rule checks flagged 26/35 resting pairs at 3.4-4.9 mm, a
# depth every re-settle reproduces, so fruit_bottle_bluebin exhausted its initializer budget on 3.6 mm)
OBJECT_SURFACE_TOL_M = 0.001  # an object crossing a FOREIGN surface (wall, other slab)
MAIN_SUPPORT_SURFACE_TOL_M = 0.02  # main support crossing an unrelated surface
# Isaac hull allowances (composition gate); hulls are fatter than meshes, so larger.
HULL_SUBTOL_M = 0.008
HULL_SIBLING_TOL_M = 0.020
# |z| of the unit separating direction below which a contact is lateral (side-by-side).
LATERAL_MAX_DZ = 0.5

RESTING_ON_SUPPORT = "resting_on_support"
LATERAL_SIBLING = "lateral_sibling"
OBJECT_OBJECT = "object_object"
OBJECT_SURFACE = "object_surface"
SURFACE_SURFACE = "surface_surface"

_TOL = {
    RESTING_ON_SUPPORT: RESTING_ON_SUPPORT_TOL_M,
    LATERAL_SIBLING: LATERAL_SIBLING_TOL_M,
    OBJECT_OBJECT: OBJECT_OBJECT_TOL_M,
    OBJECT_SURFACE: OBJECT_SURFACE_TOL_M,
    SURFACE_SURFACE: MAIN_SUPPORT_SURFACE_TOL_M,
}


def _is_obj(name: str) -> bool:
    return str(name).startswith("obj_")


def contact_class(
    a: str,
    b: str,
    support: Optional[dict] = None,
    direction: Optional[Sequence[float]] = None,
) -> str:
    support = support or {}
    oa, ob = _is_obj(a), _is_obj(b)
    if oa and ob:
        if support.get(a) == b or support.get(b) == a:
            return RESTING_ON_SUPPORT
        sa, sb = support.get(a), support.get(b)
        if sa is not None and sa == sb:
            if direction is None or abs(float(direction[2])) < LATERAL_MAX_DZ:
                return LATERAL_SIBLING
            return OBJECT_OBJECT  # same support but stacked: one sits ON the other
        return OBJECT_OBJECT
    if oa or ob:
        obj, surf = (a, b) if oa else (b, a)
        return RESTING_ON_SUPPORT if support.get(obj) == surf else OBJECT_SURFACE
    return SURFACE_SURFACE


def tolerance_m(cls: str) -> float:
    return _TOL[cls]


def excess_m(depth: float, cls: str) -> float:
    """Depth beyond the class allowance (0 when within it)."""
    return max(0.0, float(depth) - tolerance_m(cls))


def penetration_improved(
    before: float, after: float, tol: float = CONTACT_TOLERANCE_M
) -> bool:
    """Accept complete clearance (to ``tol``) or a measurable reduction of a real defect."""
    if before <= tol:
        return False
    return after <= tol or after <= before - min(0.001, before * 0.25)
