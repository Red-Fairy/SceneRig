"""Post-initializer cleanup for the exact main-support root mesh.

The initializer builds a ``table`` main-support as a thin top slab JOINED with a
solid base/cabinet beneath it, and the prompt asks the base to overlap UP INTO the
slab's underside (prompt_builder ``form == "table"``). Agents routinely land the
base's TOP face at exactly z=0 — coplanar with, and overlapping, the slab's top face.
Two coincident coplanar same-facing faces render fine in EEVEE (the initializer's
fast-iteration engine) but SELF-SHADOW in Cycles (every later stage's engine): shadow
rays leaving the visible top immediately hit the coincident duplicate, so the whole
tabletop turns pitch black. The texture agent then cannot fix it (geometry is out of
its scope) and the run is ruined.

This pass deletes the redundant buried duplicate only when a convex retained face
fully contains it. Convexity makes the vertex-containment proof sound; concave Boolean
topology is deliberately left untouched. The caller supplies the exact main-support
build name, so unrelated roots, detail children, and imported assets are never candidates.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _plane_basis(normal: tuple[float, float, float]) -> tuple[tuple, tuple]:
    """Two orthonormal in-plane axes for a face normal (for 2D projection)."""
    nx, ny, nz = normal
    # pick the world axis least aligned with the normal to seed the basis
    seed = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
    ux = seed[1] * nz - seed[2] * ny
    uy = seed[2] * nx - seed[0] * nz
    uz = seed[0] * ny - seed[1] * nx
    ul = math.sqrt(ux * ux + uy * uy + uz * uz) or 1.0
    u = (ux / ul, uy / ul, uz / ul)
    v = (ny * u[2] - nz * u[1], nz * u[0] - nx * u[2], nx * u[1] - ny * u[0])
    return u, v


def _project(verts: list[tuple], u: tuple, v: tuple) -> list[tuple[float, float]]:
    return [(vx * u[0] + vy * u[1] + vz * u[2], vx * v[0] + vy * v[1] + vz * v[2])
            for (vx, vy, vz) in verts]  # fmt: skip


def _poly_area(poly: list[tuple[float, float]]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return abs(a) * 0.5


def _is_convex_poly(poly: list[tuple[float, float]], *, eps: float = 1e-9) -> bool:
    """Return whether a simple 2-D polygon is non-degenerate and convex.

    Collinear boundary vertices are permitted. A concave or degenerate keeper is
    rejected because vertex containment alone does not prove polygon containment for
    those shapes; cleanup false negatives are safer than deleting visible geometry.
    """
    if len(poly) < 3:
        return False
    winding = 0
    for i in range(len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        cross = (x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)
        if abs(cross) <= eps:
            continue
        turn = 1 if cross > 0 else -1
        if winding and turn != winding:
            return False
        winding = turn
    return winding != 0


def _point_in_poly(pt: tuple[float, float], poly: list[tuple[float, float]],
                   eps: float = 1e-6) -> bool:  # fmt: skip
    """Ray-cast point-in-polygon, inclusive of the boundary."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        # on-edge (boundary) counts as inside
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if (
            abs(cross) <= eps
            and min(x0, x1) - eps <= x <= max(x0, x1) + eps
            and min(y0, y1) - eps <= y <= max(y0, y1) + eps
        ):
            return True
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xint:
                inside = not inside
    return inside


def faces_to_delete(
    faces: list[dict], *, normal_tol: float = 1e-3, plane_tol: float = 1e-4
) -> list[int]:
    """Indices of redundant coplanar faces to remove (pure geometry — no bpy).

    ``faces``: ``[{"index": int, "normal": (x,y,z), "verts": [(x,y,z), ...]}, ...]``
    in a single consistent (world) frame. A face is deleted iff it is coplanar and
    same-facing with a LARGER CONVEX face in its group and every one of its vertices
    lies inside that larger face. Convexity is required because all-vertex containment
    is not sufficient for concave polygons (notably Boolean-cut window walls).
    """
    groups: dict[tuple, list[dict]] = {}
    for f in faces:
        n = f["normal"]
        nl = math.sqrt(sum(c * c for c in n)) or 1.0
        un = (n[0] / nl, n[1] / nl, n[2] / nl)
        d = sum(un[i] * f["verts"][0][i] for i in range(3))  # plane offset
        key = (round(un[0] / normal_tol), round(un[1] / normal_tol),
               round(un[2] / normal_tol), round(d / plane_tol))  # fmt: skip
        groups.setdefault(key, []).append({**f, "_un": un})

    drop: list[int] = []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        u, v = _plane_basis(members[0]["_un"])
        for m in members:
            m["_poly"] = _project(m["verts"], u, v)
            m["_area"] = _poly_area(m["_poly"])
            m["_convex"] = _is_convex_poly(m["_poly"])
        # largest first: a face is buried if fully inside an already-KEPT larger face
        members.sort(key=lambda m: m["_area"], reverse=True)
        kept: list[dict] = []
        for m in members:
            buried = any(
                k["_convex"] and all(_point_in_poly(p, k["_poly"]) for p in m["_poly"])
                for k in kept
            )
            if buried:
                drop.append(m["index"])
            else:
                kept.append(m)
    return drop


# Blender-side pass. Runs under the initializer's blender binary; imports the pure
# ``faces_to_delete`` above (cwd is REPO_ROOT) so the geometry logic is single-sourced.
_CLEANUP_SCRIPT = r"""
import os, sys, json
sys.path.insert(0, os.getcwd())
import bpy, bmesh
from lib.tools.geometry.root_surface_cleanup import faces_to_delete

blend = sys.argv[-2]
eligible_names = set(json.loads(sys.argv[-1]))
bpy.ops.wm.open_mainfile(filepath=blend)

report = {}
for o in list(bpy.data.objects):
    if o.type != 'MESH' or o.name not in eligible_names or o.parent is not None:
        continue
    me = o.data
    mw = o.matrix_world
    M = mw.to_3x3()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    faces = []
    for f in bm.faces:
        wn = (M @ f.normal)
        wn = wn.normalized()
        wverts = [tuple(mw @ vtx.co) for vtx in f.verts]
        faces.append({"index": f.index,
                      "normal": (wn.x, wn.y, wn.z),
                      "verts": wverts})
    drop_idx = set(faces_to_delete(faces))
    if drop_idx:
        victims = [f for f in bm.faces if f.index in drop_idx]
        bmesh.ops.delete(bm, geom=victims, context='FACES')
        bm.to_mesh(me)
        me.update()
        report[o.name] = {"before": len(faces), "deleted": len(victims)}
    bm.free()

print("CLEANUP_JSON" + json.dumps(report))
if report:
    bpy.ops.wm.save_mainfile(filepath=blend)
"""


def dedup_coplanar_faces(
    blend: str, blender_cmd: str, *, eligible_names: set[str]
) -> dict:
    """Delete buried duplicates from explicitly eligible parentless mesh roots.

    The blend is saved in place only when a face is deleted. Best-effort: returns
    ``{}`` on failure so this optional repair never aborts the pipeline.
    """
    names = sorted(
        name.strip()
        for name in eligible_names
        if isinstance(name, str) and name.strip()
    )
    if not (blend and os.path.exists(blend) and names):
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_CLEANUP_SCRIPT)
        script = f.name
    try:
        r = subprocess.run(
            [
                blender_cmd,
                "--background",
                "--python",
                script,
                "--",
                blend,
                json.dumps(names),
            ],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        )  # fmt: skip
        line = next(
            (ln for ln in r.stdout.splitlines() if ln.startswith("CLEANUP_JSON")),
            None,
        )
        return json.loads(line[len("CLEANUP_JSON") :]) if line else {}
    except Exception:  # noqa: BLE001
        return {}
