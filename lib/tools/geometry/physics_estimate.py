""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from lib.tools.geometry.generative_resegment import MARK_RED

REPO_ROOT = Path(__file__).resolve().parents[3]
BLENDER = REPO_ROOT / "lib/utils/third_party/blender-4.5/blender"
MATERIALS = yaml.safe_load((REPO_ROOT / "isaac/materials.yaml").read_text())
PROMPT_PATH = REPO_ROOT / "isaac/prompts/physics_estimation.yaml"
DENSITY_BAND = 3.0  # accept mass/volume within [band/3, band*3] of the material density
N_SIDE_VIEWS = 4
MIN_VOLUME = 1e-7  # m^3, guard against degenerate meshes
MASS_PERSIST_DECIMALS = 6
MASS_FLOOR_KG = 1e-6


def persistable_mass(mass: float, lo: float, hi: float) -> tuple[float, float, float]:
    """Round a validated (mass, lo, hi) for persistence, preserving the contract.

    Rounding happens AFTER the reply was validated, so it must keep every value
    positive and finite and keep ``lo <= mass <= hi`` by construction: floor at 1 mg,
    round to 1 mg, then widen the range around the rounded mass.
    """
    mass = max(round(float(mass), MASS_PERSIST_DECIMALS), MASS_FLOOR_KG)
    lo = max(round(float(lo), MASS_PERSIST_DECIMALS), MASS_FLOOR_KG)
    hi = max(round(float(hi), MASS_PERSIST_DECIMALS), MASS_FLOOR_KG)
    return mass, min(lo, mass), max(hi, mass)
VLM_JOBS = int(os.environ.get("GRASE_VLM_PHYSICS_BATCH", "5"))
# extents-product vs mesh-volume rescale-ratio disagreement that triggers a warning
# (OBB solver noise and small decomposition drift stay well under this)
CROSSCHECK_TOL = 0.25
FALLBACK_DENSITY_KGM3 = 250.0
FALLBACK_FRICTION = 0.6
PHYSICS_ESTIMATE_MANIFEST_VERSION = 1


# --------------------------------------------------------------------------- #
# Geometry: one-time anchors + the stateless rescale rule                      #
# --------------------------------------------------------------------------- #
def mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Sealed-mesh volume via the signed-tetrahedra integral |sum a.(b x c)/6|
    (same math as the CoM solver's ``uniform_com_z``), floored at MIN_VOLUME."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    return max(abs(float(vol)), MIN_VOLUME)


def obb_extents(verts: np.ndarray) -> list[float]:
    """Sorted-descending minimal-OBB extents (m) of a vertex cloud —
    rotation-invariant, so the same object measures the same at any pose."""
    from trimesh.bounds import oriented_bounds

    _, ext = oriented_bounds(np.asarray(verts, dtype=np.float64))
    return sorted((float(x) for x in ext), reverse=True)


def glb_anchors(glb: str | Path) -> tuple[float, list[float]]:
    """(mesh volume, sorted OBB extents) of a placed GLB. Both are invariant
    under the glTF-import pre-rotation, so no frame fixup is needed."""
    import trimesh

    mesh = trimesh.load(str(glb), force="mesh")
    return (
        mesh_volume(mesh.vertices, mesh.faces),
        obb_extents(np.asarray(mesh.vertices)),
    )


def rescaled_mass(
    entry: dict,
    current_obb_extents,
    current_volume: float | None = None,
    label: str = "",
) -> tuple[float, list[float]]:
    """(mass_kg, mass_range_kg) at the object's CURRENT size: the estimate times
    the extents-product ratio vs the recorded anchor (s^3 under uniform scale).
    ``current_volume`` (same signed-tetra convention) is an optional cross-check
    only — a big disagreement means the geometry changed non-similarly."""
    est_ext = entry.get("est_obb_extents_m")
    mass, rng = float(entry["mass_kg"]), [float(x) for x in entry["mass_range_kg"]]
    if not est_ext:
        return mass, rng  # legacy entry with no anchor: use as-is
    ratio = float(np.prod(np.asarray(current_obb_extents, dtype=float))) / max(
        float(np.prod(np.asarray(est_ext, dtype=float))), 1e-12
    )
    if current_volume is not None and entry.get("est_volume_m3"):
        vratio = float(current_volume) / max(float(entry["est_volume_m3"]), 1e-12)
        if abs(vratio / max(ratio, 1e-12) - 1.0) > CROSSCHECK_TOL:
            print(
                f"[physics-estimate] WARNING {label or '?'}: extents ratio "
                f"{ratio:.3f} vs volume ratio {vratio:.3f} disagree — geometry "
                f"changed non-similarly since estimation",
                file=sys.stderr,
            )
    return mass * ratio, [x * ratio for x in rng]


def density_gate(mass: float, volume: float, material: str) -> tuple[str, float]:
    """('ok'|'high'|'low', gated_mass). 'high' clamps to band*3 x volume; 'low' is
    flag-only — hollow objects (mugs, electronics) legitimately undershoot."""
    band = MATERIALS.get(material, {}).get("density", 1000.0)
    rho = mass / volume
    if rho > band * DENSITY_BAND:
        return "high", band * DENSITY_BAND * volume
    if rho < band / DENSITY_BAND:
        return "low", mass
    return "ok", mass


def add_badge(png: Path, label: str) -> None:
    from PIL import Image, ImageDraw

    img = Image.open(png).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w = 26 + 13 * (len(label) - 1)
    draw.rectangle([4, 4, 4 + w, 30], fill="white")
    draw.text((11, 8), label, fill="black")
    img.save(png)


MIN_PHOTO_EDGE = 384
# Marking colour for the photo crop, as (rgb, NAME) — the NAME is what the prompt
# below calls it, so colour and wording come from ONE source (see the MARK_* note
# in generative_resegment.py). MARK_RED matches check_vital's crops.
PHOTO_MARK = MARK_RED


def photo_crop(image_png: Path, mask_npy: Path, crop_path: Path) -> Path | None:
    """Context crop of the capture photo with the object tinted + outlined, badge PHOTO.

    Marking is delegated to ``generative_resegment._marked_crop`` — the same
    tint+ring the resegment guards use — so the two stages cannot drift, and the
    colour NAME travels with the colour (see MARK_* there: a hardcoded prompt
    word once said GREEN while the crop drew RED for ~219 production calls).

    Window/scale exist because of the badge: the old tight 25%-pad crop came out
    ~64 px on a small object, so the fixed-size PHOTO badge covered the very
    object it labelled and the VLM saw no surroundings to judge scale by.
    ``pad_frac`` gives ~2x the object box, floored so a small object still gets
    real context, and the result is upscaled to MIN_PHOTO_EDGE.
    """
    from PIL import Image

    from lib.tools.geometry.generative_resegment import _marked_crop

    if not Path(mask_npy).exists():
        return None
    mask = np.load(mask_npy) > 0  # masks are uint8 0/255 on disk
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    img = np.asarray(Image.open(image_png).convert("RGB"))
    # 0.5 -> window ~2x the object box; the floor keeps >=32 px of surroundings
    # on the short axis so a thumbnail-sized object is not cropped to itself.
    short = max(1, min(int(xs.max() - xs.min()), int(ys.max() - ys.min())))
    _marked_crop(
        img, mask, str(crop_path), pad_frac=max(0.5, 32.0 / short), mark=PHOTO_MARK
    )

    crop = Image.open(crop_path)
    scale = MIN_PHOTO_EDGE / min(crop.width, crop.height)
    if scale > 1.0:
        crop.resize(
            (round(crop.width * scale), round(crop.height * scale)), Image.LANCZOS
        ).save(crop_path)
    add_badge(crop_path, "PHOTO")
    return crop_path


def render_views(tasks: list[dict], out: Path) -> None:
    """One headless-Blender pass: per task import the glb and render 6 transparent
    512px views (0=top, 1=bottom, 2+=sides at 20 deg elevation)."""
    script = out / "_render.py"
    script.write_text(f"""
import bpy, math, mathutils
scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene.render.engine = 'CYCLES'
scene.cycles.samples = 32
scene.render.film_transparent = True
scene.render.resolution_x = scene.render.resolution_y = 512
world = bpy.data.worlds.new('w'); scene.world = world
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.7
sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(35), 0, math.radians(30))
scene.collection.objects.link(sun)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam'))
scene.collection.objects.link(cam); scene.camera = cam
for task in {tasks!r}:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=task['glb'])
    meshes = [o for o in set(bpy.data.objects) - before if o.type == 'MESH']
    pts = [o.matrix_world @ mathutils.Vector(c) for o in meshes for c in o.bound_box]
    lo = mathutils.Vector([min(p[i] for p in pts) for i in range(3)])
    hi = mathutils.Vector([max(p[i] for p in pts) for i in range(3)])
    center, radius = (lo + hi) / 2, max(1e-3, (hi - lo).length / 2)
    d = 2.6 * radius
    el = math.radians(20)
    views = [(0, 0, d), (0, 0, -d)] + [
        (d * math.cos(el) * math.cos(a), d * math.cos(el) * math.sin(a), d * math.sin(el))
        for a in [math.radians(90 * k + 45) for k in range({N_SIDE_VIEWS})]]
    for i, off in enumerate(views):
        cam.location = center + mathutils.Vector(off)
        look = center - cam.location
        cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
        scene.render.filepath = task['prefix'] + f'_{{i}}.png'
        bpy.ops.render.render(write_still=True)
    for o in set(bpy.data.objects) - before:
        bpy.data.objects.remove(o, do_unlink=True)
print('RENDER_OK')
""")
    r = subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python", str(script)],
                       capture_output=True, text=True)
    assert "RENDER_OK" in r.stdout, r.stdout[-2000:] + r.stderr[-1000:]


def call_vlm(
    client,
    model: str,
    system_text: str,
    user_text: str,
    image_paths: list[Path],
    *,
    log_stream=None,
) -> dict:
    ""
    from lib.tools.geometry.agentic_mask import parse_json
    from lib.utils.common import get_image_base64, get_model_response

    content = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": get_image_base64(str(p))}}
        for p in image_paths
    ]
    err = ""
    for attempt in range(3):
        retry_note = (
            [{"type": "text", "text": (
                f"Your previous reply was invalid ({err}). Return ONLY the JSON "
                "object with the required keys — no prose, no code fences."
            )}]
            if err
            else []
        )
        response = get_model_response(client, {
            "model": model,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": content + retry_note},
            ],
            "max_tokens": 2000,
        }, effort=os.environ.get("GRASE_VLM_PHYSICS_EFFORT", "medium"))
        raw = response.choices[0].message.content
        try:
            out = parse_json(raw)
            if not isinstance(out, dict):
                raise ValueError(f"expected a JSON object, got {type(out).__name__}")
            if not isinstance(out.get("material"), str) or out["material"] not in MATERIALS:
                raise ValueError(f"unknown material: {out.get('material')}")
            raw_mass = out.get("mass_kg")
            if isinstance(raw_mass, bool) or not isinstance(raw_mass, (int, float)):
                raise ValueError("mass_kg must be a number")
            mass = float(raw_mass)
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError(f"non-positive mass: {out['mass_kg']}")
            raw_range = out["mass_range_kg"]
            if (
                not isinstance(raw_range, list)
                or len(raw_range) != 2
                or any(
                    isinstance(x, bool) or not isinstance(x, (int, float))
                    for x in raw_range
                )
            ):
                raise ValueError("mass_range_kg must be a two-number JSON list")
            lo, hi = (float(x) for x in raw_range)
            if not all(np.isfinite(x) and x > 0 for x in (lo, hi)) or lo > hi:
                raise ValueError("mass_range_kg must be positive, finite, and ordered")
            if not lo <= mass <= hi:
                raise ValueError("mass_kg must fall within mass_range_kg")
            if out.get("mass_source") not in {"estimated", "image_text"}:
                raise ValueError("mass_source must be estimated or image_text")
            out["mass_kg"] = mass
            out["mass_range_kg"] = [lo, hi]
            return out
        except (ValueError, KeyError, TypeError) as e:
            err = str(e)[:300]
            print(
                f"[physics call_vlm] invalid reply (attempt {attempt + 1}/3): {err}",
                file=log_stream,
            )
    raise ValueError(f"physics VLM returned invalid JSON 3x — last: {err}")


def estimate_object(
    client, model: str, prompt: str, label: str,
    renders: list[Path], crop: Path | None,
    dims, volume: float,
    *, log_stream=None,
) -> dict:
    """One object's estimate: VLM call + density gate with one re-ask. ``dims`` is
    the metric size for the prompt text; ``volume`` the sealed mesh volume (m^3)."""
    n_views = 2 + N_SIDE_VIEWS
    text = (
        f"### Object:\n"
        f"- name hint: {label}\n"
        f"- metric dimensions (m): {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f}\n"
        f"- sealed mesh volume: {volume:.6f} m^3\n"
        f"- images: {n_views} mesh renders (indexed 0-{n_views - 1})"
        + (
            f", then 1 real photo crop (PHOTO) showing the object IN CONTEXT, "
            f"tinted and outlined in {PHOTO_MARK[1]} — judge only the "
            f"{PHOTO_MARK[1]}-marked object; the surroundings are there for "
            f"material, identity, fill-state, and lighting cues, never scale "
            f"or geometry cues"
            if crop
            else ""
        )
    )
    images = renders + ([crop] if crop else [])
    call_kwargs = {"log_stream": log_stream} if log_stream is not None else {}
    pred = call_vlm(client, model, prompt, text, images, **call_kwargs)
    mass = float(pred["mass_kg"])
    gate, gated_mass = density_gate(mass, volume, pred["material"])
    if gate == "high":  # one re-ask only for physically excessive density
        rho, band = mass / volume, MATERIALS[pred["material"]]["density"]
        retry = (
            f"{text}\n\n### Sanity check failed on your previous answer:\n"
            f"You said {pred['material']}, {mass:.3f} kg -> implied density "
            f"{rho:.0f} kg/m^3, but {pred['material']} is ~{band:.0f} kg/m^3. "
            f"Reconsider (is it hollow? wrong material? wrong mass?) and return "
            f"the JSON again."
        )
        pred = call_vlm(client, model, prompt, retry, images, **call_kwargs)
        mass = float(pred["mass_kg"])
        gate, gated_mass = density_gate(mass, volume, pred["material"])
        gate = {"ok": "ok_after_reask", "high": "clamped_high", "low": "flagged_low"}[gate]
        mass = gated_mass
    elif gate == "low":
        # A low mass/contact-material density is normal for hollow vessels,
        # electronics, foam-filled objects, and mixed-material assemblies.
        gate = "flagged_low"
    # widen the range to include the (possibly gate-clamped) final mass, so a
    # persisted entry never carries mass_kg outside its own mass_range_kg
    lo, hi = (float(x) for x in pred["mass_range_kg"])
    mass, lo, hi = persistable_mass(mass, lo, hi)
    return {
        "material": pred["material"],
        "mass_kg": mass,
        "mass_range_kg": [lo, hi],
        "mass_source": pred.get("mass_source", "estimated"),
        "friction": MATERIALS[pred["material"]]["friction"],
        "density_kgm3": round(mass / volume, 1),
        "density_gate": gate,
        "reasoning": pred.get("reasoning", ""),
        "_note": f"physics_estimate {date.today()}, model={model}, "
                 f"{n_views} renders" + (" + photo crop" if crop else ""),
    }


# --------------------------------------------------------------------------- #
# Scene-level driver + the canonical file                                      #
# --------------------------------------------------------------------------- #
def load_estimates(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:  # noqa: BLE001 - estimates are best-effort everywhere
        return {}


def _estimate_manifest_path(out_json: Path) -> Path:
    return out_json.with_name("physics_estimate_manifest.json")


def _estimate_record(
    status: str,
    model: str,
    *,
    entry: dict | None = None,
    render_evidence_count: int = 0,
    photo_evidence_count: int = 0,
    failure: str | None = None,
) -> dict:
    record = {
        "status": status,
        "model": model,
        "render_evidence_count": render_evidence_count,
        "photo_evidence_count": photo_evidence_count,
        "evidence_count": render_evidence_count + photo_evidence_count,
    }
    if entry is not None:
        record.update(
            {
                "mass_kg": entry.get("mass_kg"),
                "mass_range_kg": entry.get("mass_range_kg"),
                "mass_source": entry.get("mass_source"),
                "material": entry.get("material"),
                "friction": entry.get("friction"),
                "density_kgm3": entry.get("density_kgm3"),
                "density_gate": entry.get("density_gate"),
            }
        )
    else:
        record["fallback"] = {
            "density_kgm3": FALLBACK_DENSITY_KGM3,
            "friction": FALLBACK_FRICTION,
        }
    if failure:
        record["failure"] = str(failure)[:1000]
    return record


def _write_estimate_manifest(
    objects: list[dict],
    out_json: Path,
    model: str,
    *,
    results: dict,
    cached_names: set[str] | None = None,
    estimated_evidence: dict[str, dict[str, int]] | None = None,
    failures: dict[str, str] | None = None,
    disabled: bool = False,
) -> dict:
    """Write complete per-object estimate/default provenance beside physics_vlm."""
    cached_names = cached_names or set()
    estimated_evidence = estimated_evidence or {}
    failures = failures or {}
    names = list(dict.fromkeys(o["name"] for o in objects if o.get("name")))
    records: dict[str, dict] = {}
    for name in names:
        if disabled:
            records[name] = _estimate_record("disabled_fallback", model)
        elif name in estimated_evidence:
            records[name] = _estimate_record(
                "estimated",
                model,
                entry=results.get(name),
                render_evidence_count=estimated_evidence[name]["renders"],
                photo_evidence_count=estimated_evidence[name]["photos"],
            )
        elif name in cached_names and name in results:
            note = str((results.get(name) or {}).get("_note", ""))
            records[name] = _estimate_record(
                "cached",
                model,
                entry=results[name],
                render_evidence_count=2 + N_SIDE_VIEWS,
                photo_evidence_count=int("photo crop" in note),
            )
        else:
            records[name] = _estimate_record(
                "failed_fallback",
                model,
                failure=failures.get(name, "estimate unavailable"),
            )
    estimated = sum(
        record["status"] in {"estimated", "cached"} for record in records.values()
    )
    fallback = len(records) - estimated
    manifest = {
        "schema_version": PHYSICS_ESTIMATE_MANIFEST_VERSION,
        "model": model,
        "expected_count": len(records),
        "estimated_count": estimated,
        "fallback_count": fallback,
        "coverage_status": "full" if fallback == 0 else "partial",
        "physics_vlm": str(out_json),
        "objects": records,
    }
    path = _estimate_manifest_path(out_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_disabled_manifest(
    objects: list[dict], out_json: Path, model: str = "disabled"
) -> dict:
    """Record explicit default use when the VLM stage is disabled by CLI policy."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text("{}\n")
    manifest = _write_estimate_manifest(
        objects, out_json, model, results={}, disabled=True
    )
    print(
        "PHYSICS_ESTIMATE_PARTIAL "
        f"expected={manifest['expected_count']} estimated=0 "
        f"fallback={manifest['fallback_count']} disabled=true"
    )
    return manifest


def write_failed_manifest(
    objects: list[dict], out_json: Path, model: str, failure: str
) -> dict:
    """Record heuristic fallback for every object after a stage-level failure."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text("{}\n")
    names = [o["name"] for o in objects if o.get("name")]
    manifest = _write_estimate_manifest(
        objects,
        out_json,
        model,
        results={},
        failures={name: failure for name in names},
    )
    print(
        "PHYSICS_ESTIMATE_PARTIAL "
        f"expected={manifest['expected_count']} estimated=0 "
        f"fallback={manifest['fallback_count']}"
    )
    return manifest


def estimate_scene(
    objects: list[dict],
    out_json: Path,
    model: str = "claude-opus-5",
    force: bool = False,
    jobs: int = VLM_JOBS,
    expected_names: list[str] | None = None,
    log_stream=None,
) -> dict:
    """Estimate every object and write the canonical physics_vlm.json.

    ``objects``: per object ``{"name": obj_<slug>, "render_glb": canonical pristine
    (or placed fallback), "placed_glb": metric placed mesh (size anchors),
    "mask_npy": ..., "image_png": ...}``. Per-object cache: existing entries are
    kept unless ``force``. One batched Blender render pass, then ``jobs``-wide
    concurrent VLM calls. A failed object is logged and skipped — never raises."""
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results = {} if force else load_estimates(out_json)
    all_expected_names = list(
        dict.fromkeys(
            [name for name in (expected_names or []) if name]
            + [o["name"] for o in objects if o.get("name")]
        )
    )
    manifest_objects = list(objects) + [
        {"name": name}
        for name in all_expected_names
        if name not in {o.get("name") for o in objects}
    ]
    cached_names = set(all_expected_names).intersection(results)
    out_json.write_text(json.dumps(results, indent=1) + "\n")
    todo = [o for o in objects if o["name"] not in results]
    if not todo:
        manifest = _write_estimate_manifest(
            manifest_objects,
            out_json,
            model,
            results=results,
            cached_names=cached_names,
        )
        status = (
            "PHYSICS_ESTIMATE_OK"
            if manifest["fallback_count"] == 0
            else "PHYSICS_ESTIMATE_PARTIAL"
        )
        print(
            f"{status} expected={manifest['expected_count']} "
            f"estimated={manifest['estimated_count']} "
            f"fallback={manifest['fallback_count']} "
            + ("(all cached)" if not manifest["fallback_count"] else "(cached only)"),
            file=log_stream,
        )
        return results

    render_dir = out_json.parent / "vlm_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    failures: dict[str, str] = {}
    estimated_evidence: dict[str, dict[str, int]] = {}
    try:
        render_views(
            [{"glb": str(o["render_glb"]), "prefix": str(render_dir / o["name"])}
             for o in todo],
            render_dir,
        )  # fmt: skip
        sys.path.insert(0, str(REPO_ROOT))
        from lib.utils.common import build_client

        client = build_client(model)
        prompt = yaml.safe_load(PROMPT_PATH.read_text())["prompt"]
    except Exception as exc:  # noqa: BLE001 - whole-batch setup still defaults safely
        failures.update({o["name"]: str(exc) for o in todo})
        manifest = _write_estimate_manifest(
            manifest_objects,
            out_json,
            model,
            results=results,
            cached_names=cached_names,
            failures=failures,
        )
        print(
            f"PHYSICS_ESTIMATE_PARTIAL expected={manifest['expected_count']} "
            f"estimated={manifest['estimated_count']} "
            f"fallback={manifest['fallback_count']}",
            file=log_stream,
        )
        return results
    n_views = 2 + N_SIDE_VIEWS
    lock = threading.Lock()

    def _one(o: dict) -> None:
        name = o["name"]
        try:
            volume, ext = glb_anchors(o["placed_glb"])
            renders = [render_dir / f"{name}_{i}.png" for i in range(n_views)]
            for i, p in enumerate(renders):
                add_badge(p, str(i))
            crop = photo_crop(
                Path(o["image_png"]), Path(o["mask_npy"]),
                render_dir / f"{name}_photo.png",
            )  # fmt: skip
            label = re.sub(r"_\d+$", "", name.removeprefix("obj_")).replace("_", " ")
            estimate_kwargs = (
                {"log_stream": log_stream} if log_stream is not None else {}
            )
            entry = estimate_object(
                client, model, prompt, label, renders, crop, ext, volume,
                **estimate_kwargs,
            )
            entry["est_volume_m3"] = round(volume, 7)
            entry["est_obb_extents_m"] = [round(x, 5) for x in ext]
            with lock:
                results[name] = entry
                estimated_evidence[name] = {
                    "renders": len(renders),
                    "photos": int(crop is not None),
                }
                out_json.write_text(json.dumps(results, indent=1))
            print(f"  {name}: {entry['material']} {entry['mass_kg']:.3f} kg "
                  f"[{entry['mass_range_kg']}] gate={entry['density_gate']}",
                  file=log_stream, flush=True)  # fmt: skip
        except Exception as e:  # noqa: BLE001 - one object must not sink the scene
            with lock:
                failures[name] = str(e)
            print(f"[physics-estimate] {name} failed ({e}); skipped", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        list(ex.map(_one, todo))
    manifest = _write_estimate_manifest(
        manifest_objects,
        out_json,
        model,
        results=results,
        cached_names=cached_names,
        estimated_evidence=estimated_evidence,
        failures=failures,
    )
    if manifest["fallback_count"]:
        print(
            f"PHYSICS_ESTIMATE_PARTIAL expected={manifest['expected_count']} "
            f"estimated={manifest['estimated_count']} "
            f"fallback={manifest['fallback_count']}",
            file=log_stream,
        )
    else:
        print(
            f"PHYSICS_ESTIMATE_OK expected={manifest['expected_count']} "
            f"estimated={manifest['estimated_count']}",
            file=log_stream,
        )
    return results


def objects_from_table(table: list[dict], scene_dir: Path) -> list[dict]:
    """estimate_scene object list from a placement table (in-memory or loaded).
    Render source prefers an explicit catalog/canonical render mesh, then the SAM3D
    pristine mesh (upright, meaningful top/side views), and finally the placed mesh."""
    out = []
    for o in table:
        name, glb = o.get("mesh_name"), o.get("mesh_glb")
        if not (name and glb and Path(glb).exists()):
            continue
        slug = name.removeprefix("obj_")
        explicit_render = o.get("render_glb") or o.get("asset_source_glb")
        if explicit_render and not Path(explicit_render).exists():
            explicit_render = None
        pristine = sorted((scene_dir / "meshes").glob(f"{slug}_pristine*.glb"))
        out.append({
            "name": name,
            "render_glb": explicit_render or (str(pristine[0]) if pristine else glb),
            "placed_glb": glb,
            "mask_npy": o.get("mask_path") or str(scene_dir / f"masks/{slug}.npy"),
            "image_png": str(scene_dir / "input.png"),
        })
    return out


def scene_objects_from_placement(exp: Path) -> list[dict]:
    """Object list from an experiment's saved preprocess artifacts."""
    scene = exp / "scene" if (exp / "scene").exists() else exp
    table = json.loads((scene / "placement.json").read_text())["objects"]
    return objects_from_table(table, scene)


def translate_to_isaac(exp: Path) -> int:
    """Preprocess physics_vlm.json (pipeline names, estimation-time anchors) ->
    scene/isaac/physics_vlm.json (USD prim names, mass rescaled to the export
    geometry via the dump's world meshes). ``object_identity.json`` is the sole
    pipeline-name -> USD-name bridge. Returns objects written; 0 = nothing to
    translate (missing source/dump) — callers fall back to defaults."""
    from isaac.export_identity import load_identity_manifest

    scene = exp / "scene" if (exp / "scene").exists() else exp
    src = load_estimates(scene / "physics/physics_vlm.json")
    dump_path = scene / "isaac/visual_meshes.npz"
    if not src or not dump_path.exists():
        return 0
    manifest = load_identity_manifest(scene / "isaac/object_identity.json")
    dump = np.load(dump_path)
    dump_names = {str(n) for n in dump["names"]}
    manifest_pipeline_names = {
        record["pipeline_name"] for record in manifest["objects"]
    }
    unknown_source_names = sorted(set(src) - manifest_pipeline_names)
    if unknown_source_names:
        raise ValueError(
            "preprocess physics estimates contain names absent from "
            f"object_identity.json: {unknown_source_names}"
        )
    out = {}
    for record in manifest["objects"]:
        pipeline_name = record["pipeline_name"]
        prim = record["usd_name"]
        if prim not in dump_names:
            raise ValueError(
                f"identity manifest maps {pipeline_name!r} to {prim!r}, "
                "which is absent from visual_meshes.npz"
            )
        entry = src.get(pipeline_name)
        if entry is None:
            continue
        v, f = dump[f"v_{prim}"], dump[f"f_{prim}"]
        mass, rng = rescaled_mass(
            entry, obb_extents(v), mesh_volume(v, f), label=prim
        )
        mass, lo, hi = persistable_mass(mass, rng[0], rng[1])
        out[prim] = {
            **{k: entry[k] for k in
               ("material", "mass_source", "friction", "density_gate")},
            "mass_kg": mass,
            "mass_range_kg": [lo, hi],
            "_note": f"translated from preprocess estimate "
                     f"({entry.get('_note', '')}); mass x"
                     f"{mass / max(float(entry['mass_kg']), 1e-9):.3f} extents rescale",
        }  # fmt: skip
    (scene / "isaac").mkdir(exist_ok=True)
    (scene / "isaac/physics_vlm.json").write_text(json.dumps(out, indent=1))
    return len(out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Standalone preprocess-side VLM physics "
                                             "estimation for an existing exp dir")
    ap.add_argument("exp_dir", type=Path)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=VLM_JOBS)
    args = ap.parse_args()
    exp = args.exp_dir if args.exp_dir.is_absolute() else REPO_ROOT / args.exp_dir
    scene_dir = exp / "scene" if (exp / "scene").exists() else exp
    estimate_scene(
        scene_objects_from_placement(exp),
        scene_dir / "physics/physics_vlm.json",
        model=args.model, force=args.force, jobs=args.jobs,
    )
