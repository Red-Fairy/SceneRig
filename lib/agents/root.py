"""Root agent orchestration for staged scene generation."""

import hashlib
import json
import logging
import os
import time
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from lib.agents.generator import GeneratorAgent
from lib.agents.generators import (
    CompositionAgent,
    InitializerPlannerAgent,
    LightingAgent,
    TextureAgent,
)
from lib.agents.harness_profile import (
    DEFAULT_HARNESS_PROFILE,  # noqa: F401 - backward-compatible re-export
    GPT6_SCENE_ARTIFACTS,
    HARNESS_PROFILE_NAMES,  # noqa: F401 - backward-compatible re-export
    base_manifest_profile_matches,
    nonterminal_runtime_recovery_markers,
    resolve_harness_profile,
)
from lib.agents.verifier import VerifierAgent
from lib.agents.verifiers import (
    InitializerPlannerVerifierAgent,
    LightingVerifierAgent,
    TextureVerifierAgent,
)
from lib.utils.provenance import blender_provenance

logger = logging.getLogger(__name__)

BLACK_RENDER_LUMINANCE = 2.0


def _blend_fingerprint(blend_path: Optional[str]) -> Optional[tuple]:
    """(size, mtime_ns) of a blend, or None when it cannot be stat'd."""
    if not blend_path or not os.path.exists(blend_path):
        return None
    st = os.stat(blend_path)
    return (st.st_size, st.st_mtime_ns)


def _mean_luminance(render_path: str) -> Optional[float]:
    ""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(render_path) as img:
            return float(np.asarray(img.convert("L"), dtype="float32").mean())
    except Exception as exc:
        logger.warning("Could not measure render luminance (%s): %s", render_path, exc)
        return None


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_recovery_stage(marker_paths: list[Path]) -> Optional[str]:
    """Resolve one authenticated recovery target without guessing across stages."""

    stages: set[str] = set()
    for marker_path in marker_paths:
        try:
            marker = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"runtime recovery marker became unreadable: {marker_path}"
            ) from exc
        stage = marker.get("stage", "initializer")
        if stage not in {"initializer", "composition"}:
            raise RuntimeError(
                f"runtime recovery marker has unsupported stage {stage!r}: "
                f"{marker_path}"
            )
        stages.add(stage)
    if len(stages) > 1:
        raise RuntimeError(
            "nonterminal GPT-6 runtime recovery markers span multiple stages: "
            + ", ".join(sorted(stages))
        )
    return next(iter(stages), None)


def _collect_gpt6_scene_artifact_outputs(scene_dir: Path) -> dict[str, dict]:
    """Record the paths of the GPT-6 scene artifacts that exist (no byte binding)."""

    if nonterminal_runtime_recovery_markers(scene_dir):
        raise RuntimeError("cannot finalize nonterminal GPT-6 runtime recovery")
    return {
        key: {"path": str(scene_dir / relative)}
        for key, relative in GPT6_SCENE_ARTIFACTS.items()
        if (scene_dir / relative).is_file()
    }


def _validate_render_integrity(
    render_path: Optional[str], target_image_path: Optional[str]
) -> dict[str, Any]:
    """Check that the delivered render is readable, visible, and non-uniform."""
    if not render_path or not os.path.isfile(render_path):
        return {
            "status": "render_missing",
            "errors": [f"final render file is missing: {render_path}"],
        }
    try:
        with Image.open(render_path) as image:
            image.load()
            size = image.size
            luminance_extrema = image.convert("L").getextrema()
            alpha_bbox = (
                image.getchannel("A").getbbox() if "A" in image.getbands() else True
            )
        target_size = None
        if target_image_path and os.path.isfile(target_image_path):
            with Image.open(target_image_path) as target:
                target_size = target.size
    except Exception as exc:  # noqa: BLE001 - reported as artifact-integrity failure
        return {
            "status": "render_unreadable",
            "errors": [f"final render cannot be decoded: {exc}"],
        }
    luminance = _mean_luminance(render_path)
    errors: list[str] = []
    status = "valid"
    if target_size is not None and size != target_size:
        status = "render_size_mismatch"
        errors.append(f"final render size {size} does not match target {target_size}")
    elif not alpha_bbox:
        status = "render_transparent"
        errors.append("final render has no visible-alpha pixels")
    elif luminance is None:
        status = "render_unreadable"
        errors.append("final render luminance could not be measured")
    elif luminance < BLACK_RENDER_LUMINANCE:
        status = "render_black"
        errors.append(
            f"final render is blank (mean luminance {luminance:.2f} < "
            f"{BLACK_RENDER_LUMINANCE})"
        )
    elif luminance_extrema[1] - luminance_extrema[0] < 1:
        status = "render_flat"
        errors.append(
            "final render is a uniform frame with no measurable scene variation"
        )
    return {
        "status": status,
        "errors": errors,
        "size": list(size),
        "target_size": list(target_size) if target_size else None,
        "mean_luminance": luminance,
        "luminance_extrema": list(luminance_extrema),
    }


@dataclass(frozen=True)  # noqa: MLR-04.01 — preserve this existing research constructor (positional callers).
class StageSpec:
    """One generator/verifier pair in the root scene pipeline.

    ``generator_max_rounds`` / ``verifier_max_rounds`` optionally cap that stage's
    per-attempt tool-call rounds. Per-stage CLI flags may override them; verifier
    ``None`` values fall back to the global ``--verifier-max-rounds`` flag.

    ``completion_policy`` separates how a stage is accepted from how its generator
    stopped. Appearance stages are verifier-completed; initializer/composition also
    require the current deterministic gate."""

    name: str
    generator_cls: type[GeneratorAgent]
    # None for single_pass stages, which never construct a verifier.
    verifier_cls: Optional[type[VerifierAgent]]
    generator_max_rounds: Optional[int] = None
    verifier_max_rounds: Optional[int] = None
    completion_policy: str = "verifier"
    # single_pass: ONE generator session, no verifier gate, no retry. Composition
    # uses it — its verifier only ever gated a 2nd FRESH-memory attempt that
    # re-investigated blind (lost what the 1st moved); a single longer session with
    # shared memory + the generator's own check_rules_enforced gate is strictly better.
    # The generator's in-loop end refusal (_refuse_end_reason) is the ONLY gate on a
    # single_pass stage; `approved` just reports whether that gate passed.
    single_pass: bool = False


class RootSceneAgent:
    """Top-level coordinator for the multi-agent scene pipeline.

    Today each stage agent is a thin copy of the original generator/verifier
    behavior with its own class, prompt key, and memory files. This root agent
    makes the intended control flow explicit so future work can specialize
    stages without changing the main entrypoint again.
    """

    # Generator round budgets: the initializer builds the whole scene structure and gets
    # the deep budget; the appearance stages edit an existing scene and need far less.
    # A per-stage CLI flag (--init-generator-max-rounds etc.) still wins over these.
    INITIALIZATION_STAGE = StageSpec(
        "initializer",
        InitializerPlannerAgent,
        InitializerPlannerVerifierAgent,
        generator_max_rounds=20,
        completion_policy="generator_end_rules_and_verifier",
    )
    # Objects are SAM3D meshes from preprocessing.
    # The camera is fixed to the MoGE source view, so there is no camera stage either.
    LOOP_STAGES = [
        StageSpec(
            "texture", TextureAgent, TextureVerifierAgent, generator_max_rounds=10
        ),
        StageSpec(
            "lighting", LightingAgent, LightingVerifierAgent, generator_max_rounds=10
        ),
        StageSpec(
            "composition",
            CompositionAgent,
            None,  # single_pass: no verifier is ever constructed for this stage
            single_pass=True,
            completion_policy="generator_end_and_rules",
            # generator budget is DYNAMIC (min(70, 8 + 5*n_objects)): ONE continuous
            # pose-refinement session (no 2nd attempt), ~1 investigate + 1-2 moves each.
        ),
    ]

    FINAL_SETTLE_REPAIR_MAX_ROUNDS = 16
    FINAL_SETTLE_REPAIR_WHY = (
        "The scene you delivered went through the final free physics settle (every "
        "object dynamic at once) and the objects listed here moved notably from where "
        "composition left them; their SETTLED poses are what the scene now holds. This "
        "is a SHORT repair round for exactly these objects: investigate each, decide "
        "whether its settled pose is acceptable (then leave it) or re-place it with "
        "execute_and_evaluate or move (the same physics rules apply), pass "
        "check_rules_enforced, and end. For THIS round the investigation-coverage rule in "
        "your instructions is scoped to the listed objects only (the backend enforces it "
        "that way); do not revisit the others."
    )

    def __init__(self, args: dict[str, Any]) -> None:
        self.args = args
        # Canonicalize direct/programmatic construction too; main.py performs the same
        # resolution for early CLI validation, while tests and library callers may not.
        self.harness_profile = resolve_harness_profile(args.get("harness_profile"))
        self.args["harness_profile"] = self.harness_profile["name"]
        self.args["harness_profile_manifest"] = resolve_harness_profile(
            self.harness_profile["name"]
        )
        self.stage_context: dict[str, Any] = {
            "harness_profile": resolve_harness_profile(self.harness_profile["name"]),
            "stage_artifacts": {},
            "stage_attempts": {},
            "verifier_history": {},
        }

    def _authorized_gpt6_inherited_inventory(self, moge_dir: str) -> bool:
        """Whether this run is explicitly inheriting a completed GPT-6 initializer."""

        if self.harness_profile["name"] != "gpt6_v1" or "initializer" not in set(
            self.args.get("skip_stages") or []
        ):
            return False
        manifest_value = self.args.get("base_run_manifest")
        if not manifest_value:
            raise RuntimeError(
                "GPT-6 initialized-scene inheritance requires a base_run_manifest"
            )
        manifest_path = Path(manifest_value)
        try:
            payload = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read inherited base manifest: {exc}") from exc
        if not isinstance(payload, dict) or not base_manifest_profile_matches(
            payload, self.harness_profile["name"]
        ):
            raise RuntimeError(
                "inherited base manifest does not exactly match the GPT-6 harness profile"
            )
        return True

    def _validate_pre_agent_inventory(
        self, *, recovery_complete: bool = False
    ) -> dict[str, Any]:
        ""

        moge_dir = self.args.get("moge_dir")
        if not moge_dir:
            # RootSceneAgent is also used by lightweight orchestration tests and
            # non-static-scene callers. Static-scene runs always supply moge_dir.
            return {}
        from lib.tools.geometry.inventory_contract import (
            InventoryContractError,
            authoritative_inventory,
            validate_blender_object_names,
            validate_retained_mask_artifacts,
            validate_scene_artifacts,
        )

        recovery_markers = (
            nonterminal_runtime_recovery_markers(moge_dir)
            if self.harness_profile["name"] == "gpt6_v1"
            else []
        )
        if recovery_markers:
            if recovery_complete:
                raise RuntimeError(
                    "GPT-6 stage recovery returned with nonterminal markers: "
                    + ", ".join(str(path) for path in recovery_markers)
                )
            recovery_stage = _runtime_recovery_stage(recovery_markers)
            # The graph/placement/Blend may be between atomic commit steps. Validate
            # immutable source evidence now, then let the matching stage Executor
            # restore its durable snapshots before it exposes any mutation tool.
            masks_path = Path(moge_dir) / "masks" / "masks.json"
            try:
                masks = json.loads(masks_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise InventoryContractError(
                    f"source masks are unreadable before GPT-6 recovery: {exc}"
                ) from exc
            authoritative_inventory(masks)
            validate_retained_mask_artifacts(masks, Path(moge_dir).resolve())
            logger.warning(
                "Deferring pre-agent graph/placement/Blend validation to GPT-6 "
                "%s recovery for: %s",
                recovery_stage,
                ", ".join(str(path) for path in recovery_markers),
            )
            return {
                "runtime_recovery_deferred": True,
                "runtime_recovery_stage": recovery_stage,
                "runtime_recovery_markers": [str(path) for path in recovery_markers],
            }

        inherited_runtime_inventory = (
            self.harness_profile["name"] == "gpt6_v1"
            if recovery_complete
            else self._authorized_gpt6_inherited_inventory(str(moge_dir))
        )

        # Before the initializer starts, no runtime root transaction has occurred.
        # A persisted runtime_added node here is stale staged state, not an authorized
        # extension of the fresh preprocessing inventory.
        report = validate_scene_artifacts(
            moge_dir,
            allow_runtime_additions=inherited_runtime_inventory,
            allow_runtime_inventory=inherited_runtime_inventory,
            # Every GPT-6 typed commit re-validates the physics inputs; surface a bad
            # preprocess estimate before the first agent round instead.
            validate_runtime_physics=inherited_runtime_inventory
            or self.harness_profile["name"] == "gpt6_v1",
        )
        expected_by_id = report["object_mesh_names"]
        blend = self.args.get("blender_save") or self.args.get("blender_file")
        blender = self.args.get("blender_command") or "blender"
        if not blend or not Path(blend).is_file():
            raise InventoryContractError(
                f"placement -> Blender inventory contract failed:\n  - shared blend "
                f"is missing before agent launch: {blend}"
            )

        work = Path(moge_dir) / "inventory_validation"
        work.mkdir(parents=True, exist_ok=True)
        script_path = work / "probe_pre_agent_objects.py"
        report_path = work / "pre_agent_blender_objects.json"
        if report_path.exists():
            report_path.unlink()
        script_path.write_text(
            "import json\n"
            "import bpy\n"
            f"path = {str(report_path)!r}\n"
            "names = sorted(o.name for o in bpy.data.objects "
            "if o.name.startswith('obj_'))\n"
            "with open(path, 'w') as stream:\n"
            "    json.dump({'pipeline_object_names': names}, stream, indent=2)\n"
        )
        try:
            proc = subprocess.run(
                [
                    blender,
                    "--background",
                    str(blend),
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(script_path),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "AL_LIB_LOGLEVEL": "0"},
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InventoryContractError(
                f"pre-agent Blender inventory probe failed: {exc}"
            ) from exc
        if proc.returncode != 0 or not report_path.is_file():
            detail = proc.stderr[-2000:] or proc.stdout[-2000:] or "no report"
            raise InventoryContractError(
                "pre-agent Blender inventory probe failed: " + detail
            )
        try:
            blender_report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise InventoryContractError(
                f"pre-agent Blender inventory report is invalid: {exc}"
            ) from exc
        validate_blender_object_names(
            expected_by_id, blender_report.get("pipeline_object_names", [])
        )
        return {**report, **blender_report}

    def _log_stage_time(self, stage: str, seconds: float) -> None:
        """Record one stage's wall time in ``<scene>/stage_timings.json`` (the runner seeds
        it with the preprocess entry). Best-effort — timing must never abort the pipeline."""
        md = self.args.get("moge_dir")
        if not md:
            return
        path = os.path.join(md, "stage_timings.json")
        try:
            data = json.load(open(path)) if os.path.exists(path) else {}
            data[stage] = round(seconds, 2)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.warning("stage timing log failed (%s)", e)

    async def _recover_runtime_before_agent(
        self, stage_name: str, stage_idx: int
    ) -> dict[str, Any]:
        """Start only the owning Executor, then validate its restored scene contract.

        Initializer and composition agents build prompts before connecting tools, and
        those prompts read the live graph and placement. During crash recovery those
        files may be between commit steps, so the Blender Executor must consume its WAL
        first. A short-lived, single-server client performs only that initialization;
        the normal owning-stage agent is constructed after restored artifacts validate.
        """

        from lib.agents.tool_client import ExternalToolClient
        from lib.agents.tool_defaults import (
            GENERATOR_TOOLS_DEFAULT,
            split_tool_scripts,
        )

        if stage_name not in {"initializer", "composition"}:
            raise RuntimeError(
                f"GPT-6 runtime recovery does not support stage {stage_name!r}"
            )
        stage = self._stage_spec(stage_name)
        if stage is None:
            raise RuntimeError(f"GPT-6 recovery targets unknown stage {stage_name!r}")

        recovery_args = self._build_stage_args(stage_name, stage_idx)
        configured = recovery_args.get("generator_tools") or GENERATOR_TOOLS_DEFAULT
        candidates = [
            path
            for path in split_tool_scripts(str(configured))
            if Path(path).name == "exec.py" and Path(path).parent.name == "blender"
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"GPT-6 {stage_name} recovery requires exactly one Blender exec.py "
                "generator server"
            )
        stage_dir = Path(self.args.get("output_dir", "output")) / "stages" / str(
            stage_idx
        )
        recovery_dir = stage_dir / stage.generator_cls.__name__ / "runtime_recovery"
        recovery_dir.mkdir(parents=True, exist_ok=True)
        recovery_args.update(
            {
                "stage_dir": str(stage_dir),
                "agent_output_dir": str(recovery_dir.parent),
                "attempt_idx": 1,
                "attempt_output_dir": str(recovery_dir),
                "scene_root": self.args.get("output_dir", "output"),
                "output_dir": str(recovery_dir),
            }
        )
        client = ExternalToolClient(candidates[0], recovery_args)
        try:
            await client.connect_servers()
        finally:
            await client.cleanup()
        report = self._validate_pre_agent_inventory(recovery_complete=True)
        logger.info(
            "GPT-6 %s recovery restored and validated scene inventory", stage_name
        )
        return report

    async def run(self) -> dict[str, Any]:
        """Run initialization once, then a single appearance pass on the preprocessed scene.

        Objects are SAM3D meshes already physics-settled during preprocessing; the
        pipeline is init -> texture -> lighting -> composition(+pose refinement),
        followed by optional composition certification.
        Each stage's wall time lands in ``<scene>/stage_timings.json``.
        """
        import time

        logger.info("Starting root scene pipeline")
        pre_agent_inventory = self._validate_pre_agent_inventory()
        explicit_skip = set(self.args.get("skip_stages") or [])
        skip = set(explicit_skip)
        recovery_stage = pre_agent_inventory.get("runtime_recovery_stage")
        if recovery_stage:
            stage_order = [self.INITIALIZATION_STAGE.name] + [
                stage.name for stage in self.LOOP_STAGES
            ]
            if recovery_stage not in stage_order:
                raise RuntimeError(
                    f"GPT-6 recovery targets unknown stage {recovery_stage!r}"
                )
            if recovery_stage in explicit_skip:
                raise RuntimeError(
                    f"GPT-6 recovery requires the {recovery_stage} stage, but that "
                    "stage is explicitly skipped"
                )
            auto_skipped = set(stage_order[: stage_order.index(recovery_stage)])
            skip.update(auto_skipped)
            self.args["skip_stages"] = sorted(skip)
            self.stage_context["runtime_recovery"] = {
                "stage": recovery_stage,
                "markers": pre_agent_inventory.get("runtime_recovery_markers", []),
                "auto_skipped_stages": sorted(auto_skipped - explicit_skip),
            }
            logger.warning(
                "Routing GPT-6 runtime recovery directly to %s; auto-skipping "
                "completed earlier stages: %s",
                recovery_stage,
                sorted(auto_skipped - explicit_skip) or "none",
            )
            await self._recover_runtime_before_agent(
                recovery_stage, stage_order.index(recovery_stage)
            )
            self.stage_context["runtime_recovery"][
                "post_recovery_inventory_validated"
            ] = True
        every_stage = {self.INITIALIZATION_STAGE.name} | {
            s.name for s in self.LOOP_STAGES
        }
        blend_before = (
            _blend_fingerprint(
                self.args.get("blender_save") or self.args.get("blender_file")
            )
            if every_stage - skip
            else None
        )
        initializer_idx = 0
        # GPT-6 runtime recovery may resume at a later stage after validating the
        # already-materialized scene inventory.
        if "initializer" not in skip:
            # C5: tag the spend ledger with the pipeline phase. Children (per-stage MCP
            # servers) inherit the env at spawn, so their VLM calls land under the
            # right stage too.
            os.environ["GRASE_USAGE_TAG"] = "initializer"
            t0 = time.time()
            initializer_result = await self._run_initializer_loop(
                self.INITIALIZATION_STAGE, initializer_idx
            )
            self._log_stage_time("initializer", time.time() - t0)
            self._record_stage_result(
                initializer_idx, self.INITIALIZATION_STAGE.name, initializer_result
            )
            # The initializer renders in EEVEE, which hides coincident coplanar faces
            # (e.g. a table's cabinet-base top landing at z=0, flush with the slab top).
            # Those duplicates SELF-SHADOW to pitch black under Cycles — every later
            # stage's engine — so strip them before the loop stages ever render.
            t0 = time.time()
            self._cleanup_root_surfaces()
            self._log_stage_time("root_surface_cleanup", time.time() - t0)
        else:
            logger.info("Skipping initializer for runtime recovery")
        # Camera is fixed to the reference (MoGE source) view; no camera stage. Objects are
        # already physics-settled in preprocessing (hierarchical drop, BEFORE init), so there
        # is no physics stage here; pose refinement runs LAST, inside composition, so its
        # context renders show textured and lit surfaces.
        if skip:
            logger.info("Skipping stages for runtime recovery: %s", sorted(skip))
        # SEQUENTIAL per-stage indices from the stage list (initializer=0,
        # texture=1, lighting=2, composition=3), stable under runtime recovery skips.
        for i, stage in enumerate(self.LOOP_STAGES, start=1):
            if stage.name not in skip:
                os.environ["GRASE_USAGE_TAG"] = stage.name  # C5 ledger phase
                t0 = time.time()
                stage_result = await self._run_stage(stage, i)
                self._log_stage_time(stage.name, time.time() - t0)
                self._record_stage_result(i, stage.name, stage_result)

        if "composition" not in skip:
            # Phase B: the composed scene is the DELIVERED layout (composition is the
            # final stage) — certify it rests under a free sim and bake micro-drift;
            # under gpt6_v1 a NOTABLE settle change opens ONE repair round first.
            self.stage_context["composition_certification"] = (
                await self._certify_and_repair()
            )
        else:
            self.stage_context["composition_certification"] = {
                "status": "skipped",
                "warning": "composition stage was skipped",
            }

        os.environ["GRASE_USAGE_TAG"] = "export"  # C5 ledger phase
        t0 = time.time()
        export_manifest = self._export_final_result(blend_before=blend_before)
        self._log_stage_time("export", time.time() - t0)
        # Certify was the last Isaac user: retire the run's shared settle server
        # here so a STANDALONE main.py (rerun scripts) doesn't leak the warm
        # daemon. Under the static_scene.py runner this races nothing — its own
        # run-end shutdown then finds no portfile and no-ops (it remains the
        # backstop for main.py crash paths).
        md = self.args.get("moge_dir")
        if md:
            try:
                from lib.tools.geometry.physics import shutdown_shared_settle_server

                if shutdown_shared_settle_server(Path(md) / "physics"):
                    logger.info("shared Isaac settle server shut down")
            except Exception:  # noqa: BLE001 - cleanup must not fail the run
                pass
        # A run FAILS only on a broken output (missing/invalid blend or render).
        # An unapproved stage is a quality signal, not a failure: the scene still
        # exports and is worth reviewing, so it is reported but never fails the run.
        complete = export_manifest.get("artifact_status") == "valid"
        unapproved = self._unapproved_stages(skip)
        if complete:
            logger.info(
                "Root scene pipeline finished successfully (unapproved stages=%s)",
                unapproved or "none",
            )
        else:
            logger.error(
                "Root scene pipeline FAILED on output integrity (export_status=%s): %s",
                export_manifest.get("status"),
                export_manifest.get("render_error"),
            )
        return {
            "complete": complete,
            "harness_profile": resolve_harness_profile(self.harness_profile["name"]),
            "execution_status": export_manifest.get("execution_status"),
            "artifact_status": export_manifest.get("artifact_status"),
            "quality_status": export_manifest.get("quality_status"),
            "unapproved_stages": unapproved,
            "export_status": export_manifest.get("status"),
            "manifest_path": export_manifest.get("manifest_path"),
        }

    def _unapproved_stages(self, skip: set[str]) -> list[str]:
        """Executed stages whose verifier did not approve the final attempt.

        Reported for triage only — see run() for why these do not fail the run.
        """
        expected = [(0, self.INITIALIZATION_STAGE), *enumerate(self.LOOP_STAGES, 1)]
        artifacts = self.stage_context.get("stage_artifacts") or {}
        incomplete: list[str] = []
        for stage_idx, stage in expected:
            if stage.name in skip:
                continue
            result = (artifacts.get(str(stage_idx)) or {}).get(stage.name)
            if not isinstance(result, dict) or not (
                result.get("approved") is True
                and result.get("completion_status") == "complete"
            ):
                incomplete.append(stage.name)
        return incomplete

    def _cleanup_root_surfaces(self) -> None:
        """Delete buried coplanar duplicates from the exact main-support mesh only.

        This is a narrow EEVEE-to-Cycles repair for joined tabletops/bases. Missing or
        contradictory graph identity fails closed: unrelated roots and detail children
        must never become cleanup candidates.
        """
        from lib.tools.geometry.root_surface_cleanup import dedup_coplanar_faces
        from lib.tools.geometry.surface_relations import surface_build_name

        blend = self.args.get("blender_save") or self.args.get("blender_file")
        blender = self.args.get("blender_command") or "blender"
        moge_dir = self.args.get("moge_dir")
        if not (blend and os.path.exists(blend) and moge_dir):
            return
        try:
            graph_path = Path(moge_dir) / "scene_graph.json"
            graph = json.loads(graph_path.read_text())
            nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]
            main_id = graph.get("main_support_id")
            matching = [node for node in nodes if node.get("id") == main_id]
            flagged = [node for node in nodes if node.get("main_support") is True]
            if (
                not isinstance(main_id, str)
                or not main_id
                or len(matching) != 1
                or len(flagged) != 1
                or flagged[0] is not matching[0]
                or matching[0].get("kind") != "root_surface"
            ):
                logger.warning(
                    "Skipping root-surface cleanup: current main-support identity is "
                    "missing or contradictory"
                )
                return
            explicit_name = matching[0].get("build_name")
            main_name = (
                explicit_name.strip()
                if isinstance(explicit_name, str) and explicit_name.strip()
                else surface_build_name(main_id)
            )
            report = dedup_coplanar_faces(blend, blender, eligible_names={main_name})
            if report:
                logger.info("Root-surface cleanup removed coplanar faces: %s", report)
        except Exception:  # noqa: BLE001
            logger.exception("root-surface cleanup failed; scene left as initialized")

    def _certify_composed_scene(self) -> dict[str, Any]:
        """Run final free-settle and return a quality outcome.

        Certification remains non-fatal for the Blender reconstruction: a failure,
        non-convergence, or topple becomes a prominent physics warning. It does not
        claim simulation readiness; the separate Isaac conversion owns that verdict.
        """
        from lib.tools.geometry.composition_physics import certify_composed_scene

        blend = self.args.get("blender_save") or self.args.get("blender_file")
        md = self.args.get("moge_dir")
        if not (blend and md and os.path.exists(blend)):
            return {
                "status": "not_run",
                "converged": None,
                "toppled_objects": [],
                "warning": "composition certification inputs were unavailable",
            }
        try:
            runtime_inventory_profile = (
                self.harness_profile["name"] == "gpt6_v1"
                and self.harness_profile["capabilities"].get(
                    "runtime_object_inventory", False
                )
            )
            cert_kwargs = (
                {"runtime_inventory_profile": True}
                if runtime_inventory_profile
                else {}
            )
            drift = certify_composed_scene(
                md,
                blend,
                self.args.get("blender_command") or "blender",
                **cert_kwargs,
            )
            pose_path = Path(md) / "physics" / "pose_changes.json"
            pose = json.loads(pose_path.read_text()) if pose_path.exists() else {}
            converged = pose.get("composition_certify_converged")
            actual_surfaces = pose.get(
                "actual_surface_validation", {"status": "unavailable"}
            )
            coverage = pose.get("composition_certify_coverage")
            # certify_composed_scene hard-requires exact object-name coverage before it
            # writes this block (schema_version 2 carries no status field), so its
            # presence is the contract; telemetry completeness is graded below.
            if runtime_inventory_profile and not isinstance(coverage, dict):
                raise RuntimeError("GPT-6 final physics coverage block is absent")
            telemetry_complete = not runtime_inventory_profile or (
                coverage.get("delivered_pose_telemetry", {}).get("status")
                == "complete"
            )
            toppled = sorted(
                name for name, report in drift.items() if report.get("toppled")
            )
            status = (
                "passed"
                if converged is True
                and bool(drift)
                and not toppled
                and actual_surfaces.get("status") == "passed"
                and telemetry_complete
                else "warning"
            )
            result = {
                "status": status,
                "converged": converged,
                "toppled_objects": toppled,
                "objects": drift,
                "actual_surface_validation": actual_surfaces,
                "warning": (
                    None
                    if status == "passed"
                    else (
                        (
                            "final free-settle, actual-surface validation, or "
                            "delivered pose telemetry needs review; see physics details"
                        )
                        if runtime_inventory_profile
                        else "final free-settle or actual-surface validation needs review; see physics details"
                    )
                ),
            }
            if runtime_inventory_profile:
                result["coverage"] = coverage
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("composition certify failed; scene left as approved")
            return {
                "status": "error",
                "converged": None,
                "toppled_objects": [],
                "warning": str(exc),
            }

    async def _certify_and_repair(self) -> dict[str, Any]:
        """Final certify, then — under gpt6_v1 when the settle moved something NOTABLY —
        one composition repair round and a second certify. The returned block is the
        LAST certification; ``repair_round`` records the trigger, the pre-repair
        verdict, and what is still notable afterwards (nothing loops further)."""
        os.environ["GRASE_USAGE_TAG"] = "composition_certify"  # C5 ledger phase
        t0 = time.time()
        certification = self._certify_composed_scene()
        self._log_stage_time("composition_certify", time.time() - t0)
        repair = self._final_settle_repair_request(certification)
        if repair is None:
            return certification
        if repair.get("skip"):
            # audit F-M6: an unmapped object would have scoped the coverage gate to
            # nothing (fail open); record why the round was skipped instead.
            certification["repair_round"] = {"skipped": repair["skip"]}
            return certification
        stage = self._stage_spec("composition")
        stage_idx = 1 + [s.name for s in self.LOOP_STAGES].index("composition")
        logger.warning(
            "final free-settle moved %s notably; running one composition repair round",
            [row["object"] for row in repair["objects"]],
        )
        try:
            os.environ["GRASE_USAGE_TAG"] = "composition_repair"
            t0 = time.time()
            result = await self._run_stage_single_pass(stage, stage_idx, repair=repair)
            self._log_stage_time("composition_repair", time.time() - t0)
            self.stage_context["composition_repair"] = result
            # quality reads stage_artifacts (audit F-M5): make the round visible there
            self.stage_context.setdefault("stage_artifacts", {}).setdefault(
                str(stage_idx), {}
            )["composition_repair"] = result
            os.environ["GRASE_USAGE_TAG"] = "composition_certify"
            t0 = time.time()
            recert = self._certify_composed_scene()
            self._log_stage_time("composition_recertify", time.time() - t0)
        except Exception as exc:  # noqa: BLE001 - the repair round is non-fatal (F-H2)
            logger.exception("composition repair round failed; keeping the first certification")
            certification["repair_round"] = {
                "trigger": repair["objects"],
                "status": "error",
                "error": str(exc),
            }
            return certification
        still = (
            None
            if recert.get("status") == "error"
            else (self._final_settle_repair_request(recert) or {"objects": []})
        )
        recert["repair_round"] = {
            "trigger": repair["objects"],
            "before": {
                key: certification.get(key)
                for key in ("status", "toppled_objects", "objects")
            },
            "approved": result.get("approved"),
            "termination_reason": result.get("termination_reason"),
            # None when the re-certify errored (audit F-L11): unknown, not "nothing"
            "still_notable": (
                None
                if still is None
                else [row["object"] for row in still.get("objects", [])]
            ),
        }
        return recert

    @staticmethod
    def _repair_round_warnings(certification: dict[str, Any]) -> list[dict[str, Any]]:
        """Manifest warnings for a repair round that failed or was not approved (F-M5)."""
        rr = (certification or {}).get("repair_round") or {}
        out: list[dict[str, Any]] = []
        if rr.get("status") == "error":
            out.append(
                {
                    "code": "composition_repair_error",
                    "stage": "composition",
                    "message": "the final-settle repair round raised; the pre-repair "
                    "certification was kept: " + str(rr.get("error") or "")[:200],
                }
            )
        elif rr.get("skipped"):
            out.append(
                {
                    "code": "composition_repair_skipped",
                    "stage": "composition",
                    "message": str(rr["skipped"]),
                }
            )
        elif rr and rr.get("approved") is False:
            out.append(
                {
                    "code": "composition_repair_unapproved",
                    "stage": "composition",
                    "message": "the final-settle repair round ended without passing its "
                    f"rules (termination {rr.get('termination_reason')!r}); still notable: "
                    f"{rr.get('still_notable')!r}",
                }
            )
        return out

    def _final_settle_repair_request(
        self, certification: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        ""
        if self.harness_profile["name"] != "gpt6_v1":
            return None
        objects = certification.get("objects") or {}
        if certification.get("status") in {"error", "not_run", "skipped"} or not objects:
            return None
        from lib.tools.geometry.composition_physics import (
            CERT_DXY_CAP_M,
            CERT_TILT_CAP_DEG,
        )

        ids = self._mesh_to_object_ids()
        rollable = self._rollable_mesh_names()
        rows = []
        unmapped: list[str] = []
        for name, d in sorted(objects.items()):
            dxy, tilt = float(d.get("dxy", 0.0)), float(d.get("tilt_deg", 0.0))
            dz = abs(float(d.get("dz", 0.0)))
            toppled = bool(d.get("toppled"))
            # class-aware like the certify's own topple test: a ROLLABLE body (pen, lime)
            # turning about its axis in place is rolling, not a notable settle change
            # (v5accept genai4: stylus 10.1 deg / 2 mm triggered a round that could change
            # nothing); it counts by displacement only
            tilt_notable = tilt > CERT_TILT_CAP_DEG and name not in rollable
            if not (toppled or dxy > CERT_DXY_CAP_M or dz > CERT_DXY_CAP_M or tilt_notable):
                continue
            if name not in ids:
                unmapped.append(name)
            rows.append(
                {
                    "object": ids.get(name, name),
                    "moved_mm": round(dxy * 1000.0),
                    "dropped_mm": round(float(d.get("dz", 0.0)) * 1000.0),
                    "rotated_deg": round(tilt, 1),
                    "toppled": toppled,
                }
            )
        if not rows:
            return None
        if unmapped:
            # fail CLOSED (audit F-M6): without scene-graph ids the coverage scope would
            # be empty and the round would pass with zero investigations
            return {
                "objects": rows,
                "skip": "repair round skipped: no scene-graph id for "
                + ", ".join(unmapped),
            }
        return {
            "objects": rows,
            "max_rounds": min(self.FINAL_SETTLE_REPAIR_MAX_ROUNDS, 4 + 4 * len(rows)),
        }

    def _rollable_mesh_names(self) -> set[str]:
        """Mesh names the placement table marks rollable; best-effort (empty on error)."""
        md = self.args.get("moge_dir")
        try:
            rows = json.load(open(os.path.join(md, "placement.json")))["objects"]
            return {r["mesh_name"] for r in rows if r.get("mesh_name") and r.get("rollable")}
        except Exception:  # noqa: BLE001
            return set()

    def _mesh_to_object_ids(self) -> dict[str, str]:
        """{mesh_name: scene-graph id} from the placement table (category#instance);
        best-effort — {} leaves the certify's mesh names in place."""
        md = self.args.get("moge_dir")
        try:
            rows = json.load(open(os.path.join(md, "placement.json")))["objects"]
            return {
                r["mesh_name"]: f"{r['category']}#{r['instance']}"
                for r in rows
                if r.get("mesh_name")
            }
        except Exception:  # noqa: BLE001
            return {}

    def _n_pose_objects(self) -> int:
        """Meshed+masked object count (the composition pose loop's workload)."""
        md = self.args.get("moge_dir")
        try:
            place = json.load(open(os.path.join(md, "placement.json")))["objects"]
            return sum(1 for o in place if o.get("mesh_glb"))
        except Exception:  # noqa: BLE001
            return 8  # sensible default when placement is missing

    @staticmethod
    def _generator_contract_complete(
        result: dict[str, Any], *, require_rules: bool
    ) -> bool:
        """Whether a generator satisfied the explicit completion contract.

        A voluntary ``end`` and a current rules pass are separate facts. Stages
        without check_rules_enforced leave the latter to their verifier; initializer
        and composition require both.
        """
        return bool(
            result.get("completed_voluntarily") is True
            and (not require_rules or result.get("rules_passed") is True)
            and (not require_rules or result.get("completion_ready") is True)
        )

    @classmethod
    def _stage_completion_approved(
        cls,
        stage: StageSpec,
        result: dict[str, Any],
        verifier_result: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Evaluate a StageSpec's explicit acceptance policy."""
        verifier_approved = bool(
            verifier_result is not None and verifier_result.get("approved") is True
        )
        if stage.completion_policy == "verifier":
            return verifier_approved
        if stage.completion_policy == "generator_end_rules_and_verifier":
            return verifier_approved and cls._generator_contract_complete(
                result, require_rules=True
            )
        if stage.completion_policy == "generator_end_and_rules":
            return cls._generator_contract_complete(result, require_rules=True)
        raise ValueError(
            f"unknown completion policy for {stage.name}: {stage.completion_policy}"
        )

    async def _run_initializer_loop(
        self, stage: StageSpec, stage_idx: int
    ) -> dict[str, Any]:
        """Generator-verifier loop for the initialization stage.

        The generator builds the root surfaces + lighting; the verifier approves or
        rejects with feedback on root-surface composition; we retry up to
        ``max_stage_attempts``, feeding each rejection back to the next attempt. Like the
        texture/composition stages, the shared blend is NOT reset between attempts: each
        attempt refines the live scene the previous one left. The plan is carried forward
        too — on the 2nd+ attempt the generator loads its previous plan and UPDATES it
        (rather than re-planning from scratch)."""
        max_attempts = int(self.args.get("max_stage_attempts", 3))
        logger.info(
            "Starting initializer generator-verifier loop (up to %s attempts)",
            max_attempts,
        )
        retry_feedback: Optional[dict[str, Any]] = None
        prior_plan: Optional[str] = None
        result: dict[str, Any] = {}

        for attempt_idx in range(1, max_attempts + 1):
            logger.info("Running initializer attempt %s/%s", attempt_idx, max_attempts)
            generator_args = self._build_agent_args(
                stage.name, stage_idx, stage.generator_cls.__name__, attempt_idx
            )
            verifier_args = self._build_agent_args(
                stage.name, stage_idx, stage.verifier_cls.__name__, attempt_idx
            )
            if retry_feedback is not None:
                # stage_retry_feedback is a GENERATOR channel (_append_stage_retry_context);
                # the verifier receives the prior decision through its run() argument (see
                # _run_stage_verifier), so it only needs the checklist here.
                generator_args["stage_retry_feedback"] = retry_feedback
                generator_args["stage_approval_checklist"] = retry_feedback.get(
                    "approval_checklist", []
                )
                verifier_args["stage_approval_checklist"] = retry_feedback.get(
                    "approval_checklist", []
                )
                if prior_plan:  # carry the plan forward to update it
                    generator_args["stage_prior_plan"] = prior_plan

            verifier = stage.verifier_cls(verifier_args)
            generator = stage.generator_cls(generator_args)
            try:
                await verifier.tool_client.connect_servers()
                await generator.tool_client.connect_servers()
                result = await generator.run()
                result["latest_render"] = self._latest_normal_render(result)
                result["latest_state_blend"] = self._latest_state_blend(result)
                result["initializer_object_corrections"] = (
                    self._initializer_object_correction_summary(stage_idx)
                )
                verifier_result = await self._run_stage_verifier(
                    stage, stage_idx, result, verifier
                )
                retry_feedback = verifier_result
                prior_plan = result.get("init_plan") or prior_plan  # carry plan forward
                result["attempt_idx"] = attempt_idx
                result["verifier_result"] = verifier_result
                result["approved"] = self._stage_completion_approved(
                    stage, result, verifier_result
                )
                # Hard gate (initializer only): approval requires BOTH a voluntary
                # completion signal and a current passing check_rules_enforced result.
                # A budget cutoff is procedurally incomplete even when the last useful
                # tool happened to pass the broad rules floor; the verifier still runs
                # so the next attempt receives concrete incremental feedback.
                if not result["approved"] and verifier_result.get("approved") is True:
                    logger.warning(
                        "Initializer did not complete voluntarily with a current rules pass "
                        "(budget_exhausted=%s, rules_passed=%s); "
                        "forcing not-approved despite verifier approval.",
                        result.get("hit_max_rounds"),
                        result.get("rules_passed"),
                    )
                    result["approved"] = False
                    verifier_result["approved"] = False
                result["completion_status"] = (
                    "complete" if result["approved"] else "incomplete"
                )
                self._record_stage_attempt(stage_idx, stage.name, result)
                self._record_verifier_result(
                    stage_idx, stage.name, attempt_idx, verifier_result
                )
                if result["approved"]:
                    break
                if attempt_idx == max_attempts:
                    logger.warning(
                        "Initializer not approved after %s attempts; keeping the "
                        "final attempt (attempts refine the same live scene).",
                        max_attempts,
                    )
            finally:
                await verifier.cleanup()
                await generator.cleanup()

        # Attempts refine the SAME live scene (blend not reset, plan carried
        # forward), so the final attempt is the most refined by construction —
        # no best-attempt selection (same rationale as _run_stage).
        if result and not result.get("approved"):
            result["kept_final_attempt"] = True
        logger.info("Finished initializer loop at artifact index %s", stage_idx)
        return result

    def _initializer_object_correction_summary(self, stage_idx: int) -> dict[str, Any]:
        """Compact, persistent nudge evidence for bookkeeping and the verifier.

        The complete matrices remain in the on-disk audit ledger; copying them into
        generator_result would bloat every verifier prompt. This summary survives a
        later rules check and ``end``, when nudge_object is no longer the last tool.
        """
        ledger_path = os.path.join(
            self.args.get("output_dir", "output"),
            "stages",
            str(stage_idx),
            "initializer_object_transactions",
            "ledger.json",
        )
        try:
            data = json.load(open(ledger_path))
        except (OSError, json.JSONDecodeError):
            return {"ledger_path": ledger_path, "active": [], "recent_rejections": []}

        def compact(tx: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": tx.get("id"),
                "status": tx.get("status", "active"),
                "attempt_idx": tx.get("attempt_idx"),
                "target": tx.get("target"),
                "members": sorted((tx.get("objects") or {}).keys()),
                "translation": tx.get("translation"),
                "distance": tx.get("distance"),
                "reason": tx.get("reason"),
            }

        active = [compact(tx) for tx in data.get("active_transactions", [])]
        rejected = [
            compact(event)
            for event in data.get("events", [])
            if event.get("status") == "rejected"
        ][-3:]
        return {
            "ledger_path": ledger_path,
            "active": active,
            "recent_rejections": rejected,
        }

    async def _run_stage(self, stage: StageSpec, stage_idx: int) -> dict[str, Any]:
        """Run one generator/verifier stage pair (or a single generator session for
        ``single_pass`` stages)."""
        logger.info("Starting stage %s at artifact index %s", stage.name, stage_idx)
        if stage.single_pass:
            return await self._run_stage_single_pass(stage, stage_idx)
        max_attempts = int(self.args.get("max_stage_attempts", 3))
        retry_feedback: Optional[dict[str, Any]] = None
        result: dict[str, Any] = {}

        for attempt_idx in range(1, max_attempts + 1):
            logger.info(
                "Running generator attempt %s/%s for stage %s",
                attempt_idx,
                max_attempts,
                stage.name,
            )
            generator_args = self._build_agent_args(
                stage.name, stage_idx, stage.generator_cls.__name__, attempt_idx
            )
            verifier_args = self._build_agent_args(
                stage.name, stage_idx, stage.verifier_cls.__name__, attempt_idx
            )
            if retry_feedback is not None:
                # generator-only channel; see the initializer loop above for why the
                # verifier gets its history through _run_stage_verifier instead.
                generator_args["stage_retry_feedback"] = retry_feedback
                generator_args["stage_approval_checklist"] = retry_feedback.get(
                    "approval_checklist", []
                )
                verifier_args["stage_approval_checklist"] = retry_feedback.get(
                    "approval_checklist", []
                )

            verifier = stage.verifier_cls(verifier_args)
            generator = stage.generator_cls(generator_args)
            try:
                await verifier.tool_client.connect_servers()
                await generator.tool_client.connect_servers()
                result = await generator.run()
                result["latest_render"] = self._latest_normal_render(result)
                result["latest_state_blend"] = self._latest_state_blend(result)
                verifier_result = await self._run_stage_verifier(
                    stage, stage_idx, result, verifier
                )
                retry_feedback = verifier_result
                result["attempt_idx"] = attempt_idx
                result["verifier_result"] = verifier_result
                result["approved"] = self._stage_completion_approved(
                    stage, result, verifier_result
                )
                result["completion_status"] = (
                    "complete" if result["approved"] else "incomplete"
                )
                self._record_stage_attempt(stage_idx, stage.name, result)
                self._record_verifier_result(
                    stage_idx, stage.name, attempt_idx, verifier_result
                )
                if result["approved"]:
                    break
                if attempt_idx == max_attempts:
                    logger.warning(
                        "Stage %s was not approved after %s attempts; keeping the "
                        "final attempt (attempts CONTINUE from the previous "
                        "attempt's state, so the last one is the most refined — "
                        "no best-attempt selection).",
                        stage.name,
                        max_attempts,
                    )
            finally:
                await verifier.cleanup()
                await generator.cleanup()

        # No best-attempt selection here, for the same reason as the initializer:
        # attempts refine the SAME live scene state, the blend already holds the
        # final attempt's result, and an extra selection-verifier session cost a
        # full agent run to almost always pick what was already in place.
        if result and not result.get("approved"):
            result["kept_final_attempt"] = True

        logger.info("Finished stage %s at artifact index %s", stage.name, stage_idx)
        return result

    async def _run_stage_single_pass(
        self,
        stage: StageSpec,
        stage_idx: int,
        repair: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        ""
        attempt_idx = 1 if repair is None else 2
        generator_args = self._build_agent_args(
            stage.name, stage_idx, stage.generator_cls.__name__, attempt_idx
        )
        if repair is not None:
            generator_args["max_rounds"] = min(
                int(generator_args["max_rounds"]), int(repair["max_rounds"])
            )
            generator_args["final_settle_repair"] = repair
            generator_args["stage_context"]["final_settle_repair"] = {
                "why": self.FINAL_SETTLE_REPAIR_WHY,
                "objects": repair["objects"],
            }
        generator = stage.generator_cls(generator_args)
        try:
            await generator.tool_client.connect_servers()
            result = await generator.run()
            result["latest_render"] = self._latest_normal_render(result)
            result["latest_state_blend"] = self._latest_state_blend(result)
            result["attempt_idx"] = attempt_idx
            result["repair_round"] = repair is not None
            result["approved"] = self._stage_completion_approved(stage, result)
            result["completion_status"] = (
                "complete" if result["approved"] else "incomplete"
            )
            if not result["approved"]:
                # Nothing downstream re-checks this: there is no verifier and no retry,
                # so retain the best-effort artifact but label it explicitly incomplete.
                logger.warning(
                    "%s retained as an INCOMPLETE best-effort artifact "
                    "(termination_reason=%s, rules_passed=%s); this single-pass stage "
                    "has no verifier and is not approved.",
                    stage.name,
                    result.get("termination_reason"),
                    result.get("rules_passed"),
                )
                result["kept_final_attempt"] = True
            self._record_stage_attempt(stage_idx, stage.name, result)
        finally:
            await generator.cleanup()
        logger.info(
            "Finished single-pass stage %s at artifact index %s", stage.name, stage_idx
        )
        return result

    async def _run_stage_verifier(
        self,
        stage: StageSpec,
        stage_idx: int,
        generator_result: dict[str, Any],
        verifier: VerifierAgent,
    ) -> dict[str, Any]:
        """Run the paired verifier as the hard gate for stage advancement."""
        # ONE resolution per verifier run: the same render goes into the execution block
        # (as the attached image) and the argument block (as the recorded path).
        render = self._verifier_scene_render(stage.name, generator_result)
        execution = self._verifier_execution(stage.name, generator_result, render)
        # Resolve renders and execution evidence from the complete internal result first,
        # then cross the model-facing boundary through an explicit allowlist.  A blacklist
        # repeatedly leaked new filesystem fields (mask paths, attempt directories, blend
        # paths, ledgers) and the full scene graph into every verifier prompt.
        verifier_generator_result = self._verifier_generator_summary(generator_result)
        argument = {
            "stage_name": stage.name,
            "stage_idx": stage_idx,
            "generator_result": verifier_generator_result,
            "current_scene_render": render,
            "verifier_history": self._latest_verifier_result(stage_idx, stage.name),
            "approval_checklist": self._latest_approval_checklist(
                stage_idx, stage.name
            ),
        }
        result = await verifier.run(
            {
                "argument": argument,
                "execution": execution,
            }
        )
        result = self._enforce_manual_yaw_review(generator_result, result)
        # Recorded in the artifacts so "did this verdict see the scene?" is one grep,
        # not a memory-file archaeology dig (it took a full audit last time).
        result["had_scene_render"] = bool(render)
        return result

    def _verifier_generator_summary(
        self, generator_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Return the bounded, path-free generator state exposed to a verifier.

        ``generator_result`` is also Root's artifact-discovery and audit record, so it
        intentionally remains complete and unmodified.  Only stable procedural fields
        cross this serialization boundary; image paths are handled separately as actual
        attachments and raw evidence is represented by the revision-checked summaries.
        """
        allowed_status_fields = (
            "agent_name",
            "attempt_idx",
            "last_tool_name",
            "hit_max_rounds",
            "normal_rounds_exhausted",
            "terminal_end_grace_offered",
            "terminal_end_grace_accepted",
            "completed_voluntarily",
            "termination_reason",
            "rules_passed",
            "completion_ready",
            "scene_graph_revision",
        )
        summary = {
            key: generator_result[key]
            for key in allowed_status_fields
            if key in generator_result
        }

        structured_gate_summary = self._structured_gate_summary(generator_result)
        if structured_gate_summary:
            summary["structured_gate_summary"] = structured_gate_summary

        runtime_root_edits = self._current_runtime_root_surface_edits(generator_result)
        if runtime_root_edits:
            summary["runtime_root_surface_edits"] = runtime_root_edits

        corrections = generator_result.get("initializer_object_corrections")
        if isinstance(corrections, dict):
            row_fields = (
                "id",
                "status",
                "attempt_idx",
                "target",
                "members",
                "translation",
                "distance",
                "reason",
            )

            def compact_rows(key: str, limit: int) -> list[dict[str, Any]]:
                rows = corrections.get(key)
                if not isinstance(rows, list):
                    return []
                return [
                    {field: row[field] for field in row_fields if field in row}
                    for row in rows[:limit]
                    if isinstance(row, dict)
                ]

            compact_corrections = {
                "active": compact_rows("active", 8),
                "recent_rejections": compact_rows("recent_rejections", 3),
            }
            if any(compact_corrections.values()):
                summary["initializer_object_corrections"] = compact_corrections

        return summary

    @staticmethod
    def _current_advisory_requirement(
        generator_result: dict[str, Any], *, public: bool = False
    ) -> Optional[dict[str, Any]]:
        """Return only a schema-valid requirement bound to the current gate revision."""
        if generator_result.get("advisory_requirement_current") is not True:
            return None
        evidence_revision = generator_result.get("advisory_requirement_revision")
        current_revision = generator_result.get("gate_input_revision")
        if (
            evidence_revision is None
            or current_revision is None
            or evidence_revision != current_revision
        ):
            return None
        try:
            from lib.tools.geometry.yaw_advisory import (
                normalize_advisory_requirement,
            )

            return normalize_advisory_requirement(
                generator_result.get("advisory_requirement"), public=public
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _current_yaw_resolution(
        cls, generator_result: dict[str, Any], *, public: bool = False
    ) -> Optional[dict[str, Any]]:
        """Return only a valid current resolution matching the current request id."""
        if generator_result.get("yaw_resolution_current") is not True:
            return None
        evidence_revision = generator_result.get("yaw_resolution_revision")
        current_revision = generator_result.get("gate_input_revision")
        if (
            evidence_revision is None
            or current_revision is None
            or evidence_revision != current_revision
        ):
            return None
        try:
            from lib.tools.geometry.yaw_advisory import normalize_yaw_resolution

            resolution = normalize_yaw_resolution(
                generator_result.get("yaw_resolution"), public=public
            )
        except (TypeError, ValueError):
            return None
        requirement = cls._current_advisory_requirement(generator_result)
        if (
            requirement
            and requirement.get("state") != "not_required"
            and resolution.get("advisory_id") != requirement.get("advisory_id")
        ):
            return None
        return resolution

    @classmethod
    def _enforce_manual_yaw_review(
        cls, generator_result: dict[str, Any], verifier_result: Any
    ) -> dict[str, Any]:
        """Fail closed when a manual-review advisory lacks its structured row."""
        result = dict(verifier_result) if isinstance(verifier_result, dict) else {}
        requirement = cls._current_advisory_requirement(generator_result, public=True)
        if not requirement or requirement.get("state") != "manual_review":
            # An unsolicited row has no current request identity and cannot become
            # evidence merely because the VLM emitted a plausible-looking object.
            result.pop("yaw_advisory_review", None)
            return result
        advisory_id = str(requirement.get("advisory_id"))
        try:
            from lib.tools.geometry.yaw_advisory import normalize_manual_review

            review = normalize_manual_review(
                result.get("yaw_advisory_review"), advisory_id
            )
        except (TypeError, ValueError) as exc:
            result.pop("yaw_advisory_review", None)
            result["approved"] = False
            result["yaw_advisory_review_error"] = str(exc)[:300]
            result["edit_suggestion"] = (
                "Repeat the initializer visual review and return the required structured "
                f"yaw_advisory_review row for {advisory_id}; do not infer completion "
                "from prose."
            )
            text = list(result.get("text") or [])
            text.append(
                "Initializer approval refused: current manual yaw advisory lacks a "
                "valid structured review row."
            )
            result["text"] = text
            return result
        result["yaw_advisory_review"] = review
        if review["decision"] != "acceptable":
            result["approved"] = False
            if review["decision"] == "mismatch":
                result["edit_suggestion"] = (
                    result.get("edit_suggestion")
                    or "Correct the main-support yaw using the structured review anchors."
                )
        return result

    @staticmethod
    def _current_runtime_root_surface_edits(
        generator_result: dict[str, Any], *, limit: int = 8
    ) -> list[dict[str, Any]]:
        """Return only bounded runtime-root metadata for the current graph revision."""
        if generator_result.get("runtime_root_surface_edits_current") is not True:
            return []
        current_revision = generator_result.get("scene_graph_revision")
        evidence_revision = generator_result.get("runtime_root_surface_edits_revision")
        if current_revision is None or evidence_revision != current_revision:
            return []
        raw_rows = generator_result.get("runtime_root_surface_edits")
        if not isinstance(raw_rows, list):
            return []
        allowed = {
            "action",
            "id",
            "build_name",
            "surface_type",
            "description",
            "reason",
            "target_region_norm",
            "added_attempt",
            "source",
            "scene_graph_revision",
        }
        rows: list[dict[str, Any]] = []
        for raw in raw_rows[: max(0, int(limit))]:
            if not isinstance(raw, dict):
                continue
            row_revision = raw.get("scene_graph_revision", evidence_revision)
            if row_revision != current_revision:
                continue
            row = {key: raw[key] for key in allowed if key in raw}
            if not row.get("id") or not row.get("build_name"):
                continue
            row["scene_graph_revision"] = current_revision
            rows.append(row)
        return rows

    @staticmethod
    def _compact_constraint_measurements(
        value: Any, *, limit: int = 14
    ) -> dict[str, Any]:
        """Keep only bounded scalar diagnostics useful to the verifier.

        The complete backend record remains in the generator artifact.  In the model
        handoff, nested hulls/conflict payloads add tokens without changing the
        generated-scene verdict, so retain scalars and short one-dimensional lists
        only.  High-value measurements are selected first, then other eligible keys in
        deterministic order.
        """
        if not isinstance(value, dict):
            return {}

        def _small(item: Any) -> Any:
            if item is None or isinstance(item, (bool, int, float)):
                return item
            if isinstance(item, str):
                return item[:160]
            if isinstance(item, (list, tuple)) and len(item) <= 4:
                compact = []
                for part in item:
                    if part is None or isinstance(part, (bool, int, float)):
                        compact.append(part)
                    elif isinstance(part, str):
                        compact.append(part[:80])
                    else:
                        return None
                return compact
            return None

        priority = (
            "delta_degrees_mod_90",
            "tolerance_degrees_mod_90",
            "signed_gap_m",
            "near_edge_gap_m",
            "gap_m",
            "vertical_contact_gap_m",
            "wall_run_gap_m",
            "vertical_gap_m",
            "xy_gap_m",
            "finite_xy_distance_m",
            "a_corner_distance_m",
            "b_corner_distance_m",
            "acute_normal_angle_degrees",
            "normal_angle_deg",
            "rectangularity",
            "connected",
            "has_complete_bottom",
            "target_build_name",
            "reference_build_name",
            "reference_run_source",
            "reference_plane_source",
        )
        ordered = list(priority) + sorted(key for key in value if key not in priority)
        result: dict[str, Any] = {}
        for key in ordered:
            if key not in value or len(result) >= max(0, int(limit)):
                continue
            compact = _small(value[key])
            # ``None`` is a useful explicit "measurement unavailable" value.  A
            # non-scalar/nested input also maps to None above, so distinguish them.
            if compact is None and value[key] is not None:
                continue
            result[str(key)[:80]] = compact
        return result

    @classmethod
    def _compact_constraint_results(
        cls, value: Any, *, limit: int = 32
    ) -> dict[str, Any]:
        """Return a bounded generated-scene constraint-result summary."""
        raw_rows = value if isinstance(value, list) else []
        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            if len(rows) >= max(0, int(limit)):
                break
            if not isinstance(raw, dict):
                continue
            targets = raw.get("targets")
            reasons = raw.get("reason_codes")
            row = {
                "constraint_id": raw.get("constraint_id"),
                "source_relationship_id": raw.get("source_relationship_id"),
                "kind": raw.get("kind"),
                "stage": raw.get("stage"),
                "runtime_status": raw.get("runtime_status"),
                "targets": [str(item)[:120] for item in targets[:4]]
                if isinstance(targets, list)
                else [],
                "reason_codes": [str(item)[:160] for item in reasons[:8]]
                if isinstance(reasons, list)
                else [],
            }
            measurements = cls._compact_constraint_measurements(raw.get("measurements"))
            if measurements:
                row["measurements"] = measurements
            rows.append(row)
        return {
            "total": sum(isinstance(raw, dict) for raw in raw_rows),
            "shown": len(rows),
            "truncated": max(
                0, sum(isinstance(raw, dict) for raw in raw_rows) - len(rows)
            ),
            "results": rows,
        }

    @staticmethod
    def _constraint_ladder_fully_evaluated(rule: Any) -> bool:
        """Whether the current gate reached CONTACT with no deferred level."""
        if not isinstance(rule, dict) or rule.get("schema_version") != 2:
            return False
        deferred = rule.get("deferred_levels")
        if not isinstance(deferred, list) or deferred:
            return False
        levels = rule.get("evaluated_levels")
        if not isinstance(levels, list):
            return False
        reached: set[str] = set()
        for raw in levels:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name") or "").strip().casefold()
            level = raw.get("level")
            if name in {"structure", "pose", "contact"}:
                reached.add(name)
            elif level in {1, 2, 3}:
                reached.add({1: "structure", 2: "pose", 3: "contact"}[level])
        return reached == {"structure", "pose", "contact"}

    @staticmethod
    def _expected_constraint_kinds(
        relationship: dict[str, Any], generator_result: dict[str, Any]
    ) -> Optional[set[str]]:
        ""
        relation_type = str(relationship.get("type") or "").strip().casefold()
        if relation_type == "against":
            return {"edge_parallel_to_surface", "finite_against"}
        if relation_type == "corner":
            return {"finite_wall_corner"}
        if relation_type == "perpendicular":
            return {"finite_perpendicular"}
        if relation_type != "under":
            return None

        graph = generator_result.get("scene_graph")
        if not isinstance(graph, dict) or "error" in graph:
            return None
        expected = {"finite_under"}
        lower_id = str(relationship.get("a") or "")
        upper_id = str(relationship.get("b") or "")
        nodes = {
            str(node.get("id")): node
            for node in graph.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        lower = nodes.get(lower_id) or {}
        lower_category = (
            str(lower.get("category") or lower_id.split("#", 1)[0]).strip().casefold()
        )
        if upper_id == str(graph.get("main_support_id") or "") and lower_category in {
            "floor",
            "ground",
        }:
            expected.add("main_support_form")
        return expected

    @classmethod
    def _structured_gate_summary(
        cls,
        generator_result: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        ""
        rule = generator_result.get("rule_evidence")
        yaw = generator_result.get("yaw_evidence")
        relationships = generator_result.get("relationship_evidence")
        constraints = generator_result.get("constraint_results")
        advisory_requirement = cls._current_advisory_requirement(
            generator_result, public=True
        )
        yaw_resolution = cls._current_yaw_resolution(generator_result, public=True)
        if not any(
            (
                isinstance(rule, dict),
                isinstance(yaw, dict),
                isinstance(relationships, list),
                isinstance(constraints, list),
                isinstance(advisory_requirement, dict),
                isinstance(yaw_resolution, dict),
            )
        ):
            return None

        current_flag = generator_result.get("rules_evidence_current")
        evidence_revision = generator_result.get("rules_evidence_revision")
        current_revision = generator_result.get("gate_input_revision")
        if current_flag is not True:
            return None
        if (
            evidence_revision is None
            or current_revision is None
            or evidence_revision != current_revision
        ):
            return None

        summary: dict[str, Any] = {
            "schema_version": 3,
            "current": True,
            "revision": evidence_revision,
            "all_rules_passed": generator_result.get("rules_passed"),
        }
        if advisory_requirement:
            summary["yaw_advisory"] = {
                "requirement": advisory_requirement,
                **({"resolution": yaw_resolution} if yaw_resolution else {}),
            }
        if isinstance(rule, dict):
            scalar_keep = (
                "schema_version",
                "overall_status",
                "deferred_levels",
            )
            compact_rule = {key: rule[key] for key in scalar_keep if key in rule}
            evaluated_levels = rule.get("evaluated_levels")
            if isinstance(evaluated_levels, list):
                compact_levels = []
                for raw_level in evaluated_levels:
                    if not isinstance(raw_level, dict):
                        continue
                    level = {
                        key: raw_level[key]
                        for key in ("level", "name", "status")
                        if key in raw_level
                    }
                    compact_checks = []
                    for raw_check in raw_level.get("checks") or []:
                        if not isinstance(raw_check, dict):
                            continue
                        compact_checks.append(
                            {
                                key: raw_check.get(key)
                                for key in ("name", "status")
                                if key in raw_check
                            }
                        )
                    if compact_checks:
                        level["checks"] = compact_checks
                    compact_levels.append(level)
                compact_rule["evaluated_levels"] = compact_levels
            if compact_rule:
                summary["rules"] = compact_rule

        if isinstance(yaw, dict):
            app = yaw.get("applicability") or {}
            built = yaw.get("built") or {}
            selection = yaw.get("selection") or {}
            verdict = yaw.get("verdict") or {}
            compact_yaw: dict[str, Any] = {
                key: yaw[key]
                for key in ("schema_version", "main_surface")
                if key in yaw
            }
            if isinstance(app, dict):
                compact_yaw["applicability"] = {
                    key: app[key]
                    for key in (
                        "status",
                        "reason",
                        "geometry_source",
                        "rectangularity",
                    )
                    if key in app
                }
            if isinstance(built, dict):
                compact_yaw["built"] = {
                    key: built[key]
                    for key in ("source", "rectangularity")
                    if key in built
                }
            if isinstance(selection, dict):
                compact_yaw["selection"] = {
                    key: selection[key]
                    for key in (
                        "candidate_id",
                        "source",
                        "surface",
                        "surface_ids",
                        "relationship_id",
                        "relationship_ids",
                        "relationship",
                        "confidence",
                        "enforcement",
                        "reason",
                    )
                    if key in selection
                }
                raw_conflicts = selection.get("conflicts")
                if isinstance(raw_conflicts, list):
                    compact_yaw["selection"]["conflicts"] = [
                        {
                            key: conflict[key]
                            for key in (
                                "a",
                                "b",
                                "delta_degrees_mod_90",
                                "joint_feasibility_tolerance_degrees",
                            )
                            if key in conflict
                        }
                        for conflict in raw_conflicts[:8]
                        if isinstance(conflict, dict)
                    ]
            if isinstance(verdict, dict):
                compact_yaw["verdict"] = {
                    key: verdict[key]
                    for key in (
                        "status",
                        "delta_degrees_mod_90",
                        "tolerance_degrees",
                        "reason",
                    )
                    if key in verdict
                }

            # Preserve the selected anchor's analytic strength (for example mask-PCA
            # eccentricity) without handing every rejected candidate to the verifier.
            candidate_id = (
                selection.get("candidate_id") if isinstance(selection, dict) else None
            )
            candidates = yaw.get("anchor_candidates") or []
            selected_candidate = next(
                (
                    candidate
                    for candidate in candidates
                    if isinstance(candidate, dict)
                    and candidate.get("id") == candidate_id
                ),
                None,
            )
            if isinstance(selected_candidate, dict):
                compact_yaw["selected_candidate"] = {
                    key: selected_candidate[key]
                    for key in (
                        "id",
                        "source",
                        "surface",
                        "surface_id",
                        "surface_ids",
                        "relationship_id",
                        "relationship_ids",
                        "relationship",
                        "confidence",
                        "usable",
                        "reason",
                        "metrics",
                    )
                    if key in selected_candidate
                }

            # Never forward model-authored analytic/resolution prose from yaw evidence.
            # The only accepted resolution path is the separate schema-validated,
            # revision-bound ``yaw_advisory`` record assembled above.
            summary["yaw"] = compact_yaw

        compact_constraints = cls._compact_constraint_results(constraints)
        if isinstance(constraints, list):
            summary["constraints"] = compact_constraints

        if isinstance(relationships, list):
            raw_constraints = [
                raw for raw in constraints or [] if isinstance(raw, dict)
            ]
            constraints_by_source: dict[str, list[dict[str, Any]]] = {}
            for raw in raw_constraints:
                source_id = raw.get("source_relationship_id")
                if isinstance(source_id, str) and source_id:
                    constraints_by_source.setdefault(source_id, []).append(raw)

            ladder_complete = cls._constraint_ladder_fully_evaluated(rule)
            rows: list[dict[str, Any]] = []
            hard_total = 0
            hard_certified = 0
            hard_uncertified: list[str] = []
            expected_stages = {
                "main_support_form": "STRUCTURE",
                "edge_parallel_to_surface": "POSE",
                "finite_against": "CONTACT",
                "finite_under": "CONTACT",
                "finite_wall_corner": "CONTACT",
                "finite_perpendicular": "CONTACT",
            }
            for raw in relationships:
                if not isinstance(raw, dict):
                    continue
                relationship_id = raw.get("relationship_id")
                status = (
                    str(raw.get("status")).strip().casefold()
                    if isinstance(raw.get("status"), str)
                    else None
                )
                enforcement = (
                    str(raw.get("enforcement")).strip().casefold()
                    if isinstance(raw.get("enforcement"), str)
                    else None
                )
                source_rows = (
                    constraints_by_source.get(relationship_id, [])
                    if isinstance(relationship_id, str)
                    else []
                )
                expected_kinds = cls._expected_constraint_kinds(raw, generator_result)
                actual_kinds = {
                    str(result.get("kind"))
                    for result in source_rows
                    if isinstance(result.get("kind"), str)
                }
                ids = [
                    str(result.get("constraint_id"))
                    for result in source_rows
                    if isinstance(result.get("constraint_id"), str)
                    and result.get("constraint_id")
                ]
                statuses = [
                    str(result.get("runtime_status") or "").strip().casefold()
                    for result in source_rows
                ]
                stages_match = all(
                    str(result.get("stage") or "").strip().upper()
                    == expected_stages.get(str(result.get("kind") or ""))
                    for result in source_rows
                )
                certification_reasons: list[str] = []
                is_hard = status == "confirmed" and enforcement == "hard"
                certified = False
                if not is_hard:
                    certification_reasons.append("source_verdict_not_confirmed_hard")
                else:
                    hard_total += 1
                    if not ladder_complete:
                        certification_reasons.append(
                            "constraint_ladder_not_fully_evaluated"
                        )
                    if expected_kinds is None:
                        certification_reasons.append(
                            "compiled_constraint_set_unavailable"
                        )
                    elif (
                        actual_kinds != expected_kinds
                        or len(source_rows) != len(expected_kinds)
                        or len(ids) != len(source_rows)
                        or len(ids) != len(set(ids))
                        or not stages_match
                    ):
                        certification_reasons.append(
                            "compiled_constraint_results_incomplete"
                        )
                    unsatisfied = sorted(
                        {
                            state
                            for state in statuses
                            if state not in {"pass", "not_applicable"}
                        }
                    )
                    if unsatisfied:
                        certification_reasons.append(
                            "constraint_status_not_satisfied:" + ",".join(unsatisfied)
                        )
                    certified = not certification_reasons
                    if certified:
                        hard_certified += 1
                    else:
                        hard_uncertified.append(str(relationship_id))
                if len(rows) < 32:
                    rows.append(
                        {
                            "relationship_id": relationship_id,
                            "type": raw.get("type"),
                            "a": raw.get("a"),
                            "b": raw.get("b"),
                            # These are the immutable semantic-source verdict.  Runtime
                            # fulfillment lives exclusively in the constraint rows below.
                            "status": status,
                            "enforcement": enforcement,
                            "certified": certified,
                            "constraint_ids": ids[:8],
                            "certification_reason_codes": certification_reasons,
                        }
                    )
            valid_relationship_count = sum(
                isinstance(raw, dict) for raw in relationships
            )
            summary["relationships"] = {
                "total": valid_relationship_count,
                "shown": len(rows),
                "truncated": max(0, valid_relationship_count - len(rows)),
                "hard_total": hard_total,
                "hard_certified": hard_certified,
                "hard_uncertified_ids": hard_uncertified[:32],
                "source_results": rows,
            }
        return summary

    def _verifier_scene_render(
        self, stage_name: str, generator_result: dict[str, Any]
    ) -> Optional[str]:
        """The current full-scene reference-view render the verifier judges from.

        Use the generator's current render when available; otherwise render the
        reference view here."""
        render = self._latest_normal_render(generator_result)
        if render:
            return render
        render = self._render_reference_view_for(generator_result, stage_name)
        if render:
            # keep the artifacts consistent with what the verifier actually saw
            generator_result["latest_render"] = render
            return render
        logger.warning(
            "Stage %s verifier is running with NO current scene render (the generator "
            "produced none and the fallback reference-view render failed) — it must call "
            "render_reference_view itself.",
            stage_name,
        )
        return None

    def _render_reference_view_for(
        self, generator_result: dict[str, Any], stage_name: str
    ) -> Optional[str]:
        """Render the shared blend from its LOCKED reference camera for the verifier.

        Rendered with THIS STAGE's engine (`_stage_render_engine`), not the helper's EEVEE
        default: the lighting verifier grades exposure, shadow softness, highlight clipping
        and colour temperature, so an EEVEE stand-in for a CYCLES stage would have it
        judging a different image than the one being delivered. Best-effort — one
        background Blender launch, and a failure just means the verifier renders it itself
        with render_reference_view."""
        from lib.tools.geometry.ref_render import render_reference_view

        attempt_dir = generator_result.get(
            "attempt_output_dir"
        ) or generator_result.get("output_dir")
        blend = self.args.get("blender_save") or self.args.get("blender_file")
        blender = self.args.get("blender_command")
        if not (attempt_dir and blend and blender and os.path.exists(blend)):
            return None
        # NOT under renders/: that directory is the generator's own round log, and both
        # _latest_normal_render globs expect a per-round subdirectory there.
        out_png = os.path.join(attempt_dir, "verifier_view.png")
        engine = self._stage_render_engine(stage_name)
        try:
            if render_reference_view(blend, out_png, blender, engine):
                logger.info(
                    "Rendered the reference view for the %s verifier in %s: %s",
                    stage_name,
                    engine,
                    out_png,
                )
                return out_png
        except Exception as e:  # noqa: BLE001 - a preview render must not abort the stage
            logger.warning("Verifier reference-view render failed (%s)", e)
        return None

    def _verifier_execution(
        self,
        stage_name: str,
        generator_result: dict[str, Any],
        latest_render: Optional[str] = None,
    ) -> dict[str, Any]:
        """Build verifier execution input around ``latest_render`` (resolved by
        _verifier_scene_render; re-resolved here only for direct callers/tests)."""
        if latest_render is None:
            latest_render = self._latest_normal_render(generator_result)
        # COPY: this dict is generator_result["last_tool_response"], which is ALSO
        # serialized into the verifier's argument block — mutating it in place printed
        # the cutoff note twice in every initializer verifier's first message.
        last_response = dict(generator_result.get("last_tool_response") or {})
        for raw_evidence_key in (
            "rule_evidence",
            "yaw_evidence",
            "relationship_evidence",
            "constraint_results",
            "advisory_requirement",
            "yaw_resolution",
        ):
            last_response.pop(raw_evidence_key, None)
        structured_gate = self._structured_gate_summary(generator_result)
        # Initializer only: the generator did NOT finish voluntarily when it was cut off at max
        # rounds, OR it otherwise lacked a passing check_rules_enforced (rules never run /
        # last check FAILED / edited after passing). Either way tell the verifier to reject this
        # attempt and NOT to trust the rules as pre-enforced (it still evaluates every aspect for
        # feedback). Under RootSceneAgent this path is initializer-scoped: composition
        # returns from its single-pass generator without running a verifier. Keeping
        # composition in `gated` below is defensive for direct callers/tests only.
        gated = stage_name in ("initializer", "composition")
        # A max-round cutoff is always a procedural non-finish: a gate pass is the broad
        # floor, while the stage prompt still requires the generator to finish its visual
        # comparison and voluntarily call `end`.
        maxrounds = bool(gated and generator_result.get("hit_max_rounds"))
        rules_gap = (
            gated and not maxrounds and generator_result.get("rules_passed") is False
        )
        _incremental = (
            "The scene state IS preserved into the next attempt and may be largely correct — "
            "phrase edit_suggestion as INCREMENTAL edits to named surfaces (what to move/resize/"
            "recolor, keeping what is right). NEVER tell the generator to 'regenerate', 'rebuild', "
            "or 'start over' on the scene or the attempt. "
        )
        if maxrounds:
            cutoff_note = (
                "This attempt was CUT OFF at max generator rounds before the generator voluntarily "
                "called end. Budget exhaustion never waives the stage completion contract. You MUST "
                "set approved=false for this attempt, but still evaluate every "
                "aspect and give concrete fixes + a full approval_checklist for the next attempt. "
                + _incremental
            )
        elif rules_gap:
            cutoff_note = (
                "This attempt ended WITHOUT a passing check_rules_enforced (the rules gate did not "
                "pass: it was never run, or its last result FAILED, or the scene was edited after it "
                "passed). Do NOT treat the z=0 / penetration / table-base / relationship rules as "
                "already satisfied — verify them yourself and set approved=false on any violation. "
                "Still evaluate every aspect and give concrete fixes + a full approval_checklist. "
                + _incremental
            )
        elif (
            stage_name == "initializer" and generator_result.get("rules_passed") is True
        ):
            cutoff_note = (
                "This attempt PASSED check_rules_enforced: z=0/level, penetration, table base, "
                "and every hard constraint explicitly certified by the current gate evidence "
                "passed — do not re-verify or reject those certified items. Surface "
                "SIZE/coverage passed only the gate's WIDE band (roughly "
                "half-to-double area), which excludes gross errors ONLY: the finer visual "
                "match — the main support's position, size, and yaw WITHIN that band — is "
                "yours to judge per your CALIBRATION rule (edge-specific evidence required; "
                "vague impressions still cannot reject). Also judge plumb and lighting. "
            )
            relationship_summary = (
                structured_gate.get("relationships") if structured_gate else None
            )
            if relationship_summary is not None:
                hard_total = int(relationship_summary.get("hard_total") or 0)
                hard_certified = int(relationship_summary.get("hard_certified") or 0)
                uncertified = relationship_summary.get("hard_uncertified_ids") or []
                cutoff_note += (
                    "STRUCTURED CONSTRAINT FULFILLMENT: "
                    f"{hard_certified}/{hard_total} confirmed/hard source "
                    "relationship(s) are certified by all of their current compiled "
                    "constraints. "
                )
                if uncertified:
                    cutoff_note += (
                        "These confirmed/hard relationship IDs are NOT certified in the "
                        f"generated scene: {', '.join(uncertified)}. Inspect their compact "
                        "constraint rows; deferred, unverified, failed, or inconsistent "
                        "constraints never count as fulfillment. "
                    )
            else:
                cutoff_note += (
                    "NO CURRENT SOURCE-RELATIONSHIP/CONSTRAINT SUMMARY was supplied; "
                    "do not describe any relationship as individually certified. "
                )

            yaw_summary = structured_gate.get("yaw") if structured_gate else None
            yaw_app = (yaw_summary or {}).get("applicability") or {}
            yaw_verdict = (yaw_summary or {}).get("verdict") or {}
            yaw_selection = (yaw_summary or {}).get("selection") or {}
            yaw_app_status = str(yaw_app.get("status") or "").lower()
            yaw_status = str(yaw_verdict.get("status") or "").lower()
            yaw_enforcement = str(yaw_selection.get("enforcement") or "").lower()
            yaw_advisory = (
                structured_gate.get("yaw_advisory") if structured_gate else None
            ) or {}
            yaw_requirement = yaw_advisory.get("requirement") or {}
            yaw_resolution = yaw_advisory.get("resolution") or {}
            advisory_state = str(yaw_requirement.get("state") or "").lower()
            resolution_status = str(yaw_resolution.get("status") or "").lower()
            if resolution_status == "verified_match":
                cutoff_note += (
                    "MACHINE-VERIFIED RESIDUAL YAW: the backend validated issued "
                    "source/built edge candidate correspondences for advisory "
                    f"{yaw_resolution.get('advisory_id')}; do not re-litigate that yaw. "
                )
            elif yaw_app_status in {"not_applicable", "not-applicable"}:
                cutoff_note += (
                    "YAW NOT APPLICABLE: the backend classified the main support as "
                    f"{yaw_app.get('reason') or 'having no meaningful yaw'}; do not demand "
                    "an arbitrary in-plane rotation. "
                )
            elif yaw_summary and (
                yaw_app_status == "unknown"
                or yaw_status in {"advisory", "unverified", "unknown"}
                or yaw_enforcement == "advisory"
            ):
                selected = yaw_summary.get("selected_candidate") or {}
                analytic = selected.get("metrics") or {}
                cutoff_note += (
                    "EXCEPTION — structured evidence marks the main support's YAW "
                    f"{'ADVISORY' if yaw_enforcement == 'advisory' else (yaw_status.upper() or 'UNVERIFIED')} "
                    f"({yaw_verdict.get('reason') or yaw_selection.get('reason') or yaw_app.get('reason') or 'weak/absent anchor'}; "
                    f"selected-anchor metrics={analytic or 'none'}). "
                    "Check the table's edge directions against the target photo on the "
                    "reference view (the attached render, or re-render it with "
                    "render_reference_view — NEVER from initialize_viewpoint/set_camera "
                    "views) and reject with the rotation fix if it is visibly off. "
                )
            elif yaw_summary and yaw_status:
                cutoff_note += (
                    "STRUCTURED YAW EVIDENCE: "
                    f"{yaw_status.upper()} via {yaw_selection.get('source') or 'the selected anchor'}"
                    f" (enforcement={yaw_enforcement or 'unspecified'}, "
                    f"delta_mod_90={yaw_verdict.get('delta_degrees_mod_90')}, "
                    f"tolerance={yaw_verdict.get('tolerance_degrees')}). "
                )
            else:
                yaw_note = generator_result.get("yaw_note")
                if yaw_note:
                    cutoff_note += (
                        f"EXCEPTION — the gate did NOT verify the main support's YAW ({yaw_note}). "
                        "Check the table's edge directions against the target photo on the "
                        "reference view (the attached render, or re-render it with "
                        "render_reference_view — NEVER from initialize_viewpoint/set_camera "
                        "views) and reject with the rotation fix if it is visibly off. "
                    )
            if advisory_state == "manual_review":
                cutoff_note += (
                    "MANDATORY STRUCTURED YAW REVIEW: backend advisory "
                    f"{yaw_requirement.get('advisory_id')} was genuinely not machine-"
                    "resolvable. Your end call MUST include yaw_advisory_review with "
                    "schema_version=1, this exact advisory_id, decision="
                    "acceptable|mismatch|unverified, and 1-4 evidence rows containing "
                    "anchor_type plus separate target_reading/render_reading. Prose is "
                    "not a substitute. approved=true is permitted only with decision="
                    "acceptable; missing/malformed/mismatched IDs are rejected by root. "
                )
        else:
            cutoff_note = ""

        # Always carry the current structured record, including failed-gate evidence.
        # The prose above explains policy; this compact JSON preserves exact statuses,
        # selected weak-anchor metrics, and any analytic/resolution fields for the model.
        if stage_name == "initializer" and structured_gate:
            cutoff_note += (
                "CURRENT STRUCTURED GATE SUMMARY: "
                + json.dumps(structured_gate, ensure_ascii=False, separators=(",", ":"))
                + " "
            )
        runtime_root_edits = self._current_runtime_root_surface_edits(generator_result)
        if stage_name == "initializer" and runtime_root_edits:
            cutoff_note += (
                "CURRENT RUNTIME ROOT-SURFACE REGISTRATIONS: "
                + json.dumps(
                    runtime_root_edits,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + " These roots were registered transactionally by build_root_surface; "
                "judge whether each is visually necessary, distinct, and correctly "
                "placed. Do not reject one merely because preprocessing did not list it. "
            )
        corrections = generator_result.get("initializer_object_corrections") or {}
        active_corrections = corrections.get("active") or []
        if stage_name == "initializer" and active_corrections:
            rows = "; ".join(
                f"tx {row.get('id')}: {row.get('target')} "
                f"delta={row.get('translation')} reason={row.get('reason')} "
                f"members={row.get('members')}"
                for row in active_corrections
            )
            cutoff_note += (
                "ACTIVE TRANSACTIONAL OBJECT CORRECTIONS (already integrity/contact-"
                f"checked by the backend): {rows}. Judge their visible placement against "
                "the reference; this evidence persists even though nudge_object was "
                "followed by a rules check/end. "
            )
        if latest_render:
            return {
                "image": [latest_render],
                "text": [
                    (
                        cutoff_note
                        + f"Current full-scene render after {stage_name} generator attempt. "
                        "Compare this image directly against the target image before using investigation tools."
                    )
                ],
            }
        # No render reached the verifier (the generator produced none AND the fallback
        # reference-view render failed). Say so, instead of leaving it to infer the scene
        # from the leftover tool text — a silent omission that approved four days of runs.
        no_render = (
            "NO current scene render is attached (the render failed). Your FIRST tool "
            "call must be render_reference_view — do NOT judge from scene-info text."
        )
        base = last_response or {"text": []}
        base["text"] = [t for t in (cutoff_note, no_render) if t] + list(
            base.get("text") or []
        )
        return base

    def _latest_normal_render(self, generator_result: dict[str, Any]) -> Optional[str]:
        """Return the generator attempt's current source-view full-scene render, or None.

        Two conditions, both learned the hard way:
        (a) SOURCE VIEW ONLY — a novel-view render is a different camera, so it cannot be
            compared against the target photo (the verifier prompts say as much).
        (b) NEWER THAN THE LAST EDIT — 0725_comfix_real8219 and 0726_tiltfix_room1 both
            approved the delivered lighting from a snapshot taken before any light was
            touched. A pre-edit render shows a scene that no longer exists; treat it as
            no render at all and let the caller re-render.
        Prefers the path the generator REPORTED; the renders/ glob remains for runs with
        no pseudo-GT camera set, where execute() still writes the round's render there."""
        reported = generator_result.get("last_scene_render") or {}
        path = reported.get("path")
        if path and not (reported.get("azimuth") or reported.get("elevation")):
            if self._render_is_current(path, generator_result):
                return path

        attempt_dir = generator_result.get(
            "attempt_output_dir"
        ) or generator_result.get("output_dir")
        if not attempt_dir:
            return None
        render_dir = Path(attempt_dir) / "renders"
        if not render_dir.exists():
            return None

        images = [
            path
            for path in render_dir.glob("*/*")
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and "focus_" not in path.parent.name.lower()
        ]
        # scene-camera renders first, anything else (other cameras) only as a fallback
        candidates = [path for path in images if path.name == "Camera.png"] or images
        if not candidates:
            return None
        newest = str(max(candidates, key=lambda path: path.stat().st_mtime))
        return newest if self._render_is_current(newest, generator_result) else None

    def _render_is_current(
        self, render_path: str, generator_result: dict[str, Any]
    ) -> bool:
        """True when ``render_path`` is at least as new as the attempt's last EDIT."""
        try:
            render_mtime = os.path.getmtime(render_path)
        except OSError:
            return False
        edit_blend = self._latest_edit_blend(generator_result)
        if not edit_blend:
            return (
                True  # nothing edited (or nothing recorded) -> nothing to be stale for
            )
        try:
            return render_mtime >= os.path.getmtime(edit_blend)
        except OSError:
            return True

    def _latest_edit_blend(self, generator_result: dict[str, Any]) -> Optional[str]:
        """Newest state.blend written by an EDIT (execute / move / flip snapshots).

        Excludes the read-only render snapshots (``<n>_render_current_scene/``), which copy
        the blend AFTER rendering into the same directory — counting those would make every
        bare render_current_scene look stale against its own snapshot."""
        blends = self._state_blends(generator_result)
        edits = [b for b in blends if "_render_current_scene" not in b.parent.name]
        if not edits:
            return None
        return str(max(edits, key=lambda path: path.stat().st_mtime))

    def _state_blends(self, generator_result: dict[str, Any]) -> list[Path]:
        """Every state.blend snapshot under a generator attempt's renders/ directory."""
        attempt_dir = generator_result.get(
            "attempt_output_dir"
        ) or generator_result.get("output_dir")
        if not attempt_dir:
            return []
        return list((Path(attempt_dir) / "renders").glob("*/state.blend"))

    def _latest_state_blend(self, generator_result: dict[str, Any]) -> Optional[str]:
        """Return the latest saved Blender state from a generator attempt."""
        candidates = self._state_blends(generator_result)
        if not candidates:
            return None
        return str(max(candidates, key=lambda path: path.stat().st_mtime))

    def _build_stage_args(self, stage_name: str, stage_idx: int) -> dict[str, Any]:
        """Build stage config with accumulated upstream context."""
        stage_args = dict(self.args)
        # The ExternalToolClient forwards this full dictionary in every MCP
        # initialize request. Re-resolve to give each stage an independent manifest.
        profile = getattr(self, "harness_profile", None)
        if not isinstance(profile, dict):
            profile = resolve_harness_profile(self.args.get("harness_profile"))
        stage_args["harness_profile"] = profile["name"]
        stage_args["harness_profile_manifest"] = resolve_harness_profile(
            profile["name"]
        )
        stage_args["root_stage_name"] = stage_name
        stage_args["memory_window"] = None
        if stage_name == "composition":
            # merged pose refinement: investigate/move caps + windowed memory
            stage_args["investigate_cap"] = int(self.args.get("investigate_cap") or 3)
            stage_args["memory_window"] = int(self.args.get("memory_window") or 5)
        stage_args["root_stage_idx"] = stage_idx
        stage_args["stage_context"] = self._prompt_stage_context(stage_name, stage_idx)
        if stage_name == "composition":
            stage_args["stage_context"]["physics_note"] = (
                "move() commits are physics-simulated to rest: a move that would "
                "topple or lose the photo match is REJECTED and auto-reverted (the "
                "tool says so), anything stacked on a moved object rides along and "
                "settles with it, and the penetration gate reads the simulator's "
                "collision geometry."
            )
        stage_args["render_engine"] = self._stage_render_engine(stage_name)
        return stage_args

    def _stage_spec(self, stage_name: str) -> Optional[StageSpec]:
        for spec in (self.INITIALIZATION_STAGE, *self.LOOP_STAGES):
            if spec.name == stage_name:
                return spec
        return None

    def _stage_max_rounds(self, stage_name: str, is_verifier: bool) -> int:
        """Per-attempt round budget for one agent. Precedence: the per-stage CLI flag
        (--init/texture/composition/lighting-{generator,verifier}-max-rounds), then the
        stage's StageSpec override, then the global --verifier-max-rounds flag for
        verifiers or the internal generator fallback for an unknown stage.

        Generators are pinned per stage by StageSpec (initializer 20, texture and
        lighting 10; composition is the dynamic single-session budget below), so the
        Verifiers carry NO spec override on purpose — 8 everywhere, and
        --verifier-max-rounds stays a working knob for all of them."""
        key = "init" if stage_name == "initializer" else stage_name
        role = "verifier" if is_verifier else "generator"
        cli = self.args.get(f"{key}_{role}_max_rounds")
        if cli is not None:
            return int(cli)
        spec = self._stage_spec(stage_name)
        override = None
        if spec is not None:
            override = (
                spec.verifier_max_rounds if is_verifier else spec.generator_max_rounds
            )
        if override is not None:
            return int(override)
        if stage_name == "composition" and not is_verifier:
            return min(70, 8 + 5 * self._n_pose_objects())
        return int(self.args.get("verifier_max_rounds", 8) if is_verifier else 10)

    def _build_agent_args(
        self,
        stage_name: str,
        stage_idx: int,
        agent_name: str,
        attempt_idx: int,
    ) -> dict[str, Any]:
        """Build per-agent config and allocate its artifact directory."""
        agent_args = self._build_stage_args(stage_name, stage_idx)
        stage_dir = os.path.join(
            self.args.get("output_dir", "output"), "stages", str(stage_idx)
        )
        agent_dir = os.path.join(stage_dir, agent_name)
        attempt_dir = os.path.join(agent_dir, f"attempt_{attempt_idx}")
        os.makedirs(attempt_dir, exist_ok=True)
        agent_args["stage_dir"] = stage_dir
        agent_args["agent_output_dir"] = agent_dir
        agent_args["attempt_idx"] = attempt_idx
        agent_args["attempt_output_dir"] = attempt_dir
        # Captured BEFORE output_dir is rebound to the attempt dir: prompt captions render
        # image paths relative to this root so they carry no date-coded run name, stage
        # index, attempt number, or username (see lib/utils/common.display_path).
        agent_args["scene_root"] = self.args.get("output_dir", "output")
        agent_args["output_dir"] = attempt_dir
        # A per-stage flag wins over StageSpec. Verifiers without a StageSpec override
        # use the shared --verifier-max-rounds budget.
        is_verifier = "Verifier" in agent_name
        agent_args["max_rounds"] = self._stage_max_rounds(stage_name, is_verifier)
        if is_verifier:
            own = self._own_stage_verifier_history_for_prompt(stage_name, stage_idx)
            sc = agent_args.get("stage_context")
            if own and isinstance(sc, dict):
                sc["latest_verifier_history"] = own
            # CT5: hand the incoming generator the CURRENT scene render up front, so it
            # does not spend its opening round(s) re-rendering to orient itself (the
            # measured pattern: a bare render_current_scene "to see the baseline" in
            # round 0-1 of nearly every stage). Source: the newest prior stage's
            # recorded latest_render (source view, never stale — see
            # _latest_normal_render). None for the first stage; the prompt block is
            # simply omitted then.
            entry_render = self._prev_stage_render()
            if entry_render:
                agent_args["stage_entry_render"] = entry_render
        return agent_args

    def _prev_stage_render(self) -> Optional[str]:
        """The newest prior stage's recorded source-view render, if it still exists."""
        arts = self.stage_context.get("stage_artifacts") or {}
        for idx in sorted(arts, key=int, reverse=True):
            for res in arts[idx].values():
                p = (res or {}).get("latest_render")
                if p and os.path.exists(p):
                    return p
        return None

    def _record_stage_result(
        self, stage_idx: int, stage_name: str, result: dict[str, Any]
    ) -> None:
        """Store a stage artifact summary under its pass index."""
        artifacts = self.stage_context["stage_artifacts"].setdefault(str(stage_idx), {})
        artifacts[stage_name] = result

    def _record_stage_attempt(
        self, stage_idx: int, stage_name: str, result: dict[str, Any]
    ) -> None:
        """Store every generator/verifier attempt, including rejected ones."""
        attempts_by_idx = self.stage_context["stage_attempts"].setdefault(
            str(stage_idx), {}
        )
        attempts_by_idx.setdefault(stage_name, []).append(result)

    def _record_verifier_result(
        self,
        stage_idx: int,
        stage_name: str,
        attempt_idx: int,
        verifier_result: dict[str, Any],
    ) -> None:
        """Store verifier decisions so future attempts can reason over prior requirements."""
        history = self.stage_context["verifier_history"].setdefault(str(stage_idx), {})
        history.setdefault(stage_name, []).append(
            {
                "attempt_idx": attempt_idx,
                "approved": bool(verifier_result.get("approved", False)),
                "approval_checklist": verifier_result.get("approval_checklist", []),
                "problem_images": verifier_result.get("problem_images", []),
                "suggested_fixes": verifier_result.get("suggested_fixes", []),
                "regression_check": verifier_result.get("regression_check", ""),
                "text": verifier_result.get("text", []),
            }
        )

    def _stage_verifier_history(self, stage_idx: int, stage_name: str) -> Any:
        """Return prior verifier decisions for this stage."""
        return (
            self.stage_context.get("verifier_history", {})
            .get(str(stage_idx), {})
            .get(stage_name, [])
        )

    def _latest_verifier_result(self, stage_idx: int, stage_name: str) -> Any:
        """Return only the latest verifier decision for this stage."""
        history = self._stage_verifier_history(stage_idx, stage_name)
        if not history:
            return None
        return history[-1]

    def _latest_approval_checklist(self, stage_idx: int, stage_name: str) -> Any:
        """Return the latest verifier checklist for this stage, if any."""
        history = self._stage_verifier_history(stage_idx, stage_name)
        if not history:
            return []
        return history[-1].get("approval_checklist", [])

    def _prompt_stage_context(self, stage_name: str, stage_idx: int) -> dict[str, Any]:
        """Return the agent-facing pipeline position without orchestrator bookkeeping."""
        context: dict[str, Any] = {
            "stage_order": self._prompt_stage_order(),
            "current_stage": stage_name,
        }
        runtime_roots = self._prompt_runtime_root_surfaces()
        if runtime_roots:
            context["runtime_root_surfaces"] = runtime_roots
        return context

    def _prompt_stage_order(self) -> list[str]:
        """Stage names in execution order, derived from the stage specs themselves so it
        cannot drift from what the orchestrator actually runs."""
        return [self.INITIALIZATION_STAGE.name] + [s.name for s in self.LOOP_STAGES]

    def _prompt_runtime_root_surfaces(self) -> list[dict[str, Any]]:
        """Runtime-added roots recorded by a completed stage, hoisted out of the
        per-stage nesting.

        Only the initializer can register these (``build_root_surface``), so at most one
        stage contributes. Downstream stages also see each runtime root in the live scene
        graph and in the CURRENT SCENE STATE preseed under its build name; this row set is
        the explicit handoff that they are ordinary required roots, not agent inventions.
        """
        for stage_idx in sorted(
            self.stage_context.get("stage_artifacts", {}), key=lambda k: str(k)
        ):
            for result in self.stage_context["stage_artifacts"][stage_idx].values():
                if not isinstance(result, dict):
                    continue
                rows = self._current_runtime_root_surface_edits(result)
                if rows:
                    return rows
        return []

    def _stage_artifacts_for_prompt(self) -> dict[str, Any]:
        """Compact per-stage artifacts for the pipeline_result.json manifest.

        No longer reaches any prompt (see ``_prompt_stage_context``); the name is kept
        because ``_export_final_result`` is the remaining caller and the manifest schema
        depends on this exact shape.
        """
        prompt_artifacts: dict[str, Any] = {}
        for stage_idx, stage_results in self.stage_context.get(
            "stage_artifacts", {}
        ).items():
            prompt_artifacts[stage_idx] = {}
            for stage_name, result in stage_results.items():
                prompt_artifacts[stage_idx][stage_name] = self._stage_result_for_prompt(
                    result
                )
        return prompt_artifacts

    @staticmethod
    def _compress_scene_graph(graph: Any) -> Any:
        """Compact a scene graph to just id / description / parent per node. The full graph
        (plane, mask_path, world_center, span ...) is too long for the downstream texture /
        composition prompts, which only need object identity + the support hierarchy."""
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
            return graph
        return {
            "nodes": [
                {
                    "id": n.get("id"),
                    "description": n.get("description") or n.get("category"),
                    "parent": n.get("parent") or n.get("support"),
                }
                for n in graph["nodes"]
                if isinstance(n, dict)
            ]
        }

    def _stage_result_for_prompt(self, result: dict[str, Any]) -> dict[str, Any]:
        """Strip bulky execution/memory fields from a stage result.

        The stage's verifier decision is intentionally NOT included here — it is carried once,
        per stage, in ``latest_verifier_history`` (see ``_prompt_stage_context``). Embedding it
        here too duplicated the whole verifier decision in the upstream-context JSON.
        """
        # scene_graph is NOT inlined: it was embedded once per completed stage (the
        # composition verifier carried 3 identical 1.7k-char copies) and the agents
        # that need it get it via their own per-scene data block; the json path
        # stays. attempt_output_dir always equalled output_dir.
        runtime_root_edits = self._current_runtime_root_surface_edits(result)
        compact = {
            "agent_name": result.get("agent_name"),
            "attempt_idx": result.get("attempt_idx"),
            "approved": result.get("approved"),
            "output_dir": result.get("output_dir"),
            "scene_graph_json": result.get("scene_graph_json"),
            "scene_graph_revision": result.get("scene_graph_revision"),
            # This list is already bounded and revision-checked in GeneratorAgent;
            # preserve it so texture/lighting/composition know the active graph path
            # contains initializer-registered roots, without inlining the full graph.
            "runtime_root_surface_edits": runtime_root_edits or None,
            "last_tool_name": result.get("last_tool_name"),
        }
        return {key: value for key, value in compact.items() if value is not None}

    def _own_stage_verifier_history_for_prompt(
        self, stage_name: str, stage_idx: int
    ) -> dict[str, Any]:
        ""
        history = (
            self.stage_context.get("verifier_history", {})
            .get(str(stage_idx), {})
            .get(stage_name)
        )
        if not history:
            return {}
        last = history[-1]
        return {k: v for k, v in last.items() if k not in ("text", "problem_images")}

    def _stage_render_engine(self, stage_name: str) -> str:
        """Return the Blender render engine for a stage."""
        explicit_key = f"{stage_name}_render_engine"
        if self.args.get(explicit_key):
            return str(self.args[explicit_key])
        render_engines = self.args.get("stage_render_engines")
        if isinstance(render_engines, dict) and render_engines.get(stage_name):
            return str(render_engines[stage_name])
        default_stage_engines = {
            self.INITIALIZATION_STAGE.name: "BLENDER_EEVEE_NEXT",
            "texture": "CYCLES",
            "composition": "CYCLES",
            "lighting": "CYCLES",
        }
        if stage_name in default_stage_engines:
            return default_stage_engines[stage_name]
        return str(self.args.get("render_engine") or "CYCLES")

    def _render_final_cycles_target_resolution(
        self,
        output_subdir: str = "final",
        filename: str = "final_cycles_target_resolution.png",
    ) -> str:
        """Render the accumulated scene once with Cycles at target image resolution."""
        output_dir = Path(self.args.get("output_dir", "output")) / output_subdir
        script_dir = output_dir / "scripts"
        render_dir = output_dir / "renders"
        script_dir.mkdir(parents=True, exist_ok=True)
        render_dir.mkdir(parents=True, exist_ok=True)
        blend_file = self.args.get("blender_save") or self.args.get("blender_file")
        target_image = self.args.get("target_image_path")
        if not blend_file:
            raise ValueError("Final render requires blender_file or blender_save")
        if not target_image or not os.path.exists(target_image):
            raise ValueError(
                f"Final render requires an existing target image path, got {target_image}"
            )

        with Image.open(target_image) as img:
            width, height = img.size
        render_path = render_dir / filename
        if render_path.exists():
            render_path.unlink()
        script_path = script_dir / "render_final_cycles_target_resolution.py"
        script_path.write_text(f"""import os
import sys

import bpy

render_path = {repr(str(render_path))}
width = {int(width)}
height = {int(height)}

scene = bpy.context.scene
if scene.camera is None:
    raise ValueError("Cannot render final image because the scene has no active camera")

scene.render.engine = "CYCLES"
scene.cycles.samples = min(getattr(scene.cycles, "samples", 128), 128)
scene.render.resolution_x = width
scene.render.resolution_y = height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = render_path
os.makedirs(os.path.dirname(render_path), exist_ok=True)
bpy.ops.render.render(write_still=True)
""")
        cmd = [
            self.args.get("blender_command"),
            "--background",
            blend_file,
            "--python-exit-code",
            "1",
            "--python",
            str(script_path),
        ]
        env = os.environ.copy()
        if self.args.get("gpu_devices"):
            env["CUDA_VISIBLE_DEVICES"] = str(self.args["gpu_devices"])
        env["AL_LIB_LOGLEVEL"] = "0"
        cmd_str = " ".join(shlex.quote(str(part)) for part in cmd)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=1800
        )
        if proc.returncode != 0 or not render_path.exists():
            raise RuntimeError(
                "Final verifier render failed.\n"
                f"Command: {cmd_str}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return str(render_path)

    def _validate_final_blend(self, blend_path: Path) -> dict[str, Any]:
        """Validate structural invariants on the exact delivered blend."""
        blender = self.args.get("blender_command")
        moge_dir = self.args.get("moge_dir")
        if not blender or not moge_dir:
            return {
                "status": "invalid",
                "errors": ["final validation requires Blender and scene artifacts"],
            }
        try:
            from lib.tools.geometry.inventory_contract import validate_scene_artifacts
            from lib.tools.geometry.runtime_object_repair import load_runtime_inventory

            runtime_inventory = None
            if self.harness_profile["name"] == "gpt6_v1":
                recovery_markers = nonterminal_runtime_recovery_markers(moge_dir)
                if recovery_markers:
                    raise RuntimeError(
                        "cannot finalize with nonterminal GPT-6 runtime recovery "
                        "markers: " + ", ".join(str(path) for path in recovery_markers)
                    )
                # Presence and conditional audit requirements are checked separately
                # from semantic inventory validation because pose/base physics files
                # are consumed only by inheritance and Isaac export.
                _collect_gpt6_scene_artifact_outputs(Path(moge_dir))
                if self.harness_profile["capabilities"].get(
                    "runtime_object_inventory", False
                ):
                    runtime_inventory = load_runtime_inventory(moge_dir)
            inventory_report = validate_scene_artifacts(
                moge_dir,
                # Runtime root surfaces are part of the longstanding baseline
                # initializer contract. Only the object-overlay/material extension
                # is profile-gated below.
                allow_runtime_additions=True,
                allow_runtime_inventory=(self.harness_profile["name"] == "gpt6_v1"),
                validate_runtime_physics=(
                    self.harness_profile["name"] == "gpt6_v1"
                    and self.harness_profile["capabilities"].get(
                        "runtime_object_inventory", False
                    )
                ),
            )
            expected_by_id = inventory_report["object_mesh_names"]
            expected = sorted(expected_by_id.values())
            expected_roots = sorted(inventory_report["root_build_names"].values())
        except Exception as exc:  # noqa: BLE001 - converted to final integrity result
            return {"status": "invalid", "errors": [f"inventory invalid: {exc}"]}
        if not expected and not (
            runtime_inventory is not None and runtime_inventory.get("tombstones")
        ):
            return {
                "status": "invalid",
                "errors": ["scene graph contains no required meshed objects"],
            }
        report_path = blend_path.parent / "blend_validation.json"
        script_path = blend_path.parent / "scripts" / "validate_final_blend.py"
        strict_runtime_hierarchy = self.harness_profile[
            "name"
        ] == "gpt6_v1" and self.harness_profile["capabilities"].get(
            "runtime_object_inventory", False
        )
        from lib.tools.geometry.runtime_object_repair import procedural_capture_of

        authored_empty_ids = {
            record["placement"]["mesh_name"]: record["id"]
            for record in (runtime_inventory or {}).get("objects", [])
            if record.get("mask_policy") == "excluded"
            or procedural_capture_of(record) is not None  # tracked composition replacement
        }
        script_path.parent.mkdir(parents=True, exist_ok=True)
        if report_path.exists():
            report_path.unlink()
        script_path.write_text(
            "import json\n"
            "import math\n"
            "import bpy\n\n"
            f"expected = {expected!r}\n"
            f"expected_roots = {expected_roots!r}\n"
            f"strict_runtime_hierarchy = {strict_runtime_hierarchy!r}\n"
            f"authored_empty_ids = {authored_empty_ids!r}\n"
            "renderable_root_types = {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', "
            "'VOLUME', 'CURVES', 'POINTCLOUD', 'GPENCIL', 'GREASEPENCIL'}\n"
            f"report_path = {str(report_path)!r}\n"
            "errors = []\n"
            "cameras = [o for o in bpy.data.objects if o.type == 'CAMERA']\n"
            "scene_objects = {o.name: o for o in bpy.context.scene.objects}\n"
            "def has_render_collection_path(obj):\n"
            "    def visit(layer, hidden):\n"
            "        hidden = hidden or bool(getattr(layer, 'exclude', False)) "
            "or bool(layer.collection.hide_render)\n"
            "        if not hidden and layer.collection.objects.get(obj.name) is obj:\n"
            "            return True\n"
            "        return any(visit(child, hidden) for child in layer.children)\n"
            "    return visit(bpy.context.view_layer.layer_collection, False)\n"
            "if bpy.context.scene.camera is None:\n"
            "    errors.append('scene has no active camera')\n"
            "if len(cameras) != 1:\n"
            "    errors.append(f'expected exactly one camera, found {len(cameras)}')\n"
            "pipeline_names = sorted(o.name for o in bpy.data.objects "
            "if o.name.startswith('obj_'))\n"
            "if strict_runtime_hierarchy:\n"
            "    for member_name in pipeline_names:\n"
            "        if member_name in expected:\n"
            "            continue\n"
            "        owners = [root for root in expected if member_name.startswith(root + '_') "
            "and member_name[len(root) + 1:].isdigit()]\n"
            "        if not owners:\n"
            "            continue\n"
            "        member = bpy.data.objects.get(member_name)\n"
            "        current = member.parent if member is not None else None\n"
            "        ancestors = set()\n"
            "        while current is not None and current.name not in ancestors:\n"
            "            ancestors.add(current.name)\n"
            "            current = current.parent\n"
            "        if owners[0] not in ancestors:\n"
            "            errors.append(f'{member_name} is not a descendant of its canonical ' "
            "f'object {owners[0]}')\n"
            "for name in expected:\n"
            "    obj = bpy.data.objects.get(name)\n"
            "    if obj is None:\n"
            "        errors.append(f'missing required object {name}')\n"
            "        continue\n"
            "    if scene_objects.get(name) is not obj:\n"
            "        errors.append(f'required object {name} is not linked to the "
            "active scene')\n"
            "    elif not has_render_collection_path(obj):\n"
            "        errors.append(f'required object {name} has no render-enabled "
            "collection path')\n"
            "    if obj.hide_render:\n"
            "        errors.append(f'required object {name} has hide_render enabled')\n"
            "    if hasattr(obj, 'visible_camera') and not obj.visible_camera:\n"
            "        errors.append(f'required object {name} has camera visibility disabled')\n"
            "    if strict_runtime_hierarchy and name in authored_empty_ids:\n"
            "        parts = list(obj.children_recursive)\n"
            "        if obj.type != 'EMPTY' or obj.get('grase_graph_id') != authored_empty_ids[name]:\n"
            "            errors.append(f'{name} is not its inventory-bound authored EMPTY')\n"
            "        if not parts or any(part.type != 'MESH' for part in parts):\n"
            "            errors.append(f'{name} must have nonempty MESH-only descendants')\n"
            "        for part in parts:\n"
            "            if part.type != 'MESH':\n"
            "                continue\n"
            "            if not part.data.vertices or not part.data.polygons:\n"
            "                errors.append(f'{name} has empty mesh part {part.name}')\n"
            "            if part.parent is not obj:\n"
            "                errors.append(f'{part.name} is not a direct child of {name}')\n"
            "            if scene_objects.get(part.name) is not part or not has_render_collection_path(part):\n"
            "                errors.append(f'{part.name} is not render-linked in the active scene')\n"
            "            if part.hide_render or part.hide_viewport or not part.visible_camera:\n"
            "                errors.append(f'{part.name} has disabled visibility')\n"
            "            if part.constraints:\n"
            "                errors.append(f'{part.name} has constraints')\n"
            "            values = [v for row in part.matrix_world for v in row]\n"
            "            if not all(math.isfinite(v) for v in values) or part.matrix_world.to_3x3().determinant() <= 1e-12:\n"
            "                errors.append(f'{part.name} has an invalid transform')\n"
            "    elif obj.type != 'MESH':\n"
            "        errors.append(f'{name} is {obj.type}, expected MESH')\n"
            "    if obj.parent is not None:\n"
            "        errors.append(f'{name} is parented under {obj.parent.name}')\n"
            "    if any(c.type == 'CHILD_OF' for c in obj.constraints):\n"
            "        errors.append(f'{name} has a CHILD_OF constraint')\n"
            "    values = [v for row in obj.matrix_world for v in row]\n"
            "    if not all(math.isfinite(v) for v in values):\n"
            "        errors.append(f'{name} has a non-finite transform')\n"
            "    if abs(obj.matrix_world.to_3x3().determinant()) < 1e-12:\n"
            "        errors.append(f'{name} has a singular transform')\n"
            "for name in expected_roots:\n"
            "    obj = bpy.data.objects.get(name)\n"
            "    if obj is None:\n"
            "        errors.append(f'missing required root surface {name}')\n"
            "        continue\n"
            "    if scene_objects.get(name) is not obj:\n"
            "        errors.append(f'required root surface {name} is not linked to the "
            "active scene')\n"
            "    elif not has_render_collection_path(obj):\n"
            "        errors.append(f'required root surface {name} has no render-enabled "
            "collection path')\n"
            "    if obj.hide_render:\n"
            "        errors.append(f'required root surface {name} has hide_render enabled')\n"
            "    if hasattr(obj, 'visible_camera') and not obj.visible_camera:\n"
            "        errors.append(f'required root surface {name} has camera visibility disabled')\n"
            "    if obj.type not in renderable_root_types:\n"
            "        errors.append(f'root surface {name} is non-renderable type {obj.type}')\n"
            "    if obj.parent is not None:\n"
            "        errors.append(f'root surface {name} is parented under {obj.parent.name}')\n"
            "    if any(c.type == 'CHILD_OF' for c in obj.constraints):\n"
            "        errors.append(f'root surface {name} has a CHILD_OF constraint')\n"
            "    values = [v for row in obj.matrix_world for v in row]\n"
            "    if not all(math.isfinite(v) for v in values):\n"
            "        errors.append(f'root surface {name} has a non-finite transform')\n"
            "    if abs(obj.matrix_world.to_3x3().determinant()) < 1e-12:\n"
            "        errors.append(f'root surface {name} has a singular transform')\n"
            "report = {'status': 'valid' if not errors else 'invalid', "
            "'errors': errors, 'expected_objects': expected, "
            "'expected_root_surfaces': expected_roots, "
            "'pipeline_object_names': pipeline_names, "
            "'camera_count': len(cameras)}\n"
            "with open(report_path, 'w') as stream:\n"
            "    json.dump(report, stream, indent=2)\n"
        )
        try:
            proc = subprocess.run(
                [
                    blender,
                    "--background",
                    str(blend_path),
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(script_path),
                ],
                capture_output=True,
                text=True,
                env={**os.environ, "AL_LIB_LOGLEVEL": "0"},
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "status": "invalid",
                "errors": [f"final Blender validation process failed: {exc}"],
            }
        if proc.returncode != 0 or not report_path.is_file():
            return {
                "status": "invalid",
                "errors": [
                    "final Blender validation process failed: "
                    + (proc.stderr[-2000:] or proc.stdout[-2000:] or "no report")
                ],
            }
        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "status": "invalid",
                "errors": [f"validation report invalid: {exc}"],
            }
        try:
            from lib.tools.geometry.inventory_contract import (
                validate_blender_object_names,
            )

            validate_blender_object_names(
                expected_by_id, report.get("pipeline_object_names", [])
            )
        except Exception as exc:  # noqa: BLE001 - reported as artifact invalidity
            report.setdefault("errors", []).append(str(exc))
            report["status"] = "invalid"
        return report

    def _export_final_result(
        self, *, blend_before: Optional[tuple] = None
    ) -> dict[str, Any]:
        """Persist and validate the best-effort Blender reconstruction.

        ``blend_before`` is the shared blend's fingerprint from before the agent
        stages. An unchanged scene is logged as a quality warning, then judged on the
        same exact structural/render checks as any edited artifact. Stage approval and
        physics certification are likewise quality outcomes, not Blender
        artifact-integrity failures. Isaac conversion is deliberately absent and
        reported as a separate not-run step.
        """
        output_dir = Path(self.args.get("output_dir", "output")) / "final"
        output_dir.mkdir(parents=True, exist_ok=True)

        blend_file = self.args.get("blender_save") or self.args.get("blender_file")
        final_blend = output_dir / "final.blend"
        skip = sorted(set(self.args.get("skip_stages") or []))
        unapproved = self._unapproved_stages(set(skip))
        stage_outcomes: dict[str, dict[str, Any]] = {}
        expected = [(0, self.INITIALIZATION_STAGE), *enumerate(self.LOOP_STAGES, 1)]
        artifacts = self.stage_context.get("stage_artifacts") or {}
        for stage_idx, stage in expected:
            if stage.name in skip:
                stage_outcomes[stage.name] = {
                    "status": "skipped",
                    "approved": None,
                    "quality_impact": "review_recommended",
                }
                continue
            result = (artifacts.get(str(stage_idx)) or {}).get(stage.name)
            stage_outcomes[stage.name] = {
                "status": (
                    "approved"
                    if isinstance(result, dict) and result.get("approved") is True
                    else "not_approved"
                ),
                "approved": result.get("approved")
                if isinstance(result, dict)
                else None,
                "completion_status": (
                    result.get("completion_status")
                    if isinstance(result, dict)
                    else None
                ),
                "generator_termination": (
                    result.get("termination_reason")
                    if isinstance(result, dict)
                    else None
                ),
                "quality_impact": (
                    "none"
                    if isinstance(result, dict) and result.get("approved") is True
                    else "review_recommended"
                ),
            }
        warnings: list[dict[str, Any]] = []
        for stage_name, outcome in stage_outcomes.items():
            if outcome["status"] != "approved":
                warnings.append(
                    {
                        "code": "stage_" + outcome["status"],
                        "stage": stage_name,
                        "message": (
                            f"{stage_name} was {outcome['status']}; the best-effort "
                            "Blender artifact remains usable but should be reviewed"
                        ),
                    }
                )
        certification = self.stage_context.get("composition_certification") or {
            "status": "not_run"
        }
        if certification.get("status") not in {"passed", "skipped"}:
            warnings.append(
                {
                    "code": "physics_certification_"
                    + str(certification.get("status") or "unknown"),
                    "stage": "composition",
                    "message": certification.get("warning")
                    or "final Blender free-settle needs review",
                }
            )
        warnings.extend(self._repair_round_warnings(certification))
        physics_manifest_path = (
            Path(self.args["moge_dir"]) / "physics/physics_estimate_manifest.json"
            if self.args.get("moge_dir")
            else None
        )
        physics_summary = None
        if physics_manifest_path and physics_manifest_path.is_file():
            try:
                physics_payload = json.loads(physics_manifest_path.read_text())
                physics_summary = {
                    key: physics_payload.get(key)
                    for key in (
                        "schema_version",
                        "model",
                        "coverage_status",
                        "expected_count",
                        "estimated_count",
                        "fallback_count",
                    )
                }
                if physics_payload.get("fallback_count", 0):
                    warnings.append(
                        {
                            "code": "physics_estimate_partial",
                            "stage": "preprocess",
                            "message": (
                                f"physics estimates defaulted for "
                                f"{physics_payload.get('fallback_count')} of "
                                f"{physics_payload.get('expected_count')} objects"
                            ),
                        }
                    )
            except (OSError, json.JSONDecodeError) as exc:
                warnings.append(
                    {
                        "code": "physics_estimate_manifest_invalid",
                        "stage": "preprocess",
                        "message": f"physics estimate provenance is unreadable: {exc}",
                    }
                )

        provenance = blender_provenance(self.args, Path(__file__).resolve().parents[2])
        harness_profile = resolve_harness_profile(self.harness_profile["name"])
        provenance.update(
            {
                "harness_profile": harness_profile,
                "source_image_sha256": _file_sha256(self.args.get("target_image_path")),
                "preprocess_manifest": (
                    str(Path(self.args.get("moge_dir")) / "preprocess_manifest.json")
                    if self.args.get("moge_dir")
                    else None
                ),
                "base_run_manifest": self.args.get("base_run_manifest"),
                "physics_estimate_manifest": (
                    str(physics_manifest_path) if physics_manifest_path else None
                ),
                "physics_estimate_summary": physics_summary,
            }
        )

        manifest: dict[str, Any] = {
            "schema_version": 3,
            "status": "exported",
            "execution_status": "completed",
            "artifact_status": "valid",
            "quality_status": "review_recommended" if warnings else "approved",
            # Backward-compatible alias with corrected semantics: this now means
            # the requested Blender execution produced a usable artifact, not that
            # every stage or a later simulator conversion passed.
            "pipeline_complete": True,
            "harness_profile": resolve_harness_profile(self.harness_profile["name"]),
            "blend_file": str(final_blend),
            "source_blend_file": self.args.get("input_blender_file") or blend_file,
            "live_blend_file": blend_file,
            "render_path": None,
            "render_error": None,
            "blend_changed": None,
            "blend_validation": {"status": "not_run", "errors": []},
            "render_validation": {"status": "not_run", "errors": []},
            "requested_stages": [stage.name for _, stage in expected],
            "skipped_stages": skip,
            "stage_outcomes": stage_outcomes,
            "unapproved_stages": unapproved,
            "warnings": warnings,
            "composition_certification": certification,
            "isaac_conversion": {
                "status": "not_run",
                "manifest": None,
                "message": "Isaac conversion and dynamic verification are a separate step",
            },
            "provenance": provenance,
            "stage_artifacts": self._stage_artifacts_for_prompt(),
            # the on-disk manifest keeps the FULL verifier record (post-hoc
            # analysis); only the prompt path is compacted
            "latest_verifier_history": {
                si: {sn: h[-1] for sn, h in sh.items() if h}
                for si, sh in self.stage_context.get("verifier_history", {}).items()
            },
        }

        if blend_file and os.path.exists(blend_file):
            shutil.copy2(blend_file, final_blend)
            if (
                blend_before is not None
                and _blend_fingerprint(blend_file) == blend_before
            ):
                manifest["blend_changed"] = False
                warning = {
                    "code": "blend_unmodified",
                    "stage": None,
                    "message": (
                        "no agent edit changed the shared blend; the exact delivered "
                        "artifact is still judged by structural and render validation"
                    ),
                }
                manifest["warnings"].append(warning)
                manifest["quality_status"] = "review_recommended"
                logger.warning("Shared blend was unchanged; retaining as a warning")
            elif blend_before is not None:
                manifest["blend_changed"] = True
        else:
            manifest["status"] = "blend_missing"
            manifest["artifact_status"] = "invalid"
            manifest["execution_status"] = "failed"
            manifest["pipeline_complete"] = False
            manifest["render_error"] = f"Source blend file missing: {blend_file}"

        if final_blend.exists():
            validation = self._validate_final_blend(final_blend)
            manifest["blend_validation"] = validation
            if (
                validation.get("status") != "valid"
                and manifest["artifact_status"] == "valid"
            ):
                manifest["status"] = "blend_invalid"
                manifest["artifact_status"] = "invalid"
                manifest["execution_status"] = "failed"
                manifest["pipeline_complete"] = False
                manifest["render_error"] = "; ".join(
                    str(error) for error in validation.get("errors") or []
                )

        if final_blend.exists():
            original_blender_file = self.args.get("blender_file")
            original_blender_save = self.args.get("blender_save")
            self.args["blender_file"] = str(final_blend)
            self.args["blender_save"] = str(final_blend)
            try:
                render_path = self._render_final_cycles_target_resolution(
                    output_subdir="final",
                    filename="final_render.png",
                )
                manifest["render_path"] = render_path
                render_validation = _validate_render_integrity(
                    render_path, self.args.get("target_image_path")
                )
                manifest["render_validation"] = render_validation
                manifest["render_luminance"] = render_validation.get("mean_luminance")
                if (
                    render_validation.get("status") != "valid"
                    and manifest["artifact_status"] == "valid"
                ):
                    manifest["status"] = render_validation["status"]
                    manifest["artifact_status"] = "invalid"
                    manifest["execution_status"] = "failed"
                    manifest["pipeline_complete"] = False
                    manifest["render_error"] = "; ".join(
                        render_validation.get("errors") or []
                    )
                    logger.error(
                        "Final render failed integrity validation: %s",
                        manifest["render_error"],
                    )
            except Exception as exc:
                if manifest["artifact_status"] == "valid":
                    manifest["status"] = "render_failed"
                    manifest["artifact_status"] = "invalid"
                    manifest["execution_status"] = "failed"
                    manifest["pipeline_complete"] = False
                    manifest["render_error"] = str(exc)
                logger.warning("Final render export failed: %s", exc)
            finally:
                self.args["blender_file"] = original_blender_file
                self.args["blender_save"] = original_blender_save

        manifest["outputs"] = {
            "final_blend": {"path": str(final_blend)},
            "final_render": {"path": manifest.get("render_path")},
        }
        if (
            harness_profile["name"] == "gpt6_v1"
            and harness_profile["capabilities"].get("runtime_object_inventory", False)
            and self.args.get("moge_dir")
        ):
            manifest["outputs"].update(
                _collect_gpt6_scene_artifact_outputs(Path(self.args["moge_dir"]))
            )

        manifest_path = output_dir / "pipeline_result.json"
        manifest["manifest_path"] = str(manifest_path)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\n=== Final blend export ===\n{final_blend}\n")
        if manifest.get("render_path"):
            print(f"=== Final render export ===\n{manifest['render_path']}\n")
        else:
            print(
                f"=== Final render export failed ===\n{manifest.get('render_error')}\n"
            )
        logger.info("Final result manifest written to %s", manifest_path)
        return manifest

    async def cleanup(self) -> None:
        """Cleanup hook for symmetry with the old main orchestration."""
        return None
