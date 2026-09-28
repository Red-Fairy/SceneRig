#!/usr/bin/env python3
"""Main entry for the staged static-scene agent (generator/verifier per stage).

Single-image 3D scene reconstruction. Uses MCP stdio connections instead of HTTP
servers.
"""

import argparse
import asyncio
import logging
import os
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from lib.agents.root import (
    DEFAULT_HARNESS_PROFILE,
    HARNESS_PROFILE_NAMES,
    RootSceneAgent,
    resolve_harness_profile,
)
from lib.agents.tool_defaults import (
    GENERATOR_TOOLS_DEFAULT,
    VERIFIER_TOOLS_DEFAULT,
    split_tool_scripts,
)
from lib.utils.cli import positive_int
from lib.utils.common import get_model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_model_credentials(args: dict) -> None:
    """Fill provider credentials only after the selected model is known."""
    if not args.get("api_key") or not args.get("api_base_url"):
        model_info = get_model_info(args["model"])
        args["api_key"] = args.get("api_key") or model_info["api_key"]
        args["api_base_url"] = args.get("api_base_url") or model_info["base_url"]


def _validate_args(parser: argparse.ArgumentParser, args: dict) -> None:
    """Fail before model/tool startup on invalid production inputs."""
    positive = ["verifier_max_rounds", "max_stage_attempts"]
    positive.extend(
        key
        for key in args
        if key.endswith(("_generator_max_rounds", "_verifier_max_rounds"))
    )
    for key in positive:
        value = args.get(key)
        if value is not None and int(value) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be > 0")
    for key in ("target_image_path", "moge_dir", "output_dir", "blender_file"):
        if not args.get(key):
            parser.error(f"--{key.replace('_', '-')} is required")
    for key in ("target_image_path", "moge_dir", "blender_file", "blender_script"):
        value = args.get(key)
        if value and not Path(value).exists():
            parser.error(f"--{key.replace('_', '-')} does not exist: {value}")
    for key in ("generator_tools", "verifier_tools"):
        try:
            scripts = split_tool_scripts(args[key])
        except ValueError as exc:
            parser.error(f"--{key.replace('_', '-')}: {exc}")
        for script in scripts:
            if not Path(script).is_file():
                parser.error(
                    f"--{key.replace('_', '-')} script is not an existing regular "
                    f"file: {script}"
                )
    blender_command = args.get("blender_command")
    blender_path = Path(str(blender_command)) if blender_command else None
    if not blender_command or (
        shutil.which(str(blender_command)) is None
        and not (
            blender_path and blender_path.is_file() and os.access(blender_path, os.X_OK)
        )
    ):
        parser.error(f"--blender-command is not executable/found: {blender_command}")


def _prepare_live_blend(parser: argparse.ArgumentParser, args: dict) -> None:
    """Initialize a distinct save path once, then make it the live stage input.

    Production passes the same post-preprocess path for input and save. Direct entry
    may use a distinct delivered accumulator; without this handoff every MCP process
    would reopen the immutable input and discard prior stages. Existing distinct save
    paths are rejected to avoid silently overwriting a prior run.
    """
    source = Path(args["blender_file"]).resolve()
    save_value = args.get("blender_save")
    args["input_blender_file"] = str(source)
    if not save_value:
        return
    live = Path(save_value).resolve()
    if live != source:
        if live.exists():
            parser.error(
                "--blender-save already exists and differs from --blender-file; "
                f"choose a new output path: {live}"
            )
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, live)
    args["blender_file"] = str(live)
    args["blender_save"] = str(live)


async def main() -> int:
    """Run the dual-agent interactive framework."""
    os.environ.setdefault("GRASE_RUN_OWNER_PID", str(os.getpid()))
    parser = argparse.ArgumentParser(description="Staged static-scene agent")
    parser.add_argument("--model", default="claude-opus-5", help="VLM model")
    parser.add_argument(
        "--harness-profile",
        choices=HARNESS_PROFILE_NAMES,
        default=DEFAULT_HARNESS_PROFILE,
        help="Versioned agent/tool policy profile. baseline preserves current behavior; "
        "gpt6_v1 opts into transactional object repair and code-first pose editing with physics after every edit.",
    )
    parser.add_argument(
        "--preprocess-model",
        default=None,
        help="Model used for preprocessing-time physics/material estimation. Defaults "
        "to --model for standalone main.py calls.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. By default credentials are resolved internally from --model.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="API base URL override. By default it is resolved from --model.",
    )
    parser.add_argument(
        "--verifier-max-rounds",
        type=positive_int,
        default=8,
        help="Max interaction (tool-call) rounds per verifier agent in staged generation "
        "(applies to every stage: verifiers carry no per-stage StageSpec override)",
    )
    # Composition has no verifier loop. Generator defaults live in each StageSpec;
    # these explicit per-stage flags are the only generator budget overrides.
    for _stage, _role in (
        ("init", "generator"),
        ("init", "verifier"),
        ("texture", "generator"),
        ("texture", "verifier"),
        ("composition", "generator"),
        ("lighting", "generator"),
        ("lighting", "verifier"),
    ):
        parser.add_argument(
            f"--{_stage}-{_role}-max-rounds",
            type=positive_int,
            default=None,
            help=f"Max tool-call rounds for the {_stage} {_role}",
        )
    parser.add_argument(
        "--init-code-path", default=None, help="Path to initial code file"
    )
    parser.add_argument(
        "--init-image-path", default=None, help="Path to initial images"
    )
    parser.add_argument(
        "--target-image-path", default=None, help="Path to target images"
    )
    # ML Rules CLI override: retain the existing argparse interface and its nargs="*"
    # empty-list semantics; a CLI-framework migration is outside this scoped change.
    parser.add_argument(
        "--ignore_objects",
        nargs="*",
        default=[],
        help="Object categories intentionally excluded from reconstruction and visual "
        "review in every agent stage. The runner forwards its effective ignore list; "
        "standalone runs default to no exclusions.",
    )
    parser.add_argument(
        "--moge-dir",
        default=None,
        help="Dir with MoGE preprocessing (moge/, masks/, placement.json) for "
        "metric initializer placement; camera is locked to the source view",
    )
    parser.add_argument("--target-description", default=None, help="Target description")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--task-name", default=None, help="Task name for hints extraction"
    )
    parser.add_argument(
        "--gpu-devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES"),
        help="GPU devices for Blender",
    )
    parser.add_argument("--clear-memory", action="store_true", help="Clear memory")
    parser.add_argument(
        "--max-stage-attempts",
        type=positive_int,
        default=3,
        help="Maximum verifier-gated attempts per root stage",
    )

    # Execution parameters
    parser.add_argument(
        "--blender-command",
        default="lib/utils/third_party/blender-4.5/blender",
        help="Blender command path",
    )
    parser.add_argument("--blender-file", default=None, help="Blender template file")
    parser.add_argument(
        "--blender-script",
        default="data/static_scene/generator_script.py",
        help="Blender execution script",
    )
    parser.add_argument("--blender-save", default=None, help="Save blender file")

    # Tool servers
    parser.add_argument(
        "--generator-tools",
        default=GENERATOR_TOOLS_DEFAULT,
        help="Comma-separated list of generator tool server scripts",
    )
    parser.add_argument(
        "--verifier-tools",
        default=VERIFIER_TOOLS_DEFAULT,
        help="Comma-separated list of verifier tool server scripts",
    )

    args = vars(parser.parse_args())
    args["backend"] = "sam3d"
    args["base_run_manifest"] = None
    args["investigate_cap"] = 3
    args["memory_window"] = 5
    args["skip_stages"] = []
    # Resolve once before any model/tool process starts. RootSceneAgent repeats this
    # canonicalization for programmatic callers that bypass the CLI.
    args["harness_profile_manifest"] = resolve_harness_profile(args["harness_profile"])
    _validate_args(parser, args)
    _prepare_live_blend(parser, args)
    _resolve_model_credentials(args)

    try:
        logger.info("Initializing root scene agent")
        root_agent = RootSceneAgent(args)
        result = await root_agent.run()
        if result.get("complete") is not True:
            logger.error(
                "Pipeline produced a broken output (export_status=%s); "
                "unapproved stages were %s",
                result.get("export_status"),
                result.get("unapproved_stages"),
            )
            return 2
        return 0
    except Exception:
        logger.exception("Error during execution")
        output_dir = args.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "fatal_error.log"), "a") as f:
                f.write(traceback.format_exc())
                f.write("\n" + "=" * 80 + "\n")
        raise


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception:
        logger.exception("Fatal error")
        sys.exit(1)
