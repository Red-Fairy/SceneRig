#!/usr/bin/env python3
"""Compatibility CLI for canonical preprocessing-time VLM physics estimation.

New runs estimate physics in ``preprocess_scene``. This entry point remains for
operators repairing an older experiment: it invokes the same canonical estimator,
then translates the result to Isaac/USD object names.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from lib.tools.geometry.physics_estimate import (  # noqa: E402
    estimate_scene,
    scene_objects_from_placement,
    translate_to_isaac,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--force", action="store_true", help="re-estimate cached entries")
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.environ.get("GRASE_VLM_PHYSICS_BATCH", "5")),
        help="maximum concurrent VLM calls",
    )
    args = parser.parse_args()
    exp = args.exp_dir if args.exp_dir.is_absolute() else REPO / args.exp_dir
    scene = exp / "scene" if (exp / "scene").exists() else exp
    canonical = scene / "physics" / "physics_vlm.json"

    if args.force or not canonical.exists():
        estimate_scene(
            scene_objects_from_placement(exp),
            canonical,
            model=args.model,
            force=args.force,
            jobs=args.jobs,
        )

    count = translate_to_isaac(exp)
    print(f"VLM_PHYSICS_OK {scene / 'isaac' / 'physics_vlm.json'} objects={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
