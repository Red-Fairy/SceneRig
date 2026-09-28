"""Machine-local executable paths with environment-variable overrides."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "lib" / "utils" / "third_party"


def _python_path(env_name: str, relative_default: str) -> str:
    return os.environ.get(env_name, str(REPO_ROOT / relative_default))


SAM3_PY = _python_path("SAM3_PYTHON", "lib/utils/third_party/sam3/.venv/bin/python")
SAM3D_PY = _python_path("SAM3D_PYTHON", "lib/utils/third_party/sam3d/.venv/bin/python")
MOLMO_PY = _python_path("MOLMO_PYTHON", "lib/utils/third_party/molmo/.venv/bin/python")

LANPAINT_QWEN_PY = _python_path(
    "LANPAINT_QWEN_PYTHON",
    "lib/utils/third_party/lanpaint-qwen/.venv/bin/python",
)
SHARP_PY = _python_path("SHARP_PYTHON", "lib/utils/third_party/sharp/.venv/bin/python")

class _ToolCommandMap(dict):
    def __missing__(self, key: str) -> str:
        return sys.executable


path_to_cmd = _ToolCommandMap(
    {
        "lib/tools/blender/exec.py": sys.executable,
        "lib/tools/blender/investigator.py": sys.executable,
        "lib/tools/generator_base.py": sys.executable,
        "lib/tools/initialize_plan.py": sys.executable,
        "lib/tools/verifier_base.py": sys.executable,
    }
)
