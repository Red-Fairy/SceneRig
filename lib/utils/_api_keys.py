"""Environment-backed API key configuration.

SceneRig does not commit personal credentials. Export the variables below in your
shell, or let ``scripts/run_e2e.sh`` load them from ``$SCENERIG_KEYS_FILE`` /
``$GRASE_KEYS_FILE`` / ``~/.zshrc``.
"""

from __future__ import annotations

import os


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


OPENAI_API_KEY = _env("OPENAI_API_KEY")
OPENAI_BASE_URL = _env("OPENAI_BASE_URL", "OPENAI_API_BASE", default="https://api.openai.com/v1")

CLAUDE_API_KEY = _env("CLAUDE_API_KEY", "ANTHROPIC_API_KEY")
CLAUDE_BASE_URL = _env("CLAUDE_BASE_URL", "ANTHROPIC_BASE_URL", default="https://api.anthropic.com/v1")

FIREWORKS_API_KEY = _env("FIREWORKS_API_KEY")
FIREWORKS_BASE_URL = _env("FIREWORKS_BASE_URL", default="https://api.fireworks.ai/inference/v1")

GEMINI_API_KEY = _env("GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_BASE_URL = _env("GEMINI_BASE_URL", default="https://generativelanguage.googleapis.com/v1beta/openai")

QWEN_API_KEY = _env("QWEN_API_KEY")
QWEN_BASE_URL = _env("QWEN_BASE_URL")
