"""Texture verifier agent scaffold."""

from typing import Any

from lib.agents.verifier import VerifierAgent


class TextureVerifierAgent(VerifierAgent):
    """Verifier copy paired with TextureAgent."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "verifier_agent_name": "TextureVerifierAgent",
                "verifier_prompt_agent_type": "texture_verifier",
                "verifier_memory_filename": "texture_verifier_memory.json",
            }
        )
        super().__init__(config)
