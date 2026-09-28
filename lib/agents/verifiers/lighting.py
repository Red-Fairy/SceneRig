"""Lighting verifier agent scaffold."""

from typing import Any

from lib.agents.verifier import VerifierAgent


class LightingVerifierAgent(VerifierAgent):
    """Verifier copy paired with LightingAgent."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "verifier_agent_name": "LightingVerifierAgent",
                "verifier_prompt_agent_type": "lighting_verifier",
                "verifier_memory_filename": "lighting_verifier_memory.json",
            }
        )
        super().__init__(config)
