"""Lighting generator agent scaffold."""

from typing import Any

from lib.agents.generator import GeneratorAgent


class LightingAgent(GeneratorAgent):
    """Live lighting-stage generator for target-matching illumination edits."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "generator_agent_name": "LightingAgent",
                "generator_prompt_agent_type": "lighting_generator",
                "generator_memory_filename": "lighting_generator_memory.json",
            }
        )
        super().__init__(config)
