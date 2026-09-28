"""Texture generator agent scaffold."""

from typing import Any

from lib.agents.generator import GeneratorAgent


class TextureAgent(GeneratorAgent):
    """Live texture-stage generator for root-surface materials and textures."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "generator_agent_name": "TextureAgent",
                "generator_prompt_agent_type": "texture_generator",
                "generator_memory_filename": "texture_generator_memory.json",
            }
        )
        super().__init__(config)
