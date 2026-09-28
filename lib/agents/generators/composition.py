"""Composition generator agent scaffold."""

from typing import Any

from lib.agents.generator import GeneratorAgent


class CompositionAgent(GeneratorAgent):
    """Live composition-stage generator for corrective layout and pose refinement."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "generator_agent_name": "CompositionAgent",
                "generator_prompt_agent_type": "composition_generator",
                "generator_memory_filename": "composition_generator_memory.json",
            }
        )
        super().__init__(config)
