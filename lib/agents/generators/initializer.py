"""Initializer/planner generator agent scaffold."""

from typing import Any

from lib.agents.generator import GeneratorAgent


class InitializerPlannerAgent(GeneratorAgent):
    """Live initializer-stage generator that plans and constructs the scene surfaces."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "generator_agent_name": "InitializerPlannerAgent",
                "generator_prompt_agent_type": "initializer_generator",
                "generator_memory_filename": "initializer_generator_memory.json",
            }
        )
        super().__init__(config)
