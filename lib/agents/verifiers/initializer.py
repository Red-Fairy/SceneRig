"""Initializer/planner verifier agent scaffold."""

from typing import Any

from lib.agents.verifier import VerifierAgent


class InitializerPlannerVerifierAgent(VerifierAgent):
    """Verifier copy paired with InitializerPlannerAgent."""

    def __init__(self, args: dict[str, Any]) -> None:
        config = dict(args)
        config.update(
            {
                "verifier_agent_name": "InitializerPlannerVerifierAgent",
                "verifier_prompt_agent_type": "initializer_verifier",
                "verifier_memory_filename": "initializer_verifier_memory.json",
            }
        )
        super().__init__(config)
