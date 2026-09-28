"""System prompts for the static_scene stage agents."""

from .static_scene.generators.composition import (
    composition_generator_system,
    static_scene_composition_generator_system,
)
from .static_scene.generators.initializer import (
    initializer_generator_system,
    static_scene_initializer_generator_system_scene_graph,
)
from .static_scene.generators.lighting import static_scene_lighting_generator_system
from .static_scene.generators.texture import (
    static_scene_texture_generator_system,
    texture_generator_system,
)
from .static_scene.verifiers.initializer import (
    initializer_verifier_system,
    static_scene_initializer_verifier_system_scene_graph,
)
from .static_scene.verifiers.lighting import static_scene_lighting_verifier_system
from .static_scene.verifiers.texture import static_scene_texture_verifier_system

SYSTEM_PROMPTS: dict[str, str] = {
    "initializer_generator": static_scene_initializer_generator_system_scene_graph,
    "texture_generator": static_scene_texture_generator_system,
    "composition_generator": static_scene_composition_generator_system,
    "lighting_generator": static_scene_lighting_generator_system,
    "initializer_verifier": static_scene_initializer_verifier_system_scene_graph,
    "texture_verifier": static_scene_texture_verifier_system,
    "lighting_verifier": static_scene_lighting_verifier_system,
}


def get_system_prompt(
    agent_type: str,
    harness_profile: str | None = None,
    harness_profile_manifest: dict | None = None,
) -> str:
    """Return the system prompt for a stage agent.

    Args:
        agent_type: Registry key, e.g. ``texture_generator``.
        harness_profile: Optional versioned harness name. Omission means baseline,
            unless a supplied manifest declares its name.
        harness_profile_manifest: Resolved profile metadata/capability map.
    """
    if agent_type not in SYSTEM_PROMPTS:
        raise ValueError(f"No system prompt registered for {agent_type!r}")
    manifest = (
        harness_profile_manifest if isinstance(harness_profile_manifest, dict) else None
    )
    selected_profile = harness_profile or (manifest or {}).get("name") or "baseline"
    if agent_type == "initializer_generator":
        return initializer_generator_system(selected_profile, manifest)
    if agent_type == "initializer_verifier":
        return initializer_verifier_system(selected_profile, manifest)
    if agent_type == "composition_generator":
        return composition_generator_system(selected_profile, manifest)
    if agent_type == "texture_generator":
        return texture_generator_system(selected_profile)
    return SYSTEM_PROMPTS[agent_type]


__all__ = ["SYSTEM_PROMPTS", "get_system_prompt"]
