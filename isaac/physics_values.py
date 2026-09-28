"""Pure precedence helpers shared by Isaac physics stamping and tests."""

from __future__ import annotations


def pick_mass_friction(
    overrides: dict, estimate: dict
) -> tuple[float | None, float | None]:
    """Resolve per-key values: explicit overrides, then VLM estimate, then caller default."""
    mass = (
        float(overrides["mass"])
        if "mass" in overrides
        else estimate.get("mass_kg")
    )
    friction = (
        float(overrides["friction"])
        if "friction" in overrides
        else estimate.get("friction")
    )
    return mass, friction
