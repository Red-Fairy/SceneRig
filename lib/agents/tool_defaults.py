"""Authoritative production MCP tool-server lists for both entrypoints."""

GENERATOR_TOOL_SCRIPTS = (
    "lib/tools/blender/exec.py",
    "lib/tools/generator_base.py",
    "lib/tools/initialize_plan.py",
)
VERIFIER_TOOL_SCRIPTS = (
    "lib/tools/blender/investigator.py",
    "lib/tools/verifier_base.py",
)

GENERATOR_TOOLS_DEFAULT = ",".join(GENERATOR_TOOL_SCRIPTS)
VERIFIER_TOOLS_DEFAULT = ",".join(VERIFIER_TOOL_SCRIPTS)


def split_tool_scripts(value: str) -> list[str]:
    """Trim a comma-separated override and reject empty path entries."""
    scripts = [part.strip() for part in value.split(",")]
    if not scripts or any(not script for script in scripts):
        raise ValueError("tool script list contains an empty path")
    return scripts
