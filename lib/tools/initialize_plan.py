"""Initialize Plan MCP Server.

Stores the detailed scene plan that guides the Generator agent's subsequent
actions. Only the static_scene initializer stage plans; every other stage edits
an already-built scene and gets no plan tool.
"""

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Tool parameter descriptions.
#
# These are the long instruction strings the model reads as the tool schema.
# They are kept here as module-level constants (one logical line per physical
# line, implicitly concatenated) so they stay short and easy to edit. Editing
# the wording below changes what the planner is told to produce.
# ---------------------------------------------------------------------------

# static_scene_plan_tool: detailed_plan parameter. Only the INITIALIZER plans in static_scene
# (objects are already imported + placed), so the plan is ONLY about the root surfaces + lighting
# — never about creating/moving objects. Kept terse to match the trimmed initializer prompt.
_STATIC_SCENE_DETAILED_PLAN = (
    "A SHORT, imperative plan for building the current registered roots + lighting to match the "
    "reference photo. Plan ONLY the surfaces and lights — the objects are already imported "
    "and placed; do NOT plan to create, move, or edit them. Cover: (1) MAIN SUPPORT — "
    "footprint size, in-plane yaw, and xy placement so its flat TOP is LEVEL at z=0 and every "
    "object rests on it; (2) each OTHER current registered root (initially the preprocessed "
    "roots named in the per-scene data, plus any runtime-added root returned later by "
    "build_root_surface) — placed through its given point and perpendicular to its given normal (walls "
    "plumb, floor level), sized to fill the background, colored to contrast the world, "
    "including any WALL STRUCTURE worth building on each current registered wall (window/door "
    "openings and wall_N_<part> detail children). Build every current registered root exactly "
    "once; execute_and_evaluate may create or edit any such root. If the target clearly "
    "contains a DISTINCT architectural bounding wall plane absent from the current registered "
    "roots, include one short conditional build_root_surface step; do not assign its ID or "
    "plan a direct graph edit. Do not use build_root_surface for an already registered root, "
    "an extension/repositioning, an opening, or a detail. After it returns, refine that "
    "runtime-added root with execute_and_evaluate. If a runtime-added root later proves "
    "unnecessary, only remove_root_surface may remove it; never plan removal of a preprocessed root. "
    "(3) LIGHTING — one or two soft area lights + a neutral world at moderate strength. "
    "Keep it terse; you will refine against renders afterward."
)

_PLAN_TOOL_DESCRIPTION = (
    "From the given inputs, imagine and articulate the scene in detail. This "
    "tool does not return new information. It stores your detailed description "
    "as your own plan to guide subsequent actions. You must call this tool "
    "first."
)

# Tool configuration for static_scene initializer planning (surfaces + lighting only)
static_scene_plan_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "initialize_plan",
        "description": _PLAN_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "detailed_plan": {
                    "type": "string",
                    "description": _STATIC_SCENE_DETAILED_PLAN,
                }
            },
            "required": ["detailed_plan"],
        },
    },
}

# Create MCP instance
mcp = FastMCP("initialize-plan-executor")


@mcp.tool()
def initialize(args: dict[str, object]) -> dict[str, object]:
    """Initialize the plan tool for the current stage.

    Args:
        args: Configuration dictionary with the 'root_stage_name' key.

    Returns:
        Dictionary with status and the stage's tool configuration.
    """
    # ONLY the initializer plans (surfaces + lighting). The appearance stages
    # (texture/composition/lighting) edit an already-built scene, so they get NO plan
    # tool — otherwise they'd waste a round storing an out-of-scope plan.
    stage = args.get("root_stage_name")
    tool_configs = [static_scene_plan_tool] if stage == "initializer" else []
    return {
        "status": "success",
        "output": {"text": ["Initialize plan completed"], "tool_configs": tool_configs},
    }


@mcp.tool()
def initialize_plan(
    overall_description: str = "", detailed_plan: str = ""
) -> dict[str, object]:
    """Store the detailed scene plan for guiding subsequent actions.

    Args:
        overall_description: Comprehensive description of the entire scene.
        detailed_plan: Step-by-step plan for scene construction.

    Returns:
        Dictionary with the stored plan and success status.
    """
    # No exhortation appended (2026-08-07): the stored text is replayed verbatim as
    # the carried plan on 2nd+ attempts, where the flow instruction is to REVISE the
    # previous plan — a trailing "Please follow the plan carefully." contradicted that.
    output_text = detailed_plan
    return {
        "status": "success",
        "output": {"plan": [output_text], "text": ["Plan initialized successfully"]},
    }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
