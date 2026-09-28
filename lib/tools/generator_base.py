"""Generator Base MCP Server.

Provides the base MCP server for the Generator agent with an 'end' tool
to signal process completion.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("generator-base")

# Tool configurations for the agent
tool_configs: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": (
                "No-op tool used to voluntarily declare this generator stage complete. "
                "Call it only when the stage system prompt's completion conditions and "
                "any required rules gate are satisfied. Round-budget exhaustion stops an "
                "incomplete attempt automatically; it never authorizes this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "yaw_advisory_resolution": {
                        "type": ["object", "null"],
                        "description": (
                            "Initializer only: set null unless resolving the current "
                            "required advisory_requirement. Otherwise copy its ID-only "
                            "resolution fields. The agent routes "
                            "it to the backend before end; never add coordinates, angles, "
                            "confidence, tolerance, or your own pass/fail conclusion."
                        ),
                        "properties": {
                            "advisory_id": {"type": "string"},
                            "observation_token": {"type": "string"},
                            "method": {
                                "type": "string",
                                "enum": ["two_edge_families"],
                            },
                            "edge_matches": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 2,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_edge_id": {"type": "string"},
                                        "built_edge_id": {"type": "string"},
                                    },
                                    "required": [
                                        "target_edge_id",
                                        "built_edge_id",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": [
                            "advisory_id",
                            "observation_token",
                            "method",
                            "edge_matches",
                        ],
                        "additionalProperties": False,
                    }
                },
                "required": ["yaw_advisory_resolution"],
                "additionalProperties": False,
            },
        },
    }
]


@mcp.tool()
def initialize(args: dict[str, object]) -> dict[str, Any]:
    """Expose the end schema for this generator stage.

    Args:
        args: Stage configuration; only ``root_stage_name="initializer"`` enables
            the nullable embedded yaw submission.

    Returns:
        MCP initialization output with an independent stage-specific tool schema.
    """
    stage_tool_configs = deepcopy(tool_configs)
    if args.get("root_stage_name") != "initializer":
        stage_tool_configs[0]["function"]["parameters"] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    return {
        "status": "success",
        "output": {
            "text": ["Generator base initialized successfully"],
            "tool_configs": stage_tool_configs,
        },
    }


@mcp.tool()
def end() -> dict[str, object]:
    """Signal that the generation process should end."""
    return {"status": "success", "output": {"text": ["END THE PROCESS"]}}


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
