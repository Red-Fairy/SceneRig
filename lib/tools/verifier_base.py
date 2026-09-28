"""Verifier Base MCP Server.

Provides the base MCP server for the Verifier agent with an 'end' tool
to output visual difference analysis and edit suggestions.
"""

from typing import Optional

from mcp.server.fastmcp import FastMCP

from lib.tools.geometry.yaw_advisory import (
    MAX_ID_LENGTH,
    MAX_REVIEW_READING_LENGTH,
)

mcp = FastMCP("verifier-base")

# Tool configurations for the agent
tool_configs: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "end",
            "description": "If you think your observations are sufficient, call the tool to end the process and output the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "visual_difference": {
                        "type": "string",
                        "description": "Describe ONLY the differences that fall within THIS stage's scope (defined in your system prompt) between the current scene and the target/reference image. Be concrete and observational: name the specific in-scope surfaces or regions that differ and, where useful, give an image-space direction and magnitude (e.g. “the in-scope rendered region's right edge is about 8% of the frame farther image-right than in the reference”, “the wall material is flat grey but the reference shows wood grain”). Do NOT prescribe an object transform or world-axis correction here; the stage prompt defines which remedies are permitted. Do NOT raise issues outside this stage's scope — that is another stage's responsibility.",
                    },
                    "edit_suggestion": {
                        "type": "string",
                        "description": "Optional edit suggestion for the current scene, referring to the visual difference. Leave empty ('') when approving — there is nothing to fix.",
                    },
                    "approved": {
                        "type": "boolean",
                        "description": "Whether this stage result is good enough to move on. Return false if the generator must revise this same stage before the pipeline advances.",
                    },
                    "approval_checklist": {
                        "type": "array",
                        "description": "Persistent checklist of concrete requirements that must be satisfied before this stage can advance. If you reject, list the exact items the generator must fix next. If there is a prior checklist, keep fulfilled items marked as done and keep unresolved items marked as pending.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Short stable identifier, e.g. geometry_missing_part or material_visibility.",
                                },
                                "requirement": {
                                    "type": "string",
                                    "description": "Concrete requirement that can be verified later.",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "done"],
                                    "description": "Whether the current scene satisfies this item.",
                                },
                                "evidence": {
                                    "type": "string",
                                    "description": "Brief evidence from render, scene info, or investigation.",
                                },
                            },
                            "required": ["id", "requirement", "status", "evidence"],
                        },
                    },
                    "problem_images": {
                        "type": "array",
                        "description": "Images that best show the blocking issue(s). Include local paths from the prompt or investigation tool outputs and a brief note about what to inspect in each image.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Local image path that shows the issue.",
                                },
                                "note": {
                                    "type": "string",
                                    "description": "What specific issue is visible in this image.",
                                },
                            },
                            "required": ["path", "note"],
                        },
                    },
                    "suggested_fixes": {
                        "type": "array",
                        "description": "Concrete implementation suggestions that could satisfy pending checklist items. These are guidance, not approval criteria.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "issue_id": {
                                    "type": "string",
                                    "description": "Checklist id or short issue id this fix addresses.",
                                },
                                "fix_summary": {
                                    "type": "string",
                                    "description": "Short summary of the proposed fix.",
                                },
                                "implementation_hint": {
                                    "type": "string",
                                    "description": "Concrete coding/modeling hint for the generator.",
                                },
                                "affected_objects": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Object names or scene graph nodes affected by this fix.",
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": "Fix priority.",
                                },
                            },
                            "required": [
                                "issue_id",
                                "fix_summary",
                                "implementation_hint",
                                "affected_objects",
                                "priority",
                            ],
                        },
                    },
                    "regression_check": {
                        "type": "string",
                        "description": "State whether anything outside the checklist became worse compared with the previous verified attempt. If approving, explicitly confirm there are no meaningful regressions.",
                    },
                    "yaw_advisory_review": {
                        "type": "object",
                        "description": (
                            "Initializer only, and mandatory exactly when the current "
                            "structured gate summary says yaw_advisory.requirement.state="
                            "manual_review. Use the exact advisory_id; prose elsewhere "
                            "does not satisfy this row."
                        ),
                        "properties": {
                            "schema_version": {"type": "integer", "enum": [1]},
                            "advisory_id": {
                                "type": "string",
                                "maxLength": MAX_ID_LENGTH,
                            },
                            "decision": {
                                "type": "string",
                                "enum": ["acceptable", "mismatch", "unverified"],
                            },
                            "evidence": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "anchor_type": {
                                            "type": "string",
                                            "enum": [
                                                "edge",
                                                "corner",
                                                "mask_boundary",
                                                "object_gap",
                                            ],
                                        },
                                        "target_reading": {
                                            "type": "string",
                                            "maxLength": MAX_REVIEW_READING_LENGTH,
                                        },
                                        "render_reading": {
                                            "type": "string",
                                            "maxLength": MAX_REVIEW_READING_LENGTH,
                                        },
                                    },
                                    "required": [
                                        "anchor_type",
                                        "target_reading",
                                        "render_reading",
                                    ],
                                    "additionalProperties": False,
                                },
                            },
                            "reason_codes": {
                                "type": "array",
                                "maxItems": 12,
                                "items": {
                                    "type": "string",
                                    "maxLength": MAX_ID_LENGTH,
                                },
                            },
                        },
                        "required": [
                            "schema_version",
                            "advisory_id",
                            "decision",
                            "evidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                # edit_suggestion is NOT required: it has no natural content when
                # APPROVING, so the model reliably omits it there — and a required
                # field it drops made pydantic reject the whole `end` call, losing
                # the real verdict and defaulting to not-approved (0715 abc4
                # composition spuriously rejected this way). Same for the other
                # narrative/list fields, which already default in the handler.
                "required": [
                    "visual_difference",
                    "approved",
                ],
            },
        },
    },
]


@mcp.tool()
def end(
    visual_difference: str,
    approved: bool,
    edit_suggestion: str = "",
    approval_checklist: Optional[list[dict[str, str]]] = None,
    regression_check: str = "",
    problem_images: Optional[list[dict[str, str]]] = None,
    suggested_fixes: Optional[list[dict[str, object]]] = None,
    yaw_advisory_review: Optional[dict[str, object]] = None,
) -> dict[str, object]:
    """End verification and output analysis results."""
    checklist = approval_checklist or []
    problem_images = problem_images or []
    suggested_fixes = suggested_fixes or []
    checklist_text = "\n".join(
        f"- [{item.get('status', 'pending')}] {item.get('id', 'item')}: {item.get('requirement', '')} Evidence: {item.get('evidence', '')}"
        for item in checklist
    )
    problem_images_text = "\n".join(
        f"- {item.get('path', '')}: {item.get('note', '')}" for item in problem_images
    )
    suggested_fixes_text = "\n".join(
        f"- [{item.get('priority', 'medium')}] {item.get('issue_id', 'issue')}: {item.get('fix_summary', '')} Hint: {item.get('implementation_hint', '')} Objects: {', '.join(item.get('affected_objects', []) or [])}"
        for item in suggested_fixes
    )
    return {
        "status": "success",
        "output": {
            "approved": approved,
            "visual_difference": visual_difference,
            "edit_suggestion": edit_suggestion,
            "approval_checklist": checklist,
            "problem_images": problem_images,
            "suggested_fixes": suggested_fixes,
            "regression_check": regression_check,
            **(
                {"yaw_advisory_review": yaw_advisory_review}
                if yaw_advisory_review is not None
                else {}
            ),
            "text": [
                f"Approved: {approved}\n"
                f"Visual difference: {visual_difference}\n"
                f"Approval checklist:\n{checklist_text}\n"
                f"Problem images:\n{problem_images_text}\n"
                f"Suggested fixes:\n{suggested_fixes_text}\n"
                f"Regression check: {regression_check}\n"
                f"Edit suggestion: {edit_suggestion}"
            ],
        },
    }


@mcp.tool()
def initialize(args: dict[str, object]) -> dict[str, object]:
    """Initialize the verifier base."""
    return {
        "status": "success",
        "output": {
            "text": ["Verifier base initialized successfully"],
            "tool_configs": tool_configs,
        },
    }


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
