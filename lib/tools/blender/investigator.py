"""Investigator MCP Server for 3D Scene Analysis.

Provides tools for camera manipulation, scene inspection, and viewpoint
management in Blender scenes. Used by the Verifier agent to analyze
generated 3D content from multiple angles.
"""

import os
import sys
from typing import Optional

from mcp.server.fastmcp import FastMCP

try:
    from .investigator_core import Investigator3D
except ImportError:
    from investigator_core import Investigator3D

# Tool configuration for agent
render_reference_view_tool: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "render_reference_view",
        "description": (
            "Render the scene from the LOCKED REFERENCE camera — the exact "
            "viewpoint of the target photo. This is the ONLY tool whose output "
            "may be compared against the target photo — for framing, coverage and "
            "pose/yaw, and equally for exposure, contrast and material appearance "
            "(initialize_viewpoint/set_camera/investigate all "
            "use other cameras and can NEVER support those claims). Also resets "
            "the investigation camera to the reference pose. Takes no arguments. "
            "The scene CANNOT change during verification, so repeated calls return "
            "an identical image — call this at most once per attempt, and not at "
            "all if a reference-view render is already attached to your first "
            "message."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

tool_configs: list[dict[str, object]] = [
    render_reference_view_tool,
    {
        "type": "function",
        "function": {
            "name": "initialize_viewpoint",
            "description": "Adds a viewpoint to observe the listed objects. The viewpoints are added to the four corners of the bounding box of the listed objects (room-scale background planes — walls/floors/ceilings — stay visible but are excluded from the framing box, so cameras stay at object scale). This tool returns the positions and rotations of the four viewpoint cameras, as well as the rendered images of the four cameras. These are ALTERNATE views for 3D topology checks — never judge framing, coverage, pose/yaw, exposure or material appearance from them; use them to pick a pose for set_camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_names": {
                        "type": "array",
                        "description": "The names of the objects to observe. Objects must exist in the scene (you can check the scene information to see if they exist). If you want to observe the whole scene, you can pass an empty list.",
                        "items": {
                            "type": "string",
                            "description": "The name of the object to observe.",
                        },
                    }
                },
                "required": ["object_names"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_camera",
            "description": "Set the current active camera to the given location and rotation, then render from it. This is an ALTERNATE camera, never the reference view: use it for 3D topology (which side something is on, floating, penetration), and NEVER to judge framing, coverage, pose/yaw, exposure or material appearance — call render_reference_view for those. It also drops any focus tracking, so a following investigate zoom/move needs a fresh focus first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "array",
                        "description": "The location of the camera (in world coordinates)",
                        "items": {
                            "type": "number",
                            "description": "The location of the camera (in world coordinates)",
                        },
                    },
                    "rotation_euler": {
                        "type": "array",
                        "description": "The rotation of the camera (in euler angles)",
                        "items": {
                            "type": "number",
                            "description": "The rotation of the camera (in euler angles)",
                        },
                    },
                },
                "required": ["location", "rotation_euler"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "investigate",
            "description": "Investigate the scene from an ALTERNATE camera orbiting one object. Call focus FIRST (it sets the target and frames it from the reference distance); zoom and move error out until then, and render_reference_view clears the target so you must re-focus after it. Steps: zoom moves ~40% closer/farther per call (about three calls either side of the focus distance before it stops); move orbits 45 degrees horizontally or 30 degrees vertically per call, so a full circuit is 8 move calls. All three return a non-reference render: use them for 3D topology (which side, floating, penetration, hidden geometry), and NEVER to judge framing, coverage, pose/yaw, exposure or material appearance — call render_reference_view for those.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["zoom", "move", "focus"],
                        "description": "The operation to perform. focus MUST come first — it sets the target object; zoom and move return an error until then, and again after any render_reference_view or set_camera call clears the target.",
                    },
                    "object_name": {
                        "type": "string",
                        "description": "If the operation is focus, you need to provide the name of the object to focus on. The object must exist in the scene.",
                    },
                    "direction": {
                        "type": "string",
                        "enum": ["up", "down", "left", "right", "in", "out"],
                        "description": "If the operation is move or zoom, you need to provide the direction to move or zoom.",
                    },
                },
                "required": ["operation"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_scene_info",
            "description": "Read the CURRENT state of the Blender scene: objects (name, transform, visibility, world-space bbox, material_slots), materials (Principled base_color/roughness/metallic; imported materials explicitly non-editable), lights, cameras, world (background strength/colour), color_management, runtime_capabilities, and diagnostic lighting_material_limits.",
        },
    },
]

# Create MCP instance
mcp = FastMCP("scene-server")

# Global investigator instance
_investigator: Optional[Investigator3D] = None


def _gpt6_pipeline_object_names_resolver(args: dict[str, object]):
    """Return a per-execution authenticated object-name resolver for GPT-6 only."""

    if args.get("harness_profile") != "gpt6_v1":
        return None

    # Imports stay behind the exact opt-in so baseline initialization keeps its
    # historical dependency and artifact-access behavior.
    from lib.agents.harness_profile import resolve_harness_profile

    expected_profile = resolve_harness_profile("gpt6_v1")
    if (
        args.get("harness_profile_manifest") != expected_profile
        or expected_profile["capabilities"].get("runtime_object_inventory") is not True
    ):
        raise ValueError(
            "investigator requires the exact canonical GPT-6 profile manifest"
        )
    scene_dir_value = args.get("moge_dir")
    if not isinstance(scene_dir_value, str) or not scene_dir_value.strip():
        raise ValueError(
            "GPT-6 investigator requires a nonempty scene artifact directory"
        )
    scene_dir = os.path.abspath(scene_dir_value)

    def resolve_current_names() -> list[str]:
        # Do not resolve at server initialization: the generator can commit an authored
        # add/remove after both attempt servers have started but before verifier preseed.
        from lib.tools.geometry.inventory_contract import validate_scene_artifacts

        report = validate_scene_artifacts(
            scene_dir,
            allow_runtime_additions=True,
            allow_runtime_inventory=True,
        )
        return report["expected_blender_object_names"]

    return resolve_current_names


@mcp.tool()
def initialize(args: dict[str, object]) -> dict[str, object]:
    """Initialize the 3D scene investigation tool.

    Args:
        args: Configuration dictionary with 'output_dir', 'blender_file',
            'blender_save' (preferred current scene), 'blender_command',
            'blender_script', 'gpu_devices', and 'render_engine' keys.

    Returns:
        Dictionary with status and tool configurations on success.
    """
    global _investigator
    try:
        save_dir = args.get("output_dir") + "/investigator/"
        blender_script = (
            os.path.dirname(args.get("blender_script")) + "/verifier_script.py"
        )
        # Inspect the CURRENT shared scene (the generator's saved output), not the
        # initial/empty blend — so the verifier sees what the generator actually built.
        scene_blend = args.get("blender_save") or args.get("blender_file")
        pipeline_object_names_resolver = _gpt6_pipeline_object_names_resolver(args)
        _investigator = Investigator3D(
            save_dir,
            str(scene_blend),
            str(args.get("blender_command")),
            blender_script,
            str(args.get("gpu_devices")),
            str(args.get("render_engine") or "CYCLES"),
            pipeline_object_names_resolver=pipeline_object_names_resolver,
        )
        return {
            "status": "success",
            "output": {
                "text": ["Investigator3D initialized successfully"],
                "tool_configs": tool_configs,
            },
        }
    except Exception as e:
        return {"status": "error", "output": {"text": [str(e)]}}


@mcp.tool()
def get_scene_info() -> dict[str, object]:
    """Get information about the current scene.

    Returns:
        Dictionary with scene objects, materials, lights, and cameras.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.get_info()


def focus(object_name: str) -> dict[str, object]:
    """Focus camera on a specific object.

    Args:
        object_name: Name of the object to focus on.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.focus_on_object(object_name)


def zoom(direction: str) -> dict[str, object]:
    """Zoom camera in or out.

    Args:
        direction: Either 'in' or 'out'.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.zoom(direction)


def move(direction: str) -> dict[str, object]:
    """Move camera around target object.

    Args:
        direction: One of 'up', 'down', 'left', 'right'.

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.move_camera(direction)


@mcp.tool()
def initialize_viewpoint(object_names: list[str] = []) -> dict[str, object]:
    """Initialize viewpoints around specified objects.

    Args:
        object_names: List of object names to observe. Empty for all objects.

    Returns:
        Dictionary with camera positions and rendered images.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.initialize_viewpoint(object_names)


@mcp.tool()
def render_reference_view() -> dict[str, object]:
    """Render the scene from the locked reference camera (the target photo's view).

    Returns:
        Dictionary with the reference-view render.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.render_reference_view()


@mcp.tool()
def investigate(
    operation: str = "", object_name: str = "", direction: str = ""
) -> dict[str, object]:
    """Investigate the scene with camera operations.

    Args:
        operation: One of 'focus', 'zoom', or 'move'.
        object_name: Required for 'focus' operation.
        direction: Required for 'zoom' (in/out) or 'move' (up/down/left/right).

    Returns:
        Dictionary with status and rendered image.
    """
    if operation == "focus":
        if not object_name:
            return {
                "status": "error",
                "output": {"text": ["object_name is required for focus"]},
            }
        return focus(object_name=object_name)
    elif operation == "zoom":
        if direction not in ("in", "out"):
            return {
                "status": "error",
                "output": {"text": ["direction must be 'in' or 'out' for zoom"]},
            }
        return zoom(direction=direction)
    elif operation == "move":
        if direction not in ("up", "down", "left", "right"):
            return {
                "status": "error",
                "output": {"text": ["direction must be up/down/left/right for move"]},
            }
        return move(direction=direction)
    else:
        return {
            "status": "error",
            "output": {"text": [f"Unknown operation: {operation}"]},
        }


@mcp.tool()
def set_camera(
    location: list[float] = [0, 0, 0], rotation_euler: list[float] = [0, 0, 0]
) -> dict[str, object]:
    """Set camera position and rotation.

    Args:
        location: Camera location in world coordinates [x, y, z].
        rotation_euler: Camera rotation in euler angles [x, y, z].

    Returns:
        Dictionary with status and rendered image.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    return _investigator.set_camera(location, rotation_euler)


@mcp.tool()
def reload_scene() -> dict[str, object]:
    """Reload the original Blender scene file.

    Returns:
        Dictionary with status message.
    """
    global _investigator
    if _investigator is None:
        return {
            "status": "error",
            "output": {"text": ["Not initialized. Call initialize first."]},
        }
    _investigator.executor.blender_file = _investigator.blender_file
    return {"status": "success", "output": {"text": ["Scene reloaded successfully"]}}


def main() -> None:
    """Run the MCP server or execute test mode."""
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running investigator tools test...")
        test_tools()
    else:
        mcp.run()


def test_tools() -> None:
    """Test investigator tool functions using environment variable configuration."""
    print("=" * 50)
    print("Testing Scene Tools")
    print("=" * 50)

    # Read test paths from environment variables
    blender_file = os.getenv(
        "BLENDER_FILE",
        "data/static_scene/christmas1/reasonable_init/christmas1_gt.blend",
    )
    test_save_dir = os.getenv("THOUGHT_SAVE", "output/test/investigator/")
    blender_command = os.getenv(
        "BLENDER_COMMAND", "lib/utils/third_party/blender-4.5/blender"
    )
    blender_script = os.getenv(
        "BLENDER_SCRIPT", "data/static_scene/generator_script.py"
    )
    gpu_devices = os.getenv("GPU_DEVICES", "0,1,2,3,4,5,6,7")

    if not os.path.exists(blender_file):
        print(f"Blender file not found: {blender_file}")
        print("Skipping all tests.")
        return

    print(f"Using blender file: {blender_file}")

    # Test initialize
    print("\n1. Testing initialize...")
    args = {
        "output_dir": test_save_dir,
        "blender_file": blender_file,
        "blender_command": blender_command,
        "blender_script": blender_script,
        "gpu_devices": gpu_devices,
    }
    result = initialize(args)
    print(f"Result: {result}")

    # Test get scene info
    print("\n2. Testing get_scene_info...")
    scene_info = get_scene_info()
    print(f"Result: {scene_info}")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    print(f"\nTest files saved to: {test_save_dir}")


if __name__ == "__main__":
    main()
