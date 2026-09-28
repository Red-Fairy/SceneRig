"""Core classes for 3D scene investigation.

Provides the Executor and Investigator3D classes for camera manipulation,
scene inspection, and viewpoint management in Blender scenes.
"""

import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from .script_generators import (
        generate_camera_focus_script,
        generate_camera_move_script,
        generate_camera_set_script,
        generate_keyframe_script,
        generate_render_script,
        generate_scene_info_script,
        generate_viewpoint_script,
        generate_visibility_script,
    )
except ImportError:
    from script_generators import (
        generate_camera_focus_script,
        generate_camera_move_script,
        generate_camera_set_script,
        generate_keyframe_script,
        generate_render_script,
        generate_scene_info_script,
        generate_viewpoint_script,
        generate_visibility_script,
    )

# Wall-clock cap (seconds) for any Blender subprocess. A hung GPU render (under
# contention) or an LLM-generated infinite loop would otherwise freeze the MCP
# server indefinitely with no recovery. Override via GRASE_BLENDER_TIMEOUT.
BLENDER_TIMEOUT = int(os.getenv("GRASE_BLENDER_TIMEOUT", "600"))

ORBIT_AZIMUTH_STEP = math.radians(45)
ORBIT_ELEVATION_STEP = math.radians(30)
ORBIT_PHI_LIMIT = math.pi / 2 - 0.1  # never quite straight up/down (gimbal-ish views)

# zoom is MULTIPLICATIVE so it is scale-free. It was `max(1, radius -/+ 3)` in world
# units: on all 20 real focus events (radius 0.505-3.419) zoom("in") landed on exactly
# 1.0 — a teleport to the clamp floor, not a step — and on 12 of them that moved the
# camera AWAY from the object. The clamp is anchored to r0, the focus-time radius, which
# is the real reference-camera distance: known-good, already stored, and meaningful for a
# wall or floor where "object size" is not.
ZOOM_FACTOR = 0.6
RADIUS_MIN_FACTOR = 0.25  # x r0
RADIUS_MAX_FACTOR = 4.0  # x r0

ALTERNATE_VIEW_WARNING = (
    "ALTERNATE CAMERA — NOT the reference view. Use it for 3D topology only (which side "
    "of the scene something is on, floating, penetration, hidden geometry). NEVER judge "
    "framing, coverage, pose/yaw, exposure, contrast or material appearance from this "
    "image — re-render with render_reference_view for any of those."
)


def _tag_alternate_view(result: dict, label: str = "") -> dict:
    """Prefix each text of an image-bearing tool result with the alternate-camera warning.

    No-op on failures and on image-less results (an error string must stay the whole
    message). ``label`` numbers multi-image results, e.g. initialize_viewpoint's four."""
    if result.get("status") != "success":
        return result
    output = result.get("output") or {}
    if not output.get("image"):
        return result
    texts = output.get("text") or []
    head = f"{label} {ALTERNATE_VIEW_WARNING}" if label else ALTERNATE_VIEW_WARNING
    if not texts:
        output["text"] = [head]
        return result
    output["text"] = [f"{head} {t}" for t in texts]
    return result


class Executor:
    """Lightweight executor for running Blender scripts.

    Handles script execution, rendering, and result collection for
    the investigator tool.

    Attributes:
        blender_command: Command to invoke Blender.
        blender_file: Path to the Blender scene file.
        blender_script: Path to the execution script.
        base: Base directory for outputs.
        script_path: Directory for saving scripts.
        render_path: Directory for rendered images.
        blender_save: Path to save modified Blender files.
        gpu_devices: CUDA device specification.
        count: Execution counter.
    """

    def __init__(
        self,
        blender_command: str,
        blender_file: str,
        blender_script: str,
        script_save: str,
        render_save: str,
        blender_save: Optional[str] = None,
        gpu_devices: Optional[str] = None,
        render_engine: Optional[str] = None,
        pipeline_object_names_resolver: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        """Initialize the executor.

        Args:
            blender_command: Command to invoke Blender.
            blender_file: Path to the Blender scene file.
            blender_script: Path to the execution script.
            script_save: Directory for saving scripts.
            render_save: Directory for rendered images.
            blender_save: Optional path to save modified Blender files.
            gpu_devices: Optional CUDA device specification.
            render_engine: Optional Blender render engine name.
            pipeline_object_names_resolver: Optional trusted callback that returns the
                current authenticated Blender object-root names before each execution.
        """
        self.blender_command = blender_command
        self.blender_file = blender_file
        self.blender_script = blender_script
        self.base = os.path.dirname(script_save)
        self.script_path = Path(script_save)
        self.render_path = Path(render_save)
        self.blender_save = blender_save
        self.gpu_devices = gpu_devices
        self.render_engine = render_engine or "CYCLES"
        self.pipeline_object_names_resolver = pipeline_object_names_resolver
        self.count = 0

        self.script_path.mkdir(parents=True, exist_ok=True)
        self.render_path.mkdir(parents=True, exist_ok=True)

    def next_run_dir(self) -> Path:
        """Create and return the next run directory.

        Returns:
            Path to the newly created run directory.
        """
        self.count += 1
        run_dir = self.render_path / f"{self.count}"
        run_dir.mkdir(parents=True, exist_ok=True)
        for p in run_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        return run_dir

    def _execute_blender(self, code_file: Path, run_dir: Path) -> dict[str, Any]:
        """Execute a Blender script and collect results.

        Args:
            code_file: Path to the Python script to execute.
            run_dir: Directory for output files.

        Returns:
            Dictionary with status and output (images or error text).
        """
        cmd = [
            self.blender_command,
            "--background",
            self.blender_file,
            "--python",
            self.blender_script,
            "--",
            str(code_file),
            str(run_dir),
        ]
        if self.blender_save:
            cmd.append(self.blender_save)

        env = os.environ.copy()
        if self.gpu_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.gpu_devices

        # Ban blender audio error
        env["AL_LIB_LOGLEVEL"] = "0"
        env["VIGA_RENDER_ENGINE"] = self.render_engine
        if self.pipeline_object_names_resolver is not None:
            try:
                pipeline_object_names = self.pipeline_object_names_resolver()
                if (
                    not isinstance(pipeline_object_names, list)
                    or any(
                        not isinstance(name, str) or not name
                        for name in pipeline_object_names
                    )
                    or pipeline_object_names != sorted(set(pipeline_object_names))
                ):
                    raise RuntimeError(
                        "authenticated pipeline object names must be a sorted unique "
                        "list of nonempty strings"
                    )
            except Exception as exc:  # noqa: BLE001 - fail before the Blender child
                logging.error(f"Pipeline ownership refresh failed: {exc}")
                return {
                    "status": "error",
                    "output": {"text": [f"Pipeline ownership refresh failed: {exc}"]},
                }
            # Override, rather than trust, any inherited ambient value. Baseline has no
            # resolver and retains its exact historical copied-environment behavior.
            env["GRASE_PIPELINE_OBJECT_NAMES"] = json.dumps(pipeline_object_names)

        try:
            # Propagate render directory to scripts
            env["RENDER_DIR"] = str(run_dir)
            proc = subprocess.run(
                " ".join(cmd),
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                env=env,
                timeout=BLENDER_TIMEOUT,
            )
            imgs = sorted(
                [
                    str(p)
                    for p in run_dir.glob("*")
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
                ]
            )
            # If no image output
            if not os.path.exists(f"{self.base}/tmp/camera_info.json"):
                return {"status": "success", "output": {"text": [proc.stdout]}}
            # If image output
            with open(f"{self.base}/tmp/camera_info.json", "r") as f:
                camera_info = json.load(f)
                for camera in camera_info:
                    camera["location"] = [round(x, 2) for x in camera["location"]]
                    camera["rotation"] = [round(x, 2) for x in camera["rotation"]]
            return {
                "status": "success",
                "output": {
                    "image": imgs,
                    "text": [
                        "Camera parameters: " + str(camera) for camera in camera_info
                    ],
                },
            }
        except subprocess.TimeoutExpired:
            logging.error(f"Blender timed out after {BLENDER_TIMEOUT}s")
            return {
                "status": "error",
                "output": {
                    "text": [
                        f"Blender timed out after {BLENDER_TIMEOUT}s and was killed."
                    ]
                },
            }
        except subprocess.CalledProcessError as e:
            logging.error(f"Blender failed: {e.stderr}")
            return {"status": "error", "output": {"text": [e.stderr or e.stdout]}}

    def execute(self, full_code: str) -> dict[str, Any]:
        """Execute Blender code and return results.

        Args:
            full_code: Complete Python code to execute in Blender.

        Returns:
            Dictionary with status and output (images or error text).
        """
        run_dir = self.next_run_dir()
        code_file = self.script_path / f"{self.count}.py"
        with open(code_file, "w") as f:
            f.write(full_code)
        result = self._execute_blender(code_file, run_dir)
        # Remove empty run directories
        if not os.listdir(run_dir):
            shutil.rmtree(run_dir)
            self.count -= 1
        return result


class Investigator3D:
    """3D scene investigator for camera manipulation and analysis.

    Provides methods for camera control, scene inspection, and viewpoint
    management in Blender scenes.

    Attributes:
        blender_file: Path to the Blender scene file.
        blender_command: Command to invoke Blender.
        base: Base directory for outputs.
        tmp_dir: Temporary directory for intermediate files.
        executor: Blender script executor.
        target: Current target object name for camera focus.
        radius: Camera orbit radius.
        theta: Camera azimuth angle.
        phi: Camera elevation angle.
        count: Operation counter.
    """

    def __init__(
        self,
        save_dir: str,
        blender_path: str,
        blender_command: str,
        blender_script: str,
        gpu_devices: str,
        render_engine: Optional[str] = None,
        pipeline_object_names_resolver: Optional[Callable[[], list[str]]] = None,
    ) -> None:
        """Initialize the 3D investigator.

        Args:
            save_dir: Directory for saving outputs.
            blender_path: Path to the Blender scene file.
            blender_command: Command to invoke Blender.
            blender_script: Path to the execution script.
            gpu_devices: CUDA device specification.
            render_engine: Optional Blender render engine name.
            pipeline_object_names_resolver: Optional trusted callback refreshed by the
                child executor immediately before each Blender process.
        """
        self.blender_file = blender_path
        self.blender_command = blender_command
        self.base = Path(save_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.tmp_dir = self.base / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        self.executor = Executor(
            blender_command=blender_command,
            blender_file=blender_path,
            blender_script=blender_script,
            script_save=str(self.base / "scripts"),
            render_save=str(self.base / "renders"),
            blender_save=str(self.base / "current_scene.blend"),
            gpu_devices=gpu_devices,
            render_engine=render_engine,
            pipeline_object_names_resolver=pipeline_object_names_resolver,
        )

        # Camera state variables
        self.target: Optional[str] = None
        self.radius: float = 5.0
        # Focus-time radius: the anchor the zoom clamp is expressed against, so "as close
        # as you may get" scales with the scene instead of being a hardcoded 1 world unit.
        self.radius0: float = 5.0
        self.theta: float = 0.0
        self.phi: float = 0.0
        self.count: int = 0

    def _generate_scene_info_script(self) -> str:
        """Generate script to get scene information."""
        return generate_scene_info_script(f"{self.base}/tmp/scene_info.json")

    def _generate_render_script(self) -> str:
        """Generate script to render current scene once into RENDER_DIR/output.png."""
        return generate_render_script()

    def _generate_camera_focus_script(self, object_name: str) -> str:
        """Generate script to focus camera on object."""
        return generate_camera_focus_script(object_name, str(self.base))

    def _generate_camera_set_script(self, location: list, rotation_euler: list) -> str:
        """Generate script to set camera position and rotation."""
        return generate_camera_set_script(location, rotation_euler, str(self.base))

    def _generate_visibility_script(
        self, show_objects: list, hide_objects: list
    ) -> str:
        """Generate script to set object visibility and render once."""
        return generate_visibility_script(show_objects, hide_objects, str(self.base))

    def _generate_camera_move_script(
        self, target_obj_name: str, radius: float, theta: float, phi: float
    ) -> str:
        """Generate script to move camera around target object."""
        return generate_camera_move_script(
            target_obj_name, radius, theta, phi, str(self.base)
        )

    def _generate_keyframe_script(self, frame_number: int) -> str:
        """Generate script to set frame number."""
        return generate_keyframe_script(frame_number, str(self.base))

    def _generate_viewpoint_script(self, object_names: list) -> str:
        """Generate script to initialize viewpoints around objects."""
        return generate_viewpoint_script(object_names, str(self.base))

    def _execute_script(self, script_code: str, description: str = "") -> dict:
        """Execute a blender script and return results."""
        try:
            result = self.executor.execute(full_code=script_code)

            # Continue from the saved blend on the next op — but ONLY if it was actually
            # written (a read-only op may "succeed" without saving, leaving the file absent
            # and the next open failing with "Cannot read current_scene.blend").
            if result.get("status") == "success":
                if self.executor.blender_save and os.path.exists(
                    self.executor.blender_save
                ):
                    self.executor.blender_file = self.executor.blender_save

            return result
        except Exception as e:
            logging.error(f"Script execution failed: {e}")
            return {"status": "error", "output": {"text": [str(e)]}}

    def _render(self) -> dict:
        """Render current scene and return image path and camera parameters."""
        render_script = self._generate_render_script()
        return self._execute_script(render_script, "Render current scene")

    def render_reference_view(self) -> dict:
        """Render from the LOCKED REFERENCE camera — the target photo's exact
        viewpoint. Camera ops (set_camera/investigate/initialize_viewpoint)
        persist into the session's working blend, so its active camera may have
        wandered; this re-opens the PRISTINE stage blend (whose active camera IS
        the locked one) and renders it. Side effect BY DESIGN: the session
        continues from this state, so the investigation camera is reset to the
        reference pose (a lost verifier can re-anchor)."""
        self.executor.blender_file = self.blender_file
        try:  # a previous camera op's sidecar would mislabel this render's result
            os.remove(f"{self.base}/tmp/camera_info.json")
        except FileNotFoundError:
            pass
        result = self._execute_script(
            self._generate_render_script(), "Render the locked reference view"
        )
        if result.get("status") != "success":
            return result
        self.target = None
        # generate_render_script writes no camera_info sidecar, so the executor
        # returns text-only — attach the rendered png(s) from this run's dir
        run_dir = self.executor.render_path / str(self.executor.count)
        imgs = sorted(
            str(q) for q in run_dir.glob("*")
            if q.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        return {
            "status": "success",
            "output": {
                "image": imgs,
                "text": [
                    "Reference-view render (the LOCKED camera — the target photo's "
                    "exact viewpoint). Framing/coverage/pose comparisons against the "
                    "target photo are valid on THIS image. The investigation camera "
                    "has been reset to this pose."
                ],
            },
        }

    def get_info(self) -> dict:
        """Get scene information by executing a script — re-read on EVERY call.

        Deliberately NOT cached (a session-lifetime memo was removed 07-27). The tool is
        advertised as the CURRENT scene state, and the verifier's own camera tools mutate
        the session blend: `focus` attaches a TRACK_TO constraint and repositions,
        `set_camera` moves the camera, `initialize_viewpoint` adds four more, and
        `render_reference_view` re-opens the pristine blend and resets the pose. A memo
        served `cameras` describing a camera that no longer existed, with no error. It
        also cost nothing to remove: across 1,444 verifier sessions of every type,
        get_scene_info has never been called twice, so the cache never once served a hit.
        The generator's get_scene_info (exec.py) is likewise uncached — same tool name,
        same contract."""
        try:
            info_path = f"{self.base}/tmp/scene_info.json"
            try:
                os.remove(info_path)  # don't trust a stale file if extraction fails
            except FileNotFoundError:
                pass
            script = self._generate_scene_info_script()
            result = self._execute_script(script, "Extract scene information")
            # The shared Blender wrapper renders the scene AFTER writing
            # scene_info.json, so a trailing render crash (e.g. GPU contention) fails
            # the subprocess even though the info was already extracted. Trust the
            # file: if it's there, the extraction succeeded.
            if os.path.exists(info_path):
                with open(info_path, "r") as f:
                    scene_info = json.load(f)
                return {"status": "success", "output": {"text": [str(scene_info)]}}
            # Genuine failure: surface the real Blender error, not a generic message.
            err = result.get("output", {}).get("text") or [
                "Failed to extract scene information"
            ]
            return {"status": "error", "output": {"text": err}}
        except Exception as e:
            return {"status": "error", "output": {"text": [str(e)]}}

    def focus_on_object(self, object_name: str) -> dict:
        """Focus camera on a specific object."""
        self.target = object_name  # Store object name instead of object reference
        # Generate and execute focus script
        focus_script = self._generate_camera_focus_script(object_name)
        result = self._execute_script(
            focus_script, f"Focus camera on object {object_name}"
        )
        if os.path.exists(f"{self.base}/tmp/rotate_info.json"):
            with open(f"{self.base}/tmp/rotate_info.json", "r") as f:
                rotate_info = json.load(f)
                self.radius = rotate_info["radius"]
                self.radius0 = self.radius  # anchor for the zoom clamp
                self.theta = rotate_info["theta"]
                self.phi = rotate_info["phi"]
        return _tag_alternate_view(result)

    def zoom(self, direction: str) -> dict:
        """Zoom camera in or out."""
        if not self.target:
            return {
                "status": "error",
                "output": {"text": ["No target object set. Call focus first."]},
            }
        if direction == "in":
            self.radius = max(
                RADIUS_MIN_FACTOR * self.radius0, self.radius * ZOOM_FACTOR
            )
        elif direction == "out":
            self.radius = min(
                RADIUS_MAX_FACTOR * self.radius0, self.radius / ZOOM_FACTOR
            )
        return self._update_and_render()

    def move_camera(self, direction: str) -> dict:
        """Move camera around target object."""
        if not self.target:
            return {
                "status": "error",
                "output": {"text": ["No target object set. Call focus first."]},
            }
        if direction == "up":
            self.phi = min(ORBIT_PHI_LIMIT, self.phi + ORBIT_ELEVATION_STEP)
        elif direction == "down":
            self.phi = max(-ORBIT_PHI_LIMIT, self.phi - ORBIT_ELEVATION_STEP)
        elif direction == "left":
            self.theta -= ORBIT_AZIMUTH_STEP
        elif direction == "right":
            self.theta += ORBIT_AZIMUTH_STEP
        return self._update_and_render()

    def _update_and_render(self) -> dict:
        """Move the camera to the current (radius, theta, phi) and render.

        Only reached with a target set — both callers guard first, so the old
        ``if not self.target: return self._render()`` branch was unreachable."""
        # Generate script to move camera
        move_script = self._generate_camera_move_script(
            self.target, self.radius, self.theta, self.phi
        )
        return _tag_alternate_view(
            self._execute_script(move_script, f"Move camera around {self.target}")
        )

    def set_camera(self, location: list, rotation_euler: list) -> dict:
        """Set camera position and rotation.

        Clears ``self.target``: the generated script drops the focus TRACK_TO constraint
        (otherwise it would override the requested rotation), so the orbit state left by
        that focus no longer describes this camera. A following zoom/move must re-focus
        rather than orbit a target the camera is no longer tracking."""
        script = self._generate_camera_set_script(location, rotation_euler)
        result = self._execute_script(
            script, f"Set camera to location {location} and rotation {rotation_euler}"
        )
        if result.get("status") == "success":
            self.target = None
        return _tag_alternate_view(result)

    def initialize_viewpoint(self, object_names: list) -> dict:
        """Initialize viewpoints around specified objects."""
        script = self._generate_viewpoint_script(object_names)
        result = self._execute_script(
            script, f"Initialize viewpoints for objects: {object_names}"
        )
        texts = result.get("output", {}).get("text")
        if result.get("status") == "success" and texts:
            # numbered per image, sharing the one warning text with the other alt cameras
            result["output"]["text"] = [
                f"VIEWPOINT {i + 1}/{len(texts)}. {ALTERNATE_VIEW_WARNING} {t}"
                for i, t in enumerate(texts)
            ]
        return result

    def set_keyframe(self, frame_number: int) -> dict:
        """Set scene to a specific frame."""
        script = self._generate_keyframe_script(frame_number)
        return self._execute_script(script, f"Set frame to {frame_number}")

    def set_visibility(self, show_objects: list, hide_objects: list) -> dict:
        """Set visibility of objects."""
        script = self._generate_visibility_script(show_objects, hide_objects)
        return self._execute_script(
            script, f"Set visibility: show {show_objects}, hide {hide_objects}"
        )
