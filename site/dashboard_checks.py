import importlib.util
import json
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("dashboard.py")
SPEC = importlib.util.spec_from_file_location("scenerig_dashboard", MODULE_PATH)
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "output" / "sample" / "scene"
    attempt = scene / "stages/0/InitializerPlannerAgent/attempt_1"
    (attempt / "renders/1").mkdir(parents=True)
    (scene / "final/renders").mkdir(parents=True)
    (scene / "input.png").write_bytes(b"png")
    (scene / "final/renders/final_render.png").write_bytes(b"png")
    (scene / "blender_file.blend").write_bytes(b"blend")
    (scene / "task.log").write_text("SceneRig run complete:\n  Successful: 1\n  Failed: 0\n")
    memory = [
        {"role": "assistant", "content": "render scene", "tool_calls": []},
        {"role": "tool", "name": "end", "content": "Approved: True"},
    ]
    (attempt / "initializer_generator_memory.json").write_text(json.dumps(memory))
    (attempt / "renders/1/output.png").write_bytes(b"png")
    return scene


def _config(output: Path):
    return dashboard.DashboardConfig(
        output_dir=output,
        repo_root=output.parent,
        target_dir=None,
        host="127.0.0.1",
        port=8765,
    )


def test_dashboard_discovers_scene_and_attempt(tmp_path):
    scene = _scene(tmp_path)
    state = dashboard.collect_state(_config(tmp_path / "output"))
    assert state["run_count"] == 1
    assert state["tasks"][0]["path"] == str(scene)
    assert state["tasks"][0]["attempts"][0]["stage"] == "initializer"
    assert state["tasks"][0]["latest_render"]["name"] == "final_render.png"


def test_dashboard_memory_exposes_embedded_image_preview(tmp_path):
    scene = _scene(tmp_path)
    memory = scene / "stages/0/InitializerPlannerAgent/attempt_1/initializer_generator_memory.json"
    memory.write_text(json.dumps([{"role": "user", "content": "data:image/png;base64,AAAA"}]))
    payload = dashboard.collect_memory(str(memory), _config(tmp_path / "output"))
    parts = payload["messages"][0]["parts"]
    assert parts == [{"type": "image", "url": "data:image/png;base64,AAAA"}]


def test_dashboard_rejects_files_outside_output_root(tmp_path):
    _scene(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    assert dashboard.resolve_dashboard_file(str(outside), _config(tmp_path / "output")) is None
