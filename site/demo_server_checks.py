import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("demo_server.py")
SPEC = importlib.util.spec_from_file_location("scenerig_demo_server", MODULE_PATH)
demo = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


def _scene(tmp_path: Path) -> Path:
    scene = tmp_path / "output" / "sample" / "scene"
    attempt = scene / "stages/0/InitializerPlannerAgent/attempt_1"
    (attempt / "renders/1").mkdir(parents=True)
    (scene / "final/renders").mkdir(parents=True)
    (scene / "input.png").write_bytes(b"png")
    (scene / "blender_file.blend").write_bytes(b"blend")
    (scene / "final/final.blend").write_bytes(b"blend")
    (scene / "final/renders/final_render.png").write_bytes(b"png")
    (scene / "task.log").write_text("SceneRig run complete:\n  Successful: 1\n  Failed: 0\n")
    (attempt / "initializer_generator_memory.json").write_text("[]")
    (attempt / "renders/1/output.png").write_bytes(b"png")
    return scene


def _configure(output: Path) -> None:
    demo.OUTPUT_ROOT = output.resolve()
    demo.ALLOWED_ROOTS = [output.resolve()]
    demo.DASH_CONFIG.output_dir = output.resolve()
    demo._runs_cache.update({"t": 0.0, "data": None})


def test_detailed_demo_discovers_and_describes_run(tmp_path, monkeypatch):
    scene = _scene(tmp_path)
    _configure(tmp_path / "output")
    monkeypatch.setattr(demo, "scene_state", lambda *_: {"glb_url": None, "building": False})

    runs = demo._build_runs()
    assert runs[0]["run_id"] == str(scene)
    status = demo.run_status(runs[0]["run_id"])
    assert status["status"] == "complete"
    assert status["final_render_url"].endswith("final_render.png")
    assert len(demo.list_agents(runs[0]["run_id"])) == 1


def test_detailed_demo_page_is_read_only():
    page = demo.index_html()
    assert "New reconstruction" not in page
    assert "Reconstruct scene" not in page
    assert "model-viewer.min.js" in page
    assert "Agent memory (generator / verifier)" in page


def test_detailed_demo_rejects_run_path_outside_output(tmp_path):
    _configure(tmp_path / "output")
    assert demo.run_dir_for("/etc") == (tmp_path / "output" / ".invalid-run")
