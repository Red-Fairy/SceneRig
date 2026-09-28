from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image

from lib.tools.geometry.depth_refine import refine_depth


def test_refine_depth_uses_moge2_and_aligns_known_scale(tmp_path, monkeypatch):
    scene = tmp_path / "scene"
    moge = scene / "moge"
    edited = scene / "masks" / "edited"
    moge.mkdir(parents=True)
    edited.mkdir(parents=True)
    (moge / "moge.json").write_text(
        '{"intrinsics_norm": [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]}'
    )
    np.save(moge / "depth.npy", np.full((2, 2), 2.0, dtype=np.float32))
    np.save(edited / "hole.npy", np.array([[0, 0], [0, 1]], dtype=np.uint8))
    Image.new("RGB", (2, 2)).save(edited / "image.png")

    def fake_estimate(_image, out_dir, fov_x_deg):
        assert fov_x_deg > 0
        target = Path(out_dir) / "moge"
        target.mkdir(parents=True)
        path = target / "depth.npy"
        np.save(path, np.ones((2, 2), dtype=np.float32))
        return SimpleNamespace(depth_npy=str(path))

    monkeypatch.setattr("lib.tools.geometry.moge_estimate.estimate", fake_estimate)
    result = refine_depth(
        str(scene), str(edited / "image.png"), str(edited / "hole.npy"), "object"
    )

    assert result["backend"] == "moge2"
    np.testing.assert_allclose(np.load(result["out_depth"]), 2.0)
    assert np.load(result["out_points"]).shape == (2, 2, 3)
