from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lib.tools.geometry.depth_refine import refine_depth


@pytest.mark.parametrize("mask_format", ["npy", "rgba", "grayscale"])
@pytest.mark.parametrize("scale", [1, 2])
def test_refine_depth_uses_moge2_and_aligns_known_scale(
    tmp_path, monkeypatch, mask_format, scale
):
    scene = tmp_path / "scene"
    moge = scene / "moge"
    edited = scene / "masks" / "edited"
    moge.mkdir(parents=True)
    edited.mkdir(parents=True)
    (moge / "moge.json").write_text(
        '{"intrinsics_norm": [[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]}'
    )
    np.save(moge / "depth.npy", np.full((2, 2), 2.0, dtype=np.float32))
    hole = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    mask_path = edited / ("hole.npy" if mask_format == "npy" else "hole.png")
    if mask_format == "npy":
        np.save(mask_path, hole)
    elif mask_format == "rgba":
        rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
        rgba[..., 3] = (1 - hole) * 255
        Image.fromarray(rgba).save(mask_path)
    else:
        Image.fromarray(hole * 255).save(mask_path)
    Image.new("RGB", (2 * scale, 2 * scale)).save(edited / "image.png")
    predicted = np.repeat(
        np.repeat(np.array([[1, 4], [4, 4]], dtype=np.float32), scale, axis=0),
        scale,
        axis=1,
    )

    def fake_estimate(_image, out_dir, fov_x_deg):
        assert fov_x_deg > 0
        target = Path(out_dir) / "moge"
        target.mkdir(parents=True)
        path = target / "depth.npy"
        np.save(path, predicted)
        return SimpleNamespace(depth_npy=str(path))

    monkeypatch.setattr("lib.tools.geometry.moge_estimate.estimate", fake_estimate)
    result = refine_depth(
        str(scene), str(edited / "image.png"), str(mask_path), "object"
    )

    assert result["backend"] == "moge2"
    np.testing.assert_allclose(np.load(result["out_depth"]), predicted * 2)
    assert np.load(result["out_points"]).shape == (2 * scale, 2 * scale, 3)
