#!/usr/bin/env python3
"""Check the public gallery links and self-hosted 3D viewers.

Start scripts/serve.py, then run this file with the same --base-url and
--browser options as test_site.py. Requires Playwright and Chromium.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import unittest
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--base-url", default="http://127.0.0.1:8765")
parser.add_argument("--browser", default=os.environ.get("CHROMIUM_EXECUTABLE"))
OPTIONS, ARGS = parser.parse_known_args()


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        options = {"headless": True, "args": ["--no-sandbox"]}
        if OPTIONS.browser:
            options["executable_path"] = OPTIONS.browser
        cls.browser = cls.pw.chromium.launch(**options)
        cls.base = OPTIONS.base_url.rstrip("/")
        cls.gallery = json.loads((Path(__file__).resolve().parents[1] / "data/gallery.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("response", lambda response: self.errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def wait_for_model(self):
        self.page.wait_for_function("""() => {
            const model = document.querySelector('#scene-model');
            return model.loaded && document.querySelector('#viewer-loading').hidden;
        }""", timeout=90000)
        dimensions = self.page.locator("#scene-model").evaluate("model => { const d = model.getDimensions(); return [d.x, d.y, d.z]; }")
        self.assertTrue(all(value is not None and value >= 0 for value in dimensions))
        self.assertGreater(max(dimensions), 0)
        self.assertTrue(self.page.locator("#viewer-reset").is_enabled())

    def test_every_gallery_image_links_to_its_own_model(self):
        self.page.goto(self.base)
        self.page.wait_for_selector("#comparison-grid .scene-viewer-link")
        count = 0
        for group in self.gallery["groups"]:
            self.page.locator(f"[data-group='{group['id']}']").click()
            for scene in (s for s in self.gallery["scenes"] if s["group"] == group["id"]):
                self.page.select_option("#scene-select", scene["sourceId"])
                for view in ["source", "novel"]:
                    self.page.locator(f"[data-view='{view}']").click()
                    self.assertEqual(self.page.locator("#comparison-grid .scene-viewer-link").count(), len(scene["methods"]))
                    self.assertEqual(self.page.locator(".comparison-card.input a").count(), 0)
                    for method in scene["methods"]:
                        link = self.page.locator(f".comparison-card.{method['id']} .scene-viewer-link")
                        params = parse_qs(urlparse(link.get_attribute("href")).query)
                        self.assertEqual(params, {"group": [group["id"]], "scene": [scene["sourceId"]], "method": [method["id"]]})
                for method in scene["methods"]:
                    count += 1
                    model = method["model"]
                    self.assertTrue(model["src"].startswith("assets/models/"))
                    response = self.context.request.get(f"{self.base}/{model['src']}", headers={"Range": "bytes=0-19"})
                    self.assertIn(response.status, (200, 206))
                    magic, version, size = struct.unpack("<4sII", response.body()[:12])
                    self.assertEqual((magic, version), (b"glTF", 2))
                    self.assertEqual(size, model["bytes"])
        self.assertEqual(count, 192)

    def test_render_each_method_and_the_edited_mug_scene(self):
        # Both groups and all eight exports of the edited input must decode.
        for scene_id in ["scene-21", "scene-01"]:
            for group in self.gallery["groups"]:
                scene = next(s for s in self.gallery["scenes"] if s["group"] == group["id"] and s["sourceId"] == scene_id)
                for method in scene["methods"]:
                    with self.subTest(scene=scene_id, group=group["id"], method=method["id"]):
                        self.page.goto(f"{self.base}/viewer.html?group={group['id']}&scene={scene_id}&method={method['id']}")
                        self.wait_for_model()
                        self.assertEqual(self.page.locator("#viewer-method").input_value(), method["id"])
                        self.assertTrue(self.page.locator("#viewer-input").evaluate("image => image.complete && image.naturalWidth > 0"))
                        self.assertTrue(self.page.locator("#viewer-render").evaluate("image => image.complete && image.naturalWidth > 0"))
                        # A loaded model can still be invisible if a huge support
                        # plane pushes the camera's near clipping plane too far.
                        variance = self.page.locator("#scene-model").evaluate("""async model => {
                            const bitmap = await createImageBitmap(await model.toBlob());
                            const canvas = new OffscreenCanvas(64, 64);
                            const context = canvas.getContext('2d');
                            context.drawImage(bitmap, 0, 0, 64, 64);
                            const pixels = context.getImageData(0, 0, 64, 64).data;
                            let sum = 0, squares = 0, count = 0;
                            for (let i = 0; i < pixels.length; i += 4) {
                                if (pixels[i + 3] < 128) continue;
                                const value = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
                                sum += value; squares += value * value; count++;
                            }
                            bitmap.close();
                            return count ? squares / count - (sum / count) ** 2 : 0;
                        }""")
                        self.assertGreater(variance, 5, "The scene loaded but its objects are not visible")

    def test_orbit_reset_method_switch_and_return_to_gallery(self):
        self.page.goto(f"{self.base}/viewer.html?group=main&scene=scene-02&method=simfoundry")
        self.wait_for_model()
        model = self.page.locator("#scene-model")
        initial = model.evaluate("model => model.getCameraOrbit().theta")
        bounds = model.bounding_box()
        x, y = bounds["x"] + bounds["width"] / 2, bounds["y"] + bounds["height"] / 2
        self.page.mouse.move(x, y)
        self.page.mouse.down()
        self.page.mouse.move(x + 120, y + 40, steps=12)
        self.page.mouse.up()
        self.page.wait_for_function("initial => Math.abs(document.querySelector('#scene-model').getCameraOrbit().theta - initial) > 0.05", arg=initial)
        self.page.locator("#viewer-reset").click()
        self.assertAlmostEqual(model.evaluate("model => model.getCameraOrbit().theta"), initial, places=4)
        self.page.select_option("#viewer-group", "gpt6")
        self.wait_for_model()
        self.assertEqual(self.page.locator("#viewer-method").input_value(), "ours")
        self.page.select_option("#viewer-method", "codex")
        self.wait_for_model()
        self.assertIn("method=codex", self.page.url)
        self.page.go_back()
        self.wait_for_model()
        self.assertEqual(self.page.locator("#viewer-method").input_value(), "ours")
        self.page.locator("#back-to-gallery").click()
        self.page.wait_for_selector("#comparison-grid .scene-viewer-link")
        self.assertEqual(self.page.locator("#scene-select").input_value(), "scene-02")
        self.assertEqual(self.page.locator("[data-group='gpt6']").get_attribute("aria-pressed"), "true")

    def test_mobile_layout_and_invalid_link(self):
        for width in [320, 390, 768]:
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.goto(f"{self.base}/viewer.html?group=main&scene=scene-02&method=rest3d")
            self.wait_for_model()
            self.assertLessEqual(self.page.evaluate("document.documentElement.scrollWidth"), width)
            self.assertTrue(self.page.locator("#viewer-reset").is_visible())
        self.page.goto(f"{self.base}/viewer.html?group=main&scene=does-not-exist&method=ours")
        self.page.wait_for_function("document.querySelector('#viewer-loading').textContent.includes('could not be found')")
        self.assertTrue(self.page.locator("#back-to-gallery").is_visible())


if __name__ == "__main__":
    unittest.main(argv=[__file__, *ARGS])
