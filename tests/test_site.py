#!/usr/bin/env python3
"""Exercise the served project page in a real Chromium browser.

Requires the Python Playwright package and a Chromium installation. Start a
static server at the repository root, then run:

    python tests/test_site.py --base-url http://127.0.0.1:8765

Use --browser /path/to/chrome or CHROMIUM_EXECUTABLE for a custom installation.
Screenshots are saved under test-results/ by default.
"""

import argparse
import json
import os
from pathlib import Path
import unittest

from playwright.sync_api import sync_playwright


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--base-url", default="http://127.0.0.1:8765")
parser.add_argument("--browser", default=os.environ.get("CHROMIUM_EXECUTABLE"))
parser.add_argument("--screenshots", type=Path, default=Path("test-results"))
OPTIONS, UNITTEST_ARGS = parser.parse_known_args()


class ProjectPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.playwright = sync_playwright().start()
        launch_options = {"headless": True, "args": ["--no-sandbox"]}
        if OPTIONS.browser:
            launch_options["executable_path"] = OPTIONS.browser
        cls.browser = cls.playwright.chromium.launch(**launch_options)
        OPTIONS.screenshots.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()

    def setUp(self):
        self.errors = []
        self.context = self.browser.new_context(
            viewport={"width": 1440, "height": 1000}, reduced_motion="reduce"
        )
        self.page = self.context.new_page()
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on("console", lambda message: self.errors.append(message.text) if message.type == "error" else None)
        self.page.on("response", lambda response: self.errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
        self.page.goto(OPTIONS.base_url, wait_until="networkidle")
        self.page.wait_for_selector("#comparison-grid .comparison-card")
        response = self.context.request.get(f"{OPTIONS.base_url.rstrip('/')}/data/gallery.json")
        self.assertTrue(response.ok)
        self.gallery = response.json()

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [], "Browser console or network errors")

    def test_all_gallery_images_decode(self):
        paths = set()
        for scene in self.gallery["scenes"]:
            paths.add(scene["input"])
            for method in scene["methods"]:
                paths.update([method["source"], method["novel"]])
        self.assertGreaterEqual(len(paths), 400)
        paths.update(self.page.locator("img").evaluate_all("images => images.map(image => image.getAttribute('src'))"))
        images = self.page.evaluate("""async paths => {
            const results = [];
            for (let offset = 0; offset < paths.length; offset += 8) {
                const batch = await Promise.all(paths.slice(offset, offset + 8).map(async path => {
                    const image = new Image();
                    image.src = path;
                    await image.decode();
                    return {path, width: image.naturalWidth, height: image.naturalHeight};
                }));
                results.push(...batch);
            }
            return results;
        }""", sorted(paths))
        self.assertEqual(len(images), len(paths))
        self.assertTrue(all(image["width"] > 0 and image["height"] > 0 for image in images))

    def test_gallery_groups_views_and_every_scene(self):
        for group in self.gallery["groups"]:
            self.page.locator(f"[data-group='{group['id']}']").click()
            self.assertEqual(self.page.locator(f"[data-group='{group['id']}']").get_attribute("aria-pressed"), "true")
            scenes = [scene for scene in self.gallery["scenes"] if scene["group"] == group["id"]]
            self.assertEqual(self.page.locator("#scene-select option").count(), len(scenes))
            self.assertEqual(self.page.locator("#gallery-note").inner_text(), group["note"])
            for view in self.gallery["views"]:
                self.page.locator(f"[data-view='{view['id']}']").click()
                self.assertEqual(self.page.locator(f"[data-view='{view['id']}']").get_attribute("aria-pressed"), "true")
                for scene in scenes:
                    self.page.locator("#scene-select").select_option(scene["sourceId"])
                    expected = {scene["input"]} | {method[view["id"]] for method in scene["methods"]}
                    actual = self.page.locator("#comparison-grid img").evaluate_all("images => images.map(image => image.getAttribute('src'))")
                    self.assertEqual(set(actual), expected, (scene["id"], view["id"]))
                    self.assertEqual(len(actual), len(scene["methods"]) + 1)
                    ours = next(method for method in scene["methods"] if method["id"] == "ours")
                    self.assertEqual(actual[:2], [scene["input"], ours[view["id"]]], (scene["id"], view["id"]))
                    labels = self.page.locator("#comparison-grid figcaption").all_text_contents()
                    for method in scene["methods"]:
                        self.assertTrue(any(label.startswith(method["label"]) for label in labels))
                    selected = self.page.locator(".scene-thumb[aria-pressed='true']")
                    self.assertEqual(selected.count(), 1)
                    self.assertEqual(selected.get_attribute("data-scene"), scene["sourceId"])

    def test_gallery_filter_navigation_and_thumbnail(self):
        categories = sorted({scene["category"] for scene in self.gallery["scenes"]})
        for group in self.gallery["groups"]:
            self.page.locator(f"[data-group='{group['id']}']").click()
            for category in categories:
                self.page.locator("#category-select").select_option(category)
                scenes = [scene for scene in self.gallery["scenes"] if scene["group"] == group["id"] and scene["category"] == category]
                self.assertEqual(self.page.locator("#scene-select option").count(), len(scenes))
                self.assertEqual(self.page.locator(".scene-thumb").count(), len(scenes))
                self.page.locator("#scene-select").select_option(scenes[0]["sourceId"])
                self.page.locator("#scene-prev").click()
                self.assertEqual(self.page.locator("#scene-select").input_value(), scenes[-1]["sourceId"])
                self.page.locator("#scene-next").click()
                self.assertEqual(self.page.locator("#scene-select").input_value(), scenes[0]["sourceId"])
                self.page.locator(f".scene-thumb[data-scene='{scenes[-1]['sourceId']}']").click()
                self.assertEqual(self.page.locator("#scene-select").input_value(), scenes[-1]["sourceId"])
                self.assertEqual(self.page.locator("#scene-count").inner_text(), f"{len(scenes)} / {len(scenes)}")
            self.page.locator("#category-select").select_option("all")

    def test_material_slider_motion_and_keyboard(self):
        slider = self.page.locator("#clay-range")
        toggle = self.page.locator("#clay-motion")
        slider.scroll_into_view_if_needed()
        self.assertEqual(toggle.get_attribute("aria-pressed"), "false", "Reduced motion must disable automatic animation")
        initial = slider.input_value()
        self.page.wait_for_timeout(450)
        self.assertEqual(slider.input_value(), initial)
        slider.focus()
        slider.press("ArrowRight")
        moved = str(int(initial) + 1)
        self.assertEqual(slider.input_value(), moved)
        self.assertEqual(slider.get_attribute("aria-valuetext"), f"{moved} percent textured")
        self.assertEqual(toggle.get_attribute("aria-pressed"), "false")
        toggle.click()
        self.assertEqual(toggle.get_attribute("aria-pressed"), "true")
        self.page.wait_for_function("old => document.querySelector('#clay-range').value !== old", arg=moved)
        toggle.click()
        paused = slider.input_value()
        self.page.wait_for_timeout(450)
        self.assertEqual(slider.input_value(), paused)
        self.page.emulate_media(reduced_motion="no-preference")
        toggle.click()
        self.page.emulate_media(reduced_motion="reduce")
        self.page.wait_for_function("document.querySelector('#clay-motion').getAttribute('aria-pressed') === 'false'")
        self.assertEqual(toggle.get_attribute("aria-pressed"), "false")

    def test_teaser_sequence_and_real_episode_sources(self):
        teaser_text = self.page.locator("#teaser").inner_text()
        for removed in [
            "A photograph becomes an editable collection of objects and surfaces.",
            "The rear mug is upright in the image.",
            "Its reconstructed orientation matters.",
            "Same meshes. Same camera. Drag to reveal.",
            "Coffee scene: SceneRig with GPT-6 Astra.",
            "All images and videos show actual reconstructions or recorded executions.",
        ]:
            self.assertNotIn(removed, teaser_text)
        response = self.context.request.get(f"{OPTIONS.base_url.rstrip('/')}/data/robotics.json")
        self.assertTrue(response.ok)
        episodes = {item["id"]: item for item in response.json()["items"]}
        self.assertEqual(self.page.locator(".hero-application").count(), 2)
        for episode_id in ["policy-01", "replay-01"]:
            application = self.page.locator(f"[data-hero='{episode_id}']")
            self.assertEqual(application.locator("video").count(), 3)
            self.assertEqual(application.locator("figure").evaluate_all("figures => figures.map(figure => figure.dataset.role)"), ["real", "baseline", "ours"])
            for role in ["real", "baseline", "ours"]:
                figure = application.locator(f"figure[data-role='{role}']")
                self.assertEqual(figure.locator("source").get_attribute("src"), episodes[episode_id]["media"][role]["src"])
                self.assertEqual(figure.locator("video").get_attribute("poster"), episodes[episode_id]["media"][role]["start_poster"])
                self.assertTrue(figure.locator("video").evaluate("video => video.controls && video.playsInline"))
            self.assertIn("Real episode", application.locator("figure[data-role='real'] figcaption").inner_text())

        for width in [1440, 768, 390]:
            self.page.set_viewport_size({"width": width, "height": 1000})
            panels = {name: self.page.locator(f".{name}-panel").bounding_box() for name in ["input", "closeup", "scene"]}
            self.assertAlmostEqual(panels["input"]["x"], panels["closeup"]["x"], delta=1)
            self.assertGreaterEqual(panels["closeup"]["y"], panels["input"]["y"] + panels["input"]["height"], width)
            if width >= 768:
                self.assertGreater(panels["scene"]["x"], panels["input"]["x"] + panels["input"]["width"])
                self.assertAlmostEqual(panels["scene"]["y"], panels["input"]["y"], delta=1)
            else:
                self.assertGreater(panels["scene"]["y"], panels["closeup"]["y"] + panels["closeup"]["height"])
            applications = self.page.locator(".hero-application")
            first, second = applications.nth(0).bounding_box(), applications.nth(1).bounding_box()
            self.assertAlmostEqual(first["x"], second["x"], delta=1)
            self.assertGreater(second["y"], first["y"] + first["height"])
            for video in self.page.locator("#hero-robotics video").all():
                box = video.bounding_box()
                self.assertAlmostEqual(box["width"] / box["height"], 16 / 9, delta=.01)

    def test_responsive_layout_and_screenshots(self):
        for width in [320, 390, 768, 1440]:
            self.page.set_viewport_size({"width": width, "height": 1000})
            self.page.evaluate("window.scrollTo(0, 0)")
            self.page.wait_for_timeout(100)
            dimensions = self.page.evaluate("({viewport: innerWidth, page: document.documentElement.scrollWidth})")
            self.assertLessEqual(dimensions["page"], dimensions["viewport"] + 1, f"Horizontal page overflow at {width}px")
            for selector in ["#clay-range", "#clay-motion", "#scene-select", "#category-select", "#scene-prev", "#scene-next"]:
                box = self.page.locator(selector).bounding_box()
                self.assertIsNotNone(box)
                self.assertGreaterEqual(box["x"], -1, (width, selector))
                self.assertLessEqual(box["x"] + box["width"], width + 1, (width, selector))
            if width in [390, 1440]:
                self.page.evaluate("""async () => {
                    for (let y = 0; y < document.documentElement.scrollHeight; y += 800) {
                        window.scrollTo(0, y);
                        await new Promise(resolve => setTimeout(resolve, 40));
                    }
                    await Promise.all([...document.images].filter(image => image.getClientRects().length).map(image => image.decode()));
                    window.scrollTo(0, 0);
                }""")
                self.page.screenshot(path=str(OPTIONS.screenshots / f"page-{width}.png"), full_page=True)
                self.page.locator("#teaser").screenshot(path=str(OPTIONS.screenshots / f"teaser-{width}.png"), style=".site-header { visibility: hidden !important; }")
                self.page.locator(".gallery-browser").screenshot(path=str(OPTIONS.screenshots / f"gallery-{width}.png"), style=".site-header { visibility: hidden !important; }")

    def test_reconstruction_metric_baselines(self):
        selector = self.page.locator("#metric-baseline")
        self.assertEqual(selector.input_value(), "viga")
        expected = {
            "viga": ["vs. 0.283", "vs. 4.9 cm", "vs. 10.0%"],
            "rest3d": ["vs. 0.210", "vs. 4.7 cm", "vs. 31.3%"],
            "sceneconductor": ["vs. 0.197", "vs. 5.0 cm", "vs. 8.8%"],
            "simfoundry": ["vs. 0.676", "vs. 3.3 cm", "vs. 98.8%"],
        }
        for method, values in expected.items():
            selector.select_option(method)
            self.assertEqual(self.page.locator("[data-baseline-metric]").all_text_contents(), values)
            self.assertEqual(self.page.locator("#metric-baseline-name").inner_text(), selector.locator("option:checked").inner_text())
        replay = self.page.locator(".replay-results-card tbody tr")
        self.assertEqual(replay.nth(0).locator("td").all_text_contents(), ["80%", "36%"])
        self.assertEqual(replay.nth(0).locator("th").inner_text(), "All tasks")
        tasks = self.page.locator(".replay-task-rows tr:not(.replay-all-tasks)")
        expected_replay = [("Everything → bin", 3, 0), ("Cup → bowl", 4, 3),
                           ("Fruits → bowl", 4, 2), ("Marker → cup", 4, 0),
                           ("Mustard → bin", 5, 4)]
        self.assertEqual(tasks.count(), 5)
        for row, (name, ours, baseline) in zip(tasks.all(), expected_replay):
            self.assertEqual(row.locator("th").inner_text(), name)
            self.assertEqual(row.locator("td").all_text_contents(), [f"{ours}/5", f"{baseline}/5"])
            self.assertEqual(row.locator(".replay-task-bar").evaluate_all(
                "bars => bars.map(bar => bar.style.getPropertyValue('--success'))"),
                [f"{ours * 20}%", f"{baseline * 20}%"])
        self.assertNotIn("Objects placed at target", self.page.locator(".replay-results-card").inner_text())

    def test_policy_correlation_dialog(self):
        trigger = self.page.get_by_role("button", name="View correlation plot")
        dialog = self.page.locator("#policy-correlation")
        self.assertFalse(dialog.is_visible())
        rows = self.page.locator(".policy-evaluation-card .application-metrics tbody tr")
        self.assertEqual(rows.nth(0).locator("td").all_text_contents(), ["69%", "54%"])
        self.assertEqual(rows.nth(1).locator("td").all_text_contents(), ["0.92", "0.84"])
        trigger.click()
        self.assertTrue(dialog.is_visible())
        dialog.locator("img").evaluate("image => image.decode()")
        self.page.keyboard.press("Escape")
        self.assertFalse(dialog.is_visible())
        self.assertTrue(trigger.evaluate("element => element === document.activeElement"))
        trigger.click()
        self.page.get_by_role("button", name="Close correlation plot").click()
        self.assertFalse(dialog.is_visible())

    def test_pose_trace_evidence_and_navigation(self):
        response = self.context.request.get(f"{OPTIONS.base_url.rstrip('/')}/data/pose-trace.json")
        self.assertTrue(response.ok)
        trace = response.json()
        call_ids = ["inspect", "translate", "scale", "reinspect", "refine"]
        self.assertEqual([call["id"] for call in trace["calls"]], call_ids)
        self.page.wait_for_selector(".trace-step")
        self.assertEqual(self.page.locator(".trace-step").count(), len(call_ids))
        self.assertEqual(self.page.locator("#trace-context").inner_text(), trace["summary"])
        detail = self.page.locator("#trace-detail")
        previous = detail.locator("[data-trace-direction='-1']")
        following = detail.locator("[data-trace-direction='1']")

        def selected_id():
            selected = self.page.locator(".trace-step[aria-pressed='true']")
            self.assertEqual(selected.count(), 1)
            return selected.get_attribute("data-trace-call")

        for index, call in enumerate(trace["calls"]):
            self.page.locator(f"[data-trace-call='{call['id']}']").click()
            self.assertEqual(selected_id(), call["id"])
            rendered_call = detail.locator(".trace-call-code").inner_text()
            self.assertTrue(rendered_call.startswith(call["tool"] + "("))
            self.assertTrue(rendered_call.endswith(")"))
            self.assertEqual(json.loads(rendered_call[len(call["tool"]) + 1:-1]), call["arguments"])
            self.assertEqual(detail.locator(".trace-response").inner_text(), call["response"])
            self.assertEqual(detail.locator(".trace-summary").inner_text(), call["summary"])
            self.assertEqual(previous.is_disabled(), index == 0)
            self.assertEqual(following.is_disabled(), index == len(call_ids) - 1)

            images = detail.locator(".trace-images img")
            self.assertEqual(images.count(), 2)
            for image, expected in zip(images.all(), call["images"]):
                self.assertEqual(image.get_attribute("src"), expected["src"])
                self.assertEqual(image.get_attribute("alt"), expected["alt"])
                image.scroll_into_view_if_needed()
                dimensions = image.evaluate("async image => { await image.decode(); return [image.naturalWidth, image.naturalHeight]; }")
                self.assertTrue(all(dimension > 0 for dimension in dimensions))

            metrics = detail.locator(".trace-metric")
            self.assertEqual(metrics.count(), len(call["metrics"]))
            for metric, expected in zip(metrics.all(), call["metrics"]):
                self.assertEqual(metric.locator("dt").inner_text(), expected["label"])
                values = [float(value.strip()) for value in metric.locator("dd").inner_text().split("→")]
                expected_values = [expected["value"]] if "value" in expected else [expected["before"], expected["after"]]
                self.assertEqual(values, expected_values)

        # Preserve the recorded non-monotonic overlap and the neighboring-object update.
        self.page.locator("[data-trace-call='scale']").click()
        scale_iou = detail.locator(".trace-metric").filter(has_text="Spoon silhouette IoU").locator("dd")
        self.assertEqual(scale_iou.inner_text(), "0.36 → 0.30")
        self.page.locator("[data-trace-call='refine']").click()
        self.assertIn("carried along: napkin#0", detail.locator(".trace-response").inner_text())
        self.assertIn("not the final scene", self.page.locator("#trace-note").inner_text())

        # Keyboard navigation updates both selection and focus, and clamps at either end.
        self.page.locator("[data-trace-call='refine']").focus()
        for key, expected in [("Home", "inspect"), ("ArrowLeft", "inspect"), ("ArrowRight", "translate"), ("End", "refine"), ("ArrowRight", "refine"), ("ArrowLeft", "reinspect")]:
            self.page.keyboard.press(key)
            self.assertEqual(selected_id(), expected)
            self.assertEqual(self.page.evaluate("document.activeElement.dataset.traceCall"), expected)

        self.page.keyboard.press("Home")
        for expected in call_ids[1:]:
            following.click()
            self.assertEqual(selected_id(), expected)
        self.assertTrue(following.is_disabled())
        for expected in reversed(call_ids[:-1]):
            previous.click()
            self.assertEqual(selected_id(), expected)
        self.assertTrue(previous.is_disabled())

        for width in [1440, 390]:
            self.page.set_viewport_size({"width": width, "height": 1000})
            self.page.locator("[data-trace-call='scale']").click()
            self.page.locator("#pose-trace").screenshot(path=str(OPTIONS.screenshots / f"pose-trace-{width}.png"), style=".site-header { visibility: hidden !important; }")


if __name__ == "__main__":
    unittest.main(argv=[__file__, *UNITTEST_ARGS], verbosity=2)
