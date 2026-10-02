"""Check the bundled slides at the GitHub Pages subpath.

Run with a Python environment containing Playwright and Chromium:
    python tests/test_slides.py
"""

from functools import partial
from http.server import ThreadingHTTPServer
import importlib.util
import os
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "test-results"
spec = importlib.util.spec_from_file_location("preview_server", ROOT / "scripts/serve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QuietHandler(module.RangeRequestHandler):
    def log_message(self, *args):
        pass


class SlidesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        (Path(cls.folder.name) / "SceneRig").symlink_to(ROOT, target_is_directory=True)
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), partial(QuietHandler, directory=cls.folder.name)
        )
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.addClassCleanup(cls.thread.join)
        cls.addClassCleanup(cls.server.server_close)
        cls.addClassCleanup(cls.server.shutdown)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}/SceneRig/"
        cls.playwright = sync_playwright().start()
        cls.addClassCleanup(cls.playwright.stop)
        options = {"headless": True, "args": ["--no-sandbox"]}
        if os.environ.get("CHROMIUM_EXECUTABLE"):
            options["executable_path"] = os.environ["CHROMIUM_EXECUTABLE"]
        cls.browser = cls.playwright.chromium.launch(**options)
        cls.addClassCleanup(cls.browser.close)
        OUTPUT.mkdir(exist_ok=True)

    def test_project_link_and_complete_presentation(self):
        context = self.browser.new_context(viewport={"width": 1440, "height": 1000})
        self.addCleanup(context.close)
        page = context.new_page()
        page.goto(self.base, wait_until="networkidle")
        link = page.locator('.hero-links a[href="slides/"]')
        self.assertEqual(link.count(), 1)
        for width in (1440, 390, 320):
            page.set_viewport_size({"width": width, "height": 1000})
            self.assertTrue(link.is_visible())
            self.assertTrue(page.evaluate("document.documentElement.scrollWidth <= innerWidth"))
            boxes = page.locator(".hero-links > *").evaluate_all(
                "elements => elements.map(e => {const r=e.getBoundingClientRect(); "
                "return {left:r.left,right:r.right,top:r.top,bottom:r.bottom};})"
            )
            for i, first in enumerate(boxes):
                for second in boxes[i + 1:]:
                    self.assertTrue(
                        first["right"] <= second["left"] + 1
                        or second["right"] <= first["left"] + 1
                        or first["bottom"] <= second["top"] + 1
                        or second["bottom"] <= first["top"] + 1
                    )
            page.screenshot(path=str(OUTPUT / f"slides-link-{width}.png"))

        errors = []
        context.on("page", lambda tab: tab.on("pageerror", lambda error: errors.append(str(error))))
        context.on("response", lambda response: errors.append(f"HTTP {response.status}: {response.url}") if response.status >= 400 else None)
        with page.expect_popup() as popup:
            link.click()
        deck = popup.value
        deck.set_viewport_size({"width": 1440, "height": 1000})
        deck.wait_for_selector("#startup-screen", state="hidden", timeout=180000)
        self.assertEqual(deck.url, self.base + "slides/")
        self.assertEqual(deck.locator("#counter").inner_text(), "1 / 29")
        self.assertEqual(deck.evaluate("SLIDES.length"), 29)
        requests = deck.evaluate("performance.getEntriesByType('resource').map(r => r.name)")
        self.assertTrue(all(url.startswith((self.base, "data:", "blob:")) for url in requests), requests)
        deck.locator("#next").click()
        deck.wait_for_function("current === 1")
        deck.keyboard.press("ArrowLeft")
        deck.wait_for_function("current === 0")

        videos = 0
        for number in range(1, 30):
            deck.evaluate("n => location.hash = String(n)", number)
            deck.wait_for_function("n => current === n - 1", arg=number)
            self.assertEqual(deck.locator("#counter").inner_text(), f"{number} / 29")
            self.assertGreater(deck.locator("#slide").evaluate("el => el.childElementCount"), 0)
            video = deck.locator("#slide video")
            if video.count():
                videos += 1
                video.evaluate("async v => {v.muted=true; v.currentTime=0; await v.play();}")
                deck.wait_for_function("document.querySelector('#slide video').currentTime > 0.1")
                self.assertTrue(video.evaluate("v => v.videoWidth > 0 && v.videoHeight > 0 && !v.error"))
                video.evaluate("v => {v.pause(); v.currentTime=Math.min(1, v.duration / 2);}")
                deck.wait_for_function("!document.querySelector('#slide video').seeking")
            if number in (1, 9, 13, 29):
                deck.wait_for_timeout(400)
                deck.screenshot(path=str(OUTPUT / f"slides-{number:02d}.png"))
        self.assertGreaterEqual(videos, 20)

        with deck.expect_popup() as popup:
            deck.locator("#audience-open").click()
        audience = popup.value
        audience.wait_for_selector("#startup-screen", state="hidden", timeout=180000)
        self.assertIn("/SceneRig/slides/?audience=1", audience.url)
        deck.evaluate("location.hash = '2'")
        audience.wait_for_function("current === 1")
        self.assertTrue(audience.locator("body").evaluate("el => el.classList.contains('audience')"))
        self.assertFalse(audience.locator(".notes").is_visible())

        for width in (390, 320):
            deck.set_viewport_size({"width": width, "height": 844})
            deck.wait_for_timeout(200)
            frame = deck.locator(".slide-frame").bounding_box()
            self.assertGreater(frame["width"], 0)
            self.assertGreaterEqual(frame["x"], -1)
            self.assertLessEqual(frame["x"] + frame["width"], width + 1)
            deck.screenshot(path=str(OUTPUT / f"slides-mobile-{width}.png"))
        self.assertEqual(errors, [])
        print(f"Verified 29 slides, {videos} video slides, audience sync, and mobile layout.")


if __name__ == "__main__":
    unittest.main()
