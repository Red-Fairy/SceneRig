"""Exercise the teaser-only modal against a served project page."""
import argparse
import os
import unittest

from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--base-url', default='http://127.0.0.1:8765')
parser.add_argument('--browser', default=os.environ.get('CHROMIUM_EXECUTABLE'))
OPTIONS, ARGS = parser.parse_known_args()


class TeaserViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        args = {'headless': True, 'args': ['--no-sandbox']}
        if OPTIONS.browser:
            args['executable_path'] = OPTIONS.browser
        cls.browser = cls.pw.chromium.launch(**args)

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.context = self.browser.new_context(viewport={'width': 1440, 'height': 1000}, reduced_motion='reduce')
        self.page = self.context.new_page()
        self.errors = []
        self.requests = []
        self.page.on('pageerror', lambda e: self.errors.append(str(e)))
        self.page.on('request', lambda r: self.requests.append(r.url))
        self.page.goto(OPTIONS.base_url)
        self.page.wait_for_function('!document.querySelector("#simulation-source").disabled')

    def tearDown(self):
        self.context.close()
        self.assertEqual(self.errors, [])

    def open_viewer(self):
        self.page.locator('#simulation-viewer').click()
        self.page.wait_for_selector('#teaser-viewer[open]')

    def wait_for_model(self):
        self.page.wait_for_function('''() =>
            document.querySelector('#teaser-viewer-canvas model-viewer')?.loaded &&
            document.querySelector('#teaser-viewer-status').hidden''', timeout=90000)

    def test_all_methods_use_headline_scenes_without_navigation(self):
        self.assertFalse(any('.glb' in url or 'model-viewer.min.js' in url for url in self.requests))
        original_url = self.page.url
        for method, model in [('scenerig', 'ours'), ('viga', 'viga'), ('simfoundry', 'simfoundry')]:
            self.page.select_option('#simulation-source', method)
            self.open_viewer()
            self.wait_for_model()
            self.assertEqual(self.page.url, original_url)
            self.assertEqual(len(self.context.pages), 1)
            self.assertTrue(self.page.locator('#teaser-viewer-canvas model-viewer').evaluate("e => e.src.startsWith('blob:')"))
            self.assertTrue(any(url.endswith(f'assets/models/main/scene-01/{model}.glb.gz') for url in self.requests))
            self.assertIn('Claude Opus 5', self.page.locator('.teaser-viewer-heading').text_content())
            self.assertEqual(self.page.locator('#teaser-viewer select').count(), 0)
            self.page.locator('#teaser-viewer-reset').click()
            self.page.locator('#teaser-viewer-close').click()
            self.page.wait_for_selector('#teaser-viewer-canvas model-viewer', state='detached')
            self.assertEqual(self.page.locator('#teaser-viewer-canvas model-viewer').count(), 0)
            self.assertTrue(self.page.locator('#simulation-viewer').evaluate('e => e === document.activeElement'))
        # Reconstruction-gallery links still lead to the standalone viewer.
        self.assertTrue(self.page.locator('.scene-viewer-link').first.get_attribute('href').startswith('viewer.html?'))

    def test_modal_dismissal_playback_and_mobile_layout(self):
        self.page.locator('#simulation-play').scroll_into_view_if_needed()
        self.page.wait_for_function('!document.querySelector("#simulation-play").disabled')
        self.page.locator('#simulation-play').click()
        self.page.wait_for_selector('#simulation-status.playing')
        self.open_viewer()
        self.page.wait_for_function('document.querySelector("#simulation-rgb").paused')
        self.page.keyboard.press('Escape')
        self.page.wait_for_function('!document.querySelector("#teaser-viewer").open')
        self.page.wait_for_selector('#simulation-status.playing')
        self.open_viewer()
        self.wait_for_model()
        for width in [1440, 768, 390, 320]:
            self.page.set_viewport_size({'width': width, 'height': 850})
            dialog = self.page.locator('#teaser-viewer')
            rect = dialog.bounding_box()
            self.assertGreaterEqual(rect['x'], 0)
            self.assertLessEqual(rect['x'] + rect['width'], width)
            self.assertTrue(dialog.evaluate('e => e.scrollWidth <= e.clientWidth'))
            self.assertEqual(self.page.locator('body').evaluate('e => getComputedStyle(e).overflow'), 'hidden')
            self.page.keyboard.press('Tab')
            self.assertTrue(dialog.evaluate('e => e.contains(document.activeElement)'))
        self.page.mouse.click(2, 2)
        self.page.wait_for_function('!document.querySelector("#teaser-viewer").open')

    def test_failed_model_can_be_retried(self):
        pattern = '**/assets/models/main/scene-01/ours.glb.gz*'
        self.page.route(pattern, lambda route: route.abort())
        self.open_viewer()
        self.page.wait_for_selector('#teaser-viewer-retry:not([hidden])', timeout=60000)
        self.assertIn('could not load', self.page.locator('#teaser-viewer-status').inner_text())
        self.page.unroute(pattern)
        self.page.locator('#teaser-viewer-retry').click()
        self.wait_for_model()

    def test_closing_during_download_and_reopening(self):
        pattern = '**/assets/models/main/scene-01/ours.glb.gz*'
        pending = []
        self.page.route(pattern, lambda route: pending.append(route))
        self.open_viewer()
        self.page.wait_for_timeout(500)
        self.page.locator('#teaser-viewer-close').click()
        for route in pending:
            route.abort()
        self.page.unroute(pattern)
        self.open_viewer()
        self.wait_for_model()


if __name__ == '__main__':
    unittest.main(argv=[__file__, *ARGS])
