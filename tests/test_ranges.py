"""HTTP integration tests for the local preview server; no external dependencies."""

from functools import partial
from http.client import HTTPConnection
import importlib.util
from pathlib import Path
import tempfile
from threading import Thread
from http.server import ThreadingHTTPServer
import unittest


spec = importlib.util.spec_from_file_location("preview_server", Path(__file__).resolve().parents[1] / "scripts/serve.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QuietHandler(module.RangeRequestHandler):
    def log_message(self, *args):
        pass


class ByteRangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.content = bytes(range(256)) * 1024
        (Path(cls.folder.name) / "video.mp4").write_bytes(cls.content)
        (Path(cls.folder.name) / "empty.mp4").write_bytes(b"")
        (Path(cls.folder.name) / "index.html").write_text("<h1>Preview</h1>")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=cls.folder.name))
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.folder.cleanup()

    def request(self, headers=None, method="GET", path="/video.mp4"):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse()
        result = response.status, response.headers, response.read()
        connection.close()
        return result

    def test_full_response_and_head(self):
        for method in ["GET", "HEAD"]:
            with self.subTest(method=method):
                status, headers, body = self.request(method=method)
                self.assertEqual(status, 200)
                self.assertEqual(int(headers["Content-Length"]), len(self.content))
                self.assertEqual(headers["Accept-Ranges"], "bytes")
                self.assertEqual(headers["Content-Type"], "video/mp4")
                self.assertEqual(body, self.content if method == "GET" else b"")
        status, headers, body = self.request({"Range": "bytes=20-29"}, method="HEAD")
        self.assertEqual(status, 200)  # RFC 9110 defines Range only for GET.
        self.assertEqual(int(headers["Content-Length"]), len(self.content))
        self.assertEqual(body, b"")

    def test_single_open_and_suffix_ranges(self):
        total = len(self.content)
        cases = [("bytes=20-29", 20, 29), ("bytes=100000-", 100000, total - 1),
                 ("bytes=-100", total - 100, total - 1), ("bytes=-999999", 0, total - 1),
                 ("bytes=262140-999999", 262140, total - 1), ("bytes=0-0", 0, 0)]
        for requested, start, end in cases:
            with self.subTest(requested=requested):
                status, headers, body = self.request({"Range": requested})
                self.assertEqual(status, 206)
                self.assertEqual(headers["Content-Range"], f"bytes {start}-{end}/{total}")
                self.assertEqual(int(headers["Content-Length"]), end - start + 1)
                self.assertEqual(body, self.content[start:end + 1])

    def test_invalid_and_unsatisfiable_ranges(self):
        for requested in ["bytes=999999-", "bytes=10-5", "bytes=-0", "bytes=-", "bytes=abc"]:
            with self.subTest(requested=requested):
                status, headers, body = self.request({"Range": requested})
                self.assertEqual(status, 416)
                self.assertEqual(headers["Content-Range"], f"bytes */{len(self.content)}")
                self.assertEqual(headers["Content-Length"], "0")
                self.assertEqual(body, b"")
        status, headers, body = self.request({"Range": "bytes=0-"}, path="/empty.mp4")
        self.assertEqual(status, 416)
        self.assertEqual(headers["Content-Range"], "bytes */0")

    def test_unsupported_ranges_use_complete_response(self):
        for requested in ["bytes=0-3,10-14", "items=0-3"]:
            status, headers, body = self.request({"Range": requested})
            self.assertEqual(status, 200)
            self.assertEqual(body, self.content)

    def test_conditional_and_normal_static_handling(self):
        _, headers, _ = self.request(method="HEAD")
        modified = headers["Last-Modified"]
        status, _, body = self.request({"Range": "bytes=0-3", "If-Range": modified})
        self.assertEqual((status, body), (206, self.content[:4]))
        status, _, body = self.request({"Range": "bytes=0-3", "If-Range": "Wed, 01 Jan 2020 00:00:00 GMT"})
        self.assertEqual((status, body), (200, self.content))
        status, _, body = self.request({"Range": "bytes=0-3", "If-Modified-Since": modified})
        self.assertEqual((status, body), (304, b""))
        status, _, body = self.request(path="/")
        self.assertEqual((status, body), (200, b"<h1>Preview</h1>"))
        status, _, _ = self.request({"Range": "bytes=0-3"}, path="/missing.mp4")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
