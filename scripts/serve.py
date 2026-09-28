#!/usr/bin/env python3
"""Preview the static site with byte-range support for native video seeking.

Uses Python's normal static-file handler for paths, directories, MIME types,
HEAD, and conditional requests. Adds single byte ranges; unsupported units and
multipart ranges fall back to a complete response (RFC 9110, section 14.2).
"""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re


class RangeRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        self.byte_range = None
        requested = self.headers.get("Range", "")
        path = self.translate_path(self.path)
        # Range applies only to GET. Retain the standard handler's cache logic.
        if (self.command != "GET" or not requested.startswith("bytes=")
                or "," in requested or not os.path.isfile(path)
                or "If-Modified-Since" in self.headers or "If-None-Match" in self.headers):
            return super().send_head()
        stat = os.stat(path)
        modified = self.date_time_string(stat.st_mtime)
        if "If-Range" in self.headers and self.headers["If-Range"] != modified:
            return super().send_head()
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", requested.strip())
        start, end = 0, -1
        if match and any(match.groups()):
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last) if last else stat.st_size - 1, stat.st_size - 1)
            elif int(last) > 0:
                start = max(0, stat.st_size - int(last))
                end = stat.st_size - 1
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{stat.st_size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        source = open(path, "rb")
        source.seek(start)
        self.byte_range = (start, end)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", modified)
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        # Browsers routinely cancel the old transfer when users seek a video.
        try:
            if self.byte_range is None:
                return super().copyfile(source, outputfile)
            remaining = self.byte_range[1] - self.byte_range[0] + 1
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    handler = partial(RangeRequestHandler, directory=str(args.directory.resolve()))
    with ThreadingHTTPServer((args.bind, args.port), handler) as server:
        print(f"Preview: http://{args.bind}:{server.server_port}", flush=True)
        server.serve_forever()
