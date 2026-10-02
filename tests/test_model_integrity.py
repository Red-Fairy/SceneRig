"""Verify compressed models reproduce the original GLB bytes exactly."""

import gzip
import hashlib
import json
from pathlib import Path
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ModelIntegrityTests(unittest.TestCase):
    def test_every_model_matches_its_original_hash(self):
        gallery = json.loads((ROOT / "data/gallery.json").read_text())
        count = 0
        for scene in gallery["scenes"]:
            for method in scene["methods"]:
                model = method["model"]
                with self.subTest(source=model["src"]):
                    packed = (ROOT / model["src"]).read_bytes()
                    self.assertEqual(len(packed), model["bytes"])
                    raw = gzip.decompress(packed) if model["src"].endswith(".gz") else packed
                    magic, version, size = struct.unpack("<4sII", raw[:12])
                    self.assertEqual((magic, version, size), (b"glTF", 2, len(raw)))
                    if model["src"].endswith(".gz"):
                        self.assertEqual(len(raw), model["uncompressedBytes"])
                        self.assertEqual(hashlib.sha256(raw).hexdigest(), model["sha256"])
                    count += 1
        self.assertEqual(count, 192)


if __name__ == "__main__":
    unittest.main()
