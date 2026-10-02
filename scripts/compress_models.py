"""Losslessly gzip gallery GLBs and update their download metadata."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = ROOT / "data/gallery.json"
    gallery = json.loads(manifest.read_text())
    originals = []
    before = after = 0
    for scene in gallery["scenes"]:
        for method in scene["methods"]:
            model = method["model"]
            source = ROOT / model["src"]
            if source.suffix == ".gz":
                continue
            raw = source.read_bytes()
            packed = gzip.compress(raw, compresslevel=9, mtime=0)
            if len(packed) >= len(raw):
                continue
            assert gzip.decompress(packed) == raw
            target = source.with_suffix(source.suffix + ".gz")
            target.write_bytes(packed)
            assert gzip.decompress(target.read_bytes()) == raw
            model.update(src=target.relative_to(ROOT).as_posix(), bytes=len(packed),
                         uncompressedBytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            originals.append(source)
            before += len(raw)
            after += len(packed)
    manifest.write_text(json.dumps(gallery, indent=2, ensure_ascii=False) + "\n")
    for source in originals:
        source.unlink()
    print(f"{len(originals)} models: {before:,} -> {after:,} bytes; saved {before - after:,} bytes")


if __name__ == "__main__":
    main()
