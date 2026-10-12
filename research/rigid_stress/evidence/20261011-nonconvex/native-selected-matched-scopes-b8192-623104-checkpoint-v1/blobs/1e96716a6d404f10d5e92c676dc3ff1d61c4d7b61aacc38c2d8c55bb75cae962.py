"""Publish exact large result bytes in bounded gzip chunks with reconstruction hashes."""

import gzip
import hashlib
import json
import os
from pathlib import Path

from publish_native_checkpoint_v1 import publish


def publish_bounded(root, destination, runs, extras, summary, limitations):
    stage = root / "bounded-publication" / destination.name
    assert not stage.exists()
    stage.mkdir(parents=True)
    staged_runs, indexes = [], []
    for run in runs:
        staged = stage / run.relative_to(root)
        for current, directories, names in os.walk(run):
            directories[:] = [name for name in directories if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
            for name in names:
                original = Path(current) / name
                target = staged / original.relative_to(run)
                target.parent.mkdir(parents=True, exist_ok=True)
                if original.suffix != ".json" or original.stat().st_size < 200_000_000:
                    os.link(original, target)
                    continue
                parts, digest, offset = [], hashlib.sha256(), 0
                with original.open("rb") as stream:
                    while data := stream.read(32 * 1024 * 1024):
                        digest.update(data)
                        path = target.with_name(target.name + f".part{len(parts):04d}.gz")
                        stored = gzip.compress(data, mtime=0)
                        path.write_bytes(stored)
                        assert gzip.decompress(path.read_bytes()) == data
                        parts.append({"file": path.name, "offset": offset, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest(),
                                      "stored_sha256": hashlib.sha256(stored).hexdigest()})
                        offset += len(data)
                index = {"original": str(original), "bytes": offset, "sha256": digest.hexdigest(),
                         "encoding": "concatenated-decompressed-gzip-byte-ranges", "parts": parts}
                index_path = target.with_name(target.name + ".chunks.json")
                index_path.write_text(json.dumps(index, indent=2) + "\n")
                indexes.append(index_path)
        staged_runs.append(staged)
    summary = {**summary, "sharded_original_files": [json.loads(path.read_text()) for path in indexes]}
    extras = [*extras, Path(__file__)]
    publish(root, destination, staged_runs, extras, summary,
            limitations + " JSON files above200MB are preserved as exact ordered32MiB byte ranges, individually gzip-compressed, with original/part/stored SHA256 and offsets. Join decompressed ranges in index order to restore the original JSON exactly; no records are reduced or excluded. Runtime staging uses hard links for unchanged small originals.")
    manifest = json.loads((destination / "manifest.json").read_text())
    artifacts = {row["original"]: row for row in manifest["files"]}
    for index_path in indexes:
        index = json.loads(index_path.read_text())
        digest, offset = hashlib.sha256(), 0
        for part in index["parts"]:
            row = artifacts[str(index_path.parent / part["file"])]
            assert row["encoding"] == "identity"
            stored = (destination / row["artifact"]).read_bytes()
            assert hashlib.sha256(stored).hexdigest() == part["stored_sha256"]
            data = gzip.decompress(stored)
            assert hashlib.sha256(data).hexdigest() == part["sha256"] and len(data) == part["bytes"]
            assert offset == part["offset"]
            digest.update(data)
            offset += len(data)
        assert offset == index["bytes"] and digest.hexdigest() == index["sha256"]
    print("Published chunk reconstruction verified against original complete JSON bytes.", flush=True)
