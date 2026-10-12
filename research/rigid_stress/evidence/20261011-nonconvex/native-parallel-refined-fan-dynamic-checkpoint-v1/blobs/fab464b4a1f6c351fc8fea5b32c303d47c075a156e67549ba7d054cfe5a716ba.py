"""Allocated-CPU checkpoint writer preserving raw, stored and decompressed hashes."""
import gzip
import hashlib
import json
import os
from pathlib import Path


def publish(root, destination, runs, extras, summary, limitations):
    assert not destination.exists()
    files = list(extras)
    for run in runs:
        for current, directories, names in os.walk(run):
            directories[:] = [name for name in directories if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
            files.extend(Path(current) / name for name in names)
        for line in (run / "source.sha256").read_text().splitlines():
            path = Path(line.split("  ", 1)[1])
            if path.is_absolute() and path.parent == root:
                files.append(path)
    destination.mkdir(parents=True)
    entries, blobs = [], set()
    for original in sorted(set(files)):
        content = original.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        compress = original.suffix in {".json", ".log"} and len(content) >= 200000
        stored = gzip.compress(content, mtime=0) if compress else content
        artifact = destination / "blobs" / (digest + original.suffix + (".gz" if compress else ""))
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(stored)
        blobs.add(artifact)
        entries.append({"original": str(original), "run_path": str(original.relative_to(root)),
                        "artifact": str(artifact.relative_to(destination)), "encoding": "gzip" if compress else "identity",
                        "bytes": len(content), "sha256": digest, "stored_bytes": len(stored),
                        "stored_sha256": hashlib.sha256(stored).hexdigest()})
    manifest = destination / "manifest.json"
    manifest.write_text(json.dumps({"scope": __doc__, "summary": summary, "limitations": limitations,
                                    "files": entries}, indent=2) + "\n")
    for row in entries:
        assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
        content = (destination / row["artifact"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
        restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
        assert hashlib.sha256(restored).hexdigest() == row["sha256"]
    print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs),
                      "stored_bytes": sum(path.stat().st_size for path in blobs),
                      "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      "summary": summary}, indent=2), flush=True)
