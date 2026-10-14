"""Publish exact source/results bytes with bounded, reconstructible JSON chunks."""

import argparse
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
            directories[:] = [
                name for name in directories if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}
            ]
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
        entries.append(
            {
                "original": str(original),
                "run_path": str(original.relative_to(root)),
                "artifact": str(artifact.relative_to(destination)),
                "encoding": "gzip" if compress else "identity",
                "bytes": len(content),
                "sha256": digest,
                "stored_bytes": len(stored),
                "stored_sha256": hashlib.sha256(stored).hexdigest(),
            }
        )
    manifest = destination / "manifest.json"
    manifest.write_text(
        json.dumps({"scope": __doc__, "summary": summary, "limitations": limitations, "files": entries}, indent=2)
        + "\n"
    )
    for row in entries:
        assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
        content = (destination / row["artifact"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
        restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
        assert hashlib.sha256(restored).hexdigest() == row["sha256"]
    print(
        json.dumps(
            {
                "entries": len(entries),
                "unique_blobs": len(blobs),
                "stored_bytes": sum(path.stat().st_size for path in blobs),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "summary": summary,
            },
            indent=2,
        ),
        flush=True,
    )


def publish_bounded(root, destination, runs, extras, summary, limitations, staging_root):
    stage = staging_root / destination.name
    assert not stage.exists()
    stage.mkdir(parents=True)
    staged_runs, indexes = [], []
    for run in runs:
        staged = stage / run.relative_to(root)
        for current, directories, names in os.walk(run):
            directories[:] = [
                name for name in directories if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}
            ]
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
                        parts.append(
                            {
                                "file": path.name,
                                "offset": offset,
                                "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(),
                                "stored_sha256": hashlib.sha256(stored).hexdigest(),
                            }
                        )
                        offset += len(data)
                index = {
                    "original": str(original),
                    "bytes": offset,
                    "sha256": digest.hexdigest(),
                    "encoding": "concatenated-decompressed-gzip-byte-ranges",
                    "parts": parts,
                }
                index_path = target.with_name(target.name + ".chunks.json")
                index_path.write_text(json.dumps(index, indent=2) + "\n")
                indexes.append(index_path)
        staged_runs.append(staged)
    summary = {**summary, "sharded_original_files": [json.loads(path.read_text()) for path in indexes]}
    extras = list(extras)
    publish(
        root,
        destination,
        staged_runs,
        extras,
        summary,
        limitations
        + " JSON files above200MB are preserved as exact ordered32MiB byte ranges, individually gzip-compressed, with original/part/stored SHA256 and offsets. Join decompressed ranges in index order to restore the original JSON exactly; no records are reduced or excluded. Runtime staging uses hard links for unchanged small originals.",
    )
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--extra", type=Path, nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limitations", required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    publish_bounded(
        args.runtime_root,
        args.destination,
        args.run,
        [*args.extra, args.summary],
        summary,
        args.limitations,
        args.staging_root,
    )


if __name__ == "__main__":
    main()
