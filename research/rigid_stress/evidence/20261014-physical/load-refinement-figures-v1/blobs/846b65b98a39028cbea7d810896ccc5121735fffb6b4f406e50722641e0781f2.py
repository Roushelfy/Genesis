"""Stream-check immutable publications, runtime originals and ordered JSON chunks."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


def digest(path, compressed=False):
    hasher = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else Path.open
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def verify_manifest(path, originals, cache, repair_missing=False, allow_appended_allocation_logs=False):
    manifest = json.loads(path.read_text())
    if "files" in manifest:
        entries = [
            {"encoding": "identity", "stored_sha256": entry["sha256"], "stored_bytes": entry["bytes"], **entry}
            for entry in manifest["files"]
        ]
    else:
        entries = []
        for entry in manifest["artifacts"]:
            relative = entry["path"] if "path" in entry else entry["file"]
            artifact = path.parent / relative
            size = entry["bytes"] if "bytes" in entry else artifact.stat().st_size
            entries.append(
                {
                    "artifact": relative,
                    "original": entry["original"] if "original" in entry else entry["raw"],
                    "encoding": "identity",
                    "sha256": entry["sha256"],
                    "bytes": size,
                    "stored_sha256": entry["sha256"],
                    "stored_bytes": size,
                }
            )
    repairs = []
    appended_logs = []
    for entry in entries:
        artifact = path.parent / entry["artifact"]
        assert artifact.resolve().is_relative_to(path.parent.resolve()), artifact
        assert entry["encoding"] in ("identity", "gzip"), entry
        expected_stored = (entry["stored_sha256"], entry["stored_bytes"])
        if repair_missing and not artifact.exists():
            # Restore only byte-identical, uncompressed publication files.
            # Compressed representations need their original publisher.
            assert entry["encoding"] == "identity", entry
            original = Path(entry["original"])
            assert digest(original) == expected_stored, original
            artifact.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, artifact)
            repairs.append({"artifact": str(artifact), "original": str(original), "sha256": expected_stored[0]})
            print(json.dumps({"restored_missing_artifact": repairs[-1]}), flush=True)
        key = (artifact, False)
        if key not in cache:
            cache[key] = digest(artifact)
        assert cache[key] == expected_stored, artifact
        key = (artifact, entry["encoding"] == "gzip")
        if key not in cache:
            cache[key] = digest(artifact, compressed=key[1])
        assert cache[key] == (entry["sha256"], entry["bytes"]), artifact
        if originals:
            original = Path(entry["original"])
            key = (original, False)
            if key not in cache:
                cache[key] = digest(original)
            if cache[key] != (entry["sha256"], entry["bytes"]):
                assert (
                    allow_appended_allocation_logs
                    and "allocation" in original.name.lower()
                    and original.suffix == ".log"
                ), original
                assert cache[key][1] > entry["bytes"], original
                opener = gzip.open if entry["encoding"] == "gzip" else Path.open
                with opener(artifact, "rb") as archived, original.open("rb") as current:
                    for block in iter(lambda: archived.read(8 * 1024 * 1024), b""):
                        assert current.read(len(block)) == block, original
                appended_logs.append(
                    {
                        "original": str(original),
                        "archived_sha256": entry["sha256"],
                        "archived_bytes": entry["bytes"],
                        "current_sha256": cache[key][0],
                        "current_bytes": cache[key][1],
                        "archived_content_is_exact_prefix": True,
                    }
                )
    chunks_verified = 0
    for index in manifest.get("summary", {}).get("sharded_original_files", []):
        complete = hashlib.sha256()
        offset = 0
        for part in index["parts"]:
            assert part["offset"] == offset, part
            candidates = [
                entry
                for entry in entries
                if Path(entry["original"]).name == part["file"]
                and Path(entry["original"]).parent.name == Path(index["original"]).parent.name
            ]
            # Several scopes share result.json names. The recorded content
            # hash identifies the complete original byte range unambiguously.
            candidates = [entry for entry in candidates if entry["sha256"] == part["stored_sha256"]]
            assert candidates, part
            artifact = path.parent / candidates[0]["artifact"]
            hasher = hashlib.sha256()
            size = 0
            with gzip.open(artifact, "rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    complete.update(block)
                    hasher.update(block)
                    size += len(block)
            assert (hasher.hexdigest(), size) == (part["sha256"], part["bytes"]), part
            offset += size
        assert (complete.hexdigest(), offset) == (index["sha256"], index["bytes"]), index["original"]
        if originals:
            original = Path(index["original"])
            key = (original, False)
            if key not in cache:
                cache[key] = digest(original)
            assert cache[key] == (index["sha256"], index["bytes"]), original
        chunks_verified += 1
    return {
        "manifest": str(path),
        "manifest_sha256": digest(path)[0],
        "verified_entries": len(entries),
        "verified_originals": len(entries) - len(appended_logs) if originals else 0,
        "verified_original_prefixes": len(appended_logs),
        "appended_allocation_logs": appended_logs,
        "reconstructed_complete_json_files": chunks_verified,
        "restored_missing_artifacts": repairs,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--originals", action="store_true")
    parser.add_argument(
        "--repair-missing", action="store_true", help="Restore missing identity files from verified originals."
    )
    parser.add_argument(
        "--allow-appended-allocation-logs",
        action="store_true",
        help="Verify exact archived prefixes of completed allocation logs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = {}
    results = []
    for path in args.manifest:
        result = verify_manifest(path, args.originals, cache, args.repair_missing, args.allow_appended_allocation_logs)
        results.append(result)
        print(json.dumps(result), flush=True)
    output = {
        "passed": True,
        "checkpoints": results,
        "verified_entries": sum(row["verified_entries"] for row in results),
        "verified_originals": sum(row["verified_originals"] for row in results),
        "verified_original_prefixes": sum(row["verified_original_prefixes"] for row in results),
        "reconstructed_complete_json_files": sum(row["reconstructed_complete_json_files"] for row in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
