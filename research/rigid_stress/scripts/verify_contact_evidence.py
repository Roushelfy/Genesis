"""Stream-verify publication hashes and optionally the preserved runtime originals."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
            size += len(block)
    return hasher.hexdigest(), size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--originals", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    count = 0
    for entry in manifest["artifacts"]:
        actual = digest(args.manifest.parent / entry["artifact"])
        assert actual == (entry["sha256"], entry["bytes"]), entry["artifact"]
        if args.originals:
            actual = digest(Path(entry["runtime_path"]))
            assert actual == (entry["original_sha256"], entry["original_bytes"]), entry["runtime_path"]
        count += 1
    result = {
        "manifest_sha256": digest(args.manifest)[0],
        "verified_artifacts": count,
        "verified_originals": count if args.originals else 0,
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
