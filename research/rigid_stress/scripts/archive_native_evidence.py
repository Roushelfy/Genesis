"""Archive selected completed native artifacts without duplicated printed JSON."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()
    manifest_path = args.destination / "optimization-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {item["runtime_name"]: item for item in manifest["files"]}
    for name in args.names:
        source = args.run / name
        data = source.read_bytes()
        archived = name
        representation = "Complete original artifact."
        if name.endswith(".log") and len(data) > 1000000:
            text = data.decode()
            position = text.find('\n{\n  "scope"')
            if position >= 0:
                data = (text[:position] + "\n").encode()
                archived += ".preamble.txt"
                representation = "Exact preamble; duplicated printed JSON is archived in its original JSON file."
        if len(data) > 1000000:
            data = gzip.compress(data, compresslevel=9, mtime=0)
            archived += ".gz"
        relative = "raw/" + archived
        (args.destination / relative).write_bytes(data)
        original = source.read_bytes()
        entries[name] = {
            "runtime_name": name,
            "artifact": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "original_bytes": len(original),
            "archived_bytes": len(data),
            "source_revision": args.revision,
            "representation": representation,
        }
    manifest["files"] = list(entries.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
