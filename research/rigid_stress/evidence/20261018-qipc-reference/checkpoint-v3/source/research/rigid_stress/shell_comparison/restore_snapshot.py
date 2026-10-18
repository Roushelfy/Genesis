"""Restore one exact analysis snapshot from the compact evidence archive, without extraction of arbitrary paths."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as archive:
        entries = json.load(archive.extractfile("analysis-index.json"))[args.campaign]
        for entry in entries:
            relative = Path(entry["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            payload = archive.extractfile(f"analysis-blobs/{entry['sha256']}").read()
            assert len(payload) == entry["bytes"] and hashlib.sha256(payload).hexdigest() == entry["sha256"]
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)


if __name__ == "__main__":
    main()
