"""Restore one compact unsharded run from the practical evidence manifest into data storage."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.run and "/" not in args.run
    manifest = json.loads(args.manifest.read_text())
    selected = {}
    for row in manifest["files"]:
        parts = PurePosixPath(row["original"]).parts
        if args.run not in parts:
            continue
        relative = Path(*parts[parts.index(args.run) + 1 :])
        assert relative.parts and not relative.is_absolute() and ".." not in relative.parts
        assert not relative.name.endswith(".chunks.json"), "This helper restores compact unsharded runs only"
        if str(relative) in selected:
            assert selected[str(relative)]["sha256"] == row["sha256"]
        selected[str(relative)] = row
    assert selected, f"No archived files for {args.run}"
    assert not args.output.exists(), args.output
    args.output.mkdir(parents=True)
    total = 0
    for relative, row in sorted(selected.items()):
        artifact = args.manifest.parent / row["artifact"]
        stored = artifact.read_bytes()
        assert hashlib.sha256(stored).hexdigest() == row["stored_sha256"]
        assert row["encoding"] in {"gzip", "identity"}
        content = gzip.decompress(stored) if row["encoding"] == "gzip" else stored
        assert len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        total += len(content)
    before = args.output / "source.sha256"
    if before.exists():
        assert before.read_bytes() == (args.output / "source-after.sha256").read_bytes()
        for line in before.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            assert hashlib.sha256((args.output / "source" / relative).read_bytes()).hexdigest() == expected
    print(
        json.dumps(
            {
                "restored_run": args.run,
                "output": str(args.output),
                "files": len(selected),
                "bytes": total,
                "all_original_file_hashes_verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
