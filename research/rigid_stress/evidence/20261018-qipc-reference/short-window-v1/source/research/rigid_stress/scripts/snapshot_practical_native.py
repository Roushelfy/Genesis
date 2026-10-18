"""Copy a self-contained native application snapshot, or verify its exact source bytes."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[3])
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path)
    target.add_argument("--verify-against", type=Path, help="Existing snapshot.json; do not copy files.")
    args = parser.parse_args()
    root = args.source_root.resolve()
    files = {}
    for directory in ["genesis", "examples/rigid", "examples/speed_benchmark"]:
        for current, directories, names in os.walk(root / directory):
            directories[:] = [
                name for name in directories if name not in {"__pycache__", ".pytest_cache", "cache", ".git"}
            ]
            for name in names:
                source = Path(current) / name
                assert not source.is_symlink(), f"Unexpected source symlink: {source}"
                files[str(source.relative_to(root))] = source
    for relative in [
        "pyproject.toml",
        "research/rigid_stress/scripts/run_practical_native_benchmark.sh",
        "research/rigid_stress/scripts/cluster_env.sh",
        "research/rigid_stress/docs/NATIVE_API.md",
    ]:
        files[relative] = root / relative
    patch = subprocess.check_output(["git", "diff", "HEAD", "--", "genesis", "examples"], cwd=root)
    rows = []
    for relative, source in sorted(files.items()):
        assert source.is_file()
        rows.append(
            {
                "path": relative,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    rows.append({"path": "source.patch", "bytes": len(patch), "sha256": hashlib.sha256(patch).hexdigest()})
    rows.sort(key=lambda row: row["path"])
    if args.verify_against is not None:
        previous = json.loads(args.verify_against.read_text())
        expected = {row["path"]: (row["bytes"], row["sha256"]) for row in previous["files"]}
        actual = {row["path"]: (row["bytes"], row["sha256"]) for row in rows}
        assert actual == expected, {
            "added": sorted(actual.keys() - expected.keys()),
            "missing": sorted(expected.keys() - actual.keys()),
            "changed": sorted(path for path in actual.keys() & expected.keys() if actual[path] != expected[path]),
        }
        print(
            json.dumps(
                {
                    "source_files_match_existing_snapshot_exactly": True,
                    "files": len(rows),
                    "bytes": sum(row["bytes"] for row in rows),
                    "manifest": str(args.verify_against),
                }
            )
        )
        return
    output = args.output.resolve()
    assert not output.exists(), output
    output.mkdir(parents=True)
    for relative, source in files.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / "source.patch").write_bytes(patch)
    for row in rows:
        assert hashlib.sha256((output / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    manifest = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_root": str(root),
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "files": rows,
        "scope": __doc__,
        "goal_completion": False,
    }
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    hashes = "".join(row["sha256"] + "  " + row["path"] + "\n" for row in rows)
    (output / "snapshot-files.sha256").write_text(hashes)
    print(
        json.dumps(
            {
                "snapshot": str(output),
                "files": len(rows),
                "bytes": sum(row["bytes"] for row in rows),
                "hash_manifest_sha256": hashlib.sha256(hashes.encode()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
