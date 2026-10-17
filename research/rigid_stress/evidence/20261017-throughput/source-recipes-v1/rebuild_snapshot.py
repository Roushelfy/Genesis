"""Rebuild exact benchmark source bytes from a committed tree and archived overrides using stdlib only."""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        parser.error("Output directory and report must both be fresh paths.")
    recipe = json.loads((args.recipe / "recipe.json").read_text())
    manifest = json.loads((args.recipe / "snapshot.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in paths):
        parser.error("Snapshot source paths must remain relative to the output directory.")
    overrides = set(recipe["overrides"])
    assert overrides <= paths
    args.output.mkdir(parents=True)
    command = [
        "git",
        "archive",
        "--format=tar",
        recipe["reconstruction_base_revision"],
        "--",
        *sorted(paths - overrides),
    ]
    process = subprocess.Popen(command, cwd=args.repository, stdout=subprocess.PIPE)
    restored = set()
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            if member.isfile():
                assert member.name in paths - overrides
                target = args.output / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                target.chmod(member.mode)
                restored.add(member.name)
            else:
                assert member.isdir()
    assert process.wait() == 0
    assert restored == paths - overrides
    for relative in overrides:
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.recipe / "overrides" / relative, target)
    for row in manifest["files"]:
        restored = args.output / row["path"]
        assert restored.stat().st_size == row["bytes"]
        assert hashlib.sha256(restored.read_bytes()).hexdigest() == row["sha256"], row["path"]
    for name in ("snapshot.json", "snapshot-files.sha256"):
        shutil.copy2(args.recipe / name, args.output / name)
    actual = {str(path.relative_to(args.output)) for path in args.output.rglob("*") if path.is_file()}
    assert actual == paths | {"snapshot.json", "snapshot-files.sha256"}
    result = {
        "scope": __doc__,
        "command": command,
        "reconstruction_base_revision": recipe["reconstruction_base_revision"],
        "recipe_sha256": hashlib.sha256((args.recipe / "recipe.json").read_bytes()).hexdigest(),
        "snapshot_manifest_sha256": hashlib.sha256((args.output / "snapshot-files.sha256").read_bytes()).hexdigest(),
        "source_files_verified": len(paths),
        "extra_files": [],
        "missing_files": [],
        "passed": True,
        "goal_completion": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"source_files_verified": len(paths), "passed": True}))


if __name__ == "__main__":
    main()
