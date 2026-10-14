"""Reconstruct a regression's recorded clean Git source when its file list was incomplete."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    revision = (args.run / "base-revision.txt").read_text().strip()
    assert not (args.run / "source.patch").read_bytes(), "The initial tracked patch must be empty."
    imported = str(Path.cwd() / "genesis/__init__.py")
    assert imported in (args.run / "import.txt").read_text()
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            ":(glob)genesis/**/*.py",
            ":(glob)tests/**/*.py",
            ":(glob)research/rigid_stress/*.py",
        ],
        text=True,
    ).splitlines()
    assert len(paths) > 400
    assert not subprocess.check_output(["git", "diff", revision, "--", *paths])
    destination = args.run / "recorded-git-source"
    assert not destination.exists()
    destination.mkdir()
    archive = subprocess.check_output(["git", "archive", "--format=tar", revision, *paths])
    entries = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream:
            if not member.isfile():
                continue
            target = destination / member.name
            assert target.resolve().is_relative_to(destination.resolve())
            content = stream.extractfile(member).read()
            assert content == Path(member.name).read_bytes(), member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            entries.append({"path": member.name, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    assert len(entries) == len(paths)
    record = {
        "revision": revision,
        "initial_tracked_patch_empty": True,
        "verified_worktree_import": imported,
        "source_files": entries,
        "provenance": "Exact initial clean Git revision, verified against current tracked sources. Supplemental files were reconstructed after test start; they are not claimed as pretest file copies.",
        "reason": "The original source-list process substitution invoked unavailable rg. Numerical tests still ran from the recorded worktree and clean Git revision; initial revision/diff and import records were retained.",
    }
    (args.run / "recorded-git-source.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"pinned_source_files": len(entries), "revision": revision}), flush=True)


if __name__ == "__main__":
    main()
