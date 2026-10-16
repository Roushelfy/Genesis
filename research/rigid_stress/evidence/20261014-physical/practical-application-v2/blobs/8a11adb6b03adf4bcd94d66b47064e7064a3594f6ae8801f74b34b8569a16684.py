"""Publish closed compact application evidence, preserving failures and original bytes."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from research.rigid_stress.scripts.publish_nonconvex_checkpoint import publish_bounded


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True, help="Physical-validation run root.")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    p = args.runtime_root
    plan = json.loads(args.plan.read_text())
    meta = p / f"{args.destination.name}-publication"
    meta.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.plan, meta / "plan.json")
    shutil.copy2(__file__, meta / "publisher.py")
    runs = [p / row["run"] for row in plan["jobs"]]
    ids = ",".join(str(row["job"]) for row in plan["jobs"])
    query = ["sacct", "-n", "-P", "-X", "-j", ids]
    account = subprocess.check_output([*query, "--format=JobID%40,State%40,ExitCode,Elapsed,NodeList"], text=True)
    commands = subprocess.check_output([*query, "--format=JobID%40,SubmitLine%30000"], text=True)
    (meta / "jobs.txt").write_text(account)
    (meta / "job-commands.txt").write_text(commands)
    states = {line.split("|")[0].strip(): line.split("|")[1].strip() for line in account.splitlines() if line}
    extras = {meta / "plan.json", meta / "publisher.py", meta / "jobs.txt", meta / "job-commands.txt"}
    native_root = Path(plan["native_snapshot"])
    native = json.loads((native_root / "snapshot.json").read_text())
    for row in native["files"]:
        assert (native_root / row["path"]).stat().st_size == row["bytes"]
        assert digest(native_root / row["path"]) == row["sha256"]
    native_paths = [row["path"] for row in native["files"] if row["path"] != "source.patch"]
    native_blobs = {}
    for start in range(0, len(native_paths), 200):
        tree = subprocess.check_output(
            ["git", "ls-tree", "-r", "-z", plan["native_source_checkpoint"], "--", *native_paths[start : start + 200]],
            cwd=plan["source_checkout"],
        )
        for entry in tree.split(b"\0"):
            if entry:
                details, relative = entry.split(b"\t", 1)
                native_blobs[relative.decode()] = details.decode().split()[2]
    assert set(native_blobs) == set(native_paths)
    for relative, expected in native_blobs.items():
        path = native_root / relative
        blob = hashlib.sha1(("blob " + str(path.stat().st_size) + "\0").encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                blob.update(block)
        assert blob.hexdigest() == expected, relative
    assert {str(file.relative_to(native_root)) for file in native_root.rglob("*") if file.is_file()} == {
        row["path"] for row in native["files"]
    } | {"snapshot.json", "snapshot-files.sha256"}
    # All898 actual source/assets files are tracked at the declared source
    # checkpoint and independently matched to this snapshot. Do not republish
    # 225MB of unchanged package assets for every compact application packet.
    extras.update(native_root / name for name in ["snapshot.json", "snapshot-files.sha256", "source.patch"])
    identities, records, inputs = [], [], {}

    def retain(path, expected=None):
        actual = digest(path)
        assert expected is None or actual == expected, path
        inputs[str(path)] = actual
        if not any(path.is_relative_to(run) for run in runs):
            assert path.stat().st_size < 50_000_000, path
            if not path.is_relative_to(p.parent):
                target = meta / "external-inputs" / (actual + path.suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                assert digest(target) == actual
                path = target
            extras.add(path)

    def source_closed(run):
        before = run / "source.sha256"
        assert before.read_bytes() == (run / "source-after.sha256").read_bytes()
        for line in before.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            source = run / "source" / relative
            assert digest(source) == expected
            identities.append({"run": str(run), "source": relative, "sha256": expected})
        before = run / "input-source.sha256"
        if before.exists():
            assert before.read_bytes() == (run / "input-source-after.sha256").read_bytes()
            for line in before.read_text().splitlines():
                expected, relative = line.split("  ", 1)
                original = Path(relative)
                captured = run / "source" / (original.name if original.is_absolute() else original)
                assert digest(captured) == expected, captured

    def native_closed(run, benchmark):
        names = (
            ("snapshot-before-verification.txt", "snapshot-after-verification.txt")
            if benchmark
            else ("native-snapshot-before.txt", "native-snapshot-after.txt")
        )
        before, after = [run / name for name in names]
        assert before.read_bytes() == after.read_bytes()
        assert len(before.read_text().splitlines()) == len(native["files"])
        assert all(line.endswith(": OK") for line in before.read_text().splitlines())
        assert (run / "snapshot-files.sha256").read_bytes() == (native_root / "snapshot-files.sha256").read_bytes()
        if benchmark:
            assert json.loads((run / "snapshot-inventory-after.json").read_text())["passed"]

    for item, run in zip(plan["jobs"], runs, strict=True):
        job = str(item["job"]) + "_0"
        assert states[job] == item["state"], (job, states.get(job))
        assert int((run / "returncode.txt").read_text()) == item["returncode"]
        if item["kind"] == "cost_group":
            assert (run / "conditions-before.sha256").read_bytes() == (run / "conditions-after.sha256").read_bytes()
            for scope in ["rigid", "live", "policy-rigid", "policy"]:
                child = run / scope
                assert (child / "returncode.txt").read_text().strip() == "0"
                native_closed(child, True)
        elif item["kind"] == "native_benchmark":
            native_closed(run, True)
        else:
            source_closed(run)
            if item["kind"] == "native_reference":
                native_closed(run, False)
        for report in run.rglob("summary.json"):
            if "cache" in report.parts or "tmp" in report.parts:
                continue
            value = json.loads(report.read_text())
            for path, expected in value.get("input_sha256", {}).items():
                retain(Path(path), expected)
        extras.update(p.glob(f"*{item['job']}_0.log"))
        records.append(item)
    for relative in plan["extra_metadata"]:
        retain(p / relative)
    summary = {
        "scope": __doc__,
        "closed_jobs": records,
        "source_identities": identities,
        "native_source_checkpoint": plan["native_source_checkpoint"],
        "native_snapshot_sha256": digest(native_root / "snapshot-files.sha256"),
        "native_source_files": len(native["files"]),
        "native_source_assets_reconstruct_from_git": True,
        "input_sha256": inputs,
        "engineering_scope": plan["engineering_scope"],
        "goal_completion": False,
    }
    (meta / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    extras.add(meta / "summary.json")
    source_runs = []
    for item, run in zip(plan["jobs"], runs, strict=True):
        if item["kind"] not in {"cost_group", "native_benchmark"}:
            source_runs.append(run)
            continue
        # Native benchmarks use complete immutable snapshot manifests. Retain
        # their small actual originals as extras without inventing provenance.
        for current, directories, names in os.walk(run):
            directories[:] = [
                name for name in directories if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}
            ]
            for name in names:
                original = Path(current) / name
                assert original.stat().st_size < 50_000_000, original
                extras.add(original)
    publish_bounded(
        p.parent, args.destination, source_runs, sorted(extras), summary, plan["limitations"], p / "publication-staging"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "research.rigid_stress.scripts.verify_runtime_publication",
            "--manifest",
            str(args.destination / "manifest.json"),
            "--runtime-root",
            str(p.parent),
            "--staging-root",
            str(p / "publication-staging" / args.destination.name),
            "--output",
            str(meta / "runtime-verification.json"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
