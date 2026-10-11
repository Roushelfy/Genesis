"""Publish closed witness runs and verify their originals and snapshots."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(run):
    before = raw / run / "source.sha256"
    assert before.read_bytes() == (raw / run / "source-after.sha256").read_bytes(), run
    sources = dict(line.split("  ", 1)[::-1] for line in before.read_text().splitlines())
    for name, expected in sources.items():
        assert digest(raw / run / "source" / name) == expected, (run, name)
    return sources


parser = argparse.ArgumentParser()
parser.add_argument("--source-revision", required=True)
parser.add_argument("--destination", required=True)
args = parser.parse_args()
repo = Path("/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery")
raw = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
destination = Path(args.destination)
assert not destination.exists()
checks = (
    ("witness-v5/cpu.log", "5 passed"),
    ("witness-v5/cuda.log", "5 passed"),
    ("witness-manifold-v1/cpu.log", "1 passed"),
    ("witness-manifold-v1/cuda.log", "1 passed"),
    ("witness-regression-v1/stress-cuda.log", "64 passed"),
    ("witness-regression-v1/stress-cpu.log", "43 passed, 21 skipped"),
    ("witness-regression-v1/collision-cuda.log", "6 passed"),
)
for name, expected in checks:
    output = re.sub(r"\x1b\[[0-9;]*m", "", (raw / name).read_text())
    assert expected in output and " failed" not in output, name
runs = ("witness-v4", "witness-v5", "witness-manifold-v1", "witness-regression-v1")
sources_by_run = {run: source_hashes(run) for run in runs}
sources = sources_by_run["witness-v5"]
engine_sources = {name: expected for name, expected in sources.items() if name.startswith("genesis/")}
for name, expected in engine_sources.items():
    assert all(items[name] == expected for items in sources_by_run.values()), name
    assert digest(repo / name) == expected, name
candidate = "tests/rigid/test_stress_witness_v5.py"
final_test = "tests/rigid/test_stress_witness.py"
assert digest(repo / final_test) == sources[candidate]
assert digest(repo / "tests/rigid/test_stress_nonconvex.py") == sources["tests/rigid/test_stress_nonconvex.py"]
manifold_test = "tests/rigid/test_stress_witness_manifold.py"
assert digest(repo / manifold_test) == sources_by_run["witness-manifold-v1"][manifold_test]
failed_output = re.sub(r"\x1b\[[0-9;]*m", "", (raw / "witness-v4/cuda.log").read_text())
assert "1 failed, 4 passed" in failed_output and "array([1, 2]" in failed_output and "array([2, 1]" in failed_output

artifacts = []
destination.mkdir(parents=True)


def copy_artifact(original, relative):
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, target)
    expected = digest(original)
    assert digest(target) == expected
    artifacts.append({"path": str(relative), "original": str(original), "sha256": expected})


for run in runs:
    for original in sorted((raw / run).iterdir()):
        if original.is_file():
            copy_artifact(original, Path(run) / original.name)
for name in (
    "run_witness_v4.sh", "run_witness_v5.sh", "run_witness_manifold_v1.sh",
    "run_witness_regression_v1.sh", "archive_witness_checkpoint.py",
):
    copy_artifact(raw / name, name)
for name in sources:
    published_name = final_test if name == candidate else name
    copy_artifact(raw / "witness-v5/source" / name, Path("source") / published_name)
copy_artifact(raw / "witness-manifold-v1/source" / manifold_test, Path("source") / manifold_test)
copy_artifact(raw / "witness-v4/source" / final_test, Path("failed-v4-source") / final_test)
copy_artifact(raw / "archive-support/tests/rigid/test_stress.py", "source/tests/rigid/test_stress.py")
copy_artifact(raw / "archive-support/research/rigid_stress/scripts/check_nonconvex_witness.sh", "check_nonconvex_witness.sh")
manifest = {
    "source_revision": args.source_revision,
    "requirement_revision": "1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "geometry_checkpoint_revision": "506b4964efc1e82bffd2ab7868db68bc7ec1bd72",
    "native_nonconvex_goal_complete": False,
    "scope": "Raw collision witness retention, state lifecycle, and existing stress/collision regression only.",
    "test_source_rename": {"tested": candidate, "committed": final_test, "sha256": sources[candidate]},
    "failed_check": {
        "log": "witness-v4/cuda.log",
        "cause": "The added comparison assumed deterministic GPU atomic contact-slot ordering across two independent pairs.",
        "correction": "Canonical contact identity/position alignment; integer identities and numerical budgets retained.",
    },
    "remaining": [
        "Reference boundary face/barycentric association", "SDF uncertainty and aggregate sources",
        "Intrinsic footprint and vector traction integration", "New-law contact/elastic/lifecycle acceptance",
        "New-law throughput and measured optimization",
    ],
    "checks": [{"log": name, "result": expected} for name, expected in checks],
    "artifacts": artifacts,
}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
for artifact in artifacts:
    assert digest(destination / artifact["path"]) == artifact["sha256"]
    assert digest(Path(artifact["original"])) == artifact["sha256"]
print(json.dumps({"artifacts_verified": len(artifacts), "manifest_sha256": digest(destination / "manifest.json")}))
