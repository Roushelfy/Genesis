"""Publish frozen source/traction checks, including the failed numerical oracle."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--source-revision", required=True)
parser.add_argument("--repo", type=Path, required=True)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
args = parser.parse_args()
assert not args.destination.exists()
runs = (
    "source-association-v1", "traction-v1", "traction-v2",
    "contact-checkpoint-v1-cpu", "contact-checkpoint-v1-cuda",
)
checks = (
    ("source-association-v1/cpu.log", "16 passed", False),
    ("source-association-v1/cuda.log", "16 passed", False),
    ("traction-v1/cpu.log", "1 failed, 2 passed", True),
    ("traction-v1/cuda.log", "1 failed, 2 passed", True),
    ("traction-v2/cpu.log", "4 passed", False),
    ("traction-v2/cuda.log", "4 passed", False),
    ("contact-checkpoint-v1-cpu/tests.log", "20 passed", False),
    ("contact-checkpoint-v1-cuda/tests.log", "20 passed", False),
)
for name, expected, has_expected_failure in checks:
    output = re.sub(r"\x1b\[[0-9;]*m", "", (args.raw / name).read_text())
    assert expected in output, name
    assert (" failed" in output) == has_expected_failure, name
sources_by_run = {}
for run in runs:
    before = args.raw / run / "source.sha256"
    assert before.read_bytes() == (args.raw / run / "source-after.sha256").read_bytes(), run
    sources = dict(line.split("  ", 1)[::-1] for line in before.read_text().splitlines())
    for name, expected in sources.items():
        assert digest(args.raw / run / "source" / name) == expected, (run, name)
    sources_by_run[run] = sources
final_sources = sources_by_run["contact-checkpoint-v1-cpu"]
assert final_sources == sources_by_run["contact-checkpoint-v1-cuda"]
for name, expected in final_sources.items():
    assert digest(args.repo / name) == expected, name
for run, sources in sources_by_run.items():
    for name, expected in sources.items():
        if name.startswith("genesis/"):
            assert final_sources[name] == expected, (run, name)

artifacts = []
args.destination.mkdir(parents=True)


def copy_artifact(original, relative):
    target = args.destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, target)
    expected = digest(original)
    assert digest(target) == expected
    artifacts.append({"path": str(relative), "original": str(original), "sha256": expected})


for run in runs:
    for original in sorted((args.raw / run).iterdir()):
        if original.is_file():
            copy_artifact(original, Path(run) / original.name)
    for name in sources_by_run[run]:
        copy_artifact(args.raw / run / "source" / name, Path(run) / "source" / name)
for name in (
    "run_source_association_v1.sh", "run_traction_v1.sh", "run_traction_v2.sh",
    "run_contact_checkpoint_v1.sh", "contact-checkpoint-v1-allocation.log",
    "archive_contact_checkpoint.py",
):
    copy_artifact(args.raw / name, name)
copy_artifact(
    args.raw / "archive-support/check_nonconvex_contact.sh",
    "check_nonconvex_contact.sh",
)
manifest = {
    "source_revision": args.source_revision,
    "requirement_revision": "1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "native_nonconvex_goal_complete": False,
    "scope": "Reference-source helpers and standalone vector traction/scatter/elasticity; not the live new-law pipeline.",
    "test_consolidation": {
        "initial": "traction-v2/source/tests/rigid/test_stress_traction_limits.py",
        "final": "contact-checkpoint-v1-cpu/source/tests/rigid/test_stress_traction.py",
        "verification": "Final combined module passes twenty CPU and twenty CUDA checks.",
    },
    "failed_check": {
        "logs": ["traction-v1/cpu.log", "traction-v1/cuda.log"],
        "cause": "Nonsmooth three-dimensional SLSQP oracle reported a line-search failure.",
        "correction": "Independent two-variable convex QP with linear constraints plus Moreau and finite-difference checks.",
        "unchanged_acceptance": "Force/moment 1e-8; complete residual max(1e-11 N, 1e-8 ||rhs||); stress rtol 1e-4, atol 1e-3 Pa.",
    },
    "remaining": [
        "General primitive/SDF/decomposition and aggregate-source association",
        "Intrinsic profile and adaptive local integration on device",
        "Live vector traction, diagnostic and lifecycle integration",
        "Franka/low-egg acceptance and four-scope matched throughput/optimization",
    ],
    "checks": [{"log": name, "result": expected} for name, expected, _ in checks],
    "artifacts": artifacts,
}
manifest_path = args.destination / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
for artifact in artifacts:
    assert digest(args.destination / artifact["path"]) == artifact["sha256"]
    assert digest(Path(artifact["original"])) == artifact["sha256"]
print(json.dumps({"artifacts_verified": len(artifacts), "manifest_sha256": digest(manifest_path)}), flush=True)
