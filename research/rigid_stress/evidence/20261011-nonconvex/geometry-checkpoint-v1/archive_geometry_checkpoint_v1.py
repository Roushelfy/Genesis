"""Verify and publish only closed nonconvex geometry checkpoint artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--published", type=Path, required=True)
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--revision", required=True)
args = parser.parse_args()

ansi = re.compile(r"\x1b\[[0-9;]*m")

def counts(relative):
    content = ansi.sub("", (args.raw / relative).read_text())
    final = content.rstrip().splitlines()[-1]
    return {kind: int(value) for value, kind in re.findall(r"(\d+) (passed|failed|skipped|deselected)", final)}

cases = [
    ("checkpoint-v1-cpu/tests.log", 12, 0, 0),
    ("checkpoint-v1-cuda/tests.log", 12, 0, 0),
    ("existing-cuda-v1.log", 52, 0, 0),
    ("boundary-com-mutation-v2.log", 0, 2, 0),
    ("path-euclidean-mutation-v3.log", 0, 2, 0),
]
checked = []
for relative, passed, failed, skipped in cases:
    result = counts(relative)
    assert result.get("passed", 0) == passed, (relative, result)
    assert result.get("failed", 0) == failed, (relative, result)
    assert result.get("skipped", 0) == skipped, (relative, result)
    checked.append({"file": relative, "counts": result})

assert "CHECKPOINT_COMBINED_V1_COMPLETE" in (args.raw / "checkpoint-allocation-v1.log").read_text()
source_hashes = {}
for backend in ("cpu", "cuda"):
    lines = (args.raw / f"checkpoint-v1-{backend}/source.sha256").read_text().splitlines()
    recorded = {}
    for line in lines:
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip()
        actual = hashlib.sha256((args.source_root / relative).read_bytes()).hexdigest()
        assert actual == digest, (relative, actual, digest)
        recorded[relative] = digest
    if source_hashes:
        assert source_hashes == recorded
    source_hashes = recorded

selected = [
    "checkpoint-v1-cpu/tests.log", "checkpoint-v1-cpu/import.log", "checkpoint-v1-cpu/source.sha256",
    "checkpoint-v1-cuda/tests.log", "checkpoint-v1-cuda/import.log", "checkpoint-v1-cuda/source.sha256",
    "checkpoint-v1-cuda/gpu.csv", "checkpoint-allocation-v1.log", "run_checkpoint_combined_v1.sh",
    "existing-cuda-v1.log", "run_existing_regression_v1.sh",
    "boundary-com-mutation-v2.log", "boundary_com_mutation.py", "run_boundary_tests_v2.sh",
    "boundary-allocation-v2.log", "boundary-cpu-v2.log", "boundary-cuda-v2.log",
    "boundary-allocation-v1.log", "boundary-cpu-v1.log", "run_boundary_tests_v1.sh",
    "path-euclidean-mutation-v3.log", "path_euclidean_mutation.py", "run_combined_tests_v3.sh",
    "combined-allocation-v3.log", "combined-cpu-v3.log", "combined-cuda-v3.log", "combined-source-v3.sha256",
    "run_path_tests_v1.sh", "paths-allocation-v1.log", "paths-cpu-v1.log", "paths-cuda-v1.log",
    "run_nonconvex_elastic_v1.sh", "elastic-allocation-v1.log", "elastic-cpu-v1.log", "elastic-cuda-v1.log",
    "elastic-source-v1.sha256", "run_source_tests_v1.sh", "source-allocation-v1.log", "source-cpu-v1.log",
    "source-source-v1.sha256", "run_source_tests_v2.sh", "source-allocation-v2.log", "source-cpu-v2.log",
    "source-cuda-v2.log", "source-source-v2.sha256", "archive_geometry_checkpoint_v1.py",
]
cpu_regression_complete = False
allocation = (args.raw / "existing-allocation-v1.log").read_text()
if "EXISTING_V1_COMPLETE" in allocation:
    result = counts("existing-cpu-v1.log")
    assert result == {"passed": 31, "skipped": 21}, result
    checked.append({"file": "existing-cpu-v1.log", "counts": result})
    selected += ["existing-cpu-v1.log", "existing-allocation-v1.log"]
    cpu_regression_complete = True

assert not args.published.exists(), "Preserve the existing publication; choose a new checkpoint directory."
args.published.mkdir(parents=True)
entries = []
for relative in selected:
    original, publication = args.raw / relative, args.published / relative
    assert original.is_file(), original
    publication.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, publication)
    raw_digest = hashlib.sha256(original.read_bytes()).hexdigest()
    pub_digest = hashlib.sha256(publication.read_bytes()).hexdigest()
    assert raw_digest == pub_digest
    entries.append({"file": relative, "raw": str(original), "sha256": raw_digest, "bytes": original.stat().st_size})

summary = {
    "source_revision": args.revision,
    "source_sha256": source_hashes,
    "checked_tests": checked,
    "existing_cpu_regression_complete": cpu_regression_complete,
    "native_nonconvex_contact_goal_complete": False,
    "limits": [
        "Only the material-normal fix is active in live stress recovery.",
        "Source and intrinsic-path helpers are not integrated into live collision/load mapping.",
        "Vector traction fitting, its contact scatter/CPU oracles/lifecycle and matched throughput remain pending.",
        "Port paths are upper bounds; no final subdivision count or physical convergence is established.",
    ],
    "artifact_count": len(entries),
    "artifacts": entries,
}
manifest = args.published / "manifest.json"
manifest.write_text(json.dumps(summary, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256((args.published / entry["file"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256(Path(entry["raw"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"verified_artifacts": len(entries), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  "existing_cpu_regression_complete": cpu_regression_complete}))
