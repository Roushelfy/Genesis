"""Closed native body-prior/frontier baseline and wider-seed failure classification checkpoint."""

import gzip
import hashlib
import json
import os
from pathlib import Path

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-body-baseline-checkpoint-v1")
assert not destination.exists()
baseline = root / "native-body-baseline-v1"
runs = sorted(path for path in baseline.iterdir() if path.is_dir())
assert len(runs) == 7
labels = ("b2048-623102", "b4096-623103")
runs += [root / f"native-body-failure-capture-v1/{label}" for label in labels]
runs += [root / f"native-body-failure-audit-v{version}/{label}" for version in (1, 2, 3) for label in labels]
runs += [root / f"validate-native-body-baseline-v{version}" for version in (1, 2, 3, 4)]
for run in runs:
    assert (run / "returncode.txt").exists(), str(run)
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), str(run)
summary = json.loads((baseline / "validated-summary.json").read_text())
failures = {}
for label in labels:
    capture = root / f"native-body-failure-capture-v1/{label}"
    metadata = json.loads((capture / "failure/metadata.json").read_text())
    assert metadata["unsupported_predicate_environments"] == []
    assert len(metadata["failed_environments"]) == 1
    audit = root / f"native-body-failure-audit-v3/{label}"
    assert (audit / "returncode.txt").read_text().strip() == "0"
    classified = json.loads((audit / "results/summary.json").read_text())
    assert classified["packed_count"] == classified["positive_count"] == 586
    failures[label] = {"capture": {key: metadata[key] for key in (
        "step_call", "workload_tick_before_step", "selected_environment", "failed_environments",
        "unsupported_predicate_environments", "description_before_rigid_integration")}, "audit": classified}
assert failures[labels[0]]["audit"]["polygons"][0]["success"]
assert failures[labels[0]]["audit"]["polygons"][0]["normalized_wrench_error"] < 1e-8
assert failures[labels[1]]["audit"]["polygons"][1]["status"] == 2
files = [baseline / "validated-summary.json"]
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
patterns = ("run_native_body_baseline_v1.sh", "run_native_body_b1024_scopes_v1.sh", "run_native_body_varied_v1.sh",
            "run_native_body_failure_capture_v1.sh", "probe_native_policy_failure_v2.py",
            "audit_native_body_failure_v[123].py", "run_audit_native_body_failure_v[123].sh",
            "validate_native_body_baseline_v[1234].py", "run_validate_native_body_baseline_v[1234].sh",
            "analyze_failure_cloud_weights_v[12].py", "failure-cloud-weight-analysis-v[12].json",
            "allocation-body*.log", "allocation-native-body*.log", "archive_native_body_baseline_v1.py",
            "run_archive_native_body_baseline_v1.sh")
for pattern in patterns:
    files.extend(root.glob(pattern))
destination.mkdir(parents=True)
entries, blobs = [], set()
for original in sorted(set(files)):
    content = original.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    compressed = original.suffix in {".json", ".log"} and len(content) >= 200000
    stored = gzip.compress(content, mtime=0) if compressed else content
    artifact = destination / "blobs" / (digest + original.suffix + (".gz" if compressed else ""))
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        artifact.write_bytes(stored)
    blobs.add(artifact)
    entries.append({"original": str(original), "run_path": str(original.relative_to(root)),
                    "artifact": str(artifact.relative_to(destination)), "encoding": "gzip" if compressed else "identity",
                    "bytes": len(content), "sha256": digest, "stored_bytes": len(stored),
                    "stored_sha256": hashlib.sha256(stored).hexdigest()})
manifest = {"scope": __doc__, "baseline": summary, "failures": failures,
            "weight_analysis": json.loads((root / "failure-cloud-weight-analysis-v2.json").read_text()),
            "limitations": "Only copiedB1024 yields valid four-scope rates. The wider independent B2048/B4096 live runs aborted before timing; their policy scopes were never reached. Captured contact inputs have no unsupported predicates;single-source cases remain unresolved. B2048innerLP proves sampled cone feasibility,but native and independent dual optimizers stagnate with extremely small weights. B4096outerLP infeasibility establishes only sampled,not continuous,cone infeasibility. No completed repair,grasp success,large-batch rate,peak throughput or plateau is claimed. Diagnostic loading/schema failures retained separately from numerical findings.",
            "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    stored = (destination / entry["artifact"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    restored = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
    assert hashlib.sha256(restored).hexdigest() == entry["sha256"]
print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs),
                  "stored_bytes": sum(artifact.stat().st_size for artifact in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2), flush=True)
