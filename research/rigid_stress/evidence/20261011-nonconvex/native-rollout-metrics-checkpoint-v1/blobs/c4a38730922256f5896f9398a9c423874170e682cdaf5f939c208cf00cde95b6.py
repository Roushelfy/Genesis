"""Timed native benchmark grasp outcomes and per-environment work collectors, including the corrected API attempt."""
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-rollout-metrics-checkpoint-v1")
assert not destination.exists()
names = [f"native-rollout-metrics-validation-v{i}/b16-live" for i in (1, 2)]
names += [f"native-rollout-metrics-validation-v3/b16-{scope}" for scope in ("rigid", "live", "policy")]
runs = [root / name for name in names]
for i, run in enumerate(runs):
    assert (run / "returncode.txt").read_text().strip() == ("1" if i == 0 else "0"), run
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), run
assert "unexpected keyword argument 'copy'" in (runs[0] / "benchmark.log").read_text()
results, summary = {}, {}
for scope in ("rigid", "live", "policy"):
    result = json.loads((root / f"native-rollout-metrics-validation-v3/b16-{scope}/result.json").read_text())
    assert result["scope"] == scope and result["envs"] == 16 and result["seed"] == 623102
    assert len(result["repeats"]) == len(result["trajectory_metrics"]["repeats"]) == 1
    row, metric = result["repeats"][0], result["trajectory_metrics"]["repeats"][0]
    assert row["steps"] == 1280 and row["resets"] == 19 and row["invalid_environment_steps"] == 0
    assert metric["native_buffer_bytes"] == 1696
    totals, per_env = metric["totals"], metric["per_environment"]
    assert per_env["observed_steps"] == [1280] * 16 and totals["observed_steps"] == 20480
    assert totals["lift_window_steps"] == totals["lift_grasp_pass_steps"] + totals["lift_grasp_fail_steps"]
    assert totals["completed_lift_success_cycles"] == 25 and totals["completed_lift_failure_cycles"] == 0
    assert totals["weak_grip_descents"] > 0 and totals["release_descents"] > 0
    assert totals["invalid_stress_steps"] == row["invalid_environment_steps"]
    assert all(len(values) == 16 for values in per_env.values())
    for name, total in totals.items():
        assert sum(per_env[name]) == total
    for name, values in metric["work_percentiles"].items():
        np.testing.assert_allclose(list(values.values()), np.percentile(per_env[name], (0, 50, 90, 99, 100)), rtol=0, atol=0)
    if scope == "rigid":
        assert totals["fit_evaluation_sum"] == totals["loaded_steps"] == 0
    else:
        assert totals["fit_evaluation_sum"] > 0 and totals["loaded_steps"] > 0
    if scope == "policy":
        assert result["policy"]["warmup_reference_max_error_rad"] < result["policy"]["control_error_budget_rad"]
    results[scope] = result
    summary[scope] = {"row": row, "totals": totals, "tail": metric["work_percentiles"], "native_metric_bytes": metric["native_buffer_bytes"], "source_revision": result["source_revision"], "gpu_uuid": result["gpu_uuid"]}
assert len({result["gpu_uuid"] for result in results.values()}) == 1
np.testing.assert_allclose(results["rigid"]["egg_position_m"], results["live"]["egg_position_m"], rtol=0, atol=1e-12)
for name in ("lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps", "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps", "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps"):
    assert results["rigid"]["trajectory_metrics"]["repeats"][0]["per_environment"][name] == results["live"]["trajectory_metrics"]["repeats"][0]["per_environment"][name]
wrapped = json.loads((runs[1] / "result.json").read_text())["trajectory_metrics"]["repeats"][0]
assert wrapped["totals"] == results["live"]["trajectory_metrics"]["repeats"][0]["totals"]
files = []
for run in runs:
    for current, dirs, entries in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in entries)
    for line in (run / "source.sha256").read_text().splitlines():
        path = Path(line.split("  ", 1)[1])
        if path.is_absolute() and path.parent == root:
            files.append(path)
files.extend(root / name for name in ("rollout-metrics-validation-v1-allocation.log", "rollout-metrics-validation-v2-allocation.log", "rollout-metrics-validation-v3-allocation.log", "run_rollout_metrics_validation_pair_v3.sh", "archive_native_rollout_metrics_v1.py", "run_archive_native_rollout_metrics_v1.sh"))
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
    entries.append({"original": str(original), "run_path": str(original.relative_to(root)), "artifact": str(artifact.relative_to(destination)), "encoding": "gzip" if compressed else "identity", "bytes": len(content), "sha256": digest, "stored_bytes": len(stored), "stored_sha256": hashlib.sha256(stored).hexdigest()})
path = destination / "manifest.json"
path.write_text(json.dumps({"scope": __doc__, "summary": summary, "files": entries,
                           "limitations": "The v1 accessor API error is preserved, corrected wrapper v2 and direct native CLI v3 pass on actual GPU. Collector cost is timed for all three direct rollout scopes. Geometric lift plus both nonzero finger contacts is an explicit task metric, rather than an inference from numerical success. Tail values are per-environment work proxies, not attributed GPU wall times. Boundary reporting is outside repeats. This B16 validation is a benchmark instrumentation check; broader seeds/batches/four-scope evidence, dynamic profiling, remaining candidate evaluations and plateau remain pending under the active goal."}, indent=2) + "\n")
for row in entries:
    assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
    content = (destination / row["artifact"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
    restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
    assert hashlib.sha256(restored).hexdigest() == row["sha256"]
print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs), "stored_bytes": sum(path.stat().st_size for path in blobs), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": summary}, indent=2), flush=True)
