"""Qualify an isolated exact-sample fan launch candidate, with copied B32/B1024 live workloads."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/fan-frontier-checkpoint-v1")
assert not destination.exists()
runs = [root / name for name in ("fan-frontier-regression-v1/cpu", "fan-frontier-regression-v1/cuda",
                                "fan-frontier-benchmark-v1", "fan-frontier-b1024-benchmark-v1")]
for run in runs:
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
for run in runs[:2]:
    assert "56 passed" in (run / "tests.log").read_text()
summary = {}
for run, batch in zip(runs[2:], (32, 1024)):
    results = {}
    for variant in ("native", "frontier"):
        profile = json.loads((run / f"profile/{variant}/result.json").read_text())
        live = json.loads((run / f"live/{variant}/result.json").read_text())
        for result in (profile, live):
            assert result["envs"] == batch and result["precision"] == "64" and result["quadrature"] == 10
            assert result["dt"] == 0.01 and result["substeps"] == 1
            assert result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
            for row in result["warmup_quality"]:
                fraction = np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))
                assert np.max(fraction) <= 1
        assert len(live["repeats"]) == 3
        for row in live["repeats"]:
            assert row["steps"] == 1280 and row["resets"] == 37 * batch // 32
            assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
            assert max(row["failed_steps_per_environment"]) == 0
        rate = batch * 1280 * 3 / sum(row["seconds"] for row in live["repeats"])
        check = profile["recovery_snapshot_check"]
        assert check["original_point_max_difference_m"] == 0
        assert check["normalized_force_error_max"] <= 1e-8 and check["normalized_moment_error_max"] <= 1e-8
        budget = np.max(np.array(profile["complete_residual_N"]) /
                        np.maximum(1e-11, 1e-8 * np.array(profile["rhs_norm_N"])))
        assert budget <= 1
        extra = 0
        if variant == "frontier":
            records = [json.loads(line) for line in (run / f"live/{variant}/benchmark.log").read_text().splitlines()
                       if line.startswith('{"candidate_note"')]
            assert len(records) == 1
            extra = records[0]["frontier_buffer_bytes"]
            assert extra == 4 * (80 * 8 * batch + 8 * batch + 1)
        results[variant] = {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / batch,
                            "repeat_rates": [row["transitions_per_second"] for row in live["repeats"]],
                            "resets_per_repetition": live["repeats"][0]["resets"],
                            "gpu_uuid": live["gpu_uuid"], "stage_wall_ms": profile["stage_wall_ms"],
                            "snapshot_check": check, "complete_residual_budget_fraction": float(budget),
                            "native_buffer_bytes": live["native_buffer_bytes"], "extra_frontier_bytes": extra,
                            "total_native_candidate_bytes": live["native_buffer_bytes"] + extra,
                            "device_free_bytes": live["device_free_bytes"],
                            "torch_peak_allocated_bytes": live["torch_peak_allocated_bytes"]}
    assert results["native"]["gpu_uuid"] == results["frontier"]["gpu_uuid"]
    results["live_speedup"] = results["frontier"]["valid_env_steps_per_second"] / results["native"]["valid_env_steps_per_second"]
    summary[str(batch)] = results
files = []
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
files += [root / name for name in ("fan_frontier_candidate_v1.py", "probe_fan_frontier_v1.py",
                                  "generate_fan_frontier_candidate_v1.py", "run_generate_fan_frontier_v1.sh",
                                  "run_fan_frontier_regression_v1.sh", "run_fan_frontier_benchmark_v1.sh",
                                  "run_fan_frontier_b1024_benchmark_v1.sh", "archive_fan_frontier_v1.py",
                                  "run_archive_fan_frontier_v1.sh")]
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
manifest = {"scope": __doc__, "summary": summary,
            "limitations": "Isolated prototype scheduling; native production integration remains to qualify. Identical geometry, quadrature, profile, intrinsic paths, friction, wrench and residual budgets. All candidate work and the additional maximum readback are timed. Extra workspace is explicitly counted because the old native enumerator excludes this prototype. One 32-condition bank is copied; policy, broader seeds, batch optimum and optimization plateau remain unverified.",
            "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    stored = (destination / entry["artifact"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    restored = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
    assert hashlib.sha256(restored).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "unique_blobs": len(blobs),
                  "unique_blob_bytes": sum(path.stat().st_size for path in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": summary}, indent=2), flush=True)
