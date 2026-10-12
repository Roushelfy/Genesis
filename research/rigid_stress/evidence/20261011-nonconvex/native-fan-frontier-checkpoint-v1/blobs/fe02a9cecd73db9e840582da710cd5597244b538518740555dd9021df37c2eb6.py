"""Native typed frontier qualification and matched B1024 live scheduling; no load-law change."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-fan-frontier-checkpoint-v1")
assert not destination.exists()
runs = [root / "native-fan-frontier-regression-v1/cuda", root / "native-fan-frontier-benchmark-v1"]
for run in runs:
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
assert "56 passed" in (runs[0] / "tests.log").read_text()
summary = {}
for variant in ("dense", "frontier"):
    profile = json.loads((runs[1] / f"profile/{variant}/result.json").read_text())
    live = json.loads((runs[1] / f"live/{variant}/result.json").read_text())
    for result in (profile, live):
        assert result["envs"] == 1024 and result["precision"] == "64" and result["quadrature"] == 10
        assert result["dt"] == 0.01 and result["substeps"] == 1
        assert result["surface_fan_frontier"] == (variant == "frontier")
        assert result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
        assert result["load_model"].endswith("failed_disk_nine_panel_refinement")
        for row in result["warmup_quality"]:
            fraction = np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))
            assert np.max(fraction) <= 1
    assert len(live["repeats"]) == 3
    for row in live["repeats"]:
        assert row["steps"] == 1280 and row["resets"] == 1184
        assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
        assert len(row["failed_steps_per_environment"]) == 1024 and max(row["failed_steps_per_environment"]) == 0
    rate = 1024 * 1280 * 3 / sum(row["seconds"] for row in live["repeats"])
    check = profile["recovery_snapshot_check"]
    assert check["original_point_max_difference_m"] == 0
    assert max(check["normalized_force_error_max"], check["normalized_moment_error_max"]) <= 1e-8
    fraction = np.max(np.array(profile["complete_residual_N"]) /
                      np.maximum(1e-11, 1e-8 * np.array(profile["rhs_norm_N"])))
    assert fraction <= 1
    summary[variant] = {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / 1024,
                        "repeat_rates": [row["transitions_per_second"] for row in live["repeats"]],
                        "gpu_uuid": live["gpu_uuid"], "stage_wall_ms": profile["stage_wall_ms"],
                        "snapshot_check": check, "complete_residual_budget_fraction": float(fraction),
                        "native_buffer_bytes": live["native_buffer_bytes"],
                        "device_free_bytes": live["device_free_bytes"],
                        "torch_peak_allocated_bytes": live["torch_peak_allocated_bytes"]}
assert summary["dense"]["gpu_uuid"] == summary["frontier"]["gpu_uuid"]
summary["live_speedup"] = summary["frontier"]["valid_env_steps_per_second"] / summary["dense"]["valid_env_steps_per_second"]
files = []
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
files += [root / name for name in ("run_native_fan_frontier_regression_v1.sh",
                                  "run_native_fan_frontier_benchmark_v1.sh",
                                  "archive_native_fan_frontier_v1.py", "run_archive_native_fan_frontier_v1.sh")]
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
            "limitations": "Copied 32-condition bank; no diverse policy, optimum or plateau claim. All frontier preparation/readback is timed and its buffers participate in native enumeration. Original componentwise load law is unchanged. Source-prior integration and policy are separate work.",
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
