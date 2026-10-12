"""Publish current-source copied B1024 profiling and valid B2048 live throughput;no optimum claim."""

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/selected-scale-checkpoint-v1")
assert not destination.exists()
run = root / "selected-copied-scale-v1"
for scope in ("profile-b1024", "live-b2048"):
    case = run / scope
    assert (case / "returncode.txt").read_text().strip() == "0"
    assert (case / "source.sha256").read_bytes() == (case / "source-after.sha256").read_bytes()
profile = json.loads((run / "profile-b1024/result.json").read_text())
live = json.loads((run / "live-b2048/result.json").read_text())
assert profile["envs"] == 1024 and live["envs"] == 2048
assert profile["source_revision"].startswith("cad0c46c") and live["source_revision"].startswith("cad0c46c")
for result in (profile, live):
    assert result["conditions_sha256"] == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
    fraction = max(float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"])
    assert fraction <= 1
for row in live["repeats"]:
    assert row["steps"] == 1280 and row["resets"] == 2368
    assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
    assert max(row["failed_steps_per_environment"]) == 0
rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in live["repeats"])
check = profile["recovery_snapshot_check"]
assert check["original_point_max_difference_m"] == 0 and check["normalized_force_error_max"] <= 1e-8 and check["normalized_moment_error_max"] <= 1e-8
budget = float(np.max(np.array(profile["complete_residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(profile["rhs_norm_N"]))))
assert budget <= 1
summary = {"profile_b1024": {"stage_wall_ms": profile["stage_wall_ms"], "recovery_snapshot_check": check, "complete_residual_budget_fraction": budget, "native_buffer_bytes": profile["native_buffer_bytes"], "gpu_uuid": profile["gpu_uuid"]}, "live_b2048": {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048, "repeats": [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in live["repeats"]], "native_buffer_bytes": live["native_buffer_bytes"], "device_free_bytes": live["device_free_bytes"], "torch_peak_allocated_bytes": live["torch_peak_allocated_bytes"], "gpu_uuid": live["gpu_uuid"]}}
files, skip = [], {"cache", "tmp", "__pycache__", ".pytest_cache"}
for current, dirs, names in os.walk(run):
    dirs[:] = [name for name in dirs if name not in skip]
    files.extend(Path(current) / name for name in names)
files += [root / name for name in ("run_selected_copied_scale_v1.sh", "archive_selected_scale_v1.py", "run_archive_selected_scale_v1.sh")]
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
manifest = {"scope": __doc__, "source_revision": "cad0c46c", "summary": summary, "limitations": "One32-condition bank is copied across all environments,including delays. Existing componentwise footprint/FP64/Q10/s8/error budgets are unchanged. B1024 profile and B2048 live run on distinct allocated GPUs and are not a matched batch-speedup comparison. Profiling replays an actual contact snapshot and includes graph/stage wall timing;it is not live throughput. B2048 reports every environment/reset/failure counter for three complete repetitions. No new independent per-stepCPUoracle,diverse seeds,policy acceptance,maximum throughput or plateau is claimed. Full raw JSON/logs are retained losslessly with explicit gzip encoding where large;original and stored hashes are separate.", "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    stored = (destination / entry["artifact"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    restored = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
    assert hashlib.sha256(restored).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "unique_blobs": len(blobs), "unique_blob_bytes": sum(path.stat().st_size for path in blobs), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "summary": summary}, indent=2), flush=True)
