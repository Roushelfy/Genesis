"""Nonnegative final-segment path-bound pruning, unchanged-budget oracle checks and matched B2048 live ablation."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-port-bounds-checkpoint-v1")
assert not destination.exists()
names = ["native-port-bounds-regression-v1/cpu", "native-port-bounds-regression-v2/cuda"]
names += [f"native-port-bounds-v2/b2048-varied-{scope}-{mode}-623102" for scope in ("profile", "live") for mode in ("off", "on")]
names += [f"native-port-bounds-v1/b2048-varied-profile-{mode}-623102" for mode in ("off", "on")]
runs = [root / name for name in names]
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0", run
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), run
fields, tests = [], {}
for backend in ("cpu", "cuda"):
    run = root / ("native-port-bounds-regression-v1/cpu" if backend == "cpu" else "native-port-bounds-regression-v2/cuda")
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    n_tests = 23
    assert suite.attrib["tests"] == str(n_tests) and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    tests[backend] = {"passed": n_tests, "seconds": float(suite.attrib["time"])}
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
assert len(fields) == 16 and sum(len(row["native_peak_Pa"]) for row in fields) == 48
profiles, live = {}, {}
for mode in ("off", "on"):
    profile = json.loads((root / f"native-port-bounds-v2/b2048-varied-profile-{mode}-623102/result.json").read_text())
    result = json.loads((root / f"native-port-bounds-v2/b2048-varied-live-{mode}-623102/result.json").read_text())
    assert profile["nonnegative_port_bounds"] == result["nonnegative_port_bounds"] == (mode == "on")
    assert profile["recovery_snapshot_check"]["nodal_force_max_difference_N"] <= 1e-11
    assert profile["recovery_snapshot_check"]["peak_max_difference_Pa"] <= 1e-3
    assert max(profile["recovery_snapshot_check"][key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
    assert result["envs"] == 2048 and result["scope"] == "live" and result["seed"] == 623102
    assert result["conditions"] is None and result["varied"] and result["intrinsic_load_partition"] == "body_source_prior"
    assert result["dt"] == .01 and result["substeps"] == 1 and result["precision"] == "64"
    assert len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["steps"] == 1280 and row["resets"] == 2323
        assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
        assert len(row["failed_steps_per_environment"]) == 2048 and max(row["failed_steps_per_environment"]) == 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
    assert max(budgets) <= 1
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
    profiles[mode] = {"stage_wall_ms": profile["stage_wall_ms"], "snapshot": profile["recovery_snapshot_check"], "gpu_uuid": profile["gpu_uuid"], "native_buffer_bytes": profile["native_buffer_bytes"]}
    live[mode] = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                  "repeats": result["repeats"], "source_revision": result["source_revision"], "gpu_uuid": result["gpu_uuid"],
                  "native_buffer_bytes": result["native_buffer_bytes"], "device_free_bytes": result["device_free_bytes"],
                  "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"], "warmup_full_residual_budget_fraction_max": max(budgets)}
assert profiles["off"]["gpu_uuid"] == profiles["on"]["gpu_uuid"]
assert live["off"]["gpu_uuid"] == live["on"]["gpu_uuid"]
summary = {"tests": tests, "independent_full_fields": 48, "fields": fields, "profiles": profiles, "live": live,
           "live_speedup": live["on"]["effective_env_steps_per_second"] / live["off"]["effective_env_steps_per_second"],
           "force_or_moment_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "complete_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
files = []
for run in runs:
    for current, dirs, entries in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in entries)
    for line in (run / "source.sha256").read_text().splitlines():
        path = Path(line.split("  ", 1)[1])
        if path.is_absolute() and path.parent == root:
            files.append(path)
files.extend(root.glob("port-bounds-*-allocation.log"))
files.extend(root / name for name in ("run_port_bounds_combined_v2.sh", "run_port_bounds_live_pair_v2.sh", "archive_native_port_bounds_v1.py", "run_archive_native_port_bounds_v1.sh"))
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
    entries.append({"original": str(original), "run_path": str(original.relative_to(root)), "artifact": str(artifact.relative_to(destination)),
                    "encoding": "gzip" if compressed else "identity", "bytes": len(content), "sha256": digest,
                    "stored_bytes": len(stored), "stored_sha256": hashlib.sha256(stored).hexdigest()})
manifest = {"scope": __doc__, "summary": summary,
            "limitations": "Goal remains active. Nonnegative final-segment length makes the accumulated port distance a lower bound on every completed candidate. Skipping candidates already no smaller than the current best preserves the same minimum and all samples/weights/model/budgets. CPU23 cases qualify pruning with the unchanged CPU sampler. CUDA23 cases qualify the combined pruning plus selected sample-major route. Both benchmark pairs close collider/stress sources. Earlier isolated environment-major profile evidence is also preserved. Current live comparison uses identical selected sample-major scheduling and seed reuse across both routes. Broader seeds/batches, current four-scope measurement, timed grasp outcomes, tail work and remaining applicable optimization/plateau are pending. No physical convergence or RL training throughput claim.",
            "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for row in entries:
    assert hashlib.sha256(Path(row["original"]).read_bytes()).hexdigest() == row["sha256"]
    content = (destination / row["artifact"]).read_bytes()
    assert hashlib.sha256(content).hexdigest() == row["stored_sha256"]
    restored = gzip.decompress(content) if row["encoding"] == "gzip" else content
    assert hashlib.sha256(restored).hexdigest() == row["sha256"]
print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs), "stored_bytes": sum(path.stat().st_size for path in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tests": tests,
                  "full_fields": 48, "maximum_wrench_error": summary["force_or_moment_error_max"],
                  "maximum_full_residual_budget_fraction": summary["complete_residual_budget_fraction_max"],
                  "maximum_peak_difference_Pa": summary["peak_difference_Pa_max"], "live_speedup": summary["live_speedup"],
                  "live": live}, indent=2), flush=True)
