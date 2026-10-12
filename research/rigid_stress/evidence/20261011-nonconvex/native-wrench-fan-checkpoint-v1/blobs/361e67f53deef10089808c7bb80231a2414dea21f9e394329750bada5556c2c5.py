"""Qualified native Q10 wrench-line fan repair, successful independent B2048 live run and certified B4096 limitation."""

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-wrench-fan-checkpoint-v1")
assert not destination.exists()
labels = ("b2048-623102", "b4096-623103")
runs = [root / f"wrench-fan-focus-v1/{label}" for label in labels]
runs += [root / f"native-wrench-fan-regression-v1/{backend}" for backend in ("cpu", "cuda")]
runs += [root / f"native-wrench-fan-live-v1/b{batch}-varied-live-frontier-{seed}" for batch, seed in ((2048, 623102), (4096, 623103))]
runs += [root / "native-wrench-fan-failure-capture-v1/b4096-623103", root / "wrench-fan-continuum-audit-v1"]
for run in runs:
    assert (run / "returncode.txt").exists()
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
fields = []
for backend in ("cpu", "cuda"):
    run = root / f"native-wrench-fan-regression-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert "64 passed" in (run / "tests.log").read_text()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "64" and suite.attrib["failures"] == "0" and suite.attrib["errors"] == "0"
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"])
                      for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
assert len(fields) == 12 and sum(len(field["native_peak_Pa"]) for field in fields) == 36
live = root / "native-wrench-fan-live-v1/b2048-varied-live-frontier-623102"
assert (live / "returncode.txt").read_text().strip() == "0"
result = json.loads((live / "result.json").read_text())
assert result["envs"] == 2048 and result["scope"] == "live" and result["conditions"] is None
assert result["seed"] == 623102 and result["varied"] and result["surface_fan_frontier"]
assert "failed_wrench_fan_partition" in result["load_model"] and result["intrinsic_load_partition"] == "body_source_prior"
assert result["dt"] == 0.01 and result["substeps"] == 1 and result["precision"] == "64"
for row in result["repeats"]:
    assert row["steps"] == 1280 and row["resets"] == 2323
    assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
    assert len(row["failed_steps_per_environment"]) == 2048 and max(row["failed_steps_per_environment"]) == 0
budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
           for row in result["warmup_quality"]]
assert max(budgets) <= 1
rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
proof_run = root / "wrench-fan-continuum-audit-v1"
assert (proof_run / "returncode.txt").read_text().strip() == "0"
proof = json.loads((proof_run / "summary.json").read_text())
assert proof["source_count"] == 2 and proof["selected_environment"] == 3092
assert proof["bounds"][1]["certified_physical_incompatibility"]
assert proof["bounds"][1]["native_ray_continuous_polar_upper_bound"] < 0
summary = {"CPU_passed": 64, "CUDA_passed": 64, "independent_full_fields": 36,
           "force_or_moment_error_max": max(max(field["force_errors"] + field["moment_errors"]) for field in fields),
           "complete_residual_budget_fraction_max": max(max(field["complete_residual_budget_fraction"]) for field in fields),
           "peak_difference_Pa_max": max(max(field["peak_difference_Pa"]) for field in fields), "fields": fields,
           "B2048_live": {"valid_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                          "repeat_rates": [row["transitions_per_second"] for row in result["repeats"]],
                          "resets_per_repetition": 2323, "invalid_environment_steps": 0,
                          "warmup_complete_residual_budget_fraction_max": max(budgets),
                          "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"],
                          "device_free_bytes": result["device_free_bytes"],
                          "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"],
                          "actual_friction_ranges": [row["contact_friction_range"] for row in result["warmup_quality"]]},
           "B4096_continuous_limitation": proof}
files = []
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
for name in ("wrench_fan_focus_v1.py", "probe_wrench_fan_focus_v1.py", "run_probe_wrench_fan_focus_v1.sh",
             "run_native_wrench_fan_regression_v1.sh", "run_native_wrench_fan_live_v1.sh",
             "run_native_wrench_fan_failure_capture_v1.sh", "probe_native_policy_failure_v2.py",
             "audit_wrench_fan_continuum_v1.py", "run_audit_wrench_fan_continuum_v1.sh", "audit_continuum_wrench_v2.py",
             "publish_single_source_fixtures_v1.py", "allocation-singlefixturev1.log",
             "archive_native_wrench_fan_v1.py", "run_archive_native_wrench_fan_v1.sh"):
    files.append(root / name)
files.extend(root.glob("allocation-*wrenchfan*.log"))
files.extend(root.glob("allocation-*fanfocus*.log"))
files.extend(root.glob("allocation-bodycontinuumv1.log"))
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
            "limitations": "B2048live is one independent seed with complete repeats,not diverse four-scope/plateau acceptance. B4096moves past its original single-source failure but aborts in its first timed repetition at a two-source continuous-model incompatibility. Input geometry/provenance/physics and explicit failed-episode accounting require further review. No validB4096throughput or completed goal is claimed. Native defaults and public error semantics are retained;no radius/friction/force/budget changes make the proof disappear.", "files": entries}
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
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "full_fields": summary["independent_full_fields"],
                  "max_wrench_error": summary["force_or_moment_error_max"],
                  "max_complete_residual_budget_fraction": summary["complete_residual_budget_fraction_max"],
                  "max_peak_difference_Pa": summary["peak_difference_Pa_max"],
                  "B2048_rate": rate, "B2048_batch_rate": rate / 2048}, indent=2), flush=True)
