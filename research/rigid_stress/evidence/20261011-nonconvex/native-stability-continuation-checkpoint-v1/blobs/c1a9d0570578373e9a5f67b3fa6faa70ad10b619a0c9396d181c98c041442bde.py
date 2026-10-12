"""Canonical shared-feature mapping, failed-fan continuation and disclosed wide rollout/producer diagnostics."""

import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-stability-continuation-checkpoint-v1")
assert not destination.exists()
run_names = [
    "native-certified-rollout-v1/b4096-varied-live-frontier-623103",
    "native-certified-rollout-v1/b4096-varied-policy-frontier-623103",
    "native-certified-failure-capture-v1/b4096-623103",
    "native-certified-failure-capture-v2/b4096-623103",
    "native-wrench-fan-live-v1/b2048-varied-profile-frontier-623102",
    "native-snapshot-repeat-capture-v1/b2048-623102",
    "native-canonical-feature-v1/b2048-varied-profile-frontier-623102",
    "snapshot-producer-audit-v1", "failed-rigid-contact-frame-audit-v2", "producer-support-audit-v1",
    "publish-shared-edge-fixture-v1", "publish-slow-fan-fixture-v1",
    "native-repair-rollout-v1/b4096-varied-live-frontier-623103",
]
run_names += [f"native-group-failure-audit-v{i}" for i in range(1, 6)]
run_names += [f"captured-gjk-patch-probe-v{i}" for i in range(1, 4)]
run_names += [f"shared-edge-diagnostic-v{i}" for i in range(1, 5)]
run_names += [f"native-{stage}-regression-v1/{backend}" for stage in ("canonical", "repair") for backend in ("cpu", "cuda")]
runs = [root / name for name in run_names]
for run in runs:
    assert (run / "returncode.txt").exists(), run
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes(), run

fields = []
for backend in ("cpu", "cuda"):
    run = root / f"native-repair-regression-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "15" and suite.attrib["failures"] == suite.attrib["errors"] == "0"
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"])
                      for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
assert len(fields) == 16 and sum(len(row["native_peak_Pa"]) for row in fields) == 48

def rollout_summary(run_name, expected_invalid, expected_resets):
    run = root / run_name
    assert (run / "returncode.txt").read_text().strip() == "0"
    result = json.loads((run / "result.json").read_text())
    handling = json.loads((run / "certified-failure-responses.json").read_text())
    assert result["envs"] == 4096 and result["seed"] == 623103 and result["varied"]
    assert result["conditions"] is None and result["surface_fan_frontier"]
    assert result["intrinsic_load_partition"] == "body_source_prior"
    assert result["dt"] == 0.01 and result["substeps"] == 1 and result["precision"] == "64"
    assert len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["steps"] == 1280 and row["invalid_environment_steps"] == expected_invalid
        assert row["resets"] == expected_resets and row["scatter_overflow_calls"] == 0
        assert sum(row["failed_steps_per_environment"]) == expected_invalid
    assert len(handling["events"]) == 3 * expected_invalid
    for event in handling["events"]:
        assert event["environments"] == [3092] and event["status"] == [36]
        assert event["peer_isolation_verified"] == [0, 4095]
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
               for row in result["warmup_quality"]]
    assert max(budgets) <= 1
    valid_steps = sum(4096 * row["steps"] - row["invalid_environment_steps"] for row in result["repeats"])
    rate = valid_steps / sum(row["seconds"] for row in result["repeats"])
    return {"scope": result["scope"], "source_revision": result["source_revision"],
            "valid_env_steps_per_second": rate,
            "batch_steps_per_second": sum(row["steps"] for row in result["repeats"]) / sum(row["seconds"] for row in result["repeats"]),
            "repeat_rates": [row["transitions_per_second"] for row in result["repeats"]],
            "invalid_environment_steps_per_repetition": expected_invalid, "resets_per_repetition": expected_resets,
            "failure_handling": handling, "policy": result["policy"],
            "warmup_complete_residual_budget_fraction_max": max(budgets),
            "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"],
            "device_free_bytes": result["device_free_bytes"], "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"]}

profile_run = root / "native-canonical-feature-v1/b2048-varied-profile-frontier-623102"
assert (profile_run / "returncode.txt").read_text().strip() == "0"
profile = json.loads((profile_run / "result.json").read_text())
oracle = json.loads((root / "native-group-failure-audit-v5/summary.json").read_text())
assert (root / "native-group-failure-audit-v5/returncode.txt").read_text().strip() == "0"
assert oracle["independent_full_field"]["complete_residual_budget_fraction"] <= 1
assert oracle["independent_full_field"]["peak_difference_Pa"] <= 1e-3
summary = {
    "CPU_passed": 15, "CUDA_passed": 15, "independent_full_fields": 48, "fields": fields,
    "force_or_moment_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
    "complete_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
    "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
    "independent_env645_audit": oracle,
    "B2048_canonical_snapshot_profile": {key: profile[key] for key in
        ("source_revision", "recovery_snapshot_check", "stage_wall_ms", "pressure_subdivision_wall_ms", "gpu_uuid")},
    "B4096_live_with_certified_failures": rollout_summary("native-repair-rollout-v1/b4096-varied-live-frontier-623103", 1, 4644),
    "B4096_prior_policy_rollout": rollout_summary("native-certified-rollout-v1/b4096-varied-policy-frontier-623103", 0, 4643),
    "producer_support_audit": json.loads((root / "producer-support-audit-v1/summary.json").read_text()),
    "material_frame_audit": json.loads((root / "failed-rigid-contact-frame-audit-v2/summary.json").read_text()),
    "captured_GJK_patch_trace": json.loads((root / "captured-gjk-patch-probe-v3/summary.json").read_text()),
}
files = []
for run in runs:
    for current, dirs, names in os.walk(run):
        dirs[:] = [name for name in dirs if name not in {"cache", "tmp", "__pycache__", ".pytest_cache"}]
        files.extend(Path(current) / name for name in names)
    # Preserve standalone drivers named in each frozen source closure.
    for line in (run / "source.sha256").read_text().splitlines():
        path = Path(line.split("  ", 1)[1])
        if path.is_absolute() and path.parent == root:
            files.append(path)
for pattern in ("allocation-*canonical*.log", "allocation-*certified*.log", "allocation-*group*.log",
                "allocation-*fixture*.log", "allocation-*repair*.log", "allocation-*producer*.log",
                "allocation-*snapshot*.log", "allocation-gjkPatchProbeV[123].log", "allocation-*shared*.log",
                "allocation-*edge*.log", "allocation-*slow*.log", "*shared-edge*allocation*.log"):
    files.extend(root.glob(pattern))
files += [root / "archive_native_stability_continuation_v1.py", root / "run_archive_native_stability_continuation_v1.sh"]
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
            "limitations": "Goal remains active. Current repair is fifteen CPU and fifteen CUDA targeted cases, not a new full-suite rerun; older complete canonical matrices retain their new-test failures and all old cases passed. Current B4096 live counts one continuous-model incompatible environment per repetition, includes its diagnostic/reset cost and subtracts it from effective throughput. Legal-grasp robustness is still incomplete because original rigid producer inputs require correction. The earlier policy run uses core688edabe before canonical/continuation fixes and has zero invalid steps; it is policy inference rollout, not RL training, and is not a matched slowdown baseline. Independent conic/root solver status flags were inconclusive; full physical wrench and full field acceptance establish the oracle qualification. More seeds, grasp outcomes, per-environment tail costs, matched four-scope current baselines, profiling, optimization and plateau evidence remain pending. Final numerical budgets, original loads, R/mu and native public defaults/errors are unchanged. No high physical meshes/convergence studies.",
            "files": entries}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    stored = (destination / entry["artifact"]).read_bytes()
    assert hashlib.sha256(stored).hexdigest() == entry["stored_sha256"]
    restored = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
    assert hashlib.sha256(restored).hexdigest() == entry["sha256"]
print(json.dumps({"entries": len(entries), "unique_blobs": len(blobs), "stored_bytes": sum(path.stat().st_size for path in blobs),
                  "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "full_fields": summary["independent_full_fields"],
                  "max_wrench_error": summary["force_or_moment_error_max"],
                  "max_complete_residual_budget_fraction": summary["complete_residual_budget_fraction_max"],
                  "max_peak_difference_Pa": summary["peak_difference_Pa_max"],
                  "B4096_live": summary["B4096_live_with_certified_failures"]["valid_env_steps_per_second"],
                  "B4096_policy": summary["B4096_prior_policy_rollout"]["valid_env_steps_per_second"]}, indent=2), flush=True)
