"""Allocated CPU qualification, honest failed-run preservation and B2048/B8192 sample-storage ablations."""
import ast
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs, reports, fields, pairs = [], {}, [], {}
for backend in ("cpu", "cuda"):
    run = root / f"native-compact-traction-profiles-regression-v2/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "31" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[backend] = dict(suite.attrib)
    for case in suite.findall("testcase"):
        if "wrench_cones_full_fields" in case.attrib["name"]:
            values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
            assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
            assert max(values["complete_residual_budget_fraction"]) <= 1
            assert max(values["peak_difference_Pa"]) <= 1e-3
            fields.append({"backend": backend, "case": case.attrib["name"], **values})
    runs.append(run)
assert len(fields) == 32 and sum(len(row["native_peak_Pa"]) for row in fields) == 96
keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
        "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
        "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps")
for batch, seed, version in ((2048, 623102, 3), (8192, 734001, 4)):
    values, summaries = {}, {}
    for mode in ("off", "on"):
        run = root / f"native-compact-traction-profiles-v{version}/b{batch}-varied-live-{mode}-{seed}"
        assert (run / "returncode.txt").read_text().strip() == "0"
        assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
        value = json.loads((run / "result.json").read_text())
        assert value["envs"] == batch and value["seed"] == seed and value["scope"] == "live"
        assert value["dt"] == .01 and value["substeps"] == 1 and value["precision"] == "64"
        assert value["quadrature"] == 10 and value["intrinsic_load_partition"] == "body_source_prior"
        assert len(value["repeats"]) == 3
        for repeat in value["repeats"]:
            assert repeat["steps"] == 1280 and repeat["invalid_environment_steps"] == 0
            assert repeat["scatter_overflow_calls"] == 0 and max(repeat["failed_steps_per_environment"]) == 0
        for row in value["trajectory_metrics"]["repeats"]:
            assert row["totals"]["observed_steps"] == batch * 1280
            assert row["totals"]["invalid_stress_steps"] == row["totals"]["completed_lift_failure_cycles"] == row["totals"]["lift_grasp_fail_steps"] == 0
        budgets = [np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))
                   for row in value["warmup_quality"] if "residual_N" in row]
        assert max(budgets) <= 1
        rate = batch * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
        summary = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / batch,
                   "native_buffer_bytes": value["native_buffer_bytes"], "device_free_bytes": value["device_free_bytes"],
                   "setup_seconds": value["setup_seconds"], "gpu_uuid": value["gpu_uuid"],
                   "warmup_full_residual_budget_fraction_max": float(max(budgets)),
                   "repeats": [{key: item for key, item in row.items() if key != "failed_steps_per_environment"} for row in value["repeats"]],
                   "metrics": [{key: item for key, item in row.items() if key != "per_environment"} for row in value["trajectory_metrics"]["repeats"]]}
        if version == 4:
            with (run / "device-telemetry.csv").open() as stream:
                rows = list(csv.DictReader(stream, skipinitialspace=True))
            assert rows and all(row["uuid"] == value["gpu_uuid"] for row in rows)
            summary["device_memory_max_MiB_sampled_200ms"] = max(float(row["memory.used [MiB]"].split()[0]) for row in rows)
            summary["telemetry_samples"] = len(rows)
        values[mode], summaries[mode] = value, summary
        runs.append(run)
    assert values["off"]["gpu_uuid"] == values["on"]["gpu_uuid"]
    difference = np.max(abs(np.array(values["off"]["egg_position_m"]) - np.array(values["on"]["egg_position_m"])))
    assert difference <= 1e-12
    for original, compact in zip(values["off"]["trajectory_metrics"]["repeats"], values["on"]["trajectory_metrics"]["repeats"]):
        for key in keys:
            assert original["per_environment"][key] == compact["per_environment"][key]
    pairs[str(batch)] = {"seed": seed, "routes": summaries,
                        "speedup": summaries["on"]["effective_env_steps_per_second"] / summaries["off"]["effective_env_steps_per_second"],
                        "native_bytes_saved": summaries["off"]["native_buffer_bytes"] - summaries["on"]["native_buffer_bytes"],
                        "rigid_position_difference_m": float(difference), "rigid_counters_identical": True}
superseded = {}
for backend, failures in (("cpu", 8), ("cuda", 5)):
    run = root / f"native-compact-traction-profiles-regression-v1/{backend}"
    assert (run / "returncode.txt").read_text().strip() == "1"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(run / "report.xml").getroot().find("testsuite")
    assert int(suite.attrib["failures"]) == failures and suite.attrib["errors"] == "0"
    superseded[backend] = dict(suite.attrib)
    runs.append(run)
for version in (1, 2):
    run = root / f"native-compact-traction-profiles-v{version}/b16-varied-live-on-623102"
    assert (run / "returncode.txt").read_text().strip() == ("1" if version == 1 else "0")
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    if version == 1:
        assert "not a valid shape for ndarray" in (run / "benchmark.log").read_text()
    runs.append(run)
extras = [root / name for name in ("archive_native_compact_traction_profiles_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_compact_traction_profiles_v1.sh", "run_compact_traction_profiles_pair_v3.sh", "run_compact_traction_profiles_pair_v4.sh", "native-compact-traction-profiles-pair-v3-allocation.log", "native-compact-traction-profiles-pair-v4-b8192-734001-allocation.log")]
extras += [root / f"native-compact-traction-profiles-tests-v{version}-{backend}-allocation.log" for version in (1, 2) for backend in ("cpu", "cuda")]
summary = {"reports": reports, "independent_full_fields": 96, "fields": fields, "pairs": pairs,
           "superseded_fixture_failures": superseded,
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "global_peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-compact-traction-profiles-checkpoint-v1"), runs, extras, summary,
        "Prototype matchesa6dc0fbc plus complete-sample capacity growth. Physical Q10/s8 footprint law/radius/friction and all final budgets are unchanged; count every physical sample, grow before fit, copy successful unaffected profiles and replay producers before consumers. CUDA-only frontier compaction; CPU route remains full storage. Minimum1-slot placeholder is the smallest tensor allowed by Quadrants. B16v1 zero-size allocation failure is preserved. First expanded-native full-field harness mistakenly reused aggregate forces as raw loads and used a CUDA cached entrypoint with disabled CPU arrays; these failed fixtures are preserved. Corrected fresh-raw input/backend entrypoint yieldsCPU31/CUDA31 and96 independent full fields without relaxing acceptance. Each matched live pair includes600 warmup and three1280-step changing trajectories/all environments/reset/history/fallback/metrics. B2048 telemetry failed from missing UUID prefix; no peak-memory claim there. B8192 monitor uses actual assigned UUID and200ms sampling, includes setup and trajectories, and is a sampled bound rather than exact transient maximum. All source/raw/stored hashes verified. Prototype has measured memory savings but is not selected yet; fused capacity checks, serial-tail candidate, broader final scopes/seeds and plateau remain pending. No physical convergence or RL training claim.")
