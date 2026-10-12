"""Publish complete-graph FP64 checks and matched graph/dispatch ablations on allocated CPU."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
regression = root / "native-contact-graph-regression-v1/cuda"
profile_runs = [root / f"native-contact-dispatch-v2/b2048-varied-profile-{mode}-623102" for mode in ("off", "fast", "graph", "both")]
profile_runs += [root / f"native-contact-graph-v3/b2048-varied-profile-{mode}-623102" for mode in ("off", "on")]
live_runs = [root / f"native-contact-scheduling-live-v1/b2048-varied-live-{mode}-623102" for mode in ("off", "both", "whole", "whole-fast")]
smoke = root / "native-contact-graph-v2/b16-varied-profile-on-623102"
failed_smoke = root / "native-contact-graph-v1/b16-varied-profile-on-623102"
runs = [regression, smoke, failed_smoke, *profile_runs, *live_runs]
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == ("1" if run == failed_smoke else "0")
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
suite = ElementTree.parse(regression / "report.xml").getroot().find("testsuite")
assert suite.attrib["tests"] == "23" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
fields = []
for case in suite.findall("testcase"):
    if "wrench_cones_full_fields" in case.attrib["name"]:
        values = {item.attrib["name"]: ast.literal_eval(item.attrib["value"]) for item in case.find("properties").findall("property")}
        assert max(values["force_errors"] + values["moment_errors"]) <= 1e-8
        assert max(values["complete_residual_budget_fraction"]) <= 1
        assert max(values["peak_difference_Pa"]) <= 1e-3
        fields.append({"case": case.attrib["name"], **values})
assert len(fields) == 8 and sum(len(row["native_peak_Pa"]) for row in fields) == 24
profiles = {}
for run in profile_runs:
    value = json.loads((run / "result.json").read_text())
    snapshot = value["recovery_snapshot_check"]
    assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
    assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
    profiles[str(run.relative_to(root))] = {"stage_wall_ms": value["stage_wall_ms"], "snapshot": snapshot,
                                         "gpu_uuid": value["gpu_uuid"], "native_buffer_bytes": value["native_buffer_bytes"]}
assert len({row["gpu_uuid"] for key, row in profiles.items() if "dispatch" in key}) == 1
assert len({row["gpu_uuid"] for key, row in profiles.items() if "graph-v3" in key}) == 1
live, results = {}, {}
rigid_keys = ("observed_steps", "lift_window_steps", "lift_grasp_pass_steps", "lift_grasp_fail_steps",
              "completed_lift_success_cycles", "completed_lift_failure_cycles", "weak_grip_steps",
              "weak_grip_descents", "release_steps", "release_descents", "two_finger_contact_steps")
for run in live_runs:
    mode = run.name.removeprefix("b2048-varied-live-").removesuffix("-623102")
    value = json.loads((run / "result.json").read_text())
    assert value["envs"] == 2048 and value["scope"] == "live" and value["seed"] == 623102
    assert value["dt"] == .01 and value["substeps"] == 1 and value["precision"] == "64"
    assert value["intrinsic_load_partition"] == "body_source_prior" and len(value["repeats"]) == 3
    for repeat in value["repeats"]:
        assert repeat["steps"] == 1280 and repeat["resets"] == 2323
        assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert max(repeat["failed_steps_per_environment"]) == 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in value["warmup_quality"]]
    assert max(budgets) <= 1
    metrics = value["trajectory_metrics"]
    assert len(metrics["repeats"]) == 3
    for row in metrics["repeats"]:
        assert row["totals"]["observed_steps"] == 2048 * 1280
        assert row["totals"]["invalid_stress_steps"] == row["totals"]["lift_grasp_fail_steps"] == 0
        assert row["totals"]["completed_lift_success_cycles"] == 3142 and row["totals"]["completed_lift_failure_cycles"] == 0
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in value["repeats"])
    live[mode] = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                  "repeats": [{key: item for key, item in row.items() if key != "failed_steps_per_environment"} for row in value["repeats"]],
                  "trajectory_metrics": {"repeats": [{key: item for key, item in row.items() if key != "per_environment"} for row in metrics["repeats"]]},
                  "setup_seconds": value["setup_seconds"], "native_buffer_bytes": value["native_buffer_bytes"],
                  "gpu_uuid": value["gpu_uuid"], "device_free_bytes": value["device_free_bytes"],
                  "warmup_full_residual_budget_fraction_max": max(budgets)}
    results[mode] = value
assert len({row["gpu_uuid"] for row in results.values()}) == 1
positions = {}
for mode, value in results.items():
    difference = float(np.max(np.abs(np.array(value["egg_position_m"]) - np.array(results["off"]["egg_position_m"]))))
    assert difference <= 1e-12
    positions[mode] = difference
    for baseline, candidate in zip(results["off"]["trajectory_metrics"]["repeats"], value["trajectory_metrics"]["repeats"]):
        for key in rigid_keys:
            assert baseline["per_environment"][key] == candidate["per_environment"][key]
speedups = {mode: row["effective_env_steps_per_second"] / live["off"]["effective_env_steps_per_second"] for mode, row in live.items()}
summary = {"cuda_passed": 23, "cuda_seconds": float(suite.attrib["time"]), "recorded_independent_full_fields": 24,
           "fields": fields, "profiles": profiles, "live": live, "live_speedups": speedups,
           "rigid_position_difference_m": positions, "identical_rigid_grasp_contact_descent_counters": True,
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields),
           "decision": "Graph candidates are not selected: no dynamic end-to-end gain in this matched three-repeat checkpoint. Fastcache-only live comparison is a separate pending candidate.",
           "failed_smoke": "v1 incorrectly unwrapped a Python dispatcher instead of its Quadrants _impl kernel; corrected v2 preserves the original numerical function body."}
extras = list(root.glob("native-contact-graph-*-allocation.log")) + list(root.glob("native-contact-dispatch-*-allocation.log"))
extras += [root / "native-contact-scheduling-live-all-v1-allocation.log"]
extras += [root / name for name in ("archive_native_contact_graph_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_contact_graph_v1.sh", "run_native_contact_graph_profiles_v3.sh", "run_native_contact_dispatch_profiles_v2.sh", "run_native_contact_scheduling_live_all_v1.sh")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-contact-graphs-rejected-v1"), runs, extras, summary,
        "Production remains unchanged 6a861556. All graph numerical bodies reuse original Quadrants .fn functions, original Q10/s8 load law and final budgets. Complete graph passes CUDA23 and 24 independent CPU FP64 fields, actual scene/lifecycle paths and constrained/refined/hard contacts; the prior fields are freshly reconstructed from original physical sources and run through the graph. Snapshot contact-wall improvements do not establish rollout gains. Matched live includes timed metrics and all environments/resets/fallbacks. Work percentiles are parallel proxies, not GPU wall attribution. Single B2048 seed; dynamic profiling, fastcache-only comparison, wider matched scopes/seeds and plateau remain pending. No physical convergence or RL training claim.")
