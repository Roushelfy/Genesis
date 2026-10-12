"""Device frontier maximum, current-capacity GPU oracle checks and same-GPU live ablation."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / "native-device-frontier-count-regression-v1/cuda"]
runs += [root / f"native-device-frontier-count-v1/b2048-varied-{scope}-{mode}-623102" for scope in ("profile", "live") for mode in ("off", "on")]
for run in runs:
    assert (run / "returncode.txt").read_text().strip() == "0"
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
suite = ElementTree.parse(runs[0] / "report.xml").getroot().find("testsuite")
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
profiles, live = {}, {}
for mode in ("off", "on"):
    profile = json.loads((root / f"native-device-frontier-count-v1/b2048-varied-profile-{mode}-623102/result.json").read_text())
    result = json.loads((root / f"native-device-frontier-count-v1/b2048-varied-live-{mode}-623102/result.json").read_text())
    assert profile["device_frontier_count"] == result["device_frontier_count"] == (mode == "on")
    snapshot = profile["recovery_snapshot_check"]
    assert snapshot["nodal_force_max_difference_N"] <= 1e-11 and snapshot["peak_max_difference_Pa"] <= 1e-3
    assert max(snapshot[key] for key in ("normalized_force_error_max", "normalized_moment_error_max")) <= 1e-8
    assert result["envs"] == 2048 and result["scope"] == "live" and result["seed"] == 623102
    assert result["dt"] == .01 and result["substeps"] == 1 and result["precision"] == "64"
    assert result["intrinsic_load_partition"] == "body_source_prior" and len(result["repeats"]) == 3
    for row in result["repeats"]:
        assert row["steps"] == 1280 and row["resets"] == 2323
        assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
        assert max(row["failed_steps_per_environment"]) == 0
    budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
    assert max(budgets) <= 1
    rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
    profiles[mode] = {"stage_wall_ms": profile["stage_wall_ms"], "snapshot": snapshot,
                      "gpu_uuid": profile["gpu_uuid"], "native_buffer_bytes": profile["native_buffer_bytes"]}
    live[mode] = {"effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
                  "repeats": [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in result["repeats"]],
                  "gpu_uuid": result["gpu_uuid"], "native_buffer_bytes": result["native_buffer_bytes"],
                  "device_free_bytes": result["device_free_bytes"], "warmup_full_residual_budget_fraction_max": max(budgets)}
assert profiles["off"]["gpu_uuid"] == profiles["on"]["gpu_uuid"]
assert live["off"]["gpu_uuid"] == live["on"]["gpu_uuid"]
summary = {"cuda_passed": 23, "cuda_seconds": float(suite.attrib["time"]), "recorded_independent_full_fields": 24,
           "fields": fields, "profiles": profiles, "live": live,
           "live_speedup": live["on"]["effective_env_steps_per_second"] / live["off"]["effective_env_steps_per_second"],
           "wrench_error_max": max(max(row["force_errors"] + row["moment_errors"]) for row in fields),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in fields),
           "peak_difference_Pa_max": max(max(row["peak_difference_Pa"]) for row in fields)}
extras = list(root.glob("device-frontier-count-*-allocation.log"))
extras += [root / name for name in ("archive_native_device_frontier_count_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_device_frontier_count_v1.sh", "run_device_frontier_count_profile_pair_v1.sh", "run_device_frontier_count_qualification_v1.sh")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-device-frontier-count-checkpoint-v1"), runs, extras, summary,
        "All actual frontier counts remain on device and the sampler consumes the current maximum directly. Q10/sample identities, radius/profile/friction, exact capacity and selected EPA/seed/sample-major/port pruning are identical across both routes. Final wrench/gauge residual/peak acceptance is unchanged. CPU route is unchanged and CUDA23/24 recorded independent fields pass. Fusion is absent from this isolated comparison. Current-source combined qualification, broad scopes/batches/seeds, dynamic and tail profiles, graph/dispatch candidates and plateau remain pending under the active goal. No physical convergence or RL training claim.")
