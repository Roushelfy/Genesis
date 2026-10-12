"""Qualify exact capacity combined with selected port pruning and timed rollout metrics."""
import ast
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
runs = [root / name for name in ("native-current-capacity-qualification-v1/cuda", "native-current-capacity-live-v1/b2048-live")]
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
result = json.loads((runs[1] / "result.json").read_text())
assert result["envs"] == 2048 and result["scope"] == "live" and result["seed"] == 623102
assert result["dt"] == .01 and result["substeps"] == 1 and result["precision"] == "64"
assert result["intrinsic_load_partition"] == "body_source_prior"
assert len(result["repeats"]) == len(result["trajectory_metrics"]["repeats"]) == 3
for row, metrics in zip(result["repeats"], result["trajectory_metrics"]["repeats"]):
    assert row["steps"] == 1280 and row["resets"] == 2323
    assert row["invalid_environment_steps"] == row["scatter_overflow_calls"] == 0
    assert max(row["failed_steps_per_environment"]) == 0
    assert metrics["totals"]["observed_steps"] == 2048 * 1280
    assert metrics["totals"]["completed_lift_success_cycles"] == 3142
    assert metrics["totals"]["completed_lift_failure_cycles"] == 0
    assert metrics["totals"]["invalid_stress_steps"] == 0
budgets = [float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))) for row in result["warmup_quality"]]
assert max(budgets) <= 1
rate = 2048 * 1280 * 3 / sum(row["seconds"] for row in result["repeats"])
summary = {"cuda_passed": 23, "cuda_seconds": float(suite.attrib["time"]), "independent_full_fields": 24,
           "fields": fields, "effective_env_steps_per_second": rate, "batch_steps_per_second": rate / 2048,
           "repeats": [{key: value for key, value in row.items() if key != "failed_steps_per_environment"} for row in result["repeats"]],
           "trajectory_totals": [metrics["totals"] for metrics in result["trajectory_metrics"]["repeats"]],
           "work_percentiles": [metrics["work_percentiles"] for metrics in result["trajectory_metrics"]["repeats"]],
           "native_metric_buffer_bytes": result["trajectory_metrics"]["repeats"][0]["native_buffer_bytes"],
           "native_buffer_bytes": result["native_buffer_bytes"], "gpu_uuid": result["gpu_uuid"],
           "warmup_residual_budget_fraction_max": max(budgets), "source_revision": result["source_revision"]}
extras = [root / name for name in ("archive_native_current_capacity_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_current_capacity_v1.sh", "run_native_current_capacity_qualification_v1.sh", "current-capacity-qualification-v1-allocation.log")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-current-capacity-checkpoint-v1"), runs, extras, summary,
        "Current source includes qualified EPA/seed/sample-major/port-bound selection, exact capacity and timed trajectory metrics. CUDA23 and24 independent recorded full fields qualify the combination. CPU23 isolated capacity and CPU23 port-bound evidence are separate checkpoints. B2048 live counts all1280 steps, resets, metric kernels, actual source updates, fit/refinement, history and fallback. Work percentiles are parallel work proxies. This is a combined qualification, not a matched speed comparison or final four-scope/large-batch plateau. Goal remains active.")
