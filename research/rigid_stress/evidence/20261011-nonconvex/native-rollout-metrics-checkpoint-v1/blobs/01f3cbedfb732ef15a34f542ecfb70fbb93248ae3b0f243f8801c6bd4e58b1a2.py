"""Franka benchmark with timed device-side grasp and per-environment work collectors."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.speed_benchmark import rigid_stress as benchmark
from rollout_metrics_v2 import RolloutMetrics

scope = sys.argv[sys.argv.index("--scope") + 1]
assert scope in ("rigid", "live", "policy", "policy-rigid")
original_init = benchmark.FrankaEgg.__init__
original_restart = benchmark.FrankaEgg.restart
original_step = benchmark.FrankaEgg.step
monitors, records = {}, []


def tracked_init(workload, *args, **kwargs):
    original_init(workload, *args, **kwargs)
    monitors[workload] = RolloutMetrics(workload)


def tracked_restart(workload):
    monitor = monitors[workload]
    records.append(monitor.summary())
    original_restart(workload)
    monitor.reset()


def tracked_step(workload, residual_action=None):
    original_step(workload, residual_action)
    monitors[workload].record()


with (
    patch.object(benchmark.FrankaEgg, "__init__", tracked_init),
    patch.object(benchmark.FrankaEgg, "restart", tracked_restart),
    patch.object(benchmark.FrankaEgg, "step", tracked_step),
):
    benchmark.main()
assert len(monitors) == 1
records.append(next(iter(monitors.values())).summary())
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
assert len(records) == len(result["repeats"]) + 1
for record, row in zip(records[1:], result["repeats"]):
    assert np.array(record["per_environment"]["observed_steps"]).tolist() == [row["steps"]] * result["envs"]
    assert record["totals"]["observed_steps"] == row["steps"] * result["envs"]
    assert record["totals"]["lift_window_steps"] == record["totals"]["lift_grasp_pass_steps"] + record["totals"]["lift_grasp_fail_steps"]
    assert record["totals"]["invalid_stress_steps"] == row["invalid_environment_steps"]
result["trajectory_metrics"] = {"collector": __doc__, "warmup": records[0], "repeats": records[1:]}
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"trajectory_metrics": [{"totals": record["totals"], "work_percentiles": record["work_percentiles"]} for record in records]}, indent=2), flush=True)
