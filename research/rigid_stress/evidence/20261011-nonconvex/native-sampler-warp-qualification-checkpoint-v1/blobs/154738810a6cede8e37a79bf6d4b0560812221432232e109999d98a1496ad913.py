"""Matched complete changing rollout with only fan counter aggregation toggled."""
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

from genesis.engine.solvers.rigid.stress import surface_frontier

mode = sys.argv.pop(1)
assert mode in ("off", "on")
output = Path(sys.argv[sys.argv.index("--output") + 1])
with patch.object(surface_frontier, "_AGGREGATE_FAN_COUNTERS", mode == "on"):
    runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__", alter_sys=True)
result = json.loads(output.read_text())
result["sampler_warp_counters"] = mode == "on"
result["sampler_warp_note"] = __doc__
output.write_text(json.dumps(result, indent=2) + "\n")
