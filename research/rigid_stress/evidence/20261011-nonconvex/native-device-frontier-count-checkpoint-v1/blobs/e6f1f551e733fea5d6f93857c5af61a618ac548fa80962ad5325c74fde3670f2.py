"""Device-resident fan frontier loop bound removes the per-step host maximum read."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import surface_frontier

mode = sys.argv.pop(1)
assert mode in ("off", "on")
original_profiles = surface_frontier.kernel_surface_fan_profiles


def selected_profiles(*args):
    return original_profiles(*args, device_maximum=mode == "on")


with patch.object(surface_frontier, "kernel_surface_fan_profiles", selected_profiles):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["device_frontier_count"] = mode == "on"
result["device_frontier_count_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
