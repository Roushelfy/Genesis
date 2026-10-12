"""Exact nonnegative final-segment lower-bound pruning after current-step seed reuse."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from baseline_footprint_point_distance_v1 import func_footprint_point_distance as baseline_distance
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import fan_refinement, footprint, surface_frontier

mode = sys.argv.pop(1)
assert mode in ("on", "off")
selected = footprint.func_footprint_point_distance if mode == "on" else baseline_distance
with (
    patch.object(footprint, "func_footprint_point_distance", selected),
    patch.object(surface_frontier, "func_footprint_point_distance", selected),
    patch.object(fan_refinement, "func_footprint_point_distance", selected),
):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["nonnegative_port_bounds"] = mode == "on"
result["port_bounds_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
