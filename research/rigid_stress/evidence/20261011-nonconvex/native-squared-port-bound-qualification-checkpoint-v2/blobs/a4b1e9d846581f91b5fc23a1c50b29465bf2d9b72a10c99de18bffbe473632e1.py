"""Evaluate a conservative squared remaining-path bound; selected coalesced layout and valid-source capture are enabled in both routes."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import footprint

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(footprint, "_SQUARED_PORT_DISTANCE_BOUND", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["squared_port_distance_bound"] = mode == "on"
result["squared_port_distance_bound_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
