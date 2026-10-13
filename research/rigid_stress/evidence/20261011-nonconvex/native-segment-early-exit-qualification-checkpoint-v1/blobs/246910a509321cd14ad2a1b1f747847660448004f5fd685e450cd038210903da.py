"""Evaluate exiting monotone-inactive surface walks; four selected optimizations are enabled in both routes."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import paths

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(paths, "_STOP_INACTIVE_SURFACE_SEGMENT", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["segment_early_exit"] = mode == "on"
result["segment_early_exit_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
