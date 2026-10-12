"""Ablate current-step chart-position reuse with counted complete sampling and eight-lane retry fixed."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(intrinsic, "_CACHE_FAN_POSITIONS", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["cached_fan_positions"] = mode == "on"
result["cached_fan_positions_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
