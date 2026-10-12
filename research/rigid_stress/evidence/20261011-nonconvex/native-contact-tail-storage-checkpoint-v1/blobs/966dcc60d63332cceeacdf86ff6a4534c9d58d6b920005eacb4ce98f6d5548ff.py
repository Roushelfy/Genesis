"""Ablate warp-parallel refined fan sampling with the same physical sample cloud."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import fan_refinement

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(fan_refinement, "_PARALLEL_FAN_REFINEMENT", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["parallel_refined_fan"] = mode == "on"
result["parallel_refined_fan_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
