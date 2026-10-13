"""Evaluate FP32 warm Newton Hessians with FP64 loads, primal gradients, cold fallback and final budgets."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import cooperative_fit

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(cooperative_fit, "_FP32_WARM_JACOBIAN", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["fp32_warm_jacobian"] = mode == "on"
result["fp32_warm_jacobian_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
