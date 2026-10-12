"""Ablate a complete cooperative refit of changed fan supports with masked serial fallback."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import traction

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(traction, "_COOPERATIVE_REFINED_FANS", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["cooperative_refined_fans"] = mode == "on"
result["cooperative_refined_fans_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
