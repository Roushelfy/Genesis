"""Ablate bounded sample-lane retry against the complete serial fallback at unchanged final budgets."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import traction

lanes = int(sys.argv.pop(1))
assert lanes in (0, 1, 2, 4, 8, 16, 32)
with patch.object(traction, "_RETRY_SAMPLE_LANES", lanes):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["retry_sample_lanes"] = lanes
result["retry_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
