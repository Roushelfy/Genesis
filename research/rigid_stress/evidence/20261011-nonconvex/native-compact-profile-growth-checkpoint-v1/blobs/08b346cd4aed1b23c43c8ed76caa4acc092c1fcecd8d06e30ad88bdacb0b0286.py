"""Ablate lossless actual-sample capacity growth, retaining the full physical quadrature and all final checks."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(intrinsic, "_COMPACT_TRACTION_PROFILES", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["compact_traction_profiles"] = mode == "on"
result["compact_traction_profiles_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
