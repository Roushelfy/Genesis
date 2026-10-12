"""Compare Q-contiguous scalar cached positions and profile/index layouts."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import traction

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(traction, "_SAMPLE_CONTIGUOUS_CACHED_POSITIONS", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["sample_contiguous_cached_positions"] = mode == "on"
result["sample_contiguous_cached_positions_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
