"""Evaluate clearing previous/current raw-contact diagnostics; valid-source capture is enabled in both routes."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(intrinsic, "_CLEAR_ACTIVE_RAW_DIAGNOSTICS", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["active_raw_diagnostics"] = mode == "on"
result["active_raw_diagnostics_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
