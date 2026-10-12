"""Compare valid-slot source initialization without changing current-contact capture."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import provenance

mode = sys.argv.pop(1)
assert mode in ("off", "on")
with patch.object(provenance, "_CAPTURE_VALID_SOURCES_ONLY", mode == "on"):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["valid_source_capture_only"] = mode == "on"
result["valid_source_capture_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
