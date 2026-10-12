"""First-triangle segment loop bound ablation with unchanged Quadrants distance arithmetic."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from baseline_surface_segment_v1 import func_surface_segment as baseline_segment
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import footprint, paths

mode = sys.argv.pop(1)
assert mode in ("on", "off")
selected = paths.func_surface_segment if mode == "on" else baseline_segment
with patch.object(paths, "func_surface_segment", selected), patch.object(footprint, "func_surface_segment", selected):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["first_triangle_segment_bounds"] = mode == "on"
result["segment_bounds_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
