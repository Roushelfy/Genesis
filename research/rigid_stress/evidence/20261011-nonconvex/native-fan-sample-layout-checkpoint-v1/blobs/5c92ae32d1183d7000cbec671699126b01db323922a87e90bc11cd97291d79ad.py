"""Exact Q10 fan sampler task-order ablation after current-step seed reuse."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from baseline_fan_samples_v1 import kernel_fan_frontier_samples as baseline_samples
from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import surface_frontier

mode = sys.argv.pop(1)
assert mode in ("on", "off")
selected = surface_frontier.kernel_fan_frontier_samples if mode == "on" else baseline_samples
with patch.object(surface_frontier, "kernel_fan_frontier_samples", selected):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["fan_sample_major"] = mode == "on"
result["fan_sample_layout_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
