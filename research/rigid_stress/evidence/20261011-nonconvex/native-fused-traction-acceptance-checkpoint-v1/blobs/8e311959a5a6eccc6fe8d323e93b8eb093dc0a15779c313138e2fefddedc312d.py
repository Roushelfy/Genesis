"""One final projection pass for original-point wrench, local cones and P2 scatter."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic

mode = sys.argv.pop(1)
assert mode in ("off", "on")
surface_factory = intrinsic.create_surface_fan_tractions
anchor_factory = intrinsic.create_anchor_fan_tractions


def surface_tractions(*args, **kwargs):
    kwargs["cached_acceptance"] = mode == "on"
    return surface_factory(*args, **kwargs)


def anchor_tractions(*args, **kwargs):
    kwargs["cached_acceptance"] = mode == "on"
    return anchor_factory(*args, **kwargs)


with (
    patch.object(intrinsic, "create_surface_fan_tractions", surface_tractions),
    patch.object(intrinsic, "create_anchor_fan_tractions", anchor_tractions),
):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(path.read_text())
result["fused_traction_acceptance"] = mode == "on"
result["fused_traction_acceptance_note"] = __doc__
path.write_text(json.dumps(result, indent=2) + "\n")
