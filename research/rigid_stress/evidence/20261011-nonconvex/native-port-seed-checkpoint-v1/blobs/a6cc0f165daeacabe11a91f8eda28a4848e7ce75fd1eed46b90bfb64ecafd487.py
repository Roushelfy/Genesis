"""Exact current-step anchor-port seed reuse ablation; all numerics remain Quadrants."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

from examples.speed_benchmark import rigid_stress as benchmark
from genesis.engine.solvers.rigid.stress import intrinsic

mode = sys.argv.pop(1)
assert mode in ("on","off")
original_create = intrinsic.create_footprints

def create_plain(n_ports,n_contacts,n_envs,n_face_ports=None):
    return original_create(n_ports,n_contacts,n_envs)

with patch.object(intrinsic,"create_footprints",original_create if mode == "on" else create_plain):
    benchmark.main()
path = Path(sys.argv[sys.argv.index("--output")+1])
result = json.loads(path.read_text())
result["current_step_port_seed_reuse"] = mode == "on"
result["port_seed_reuse_note"] = __doc__
path.write_text(json.dumps(result,indent=2)+"\n")
