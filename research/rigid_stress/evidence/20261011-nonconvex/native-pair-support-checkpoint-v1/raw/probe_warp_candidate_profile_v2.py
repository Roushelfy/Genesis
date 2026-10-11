"""Intrusive matched-stage benchmark of a Quadrants scheduling candidate."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v2 import kernel_fit_tractions

candidate = Path(__file__).with_name("traction_warp_candidate_v2.py")
with patch.object(intrinsic, "kernel_fit_tractions", kernel_fit_tractions):
    runpy.run_module("examples.speed_benchmark.rigid_stress", run_name="__main__")
output = Path(sys.argv[sys.argv.index("--output") + 1])
result = json.loads(output.read_text())
result["experimental_fit_scheduler"] = "cuda_warp_candidate_v2"
result["source_sha256"][str(candidate)] = hashlib.sha256(candidate.read_bytes()).hexdigest()
result["source_sha256"][str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"experimental_fit_scheduler": result["experimental_fit_scheduler"]}), flush=True)
