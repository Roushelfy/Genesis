"""Measure the complete new-law pipeline with experimental sample scheduling."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v2 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

with patch.object(intrinsic, 'kernel_fit_tractions', kernel_fit_tractions), \
     patch.object(intrinsic, 'kernel_commit_intrinsic_contacts', kernel_commit_intrinsic_contacts), \
     patch.object(intrinsic, 'kernel_scatter_tractions', kernel_scatter_tractions):
    runpy.run_module('examples.speed_benchmark.rigid_stress', run_name='__main__')
output = Path(sys.argv[sys.argv.index('--output')+1])
result = json.loads(output.read_text())
result['experimental_fit_scheduler'] = 'cuda_warp_candidate_v2'
result['experimental_sample_scheduler'] = 'cuda_warp_candidate_v1'
for name in ('traction_warp_candidate_v2.py', 'traction_sample_warp_candidate_v1.py', Path(__file__).name):
    path = Path(__file__).with_name(name)
    result['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({'experimental_sample_scheduler':result['experimental_sample_scheduler']}), flush=True)
