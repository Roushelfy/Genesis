"""Complete intrusive live validation of the experimental new-law sample scheduling."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v2 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

paths=[Path(__file__), Path(__file__).with_name('traction_warp_candidate_v2.py'),
       Path(__file__).with_name('traction_sample_warp_candidate_v1.py')]
before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
with patch.object(intrinsic, 'kernel_fit_tractions', kernel_fit_tractions), \
     patch.object(intrinsic, 'kernel_commit_intrinsic_contacts', kernel_commit_intrinsic_contacts), \
     patch.object(intrinsic, 'kernel_scatter_tractions', kernel_scatter_tractions):
    runpy.run_module('research.rigid_stress.probe_nonconvex_online', run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1])/'summary.json'
after={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
assert before==after
result=json.loads(output.read_text())
result['experimental_schedulers']={'fit':'warp_v2', 'commit_scatter':'sample_warp_v1'}
result['source_sha256'].update(before)
result['source_sha256_after'].update(after)
output.write_text(json.dumps(result, indent=2)+'\n')
