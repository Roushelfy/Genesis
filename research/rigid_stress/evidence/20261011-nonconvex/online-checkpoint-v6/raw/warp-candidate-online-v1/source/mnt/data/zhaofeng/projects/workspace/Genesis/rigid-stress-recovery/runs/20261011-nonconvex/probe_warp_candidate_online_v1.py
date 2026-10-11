"""Evaluate cooperative fitting on unchanged public live validation and independent assertions."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

from traction_warp_candidate_v1 import kernel_fit_tractions

output = Path(sys.argv[sys.argv.index('--output') + 1])
try:
    with mock.patch('genesis.engine.solvers.rigid.stress.intrinsic.kernel_fit_tractions', kernel_fit_tractions):
        runpy.run_module('research.rigid_stress.probe_nonconvex_online', run_name='__main__')
finally:
    summary = output / 'summary.json'
    if summary.exists():
        result = json.loads(summary.read_text())
        result['experimental_fit_scheduler'] = 'cuda_warp_candidate_v1'
        result['candidate_source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name('traction_warp_candidate_v1.py'))}
        summary.write_text(json.dumps(result, indent=2) + '\n')
