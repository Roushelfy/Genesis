"""Test or profile symmetric Hessian and single-lane direction reuse."""
import hashlib
import json
import runpy
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v7 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            for name, function in [('kernel_fit_tractions', kernel_fit_tractions), ('kernel_scatter_tractions', kernel_scatter_tractions)]:
                if name in item.module.__dict__:
                    item.module.__dict__[name] = function

with patch.object(intrinsic, 'kernel_fit_tractions', kernel_fit_tractions), \
     patch.object(intrinsic, 'kernel_commit_intrinsic_contacts', kernel_commit_intrinsic_contacts), \
     patch.object(intrinsic, 'kernel_scatter_tractions', kernel_scatter_tractions):
    if '--test' in sys.argv:
        status = pytest.main(['-q', '-s', '-n', '0', '--tb=short', '--backend', 'cuda',
            'tests/rigid/test_stress_shared_support.py', 'tests/rigid/test_stress_shared_lifecycle.py',
            'tests/rigid/test_stress_surface_fan.py', 'tests/rigid/test_stress_traction.py'], plugins=[CandidatePlugin()])
        raise SystemExit(status)
    runpy.run_module('examples.speed_benchmark.rigid_stress', run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1])
result=json.loads(output.read_text())
result['experimental_fit_scheduler']='unregularized_newton_with_singular_ridge_fallback_v7'
result['experimental_sample_scheduler']='cuda_warp_candidate_v1'
for name in ('traction_warp_candidate_v7.py','traction_sample_warp_candidate_v1.py',Path(__file__).name):
    path=Path(__file__).with_name(name)
    result['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
output.write_text(json.dumps(result,indent=2)+'\n')
