"""Check experimental sample scheduling against the independent contact assertions."""
from unittest.mock import patch
import pytest
import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v2 import kernel_fit_tractions
from traction_sample_warp_candidate_v1 import kernel_commit_intrinsic_contacts, kernel_scatter_tractions

class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            for name, function in [('kernel_fit_tractions', kernel_fit_tractions),
                                   ('kernel_scatter_tractions', kernel_scatter_tractions)]:
                if name in item.module.__dict__:
                    item.module.__dict__[name] = function

with patch.object(intrinsic, 'kernel_fit_tractions', kernel_fit_tractions), \
     patch.object(intrinsic, 'kernel_commit_intrinsic_contacts', kernel_commit_intrinsic_contacts), \
     patch.object(intrinsic, 'kernel_scatter_tractions', kernel_scatter_tractions):
    status = pytest.main(['-q', '-n', '0', '--tb=short', '--backend', 'cuda',
        'tests/rigid/test_stress_shared_support.py', 'tests/rigid/test_stress_shared_lifecycle.py',
        'tests/rigid/test_stress_surface_fan.py', 'tests/rigid/test_stress_traction.py'], plugins=[CandidatePlugin()])
raise SystemExit(status)
