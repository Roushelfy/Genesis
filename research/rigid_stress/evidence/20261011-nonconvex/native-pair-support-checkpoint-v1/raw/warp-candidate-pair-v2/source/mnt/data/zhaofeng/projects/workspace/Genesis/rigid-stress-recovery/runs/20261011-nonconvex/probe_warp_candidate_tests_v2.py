"""Exercise the declared shared support and unchanged independent assertions with CUDA warp fitting."""
from unittest.mock import patch
import pytest
import genesis.engine.solvers.rigid.stress.intrinsic as intrinsic
from traction_warp_candidate_v2 import kernel_fit_tractions

class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            if 'kernel_fit_tractions' in item.module.__dict__:
                item.module.kernel_fit_tractions = kernel_fit_tractions

with patch.object(intrinsic, 'kernel_fit_tractions', kernel_fit_tractions):
    status = pytest.main(['-q', '-n', '0', '--tb=short', '--backend', 'cuda',
        'tests/rigid/test_stress_shared_support.py', 'tests/rigid/test_stress_shared_lifecycle.py',
        'tests/rigid/test_stress_surface_fan.py', 'tests/rigid/test_stress_traction.py'], plugins=[CandidatePlugin()])
raise SystemExit(status)
