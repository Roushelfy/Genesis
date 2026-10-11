"""Run unchanged CPU/CUDA stress oracle assertions with the candidate path predicate."""
import sys
import pytest
from surface_segment_candidate_v1 import func_surface_segment
from genesis.engine.solvers.rigid.stress import footprint, paths


class CandidatePlugin:
    def pytest_collection_modifyitems(self, items):
        footprint.func_surface_segment = func_surface_segment
        paths.func_surface_segment = func_surface_segment


raise SystemExit(pytest.main(['-q', '-n', '0', '--tb=short',
    'tests/rigid/test_stress_surface_fan.py', '--backend', sys.argv[1]], plugins=[CandidatePlugin()]))
