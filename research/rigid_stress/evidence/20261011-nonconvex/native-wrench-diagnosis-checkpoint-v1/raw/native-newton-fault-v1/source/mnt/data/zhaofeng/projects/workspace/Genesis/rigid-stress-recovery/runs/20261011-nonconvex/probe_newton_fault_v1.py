"""Check that restoring mandatory ridge rejects the newly feasible first-failure regression."""
from pathlib import Path
import json

import quadrants as qd
import pytest
import genesis as gs
import genesis.engine.solvers.rigid.stress.traction as traction

original_direction = traction.func_traction_direction

@qd.func
def mandatory_ridge_direction(matrix: qd.types.matrix(6,6), gradient: qd.types.vector(6), ridge: float):
    ridge = 1e-12*qd.max(1.0,matrix.trace()/6.0)
    return original_direction(matrix,gradient,ridge)

traction.func_traction_direction = mandatory_ridge_direction
status = pytest.main(['-q','-s','-n','0','--tb=short','--backend','cuda',
    'tests/rigid/test_stress_surface_fan.py::test_saved_edge_contact_uses_neighbor_finite_area_and_cpu_oracle[517-2]'])
root=Path(__file__).parent
(root/'native-newton-fault-v1/result.json').write_text(json.dumps({'pytest_status':status,
    'expected':'The unchanged successful-recovery assertion rejects the mandatory-ridge control.'},indent=2)+'\n')
if status == 0:
    raise RuntimeError('The regression survived restoration of the diagnosed fitting defect.')
