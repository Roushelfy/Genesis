"""Explicit producer ablation of near-parallel patch feature tolerances; no stress load filtering."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest import mock
import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg


class ExactAlignmentFrankaEgg(FrankaEgg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        info = self.scene.rigid_solver.collider.collider_info.gjk
        info.contact_face_tol.fill(1.0 - 64.0 * np.finfo(gs.np_float).eps)
        info.contact_edge_tol.fill(1e-10)
        print(json.dumps({'diagnostic_patch_face_cosine_min': 1.0-64.0*np.finfo(gs.np_float).eps,
                          'diagnostic_patch_edge_tolerance': 1e-10}), flush=True)


output = Path(sys.argv[sys.argv.index('--output')+1])
try:
    with mock.patch('examples.rigid.franka_egg_stress.FrankaEgg', ExactAlignmentFrankaEgg):
        runpy.run_module('research.rigid_stress.probe_nonconvex_online', run_name='__main__')
finally:
    summary = output / 'summary.json'
    if summary.exists():
        result = json.loads(summary.read_text())
        result['diagnostic_collision_patch_alignment'] = {'face_cosine_min': 1.0-64.0*np.finfo(gs.np_float).eps,
                                                        'edge_tolerance': 1e-10}
        result['driver_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        summary.write_text(json.dumps(result, indent=2)+'\n')
