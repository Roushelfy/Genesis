"""FP64 producer control retaining patch witnesses instead of overriding their normals with one EPA normal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from unittest import mock

candidate_path = Path(__file__).with_name('gjk_patch_normal_candidate_v1.py')
spec = importlib.util.spec_from_file_location('genesis.engine.solvers.rigid.collider.gjk_patch_candidate', candidate_path)
candidate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = candidate
spec.loader.exec_module(candidate)
output = Path(sys.argv[sys.argv.index('--output')+1])
try:
    with mock.patch('genesis.engine.solvers.rigid.collider.gjk.func_gjk_contact', candidate.func_gjk_contact):
        runpy.run_module('research.rigid_stress.probe_nonconvex_online', run_name='__main__')
finally:
    summary = output / 'summary.json'
    if summary.exists():
        result = json.loads(summary.read_text())
        result['diagnostic_patch_normal'] = 'FP64 patch-witness normal; all existing manifold witnesses retained'
        result['candidate_source_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in (Path(__file__), candidate_path)}
        summary.write_text(json.dumps(result, indent=2)+'\n')
