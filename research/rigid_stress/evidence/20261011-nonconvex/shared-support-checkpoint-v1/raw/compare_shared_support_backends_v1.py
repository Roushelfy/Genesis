"""Compare canonical nodal loads and independent full fields from the two native union-support backends."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import splu
from research.rigid_stress.mechanics import P2Shell
from research.rigid_stress.field_cpu import stress_field
from genesis.options.rigid_stress import RigidStressOptions

root = Path(__file__).parent
cpu_root, gpu_root = (root / run / 'results' for run in ('shared-plane-support-v5', 'shared-plane-support-cuda-v1'))
cpu, gpu = (json.loads((folder / 'summary.json').read_text()) for folder in (cpu_root, gpu_root))
output = root / 'shared-support-backends-v1'
output.mkdir(exist_ok=False)
options = RigidStressOptions(mesh=Path('examples/rigid/assets/hollow_egg/level1/elastic.npz'), method='direct')
with np.load(options.mesh) as saved:
    oracle = P2Shell(saved['vertices'], saved['tetrahedra'], saved['surface_triangles'],
        options.young, options.poisson, options.density, 2, factor_backend='none')
assert len(cpu['records']) == len(gpu['records']) == 2
records = []
for cpu_record, gpu_record in zip(cpu['records'], gpu['records']):
    assert cpu_record['original_inputs'] == gpu_record['original_inputs'] or all(
        {key: value for key, value in first.items() if key != 'samples'} ==
        {key: value for key, value in second.items() if key != 'samples'}
        for first, second in zip(cpu_record['original_inputs'], gpu_record['original_inputs']))
    name = f"{cpu_record['run']}-{cpu_record['tick']}-support.npz"
    with np.load(cpu_root / name) as cpu_saved, np.load(gpu_root / name) as gpu_saved:
        for first, second in zip(cpu_record['fit'], gpu_record['fit']):
            assert first['case'] == second['case']
            label = first['case']
            cpu_nodal, gpu_nodal = (saved[f'{label}-native-nodal'] for saved in (cpu_saved, gpu_saved))
            np.testing.assert_allclose(cpu_nodal, gpu_nodal, rtol=1e-4, atol=1e-10)
            np.testing.assert_allclose(first['oracle']['native_peak_Pa'], second['oracle']['native_peak_Pa'], rtol=1e-4, atol=1e-3)
            records.append({'tick': cpu_record['tick'], 'case': label,
                'maximum_nodal_difference_N': float(np.max(np.abs(cpu_nodal-gpu_nodal))),
                'relative_nodal_difference': float(np.linalg.norm(cpu_nodal-gpu_nodal)/np.linalg.norm(cpu_nodal)),
                'native_peak_difference_Pa': float(abs(first['oracle']['native_peak_Pa']-second['oracle']['native_peak_Pa']))})
(output / 'summary.json').write_text(json.dumps({'scope': __doc__, 'records': records}, indent=2)+'\n')
print(json.dumps({'scope': __doc__, 'records': records}), flush=True)
