"""Prepare bounded projection controls after an unsubmitted task-plan syntax failure."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

n = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
w = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees')
before, after = w / 'hard_contact_surface_basis', w / 'hard_contact_edge_projection'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=after, text=True).strip() == '1c872039165fad4dc4b3c7e642ca30126050dae8'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=after)
partial = []
for suffix in ['cpu', 'gpu', 'profile']:
    root = n / f'hard-edge-projection-{suffix}-v1'
    assert root.exists() and not (root / 'submission.json').exists() and not (root / 'job.sh').exists()
    partial.append({'path': str(root), 'submitted': False, 'files': sum(p.is_file() for p in root.rglob('*'))})
with (n / 'meta/edge-projection-preflight-v1.json').open('x') as f:
    json.dump({'reason': 'Task-plan preparation raised SyntaxError before writing task files. Three submit calls then failed with FileNotFoundError while reading those absent plans. Only analysis-source copies exist; no sbatch invocation occurred. Partial directories are retained. The runner now reads task JSON before creating output directories.', 'partial_directories': partial}, f, indent=2)
    f.write('\n')
mesh = n / 'hard-native-curved-material-v3/group1/curved-material/estimated.npz'
old = json.loads((n / 'hard-curved-source-geometry-v2/tasks.json').read_text())
plans = {}
for name, configs in [('hard-edge-projection-cpu-v2', [('cpu', '64')]),
                      ('hard-edge-projection-gpu-v2', [('gpu', '64'), ('gpu', '32')])]:
    groups = []
    for index, (backend, precision) in enumerate(configs):
        args = ['--backend', backend, '--precision', precision, '--mesh', str(mesh)]
        group = [
            {'name': 'baseline', 'module': 'research.rigid_stress.qipc_comparison.check_material_projection', 'args': args, 'source': str(before)},
            {'name': 'candidate', 'module': 'research.rigid_stress.qipc_comparison.check_material_projection', 'args': args + ['--baseline', str(n / name / f'group{index}/baseline/fields.npz')]},
        ]
        if precision == '64':
            group.append(old[0 if backend == 'cpu' else 1][0])
            group.append({'name': 'bounds', 'module': 'research.rigid_stress.qipc_comparison.check_source_bounds', 'args': ['--backend', 'cpu' if backend == 'cpu' else 'cuda', '--source', str(after)]})
        groups.append(group)
    plans[name] = groups
args = json.loads((n / 'hard-native-surface-unroll-profile-v1/tasks.json').read_text())[0][0]['args']
profile = []
for name, source in [('baseline', before), ('candidate', after)]:
    profile.append({'name': name, 'module': 'research.rigid_stress.qipc_comparison.profile_hard_franka', 'args': args, 'source': str(source)})
plans['hard-edge-projection-profile-v2'] = [profile]
for name, groups in plans.items():
    with (n / 'meta' / f'{name}-tasks.json').open('x') as f:
        json.dump(groups, f, indent=2)
        f.write('\n')
heads = json.loads((n / 'meta/surface-basis-terminal-v1.json').read_text())['frozen_heads']
heads[after.name] = '1c872039165fad4dc4b3c7e642ca30126050dae8'
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=w / name, text=True).strip() == sha
preserved = 0
for name, data in json.loads((n / 'meta/hard-scene-pilot-start-v1.json').read_text())['preservation']['preserved'].items():
    for path, sha in data['files'].items():
        assert hashlib.sha256((w / name / path).read_bytes()).hexdigest() == sha
        preserved += 1
with (n / 'meta/edge-projection-start-v1.json').open('x') as f:
    json.dump({'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'frozen_heads': heads, 'preserved_files': preserved, 'candidate': heads[after.name], 'baseline': heads[before.name]}, f, indent=2)
    f.write('\n')
print({k: len(v) for k, v in plans.items()})
