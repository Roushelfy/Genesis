"""Retain native curved material operators and captured-source traction controls."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_curved_material'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('aa9cc8298')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [f'hard-native-curved-material-v{i}' for i in range(1, 4)]
names += ['hard-curved-material-corner-v1']
names += [f'hard-curved-hertz-residual-v{i}' for i in range(1, 4)]
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        code = (root / 'returncode.txt').read_text().strip()
        expected = ['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0']
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code)
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if code == '0':
                assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
allowed = [ALIAS, '797a982b1', '132dd4d0b', 'e7e49b738']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('check_native_curved_material', 'hard-native-curved-material-v3'),
                         ('study_curved_hertz_residual', 'hard-curved-hertz-residual-v3')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preservation = read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved']
preserved = 0
for name, data in preservation.items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/quadric-curvature-terminal-v1.json')['frozen_heads']
heads['hard_contact_curved_material'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
native = [read(N / f'hard-native-curved-material-v3/group{i}/curved-material/result.json') for i in [0, 1]]
assert all(row['passed'] and row['inverted_geometry_rejected'] for row in native)
residual = read(N / 'hard-curved-hertz-residual-v3/group0/residual/result.json')
assert residual['passed']
corner = read(N / 'hard-curved-material-corner-v1/group0/corner/result.json')
assert corner['passed'] and len(corner['checks']) == 2
scope = ('Native optional quadratic edge geometry now uses consistent curved volume stiffness/mass, boundary '
    'quadrature and complete corner/query stress gradients. Independent CPU/GPU same-load checks, max/full, '
    'cached/uncached gradients, released/rotating states and inverted geometry rejection pass, alongside '
    'unchanged affine shell, rotated normal, live friction/history/reset and corner-cancellation controls. '
    'The actual first-loaded normal source is separately tested with curved point versus distributed prescribed-Hertz '
    'boundary corrections. Linear source-centered fan integration removes a measured broad-range quadrature '
    'error; the independently refined distributed correction keeps its own wrench and field gates. This checks '
    'elasticity and load decomposition, not unknown contact-pressure accuracy, moving curved friction or admitted '
    'Franka throughput. Original compile failures and inaccurate quadrature attempts remain captured. '
    'Curved-source native transfer, material probes and moving max/full scene admission remain open.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'native_material_checks': native, 'source_residual_checks': residual, 'corner_checks': corner,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/curved-material-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
test_path = N / 'meta/curved-material-corner-test-v1.py'
with test_path.open('xb') as stream:
    stream.write((C3 / 'tests/rigid/test_stress_corner_precision.py').read_bytes())
assert test_path.read_bytes() == subprocess.check_output(['git', 'show', ALIAS + ':tests/rigid/test_stress_corner_precision.py'], cwd=C3)
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path, test_path]
external = {N / 'hard-native-quadric-franka-v1/group0/franka/frame0010.npz',
            N / 'hard-quadratic-rest-surface-v1/group0/geometry/geometry.npz'}
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    files.extend(root.glob('group*/corner/*.xml'))
    external.update(root.glob('group*/**/*.npz'))
    external.update(root.glob('group*/**/*.obj'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-curved-material-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-curved-material-v1/README.md').write_text('# Native curved material and source controls\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#native-curved-material-and-source-integration). '
    'Replay the captured task groups with the pinned engine, Slurm rtx-mid, image1_26 and the existing data-side '
    'environment. Use fresh explicit data outputs and private node caches. Final native and residual checks are '
    'in v3; earlier captures retain compilation failures and inaccurate quadrature. Decode gzip separately and '
    'leave immutable packet bytes intact. Full fields and generated meshes remain at hashed data paths.\n')
with (N / 'meta/curved-material-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
