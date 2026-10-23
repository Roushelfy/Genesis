"""Retain loaded-region acceleration and source/face-localized boundary diagnostics."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_boundary_hyperbolic'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-hard-boundary-sources-v1', 'hard-native-hard-input-profile-v1',
         'hard-native-hard-boundary-inspection-v1', 'hard-native-hard-boundary-polar-v1',
         'hard-native-hard-input-profile-v2', 'hard-native-hard-active-allocation-v1',
         'hard-native-hard-boundary-closest-v1', 'hard-native-hard-active-scaling-v1',
         'hard-native-hard-active-profile-v1', 'hard-native-hard-boundary-faces-v1',
         'hard-native-hard-boundary-radial-v1', 'hard-native-hard-boundary-hyperbolic-v1',
         'hard-native-hard-boundary-hyperbolic-v2', 'hard-native-hard-boundary-hyperbolic-v3']
for name in ['hard-native-hard-hyperbolic-scene-v1', 'hard-native-hard-hyperbolic-faces-v1', 'hard-native-hard-friction110-v1']:
    if (N / name / 'submission.json').exists(): names.append(name)
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
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code, observed[f'{job}_{group}'])
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if code == '0': assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/native-hard-scene-terminal-v1.json')['frozen_heads']
for name in ['hard_contact_boundary_polar', 'hard_contact_active_allocation', 'hard_contact_boundary_closest',
             'hard_contact_boundary_radial', 'hard_contact_boundary_hyperbolic']:
    heads[name] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip()
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
costs = []
for group in [0, 1]:
    root = N / f'hard-native-hard-active-scaling-v1/group{group}'
    baseline, candidate = (read(root / mode / 'timing.json') for mode in ['baseline', 'candidate'])
    conformance = read(root / 'candidate/conformance.json')
    assert conformance['passed'] and len(conformance['fields']) == 25
    costs.append({'batch': baseline['batch'], 'baseline': baseline, 'candidate': candidate,
                  'speedup': baseline['median_seconds'] / candidate['median_seconds'], 'conformance': conformance})
validation = {}
for backend, relative in [('gpu', 'hard-native-hard-active-allocation-v1/group2/gpu'),
                          ('cpu', 'hard-native-hard-active-scaling-v1/group2/cpu')]:
    result = read(N / relative / 'result.json')
    assert result['passed'] and result['sparse_permuted_750_slot_unload_restoration']
    validation[backend] = result
scope_path = N / 'meta/native-hard-input-boundary-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'paired_input_costs': costs, 'allocation_validation': validation,
    'boundary_sources': read(N / 'hard-native-hard-boundary-sources-v1/group0/sources/result.json'),
    'boundary_faces': read(N / 'hard-native-hard-boundary-faces-v1/group0/sources/result.json'),
    'boundary_hyperbolic': read(N / 'hard-native-hard-boundary-hyperbolic-v3/group1/combined/result.json'),
    'actual_scene_outcomes': read(N / 'meta/native-hard-input-boundary-scene-summary-v1.json'),
    'friction110_replay': read(N / 'hard-native-hard-friction110-v1/group0/cpu/result.json'),
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-hard-input-boundary-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files = [SCRIPT, HELPER_SOURCE, HELPER, REPLAY_HELPER, receipt_path, scope_path, N / 'meta/native-hard-input-boundary-scene-summary-v1.json']
external = set()
def add_identities(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values(): add_identities(item)
    elif isinstance(value, list):
        for item in value: add_identities(item)
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    for pattern in ['*.npz', '*.obj', '*.urdf']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts: add_identities(read(path))
large = {path for path in files if path.stat().st_size > 2*1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hard-input-boundary-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-hard-input-boundary-v1/README.md').write_text('# Loaded-region allocation and moving boundary integration\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is an experimental ancestor. '
    'See [measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#loaded-region-allocation-and-moving-boundary-diagnosis). '
    'Replay the captured tasks with their pinned engine and analysis sources, Slurm rtx-mid, image1_26, '
    'the data-side Python environment and private node caches. Full arrays remain at hashed data paths. '
    'Input-only costs and rejected stress frames are excluded from useful complete scene throughput.\n')
with (N / 'meta/native-hard-input-boundary-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
