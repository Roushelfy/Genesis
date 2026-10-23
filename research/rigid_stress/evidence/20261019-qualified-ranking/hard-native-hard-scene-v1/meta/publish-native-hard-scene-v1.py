"""Retain actual sharp-edge integration, failure replays and unchanged acceptance gates."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_smooth_minimum'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-hard-scene-v1', 'hard-native-hard-transition-v1',
         'hard-native-hard-scene-capture-v1', 'hard-native-hard-scene-replay-v1',
         'hard-native-hard-scene-capture-v2', 'hard-native-hard-scene-replay-v2',
         'hard-native-hard-boundary-replay-v3', 'hard-native-hard-boundary-replay-v4',
         'hard-native-hard-scene-convergence-v1', 'hard-native-hard-scene-quadrature-v1',
         'hard-native-hard-smooth-geometry-v1', 'hard-native-hard-smooth-minimum-v1',
         'hard-native-hard-boundary-motion-v1']
if (N / 'hard-native-hard-franka-timing-v1/submission.json').exists():
    names.append('hard-native-hard-franka-timing-v1')
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
            if code == '0':
                assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/native-hard-composition-terminal-v1.json')['frozen_heads']
for name in ['hard_contact_edge_scene', 'hard_contact_edge_convergence', 'hard_contact_edge_quadrature', 'hard_contact_smooth_minimum']:
    heads[name] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip()
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
scenes = {}
for campaign in ['hard-native-hard-scene-v1', 'hard-native-hard-scene-convergence-v1', 'hard-native-hard-scene-quadrature-v1', 'hard-native-hard-smooth-minimum-v1']:
    scenes[campaign] = {}
    for group, mode in enumerate(['full', 'max']):
        result = read(N / campaign / f'group{group}/{mode}/result.json')
        rows = result['rows']
        scenes[campaign][mode] = {'passed': result['passed'], 'captured_steps': len(rows),
            'all_environment_admitted_steps': sum(all(row['accepted']) for row in rows),
            'last_step': rows[-1]['step'], 'last_accepted': rows[-1]['accepted'],
            'last_input_status': rows[-1]['input_status'], 'last_recovery_status': rows[-1]['recovery_status'],
            'accepted_sharp_environment_steps': sum(sum(bool(ok) and count > 0 for ok, count in zip(row['accepted'], row.get('sharp_edge_regions', [0]*4))) for row in rows)}
validation = {}
for group, backend in [(2, 'cpu'), (3, 'gpu')]:
    for module in ['check_native_friction_history', 'check_native_friction_binding', 'check_hard_peak_semantics', 'check_curved_force_line', 'check_curved_friction_binding']:
        path = N / 'hard-native-hard-scene-convergence-v1' / f'group{group}/{backend}-{module}/result.json'
        result = read(path)
        assert result['passed']
        validation[f'{backend}-{module}'] = {'passed': True, 'rows': len(result['rows'])}
friction = {backend: read(N / 'hard-native-hard-scene-replay-v2' / f'group{group}/{backend}-friction/result.json') for group, backend in enumerate(['cpu', 'gpu'])}
assert all(row['passed'] for row in friction.values())
for row in friction.values():
    selected = next(item for item in row['rows'] if item['penalty'] == 1)
    assert selected['status'] == [0]*4 and selected['iterations'][0] == 398
boundary = read(N / 'hard-native-hard-boundary-replay-v4/group0/boundary/result.json')
geometry = read(N / 'hard-native-hard-smooth-geometry-v1/group0/geometry/result.json')
scope_path = N / 'meta/native-hard-scene-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'actual_scene_outcomes': scenes, 'validation': validation,
    'friction_replays': friction, 'boundary_quadrature_replay': boundary, 'smooth_geometry_diagnostic': geometry,
    'moving_boundary_replay': read(N / 'hard-native-hard-boundary-motion-v1/group0/boundary/result.json'),
    'smooth_minimum_controls': {name: read(N / 'hard-native-hard-smooth-minimum-v1' / rel) for name, rel in [('rotated_binding', 'group2/binding/result.json'), ('cpu', 'group3/cpu/result.json'), ('gpu', 'group4/gpu/result.json')]},
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-hard-scene-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files = [SCRIPT, HELPER_SOURCE, HELPER, REPLAY_HELPER, receipt_path, scope_path,
         N / 'meta/native-hard-scene-implementation-start-v1.json']
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
    for pattern in ['*.npz', '*.obj', '*.urdf']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts: add_identities(read(path))
large = {path for path in files if path.stat().st_size > 2*1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hard-scene-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-hard-scene-v1/README.md').write_text('# Native sharp-edge scene integration\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public experimental ancestor. '
    'See [measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#actual-sharp-edge-scene-integration). '
    'Replay captured tasks with their pinned engine and analysis sources, Slurm rtx-mid, image1_26, '
    'the established data-side Python environment and private node caches. Runtime arrays remain at hashed data paths. '
    'Failures are retained and excluded from useful stress throughput.\n')
with (N / 'meta/native-hard-scene-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
