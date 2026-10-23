"""Retain boundary quadrature scheduling and the single-region wrench failure."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_boundary_lanes'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-geometry-cycle-v1', 'hard-native-allocation428-v1']
names += ['hard-native-active-boundary-' + suffix + '-v1' for suffix in ['controls', 'timing', 'profile', 'lifecycle', 'cycle-timing']]
names += ['hard-native-boundary-lanes-' + suffix + '-v1' for suffix in ['controls', 'scene', 'timing', 'profile', 'differences', 'observations', 'capture428']]
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        code_path = root / 'returncode.txt'
        code = code_path.read_text().strip() if code_path.is_file() else None
        actual_state, exit_code = observed[f'{job}_{group}']
        assert actual_state in ['COMPLETED', 'FAILED', 'TIMEOUT'], (name, actual_state)
        if actual_state == 'COMPLETED': assert code == '0' and exit_code == '0:0'
        elif actual_state == 'FAILED': assert code == '1' and exit_code == '1:0'
        else: assert code in [None, '124', '137', '143']
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if actual_state == 'COMPLETED': assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': actual_state, 'exit_code': exit_code,
                            'wrapper_return_code': code, 'tasks': task_states})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/native-geometry-search-terminal-v1.json')['frozen_heads']
for name in ['hard_contact_active_boundary', 'hard_contact_boundary_lanes']:
    heads[name] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip()
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
controls = {}
for kind in ['active-boundary', 'boundary-lanes']:
    controls[kind] = {}
    for group, backend in [(0, 'cpu'), (1, 'gpu')]:
        controls[kind][backend] = {}
        for task in ['shell-q16', 'normal-component'] + (['live-friction'] if backend == 'gpu' else []):
            result = read(N / f'hard-native-{kind}-controls-v1/group{group}/{task}/result.json')
            assert result['passed']
            controls[kind][backend][task] = result
lifecycle = {}
for group, backend in [(0, 'cpu'), (1, 'gpu')]:
    lifecycle[backend] = {}
    for task in ['baseline', 'candidate']:
        result = read(N / f'hard-native-active-boundary-lifecycle-v1/group{group}/{task}/result.json')
        assert result['passed'] and result['release_and_reloading_isolation_passed']
        lifecycle[backend][task] = result
scene_timing, profiles = {}, {}
for kind in ['active-boundary', 'boundary-lanes']:
    scene_timing[kind] = []
    for group, batch_size in [(0, 4), (1, 64)]:
        result = read(N / f'hard-native-{kind}-timing-v1/group{group}/paired-b{batch_size}/result.json')
        assert result['passed'] and len(result['rows']) == 3
        scene_timing[kind].extend(result['rows'])
    result = read(N / f'hard-native-{kind}-profile-v1/group0/profile/result.json')
    assert result['passed'] and not result['primary_timing']
    profiles[kind] = result
scenes = {}
for group, mode in [(0, 'full'), (1, 'max')]:
    result = read(N / f'hard-native-boundary-lanes-scene-v1/group{group}/{mode}/result.json')
    assert result['passed'] and len(result['rows']) == 130
    assert all(all(row['accepted']) for row in result['rows'])
    scenes[mode] = {'frames': 130, 'admitted_environment_steps': 520}
observations = []
for group in [0, 1]:
    result = read(N / f'hard-native-boundary-lanes-observations-v1/group{group}/observations/result.json')
    assert result['passed']
    observations.append(result)
cycle = {}
for task, campaign in [('baseline', 'hard-native-active-boundary-cycle-timing-v1'),
                       ('candidate', 'hard-native-active-boundary-cycle-timing-v1'),
                       ('capture428', 'hard-native-boundary-lanes-capture428-v1')]:
    result = read(N / f'{campaign}/group0/{task}/result.json')
    assert not result['passed'] and result['failure'] == {'batch': 4, 'mode': 'max', 'step': 428}
    frame = read(N / f'{campaign}/group0/{task}/failure-max-b4/frame.json')
    assert frame['accepted'] == [False, True, True, True] and frame['input_status'] == [32, 0, 0, 0]
    cycle[task] = {'failure': result['failure'], 'admission': frame['accepted'], 'input_status': frame['input_status']}
certificates = {}
for group, backend in [(0, 'cpu'), (1, 'gpu')]:
    result = read(N / f'hard-native-allocation428-v1/group{group}/{backend}/result.json')
    assert result['passed'] and result['rows'][0]['linear_infeasibility_proved_at_native_1e_12_gate']
    assert result['rows'][0]['linear_rank'] == 3 and result['rows'][0]['certificate_null_error'] == 0
    assert all(row['native_status'] == 0 for row in result['rows'][1:])
    certificates[backend] = result

differences = {}
for campaign in ['hard-native-boundary-lanes-differences-v1']:
    for path in sorted((N / campaign).glob('group*/*/result.json')):
        result = read(path)
        assert result['passed'] and result['diagnostic_only']
        fields = {}
        for row in result['rows']:
            for value in row['fields']:
                field = fields.setdefault(value['field'], {'maximum_absolute_difference': 0.,
                    'maximum_relative_l2_difference': 0., 'nonconformant_frames': 0, 'nonexact_frames': 0})
                field['maximum_absolute_difference'] = max(field['maximum_absolute_difference'], value['maximum_absolute_difference'])
                field['maximum_relative_l2_difference'] = max(field['maximum_relative_l2_difference'], value['relative_l2_difference'])
                field['nonconformant_frames'] += not value['conformant']
                field['nonexact_frames'] += not value['exact']
        differences[str(path.relative_to(N))] = {'frames': len(result['rows']),
            'all_fields_conformant': result['all_fields_conformant'], 'fields': fields}
bindings = []
for module, campaign in [('check_live_curved_hertz_franka', 'hard-native-boundary-lanes-capture428-v1'),
                         ('check_native_hertz_recovery', 'hard-native-boundary-lanes-controls-v1'),
                         ('time_hard_franka', 'hard-native-boundary-lanes-timing-v1'),
                         ('compare_hard_frames', 'hard-native-boundary-lanes-differences-v1'),
                         ('inspect_hard_allocation', 'hard-native-allocation428-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope_path = N / 'meta/native-boundary-lanes-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'boundary_controls': controls, 'release_lifecycle_controls': lifecycle,
    'paired_scene_timing': scene_timing, 'separate_scene_profiles': profiles,
    'actual_scene_outcomes': scenes, 'public_observation_conformance': observations,
    'raw_field_differences': differences, 'cycle_failures': cycle, 'allocation_certificates': certificates,
    'original_timeout': read(N / 'meta/native-geometry-cycle-timeout-verified-v1.json'),
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-boundary-lanes-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files = [SCRIPT, HELPER_SOURCE, HELPER, REPLAY_HELPER, receipt_path, scope_path, N / 'meta/native-geometry-cycle-timeout-verified-v1.json']
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
    for pattern in ['*.npz', '*.obj', '*.urdf', '*.pstats']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts: add_identities(read(path))
large = {path for path in files if path.stat().st_size > 2*1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-boundary-lanes-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-boundary-lanes-v1/README.md').write_text('# Boundary quadrature scheduling and single-region wrench limits\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is an experimental ancestor. '
    'See [measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#boundary-quadrature-scheduling). '
    'Replay captured tasks with their pinned engine and analysis sources, Slurm rtx-mid, image1_26, '
    'the data-side Python environment and private node caches. Full arrays remain at hashed data paths. '
    'The original timed-out full-field runs and actual frame428 refusals remain retained. Short-window throughput, '
    'raw numerical differences and the independent point-force representability certificate have separate scopes.\n')
with (N / 'meta/native-boundary-lanes-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
