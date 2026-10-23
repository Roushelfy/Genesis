"""Publish continuous support and force-cone evidence for the frame428 refusal."""
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_boundary_lanes'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS == '06cf62f1c44a13465ac3373fb52eee013d8db44e'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-region-moments428-v1', 'hard-region-moments428-capture-v1',
         'hard-region-moments428-v2', 'hard-region-moments428-v3',
         'hard-region-application428-v1', 'hard-region-cone428-v1']
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
        expected = ['FAILED', '1:0'] if name in names[:3] else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected
        assert code == ('1' if expected[0] == 'FAILED' else '0')
        task_states = []
        for task in tasks:
            path = root / task['name'] / 'result.json'
            result = read(path) if path.is_file() else None
            passed = bool(result and result.get('passed', False))
            if code == '0': assert passed, (name, group, task)
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1],
                            'wrapper_return_code': code, 'tasks': task_states})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
heads = read(N / 'meta/native-boundary-lanes-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
capture_root = N / 'hard-region-moments428-capture-v1/group0/capture428'
failure = read(capture_root / 'result.json')
assert not failure['passed'] and failure['failure'] == {'batch': 4, 'mode': 'max', 'step': 428}
frame = read(capture_root / 'failure-max-b4/frame.json')
assert frame['accepted'] == [False, True, True, True] and frame['input_status'] == [32, 0, 0, 0]
moments = [read(N / f'hard-region-moments428-v3/group{group}/order{order}/result.json') for group, order in [(0, 24), (1, 48)]]
for result in moments:
    assert result['passed'] and len(result['rows']) == 1
    row = result['rows'][0]
    assert row['independent_and_native_pressure_scales_agree'] and row['reconstructed_pressure_center_matches_capture']
    assert not row['positive_pressure_centroid_inside_planar_support'] and not row['linear_pressure_tilt_nonnegative']
    assert not row['curved_proportional_cone_admitted']
maximum_change = max(abs(a-b) for a,b in zip(moments[0]['rows'][0]['curved_proportional_free_moment_nm'], moments[1]['rows'][0]['curved_proportional_free_moment_nm']))
assert maximum_change < 2e-21
application = read(N / 'hard-region-application428-v1/group0/application/result.json')
assert application['passed'] and application['raw_contact_association_checked'] and not application['ground_hertz_reconstructed']
assert len([row for row in application['rows'] if row['environment'] == 0]) == 2
cone = read(N / 'hard-region-cone428-v1/group0/certificate/result.json')
assert cone['passed'] and len(cone['rows']) == 2
for row in cone['rows']:
    assert row['bounds']['height_boundary_boxes'] == 0
    assert row['force_certificate']['continuous_force_excluded']
    assert row['force_certificate']['maximum_unit_traction_projection'] < 0
    assert row['force_certificate']['target_force_projection_n'] > 0
last = cone['rows'][-1]['force_certificate']
lower = math.sin(math.radians(last['force_angle_degrees'] - last['enlarged_cone_angle_degrees']))
bindings = []
for module, campaign in [('check_live_curved_hertz_franka', 'hard-region-moments428-capture-v1'),
                         ('inspect_hard_region_moments', 'hard-region-moments428-v3'),
                         ('inspect_contact_application', 'hard-region-application428-v1'),
                         ('check_curved_traction_certificate', 'hard-region-cone428-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope_path = N / 'meta/region-moments428-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'native_failure': failure['failure'],
    'pressure_and_curved_moment_checks': moments, 'maximum_quadrature_moment_change_nm': maximum_change,
    'application_point_audit': application, 'continuous_cone_certificate': cone,
    'minimum_relative_force_change_lower_bound': lower, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/region-moments428-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files = [SCRIPT, HELPER_SOURCE, HELPER, REPLAY_HELPER, receipt_path, scope_path]
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
    for pattern in ['*.npz', '*.obj', '*.urdf', '*.pstats']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts: add_identities(read(path))
large = {path for path in files if path.stat().st_size > 2*1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-region-moments428-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-region-moments428-v1/README.md').write_text('# Single-region pressure support and continuous traction limits\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') remains unchanged. '
    'See [findings and physical scope](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#single-region-pressure-support-and-traction-limits). '
    'Replay captured tasks with pinned engine/analysis sources, Slurm rtx-mid, image1_26 and the data-side Python environment. '
    'The original invalid-scratch readers and native refusal remain separate from the successful diagnostic controls. '
    'Full arrays stay at the hashed data paths.\n')
with (N / 'meta/region-moments428-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
