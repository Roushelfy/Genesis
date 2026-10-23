"""Publish contact-frame and elastic-displacement evidence at the sharp-edge transition."""
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
names = ['edge-contact-frame428-v1', 'edge-shell-transition-capture-v1',
         'edge-shell-transition-tilt-v1', 'edge-self-displacement428-v1', 'edge-self-displacement428-v2']
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
        expected = ['FAILED', '1:0'] if name == 'edge-shell-transition-capture-v1' else ['COMPLETED', '0:0']
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
heads = read(N / 'meta/region-moments428-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
capture_root = N / 'edge-shell-transition-capture-v1/group0/capture428'
failure = read(capture_root / 'result.json')
assert not failure['passed'] and failure['failure'] == {'batch': 4, 'mode': 'max', 'step': 428}
for step in [419, 420, 427]:
    frame = read(capture_root / f'capture-max-b4-step{step}/frame.json')
    assert all(frame['accepted']) and frame['input_status'] == [0, 0, 0, 0]
frames = read(N / 'edge-contact-frame428-v1/group0/frames/result.json')
assert frames['passed'] and len(frames['rows']) == 1
row = frames['rows'][0]
assert row['frame_change_preserves_force_and_moment']
assert row['opposing_face_shear_normal_ratio'] > row['material_shear_normal_ratio'] > row['friction']
tilts, summaries = [], []
for group, step in enumerate([419, 420, 427]):
    result = read(N / f'edge-shell-transition-tilt-v1/group{group}/step{step}/result.json')
    assert result['passed'] and result['environment'] == 0
    point = {r['region']: r for r in result['rows'] if r['particular_nodal_model'] == 'point'}
    moments = {r['region']: r for r in result['rows'] if r['particular_nodal_model'] == 'four_moments'}
    assert point.keys() == moments.keys() and len(point) == 4
    assert all(r['mass_rigid_projection_relative_residual'] < 1e-10 for r in result['rows'])
    summaries.append({'step': step,
        'maximum_rotation_degrees': max(r['first_order_normal_rotation_degrees'] for r in moments.values()),
        'maximum_along_shift_m': max(abs(r['smooth_along_center_shift_m']) for r in moments.values()),
        'maximum_normal_displacement_m': max(abs(r['smooth_displacement_local_m'][2]) for r in moments.values()),
        'maximum_point_moment_rotation_change_degrees': max(abs(point[i]['first_order_normal_rotation_degrees'] - r['first_order_normal_rotation_degrees']) for i, r in moments.items()),
        'maximum_point_moment_displacement_change_m': max(abs(x-y) for i, r in moments.items() for x,y in zip(point[i]['smooth_displacement_local_m'], r['smooth_displacement_local_m']))})
    tilts.append(result)
displacements = [read(N / f'edge-self-displacement428-v{version}/group0/displacement/result.json') for version in [1, 2]]
assert all(r['passed'] for r in displacements)
rows = displacements[1]['rows']
change = math.sqrt(sum((a-b)**2 for a,b in zip(rows[0]['displacement_local_m'], rows[1]['displacement_local_m'])))
relative_change = change / rows[1]['displacement_norm_m']
assert relative_change < 1e-6
bindings = []
for module, campaign in [('check_live_curved_hertz_franka', 'edge-shell-transition-capture-v1'),
                         ('time_hard_franka', 'edge-shell-transition-capture-v1'),
                         ('inspect_edge_contact_frame', 'edge-contact-frame428-v1'),
                         ('inspect_live_shell_tilt', 'edge-shell-transition-tilt-v1'),
                         ('inspect_edge_contact_displacement', 'edge-self-displacement428-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope_path = N / 'meta/edge-shell-transition-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'native_failure': failure['failure'],
    'contact_frame_check': frames, 'smooth_shell_corrections': tilts, 'smooth_shell_summary': summaries,
    'self_displacements': displacements, 'self_displacement_refinement_relative_change': relative_change,
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/edge-shell-transition-terminal-v1.json'
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
packet = freeze('edge-shell-transition-v1', files, sorted(external-set(files)), scope)
(DEST / 'edge-shell-transition-v1/README.md').write_text('# Sharp-edge contact frames and shell displacement\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') remains unchanged. '
    'See [measurements and physical scope](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#contact-frames-and-shell-displacement). '
    'Replay captured tasks with pinned engine/analysis sources, Slurm rtx-mid, image1_26 and the data-side Python environment. '
    'The native frame428 refusal remains separate from the admitted frame419/420/427 displacement checks. '
    'Full arrays stay at the hashed data paths.\n')
with (N / 'meta/edge-shell-transition-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
