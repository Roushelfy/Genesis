"""Publish corrected opposing-plane geometry and adjacent-face wrench certificates."""
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
names = ['edge-contact-frame428-v2', 'box-edge-wrench428-v1', 'box-edge-wrench428-v2', 'box-edge-wrench428-v3', 'box-edge-wrench428-v4']
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
        expected = ['FAILED', '1:0'] if name in ['box-edge-wrench428-v1', 'box-edge-wrench428-v3'] else ['COMPLETED', '0:0']
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
heads = read(N / 'meta/edge-shell-transition-terminal-v1.json')['frozen_heads']
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
old_plane = read(N / 'edge-contact-frame428-v1/group0/frames/result.json')
plane = read(N / 'edge-contact-frame428-v2/group0/frames/result.json')
assert plane['passed'] and len(plane['rows']) == 1
row, old = plane['rows'][0], old_plane['rows'][0]
for key in ['opposing_face_force_n', 'material_shear_normal_ratio', 'opposing_face_shear_normal_ratio', 'friction']:
    assert row[key] == old[key]
assert old['required_centroid_inside_opposing_face'] and not row['required_centroid_inside_opposing_face']
assert row['frame_change_preserves_force_and_moment']
assert max(abs(p[axis]) for p in row['opponent_witnesses_from_opposing_edge_m'] for axis in [0, 2]) < 1e-12
cone_results = [read(N / f'box-edge-wrench428-v{version}/group0/cones/result.json') for version in [2, 4]]
for result in cone_results:
    assert result['passed'] and len(result['rows']) == 16
    assert result['force_only']['classification'] == 'feasible'
    for row in result['rows']:
        check = row['certificate']
        assert check['classification'] in ['excluded', 'feasible']
        if check['classification'] == 'excluded':
            assert check['minimum_continuous_cone_projection'] > 1e-12
            assert check['target_normalized_projection'] < -1e-8
        else:
            assert check['normalized_wrench_residual'] < 1e-8
            assert check['minimum_circular_cone_margin_n'] > -1e-9
assert [r['certificate']['classification'] for r in cone_results[0]['rows']] == [r['certificate']['classification'] for r in cone_results[1]['rows']]
assert cone_results[1]['one_face']['classification'] == 'excluded'
assert cone_results[1]['original_minimum_side_depth_bound_m'] > 11e-6
assert cone_results[1]['witness_minimum_half_length_at_width_depth_m'] > 77e-6
assert all(r['certificate']['classification'] == 'excluded' for r in cone_results[1]['rows'][:8])
bindings = []
for module, campaign in [('inspect_edge_contact_frame', 'edge-contact-frame428-v2'),
                         ('check_box_edge_wrench', 'box-edge-wrench428-v4')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope_path = N / 'meta/box-edge-wrench428-scope-v1.txt'
scope = scope_path.read_text().strip()
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'frozen_heads': heads,
    'preserved_files': preserved, 'source_bindings': bindings, 'original_plane_reader': old_plane,
    'corrected_plane_reader': plane, 'cone_refinements': cone_results,
    'same_64_and_128_facet_classifications': True, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/box-edge-wrench428-terminal-v1.json'
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
packet = freeze('box-edge-wrench428-v1', files, sorted(external-set(files)), scope)
(DEST / 'box-edge-wrench428-v1/README.md').write_text('# Corrected opposing plane and adjacent-face wrench limits\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') remains unchanged. '
    'See [geometry correction and reaction bounds](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#adjacent-face-wrench-limits). '
    'Replay captured tasks with pinned engine/analysis sources, Slurm rtx-mid, image1_26 and the data-side Python environment. '
    'The prior plane reader omitted region_gap. Its result remains preserved, while the current geometry is independently checked against raw opponent witnesses. '
    'Original dtype and HiGHS failures remain separate from successful physical-cone certificates. Full arrays stay at the hashed data paths.\n')
with (N / 'meta/box-edge-wrench428-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
