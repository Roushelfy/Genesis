"""Retain curved material walks, friction binding, first-exit controls and component cost."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_curved_walk'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('3025c965b')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-curved-walk-feature-inspection-v1', 'hard-curved-walk-feature-inspection-v2', 'hard-curved-walk-controls-v1', 'hard-curved-walk-baseline-v1', 'hard-curved-walk-friction-v1', 'hard-curved-walk-friction-baseline-v1', 'hard-curved-walk-franka-v1', 'hard-curved-walk-first-exit-v1', 'hard-curved-walk-first-exit-baseline-v1', 'hard-curved-walk-timing-v1', 'hard-curved-walk-scene-comparison-v1', 'hard-curved-walk-scene-comparison-v2']
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
allowed = [ALIAS, 'ad6b71984']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('inspect_hertz_force_line', 'hard-curved-walk-feature-inspection-v2'), ('check_curved_surface_walk', 'hard-curved-walk-controls-v1'), ('check_curved_friction_binding', 'hard-curved-walk-friction-v1'), ('check_curved_walk_first_exit', 'hard-curved-walk-first-exit-v1'), ('time_curved_surface_walk', 'hard-curved-walk-timing-v1'), ('inspect_curved_walk_scene', 'hard-curved-walk-scene-comparison-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/matched-collision-terminal-v1.json')['frozen_heads']
heads['hard_contact_curved_walk'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha

controls = [read(N / f'hard-curved-walk-controls-v1/group{i}/walk/result.json') for i in [0, 1]]
assert all(result['passed'] and all(row['valid_queries'] == 56 for row in result['rows']) for result in controls)
negative = read(N / 'hard-curved-walk-baseline-v1/group0/walk/result.json')
assert negative['expected_baseline_curved_failure_observed']
friction = [read(N / f'hard-curved-walk-friction-v1/group{i}/{name}/result.json')
            for i in [0, 1] for name in ['curved-friction', 'affine-friction']]
assert all(result['passed'] for result in friction)
friction_negative = read(N / 'hard-curved-walk-friction-baseline-v1/group0/curved-friction/result.json')
assert friction_negative['expected_baseline_failure_step'] == 1
first_exit = [read(N / f'hard-curved-walk-first-exit-v1/group{i}/exit/result.json') for i in [0, 1]]
assert all(row['passed'] and row['faces'] == [-1, -1, 0] for row in first_exit)
first_exit_negative = read(N / 'hard-curved-walk-first-exit-baseline-v1/group0/exit/result.json')
assert first_exit_negative['expected_baseline_shortcut_observed']
scenes = []
for group in [0, 1]:
    result = read(N / f'hard-curved-walk-franka-v1/group{group}/franka/result.json')
    rows = result['rows']
    assert not result['passed'] and all(all(row['accepted']) for row in rows[:-1])
    assert rows[-1]['step'] == (108 if group == 0 else 106)
    row = rows[-1]
    scenes.append({'group': group, 'collision_divisions': 0 if group == 0 else 8,
        'frames': len(rows), 'fully_accepted_prefix_steps': len(rows)-1,
        'fully_accepted_prefix_loaded_environment_steps': sum(sum(r['loaded']) for r in rows[:-1]),
        'first_invalid_step': row['step'], 'accepted': row['accepted'], 'input_status': row['input_status'],
        'recovery_status': row['recovery_status'],
        'source_failures': [[i, j, status] for i, statuses in enumerate(row['source_status'])
                            for j, status in enumerate(statuses) if status != 0],
        'region_failures': [[i, j, status] for i, statuses in enumerate(row['region_status'])
                            for j, status in enumerate(statuses) if status not in [0, 1]]})
assert scenes[-1]['input_status'] == [1, 0, 0, 0] and not scenes[-1]['source_failures']
inspection = [read(N / f'hard-curved-walk-feature-inspection-v2/group0/force-line-{i}/result.json') for i in range(3)]
assert all(row['passed'] for row in inspection)
assert all(row['collision_normal_dominant_axis_is_selected_face'] for result in inspection for row in result['rows'])
timing = read(N / 'hard-curved-walk-timing-v1/group0/timing/result.json')
assert timing['passed'] and len(timing['rows']) == 6
comparisons = [read(N / f'hard-curved-walk-scene-comparison-v2/group0/comparison-{i}/result.json') for i in [0, 1]]
assert all(row['passed'] for row in comparisons)
scope = ('Native projected walks now evaluate regular quadratic material faces and their first outgoing curved edge. '
    'A positive Bernstein-Jacobian bound retains facing surfaces; adjacency controls continuity. Affine geometry keeps its '
    'prior path. CPU/CUDA independent position, rotation, reversed-normal, disconnected and nonconvex first-exit controls '
    'pass. The old source displaced curved positions by1.62mm and took an invalid endpoint shortcut. Supplied curved '
    'friction histories now survive movement, edge crossing and slot permutation, with isolated release/recontact and '
    'independent radial-law comparison; the old source loses all four environments at step1. Existing affine friction '
    'controls pass. Actual original/refined Franka still first reject at108/106; same-pose geometry shows correct dominant '
    'BOX face selection but refined normal roots outside by24.76/34.26 Hertz widths. Original sign-label inspection '
    'and corrected opponent-normal interpretation are both retained. Warmed device-resident geometry component timing '
    'does not establish full accepted stress throughput. Wrench-preserving curved force-line centers and actual local '
    'normal/curvature, finite-edge pressure/global peaks, and accepted moving max/full grasp throughput remain open. '
    'No pressure model or acceptance threshold changes, no global stress or whole-goal completion claim.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'geometry_controls': controls, 'negative_geometry_control': negative, 'friction_controls': friction,
    'negative_friction_control': friction_negative, 'first_exit_controls': first_exit,
    'negative_first_exit_control': first_exit_negative, 'actual_scene_admission': scenes,
    'scene_comparisons': comparisons, 'force_line_inspection': inspection, 'component_timing': timing,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/curved-walk-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    for path in root.glob('group*/**/*.npz'):
        external.add(path)
    for path in root.glob('group*/**/start.json'):
        start = read(path)
        for item in start.get('inputs', []) + start.get('collision_assets', []):
            external.add(Path(item['path']))
        for key in ['mesh', 'collision_urdf', 'conditions', 'input']:
            if start.get(key) is not None:
                external.add(Path(start[key]['path']))
for comparison in comparisons:
    external.update(Path(item['path']) for row in comparison['frames'] for item in [row['before'], row['after']])
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-curved-walk-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-curved-walk-v1/README.md').write_text('# Curved material walks and friction binding\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor; '
    'original sourcead6b71984 remains a frozen negative control. See '
    '[measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#curved-material-walks-and-friction-binding). '
    'Replay captured tasks with pinned engine source, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh explicit data outputs and private node caches. Preserve all FAILED states and original captures. '
    'Full arrays and generated meshes remain at hashed data paths.\n')
with (N / 'meta/curved-walk-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
