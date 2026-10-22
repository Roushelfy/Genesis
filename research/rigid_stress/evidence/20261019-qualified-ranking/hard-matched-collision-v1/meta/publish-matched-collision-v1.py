"""Retain matched collision geometry and native curved source binding."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_matched_collision'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('ad6b71984')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-matched-collision-assets-v1', 'hard-matched-collision-franka-v1',
         'hard-curved-source-geometry-v1', 'hard-curved-source-geometry-v2',
         'hard-matched-collision-inspection-v1', 'hard-matched-collision-franka-v2',
         'hard-matched-collision-baseline-v1', 'hard-curved-source-geometry-v3', 'hard-curved-source-geometry-v4']
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
allowed = [ALIAS, '42993eb5f']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('prepare_curved_collision', 'hard-matched-collision-assets-v1'),
                         ('check_live_curved_hertz_franka', 'hard-matched-collision-franka-v2'),
                         ('check_curved_source_geometry', 'hard-curved-source-geometry-v4'),
                         ('inspect_hertz_boundary_capture', 'hard-matched-collision-baseline-v1'),
                         ('inspect_hertz_force_line', 'hard-matched-collision-inspection-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/finite-edge-terminal-v1.json')['frozen_heads']
heads['hard_contact_matched_collision'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
assets = read(N / 'hard-matched-collision-assets-v1/group0/assets/result.json')
assert assets['passed'] and [row['triangles'] for row in assets['rows']] == [80, 320, 1280, 5120]
checks = [read(N / f'hard-curved-source-geometry-v4/group{i}/sources/result.json') for i in [0, 1]]
assert all(row['passed'] for row in checks)
assert all(row['passed'] for result in checks for row in result['rows'])
assert all(row['accepted_queries'] == 48 and row['rejected_queries'] == 24
           for result in checks for row in result['rows'] if row['kind'].endswith('-offsets'))
regression = read(N / 'hard-curved-source-geometry-v2/group2/regression/result.json')
assert regression['passed']
baseline = read(N / 'hard-matched-collision-baseline-v1/group0/baseline/result.json')
assert baseline['passed'] and len(baseline['frames']) == 109
scenes = []
for version in [1, 2]:
    for group, divisions in enumerate([0, 2, 8]):
        result = read(N / f'hard-matched-collision-franka-v{version}/group{group}/franka/result.json')
        rows = result['rows']
        assert not result['passed'] and all(all(row['accepted']) for row in rows[:-1])
        assert rows[-1]['step'] == (108 if group == 0 else 106)
        row = rows[-1]
        scenes.append({'version': version, 'group': group, 'collision_divisions': divisions,
            'frames': len(rows), 'fully_accepted_prefix_steps': len(rows)-1,
            'fully_accepted_prefix_loaded_environment_steps': sum(sum(r['loaded']) for r in rows[:-1]),
            'first_invalid_step': row['step'], 'accepted': row['accepted'],
            'input_status': row['input_status'], 'recovery_status': row['recovery_status'],
            'source_failures': [[i, j, status] for i, statuses in enumerate(row['source_status'])
                                for j, status in enumerate(statuses) if status != 0],
            'region_failures': [[i, j, status] for i, statuses in enumerate(row['region_status'])
                                for j, status in enumerate(statuses) if status not in [0, 1]]})
assert scenes[-1]['input_status'] == [1, 0, 0, 0] and not scenes[-1]['source_failures']
inspection = [read(N / f'hard-matched-collision-inspection-v1/group0/force-line-{i}/result.json') for i in range(3)]
assert all(row['passed'] for row in inspection)
scope = ('Matched collision assets tessellate the actual quadratic outer material boundary at1/2/4/8 divisions '
    'with preserved URDF inertia and controller conditions. Sampled position errors decrease quadratically '
    'to31.36um at8divisions. Original optional-asset baseline retains exact rigid inputs and acceptance over109 '
    'frames; constitutive differences pass explicit numerical budgets after an original exact-direction failure. '
    'Native source binding now evaluates curved positions, closest material patches and shared quadratic edges. '
    'CPU/CUDA corner/midpoint/interior/shared-edge and48 accepted/24 refused offset controls per material pass, '
    'as do original shared-edge, U-shape and cavity uncertainty regressions. Original native curved-source refusal '
    'and the expanded worker variable-reuse error remain retained. Actual8-division Franka clears source binding '
    'refusal at first gripper contact but still rejects two finite-face normal roots at106. Original/coarser '
    'collision witnesses can exceed the unchanged100um material-projection limit. Finite-face load/support, '
    'curved moving-source/friction binding, pressure/global-peak accuracy and accepted full grasp throughput '
    'remain open. No production tolerance is relaxed and no speed or whole-goal completion is claimed.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'collision_assets': assets, 'native_source_checks': checks, 'affine_nonconvex_regression': regression,
    'original_asset_baseline': baseline, 'actual_scene_admission': scenes, 'force_line_inspection': inspection,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/matched-collision-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    files.extend(root.glob('group*/**/*.xml'))
    for suffix in ['npz', 'obj', 'urdf']:
        external.update(root.glob(f'group*/**/*.{suffix}'))
    for path in root.glob('group*/**/start.json'):
        start = read(path)
        for item in start.get('inputs', []) + start.get('collision_assets', []):
            external.add(Path(item['path']))
        for key in ['mesh', 'collision_urdf', 'conditions']:
            if start.get(key) is not None:
                external.add(Path(start[key]['path']))
external.update(Path(item['path']) for row in baseline['frames'] for item in [row['before'], row['after']])
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-matched-collision-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-matched-collision-v1/README.md').write_text('# Matched collision geometry and curved source binding\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor; '
    'the original optional-asset source42993eb5f remains reproducible. See '
    '[measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#matched-collision-geometry-and-curved-source-binding). '
    'Replay captured tasks with pinned engine source, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh explicit data outputs and private node caches. Preserve all FAILED states and original captures. '
    'Decode gzip separately. Full arrays, collision assets and scene frames stay at hashed data paths.\n')
with (N / 'meta/matched-collision-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
