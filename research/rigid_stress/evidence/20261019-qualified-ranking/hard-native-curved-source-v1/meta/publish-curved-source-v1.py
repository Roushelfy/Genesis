"""Retain native curved source fields and actual Franka admission failures."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_curved_source'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('ce57a7fb2')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-curved-source-v1', 'hard-native-curved-source-v2', 'hard-live-curved-franka-v1',
         'hard-live-curved-franka-grasp-v1', 'hard-live-curved-franka-geometry-v1', 'hard-live-curved-franka-geometry-v2',
         'hard-curved-franka-region-v1']
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
allowed = [ALIAS, 'df9cf92cb']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('check_native_curved_source', 'hard-native-curved-source-v2'),
                         ('check_live_curved_hertz_franka', 'hard-live-curved-franka-geometry-v2'),
                         ('inspect_hertz_franka_region', 'hard-curved-franka-region-v1')]:
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
heads = read(N / 'meta/curved-material-terminal-v1.json')['frozen_heads']
heads['hard_contact_curved_source'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
native = [read(N / f'hard-native-curved-source-v2/group{i}/curved-source/result.json') for i in [0, 1]]
assert all(row['passed'] for row in native)
scenes = []
for name, group in [('hard-live-curved-franka-v1', 0), ('hard-live-curved-franka-v1', 1),
                    ('hard-live-curved-franka-grasp-v1', 0), ('hard-live-curved-franka-geometry-v2', 0)]:
    result = read(N / name / f'group{group}/franka/result.json')
    rows = result['rows']
    first = next((row for row in rows if not all(row['accepted'])), None)
    prefix = rows if first is None else rows[:first['step']]
    scenes.append({'campaign': name, 'group': group, 'passed': result['passed'], 'captured_frames': len(rows),
        'accepted_loaded_environment_steps': sum(sum(a and l for a, l in zip(row['accepted'], row['loaded'])) for row in rows),
        'fully_accepted_prefix_steps': len(prefix), 'fully_accepted_prefix_loaded_environment_steps': sum(sum(row['loaded']) for row in prefix),
        'first_invalid': None if first is None else {key: first[key] for key in ['step', 'loaded', 'accepted', 'input_status', 'recovery_status']}})
assert all(row['passed'] and row['accepted_loaded_environment_steps'] == 111 for row in scenes[:2])
assert all(not row['passed'] and row['first_invalid']['step'] == 108 for row in scenes[2:])
diagnostic = read(N / 'hard-live-curved-franka-geometry-v2/group0/franka/geometry.json')
assert diagnostic['step'] == 108
region = read(N / 'hard-curved-franka-region-v1/group0/region/result.json')
assert region['passed']
scope = ('Native quadratic-surface source traction and material-following Hertz probes pass independent CPU/CUDA '
    'same-prescribed-pressure load, corner, probe, sampled peak, max/full, release and rejection-isolation controls. '
    'Affine shell and actual planar friction controls remain accepted. Three curved radial subdivisions repair the '
    'original quadrature failure without changing the declared load gate. Actual B4 full-output Franka ground/approach '
    'accepts30steps for both geometry assets,111 loaded environment-steps each. The extended estimated-geometry '
    'grasp first invalidates environment0 at step108 with region geometry status2 and raises the production exception '
    'at110. Its108-step fully accepted prefix contains423 loaded environment-steps. Initial-pair geometry diagnosis '
    'and its original reader failure are preserved. Component B3 normal costs exclude collision inputs/friction. '
    'Physical pressure/discretization/global-maximum qualification, moving gripper geometry/source/history admission '
    'and matched accepted Franka max/full throughput remain open.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'native_source_checks': native, 'actual_scene_admission': scenes, 'initial_pair_diagnostic': diagnostic, 'region_inspection': region,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/curved-source-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = {N / 'hard-native-quadric-franka-v1/group0/franka/frame0010.npz',
            N / 'hard-quadratic-rest-surface-v1/group0/geometry/geometry.npz',
            N / 'hard-curved-hertz-residual-v3/group0/residual/fields.npz'}
for kind in ['analytic', 'estimated']:
    external.add(N / f'hard-native-curved-material-v3/group1/curved-material/{kind}.npz')
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    external.update(root.glob('group*/**/*.obj'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-curved-source-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-curved-source-v1/README.md').write_text('# Native curved sources and actual Franka admission\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#native-curved-hertz-sources-and-moving-material-probes). '
    'Replay captured task groups with the pinned engine, Slurm rtx-mid, image1_26 and the existing data-side '
    'environment. Use fresh explicit data outputs and private node caches. Native v2 passes its declared gates. '
    'The grasp and geometry campaigns intentionally retain actual rejection; their FAILED scheduler states are '
    'part of the evidence. Decode gzip separately and leave packet bytes intact. Full fields, scene frames and '
    'generated meshes remain at hashed data paths.\n')
with (N / 'meta/curved-source-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
