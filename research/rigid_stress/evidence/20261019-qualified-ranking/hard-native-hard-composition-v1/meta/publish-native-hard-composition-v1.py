"""Retain native hard input binding, complete-field composition and cone-boundary stability."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_edge_stability'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('8324e7d53')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [f'hard-native-hard-binding-v{i}' for i in range(1, 4)]
names += [f'hard-native-hard-recovery-v{i}' for i in range(3, 6)]
names += [f'hard-native-hard-stability-v{i}' for i in range(1, 3)]
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
bindings = []
for module, campaign in [('check_native_hard_binding', 'hard-native-hard-stability-v2'),
                         ('check_native_hard_recovery', 'hard-native-hard-recovery-v5'),
                         ('check_live_curved_hertz_franka', 'hard-native-hard-binding-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/native-edge-geometry-terminal-v1.json')['frozen_heads']
for name in ['hard_contact_edge_binding', 'hard_contact_edge_recovery', 'hard_contact_edge_stability']:
    heads[name] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip()
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
binding = read(N / 'hard-native-hard-stability-v2/group0/binding/result.json')
allocation = [read(N / f'hard-native-hard-stability-v2/group1/{backend}/result.json') for backend in ['cpu', 'gpu']]
fields = [read(N / f'hard-native-hard-recovery-v5/group{i}/{backend}/result.json') for i, backend in enumerate(['gpu', 'cpu'])]
live = read(N / 'hard-native-hard-binding-v2/group1/franka/hard0106.json')
assert binding['passed'] and binding['world_frame_invariance'] and binding['partial_reset_and_reload']
assert binding['environment_status'] == [0, 0, 0, 0]
assert all(row['passed'] and row['maximum_force_relative'] < 1e-8 for row in allocation)
assert all(row['passed'] and row['max_output']['agrees_with_full'] for row in fields)
assert max(item['volume_tensor_relative'] for row in fields for item in row['rows']) < 5e-6
assert max(item['common_probes']['tensor_relative'] for row in fields for item in row['rows']) < 1.4e-5
assert live['status'] == [0, 1, 1, 1]
scope = ('Native sharp-edge inputs now resolve material geometry, group five collision samples into three '
    'physical regions, predict pressure and update constrained region forces. CPU/GPU replay verifies the '
    'original total wrench, merged-sample invariance, unload/isolation, partial reset and restoration. '
    'Actual Franka frame106 passes this input component while the unchanged main Hertz path still refuses '
    'the frame. Rotated/translated copies exposed a cone-boundary Newton instability. Dual-objective '
    'backtracking repairs it without changing acceptance tolerances, retaining independent CPU/GPU '
    'constrained-minimum agreement within3.36e-10. Native mixed Hertz/edge complete-field composition '
    'passes CPU/GPU64, including89 material probes per source, all750 slots and four positive environments. '
    'Against the same approximate model assembled offline, volume tensor difference is4.87e-6 and common '
    'probe difference1.30e-5. Max/full sampled peaks agree. This checks numerical composition; independent '
    'physical stress errors, changing edge friction, actual main-pipeline admission and matched accepted '
    'pure-rigid/max/full throughput remain required. Edge pressure is unbounded at the continuum edge; '
    'automatic finite edge probes start at0.01 widths. All failures and executed sources are retained. Goal active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'native_binding': binding, 'native_force_allocation_controls': allocation, 'native_complete_fields': fields,
    'live_input_status': live['status'], 'live_scene_admitted': False,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False,
    'submission_preflights': {'hard-native-hard-recovery-v1': 'No scheduler submission. Sparse checkout omitted pyproject.toml.',
                             'hard-native-hard-recovery-v2': 'No scheduler submission. Old Git treated set --cone as a path. init --cone restored the required checkout before v3.'}}

receipt_path = N / 'meta/native-hard-composition-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
def add_identities(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values():
            add_identities(item)
    elif isinstance(value, list):
        for item in value:
            add_identities(item)
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    for pattern in ['*.npz', '*.obj', '*.urdf']:
        external.update(root.glob('group*/**/' + pattern))
    for path in root.glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts:
            add_identities(read(path))

external -= set(files)
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hard-composition-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-hard-composition-v1/README.md').write_text('# Native hard-contact inputs and complete-field composition\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#native-hard-inputs-and-complete-field-composition). '
    'Replay captured tasks with pinned sources, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Full arrays remain at hashed data paths. '
    'Original failures remain retained; source acceptance targets remain unchanged.\n')
with (N / 'meta/native-hard-composition-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
