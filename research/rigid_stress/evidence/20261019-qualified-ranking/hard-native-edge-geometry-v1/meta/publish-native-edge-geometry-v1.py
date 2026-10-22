"""Retain continuous sharp-edge fields, native evaluation and curved-shell source response."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_edge_geometry'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('6c2e887be')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-edge-geometry-v1', 'hard-native-force-allocation-v1', 'hard-native-force-allocation-v2']
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
allowed = [ALIAS, 'f572459433968382b40ef8d518b695f7202f0784']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('check_native_edge_geometry', 'hard-native-edge-geometry-v1'),
                         ('check_native_force_allocation', 'hard-native-force-allocation-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/contact-application-terminal-v1.json')['frozen_heads']
heads['hard_contact_edge_geometry'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
geometry = [read(N / f'hard-native-edge-geometry-v1/group0/{backend}/result.json') for backend in ['cpu', 'gpu']]
allocation = [read(N / f'hard-native-force-allocation-v1/group0/{backend}/result.json') for backend in ['cpu', 'gpu']]
timing = read(N / 'hard-native-force-allocation-v2/group0/gpu/result.json')
assert all(row['passed'] for row in [*geometry, *allocation, timing])
assert max(row['maximum_force_relative'] for row in allocation) < 1e-8
assert timing['maximum_independently_reconstructed_wrench_residual'] < 1e-10
assert [row['environments'] for row in timing['component_timing']] == [1, 64, 1024]
scope = ('Native Quadrants box-edge geometry follows connected actual quadratic material faces and uses '
    'analytic height derivatives. CPU/GPU64 controls reproduce independently minimized captured edge '
    'positions within0.206nanometres and central derivatives within6.85e-7 relative, including rigid '
    'rotation/translation, duplicate samples, invalid/interior/corner isolation and restoration. '
    'A native six-coordinate dual Newton force allocator projects each region onto its Coulomb cone '
    'and preserves the supplied original total wrench, including free couples. Twelve CPU/GPU64 '
    'scaled/rotated controls match independent constrained minima within3.36e-10 relative after3-4 updates. '
    'Independent wrench reconstruction is within2.47e-13; original6.22% load correction remains visible. '
    'Invalid/singular/unloaded isolation and restoration pass. Homogeneous B1/64/1024 three-region '
    'GPU force allocation measures0.1983/0.1988/0.2173ms inclusive of dispatch and synchronization. '
    'This excludes geometry, grouping, support updates, full recovery and scenes. These native components '
    'are still experimental supplied-input controls; actual sharp-edge scene binding, changing friction '
    'history, independent complete physical accuracy and admitted matched max/full scene throughput '
    'remain required. Sharp-edge continuum peaks remain unbounded. Goal active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'native_geometry_controls': geometry, 'native_force_allocation_controls': allocation,
    'allocation_component_timing': timing, 'preserved_files': preserved, 'frozen_heads': heads,
    'scope': scope, 'whole_goal_complete': False}

receipt_path = N / 'meta/native-edge-geometry-terminal-v1.json'
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
packet = freeze('hard-native-edge-geometry-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-edge-geometry-v1/README.md').write_text('# Native sharp-edge geometry and constrained region forces\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#native-edge-geometry-and-force-allocation). '
    'Replay captured tasks with pinned sources, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Full arrays remain at hashed data paths. '
    'Original failures remain retained; source acceptance targets remain unchanged.\n')
with (N / 'meta/native-edge-geometry-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
