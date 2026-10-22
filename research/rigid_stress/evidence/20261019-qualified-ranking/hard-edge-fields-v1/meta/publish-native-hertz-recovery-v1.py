"""Freeze native normal recovery, far-field cost and actual failed controls."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_farfield'
ALIAS = '67c9de0b2ab02ca4adaa47773a0b6cd17692eebd'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()

names = [f'hard-native-hertz-recovery-v{i}' for i in range(1, 6)]
names += [f'hard-hertz-surface-depth-v{i}' for i in range(1, 3)]
names += ['hard-native-hertz-rotation-inspection-v1', 'hard-native-hertz-recovery-timing-v1']
names += [f'hard-native-hertz-farfield-controls-v{i}' for i in range(3, 6)]
names += [f'hard-native-hertz-farfield-paired-v{i}' for i in range(3, 6)]
names += ['hard-hertz-farfield-kernel-v3']
names += [f'hard-native-hertz-final-inspection-v{i}' for i in range(1, 3)]
names += ['hard-native-hertz-component-regression-v1']
names += [f'hard-native-hertz-refusal-v{i}' for i in range(1, 5)]
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
successful_tasks = 0
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        code = (root / 'returncode.txt').read_text().strip()
        expected = ['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0']
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected, (name, group, code, observed)
        task_states = []
        for task in tasks:
            result_path = root / task['name'] / 'result.json'
            result = read(result_path) if result_path.is_file() else None
            passed = bool(result and result.get('passed', result.get('execution_completed', False)))
            if code == '0':
                assert passed, (name, group, task, result)
            successful_tasks += passed
            task_states.append({'name': task['name'], 'result_exists': result is not None, 'passed': passed})
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1], 'tasks': task_states})

# Reconstruct every captured source as public ancestor plus its original patch.
# A private temporary index leaves all source worktrees untouched.
blob_hashes = {}
source_trees = {}
replayed = []
batch = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=C3, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    with tempfile.TemporaryDirectory(prefix='hertz-source-replay-', dir=N / 'meta') as temporary:
        for name in names:
            root = N / name
            source = read(root / 'source.json')
            patch = root / 'source.patch'
            assert hashlib.sha256(patch.read_bytes()).hexdigest() == source['patch_sha256']
            key = (source['revision'], source['patch_sha256'])
            if key not in source_trees:
                env = os.environ.copy()
                env['GIT_INDEX_FILE'] = str(Path(temporary) / 'index')
                subprocess.run(['git', 'read-tree', source['revision']], cwd=C3, env=env, check=True)
                if patch.stat().st_size:
                    subprocess.run(['git', 'apply', '--cached', '--whitespace=nowarn', str(patch)], cwd=C3, env=env, check=True)
                tree = subprocess.check_output(['git', 'write-tree'], cwd=C3, env=env, text=True).strip()
                entries = {}
                for line in subprocess.check_output(['git', 'ls-tree', '-rz', tree], cwd=C3).split(b'\0'):
                    if line:
                        meta, relative = line.split(b'\t', 1)
                        entries[relative.decode()] = meta.decode().split()
                source_trees[key] = (tree, entries)
            tree, entries = source_trees[key]
            for row in source['files']:
                mode, kind, blob = entries[row['path']]
                assert kind == 'blob'
                if blob not in blob_hashes:
                    batch.stdin.write((blob + '\n').encode())
                    batch.stdin.flush()
                    header = batch.stdout.readline().decode().strip().split()
                    assert header[:2] == [blob, 'blob']
                    content = batch.stdout.read(int(header[2]))
                    assert batch.stdout.read(1) == b'\n'
                    blob_hashes[blob] = (len(content), hashlib.sha256(content).hexdigest())
                assert blob_hashes[blob] == (row['bytes'], row['sha256']), (name, row['path'])
            for relative, commit in source['gitlinks'].items():
                assert entries[relative] == ['160000', 'commit', commit]
            replayed.append({'campaign': name, 'recorded_revision': source['revision'],
                'recorded_patch_sha256': source['patch_sha256'], 'reconstructed_tree': tree,
                'all_captured_source_files_exact': len(source['files'])})
finally:
    batch.stdin.close()
    batch.wait()

alias_tree = subprocess.check_output(['git', 'rev-parse', ALIAS + '^{tree}'], cwd=C3, text=True).strip()
for name in ['hard-native-hertz-farfield-controls-v5', 'hard-native-hertz-farfield-paired-v5',
             'hard-native-hertz-final-inspection-v2', 'hard-native-hertz-refusal-v4']:
    assert next(row for row in replayed if row['campaign'] == name)['reconstructed_tree'] == alias_tree

owned = {
    'hertz_stress': 'hard-native-hertz-farfield-controls-v5',
    'check_native_hertz_recovery': 'hard-native-hertz-farfield-controls-v5',
    'check_hertz_surface_depth': 'hard-hertz-surface-depth-v2',
    'inspect_hertz_rotation_failure': 'hard-native-hertz-rotation-inspection-v1',
    'time_native_hertz_recovery': 'hard-native-hertz-farfield-paired-v5',
    'check_hertz_farfield': 'hard-hertz-farfield-kernel-v3',
    'inspect_native_hertz_recovery': 'hard-native-hertz-final-inspection-v2',
    'check_hertz_recovery_refusal': 'hard-native-hertz-refusal-v4',
}
bindings = []
for module, name in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / name / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})

inspection = read(N / 'hard-native-hertz-final-inspection-v2/group0/inspection/result.json')
assert inspection['passed'] and not inspection['general_pressure_friction_or_scene_qualified']
assert all(row['repeat_ranges_separate'] for row in inspection['same_allocation_timings'] if row['batch'] >= 64)
assert all(row['admitted_all_audit_frames'] == 24 * row['batch'] for row in inspection['same_allocation_timings'])
maxima = {key: max(row[key] for case in inspection['independent_prescribed_load_checks'] for row in case['rows'])
    for key in ['higher_rule_complete_field_relative', 'higher_rule_volume_relative',
                'higher_rule_vm_peak_relative', 'higher_rule_tensile_peak_relative']}
for group in [0, 1]:
    low = read(N / f'hard-native-hertz-refusal-v4/group{group}/low/result.json')
    check = read(N / f'hard-native-hertz-refusal-v4/group{group}/check/result.json')
    assert low['passed'] and low['controlled_low_rule_refusal'] and low['status'] == [4, 3, 3]
    assert check['passed'] and check['initial_status'] == [0, 3, 3]
    assert check['unloaded_centrifugal_fields_preserved'] and check['source_identity_preserved_on_refusal']

def add_input_ids(value, external):
    if isinstance(value, dict):
        if {'path', 'sha256', 'bytes'} <= value.keys() and Path(value['path']).is_absolute():
            path = Path(value['path'])
            assert identity(path) == {key: value[key] for key in ['path', 'bytes', 'sha256']}
            external.add(path)
        for item in value.values():
            add_input_ids(item, external)
    elif isinstance(value, list):
        for item in value:
            add_input_ids(item, external)

files = [SCRIPT, HELPER, HELPER_SOURCE, N / 'meta/hertz-stress-confocal-before-tight-bracket-v1.py',
    N / 'meta/hertz-farfield-prelaunch-failure-v1.json', N / 'meta/hertz-farfield-sparse-prelaunch-v1.json']
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    external.update((N / name).glob('group*/**/*.npz'))
    for path in (N / name).glob('group*/**/*.json'):
        if 'cache' not in path.parts and 'tmp' not in path.parts:
            add_input_ids(read(path), external)

prelaunch = ['hard-native-hertz-farfield-controls-v1', 'hard-native-hertz-farfield-paired-v1',
             'hard-hertz-farfield-kernel-v1', 'hard-native-hertz-farfield-controls-v2']
for name in prelaunch:
    root = N / name
    assert not (root / 'submission.json').exists()
    files += [path for path in root.glob('*.json') if path.is_file()]
    for module in ['check_native_hertz_recovery', 'time_native_hertz_recovery', 'check_hertz_farfield']:
        path = root / f'analysis-source/research/rigid_stress/qipc_comparison/{module}.py'
        if path.is_file():
            files.append(path)

scope = ('Normal-only experimental Quadrants Hertz plus shared finite-domain regular recovery. '
    'Final CPU/GPU exact/far/rotated complete fields, sampled VM/principal peaks, max/full, refusal, '
    'release, moving supplied centers, step invalidation and centrifugal controls pass. '
    'An independent same-coarse-mesh prescribed-Hertz P2 solve gives maximum0.01974% volume tensor '
    'and0.001842%/0.000296% tensile/VM peak changes; this is not independent pressure or continuum accuracy. '
    'Tiny positive-depth confocal brackets are repaired without gate relaxation; original root, '
    'point-side, point-location, NumPy, zero-query and insufficient-quadrature failures are retained. '
    'Four moment-matched far sources have1.44e-6p0 independent pressure-integral error. '
    'Same-allocation B1024 component max/full time12.164/12.214 to7.389/7.429ms falls39.3%/39.2%, '
    'with all24576 audited frames and17014 loaded frames admitted. Low-batch repeat overlap remains visible. '
    'No automatic moving probe, live scene, general curved/inner/nonconvex or vector-friction qualification follows.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'submitted_campaigns': len(names), 'successful_task_results': successful_tasks,
    'prelaunch_directories_without_submission': prelaunch, 'captured_source_replay': replayed,
    'current_worker_bindings': bindings, 'public_engine_alias': ALIAS, 'public_engine_tree': alias_tree,
    'initial_exact_engine_alias': '77ec4aab4c7d8f74c3ebbf71fce0a35955baf8b6',
    'independent_prescribed_load_maxima': maxima, 'same_allocation_timings': inspection['same_allocation_timings'],
    'runtime_data_under_data_and_private_compiler_caches_node_local': True,
    'public_stress_options_connected_to_component': False, 'general_pressure_friction_or_scene_qualified': False,
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/hard-native-hertz-recovery-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files.append(receipt_path)
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hertz-recovery-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-hertz-recovery-v1/README.md').write_text('# Native normal recovery and far-field controls\n\n' + scope +
    '\n\nPublic experimental engine source: [`67c9de0b2`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + '). '
    'Checkout this exact source; task HEAD stress options are not connected to the component. '
    'The receipt reconstructs every historical engine snapshot as its public ancestor plus captured patch and verifies every source file. '
    'Captured analysis dependencies and exact task arguments are retained. Decode gzip files into a separate replay directory. '
    'Use the existing data-side scientific Python, Slurm `rtx-mid`, image `1_26`, an explicit fresh data output and private node-local compiler caches. '
    'Full fields, independent references and every-frame timing audits remain at externally hashed data paths. '
    'The report is [NATIVE_HERTZ_RECOVERY_20261022.md](../../../docs/NATIVE_HERTZ_RECOVERY_20261022.md). '
    'The complete cheap estimator and matching scene accuracy/throughput goal remains active.\n')
with (N / 'meta/native-hertz-recovery-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
