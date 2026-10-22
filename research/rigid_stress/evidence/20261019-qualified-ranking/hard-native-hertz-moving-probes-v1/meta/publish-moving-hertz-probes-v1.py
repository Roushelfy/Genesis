"""Freeze moving material probes and their actual strict/engineering results."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
PARENT = N / 'meta/publish-native-hertz-recovery-v1.py'
exec(compile(PARENT.read_text().split('\nC3 = ', 1)[0], str(PARENT), 'exec'))
C3 = WORK.parent / 'hard_contact_moving_probes'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS != '67c9de0b2ab02ca4adaa47773a0b6cd17692eebd'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [f'hard-native-hertz-moving-probes-v{i}' for i in range(1, 6)]
names += ['hard-native-hertz-moving-roundoff-v1']
names += [f'hard-native-hertz-moving-paired-v{i}' for i in range(1, 3)]
names += [f'hard-native-hertz-moving-inspection-v{i}' for i in range(1, 3)]
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

# Reuse the published exact ancestor-plus-patch replay, including every source
# file hash. Test files are separately bound below because the runner's engine
# inventory intentionally does not include the tests directory.
replay_code = PARENT.read_text().split('# Reconstruct every captured source', 1)[1].split('\nalias_tree = ', 1)[0]
exec(compile('# Reconstruct every captured source' + replay_code, str(PARENT), 'exec'))
alias_genesis = subprocess.check_output(['git', 'rev-parse', ALIAS + ':genesis'], cwd=C3, text=True).strip()
for row in replayed:
    if row['campaign'] in ['hard-native-hertz-moving-probes-v3', 'hard-native-hertz-moving-probes-v4',
        'hard-native-hertz-moving-probes-v5', 'hard-native-hertz-moving-paired-v1',
        'hard-native-hertz-moving-paired-v2', 'hard-native-hertz-moving-inspection-v1',
        'hard-native-hertz-moving-inspection-v2']:
        tree = subprocess.check_output(['git', 'rev-parse', row['reconstructed_tree'] + ':genesis'], cwd=C3, text=True).strip()
        assert tree == alias_genesis

inspection = read(N / 'hard-native-hertz-moving-inspection-v2/group0/inspection/result.json')
assert inspection['preexisting_goal_engineering_targets_met']
assert not inspection['automatic_collision_input_or_scene_qualified']
assert not inspection['unknown_pressure_or_general_friction_qualified']
assert inspection['strict_reference_targets'] == {'field': 1e-3, 'peak': 1e-2}
assert inspection['preexisting_goal_engineering_targets'] == {'tensor': .05, 'peak': .10}
assert len(inspection['same_allocation_timings']) == 8
assert all(row['admitted_total_frames'] == 24 * row['batch'] for row in inspection['same_allocation_timings'])
for group in [0, 1]:
    for name in ['hard-native-hertz-moving-probes-v3', 'hard-native-hertz-moving-probes-v5']:
        for case in ['moving', 'rotated']:
            result = read(N / name / f'group{group}/{case}/result.json')
            assert result['passed'] and result['query_failure_and_step_invalidation_passed']
            assert result['disconnected_material_location']['passed']
    result = read(N / f'hard-native-hertz-moving-probes-v3/group{group}/topology/result.json')
    assert result['passed'] and result['returncode'] == 0
    assert result['test_source'] == identity(C3 / 'tests/rigid/test_stress_nonconvex.py')
    assert b'tests="4"' in (N / f'hard-native-hertz-moving-probes-v3/group{group}/topology/tests.xml').read_bytes()

owned = {
    'check_moving_hertz_probes': 'hard-native-hertz-moving-probes-v5',
    'time_native_hertz_recovery': 'hard-native-hertz-moving-paired-v2',
    'check_hertz_probe_topology': 'hard-native-hertz-moving-probes-v3',
    'inspect_moving_probe_roundoff': 'hard-native-hertz-moving-roundoff-v1',
    'inspect_moving_hertz_probes': 'hard-native-hertz-moving-inspection-v2',
}
bindings = []
for module, name in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / name / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
test_copy = N / 'meta/moving-probes-test_stress_nonconvex-v1.py'
if test_copy.exists():
    assert test_copy.read_bytes() == (C3 / 'tests/rigid/test_stress_nonconvex.py').read_bytes()
else:
    test_copy.write_bytes((C3 / 'tests/rigid/test_stress_nonconvex.py').read_bytes())

scope = ('Optional source-relative89-point native normal probe bank follows moving centers, axes and frames. '
    'All runtime generation, material-face walking, regular/Hertz field evaluation and combined max/full reductions are Quadrants. '
    'Actual anchor ownership and vertex-identity tetrahedral neighbors prevent crossing disconnected material; no global relocalization is used. '
    'CPU/GPU changing-source, release, sampled maxima, geometry/field rotation, same-step failure and disconnected-location controls pass. '
    'Four focused topology tests pass on each backend; the original fixed-query numerical control is preserved. '
    'Surface self-fields retain exact parametric coordinates after an observed1e-18m world-roundoff failure. '
    'Final q6 controls pass; q4 passes the measured trajectory and unrotated controls but retains a rotated1.048e-4 wrench refusal against the unchanged1e-4 gate. '
    'Independent prescribed-load same-coarse-mesh probes/corners and saved sampled peaks meet the preexisting5%/10% engineering budgets. '
    'The separate stricter0.1%-field/1%-peak inspection target and its actual failed states remain explicit in the receipt. '
    'Actual benchmark input motion, per-frame validity and exact/far/q6/q4 costs are retained on the same allocation. '
    'These are normal component controls and costs, not automatic live contacts, continuum convergence, unknown pressure or general friction qualification.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings,
    'public_engine_alias': ALIAS, 'public_engine_genesis_tree': alias_genesis,
    'test_source': identity(test_copy), 'test_results_bound_to_public_source': True,
    'strict_reference_targets': inspection['strict_reference_targets'],
    'strict_reference_targets_met': inspection['strict_reference_targets_met'],
    'preexisting_goal_engineering_targets': inspection['preexisting_goal_engineering_targets'],
    'preexisting_goal_engineering_targets_met': inspection['preexisting_goal_engineering_targets_met'],
    'maximum_control_changes': inspection['maximum_control_changes'],
    'same_allocation_timings': inspection['same_allocation_timings'],
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/moving-hertz-probes-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, PARENT, HELPER_SOURCE, HELPER, test_copy, receipt_path]
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    files.extend((N / name).glob('group*/**/tests.xml'))
    external.update((N / name).glob('group*/**/*.npz'))
    for path in (N / name).glob('group*/**/start.json'):
        value = read(path)
        for entry in value.values():
            if isinstance(entry, dict) and {'path', 'bytes', 'sha256'} <= entry.keys():
                source = Path(entry['path'])
                assert identity(source) == {key: entry[key] for key in ['path', 'bytes', 'sha256']}
                external.add(source)
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hertz-moving-probes-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-hertz-moving-probes-v1/README.md').write_text('# Moving normal material probes\n\n' + scope +
    '\n\nPublic experimental engine: [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + '). '
    'Use this exact source and captured workers/task lists. Historical engine patches reconstruct from the public parent67c9de0b2; focused test sources are independently retained and hash-bound. '
    'Decode gzip files into a separate replay directory. Run scientific work with the existing data-side physical Python, Slurm `rtx-mid`, image `1_26`, an explicit fresh data output and private node-local caches. '
    'Native/reference fields and complete per-frame benchmark audits remain at externally hashed data paths. '
    'See [the report](../../../docs/NATIVE_MOVING_HERTZ_PROBES_20261022.md). '
    'The estimator is not connected to automatic collision inputs or public stress options; the complete active goal remains open.\n')
with (N / 'meta/moving-hertz-probes-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
