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
C3 = WORK.parent / 'hard_contact_edge_farfield'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('4a8125c4c')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-edge-wrench-allocation-v1', 'hard-edge-farfield-controls-v1',
         'hard-native-edge-farfield-gpu-v1', 'hard-edge-farfield-response-v1',
         'hard-native-edge-farfield-gpu-v2', 'hard-edge-precision-v1',
         'hard-edge-allocation-response-v1', 'hard-edge-farfield-comparison-v1',
         'hard-native-edge-farfield-regression-v1']
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
allowed = [ALIAS, 'e64780805847beb0d8f902d3a05392976ce93cd8']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('study_edge_wrench_allocation', 'hard-edge-allocation-response-v1'),
                         ('study_edge_farfield', 'hard-edge-farfield-controls-v1'),
                         ('check_native_edge_farfield', 'hard-native-edge-farfield-gpu-v2'),
                         ('inspect_edge_precision', 'hard-edge-precision-v1'),
                         ('study_edge_source_response', 'hard-edge-allocation-response-v1'),
                         ('compare_edge_field_response', 'hard-edge-farfield-comparison-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/edge-fields-terminal-v1.json')['frozen_heads']
heads['hard_contact_edge_farfield'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
references = read(N / 'hard-edge-farfield-controls-v1/group1/controls/result.json')
paired = [read(N / f'hard-native-edge-farfield-gpu-v1/group0/64-{label}/result.json')
          for label in ['before', 'candidate', 'after']]
single = read(N / 'hard-native-edge-farfield-gpu-v2/group0/32-candidate/result.json')
precision = read(N / 'hard-edge-precision-v1/group0/precision/result.json')
comparison = read(N / 'hard-edge-farfield-comparison-v1/group0/comparison/result.json')
regression = [read(N / f'hard-native-edge-farfield-regression-v1/group{i}/fields/result.json') for i in [0, 1]]
allocation = read(N / 'hard-edge-allocation-response-v1/group0/allocation/result.json')
response = read(N / 'hard-edge-allocation-response-v1/group0/response/result.json')
assert all(r['passed'] for r in [references, *paired, single, precision, comparison, *regression, allocation, response])
assert precision['maximum_relative_error'] > 5e-4 and single['maximum_numerical_point_error'] < 5e-4
assert comparison['regular_displacement_identical']
assert allocation['constrained']['qualified_constrained_minimum']
assert response['rows'][-1]['all_contact_allocation_response']['common_probes']['tensor_relative'] > 0.05
scope = ('Four equal remote pressure-moment quadrature sources reduce the sharp-edge particular tensor cost '
    'beyond eight patch dimensions. Eleven independent continuous prescribed-pressure controls have at most '
    '0.02690% point tensor error; this checks field integration, not pressure/contact physical accuracy. '
    'Native 64/32 numerical errors are9.18e-11/1.09e-5 and prior surface/isolation checks remain admitted. '
    'Same-allocation 64-bit baseline-before/candidate/baseline-after for65536 queries measures2.829/1.249/2.826ms, '
    'excluding full recovery and scenes. Original32-bit remote cancellation fails the unchanged numerical target '
    'and is retained. The candidate passes. Complete-shell remote-rule comparison has maximum coarse-volume '
    'change0.00209% with identical regular displacement. An all-contact fixed-support force allocation using '
    'actual material normals and reconstructed ground geometry preserves total wrench and Coulomb cones with '
    'up to6.22% force change. Its combined near-probe tensor changes5.67%, exceeding the5% budget; no runtime '
    'source gate is changed. The earlier origin-centered ground-cache trial is invalid geometry evidence, '
    'not physical allocation evidence. Self-consistent pressure/support, collision provenance, independent '
    'complete physical stress accuracy and accepted paired max/full grasp throughput remain open. Goal active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'continuous_remote_fields': references, 'paired64_component_timing': paired,
    'single32_component_timing': single, 'baseline32_failure_diagnostic': precision,
    'complete_shell_comparison': comparison, 'surface_isolation_regression': regression,
    'all_contact_force_allocation': allocation, 'fixed_support_stress_response': response,
    'invalid_cached_ground_trial': 'hard-edge-farfield-response-v1/group0/allocation; unassigned ground center at origin, excluded from physical conclusions',
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/edge-farfield-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    for path in root.glob('group*/**/start.json'):
        for item in read(path).get('inputs', []):
            external.add(Path(item['path']))
external -= set(files)
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-edge-farfield-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-edge-farfield-v1/README.md').write_text('# Remote sharp-edge fields and constrained load sensitivity\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#remote-field-quadrature-and-paired-component-cost). '
    'Replay captured tasks with pinned sources, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Full arrays remain at hashed data paths. '
    'Original failures remain retained; source acceptance targets remain unchanged.\n')
with (N / 'meta/edge-farfield-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
