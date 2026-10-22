"""Freeze source-bound conformance, paired scenes and independent friction."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_bounded_sources'
ALIAS = '5a28cf83734756020776ff885a39233ccd567913'
BASELINE = 'fc63cf32ea74444168fd328e0bf96ae9b2f0b5f7'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-source-bounds-controls-v2', 'hard-source-bounds-paired-v2',
    'hard-source-bounds-trace-v2', 'hard-tangential-contact-pilot-v1',
    'hard-tangential-contact-pilot-v2', 'hard-tangential-contact-reference-v1',
    'hard-source-and-tangent-inspection-v1', 'hard-tangential-contact-inspection-v2',
    'hard-source-bounds-paired-v3', 'hard-source-bounds-reversed-inspection-v1']
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

# Reuse exact source replay, extending it to the paired child's independently
# captured baseline. Private indices leave every executed worktree unchanged.
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
replay_code = replay_code.replace(
    "[root / 'source.json', *sorted(root.glob('extra-source*.json'))]",
    "[root / 'source.json', *sorted(root.glob('extra-source*.json')), *sorted(root.glob('group*/paired/reference-source.json'))]")
replay_code = replay_code.replace("'inventory': inventory.name", "'inventory': str(inventory.relative_to(root))")
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
alias_tree = subprocess.check_output(['git', 'rev-parse', ALIAS + '^{tree}'], cwd=C3, text=True).strip()
baseline_tree = subprocess.check_output(['git', 'rev-parse', BASELINE + '^{tree}'], cwd=C3, text=True).strip()
for row in replayed:
    if row['campaign'] in ['hard-tangential-contact-pilot-v1', 'hard-tangential-contact-pilot-v2',
            'hard-tangential-contact-reference-v1'] or row['inventory'].endswith('reference-source.json'):
        assert row['reconstructed_tree'] == baseline_tree, row
    else:
        assert row['reconstructed_tree'] == alias_tree, row

source_checks = []
test_files = []
for group in [0, 1]:
    root = N / f'hard-source-bounds-controls-v2/group{group}'
    result = read(root / 'sources/result.json')
    assert result['passed'] and result['returncode'] == 0
    suites = ET.parse(root / 'sources/tests.xml').getroot().findall('testsuite')
    assert sum(int(row.attrib['tests']) for row in suites) == 7
    assert all(row.attrib['failures'] == row.attrib['errors'] == '0' for row in suites)
    for entry in result['test_sources']:
        assert identity(Path(entry['path'])) == entry
    source_checks.append({'backend_group': group, 'passed_tests': 7, 'test_sources': result['test_sources']})
    assert read(root / 'rotated/result.json')['passed']
for relative in ['tests/rigid/test_stress_sources.py', 'tests/rigid/test_stress_nonconvex.py']:
    original = C3 / relative
    copied = N / 'meta' / ('source-bounds-' + original.name)
    if copied.exists():
        assert copied.read_bytes() == original.read_bytes()
    else:
        copied.write_bytes(original.read_bytes())
    assert subprocess.check_output(['git', 'show', BASELINE + ':' + relative], cwd=C3) == original.read_bytes()
    test_files.append(copied)

first = read(N / 'hard-source-and-tangent-inspection-v1/group0/inspection/result.json')
reverse = read(N / 'hard-source-bounds-reversed-inspection-v1/group0/inspection/result.json')
assert first['passed'] and reverse['passed']
assert len(first['same_allocation_timings']) == 12 and len(reverse['same_allocation_timings']) == 9
assert all(row['admitted_environment_steps'] == 24 * row['batch'] for row in first['same_allocation_timings'] + reverse['same_allocation_timings'] if row['mode'] != 'rigid')
physics = read(N / 'hard-tangential-contact-inspection-v2/group0/inspection/result.json')
assert physics['passed'] and physics['nu0_analytical_controls_within_2p']
assert physics['all_combined_reference_changes_within_2p']
assert not physics['all_strict_tangent_reference_changes_within_2p']
history = [row for row in physics['comparisons'] if row['model'] == 'radial_history16']
proportional = [row for row in physics['comparisons'] if row['model'] == 'proportional']
assert len(history) == len(proportional) == 16
assert all(row['preexisting_combined_5p_peak_10p_targets_met_on_bank'] for row in history)
assert not all(row['strict_tangent_5p_target_met'] for row in history)
assert not all(row['preexisting_combined_5p_peak_10p_targets_met_on_bank'] for row in proportional)
for group in range(5):
    row = read(N / f'hard-tangential-contact-reference-v1/group{group}/study/result.json')
    assert row['passed'] and row['numerical_reference_execution_passed']
    assert len(row['rows']) == 8 and len(row['history']) == 6 * row['subdivisions']

owned = {'check_source_bounds': 'hard-source-bounds-controls-v2',
    'inspect_source_bounds': 'hard-source-bounds-reversed-inspection-v1',
    'time_live_hertz_normal': 'hard-source-bounds-paired-v3',
    'tangential_contact_reference': 'hard-tangential-contact-reference-v1',
    'study_tangential_contact_reference': 'hard-tangential-contact-reference-v1',
    'inspect_tangential_contact_reference': 'hard-tangential-contact-inspection-v2'}
bindings = []
for module, campaign in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_worker = N / campaign / 'analysis-source' / relative
    assert captured_worker.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(captured_worker), 'current': identity(WORK / relative), 'exact_match': True})

scope = ('Existing immutable face boxes prune exact witnessed-source nearest and ambiguity queries, preserving canonical order, '
    'topology, source/wrench, transform tolerances and all native acceptance. Seven source tests perCPU/CUDA and actual '
    'rotated complete-field/source/probe captures pass against frozen baselinefc63cf32e. '
    'Two same-allocation rigid/max/full source orders, includingB4096, retain raw timing repetitions and every-frame audit identities; '
    'the firstB1 max regression remains explicit. Separate original-producer CUDA profiles and unchanged logical-memory counts remain diagnostic. '
    'An independent incremental vector Cerruti/Coulomb contact reference covers supplied circular Hertz pressure, variable normal force, '
    'vector reversal and residual history with matched17/25 spatial and4/8 time controls. '
    'Combined sampled tensor/peak changes meet the predeclared2% reference-change gate and nu0 monotonic analytical controls pass. '
    'Radial16 history meets existing5% combined/10% peak budgets on96 positive-depth points; proportional shear fails them. '
    'Additional strict tangent-only accuracy/refinement targets remain false, especially in weak residual fields. '
    'Original sparse-source prelaunch, coarse stress-quadrature and NumPy-bool serialization failures remain unmodified. '
    'These are normal scene performance and conditional positive-depth friction controls, not native friction integration, '
    'surface/global/full-volume accuracy, curved/inner/hollow-egg qualification or accepted Franka throughput; the complete goal remains active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings,
    'public_engine_alias': ALIAS, 'public_engine_tree': alias_tree, 'frozen_baseline_alias': BASELINE,
    'source_tests': source_checks, 'source_test_copies': [identity(path) for path in test_files],
    'same_allocation_first_order': first['same_allocation_timings'],
    'same_allocation_reverse_order': reverse['same_allocation_timings'],
    'physical_reference_variations': physics['variations'],
    'physical_candidate_comparisons': physics['comparisons'],
    'combined_reference_changes_within_2p': True,
    'strict_tangent_reference_changes_within_2p': False,
    'radial_history_combined_engineering_targets_on_bank': True,
    'radial_history_strict_tangent_targets': False,
    'proportional_combined_engineering_targets_on_bank': False,
    'native_friction_surface_global_or_franka_qualified': False,
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/source-bounds-and-friction-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path, *test_files,
    N / 'meta/hard-bounded-sources-start-v1.json', N / 'meta/hard-source-bounds-prelaunch-v1.json']
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    files.extend((N / name).glob('group*/**/*.xml'))
    files.extend((N / name).glob('group*/paired/reference-source.*'))
    external.update((N / name).glob('group*/**/*.npz'))
    external.update((N / name).glob('group*/**/*-trace.json'))
for name in ['hard-source-bounds-controls-v1', 'hard-source-bounds-paired-v1', 'hard-source-bounds-trace-v1']:
    assert not (N / name / 'submission.json').exists()
    files.extend(path for path in sources(name) if path.is_file())
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-source-bounds-and-friction-reference-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-source-bounds-and-friction-reference-v1/README.md').write_text('# Native source bounds and independent vector friction\n\n' + scope +
    '\n\nPublic native experimental source: [`5a28cf837`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + '). '
    'Frozen baselinefc63cf32e and every primary/child source capture are exactly replayed. '
    'See [native source cost](../../../docs/NATIVE_SOURCE_BOUNDS_20261022.md) and '
    '[independent friction](../../../docs/TANGENTIAL_CONTACT_REFERENCE_20261022.md). '
    'Decode gzip captures separately. Use captured task recipes with Slurm `rtx-mid`, image `1_26`, '
    'the existing data-side physical environment, explicit fresh data outputs and private node-local compiler caches. '
    'Full trajectories, fields, raw contact histories, benchmark audits and CUDA traces remain at externally hashed data paths. '
    'The complete cheap hard-contact estimator and accepted moving Franka scene objective remains active.\n')
with (N / 'meta/source-bounds-and-friction-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
