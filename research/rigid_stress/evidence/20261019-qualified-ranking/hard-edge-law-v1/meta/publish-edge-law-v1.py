"""Retain coupled sharp-edge references, unfitted pressure law and native scale costs."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_edge_law'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('88d9e891a')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-edge-shear-coupling-v1', 'hard-edge-shear-coupling-v2', 'hard-edge-asymptotic-law-v1', 'hard-native-edge-law-cpu-v1', 'hard-native-edge-law-gpu-v1']
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
allowed = [ALIAS, '601cde58f']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('hertz_reference', 'hard-native-edge-law-gpu-v1'), ('study_far_edge_contact', 'hard-native-edge-law-gpu-v1'), ('edge_asymptotic', 'hard-native-edge-law-gpu-v1'), ('study_edge_shear_coupling', 'hard-edge-shear-coupling-v2'), ('study_edge_asymptotic_law', 'hard-edge-asymptotic-law-v1'), ('check_native_edge_law', 'hard-native-edge-law-gpu-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/force-line-terminal-v1.json')['frozen_heads']
heads['hard_contact_edge_law'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha

coupling = [read(N / f'hard-edge-shear-coupling-v2/group{i}/coupling/result.json') for i in [1, 2]]
assert all(result['passed'] and result['controls']['frictional_strip_equation_checked'] for result in coupling)
assert all(row['trial_targets_met'] and not row['normal_only_approximation_targets_met'] for result in coupling for row in result['rows'])
unfitted = [read(N / f'hard-edge-asymptotic-law-v1/group{i}/law/result.json') for i in [0, 1]]
assert all(result['passed'] and result['dimensional_controls_passed'] for result in unfitted)
assert all(row['trial_targets_met'] for result in unfitted for row in result['rows'])
native = [read(N / f'hard-native-edge-law-{backend}-v1/group{i}/edge/result.json') for backend in ['cpu', 'gpu'] for i in [0, 1]]
assert all(result['passed'] and result['valid_cases'] == 274 and result['invalid_unloaded_isolation_and_reset_checked'] and result['sharp_edge_unbounded_pressure_checked'] for result in native)
assert all(len(result['benchmarks']) == (3 if result['backend'] == 'gpu' else 0) for result in native)
scope = ('A nonsymmetric unilateral half-space reference includes shear-induced normal displacement with independent '
    'cell integration, manufactured complementarity and original normal-force regression checks. A derived sharp-edge '
    'strip exponent and an unfitted logarithmic scale equation remove offline pressure/support fitting on two actual '
    'curved gaps and reversed loads. Grid72 pressure errors3.4-5.8%, sampled positive-depth tensor errors1.3-1.6% and '
    'sample peak errors1.85-2.13% meet the declared local reference budgets. Native Quadrants supplied-region scales '
    'and pressure evaluation pass274 cases/backend/precision, singular-pressure limits and isolated invalid/unloaded '
    'controls. Resident scale dispatch cost is measured separately. Pressure remains singular at the sharp edge. '
    'Actual source/wrench consistency, longitudinal-shear generality, friction history, complete curved-shell fields '
    'and accepted moving pure-rigid/max/full throughput remain open. No full estimator promotion or goal completion.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'coupled_reference_and_frozen_normal_fit_trials': coupling, 'unfitted_pressure_law': unfitted,
    'native_controls_and_component_timing': native, 'preserved_files': preserved, 'frozen_heads': heads,
    'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/edge-law-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
physical = N / 'meta/native-edge-law-physical-inputs-v1.json'
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path, physical]
external = set()
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    for path in root.glob('group*/**/start.json'):
        for item in read(path).get('inputs', []):
            external.add(Path(item['path']))
external.discard(physical)
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-edge-law-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-edge-law-v1/README.md').write_text('# Sharp-edge pressure and native scale estimates\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md). '
    'Replay captured tasks with pinned engine source, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Original captures remain intact. Full arrays remain at hashed '
    'data paths. Two precision timing campaigns use separate GPU allocations.\n')
with (N / 'meta/edge-law-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
