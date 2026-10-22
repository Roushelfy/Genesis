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
C3 = WORK.parent / 'hard_contact_edge_fields'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS == 'e64780805847beb0d8f902d3a05392976ce93cd8'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-edge-fields-v1', 'hard-native-edge-fields-cpu-v1', 'hard-native-edge-fields-gpu-v1',
         'hard-native-edge-fields-cpu-v2', 'hard-native-edge-fields-gpu-v2', 'hard-edge-source-response-v1',
         'hard-edge-source-response-v2', 'hard-edge-shell-tilt-v1']
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
allowed = [ALIAS, '7a77a5b15ff633ba8bcf0cd06328bf88f890a363', '88d9e891aa321010cbd0dd50777e978b7d84fcc0']
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('edge_fields', 'hard-native-edge-fields-gpu-v2'), ('study_edge_fields', 'hard-edge-fields-v1'),
                         ('check_native_edge_fields', 'hard-native-edge-fields-gpu-v2'),
                         ('study_far_edge_contact', 'hard-edge-source-response-v2'),
                         ('study_edge_source_response', 'hard-edge-source-response-v2'),
                         ('inspect_edge_shell_tilt', 'hard-edge-shell-tilt-v1')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preserved = 0
for name, data in read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/edge-law-terminal-v1.json')['frozen_heads']
heads['hard_contact_edge_fields'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
references = [read(N / f'hard-edge-fields-v1/group{i}/fields/result.json') for i in [1, 2]]
assert all(result['passed'] and result['composite32_targets_met'] for result in references)
native = [read(N / f'hard-native-edge-fields-{backend}-v2/group{i}/fields/result.json') for backend in ['cpu', 'gpu'] for i in [0, 1]]
assert all(result['passed'] and result['queries'] == 768 and result['invalid_singular_isolation_restore_checked']
           and result['original_hertz_quadrature_checked'] for result in native)
assert all(len(result['benchmarks']) == (3 if result['backend'] == 'gpu' else 0) for result in native)
response = read(N / 'hard-edge-source-response-v2/group0/response/result.json')
tilt = read(N / 'hard-edge-shell-tilt-v1/group0/tilt/result.json')
assert response['passed'] and tilt['passed'] and tilt['mass_rigid_projection_checked']
assert all(row['common_probes']['tensor_relative'] > 1 for row in response['rows'][-1]['source_response'])
scope = ('Continuous prescribed sharp-edge pressure integration validates a local exact strip plus finite parabolic '
    'centroid-line stress correction down to0.1 transverse width. Four local cases have0.164-0.524% sampled tensor '
    'errors; this is field integration accuracy, separate from contact-pressure and shell model accuracy. Native '
    'CPU/CUDA32/64 controls cover768 queries each, invalid/singular isolation, restoration and the original Hertz '
    'quadrature; endpoint cancellation failures remain preserved and the fixed expression passes unchanged targets. '
    'Resident tensor evaluation cost is measured on separate precision allocations, excluding full recovery/scenes. '
    'Native curved-shell point corrections plus edge fields compare pressure-centroid and rigid-force-line sources. '
    'Shifts92.98/369.66um change common near-probe tensors about141% despite small peak-magnitude changes; coarse '
    'volume changes1.54/62.41% are explicitly unresolved near the footprint. A two-gripper-source smooth shell-tilt '
    'check predicts about0.02um shifts, far below required64.5/356um along-edge changes. No source gate is loosened. '
    'Actual contact binding, constrained load allocation, independent complete physical shell/peak error and '
    'accepted matched pure-rigid/max/full grasp throughput remain open. Goal active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'continuous_pressure_fields': references, 'native_controls_and_component_timing': native,
    'curved_shell_source_response': response, 'smooth_shell_tilt': tilt,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/edge-fields-terminal-v1.json'
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
    for path in root.glob('group*/**/references.json'):
        for item in read(path).get('files', []):
            external.add(Path(item['path']))
external.discard(physical)
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-edge-fields-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-edge-fields-v1/README.md').write_text('# Sharp-edge fields and curved-shell source response\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [derivation, measurements and limits](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#sharp-edge-particular-stress). '
    'Replay captured tasks with pinned sources, Slurm rtx-mid, image1_26, the existing data-side environment, '
    'fresh data outputs and private node caches. Full arrays remain at hashed data paths. '
    'Original failures remain retained; no native source acceptance threshold was changed.\n')
with (N / 'meta/edge-fields-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
