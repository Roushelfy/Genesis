"""Retain actual Franka refusal, finite-shell controls and optional-buffer repair."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_shell_recovery'
ALIAS = 'd26b5b3d46efe614ed715fef293f8cc49bcdf0c7'
BASELINE = '7958052904398bfa31842ea6a2c40ecec1220de1'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-friction-franka-admission-v1'] + [f'hard-native-hertz-shell-admission-v{i}' for i in range(1, 5)]
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
        assert code in ['0', '1'] and observed[f'{job}_{group}'] == expected
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
trees = {label: subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip()
         for label, sha in [('final', ALIAS), ('baseline', BASELINE)]}
for row in replayed:
    assert row['reconstructed_tree'] == trees['baseline' if row['campaign'] in names[:3] else 'final'], row
franka = read(N / names[0] / 'group0/franka/result.json')
assert not franka['passed'] and franka['rows'][-1]['step'] == 10
admitted_loaded = sum(sum(loaded and accepted for loaded, accepted in zip(row['loaded'], row['accepted']))
                     for row in franka['rows'])
assert admitted_loaded == 0
inspection = read(N / names[2] / 'group0/franka-inspection/result.json')
assert inspection['passed'] and all(not row['original_normal_gate_passed'] for row in inspection['active_regions'])
assert all(value == 0 for value in inspection['input_status'])
assert all(value == 2 for value in inspection['recovery_status'])
for group in [0, 1]:
    old = read(N / names[3] / f'group{group}/shell-q8/result.json')
    assert not old['passed'] and old['status'] == [4] * 4 and min(old['wrench_error']) > 1e-4
    normal = read(N / names[3] / f'group{group}/normal-component/result.json')
    assert normal['passed']
    final = read(N / names[4] / f'group{group}/shell-q16/result.json')
    assert final['passed'] and final['status'] == [0] * 4 and max(final['wrench_error']) < 1e-4
    assert all(max(row.values()) < 1e-7 for row in final['same_load_errors'])
bindings = []
for module, campaign in [('check_live_hertz_franka', names[0]), ('inspect_hertz_franka_admission', names[2]),
                         ('check_native_hertz_shell', names[4])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_worker = N / campaign / 'analysis-source' / relative
    assert captured_worker.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(captured_worker), 'current': identity(WORK / relative), 'exact_match': True})
preservation = read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved']
preserved = 0
for name, data in preservation.items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/native-friction-live-preservation-v2.json')['frozen_heads']
heads['hard_contact_shell_recovery'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
scope = ('Actual native Franka level1 first loaded step10 refuses all four environments because smooth source normals differ22.63569 degrees '
    'from witnessed facet normals. Input/history checks pass and original source identity/frame/half-space diagnostics remain. '
    'Four separate facet-centered normal shell controls reject order8 boundary integration under the original1e-4 wrench gate; order16 '
    'admits all and passes complete independent same-load P2 corner/probe comparisons below1e-7 on CPU/CUDA. '
    'The optional normal-only shear container is repaired using standard inactive scalar-shaped placeholders, with existing rotated '
    'normal-component CPU/CUDA checks. Original worker axis, empty-array and round-trip edge-reference failures remain preserved. '
    'This provides no independent shell pressure, curved-source or shell-friction error and no accepted Franka throughput. The complete goal remains active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states, 'captured_source_replay': replayed,
    'public_engine_alias': ALIAS, 'baseline_alias': BASELINE, 'current_worker_bindings': bindings,
    'franka_loaded_admitted_environment_steps': admitted_loaded, 'franka_geometry_inspection': inspection,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-hertz-shell-admission-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    external.update((N / name).glob('group*/**/*.npz'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hertz-shell-admission-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-hertz-shell-admission-v1/README.md').write_text('# Native hollow-egg recovery and Franka refusal\n\n' + scope +
    '\n\nSource [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') and original795805290 are public ancestors. '
    'See [results and reproduction](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md). Captured tasks run with Slurm rtx-mid, '
    'image1_26, the existing data-side physical environment and explicit fresh data outputs/private node scratch. '
    'Full arrays remain at their hashed data paths. Decode gzip separately. The whole goal remains active.\n')
with (N / 'meta/native-hertz-shell-admission-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
