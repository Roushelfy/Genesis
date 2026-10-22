"""Retain cavity coverage, focused boundary integration and independent fields."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_complete_boundary'
ALIAS = '2e39f5f752136de5d7616b3436063f1bf480814b'
BASELINE = 'd26b5b3d46efe614ed715fef293f8cc49bcdf0c7'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [f'hard-native-hertz-shell-quadrature-v{i}' for i in range(1, 4)] + [
    'hard-native-hertz-complete-boundary-v1', 'hard-native-hertz-complete-boundary-v2',
    'hard-native-hertz-complete-boundary-inspection-v1']
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
    if row['campaign'] in names[:3]:
        assert row['reconstructed_tree'] == trees['baseline'], row
    elif row['campaign'] in names[4:]:
        assert row['reconstructed_tree'] == trees['final'], row
    else:
        changed = subprocess.check_output(['git', 'diff', '--name-only', row['reconstructed_tree'], trees['final']], cwd=C3, text=True).splitlines()
        assert changed == ['genesis/engine/solvers/rigid/stress/hertz_recovery.py'], changed
        trees['unsplit_near_rule'] = row['reconstructed_tree']
quadrature = read(N / names[2] / 'group0/quadrature/result.json')
assert quadrature['passed'] and quadrature['asset_selected_faces'] == 80 and quadrature['complete_material_boundary_faces'] == 160
assert quadrature['candidates'] == {'focused12': False, 'focused16x2': True}
for group in [0, 1]:
    old = read(N / names[3] / f'group{group}/shell-q16/result.json')
    assert not old['passed'] and old['status'] == [4, 4, 0, 4]
    assert read(N / names[3] / f'group{group}/normal-component/result.json')['passed']
    final = read(N / names[4] / f'group{group}/shell-q16/result.json')
    assert final['passed'] and final['status'] == [0] * 4 and max(final['wrench_error']) < 1e-4
    assert final['selected_contact_faces'] == 80 and final['complete_material_boundary_faces'] == 160
assert read(N / names[3] / 'group1/live-friction/result.json')['passed']
inspection = read(N / names[5] / 'group0/inspection/result.json')
assert inspection['passed'] and len(inspection['rows']) == 8
assert all(row['corner_field_relative'] < 1e-7 and row['probe_field_relative'] < 1e-7 for row in inspection['rows'])
bindings = []
for module, campaign in [('study_hertz_shell_quadrature', names[2]), ('check_native_hertz_shell', names[4]),
                         ('inspect_complete_hertz_shell', names[5])]:
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
heads = read(N / 'meta/native-hertz-shell-admission-terminal-v1.json')['frozen_heads']
heads['hard_contact_complete_boundary'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
scope = ('The original level1 asset selects80 outer faces from160 material-boundary faces. Same-load selected-surface checks '
    'do not establish complete boundary correction. Hertz model construction now appends all omitted material faces while '
    'preserving existing face indices, including cavity boundaries and their surface operators. Near-wall point-traction '
    'integration uses closest-point triangle fans and a rational radial map, with two subintervals in each coordinate. '
    'Four facet-centered normal shell cases pass CPU/CUDA and independent higher-order full-boundary P2 fields, original '
    '1e-4 authored-origin wrench and1e-7 field gates. Original uniform-reference, moment-definition and unsplit-native '
    'failures remain. The earlier native variant also passes rotated normal-component CPU/CUDA and planar actual-friction CUDA controls. '
    'This supplies numerical complete-boundary recovery under the declared normal pressure model, not curved-pressure, '
    'moving shell-friction or accepted Franka accuracy/throughput. The whole goal remains active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states, 'captured_source_replay': replayed,
    'public_engine_alias': ALIAS, 'selected_surface_baseline': BASELINE, 'source_trees': trees,
    'current_worker_bindings': bindings, 'independent_quadrature': quadrature, 'independent_native_inspection': inspection,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-hertz-complete-boundary-terminal-v1.json'
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
packet = freeze('hard-native-hertz-complete-boundary-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-hertz-complete-boundary-v1/README.md').write_text('# Complete material boundary and near-wall integration\n\n' + scope +
    '\n\nSource [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') and selected-surface baseline '
    'd26b5b3d4 are public ancestors. See [results and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md). '
    'Use captured tasks with Slurm rtx-mid, image1_26, the existing data-side physical environment and fresh explicit data outputs. '
    'Caches use private node scratch. Full fields remain at hashed data paths; decode gzip separately. The goal remains active.\n')
with (N / 'meta/native-hertz-complete-boundary-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
