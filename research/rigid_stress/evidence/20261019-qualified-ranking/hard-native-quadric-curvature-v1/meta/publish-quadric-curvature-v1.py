"""Retain native quadric curvature and independent curved-geometry controls."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_quadric_curvature'
ALIAS = 'e7e49b738e425f9a25227929262122bcd1f35340'
BASELINE = '32c1f3ecff20813a7567e03f4ed725686b57f21d'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip() == ALIAS
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-quadratic-rest-surface-v1', 'hard-curved-contact-gap-v1', 'hard-vertex-curvature-v1']
names += [f'hard-native-quadric-curvature-v{i}' for i in range(1, 4)]
names += [f'hard-curved-shell-operators-v{i}' for i in range(1, 5)]
names += ['hard-native-quadric-franka-v1', 'hard-native-quadric-contact-v1']
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
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip()
                 for sha in [ALIAS, BASELINE]}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [
    ('study_quadratic_rest_surface', names[0]), ('study_curved_contact_gap', names[1]),
    ('study_vertex_curvature', names[2]), ('check_quadric_curvature', 'hard-native-quadric-curvature-v3'),
    ('check_curved_shell_operators', 'hard-curved-shell-operators-v4'),
    ('check_quadric_contact', 'hard-native-quadric-contact-v1'),
]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
preservation = read(N / 'meta/hard-scene-pilot-start-v1.json')['preservation']['preserved']
preserved = 0
for name, data in preservation.items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
heads = read(N / 'meta/parallel-boundary-terminal-v1.json')['frozen_heads']
heads['hard_contact_quadric_curvature'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
contact = read(N / 'hard-native-quadric-contact-v1/group0/contact/result.json')
operators = read(N / 'hard-curved-shell-operators-v4/group0/operators/result.json')
assert contact['passed'] and contact['rows'][1]['targets_met'] and operators['passed']
assert all(row['accepted_loaded_environments'] == 0 for row in contact['rows'])
assert len(operators['rows']) == 4 and all(row['material_probes'] == 89 for row in operators['rows'])
native = [read(N / f'hard-native-quadric-curvature-v3/group{i}/quadric/result.json') for i in [0, 1]]
assert all(row['passed'] for row in native)
scope = ('Native fixed-shape implicit quadric curvature improves the actual first-loaded Franka pressure '
    'comparison from6.264% to1.683% against the retained independent smooth-gap half-space reference. '
    'CPU/GPU fits, signed normals, explicit flat fallback, disconnected surface selection and existing region '
    'controls pass. Warm constructor costs quantify the added initialization only. Whole coarse egg vertex '
    'curvature still has9.138% RMS error; actual Franka recovery remains refused with zero accepted loaded '
    'updates. Separate normal-constrained quadratic rest geometry exposes curvature/edge-normal limits and '
    'passes independent curved P2 uniform-traction, consistent-mass and material-probe controls. These do not '
    'qualify contact-shell stress, moving friction or accepted Franka throughput. Every original failed harness '
    'attempt and the actual engine refusal retain their FAILED scheduler states. The whole goal remains active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings,
    'public_engine_alias': ALIAS, 'baseline': BASELINE, 'actual_contact_comparison': contact,
    'curved_operator_checks': operators, 'native_curvature_checks': native,
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/quadric-curvature-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
external = {N / 'hard-native-friction-franka-admission-v1/group0/franka/frame0010.npz'}
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    external.update(root.glob('group*/**/*.obj'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-quadric-curvature-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-native-quadric-curvature-v1/README.md').write_text('# Native curvature and curved material controls\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#curvature-correction-and-curved-rest-geometry-control). '
    'Replay captured task groups with the pinned engine, Slurm rtx-mid, image1_26 and the existing data-side environment. '
    'Use fresh explicit data outputs and private node caches. Decode gzip separately; full fields and generated meshes '
    'remain at hashed data paths. Final successful curvature/elastic workers are in v3/v4; original failures remain in '
    'earlier captures. The actual Franka rollout intentionally retains its geometry rejection.\n')
with (N / 'meta/quadric-curvature-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
