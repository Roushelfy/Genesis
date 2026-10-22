"""Retain same-pose finite boundaries, independent edge trials and load-line diagnosis."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
exec(compile(REPLAY_HELPER.read_text().split('\nC3 = ', 1)[0], str(REPLAY_HELPER), 'exec'))
C3 = WORK.parent / 'hard_contact_finite_edge'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS.startswith('660c85e98')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-finite-edge-franka-boundary-v2', 'hard-finite-edge-lookup-v1',
         'hard-finite-edge-lookup-v2', 'hard-finite-edge-lookup-v3', 'hard-finite-edge-boundary-check-v1',
         'hard-finite-edge-capture-comparison-v1', 'hard-finite-edge-capture-comparison-v2',
         'hard-finite-edge-force-line-v1', 'hard-finite-edge-force-line-v2']
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
allowed = [ALIAS]
allowed_trees = {subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip() for sha in allowed}
assert all(row['reconstructed_tree'] in allowed_trees for row in replayed)
bindings = []
for module, campaign in [('check_live_curved_hertz_franka', 'hard-finite-edge-franka-boundary-v2'),
                         ('study_hertz_edge_lookup', 'hard-finite-edge-lookup-v3'),
                         ('check_hertz_face_boundaries', 'hard-finite-edge-boundary-check-v1'),
                         ('inspect_hertz_boundary_capture', 'hard-finite-edge-capture-comparison-v2'),
                         ('inspect_hertz_force_line', 'hard-finite-edge-force-line-v2')]:
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
heads = read(N / 'meta/curved-source-terminal-v1.json')['frozen_heads']
heads['hard_contact_finite_edge'] = ALIAS
for name, sha in heads.items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
boundaries = [read(N / f'hard-finite-edge-boundary-check-v1/group{i}/boundaries/result.json') for i in [0, 1]]
assert all(row['passed'] for row in boundaries)
comparison = read(N / 'hard-finite-edge-capture-comparison-v2/group0/comparison/result.json')
assert comparison['passed'] and len(comparison['frames']) == 109
scene = read(N / 'hard-finite-edge-franka-boundary-v2/group0/franka/result.json')
assert not scene['passed'] and len(scene['rows']) == 109
assert all(all(row['accepted']) for row in scene['rows'][:-1])
assert scene['rows'][-1]['step'] == 108 and not all(scene['rows'][-1]['accepted'])
lookup = [read(N / f'hard-finite-edge-lookup-v{i}/group0/lookup/result.json') for i in [1, 2, 3]]
assert all(not row['passed'] for row in lookup)
assert all(row['engineering_targets_met'] for row in lookup[0]['rows'] if row['case'].startswith('circular-'))
assert all(row['variational']['engineering_targets_met'] for row in lookup[2]['rows'])
force_line = read(N / 'hard-finite-edge-force-line-v2/group0/force-line/result.json')
assert force_line['passed'] and len(force_line['rows']) == 1
assert force_line['rows'][0]['region_status'] == 2 and not force_line['rows'][0]['footprint_contained_in_face']
unsubmitted = 'hard-finite-edge-franka-boundary-v1'
assert not (N / unsubmitted / 'submission.json').exists()
assert not (N / unsubmitted / 'source.json').exists()
scope = ('Native finite-box half-planes are retained at the contact evaluation pose with unchanged production gates. '
    'CPU/CUDA boundary, rotation, release and restoration controls pass. All109 actual Franka captures retain '
    'positions, forces, histories and acceptance, with only roundoff field/curvature differences. First rejection '
    'remains108; same-substep normal-root excess1.816555um supersedes post-integration2.051902um. '
    'Circular nine-node pressure lookup passes independent held-out circular controls but fails flattened ellipses. '
    'An offline variational support/mixture/tangential-shift trial passes selected actual-root and flattened controls; '
    'it is not a qualified native model or runtime cost. Correct material/world transforms replace the original '
    'actual-contact diagnostic. The actual force-line curved-surface intersection preserves its wrench but lies '
    '29.05um outside a different finger edge,197.2um from the collision witness. Original frame/connectivity-reader '
    'failures and one unsubmitted sparse-checkout preflight are retained. Collision/material correspondence, '
    'load-compatible finite support, global peaks/friction and accepted max/full scene throughput remain open.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings, 'public_engine_alias': ALIAS,
    'native_boundary_checks': boundaries, 'capture_comparison': comparison,
    'first_invalid_step': 108, 'edge_pressure_trials': lookup, 'force_line_inspection': force_line,
    'unsubmitted_preflight': {'campaign': unsubmitted, 'submitted': False,
        'reason': 'Tracked-source preflight found missing pyproject.toml in new sparse checkout. Git2.34.1 had treated --cone as a literal pattern; worktree-local cone configuration and explicit paths restored tracked root files before v2 capture.'},
    'superseded_diagnostics': {'lookup_v1': 'Actual force/position material-world frame conversion was missing. Circular training and independent circular controls are unaffected.',
        'capture_comparison_v1': 'Exact curvature comparison failed at step108, absolute7.11e-15/m. Corrected reader explicitly allows relative1e-13; production gates unchanged.',
        'force_line_v1': 'Surface connectivity axes were transposed; v2 uses the recorded six-by-face layout.'},
    'preserved_files': preserved, 'frozen_heads': heads, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/finite-edge-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path,
         N / 'meta/finite-edge-frame-bytes-v1.json']
files.extend(path for path in sources(unsubmitted) if path.is_file())
external = {N / 'hard-edge-v2/group0/study/result.json'}
external.update(Path(item['path']) for row in comparison['frames'] for item in [row['before'], row['after']])
for name in names:
    root = N / name
    files.extend(sources(name) + terminal(name))
    files.extend(root.glob('extra-source*'))
    external.update(root.glob('group*/**/*.npz'))
    external.update(root.glob('group*/**/*.obj'))
# Inputs outside these campaigns remain bound by exact identities without duplication.
for name in ['hard-finite-edge-lookup-v3', 'hard-finite-edge-franka-boundary-v2']:
    for path in (N / name).glob('group*/**/start.json'):
        start = read(path)
        for item in start.get('inputs', []):
            external.add(Path(item['path']))
        if 'mesh' in start:
            external.add(Path(start['mesh']['path']))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-finite-edge-v1', files, sorted(external-set(files)), scope)
(DEST / 'hard-finite-edge-v1/README.md').write_text('# Same-pose finite boundaries and independent edge trials\n\n' + scope +
    '\n\nEngine [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + ') is a public ancestor. '
    'See [measurements and limits](../../../docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#same-substep-finite-face-boundaries-and-edge-pressure). '
    'Replay captured tasks with the pinned engine, Slurm rtx-mid, image1_26, the existing data-side '
    'environment, fresh explicit data outputs and private node caches. Original FAILED states remain evidence; '
    'offline fitted rows passing do not make the original circular lookup pass. Decode gzip separately and leave '
    'packet bytes intact. Full fields, scene frames and generated meshes remain at hashed data paths.\n')
with (N / 'meta/finite-edge-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
