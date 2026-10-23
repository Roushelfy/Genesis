"""Freeze actual native normal lifecycle, clock controls and matched scene cost."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
PARENT = N / 'meta/publish-native-hertz-recovery-v1.py'
exec(compile(PARENT.read_text().split('\nC3 = ', 1)[0], str(PARENT), 'exec'))
C3 = WORK.parent / 'hard_contact_live_normal'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
assert ALIAS == 'fc63cf32ea74444168fd328e0bf96ae9b2f0b5f7'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = [f'hard-native-hertz-live-normal-v{i}' for i in range(1, 4)]
names += [f'hard-native-hertz-live-inspection-v{i}' for i in range(1, 3)]
names += [f'hard-native-hertz-live-lifecycle-v{i}' for i in range(1, 5)]
names += [f'hard-native-hertz-live-failure-inspection-v{i}' for i in range(1, 3)]
names += ['hard-native-hertz-live-regression-v1', 'hard-native-hertz-walk-roundoff-cpu-v1',
          'hard-native-hertz-walk-roundoff-gpu-v1', 'hard-native-hertz-walk-roundoff-v2',
          'hard-native-stress-clock-cpu-v1', 'hard-native-stress-clock-gpu-v1',
          'hard-native-hertz-live-timing-v1', 'hard-native-hertz-live-default-v1',
          'hard-native-hertz-live-trace-v1']
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

# Replay all primary and per-task source overrides through private data-side
# Git indices. Every tracked file hash, including the frozen negative controls,
# must match its actual executed source inventory.
replayed, blob_hashes, trees = [], {}, {}
batch = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=C3, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    with tempfile.TemporaryDirectory(prefix='live-hertz-replay-', dir=N / 'meta') as temporary:
        for name in names:
            root = N / name
            for inventory in [root / 'source.json', *sorted(root.glob('extra-source*.json'))]:
                source = read(inventory)
                patch = inventory.with_suffix('.patch')
                assert hashlib.sha256(patch.read_bytes()).hexdigest() == source['patch_sha256']
                key = (source['revision'], source['patch_sha256'])
                if key not in trees:
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
                    trees[key] = (tree, entries)
                tree, entries = trees[key]
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
                    assert blob_hashes[blob] == (row['bytes'], row['sha256']), (name, inventory.name, row['path'])
                for relative, commit in source['gitlinks'].items():
                    assert entries[relative] == ['160000', 'commit', commit]
                replayed.append({'campaign': name, 'inventory': inventory.name, 'recorded_revision': source['revision'],
                    'recorded_patch_sha256': source['patch_sha256'], 'reconstructed_tree': tree,
                    'all_captured_source_files_exact': len(source['files'])})
finally:
    batch.stdin.close()
    batch.wait()

measured_alias = 'b5c28c271c888d082ebabd7a58294ba9ff48174e'
measured_tree = subprocess.check_output(['git', 'rev-parse', measured_alias + '^{tree}'], cwd=C3, text=True).strip()
final_tree = subprocess.check_output(['git', 'rev-parse', ALIAS + '^{tree}'], cwd=C3, text=True).strip()
for row in replayed:
    if row['campaign'] in ['hard-native-hertz-live-lifecycle-v4', 'hard-native-hertz-live-inspection-v2',
            'hard-native-hertz-live-timing-v1']:
        assert row['reconstructed_tree'] == measured_tree
    if row['campaign'] in ['hard-native-hertz-live-default-v1', 'hard-native-hertz-live-trace-v1']:
        assert row['reconstructed_tree'] == final_tree
assert subprocess.check_output(['git', 'diff', '--name-only', measured_alias, ALIAS], cwd=C3, text=True).splitlines() == ['genesis/options/rigid_stress.py']

for backend in ['cpu', 'gpu']:
    root = N / f'hard-native-stress-clock-{backend}-v1'
    for route in ['hertz', 'dpr']:
        original, repaired = [read(root / f'group{i}/{route}/result.json') for i in range(2)]
        assert original['passed'] and original['source_expected_last_only'] and original['maximum_relative_miss'] > 1e-7
        assert repaired['passed'] and repaired['complete_physical_substep_aggregation'] and repaired['maximum_relative_miss'] == 0
lifecycle = [read(N / f'hard-native-hertz-live-lifecycle-v4/group{i}/lifecycle/result.json') for i in range(2)]
defaults = [read(N / f'hard-native-hertz-live-default-v1/group{i}/check/result.json') for i in range(2)]
assert all(row['passed'] and row['default_quadrature_used'] and row['quadrature'] == 8 for row in defaults)
assert all(row['passed'] and row['released_centrifugal_field_relative'] < 1e-7 for row in lifecycle)
inspection = [read(N / f'hard-native-hertz-live-inspection-v2/group0/{backend}/result.json') for backend in ['cpu', 'gpu']]
assert all(row['passed'] and len(row['rows']) == 32 for row in inspection)
timing = read(N / 'hard-native-hertz-live-timing-v1/group0/timing/result.json')
assert timing['passed'] and len(timing['rows']) == 12
assert all(row['admitted_audit_environment_steps'] == 24 * row['batch'] for row in timing['rows'] if row['mode'] != 'rigid')
trace = read(N / 'hard-native-hertz-live-trace-v1/group0/trace/result.json')
assert trace['passed'] and not trace['primary_timing'] and [row['batch'] for row in trace['rows']] == [1, 1024]

owned = {
    'check_live_hertz_normal': 'hard-native-hertz-live-default-v1',
    'check_live_hertz_lifecycle': 'hard-native-hertz-live-lifecycle-v4',
    'inspect_live_hertz_normal': 'hard-native-hertz-live-inspection-v2',
    'inspect_live_hertz_failures': 'hard-native-hertz-live-failure-inspection-v2',
    'check_hertz_probe_walk_roundoff': 'hard-native-hertz-walk-roundoff-v2',
    'check_native_stress_clock': 'hard-native-stress-clock-cpu-v1',
    'time_live_hertz_normal': 'hard-native-hertz-live-timing-v1',
    'trace_live_hertz_normal': 'hard-native-hertz-live-trace-v1',
    'native_hertz_transfer': 'hard-native-hertz-live-regression-v1',
    'check_native': 'hard-native-hertz-live-regression-v1',
}
bindings = []
for module, name in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    saved = N / name / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
test_copy = N / 'meta/live-normal-test_stress-v1.py'
test_bytes = (C3 / 'tests/rigid/test_stress.py').read_bytes()
assert subprocess.check_output(['git', 'show', 'a599b9c5033da7f6d4e40b79eb9be9a9058d0be0:tests/rigid/test_stress.py'], cwd=C3) == test_bytes
if test_copy.exists():
    assert test_copy.read_bytes() == test_bytes
else:
    test_copy.write_bytes(test_bytes)

scope = ('Actual rigid-solver Hertz normal contact route uses current collider provenance, region curvature/wrench, '
    'native normal law, regular elastic recovery and89 moving material probes, with public max/full/principal/probe APIs. '
    'CPU/GPU rotated input, independent same-load P2 fields, substeps, manual cadence, isolation, partial resets, '
    'state restore, copied/borrowed views and release/reload controls pass. '
    'An inherited checkpoint-ring/physical-substep clock bug is repaired for Hertz and original DPR routes; exact old-source negative controls remain frozen. '
    'The near-vertex material walk is repaired without changing its1e-10 barycentric gate,64-step bound or material adjacency. '
    'Released residual centrifugal fields agree independently; zero angular velocity gives exact zero. '
    'The original constructor, quadrature, locator, clock, tiny-release test assumption and replay-check failures retain actual FAILED states. '
    'Actual paired B1/64/256/1024 headless scene timings include two physical substeps, input motion and required native acceptance; all audited frames are admitted. '
    'Measured order8 replaces the prior experimental order6 default. Final default/rotation and separate original-producer CPU/CUDA profiles are captured. '
    'This is a centered normal sphere/flat-link control; curved/inner material response, physical friction/history and accepted Franka throughput remain active requirements.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'current_worker_bindings': bindings,
    'public_engine_alias': ALIAS, 'public_engine_tree': final_tree,
    'actual_timing_and_lifecycle_source_alias': measured_alias,
    'final_default_only_source_change': 'Hertz helper quadrature6 to8; measured fixtures already use8 explicitly.',
    'frozen_negative_clock_source': '69f89c0645ac1f23c9e00159bfc417c2171fcd0a',
    'test_source': identity(test_copy), 'lifecycle_results': lifecycle,
    'independent_rotated_field_maxima': {key: max(row[key] for result in inspection for row in result['rows'])
        for key in ['probe_relative', 'corner_relative']},
    'same_allocation_timings': timing['rows'],
    'profile_result': identity(N / 'hard-native-hertz-live-trace-v1/group0/trace/result.json'),
    'runtime_data_under_data_and_private_compiler_caches_node_local': True,
    'public_options_connected_in_pinned_engine_alias': True,
    'general_friction_curved_or_franka_qualified': False, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/live-hertz-normal-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, PARENT, HELPER_SOURCE, HELPER, receipt_path, test_copy,
    N / 'meta/hertz-surface-walk-native-move-v1.json',
    N / 'meta/hard-native-hertz-live-normal-start-v1.json']
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    files.extend((N / name).glob('extra-source*'))
    files.extend((N / name).glob('group*/**/*.xml'))
    external.update((N / name).glob('group*/**/*.npz'))
    # CUDA Chrome traces are research profiler artifacts, never production input.
    external.update((N / name).glob('group*/**/*-trace.json'))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-hertz-live-normal-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-hertz-live-normal-v1/README.md').write_text('# Actual native normal contact lifecycle\n\n' + scope +
    '\n\nPublic experimental engine: [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + '). '
    'Use this exact source for the new public Hertz options; task HEAD keeps the established engine baselines. '
    'The terminal receipt reconstructs every primary/override engine capture and verifies every inventoried source file. '
    'Historical negative controls, captured analysis modules/task lists, logs and separately bound regression tests are retained. '
    'Full native/reference/benchmark arrays and intrusive CUDA traces remain at externally hashed data locations. '
    'Decode gzip to a separate replay directory and run scientific work with Slurm `rtx-mid`, image `1_26`, '
    'the existing data-side environment, explicit fresh data outputs and private node scratch. '
    'See [the report](../../../docs/NATIVE_LIVE_HERTZ_NORMAL_20261022.md). '
    'The complete active cheap hard-contact estimator goal remains open.\n')
with (N / 'meta/live-hertz-normal-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
