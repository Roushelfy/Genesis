"""Freeze actual native friction, history lifecycle and paired far-field cost."""
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
C3 = WORK.parent / 'hard_contact_friction_moments'
ALIAS = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=C3, text=True).strip()
BASELINE = '7958052904398bfa31842ea6a2c40ecec1220de1'
CORE = '07bf46ce865e88b6bc80fafb0d9eb9d44c37d811'
UNGUARDED = 'dd0125c1f50478958c6dd49738427a67fb2796f0'
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C3, text=True).strip()
names = ['hard-native-friction-core-v1', 'hard-native-friction-binding-pilot-v1',
    'hard-native-friction-live-pilot-v1', 'hard-native-friction-field-inspection-v1',
    'hard-native-friction-live-timing-v1', 'hard-native-friction-trace-v1',
    'hard-native-friction-binding-controls-v1', 'hard-native-friction-binding-controls-v2',
    'hard-native-friction-live-scaling-v1', 'hard-native-friction-moments-controls-v1',
    'hard-native-friction-moments-diagnostic-v1', 'hard-native-friction-moments-gpu-diagnostic-v1',
    'hard-native-friction-moments-controls-v2', 'hard-native-friction-moments-inspection-v1',
    'hard-native-friction-public-lifecycle-v1', 'hard-native-friction-moments-paired-v2']
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
replay_code = replay_code.replace(
    "[root / 'source.json', *sorted(root.glob('extra-source*.json'))]",
    "[root / 'source.json', *sorted(root.glob('extra-source*.json')), *sorted(root.glob('group*/paired/reference-source.json'))]")
replay_code = replay_code.replace("'inventory': inventory.name", "'inventory': str(inventory.relative_to(root))")
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
trees = {label: subprocess.check_output(['git', 'rev-parse', sha + '^{tree}'], cwd=C3, text=True).strip()
         for label, sha in [('final', ALIAS), ('baseline', BASELINE), ('core', CORE), ('unguarded', UNGUARDED)]}
for row in replayed:
    assert row['reconstructed_tree'] in trees.values(), row
    if row['campaign'] == 'hard-native-friction-core-v1':
        assert row['reconstructed_tree'] == trees['core'], row
    elif row['inventory'].endswith('reference-source.json'):
        assert row['reconstructed_tree'] == trees['baseline'], row
    elif row['campaign'] in names[1:9]:
        assert row['reconstructed_tree'] == trees['baseline'], row
    elif row['campaign'] in names[12:]:
        assert row['reconstructed_tree'] == trees['final'], row
for group in [0, 1]:
    assert read(N / f'hard-native-friction-live-pilot-v1/group{group}/friction-scene/result.json')['physical_checkpoint_replay_passed']
    assert read(N / f'hard-native-friction-binding-controls-v1/group{group}/rotated-live/result.json')['passed']
    assert read(N / f'hard-native-friction-binding-controls-v2/group{group}/binding/result.json')['duplicate_correspondence_loss_latch_and_release_passed']
    assert read(N / f'hard-native-friction-moments-controls-v2/group{group}/tensor-moments/result.json')['far_bank_max_relative_to_uncompressed'] < 1e-6
    assert read(N / f'hard-native-friction-moments-inspection-v1/group0/conformance-{group}/result.json')['passed']
    assert read(N / f'hard-native-friction-moments-inspection-v1/group0/p2-{group}/result.json')['passed']
public = read(N / 'hard-native-friction-public-lifecycle-v1/group0/public-lifecycle/result.json')
assert public['passed'] and all(row['returncode'] == 0 for row in public['checks'])
for backend in ['cpu', 'cuda']:
    suites = ET.parse(N / f'hard-native-friction-public-lifecycle-v1/group0/public-lifecycle/{backend}.xml').getroot().findall('testsuite')
    assert sum(int(row.attrib['tests']) for row in suites) == 2
    assert all(row.attrib['failures'] == row.attrib['errors'] == '0' for row in suites)
paired = read(N / 'hard-native-friction-moments-paired-v2/group0/paired/result.json')
assert paired['passed'] and paired['same_allocation']
paired_rows = []
for label in ['reference', 'current']:
    result = read(N / f'hard-native-friction-moments-paired-v2/group0/paired/{label}/result.json')
    assert result['passed'] and result['force_driven_friction']
    for row in result['rows']:
        if row['mode'] != 'rigid':
            assert row['admitted_audit_environment_steps'] == row['admitted_loaded_audit_steps'] == row['audit_environment_steps']
        paired_rows.append({'source_variant': label, **row})
owned = {
    'check_native_friction_history': 'hard-native-friction-core-v1',
    'check_live_hertz_normal': 'hard-native-friction-live-pilot-v1',
    'native_hertz_tangential': 'hard-native-friction-core-v1',
    'hertz_tangential': 'hard-native-friction-moments-controls-v2',
    'check_native_friction_binding': 'hard-native-friction-binding-controls-v2',
    'check_native_friction_stress': 'hard-native-friction-moments-controls-v2',
    'check_live_hertz_friction': 'hard-native-friction-moments-controls-v2',
    'inspect_live_hertz_friction': 'hard-native-friction-moments-inspection-v1',
    'compare_live_hertz_friction': 'hard-native-friction-moments-inspection-v1',
    'check_native_hertz_recovery': 'hard-native-friction-moments-controls-v2',
    'time_live_hertz_normal': 'hard-native-friction-moments-paired-v2',
    'trace_live_hertz_normal': 'hard-native-friction-trace-v1',
    'check_native': 'hard-native-friction-public-lifecycle-v1'}
bindings = []
for module, campaign in owned.items():
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    captured_worker = N / campaign / 'analysis-source' / relative
    assert captured_worker.read_bytes() == (WORK / relative).read_bytes(), module
    bindings.append({'captured': identity(captured_worker), 'current': identity(WORK / relative), 'exact_match': True})
scope = ('Native normal Hertz plus radial16 tangential history from actual solved rigid contacts, material correspondence/frame transport, '
    'force-line boundary centers and unchanged force/moment/cone/wrench acceptance. Physical history advances under held manual observations, '
    'selected resets remain isolated and physical checkpoint restore preserves memory while invalidating derived output. '
    'CPU/GPU core, actual and rotated force-driven scenes, synthetic two-region material/slot controls, complete same-load independent P2 fields, '
    'public DPR checkpoint lifecycle regressions and actual rigid/max/full B1/B64/B256/B1024 scenes are retained with all failures. '
    'Four-ring far-field moment interpolation preserves weak zero-resultant history; a diagnosed original CPU/GPU exact-switch branch disagreement '
    'is repaired with a machine-roundoff squared-distance band without changing stress gates. '
    'Source-matched complete-field conformance and same-allocation B1/B64 timings evaluate that separate optimization. '
    'These are supported planar half-space scenes and supplied-traction elasticity controls, not new independent pressure qualification, '
    'physical moving-footprint transport validation, curved/inner/hollow-egg qualification or accepted Franka throughput. The complete goal remains active.')
receipt = {'passed': True, 'actual_scheduler': accounting, 'campaign_states': states,
    'captured_source_replay': replayed, 'public_engine_alias': ALIAS, 'uncompressed_engine_alias': BASELINE,
    'core_engine_alias': CORE, 'unguarded_moments_alias': UNGUARDED, 'source_trees': trees,
    'current_worker_bindings': bindings, 'same_allocation_timings': paired_rows,
    'public_lifecycle': public, 'scope': scope, 'whole_goal_complete': False}
receipt_path = N / 'meta/native-friction-live-terminal-v1.json'
with receipt_path.open('x') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
files = [SCRIPT, REPLAY_HELPER, PARENT, HELPER_SOURCE, HELPER, receipt_path]
files.extend(path for path in (N / 'meta').glob('native-friction*') if path.is_file() and path.suffix in ['.json', '.py', '.log'])
external = set()
for name in names:
    files.extend(sources(name) + terminal(name))
    files.extend((N / name).glob('extra-source*.*'))
    files.extend((N / name).glob('group*/**/*.xml'))
    files.extend((N / name).glob('group*/paired/reference-source.*'))
    external.update((N / name).glob('group*/**/*.npz'))
    external.update((N / name).glob('group*/**/*-trace.json'))
cancelled = 'hard-native-friction-moments-paired-v1'
assert not any((N / cancelled).glob('group*/returncode.txt'))
files.extend(sources(cancelled))
files = [path for path in files if path not in external]
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-native-friction-live-v1', files, sorted(external - set(files)), scope)
(DEST / 'hard-native-friction-live-v1/README.md').write_text('# Native friction history and actual rigid observations\n\n' + scope +
    '\n\nFinal experimental source: [`' + ALIAS[:9] + '`](https://github.com/Roushelfy/Genesis/commit/' + ALIAS + '). '
    'Uncompressed795805290, core07bf46ce8 and unguardeddd0125c1f are retained public ancestors. '
    'See [executed checks, timings and limits](../../../docs/NATIVE_LIVE_HERTZ_FRICTION_20261022.md). '
    'Replay captured module/argument lists with Slurm rtx-mid, image1_26, the existing data-side physical environment, '
    'fresh explicit data outputs and private node-local caches. Decode gzip separately; complete arrays/traces remain on data. '
    'No full-goal completion or accepted Franka throughput is claimed.\n')
with (N / 'meta/native-friction-live-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2)
    stream.write('\n')
print(json.dumps(packet))
