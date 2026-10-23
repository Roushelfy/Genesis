"""Freeze complete-domain sharp-edge stress evidence, including failed readouts."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tempfile

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
C3 = WORK.parent / 'hard_contact_traction_vectors'
ALIAS = 'e2979aed2a13125a5a3b092c9cb4dddb43de9291'
names = list(read(META / 'edge-shell-fields-launch-v4.json'))
cancelled = 'edge-shell-fields-inspection-v1'
failed = 'edge-shell-fields-v1'
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        actual = observed[f'{job}_{group}']
        if name == cancelled:
            assert actual[0].startswith('CANCELLED')
            assert not (root / 'inspection/start.json').exists()
            states[name].append({'job': f'{job}_{group}', 'state': actual[0],
                'exit_code': actual[1], 'no_scientific_execution': True})
            continue
        expected = ['FAILED', '1:0'] if name == failed else ['COMPLETED', '0:0']
        assert actual == expected, (name, group, actual)
        assert (root / 'returncode.txt').read_text().strip() == ('1' if name == failed else '0')
        for task in tasks:
            if name == failed:
                assert (root / task['name'] / 'particular.npz').is_file()
                assert not (root / task['name'] / 'result.json').exists()
            else:
                result = read(root / task['name'] / 'result.json')
                assert result.get('passed', result.get('execution_completed', False))
        states[name].append({'job': f'{job}_{group}', 'state': actual[0], 'exit_code': actual[1]})

REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
assert {row['recorded_revision'] for row in replayed} == {ALIAS}
start = read(META / 'edge-shell-fields-start-v1.json')
for name, sha in start['frozen_heads'].items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in start['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
inspection = read(N / 'edge-shell-fields-inspection-v2/group0/inspection/result.json')
assert inspection['execution_completed']
assert not inspection['finite_surface_global_peak'] and not inspection['native_scene_accuracy_qualified']
peak_inspection = read(N / 'edge-shell-peaks-v1/group0/peaks/result.json')
assert peak_inspection['execution_completed'] and not peak_inspection['certified_global_maximum']
diagnosis = read(N / 'edge-shell-integral-diagnosis-v1/group0/diagnosis/result.json')
assert diagnosis['execution_completed']
bindings = []
for stem, campaign in [
    ('check_rectangle_normal_stress', 'edge-shell-rectangle-controls-v2'),
    ('rectangle_normal_stress', 'edge-shell-rectangle-controls-v2'),
    ('edge_normal_batch', 'edge-shell-rectangle-controls-v2'),
    ('edge_shell_sampling', 'edge-shell-sampling-v1'),
    ('inspect_edge_shell_locations', 'edge-shell-locations-v1'),
    ('study_edge_shell_fields', 'edge-shell-fields-v2'),
    ('inspect_edge_shell_fields', 'edge-shell-fields-inspection-v2'),
    ('inspect_edge_shell_peaks', 'edge-shell-peaks-v1'),
    ('diagnose_edge_shell_integrals', 'edge-shell-integral-diagnosis-v1'),
]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope = ('Complete original-domain stress integrals for the independently solved5mN frictionless sharp edge '
    'on the existing500um-wall hollow polar land. Two exact material partitions, three unchanged body meshes, '
    '36/48 pressure cells,3/5 source tables and9/17 far-source integration distinguish numerical changes. '
    'Volume tensor and material-column membrane/bending include the contact region. Peak cutoffs apply only '
    'to peak searches, never field integrals. Failed cross-mesh boundary lookups remain preserved; own-element '
    'unsmoothed corner stresses replace only those lookups, and completed particular arrays are reused exactly. '
    'See the measured gates and variations in the inspection; execution completion does not imply every '
    'physical-error target passed. No finite surface peak, native scene accuracy, nonlinear material, '
    'changing-friction or whole-goal qualification. All55 engine heads and12 original unfinished files preserved.')
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'unchanged_engine_alias': ALIAS,
    'frozen_heads': start['frozen_heads'], 'preserved_files': preserved,
    'source_bindings': bindings, 'inspection': inspection, 'peak_inspection': peak_inspection,
    'integral_diagnosis': diagnosis,
    'reuse_contract': read(META / 'edge-shell-fields-reuse-v1.json'),
    'scope': scope, 'whole_goal_complete': False}
p = META / 'edge-shell-fields-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect([name for name in names if name != cancelled])
files += sources(cancelled) + terminal(cancelled)
files += [p, HELPER_SOURCE, REPLAY_HELPER]
files += [META / name for name in [
    'edge-shell-fields-start-v1.json', 'edge-shell-fields-storage-v1.txt',
    'edge-shell-fields-launch-v1.json', 'edge-shell-fields-launch-v2.json',
    'edge-shell-fields-launch-v3.json', 'edge-shell-fields-launch-v4.json',
    'edge-shell-fields-reuse-v1.json', 'edge-shell-fields-inspection-v1-cancel.json',
]]
for name in names:
    files += [META / f'{name}-tasks.json', META / f'{name}-submit.log']

def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            path = Path(value['path'])
            actual = identity(path)
            assert all(actual[key] == value[key] for key in ['bytes', 'sha256'])
            external.add(path.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)

for name in names:
    for path in (N / name).glob('group*/**/start.json'): input_receipts(read(path))
input_receipts(inspection)
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('sharp-edge-shell-fields-v1', files, sorted(external), scope)
(DEST / 'sharp-edge-shell-fields-v1/README.md').write_text(
    '# Complete-domain single-edge hollow-shell fields\n\n' + scope +
    '\n\nSee [the physical control](../../../docs/SHARP_EDGE_SHELL_CONTROL_20261023.md). '
    'All campaigns are terminal, including failedv1 readouts and the cancelled pending inspector. '
    'Captured modules and source inventories retain their exact executed versions. '
    'Runtime arrays remain on data; original particular fields are reused in place. '
    'Scientific work used CPU Slurm rtx-mid/image1_26, the existing data-side Python and private node caches.\n')
with (META / 'edge-shell-fields-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
