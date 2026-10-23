"""Freeze exact-column and singular-window integration, retaining earlier failures."""
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
names = list(read(META / 'edge-shell-integrals-launch-v4.json'))
cancelled = {'edge-shell-integrals-inspection-v1', 'edge-shell-integrals-inspection-v3'}
failed = {'edge-shell-exact-columns-v1', 'edge-shell-exact-columns-v2'}
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        if name in cancelled:
            actual = observed[f'{job}_{group}']
            assert actual[0].startswith('CANCELLED') and not (root / 'inspection/start.json').exists()
            states[name].append({'job': f'{job}_{group}', 'state': actual[0], 'exit_code': actual[1], 'no_scientific_execution': True})
            continue
        failed_group = name in failed or (name == 'edge-shell-volume-window-v2' and group == 0)
        expected = ['FAILED', '1:0'] if failed_group else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected
        assert (root / 'returncode.txt').read_text().strip() == ('1' if failed_group else '0')
        for task in tasks:
            if failed_group:
                assert not (root / task['name'] / 'result.json').exists()
            else:
                assert read(root / task['name'] / 'result.json')['execution_completed']
        states[name].append({'job': f'{job}_{group}', 'state': expected[0], 'exit_code': expected[1]})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
assert {row['recorded_revision'] for row in replayed} == {ALIAS}
start = read(META / 'edge-shell-exact-columns-start-v1.json')
for name, sha in start['frozen_heads'].items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in start['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
inspection = read(N / 'edge-shell-integrals-inspection-v4/group0/inspection/result.json')
previous_inspection = read(N / 'edge-shell-integrals-inspection-v2/group0/inspection/result.json')
assert inspection['execution_completed'] and not inspection['native_scene_accuracy_qualified']
for window in ['edge-shell-volume-window-v3/group0/window', 'edge-shell-volume-window-v2/group1/window']:
    window_start = read(N / window / 'start.json')
    mesh_receipt = window_start['inputs'][0]
    for original_group in range(2):
        assert mesh_receipt in read(N / f'edge-shell-fields-v2/group{original_group}/fields/start.json')['inputs']
bindings = []
for stem, campaign in [
    ('layered_column_integrals', 'edge-shell-exact-columns-v3'),
    ('study_edge_shell_exact_columns', 'edge-shell-exact-columns-v3'),
    ('edge_volume_window', 'edge-shell-volume-window-v3'),
    ('study_edge_volume_window', 'edge-shell-volume-window-v3'),
    ('inspect_edge_shell_integrals', 'edge-shell-integrals-inspection-v4'),
]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope = ('Same independent5mN sharp-edge hollow-shell control. Exact Cartesian tetrahedron intersections integrate '
    'the saved piecewise-linear regular stress through the original complete material-column banks. '
    'Two failed candidate-prism searches remain intact; the existing full-mesh interval query resolves14 columns '
    'at unchanged coordinates and tolerances. A compact smooth local subtraction identity retains the entire volume '
    'while polar quadrature resolves particular-field energy near the singular edge. Two independent quadrature '
    'orders and both original global partitions retain separate convergence and physical-reference gates. '
    'No new elastic solve, stress law, native engine, solver tolerance or throughput claim. '
    'Observed variation sums are not certified continuum bounds; whole-goal and native-scene accuracy remain open.')
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'unchanged_engine_alias': ALIAS,
    'frozen_heads': start['frozen_heads'], 'preserved_files': preserved,
    'source_bindings': bindings, 'inspection': inspection, 'unaligned_window_inspection': previous_inspection,
    'scope': scope, 'whole_goal_complete': False}
p = META / 'edge-shell-integrals-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect([name for name in names if name not in cancelled])
for name in sorted(cancelled): files += sources(name) + terminal(name)
files += [p, Path(__file__), HELPER_SOURCE, REPLAY_HELPER, META / 'edge-shell-exact-columns-start-v1.json',
          META / 'edge-shell-exact-columns-storage-v1.txt', META / 'edge-shell-integrals-launch-v1.json',
          META / 'edge-shell-integrals-launch-v2.json', META / 'edge-shell-integrals-launch-v3.json',
          META / 'edge-shell-integrals-inspection-v1-cancel.json', META / 'edge-shell-integrals-inspection-v3-cancel.json',
          META / 'edge-shell-integrals-launch-v4.json']
for name in names:
    files += [META / f'{name}-tasks.json', META / f'{name}-submit.log']
checked_inputs = {}
def input_receipts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= value.keys():
            path = Path(value['path']); key = path.resolve()
            if key not in checked_inputs: checked_inputs[key] = identity(path)
            actual = checked_inputs[key]
            assert all(actual[key] == value[key] for key in ['bytes', 'sha256'])
            external.add(path.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)
for name in names:
    for path in (N / name).glob('group*/**/start.json'): input_receipts(read(path))
input_receipts(inspection)
files += [META / 'document-edge-integrals-and-batch2048-v1.py', META / 'verify-edge-shell-integrals-delivery-v1.py']
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('sharp-edge-shell-integrals-v1', files, sorted(external), scope)
(DEST / 'sharp-edge-shell-integrals-v1/README.md').write_text(
    '# Sharp-edge complete-domain integral refinement\n\n' + scope +
    '\n\nSee [the physical control](../../../docs/SHARP_EDGE_SHELL_CONTROL_20261023.md). '
    'Every campaign is terminal, including both failed column variants, the rejected low-order window and two cancelled pending inspectors. Scientific work used CPU Slurm '
    'rtx-mid/image1_26, the existing data-side Python and private node caches. All55 source heads '
    'and12 original unfinished files remain preserved; runtime arrays stay on data.\n')
with (META / 'edge-shell-integrals-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
