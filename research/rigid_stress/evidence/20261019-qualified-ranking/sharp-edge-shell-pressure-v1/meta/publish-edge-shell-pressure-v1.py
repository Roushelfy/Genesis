"""Freeze the independent single-edge hollow-shell pressure control."""
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
names = ['edge-shell-pressure-pilot-v1', 'edge-shell-pressure-refinement-v1', 'edge-shell-pressure-inspection-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in names]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
states = {}
for name, job in zip(names, jobs):
    states[name] = []
    for group, tasks in enumerate(read(N / name / 'tasks.json')):
        root = N / name / f'group{group}'
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
        assert (root / 'returncode.txt').read_text().strip() == '0'
        for task in tasks:
            assert read(root / task['name'] / 'result.json')['execution_completed']
        states[name].append({'job': f'{job}_{group}', 'state': 'COMPLETED', 'exit_code': '0:0'})
REPLAY_HELPER = N / 'meta/publish-live-hertz-normal-v1.py'
replay_code = '# Replay all primary' + REPLAY_HELPER.read_text().split('# Replay all primary', 1)[1].split('\nmeasured_alias = ', 1)[0]
exec(compile(replay_code, str(REPLAY_HELPER), 'exec'))
assert {row['recorded_revision'] for row in replayed} == {ALIAS}
start = read(META / 'edge-shell-pressure-start-v1.json')
for name, sha in start['frozen_heads'].items():
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK.parent / name, text=True).strip() == sha
preserved = 0
for name, data in start['preserved'].items():
    for relative, sha in data['files'].items():
        assert hashlib.sha256((WORK.parent / name / relative).read_bytes()).hexdigest() == sha
        preserved += 1
inspection = read(N / names[-1] / 'group0/inspection/result.json')
assert inspection['reference_variation_targets_met'] and inspection['sampled_error_budget_met']
assert not inspection['independent_complete_stress_qualified'] and not inspection['global_peak_qualified']
bindings = []
for stem, campaign in [('study_edge_shell_pressure', names[0]), ('study_edge_shell_pressure', names[1]),
                       ('inspect_edge_shell_pressure', names[2])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope = ('Independent pressure/support for a 5mN frictionless sharp edge on the existing 500um-wall hollow polar land. '
    'Fixed slope0.02 and along curvature80/m, fixed orientation and output reaction moments. '
    'Three body meshes,36/48 cell grids and3/5 source tables retain force/gap/complementarity checks. '
    'Coarse24 pilot remains separate. A saved-array inspection verifies fields and a third integration order. '
    'Fine cell-pressure error4.1846%; coarse-candidate/fine-reference sampled tensor/tensile/VM errors1.5313/2.6961/1.9876%. '
    'All140 samples are7.687--61.497um deep. The growing edge cell-pressure maximum is not a finite surface peak. '
    'Complete-volume, membrane/bending, global peaks, changing friction, native configuration matching and full-cycle '
    'scene admission remain open. No engine, numerical acceptance or runtime pressure solve changes.')
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'unchanged_engine_alias': ALIAS,
    'frozen_heads': start['frozen_heads'], 'preserved_files': preserved,
    'source_bindings': bindings, 'inspection': inspection, 'scope': scope, 'whole_goal_complete': False}
p = META / 'edge-shell-pressure-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, HELPER_SOURCE, REPLAY_HELPER, META / 'edge-shell-pressure-start-v1.json',
          META / 'edge-shell-pressure-storage-v1.txt']
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
packet = freeze('sharp-edge-shell-pressure-v1', files, sorted(external), scope)
(DEST / 'sharp-edge-shell-pressure-v1/README.md').write_text(
    '# Single-edge hollow-shell pressure control\n\n' + scope +
    '\n\nSee [the physical control](../../../docs/SHARP_EDGE_SHELL_CONTROL_20261023.md). '
    'All three campaigns are terminal. Captured modules and source inventories retain exact executed versions. '
    'Runtime arrays remain on data. All55 engine heads and12 original unfinished files remain preserved. '
    'Scientific work used CPU Slurm rtx-mid/image1_26, the existing data-side Python and private node caches.\n')
with (META / 'edge-shell-pressure-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
