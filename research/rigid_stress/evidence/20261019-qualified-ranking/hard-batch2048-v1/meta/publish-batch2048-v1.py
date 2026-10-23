"""Freeze matched native B1024/B2048 timing and independently sampled memory."""
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
names = ['hard-batch-scaling2048-v1', 'hard-batch-scaling2048-inspection-v1']
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
            assert read(root / task['name'] / 'result.json')['passed']
        states[name].append({'job': f'{job}_{group}', 'state': 'COMPLETED', 'exit_code': '0:0'})
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
inspection = read(N / names[1] / 'group0/inspection/result.json')
assert inspection['execution_completed'] and inspection['passed']
assert inspection['batches'] == [1024, 2048] and inspection['independent_initial_conditions'] == 4
assert len(inspection['primary_rows']) == 3
bindings = []
for stem, campaign in [
    ('time_hard_franka', names[0]), ('memory_hard_franka', names[0]),
    ('inspect_hard_batch_scaling', names[1]),
]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{stem}.py')
    saved = N / campaign / 'analysis-source' / relative
    assert saved.read_bytes() == (WORK / relative).read_bytes()
    bindings.append({'captured': identity(saved), 'current': identity(WORK / relative), 'exact_match': True})
scope = ('Same-allocation B1024/B2048 rigid/max/full moving-scene timing with unchanged native engine, '
    '64-bit precision, surface order16 and the same four-condition bank. Three continuous100..129 windows '
    'retain runtime admission. Independent saved-array checks cover all130 audited frames/environments '
    'across batches, within replicated condition banks and against separate fresh-process full memory sampling. '
    'GPU polls include non-Torch allocations but cannot certify a continuous peak or capacity limit. '
    'No independently diverse population, best batch, full-cycle admission or physical accuracy claim. '
    'The original frame428 refusal and all previous baselines remain unchanged.')
receipt = {'passed': True, 'campaign_states': states, 'actual_scheduler': accounting,
    'captured_source_replay': replayed, 'unchanged_engine_alias': ALIAS,
    'frozen_heads': start['frozen_heads'], 'preserved_files': preserved,
    'source_bindings': bindings, 'inspection': inspection,
    'scope': scope, 'whole_goal_complete': False}
p = META / 'batch2048-terminal-v1.json'
with p.open('x') as stream:
    json.dump(receipt, stream, indent=2); stream.write('\n')
files, external = collect(names)
files += [p, Path(__file__), HELPER_SOURCE, REPLAY_HELPER,
    META / 'edge-shell-exact-columns-start-v1.json', META / 'edge-shell-exact-columns-storage-v1.txt',
    META / 'hard-batch-scaling2048-v1-memory-full-task.json']
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
            if not path.resolve().is_relative_to(WORK):
                external.add(path.resolve())
        else:
            for item in value.values(): input_receipts(item)
    elif isinstance(value, list):
        for item in value: input_receipts(item)
input_receipts(receipt)
for name in names:
    for path in (N / name).glob('group*/**/start.json'): input_receipts(read(path))
files += [META / 'document-edge-integrals-and-batch2048-v1.py', META / 'verify-edge-shell-integrals-delivery-v1.py']
files = sorted(set(files))
large = {path for path in files if path.stat().st_size > 2 * 1024**2}
files = [path for path in files if path not in large]
external.update(large)
external = {path.resolve() for path in external} - {path.resolve() for path in files}
packet = freeze('hard-batch2048-v1', files, sorted(external), scope)
(DEST / 'hard-batch2048-v1/README.md').write_text(
    '# Native B1024/B2048 throughput and sampled memory\n\n' + scope +
    '\n\nSee [the measurements](../../../docs/SHARP_EDGE_PRESSURE_20261022.md#b2048-throughput-and-sampled-memory). '
    'All campaigns are terminal. Slurm rtx-mid/image1_26, the existing data-side Python and private node caches '
    'were used. All55 engine heads and12 original unfinished files remain preserved. '
    'Runtime arrays stay on data; no engine, tolerance, geometry or material change was introduced.\n')
with (META / 'batch2048-publication-v1.json').open('x') as stream:
    json.dump(packet, stream, indent=2); stream.write('\n')
print(json.dumps(packet))
