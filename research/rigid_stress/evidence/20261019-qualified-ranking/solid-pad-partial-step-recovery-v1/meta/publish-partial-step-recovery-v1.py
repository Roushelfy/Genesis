"""Freeze terminal partial-level recovery controls; never capture live fine outputs."""
import ast
import datetime
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = RUN / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
# Reuse the already published deterministic encoding/manifest implementation only.
# The mutating campaign-specific body of that publisher is never executed.
prefix = HELPER.read_text().split('\nimport ast\n', 1)[0]
exec(compile(prefix, str(HELPER), 'exec'))
read = lambda path: json.loads(path.read_text())
campaigns = ['solid-pad-step300-recovery-control-v1', 'solid-pad-step225-recovery-control-v1']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(
    ['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader', '--format=JobID,State,ExitCode,Elapsed,End'], text=True)
observed = {parts[0]: (parts[1].split()[0], parts[2]) for line in accounting.splitlines()
            if len(parts := line.split('|')) >= 3}
files = [pathlib.Path(__file__), HELPER]
external = set()
controls = {}
for campaign, job, inherited in zip(campaigns, jobs, [300, 225]):
    root = RUN / campaign
    groups = read(root / 'tasks.json')
    for index, tasks in enumerate(groups):
        assert observed[f'{job}_{index}'] == ('COMPLETED', '0:0'), (job, observed)
        assert (root / f'group{index}/returncode.txt').read_text().strip() == '0'
        for task in tasks:
            path = root / f'group{index}' / task['name'] / 'result.json'
            result = read(path)
            if task['name'] in ('reference', 'equivalence-level1'):
                continue
            if task['name'] == 'moment':
                assert result['passed'] and result['history_admitted'] and len(result['frames']) == 2
                continue
            assert result['passed'] and result['checks'] and all(result['checks'].values()), str(path)
            controls[f'{campaign}/group{index}/{task["name"]}'] = len(result['checks'])
            for receipt in result.get('receipts', []):
                assert identity(pathlib.Path(receipt['path'])) == receipt, receipt
                if receipt['path'].endswith('.npz'):
                    external.add(pathlib.Path(receipt['path']))
    result = read(root / 'group0/reference/result.json')
    assert result['completed'] and result['mechanical_history_passed'] and result['steps'] == 400
    assert result['inherited_steps'] == inherited and result['newly_executed_steps'] == 400 - inherited
    lineage = read(root / 'group0/lineage-controls/result.json')['actual_coarse_lineage']
    assert len(lineage['ancestry']) == 2
    assert [row['inherited_steps'] for row in lineage['ancestry']] == [inherited, 200]
    source = root / 'analysis-source'
    inventory = {row['path']: row for row in read(root / 'analysis-files.json')}
    queue = [task['module'] for group in groups for task in group]
    seen = set()
    while queue:
        module = queue.pop()
        relative = pathlib.Path(*module.split('.')).with_suffix('.py')
        path = source / relative
        if str(relative) in seen or not path.is_file():
            continue
        seen.add(str(relative))
        actual = identity(path)
        assert all(actual[key] == inventory[str(relative)][key] for key in ['bytes', 'sha256'])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                base = '.'.join(module.split('.')[:-node.level]) if node.level else ''
                target = '.'.join(part for part in [base, node.module] if part)
                if target.startswith('research.rigid_stress'):
                    queue.append(target)
                    queue.extend(target + '.' + item.name for item in node.names)
            elif isinstance(node, ast.Import):
                queue.extend(item.name for item in node.names if item.name.startswith('research.rigid_stress'))
    files += captured(campaign, [str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in seen])
    files += terminal(campaign)
    files.append(root / 'group0/reference/steps.jsonl')
    files.append(root / 'group0/step-selection-controls/selection-fixture/checkpoints/step399.npz')
    for step in (375, 400):
        external.add(RUN / 'solid-pad-periodic-checkpoint-control-v1/group0/reference/checkpoints' / f'step{step}.npz')

for step in (300, 225):
    files += [RUN / f'solid-pad-step{step}-preparation-v1/result.json',
              META / f'solid-pad-step{step}-recovery-contract-v1.json',
              META / f'solid-pad-step{step}-recovery-tasks-v1.json']
files.append(META / 'step300-physics-and-captures-unchanged-v1.json')
assert read(files[-1])['passed']
old = RUN / 'solid-pad-periodic-state-reinspection-v1/group0/admission/result.json'
assert read(old)['passed'] and (RUN / 'solid-pad-periodic-checkpoint-control-v1/group0/returncode.txt').read_text().strip() == '1'
for receipt in read(old)['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
fine_jobs = ['730978_0', '730979_0', '730980_0']
current = subprocess.check_output(
    ['sacct', '-j', ','.join(fine_jobs), '-P', '--noheader', '--format=JobID,State,ExitCode,Elapsed,End'], text=True)
receipt = {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'accounting': accounting, 'controls': controls, 'fine_job_accounting_at_publication': current,
    'actual_continuations': [{'inherited_steps': 300, 'new_steps': 100}, {'inherited_steps': 225, 'new_steps': 175}],
    'all_prior_numeric_targets_preserved': True, 'original_failed_periodic_reader_preserved': True,
    'live_fine_sources_and_outputs_unchanged': True,
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
}
path = META / 'partial-step-recovery-terminal-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(path)
scope = ('Actual terminal step300 and loading-phase step225 full-state recovery controls complete400 steps as300+100 and225+175. '
         'Both retain original mechanics, both equilibrium windows, own current-momentum audits and unchanged1e-8 complete-field/point-force/raw/common-peak limits and1e-10m/s velocity limit against the uninterrupted400-step baseline. '
         'Explicit saved-state/source/clock/plate binding, damaged newest-write fallback, final-state refusal, legacy endpoint compatibility and two-link lineage with ancestor/cycle rejection are exercised. '
         'Recursive ancestry admits only the exact available prefix of a terminal intermediate parent; it does not invent a completed parent result. '
         'Only algorithmic caches are rebuilt; driver and ten physics modules are byte unchanged. Original JSON-reader failure and all live fine captures remain intact. '
         'No fine reference, final physical rank or final artwork is admitted here. Existing fine continuations and their strict downstream queue remain unchanged.')
row = freeze('solid-pad-partial-step-recovery-v1', files, sorted(external), scope)
path = META / 'partial-step-recovery-publication-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
