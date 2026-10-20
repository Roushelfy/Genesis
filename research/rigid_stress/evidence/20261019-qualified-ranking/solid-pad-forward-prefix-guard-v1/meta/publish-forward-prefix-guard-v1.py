"""Freeze the actual old/new forward-prefix counterexample without live fine outputs."""
import ast
import datetime
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = RUN / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
exec(compile(HELPER.read_text().split('\nimport ast\n', 1)[0], str(HELPER), 'exec'))
read = lambda path: json.loads(path.read_text())
campaign = 'solid-pad-forward-prefix-controls-v1'
root = RUN / campaign
result_path = root / 'group0/controls/result.json'
result = read(result_path)
assert result['passed'] and len(result['checks']) == 20 and all(result['checks'].values())
assert (root / 'group0/returncode.txt').read_text().strip() == '0'
job = str(read(root / 'submission.json')['job']).split(';')[0]
accounting = subprocess.check_output(['sacct', '-j', job + '_0', '-P', '--noheader',
                                      '--format=JobID,State,ExitCode,Elapsed,End'], text=True)
assert any(line.split('|')[:3] == [job + '_0', 'COMPLETED', '0:0'] for line in accounting.splitlines())
for receipt in result['receipts']:
    assert identity(pathlib.Path(receipt['path'])) == receipt
inspection = read(META / 'forward-prefix-source-inspection-v1.json')
assert inspection['passed'] and inspection['restoration_function_AST_unchanged']
refusal = read(RUN / 'solid-pad-inherited-endpoint-refusal-v1/result.json')
assert not refusal['passed'] and 'does not advance' in refusal['error'] and 'ancestor' in refusal['error']

inventory = {row['path']: row for row in read(root / 'analysis-files.json')}
queue = [task['module'] for group in read(root / 'tasks.json') for task in group]
seen = set()
while queue:
    module = queue.pop()
    relative = pathlib.Path(*module.split('.')).with_suffix('.py')
    path = root / 'analysis-source' / relative
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

files = [pathlib.Path(__file__), HELPER, META / 'forward-prefix-source-inspection-v1.json',
         META / 'solid-pad-forward-prefix-controls-v1-tasks.json',
         RUN / 'solid-pad-inherited-endpoint-refusal-v1/result.json']
files += captured(campaign, [str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in seen])
files += terminal(campaign)
# Bind and retain the exact pre-fix module that accepts the real-data counterexample.
legacy = pathlib.Path(next(row['path'] for row in result['receipts']
                           if '/solid-pad-step225-recovery-control-v1/analysis-source/' in row['path']))
files.append(legacy)
external = set()
for name in ['solid-pad-step225-preparation-v1', 'solid-pad-step300-preparation-v1',
             'solid-pad-resume-preparation-v1', 'solid-pad-stvk-canonical125-final-resume-preparation-v1',
             'solid-pad-stvk-canonical125-contact-resume-preparation-v1',
             'solid-pad-stvk-canonical125-halfdt-resume-preparation-v1']:
    path = RUN / name / 'result.json'
    files.append(path)
    proof = read(path)
    for receipt in proof['receipts']:
        assert identity(pathlib.Path(receipt['path'])) == receipt
        if receipt['path'].endswith('.npz'):
            external.add(pathlib.Path(receipt['path']))
receipt = {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'actual_scheduler_accounting': accounting, 'controls': result['checks'],
    'actual_old_validator_and_latest_selector_accept_copied200_endpoint': True,
    'new_preparation_and_validation_refuse_before_model_setup': True,
    'all_six_original_or_advancing_prefix_receipts_still_pass': True,
    'driver_ten_physics_modules_and_restoration_arithmetic_unchanged': True,
    'scope': 'Align early metadata admission with the existing strictly advancing ancestry requirement. If a terminal intermediate attempt adds no new saved state, reuse the terminal ancestor receipt; do not build another expensive model from a duplicate inherited prefix.',
    'no_live_fine_source_or_output_changed': True,
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
}
path = META / 'forward-prefix-guard-terminal-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(path)
scope = ('Actual old/new metadata regression731307 passes20 checks. The exact old validator and latest selector accept a copied200-step endpoint from a parent that already inherited those200 steps; the unchanged final ancestry gate would reject that nonadvancing link after expensive integration. '
         'Preparation and validation now reject the same link before model setup and direct recovery to its terminal ancestor receipt. Actual terminal preparation refusal, root200, advancing225/300, all three original fine200/200/400 receipts and valid explicit/latest selectors are verified. '
         'The restoration function AST, physical driver and ten numerical modules remain unchanged; no FEM trajectory or numerical conformance sweep is rerun. '
         'No live fine snapshot/output is changed or published. This is recovery metadata admission only, not fine-reference qualification, final ranking or goal completion.')
row = freeze('solid-pad-forward-prefix-guard-v1', files, sorted(external), scope)
with (META / 'forward-prefix-guard-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
