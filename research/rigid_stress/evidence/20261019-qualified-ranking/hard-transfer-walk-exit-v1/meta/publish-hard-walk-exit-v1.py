"""Retain the declined walk early-exit experiment and its original GPU failure."""

from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-scene-walk-stop-v1', 'hard-walk-baseline-diagnostic-v1',
             'hard-walk-stop-diagnostic-v1', 'hard-walk-baseline-context-v1', 'hard-walk-stop-context-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(N / campaign / 'tasks.json'))):
        failure = campaign == 'hard-scene-walk-stop-v1' and group == 0
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if failure else ['COMPLETED', '0:0'])
        assert (N / campaign / f'group{group}/returncode.txt').read_text().strip() == ('1' if failure else '0')
for group in [2, 3]:
    for task in ['comparison', 'max-conformance', 'full-conformance']:
        assert read(N / f'hard-scene-walk-stop-v1/group{group}/{task}/result.json')['passed']
for name in campaigns[1:3]:
    result = read(N / name / 'group0/diagnostic/result.json')
    assert result['execution_completed'] and result['all_patches_admitted']
for name in campaigns[3:]:
    assert read(N / name / 'group0/transfer/result.json')['passed']
assert not (N / 'hard-scene-walk-stop-v1/group0/transfer/result.json').exists()
assert not (N / 'hard-scene-walk-stop-v1/group0/bridge').exists()
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
           'baseline_scene': [read(N / 'hard-scene-stable-v3/group4/comparison/result.json'), read(N / 'hard-scene-large-batch-v1/group1/comparison/result.json')],
           'early_exit_scene': [read(N / f'hard-scene-walk-stop-v1/group{group}/comparison/result.json') for group in [2, 3]],
           'original_gpu_microbenchmark_failed': True,
           'later_baseline_and_early_exit_standalone_and_complete_context_checks_passed': True,
           'original_failure_root_cause_resolved': False,
           'selected': False,
           'public_engine_alias': 'd4503b9e42a7f8f8555d128b437bb45b3193cf75',
           'scope': 'Declined explicit early exit in the existing projected boundary walk. B4/B1024 complete scenes preserve every-step statuses/peaks, exact rigid trajectories and saved max/full tensors under the existing numerical gates; no measured transfer acceleration is observed. The initial GPU component run completes seven geometric cases then fails the original homogeneous microbenchmark assertion before reporting timings. CPU controls pass. Separate original/early-exit standalone and complete-context GPU controls later pass without any budget change, leaving the original failure cause unresolved. Original failures, all captured variants and arrays remain retained. The subsequent inactive-sample bandwidth experiment is excluded.',
           'whole_goal_complete': False}
terminal_path = N / 'meta/hard-walk-exit-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
for campaign in campaigns:
    files.extend((N / campaign).glob('group*/**/*.jsonl.gz'))
    external.update((N / campaign).glob('group*/**/*.obj'))
large = {path for path in files if path.stat().st_size > 2 * 1024 * 1024}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-transfer-walk-exit-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Declined projected-walk early exit\n\n' + receipt['scope'] +
    '\n\nReplay each captured task group with its own captured workers, public engine alias `d4503b9e4`, the pinned scientific Python, `rtx-mid`, container `1_26`, and fresh explicit data-side outputs/private caches. Baseline and early-exit diagnostic/context campaigns intentionally use different captured transfer modules; retain those identities. The first failed GPU assertion is preserved and is not converted to a passing control by the later observations. Large full arrays, histories and generated meshes remain hashed on data. This implementation-only experiment changes no physical model or acceptance threshold and is not selected.\n')
with (N / 'meta/hard-walk-exit-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
