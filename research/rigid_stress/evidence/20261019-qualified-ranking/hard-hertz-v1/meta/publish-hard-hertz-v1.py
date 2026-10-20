"""Retain executed cheap hard-contact comparisons and their actual scope."""

import hashlib
import json
from pathlib import Path
import subprocess

RUN = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = RUN / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))

campaigns = ['hard-hertz-v1', 'hard-hertz-interaction-v1', 'hard-hertz-figure-v1']
jobs = [read(RUN / name / 'submission.json')['job'] for name in campaigns]
accounting = subprocess.check_output(
    ['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True
)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
        assert (root / f'group{group}/returncode.txt').read_text().strip() == '0'
        for task in tasks:
            assert read(root / f'group{group}' / task['name'] / 'result.json')['passed']

working = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking')
matched = []
for module, campaign in (
    ('native_hertz', campaigns[0]), ('hertz_reference', campaigns[0]),
    ('check_native_hertz', campaigns[0]), ('study_hard_hertz', campaigns[0]),
    ('check_hertz_interaction', campaigns[1]), ('plot_hard_hertz', campaigns[2]),
):
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    snapshot_path = RUN / campaign / 'analysis-source' / relative
    assert snapshot_path.read_bytes() == (working / relative).read_bytes()
    matched.append({'module': module, 'sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest()})

receipt = {
    'passed': True, 'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting, 'current_sources_match_executed': matched,
    'native_cases_per_backend': 101, 'independent_bem_cases': 6, 'interaction_reused_references': 2,
    'native_gpu_cost_scope': 'Resident supplied-region curvature/force-to-ellipse lookup; geometry, grouping, pressure scatter, elasticity and scene excluded.',
    'physical_scope': 'Unilateral normal hard contact in an elastic half-space. Sampled subsurface tensors exclude surface/global peaks; no shell or friction claim.',
    'whole_goal_complete': False,
    'next': 'Cached current-region curvature and witnessed-boundary transfer for the cheap native estimator, matched shell/friction controls and actual scene throughput. No further soft-pad or dense elastic-basis expansion.',
}
terminal_path = RUN / 'meta/hard-hertz-terminal-v1.json'
assert read(terminal_path) == receipt
files, external = collect(campaigns)
files.extend([RUN / 'meta/publish-hard-hertz-first-attempt-v1.py', RUN / 'meta/hard-hertz-publication-first-attempt-v1.txt', HELPER_SOURCE, terminal_path, RUN / 'meta/hard-hertz-visual-review-v1.json', RUN / 'meta/hard-hertz-nature-style-v1.py'])
figure = RUN / campaigns[2] / 'group0/figure'
files.extend(figure / name for name in ('hard-hertz.svg', 'hard-hertz.pdf', 'hard-hertz.png', 'preview.png'))
scope = (
    'Cheap hard-contact results under the superseding remote goal75d534428. '
    'Six independent unilateral square-cell Boussinesq contact solves on two grids; circular/rotated elliptical/two-region cases. '
    'Two reused-reference far-field correction comparisons. Native Quadrants elliptic lookup validates101 cases on each CPU/GPU backend, '
    'with explicit invalid/unloaded/unsupported/isolation controls and resident lookup timing for1/128/512/8192 supplied physical regions. '
    'Inspected six-panel pressure maps and editable exports retained. These are half-space normal-contact components, '
    'not hollow-shell/friction/global-peak qualification, automatic source grouping, production lifecycle or actual scene throughput. '
    'Prior pad results remain unchanged historical evidence; no new pad or dense elastic-basis campaign is required.'
)
packet = freeze('hard-hertz-v1', files, sorted(external), scope)
(DEST / packet['packet'] / 'README.md').write_text(
    '# Cheap hard-contact estimates\n\n' + scope + '\n\n'
    'The manifest records exact encoded/original payload hashes and external array identities. '
    'Executed workers, task arguments, engine inventory, logs, terminal accounting and visual inspection are retained. '
    'Full numerical arrays stay on data and private caches use compute-node scratch.\n'
)
print(json.dumps(packet, indent=2))
