"""Retain native curvature/source controls and rejected live region throughput."""
from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
H = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cheap_hard_contact')
campaigns = [f'hard-hertz-inputs-v{i}' for i in range(1, 7)] + [f'hard-hertz-live-v{i}' for i in range(1, 5)]
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
failed = {'hard-hertz-inputs-v1', 'hard-hertz-inputs-v2', 'hard-hertz-inputs-v3', 'hard-hertz-live-v1'}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    for group, tasks in enumerate(read(root / 'tasks.json')):
        expected = ['FAILED', '1:0'] if campaign in failed else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected, (campaign, observed[f'{job}_{group}'])
        assert ((root / f'group{group}/returncode.txt').read_text().strip() != '0') == (campaign in failed)
        if campaign not in failed:
            for task in tasks:
                result = read(root / f'group{group}' / task['name'] / 'result.json')
                assert result.get('passed', result.get('execution_completed', False))

matched = []
for module, campaign in [('check_hertz_regions', 'hard-hertz-inputs-v6'), ('live_hertz_inputs', 'hard-hertz-live-v4'), ('compare_live_hertz_inputs', 'hard-hertz-live-v4')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    executed = RUN / campaign / 'analysis-source' / relative
    assert executed.read_bytes() == (WORK / relative).read_bytes()
    matched.append(identity(executed))
for relative in ['genesis/engine/solvers/rigid/stress/curvature.py', 'genesis/engine/solvers/rigid/stress/hertz_regions.py']:
    current = identity(H / relative)
    for campaign in ['hard-hertz-inputs-v6', 'hard-hertz-live-v4']:
        expected = next(row for row in read(RUN / campaign / 'source.json')['files'] if row['path'] == relative)
        assert all(current[key] == expected[key] for key in ['bytes', 'sha256'])
    matched.append(current)
control = [read(RUN / f'hard-hertz-inputs-v6/group{i}/regions/result.json') for i in range(2)]
assert all(row['passed'] and len(row['region_cases']) == 12 for row in control)
live = read(RUN / 'hard-hertz-live-v4/group0/comparison/result.json')
assert live['motion_matches']
assert live['source_status_counts'] == [887, 0, 8006, 0, 1000, 0]
assert live['invalid_environment_steps'] == [489, 477, 480, 480]
receipt = {
    'passed': True,
    'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting,
    'current_sources_match_executed': matched,
    'controls': control,
    'live_comparison': live,
    'before_opposing_box_correction': read(RUN / 'hard-hertz-live-v3/group0/comparison/result.json'),
    'whole_goal_complete': False,
    'scope': 'Primitive analytical curvature and explicit smooth-normal mesh estimates; local geometric-minimum source coalescing with exact force/moment preservation. Twelve CPU/GPU controls pass, including disconnected minima, splitting/order/side invariance, rotation, release, invalid-source quarantine recovery and a one-millimeter sphere/box overlap. Actual changing Franka scenes preserve all trajectories exactly but reject most loaded environment steps; measured input-stage cost is not admitted usable stress throughput. No public stress integration, finite-shell pressure qualification, friction-history law or max/full scene claim.',
    'next': 'Use the retained same-substep evidence to resolve finite-face/link-union geometry and coarse normal-field convergence, and reduce input cost before integrating the useful cheap estimator. The opposing-box correction passes its analytic overlap control but does not improve loaded scene-step admission. Keep mesh smoothness, local-minimum convergence and actual wrench/pressure compatibility explicit. Continue finite-shell and tangential accuracy separately; no default coupled Newton-PCG or pad expansion.',
}
terminal_path = RUN / 'meta/hard-hertz-inputs-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
for campaign in campaigns:
    root = RUN / campaign
    files.extend(root.glob('group*/**/*.jsonl.gz'))
    external.update(root.glob('group*/**/*.obj'))
packet = freeze('hard-hertz-inputs-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text('# Current curvature and live hard-contact inputs\n\n' + receipt['scope'] + '\n\nThe manifest retains exact captured source variants, original failures, terminal scheduler state and compact histories. Full control/trajectory arrays and generated geometry remain hashed on data. Runtime caches were node-local.\n')
with (RUN / 'meta/hard-hertz-inputs-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
