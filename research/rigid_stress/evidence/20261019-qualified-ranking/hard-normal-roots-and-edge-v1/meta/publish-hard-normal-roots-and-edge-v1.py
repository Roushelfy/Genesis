"""Retain direct planar roots, matched live cost and finite-edge pressure controls."""
from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
H = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/cheap_hard_contact')
campaigns = ['hard-normal-roots-v1', 'hard-normal-live-v1', 'hard-edge-v1', 'hard-edge-v2']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0'], (campaign, observed[f'{job}_{group}'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == '0'
        for task in tasks:
            result = read(root / f'group{group}' / task['name'] / 'result.json')
            assert result.get('passed', result.get('execution_completed', False))

matched = []
for module, campaign in [('check_hertz_regions', 'hard-normal-roots-v1'), ('live_hertz_inputs', 'hard-normal-live-v1'), ('compare_live_hertz_inputs', 'hard-normal-live-v1'), ('study_hard_edge', 'hard-edge-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    executed = RUN / campaign / 'analysis-source' / relative
    assert executed.read_bytes() == (WORK / relative).read_bytes()
    matched.append(identity(executed))
for relative in ['genesis/engine/solvers/rigid/stress/curvature.py', 'genesis/engine/solvers/rigid/stress/hertz_regions.py', 'genesis/engine/solvers/rigid/stress/normal_roots.py']:
    current = identity(H / relative)
    for campaign in ['hard-normal-roots-v1', 'hard-normal-live-v1']:
        expected = next(row for row in read(RUN / campaign / 'source.json')['files'] if row['path'] == relative)
        assert all(current[key] == expected[key] for key in ['bytes', 'sha256'])
    matched.append(current)

controls = [read(RUN / f'hard-normal-roots-v1/group{i}/regions/result.json') for i in range(2)]
for control in controls:
    assert control['passed'] and len(control['region_cases']) == 12 and len(control['curvature']) == 4
    assert max(row['normal_root_forward_residual_max'] for row in control['curvature']) < 1e-12
live = read(RUN / 'hard-normal-live-v1/group0/comparison/result.json')
assert live['motion_matches'] and live['source_status_counts'] == [3020, 0, 6873, 0, 0, 0]
assert sum(live['valid_loaded_environment_steps']) == 1381
assert sum(live['loaded_environment_steps']) == 2808
assert max(live['sampled_moment_over_force_major_axis']) > 100
edge = read(RUN / 'hard-edge-v2/group0/study/result.json')
assert edge['passed'] and len(edge['rows']) == 8
for row in edge['rows']:
    assert row['variants']['mixed_ritz']['engineering_targets_met']
    assert not row['variants']['load_consistent_hertz_cap']['engineering_targets_met']
    assert row['cap_quadrature_force_relative_before_normalization'] < 1e-5
    for fit in row['fitting'].values():
        assert not fit['at_search_boundary']
        assert fit['quadrature_20_to_40_force_l2_relative'] < 1e-4

receipt = {
    'passed': True,
    'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting,
    'current_sources_match_executed': matched,
    'controls': controls,
    'live_comparison': live,
    'edge_corrected': edge,
    'original_edge_diagnostic': identity(RUN / 'hard-edge-v1/group0/study/result.json'),
    'original_edge_limit': 'At d/a0=2 the original Hertz cap exceeds the y search domain and is clipped before normalization. Original ordinary Gauss quadrature also under-resolves the edge trial singularity. Retain the diagnostic, but supersede its provisional pressure-target judgments with the expanded-domain transformed-quadrature run.',
    'whole_goal_complete': False,
    'scope': 'Native direct inversion of fixed mesh normal fields for planar opponents preserves twelve CPU/GPU source-region controls and passes128 forward-normal queries per backend. Actual matched Franka trajectories remain identical; input cost falls to4.025ms versus2.249ms pure rigid, with1381/2808 loaded environment steps passing geometric checks. Finite-face rejection, coarse geometry error and large incompatible source moments remain explicit. A separate mixed regular/edge variational pressure trial passes eight predeclared cell-pressure/subsurface-sample targets; offline fitting is not native runtime, shell accuracy or continuum-peak qualification.',
    'next': 'Continue cheap native estimation with explicit wrench/pressure compatibility, finite-face and source allocation limits, tangential history and finite-shell stress. The straight-edge mixed trial merits a bounded dimensionless surrogate study, not runtime BEM minimization or broad geometry tables without evidence. Actual useful max/full scene throughput remains open; no default coupled Newton-PCG or pad expansion.',
}
terminal_path = RUN / 'meta/hard-normal-roots-and-edge-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
for campaign in campaigns:
    root = RUN / campaign
    files.extend(root.glob('group*/**/*.jsonl.gz'))
    external.update(root.glob('group*/**/*.obj'))
packet = freeze('hard-normal-roots-and-edge-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text('# Direct normal roots and finite-edge pressure controls\n\n' + receipt['scope'] + '\n\nThe manifest retains exact executed sources, terminal scheduler state, corrected and original diagnostic results, and compact moving-scene histories. Full arrays and generated geometry remain hashed on data. Private compiler caches were node-local. The original finite-edge trial-domain and quadrature limitations remain recorded; only the final corrected study supports its stated cell/sample comparisons.\n')
with (RUN / 'meta/hard-normal-roots-and-edge-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
