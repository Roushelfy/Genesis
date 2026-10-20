"""Publish executed elliptic near-field controls, matching cost and finite-shell limits."""

import json
from pathlib import Path
import subprocess

RUN = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = RUN / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))

campaigns = [f'hard-hertz-nearfield-v{i}' for i in range(1, 4)]
campaigns += [f'hard-hertz-native-stress-v{i}' for i in range(1, 6)]
failed = {'hard-hertz-nearfield-v1', 'hard-hertz-native-stress-v1', 'hard-hertz-native-stress-v2'}
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(
    ['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList'],
    text=True,
)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    code = '1' if campaign in failed else '0'
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if code == '1' else ['COMPLETED', '0:0'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == code
        if code == '0':
            for task in tasks:
                assert read(root / f'group{group}' / task['name'] / 'result.json')['passed']

matched = []
for module, campaign in (
    ('hertz_stress', 'hard-hertz-nearfield-v3'),
    ('study_hertz_nearfield', 'hard-hertz-nearfield-v3'),
    ('native_hertz_stress', 'hard-hertz-native-stress-v5'),
    ('check_native_hertz_stress', 'hard-hertz-native-stress-v5'),
):
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    snapshot_path = RUN / campaign / 'analysis-source' / relative
    assert snapshot_path.read_bytes() == (WORK / relative).read_bytes()
    matched.append({'module': module, 'sha256': identity(snapshot_path)['sha256']})
assert (RUN / 'hard-hertz-native-stress-v5/analysis-source/research/rigid_stress/qipc_comparison/hertz_stress.py').read_bytes() == (
    WORK / 'research/rigid_stress/qipc_comparison/hertz_stress.py'
).read_bytes()

study = read(RUN / 'hard-hertz-nearfield-v3/group0/study/result.json')
assert study['passed'] and not study['whole_goal_complete']
assert all(study['checks'].values())
assert [row['axis_ratio'] for row in study['rows']] == [1, 0.73, 0.25, 0.05]
for row in study['rows']:
    assert row['independent_pressure_integration_relative_errors'][-1] < 4.60e-7
    assert identity(Path(row['arrays']['path'])) == row['arrays']
native = [read(RUN / f'hard-hertz-native-stress-v5/group{i}/check/result.json') for i in range(3)]
assert [(row['backend'], row['quadrature_order']) for row in native] == [('gpu', 16), ('cpu', 16), ('gpu', 48)]
for row in native:
    assert row['passed'] and row['release_invalid_input_isolation_and_recovery']
    assert row['regions'] == 37 and row['probes_per_region'] == 77
    assert row['maximum_implementation_tensor_difference_over_peak_pressure'] < 1.23e-8
    assert row['maximum_high_order_tensor_difference_over_peak_pressure'] < 0.0001805
    assert row['maximum_far_field_tensor_relative_difference'] < 1.01e-8
    assert [cost['regions'] for cost in row['benchmark']] == [1, 128, 512]
    assert identity(Path(row['arrays']['path'])) == row['arrays']

gap = study['coarse_scale_gap']
assert identity(Path(gap['source']['path'])) == gap['source']
assert 0.0051 < gap['coarse_over_halfspace_von_mises'] < 0.0053
assert 0.0080 < gap['coarse_over_halfspace_tensile'] < 0.0083
receipt = {
    'passed': True,
    'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting,
    'current_sources_match_executed': matched,
    'cpu_potential_control': study,
    'native_control_and_cost': native,
    'scope': 'Normal elliptic half-space tensor evaluation including the surface, edge, axis and distant material probes. Numerical evaluation and independent same-pressure integration are verified. The reused coarse-shell comparison diagnoses a scale gap between elastic domains; it is not a shell physical-error measurement or a validated field combination.',
    'whole_goal_complete': False,
    'next': 'Validate particular Hertz stress plus a finite-domain residual correction against matched resolved-pressure elasticity on a small flat-faced body, then a hollow polar-land egg. Subtract the particular boundary traction before solving the residual; retain inertial relief and boundary integration errors. Arbitrary curved/nonconvex use needs a valid local domain or a cutoff with its equilibrium residual. Continue current-region/curvature preparation, a declared tangential history model and actual moving-scene throughput. Do not restart pad expansion or a point-load peak sweep.',
}
terminal_path = RUN / 'meta/hard-hertz-stress-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path, Path(gap['source']['path'])])
source = read(Path(gap['source']['path']))
egg = next(row for row in source['rows'] if row['case'] == 'egg')
external.add(Path(egg['arrays']['path']))
scope = (
    'Elliptic Hertz normal stress from a one-dimensional potential integral, with exact surface traction and a scaled near/far integration map. '
    'Final independent CPU controls cover four axis ratios, circular formulas for three Poisson ratios and pressure-area integration. '
    'The original slender-ellipse reference-resolution failure is retained; its refined comparison reaches4.60e-7 relative tensor difference. '
    'Final native16-point CPU/GPU controls each evaluate2849 complete tensors and retain release, invalid inputs, region isolation and recovery. '
    'Maximum sampled difference from192-point evaluation is0.01805% of peak pressure; distant-probe relative difference is at most1.01e-8. '
    'Matching resident77-probe/region timings cover1/128/512 regions, including a48-point GPU control;16 points reduce observed GPU cost by39.6-45.2%. '
    'Original native initialization/type failures and previous successful configurations remain preserved. '
    'The retained0.2N ellipse gives a16.5MPa axial half-space von-Mises candidate versus0.086MPa in the complete coarse shell field. '
    'That is a scale-gap diagnosis across distinct domains, not shell accuracy. Finite-domain combination, tangential history, physical current-region inputs and moving-scene max/full cost remain open.'
)
packet = freeze('hard-hertz-stress-v1', files, sorted(external), scope)
(DEST / packet['packet'] / 'README.md').write_text(
    '# Normal Hertz near-field tensors\n\n' + scope + '\n\n'
    'The manifest retains encoded/original payload hashes, actual captured worker sources, task arguments, logs, '
    'terminal accounting and external array identities. Runtime arrays remain on data; private caches use compute-node scratch.\n'
)
with (RUN / 'meta/hard-hertz-stress-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
