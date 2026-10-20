"""Retain finite Hertz transfer controls and explicitly limited wrench sensitivity."""

import hashlib
import json
from pathlib import Path
import subprocess

RUN = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = RUN / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))

campaigns = [f'hard-hertz-transfer-v{i}' for i in range(1, 5)]
campaigns += ['hard-hertz-wrench-v1', 'hard-hertz-wrench-v2']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(
    ['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList'],
    text=True,
)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    code = '1' if campaign in campaigns[:2] else '0'
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if code == '1' else ['COMPLETED', '0:0'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == code
        if code == '0':
            for task in tasks:
                assert read(root / f'group{group}' / task['name'] / 'result.json')['passed']

matched = []
for module, campaign in (
    ('native_hertz_transfer', 'hard-hertz-transfer-v4'),
    ('check_hertz_transfer', 'hard-hertz-transfer-v4'),
    ('study_hertz_wrench', 'hard-hertz-wrench-v2'),
):
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    snapshot_path = RUN / campaign / 'analysis-source' / relative
    assert snapshot_path.read_bytes() == (WORK / relative).read_bytes()
    matched.append({'module': module, 'sha256': hashlib.sha256(snapshot_path.read_bytes()).hexdigest()})

results = [read(RUN / f'hard-hertz-transfer-v4/group{i}/check/result.json') for i in (0, 1)]
assert [result['backend'] for result in results] == ['gpu', 'cpu']
cases = ['face', 'edge', 'vertex', 'cavity', 'u', 'egg', 'rotated_egg']
for result in results:
    assert result['passed'] and not result['whole_goal_complete']
    assert [row['case'] for row in result['rows']] == cases
    assert all(row['tensor_relative_error'] < 1.42e-11 for row in result['rows'])
    assert all(row['quadrature_complete_tensor_relative'] < 0.001989 for row in result['rows'])
    assert [row['regions_and_environments'] for row in result['benchmark']] == [1, 128, 512]
    for row in result['rows']:
        for key in ('arrays', 'mesh'):
            assert identity(Path(row[key]['path'])) == row[key]

wrench = read(RUN / 'hard-hertz-wrench-v2/group0/study/result.json')
assert wrench['passed'] and wrench['strict_original_preserved'] and not wrench['engineering_admitted']
assert identity(Path(wrench['arrays']['path'])) == wrench['arrays']
assert 0.000958 < wrench['complete_tensor_relative_change'] < 0.000959
for metric in wrench['metrics'].values():
    assert metric['cone_violations'] == 60 and metric['contact_plane_cone_violations'] == 0
    assert metric['minimum_sample_compression_n'] > 0 and metric['contact_plane_minimum_compression_n'] > 0

receipt = {
    'passed': True,
    'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting,
    'current_sources_match_executed': matched,
    'native_cases_per_backend': 7,
    'maximum_same_load_tensor_relative_error': max(row['tensor_relative_error'] for result in results for row in result['rows']),
    'maximum_dense_rule_tensor_relative_difference': max(row['quadrature_complete_tensor_relative'] for result in results for row in result['rows']),
    'additional_final_source_controls': ['invalid environment index', 'failed region rejects shared environment', 'shear with zero normal load'],
    'benchmarks': {result['backend']: result['benchmark'] for result in results},
    'timing_scope': 'Resident supplied-region estimate, 96-point topology walk, full-wrench check, nodal zero/scatter. Homogeneous coarse egg vertex; excludes elasticity, input/grouping/curvature preparation, JIT, copies and actual scene dynamics.',
    'wrench_candidate': wrench,
    'physical_scope': 'Same-load P2 implementation controls and finite pressure-integration sensitivity. Separate reference-facet versus fixed contact-plane cone diagnostics are different traction-frame models. No independent hard-shell, friction-history or global-peak accuracy claim.',
    'whole_goal_complete': False,
    'next': 'Assess cheap Hertz near-field stress against the coarse shell field before claiming global peaks; resolve overlap and full-output semantics. Connect cached rest-curvature/current physical-region inputs, declare and test a cheap tangential history law, then measure matching moving-scene pure-rigid/max/full throughput. Do not resume soft-pad or dense elastic-basis expansion.',
}
terminal_path = RUN / 'meta/hard-hertz-transfer-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
scope = (
    'Finite physical Hertz footprints use96 samples and witnessed material-topology walks, followed by signed P2 nodal scatter. '
    'Seven native CPU/GPU controls each cover flat face/edge/vertex, cavity, U gap refusal, coarse egg and common rotation/translation. '
    'Final source also checks bad environment indices, multiple-region failure isolation and zero-normal shear refusal. '
    'Independent same-load complete P2 fields agree within1.42e-11; a dense12x1024 rule changes complete fields by at most0.199%. '
    'Matching resident estimate/transfer timing covers1/128/512 repeated supplied regions, excluding elasticity and scene dynamics. '
    'Original dependency and geometry failures are retained. A separate coarse-egg6x6 wrench correction changes its complete P2 field by0.09582%, '
    'but both prior and corrected loads violate60/96 reference-facet mu=0.5 cones. Fixed contact-plane cones pass with different normals; '
    'this is an explicitly different friction-frame model, not an admission or physical-history validation. '
    'No automatic input preparation, shell hard-contact accuracy, resolved near-field/global peak or production scene-throughput claim is made. '
    'Full arrays remain on data; prior strict transfer/baselines remain unchanged.'
)
packet = freeze('hard-hertz-transfer-v1', files, sorted(external), scope)
(DEST / packet['packet'] / 'README.md').write_text(
    '# Finite Hertz footprint transfer\n\n' + scope + '\n\n'
    'The manifest records encoded/original payload hashes and external array identities. '
    'Executed workers, task arguments, engine inventories, logs and terminal accounting are retained. '
    'Runtime arrays stay on data and private caches use compute-node scratch.\n'
)
with (RUN / 'meta/hard-hertz-transfer-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
