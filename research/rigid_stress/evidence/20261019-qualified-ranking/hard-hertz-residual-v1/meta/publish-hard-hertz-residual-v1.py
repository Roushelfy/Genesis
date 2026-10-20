"""Freeze finite-box decomposition and matched native correction controls."""

import json
from pathlib import Path
import subprocess

RUN = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = RUN / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))

campaigns = [f'hard-hertz-residual-v{i}' for i in range(1, 4)]
campaigns += [f'hard-hertz-box-v{i}' for i in range(1, 5)]
failed = set(campaigns) - {'hard-hertz-residual-v3', 'hard-hertz-box-v3', 'hard-hertz-box-v4'}
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(
    ['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
     '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True,
)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    for group, tasks in enumerate(read(root / 'tasks.json')):
        expected = ['FAILED', '1:0'] if campaign in failed else ['COMPLETED', '0:0']
        assert observed[f'{job}_{group}'] == expected, (campaign, observed[f'{job}_{group}'])
        code = (root / f'group{group}/returncode.txt').read_text().strip()
        assert (code != '0') == (campaign in failed)
        if campaign not in failed:
            for task in tasks:
                assert read(root / f'group{group}' / task['name'] / 'result.json')['passed']

matched = []
for module, campaign in (
    ('study_hertz_residual', 'hard-hertz-residual-v3'),
    ('native_hertz_box', 'hard-hertz-box-v4'),
    ('check_hertz_box', 'hard-hertz-box-v4'),
):
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    executed = RUN / campaign / 'analysis-source' / relative
    assert executed.read_bytes() == (WORK / relative).read_bytes(), module
    matched.append({'module': module, 'sha256': identity(executed)['sha256']})

study = read(RUN / 'hard-hertz-residual-v3/group0/study/result.json')
assert study['corrected_targets_met'] and not study['whole_goal_complete']
assert [row['tetrahedra'] for row in study['rows']] == [192, 6048, 12288]
for row in study['rows']:
    assert identity(Path(row['arrays']['path'])) == row['arrays']
    assert max(method['elastic_residual_relative'] for method in row['methods'].values()) < 1e-7
native = [read(RUN / f'hard-hertz-box-v4/group{i}/{name}/result.json')
          for i in range(2) for name in ('hertz', 'point')]
assert [(row['backend'], row['remote_traction']) for row in native] == [
    (b, t) for b in ('gpu', 'cpu') for t in ('hertz', 'point')]
for row in native:
    assert row['passed'] and not row['whole_goal_complete']
    assert identity(Path(row['arrays']['path'])) == row['arrays']
    assert row['queries'] == 861 and row['scalar_nodes'] == 405
    assert row['moving_center_curvature_tensor_relative'] < 1e-7
    assert [(r['batch'], r['output']) for r in row['benchmark']] == [
        (b, mode) for b in (1, 128, 512) for mode in ('max', 'full')]
    for case in row['rows']:
        assert case['same_load_tensor_relative'] < 1e-7
        assert case['higher_rule_tensor_relative'] < 1e-3
        assert case['complete_volume_tensor_relative'] < 1e-3
        if row['remote_traction'] == 'point':
            assert case['independent_point_traction_relative'] < 1e-4
        assert case['wrench_error_relative'] < 1e-4

receipt = {
    'passed': True,
    'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting,
    'current_sources_match_executed': matched,
    'finite_box': study,
    'native_box': native,
    'original_native_cost': [read(RUN / f'hard-hertz-box-v3/group{i}/check/result.json') for i in range(2)],
    'whole_goal_complete': False,
    'scope': 'Prescribed normal finite-area Hertz pressure on a finite flat-faced free body. The residual subtraction preserves inertial relief and avoids duplicate loading. Corrected-mesh agreement does not independently qualify unknown pressure or direct FEM continuum peaks. Native normal-box cost includes estimate, boundary residual, elastic solve, complete numerical residual, combined sampled field and max/full output; no moving scene or automatic physical-region input claim.',
    'next': 'Keep the cheap hard-contact priority. Use the finite-body correction on a small hollow polar-land egg and targeted hard-contact pressure controls, while connecting current physical regions/rest curvature and an explicit inexpensive tangential/history law to the native scene. General curved/nonconvex geometry needs a valid local half-space domain or controlled localization. Actual paired moving-scene pure-rigid/max/full throughput remains required. No further soft-pad expansion, full vector elastic basis or default coupled Newton-PCG path.',
}
terminal_path = RUN / 'meta/hard-hertz-residual-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
scope = (
    'Finite free-box Hertz particular stress plus a traction-subtracted P2 residual, using192/6048/12288 tetrahedra, '
    '24192 volume-weighted queries,9123 peak probes and all element corners. Coarse/fine corrected field difference is1.575%; '
    'the independent direct-pressure FEM comparison remains resolution-limited at8.56%, with8.03% direct mesh change. '
    'Both earlier8GB memory failures and the final bounded-memory completion are preserved. '
    'Matched CPU/GPU native normal-box controls combine boundary subtraction, elastic recovery and full/max fields at861 supplied queries; '
    'module timings cover1/128/512 environments, with release, invalid input, isolation, recovery, center motion and curvature rotation. '
    'The original compile-time-shape failure and mismatched-ellipse numerical comparison are retained. '
    'Lookup-induced field changes are separate from same-ellipse implementation/integration errors. '
    'A separately measured closed-form point-force approximation is used only on the distant box boundary, '
    'while retaining finite Hertz near-field tensors. Its volume-weighted difference from refined Hertz boundary correction is below0.024%, '
    'with independent complex-step Boussinesq traction checks and matched native timings. '
    'Unknown shell pressure, tangential history, arbitrary moving-contact production integration and scene throughput remain open.'
)
packet = freeze('hard-hertz-residual-v1', files, sorted(external), scope)
(DEST / packet['packet'] / 'README.md').write_text(
    '# Finite-box Hertz residual and native module cost\n\n' + scope + '\n\n'
    'The manifest records original/encoded payload hashes, captured executed sources, configurations, failures, '
    'terminal scheduler receipts and externally hashed full arrays. Runtime data stays under the established data root; '
    'private compiler caches use compute-node scratch.\n'
)
with (RUN / 'meta/hard-hertz-residual-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
