"""Retain elliptic tangential response, history discrepancy and measured module cost."""
from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-tangential-v1', 'hard-native-tangential-v1', 'hard-native-tangential-v2']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    failed = campaign == 'hard-native-tangential-v1'
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if failed else ['COMPLETED', '0:0'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == ('1' if failed else '0')
        if not failed:
            for task in tasks:
                assert read(root / f'group{group}' / task['name'] / 'result.json')['passed']
matched = []
for module, campaign in [('hertz_tangential', 'hard-tangential-v1'), ('study_hertz_tangential', 'hard-tangential-v1'), ('native_hertz_tangential', 'hard-native-tangential-v2'), ('native_hertz_stress', 'hard-native-tangential-v2'), ('check_native_hertz_stress', 'hard-native-tangential-v2')]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    executed = RUN / campaign / 'analysis-source' / relative
    assert executed.read_bytes() == (WORK / relative).read_bytes()
    matched.append(identity(executed))
study = read(RUN / 'hard-tangential-v1/group0/study/result.json')
assert study['passed'] and len(study['history']) == 8
assert max(row['independent_cerruti_relative'] for row in study['elliptic_controls']) < 6e-9
zero = [row for row in study['history'] if row['force_over_mu_normal'] == 0]
assert len(zero) == 2 and all(row['sample_tensile_peak_relative'] > 0.47 for row in zero)
native = []
for group in range(2):
    tangent = read(RUN / f'hard-native-tangential-v2/group{group}/tangent/result.json')
    normal = read(RUN / f'hard-native-tangential-v2/group{group}/normal/result.json')
    assert tangent['passed'] and tangent['regions'] * tangent['probes_per_region'] == 2923
    assert tangent['maximum_high_order_tensor_difference_over_peak_pressure'] < 5e-3
    assert normal['passed'] and normal['regions'] * normal['probes_per_region'] == 2849
    assert normal['maximum_high_order_tensor_difference_over_peak_pressure'] < 1e-3
    native.append({'tangential': tangent, 'normal_shared_root_regression': normal})
receipt = {
    'passed': True, 'superseding_goal_commit': '75d534428e83072ac6fdcb17c5bcde64d01009e9',
    'scheduler_accounting': accounting, 'current_sources_match_executed': matched,
    'study': study, 'native': native, 'whole_goal_complete': False,
    'scope': 'Elliptic tangential traction response is checked against independent Cerruti point-force integration. Native CPU/GPU controls cover2923 tensors each, finite surface/edge/axis limits, two traction components, release/refusal/isolation/recovery, and shared normal-root regression. GPU supplied-basis update costs0.290-0.460ms for79 probes at1-512 regions;16-point error is at most0.484% of normal peak pressure on the checked set. A separate constant-normal-load collinear Cattaneo-Mindlin cycle exposes47.76% sampled tensile-peak error when a proportional law discards history. This is not a native history law, coupled friction qualification, complete-volume/continuum-peak shell accuracy or scene throughput.',
    'failed_test': 'Both original native groups failed at a NumPy advanced-index axis-order mismatch in the surface-traction assertion. Explicit array selection and per-region pressure normalization repair the test; the numerical tangential kernel is unchanged. Both final groups complete tangent and normal tasks.',
    'next': 'Use finite Hertz traction bases for an inexpensive force-controlled history approximation; qualify normal-load changes, direction changes, contact identity/reset and resultant/traction bounds. Do not adopt memoryless proportional traction as an accurate friction law. Continue finite-shell correction and useful actual max/full scene integration; no default coupled Newton-PCG or pad expansion.',
}
terminal_path = RUN / 'meta/hard-hertz-tangential-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path])
packet = freeze('hard-hertz-tangential-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text('# Elliptic tangential stress and friction-history discrepancy\n\n' + receipt['scope'] + '\n\nThe manifest retains exact executed sources, original failed assertion and final controls, terminal scheduler state and full result tables. Complete tensor/cycle arrays remain hashed on data. Private compiler caches were node-local. The history comparison uses the stated Cattaneo-Mindlin approximation, not a qualified coupled rigid-indenter friction reference.\n')
with (RUN / 'meta/hard-hertz-tangential-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
