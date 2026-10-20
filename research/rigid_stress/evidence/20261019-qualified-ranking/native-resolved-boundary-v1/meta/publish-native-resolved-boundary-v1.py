"""Freeze completed native boundary-transfer implementation controls."""
from pathlib import Path
import xml.etree.ElementTree as ET

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = [f'native-resolved-boundary-tests-v{i}' for i in range(1, 5)]
campaigns.append('native-resolved-physical-fields-v1')
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
tests = []
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        code = (RUN / campaign / f'group{group}/returncode.txt').read_text().strip()
        assert observed[f'{job}_{group}'] == (['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0'])
        failed = (campaign == 'native-resolved-boundary-tests-v1' and group == 0) or campaign == 'native-resolved-boundary-tests-v2'
        assert code == ('1' if failed else '0')
        path = RUN / campaign / f'group{group}/tests/junit.xml'
        if path.exists():
            cases = ET.parse(path).findall('.//testcase')
            for case in cases:
                assert case.find('error') is None and case.find('skipped') is None
            tests.append({'campaign': campaign, 'group': group,
                'passed': sum(case.find('failure') is None for case in cases),
                'failed': sum(case.find('failure') is not None for case in cases),
                'receipt': identity(path)})
assert [(row['passed'], row['failed']) for row in tests] == [(2, 2), (6, 0), (1, 1), (1, 0), (2, 0), (4, 0)]
fields = read(RUN / 'native-resolved-physical-fields-v1/group0/control/result.json')
assert fields['passed'] and len(fields['rows']) == 2
assert max(value for row in fields['rows'] for value in row['relative_errors'].values()) <= 1e-6
files, external = collect(campaigns)
files += [BASE]
files += list(META.glob('native-resolved-boundary-*.json'))
files += list(META.glob('native-resolved-boundary-*.py'))
files += list(META.glob('native-resolved-boundary-*.patch'))
files += list(META.glob('native-resolved-physical-fields-*.json'))
for campaign in campaigns:
    files += list((RUN / campaign).glob('group*/tests/junit.xml'))
for row in fields['rows']:
    for key in ('physical', 'cpu_field', 'field', 'sources'):
        item = row[key]
        path = Path(item['path'])
        assert identity(path) == item
        external.add(path)
receipt = {
    'passed': True, 'actual_scheduler_accounting': accounting,
    'captured_imports': cones, 'tests': tests, 'complete_field_cases': fields['rows'],
    'engine_source': str(Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/resolved-boundary-production')),
    'effective_unique_checks': 'GPU and CPU nonconvex/cavity work tests; both-backend max/full checkpoint and post-reset isolation tests; six existing manual-pressure regressions.',
    'original_test_setup_and_assertion_api_failures_retained': True,
    'continuous_research_history_in_separate_live_campaign': 'native-resolved-history-v1',
    'production_automatic_pressure_model_admitted': False, 'production_tolerance_changed': False,
    'whole_goal_complete': False,
}
p = META / 'native-resolved-boundary-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packet = freeze('native-resolved-boundary-v1', files, sorted(external),
    'Native load_transfer=\"resolved\" routes declared integrated samples through witnessed material-boundary P2 shapes without a second finite disk. Source geometry/ambiguity, supplied normal/friction and original-point wrench acceptance remain explicit. Nonconvex and cavity quadratic virtual work, force/moment, release and failure isolation pass on GPU and CPU; public max/full checkpoint/reset and next-step isolation pass on both backends. Six existing manual-pressure regressions pass. Actual normal/friction complete fields match the independent same-load CPU resolved transfer to1.26e-11. Original unsupported zero-friction test setup and missing assertion-API argument failures remain retained; repaired assertions are exact or keep their original numerical thresholds. Runtime force assembly, balance, elasticity and observation remain native Quadrants. This is the conservative-transfer component, not automatic physical support/pressure prediction or production throughput. A separately submitted400-step bounded-input research history is excluded while live. Original engine baselines and unrelated work are preserved.')
with (META / 'native-resolved-boundary-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
