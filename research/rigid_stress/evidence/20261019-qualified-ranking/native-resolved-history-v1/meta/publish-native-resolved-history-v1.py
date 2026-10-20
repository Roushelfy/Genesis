"""Freeze the completed native resolved-transfer history and independent comparisons."""
from pathlib import Path

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = ['native-resolved-history-v1', 'native-resolved-history-admission-v1',
             'native-resolved-history-comparison-v1']
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
for campaign, job in zip(campaigns, jobs):
    assert (RUN / campaign / 'group0/returncode.txt').read_text().strip() == '0'
    assert observed[f'{job}_0'] == ['COMPLETED', '0:0']
admission = read(RUN / campaigns[1] / 'group0/admission/result.json')
comparison = read(RUN / campaigns[2] / 'group0/comparison/result.json')
assert admission['passed'] and comparison['passed'] and comparison['observations'] == 400
assert len(admission['cases']) == 2
for case in admission['cases']:
    assert case['steps'] == 400 and case['mechanical_history_passed'] and case['all_windows_passed']
    assert case['all_stress_observations_valid']
    for field in case['fields']:
        assert field['adapter_passed'] and field['recovered_surface_relative_wrench_error'] <= 1e-7
assert [row['step'] for row in comparison['corrections']] == [37, 38]
assert comparison['maximum_peak_mixed_scaled_difference'] <= 1
for row in comparison['corrections']:
    assert row['force_change_bound_n'] <= row['physical_force_budget_n']
for field in comparison['fields']:
    assert max(field['errors'].values()) <= 1e-6
files, external = collect(campaigns)
files += [BASE]
files += list(META.glob('native-resolved-history*-tasks-v1.json'))
for item in comparison['receipts']:
    path = Path(item['path'])
    assert identity(path) == item
    if path.suffix == '.npz':
        external.add(path)
    else:
        files.append(path)
for case in admission['cases']:
    for key in ('run', 'journal'):
        path = Path(case[key]['path'])
        assert identity(path) == case[key]
        files.append(path)
    for field in case['fields']:
        for key in ('field', 'geometry'):
            path = Path(field[key]['path'])
            assert identity(path) == field[key]
            external.add(path)
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
    'captured_imports': cones, 'observations': 400,
    'all_steps_and_mechanical_windows_passed': True,
    'complete_field_errors': comparison['fields'],
    'maximum_peak_absolute_difference_pa': comparison['maximum_peak_absolute_difference_pa'],
    'maximum_peak_mixed_scaled_difference': comparison['maximum_peak_mixed_scaled_difference'],
    'corrections': comparison['corrections'],
    'intermediate_complete_original_source_arrays_captured': False,
    'new_per_event_correction_stress_bound_claimed': False,
    'production_automatic_pressure_model_admitted': False,
    'production_tolerance_changed': False, 'whole_goal_complete': False}
p = META / 'native-resolved-history-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packet = freeze('native-resolved-history-v1', files, sorted(external),
    'The production resolved-boundary transfer completes all400 observations in the declared normal/friction research history with explicit bounded source-input reconciliation. Independent mechanical/source admission and every-step peak/endpoint-field comparisons pass. Maximum peak difference is1.13e-5Pa, only0.0369 of the predeclared1e-6Pa+1e-8-relative allowance; endpoint complete stresses differ from the earlier resolved research adapter by at most3.24e-11. Observations37/38 retain force-correction bounds5.15e-21N/2.22e-20N, below their declared physical budgets. Complete original/corrected source arrays exist at endpoint keyframes; intermediate event arrays were not captured, so no new event-specific stress-perturbation bound is claimed. Earlier exact P2 correction controls remain separate. Original inputs, strict failures and engine baselines are retained. This is continuous numerical transfer/history conformance, not automatic physical pressure prediction, new continuum-error qualification or native production throughput.')
with (META / 'native-resolved-history-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
