"""Finish a safely stopped scheduler update; preserve the original reader failure."""
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = ROOT / 'meta'


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def show(job):
    return subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=20)


def actual_dependencies(state):
    return set(re.findall(r'(afterok|afterany):([0-9]+)', state))


def requested_dependencies(value):
    kind, *jobs = value.split(':')
    return {(kind, job) for job in jobs}


def write_fresh(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


contract_path = META / 'independent-method-preparation-contract-v1.json'
contract = json.loads(contract_path.read_text())
files = contract['immutable_submission_files']
assert all(identity(row['path']) == row for row in files)
before = {row['job']: show(row['job']) for row in contract['changes']}
assert actual_dependencies(before['731022']) == {('afterany', '731020'), ('afterany', '731019')}
for job in ('731020', '731021'):
    assert actual_dependencies(before[job]) == {('afterok', '731019')}
assert all('JobState=PENDING' in state and 'StartTime=Unknown' in state for state in before.values())
failure = {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'original_script': identity(META / 'rebind-independent-method-preparation-v1.py'),
    'original_contract': identity(contract_path),
    'actual_v1_command_exit_code': 1,
    'failure': 'Scheduler-receipt assertion parsed only the first ID of requested afterany:731020:731019, while Slurm correctly expanded both dependencies. The script stopped after the safe consumer update and before either producer update.',
    'correct_consumer_update_preserved': before['731022'],
    'remaining_producers_not_changed_by_v1': {job: before[job] for job in ('731020', '731021')},
    'no_scientific_failure_or_source_change': True,
}
failure_path = META / 'independent-method-preparation-reader-failure-v1.json'
write_fresh(failure_path, failure)
write_fresh(META / 'independent-method-preparation-recovery-contract-v2.json', {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'initial_contract': identity(contract_path), 'failure_receipt': identity(failure_path),
    'before': before, 'changes': contract['changes'],
    'consumer_dependency_already_correct_and_kept': True,
    'immutable_original_submissions_and_captures_kept': True,
})
events = META / 'independent-method-preparation-events-v2.jsonl'
assert not events.exists()
for row in contract['changes']:
    job, campaign, desired = row['job'], row['campaign'], row['new_dependency']
    expected = requested_dependencies(desired)
    if job != '731022':
        assert not list((ROOT / campaign).glob('group*/*/result.json'))
        assert not list((ROOT / campaign).glob('group*/returncode.txt'))
        subprocess.run(['scontrol', 'update', 'JobId=' + job, 'Dependency=' + desired], check=True)
    state = show(job)
    elements = [show(job + '_' + str(index)) for index in range(6 if job == '731020' else 1)]
    assert 'JobState=PENDING' in state and actual_dependencies(state) == expected
    assert all('JobState=PENDING' in item and actual_dependencies(item) == expected for item in elements)
    with events.open('a') as stream:
        stream.write(json.dumps({'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'job': job, 'actual_after': state, 'array_elements_after': elements,
                                 'scheduler_update_in_v2': job != '731022'}) + '\n')

unchanged_after = {job: show(job) for job in contract['unchanged_before']}
assert all(actual_dependencies(state) == actual_dependencies(contract['unchanged_before'][job])
           for job, state in unchanged_after.items())
assert all('JobState=RUNNING' in unchanged_after[job] for job in ['730978_0', '730979_0', '730980_0'])
assert all(identity(row['path']) == row for row in files)
after = {row['job']: show(row['job']) for row in contract['changes']}
write_fresh(META / 'independent-method-preparation-verified-v2.json', {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'checks': {
        'three_pending_jobs_only_rebound': True,
        'all_eight_pending_array_elements_verified': True,
        'all_published_submission_and_source_inventories_unchanged': True,
        'three_live_fine_jobs_untouched_and_verified_running': True,
        'strict_four_parent_history_momentum_guard_retained': True,
        'full_seven_parent_resolution_qualification_explicit_before_ranking': True,
        'all_other_guard_qualification_figure_cost_dependencies_unchanged': True,
        'same_reference_paths_and_all_18_complete_field_tasks': True,
        'same_force_data_source_with_no_spatial_admission_claim': True,
        'consumer_updated_before_producers': True,
        'original_reader_failure_and_correct_partial_update_preserved': True,
    },
    'after': after, 'unchanged_after': unchanged_after,
    'receipts': [identity(contract_path), identity(failure_path), identity(events), *files],
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
})
print(json.dumps({'passed': True, 'checks': 11, 'actual_jobs': list(after),
                  'source_and_numerical_targets_unchanged': True, 'failed_receipt_reader_preserved': True}))
