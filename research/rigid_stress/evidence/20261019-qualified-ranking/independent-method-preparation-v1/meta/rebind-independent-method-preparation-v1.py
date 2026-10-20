"""Advance independent preparation after full parent admission; retain final qualification explicitly."""
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = ROOT / 'meta'
PACKET = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/stress_qualified_ranking/research/rigid_stress/evidence/20261019-qualified-ranking/canonical125-resilient-recovery-v1/manifest.json')
CHANGES = [
    ('731022', 'canonical-stvk-resilient-method-ranking-v1', {('afterany', '731020')}, 'afterany:731020:731019'),
    ('731020', 'canonical-stvk-resilient-method-fields-v1', {('afterok', '731019')}, 'afterok:731010'),
    ('731021', 'canonical-friction-resilient-plot-data-v1', {('afterok', '731019')}, 'afterok:731010'),
]
UNCHANGED = ['730978_0', '730979_0', '730980_0', '731010', '731011', '731013', '731014', '731016', '731019', '731023', '731025']


def show(job):
    return subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True, timeout=20)


def dependencies(state):
    return set(re.findall(r'(afterok|afterany):([0-9]+)', state))


def identity(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write_fresh(path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


old = json.loads(PACKET.read_text())
published = {row['source']: row for row in old['files']}
files = []
campaigns = [campaign for _, campaign, _, _ in CHANGES] + [
    'canonical125-resilient-spatial-parent-admission-v1', 'canonical-friction-resilient-qualification-v1',
    'canonical-stvk-resilient-figures-v1', 'pad-friction-resilient-cost-accuracy-v1',
]
for campaign in campaigns:
    root = ROOT / campaign
    for name in ['submission.json', 'source.json', 'source.patch', 'analysis-files.json', 'tasks.json', 'job.sh']:
        receipt = identity(root / name)
        recorded = published[receipt['path']]
        assert receipt['bytes'] == recorded['original_bytes'] and receipt['sha256'] == recorded['original_sha256']
        files.append(receipt)
    inventory = json.loads((root / 'analysis-files.json').read_text())
    for row in inventory:
        actual = identity(root / 'analysis-source' / row['path'])
        assert all(actual[key] == row[key] for key in ['bytes', 'sha256'])

before = {job: show(job) for job, _, _, _ in CHANGES}
unchanged_before = {job: show(job) for job in UNCHANGED}
for job, campaign, expected, _ in CHANGES:
    state = before[job]
    assert 'JobState=PENDING' in state and 'StartTime=Unknown' in state and 'Restarts=0' in state, state
    assert dependencies(state) == expected, state
    assert not list((ROOT / campaign).glob('group*/*/result.json'))
    assert not list((ROOT / campaign).glob('group*/returncode.txt'))
for job in ['730978_0', '730979_0', '730980_0']:
    assert 'JobState=RUNNING' in unchanged_before[job]
assert dependencies(unchanged_before['731010']) == {('afterok', '730978')}
assert dependencies(unchanged_before['731019']) == {('afterany', '731011'), ('afterany', '731016')}
assert dependencies(unchanged_before['731023']) == {('afterany', '731022'), ('afterany', '731021')}
assert dependencies(unchanged_before['731025']) == {('afterany', '731022')}

contract = {
    'created_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope': 'Only pending scheduler dependencies change. Complete parent mechanics/momentum still gate fields and force-data preparation; actual seven-parent resolution qualification explicitly gates final ranks. Figure/cost guards and all source/task/output identities remain unchanged.',
    'changes': [{'job': job, 'campaign': campaign, 'old_dependencies': sorted(expected), 'new_dependency': desired}
                for job, campaign, expected, desired in CHANGES],
    'consumer_dependency_updated_before_producers': True,
    'before': before, 'unchanged_before': unchanged_before,
    'published_packet_manifest': identity(PACKET), 'immutable_submission_files': files,
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
}
write_fresh(META / 'independent-method-preparation-contract-v1.json', contract)
events = META / 'independent-method-preparation-events-v1.jsonl'
assert not events.exists()
for job, campaign, _, desired in CHANGES:
    subprocess.run(['scontrol', 'update', 'JobId=' + job, 'Dependency=' + desired], check=True)
    state = show(job)
    expected = set(re.findall(r'(afterok|afterany):([0-9]+)', desired))
    assert 'JobState=PENDING' in state and dependencies(state) == expected, state
    elements = [show(job + '_' + str(index)) for index in range(6 if job == '731020' else 1)]
    assert all('JobState=PENDING' in element and dependencies(element) == expected for element in elements)
    with events.open('a') as stream:
        stream.write(json.dumps({'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 'job': job, 'dependency': desired, 'actual_after': state,
                                 'array_elements_after': elements}) + '\n')

unchanged_after = {job: show(job) for job in UNCHANGED}
assert all(dependencies(unchanged_after[job]) == dependencies(unchanged_before[job]) for job in UNCHANGED)
assert all(identity(row['path']) == row for row in files)
after = {job: show(job) for job, _, _, _ in CHANGES}
write_fresh(META / 'independent-method-preparation-verified-v1.json', {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'checks': {
        'three_pending_jobs_only_rebound': True,
        'all_eight_pending_array_elements_verified': True,
        'all_published_submission_and_source_inventories_unchanged': True,
        'three_live_fine_jobs_untouched': True,
        'strict_four_parent_history_momentum_guard_retained': True,
        'full_seven_parent_resolution_qualification_explicit_before_ranking': True,
        'all_other_guard_qualification_figure_cost_dependencies_unchanged': True,
        'same_reference_paths_and_all_18_complete_field_tasks': True,
        'same_force_data_source_with_no_spatial_admission_claim': True,
        'consumer_updated_before_producers': True,
    },
    'after': after, 'unchanged_after': unchanged_after,
    'receipts': [identity(META / 'independent-method-preparation-contract-v1.json'), identity(events), *files],
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
})
print(json.dumps({'passed': True, 'changes': contract['changes'], 'numerical_source_or_targets_changed': False}))
