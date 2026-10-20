"""Publish immutable scheduling/source receipts without copying any live numerical output."""
import datetime
import json
import pathlib
import re
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = RUN / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
exec(compile(HELPER.read_text().split('\nimport ast\n', 1)[0], str(HELPER), 'exec'))
read = lambda path: json.loads(path.read_text())
verified = read(META / 'independent-method-preparation-verified-v2.json')
assert verified['passed'] and len(verified['checks']) == 11 and all(verified['checks'].values())
for row in verified['receipts']:
    assert identity(pathlib.Path(row['path'])) == row
original = read(META / 'independent-method-preparation-contract-v1.json')
old_manifest = pathlib.Path(original['published_packet_manifest']['path'])
assert identity(old_manifest) == original['published_packet_manifest']
published = {row['source']: row for row in read(old_manifest)['files']}
inventories = []
for receipt in original['immutable_submission_files']:
    if not receipt['path'].endswith('/analysis-files.json'):
        continue
    root = pathlib.Path(receipt['path']).parent
    inventory = read(pathlib.Path(receipt['path']))
    for row in inventory:
        actual = identity(root / 'analysis-source' / row['path'])
        assert all(actual[key] == row[key] for key in ('bytes', 'sha256'))
    inventories.append({'campaign': root.name, 'files': len(inventory), 'inventory': receipt})
guards = []
for campaign, module in [
    ('canonical125-resilient-spatial-parent-admission-v1', 'require_solid_pad_admission.py'),
    ('canonical-friction-resilient-qualification-v1', 'qualify_friction_reference.py'),
    ('canonical-stvk-resilient-method-fields-v1', 'contact_fem_exact_columns.py'),
    ('canonical-friction-resilient-plot-data-v1', 'friction_plot_data.py'),
    ('canonical-stvk-resilient-method-ranking-v1', 'rank_stvk_pad_methods.py'),
    ('canonical-stvk-resilient-figures-v1', 'plot_stvk_pad_figures.py'),
]:
    path = RUN / campaign / 'analysis-source/research/rigid_stress/qipc_comparison' / module
    actual = identity(path)
    row = published[str(path)]
    assert actual['bytes'] == row['original_bytes'] and actual['sha256'] == row['original_sha256']
    guards.append(actual)
states = {}
for row in original['changes']:
    job = row['job']
    actual = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    kind, *jobs = row['new_dependency'].split(':')
    assert set(re.findall(r'(afterok|afterany):([0-9]+)', actual)) == {(kind, item) for item in jobs}
    assert 'JobState=PENDING' in actual
    states[job] = actual
failure = read(META / 'independent-method-preparation-reader-failure-v1.json')
assert failure['actual_v1_command_exit_code'] == 1 and failure['no_scientific_failure_or_source_change']
validation = {
    'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'eleven_actual_scheduler_wiring_checks': verified['checks'],
    'all_seven_captured_inventories_unchanged_after_rebinding': inventories,
    'six_published_computation_and_acceptance_modules_unchanged': guards,
    'current_changed_job_states': states,
    'no_live_output_frozen': True, 'no_new_numerical_results_or_timing_claim': True,
    'fine_reference_qualified': False, 'accepted_reference_frames': [], 'whole_goal_complete': False,
}
validation_path = META / 'independent-method-preparation-publication-validation-v1.json'
with validation_path.open('x') as stream:
    stream.write(json.dumps(validation, indent=2) + '\n')
names = [
    'rebind-independent-method-preparation-v1.py', 'rebind-independent-method-preparation-v2.py',
    'independent-method-preparation-contract-v1.json', 'independent-method-preparation-reader-failure-v1.json',
    'independent-method-preparation-recovery-contract-v2.json', 'independent-method-preparation-events-v2.jsonl',
    'independent-method-preparation-verified-v2.json', 'final-native-source-delivery-inspection-v1.json',
    'final-ranking-pairwise-source-inspection-v1.json',
]
files = [META / name for name in names] + [pathlib.Path(__file__), HELPER, validation_path]
scope = ('Independent method-field integration731020 and physical force/traction data preparation731021 now wait on strict complete-parent history/momentum admission731010, allowing them to run before the remaining contact/time resolution comparisons finish. '
         'Final ranking731022 explicitly waits for both method fields731020 and full seven-parent/36-comparison qualification731019. Final figure/cost dependencies and every original acceptance target remain unchanged. '
         'All eight pending array elements pass scheduler inspection;11 wiring checks, seven full captured-source inventories and six published computation/acceptance modules are verified unchanged. '
         'The first receipt reader misparsed the compact two-ID dependency after the consumer was correctly updated; its nonzero exit/source and correct partial scheduler state are preserved, and only the two remaining producer dependencies are subsequently updated. '
         'Additional read-only inspections confirm the complete delivered Genesis package equals measured1ff39b077, includes the MPR repair/regression, and the exact pending ranking source retains pairwise strict interval comparisons without overlap clustering. '
         'This is scheduling/source evidence only. No live fine output, new numerical error/rank, speedup measurement or final artwork is published; all required fine qualification and final delivery remain open.')
row = freeze('independent-method-preparation-v1', files, [], scope)
path = META / 'independent-method-preparation-publication-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
