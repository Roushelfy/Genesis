"""Freeze inactive-sample bandwidth controls after final paired scene admission."""

from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-scene-masked-samples-v1', 'hard-scene-sample-bandwidth-v1',
             'hard-scene-sample-bandwidth-reinspection-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for campaign, job in zip(campaigns, jobs):
    for group, tasks in enumerate(read(N / campaign / 'tasks.json')):
        preempted = campaign == campaigns[1] and group == 2
        assert observed[f'{job}_{group}'] == (['PREEMPTED', '0:0'] if preempted else ['COMPLETED', '0:0'])
        code = N / campaign / f'group{group}/returncode.txt'
        if preempted:
            assert not code.exists()
        else:
            assert code.read_text().strip() == '0'
        for task in tasks:
            path = N / campaign / f'group{group}' / task['name'] / 'result.json'
            if preempted and task['name'] == 'full-conformance':
                assert not path.exists()
                continue
            result = read(path)
            assert result.get('passed', result.get('execution_completed', False)), (campaign, group, task['name'])
final = N / campaigns[1]
reinspection = N / campaigns[2] / 'group0/full-conformance/result.json'
assert (N / campaigns[2] / 'analysis-source/research/rigid_stress/qipc_comparison/compare_hertz_implementations.py').read_bytes() == (final / 'analysis-source/research/rigid_stress/qipc_comparison/compare_hertz_implementations.py').read_bytes()
completed_artifacts = []
for task in ['cleared-max', 'masked-max', 'cleared-full', 'masked-full', 'rigid']:
    path = final / f'group2/{task}/result.json'
    result = read(path)
    assert result['execution_completed'] and len(result['rollouts']) == 4
    assert result['rollouts'][0]['audited'] and all(not row['audited'] and not row['research_audit_in_timing'] for row in result['rollouts'][1:])
    assert read(path.parent / 'rollouts.json') == result['rollouts']
    for row in [result['trajectory'], *result['snapshots']]:
        assert identity(Path(row['path'])) == row
    completed_artifacts.append(identity(path))
for variant in ['cleared', 'masked']:
    comparison = read(final / f'group2/{variant}-comparison/result.json')
    for name in ['max', 'full', 'rigid']:
        task = f'{variant}-{name}' if name != 'rigid' else 'rigid'
        assert identity(final / f'group2/{task}/result.json') == comparison['results'][name]
for mode in ['max', 'full']:
    comparison = read(final / 'group2/max-conformance/result.json' if mode == 'max' else reinspection)
    assert comparison['baseline'] == identity(final / f'group2/cleared-{mode}/result.json')
    assert comparison['candidate'] == identity(final / f'group2/masked-{mode}/result.json')
bindings = []
for module in ['native_hertz_transfer', 'live_hertz_scene', 'check_hertz_scene',
               'check_hertz_transfer', 'live_hertz_cleared_samples', 'compare_hertz_implementations']:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    path = final / 'analysis-source' / relative
    assert path.read_bytes() == (WORK / relative).read_bytes()
    bindings.append(identity(path))
paired = {name: read(final / f'group2/{name}-comparison/result.json') for name in ['cleared', 'masked']}
assert paired['cleared']['counts'] == paired['masked']['counts']
rows = []
identities = []
for mode in ['max', 'full']:
    for variant in ['cleared', 'masked']:
        result = read(final / f'group2/{variant}-{mode}/result.json')
        assert result['rollouts'][0]['inputs']['cleared_inactive_sample_geometry'] is (variant == 'cleared')
        identities.append(read(final / f'group2/{variant}-{mode}/start.json')['identity'])
    baseline = paired['cleared']['timing'][mode]
    candidate = paired['masked']['timing'][mode]
    # Require separated warmed replicate ranges, not a small median fluctuation.
    assert max(candidate['seconds_per_step']) < min(baseline['seconds_per_step'])
    rows.append({'mode': mode, 'baseline': baseline, 'candidate': candidate,
                 'inclusive_time_reduction': 1 - candidate['median_seconds_per_step'] / baseline['median_seconds_per_step'],
                 'added_time_reduction': 1 - candidate['added_seconds_per_step'] / baseline['added_seconds_per_step'],
                 'masked_replicates_all_below_cleared_replicates': True})
assert len({row['hostname'] for row in identities}) == 1
assert len({row['environment']['SLURM_JOB_ID'] for row in identities}) == 1
assert len({row['nvidia_smi_visible_inventory'] for row in identities}) == 1
for mode in ['max', 'full']:
    assert read(final / 'group2/max-conformance/result.json' if mode == 'max' else reinspection)['compared_scene_fields']
C = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_reproducible')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=C, text=True).strip()
for row in read(final / 'source.json')['files']:
    current = identity(C / row['path'])
    assert all(current[key] == row[key] for key in ['bytes', 'sha256'])
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
           'original_paired_group_preempted': True,
           'original_paired_returncode_marker_absent': True,
           'five_completed_same_allocation_measurement_artifacts': completed_artifacts,
           'missing_read_only_full_comparison_completed_separately': identity(reinspection),
           'original_results_and_scheduler_state_not_rewritten': True,
           'current_sources_match_executed': bindings, 'paired_scene_comparisons': paired,
           'paired_cost_changes': rows,
           'original_inactive_sample_clearing_control_preserved': True,
           'invalid_face_tags_refreshed_each_update': True,
           'logical_sample_capacity_unchanged': True,
           'B1024_sample_slots': 96 * 750 * 1024,
           'B1024_six_float_geometry_clear_bytes': 96 * 750 * 1024 * 6 * 8,
           'public_engine_alias': 'd4503b9e42a7f8f8555d128b437bb45b3193cf75',
           'selected_research_scene_path': True,
           'scope': 'Avoid stores of unused sample coordinates and barycentrics while refreshing negative face invalidation tags on every update. Active mapping, original force/moment/facet-cone gates, full output and reset/release isolation are unchanged. CPU/GPU independent controls, B4/B1024 complete histories and retained fields pass. Final B1024 original-clearing and masked max/full variants share one allocation, rigid control, conditions, source and three warmed repeats; all masked replicate times lie below all cleared replicates. The allocation was preempted after all five measured scene tasks, both scene comparisons and max conformance completed; only the missing read-only full conformance is completed in a separate CPU job with exact source/input bindings. The preempted state and absent returncode marker remain unchanged. This saves bandwidth, not allocated capacity. Old clearing remains an explicit research timing control. Raw rates include rejected loads and do not qualify physical pressure, shell/near-field or friction-history accuracy, nor a public automatic estimator.',
           'whole_goal_complete': False}
terminal_path = N / 'meta/hard-sample-bandwidth-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files = [SCRIPT, HELPER, HELPER_SOURCE, terminal_path, N / 'meta/hard-sample-bandwidth-launch-v1.json',
         N / 'meta/publish-hard-sample-bandwidth-before-preemption-v1.py']
external = set()
for campaign in campaigns:
    files.extend(sources(campaign) + terminal(campaign))
    files.extend((N / campaign).glob('group*/**/*.jsonl'))
    files.extend((N / campaign).glob('group*/**/*.jsonl.gz'))
    external.update((N / campaign).glob('group*/**/*.npz'))
    external.update((N / campaign).glob('group*/**/*.obj'))
large = {path for path in files if path.stat().st_size > 2 * 1024 * 1024}
files = [path for path in files if path not in large]
external.update(large)
packet = freeze('hard-transfer-sample-bandwidth-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Inactive-sample bandwidth and paired complete stress scenes\n\n' + receipt['scope'] +
    '\n\nReplay the captured task groups with their captured workers, public engine alias `d4503b9e4`, the pinned scientific Python, `rtx-mid`, container `1_26`, and fresh explicit data-side outputs/private caches. The final paired task order interleaves original-clearing and masked max/full modes before the shared pure-rigid control. Its measured tasks finish before scheduler preemption; the read-only full comparison is completed by the separately recorded reinspection task without rerunning dynamics or changing original outputs. Use `--masked-samples` for the component control; the original clearing baseline remains the constructor default for other consumers. Faces below zero invalidate their geometry storage. Full fields, trajectories, histories and generated meshes remain externally hashed on data. Logical allocated capacity is unchanged; store-byte accounting is distinct from measured device DRAM traffic. All original rejection/precision gates and physical limitations remain intact.\n')
with (N / 'meta/hard-sample-bandwidth-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
