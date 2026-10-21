"""Freeze measured native input optimization and matched implementation controls."""
from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
O = Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/hard_contact_optimized')
campaigns = ['hard-input-profile-v1', 'hard-input-grouping-v1', 'hard-input-precomputed-v1', 'hard-input-precomputed-v2', 'hard-input-precomputed-v3']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P', '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    failed = campaign in ['hard-input-precomputed-v1', 'hard-input-precomputed-v2']
    for group, tasks in enumerate(read(root / 'tasks.json')):
        assert observed[f'{job}_{group}'] == (['FAILED', '1:0'] if failed else ['COMPLETED', '0:0'])
        assert (root / f'group{group}/returncode.txt').read_text().strip() == ('1' if failed else '0')
        if not failed:
            for task in tasks:
                result = read(root / f'group{group}' / task['name'] / 'result.json')
                assert result.get('passed', result.get('execution_completed', False))
matched = []
final = RUN / 'hard-input-precomputed-v3'
for module in ['live_hertz_inputs', 'check_hertz_grouping', 'compare_live_hertz_inputs', 'compare_hertz_implementations', 'check_precomputed_surface', 'check_hertz_regions']:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    executed = final / 'analysis-source' / relative
    assert executed.read_bytes() == (WORK / relative).read_bytes()
    matched.append(identity(executed))
for relative in ['genesis/engine/solvers/rigid/stress/curvature.py', 'genesis/engine/solvers/rigid/stress/hertz_regions.py']:
    current = identity(O / relative)
    expected = next(row for row in read(final / 'source.json')['files'] if row['path'] == relative)
    assert all(current[key] == expected[key] for key in ['bytes', 'sha256'])
    assert (O / relative).read_bytes() == (WORK / relative).read_bytes()
    matched.append(current)
rows = []
for campaign in ['hard-input-profile-v1', 'hard-input-grouping-v1', 'hard-input-precomputed-v3']:
    for i, batch in enumerate([4, 64]):
        inputs = read(RUN / campaign / f'group{i}/inputs/result.json')
        comparison = read(RUN / campaign / f'group{i}/comparison/result.json')
        assert comparison['motion_matches']
        audit = inputs['rollouts'][0]['inputs']
        assert max(audit['maximum_force_discrepancy_n']) < 1e-10
        assert max(audit['maximum_moment_discrepancy_nm']) < 1e-10
        assert all(not r['research_audit_in_timing'] for r in inputs['rollouts'][1:])
        if campaign != 'hard-input-profile-v1':
            assert all(v == 0 for row in inputs['grouping_errors'] for v in row)
        rows.append({'campaign': campaign, 'envs': batch,
            'input_scene_seconds': comparison['timing']['inputs'],
            'pure_rigid_seconds': comparison['timing']['rigid'],
            'added_scene_seconds': comparison['incremental_scene_step_seconds'],
            'aggregate_scene_env_steps_per_second': batch / comparison['timing']['inputs']['median_scene_step_seconds'],
            'loaded_steps': sum(comparison['loaded_environment_steps']),
            'valid_loaded_steps': sum(comparison['valid_loaded_environment_steps']),
            'source_status_counts': comparison['source_status_counts'],
            'phase_profile': inputs['phase_profile'],
            'curvature_build_seconds_including_first_compile': inputs['curvature_build_seconds_including_first_compile']})
controls = []
for i in range(2):
    surface = read(final / f'group{i}/surface/result.json')
    regions = read(final / f'group{i}/regions/result.json')
    conformance = read(final / f'group{i}/conformance/result.json')
    grouping = read(final / f'group{i}/grouping_conformance/result.json')
    assert all(x['passed'] for x in [surface, regions, conformance, grouping])
    assert len(regions['region_cases']) == 12
    controls.append({'surface': surface, 'regions': regions, 'conformance': conformance, 'grouping_conformance': grouping})
receipt = {'passed': True, 'remote_goal_commit': 'c1c82eea667f986b7cf494eda5230904a1a752a7',
    'scheduler_accounting': accounting, 'current_sources_match_executed': matched,
    'rows': rows, 'controls': controls, 'whole_goal_complete': False,
    'scope': 'Profile-driven native contact-input optimization: parallel slot clearing, actual-contact-count deterministic region grouping, and shared fixed-geometry barycentric preprocessing. Matched B4/B64 independently evolving Franka inputs retain full-history geometric validity and motion, complete sampled outputs, and exact every-step group/force/moment controls. Intrusive synchronized phases and primary warmed inclusive scene timings are separate. No elastic recovery, maximum/full stress output or new physical-pressure/friction admission is inferred from input-only timings.',
    'failed_test': 'The first precomputed-surface control omitted batch_fixed_verts=True for per-environment fixed-mesh poses. Both groups fail before query execution. The second campaign corrects that fixture and completes numerical controls/timing, but fails exact equality of research probe normals after reassociating the general curvature barycentric product. The final candidate restores the original general-curvature arithmetic and limits cached transforms to nearest-face search; all original conformance assertions remain unchanged.',
    'next': 'Continue collision provenance and continuous-contact cache/controlled fallback, fusion and useful larger-batch tuning with numerical controls. Preserve finite-face and source-moment rejection, friction history, shell correction and actual complete max/full scene work. No pad expansion or default Newton-PCG integration.'}
terminal_path = RUN / 'meta/hard-input-optimization-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path, RUN / 'meta/hard-input-hardware-v1.json', RUN / 'meta/hard-input-source-counts-v1.json'])
for campaign in campaigns:
    files.extend((RUN / campaign).glob('group*/**/*.jsonl.gz'))
    external.update((RUN / campaign).glob('group*/**/*.obj'))
packet = freeze('hard-input-optimization-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text('# Native hard-contact input profiling and optimization\n\n' + receipt['scope'] + '\n\nThe manifest retains terminal scheduler state, exact task/import/source inventories, the initial fixture failure and second candidate numerical failure, full compressed audit histories and final numerical/performance receipts. Complete arrays and generated meshes stay hashed on data; private caches were compute-node local. The prior friction-history work remains separate and unfinished.\n')
with (RUN / 'meta/hard-input-optimization-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet, indent=2))
