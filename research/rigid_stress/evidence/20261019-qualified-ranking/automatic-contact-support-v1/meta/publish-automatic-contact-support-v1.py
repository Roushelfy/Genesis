"""Freeze the first actual CPU support/pressure/elastic contrasts and inspected figure."""
import ast
import datetime
import json
import pathlib
import subprocess

RUN = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
META = RUN / 'meta'
HELPER = META / 'publish-canonical125-resilient-recovery-v1.py'
exec(compile(HELPER.read_text().split('\nimport ast\n', 1)[0], str(HELPER), 'exec'))
read = lambda path: json.loads(path.read_text())

success = [('automatic-contact-support-cpu-v4', '0', 'study'),
           ('automatic-contact-support-cpu-v4', '1', 'study'),
           ('distributed-pad-transfer-cpu-v1', '0', 'comparison'),
           ('contact-region-figure-v1', '0', 'figure')]
for campaign, group, name in success:
    root = RUN / campaign
    assert read(root / f'group{group}/{name}/result.json')['passed']
    assert (root / f'group{group}/returncode.txt').read_text().strip() == '0'

campaigns = ['automatic-contact-support-cpu-v1', 'automatic-contact-support-cpu-v2',
             'automatic-contact-support-cpu-v3', 'automatic-contact-support-cpu-v4',
             'distributed-pad-transfer-cpu-v1', 'contact-region-figure-v1']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, group, name in success:
    job = str(read(RUN / campaign / 'submission.json')['job']).split(';')[0]
    assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
for job, group in [(jobs[0], '0'), (jobs[0], '1'), (jobs[1], '0'), (jobs[1], '1'), (jobs[2], '0')]:
    assert observed[f'{job}_{group}'] == ['FAILED', '1:0']
assert observed[f'{jobs[2]}_1'] == ['COMPLETED', '0:0']

files = [pathlib.Path(__file__), HELPER, META / 'automatic-contact-support-tasks-v1.json',
         META / 'distributed-pad-transfer-tasks-v1.json', META / 'contact-region-figure-tasks-v1.json',
         META / 'contact-region-nature-style-v1.py', META / 'contact-region-figure-visual-qa-v1.json']
external = set()
cones = {}
for campaign in campaigns:
    root = RUN / campaign
    inventory = {row['path']: row for row in read(root / 'analysis-files.json')}
    queue = [task['module'] for group in read(root / 'tasks.json') for task in group]
    seen = set()
    while queue:
        module = queue.pop()
        relative = pathlib.Path(*module.split('.')).with_suffix('.py')
        path = root / 'analysis-source' / relative
        if str(relative) in seen or not path.is_file():
            continue
        seen.add(str(relative))
        actual = identity(path)
        assert all(actual[key] == inventory[str(relative)][key] for key in ['bytes', 'sha256'])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                base = '.'.join(module.split('.')[:-node.level]) if node.level else ''
                target = '.'.join(part for part in [base, node.module] if part)
                if target.startswith('research.rigid_stress'):
                    queue.append(target)
                    queue.extend(target + '.' + item.name for item in node.names)
            elif isinstance(node, ast.Import):
                queue.extend(item.name for item in node.names if item.name.startswith('research.rigid_stress'))
    cones[campaign] = sorted(seen)
    files += captured(campaign, [str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in seen])
    files += terminal(campaign)
    external.update(root.glob('group*/**/*.npz'))

support = [read(RUN / f'automatic-contact-support-cpu-v4/group{g}/study/result.json') for g in [0, 1]]
for result in support:
    proof = result['reference_support']
    assert identity(pathlib.Path(proof['frame']['path'])) == proof['frame']
    external.add(pathlib.Path(proof['frame']['path']))
    for case in result['double_spreading']:
        for key in ['original', 'physical_state', 'actual_quadrature', 'mesh', 'face_maps']:
            row = case[key]
            assert identity(pathlib.Path(row['path'])) == row
            external.add(pathlib.Path(row['path']))
        path = pathlib.Path(case['capture_receipt']['path'])
        assert identity(path) == case['capture_receipt']
        files.append(path)
        original_run = pathlib.Path(case['original']['path']).parent.parent
        files += [original_run / name for name in ['start.json', 'result.json']]

for campaign, module in [('automatic-contact-support-cpu-v4', 'study_contact_regions.py'),
                         ('distributed-pad-transfer-cpu-v1', 'compare_distributed_pad_transfer.py'),
                         ('contact-region-figure-v1', 'plot_contact_region_study.py')]:
    relative = pathlib.Path('research/rigid_stress/qipc_comparison') / module
    assert (RUN / campaign / 'analysis-source' / relative).read_bytes() == (WORK / relative).read_bytes()
figure = RUN / 'contact-region-figure-v1/group0/figure'
visual = read(META / 'contact-region-figure-visual-qa-v1.json')
assert visual['passed'] and identity(pathlib.Path(visual['preview']['path'])) == visual['preview']
figure_result = read(figure / 'result.json')
assert not figure_result['rendered_text_issues'] and figure_result['svg_text_editable']
assert b'/FontFile2' in (figure / 'contact-pressure.pdf').read_bytes()
assert '<text' in (figure / 'contact-pressure.svg').read_text()
for row in figure_result['exports'] + [figure_result['preview']]:
    assert identity(pathlib.Path(row['path'])) == row
    files.append(pathlib.Path(row['path']))

transfer = read(RUN / 'distributed-pad-transfer-cpu-v1/group0/comparison/result.json')
receipt = {'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'passed': True,
    'actual_scheduler_accounting': accounting, 'captured_research_imports': cones,
    'successful_cpu_support_jobs': ['731403_0', '731403_1'], 'paired_elastic_job': '731405_0',
    'inspected_pressure_figure_job': '731411_0',
    'original_failures_retained': ['prefix_cumsum_force_error', 'old_frame_rotation_member_absent', 'outer_only_matching_refuses_inner_DPR_facets'],
    'native_engine_and_original_histories_unchanged': True,
    'same_load_cpu_native_stress_maximum_relative_difference': max(c['cpu_native_same_load_stress_relative_error'] for c in transfer['cases']),
    'conservative_transfer_stress_changes': [c['transfer_induced_stress_relative_change'] for c in transfer['cases']],
    'pdf_embedded_truetype_marker_present': True, 'visual_qa': identity(META / 'contact-region-figure-visual-qa-v1.json'),
    'fine_reference_qualified': False, 'automatic_native_model_admitted': False, 'whole_goal_complete': False}
p = META / 'automatic-contact-support-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
scope = ('First compact CPU contact-region study under the2026-10-20 extension. Analytic independent-column pressure/area/axis, source-partition and nonconvex loading/unloading controls pass; rigid-gap prediction is compared with an actual coarse physical normal state, while saved deformation only checks its pad law. '
         'Actual exported native quadrature identifies both boundary layers: the retained2mm DPR baseline transfers about42% of normal load to the inner boundary, and SCR redistributes about19–20% beyond the original active faces. Total wrench gates still pass. '
         'Conservative outer P2 transfer preserves force, moment and virtual work; paired independent elasticity matches original same-load native stress to1.06e-11 before measuring an8.70% transfer-induced complete-field change. This is not a continuum-error improvement claim. '
         'The common-scale editable pressure figure has complete nonzero polar support and passes rendered visual/text checks; original scalar-sum, old-format reader and outer-only matching failures remain intact. Full runtime fields/maps/quadrature remain externally hashed on data. '
         'No production engine, baseline source, submitted fine writer, physical law, tolerance or final ranking is changed. Coupled automatic compliance, SCR repair/attribution, native integration, measured tuning and the final fixed-radius comparison remain required; the extended goal is active.')
row = freeze('automatic-contact-support-v1', files, sorted(external), scope)
with (META / 'automatic-contact-support-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(row, indent=2) + '\n')
print(json.dumps(row))
