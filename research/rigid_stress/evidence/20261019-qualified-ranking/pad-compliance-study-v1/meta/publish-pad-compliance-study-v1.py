"""Freeze actual coupled-compliance contrasts, independent controls and inspected pressure maps."""
import csv
import datetime
import json
import pathlib
import subprocess
import xml.etree.ElementTree as ET

BASE = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))
campaigns = ['pad-compliance-study-v1', 'pad-compliance-controls-v1', 'pad-compliance-figure-v1']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
        assert (RUN / campaign / f'group{group}/returncode.txt').read_text().strip() == '0'
study = read(RUN / 'pad-compliance-study-v1/group0/study/result.json')
control = read(RUN / 'pad-compliance-controls-v1/group0/control/result.json')
quadrature = read(RUN / 'pad-compliance-controls-v1/group1/quadrature/result.json')
figure_root = RUN / 'pad-compliance-figure-v1/group0/figure'
figure = read(figure_root / 'result.json')
assert all(result['passed'] for result in [study, control, quadrature, figure])
assert len(study['cases']) == 8 and len(study['records']) == 48 and len(study['comparisons']) == 32
assert len(quadrature['rows']) == 24
for record in study['records']:
    assert record['search_contains_actual_elastic_compression']
    assert record['own_pad_law_relative_error'] <= 1e-7
    assert record['equation_relative_error'] <= 1e-7 and record['balance_correction_relative'] <= 1e-8
    if record['mode'] == 'coupled':
        assert record['full_elastic_pad_law_relative_error'] <= 1e-7
assert all(not row['search_contains_actual_elastic_compression'] for row in control['records'][:3])
assert all(row['search_contains_actual_elastic_compression'] for row in control['records'][3:])
assert [row['search_radius_m'] for row in control['records']] == [0.001] * 3 + [0.003] * 3
independent = control['independent_controls']
assert independent['full_elastic_fixed_point_force_relative_error'] <= 1e-7
assert independent['split_source_nodal_force_relative_error'] <= 1e-7
assert independent['same_nominal_baseline_stress_relative_error'] <= 1e-6
assert not figure['rendered_text_issues']
for identity_row in figure['exports'] + [figure['preview']]:
    assert identity(pathlib.Path(identity_row['path'])) == identity_row
svg = figure_root / 'pad-compliance-pressure.svg'
root = ET.parse(svg).getroot()
actual_size = [float(root.attrib[key][:-2]) * 25.4 / 72 for key in ['width', 'height']]
assert abs(actual_size[0] - 183) < 1e-5 and abs(actual_size[1] - 142) < 1e-5
assert '<text' in svg.read_text() and b'/FontFile2' in (figure_root / 'pad-compliance-pressure.pdf').read_bytes()
qa = {'passed': True, 'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'preview': figure['preview'], 'actual_svg_size_mm': actual_size, 'preview_inspected': True,
    'all_six_panels_and_physical_units_readable': True, 'no_clipping_or_overlap_observed': True,
    'separate_material_rows_share_scales_within_each_row': True,
    'all_positive_upper_pressure_samples_visible': True,
    'editable_text_and_embedded_fonts_checked': True, 'remaining_visual_issues': [],
    'interpretation': 'Nominal soft-pad pressure patterns agree closely; the flexible-link/stiff-pad coupled model redistributes pressure toward the finite-land edge. The diagonal model stays close to the rigid predictor. Full two-sided errors and quadrature sensitivity determine the quantitative comparison.'}
qa_path = META / 'pad-compliance-figure-visual-qa-v1.json'
with qa_path.open('x') as stream:
    stream.write(json.dumps(qa, indent=2) + '\n')
table = META / 'pad-compliance-model-errors-v1.csv'
rows = [{**row, **study['cases'][row['case']]} for row in study['comparisons']]
with table.open('x', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
files, external = collect(campaigns)
files += [BASE, qa_path, table, META / 'pad-compliance-cases-v1.json',
    META / 'pad-compliance-control-cases-v1.json', META / 'contact-region-nature-style-v1.py']
files += [p for p in figure_root.iterdir() if p.suffix in ['.svg', '.pdf', '.png', '.md']]
start = read(RUN / 'pad-compliance-study-v1/group0/study/start.json')
for key in ['mesh', 'seed_operator', 'baseline_frame']:
    row = start[key]; path = pathlib.Path(row['path']); assert identity(path) == row; external.add(path)
for campaign, module in [('pad-compliance-controls-v1', 'study_pad_compliance.py'),
                         ('pad-compliance-controls-v1', 'compare_pad_compliance.py'),
                         ('pad-compliance-figure-v1', 'plot_pad_compliance.py')]:
    relative = pathlib.Path('research/rigid_stress/qipc_comparison') / module
    assert (RUN / campaign / 'analysis-source' / relative).read_bytes() == (WORK / relative).read_bytes()
receipt = {'passed': True, 'actual_scheduler_accounting': accounting, 'captured_imports': cones,
    'material_load_asymmetry_cases': 8, 'actual_model_fields': 48, 'model_error_rows': 32,
    'separate_quadrature_sensitivity_rows': 24, 'independent_controls': independent,
    'actual_small_search_refusals_and_expansion': True, 'visual_qa': identity(qa_path),
    'physical_reference_qualified': False, 'production_automatic_model_admitted': False,
    'thickness_and_moving_friction_studies_complete': False, 'whole_goal_complete': False}
path = META / 'pad-compliance-study-terminal-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(path)
packet = freeze('pad-compliance-study-v1', files, sorted(external),
    'Eight actual stiffness/load/asymmetric-pad cases compare rigid, diagonal and coupled free-link compliance at centroid and degree6 quadrature:48 full fields,32 model contrasts and24 separate sampling contrasts. Coupled pressure satisfies its complete elastic pad law; approximate models retain their measured full-law residuals. Independent full-elastic fixed-point and split-source controls agree with the condensed solution, and an actual1mm search misses compression before expanding to3mm. The nominal rigid approximation is close, while flexible-link/stiff-pad pressure and field errors reach24.47% and6.68%; diagonal compliance barely improves them. Contact sampling itself changes coupled full stress by1.94–2.82%, and remains distinct from model error. Six-panel pressure maps pass direct rendered review and exact183x142mm export checks. Operators and full arrays stay on data. This is one coarse linear-elastic geometry with static normal pads; thickness, changing nonconvex/friction histories, continuum qualification, production native integration and incremental measured throughput remain required.')
with (META / 'pad-compliance-study-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
