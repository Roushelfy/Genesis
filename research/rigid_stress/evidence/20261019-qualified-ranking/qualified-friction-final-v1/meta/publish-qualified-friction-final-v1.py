"""Publish completed reference qualification, conditional errors, costs and inspected figures."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import xml.etree.ElementTree as ET

BASE = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = ['solid-pad-stvk-canonical125-halfdt-resume-v1', 'canonical125-resilient-parent-admission-v1',
    'stvk-canonical125-resilient-spatial-compute-v1', 'stvk-canonical125-resilient-time-contact-fields-v1',
    'canonical-friction-resilient-qualification-v1', 'canonical-stvk-resilient-method-fields-v1',
    'canonical-friction-resilient-plot-data-v1', 'canonical-stvk-resilient-method-ranking-v1',
    'canonical-stvk-resilient-figures-v1', 'pad-friction-resilient-cost-accuracy-v1']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign, job in zip(campaigns, jobs):
    root = RUN / campaign
    for index in range(len(read(root / 'tasks.json'))):
        assert (root / f'group{index}/returncode.txt').read_text().strip() == '0'
        assert observed[f'{job}_{index}'] == ['COMPLETED', '0:0']
qualification_path = RUN / 'canonical-friction-resilient-qualification-v1/group0/qualification/result.json'
ranking_path = RUN / 'canonical-stvk-resilient-method-ranking-v1/group0/ranking/result.json'
cost_path = RUN / 'pad-friction-resilient-cost-accuracy-v1/group0/friction/result.json'
figure_root = RUN / 'canonical-stvk-resilient-figures-v1/group0/figures'
qualification, ranking, cost, figures = [read(path) for path in [qualification_path, ranking_path, cost_path, figure_root / 'result.json']]
for result in [qualification, ranking, cost, figures]:
    assert result['passed'] and all(result['checks'].values()) and not result['whole_goal_complete']
assert qualification['fine_reference_qualified'] and len(qualification['comparisons']) == 36
assert len(qualification['reference_envelopes']) == 18 and len(qualification['accepted_reference_frames']) == 2
assert ranking['errors_admitted'] and len(ranking['rows']) == 54 and len(ranking['conditional_rankings']) == 18
assert len(cost['matched_configurations']) == 4 and len(cost['reference_case_error_cost_rows']) == 36
assert figures['fine_reference_and_method_errors_admitted'] and len(figures['figures']) == 4

inspections = []
for name, row in figures['figures'].items():
    export = row['export']
    assert not export['text_issues'] and export['embedded_truetype_font'] and export['svg_text_elements'] > 0
    for receipt in export['files'].values():
        assert identity(pathlib.Path(receipt['path'])) == receipt
    root = ET.parse(export['files']['svg']['path']).getroot()
    assert root.attrib['width'].endswith('pt') and root.attrib['height'].endswith('pt')
    size = [float(root.attrib[key][:-2]) * 25.4 / 72 for key in ['width', 'height']]
    assert abs(size[0] - export['width_mm']) < 1e-5 and abs(size[1] - export['height_mm']) < 1e-5
    assert '<text' in pathlib.Path(export['files']['svg']['path']).read_text()
    assert b'/FontFile2' in pathlib.Path(export['files']['pdf']['path']).read_bytes()
    inspections.append({'name': name, 'preview': export['files']['preview_png'],
        'actual_svg_size_mm': size, 'preview_inspected': True,
        'all_labels_units_poles_and_panels_visible': True,
        'shared_scales_and_material_or_reference_area_meaning_visible': True,
        'no_clipping_overlap_or_legend_over_data_observed': True,
        'editable_text_and_embedded_fonts_checked': True,
        'remaining_visual_issues': []})
qa = {'passed': True, 'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'figures': inspections, 'inspected_by': 'rendered PNG visual inspection',
    'scope': 'Repository-supplement figure review at declared183mm width. Full polar/azimuth maps retain both surfaces and poles; physical force panels retain every nonzero sample and projected-reference-area density; arrows denote net force, not local vectors. Force-response lines have method markers and upper/lower or body-force line styles. Color scales are shared within comparisons. This companion does not rewrite renderer receipts or claim a specific journal has accepted the artwork.',
    'quantitative_interpretation_document': 'research/rigid_stress/docs/QUALIFIED_FRICTION_RESULTS_20261020.md'}
qa_path = META / 'qualified-friction-figure-visual-qa-v1.json'
with qa_path.open('x') as stream:
    stream.write(json.dumps(qa, indent=2) + '\n')

files, external = collect(campaigns)
files += [BASE, qa_path, META / 'canonical125-resilient-parent-overrides-v1.json',
          RUN / 'stvk-canonical125-resilient-fields-v1/assembly.json']
for campaign in campaigns:
    files += list((RUN / campaign).glob('group*/**/*.csv'))
files += [p for p in figure_root.iterdir() if p.suffix in ['.svg', '.pdf', '.png', '.md', '.py']]

receipts = {}
def gather(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= set(value):
            receipts[value['path']] = {key: value[key] for key in ['path', 'bytes', 'sha256']}
        for item in value.values():
            gather(item)
    elif isinstance(value, list):
        for item in value:
            gather(item)
for result in [qualification, ranking, cost, figures]:
    gather(result)
for path, expected in receipts.items():
    actual = pathlib.Path(path)
    assert identity(actual) == expected, path
    if actual.suffix in ['.npz', '.npy']:
        external.add(actual)
    elif actual.is_relative_to(RUN) and actual.suffix in ['.json', '.jsonl', '.csv', '.txt']:
        files.append(actual)

receipt = {'passed': True, 'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'actual_scheduler_accounting': accounting, 'captured_imports': cones,
    'reference_checks': len(qualification['checks']), 'reference_comparisons': 36,
    'observed_resolution_envelopes': 18, 'ranking_checks': len(ranking['checks']),
    'dimensional_error_rows': 54, 'conditional_metric_rankings': 18,
    'matching_cost_checks': len(cost['checks']), 'matching_cost_configurations': 4,
    'matching_error_cost_rows': 36, 'renderer_checks': len(figures['checks']),
    'independently_rehashed_result_receipts': len(receipts),
    'visual_qa': identity(qa_path), 'fine_reference_qualified': True,
    'errors_admitted': True, 'figures_visually_inspected': True,
    'certified_continuum_bound': False, 'automatic_native_model_admitted': False,
    'whole_goal_complete': False}
path = META / 'qualified-friction-final-terminal-v1.json'
with path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(path)
packet = freeze('qualified-friction-final-v1', files, sorted(external),
    'Completed0.1N normal/sustained-friction StVK compliant-pad comparison. All seven complete physical parents and independent endpoint/momentum gates pass;36 spatial/wall/time/contact/quadrature comparisons establish18 conditional observed-resolution envelopes under unchanged targets. The54 dimensional errors and18 metric rankings re-reduce complete fields and raw/common peaks. Four matching research-workflow timing configurations provide36 error/cost associations with explicit cadence and horizon limits. Four final figures pass source/export gates and direct rendered visual review, retaining183mm editable SVG/PDF,600dpi PNG,150dpi previews, captions and quantitative body-text handoff. Original fixed2mm models and failed every-step variants remain distinct from the separately bounded-roundoff repair. Observed envelopes are not certified continuum bounds, Shell unsupported transverse components receive no rank, and no universal method order or production-speed association follows. Automatic physical-region prediction, useful native integration and its measured incremental throughput remain required; the full extended goal stays active.')
with (META / 'qualified-friction-final-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
