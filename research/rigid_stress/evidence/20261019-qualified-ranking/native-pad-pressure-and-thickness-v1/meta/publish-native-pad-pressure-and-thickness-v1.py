"""Freeze native pressure-stage conformance and the completed physical thickness contrast."""
from pathlib import Path

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = ['native-pad-newton-inputs-v1', 'native-pad-newton-v1', 'native-pad-newton-fields-v1',
    'pad-thickness-study-v1', 'native-pad-thickness-inputs-v1', 'native-pad-thickness-v1',
    'native-pad-thickness-fields-v1', 'pad-thickness-figure-draft-v1', 'pad-thickness-figure-v1']
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,MaxRSS'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        assert (RUN / campaign / f'group{group}/returncode.txt').read_text().strip() == '0'
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
native = []
for campaign in ('native-pad-newton-v1', 'native-pad-thickness-v1'):
    for path in sorted((RUN / campaign).glob('group*/*/result.json')):
        result = read(path)
        assert result['passed'] and len(result['rows']) == 2
        for row in result['rows']:
            assert max(row['pressure_relative_errors']) <= 1e-6
            assert max(row['independent_pad_law_errors']) <= 1e-7
            assert max(row['wrench_errors']) <= 1e-8 and row['maximum_gradient'] <= 1e-10
            assert all(row[key] for key in ('warm_unchanged_exact', 'release_exact',
                'failure_isolation_exact', 'partial_recovery_preserves_others_exact'))
        native.append({'receipt': identity(path), 'backend': result['backend'], 'rows': result['rows']})
assert sum(len(row['pressure_relative_errors']) for case in native for row in case['rows']) == 48
fields = []
for campaign in ('native-pad-newton-fields-v1', 'native-pad-thickness-fields-v1'):
    for path in sorted((RUN / campaign).glob('group*/fields/result.json')):
        result = read(path)
        assert result['passed']
        assert max(row['pressure_force_relative'] for row in result['backend_errors']) <= 1e-10
        for row in result['rows']:
            assert max(row['complete_stress_relative_error'], row['nodal_force_relative_error'],
                       *row['peak_relative_errors'].values()) <= 1e-6
            assert row['equation_relative_error'] <= 1e-7 and row['balance_relative_error'] <= 1e-8
        fields.extend(result['rows'])
assert len(fields) == 48
for group in range(2):
    result = read(RUN / f'pad-thickness-study-v1/group{group}/study/result.json')
    assert result['passed'] and len(result['records']) == 12
    for row in result['records']:
        assert row['search_contains_actual_elastic_compression']
        assert row['own_pad_law_relative_error'] <= 1e-7 and row['wrench_relative_error'] <= 1e-8
        if row['mode'] == 'coupled':
            assert row['full_elastic_pad_law_relative_error'] <= 1e-7
    sampling = read(RUN / f'native-pad-thickness-inputs-v1/group{group}/sampling/result.json')
    assert sampling['passed'] and len(sampling['rows']) == 6
figure = read(RUN / 'pad-thickness-figure-v1/group0/figure/result.json')
qa = read(META / 'pad-thickness-final-visual-qa-v1.json')
assert figure['passed'] and len(figure['model_errors']) == 12 and qa['passed']
assert figure['editable_source_matches_inspected_draft'] and qa['preview_and_svg_bitwise_match_inspected_source']
binding = read(META / 'native-pad-pressure-final-source-v1.json')
assert binding['passed'] and len(binding['files']) == 6 and all(row['ast_identical'] for row in binding['files'])
files, external = collect(campaigns)
files.append(BASE)
names = ['native-pad-newton-inputs-tasks-v1.json', 'native-pad-newton-tasks-v1.json',
    'native-pad-newton-fields-tasks-v1.json', 'pad-thickness-cases-v1.json',
    'pad-thickness-study-tasks-v1.json', 'native-pad-thickness-inputs-v1-tasks.json',
    'native-pad-thickness-v1-tasks.json', 'native-pad-thickness-fields-v1-tasks.json',
    'pad-thickness-figure-contract-v1.json', 'pad-thickness-figure-draft-tasks-v1.json',
    'pad-thickness-draft-visual-qa-v1.json', 'pad-thickness-figure-tasks-v1.json',
    'pad-thickness-final-visual-qa-v1.json', 'native-pad-pressure-final-source-v1.json',
    'contact-region-nature-style-v1.py']
files.extend(META / name for name in names)
for campaign in ('pad-thickness-figure-draft-v1', 'pad-thickness-figure-v1'):
    files.extend(path for path in (RUN / campaign / 'group0/figure').iterdir()
                 if path.suffix in ('.svg', '.pdf', '.png', '.md', '.csv'))
for item in figure['source_receipts']:
    path = Path(item['path'])
    assert identity(path) == item
    if path.suffix == '.npz':
        external.add(path)
    else:
        files.append(path)
for row in fields:
    item = row['reference']
    path = Path(item['path'])
    assert identity(path) == item
    external.add(path)
preparation = read(RUN / 'native-pad-newton-inputs-v1/group0/inputs/start.json')
for item in preparation['inputs']:
    path = Path(item['path'])
    assert identity(path) == item
    external.add(path)
receipt = {'passed': True, 'actual_scheduler_accounting': accounting, 'captured_imports': cones,
    'native_pressure_cases': 48, 'independent_complete_fields': 48,
    'maximum_complete_stress_relative_error': max(row['complete_stress_relative_error'] for row in fields),
    'maximum_raw_peak_relative_error': max(value for row in fields for value in row['peak_relative_errors'].values()),
    'native_stage_controls': native, 'new_physical_thickness_fields': 24,
    'thickness_model_errors': figure['model_errors'], 'visual_qa': qa,
    'cpu_prepared_compliance_geometry_and_cold_guesses_remain': True,
    'production_automatic_model_admitted': False, 'production_tolerance_changed': False,
    'controlled_native_throughput_admitted': False, 'whole_goal_complete': False}
p = META / 'native-pad-pressure-and-thickness-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packet = freeze('native-pad-pressure-and-thickness-v1', files, sorted(external),
    'The reduced physical pad pressure stage now runs in Quadrants with batched Newton, matrix-free preconditioned conjugate gradients, line search and separate current-frame acceptance. CPU/GPU pressure outputs reproduce48 supplied physical cases and48 independently solved complete elastic fields across250/500/1000um walls. Exact separated release, unchanged warm repeats, invalid-spring isolation and selective recovery pass atB8/B2. Operators, geometry and cold rigid guesses remain CPU-prepared validation inputs; the public automatic producer, native operator preparation, friction/lifecycle integration and controlled module/scene throughput remain required. Two new physical thicknesses retain24 rigid/diagonal/coupled normal fields with independent operators, full elasticity and omitted-face checks. AtE10GPa and pad stiffness1e10N/m3, rigid pressure/stress errors are15.52%/5.62%,2.69%/0.696%,0.489%/0.0965% for250/500/1000um. Errors compare models within each own thickness, without subtracting fields on different physical domains or claiming continuum qualification. Source quadrature controls remain separate. The inspected183x142mm pressure-map SVG/preview is bitwise preserved in final PDF/600dpi PNG export. All numerical data stay at hashed data-side paths, and production defaults are unchanged.')
with (META / 'native-pad-pressure-and-thickness-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
