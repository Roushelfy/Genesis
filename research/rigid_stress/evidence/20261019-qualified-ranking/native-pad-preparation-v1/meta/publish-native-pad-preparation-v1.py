"""Freeze native condensation controls, basis construction and independently checked cold solves."""
from pathlib import Path

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))
campaigns = ['native-pad-compliance-v1', 'native-pad-basis-v1', 'native-pad-basis-fields-v1']
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,MaxRSS'], text=True)
observed = {p[0]: p[1:3] for line in accounting.splitlines() if len(p := line.split('|')) >= 3}
for c, job in zip(campaigns, jobs):
    for i in range(len(read(RUN / c / 'tasks.json'))):
        assert (RUN / c / f'group{i}/returncode.txt').read_text().strip() == '0'
        assert observed[f'{job}_{i}'] == ['COMPLETED', '0:0']
operators, native, fields, backend_errors = [], [], [], []
for path in sorted((RUN / campaigns[0]).glob('group*/*/result.json')):
    result = read(path)
    assert result['passed'] and len(result['rows']) == 2
    for row in result['rows']:
        assert max(row['relative_compliance_error'], row['relative_factor_error'],
                   row['frame_material_error']) <= 1e-10
        assert row['native_reciprocity_error'] <= 1e-7
    operators.extend(result['rows'])
for path in sorted((RUN / campaigns[1]).glob('group*/*/result.json')):
    result = read(path)
    assert result['passed'] and result['native_basis_and_cold_initialization']
    for row in result['rows']:
        assert row['native_mapping_relative_error'] <= 1e-10 and row['maximum_gradient'] <= 1e-10
        assert max(row['pressure_relative_errors']) <= 1e-6
        assert max(row['independent_pad_law_errors']) <= 1e-7
        assert max(row['wrench_errors']) <= 1e-8
        assert all(row[key] for key in ('warm_unchanged_exact', 'release_exact',
            'failure_isolation_exact', 'partial_recovery_preserves_others_exact'))
    native.extend(result['rows'])
assert len(operators) == 4 and sum(len(row['pressure_relative_errors']) for row in native) == 48
for path in sorted((RUN / campaigns[2]).glob('group*/fields/result.json')):
    result = read(path)
    assert result['passed']
    for row in result['rows']:
        assert max(row['complete_stress_relative_error'], row['nodal_force_relative_error'],
                   *row['peak_relative_errors'].values()) <= 1e-6
        assert row['equation_relative_error'] <= 1e-7 and row['balance_relative_error'] <= 1e-8
    assert max(row['pressure_force_relative'] for row in result['backend_errors']) <= 1e-10
    fields.extend(result['rows'])
    backend_errors.extend(result['backend_errors'])
assert len(fields) == 48
files, external = collect(campaigns)
files += [BASE, *[META / f'{c}-tasks.json' for c in campaigns]]
for row in native:
    path = Path(row['inputs']['path'])
    assert identity(path) == row['inputs']
    external.add(path)
for row in fields:
    path = Path(row['reference']['path'])
    assert identity(path) == row['reference']
    external.add(path)
binding = []
for name in ['native_pad_operator', 'native_pad_newton', 'check_native_pad_newton', 'check_native_pad_compliance']:
    relative = Path('research/rigid_stress/qipc_comparison') / f'{name}.py'
    captured_path = RUN / campaigns[0] / 'analysis-source' / relative
    final_path = WORK / relative
    assert captured_path.read_bytes() == final_path.read_bytes()
    files.append(captured_path)
    binding.append({'final': identity(final_path), 'captured': identity(captured_path)})
receipt = {'passed': True, 'actual_scheduler_accounting': accounting, 'captured_imports': cones,
    'final_sources_match_captured_bytes': binding, 'native_condensation_controls': operators,
    'native_basis_pressure_cases': 48, 'independent_full_fields': 48,
    'maximum_complete_stress_relative_error': max(row['complete_stress_relative_error'] for row in fields),
    'maximum_raw_peak_relative_error': max(value for row in fields for value in row['peak_relative_errors'].values()),
    'native_pressure_controls': native, 'backend_errors': backend_errors,
    'large_physical_operator_still_cpu_prepared': True,
    'geometry_source_maps_still_supplied': True, 'production_automatic_model_admitted': False,
    'controlled_native_throughput_admitted': False, 'whole_goal_complete': False}
p = META / 'native-pad-preparation-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packet = freeze('native-pad-preparation-v1', files, sorted(external),
    'Native balanced unit-load condensation, mass-gauge alignment, compliance Cholesky, sparse source mapping and opposing-pad cold initialization. Four compact CPU/GPU operator controls agree with independent full elasticity under arbitrary nodal directions and a translated/rotated material-scaled variant. All48 physical pressure cases across250/500/1000um walls, q0/q6 andB8/B2 pass with native factor/map/cold preparation, including warm repeats, exact release, invalid-input isolation and selective recovery. Actual pressures drive48 independent complete-field checks. The large834-node compliance and geometry maps are still supplied validation inputs; compact native condensation is not a measured large-mesh preparation result. Native conservative cold starts cost more Newton updates than supplied rigid guesses. This is numerical component validation, not completed public automatic geometry/friction/lifecycle integration or controlled production throughput. Original defaults and physical admission targets remain unchanged.')
with (META / 'native-pad-preparation-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
