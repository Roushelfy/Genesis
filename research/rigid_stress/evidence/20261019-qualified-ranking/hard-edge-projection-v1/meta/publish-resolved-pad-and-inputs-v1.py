"""Freeze terminal resolved-pressure, source-input and fine-parent evidence."""
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
cones = {}


def sources(campaign):
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
    return captured(campaign, [str(pathlib.Path(path).relative_to('research/rigid_stress')) for path in seen])


def collect(campaigns):
    files = [SCRIPT, HELPER]
    external = set()
    for campaign in campaigns:
        root = RUN / campaign
        groups = read(root / 'tasks.json')
        for index in range(len(groups)):
            assert (root / f'group{index}/returncode.txt').is_file()
        files += sources(campaign) + terminal(campaign)
        files += [p for p in root.glob('group*/**/*.jsonl') if 'cache' not in p.parts and 'tmp' not in p.parts]
        external.update(root.glob('group*/**/*.npz'))
    return files, external


def bounded_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'tangent_relative_l2', 'membrane_relative_l2', 'bending_relative_l2',
                       'outer_tangent_relative_l2', 'inner_tangent_relative_l2',
                       'raw_mises_relative_change', 'raw_tension_relative_change',
                       'relative_l2_error', 'relative_difference'} and isinstance(item, (int, float)):
                yield key, item
            else:
                yield from bounded_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from bounded_values(item)


resolved = ['distributed-pad-native-control-v1', 'distributed-pad-native-control-v2',
            'distributed-pad-every-step-v1', 'distributed-pad-history-admission-v1',
            'distributed-pad-complete-fields-v1']
inputs = [f'pad-source-roundoff-cpu-v{i}' for i in range(1, 5)]
inputs += [f'native-wrench-attribution-v{i}' for i in range(1, 6)]
inputs += ['pad-roundoff-every-step-v1', 'dpr-roundoff-admission-v1', 'scr-roundoff-admission-v1']
fine = ['solid-pad-stvk-canonical125-final-resume-v3',
        'solid-pad-stvk-canonical125-contact-resume-v1',
        'canonical125-resilient-spatial-parent-admission-v1',
        'contact125-resilient-pair-parent-admission-v1']
jobs = [str(read(RUN / name / 'submission.json')['job']).split(';')[0] for name in resolved + inputs + fine]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {line.split('|')[0]: line.split('|')[1:3] for line in accounting.splitlines() if '|' in line}
for campaign in resolved + inputs + fine:
    root = RUN / campaign
    job = str(read(root / 'submission.json')['job']).split(';')[0]
    for index in range(len(read(root / 'tasks.json'))):
        code = (root / f'group{index}/returncode.txt').read_text().strip()
        assert observed[f'{job}_{index}'] == (['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0'])

control = read(RUN / 'distributed-pad-native-control-v2/group0/control/result.json')
assert control['passed'] and len(control['checks']) == 8 and all(control['checks'].values())
admission = read(RUN / 'distributed-pad-history-admission-v1/group0/admission/result.json')
assert admission['passed']
for case in admission['cases']:
    assert case['steps'] == 400 and case['mechanical_history_passed'] and case['all_windows_passed']
    assert case['all_stress_observations_valid']
    for field in case['fields']:
        assert field['adapter_passed'] and field['recovered_surface_relative_wrench_error'] <= 1e-7
metrics = []
for group in range(4):
    result = read(RUN / f'distributed-pad-complete-fields-v1/group{group}/comparison/result.json')
    values = list(bounded_values(result))
    assert values and all(abs(value) <= 1e-6 for key, value in values)
    metrics += values
files, external = collect(resolved)
receipt = {'passed': True, 'native_control_checks': control['checks'],
    'all_complete_material_column_metrics_below_1e_6': True,
    'maximum_complete_field_metric': max(abs(value) for key, value in metrics),
    'recovered_surface_wrench_bound_checked_separately': True,
    'captured_imports': {name: cones[name] for name in resolved},
    'actual_scheduler_accounting': accounting,
    'production_automatic_model_admitted': False, 'fine_reference_qualified': False,
    'whole_goal_complete': False}
p = META / 'resolved-pad-native-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packets = [freeze('resolved-pad-native-v1', files, sorted(external),
    'Actual Quadrants transfer of separately admitted physical outer-centroid pressures. Eight max/full/zero/refusal/recovery controls pass; the original fixed-radius replay is retained. A complete400-step normal/friction history and independent mechanical/source audit pass. Four q9/q25 endpoint material-column comparisons retain unsmoothed outer/inner/full3D fields, membrane/bending and common peaks, all below1e-6 numerical conformance. This is a B1 research adapter for the declared centroid quadrature, not automatic production pressure prediction or continuum-error qualification. Original engine and manual baselines are unchanged.')]

roundoff = read(RUN / 'pad-source-roundoff-cpu-v4/group0/control/result.json')
assert roundoff['passed']
for case in roundoff['cases']:
    assert case['baseline_correction_bitwise_preserved'] and case['frame_rotation_conformance_passed']
    assert case['finite_incompatible_load_rejected'] and case['already_admitted_sources_bitwise_preserved']
    assert case['force_change_bound_n'] <= case['physical_force_budget_n']
    assert case['maximum_absolute_principal_stress_perturbation_pa'] <= 1e-6
    assert case['elastic_equation_relative_residual'] <= 1e-7
histories = []
for group, method in enumerate(['dpr', 'scr']):
    root = RUN / f'pad-roundoff-every-step-v1/group{group}/{method}'
    result, cost = read(root / 'result.json'), read(root / 'cost.json')
    rows = [json.loads(line) for line in (root / 'steps.jsonl').read_text().splitlines()]
    assert result['completed'] and result['steps'] == len(rows) == 1000
    assert result['mechanical_history_passed'] and not result['history_failed_steps']
    assert all(row['stress']['native_valid'] for row in rows)
    assert all(window['mechanical_window_passed'] for window in result['windows'])
    changes = [{'step': row['step'], **row['stress']['source_roundoff']}
               for row in rows if row['stress']['source_roundoff']['corrected_sources']]
    assert [row['step'] for row in changes] == [37, 38]
    assert all(row['force_change_bound_n'] <= row['physical_force_budget_n'] for row in changes)
    for name in [f'{method}-audit', 'primary-audit']:
        audit = read(RUN / f'{method}-roundoff-admission-v1/group0/{name}/result.json')
        assert audit['passed']
        if name == 'primary-audit':
            assert len(audit['cases'][0]['checks']) == 8 and all(audit['cases'][0]['checks'].values())
    histories.append({'method': method, 'steps': len(rows), 'changes': changes,
        'median_complete_workflow_ms_per_step': cost['median_latency_ms'],
        'zero_load_step38_raw_vm_peak_pa': rows[37]['stress']['raw_vm_peak_pa'],
        'zero_load_step38_rhs_normalized_residual': rows[37]['stress']['elastic_equation_relative_residual']})
for group in [0, 1]:
    proof = read(RUN / f'native-wrench-attribution-v5/group{group}/attribution/result.json')
    assert proof['passed'] and not proof['SCR_repair_admitted']
    cert = proof['rows'][0]['independent_continuous_force_certificate']
    assert cert['independent_force_infeasibility_proven'] == (group == 1)
files, external = collect(inputs)
files += [META / 'roundoff-admission-plans-v1.json']
for group in [0, 1]:
    files.append(RUN / f'native-diverse-rejection-audit-v5/group{group}/audit/result.json')
    external.add(RUN / f'native-diverse-rejection-capture-v2/group{group}/scr-max/rejected-state.npz')
for case in roundoff['cases']:
    external.add(pathlib.Path(case['frame']['path']))
receipt = {'passed': True, 'recorded_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'actual_scheduler_accounting': accounting, 'complete_every_step_histories': histories,
    'captured_imports': {name: cones[name] for name in inputs},
    'original_failed_inputs_and_strict_continuous_certificates_retained': True,
    'final_rotation_helper_reproduces_original_two_unrotated_corrections_bitwise': True,
    'live_history_sources_not_replaced_by_final_helper': True,
    'zero_rhs_relative_residual_is_not_a_meaningful_absolute_accuracy_measure': True,
    'physical_accuracy_qualified': False, 'production_repair_admitted': False, 'whole_goal_complete': False}
p = META / 'source-roundoff-and-wrench-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packets.append(freeze('source-roundoff-and-wrench-attribution-v1', files, sorted(external),
    'Separate roundoff and finite SCR model conflicts. Final independent actual-input controls bound force/moment/stress effects, scales/rotation and finite-load refusal, preserving earlier failures. Both original1000-step every-step DPR/SCR paths complete with only observations37/38 reconciled, unchanged native tolerances and their own complete mechanical/eight-gate primary-cost admission. Complete research-workflow costs include full host readback and audits; they are not production rates or speedups over sparse observations. At zero load a sub1e-6Pa numerical stress tail and singular RHS-relative diagnostic remain explicitly retained. Diverse SCR env330 receives an independent continuous force-only infeasibility proof; env9 force-only remains unresolved, while its prior full-wrench certificate remains authoritative. Source-normal/application contrasts change the declared model and do not admit an SCR accuracy repair. Captured trajectories use the original helper; final rotation controls prove the original two failed unrotated corrections remain bitwise identical. All original failures/certificates remain immutable; no production model is changed.'))

checks = []
for campaign, count in [('canonical125-resilient-spatial-parent-admission-v1', 4),
                        ('contact125-resilient-pair-parent-admission-v1', 2)]:
    result = read(RUN / campaign / 'group0/admission/result.json')
    assert result['passed'] and len(result['runs']) == count
    for row in result['runs']:
        assert len(row['checks']) == 10 and all(row['checks'].values())
        checks.append(row['checks'])
files, external = collect(fine)
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
    'completed_final_and_contact_steps': [400, 400], 'strict_parent_checks': checks,
    'captured_imports': {name: cones[name] for name in fine},
    'half_dt_live_outputs_excluded': True, 'fine_reference_qualified': False,
    'method_rank_admitted': False, 'whole_goal_complete': False}
p = META / 'canonical125-final-contact-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packets.append(freeze('canonical125-final-contact-terminal-v1', files, sorted(external),
    'Actual terminal125um three-wall q10 main and q6 contact controls each complete their original400 steps after validated200-step continuation. Their original mechanics and independent momentum gates pass. Strict four-parent spatial admission passes40 checks and q6/q10 pair admission passes20, releasing existing comparison tasks. Terminal captured sources, journals, receipts and checkpoint metadata are retained; full checkpoint/endpoint arrays remain on data. Half-dt remains live and excluded. This is not final seven-parent/36-comparison qualification, a54-error/18-rank result or final figure admission; the full original and extended goals remain active.'))
with (META / 'resolved-pad-and-inputs-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packets, indent=2) + '\n')
print(json.dumps(packets))
