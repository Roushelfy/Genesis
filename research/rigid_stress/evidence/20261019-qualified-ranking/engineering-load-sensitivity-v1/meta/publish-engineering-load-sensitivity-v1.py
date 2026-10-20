"""Freeze admitted resolved-boundary errors and the source-local engineering study."""
import json
import pathlib
import subprocess

BASE = pathlib.Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))
campaigns = ['resolved-pad-qualified-reference-v1', 'resolved-pad-ranking-v1',
    'nearby-scr-loads-v1', 'nearby-scr-loads-control-v2', 'nearby-scr-loads-control-v3',
    'nearby-scr-loads-q25-v4', 'nearby-scr-loads-q81-v4', 'nearby-scr-loads-projection-v5',
    'nearby-scr-sampling-v1', 'nearby-scr-loads-fine-v6', 'nearby-scr-loads-projection-precision-v7',
    'nearby-scr-loads-analytic-v8', 'nearby-scr-fine-sampling-v4']
cancelled = ['resolved-pad-decomposition-v1', 'nearby-scr-fine-sampling-v2', 'nearby-scr-fine-sampling-v3']
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns + cancelled]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
for campaign, job in zip(campaigns + cancelled, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        state, exit_code = observed[f'{job}_{group}']
        assert state in ['COMPLETED', 'FAILED'] or state.startswith('CANCELLED'), (campaign, state)
        code = RUN / campaign / f'group{group}/returncode.txt'
        if code.exists():
            if code.read_text().strip() == '0':
                assert state == 'COMPLETED' and exit_code == '0:0'
            else:
                assert state == 'FAILED' or state.startswith('CANCELLED')
        else:
            assert state.startswith('CANCELLED')
rank = read(RUN / 'resolved-pad-ranking-v1/group0/ranking/result.json')
sampling = read(RUN / 'nearby-scr-fine-sampling-v4/group0/comparison/result.json')
assert rank['passed'] and rank['errors_admitted'] and len(rank['rows']) == 18
assert sampling['passed'] and len(sampling['rows']) == 2
assert not sampling['engineering_approximation_admitted']
selected = ['nearby-scr-loads-fine-v6/group0/study', 'nearby-scr-loads-analytic-v8/group0/study',
            'nearby-scr-loads-fine-v6/group2/study', 'nearby-scr-loads-fine-v6/group3/study']
for directory in selected:
    result = read(RUN / directory / 'result.json')
    assert result['passed'] and len(result['rows']) == 1
    row = result['rows'][0]
    assert all(not budget['accepted'] for budget in row['budgets'])
    assert all(source['status'] == 'PrimalInfeasible' for budget in row['budgets'][:-1] for source in budget['sources'])
    diagnostic = row['budgets'][-1]
    assert diagnostic['variant'] == 'unbudgeted-projection' and diagnostic['physically_evaluable']
    assert max(diagnostic[k] for k in ['linear_superposition_error', 'equation_relative_residual',
        'delta_equation_relative_residual']) <= 1e-7
    assert identity(pathlib.Path(diagnostic['field']['path'])) == diagnostic['field']
files, external = collect(campaigns)
for campaign in cancelled:
    files += sources(campaign) + terminal(campaign)
    external.update((RUN / campaign).glob('group*/**/*.npz'))
files.append(BASE)
files += list((RUN / 'resolved-pad-ranking-v1/group0/ranking').glob('*.csv'))
for pattern in ['nearby-scr-*.json', 'nearby-load-and-transfer-*.json',
                'resolved-pad-qualified-reference-*.json', 'resolved-pad-ranking-*.json',
                'resolved-pad-decomposition-v1-cancellation.json']:
    files += list(META.glob(pattern))
for detail in rank['details']:
    for raw in detail['raw']:
        external.update(pathlib.Path(raw[k]['path']) for k in ['mesh', 'field'])
for directory in selected:
    start = read(RUN / directory / 'start.json')
    for key in ['state', 'mesh']:
        row = start[key]; path = pathlib.Path(row['path']); assert identity(path) == row; external.add(path)
    files.append(pathlib.Path(start['certificate']['path']))
terminal_receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
    'captured_imports': cones, 'resolved_reference_errors': 18,
    'source_local_grid': [0, 0.001, 0.01, 0.1], 'all_source_local_grid_cases_rejected': True,
    'refined_stress_sampling': sampling['rows'], 'selected_fine_studies': selected,
    'original_failures_and_unsuccessful_optimizers_retained': True,
    'actual_inertial_state_unavailable': True, 'same_linear_correction_includes_equilibrium_projection': True,
    'aggregate_budget_cooperative_study_complete': False,
    'signed_transfer_decomposition_pending': True, 'production_approximation_admitted': False,
    'whole_goal_complete': False}
p = META / 'engineering-load-sensitivity-terminal-v1.json'
with p.open('x') as stream: stream.write(json.dumps(terminal_receipt, indent=2) + '\n')
files.append(p)
packet = freeze('engineering-load-sensitivity-v1', files, sorted(external),
    'Eighteen admitted resolved-boundary native errors use the existing qualified physical reference and own400-step history. Corrected transfer has10.56% full tangential error versus7.94% for fixed-radius DPR; conditional overlap remains for full tangent/membrane/inner metrics, while fixed-radius is lower for bending/outer/peaks. This does not justify wrong-boundary loads or assign isolated elasticity error. Separate source-local SCR budgets0/0.001/0.01/0.1 reject both retained diverse first failures at each evaluated sampling. An explicitly unbudgeted nearest-prior cone projection changes full3D stress by29.81%/33.51%;225-to625 sampling changes the correction by0.170%/0.316%. Full P2 tensors, resultants, raw peaks/positions, actual source/global wrench corrections and numerical/analytic optimality checks are retained. Failed mesh-convention controls, underresolved sampling, failed optimizer accuracy and rejected AlmostSolved states remain immutable. Exact analytic projection resolves one diagnostic only; it never admits an engineering observation. Absolute fields are contact-only because inertial inputs were not captured; linear correction includes consistent-mass equilibrium changes. A cooperative aggregate-budget study, actual inertial baseline, native integration and matched production costs remain required. Live signed transfer decomposition is excluded.')
with (META / 'engineering-load-sensitivity-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
