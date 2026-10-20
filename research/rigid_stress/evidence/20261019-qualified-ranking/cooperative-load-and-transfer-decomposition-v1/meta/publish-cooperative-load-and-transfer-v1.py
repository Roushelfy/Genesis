"""Freeze completed cooperative-load controls and signed transfer attribution."""
from pathlib import Path

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = [
    'resolved-pad-decomposition-v2', 'resolved-pad-decomposition-summary-v1',
    'nearby-scr-group-budget-v1', 'nearby-scr-port-budget-v1', 'nearby-scr-port-budget-v2',
    'nearby-scr-fit-accuracy-v1', 'nearby-scr-local-port-control-v1',
    'nearby-scr-group-comparison-v1', 'nearby-scr-group-comparison-v2', 'nearby-scr-group-comparison-v3',
    'nearby-scr-group-figure-v1',
]
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        code = (RUN / campaign / f'group{group}/returncode.txt').read_text().strip()
        assert observed[f'{job}_{group}'] == (['COMPLETED', '0:0'] if code == '0' else ['FAILED', '1:0'])
        assert code == ('1' if campaign == 'nearby-scr-group-comparison-v2' else '0')
decomposition = read(RUN / 'resolved-pad-decomposition-summary-v1/group0/summary/result.json')
comparison = read(RUN / 'nearby-scr-group-comparison-v3/group0/comparison/result.json')
qa = read(META / 'nearby-scr-group-figure-qa-v1.json')
assert decomposition['passed'] and len(decomposition['rows']) == 10
assert comparison['passed'] and len(comparison['rows']) == 7 and len(comparison['sensitivities']) == 42
assert len(comparison['source_local_controls']) == 10 and len(comparison['independent_aggregate_checks']) == 7
assert all(not row['engineering_stress_accepted'] for row in comparison['rows'])
assert qa['passed'] and qa['direct_preview_inspected']
for row in [qa['result'], qa['preview'], qa['source_data'], *qa['exports']]:
    assert identity(Path(row['path'])) == row
files, external = collect(campaigns)
files += [BASE, META / 'contact-region-nature-style-v1.py']
for campaign in campaigns:
    files += list((RUN / campaign).glob('group*/**/*.csv'))
for pattern in ('nearby-scr-group-*.json', 'nearby-scr-port-*.json', 'nearby-scr-fit-*.json',
                'nearby-scr-local-port-*.json', 'resolved-pad-decomposition-tasks-v2.json',
                'resolved-pad-decomposition-summary-tasks-v1.json'):
    files += list(META.glob(pattern))
figure = RUN / 'nearby-scr-group-figure-v1/group0/figure'
files += [p for p in figure.iterdir() if p.suffix in ('.svg', '.pdf', '.png', '.md')]
for group in (0, 2):
    start = read(RUN / f'nearby-scr-fit-accuracy-v1/group{group}/study/start.json')
    for key in ('state', 'mesh', 'certificate'):
        row = start[key]
        path = Path(row['path'])
        assert identity(path) == row
        if path.suffix == '.npz':
            external.add(path)
        else:
            files.append(path)
for row in decomposition['receipts']:
    path = Path(row['path'])
    assert identity(path) == row
    if path.suffix == '.npz':
        external.add(path)
terminal_receipt = {
    'passed': True, 'actual_scheduler_accounting': accounting, 'captured_imports': cones,
    'signed_decomposition_rows': 10, 'evaluated_aggregate_rows': 7,
    'source_local_control_rows': 10, 'numerical_and_sampling_sensitivities': 42,
    'independent_aggregate_anchor_checks': comparison['independent_aggregate_checks'],
    'historical_anchor_label_corrected_without_changing_frozen_inputs': True,
    'centroid_path_results_not_original_support_minima': True,
    'failed_optimizers_and_failed_reader_retained': True,
    'source_local_vs_group_support_and_priors_matched': True,
    'engineering_stress_accepted': False, 'production_tolerance_changed': False,
    'original_inertial_state_unavailable': True, 'linear_correction_includes_equilibrium_projection': True,
    'figure_directly_inspected': True, 'whole_goal_complete': False,
}
p = META / 'cooperative-load-and-transfer-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(terminal_receipt, indent=2) + '\n')
files.append(p)
packet = freeze('cooperative-load-and-transfer-decomposition-v1', files, sorted(external),
    'Ten signed tensor decompositions show partial cancellation of remaining reference error by the old wrong-boundary DPR transfer. Improved topology-preserving surface paths admit nearby cooperative SCR loads within1%/0.1% aggregate force/moment budgets for env9/env330, with8.76%/14.66% complete-stress changes and actual individual-source redistribution. Strict originals remain rejected; every candidate has engineering_stress_accepted=false. Seven complete-field precision comparisons, fourteen equivalent-conditioning comparisons, separate angular/port sensitivities and a matched source-local control retain every unsuccessful optimizer. The earlier29.81%/33.51% unbudgeted stress changes apply only to the narrower centroid-path subset. Full tensors/resultants, raw peak values/locations, source/global wrench changes and consistent-mass equilibrium corrections are retained. Absolute stresses remain contact-only because original inertia was not captured. Historical start metadata names an origin moment incorrectly; actual optimization uses failure.position, independently rechecked without changing frozen records. Comparison-v2 retains a metadata-key failure;v3 is the successful reader. Six-panel maps include inspected SVG/PDF/PNG and plotted data. Production automatic integration, admitted engineering stress budgets and matched physical/cost validation remain open.')
with (META / 'cooperative-load-and-transfer-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
