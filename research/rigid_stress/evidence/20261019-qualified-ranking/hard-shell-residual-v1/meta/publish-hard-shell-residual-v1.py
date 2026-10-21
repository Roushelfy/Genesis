"""Freeze the hollow-shell decomposition and its retained independent-reference refusal."""

from pathlib import Path
import json
import subprocess

N = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
HELPER_SOURCE = N / 'meta/publish-resolved-pad-and-inputs-v1.py'
exec(compile(HELPER_SOURCE.read_text().split('\nresolved = ', 1)[0], str(HELPER_SOURCE), 'exec'))
campaigns = ['hard-shell-residual-v1', 'hard-shell-residual-inspection-v1']
jobs = [str(read(N / name / 'submission.json')['job']).split(';')[0] for name in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-n', '-P',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,ReqMem,MaxRSS'], text=True)
observed = {row.split('|')[0]: row.split('|')[1:3] for row in accounting.splitlines() if '|' in row}
for name, job in zip(campaigns, jobs):
    assert observed[f'{job}_0'] == ['COMPLETED', '0:0']
    assert (N / name / 'group0/returncode.txt').read_text().strip() == '0'
study = read(N / 'hard-shell-residual-v1/group0/study/result.json')
inspection = read(N / 'hard-shell-residual-inspection-v1/group0/inspection/result.json')
assert study['execution_completed'] and inspection['execution_completed']
assert study['corrected_internal_targets_met'] and inspection['retained_field_peak_reductions_passed']
assert not inspection['independent_direct_reference_change_gate_passed']
assert not inspection['coarse_corrected_vs_fine_direct_error_gate_passed']
assert not inspection['conditional_prescribed_pressure_stress_gate_passed']
assert not inspection['pressure_support_prediction_qualified']
bindings = []
for module, campaign in [('study_hertz_shell_residual', campaigns[0]), ('inspect_hertz_shell_residual', campaigns[1])]:
    relative = Path(f'research/rigid_stress/qipc_comparison/{module}.py')
    path = N / campaign / 'analysis-source' / relative
    assert path.read_bytes() == (WORK / relative).read_bytes()
    bindings.append(identity(path))
for row in study['rows']:
    assert all(identity(Path(row[key]['path'])) == row[key] for key in ['mesh', 'fields'])
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
           'current_workers_match_executed_captures': bindings,
           'study': study, 'inspection': inspection,
           'public_scientific_source': 'd4503b9e42a7f8f8555d128b437bb45b3193cf75',
           'public_source_binding_packet': 'hard-scene-public-source-binding-v1',
           'scope': 'Prescribed normal Hertz pressure on one fixed faceted full hollow polar-land egg, with identical outer/inner boundaries and volume across three local/wall refinements. Particular boundary subtraction includes the cavity and preserves consistent-mass inertial relief. Complete common tensors, material-column membrane/bending, all element-corner stresses and near-field sampled maxima/locations are retained. Corrected internal refinement passes the declared targets; direct-pressure reference field/tensile-peak changes and candidate-versus-direct gates fail. No independent pressure/support or physical-stress accuracy, cheap native shell correction, friction history or public estimator is admitted.',
           'whole_goal_complete': False}
terminal_path = N / 'meta/hard-shell-residual-terminal-v1.json'
with terminal_path.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files, external = collect(campaigns)
files.extend([HELPER_SOURCE, terminal_path, N / 'meta/hard-scene-next-phase-launch-v1.json'])
packet = freeze('hard-shell-residual-v1', files, sorted(external), receipt['scope'])
(DEST / packet['packet'] / 'README.md').write_text(
    '# Prescribed-pressure hollow-shell stress decomposition\n\n' + receipt['scope'] +
    '\n\nReplay the captured task groups in order with the existing Slurm submitter, `rtx-mid`, container `1_26`, one CPU core, the pinned scientific Python and fresh explicit data-side output/cache paths. Parent737295 used 64 GiB/45 minutes; the read-only inspection737336 used 8 GiB/10 minutes with `afterok:737295`. The public engine alias is in the terminal receipt and its exact binding packet; captured research workers are authoritative. Keep the full free shell, flat300um/blend600um poles, 500um wall, E10GPa, nu0.3, density2000kg/m3, Fn0.2N and the original ellipse. All meshes and complete fields remain externally hashed on data. No point-load sweep, new pad model, solver-tolerance change or active scene-walk experiment is included. Internal corrected convergence and independent physical/reference admission remain separate.\n')
with (N / 'meta/hard-shell-residual-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
