"""Freeze completed coupled pad elimination, PCG forcing and comparison controls."""
from pathlib import Path

BASE = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/meta/publish-resolved-pad-and-inputs-v1.py')
exec(compile(BASE.read_text().split('\nresolved = ', 1)[0], str(BASE), 'exec'))

campaigns = ['reduced-pad-equilibrium-v1', 'reduced-pad-pcg-v1', 'reduced-pad-comparison-v1']
jobs = [str(read(RUN / c / 'submission.json')['job']).split(';')[0] for c in campaigns]
accounting = subprocess.check_output(['sacct', '--array', '-j', ','.join(jobs), '-P', '--noheader',
    '--format=JobID,State,ExitCode,Elapsed,End,NodeList,MaxRSS'], text=True)
observed = {parts[0]: parts[1:3] for line in accounting.splitlines() if len(parts := line.split('|')) >= 3}
for campaign, job in zip(campaigns, jobs):
    for group in range(len(read(RUN / campaign / 'tasks.json'))):
        assert (RUN / campaign / f'group{group}/returncode.txt').read_text().strip() == '0'
        assert observed[f'{job}_{group}'] == ['COMPLETED', '0:0']
comparison = read(RUN / campaigns[2] / 'group0/comparison/result.json')
assert comparison['passed'] and len(comparison['field_comparisons']) == 48
assert len(comparison['changing_load_controls']) == len(comparison['paired_cpu_timings']) == 3
files, external = collect(campaigns)
files += [BASE, META / 'reduced-pad-equilibrium-tasks-v1.json',
          META / 'reduced-pad-pcg-tasks-v1.json', META / 'reduced-pad-comparison-tasks-v1.json']
parents = []
for relative in ('reduced-pad-equilibrium-v1/group0/control',
                 'reduced-pad-pcg-v1/group0/pcg-fixed', 'reduced-pad-pcg-v1/group1/pcg-adaptive'):
    root = RUN / relative
    result, start = (read(root / name) for name in ('result.json', 'start.json'))
    assert result['passed'] and len(result['records']) == 16 and len(result['changing_load_controls']) == 4
    parents.append(result)
    for item in start['inputs']:
        path = Path(item['path'])
        assert identity(path) == item
        if path.suffix == '.npz':
            external.add(path)
        else:
            files.append(path)
    for row in result['records']:
        assert row['rounds'][-1]['gradient_norm'] <= 1e-10
        assert row['complete_stress_relative_error'] <= 1e-6
        assert row['pressure_area_relative_error'] <= 1e-6
        assert row['full_elastic_pad_law_relative_error'] <= 1e-7
        assert row['wrench_relative_error'] <= 1e-8
        item = row['baseline']
        path = Path(item['path'])
        assert identity(path) == item
        external.add(path)
receipt = {'passed': True, 'actual_scheduler_accounting': accounting,
    'captured_imports': cones, 'new_complete_fields': 48, 'retained_coupled_reference_fields': 16,
    'newton_updates': [sum(len(row['rounds']) - 1 for row in parent['records']) for parent in parents],
    'pcg_iterations': [sum(step.get('linear_iterations', 0) for row in parent['records'] for step in row['rounds']) for parent in parents],
    'paired_cpu_timings': comparison['paired_cpu_timings'],
    'changing_load_controls': comparison['changing_load_controls'],
    'direct_v1_setup_metadata': 'operator_setup_seconds repeats the last setup duration on reused cases; actual rebuilds are cases0,3,6,7 at each quadrature. Later captured workers report zero on reuse. Do not sum v1 repeated entries.',
    'production_model_admitted': False, 'production_tolerance_changed': False, 'whole_goal_complete': False}
p = META / 'reduced-pad-equilibrium-terminal-v1.json'
with p.open('x') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
files.append(p)
packet = freeze('reduced-pad-equilibrium-v1', files, sorted(external),
    'Exact elimination of local physical pad springs reduces the degree6 coupled primal from5446 to838 unknowns while retaining cross-region elasticity and all four free plate/link coordinates. Three CPU solvers each reproduce16 retained coupled pressure/stress fields with unchanged physical/numerical admission; maximum direct-versus-reference stress and pressure differences are1.51e-10/2.28e-10. Direct reduced Newton paired with the full QP in one process takes0.0600/0.0924/0.0929s versus23.47/23.64/24.58s on nominal/flexible-stiff/gradient cases, including cold rigid initialization and excluding separately reported fixed operators/FEM output. Three repeats are warm process calls, not independent-process repetitions or native throughput. Fixed1e-6 PCG and a labelled min(0.1,sqrt(gradient)) heuristic retain the same1e-10 outer gradient and independent pad-law/wrench gates. They require36/74 Newton updates and268/304 PCG iterations over16 fields; the looser heuristic is not an admitted speed improvement. Each solver passes four prescribed normal load/tilt states, warm-start/full-QP agreement and return-to-initial-pressure closure. This uses a fixed candidate region; it is not general moving/nonconvex search, friction history, thickness qualification or automatic native integration. Complete fields, exact sources and original full-QP references are retained; large arrays stay on data.')
with (META / 'reduced-pad-equilibrium-publication-v1.json').open('x') as stream:
    stream.write(json.dumps(packet, indent=2) + '\n')
print(json.dumps(packet))
