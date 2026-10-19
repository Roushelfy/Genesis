"""Audit terminal receipts and scalar refinement results; no numerical recomputation."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
OUT = ROOT/'meta/friction-spatial250-v1-audit.json'
receipts = {}

def receipt(path):
    path = Path(path)
    if str(path) not in receipts:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        receipts[str(path)] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return receipts[str(path)]

def read(path):
    receipt(path)
    return json.loads(path.read_text())

scheduler = subprocess.check_output(['sacct', '-j', '728256,728389,728644,728678', '-X', '-n', '-P', '--format=JobID,State,ExitCode'], text=True)
for job in ('728256_0', '728389_0', '728644_0', '728678_0', '728678_1'):
    assert f'{job}|COMPLETED|0:0' in scheduler.splitlines(), job
fields = {}
for name in ('solid-pad-stvk-spatial500-v2', 'solid-pad-stvk-spatial250-cache-v1'):
    base = ROOT/name/'group0'
    assert (base/'returncode.txt').read_text().strip() == '0'
    result = read(base/'reference/result.json')
    assert result['completed'] and result['steps'] == 400 and result['mechanical_history_passed']
    assert all(result['history_checks'].values())
    assert len(result['windows']) == 2 and all(w['mechanical_window_passed'] for w in result['windows'])
    moment = read(base/'moment/result.json')
    assert moment['passed'] and moment['history_admitted'] and moment['run'] == receipt(base/'reference/result.json')
    assert len(moment['frames']) == 2
    for frame in moment['frames']:
        assert frame['source_transfer_passed'] and frame['physical_momentum_passed']
        actual = receipt(frame['field']['path'])
        assert actual == frame['field']
        fields[actual['path']] = actual
    read(base/'reference/start.json')
    assert read(base/'analytic/result.json')['passed']

metrics = ('tangent_relative_l2', 'membrane_relative_l2', 'bending_relative_l2', 'outer_tangent_relative_l2', 'inner_tangent_relative_l2')
rows = []
for level in range(2):
    group = ROOT/'stvk-spatial250-fields-v1'/f'group{level}'
    assert (group/'returncode.txt').read_text().strip() == '0'
    comparisons = {}
    for q in (9, 25):
        data = read(group/f'surface500to250-level{level}-q{q}/result.json')
        assert data['surface_quadrature_points_per_face'] == q and data['depth_quadrature_points'] == 4
        assert data['physical_volume_weighting'] and data['common_peaks_evaluated']
        assert len(data['cases']) == 1
        case = data['cases'][0]
        assert case['load_n'] == .1 and len(case['raw']) == 2 and len(case['adjacent_comparisons']) == 1
        for raw in case['raw']:
            assert fields[raw['field']['path']] == raw['field']
            assert receipt(raw['mesh']['path']) == raw['mesh']
        comparisons[q] = case['adjacent_comparisons'][0]
    c = comparisons[25]
    row = {'level': level, 'field_changes': {key: c[key] for key in metrics}, 'raw_mises_relative_change': c['raw_mises_relative_change'], 'raw_tension_relative_change': c['raw_tension_relative_change'], 'common_peak_changes': {p['metric']: p['relative_difference'] for p in c['common_spatial_measures']['tangent_peaks']}}
    row['field_target_passed'] = all(c[key] <= .02 for key in metrics)
    row['raw_peak_target_passed'] = max(c['raw_mises_relative_change'], c['raw_tension_relative_change']) <= .05
    row['common_peak_target_passed'] = max(row['common_peak_changes'].values()) <= .05
    row['q9_q25_field_metric_absolute_differences'] = {key: abs(comparisons[9][key] - c[key]) for key in metrics}
    row['q9_q25_actual_peak_relative_changes'] = {f'{a["metric"]}_{side}': abs(a[side]['stress_pa'] - b[side]['stress_pa'])/abs(b[side]['stress_pa']) for a,b in zip(comparisons[9]['common_spatial_measures']['tangent_peaks'], c['common_spatial_measures']['tangent_peaks']) for side in ('actual', 'reference')}
    rows.append(row)
plot = read(ROOT/'friction-plot-data500-control-v1/group0/data/result.json')
assert plot['passed'] and len(plot['checks']) == 5 and all(plot['checks'].values())
assert plot['trajectory_rows'] == 1600 and len(plot['traction']) == 8
assert not plot['spatial_reference_qualified'] and not plot['whole_goal_complete']
for entry in plot['receipts'] + [plot['force_history']] + [t['output'] for t in plot['traction']]:
    assert receipt(entry['path']) == entry
result = {'metadata_and_hash_audit_passed': True, 'scheduler': scheduler, 'verified_reference_fields': list(fields.values()), 'refinement': rows, 'plot_data_checks_passed': True, 'receipts': list(receipts.values()), 'spatial_reference_qualified': False, 'whole_goal_complete': False, 'scope': 'Terminal500/250um mechanics and current-momentum receipts, independently rehashed endpoint fields, q9/q25 scalar comparisons and plot-data input/output hashes. Both spatial field gates fail. No new scientific solve or figure rendered; no final method rank or continuum certification.'}
with OUT.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps({'verified_receipts':len(receipts),'refinement':rows}, indent=2))
