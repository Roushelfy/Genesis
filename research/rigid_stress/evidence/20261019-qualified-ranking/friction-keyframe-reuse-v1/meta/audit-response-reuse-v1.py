"""Check already computed Slurm results against the predeclared reuse gates."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking')
RUN = ROOT / 'solid-pad-stvk-response-reuse-control-v1'
receipts = []

def read(path):
    data = path.read_bytes()
    receipts.append({'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    return json.loads(data)

assert (RUN / 'group0/returncode.txt').read_text().strip() == '0'
targets = read(ROOT / 'meta/solid-pad-stvk-response-reuse-control-v1-targets.json')
for name in ('analytic', 'contact-analytic', 'moment'):
    assert read(RUN / f'group0/{name}/result.json')['passed']
history = read(RUN / 'group0/reference/result.json')
assert history['completed'] and history['steps'] == 400 and history['mechanical_history_passed']
assert all(history['history_checks'].values())
assert all(w['mechanical_window_passed'] and all(w['checks'].values()) for w in history['windows'])
assert len(history['windows']) == 2
rows = []
for level in range(2):
    result = read(RUN / f'group0/equivalence-level{level}/result.json')
    assert result['common_peaks_evaluated'] and result['physical_volume_weighting']
    change = result['cases'][0]['adjacent_comparisons'][0]
    fields = {key: value for key, value in change.items() if key.endswith('_relative_l2')}
    raw = {key: value for key, value in change.items() if key.startswith('raw_')}
    common = change['common_spatial_measures']
    peaks = {p['metric']: p['relative_difference'] for p in common['tangent_peaks']}
    extra = {p['component']: p['relative_l2_error'] for p in common['full3d_only_components']}
    assert len(fields) == 5 and len(raw) == 2 and len(peaks) == 2 and len(extra) == 2
    assert all(0 <= value <= targets['maximum_complete_field_relative_difference'] for value in (*fields.values(), *extra.values()))
    assert all(0 <= value <= targets['maximum_raw_peak_relative_difference'] for value in (*raw.values(), *peaks.values()))
    rows.append({'level': level, 'fields': fields, 'raw_peaks': raw, 'common_tangent_peaks': peaks, 'full3d_only_components': extra})
output = ROOT / 'meta/solid-pad-stvk-response-reuse-admission-v1.json'
assert not output.exists()
output.write_text(json.dumps({'passed': True, 'targets': targets, 'comparisons': rows, 'receipts': receipts, 'scope': 'Metadata-only verification of completed Slurm analytic, full-history, current-momentum and field controls. Reuse avoids one redundant factor solve; no physical or solver acceptance threshold changes. This is not spatial-reference qualification or a repeated timing claim.'}, indent=2) + '\n')
print(json.dumps({'passed': True, 'output': str(output)}))
