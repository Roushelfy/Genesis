"""Validate closed shared-support trajectory evidence and summarize unchanged acceptance budgets."""
import hashlib
import json
from pathlib import Path
root = Path(__file__).parent
records = []
for batch, seed in [(16, 623001), (32, 623101)]:
    run = root / 'online-native-pair-support-v1' / f'b{batch}-seed{seed}'
    summary = json.loads((run / 'results/summary.json').read_text())
    assert summary['completed_steps'] == 1280
    assert summary['source_sha256'] == summary['source_sha256_after']
    rows, captures = summary['records'], summary['oracle_captures']
    assert len(rows) == 1280 and len(captures) == 9
    assert not (run / 'results/failure').exists()
    entry = {'batch':batch, 'seed':seed, 'steps':1280, 'env_steps':1280*batch,
        'oracle_captures':len(captures), 'cpu_full_fields':batch*len(captures),
        'max_normalized_original_force_error':max(row['max_force_error'] for row in rows),
        'max_normalized_original_moment_error':max(row['max_moment_error'] for row in rows),
        'max_complete_residual_N':max(row['full_residual_max_N'] for row in captures),
        'max_fit_evaluations':max(row['max_evaluations'] for row in rows),
        'max_source_samples':max(row['max_samples'] for row in rows),
        'reset_environments':rows[-1]['reset_environments'],
        'contact_slot_changes':sum(row['contact_slot_changes'] for row in rows),
        'measured_combined_mu':sorted({mu for row in rows for mu in row['mu_values']})}
    assert entry['max_normalized_original_force_error'] <= 1e-8
    assert entry['max_normalized_original_moment_error'] <= 1e-8
    records.append(entry)
for folder in ['online-native-pair-support-v1', 'native-pair-lifecycle-v4', 'native-pair-support-regression-v1/cpu', 'native-pair-support-regression-v1/cuda']:
    run = root / folder
    assert (run / 'source.sha256').read_bytes() == (run / 'source-after.sha256').read_bytes()
for backend in ['cpu', 'cuda']:
    assert '47 passed' in (root / f'native-pair-support-regression-v1/{backend}/tests.log').read_text()
assert '1 passed' in (root / 'native-pair-lifecycle-v4/tests.log').read_text()
result = {'scope':__doc__, 'trajectory_records':records, 'final_wrench_budget':1e-8,
    'complete_residual_budget':'max(1e-11 N, 1e-8 ||rhs||), including six gauge rows',
    'stress_budget':{'rtol':1e-4, 'atol_Pa':1e-3},
    'production_law':'explicit pair-union mixture of unchanged intrinsic Q10 squared-coordinate disks with per-destination friction and per-original-wrench acceptance',
    'throughput_claim':False}
output = root / 'native-pair-support-summary-v1'
output.mkdir(exist_ok=False)
(output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result), flush=True)
