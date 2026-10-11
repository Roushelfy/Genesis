"""Publish closed producer, scheduling and exact first-failure evidence, preserving failed controls."""
import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).parent
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/native-scheduling-diagnosis-checkpoint-v1')
assert not destination.exists(), destination
folders = [*[f'actual-nonconvex-producers-v{i}' for i in range(1, 5)],
           'sample-warp-candidate-pair-v1', 'sample-warp-candidate-online-v1',
           'symmetric-warp-v3-v1', 'symmetric-warp-v4-v1',
           'sample-warp-b1024-profile-v1', 'b1024-failure-capture-v1', 'b1024-stage-failure-v1',
           'exact-517-replay-v1', 'exact-517-replay-v2']
files, external = [], []
skip = {'cache', 'tmp', '__pycache__', '.pytest_cache'}
for folder in folders:
    directory = root / folder
    assert directory.is_dir(), directory
    for current, dirs, names in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in skip]
        for name in names:
            path = Path(current) / name
            if path.stat().st_size > 20_000_000:
                external.append({'original': str(path), 'bytes': path.stat().st_size,
                                 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'reason': 'Large later-frame native snapshot stays in the project data directory.'})
            else:
                files.append(path)
    allocation = root / (folder + '-allocation.log')
    if allocation.exists():
        files.append(allocation)
names = [*[f'probe_actual_nonconvex_producers_v{i}.py' for i in range(1, 5)],
         *[f'run_actual_nonconvex_producers_v{i}.sh' for i in range(1, 5)],
         *[f'traction_warp_candidate_v{i}.py' for i in range(2, 7)],
         'traction_sample_warp_candidate_v1.py', 'probe_sample_warp_candidate_tests_v1.py',
         'probe_sample_warp_candidate_profile_v1.py', 'probe_sample_warp_candidate_online_v1.py',
         'run_sample_warp_candidate_pair_v1.sh', 'run_sample_warp_candidate_online_v1.sh',
         'run_sample_warp_b1024_profile_v1.sh',
         *[f'probe_warp_v{i}_v1.py' for i in (3, 4)], *[f'run_warp_v{i}_v1.sh' for i in (3, 4)],
         *[f'probe_b1024_{name}_v1.py' for name in ('failure_capture', 'stage_failure')],
         *[f'run_b1024_{name}_v1.sh' for name in ('failure_capture', 'stage_failure')],
         *[f'probe_exact_517_v{i}.py' for i in (1, 2)], *[f'run_exact_517_v{i}.sh' for i in (1, 2)],
         'archive_online_checkpoint_v10.py', 'run_archive_online_checkpoint_v10.sh']
files.extend(root / name for name in names)
producers = []
for case in ('primitive-cube', 'decomposition-u', 'sdf-u', 'sdf-cavity'):
    summary = json.loads((root / f'actual-nonconvex-producers-v4/{case}/results/summary.json').read_text())
    assert summary['complete_steps'] == 16 and len(summary['records']) == 16
    assert summary['source_sha256'] == summary['source_sha256_after']
    records = summary['records']
    producers.append({'case': case, 'complete_steps': 16, 'env_steps': 48, 'cpu_full_fields': 48,
                      'force_error_max': max(r['force_error_max'] for r in records),
                      'moment_error_max': max(r['moment_error_max'] for r in records),
                      'complete_residual_max_N': max(r['complete_residual_max_N'] for r in records),
                      'actual_contact_mu': sorted(set(v for r in records for v in r['friction_coefficients']))})
assert '16 passed' in (root / 'sample-warp-candidate-pair-v1/tests.log').read_text()
assert '16 passed' in (root / 'symmetric-warp-v4-v1/tests.log').read_text()
for batch, seed in ((16, 623001), (32, 623101)):
    summary = json.loads((root / f'sample-warp-candidate-online-v1/b{batch}-seed{seed}/results/summary.json').read_text())
    assert summary['completed_steps'] == 1280 and len(summary['oracle_captures']) == 9
    assert summary['source_sha256'] == summary['source_sha256_after']
exact = json.loads((root / 'exact-517-replay-v2/results/summary.json').read_text())
assert all(r['status'][0][0] == 3 for r in exact['records'][:5])
assert all(r['status'][0][0] == 0 for r in exact['records'][5:])
assert len(exact['cpu_oracles']) == 3
for folder in folders[4:]:
    before, after = root / folder / 'source.sha256', root / folder / 'source-after.sha256'
    assert before.read_bytes() == after.read_bytes(), folder
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    assert original.is_file(), original
    artifact = destination / 'raw' / original.relative_to(root)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({'original': str(original), 'artifact': str(artifact.relative_to(destination)),
                    'bytes': original.stat().st_size, 'sha256': digest})
manifest = {'scope': __doc__, 'source_base_revision': '9db33909a53563791719e7cfe93759e29eb48aaa',
            'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
            'note': 'Scheduling remains experimental. Exact first error is tick517; scalar replay of later tick520 does not establish first-frame feasibility. Exact Q10 CPU fit establishes feasibility; ridge ablation diagnoses numerical stagnation. General native acceptance and large-batch throughput remain unfinished.',
            'producer_summary': producers, 'external_files': external, 'files': entries}
path = destination / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination / entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
for entry in external:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(e['bytes'] for e in entries),
                  'external_files': external, 'producer_summary': producers,
                  'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
