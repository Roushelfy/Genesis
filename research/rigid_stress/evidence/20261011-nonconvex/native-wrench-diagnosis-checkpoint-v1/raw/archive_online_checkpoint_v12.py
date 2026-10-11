"""Publish feasible-fit regression and continuous wrench certificates with their failed controls."""

import hashlib
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).parent
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/native-wrench-diagnosis-checkpoint-v1')
assert not destination.exists(), destination
folders = [
    'singular-fallback-warp-v7-v1', 'warp-v7-online-b1024-v1',
    'unified-cooperative-v8-v1', 'trust-cooperative-v9-v1',
    'exact-602-replay-v1', 'exact-602-socp-v1', 'exact-602-pair-socp-v1',
    'exact-602-continuous-v1', 'exact-602-search-continuous-v1',
    'certificate-search-v1', 'continuous-wrench-candidate-v1',
    'polar-search-candidate-v1', 'native-newton-regression-v1',
    'native-newton-regression-v2', 'native-newton-regression-v3', 'native-newton-regression-v4',
    'native-newton-fault-v1', 'native-producers-v5', 'native-producers-v7',
    'archive-wrench-diagnosis-checkpoint-v1',
]
skip = {'cache', 'tmp', 'tools', '__pycache__', '.pytest_cache'}
files, external = [], []
postrun_hash_records = []
for folder in folders:
    directory = root / folder
    assert directory.is_dir(), directory
    for current, dirs, names in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in skip]
        for name in names:
            original = Path(current) / name
            if original.stat().st_size > 20_000_000:
                external.append({'original': str(original), 'bytes': original.stat().st_size,
                                 'sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
                                 'reason': 'Large runtime capture remains data-side.'})
            else:
                files.append(original)
    for before in directory.rglob('source.sha256'):
        if not any(part in skip for part in before.relative_to(directory).parts):
            after = before.with_name('source-after.sha256')
            if folder in ('exact-602-continuous-v1', 'exact-602-search-continuous-v1'):
                assert not after.exists()
                for line in before.read_text().splitlines():
                    digest, filename = line.split('  ', 1)
                    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest
                postrun_hash_records.append({'record': str(before),
                    'scope': 'Offline script/output checksums were recorded after the run, not before/after source closure.'})
            else:
                assert before.read_bytes() == after.read_bytes(), before

names = [
    'traction_sample_cooperative_candidate_v2.py', 'traction_cooperative_candidate_v8.py',
    'traction_trust_candidate_v9.py', 'continuous_wrench_candidate_v1.py', 'polar_search_candidate_v1.py',
    'probe_warp_v7_v1.py', 'run_warp_v7_v1.sh', 'probe_warp_v7_online_v1.py', 'run_warp_v7_online_v1.sh',
    'probe_cooperative_v8_v1.py', 'run_cooperative_v8_v1.sh', 'probe_trust_v9_v1.py', 'run_trust_v9_v1.sh',
    'run_newton_regression_v1.sh', 'run_newton_regression_v2.sh', 'run_newton_regression_v3.sh', 'run_newton_regression_v4.sh',
    'probe_newton_fault_v1.py', 'run_newton_fault_v1.sh', 'run_native_producers_v5.sh',
    'probe_continuous_wrench_v1.py', 'run_continuous_wrench_v1.sh',
    'probe_polar_search_v1.py', 'run_polar_search_v1.sh',
    'inspect_602_v1.py', 'run_inspect_602_v1.sh', 'inspect-602-v1.log',
    'probe_certificate_search_v1.py', 'run_certificate_search_v1.sh',
    'extract_517_fixture_v1.py', 'run_extract_517_fixture_v1.sh', 'extract-517-fixture-v1.log',
    'archive_online_checkpoint_v11.py', 'run_archive_online_checkpoint_v11.sh',
    'archive_online_checkpoint_v12.py', 'run_archive_online_checkpoint_v12.sh',
    'certificate-search-v1.log', 'exact-602-continuous-v1.log', 'exact-602-search-continuous-v1.log',
]
names.extend(f'traction_warp_candidate_v{i}.py' for i in range(2, 8))
for kind in ('', '_socp', '_pair_socp', '_continuous', '_search_continuous'):
    names.extend((f'probe_exact_602{kind}_v1.py', f'run_exact_602{kind}_v1.sh'))
files.extend(root / name for name in names)
files.extend(root.glob('*newton*allocation.log'))
files.extend(root.glob('*producer*allocation.log'))
files.extend(root.glob('*602*allocation.log'))
files.extend(root.glob('*polar*allocation.log'))
files.extend(root.glob('*continuous-wrench*allocation.log'))
files.extend(root.glob('*cooperative-v[89]*allocation.log'))
files.extend(root.glob('*warp-v7*allocation.log'))
files.extend(root.glob('*certificate-search*allocation.log'))
files.extend(root.glob('archive-wrench-diagnosis-checkpoint-v1-allocation.log'))

for backend in ('cpu', 'cuda'):
    assert '50 passed' in (root / f'native-newton-regression-v2/{backend}/tests.log').read_text()
    assert '7 passed' in (root / f'native-newton-regression-v3/{backend}/tests.log').read_text()
    assert '7 passed' in (root / f'native-newton-regression-v4/{backend}/tests.log').read_text()
    assert '16 passed' in (root / f'unified-cooperative-v8-v1/{backend}/tests.log').read_text()
    assert '7 passed' in (root / f'trust-cooperative-v9-v1/{backend}/tests.log').read_text()
    polar = json.loads((root / f'polar-search-candidate-v1/{backend}/results/summary.json').read_text())
    assert polar['search']['valid'] == [[False, True]]
    assert all(record['valid'] == [[False, True]] for record in polar['records'][1:])
    assert polar['records'][0]['valid'] == [[False, False]]
assert '2 failed' in (root / 'native-newton-regression-v1/cpu/tests.log').read_text()
assert '50 passed' in (root / 'native-newton-regression-v1/cuda/tests.log').read_text()
assert '1 failed' in (root / 'native-newton-fault-v1/tests.log').read_text()
for epoch in ('native-producers-v5', 'native-producers-v7'):
    for case in ('primitive-cube', 'decomposition-u', 'sdf-u', 'sdf-cavity'):
        summary = json.loads((root / f'{epoch}/{case}/results/summary.json').read_text())
        assert summary['complete_steps'] == 16 and len(summary['records']) == 16
        assert summary['source_sha256'] == summary['source_sha256_after']
socp = json.loads((root / 'exact-602-socp-v1/summary.json').read_text())
assert socp['records'][0]['status'] == 'Solved'
assert socp['records'][1]['status'] == 'PrimalInfeasible'
assert socp['records'][1]['certificate_polar_max'] < 0 < socp['records'][1]['certificate_margin']
for epoch in ('exact-602-continuous-v1', 'exact-602-search-continuous-v1'):
    summary = json.loads((root / epoch / 'summary.json').read_text())
    assert summary['conservative_polar_upper_bound'] < 0 < summary['certificate_margin']
pair = json.loads((root / 'exact-602-pair-socp-v1/summary.json').read_text())
assert pair['records'][0]['status'] == 'Solved'
online = json.loads((root / 'warp-v7-online-b1024-v1/b1024-seed623201/results/summary.json').read_text())
assert online['completed_steps'] == 602

for original in files:
    assert original.is_file(), original
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
manifest = {
    'scope': __doc__, 'source_base_revision': 'ef388a7ed967d3b419f4a20c987338659f17756d',
    'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note': 'Feasible step517 repair preserves final budgets. Step602 original component is continuously incompatible. Pair-total partitioning remains an unselected modeling alternative. Cooperative scheduling and continuous diagnostic kernels remain experimental; large-batch acceptance, four-scope throughput and plateau remain unfinished.',
    'native_regression': {'cpu': 50, 'cuda': 50, 'inline_targeted_cpu': 7, 'inline_targeted_cuda': 7,
                          'final_single_call_targeted_cpu': 7, 'final_single_call_targeted_cuda': 7},
    'socp': socp, 'pair_socp': pair, 'external_files': external, 'files': entries,
    'postrun_hash_records': postrun_hash_records,
    'offline_search_source_scope': 'certificate-search-v1 and inspect-602/extraction scripts have archived source and outputs but no contemporaneous before/after source closure. Closed native CPU/CUDA certificate runs independently reproduce separation from the exact input.',
}
path = destination / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2) + '\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination / entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(entry['bytes'] for entry in entries),
                  'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
