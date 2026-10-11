"""Freeze native pair support, complete trajectories, lifecycle/data/rigid isolation and matched scheduling evidence."""
import hashlib
import json
import shutil
from pathlib import Path
root = Path(__file__).parent
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/native-pair-support-checkpoint-v1')
assert not destination.exists(), destination
folders = ['native-pair-support-regression-v1', *[f'native-pair-lifecycle-v{i}' for i in range(1, 6)],
    'online-native-pair-support-v1', 'native-pair-support-summary-v1', 'native-pair-safety-v1',
    'native-pair-safety-v2', 'rigid-immutability-v3', 'warp-candidate-pair-v1', 'warp-candidate-pair-v2']
files = []
for folder in folders:
    directory = root / folder
    assert directory.is_dir(), directory
    for before in directory.rglob('source.sha256'):
        after = before.with_name('source-after.sha256')
        assert after.exists() and before.read_bytes() == after.read_bytes(), before
    files.extend(p for p in directory.rglob('*') if p.is_file() and not
        {'cache', 'tmp', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(directory).parts))
    allocation = root / (folder+'-allocation.log')
    if allocation.exists():
        files.append(allocation)
names = ['extract_pair_support_fixtures_v1.py', 'run_extract_pair_support_fixtures_v1.sh',
    'extract-pair-support-fixtures-v1-allocation.log', 'run_native_pair_support_regression_v1.sh',
    *[f'run_native_pair_lifecycle_v{i}.sh' for i in range(1, 6)], 'run_online_native_pair_support_v1.sh',
    'summarize_native_pair_support_v1.py', 'run_summarize_native_pair_support_v1.sh',
    'run_native_pair_safety_v1.sh', 'run_native_pair_safety_v2.sh', 'probe_recovery_rigid_immutability_v3.py',
    'traction_warp_candidate_v2.py', 'probe_warp_candidate_tests_v2.py', 'probe_warp_candidate_profile_v2.py',
    'run_warp_candidate_pair_v1.sh', 'run_warp_candidate_pair_v2.sh', 'archive_online_checkpoint_v9.py',
    'run_archive_online_checkpoint_v9.sh']
files.extend(root / name for name in names)
for backend in ['cpu', 'cuda']:
    assert '47 passed' in (root / f'native-pair-support-regression-v1/{backend}/tests.log').read_text()
assert '2 passed' in (root / 'native-pair-lifecycle-v5/tests.log').read_text()
assert '2 passed' in (root / 'native-pair-safety-v2/tests.log').read_text()
for batch, seed in [(16, 623001), (32, 623101)]:
    summary = json.loads((root / f'online-native-pair-support-v1/b{batch}-seed{seed}/results/summary.json').read_text())
    assert summary['completed_steps'] == 1280 and len(summary['oracle_captures']) == 9
    assert summary['source_sha256'] == summary['source_sha256_after']
immutable = json.loads((root / 'rigid-immutability-v3/summary.json').read_text())
assert immutable['completed_steps'] == 1280 and len(immutable['records']) == 33
assert all(record['before_sha256'] == record['after_sha256'] for record in immutable['records'])
assert '15 passed' in (root / 'warp-candidate-pair-v1/tests.log').read_text()
for variant in ['scalar', 'warp']:
    profile = json.loads((root / f'warp-candidate-pair-v2/{variant}.json').read_text())
    assert profile['intrinsic_support'] == 'pair_union'
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    assert original.is_file(), original
    artifact = destination / 'raw' / original.relative_to(root)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({'original':str(original), 'artifact':str(artifact.relative_to(destination)),
        'bytes':original.stat().st_size, 'sha256':digest})
manifest = {'scope':__doc__, 'source_base_revision':'1977fca50c51ff2011b83deda8a06316ff5ab5b0',
    'requirement_revision':'1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note':'All original component wrenches and constituent radii retained. Destination friction varies. Complete B16/B32 trajectories and independent same-mesh CPU fields pass. Nested support DataKind repair is Python metadata only; preceding numerical runs and failed integration controls remain immutable. No high-mesh convergence or large-batch plateau/completion claim.',
    'files':entries}
path = destination / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination / entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts':len(entries), 'bytes':sum(e['bytes'] for e in entries),
    'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
