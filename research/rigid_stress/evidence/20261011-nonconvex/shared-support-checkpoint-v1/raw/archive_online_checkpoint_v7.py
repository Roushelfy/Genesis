"""Preserve explicit shared finite-support candidate fits, independent CPU/CUDA recovery and failed controls."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).parent
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/shared-support-checkpoint-v1')
if destination.exists():
    raise FileExistsError(destination)
folders = [f'shared-plane-support-v{i}' for i in range(1, 6)] + ['shared-plane-support-cuda-v1', 'shared-support-backends-v1']
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
names = [f'diagnose_shared_plane_support_v{i}.py' for i in range(1, 6)] + [f'run_shared_plane_support_v{i}.sh' for i in range(1, 6)] + [
    'traction_union_candidate_v1.py', 'cpu_union_oracle_candidate_v1.py', 'run_shared_plane_support_cuda_v1.sh',
    'compare_shared_support_backends_v1.py', 'archive_online_checkpoint_v7.py', 'run_archive_online_checkpoint_v7.sh']
files.extend(root / name for name in names)
for run, backend in [('shared-plane-support-v5', 'cpu'), ('shared-plane-support-cuda-v1', 'cuda')]:
    summary = json.loads((root / run / 'results/summary.json').read_text())
    assert summary['backend'] == backend
    assert len(summary['records']) == 2
    assert all(len(record['fit']) == 3 and all(fit['native'][0]['status'] == 0 and 'oracle' in fit for fit in record['fit'])
        for record in summary['records'])
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
manifest = {'scope': __doc__, 'source_base_revision': 'e1ab516321884034e73da93bbb7dac3e5e8a37c9',
    'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note': 'No production selection or throughput claim. The explicit union keeps each constituent intrinsic disk radius and verifies individual original wrenches as well as an aggregate comparison. Six CPU and six CUDA full recovery cases pass against independent FP64 convex-dual/root-polished oracle; old failed controls retained. Public configuration, complete same-pair pooling/streaming and broad live/lifecycle validation remain necessary.',
    'files': entries}
path = destination / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination / entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(e['bytes'] for e in entries),
    'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
