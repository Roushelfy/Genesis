"""Archive squared finite-contact integration, coordinate-roundoff repair and unresolved producer controls."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v6')
if destination.exists():
    raise FileExistsError(destination)
folders = [
    'surface-fan-square-regression-v1', 'surface-fan-square-regression-v2',
    'online-square-fan-v1', 'square-fan-oracle-failure-v1',
    'square-profiles-v1', 'square-profiles-v2', 'surface-segment-candidate-v1',
    'surface-fan-square-fit-v1', 'continuum-wrench-v3',
    'warp-candidate-tests-v2', 'warp-candidate-online-v1',
    'exact-patch-alignment-v1', 'patch-normals-v1', 'source-replay-v1', 'source-replay-v2', 'source-replay-v3',
    'source-replay-audit-v1', 'source-replay-audit-launch-v1',
]
files = []
for folder in folders:
    path = root / folder
    assert path.is_dir(), path
    for before in path.rglob('source.sha256'):
        after = before.with_name('source-after.sha256')
        assert after.exists() and before.read_bytes() == after.read_bytes(), before
    files.extend(p for p in path.rglob('*') if p.is_file() and
        not {'cache', 'tmp', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(path).parts))
    allocation = root / (folder + '-allocation.log')
    if allocation.exists():
        files.append(allocation)
names = [
    'extract_patch_fixtures_v1.py', 'diagnose_native_surface_fans_v11.py',
    'diagnose_square_profiles_v1.py', 'diagnose_square_profiles_v2.py',
    'surface_segment_candidate_v1.py', 'probe_surface_segment_candidate_v1.py',
    'probe_exact_patch_alignment_v1.py', 'audit_generated_cones_v1.py',
    'gjk_patch_normal_candidate_v1.py', 'probe_patch_normals_v1.py',
    'probe_source_replay_v1.py', 'probe_source_replay_v2.py', 'probe_source_replay_v3.py',
    'audit_source_replay_v1.py', 'run_source_replay_audit_v1.sh',
    'run_surface_fan_square_regression_v1.sh', 'run_surface_fan_square_regression_v2.sh',
    'run_online_square_fan_v1.sh', 'run_square_fan_oracle_failure_v1.sh',
    'run_square_profiles_v1.sh', 'run_square_profiles_v2.sh',
    'run_surface_segment_candidate_v1.sh', 'run_surface_fan_square_fit_v1.sh',
    'run_continuum_wrench_v3.sh', 'run_warp_candidate_tests_v2.sh',
    'run_warp_candidate_online_v1.sh', 'run_exact_patch_alignment_v1.sh',
    'run_patch_normals_v1.sh', 'run_source_replay_v1.sh', 'run_source_replay_v2.sh', 'run_source_replay_v3.sh',
    'archive_online_checkpoint_v6.py', 'run_archive_online_checkpoint_v6.sh',
]
for name in names:
    path = root / name
    assert path.is_file(), path
    files.append(path)
for backend in ('cpu', 'cuda'):
    log = (root / 'surface-fan-square-regression-v2' / backend / 'tests.log').read_text()
    assert '45 passed' in log and 'failed' not in log, backend
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    relative = original.relative_to(root)
    artifact = destination / 'raw' / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({'original': str(original), 'artifact': str(artifact.relative_to(destination)),
        'bytes': original.stat().st_size, 'sha256': digest})
manifest = {'scope': __doc__,
    'source_base_revision': '897b5172fca34125046e0c1c6544e3e2ba742bc8',
    'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note': 'The final squared-coordinate/operand-aware-plane source passes 45 CPU and 45 CUDA cases. Earlier profile-oracle failures are retained. Live B32 and experimental warp B16 trajectories remain incomplete at physically incompatible producer inputs. Exact patch alignment and patch-witness-normal controls do not repair tick710. No new end-to-end throughput or optimization plateau claim.',
    'files': entries}
path = destination / 'manifest.json'
path.write_text(json.dumps(manifest, indent=2)+'\n')
for entry in entries:
    assert hashlib.sha256(Path(entry['original']).read_bytes()).hexdigest() == entry['sha256']
    assert hashlib.sha256((destination / entry['artifact']).read_bytes()).hexdigest() == entry['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(e['bytes'] for e in entries),
    'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
