"""Publish complete selected local-integration and numerical-comparison evidence."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v3')
if destination.exists():
    raise FileExistsError(destination)
folders = [
    'continuum-cone-v5', 'continuum-cone-v6', 'online-mesh64-gjk-v1',
    'nearby-fan-diagnosis-v1',
    'surface-fan-fit-v1', 'surface-fan-fit-v2', 'surface-fan-fit-v3', 'surface-fan-fit-v4', 'surface-fan-fit-v5',
    'surface-fan-regression-v1', 'surface-fan-regression-v2', 'surface-fan-regression-v3',
    'surface-fan-oracle-v3',
    'online-surface-fan-v1', 'online-surface-fan-v2', 'online-surface-fan-v3',
    'dynamics-surface-fan-v1', 'rigid-twin-v1', 'rigid-immutability-v1', 'rigid-immutability-v2',
]
files = []
for folder in folders:
    path = root / folder
    assert path.is_dir(), path
    for before in path.rglob('source.sha256'):
        after = before.with_name('source-after.sha256')
        assert after.exists() and before.read_bytes() == after.read_bytes(), before
    files.extend(p for p in path.rglob('*') if p.is_file() and not {'cache', 'tmp', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(path).parts))
prefixes = (
    'run_continuum_cones_v5', 'run_continuum_cones_v6', 'audit_continuum_cones_v5', 'audit_continuum_cones_v6',
    'run_online_mesh64_gjk_v1', 'audit_nearby_fans_v1', 'run_nearby_fan_diagnosis_v1',
    'diagnose_native_surface_fans_', 'run_surface_fan_fit_', 'run_surface_fan_regression_',
    'run_surface_fan_oracle_v3', 'run_online_surface_fan_', 'extract_surface_fan_fixture',
    'run_dynamics_surface_fan_v1', 'run_rigid_twin_v1', 'probe_rigid_twin_v1',
    'run_rigid_immutability_', 'probe_recovery_rigid_immutability_',
)
files.extend(p for p in root.iterdir() if p.is_file() and (p.name.startswith(prefixes) or any(p.name == folder + '-allocation.log' for folder in folders)))
files.extend([Path(__file__), root / 'run_archive_online_checkpoint_v3.sh', root / 'summarize_surface_fan_checkpoint_v1.py', root / 'surface-fan-checkpoint-summary.json'])
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    relative = original.relative_to(root)
    artifact = destination / 'raw' / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({'original': str(original), 'artifact': str(artifact.relative_to(destination)), 'bytes': original.stat().st_size, 'sha256': digest})
manifest = {
    'scope': __doc__, 'source_base_revision': '463ac6c010d0c1b3e4cf58d599f727ce16966f9b',
    'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note': 'Exact complete artifacts including development failures. Source snapshots and before/after hashes identify each tested state. Failed or partial trajectories are not acceptance. The nonconvex goal and new-law performance work remain incomplete.',
    'files': entries,
}
manifest_path = destination / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
for item in entries:
    assert hashlib.sha256(Path(item['original']).read_bytes()).hexdigest() == item['sha256']
    assert hashlib.sha256((destination / item['artifact']).read_bytes()).hexdigest() == item['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(x['bytes'] for x in entries), 'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}), flush=True)
