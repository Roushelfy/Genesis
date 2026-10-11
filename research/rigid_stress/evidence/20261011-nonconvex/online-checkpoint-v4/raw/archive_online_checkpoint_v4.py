"""Publish initial intrinsic profiling and the diagnosed B16 source-load failure."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v4')
if destination.exists():
    raise FileExistsError(destination)
folders = [
    'profile-surface-fan-v1', 'profile-surface-fan-v2', 'online-surface-fan-v4',
    'online-surface-fan-gjk-v1', 'online-surface-fan-mpr-accuracy-v1', 'continuum-cone-v7',
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
    'run_profile_surface_fan_', 'run_online_surface_fan_v4', 'run_online_surface_fan_gjk_v1',
    'run_online_surface_fan_mpr_accuracy_v1', 'probe_nonconvex_mpr_accuracy_v1',
    'audit_continuum_cones_v7', 'continuum-cone-v7.log',
)
files.extend(p for p in root.iterdir() if p.is_file() and (p.name.startswith(prefixes) or any(p.name == folder + '-allocation.log' for folder in folders)))
files.extend([Path(__file__), root / 'run_archive_online_checkpoint_v4.sh'])
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
    'scope': __doc__, 'source_base_revision': '209ad577384a2548bb2173498322fbc9fabd9d03',
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
