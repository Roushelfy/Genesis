"""Preserve direct-patch source audits, longer trajectory failures, finite-support diagnoses and warp profiling."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex')
destination = Path('research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v5')
if destination.exists():
    raise FileExistsError(destination)
folders = [
    'online-source-geometry-v1', 'source-geometry-audit-v1',
    'online-surface-fan-gjk-patch-v1', 'online-surface-fan-gjk-patch-v2', 'online-surface-fan-gjk-patch-v3',
    'surface-fan-fit-v6', 'surface-fan-fit-v7', 'surface-fan-fit-v8', 'surface-fan-stable-v1',
    'surface-fan-power-v1', 'continuum-wrench-v1', 'continuum-wrench-v2',
    'warp-candidate-tests-v1', 'warp-candidate-profile-v1',
]
files = []
for folder in folders:
    path = root / folder
    assert path.is_dir(), path
    for before in path.rglob('source.sha256'):
        after = before.with_name('source-after.sha256')
        assert after.exists() and before.read_bytes() == after.read_bytes(), before
    files.extend(p for p in path.rglob('*') if p.is_file() and not {'cache', 'tmp', '__pycache__', '.pytest_cache'}.intersection(p.relative_to(path).parts))
names = [
    'probe_nonconvex_source_geometry_v1.py', 'audit_source_geometry_v1.py', 'probe_nonconvex_gjk_patch_v1.py',
    'franka_egg_stress_patch_v1.py', 'traction_warp_candidate_v1.py', 'probe_warp_candidate_tests_v1.py',
    'probe_warp_candidate_profile_v1.py', 'diagnose_native_surface_fans_v6.py',
    'diagnose_native_surface_fans_v8.py', 'diagnose_native_surface_fans_v9.py',
    'diagnose_native_surface_fans_v10.py', 'audit_continuum_wrench_v1.py', 'audit_continuum_wrench_v2.py',
    'run_online_source_geometry_v1.sh', 'run_online_surface_fan_gjk_patch_v1.sh',
    'run_online_surface_fan_gjk_patch_v2.sh', 'run_online_surface_fan_gjk_patch_v3.sh',
    'run_surface_fan_fit_v6.sh', 'run_surface_fan_fit_v7.sh', 'run_surface_fan_fit_v8.sh',
    'run_surface_fan_stable_v1.sh', 'run_surface_fan_power_v1.sh',
    'run_continuum_wrench_v1.sh', 'run_continuum_wrench_v2.sh',
    'run_warp_candidate_tests_v1.sh', 'run_warp_candidate_profile_v1.sh',
    'archive_online_checkpoint_v5.py', 'run_archive_online_checkpoint_v5.sh',
    'archive-online-checkpoint-v5/archive.log', 'archive-online-checkpoint-v5-allocation.log',
]
for name in names:
    original = root / name
    assert original.is_file(), original
    files.append(original)
files.extend(p for p in root.iterdir() if p.is_file() and any(p.name == folder + '-allocation.log' for folder in folders))
scalar = json.loads((root / 'warp-candidate-profile-v1/scalar.json').read_text())
warp = json.loads((root / 'warp-candidate-profile-v1/warp.json').read_text())
assert scalar['arguments']['save_conditions'] == warp['conditions']
assert hashlib.sha256(Path(warp['conditions']).read_bytes()).hexdigest() == warp['conditions_sha256']
for key in ('envs', 'stress_configurations', 'contact_model', 'collision_method', 'contact_patch',
            'precision', 'quadrature', 'dt', 'substeps', 'history', 'output_mode'):
    assert scalar[key] == warp[key], key
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
manifest = {
    'scope': __doc__, 'source_base_revision': '1fe4f804d19425772cdb6c6faf53326f2fbf2a00',
    'requirement_revision': '1d6babf77848c46ba616da8bbb7062260a7ccd09',
    'note': 'Exact complete artifacts including legal-contact development failures. Power-two integration passes saved native wrench fits; its full stress oracle and longer live tests are pending in the next source state. One frozen stage profile is not measured end-to-end rollout speedup or an optimization plateau. Active nonconvex goal remains incomplete.',
    'matched_profile': {'scalar_contact_ms': scalar['stage_wall_ms']['intrinsic_contact_pipeline'],
                        'warp_contact_ms': warp['stage_wall_ms']['intrinsic_contact_pipeline'],
                        'scalar_to_warp_stage_ratio': scalar['stage_wall_ms']['intrinsic_contact_pipeline'] / warp['stage_wall_ms']['intrinsic_contact_pipeline']},
    'files': entries,
}
manifest_path = destination / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
for item in entries:
    assert hashlib.sha256(Path(item['original']).read_bytes()).hexdigest() == item['sha256']
    assert hashlib.sha256((destination / item['artifact']).read_bytes()).hexdigest() == item['sha256']
print(json.dumps({'artifacts': len(entries), 'bytes': sum(x['bytes'] for x in entries),
                  'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  'matched_profile': manifest['matched_profile']}), flush=True)
