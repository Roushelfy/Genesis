"""Publish complete selected checkpoint artifacts, excluding runtime caches."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v2")
if destination.exists():
    raise FileExistsError(destination)
folders = [
    "source-regression-v5", "online-mesh64-v4", "dynamics-mesh64-v3",
    "mesh64-fit-diagnosis-v1", "mesh64-fit-diagnosis-v2", "mesh64-fit-diagnosis-v3", "continuum-cone-v4",
]
files = []
for folder in folders:
    path = root / folder
    if path.exists():
        files.extend(p for p in path.rglob("*") if p.is_file() and not {"cache", "tmp", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(path).parts))
prefixes = ("run_source_regression_v5", "run_online_mesh64_v4", "run_dynamics_mesh64_v3", "run_mesh64_fit_diagnosis_", "diagnose_native_mesh64_fit_", "run_continuum_cones_v4", "audit_continuum_cones_v4", "run_fixture_edge_v2", "export_fixture_edge_v2")
files.extend(p for p in root.iterdir() if p.is_file() and (p.name.startswith(prefixes) or any(p.name == folder + "-allocation.log" for folder in folders)))
files.extend([Path(__file__), root / "run_archive_online_checkpoint_v2.sh"])
destination.mkdir(parents=True)
entries = []
for original in sorted(set(files)):
    relative = original.relative_to(root)
    artifact = destination / "raw" / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append({"original": str(original), "artifact": str(artifact.relative_to(destination)), "bytes": original.stat().st_size, "sha256": digest})
manifest = {
    "scope": __doc__, "source_base_revision": "026974987ee6a4e1e1a21bd0d414a103be700a71", "requirement_revision": "1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "note": "Exact complete artifacts, including development failures. Run-specific source files/hashes identify tested states; a failed or partial trajectory is not acceptance. No new-law throughput claim.",
    "files": entries,
}
manifest_path = destination / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
for item in entries:
    assert hashlib.sha256(Path(item["original"]).read_bytes()).hexdigest() == item["sha256"]
    assert hashlib.sha256((destination / item["artifact"]).read_bytes()).hexdigest() == item["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(x["bytes"] for x in entries), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}), flush=True)
