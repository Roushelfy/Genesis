"""Publish complete selected checkpoint artifacts, excluding runtime caches."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex")
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/online-checkpoint-v1")
if destination.exists():
    raise FileExistsError(destination)
folders = [
    "packed-regression-v1-cpu", "packed-regression-v1-cuda", "packed-regression-v2-cpu", "packed-regression-v2-cuda",
    "intrinsic-regression-v1-cpu", "intrinsic-regression-v1-cuda", "intrinsic-regression-v2-cpu", "intrinsic-regression-v2-cuda",
    "intrinsic-regression-v3-cpu", "intrinsic-regression-v4-cpu", "intrinsic-regression-v4-cuda",
    "online-v1", "online-v2", "online-failure-capture-v1", "budget-diagnosis-v1", "budget-diagnosis-v2", "budget-diagnosis-v3",
    "continuum-cone-v1", "continuum-cone-v2", "continuum-cone-v3", "source-edge-v1", "source-edge-v2",
    "online-mesh64-v1", "online-mesh64-v2", "online-mesh64-v3", "dynamics-mesh64-v1", "dynamics-mesh64-v2",
]
files = []
for folder in folders:
    path = root / folder
    if path.exists():
        files.extend(p for p in path.rglob("*") if p.is_file() and not {"cache", "tmp", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(path).parts))
prefixes = ("run_packed_", "run_intrinsic_", "run_online_", "run_budget_", "run_continuum_cones_", "run_source_edge_", "run_dynamics_", "run_fixture_", "export_fixture_", "audit_continuum_cones_", "audit_source_edge_", "diagnose_native_budget_", "nonconvex_", "packed-regression-", "intrinsic-regression-", "online-", "budget-diagnosis-", "continuum-cone-", "source-edge-", "dynamics-", "fixture-")
files.extend(p for p in root.iterdir() if p.is_file() and (p.name.startswith(prefixes) or any(p.name == folder + "-allocation.log" for folder in folders)))
files.extend([Path(__file__), root / "run_archive_online_checkpoint_v1.sh"])
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
    "scope": __doc__, "source_base_revision": "c308371066b91976ae83c77b0c5b53b0a88421f9", "requirement_revision": "1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "note": "Exact complete artifacts, including development failures. Run-specific source files/hashes identify tested states; a failed or partial trajectory is not acceptance. No new-law throughput claim.",
    "files": entries,
}
manifest_path = destination / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
for item in entries:
    assert hashlib.sha256(Path(item["original"]).read_bytes()).hexdigest() == item["sha256"]
    assert hashlib.sha256((destination / item["artifact"]).read_bytes()).hexdigest() == item["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(x["bytes"] for x in entries), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}), flush=True)
