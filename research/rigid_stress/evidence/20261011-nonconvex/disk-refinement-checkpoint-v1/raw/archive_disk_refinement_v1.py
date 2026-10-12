"""Publish closed native disk refinement, matched scheduling and actual policy failure evidence."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/disk-refinement-checkpoint-v1")
assert not destination.exists()
folders = [
    "native-circle-refinement-probe-v1", "native-circle-refinement-suite-v1",
    "native-circle-refinement-suite-v2", "native-circle-refinement-suite-v3",
    "polar-cap-refinement-v1", "polar-cap-refinement-v2", "polar-cap-refinement-v3",
    "encoded-circle-refinement-v1", "native-disk-torque-v1", "polar-torque-refinement-v1",
    "verified-circle-matched-b32-v1", "policy-failure-capture-v1", "policy-failure-capture-v2",
    "audit-policy-failure-inputs-v1", "policy-continuous-v1", "policy-triangle-clip-v1",
]
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
files = []
for folder in folders:
    assert (root / folder).is_dir(), folder
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
for backend in ("cpu", "cuda"):
    assert "54 passed" in (root / f"native-circle-refinement-suite-v3/{backend}/combined.log").read_text()
    assert "2 passed" in (root / f"native-circle-refinement-suite-v3/{backend}/targeted.log").read_text()
for pattern in (
    "run_native_circle_refinement_*v1.sh", "run_native_circle_refinement_suite_v2.sh", "run_native_circle_refinement_suite_v3.sh",
    "run_polar_cap_refinement_v*.sh", "probe_polar_cap_refinement_v*.py", "polar_cap_refinement_candidate_v*.py",
    "run_polar_torque_refinement_v1.sh", "probe_polar_torque_refinement_v1.py",
    "run_native_disk_torque_v1.sh", "probe_native_disk_torque_v1.py",
    "run_encoded_circle_refinement_v1.sh", "probe_encoded_circle_refinement_v1.py", "encoded_circle_refinement_candidate_v1.py",
    "run_verified_circle_matched_scopes_v1.sh", "probe_cooperative_fallback_candidate_v1.py",
    "traction_original_point_candidate_v10.py", "traction_sample_cooperative_candidate_v2.py",
    "traction_scalar_fallback_candidate_v1.py", "traction_cooperative_fallback_candidate_v1.py",
    "run_policy_failure_capture_v*.sh", "probe_policy_failure_capture_v*.py",
    "run_audit_policy_failure_inputs_v1.sh", "audit_policy_failure_inputs_v1.py",
    "run_policy_continuous_v1.sh", "probe_policy_continuous_v1.py",
    "run_policy_triangle_clip_v1.sh", "probe_policy_triangle_clip_v1.py", "triangle_disk_clip_candidate_v1.py",
    "archive_disk_refinement_v1.py", "run_archive_disk_refinement_v1.sh",
):
    matched = list(root.glob(pattern))
    assert matched, pattern
    files.extend(matched)
rates = {}
for variant in ("scalar", "cooperative-fallback"):
    rates[variant] = {}
    base = root / "verified-circle-matched-b32-v1" / variant
    for scope in ("rigid", "recovery", "live"):
        result = json.loads((base / scope / "result.json").read_text())
        assert result["envs"] == 32
        assert all(row["invalid_environment_steps"] == 0 for row in result["repeats"])
        assert all(row["scatter_overflow_calls"] == 0 for row in result["repeats"])
        rates[variant][scope] = [row["transitions_per_second"] for row in result["repeats"]]
    assert not (base / "policy/result.json").exists()
    assert "Native stress contact recovery rejected a load" in (base / "policy/benchmark.log").read_text()
metadata = json.loads((root / "policy-failure-capture-v2/failure/metadata.json").read_text())
continuous = json.loads((root / "policy-continuous-v1/results/summary.json").read_text())
assert continuous["classification"] == "continuum_unresolved" and continuous["conservative_polar_upper_bound"] > 0
destination.mkdir(parents=True)
entries, external = [], []
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000 or (original.suffix == ".npz" and original.stat().st_size >= 1_000_000):
        external.append({"original": str(original), "bytes": original.stat().st_size, "sha256": digest})
    else:
        artifact = destination / "raw" / original.relative_to(root)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, artifact)
        entries.append({"original": str(original), "artifact": str(artifact.relative_to(destination)),
                        "bytes": original.stat().st_size, "sha256": digest})
manifest = {
    "scope": __doc__, "source_base_revision": "f26f09d8780b334464e9798307c70a3e3e71c8f7",
    "native_disk_refinement": {"cpu_pass": 54, "cuda_pass": 54, "cpu_seconds": 454.41, "cuda_seconds": 672.04},
    "matched_preceding_circle_model": {"envs": 32, "rates": rates,
        "live_mean_gain": float(np.mean(rates["cooperative-fallback"]["live"]) / np.mean(rates["scalar"]["live"])),
        "policy": "both variants rejected a load; no valid complete policy throughput"},
    "actual_policy_failure": metadata, "continuous_ray_audit": continuous,
    "failed_epochs": "First native import lacks supported QD kernel signature; first test helper rejects empty shape tuples; extended prototype fields do not flatten; centroid-only torque misses sampled capacity; initial clip audit has CPU/native selected-count disagreement. None counts as acceptance.",
    "note": "Goal remains incomplete. Native cooperative selection, new-model baseline, general refinement, policy acceptance and larger batch/plateau are open.",
    "files": entries, "external_files": external,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert Path(entry["original"]).stat().st_size == entry["bytes"]
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(e["bytes"] for e in entries),
    "external_artifacts": len(external), "external_bytes": sum(e["bytes"] for e in external),
    "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
