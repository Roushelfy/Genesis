"""Publish experimental sample scheduling and the production verified-circle integration repair, with failed epochs."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/contact-scheduling-circle-checkpoint-v1")
assert not destination.exists(), destination
folders = [
    "original-point-cooperative-v10-v1",
    "original-point-scalar-b32-profile-v1",
    "original-point-cooperative-b32-profile-v2",
    "original-point-cooperative-online-b16-v2",
    "original-point-cooperative-online-b32-v2",
    "sample-contiguous-layout-v11-v1",
    "sample-contiguous-layout-v11-b32-profile-v1",
    "original-point-matched-live-b32-v1",
    "resolved-flat-disk-v1",
    "circle-disk-candidate-v1",
    "circle-disk-candidate-v2",
    "native-circle-integration-v1",
    "native-circle-integration-v2",
]
files = []
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
for folder in folders:
    directory = root / folder
    assert directory.is_dir(), directory
    for current, dirs, names in os.walk(directory):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in directory.rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(directory).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
for backend in ("cpu", "cuda"):
    assert "51 passed" in (root / f"original-point-cooperative-v10-v1/{backend}/tests.log").read_text()
    assert "52 passed" in (root / f"native-circle-integration-v2/combined/{backend}/tests.log").read_text()
    failed = (root / f"circle-disk-candidate-v1/{backend}/probe.log").read_text()
    assert "TypeError" in failed and "8 were given" in failed
    circle = json.loads((root / f"circle-disk-candidate-v2/{backend}/results/summary.json").read_text())
    assert circle["sample_count"] == 100 and circle["evaluations"] == 8
    assert max(circle["force_error"], circle["moment_error"]) < 1e-8
assert "51 passed" in (root / "sample-contiguous-layout-v11-v1/cuda/tests.log").read_text()
assert "1 passed" in (root / "native-circle-integration-v1/targeted/cpu/tests.log").read_text()
assert "1 failed" in (root / "native-circle-integration-v1/targeted/cuda/tests.log").read_text()
assert "Cancelled job 706618" in (root / "native-circle-integration-v1/combined/cpu/termination.txt").read_text()
condition = root / "original-point-scalar-b32-profile-v1/conditions.npz"
condition_sha = hashlib.sha256(condition.read_bytes()).hexdigest()
assert condition_sha == "156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930"
matched = [
    json.loads((root / f"original-point-matched-live-b32-v1/{variant}/live.json").read_text())
    for variant in ("scalar", "cooperative")
]
for key in (
    "gpu_uuid",
    "envs",
    "dt",
    "seed",
    "warmup_steps",
    "precision",
    "quadrature",
    "load_model",
    "contact_model",
    "intrinsic_support",
    "effective_contact_block_size",
    "conditions_sha256",
    "stress_configurations",
    "output_mode",
    "history",
    "reset_mode",
    "collision_method",
    "contact_patch",
):
    assert matched[0][key] == matched[1][key], key
assert matched[0]["envs"] == 32 and matched[0]["conditions_sha256"] == condition_sha
rates = []
for item in matched:
    assert item["scope"] == "live" and len(item["repeats"]) == 3
    variant_rates = []
    for repeat in item["repeats"]:
        assert repeat["steps"] == 1280 and repeat["resets"] == 37
        assert repeat["invalid_environment_steps"] == 0 and repeat["scatter_overflow_calls"] == 0
        assert repeat["failed_steps_per_environment"] == [0] * 32
        assert np.isclose(repeat["transitions_per_second"], 32 * 1280 / repeat["seconds"], rtol=1e-12)
        assert np.isclose(repeat["batch_steps_per_second"], 1280 / repeat["seconds"], rtol=1e-12)
        variant_rates.append(repeat["transitions_per_second"])
    rates.append(variant_rates)
live_rows = []
for envs in (16, 32):
    live = json.loads((root / f"original-point-cooperative-online-b{envs}-v2/results/summary.json").read_text())
    assert live["completed_steps"] == len(live["records"]) == 1280
    assert len(live["oracle_captures"]) == 9
    assert all(len(capture["cpu_peaks_Pa"]) == envs for capture in live["oracle_captures"])
    assert all(max(record["max_force_error"], record["max_moment_error"]) <= 1e-8 for record in live["records"])
    mu = [value for record in live["records"] for value in record["mu_values"]]
    live_rows.append(
        {
            "envs": envs,
            "steps": 1280,
            "seed": live["arguments"]["seed"],
            "cpu_full_fields": 9 * envs,
            "resets": live["records"][-1]["reset_environments"],
            "max_force_error": max(record["max_force_error"] for record in live["records"]),
            "max_moment_error": max(record["max_moment_error"] for record in live["records"]),
            "actual_mu_min": min(mu),
            "actual_mu_max": max(mu),
        }
    )
profiles = []
for folder in (
    "original-point-scalar-b32-profile-v1",
    "original-point-cooperative-b32-profile-v2",
    "sample-contiguous-layout-v11-b32-profile-v1",
):
    profile = json.loads((root / folder / "profile.json").read_text())
    check = profile["recovery_snapshot_check"]
    assert check["original_point_max_difference_m"] == 0
    assert max(check["normalized_force_error_max"], check["normalized_moment_error_max"]) <= 1e-8
    assert (np.array(profile["complete_residual_N"]) <= np.maximum(1e-11, 1e-8 * np.array(profile["rhs_norm_N"]))).all()
    profiles.append(
        {
            "candidate": folder,
            "gpu_uuid": profile["gpu_uuid"],
            "intrinsic_ms": profile["stage_wall_ms"]["intrinsic_contact_pipeline"],
            "stage_wall_ms": profile["stage_wall_ms"],
            "pressure_subdivision_wall_ms": profile["pressure_subdivision_wall_ms"],
            "snapshot_check": check,
        }
    )
legal = json.loads((root / "resolved-flat-disk-v1/results/summary.json").read_text())
assert legal["radius_m"] == 1e-5 and legal["friction"] == 0.5 and legal["sampled_hull_gap_m"] > 4e-6
assert legal["positive_area_patch_radius_m"] > 0 and legal["diagnostic_status"] != 6
summary = {
    "matched_hardware_uuid": matched[0]["gpu_uuid"],
    "scalar_env_steps_per_second": rates[0],
    "cooperative_env_steps_per_second": rates[1],
    "mean_rate_speedup": float(np.mean(rates[1]) / np.mean(rates[0])),
    "profiles": profiles,
    "experimental_live_validation": live_rows,
    "legal_sampling_defect": legal,
    "native_circle_regression": {"cpu": 52, "cuda": 52},
    "note": "Scheduling/layout measurements use unchanged source 047d7de9 plus explicitly archived experimental native kernels. Production circle integration is a separate footprint integration change. No new-model speedup, large-batch four-scope acceptance or plateau is claimed.",
}
names = [
    "traction_original_point_candidate_v10.py",
    "traction_sample_cooperative_candidate_v2.py",
    "probe_original_point_candidate_v10.py",
    "probe_original_point_online_v10.py",
    "probe_sample_layout_candidate_v11.py",
    "probe_resolved_flat_disk_v1.py",
    "circle_disk_candidate_v1.py",
    "circle_disk_candidate_v2.py",
    "probe_circle_disk_candidate_v1.py",
    "probe_circle_disk_candidate_v2.py",
    "run_original_point_candidate_v10.sh",
    "run_original_point_scalar_profile_v1.sh",
    "run_original_point_cooperative_profile_v1.sh",
    "run_original_point_cooperative_profile_v2.sh",
    "run_original_point_cooperative_online_v1.sh",
    "run_original_point_cooperative_online_v2.sh",
    "run_sample_layout_candidate_v11.sh",
    "run_sample_layout_profile_v11_v1.sh",
    "run_original_point_matched_live_v1.sh",
    "run_resolved_flat_disk_v1.sh",
    "run_circle_disk_candidate_v1.sh",
    "run_circle_disk_candidate_v2.sh",
    "run_native_circle_integration_v1.sh",
    "run_native_circle_integration_v2.sh",
    "archive_contact_scheduling_v1.py",
    "run_archive_contact_scheduling_v1.sh",
]
files.extend(root / name for name in names)
for pattern in (
    "original-point-*allocation.log",
    "sample-contiguous-layout-*allocation.log",
    "resolved-flat-disk-*allocation.log",
    "circle-disk-candidate-*allocation.log",
    "native-circle-integration-*allocation.log",
):
    files.extend(root.glob(pattern))
assert all(original.is_file() for original in files)
destination.mkdir(parents=True)
entries, external_entries = [], []
for original in sorted(set(files)):
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if original.stat().st_size >= 20_000_000 or (original.suffix == ".npz" and original.stat().st_size >= 1_000_000):
        external_entries.append({"original": str(original), "bytes": original.stat().st_size, "sha256": digest})
        continue
    artifact = destination / "raw" / original.relative_to(root)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, artifact)
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == digest
    entries.append(
        {
            "original": str(original),
            "artifact": str(artifact.relative_to(destination)),
            "bytes": original.stat().st_size,
            "sha256": digest,
        }
    )
manifest = {
    "scope": __doc__,
    "source_base_revision": "047d7de9e4650e4f31631cdd909659660f16d9e9",
    "requirement_revision": "1d6babf77848c46ba616da8bbb7062260a7ccd09",
    "validation": summary,
    "storage_note": "Runtime NPZ snapshots at least 1 MB and any artifact at least 20 MB stay in the project data directory with verified size/SHA256. Small raw sources, logs, conditions and fixtures are copied to Git. Failed and interrupted epochs are explicit.",
    "files": entries,
    "external_files": external_entries,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external_entries:
    assert Path(entry["original"]).stat().st_size == entry["bytes"]
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(
    json.dumps(
        {
            "validation": summary,
            "artifacts": len(entries),
            "bytes": sum(entry["bytes"] for entry in entries),
            "external_artifacts": len(external_entries),
            "external_bytes": sum(entry["bytes"] for entry in external_entries),
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    ),
    flush=True,
)
