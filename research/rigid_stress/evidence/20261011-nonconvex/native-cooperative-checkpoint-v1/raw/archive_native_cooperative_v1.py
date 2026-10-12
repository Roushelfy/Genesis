"""Publish closed native scheduling checks, current-law timings and explicitly unselected alternatives."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-cooperative-checkpoint-v1")
assert not destination.exists()
folders = [
    "native-cooperative-regression-v1", "native-cooperative-profiles-v1", "native-cooperative-profiles-v2",
    "native-cooperative-kernels-v1", "native-cooperative-live-v1", "native-cooperative-validation-v1",
    "validate-native-cooperative-v1", "workspace-seed-regression-v1", "workspace-seed-live-v1",
    "policy-pair-aggregate-socp-v1", "policy-boundary-certificate-v1", "policy-boundary-certificate-v2",
    "policy-boundary-continuous-v1",
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
    assert "55 passed" in (root / f"native-cooperative-regression-v1/{backend}/tests.log").read_text()
assert "55 passed" in (root / "workspace-seed-regression-v1/cuda/tests.log").read_text()
quality = json.loads((root / "native-cooperative-validation-v1/summary.json").read_text())
assert quality["maximum_complete_residual_budget_fraction"] < 1.0
assert quality["maximum_cross_scheduler_peak_budget_fraction"] < 1.0


def compare_live(folder, variants):
    results = [json.loads((root / folder / variant / "result.json").read_text()) for variant in variants]
    reference, candidate = results
    for name in (
        "gpu_uuid", "dt", "substeps", "envs", "conditions_sha256", "condition_count", "load_model", "precision",
        "dofs", "tetrahedra", "quadrature", "output_mode", "method_selected", "native_buffer_bytes",
        "effective_contact_block_size", "intrinsic_support", "egg_position_m",
    ):
        assert reference[name] == candidate[name], (folder, name)
    assert len(reference["warmup_quality"]) == len(candidate["warmup_quality"])
    for left, right in zip(reference["warmup_quality"], candidate["warmup_quality"]):
        for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts",
                     "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
            assert left[name] == right[name], (folder, left["tick"], name)
        np.testing.assert_allclose(right["maximum_stress_Pa"], left["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
        for row in (left, right):
            assert (np.array(row["residual_N"]) <= np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))).all()
    np.testing.assert_allclose(candidate["maximum_stress_Pa"], reference["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
    reports = {}
    for variant, result in zip(variants, results):
        assert len(result["repeats"]) == 3
        for row in result["repeats"]:
            assert row["steps"] == 1280 and row["resets"] == 37
            assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
            assert len(row["failed_steps_per_environment"]) == 32
            assert max(row["failed_steps_per_environment"]) == 0
            assert abs(row["transitions_per_second"] - 32 * row["batch_steps_per_second"]) < 1e-9
        seconds = sum(row["seconds"] for row in result["repeats"])
        reports[variant] = {
            "aggregate_env_steps_per_second": 3 * 1280 * 32 / seconds,
            "aggregate_batch_steps_per_second": 3 * 1280 / seconds,
            "repeats": result["repeats"], "gpu_uuid": result["gpu_uuid"],
            "native_buffer_bytes": result["native_buffer_bytes"],
        }
    gain = reports[variants[1]]["aggregate_env_steps_per_second"] / reports[variants[0]]["aggregate_env_steps_per_second"]
    return {"variants": reports, "aggregate_speedup": gain}


live = compare_live("native-cooperative-live-v1", ("scalar", "cooperative"))
seeded = compare_live("workspace-seed-live-v1", ("native", "seeded"))
for backend in ("cpu", "cuda"):
    ray = json.loads((root / f"policy-boundary-certificate-v2/{backend}/results/summary.json").read_text())[-1]
    assert ray["status"] == [3, 6]
    assert ray["search_bound"][1] < -4e-4 and ray["certificate_margin"][1] > 1e-3
continuous = json.loads((root / "policy-boundary-continuous-v1/results/summary.json").read_text())
assert continuous["classification"] == "continuous_incompatible"
assert continuous["conservative_polar_upper_bound"] < -4e-4
aggregate = json.loads((root / "policy-pair-aggregate-socp-v1/results/summary.json").read_text())[0]
assert aggregate["status"] == "Solved" and aggregate["force_error"] < 1e-11 and aggregate["moment_error"] < 1e-11
for name in (
    "run_native_cooperative_regression_v1.sh", "run_native_cooperative_profiles_v1.sh", "run_native_cooperative_profiles_v2.sh",
    "run_native_cooperative_kernels_v1.sh", "run_native_cooperative_live_v1.sh",
    "validate_native_cooperative_v1.py", "run_validate_native_cooperative_v1.sh",
    "traction_workspace_seed_candidate_v1.py", "probe_workspace_seed_v1.py", "run_workspace_seed_regression_v1.sh", "run_workspace_seed_live_v1.sh",
    "probe_policy_pair_aggregate_socp_v1.py", "run_policy_pair_aggregate_socp_v1.sh",
    "continuum_boundary_certificate_candidate_v1.py", "probe_policy_boundary_certificate_v1.py", "probe_policy_boundary_certificate_v2.py",
    "run_policy_boundary_certificate_v1.sh", "run_policy_boundary_certificate_v2.sh", "probe_policy_boundary_continuous_v1.py", "run_policy_boundary_continuous_v1.sh",
    "archive_native_cooperative_v1.py", "run_archive_native_cooperative_v1.sh",
):
    files.append(root / name)
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
        entries.append({"original": str(original), "artifact": str(artifact.relative_to(destination)), "bytes": original.stat().st_size, "sha256": digest})
manifest = {
    "scope": __doc__, "source_base_revision": "a509a6e8639026e5031e97d1cafbbd740a2dd54d",
    "selected_native_scheduler": "CUDA sample-cooperative fit, validation and signed scatter with canonical masked cold retry; CPU scalar",
    "native_regression": {"cpu_pass": 55, "cuda_pass": 55, "cpu_seconds": 460.57, "cuda_seconds": 681.45},
    "same_input_profile_validation": quality, "native_matched_live": live,
    "unselected_workspace_dual_guess": seeded,
    "unselected_native_boundary_diagnosis": {"independent_cpu_continuous_bound": continuous, "note": "Diagnostic cloud is not production-selected. Depth8 remains unresolved; depth12 proves the original component incompatible without changing its inputs."},
    "unselected_aggregate_model_feasibility": aggregate,
    "limitations": "Only the current repaired law and B32 live scheduling are selected. Workspace dual guesses, diagnostic-cloud storage and aggregate-pair partitioning remain unselected. No complete current-law policy or large-batch four-scope throughput, plateau, physical convergence or overall goal completion is claimed. Failed profile construction and unresolved depth8 diagnostics are retained.",
    "files": entries, "external_files": external,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "native_live": live, "unselected_seeded_live": seeded}, indent=2), flush=True)
