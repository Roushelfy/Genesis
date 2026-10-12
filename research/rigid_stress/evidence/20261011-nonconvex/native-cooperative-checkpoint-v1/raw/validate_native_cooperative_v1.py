"""Validate closed scalar/cooperative profiles and their unchanged numerical budgets on CPU compute."""

import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = root / "native-cooperative-validation-v1"
destination.mkdir(exist_ok=False)
folder = root / "native-cooperative-profiles-v2"
assert (folder / "source.sha256").read_bytes() == (folder / "source-after.sha256").read_bytes()
results = [json.loads((folder / variant / "result.json").read_text()) for variant in ("scalar", "cooperative")]
baseline, candidate = results
for name in (
    "gpu_uuid", "dt", "substeps", "envs", "level", "dofs", "tetrahedra", "quadrature", "load_model",
    "conditions_sha256", "condition_count", "precision", "method_selected", "inverse_precision",
    "surface_inverse_selected", "output_mode", "intrinsic_support", "effective_contact_block_size",
    "full_surface_samples", "native_buffer_bytes",
):
    assert baseline[name] == candidate[name], name
assert not baseline["cooperative_tractions"] and candidate["cooperative_tractions"]
assert len(baseline["warmup_quality"]) == len(candidate["warmup_quality"])
maximum_peak_difference = 0.0
maximum_peak_budget_fraction = 0.0
maximum_residual_budget_fraction = 0.0
maximum_force_error = 0.0
maximum_moment_error = 0.0
quality_states = 0
for result in results:
    assert result["recovery_snapshot_check"]["original_point_max_difference_m"] == 0.0
    assert result["recovery_snapshot_check"]["normalized_force_error_max"] <= 1e-8
    assert result["recovery_snapshot_check"]["normalized_moment_error_max"] <= 1e-8
    maximum_force_error = max(maximum_force_error, result["recovery_snapshot_check"]["normalized_force_error_max"])
    maximum_moment_error = max(maximum_moment_error, result["recovery_snapshot_check"]["normalized_moment_error_max"])
    residual = np.array(result["complete_residual_N"])
    rhs = np.array(result["rhs_norm_N"])
    fraction = np.max(residual / np.maximum(1e-11, 1e-8 * rhs))
    assert fraction <= 1.0
    maximum_residual_budget_fraction = max(maximum_residual_budget_fraction, fraction)
    for row in result["warmup_quality"]:
        residual, rhs = np.array(row["residual_N"]), np.array(row["rhs_N"])
        fraction = np.max(residual / np.maximum(1e-11, 1e-8 * rhs))
        assert fraction <= 1.0
        maximum_residual_budget_fraction = max(maximum_residual_budget_fraction, fraction)
        quality_states += len(rhs)
for left, right in zip(baseline["warmup_quality"], candidate["warmup_quality"]):
    for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts",
                 "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
        assert left[name] == right[name], (left["tick"], name)
    left_peak, right_peak = np.array(left["maximum_stress_Pa"]), np.array(right["maximum_stress_Pa"])
    difference = np.abs(left_peak - right_peak)
    fraction = np.max(difference / (1e-3 + 1e-4 * np.abs(left_peak)))
    assert fraction <= 1.0
    maximum_peak_difference = max(maximum_peak_difference, np.max(difference))
    maximum_peak_budget_fraction = max(maximum_peak_budget_fraction, fraction)
peak_difference = np.abs(np.array(baseline["maximum_stress_Pa"]) - np.array(candidate["maximum_stress_Pa"]))
fraction = np.max(peak_difference / (1e-3 + 1e-4 * np.abs(baseline["maximum_stress_Pa"])))
assert fraction <= 1.0
maximum_peak_difference = max(maximum_peak_difference, np.max(peak_difference))
maximum_peak_budget_fraction = max(maximum_peak_budget_fraction, fraction)
for backend in ("cpu", "cuda"):
    test_folder = root / "native-cooperative-regression-v1" / backend
    assert (test_folder / "source.sha256").read_bytes() == (test_folder / "source-after.sha256").read_bytes()
    assert "55 passed" in (test_folder / "tests.log").read_text()
summary = {
    "profiles": str(folder), "same_gpu_uuid": baseline["gpu_uuid"], "law": baseline["load_model"],
    "native_tests": {"cpu_pass": 55, "cuda_pass": 55}, "quality_environment_states": quality_states,
    "maximum_cross_scheduler_peak_difference_Pa": maximum_peak_difference,
    "maximum_cross_scheduler_peak_budget_fraction": maximum_peak_budget_fraction,
    "maximum_complete_residual_budget_fraction": maximum_residual_budget_fraction,
    "maximum_normalized_original_point_force_error": maximum_force_error,
    "maximum_normalized_original_point_moment_error": maximum_moment_error,
    "stage_wall_ms": {variant: result["stage_wall_ms"] for variant, result in zip(("scalar", "cooperative"), results)},
    "intrinsic_snapshot_wall_speedup": baseline["stage_wall_ms"]["intrinsic_contact_pipeline"] / candidate["stage_wall_ms"]["intrinsic_contact_pipeline"],
    "note": "Profiling has stage wall timing only: this epoch did not enable QD kernel profiling. Cross-scheduler agreement supplements the independent CPU oracle tests; it is not an independent oracle itself. Actual live/policy/large-batch throughput is separate.",
    "inputs": {str(folder / variant / "result.json"): hashlib.sha256((folder / variant / "result.json").read_bytes()).hexdigest() for variant in ("scalar", "cooperative")},
}
(destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
