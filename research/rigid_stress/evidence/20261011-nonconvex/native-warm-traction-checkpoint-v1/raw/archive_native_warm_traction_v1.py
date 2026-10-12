"""Publish actual native warm-start qualification and same-GPU cold/warm profile/live comparisons."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/native-warm-traction-checkpoint-v1")
assert not destination.exists()
folders = ["native-warm-traction-regression-v1", "native-warm-traction-benchmark-v1"]
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
files = []
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
assert "55 passed" in (root / "native-warm-traction-regression-v1/cuda/tests.log").read_text()
reports = {}
for scope in ("profile", "live"):
    results = [json.loads((root / f"native-warm-traction-benchmark-v1/{scope}/{variant}/result.json").read_text()) for variant in ("cold", "warm")]
    left, right = results
    for name in ("gpu_uuid", "dt", "substeps", "envs", "conditions_sha256", "condition_count", "seed", "load_model", "precision", "dofs", "tetrahedra", "quadrature", "output_mode", "method_selected", "native_buffer_bytes", "effective_contact_block_size", "intrinsic_support", "egg_position_m"):
        assert left[name] == right[name], (scope, name)
    assert not left["warm_start_tractions"] and right["warm_start_tractions"]
    assert left["cooperative_tractions"] and right["cooperative_tractions"]
    assert len(left["warmup_quality"]) == len(right["warmup_quality"])
    maximum_peak_difference = 0.0
    maximum_residual_budget_fraction = 0.0
    for a, b in zip(left["warmup_quality"], right["warmup_quality"]):
        for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts", "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
            assert a[name] == b[name], (scope, a["tick"], name)
        np.testing.assert_allclose(b["maximum_stress_Pa"], a["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
        maximum_peak_difference = max(maximum_peak_difference, np.max(np.abs(np.array(a["maximum_stress_Pa"]) - np.array(b["maximum_stress_Pa"]))))
        for row in (a, b):
            fraction = np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))
            assert fraction <= 1.0
            maximum_residual_budget_fraction = max(maximum_residual_budget_fraction, fraction)
    np.testing.assert_allclose(right["maximum_stress_Pa"], left["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
    report = {"gpu_uuid": left["gpu_uuid"], "maximum_logged_peak_difference_Pa": maximum_peak_difference, "maximum_complete_residual_budget_fraction": maximum_residual_budget_fraction, "native_buffer_bytes": left["native_buffer_bytes"]}
    if scope == "profile":
        for result in results:
            qa = result["recovery_snapshot_check"]
            assert qa["original_point_max_difference_m"] == 0.0
            assert qa["normalized_force_error_max"] <= 1e-8 and qa["normalized_moment_error_max"] <= 1e-8
        report["stage_wall_ms"] = {variant: result["stage_wall_ms"] for variant, result in zip(("cold", "warm"), results)}
        report["note"] = "Kernel profiler enabled: wall times are intrusive snapshot costs, not rollout throughput."
    else:
        variants = {}
        for variant, result in zip(("cold", "warm"), results):
            assert len(result["repeats"]) == 3
            for row in result["repeats"]:
                assert row["steps"] == 1280 and row["resets"] == 37
                assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
                assert len(row["failed_steps_per_environment"]) == 32 and max(row["failed_steps_per_environment"]) == 0
            elapsed = sum(row["seconds"] for row in result["repeats"])
            variants[variant] = {"aggregate_env_steps_per_second": 32 * 1280 * 3 / elapsed, "aggregate_batch_steps_per_second": 1280 * 3 / elapsed, "repeats": result["repeats"]}
        report["variants"] = variants
        report["aggregate_speedup"] = variants["warm"]["aggregate_env_steps_per_second"] / variants["cold"]["aggregate_env_steps_per_second"]
    reports[scope] = report
for name in ("run_native_warm_traction_regression_v1.sh", "run_native_warm_traction_benchmark_v1.sh", "archive_native_warm_traction_v1.py", "run_archive_native_warm_traction_v1.sh"):
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
manifest = {"scope": __doc__, "source_base_revision": "c560c5e4426aac8dabe9dc6994767f6c7da32ea1", "native_cuda_regression_pass": 55, "cpu_route": "Canonical CPU arithmetic is unchanged from c560c5e4 and its55-case CPU source epoch; no new CPU suite is claimed.", "reports": reports, "independent_live_cpu_fields": "Separate traction-guesses-checkpoint-v1:432 fields,56 saved-state resets and two seeds on the identical warm-guess algorithm.", "note": "Select previous successful workspace duals only as initial guesses. No source identity or previous acceptance is assumed; every current problem and all cold/refinement fallback remain checked. This is B32 live inference-free throughput, not four-scope large-batch acceptance, a plateau or overall completion. Gradient-only trial evaluation and parallel metadata clearing remain separate experiments.", "files": entries, "external_files": external}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reports": reports}, indent=2), flush=True)
