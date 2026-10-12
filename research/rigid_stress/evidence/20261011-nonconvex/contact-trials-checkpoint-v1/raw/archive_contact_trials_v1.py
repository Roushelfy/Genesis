"""Publish parallel contact clearing, rejected gradient trials and on-demand physical certificates."""

import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

root = Path(__file__).parent
destination = Path("research/rigid_stress/evidence/20261011-nonconvex/contact-trials-checkpoint-v1")
assert not destination.exists()
folders = [
    "parallel-contact-clear-regression-v1", "parallel-contact-clear-benchmark-v1",
    "native-parallel-clear-regression-v1", "gradient-trial-regression-v1",
    "gradient-trial-benchmark-v1", "boundary-penalty-certificate-v1",
    "boundary-penalty-continuous-v1",
]
skip = {"cache", "tmp", "__pycache__", ".pytest_cache"}
files = []
for folder in folders:
    for current, dirs, names in os.walk(root / folder):
        dirs[:] = [name for name in dirs if name not in skip]
        files.extend(Path(current) / name for name in names)
    for before in (root / folder).rglob("source.sha256"):
        if not any(part in skip for part in before.relative_to(root / folder).parts):
            assert before.read_bytes() == before.with_name("source-after.sha256").read_bytes(), before
for folder, backend, count in [
    ("parallel-contact-clear-regression-v1", "cuda", 9),
    ("native-parallel-clear-regression-v1", "cpu", 9),
    ("native-parallel-clear-regression-v1", "cuda", 9),
    ("gradient-trial-regression-v1", "cuda", 55),
]:
    assert f"{count} passed" in (root / folder / backend / "tests.log").read_text()

def packing_ast(path):
    return ast.dump(next(node for node in ast.parse(path.read_text()).body
                         if isinstance(node, ast.FunctionDef) and node.name == "kernel_pack_intrinsic_contacts"))

assert packing_ast(Path("genesis/engine/solvers/rigid/stress/intrinsic.py")) == packing_ast(root / "parallel_contact_clear_candidate_v1.py")
reports = {}
for experiment, variant in [("parallel-contact-clear", "parallel_clear"), ("gradient-trial", "gradient_trial")]:
    scopes = {}
    for scope in ("profile", "live"):
        results = [json.loads((root / f"{experiment}-benchmark-v1/{scope}/{name}/result.json").read_text()) for name in ("native", variant)]
        left, right = results
        for name in ("gpu_uuid", "dt", "substeps", "envs", "conditions_sha256", "condition_count", "seed", "load_model", "precision", "dofs", "tetrahedra", "quadrature", "output_mode", "method_selected", "native_buffer_bytes", "effective_contact_block_size", "intrinsic_support", "egg_position_m", "cooperative_tractions", "warm_start_tractions"):
            assert left[name] == right[name], (experiment, scope, name)
        assert left["cooperative_tractions"] and left["warm_start_tractions"]
        assert len(left["warmup_quality"]) == len(right["warmup_quality"]) == 60
        maximum_peak_difference = 0.0
        maximum_residual_budget_fraction = 0.0
        for a, b in zip(left["warmup_quality"], right["warmup_quality"]):
            for name in ("tick", "phase", "height_m", "finger_contacts", "nonzero_egg_contacts", "contacts", "tangential_force_N", "contact_radius_range_m", "contact_friction_range"):
                assert a[name] == b[name], (experiment, scope, a["tick"], name)
            np.testing.assert_allclose(b["maximum_stress_Pa"], a["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
            maximum_peak_difference = max(maximum_peak_difference, float(np.max(np.abs(np.array(a["maximum_stress_Pa"]) - np.array(b["maximum_stress_Pa"])))))
            for row in (a, b):
                fraction = float(np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"]))))
                assert fraction <= 1.0
                maximum_residual_budget_fraction = max(maximum_residual_budget_fraction, fraction)
        np.testing.assert_allclose(right["maximum_stress_Pa"], left["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
        report = {"gpu_uuid": left["gpu_uuid"], "maximum_logged_peak_difference_Pa": maximum_peak_difference, "maximum_complete_residual_budget_fraction": maximum_residual_budget_fraction, "native_buffer_bytes": left["native_buffer_bytes"]}
        if scope == "profile":
            for result in results:
                qa = result["recovery_snapshot_check"]
                assert qa["original_point_max_difference_m"] == 0.0
                assert qa["normalized_force_error_max"] <= 1e-8 and qa["normalized_moment_error_max"] <= 1e-8
            report["stage_wall_ms"] = {name: result["stage_wall_ms"] for name, result in zip(("native", variant), results)}
            report["note"] = "Kernel profiler enabled: intrusive snapshot costs do not measure rollout throughput."
        else:
            variants = {}
            for name, result in zip(("native", variant), results):
                assert len(result["repeats"]) == 3
                for row in result["repeats"]:
                    assert row["steps"] == 1280 and row["resets"] == 37
                    assert row["invalid_environment_steps"] == 0 and row["scatter_overflow_calls"] == 0
                    assert len(row["failed_steps_per_environment"]) == 32 and max(row["failed_steps_per_environment"]) == 0
                elapsed = sum(row["seconds"] for row in result["repeats"])
                variants[name] = {"aggregate_env_steps_per_second": 32 * 1280 * 3 / elapsed, "aggregate_batch_steps_per_second": 1280 * 3 / elapsed, "repeats": result["repeats"]}
            report["variants"] = variants
            report["aggregate_speedup"] = variants[variant]["aggregate_env_steps_per_second"] / variants["native"]["aggregate_env_steps_per_second"]
        scopes[scope] = report
    reports[experiment] = scopes
assert reports["parallel-contact-clear"]["live"]["aggregate_speedup"] > 1.0
assert reports["gradient-trial"]["live"]["aggregate_speedup"] < 1.0

diagnostics = {}
for backend in ("cpu", "cuda"):
    rows = json.loads((root / f"boundary-penalty-certificate-v1/{backend}/results/summary.json").read_text())
    assert rows[0]["status"] == [0, 3] and rows[-1]["status"] == [0, 6]
    assert rows[-1]["search_bound"][1] < 0 and rows[-1]["certificate_margin"][1] > 0
    for name in ("force_error", "moment_error"):
        assert rows[0][name][0] == rows[-1][name][0]
    diagnostics[backend] = rows[-1]
audit = json.loads((root / "boundary-penalty-continuous-v1/results/summary.json").read_text())
assert audit["classification"] == "continuous_incompatible" and audit["conservative_polar_upper_bound"] < 0
diagnostics["independent_cpu_continuous_audit"] = audit
for name in (
    "parallel_contact_clear_candidate_v1.py", "probe_parallel_contact_clear_v1.py",
    "run_parallel_contact_clear_regression_v1.sh", "run_parallel_contact_clear_benchmark_v1.sh",
    "run_native_parallel_clear_regression_v1.sh", "traction_gradient_trial_candidate_v1.py",
    "probe_gradient_trial_v1.py", "run_gradient_trial_regression_v1.sh", "run_gradient_trial_benchmark_v1.sh",
    "boundary_wrench_penalty_candidate_v1.py", "probe_boundary_penalty_certificate_v1.py",
    "run_boundary_penalty_certificate_v1.sh", "probe_boundary_penalty_continuous_v1.py",
    "run_boundary_penalty_continuous_v1.sh", "triangle_disk_clip_candidate_v2.py",
    "archive_contact_trials_v1.py", "run_archive_contact_trials_v1.sh",
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
    "scope": __doc__, "source_base_revision": "5510f21e", "reports": reports,
    "native_parallel_clear_tests": {"cpu": 9, "cuda": 9}, "gradient_trial_cuda_tests": 55,
    "on_demand_diagnostics": diagnostics,
    "selection": "Parallel metadata clearing is selected; its native function AST exactly matches the measured candidate. Gradient-only trials are rejected by complete live cost. On-demand boundary witnesses are FP64 diagnostic prototypes, not selected production integration or physical contact repair.",
    "limitations": "B32, one condition bank and seed per matched experiment; no policy, large-batch acceptance, plateau or overall completion is claimed. Physical-incompatibility proof for the declared per-component law does not make normal-grasp failure acceptable. Triangle/disk clipping and aggregate wrench redistribution remain unselected.",
    "files": entries, "external_files": external,
}
path = destination / "manifest.json"
path.write_text(json.dumps(manifest, indent=2) + "\n")
for entry in entries:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
    assert hashlib.sha256((destination / entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"]
for entry in external:
    assert hashlib.sha256(Path(entry["original"]).read_bytes()).hexdigest() == entry["sha256"]
print(json.dumps({"artifacts": len(entries), "bytes": sum(entry["bytes"] for entry in entries), "external_artifacts": len(external), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "reports": reports, "on_demand_diagnostics": diagnostics}, indent=2), flush=True)
