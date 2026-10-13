"""Allocated-CPU dynamic latency/work audit and independent elasticity oracle for captured GPU fields."""
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.testing import assert_allclose
from publish_native_checkpoint_v1 import publish
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from scipy.sparse.linalg import splu

root = Path(__file__).parent
run = root / "native-final-sampler-wrench-oracle-v1/b2048-623102"
assert (run / "returncode.txt").read_text().strip() == "0"
assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
result = json.loads((run / "result.json").read_text())
assert result["envs"] == 2048 and result["seed"] == 623102
assert result["dt"] == .01 and result["substeps"] == 1 and len(result["passes"]) == 1
with np.load(run / "source/examples/rigid/assets/hollow_egg/level1/elastic.npz") as asset:
    oracle = P2Shell(asset["vertices"], asset["tetrahedra"], asset["surface_triangles"], 1e10, .3, 2000, 2, factor_backend="none")
factor = splu(oracle.k[oracle.free][:, oracle.free].tocsc())
checks = []
for sample in sorted(run.glob("fields-*.npz")):
    with np.load(sample) as saved:
        native_rhs = saved["rhs"].reshape(len(saved["envs"]), oracle.ndof).T
        valid = saved["contact_valid"]
        contact_force = np.where(valid[:, :, None], saved["contact_force"], 0)
        contact_position = np.where(valid[:, :, None], saved["contact_position"], 0)
        contact_free_moment = np.where(valid[:, :, None], saved["contact_free_moment"], 0)
        original_force = contact_force.sum(axis=1)
        original_moment = (np.cross(contact_position, contact_force) + contact_free_moment).sum(axis=1)
        nodal_force = saved["force"].sum(axis=1)
        nodal_moment = np.cross(oracle.xyz[None], saved["force"]).sum(axis=1)
        magnitudes = np.linalg.norm(contact_force, axis=2)
        force_scale = np.maximum(1e-30, magnitudes.sum(axis=1))
        reference = (contact_position * magnitudes[:, :, None]).sum(axis=1) / force_scale[:, None]
        # Propagate each unchanged original-point force/moment budget to the
        # common reference: dM_a = dM_p + (p-a) cross dF. A single contact
        # has a=p and exactly the original radius-scaled moment budget.
        lever = np.linalg.norm(contact_position - reference[:, None], axis=2)
        moment_scale = np.maximum(1e-30, (magnitudes * (saved["contact_radius"] + lever) + np.linalg.norm(contact_free_moment, axis=2)).sum(axis=1))
        force_errors = np.linalg.norm(nodal_force - original_force, axis=1) / force_scale
        moment_errors = np.linalg.norm(nodal_moment - original_moment - np.cross(reference, nodal_force - original_force), axis=1) / moment_scale
        assert max(force_errors.max(), moment_errors.max()) <= 1e-8
        omega = saved["omega"][:, None]
        centrifugal = -np.cross(omega, np.cross(omega, (oracle.xyz - oracle.com)[None]))
        raw = saved["force"].reshape(len(saved["envs"]), oracle.ndof).T
        rhs = oracle.balance(raw + oracle.m @ centrifugal.reshape(len(saved["envs"]), oracle.ndof).T)
        rhs_budget = np.maximum(1e-11, 1e-8 * np.linalg.norm(rhs, axis=0))
        rhs_difference_fraction = np.linalg.norm(native_rhs - rhs, axis=0) / rhs_budget
        assert rhs_difference_fraction.max() <= 1
        displacement = saved["displacement"].reshape(len(saved["envs"]), oracle.ndof).T
        expected = np.zeros_like(rhs)
        expected[oracle.free] = factor.solve(rhs[oracle.free])
        residual = np.linalg.norm(oracle.k @ displacement - rhs, axis=0)
        budget = np.maximum(1e-11, 1e-8 * np.linalg.norm(rhs, axis=0))
        assert np.max(residual / budget) <= 1
        peak_difference = []
        tensor_difference, vm_difference = [], []
        for i_b in range(len(saved["envs"])):
            expected_tensor, expected_vm = stress_field(oracle.glambda, oracle.elements, expected[:, i_b], 1e10, .3)
            actual_tensor, actual_vm = stress_field(oracle.glambda, oracle.elements, displacement[:, i_b], 1e10, .3)
            assert_allclose(actual_tensor, expected_tensor, rtol=1e-4, atol=1e-3)
            assert_allclose(actual_vm, expected_vm, rtol=1e-4, atol=1e-3)
            assert_allclose(saved["peak"][i_b], expected_vm.max(), rtol=1e-4, atol=1e-3)
            peak_difference.append(float(abs(saved["peak"][i_b] - expected_vm.max())))
            tensor_difference.append(float(np.max(abs(actual_tensor - expected_tensor))))
            vm_difference.append(float(np.max(abs(actual_vm - expected_vm))))
        checks.append({"sample": sample.name, "envs": saved["envs"].tolist(),
                       "original_aggregate_force_error": force_errors.tolist(), "original_aggregate_moment_error": moment_errors.tolist(),
                       "independent_inertia_relief_rhs_difference_budget_fraction": rhs_difference_fraction.tolist(),
                       "complete_residual_budget_fraction": (residual / budget).tolist(),
                       "global_peak_difference_Pa": peak_difference, "tensor_difference_Pa": tensor_difference,
                       "von_mises_difference_Pa": vm_difference})
assert len(checks) >= 10
passes, raw = [], []
for row in result["passes"]:
    assert row["invalid_environment_steps"] == 0 and row["resets"] == 2323
    with np.load(run / row["raw_npz"]) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    raw.append(data)
    assert data["work"].shape == (1280, 2048, 9)
    assert np.max(data["work"][:, :, 8]) == 0
    metrics = row["metrics"]
    assert metrics["totals"]["observed_steps"] == 2048 * 1280
    assert metrics["totals"]["completed_lift_success_cycles"] == 3142
    assert metrics["totals"]["completed_lift_failure_cycles"] == metrics["totals"]["lift_grasp_fail_steps"] == 0
    assert np.max(data["residual_N"] / np.maximum(1e-11, 1e-8 * data["rhs_N"])) <= 1
    summary = {key: value for key, value in row.items() if key != "metrics"}
    summary["metrics"] = {key: value for key, value in metrics.items() if key != "per_environment"}
    retry = np.max(data["work"][:, :, 5], axis=1) > 0
    summary["mean_step_ms_no_retry"] = float(np.mean(data["step_seconds"][~retry]) * 1000)
    summary["mean_step_ms_retry"] = float(np.mean(data["step_seconds"][retry]) * 1000) if np.any(retry) else None
    summary["environment_max_evaluations_percentiles"] = np.percentile(np.max(data["work"][:, :, 2], axis=0), [0, 50, 90, 99, 100]).tolist()
    passes.append(summary)
summary = {"passes": passes, "oracle": checks, "captured_independent_full_fields": sum(len(row["envs"]) for row in checks),
           "global_peak_difference_Pa_max": max(max(row["global_peak_difference_Pa"]) for row in checks),
           "full_residual_budget_fraction_max": max(max(row["complete_residual_budget_fraction"]) for row in checks),
           "gpu_uuid": result["gpu_uuid"], "source_revision": result["source_revision"],
           "fit_attempts": result["fit_attempts"] }
groups = {}
for buffer in result["native_buffers"]:
    if buffer["alias_of"] is None:
        group = ".".join(buffer["name"].split(".")[:4])
        groups[group] = groups.get(group, 0) + buffer["bytes"]
summary["native_buffer_bytes"] = result["native_buffer_bytes"]
summary["original_aggregate_wrench_error_max"] = max(max(row["original_aggregate_force_error"] + row["original_aggregate_moment_error"]) for row in checks)
summary["independent_relief_rhs_difference_budget_fraction_max"] = max(max(row["independent_inertia_relief_rhs_difference_budget_fraction"]) for row in checks)
summary["native_buffer_group_bytes"] = groups
summary["largest_native_buffers"] = sorted(result["native_buffers"], key=lambda row: row["bytes"], reverse=True)[:24]
runs = [run]
reports = {}
for backend in ("cpu",):
    regression = root / f"native-final-combined-regression-v2/{backend}"
    assert (regression / "returncode.txt").read_text().strip() == "0"
    assert (regression / "source.sha256").read_bytes() == (regression / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(regression / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "35" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[backend] = dict(suite.attrib)
    runs.append(regression)
summary["reports"] = reports
summary["cpu_route_note"] = "CPU35 is inherited from the unchanged CPU numerical/dispatch path. Surface frontier allocation and warp counters are CUDA-only. Clean-source CUDA qualification is separately published."
extras = [root / name for name in ("archive_native_final_sampler_wrench_oracle_v1.py",
    "publish_native_checkpoint_v1.py", "run_archive_native_final_sampler_wrench_oracle_v1.sh",
    "native-final-sampler-wrench-oracle-v1-allocation.log", "native-final-combined-regression-v2-cpu-allocation.log")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-final-sampler-wrench-oracle-checkpoint-v1"), runs, extras, summary,
        "Clean coalesced-position/valid-source/warm-Hessian/sparse-diagnostic/warp-counter combination with exact snapshots/patches. Low-shell FP64/Q10/s8/max B2048/623102,600warmup and complete1280-step varied trajectory. Unchanged final budgets. Independent CPU checks the original contact/free-moment wrench after signed P2 scatter, propagating each unchanged original-point moment/force budget to a common reference. It independently assembles mass, centrifugal loading, inertia relief, complete RHS, stiffness/gauge/factor and tensor/von Mises/global peak. Individual dynamic cone/sample fits are separately checked by independent fixture oracles. Includes inherited35-case qualification of the unchanged CPU numerical route; clean35CUDA is published separately. Fences/exports perturb latencies and nested fan calls overlap. All environments, metrics, tails and fields are retained. Final-source wider scopes/copied cohorts and plateau are separate requirements. No physical convergence or full RL training claim.")
