"""Allocated-CPU dynamic latency/work audit and independent elasticity oracle for captured GPU fields."""
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse.linalg import splu

from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
run = root / "native-hard-fit-attempts-v2/b2048-623102"
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
        rhs = saved["rhs"].reshape(len(saved["envs"]), oracle.ndof).T
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
runs = [run]
reports = {}
for backend in ("cuda",):
    regression = root / f"native-sample-lane-retry-regression-v2/{backend}"
    assert (regression / "returncode.txt").read_text().strip() == "0"
    assert (regression / "source.sha256").read_bytes() == (regression / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(regression / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "23" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[backend] = dict(suite.attrib)
    runs.append(regression)
summary["reports"] = reports
extras = [root / name for name in ("archive_native_sample_lane_retry_selected_v2.py", "publish_native_checkpoint_v1.py", "run_archive_native_sample_lane_retry_selected_v2.sh", "native-hard-fit-attempts-v2-b2048-allocation.log", "native-sample-lane-retry-tests-v2-cuda-allocation.log")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-sample-lane-retry-selected-checkpoint-v2"), runs, extras, summary,
        "Production-form8-sample-lane retry removes the experimental flag; every physical sample contributes once and all final budgets stay unchanged. CUDA23 passes including24 independent contact/nodal/full-field samples and successful-state bitwise retry isolation. CPU engine route is unchanged. Low-shell FP64/Q10/s8, one complete1280-step changing trajectory after600 warmup. Fences and four hard-frame pre-attempt exports perturb scheduling; these are diagnostic serialized call latencies, not headline rates. Work/field exports happen outside kernel call latency. All environments/hard frames remain included. Independent CPU assembles identical mesh stiffness, uses its own gauge/factor and checks full residual/tensor/von Mises/global maximum budgets on160 captured GPU fields. This oracle receives captured balanced RHS and does not independently reconstruct dynamic fitting or relief; separate fixture oracles remain in the matrix. Private base90b2964 source patch reproduces currenta6dc0fbc math plus selected retry; numerical source closure is authoritative. Primary merge, combined compact storage qualification, broader scopes/seeds, remaining candidates and plateau are pending. No individual environment GPU wall-time attribution, physical convergence or RL training claim.")
