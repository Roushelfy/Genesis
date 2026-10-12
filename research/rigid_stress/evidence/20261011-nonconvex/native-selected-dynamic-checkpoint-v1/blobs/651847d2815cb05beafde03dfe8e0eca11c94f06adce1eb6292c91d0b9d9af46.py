"""Allocated-CPU dynamic latency/work audit and independent elasticity oracle for captured GPU fields."""
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from numpy.testing import assert_allclose
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell
from scipy.sparse.linalg import splu

from publish_native_checkpoint_v1 import publish

root = Path(__file__).parent
run = root / "native-selected-dynamic-v1/b16384-623102"
assert (run / "returncode.txt").read_text().strip() == "0"
assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
result = json.loads((run / "result.json").read_text())
assert result["envs"] == 16384 and result["seed"] == 623102
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
    assert row["invalid_environment_steps"] == 0 and row["resets"] == 18571
    with np.load(run / row["raw_npz"]) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    raw.append(data)
    assert data["work"].shape == (1280, 16384, 9)
    assert np.max(data["work"][:, :, 8]) == 0
    metrics = row["metrics"]
    assert metrics["totals"]["observed_steps"] == 16384 * 1280
    assert metrics["totals"]["completed_lift_success_cycles"] == 25124
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
summary["native_buffer_group_bytes"] = groups
summary["largest_native_buffers"] = sorted(result["native_buffers"], key=lambda row: row["bytes"], reverse=True)[:24]
runs = [run]
reports = {}
for backend in ("cuda",):
    regression = root / f"native-compact-cache-fan-selected-regression-v1/{backend}"
    assert (regression / "returncode.txt").read_text().strip() == "0"
    assert (regression / "source.sha256").read_bytes() == (regression / "source-after.sha256").read_bytes()
    suite = ElementTree.parse(regression / "report.xml").getroot().find("testsuite")
    assert suite.attrib["tests"] == "31" and suite.attrib["errors"] == suite.attrib["failures"] == suite.attrib["skipped"] == "0"
    reports[backend] = dict(suite.attrib)
    runs.append(regression)
summary["reports"] = reports
extras = [root / name for name in ("archive_native_selected_dynamic_v1.py", "publish_native_checkpoint_v1.py", "run_archive_native_selected_dynamic_v1.sh", "native-selected-dynamic-v1-b16384-allocation.log", "native-compact-cache-fan-selected-tests-v1-cuda-allocation.log")]
publish(root, Path("research/rigid_stress/evidence/20261011-nonconvex/native-selected-dynamic-checkpoint-v1"), runs, extras, summary,
        "Production92ec3e0c counted positions/complete exact growth/parallel fan source, low-shell FP64/Q10/s8/max output, B16384/623102 with600 warmup and one complete1280-step varied trajectory. Separate production31 CPU/31 CUDA/96-field fixture qualification remains published; this archive includes its31-case CUDA report. Every original contact, topology/radius/profile/friction and final budget is unchanged. Fences and hard-frame pre-attempt exports perturb scheduling; these are serialized diagnostic latencies, not headline rates. Nested fan wrapper/chart/sample calls overlap and must not be added. Work/field exports occur outside measured kernel latency. Complete environment metrics, tails and buffer accounting are retained. Independent CPU assembles the same mesh stiffness, uses its own gauge/factor and checks every captured GPU field's full residual/tensor/von Mises/global peak. Captured balanced RHS does not independently reconstruct dynamic fitting/relief; separate fixture oracles cover those paths. Final matched wide/copy scopes, remaining applicable layout/capture decisions and plateau remain pending. No individual environment GPU wall-time attribution, physical convergence or RL training claim.")
