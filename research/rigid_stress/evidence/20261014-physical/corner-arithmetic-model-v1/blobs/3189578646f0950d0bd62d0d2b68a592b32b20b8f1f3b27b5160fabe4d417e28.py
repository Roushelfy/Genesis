"""Check actual native corner kernels against complete independently verified decimal polynomial fields."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_peak_impl
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .summarize_fine_references import closed_source
from .verify_corner_polynomial_decimal import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decimal-audit", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.decimal_audit)
    assert int((args.decimal_audit / "returncode.txt").read_text()) == 0
    summary_path = args.decimal_audit / "results/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["exact_rational_ten_monomial_corner_qualification_passed"] and summary["decimal_precision"] >= 64
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    inputs = {str(summary_path): hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    diagnostics = list(dict.fromkeys(row["diagnostic_run"] for row in summary["rows"]))
    rows = []
    for i_case, diagnostic in enumerate(diagnostics):
        run = Path(diagnostic)
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        diagnostic_summary = run / "results/summary.json"
        inputs[str(diagnostic_summary)] = hashlib.sha256(diagnostic_summary.read_bytes()).hexdigest()
        source = Path(json.loads(diagnostic_summary.read_text())["replay_run"])
        closed_source(source)
        assert int((source / "returncode.txt").read_text()) == 1
        mesh = source / "results/body_mesh.npz"
        inputs[str(mesh)] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        options = RigidStressOptions(mesh=mesh, method="pcg", output_mode="full")
        model = StressModel(options)
        state = model.create_state(1, "full")
        geometry = run / "results/geometry.npz"
        inputs[str(geometry)] = hashlib.sha256(geometry.read_bytes()).hexdigest()
        with np.load(geometry) as saved:
            vertices, elements = saved["native_vertices_m"].copy(), saved["native_elements"].copy()
            np.testing.assert_array_equal(qd_to_numpy(model.info.vertices, copy=True), vertices)
            np.testing.assert_array_equal(qd_to_numpy(model.info.elements, copy=True), elements)
        for original in summary["rows"]:
            if original["diagnostic_run"] != diagnostic:
                continue
            mode = original["mode"]
            target_path = args.decimal_audit / f"results/case{i_case:02d}-{mode}.npz"
            field_path = run / f"results/{mode}.npz"
            for path in [target_path, field_path]:
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(target_path) as saved:
                target = saved["decimal_polynomial_components_Pa"].copy()
                old_reference = saved["original_reference_components_Pa"].copy()
            with np.load(field_path) as saved:
                displacement = saved["frozen_native_displacement_m"].copy()
                old_native = saved["stored_native_components_Pa"].copy()
            state.displacement.from_numpy(displacement[:, None])
            state.valid.fill(True)
            for is_cached in [False, True]:
                kernel_peak_impl(options.young, options.poisson, state, model.info, is_cached, True)
                tensor = qd_to_numpy(state.stress_tensor, transpose=True, copy=True)[0]
                peak = float(qd_to_numpy(state.peak, copy=True)[0])
                metrics = compare(tensor, target)
                target_vm = np.sqrt(
                    0.5
                    * (
                        (target[..., 0] - target[..., 1]) ** 2
                        + (target[..., 1] - target[..., 2]) ** 2
                        + (target[..., 2] - target[..., 0]) ** 2
                    )
                    + 3 * np.sum(target[..., 3:] ** 2, axis=-1)
                )
                target_peak = float(target_vm.max())
                row = {
                    "diagnostic_run": diagnostic,
                    "mode": mode,
                    "tetrahedra": len(elements),
                    "is_cached": is_cached,
                    "native_vs_decimal_polynomial": metrics,
                    "old_native_vs_decimal_polynomial": compare(old_native, target),
                    "old_reference_vs_decimal_polynomial": compare(old_reference, target),
                    "raw_global_native_peak_Pa": peak,
                    "raw_global_decimal_peak_Pa": target_peak,
                    "raw_global_peak_existing_budget_passed": bool(np.isclose(peak, target_peak, rtol=1e-4, atol=1e-3)),
                    "all_existing_component_budgets_passed": metrics["maximum_existing_component_budget_fraction"] <= 1,
                }
                rows.append(row)
                np.savez_compressed(
                    args.output / f"case{i_case:02d}-{mode}-cached{int(is_cached)}.npz",
                    native_components_Pa=tensor,
                    decimal_polynomial_components_Pa=target,
                )
                print(json.dumps(row), flush=True)
            if i_case == len(diagnostics) - 1 and mode == "full_inertia":
                fraction = abs(old_native - target) / (1e-5 + 1e-9 * abs(target))
                i_element = np.unravel_index(np.argmax(fraction), fraction.shape)[0]
                local_nodes = elements[i_element]
                np.savez_compressed(
                    args.output / "regression_fixture.npz",
                    vertices=vertices[local_nodes[:4]],
                    tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int32),
                    surface_triangles=np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]], dtype=np.int32),
                    displacement_m=displacement[local_nodes],
                    expected_corner_components_Pa=target[i_element],
                    original_native_components_Pa=old_native[i_element],
                )
                (args.output / "regression_fixture_provenance.json").write_text(
                    json.dumps(
                        {
                            "diagnostic_run": diagnostic,
                            "original_element": int(i_element),
                            "mode": mode,
                            "old_maximum_budget_fraction": float(fraction[i_element].max()),
                            "decimal_reference": str(target_path),
                            "decimal_reference_sha256": inputs[str(target_path)],
                        },
                        indent=2,
                    )
                    + "\n"
                )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "backend": args.backend,
        "decimal_audit": str(args.decimal_audit),
        "rows": rows,
        "input_sha256": inputs,
        "all_existing_component_budgets_passed": all(row["all_existing_component_budgets_passed"] for row in rows),
        "new_equilibrium_solve_or_physical_state": False,
        "goal_completion": False,
        "limitations": "Every actual cached/uncached native corner component is checked against the independently verified decimal polynomial field at the unchanged1e-5Pa+1e-9*abs(stress) budget. Frozen original solver states and actual old failed exits remain; old float64 oracle discrepancies are separately retained. This postprocessing qualification accepts neither new physical states nor spatial/contact/model/feedback convergence or warmed cost.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["all_existing_component_budgets_passed"], rows
    assert all(row["raw_global_peak_existing_budget_passed"] for row in rows), rows


if __name__ == "__main__":
    main()
