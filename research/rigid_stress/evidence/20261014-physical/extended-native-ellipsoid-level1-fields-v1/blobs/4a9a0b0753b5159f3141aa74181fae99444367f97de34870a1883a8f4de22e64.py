"""Compare frozen native corner fields with independent extended-precision geometry and stress evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skfem import Basis, ElementTetP2, MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.solve import kernel_peak_impl
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.misc import qd_to_numpy

from .summarize_fine_references import closed_source


def stress_components(derivative, young, poisson):
    """Evaluate all six symmetric stress components in the input arithmetic."""
    lame_mu = young / (2 * (1 + poisson))
    lame_lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    tensor = lame_mu * (derivative + derivative.swapaxes(-1, -2))
    for i_axis in range(3):
        tensor[..., i_axis, i_axis] += lame_lambda * np.trace(derivative, axis1=-2, axis2=-1)
    return tensor[..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]]


def difference(first, reference):
    """Retain complete component errors against the existing cached-field budget."""
    error = abs(first - reference)
    budget = 1e-5 + 1e-9 * abs(reference)
    position = np.unravel_index(np.argmax(error / budget), error.shape)
    return {
        "maximum_absolute_difference_Pa": float(error.max()),
        "maximum_existing_component_budget_fraction": float((error / budget).max()),
        "worst_element_corner_component": [int(value) for value in position],
        "first_at_worst_Pa": float(first[position]),
        "reference_at_worst_Pa": float(reference[position]),
        "whole_field_relative_L2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-run", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    closed_source(args.replay_run)
    producer_code = int((args.replay_run / "returncode.txt").read_text())
    assert producer_code == 1
    summary_path = args.replay_run / "results/summary.json"
    summary = json.loads(summary_path.read_text())
    assert not summary["all_existing_gates_passed"]
    assert all(row["complete_equilibrium_passed"] and row["independent_tensor_passed"] for row in summary["rows"])
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def retain(path):
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path not in inputs or inputs[path] == expected
        inputs[path] = expected

    retain(summary_path)
    mesh_path = args.replay_run / "results/body_mesh.npz"
    retain(mesh_path)
    with np.load(mesh_path) as saved:
        vertices, tetrahedra = saved["vertices"].copy(), saved["tetrahedra"].copy()
    mesh = MeshTet(vertices.T, tetrahedra.T, sort_t=False)
    scalar_basis = Basis(mesh, ElementTetP2(), quadrature=(np.eye(4)[1:], np.ones(4)))
    reference_gradients = np.stack([entry[0].grad for entry in scalar_basis.basis]).transpose(2, 3, 0, 1)
    gs.init(backend=gs.cuda if args.backend == "cuda" else gs.cpu, precision="64", logging_level="warning")
    options = RigidStressOptions(mesh=mesh_path, method="pcg", output_mode="full")
    assert options.young == 1e10 and options.poisson == 0.3 and options.density == 2000
    model = StressModel(options)
    state = model.create_state(1, "full")
    native_vertices = qd_to_numpy(model.info.vertices, copy=True)
    native_elements = qd_to_numpy(model.info.elements, copy=True)
    native_gradients = qd_to_numpy(model.info.gradients, copy=True)
    corner_nodes = qd_to_numpy(model.info.corner_nodes, copy=True)
    corner_gradients = qd_to_numpy(model.info.corner_gradients, copy=True)
    edges = qd_to_numpy(model.info.edges, copy=True)
    assert native_vertices.shape == (scalar_basis.N, 3)
    assert native_elements.shape == (len(tetrahedra), 10)
    assert corner_nodes.shape == (len(tetrahedra), 4, 7)
    assert corner_gradients.shape == (len(tetrahedra), 4, 7, 3)
    np.testing.assert_array_equal(native_vertices[: len(vertices)], vertices)
    np.testing.assert_array_equal(native_elements[:, :4], tetrahedra)
    distances, native_to_reference = cKDTree(scalar_basis.doflocs.T).query(native_vertices)
    assert distances.max() <= 1e-12 and len(np.unique(native_to_reference)) == scalar_basis.N

    # Cofactors use the exact saved float64 coordinates cast before subtraction.
    # This independent geometry calculation avoids a float64 matrix inverse.
    points = np.asarray(vertices[tetrahedra], dtype=np.longdouble)
    first, second, third = [points[:, i_corner] - points[:, 0] for i_corner in [1, 2, 3]]
    determinant = np.einsum("ei,ei->e", first, np.cross(second, third))
    wide_gradients = (
        np.stack([np.cross(second, third), np.cross(third, first), np.cross(first, second)], axis=1)
        / determinant[:, None, None]
    )
    wide_gradients = np.concatenate([-wide_gradients.sum(axis=1)[:, None], wide_gradients], axis=1)
    wide_shape_gradients = np.empty((len(tetrahedra), 4, 10, 3), dtype=np.longdouble)
    for i_corner in range(4):
        for i_local in range(10):
            if i_local < 4:
                wide_shape_gradients[:, i_corner, i_local] = (4 * (i_corner == i_local) - 1) * wide_gradients[
                    :, i_local
                ]
            else:
                i_first, i_second = edges[i_local - 4]
                wide_shape_gradients[:, i_corner, i_local] = 4 * (
                    (i_corner == i_first) * wide_gradients[:, i_second]
                    + (i_corner == i_second) * wide_gradients[:, i_first]
                )

    np.savez_compressed(
        args.output / "geometry.npz",
        native_vertices_m=native_vertices,
        native_elements=native_elements,
        native_affine_gradients=native_gradients,
        native_corner_nodes=corner_nodes,
        native_corner_gradients=corner_gradients,
        independent_float64_corner_gradients=reference_gradients,
        independent_extended_affine_gradients=wide_gradients,
        independent_extended_corner_gradients=wide_shape_gradients,
        reference_element_nodes=scalar_basis.element_dofs.T,
        native_to_reference_nodes=native_to_reference,
    )
    rows = []
    for original in summary["rows"]:
        mode = original["mode"]
        path = args.replay_run / f"results/{mode}.npz"
        retain(path)
        with np.load(path) as saved:
            vector = saved["native_displacement_m"].copy()
            stored_native = saved["native_cached_corner_components_Pa"].copy()
            stored_reference = saved["native_corner_tensor_Pa"][..., [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]].copy()
        reference_displacement = vector.reshape(-1, 3)
        native_displacement = reference_displacement[native_to_reference]
        state.displacement.from_numpy(native_displacement[:, None])
        state.valid.fill(True)
        native_fields = {}
        for is_cached in [True, False]:
            kernel_peak_impl(options.young, options.poisson, state, model.info, is_cached, True)
            native_fields["cached" if is_cached else "uncached"] = qd_to_numpy(
                state.stress_tensor, transpose=True, copy=True
            )[0]
        native_wide_derivative = np.einsum(
            "ecli,eclj->ecij",
            np.asarray(native_displacement[corner_nodes], dtype=np.longdouble),
            np.asarray(corner_gradients, dtype=np.longdouble),
            optimize=False,
        )
        reference_wide_derivative = np.einsum(
            "eli,eclj->ecij",
            np.asarray(reference_displacement[scalar_basis.element_dofs.T], dtype=np.longdouble),
            np.asarray(reference_gradients, dtype=np.longdouble),
            optimize=False,
        )
        exact_derivative = np.einsum(
            "eli,eclj->ecij",
            np.asarray(native_displacement[native_elements], dtype=np.longdouble),
            wide_shape_gradients,
            optimize=False,
        )
        native_wide = stress_components(
            native_wide_derivative, np.longdouble(options.young), np.longdouble(options.poisson)
        )
        reference_wide = stress_components(
            reference_wide_derivative, np.longdouble(options.young), np.longdouble(options.poisson)
        )
        exact_components = stress_components(
            exact_derivative, np.longdouble(options.young), np.longdouble(options.poisson)
        )
        row = {
            "mode": mode,
            "original_complete_equilibrium_passed": original["complete_equilibrium_passed"],
            "original_cached_component_budget_fraction": original["cached_corner_stress_max_existing_budget_fraction"],
            "rebuilt_cached_vs_stored_cached": difference(native_fields["cached"], stored_native),
            "rebuilt_uncached_vs_stored_reference": difference(native_fields["uncached"], stored_reference),
            "native_extended_accumulation_vs_reference_extended": difference(native_wide, reference_wide),
            "native_gradient_extended_vs_exact_geometry_extended": difference(native_wide, exact_components),
            "reference_gradient_extended_vs_exact_geometry_extended": difference(reference_wide, exact_components),
            "cached_vs_native_extended_accumulation": difference(native_fields["cached"], native_wide),
            "stored_reference_vs_reference_extended_accumulation": difference(stored_reference, reference_wide),
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"{mode}.npz",
            stored_native_components_Pa=stored_native,
            stored_reference_components_Pa=stored_reference,
            rebuilt_native_cached_components_Pa=native_fields["cached"],
            rebuilt_native_uncached_components_Pa=native_fields["uncached"],
            native_gradient_extended_accumulation_components_Pa=native_wide,
            reference_gradient_extended_accumulation_components_Pa=reference_wide,
            independent_extended_geometry_components_Pa=exact_components,
            frozen_native_displacement_m=native_displacement,
        )
        print(json.dumps(row), flush=True)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    result = {
        "scope": __doc__,
        "replay_run": str(args.replay_run),
        "actual_replay_wrapper_code": producer_code,
        "tetrahedra": len(tetrahedra),
        "extended_dtype": str(np.dtype(np.longdouble)),
        "extended_mantissa_bits": int(np.finfo(np.longdouble).nmant),
        "rows": rows,
        "input_sha256": {str(path): expected for path, expected in inputs.items()},
        "new_equilibrium_solve_or_physical_state": False,
        "goal_completion": False,
        "limitations": "Frozen displacements and both actual failed replay codes remain unchanged. Complete cached/uncached native fields, independent extended accumulations and cofactor geometry retain every corner/component. Existing component budgets provide diagnostics, without replacing their original reference or accepting a failed gate. This isolates arithmetic/geometry contributions before any production change; spatial/contact/model/feedback and warmed accuracy/cost remain open.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
