"""Check experimental shell bending prolongation on planar triangle macroelements inside Slurm.

For a transverse displacement w and normal n, Kirchhoff rotations satisfy theta=(w_y,-w_x,0) in the local plane.
The normal component of (theta_a-theta_b) x (x_b-x_a) is minus the edge contraction of the Hessian of w. Its
coefficients 1/8 at edge midpoints and 1/18 at triangle centroids exactly remove linear interpolation error for
quadratic w. These planar controls establish kinematics and numerical consistency. Curved-shell accuracy requires
separate comparison with the unchanged mother discretization.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy import sparse

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import create_stress_state
from genesis.engine.solvers.rigid.stress.feature_basis import FeatureBasis
from genesis.engine.solvers.rigid.stress.feature_operator import FeatureOperator
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.solve import kernel_direct_init, kernel_full_residual
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json
from .check_local_feature_spaces import put_columns, relative, rigid_modes, vector_columns


def check_patch(output: Path, name: str, first_axis: np.ndarray, second_axis: np.ndarray):
    normal = np.cross(first_axis, second_axis)
    local_vertices = np.array([[-0.04, -0.02], [0.05, -0.015], [-0.01, 0.045]])
    original = local_vertices[:, :1] * first_axis + local_vertices[:, 1:] * second_axis
    edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=gs.np_int)
    vertices = np.concatenate((original, original[edges].mean(axis=1), original.mean(axis=0, keepdims=True)))
    quads = np.array([[0, 3, 6, 5], [1, 4, 6, 3], [2, 5, 6, 4]], dtype=gs.np_int)
    parents = np.full((len(vertices), 3), -1, dtype=gs.np_int)
    parents[:3, 0] = np.arange(3)
    parents[3:6, :2] = edges
    parents[6] = np.arange(3)
    asset = output / f"{name}.npz"
    np.savez(asset, vertices=vertices, quads=quads, directors=np.broadcast_to(normal, vertices.shape).copy())
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset,
        backend="shell",
        method="direct",
        surface_inverse=False,
        factor_scaling=True,
        output_mode="full",
        young=1e7,
        shell_thickness=0.003,
        density=1000.0,
        tolerance=1e-8,
        absolute_tolerance=1e-11,
    )
    mother = ShellModel(options)
    fixed_nodes = np.array([3], dtype=gs.np_int)
    linear = FeatureBasis(mother, parents, fixed_nodes)
    bending = FeatureBasis(mother, parents, fixed_nodes, shell_bending=True)
    np.testing.assert_array_equal(linear.coarse_nodes, bending.coarse_nodes)
    np.testing.assert_array_equal(linear.parent_nodes, bending.parent_nodes)
    np.testing.assert_array_equal(linear.transform.reduced_free, bending.transform.reduced_free)
    assert fixed_nodes[0] in bending.coarse_nodes
    mother_vertices = qd_to_numpy(mother.info.vertices, copy=True)
    mother_directors = qd_to_numpy(mother.shell.directors, copy=True)
    mother_free = qd_to_numpy(mother.info.is_free, copy=True)
    mother_pins = qd_to_numpy(mother.info.pins, copy=True)

    x, y = vertices @ first_axis, vertices @ second_axis
    transverse = np.column_stack((0.5 * x**2, x * y, 0.5 * y**2))
    gradient_x = np.column_stack((x, y, np.zeros(len(x))))
    gradient_y = np.column_stack((np.zeros(len(y)), x, y))
    quadratic = np.stack(
        (
            normal[None, :, None] * transverse[:, None, :],
            first_axis[None, :, None] * gradient_y[:, None, :]
            - second_axis[None, :, None] * gradient_x[:, None, :],
        ),
        axis=1,
    ).reshape((6 * len(vertices), 3))
    modes = rigid_modes(vertices, blocks_per_node=2)
    analytic = np.column_stack((modes, quadratic))
    coarse_analytic = analytic.reshape((len(vertices), 6, 9))[bending.coarse_nodes].reshape(
        (6 * len(bending.coarse_nodes), 9)
    )
    raw_shape = (6 * len(vertices), 3 * bending.transform.n_reduced_blocks)
    linear_map = sparse.bsr_matrix(
        (qd_to_numpy(linear.transform.values), linear.transform.columns, linear.transform.row_start), shape=raw_shape
    ).tocsr()
    bending_map = sparse.bsr_matrix(
        (qd_to_numpy(bending.transform.values), bending.transform.columns, bending.transform.row_start), shape=raw_shape
    ).tocsr()
    translations = (6 * np.arange(len(vertices))[:, None] + np.arange(3)).reshape((3 * len(vertices),))
    rotations = translations + 3
    actual = bending_map @ coarse_analytic
    linear_actual = linear_map @ coarse_analytic
    drilling = np.zeros((len(bending.coarse_nodes), 6))
    drilling[:, 3:] = normal * (x[bending.coarse_nodes] + 0.3 * y[bending.coarse_nodes])[:, None]
    drilling_displacement = (bending_map - linear_map) @ drilling.reshape((6 * len(bending.coarse_nodes),))
    kinematics = {
        "rigid_modes_relative": relative(actual[:, :6], modes),
        "quadratic_transverse_relative": relative(actual[translations, 6:], quadratic[translations]),
        "linear_rotation_relative": relative(actual[rotations, 6:], quadratic[rotations]),
        "linear_space_quadratic_transverse_relative": relative(
            linear_actual[translations, 6:], quadratic[translations]
        ),
        "constant_rotation_correction_absolute": float(
            np.max(np.abs((bending_map - linear_map) @ coarse_analytic[:, :6]))
        ),
        "drilling_translation_correction_absolute": float(np.max(np.abs(drilling_displacement[translations]))),
    }
    assert kinematics["linear_space_quadratic_transverse_relative"] > 1e-3, kinematics
    assert max(value for key, value in kinematics.items() if not key.startswith("linear_space_")) < 1e-12, kinematics
    cardinal_rows = (6 * bending.coarse_nodes[:, None] + np.arange(6)).reshape((6 * len(bending.coarse_nodes),))
    np.testing.assert_array_equal(bending_map[cardinal_rows].toarray(), np.eye(len(cardinal_rows)))

    n_envs = 4
    generator = np.random.default_rng(20261026)
    force = generator.normal(size=(6 * len(vertices), n_envs)) * 1e-3
    force.reshape((len(vertices), 6, n_envs))[:, 3:] *= options.shell_thickness
    force[:, 2] *= 1e-12
    force[:, 3] = 0.0
    reference = mother.create_state(n_envs, "full")
    put_columns(reference.force, force)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0)
    mother.recover(omega, reference)
    assert qd_to_numpy(reference.valid).all(), "Mother solve must pass the original equilibrium contract."
    mother_solution = vector_columns(reference.displacement)
    local_nodes = np.flatnonzero(bending.coarse_index < 0).astype(gs.np_int)
    assert 6 in local_nodes and np.isin(local_nodes, [3, 4, 5]).any()
    spaces = []
    for label, selected in (("coarse", np.empty(0, dtype=gs.np_int)), ("all_bubbles", local_nodes)):
        feature = bending.feature(selected)
        operator = FeatureOperator(mother.info, *feature)
        state = create_stress_state(feature.n_reduced_blocks, n_envs)
        operator.project(reference.rhs, state.rhs)
        kernel_direct_init(state)
        operator.factor.solve(options.young, state, operator.info)
        kernel_full_residual(
            options.young, options.tolerance, options.absolute_tolerance, state, operator.info, only_active=False
        )
        assert qd_to_numpy(state.valid).all(), (name, label)
        residual_norm = np.sqrt(qd_to_numpy(state.residual_norm_squared))
        budget = np.maximum(
            options.absolute_tolerance, options.tolerance * np.linalg.norm(vector_columns(state.rhs), axis=0)
        )
        assert (residual_norm <= budget).all(), (label, residual_norm, budget)

        vectors = generator.normal(size=(3 * feature.n_reduced_blocks, n_envs))
        vectors *= feature.reduced_free.reshape((3 * feature.n_reduced_blocks, 1))
        put_columns(state.displacement, vectors)
        lifted = mother.create_state(n_envs, "full")
        operator.lift(state.displacement, lifted.displacement)
        operator.project(reference.force, state.force)
        virtual_difference = np.abs(
            np.sum(vector_columns(lifted.displacement) * force, axis=0)
            - np.sum(vectors * vector_columns(state.force), axis=0)
        )
        virtual_scale = np.maximum(
            np.linalg.norm(vector_columns(lifted.displacement), axis=0) * np.linalg.norm(force, axis=0), 1e-30
        )
        virtual_error = float(np.max(virtual_difference / virtual_scale))
        assert virtual_error < 1e-12, virtual_error

        kernel_direct_init(state)
        operator.factor.solve(options.young, state, operator.info)
        operator.lift(state.displacement, lifted.displacement)
        mother.evaluate_stress(options, lifted)
        solution = vector_columns(lifted.displacement)
        checks = {
            "virtual_work": virtual_error,
            "mother_displacement_relative": relative(solution, mother_solution),
            "mother_tensor_relative": relative(
                qd_to_numpy(lifted.stress_tensor), qd_to_numpy(reference.stress_tensor)
            ),
            "mother_transverse_resultant_relative": relative(
                qd_to_numpy(lifted.transverse_resultant), qd_to_numpy(reference.transverse_resultant)
            ),
        }
        raw = sparse.bsr_matrix(
            (qd_to_numpy(feature.values), feature.columns, feature.row_start),
            shape=(6 * len(vertices), 3 * feature.n_reduced_blocks),
        ).tocsr()
        free = feature.reduced_free.reshape((3 * feature.n_reduced_blocks,)).astype(bool)
        rank = np.linalg.matrix_rank(raw[:, free].toarray())
        assert rank == np.count_nonzero(free)
        np.testing.assert_array_equal(solution[:, 3], 0.0)
        if label == "all_bubbles":
            assert raw.shape[0] == raw.shape[1]
            assert max(checks.values()) < 1e-8, checks
        spaces.append(
            {
                "space": label,
                "blocks": feature.n_reduced_blocks,
                "free_rank": int(rank),
                "local_nodes": selected.tolist(),
                "unscaled_reduced_residual_norm": residual_norm.tolist(),
                "original_residual_budget": budget.tolist(),
                "checks": checks,
                "coarse_mother_errors_are_spatial_diagnostics": label == "coarse",
            }
        )
    np.testing.assert_array_equal(qd_to_numpy(mother.info.vertices), mother_vertices)
    np.testing.assert_array_equal(qd_to_numpy(mother.shell.directors), mother_directors)
    np.testing.assert_array_equal(qd_to_numpy(mother.info.is_free), mother_free)
    np.testing.assert_array_equal(qd_to_numpy(mother.info.pins), mother_pins)
    return {
        "patch": name,
        "asset": file_id(asset),
        "kinematics": kinematics,
        "mother_blocks": mother.info.vertices.shape[0],
        "coarse_nodes": bending.coarse_nodes.tolist(),
        "fixed_nodes": fixed_nodes.tolist(),
        "pins": mother_pins.tolist(),
        "normal_projection": True,
        "spaces": spaces,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Scientific execution requires a scheduled Slurm job."
    assert os.environ.get("SLURM_JOB_PARTITION") == "rtx-mid", "Use the established rtx-mid partition."
    assert os.environ.get("GENESIS_IMAGE_VER") == "1_26", "Use the captured image 1_26 environment."
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "complete": False,
        "passed": False,
        "identity": runtime_identity(),
        "source": file_id(Path(__file__)),
        "curved_shell_accuracy_qualified": False,
        "tolerance": 1e-8,
        "absolute_tolerance": 1e-11,
        "patches": [],
    }
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    result["patches"].append(
        check_patch(args.output, "horizontal", np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    )
    first = np.array([1.0, 2.0, -1.0]) / np.sqrt(6.0)
    second = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    result["patches"].append(check_patch(args.output, "tilted", first, second))
    result.update(complete=True, passed=True)
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
