"""Independent CPU shell checks of native stiffness, modes, relief and complete fields."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

import genesis as gs
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu import core, geometry

from .artifacts import file_id, runtime_identity, write_json


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def control(output, name, vertices, quads, method, gauge, spinning=False):
    asset = output / f"{name}.npz"
    np.savez(asset, vertices=vertices, quads=quads)
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset, backend="shell", method=method, shell_gauge=gauge, output_mode="full", density=1000.0
    )
    native = ShellModel(options)
    oracle = core.ShellModel(vertices, quads, density=options.density, gauge=gauge)
    info = native.info
    starts, columns = qd_to_numpy(info.row_start), qd_to_numpy(info.columns)
    matrix = sparse.bsr_matrix(
        (qd_to_numpy(info.stiffness) * options.young, columns, starts), shape=oracle.K.shape
    ).tocsr()
    errors = {
        "stiffness": relative(matrix.toarray(), oracle.K.toarray()),
        "directors": relative(qd_to_numpy(native.shell.directors), oracle.directors),
        "membrane": relative(qd_to_numpy(native.shell.membrane).transpose(0, 1, 3, 2), oracle.B_membrane),
        "bending": relative(qd_to_numpy(native.shell.bending).transpose(0, 1, 3, 2), oracle.B_bending),
        "shear": relative(qd_to_numpy(native.shell.shear).transpose(0, 1, 3, 2), oracle.B_shear),
        "mass_modes": relative(qd_to_numpy(info.mass_modes).reshape((-1, 6)), oracle.mass_modes),
        "gram": relative(qd_to_numpy(info.gram), oracle.mass_gram),
        "rigid_null_residual": float(
            np.linalg.norm(matrix @ oracle.rigid_modes)
            / (np.linalg.norm(matrix.toarray()) * np.linalg.norm(oracle.rigid_modes))
        ),
    }
    rng = np.random.default_rng(1024)
    force = rng.normal(size=(oracle.ndof, 4)) * 1e-3
    force.reshape((-1, 6, 4))[:, 3:] *= options.shell_thickness
    force[:, 3] = 0.0
    state = native.create_state(4, "full")
    state.force.from_numpy(force.reshape((-1, 3, 4)).transpose(0, 2, 1).copy())
    omega = V_VEC(3, dtype=gs.qd_float, shape=(4,))
    angular = rng.normal(size=(4, 3)) * 3.0 if spinning else np.zeros((4, 3))
    angular[3] = 0.0
    omega.from_numpy(angular)
    lever = vertices - oracle.center_of_mass
    acceleration = np.cross(angular[None, :, :], np.cross(angular[None, :, :], lever[:, None, :]))
    rotational_inertia = (
        oracle.nodal_mass[:, None, None]
        * options.shell_thickness**2
        / 12.0
        * (np.eye(3)[None, :, :] - oracle.directors[:, :, None] * oracle.directors[:, None, :])
    )
    spin_momentum = np.einsum("nij,bj->nbi", rotational_inertia, angular)
    centrifugal = (
        np.concatenate(
            (-oracle.nodal_mass[:, None, None] * acceleration, -np.cross(angular[None, :, :], spin_momentum)), axis=2
        )
        .transpose(0, 2, 1)
        .reshape((-1, 4))
    )
    angular_terms = np.column_stack(
        (angular**2, angular[:, 0] * angular[:, 1], angular[:, 0] * angular[:, 2], angular[:, 1] * angular[:, 2])
    ).T
    errors["centrifugal"] = relative(qd_to_numpy(info.centrifugal).reshape((-1, 6)) @ angular_terms, centrifugal)
    reference_force = force + centrifugal
    native.recover(omega, state)
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape((4, -1)).T
    reference_displacement = oracle.solve(reference_force)
    expected_load, _coeff = oracle.balance_load(reference_force)
    actual_load = qd_to_numpy(state.rhs, transpose=True).reshape((4, -1)).T
    errors["balanced_load"] = relative(actual_load, expected_load)
    errors["complete_residual"] = float(oracle.residual(displacement, reference_force))
    reference_stress = oracle.stress(reference_displacement)
    tensors = np.moveaxis(np.stack((reference_stress["sigma_top"], reference_stress["sigma_bottom"]), axis=1), -1, 0)
    expected_field = np.stack(
        [tensors[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1
    )
    expected_field = expected_field.reshape((4, 2 * len(quads), 4, 6))
    actual_field = qd_to_numpy(state.stress_tensor, transpose=True)
    errors["complete_field"] = relative(actual_field, expected_field)
    actual_vm = qd_to_numpy(state.von_mises, transpose=True)[..., 0]
    expected_vm = np.sqrt(
        0.5
        * (
            (expected_field[..., 0] - expected_field[..., 1]) ** 2
            + (expected_field[..., 1] - expected_field[..., 2]) ** 2
            + (expected_field[..., 2] - expected_field[..., 0]) ** 2
        )
        + 3.0 * np.sum(expected_field[..., 3:] ** 2, axis=-1)
    )
    errors["complete_von_mises"] = relative(actual_vm, expected_vm)
    errors["global_peak"] = relative(qd_to_numpy(state.step_peak), expected_vm.max(axis=(1, 2)))
    e1, e2 = oracle.axes[:, :, :, 0], oracle.axes[:, :, :, 1]
    tensor_basis = np.stack(
        (
            e1[:, :, :, None] * e1[:, :, None, :],
            e2[:, :, :, None] * e2[:, :, None, :],
            e1[:, :, :, None] * e2[:, :, None, :] + e2[:, :, :, None] * e1[:, :, None, :],
        ),
        axis=2,
    )
    for label, buffer in (("N", state.membrane_resultant), ("M", state.bending_resultant)):
        tensor = np.einsum("egkij,egkb->begij", tensor_basis, reference_stress[label])
        components = np.stack([tensor[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)
        errors[label] = relative(qd_to_numpy(buffer, transpose=True), components)
    transverse = np.einsum("egij,egjb->begi", oracle.axes[:, :, :, :2], reference_stress["Q"])
    errors["Q"] = relative(qd_to_numpy(state.transverse_resultant, transpose=True), transverse)

    np.savez(
        output / f"{name}-fields.npz",
        expected=expected_field,
        actual=actual_field,
        force=force,
        displacement=displacement,
    )
    result = {
        "name": name,
        "method": method,
        "spinning": spinning,
        "gauge": gauge,
        "dofs": oracle.ndof,
        "quads": len(quads),
        "errors": errors,
        "native_valid": qd_to_numpy(state.valid).tolist(),
        "build_timings": asdict(native.build_timings),
        "asset": file_id(asset),
    }
    write_json(output / f"{name}.json", result)
    assert all(
        errors[key] < 1e-11
        for key in (
            "stiffness",
            "directors",
            "membrane",
            "bending",
            "shear",
            "mass_modes",
            "gram",
            "rigid_null_residual",
            "balanced_load",
            "centrifugal",
        )
    ), result
    assert errors["complete_residual"] < options.tolerance, result
    assert max(errors[key] for key in ("complete_field", "complete_von_mises", "global_peak", "N", "M", "Q")) < 1e-8, (
        result
    )
    assert qd_to_numpy(state.valid).all(), result
    np.testing.assert_array_equal(actual_field[3], 0.0)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    cli.add_argument("--method", choices=("direct", "inverse"), default="direct")
    cli.add_argument("--gauge", choices=("legacy", "translation"), default="translation")
    cli.add_argument("--spinning", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"), "genesis": gs.__file__},
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    fixtures = [
        ("plate", *geometry.plate_mesh(2, Lx=0.04)),
        ("sphere", *geometry.sphere_mesh(2)),
        ("egg", *geometry.egg_mesh(2, axes=(0.022, 0.022, 0.03), asym=-0.18)),
    ]
    results = [
        control(args.output, name, vertices, quads, args.method, args.gauge, args.spinning)
        for name, vertices, quads in fixtures
    ]
    write_json(args.output / "result.json", {"passed": True, "results": results})


if __name__ == "__main__":
    main()
