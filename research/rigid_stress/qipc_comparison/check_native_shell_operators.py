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


def control(output, name, vertices, quads, method, gauge):
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
    omega.fill(0.0)
    native.recover(omega, state)
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape((4, -1)).T
    reference_displacement = oracle.solve(force)
    expected_load, _coeff = oracle.balance_load(force)
    actual_load = qd_to_numpy(state.rhs, transpose=True).reshape((4, -1)).T
    errors["balanced_load"] = relative(actual_load, expected_load)
    errors["complete_residual"] = float(oracle.residual(displacement, force))
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
        )
    ), result
    assert errors["complete_residual"] < options.tolerance, result
    assert max(errors[key] for key in ("complete_field", "complete_von_mises", "global_peak")) < 1e-8, result
    assert qd_to_numpy(state.valid).all(), result
    np.testing.assert_array_equal(actual_field[3], 0.0)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    cli.add_argument("--method", choices=("direct", "inverse"), default="direct")
    cli.add_argument("--gauge", choices=("legacy", "translation"), default="translation")
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
        control(args.output, name, vertices, quads, args.method, args.gauge) for name, vertices, quads in fixtures
    ]
    write_json(args.output / "result.json", {"passed": True, "results": results})


if __name__ == "__main__":
    main()
