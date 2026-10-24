"""Bounded native shell membrane/bending checks against analytical and independent FEM references.

The sphere exercises the production inertia-relief recovery. The supported
plate is an operator fixture with prescribed constraints, not a new public
boundary-condition API. Neither case validates local three-dimensional
contact peaks. Run on a compute node after performance tuning.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse

import genesis as gs
from genesis.engine.solvers.rigid.stress.factor import StressFactor
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu import geometry, reference

from .artifacts import file_id, runtime_identity, write_json


def tensor(components):
    result = np.zeros((*components.shape[:-1], 3, 3))
    for k, (a, b) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        result[..., a, b] = components[..., k]
        result[..., b, a] = components[..., k]
    return result


def weighted_error(actual, expected, weights):
    return float(
        np.sqrt(
            np.sum(weights[..., None, None] * (actual - expected) ** 2)
            / max(np.sum(weights[..., None, None] * expected**2), 1e-300)
        )
    )


def control(output, kind, n):
    young, poisson, thickness = 1e10, 0.3, 0.0005
    radius, side = 0.025, 0.1
    pressure = 100.0 if kind == "sphere" else 10.0
    if kind == "sphere":
        vertices, quads = geometry.sphere_mesh(n, radius)
        directors = vertices / radius
        force = reference.uniform_surface_pressure(vertices, quads, pressure)
    else:
        vertices, quads = reference.plate_mesh(n, side)
        directors = np.tile([0.0, 0.0, 1.0], (len(vertices), 1))
        force = reference.plate_uniform_load(vertices, quads, pressure)
    asset = output / f"{kind}-{n}.npz"
    np.savez(asset, vertices=vertices, quads=quads, directors=directors)
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset,
        backend="shell",
        method="direct",
        output_mode="full",
        young=young,
        poisson=poisson,
        shell_thickness=thickness,
        density=1000.0,
    )
    native = ShellModel(options)
    state = native.create_state(1, "full")
    info = native.info
    matrix = sparse.bsr_matrix(
        (qd_to_numpy(info.stiffness) * young, qd_to_numpy(info.columns), qd_to_numpy(info.row_start)),
        shape=(len(force), len(force)),
    ).tocsr()
    if kind == "sphere":
        state.force.from_numpy(force.reshape((-1, 1, 3)))
        omega = V_VEC(3, dtype=gs.qd_float, shape=(1,))
        omega.fill(0.0)
        native.recover(omega, state)
        applied = qd_to_numpy(state.rhs, transpose=True).reshape(-1)
        fixed = qd_to_numpy(info.pins)
        assert qd_to_numpy(state.valid).all()
    else:
        fixed = reference.plate_simply_supported_dofs(vertices, side)
        free = np.ones(len(force), dtype=gs.np_int)
        free[fixed] = 0
        info.is_free.from_numpy(free.reshape((-1, 3)))
        factor = StressFactor(info)
        state.rhs.from_numpy(force.reshape((-1, 1, 3)))
        factor.solve(young, state, info)
        native.evaluate_stress(options, state)
        applied = force.copy()
    displacement = qd_to_numpy(state.displacement, transpose=True).reshape(-1)
    free = np.ones(len(force), dtype=bool)
    free[fixed] = False
    residual = matrix @ displacement - applied
    free_residual = float(np.linalg.norm(residual[free]) / max(np.linalg.norm(applied[free]), 1e-30))
    independent = reference.Pyfe3dReference(vertices, quads, E=young, nu=poisson, h=thickness)
    reference_u = independent.solve(applied, fixed_dofs=fixed)
    reference_fields = independent.stress(reference_u)
    fields = tensor(qd_to_numpy(state.stress_tensor, transpose=True)[0].reshape((len(quads), 2, 4, 6)))
    expected = np.stack((reference_fields["sigma_top"], reference_fields["sigma_bottom"]), axis=1)
    weights = qd_to_numpy(native.shell.weights)
    row = {
        "case": kind,
        "n": n,
        "nodes": len(vertices),
        "quadrilaterals": len(quads),
        "dofs": len(force),
        "free_row_residual": free_residual,
        "same_applied_load": True,
        "native_vs_independent_surface_tensor_L2": weighted_error(fields, expected, weights[:, None, :]),
        "load_relief_relative": float(np.linalg.norm(applied - force) / np.linalg.norm(force)),
        "asset": file_id(asset),
    }
    if kind == "sphere":
        modes = reference.rigid_modes(vertices)
        clean_u = displacement - modes @ np.linalg.lstsq(modes, displacement, rcond=None)[0]
        clean_reference_u = reference_u - modes @ np.linalg.lstsq(modes, reference_u, rcond=None)[0]
        analytical_stress = pressure * radius / (2 * thickness)
        analytical_u = pressure * radius**2 * (1 - poisson) / (2 * young * thickness)
        row.update(analytic_membrane_stress_Pa=analytical_stress, analytic_radial_displacement_m=analytical_u)
        for label, u, stress in (("native", clean_u, fields), ("independent", clean_reference_u, expected)):
            radial = np.einsum("ij,ij->i", u.reshape((-1, 6))[:, :3], directors)
            mean_stress = np.average(np.trace(stress.mean(axis=1), axis1=-2, axis2=-1) / 2, weights=weights)
            row[label + "_mean_membrane_relative_error"] = float(abs(mean_stress / analytical_stress - 1))
            row[label + "_mean_radial_relative_error"] = float(abs(radial.mean() / analytical_u - 1))
        row["complete_residual"] = float(np.linalg.norm(residual) / np.linalg.norm(applied))
    else:
        signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        gauss = signs / np.sqrt(3)
        shape = 0.25 * (1 + gauss[:, :1] * signs[:, 0]) * (1 + gauss[:, 1:] * signs[:, 1])
        positions = np.einsum("gn,enp->egp", shape, vertices[quads])
        _, _, analytic = reference.navier_square_field(
            positions.reshape((-1, 3))[:, :2], young, poisson, thickness, side, pressure
        )
        analytic = analytic.reshape((-1, 4, 3, 3))
        center = np.argmin(np.linalg.norm(vertices - [side / 2, side / 2, 0], axis=1))
        exact = reference.navier_square_center(young, poisson, thickness, side, pressure)
        row["analytic_center_displacement_m"] = exact["center_w_m"]
        for label, u, stress in (("native", displacement, fields), ("independent", reference_u, expected)):
            row[label + "_center_displacement_relative_error"] = float(abs(u[6 * center + 2] / exact["center_w_m"] - 1))
            row[label + "_bending_tensor_relative_error"] = weighted_error(stress[:, 0], analytic, weights)
        row["support_reaction_sum_N"] = float(residual.reshape((-1, 6))[:, 2].sum())
    np.savez(
        output / f"{kind}-{n}-fields.npz",
        native_displacement=displacement,
        independent_displacement=reference_u,
        original_force=force,
        applied_force=applied,
        native_stress=fields,
        independent_stress=expected,
        residual=residual,
        weights=weights,
        membrane=qd_to_numpy(state.membrane_resultant, transpose=True),
        bending=qd_to_numpy(state.bending_resultant, transpose=True),
        transverse=qd_to_numpy(state.transverse_resultant, transpose=True),
    )
    write_json(output / f"{kind}-{n}.json", row)
    assert np.isfinite(fields).all() and free_residual < 1e-8, row
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--case", choices=("sphere", "plate"), required=True)
    cli.add_argument("--sizes", type=int, nargs="+", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    write_json(args.output / "start.json", {"identity": runtime_identity(), "availability": reference.availability()})
    rows = [control(args.output, args.case, n) for n in args.sizes]
    write_json(
        args.output / "result.json",
        {
            "rows": rows,
            "availability": reference.availability(),
            "scope": "Global linear shell membrane/bending at resolved load scales; no local 3D contact-peak certification.",
            "supported_plate": "Native operators and factor with research-only prescribed supports; production sphere uses native inertia relief.",
        },
    )


if __name__ == "__main__":
    main()
