"""Compare production friction tensors with independent supplied-traction fields."""

import argparse
from math import pi
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import kernel_hertz_stress_quadrature
from genesis.engine.solvers.rigid.stress.hertz_tangential import (
    StressHertzShear,
    func_cerruti_tensor,
    func_hertz_friction_particular,
)
from genesis.utils.array_class import V, V_MAT, V_VEC

from .artifacts import runtime_identity, write_json
from .check_native_hertz_recovery import array
from .hertz_stress import elliptic_stress
from .hertz_tangential import cerruti_stress, elliptic_tangential_stress
from .inspect_live_hertz_friction import particular


@qd.kernel
def kernel_points(position: qd.Tensor, force: qd.Tensor, poisson: qd.Tensor, tensor: qd.Tensor):
    for i_p in range(position.shape[0]):
        tensor[i_p] = func_cerruti_tensor(position[i_p], force[i_p], poisson[i_p])


@qd.kernel
def kernel_fields(
    position: qd.Tensor,
    axes: qd.Tensor,
    poisson: qd.Tensor,
    quadrature: qd.Tensor,
    tensor: qd.Tensor,
    status: qd.Tensor,
    shear: StressHertzShear,
    far_field: qd.template(),
):
    for i_q, i_p in qd.ndrange(position.shape[0], position.shape[1]):
        force = 2 * pi * axes[i_p][0] * axes[i_p][1] / 3
        tensor[i_q, i_p], status[i_q, i_p] = func_hertz_friction_particular(
            i_p, position[i_q, i_p], axes[i_p], 1.0, force, poisson[i_p], quadrature, shear, far_field
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"cerruti_relative": 1e-11, "matched_primitive_absolute_over_pressure": 1e-7, "far_bank_relative": 1e-6}
    write_json(args.output / "start.json", {"identity": runtime_identity(), "targets": targets})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    rng = np.random.default_rng(507)
    points = rng.uniform([-2, -2, 0.01], [2, 2, 3], (160, 3)) * np.geomspace(1e-5, 2, 160)[:, None]
    forces = rng.normal(size=(160, 2))
    poisson = np.tile([-0.2, 0, 0.3, 0.49], 40)
    position, force = V_VEC(3, dtype=gs.qd_float, shape=(160,)), V_VEC(2, dtype=gs.qd_float, shape=(160,))
    nu, tensor = V(dtype=gs.qd_float, shape=(160,)), V_MAT(3, 3, dtype=gs.qd_float, shape=(160,))
    position.from_numpy(points)
    force.from_numpy(forces)
    nu.from_numpy(poisson)
    kernel_points(position, force, nu, tensor)
    actual = array(tensor)
    expected = np.stack(
        [cerruti_stress(p[None], np.zeros((1, 2)), f[None], n)[0] for p, f, n in zip(points, forces, poisson)]
    )
    relative = np.linalg.norm(actual - expected, axis=(1, 2)) / np.linalg.norm(expected, axis=(1, 2))
    assert relative.max() < targets["cerruti_relative"], relative.max()
    np.savez_compressed(
        args.output / "cerruti.npz", points=points, forces=forces, poisson=poisson, actual=actual, expected=expected
    )
    # Oscillating primitive coefficients include zero-net-force residual fields.
    axes_np = np.array([[1, 1], [1, 0.4], [1, 0.05], [1, 1]])
    coefficients = rng.uniform(-0.2, 0.2, (16, 4, 2))
    coefficients[:, 0] = 0
    radii = np.arange(1, 17) / 16
    coefficients[-1, 3] = -np.sum(coefficients[:-1, 3] * radii[:-1, None] ** 3, axis=0)
    unit = np.r_[
        rng.uniform([-1.5, -1.5, 0.01], [1.5, 1.5, 3], (48, 3)),
        [[0, 0, 0], [1, 0, 0], [0.5, 0, 0], [2, 1, 0], [0, 0, 20], [20, 3, 10]],
    ]
    positions = np.broadcast_to(unit[:, None], (len(unit), 4, 3)).copy()
    positions[:, :, 1] *= axes_np[None, :, 1]
    directions = rng.uniform([-1, -1, 0.1], [1, 1, 1], (12, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    far_points = np.concatenate([distance * directions for distance in [8.0, 8.0001, 12, 32]])
    positions = np.concatenate([positions, np.broadcast_to(far_points[:, None], (len(far_points), 4, 3))])
    count = len(positions)
    position = V_VEC(3, dtype=gs.qd_float, shape=(count, 4))
    axes = V_VEC(2, dtype=gs.qd_float, shape=(4,))
    nu = V(dtype=gs.qd_float, shape=(4,))
    quadrature = V_VEC(2, dtype=gs.qd_float, shape=(16,))
    tensor = V_MAT(3, 3, dtype=gs.qd_float, shape=(count, 4))
    status = V(dtype=gs.qd_bool, shape=(count, 4))
    shear = StressHertzShear(
        V(dtype=gs.qd_float, shape=(4,)),
        V_VEC(2, dtype=gs.qd_float, shape=(4,)),
        V_VEC(2, dtype=gs.qd_float, shape=(16, 4)),
    )
    position.from_numpy(positions)
    axes.from_numpy(axes_np)
    nu.fill(0.3)
    shear.friction.fill(0.4)
    shear.coefficients.from_numpy(coefficients)
    kernel_hertz_stress_quadrature(quadrature)
    kernel_fields(position, axes, nu, quadrature, tensor, status, shear, False)
    assert array(status).all()
    actual = array(tensor)
    expected = np.stack([elliptic_stress(positions[:, i], a, peak=1, order=16) for i, a in enumerate(axes_np)])
    for i_p, axes_i in enumerate(axes_np):
        for i_s, radius in enumerate(radii):
            expected[i_p] += elliptic_tangential_stress(
                positions[:, i_p], radius * axes_i, 0.4 * coefficients[i_s, i_p], peak=radius, order=16
            )
    error = np.linalg.norm(actual - expected, axis=(2, 3))
    assert error.max() < targets["matched_primitive_absolute_over_pressure"], error.max()
    kernel_fields(position, axes, nu, quadrature, tensor, status, shear, True)
    far = array(tensor)
    assert array(status).all() and np.isfinite(far).all()
    uncompressed = np.stack(
        [
            particular(positions[:, i], a, 1.0, 2 * pi * np.prod(a) / 3, 0.4, coefficients[:, i])
            for i, a in enumerate(axes_np)
        ]
    )
    far_relative = np.linalg.norm(far - uncompressed, axis=(2, 3)) / np.maximum(
        np.linalg.norm(uncompressed, axis=(2, 3)), 1e-300
    )
    assert far_relative.max() < targets["far_bank_relative"], far_relative.max()
    far_change = np.linalg.norm(far - actual, axis=(2, 3))
    np.savez_compressed(
        args.output / "combined.npz",
        positions=positions,
        axes=axes_np,
        coefficients=coefficients,
        actual=actual,
        expected=expected,
        far=far,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "targets": targets,
            "point_count": len(points),
            "cerruti_max_relative": float(relative.max()),
            "combined_tensor_count": count * 4,
            "combined_max_over_pressure": float(error.max()),
            "moment_source_change_max_over_pressure": float(far_change.max()),
            "far_bank_max_relative_to_uncompressed": float(far_relative.max()),
            "scope": "Production supplied-traction arithmetic and independent complex-step point tensor. The oscillating bank is not a physical friction-history reference. No complete-body or actual-contact accuracy admission.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
