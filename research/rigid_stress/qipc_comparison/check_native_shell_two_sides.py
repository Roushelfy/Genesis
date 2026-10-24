"""Independent two-plane pressure moments for a support ball crossing both shell sides."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.scatter import (
    create_scatter_workspace,
    kernel_pressure_moments,
    kernel_scatter_faces,
)
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu import core, geometry

from .artifacts import runtime_identity, write_json
from .check_native_shell_contacts import field_components
from .check_native_shell_operators import relative


def plane_moments(radius, height, profile):
    squared = radius**2 - height**2
    if profile == "hertz":
        return 2 * np.pi * squared**1.5 / (3 * radius), squared / 5
    points, weights = np.polynomial.legendre.leggauss(64)
    radial_squared = squared * (points + 1) / 2
    distance = (radial_squared + height**2) / radius**2
    density = np.exp(-0.5 * distance / 0.45**2) * (1 - distance) ** 2
    mass = np.pi * squared / 2 * (weights @ density)
    second = np.pi * squared / 4 * (weights @ (radial_squared * density))
    return mass, second / mass


def control(output, profile, reuse_moments):
    vertices, quads = geometry.plate_mesh(2, Lx=0.04)
    asset = output / (profile + "-mesh.npz")
    np.savez(asset, vertices=vertices, quads=quads)
    options = gs.options.RigidStressOptions.dpr(mesh=asset, backend="shell", method="direct", output_mode="full")
    native = ShellModel(options)
    oracle = core.ShellModel(vertices, quads, density=options.density, gauge="translation")
    surface = StressSurface(24, native.load_info)
    radius, n_envs = 0.0008, 2
    contacts = create_contacts(1, n_envs, radius, hertz_profile=profile == "hertz")
    state = native.create_state(n_envs, "full")
    workspace = create_scatter_workspace(n_envs, 16, reuse_moments)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    center = vertices[quads[0]].mean(axis=0)
    extent = np.ptp(vertices[quads[0]], axis=0)[:2] / 2
    signs = (vertices[quads[0], :2] - center[:2]) / extent
    linear = np.column_stack((signs / (4 * extent), np.zeros(4)))
    positions, forces, normals = (np.zeros((1, n_envs, 3)) for _ in range(3))
    expected = np.zeros((len(vertices), 6, n_envs))
    side_rows = []
    for i_b, source_side in enumerate((1, -1)):
        point = center + np.array([0.0, 0.0, source_side * options.shell_thickness / 2])
        force = 0.01 * np.array([0.12, 0.07, -source_side])
        direction = force / np.linalg.norm(force)
        first = np.cross(direction, [0.0, 1.0, 0.0])
        first /= np.linalg.norm(first)
        second = np.cross(direction, first)
        directions = np.stack((np.zeros(3), first, second)) / radius
        gram = np.zeros((3, 3))
        moments = []
        for side in (1, -1):
            height = (side - source_side) * options.shell_thickness / 2
            mass, variance = plane_moments(radius, height, profile)
            offset = np.array([0.0, 0.0, height])
            mean = np.array([1.0, offset @ first / radius, offset @ second / radius])
            covariance = np.diag([variance, variance, 0.0])
            gram += mass * (np.outer(mean, mean) + directions @ covariance @ directions.T)
            moments.append(mass * (mean[None] / 4 + linear @ covariance @ directions.T))
        coefficient = np.linalg.solve(gram, [1.0, 0.0, 0.0])
        assert coefficient[0] > np.linalg.norm(coefficient[1:])
        for side, moment in zip((1, -1), moments):
            nodal = (moment @ coefficient)[:, None] * force
            expected[quads[0], :3, i_b] += nodal
            offset = np.array([0.0, 0.0, side * options.shell_thickness / 2])
            expected[quads[0], 3:, i_b] += np.cross(offset, nodal)
            side_rows.append(dict(env=i_b, side=side, force=nodal.sum(axis=0).tolist()))
        positions[0, i_b], forces[0, i_b] = point, force
        normals[0, i_b] = [0.0, 0.0, -source_side]
    contacts.position.from_numpy(positions)
    contacts.force.from_numpy(forces)
    contacts.normal.from_numpy(normals)
    contacts.valid.fill(True)
    contacts.friction.fill(0.5)
    kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
    if reuse_moments:
        kernel_pressure_moments(contacts, state, native.load_info, surface.info, workspace, True)
        kernel_scatter_faces(contacts, state, native.load_info, surface.info, workspace, True, True)
    else:
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, native.load_info, surface.info)
    actual = qd_to_numpy(state.force, transpose=True).reshape((n_envs, -1)).T
    native.recover(omega, state)
    expected = expected.reshape((-1, n_envs))
    expected_field = field_components(oracle.stress(oracle.solve(expected)))
    actual_field = qd_to_numpy(state.stress_tensor, transpose=True).reshape(expected_field.shape)
    errors = dict(load=relative(actual, expected), field=relative(actual_field, expected_field))
    np.savez(output / (profile + "-fields.npz"), actual_load=actual, expected_load=expected, actual_field=actual_field)
    row = dict(
        profile=profile, radius=radius, errors=errors, sides=side_rows, statuses=qd_to_numpy(contacts.status).tolist()
    )
    write_json(output / (profile + ".json"), row)
    assert (qd_to_numpy(contacts.status) == 0).all() and qd_to_numpy(state.valid).all(), row
    assert errors["load"] < 1e-9 and errors["field"] < 1e-8, row
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--reuse-moments", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", dict(identity=runtime_identity(require_gpu_inventory=args.backend == "gpu")))
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    assert not args.reuse_moments or args.backend == "gpu"
    rows = [control(args.output, profile, args.reuse_moments) for profile in ("hertz", "dpr")]
    write_json(args.output / "result.json", dict(passed=True, rows=rows))


if __name__ == "__main__":
    main()
