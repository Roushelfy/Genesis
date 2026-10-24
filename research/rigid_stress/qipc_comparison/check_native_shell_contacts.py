"""Compare physical Q4 native DPR loads with the independent support-ball CPU control."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_scatter_faces
from genesis.engine.solvers.rigid.stress.shell_model import ShellModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.shell_cpu import core, geometry
from research.rigid_stress.shell_cpu.contacts import Contact, ShellDPRTransfer, _shape
from research.rigid_stress.shell_cpu.native_dpr_control import _oracle, native_shell_transfer

from .artifacts import runtime_identity, write_json
from .check_native_shell_operators import relative


def field_components(stress):
    tensor = np.moveaxis(np.stack((stress["sigma_top"], stress["sigma_bottom"]), axis=1), -1, 0)
    return np.stack([tensor[..., a, b] for a, b in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)


def control(output, name, vertices, quads, quadrature, scatter):
    asset = output / f"{name}.npz"
    np.savez(asset, vertices=vertices, quads=quads)
    options = gs.options.RigidStressOptions.dpr(
        mesh=asset, backend="shell", method="direct", output_mode="full", quadrature=quadrature
    )
    native = ShellModel(options)
    cpu = core.ShellModel(vertices, quads, density=options.density, gauge="translation")
    transfer = ShellDPRTransfer(cpu)
    oracle, _path = _oracle(Path(__file__).resolve().parents[3])
    surface = StressSurface(quadrature, native.load_info)
    n_envs, n_contacts, radius = 6, 2, 8e-5
    state = native.create_state(n_envs, "full")
    contacts = create_contacts(n_contacts, n_envs, radius, hertz_profile=True)
    workspace = create_scatter_workspace(n_envs, 16)
    omega = V_VEC(3, dtype=gs.qd_float, shape=(n_envs,))
    omega.fill(0.0)
    positions, forces, normals = (np.zeros((n_contacts, n_envs, 3)) for _ in range(3))
    valid = np.zeros((n_contacts, n_envs), dtype=bool)
    expected = np.zeros((cpu.ndof, n_envs))
    coarse = np.zeros_like(expected)
    rows = []
    # Distinct environments include both material sides, eccentric friction, a shared edge,
    # a force-line anchor offset, no contact, and rejected friction. Environment zero is asymmetric.
    for i_c, i_b, face, xi, eta, side, tangent_fraction in [
        (0, 0, 0, 0.15, -0.3, 1, 0.0),
        (1, 0, len(quads) - 1, -0.25, 0.1, 1, 0.17),
        (0, 1, 0, -0.25, 0.2, -1, 0.2),
        (0, 2, 0, 1.0, -0.1, 1, 0.1),
        (0, 3, 0, 0.3, 0.4, 1, -0.2),
        (0, 5, 0, 0.15, -0.3, 1, 2.0),
    ]:
        shape, dxi, deta = _shape(xi, eta)
        physical = transfer.faces[side][0][face]
        point = shape @ physical
        outward = side * np.cross(dxi @ physical, deta @ physical)
        outward /= np.linalg.norm(outward)
        tangent = dxi @ physical
        tangent /= np.linalg.norm(tangent)
        force = 0.01 * (-outward + tangent_fraction * tangent)
        if i_b == 3:
            point = point + 0.2 * radius * force / np.linalg.norm(force)
        positions[i_c, i_b], forces[i_c, i_b], normals[i_c, i_b] = point, force, -outward
        valid[i_c, i_b] = True
        if i_b != 5:
            contact = Contact(point, force, outward, radius=radius, side=side, friction=0.5)
            load, diagnostic = native_shell_transfer(transfer, contact, oracle, order=96)
            lower, _ = native_shell_transfer(transfer, contact, oracle, order=48)
            expected[:, i_b] += load
            coarse[:, i_b] += lower
            rows.append(dict(slot=i_c, env=i_b, side=side, diagnostic=diagnostic))
    contacts.position.from_numpy(positions)
    contacts.force.from_numpy(forces)
    contacts.normal.from_numpy(normals)
    contacts.valid.from_numpy(valid)
    contacts.friction.fill(0.5)
    kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
    kernel_pressure(contacts, surface.info)
    if scatter == "faces" and gs.backend == gs.cuda:
        kernel_scatter_faces(contacts, state, native.load_info, surface.info, workspace, True, False)
    else:
        kernel_scatter(contacts, state, native.load_info, surface.info)
    actual = qd_to_numpy(state.force, transpose=True).reshape((n_envs, -1)).T
    native.recover(omega, state)
    field = qd_to_numpy(state.stress_tensor, transpose=True).reshape((n_envs, len(quads), 2, 4, 6))
    reference_field = field_components(cpu.stress(cpu.solve(expected)))
    same_load_field = field_components(cpu.stress(cpu.solve(actual)))
    nodal = actual.reshape((-1, 6, n_envs))
    actual_force = nodal[:, :3].sum(axis=0).T
    actual_moment = (
        np.cross(vertices[:, :, None], nodal[:, :3], axisa=1, axisb=1, axisc=1).sum(axis=0).T
        + nodal[:, 3:].sum(axis=0).T
    )
    desired_force = forces[:, :5].sum(axis=0)
    desired_moment = np.cross(positions[:, :5], forces[:, :5]).sum(axis=0)
    scales = np.maximum(np.linalg.norm(forces[:, :5], axis=2).sum(axis=0), 1e-30)
    errors = dict(
        generalized_load=relative(actual, expected),
        translation_load=relative(nodal[:, :3], expected.reshape((-1, 6, n_envs))[:, :3]),
        rotational_load=relative(nodal[:, 3:], expected.reshape((-1, 6, n_envs))[:, 3:]),
        oracle_quadrature=relative(coarse, expected),
        complete_field=relative(field, reference_field),
        same_load_complete_field=relative(field, same_load_field),
        force=float(np.max(np.linalg.norm(actual_force[:5] - desired_force, axis=1) / scales)),
        moment=float(np.max(np.linalg.norm(actual_moment[:5] - desired_moment, axis=1) / (scales * radius))),
    )
    result = dict(
        name=name,
        quadrature=quadrature,
        scatter=scatter,
        errors=errors,
        status=qd_to_numpy(contacts.status).tolist(),
        valid=qd_to_numpy(state.valid).tolist(),
        contacts=rows,
    )
    write_json(output / f"{name}.json", result)
    np.savez(
        output / f"{name}-fields.npz",
        actual_load=actual,
        expected_load=expected,
        actual_field=field,
        expected_field=reference_field,
    )
    assert np.all(qd_to_numpy(contacts.status)[:, :5] == 0), result
    assert qd_to_numpy(contacts.status)[0, 5] == 1, result
    assert max(errors[k] for k in ["force", "moment", "same_load_complete_field"]) < 1e-8, result
    assert (
        max(errors[k] for k in ["generalized_load", "rotational_load", "complete_field", "oracle_quadrature"]) < 1e-6
    ), result
    np.testing.assert_array_equal(actual[:, 4:], 0.0)
    # Slot permutations, complete release and recontact must not retain the previous load.
    for label, mask, permutation in [
        ("swap", valid, [1, 0]),
        ("release", np.zeros_like(valid), [0, 1]),
        ("recontact", valid, [0, 1]),
    ]:
        contacts.position.from_numpy(positions[permutation].copy())
        contacts.force.from_numpy(forces[permutation].copy())
        contacts.normal.from_numpy(normals[permutation].copy())
        contacts.valid.from_numpy(mask[permutation].copy())
        kernel_anchor(float(np.finfo(np.float64).eps), contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, native.load_info, surface.info)
        updated = qd_to_numpy(state.force, transpose=True).reshape((n_envs, -1)).T
        np.testing.assert_allclose(updated, 0.0 if label == "release" else actual, rtol=1e-11, atol=1e-16)
    result["slot_release_recontact_passed"] = True
    write_json(output / f"{name}.json", result)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    cli.add_argument("--scatter", choices=["contacts", "faces"], default="faces")
    cli.add_argument("--quadrature", type=int, default=16)
    cli.add_argument(
        "--geometries", nargs="+", choices=["flat", "warped", "egg", "valley"], default=["flat", "warped", "egg"]
    )
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        dict(identity=runtime_identity(require_gpu_inventory=args.backend == "gpu"), genesis=gs.__file__),
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    flat, quads = geometry.plate_mesh(2, Lx=0.04)
    warped = flat.copy()
    warped[:, 2] = 2.5 * warped[:, 0] * warped[:, 1]
    valley = flat.copy()
    valley[:, 2] = 0.15 * np.abs(valley[:, 0])
    fixtures = [
        ("flat", flat, quads),
        ("valley", valley, quads),
        ("warped", warped, quads),
        ("egg", *geometry.egg_mesh(2, axes=(0.022, 0.022, 0.03), asym=-0.18)),
    ]
    results = [
        control(args.output, name, vertices, connectivity, args.quadrature, args.scatter)
        for name, vertices, connectivity in fixtures
        if name in args.geometries
    ]
    write_json(args.output / "result.json", dict(passed=True, results=results))


if __name__ == "__main__":
    main()
