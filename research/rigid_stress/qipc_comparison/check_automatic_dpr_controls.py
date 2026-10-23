"""Analytical radius, finite-disk P2 moments and original-FEM controls for automatic DPR."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.automatic_radius import func_estimate_radius
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.utils import array_class
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell

from .artifacts import runtime_identity, write_json


@qd.kernel
def kernel_radius_controls(
    contacts: qd.template(), collider: array_class.ColliderState, dyn: array_class.DynState, info: qd.template()
):
    for i_b in range(6):
        force = 0.005 * qd.Vector([0.2, 0.0, -1.0])
        if i_b == 1:
            force *= 8.0
        elif i_b == 2:
            force *= 0.0
        elif i_b == 3:
            force = qd.Vector([0.0, 0.0, 0.005])
        contacts.force[0, i_b] = force
        contacts.normal[0, i_b] = qd.Vector([0.0, 0.0, -1.0])
        collider.contact_data.geom_a[0, i_b] = 0
        collider.contact_data.geom_b[0, i_b] = 1
        if i_b == 5:
            collider.contact_data.geom_b[0, i_b] = 2
        collider.contact_data.normal[0, i_b] = qd.Vector([0.0, 0.0, 1.0])
        collider.contact_data.witness.vertex_a[0, i_b] = -1
        collider.contact_data.witness.vertex_b[0, i_b] = -1
        collider.contact_data.witness.point_a[0, i_b] = qd.Vector([0.0, 0.0, 0.05])
        collider.contact_data.witness.point_b[0, i_b] = qd.Vector([0.0, 0.0, 0.05])
        if i_b == 4:
            collider.contact_data.witness.point_a[0, i_b] = qd.Vector([0.05, 0.05, 0.05])
        func_estimate_radius(0, i_b, 0, contacts, collider, dyn, info)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(*[np.array([-0.05, 0.05])] * 3)
    vertices, tetrahedra, faces = mesh.p.T, mesh.t.T, mesh.facets[:, mesh.boundary_facets()].T
    asset = args.output / "cube.npz"
    np.savez(asset, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), show_viewer=False)
    box = scene.add_entity(gs.morphs.Box(size=(0.1, 0.1, 0.1)))
    scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(0, 0, 0.1), fixed=True))
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, -1)))
    box.base_link.configure_stress_recovery(gs.options.RigidStressOptions.dpr(mesh=asset, radius_model="hertz"))
    scene.build(n_envs=6)
    entry = scene.rigid_solver.stress_recovery.links[0]
    contacts = create_contacts(1, 6, 0.006)
    kernel_radius_controls(
        contacts, scene.rigid_solver.collider.collider_state, scene.rigid_solver.dyn_state, entry.automatic_radius
    )
    radius = qd_to_numpy(contacts.radius, transpose=True)[:, 0]
    status = qd_to_numpy(contacts.radius_status, transpose=True)[:, 0]
    np.testing.assert_array_equal(status, [1, 1, 2, 4, 3, 3])
    expected = (0.75 * np.array([0.005, 0.04]) * 0.05 / (1e10 / (1 - 0.3**2))) ** (1 / 3)
    np.testing.assert_allclose(radius[:2], expected, rtol=5e-14)
    assert np.isnan(radius[3:]).all()
    try:
        box.base_link.set_stress_contact_radius(0.01)
    except gs.GenesisException:
        pass
    else:
        raise AssertionError("The manual setter must not silently override automatic radius.")
    model, surface = entry.model, entry.surface
    nodes = qd_to_numpy(model.info.surface_nodes)
    face = 0
    normal = qd_to_numpy(surface.info.face_normal)[face]
    triangle = qd_to_numpy(model.info.vertices)[nodes[face, :3]]
    center = triangle.mean(axis=0)
    tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
    force = -0.03 * normal + 0.006 * tangent
    radii = np.array([1e-6, 1e-5, 1e-4])
    dual = qd_to_numpy(surface.info.face_dual)[face]
    gradients = np.vstack((-dual.sum(axis=0), dual))
    rows = []
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    direct = splu(oracle.k[free][:, free].tocsc())
    for profile in [False, True]:
        contacts = create_contacts(1, 3, 0.006, hertz_profile=profile)
        contacts.valid.fill(True)
        contacts.position.from_numpy(np.broadcast_to(center, (1, 3, 3)).copy())
        contacts.force.from_numpy(np.broadcast_to(force, (1, 3, 3)).copy())
        contacts.normal.from_numpy(np.broadcast_to(-normal, (1, 3, 3)).copy())
        contacts.friction.fill(0.5)
        contacts.radius.from_numpy(radii[None].copy())
        state = model.create_state(3, "full")
        kernel_anchor(np.finfo(float).eps, contacts, surface.info)
        kernel_pressure(contacts, surface.info)
        kernel_scatter(contacts, state, model.info, surface.info)
        np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
        load = qd_to_numpy(state.force, transpose=True)
        second_moment = 0.2
        if not profile:
            x, weight = np.polynomial.legendre.leggauss(64)
            r, weight = (x + 1) / 2, weight / 2
            measure = weight * r * np.exp(-0.5 * (r / 0.45) ** 2) * (1 - r**2) ** 2
            second_moment = 0.5 * np.dot(measure, r**2) / measure.sum()
        moments = np.ones((3, 3, 3)) / 9 + second_moment * radii[:, None, None] ** 2 * (gradients @ gradients.T)
        basis = np.column_stack(
            (2 * np.diagonal(moments, axis1=1, axis2=2) - 1 / 3, 4 * moments[:, [0, 0, 1], [1, 2, 2]])
        )
        # surface_nodes uses (01, 02, 12), matching the native surface basis.
        expected_load = np.zeros_like(load)
        expected_load[:, nodes[face]] = basis[:, :, None] * force
        relative = np.linalg.norm(load - expected_load) / np.linalg.norm(expected_load)
        assert relative < 1e-9, (profile, relative)
        omega = V_VEC(3, dtype=gs.qd_float, shape=(3,))
        omega.fill(0)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid).all()
        rhs = qd_to_numpy(state.rhs, transpose=True).reshape(3, -1).T
        displacement = np.zeros_like(rhs)
        displacement[free] = direct.solve(rhs[free])
        tensor = qd_to_numpy(state.stress_tensor, transpose=True)
        for i_b in range(3):
            expected_tensor, _ = stress_field(oracle.glambda, oracle.elements, displacement[:, i_b], 1e10, 0.3)
            np.testing.assert_allclose(tensor[i_b], expected_tensor, rtol=1e-8, atol=1e-8)
        rows.append({"profile": "hertz" if profile else "dpr", "relative_nodal_load_error": float(relative)})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "radius_m": radius[:2].tolist(),
            "radius_status": status.tolist(),
            "rows": rows,
            "scope": "Analytical circular radius and planar finite-pressure P2 moments; independent sparse solve of the same coarse FEM. Numerical conformance, not a shell contact-accuracy claim.",
        },
    )


if __name__ == "__main__":
    main()
