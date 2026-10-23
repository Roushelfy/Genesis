"""Analytical radius, finite-disk P2 moments and original-FEM controls for automatic DPR."""

import argparse
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import quadrants as qd
from scipy.sparse.linalg import splu
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.automatic_radius import StressRadiusInfo, func_estimate_radius
from genesis.engine.solvers.rigid.stress.contact import (
    StressContactState,
    create_contacts,
    func_force_frame,
    func_hertz_disk_moments,
    kernel_anchor,
    kernel_pressure,
    kernel_scatter,
)
from genesis.engine.solvers.rigid.stress.scatter import create_scatter_workspace, kernel_fit_scatter_moments
from genesis.engine.solvers.rigid.stress.surface import StressSurfaceInfo
from genesis.utils import array_class, geom
from genesis.utils.array_class import V_VEC
from genesis.utils.misc import qd_to_numpy
from research.rigid_stress.field_cpu import stress_field
from research.rigid_stress.mechanics import P2Shell

from .artifacts import runtime_identity, write_json


@qd.kernel
def kernel_disk_moments(
    face: int,
    point: qd.types.ndarray(),
    center: qd.types.ndarray(),
    radius: float,
    force: qd.types.ndarray(),
    surface: StressSurfaceInfo,
    gram: qd.types.ndarray(),
    load: qd.types.ndarray(),
):
    point_value = qd.Vector([point[a] for a in qd.static(range(3))])
    center_value = qd.Vector([center[a] for a in qd.static(range(3))])
    force_value = qd.Vector([force[a] for a in qd.static(range(3))])
    first, second = func_force_frame(force_value)
    result_gram, result_load = func_hertz_disk_moments(
        face,
        point_value,
        radius * radius - (point_value - center_value).norm_sqr(),
        center_value,
        radius,
        first,
        second,
        surface,
    )
    for i, j in qd.ndrange(3, 3):
        gram[i, j] = result_gram[i, j]
    for i, j in qd.ndrange(6, 3):
        load[i, j] = result_load[i, j]


def check_disk_moments(surface, triangle, normal, force):
    point = triangle.mean(axis=0)
    radius = 1e-4
    tangent = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
    bitangent = np.cross(normal, tangent)
    direction = force / np.linalg.norm(force)
    unit = np.eye(3)[np.argmin(np.abs(direction))]
    first = np.cross(direction, unit)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    edges = triangle[1:] - triangle[0]
    dual = np.linalg.solve(edges @ edges.T, edges)
    x, weight = np.polynomial.legendre.leggauss(64)
    u, weight = (x + 1) / 2, weight / 2
    angle = 2 * np.pi * u
    rows = []
    for distance in [0.0, 0.3 * radius, 0.9 * radius]:
        center = point + distance * normal
        remaining = radius**2 - distance**2
        radial = np.sqrt(remaining * (1 - u**2))
        positions = point + radial[:, None, None] * (
            np.cos(angle)[None, :, None] * tangent + np.sin(angle)[None, :, None] * bitangent
        )
        measure = 2 * np.pi * remaining**1.5 / radius * (weight * u**2)[:, None] * weight[None, :]
        local = (positions - triangle[0]) @ dual.T
        bary = np.concatenate((1 - local.sum(axis=-1, keepdims=True), local), axis=-1)
        shape = np.concatenate((bary * (2 * bary - 1), 4 * bary[..., [0, 0, 1]] * bary[..., [1, 2, 2]]), axis=-1)
        relative = (positions - center) / radius
        coordinates = np.stack((np.ones_like(measure), relative @ first, relative @ second), axis=-1)
        expected_gram = np.einsum("ij,ija,ijb->ab", measure, coordinates, coordinates)
        expected_load = np.einsum("ij,ija,ijb->ab", measure, shape, coordinates)
        gram, load = np.zeros((3, 3)), np.zeros((6, 3))
        kernel_disk_moments(0, point, center, radius, force, surface.info, gram, load)
        error_gram = np.linalg.norm(gram - expected_gram) / np.linalg.norm(expected_gram)
        error_load = np.linalg.norm(load - expected_load) / np.linalg.norm(expected_load)
        assert error_gram < 1e-11 and error_load < 1e-11, (distance, error_gram, error_load)
        rows.append(
            {
                "offset_over_radius": distance / radius,
                "relative_gram_error": error_gram,
                "relative_load_error": error_load,
            }
        )
    return rows


@qd.kernel
def kernel_radius_controls(
    contacts: StressContactState,
    collider: array_class.ColliderState,
    dyn: array_class.DynState,
    info: qd.template(),
    corner: bool,
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
            collider.contact_data.witness.point_a[0, i_b] = qd.Vector([0.05, 0.0, 0.05])
            if corner:
                collider.contact_data.witness.point_a[0, i_b][1] = 0.05
        collider.contact_data.witness.point_a[0, i_b] = geom.qd_transform_by_trans_quat(
            collider.contact_data.witness.point_a[0, i_b], dyn.geoms.pos[0, i_b], dyn.geoms.quat[0, i_b]
        )
        collider.contact_data.witness.point_b[0, i_b] = geom.qd_transform_by_trans_quat(
            collider.contact_data.witness.point_b[0, i_b], dyn.geoms.pos[0, i_b], dyn.geoms.quat[0, i_b]
        )
        collider.contact_data.normal[0, i_b] = geom.qd_transform_by_quat(
            collider.contact_data.normal[0, i_b], dyn.geoms.quat[0, i_b]
        )
        func_estimate_radius(0, i_b, 0, contacts, collider, dyn, info)


def disk_wedge_loads(model, surface, center, force, radii, profile):
    """Integrate P2 polynomials analytically over full disks and planar sectors at mesh edges/vertices."""
    nodes = qd_to_numpy(model.info.surface_nodes)
    coordinates = qd_to_numpy(model.info.vertices)
    triangles = coordinates[nodes[:, :3]]
    sectors = []
    radial_first, radial_second = 3 * np.pi / 16, 0.4
    if not profile:
        x, weights = np.polynomial.legendre.leggauss(64)
        r, weights = (x + 1) / 2, weights / 2
        measure = weights * r * np.exp(-0.5 * (r / 0.45) ** 2) * (1 - r**2) ** 2
        radial_first, radial_second = np.dot(measure, r) / measure.sum(), np.dot(measure, r**2) / measure.sum()
    for i_f, triangle in enumerate(triangles):
        edges = triangle[1:] - triangle[0]
        dual = np.linalg.solve(edges @ edges.T, edges)
        local = dual @ (center - triangle[0])
        bary = np.r_[1 - local.sum(), local]
        projection = triangle[0] + local @ (triangle[1:] - triangle[0])
        if np.linalg.norm(projection - center) > 1e-12 or bary.min() < -1e-12:
            continue
        zeros = np.flatnonzero(np.abs(bary) < 1e-12)
        if len(zeros) == 0:
            first = (triangle[1] - triangle[0]) / np.linalg.norm(triangle[1] - triangle[0])
            second = triangle[2] - triangle[0]
            angle = 2 * np.pi
        elif len(zeros) == 1:
            first = triangle[np.flatnonzero(bary > 1e-12)[0]] - center
            first /= np.linalg.norm(first)
            second = triangle[zeros[0]] - center
            angle = np.pi
        else:
            first = triangle[zeros[0]] - center
            first /= np.linalg.norm(first)
            second = triangle[zeros[1]] - center
            angle = np.arccos(np.clip(first @ second / np.linalg.norm(second), -1, 1))
        second = second - first * (first @ second)
        second /= np.linalg.norm(second)
        mean = radial_first * (np.sin(angle) * first + (1 - np.cos(angle)) * second) / angle
        angular = (
            (angle + np.sin(angle) * np.cos(angle)) * np.outer(first, first)
            + (angle - np.sin(angle) * np.cos(angle)) * np.outer(second, second)
            + np.sin(angle) ** 2 * (np.outer(first, second) + np.outer(second, first))
        ) / (2 * angle)
        gradients = np.vstack((-dual.sum(axis=0), dual))
        first_moment = radii[:, None] * (gradients @ mean)
        second_moment = radial_second * radii[:, None, None] ** 2 * (gradients @ angular @ gradients.T)
        moments = (
            np.outer(bary, bary)[None]
            + bary[None, :, None] * first_moment[:, None, :]
            + first_moment[:, :, None] * bary[None, None, :]
            + second_moment
        )
        basis = np.column_stack(
            (
                2 * np.diagonal(moments, axis1=1, axis2=2) - (bary[None] + first_moment),
                4 * moments[:, [0, 0, 1], [1, 2, 2]],
            )
        )
        sectors.append((i_f, angle, basis))
    assert sectors
    total_angle = sum(row[1] for row in sectors)
    result = np.zeros((len(radii), len(coordinates), 3))
    for i_f, angle, basis in sectors:
        result[:, nodes[i_f]] += angle / total_angle * basis[:, :, None] * force
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--reuse-moments", action="store_true")
    cli.add_argument("--tasks-per-env", type=int, default=8)
    cli.add_argument("--quadrature", type=int, default=10)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "genesis_module": gs.__file__,
            "quadrature": args.quadrature,
            "code_aster_runtime": {
                "run_aster": shutil.which("run_aster"),
                "as_run": shutil.which("as_run"),
                "code_aster_python_module": importlib.util.find_spec("code_aster") is not None,
                "scope": "Current Slurm image PATH and the selected Python environment only.",
            },
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(*[np.array([-0.05, 0.05])] * 3)
    vertices, tetrahedra, faces = mesh.p.T, mesh.t.T, mesh.facets[:, mesh.boundary_facets()].T
    asset = args.output / "cube.npz"
    np.savez(asset, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=faces)
    scene = gs.Scene(sim_options=gs.options.SimOptions(gravity=(0, 0, 0)), show_viewer=False)
    box = scene.add_entity(
        gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.65, -0.3, 0.15), quat=(np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0))
    )
    scene.add_entity(
        gs.morphs.Sphere(radius=0.05, pos=(0.65 + 0.1 / np.sqrt(2), -0.3, 0.15 + 0.1 / np.sqrt(2)), fixed=True)
    )
    scene.add_entity(gs.morphs.Plane(pos=(0, 0, -1)))
    box.base_link.configure_stress_recovery(
        gs.options.RigidStressOptions.dpr(mesh=asset, radius_model="hertz", quadrature=args.quadrature)
    )
    scene.build(n_envs=6)
    entry = scene.rigid_solver.stress_recovery.links[0]
    contacts = create_contacts(1, 6, 0.006)
    kernel_radius_controls(
        contacts,
        scene.rigid_solver.collider.collider_state,
        scene.rigid_solver.dyn_state,
        entry.automatic_radius,
        False,
    )
    radius = qd_to_numpy(contacts.radius, transpose=True, copy=True)[:, 0]
    status = qd_to_numpy(contacts.radius_status, transpose=True, copy=True)[:, 0]
    np.testing.assert_array_equal(status, [1, 1, 2, 4, 3, 3])
    expected = (0.75 * np.array([0.005, 0.04]) * 0.05 / (1e10 / (1 - 0.3**2))) ** (1 / 3)
    np.testing.assert_allclose(radius[:2], expected, rtol=5e-14)
    assert np.isnan(radius[3:]).all()
    planar = StressRadiusInfo(
        scene.rigid_solver.dyn_info,
        entry.automatic_radius.curvature,
        entry.automatic_radius.modulus,
        entry.automatic_radius.unloaded_radius,
        entry.automatic_radius.epsilon,
        True,
    )
    edge_rows = []
    for corner in (False, True):
        kernel_radius_controls(
            contacts, scene.rigid_solver.collider.collider_state, scene.rigid_solver.dyn_state, planar, corner
        )
        estimated = qd_to_numpy(contacts.radius, transpose=True, copy=True)[:, 0]
        codes = qd_to_numpy(contacts.radius_status, transpose=True, copy=True)[:, 0]
        np.testing.assert_array_equal(codes, [1, 1, 2, 4, 5, 3])
        np.testing.assert_allclose(estimated[[0, 1, 4]], expected[[0, 1, 0]], rtol=5e-14)
        assert np.isnan(estimated[[3, 5]]).all()
        edge_rows.append({"corner": corner, "radius_m": float(estimated[4]), "status": int(codes[4])})
    assert any(item.name.startswith("stress.curvature.") for item in scene.rigid_solver.stress_recovery.data)
    write_json(args.output / "box-edge-controls.json", {"passed": True, "rows": edge_rows})
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
    write_json(
        args.output / "disk-moments.json",
        {"passed": True, "rows": check_disk_moments(surface, triangle, normal, force)},
    )
    radii = np.array([1e-6, 1e-5, 1e-4])
    edge_pairs = np.array([[0, 1], [0, 2], [1, 2]])
    longest = edge_pairs[np.argmax(np.linalg.norm(triangle[edge_pairs[:, 0]] - triangle[edge_pairs[:, 1]], axis=1))]
    locations = [
        ("interior", center, force),
        ("mesh_edge", triangle[longest].mean(axis=0), force),
        ("mesh_vertex", np.full(3, 0.05), -0.03 * np.ones(3) / np.sqrt(3)),
    ]
    rows = []
    oracle = P2Shell(vertices, tetrahedra, faces, 1e10, 0.3, 2000.0, 2, factor_backend="none")
    free = np.setdiff1d(np.arange(oracle.ndof), qd_to_numpy(model.info.pins))
    direct = splu(oracle.k[free][:, free].tocsc())
    workspace = create_scatter_workspace(3, args.tasks_per_env, True) if args.reuse_moments else None
    for location, center, force in locations:
        for profile in [False, True]:
            contacts = create_contacts(1, 3, 0.006, hertz_profile=profile)
            contacts.valid.fill(True)
            contacts.position.from_numpy(np.broadcast_to(center, (1, 3, 3)).copy())
            contacts.force.from_numpy(np.broadcast_to(force, (1, 3, 3)).copy())
            contacts.normal.from_numpy(np.broadcast_to(force / np.linalg.norm(force), (1, 3, 3)).copy())
            contacts.friction.fill(0.5)
            contacts.radius.from_numpy(radii[None].copy())
            state = model.create_state(3, "full")
            kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
            if workspace is None:
                kernel_pressure(contacts, surface.info)
                kernel_scatter(contacts, state, model.info, surface.info)
            else:
                assert args.backend == "gpu"
                kernel_fit_scatter_moments(contacts, state, model.info, surface.info, workspace, True)
            np.testing.assert_array_equal(qd_to_numpy(contacts.status), 0)
            load = qd_to_numpy(state.force, transpose=True)
            expected_load = disk_wedge_loads(model, surface, center, force, radii, profile)
            relative = np.linalg.norm(load - expected_load) / np.linalg.norm(expected_load)
            row = {
                "location": location,
                "profile": "hertz" if profile else "dpr",
                "relative_nodal_load_error": float(relative),
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
            assert relative < 1e-9, row
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
