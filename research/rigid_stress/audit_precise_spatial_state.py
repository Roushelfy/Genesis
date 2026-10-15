"""Independently assemble centered contact, body inertia and jaw PD balances for a recovered state."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .franka_rigid_mode_coordinates import coordinate_map
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def assemble_state(static, config, u, previous, velocity, expected, bulk=None):
    dt = config["dt"]
    normal, current = static.normal_collisions(u)
    positions = static.contact_node_map @ (np.asarray(static.rest, dtype=np.longdouble) + u.reshape(-1, 3))
    surface = np.zeros(positions.shape)
    magnitudes = {}
    for index in range(len(normal)):
        collision = normal[index]
        assert type(collision).__name__ == "FaceVertexNormalCollision"
        ids = np.asarray(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
        ids = ids[ids >= 0]
        groups = static.contact_surface_groups[ids]
        local = np.asarray(positions[ids] - positions[ids[0]], dtype=float).reshape(-1)
        force = -static.barrier.gradient(collision, local).reshape(-1, 3)
        np.add.at(surface, ids, force)
        magnitudes[tuple(ids)] = float(np.linalg.norm(force[groups == 0].sum(axis=0)) / abs(collision.weight))
        del collision
    normal_surface = surface.copy()
    tangential = ipctk.TangentialCollisions()
    tangential.build(static.mesh, current, normal, static.barrier, static.friction)
    delta = np.asarray(static.contact_node_map @ (u - previous).reshape(-1, 3), dtype=float)
    friction = ipctk.FrictionPotential(dt * 4e-6)
    coulomb_violations = []
    for index in range(len(tangential)):
        collision = tangential[index]
        ids = np.asarray(collision.vertex_ids(static.mesh.edges, static.mesh.faces))
        ids = ids[ids >= 0]
        collision.normal_force_magnitude = magnitudes[tuple(ids)]
        force = -friction.gradient(collision, collision.dof(delta, static.mesh.edges, static.mesh.faces)).reshape(-1, 3)
        np.add.at(surface, ids, force)
        bound = collision.mu_s * abs(collision.weight) * collision.normal_force_magnitude
        violation = float(np.linalg.norm(force[static.contact_surface_groups[ids] == 0].sum(axis=0)) - bound)
        assert violation <= 1e-12
        coulomb_violations.append(violation)
        del collision
    contact = static.mesh.to_full_dof(surface.reshape(-1))
    np.testing.assert_allclose(contact, expected, rtol=1e-12, atol=1e-12)
    n = static.body.basis.N
    permutation = static.parts[0].permutation
    vector = np.empty(n, dtype=np.longdouble)
    vector[permutation] = u[:n]
    if bulk is None:
        bulk = WidePiolaBulk(static.body)
    gradient = np.zeros(len(u), dtype=np.longdouble)
    gradient[:n] = bulk.evaluate(vector, False)[1][permutation]
    mass = sparse.block_diag(
        (static.body.mass[permutation][:, permutation], sparse.csc_matrix((len(u) - n,) * 2)), format="csc"
    )
    for node, axis, physical_mass in zip(static.finger_com_nodes, static.axes, static.masses, strict=True):
        dofs = np.arange(3 * node, 3 * node + 3)
        local = physical_mass * np.eye(3) + 0.1 * np.outer(axis, axis)
        rows = np.repeat(dofs, 3)
        columns = np.tile(dofs, 3)
        mass += sparse.coo_matrix((local.ravel(), (rows, columns)), shape=mass.shape).tocsc()
        q = 0.04 + u[dofs] @ axis
        dq = (u[dofs] - previous[dofs]) @ axis
        effort = np.clip(100 * (q - 0.018) + 10 * dq / dt, -4, 4) + dq / dt
        gradient[dofs] += effort * axis
    acceleration = (u - previous - dt * velocity) / dt**2
    gradient += mass @ acceleration - contact - static.gravity_force([0, 0, -9.81])
    mapping = coordinate_map(static)
    reduced = mapping.T @ gradient
    body_residual = float(np.linalg.norm(gradient[:n]))
    complete = float(np.linalg.norm(reduced))
    assert body_residual <= 1e-9 and complete <= 1e-9
    triangles = positions[static.contact_surface_faces]
    area = (
        np.sqrt(np.sum(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) ** 2, axis=1)) / 2
    )
    lumped = np.zeros(len(positions))
    np.add.at(lumped, static.contact_surface_faces.ravel(), np.repeat(np.asarray(area, dtype=float) / 3, 3))
    assert np.all(lumped > 0)
    body_nodes = static.contact_surface_groups == 0
    active = body_nodes & (np.linalg.norm(normal_surface, axis=1) > 1e-8)
    metrics = {
        "body_tetrahedra": static.body.mesh.nelements,
        "complete_body_dynamic_residual_N": body_residual,
        "complete_mapped_dynamic_residual_N": complete,
        "force_pullback_max_difference_N": float(abs(contact - expected).max()),
        "normal_stencils": len(normal),
        "maximum_coulomb_violation_N": max(coulomb_violations, default=0),
        "active_body_contact_nodes": int(active.sum()),
        "active_lumped_contact_area_m2": float(lumped[active].sum()),
        "maximum_unsmoothed_body_normal_force_per_area_Pa": float(
            (np.linalg.norm(normal_surface, axis=1) / lumped)[body_nodes].max()
        ),
    }
    fields = {
        "contact_force_N": contact,
        "body_gradient_N": gradient[:n],
        "mapped_gradient_N": reduced,
        "body_acceleration_m_s2": acceleration[:n],
        "surface_positions_m": positions,
        "normal_surface_force_N": normal_surface,
        "shear_surface_force_N": surface - normal_surface,
        "surface_vertex_lumped_area_m2": lumped,
        "active_body_contact_nodes": np.flatnonzero(active),
    }
    return metrics, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for run in [args.source_run, args.recovery_run]:
        closed_source(run)
    assert int((args.recovery_run / "returncode.txt").read_text()) == 0
    report = json.loads((args.recovery_run / "results/summary.json").read_text())
    assert report["recovered"] and Path(report["source_run"]).resolve() == args.source_run.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.source_run)
    with np.load(args.recovery_run / "results/recovered.npz") as saved:
        u, previous, velocity, expected = [
            saved[k].copy() for k in ["displacement", "previous_displacement", "previous_velocity", "contact_force_N"]
        ]
    old_paths = sorted((args.source_run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    with np.load(old_paths[-1]) as saved:
        np.testing.assert_array_equal(previous, saved["displacement"])
        np.testing.assert_array_equal(velocity, saved["velocity"])
    metrics, fields = assemble_state(static, config, u, previous, velocity, expected)
    result = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "recovery_run": str(args.recovery_run),
        "recovered_step": len(old_paths) + 1,
        **metrics,
        "goal_completion": False,
        "limitations": "This reconstructs the recovered physical step without importing or calling the precision-aware coupled solver. Contact arithmetic, original consistent mass/armature, PD/caps/passive damping and unchanged force budgets are explicit. Full remaining history, independent old-state arithmetic comparisons and spatial/contact/geometry convergence remain required.",
    }
    np.savez_compressed(args.output / "audited_fields.npz", **fields)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
