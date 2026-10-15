"""Check actual P2 collision interpolation and active current-friction chain rules."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse

from .contact_reference import IndependentContactSystem
from .franka_contact_reference import RigidFingerContactSystem, build_from_capture
from .refined_franka_contact import refine_contact
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


class ContactOnly(IndependentContactSystem):
    def elastic_displacement(self, displacement):
        return np.zeros_like(displacement)


def contact_only(system, mapped):
    result = ContactOnly.__new__(ContactOnly)
    if mapped:
        result.mesh, result.rest = system.mesh, system.rest
        result.contact_node_map = system.contact_node_map
    else:
        result.rest = system.contact_surface_rest
        faces = system.contact_surface_faces
        edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        result.mesh = ipctk.CollisionMesh(result.rest, edges, faces)
        result.mesh.can_collide = ipctk.make_vertex_patches_filter(system.contact_surface_groups)
    result.stiffness = sparse.csc_matrix((3 * len(result.rest),) * 2)
    result.barrier, result.friction_potential, result.friction = (
        system.barrier,
        system.friction_potential,
        system.friction,
    )
    result.dhat = system.dhat
    return result


def manufactured_patch():
    body = SkfemElasticReference(
        np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, -0.01]]), np.array([[0, 2, 1, 3]]), 1e10, 0.3, 2000
    )
    front = np.array([[-0.02, -0.02, 0.001], [-0.02, 0.05, 0.001], [0.05, -0.02, 0.001]])
    far = front + [0, 0, 0.1]
    system = RigidFingerContactSystem(
        body,
        [front, far],
        [np.array([[0, 1, 2]])] * 2,
        [[0, 0, 1], [0, 0, -1]],
        [0.015, 0.015],
        [[0, 0, 0.002], [0, 0, 0.102]],
        0.002,
        0.6,
    )
    u = np.zeros(system.stiffness.shape[0])
    previous = u.copy()
    previous[: body.basis.N].reshape(-1, 3)[:, 0] -= 2e-5
    return system, u, previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--levels", type=int, default=0)
    parser.add_argument("--manufactured-patch", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    closed_source(args.reference)
    config = json.loads((args.reference / "results/configuration.json").read_text())
    closure = json.loads((Path(config["closure_run"]) / "results/configuration.json").read_text())
    system, _ = build_from_capture(Path(closure["capture"]), closure["dhat"])
    with np.load(args.reference / f"results/step{config['steps']:04d}.npz") as saved:
        u = saved["displacement"].copy()
    with np.load(args.reference / f"results/step{config['steps'] - 1:04d}.npz") as saved:
        previous = saved["displacement"].copy()
    if args.manufactured_patch:
        system, u, previous = manufactured_patch()
    system.friction_potential = ipctk.FrictionPotential(config["dt"] * 4e-6)
    system.q_target[:] = 0.018
    external = system.gravity_force([0, 0, -9.81])
    original = system.evaluate_updated_friction(u, previous, external, True)
    refine_contact(system, args.levels)
    mapped = system.evaluate_updated_friction(u, previous, external, True)
    if args.levels == 0:
        np.testing.assert_allclose(original[0], mapped[0], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(original[1], mapped[1], rtol=1e-11, atol=1e-11)
        difference = original[2] - mapped[2]
        assert abs(difference.data).max(initial=0) < 1e-7
    n = system.contact_node_map
    normal, current = system.normal_collisions(u)
    raw_normal = system.barrier.gradient(normal, system.mesh, current).reshape(-1, 3)
    negative_rows = np.asarray((n < 0).sum(axis=1)).ravel() > 0
    active_negative_rows = int(np.count_nonzero(negative_rows & (np.linalg.norm(raw_normal, axis=1) > 1e-12)))
    if args.manufactured_patch and args.levels:
        assert active_negative_rows > 0
    mapped_contact, direct_contact = contact_only(system, True), contact_only(system, False)
    a = mapped_contact.evaluate_updated_friction(u, previous, None, True)
    collision_u = (n @ u.reshape(-1, 3)).ravel()
    collision_previous = (n @ previous.reshape(-1, 3)).ravel()
    b = direct_contact.evaluate_updated_friction(collision_u, collision_previous, None, True)
    dof_map = sparse.kron(n, sparse.eye(3), format="csc")
    expected_force = dof_map.T @ b[1]
    expected_tangent = dof_map.T @ b[2] @ dof_map
    tangent_difference = a[2] - expected_tangent
    force_relative = np.linalg.norm(a[1] - expected_force) / max(np.linalg.norm(expected_force), 1e-12)
    tangent_relative = sparse.linalg.norm(tangent_difference) / sparse.linalg.norm(expected_tangent)
    assert force_relative < 1e-8 and tangent_relative < 1e-8
    direction = np.random.default_rng(7426).normal(size=len(u))
    direction /= np.linalg.norm(direction)
    tangent = a[2] @ direction
    fd = []
    for h in [1e-8, 3e-9, 1e-9]:
        gp = mapped_contact.evaluate_updated_friction(u + h * direction, previous, None, False)[1]
        gm = mapped_contact.evaluate_updated_friction(u - h * direction, previous, None, False)[1]
        fd.append(
            {
                "step_m": h,
                "relative_current_friction_tangent_error": float(
                    np.linalg.norm((gp - gm) / (2 * h) - tangent) / np.linalg.norm(tangent)
                ),
            }
        )
    assert fd[-1]["relative_current_friction_tangent_error"] < 1e-4
    # Independently prescribed quadratic trace: P2 interpolation must reproduce
    # the physical field exactly before contact or bulk error is interpreted.
    x, y, z = system.rest.T
    field = np.column_stack((x * y, y * z, z * x))
    points = n @ system.rest
    body_nodes = system.contact_surface_groups == 0
    expected = np.column_stack((points[:, 0] * points[:, 1], points[:, 1] * points[:, 2], points[:, 2] * points[:, 0]))
    trace_error = float(abs((n @ field - expected)[body_nodes]).max())
    assert trace_error < 1e-15
    force = mapped[4].reshape(-1, 3)
    sparse.save_npz(args.output / "contact_node_map.npz", n)
    np.savez_compressed(
        args.output / "contact_geometry.npz",
        rest=points,
        faces=system.contact_surface_faces,
        groups=system.contact_surface_groups,
    )
    output = {
        "scope": "Manufactured planar P2/current-friction interpolation patch" if args.manufactured_patch else __doc__,
        "levels": args.levels,
        "manufactured_patch": args.manufactured_patch,
        "active_negative_interpolation_rows": active_negative_rows,
        "bulk_dofs": len(u),
        "contact_vertices": len(points),
        "contact_triangles": len(system.contact_surface_faces),
        "quadratic_trace_max_error_m2": trace_error,
        "active_force_difference_N": float(np.linalg.norm(original[1] - mapped[1])),
        "mapped_direct_contact_force_relative_difference": float(force_relative),
        "mapped_direct_current_friction_tangent_relative_difference": float(tangent_relative),
        "current_friction_finite_differences": fd,
        "contact_total_force_N": force.sum(axis=0).tolist(),
        "contact_total_moment_Nm": np.cross(system.rest + u.reshape(-1, 3), force).sum(axis=0).tolist(),
        "goal_completion": False,
        "limitations": "Level0 active-equation agreement and P2 polynomial reproduction qualify interpolation only. Refined contact at saved old states is not a newly solved equilibrium or a physical contact tail.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
