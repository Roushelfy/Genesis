"""Qualify IPC contact interpolation using explicit full-coordinate chain rules."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    full = np.array(
        [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, -0.01], [0, 0, 0.001], [0, 0.01, 0.001], [0.01, 0, 0.001]]
    )
    node_map = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0.5, 0.5, 0, 0, 0, 0, 0],
                [0, 0.5, 0.5, 0, 0, 0, 0],
                [0.5, 0, 0.5, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
    )
    mapping = sparse.kron(node_map, sparse.eye(3), format="csc")
    rest = node_map @ full
    faces = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5], [6, 7, 8]])
    edges = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
    print(json.dumps({"constructing_displacement_map_shape": mapping.shape}), flush=True)
    mapped = ipctk.CollisionMesh(rest, edges, faces, node_map.tocsc())
    direct = ipctk.CollisionMesh(rest, edges, faces)
    for mesh in (mapped, direct):
        mesh.can_collide = ipctk.make_vertex_patches_filter(np.r_[np.zeros(6, dtype=int), np.ones(3, dtype=int)])
    np.testing.assert_allclose(mapped.vertices(node_map @ full), rest, rtol=0, atol=1e-15)
    direction = np.random.default_rng(23106).normal(size=full.shape)
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(
        mapped.vertices(node_map @ (full + 0.00001 * direction)),
        node_map @ (full + 0.00001 * direction),
        rtol=0,
        atol=1e-15,
    )
    potential = ipctk.BarrierPotential(0.002, 1e6, use_physical_barrier=True)

    def evaluate(mesh, positions):
        collisions = ipctk.NormalCollisions()
        collisions.use_area_weighting = True
        collisions.collision_set_type = ipctk.NormalCollisions.IMPROVED_MAX_APPROX
        current = mesh.vertices(node_map @ positions if mesh is mapped else positions)
        collisions.build(mesh, current, 0.002)
        return (
            float(potential(collisions, mesh, current)),
            mesh.to_full_dof(potential.gradient(collisions, mesh, current)),
            mesh.to_full_dof(potential.hessian(collisions, mesh, current)),
            len(collisions),
        )

    energy, gradient, hessian, count = evaluate(mapped, full)
    direct_energy, direct_gradient, direct_hessian, _ = evaluate(direct, rest)
    np.testing.assert_allclose(energy, direct_energy, rtol=0, atol=1e-15)
    expected_gradient = mapping.T @ direct_gradient
    expected_hessian = mapping.T @ direct_hessian @ mapping
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-12, atol=1e-13)
    difference = hessian - expected_hessian
    assert abs(difference.data).max(initial=0) < 1e-10
    force = -gradient.reshape(-1, 3)
    direct_force = -direct_gradient.reshape(-1, 3)
    np.testing.assert_allclose(force.sum(axis=0), direct_force.sum(axis=0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.cross(full, force).sum(axis=0), np.cross(rest, direct_force).sum(axis=0), rtol=0, atol=1e-13
    )
    fd_records = []
    for step in [1e-6, 3e-7, 1e-7]:
        plus, gp, _, _ = evaluate(mapped, full + step * direction)
        minus, gm, _, _ = evaluate(mapped, full - step * direction)
        expected_work = float(gradient @ direction.ravel())
        work_error = abs((plus - minus) / (2 * step) - expected_work) / max(abs(expected_work), 1e-12)
        tangent = hessian @ direction.ravel()
        tangent_error = np.linalg.norm((gp - gm) / (2 * step) - tangent) / np.linalg.norm(tangent)
        fd_records.append(
            {"step_m": step, "work_relative_error": float(work_error), "tangent_relative_error": float(tangent_error)}
        )
    assert fd_records[-1]["work_relative_error"] < 1e-5
    assert fd_records[-1]["tangent_relative_error"] < 1e-5
    sparse.save_npz(args.output / "node_map.npz", node_map)
    np.savez_compressed(
        args.output / "fields.npz",
        full_positions=full,
        collision_positions=rest,
        full_gradient=gradient,
        direct_gradient=direct_gradient,
    )
    output = {
        "scope": __doc__,
        "mapping_shape": mapping.shape,
        "normal_stencils": count,
        "max_gradient_chain_difference": float(abs(gradient - expected_gradient).max()),
        "max_hessian_chain_difference": float(abs(difference.data).max(initial=0)),
        "finite_differences": fd_records,
        "goal_completion": False,
        "limitations": "A mapped normal-contact patch validates coordinates, virtual work, tangent and total force/moment pullback. P2 interpolation, current friction derivatives, actual geometry, contact refinement and physical convergence require separate checks.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
