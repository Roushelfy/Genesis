"""Audit per-stencil normal-force scaling and precise relative contact increments."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np

from .wide_spatial_franka import WideSpatialInertia, restore_static


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    static, config = restore_static(args.run)
    system = WideSpatialInertia(static, config["dt"])
    with np.load(args.run / "results/failure.npz") as saved:
        displacement = np.asarray(saved["displacement"], dtype=np.longdouble)
        previous = np.asarray(saved["previous_displacement"], dtype=np.longdouble)
    normal, current = system.normal_collisions(displacement)
    precise = system.contact_node_map @ (np.asarray(system.rest, dtype=np.longdouble) + displacement.reshape(-1, 3))
    delta = np.asarray(system.contact_node_map @ (displacement - previous).reshape(-1, 3), dtype=float)
    old_delta = system.mesh.displace_vertices((displacement - previous).reshape(-1, 3) - system.rest)
    tangential = ipctk.TangentialCollisions()
    tangential.build(system.mesh, current, normal, system.barrier, system.friction)
    rows = []
    assembled = np.zeros_like(current)
    precise_assembled = np.zeros_like(current)
    for index in range(len(normal)):
        collision = normal[index]
        ids = np.asarray(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        groups = system.contact_surface_groups[ids]
        dofs = collision.dof(current, system.mesh.edges, system.mesh.faces)
        np.testing.assert_array_equal(np.asarray(dofs).reshape(-1, 3), current[ids])
        force = -system.barrier.gradient(collision, dofs).reshape(-1, 3)
        origin = precise[ids[0]].copy()
        centered = np.asarray(precise[ids] - origin, dtype=float).reshape(np.asarray(dofs).shape)
        local_force = -system.barrier.gradient(collision, centered).reshape(-1, 3)
        np.add.at(assembled, ids, force)
        np.add.at(precise_assembled, ids, local_force)
        row = {
            "index": index,
            "kind": type(collision).__name__,
            "ids": ids.tolist(),
            "groups": groups.tolist(),
            "weight": float(collision.weight),
            "body_normal_force_N": float(np.linalg.norm(force[groups == 0].sum(axis=0))),
            "centered_body_normal_force_N": float(np.linalg.norm(local_force[groups == 0].sum(axis=0))),
            "local_global_force_difference_N": float(np.linalg.norm(force - local_force)),
        }
        rows.append(row)
        del collision
    global_force = -system.barrier.gradient(normal, system.mesh, current).reshape(-1, 3)
    np.testing.assert_allclose(assembled, global_force, rtol=1e-12, atol=1e-14)
    shear_rows = []
    for index in range(len(tangential)):
        collision = tangential[index]
        ids = np.asarray(collision.vertex_ids(system.mesh.edges, system.mesh.faces))
        ids = ids[ids >= 0]
        matches = [row for row in rows if row["ids"] == ids.tolist()]
        assert len(matches) == 1
        base = matches[0]
        magnitude = float(collision.normal_force_magnitude)
        shear_rows.append(
            {
                "index": index,
                "kind": type(collision).__name__,
                "ids": ids.tolist(),
                "weight": float(collision.weight),
                "normal_force_magnitude": magnitude,
                "weighted_normal_force": magnitude * float(collision.weight),
                "body_normal_force_N": base["body_normal_force_N"],
                "precise_body_normal_force_N": base["centered_body_normal_force_N"],
            }
        )
        del collision
    old_shear = system.friction_potential.gradient(tangential, system.mesh, old_delta)
    direct_shear = system.friction_potential.gradient(tangential, system.mesh, delta)
    summary = {
        "scope": __doc__,
        "run": str(args.run),
        "normal": rows,
        "tangential": shear_rows,
        "assembled_normal_gradient_max_error_N": float(abs(assembled - global_force).max()),
        "global_vs_centered_contact_gradient_L2_N": float(np.linalg.norm(assembled - precise_assembled)),
        "relative_increment_max_arithmetic_difference_m": float(abs(old_delta - delta).max()),
        "shear_force_relative_increment_difference_N": float(np.linalg.norm(old_shear - direct_shear)),
        "goal_completion": False,
        "limitations": "Arithmetic and per-stencil scaling diagnostics only; no physical parameter, force tolerance or accepted field is changed. A precise force prototype still needs derivative, original-state and complete residual qualification.",
    }
    np.savez_compressed(
        args.output / "contact_fields.npz",
        global_normal_force_N=global_force,
        centered_normal_force_N=precise_assembled,
        direct_relative_increment_m=delta,
        original_relative_increment_m=old_delta,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ["normal", "tangential"]}))


if __name__ == "__main__":
    main()
