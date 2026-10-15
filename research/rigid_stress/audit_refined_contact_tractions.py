"""Audit collision-surface traction quadrature and islands without nodal-load smoothing."""

import argparse
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .franka_contact_reference import build_from_capture
from .frozen_franka_contact import freeze_body_contact
from .localized_franka_mesh import refine_local_body
from .refine_franka_reference import refine_body
from .refined_franka_contact import collision_vertices, refine_contact
from .summarize_fine_references import closed_source


def vertex_areas(positions, faces):
    triangles = positions[faces]
    area = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    return np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=len(positions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--axis", choices=("contact_refinement", "body_refinement", "local_levels"), default="contact_refinement"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    frozen = None
    for run in args.run:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "results/configuration.json").read_text())
        physical = {k: v for k, v in config.items() if k not in ("output", args.axis)}
        if frozen is None:
            frozen = physical
        assert physical == frozen
        closure = json.loads((Path(config["closure_run"]) / "results/configuration.json").read_text())
        system, _ = build_from_capture(Path(closure["capture"]), closure["dhat"])
        seed = system
        if "local_levels" in config:
            system = refine_local_body(
                system, config["local_levels"], Path(config["local_contact_run"]), config["local_radius"]
            )
        else:
            system = refine_body(system, config["body_refinement"])
        if config.get("freeze_contact", False):
            freeze_body_contact(seed, system)
            # Represent the original collision surface by an exact selection map.
            refine_contact(system, 0)
        else:
            refine_contact(system, config.get("contact_refinement", 0))
        system.friction_potential = ipctk.FrictionPotential(config["dt"] * 4e-6)
        with np.load(run / f"results/step{config['steps']:04d}.npz") as saved:
            u = saved["displacement"].copy()
            saved_force = saved["contact_force_N"].copy()
        with np.load(run / f"results/step{config['steps'] - 1:04d}.npz") as saved:
            previous = saved["displacement"].copy()
        normal, current = system.normal_collisions(u)
        tangential = ipctk.TangentialCollisions()
        tangential.build(system.mesh, current, normal, system.barrier, system.friction)
        velocity = collision_vertices(system, (u - previous).reshape(-1, 3))
        normal_gradient = system.barrier.gradient(normal, system.mesh, current)
        shear_gradient = system.friction_potential.gradient(tangential, system.mesh, velocity)
        force_n = -normal_gradient.reshape(-1, 3)
        force_t = -shear_gradient.reshape(-1, 3)
        pullback = -system.mesh.to_full_dof(normal_gradient + shear_gradient)
        np.testing.assert_allclose(pullback, saved_force, rtol=1e-10, atol=1e-11)
        area = vertex_areas(current, system.contact_surface_faces)
        rest_area = vertex_areas(system.contact_surface_rest, system.contact_surface_faces)
        assert np.all(area > 0)
        p = np.linalg.norm(force_n, axis=1) / area
        t = np.linalg.norm(force_t, axis=1) / area
        active = (system.contact_surface_groups == 0) & (np.linalg.norm(force_n, axis=1) > 1e-8)
        indices = np.flatnonzero(active)
        assert len(indices) > 0
        edges = np.unique(np.sort(system.contact_surface_faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2)), axis=0)
        edges = edges[active[edges].all(axis=1)]
        rows = np.r_[edges[:, 0], edges[:, 1]]
        cols = np.r_[edges[:, 1], edges[:, 0]]
        graph = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(current),) * 2).tocsr()[indices][
            :, indices
        ]
        count, labels = connected_components(graph, directed=False)
        islands = []
        for label in range(count):
            selected = indices[labels == label]
            force = (force_n + force_t)[selected]
            islands.append(
                {
                    "nodes": len(selected),
                    "current_lumped_area_m2": float(area[selected].sum()),
                    "force_N": force.sum(axis=0).tolist(),
                    "moment_about_shared_origin_Nm": np.cross(current[selected], force).sum(axis=0).tolist(),
                    "normal_resultant_N": force_n[selected].sum(axis=0).tolist(),
                    "shear_resultant_N": force_t[selected].sum(axis=0).tolist(),
                }
            )
        body = system.contact_surface_groups == 0
        record = {
            "run": str(run),
            "contact_refinement": config.get("contact_refinement", 0),
            "body_refinement": config["body_refinement"],
            "local_levels": config.get("local_levels", 0),
            "declared_level": config.get(args.axis, 0),
            "body_tetrahedra": system.body.mesh.nelements,
            "bulk_dofs": len(u),
            "surface_vertices": len(current),
            "surface_triangles": len(system.contact_surface_faces),
            "normal_stencils": len(normal),
            "active_body_nodes": len(indices),
            "islands": islands,
            "active_lumped_current_area_m2": float(area[active].sum()),
            "active_lumped_rest_area_m2": float(rest_area[active].sum()),
            "normal_density_peak_Pa": float(p[body].max()),
            "shear_density_peak_Pa": float(t[body].max()),
            "normal_density_l2_Pa_m": float(np.sqrt(np.sum(area[body] * p[body] ** 2))),
            "shear_density_l2_Pa_m": float(np.sqrt(np.sum(area[body] * t[body] ** 2))),
            "total_contact_force_N": (force_n + force_t).sum(axis=0).tolist(),
            "total_contact_moment_Nm": np.cross(current, force_n + force_t).sum(axis=0).tolist(),
            "saved_force_pullback_max_error_N": float(abs(pullback - saved_force).max()),
        }
        np.savez_compressed(
            args.output / f"case{len(cases):02d}-surface.npz",
            rest_positions_m=system.contact_surface_rest,
            current_positions_m=current,
            faces=system.contact_surface_faces,
            groups=system.contact_surface_groups,
            normal_force_N=force_n,
            shear_force_N=force_t,
            current_vertex_area_m2=area,
            normal_density_Pa=p,
            shear_density_Pa=t,
            active_body_nodes=indices,
            island_labels=labels,
        )
        cases.append(record)
    cases.sort(key=lambda row: row["declared_level"])
    output = {
        "scope": __doc__,
        "cases": cases,
        "goal_completion": False,
        "limitations": "Densities divide unsmoothed collision-vertex force resultants by current barycentric lumped areas. Support area is a quadrature diagnostic, not an independently resolved physical contact boundary. All source normal/friction forces and island wrenches remain visible. Total contact moment can retain finite-gap friction torque; its regularization error is separate from pullback conservation. No contact/stress tail or full-arm matching is inferred from aggregate force stability.",
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {"cases": [{k: v for k, v in c.items() if k != "islands"} for c in cases], "goal_completion": False}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
