"""Resolve matched original two-jaw controls on declared CAD or fixed-polyhedron egg geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from .audit_franka_reference import rotation
from .franka_contact_reference import RigidFingerContactSystem, build_from_capture
from .franka_rigid_mode_coordinates import coordinate_map
from .precise_stencil_spatial import PreciseStencilSpatial
from .refined_franka_contact import refine_contact
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-run", type=Path, required=True)
    parser.add_argument("--mode", choices=["cad", "polyhedron"], required=True)
    parser.add_argument("--level", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--closure-run", type=Path, required=True)
    parser.add_argument("--seed-reference", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.00125)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--control-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.dt > 0 and args.steps > 0 and args.control_steps > 0
    for run in [args.geometry_run, args.closure_run, args.seed_reference]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    closure = json.loads((args.closure_run / "results/configuration.json").read_text())
    capture = Path(closure["capture"])
    seed, frame = build_from_capture(capture, closure["dhat"])
    geometry_path = args.geometry_run / f"results/{args.mode}-level{args.level}.npz"
    geometry_report = json.loads((args.geometry_run / "results/summary.json").read_text())
    geometric = next(
        r for r in geometry_report["rows"] if r["mode"].lower() == args.mode and r["level"] == args.level
    )
    assert geometric["original_initializer_all_anchors_retained"]
    assert hashlib.sha256(geometry_path.read_bytes()).hexdigest() == geometric["mesh_sha256"]
    with np.load(geometry_path) as saved:
        vertices = saved["vertices"] @ rotation(frame["egg_quaternion_wxyz"]).T
        vertices += np.array(frame["egg_position_m"]) - frame["egg_world_com_m"]
        body = SkfemElasticReference(vertices, saved["tetrahedra"], 1e10, 0.3, 2000)
    np.testing.assert_allclose(body.total_mass, geometric["mass_kg"], rtol=1e-12, atol=1e-14)
    with np.load(args.seed_reference / "results/reference_geometry.npz") as saved:
        fixed, components = saved["fixed_initializer_points_m"].copy(), saved["fixed_initializer_components"].copy()
    distance, nodes = cKDTree(body.coordinates).query(fixed)
    assert distance.max() < 1e-12
    body.gauge_dofs = body.component_dofs[nodes, components]
    assert len(np.unique(body.gauge_dofs)) == 6
    body.free_dofs = np.setdiff1d(np.arange(body.basis.N), body.gauge_dofs)
    finger_vertices, finger_faces, centres = [], [], []
    for index, indices in enumerate(seed.finger_nodes):
        local = indices[:-1]
        selected = seed.faces[seed.groups[seed.faces[:, 0]] == index + 1]
        assert np.all(np.isin(selected, local))
        finger_vertices.append(seed.rest[local])
        finger_faces.append(selected - local[0])
        centres.append(seed.rest[indices[-1]])
    static = RigidFingerContactSystem(
        body, finger_vertices, finger_faces, seed.axes, seed.masses, centres, seed.dhat, seed.friction
    )
    static.world_coordinate_origin_m = seed.world_coordinate_origin_m.copy()
    static.fixed_initializer_points_m, static.fixed_initializer_components = fixed, components
    static = refine_contact(static, 0)
    assert not ipctk.has_intersections(static.mesh, static.contact_surface_rest)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "local_levels": 0,
        "local_radius": 5e-5,
        "body_refinement": 0,
        "contact_refinement": 0,
        "cold_closure": True,
        "freeze_contact": False,
    }
    (args.output / "configuration.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
    np.savez_compressed(
        args.output / "reference_geometry.npz",
        vertices=body.mesh.p.T,
        tetrahedra=body.mesh.t.T,
        fixed_initializer_points_m=fixed,
        fixed_initializer_components=components,
    )
    np.savez_compressed(
        args.output / "contact_geometry.npz",
        rest_positions=static.contact_surface_rest,
        faces=static.contact_surface_faces,
        groups=static.contact_surface_groups,
    )
    sparse.save_npz(args.output / "contact_node_map.npz", static.contact_node_map)
    u = np.zeros(static.stiffness.shape[0])
    held_map, held = static.coordinate_map(True)
    gravity = static.gravity_force([0, 0, -9.81])
    closure_rows = []
    for step in range(1, args.control_steps + 1):
        static.q_target[:] = 0.04 + (0.018 - 0.04) * step / args.control_steps
        u, gradient, _force, _energy, _iterations = static.solve_updated_friction(
            held,
            np.zeros(len(held)),
            1,
            tolerance=1e-10,
            initial_displacement=u,
            coordinate_map=held_map,
            applied_force=gravity,
            initial_force=gravity,
            phase=f"paired_original_closure{step}",
            failure_path=args.output / "closure_failure.npz",
        )
        residual = float(np.linalg.norm(held_map.T @ gradient))
        assert residual <= 1e-9
        closure_rows.append({"step": step, "target_m": static.q_target.tolist(), "complete_held_residual_N": residual})
        print(json.dumps(closure_rows[-1]), flush=True)
    np.savez_compressed(
        args.output / "closure_state.npz", displacement=u, fixed_points_m=fixed, fixed_components=components
    )
    provenance_files = [
        args.output / name
        for name in ["closure_state.npz", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]
    ]
    provenance = {
        "scope": "New declared geometry reference, independently solved original held initializer; no copied or resumed physical history.",
        "source_run": str(args.output.parent),
        "initial_state_and_meshes": [
            {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in provenance_files
        ],
    }
    (args.output / "precision_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    system = PreciseStencilSpatial(static, args.dt)
    system.q_target[:] = 0.018
    u = np.asarray(u, dtype=np.longdouble)
    velocity = np.zeros_like(u)
    mapping = coordinate_map(system)
    rows = []
    for step in range(1, args.steps + 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + args.dt * old_velocity
        u, gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=gravity,
            initial_force=gravity,
            phase=f"paired_dynamic{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / args.dt
        acceleration = (velocity - old_velocity) / args.dt
        complete, body_residual = (
            float(np.linalg.norm(mapping.T @ gradient)),
            float(np.linalg.norm(gradient[: body.basis.N])),
        )
        assert complete <= 1e-9 and body_residual <= 1e-9
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=force,
            applied_force_N=gravity,
            body_component_permutation=system.parts[0].permutation,
        )
        rows.append(
            {
                "step": step,
                "time_s": step * args.dt,
                "complete_dynamic_residual_N": complete,
                "complete_body_dynamic_residual_N": body_residual,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    result = {
        "scope": __doc__,
        "geometry": geometric,
        "physical_parameters": {
            "young_Pa": body.young,
            "poisson": body.poisson,
            "density_kg_m3": body.density,
            "mass_kg": body.total_mass,
            "friction": static.friction,
            "dhat_m": static.dhat,
        },
        "closure": closure_rows,
        "steps": rows,
        "goal_completion": False,
        "limitations": "Original actual finger geoms, physical held points/components, material/density, friction, actuation, timestep and force budgets are preserved. CAD and polyhedral models have their explicitly measured different represented volume/mass. Five states are a short diagnostic, not a complete application/history or converged contact reference. Every saved state still requires independent force/global-field auditing. Geometry, bulk, contact integration and regularization axes, models/feedback and verified accuracy/cost remain open.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
