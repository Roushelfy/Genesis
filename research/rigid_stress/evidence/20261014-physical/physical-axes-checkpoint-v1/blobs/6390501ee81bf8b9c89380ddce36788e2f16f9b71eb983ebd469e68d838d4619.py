"""Independent full3D body and compliant contact shoes with force-controlled rigid backplates."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import ipctk
import numpy as np
from scipy import sparse
from skfem import MeshTet

from .contact_reference import IndependentContactSystem
from .finite_traction_reference import triangle_closest_points
from .skfem_reference import SkfemElasticReference


def compliant_shoe(geometry, side, resolution, gap, thickness, lift, young):
    half_y, half_z = (0.003, 0.002) if geometry == "u" else (0.003, 0.003)
    y0 = 0.018 if geometry == "u" else 0
    tensor = MeshTet.init_tensor(
        np.linspace(0, 1, resolution),
        np.linspace(-half_y, half_y, resolution),
        np.linspace(-half_z, half_z, resolution),
    )
    depth, y, z = tensor.p
    if geometry == "u":
        surface = side * (0.008 - np.sqrt(0.003**2 - z**2))
        drive = side
    elif geometry == "egg":
        scale = 1 - 0.18 * z / 0.030
        surface = side * 0.022 * scale * np.sqrt(1 - (y / (0.022 * scale)) ** 2 - (z / 0.030) ** 2)
        drive = -side
    else:
        raise ValueError("Coupled pilot supports the full egg and rounded U")
    edge_lift = lift * ((y / half_y) ** 2 + (z / half_z) ** 2) ** 2
    vertices = np.column_stack((surface - drive * (gap + edge_lift + thickness * depth), y + y0, z))
    tetrahedra = tensor.t.T.copy()
    determinant = np.linalg.det(vertices[tetrahedra[:, 1:]] - vertices[tetrahedra[:, :1]])
    flipped = determinant < 0
    tetrahedra[flipped, 2:4] = tetrahedra[flipped, 3:1:-1]
    reference = SkfemElasticReference(vertices, tetrahedra, young, 0, 1000)
    # The affine depth coordinate reconstructs exactly under P2 nodal interpolation.
    back_vertices = tensor.p[0] == 1
    edges = reference.mesh.edges.T
    back_nodes = np.concatenate(
        (
            reference.scalar_basis.dofs.nodal_dofs[0, back_vertices],
            reference.scalar_basis.dofs.edge_dofs[0, back_vertices[edges].all(axis=1)],
        )
    )
    return reference, back_nodes, drive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--geometry", choices=("u", "egg"), required=True)
    parser.add_argument("--pad-resolution", type=int, default=3)
    parser.add_argument("--pad-young", type=float, default=1e6)
    parser.add_argument("--force", type=float, default=0.05)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--initial-gap", type=float, default=0.001)
    parser.add_argument("--pad-thickness", type=float, default=0.001)
    parser.add_argument("--edge-lift", type=float, default=1e-4)
    parser.add_argument("--force-steps", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    left, left_back, left_drive = compliant_shoe(
        args.geometry, -1, args.pad_resolution, args.initial_gap, args.pad_thickness, args.edge_lift, args.pad_young
    )
    right, right_back, right_drive = compliant_shoe(
        args.geometry, 1, args.pad_resolution, args.initial_gap, args.pad_thickness, args.edge_lift, args.pad_young
    )
    system = IndependentContactSystem(
        (("body", body), ("left_shoe", left), ("right_shoe", right)), args.dhat, 1e6, args.friction, 1e-8
    )
    body_count = body.basis.N
    # These constraints initialize the contact state and are removed in the force-controlled stage.
    body_hold = np.flatnonzero(np.isin(system.parts[0].permutation, body.gauge_dofs))
    backs = [3 * (part.first_node + back) for part, back in zip(system.parts[1:], (left_back, right_back))]
    initialization_fixed = np.concatenate((body_hold, *(np.concatenate((back, back + 1, back + 2)) for back in backs)))
    values = np.zeros(len(initialization_fixed))
    displacement = np.zeros(system.stiffness.shape[0])
    triangles = body.mesh.p.T[body.mesh.facets[:, body.mesh.boundary_facets()].T]
    preload_travel = []
    centre_y = 0.018 if args.geometry == "u" else 0
    for side, part, drive in zip((-1, 1), system.parts[1:], (left_drive, right_drive)):
        surface_x = side * (0.005 if args.geometry == "u" else 0.022)
        centre = np.array([surface_x, centre_y, 0])
        closest, distances = triangle_closest_points(triangles, centre)
        witness = closest[np.argmin(distances)]
        front_x = surface_x - drive * args.initial_gap
        preload_travel.append(drive * (witness[0] - front_x) + 3e-6)
        np.savez_compressed(
            args.output / f"{part.name}_mesh.npz", vertices=part.reference.mesh.p.T, tetrahedra=part.reference.mesh.t.T
        )
    for back, drive, travel in zip(backs, (left_drive, right_drive), preload_travel):
        values[np.isin(initialization_fixed, back)] = drive * travel
    initial_positions = system.mesh.vertices(system.rest)
    intersects = bool(ipctk.has_intersections(system.mesh, initial_positions))
    (args.output / "initial_geometry_audit.json").write_text(
        json.dumps(
            {
                "initial_intersections": intersects,
                "initial_gap_m": args.initial_gap,
            },
            indent=2,
        )
        + "\n"
    )
    if intersects:
        raise ValueError("Reference shoes intersect the material boundary before initialization")
    rigid_target = np.zeros((len(system.rest), 3))
    for part, drive, travel in zip(system.parts[1:], (left_drive, right_drive), preload_travel):
        count = len(part.reference.coordinates)
        rigid_target[part.first_node : part.first_node + count, 0] = drive * travel
    collision_free_fraction = ipctk.compute_collision_free_stepsize(
        system.mesh, initial_positions, system.mesh.vertices(system.rest + rigid_target)
    )
    approach_fraction = 0.9 * collision_free_fraction
    displacement, _, _, _, _ = system.solve_supported(
        initialization_fixed,
        approach_fraction * values,
        12,
        phase="collision_free_approach",
        failure_path=args.output / "approach_failure.npz",
    )
    init_steps = int(np.ceil((1 - approach_fraction) * max(preload_travel) / (args.dhat / 4)))
    displacement, gradient, _, _, _ = system.solve_supported(
        initialization_fixed,
        values,
        init_steps,
        initial_displacement=displacement,
        phase="contact_initialization",
        failure_path=args.output / "initialization_failure.npz",
    )
    guides = np.concatenate([np.concatenate((back + 1, back + 2)) for back in backs])
    controlled = np.concatenate(backs)
    independent = np.setdiff1d(np.arange(len(displacement)), np.concatenate((guides, controlled)))
    rows = np.concatenate((independent, *backs))
    columns = np.concatenate(
        (
            np.arange(len(independent)),
            *[np.full(len(back), len(independent) + index) for index, back in enumerate(backs)],
        )
    )
    coordinate_map = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, columns)), shape=(len(displacement), len(independent) + 2)
    )
    sparse.save_npz(args.output / "coordinate_map.npz", coordinate_map)
    target_force = np.zeros_like(displacement)
    starting_force = np.zeros_like(displacement)
    for back, drive in zip(backs, (left_drive, right_drive)):
        target_force[back] = drive * args.force / len(back)
        starting_force[back] = gradient[back].sum() / len(back)
    for part in system.parts:
        acceleration = np.zeros(part.reference.basis.N)
        acceleration[part.reference.component_dofs] = [0, 0, -9.81]
        gravity = part.reference.mass @ acceleration
        section = slice(part.first_node * 3, part.first_node * 3 + part.reference.basis.N)
        target_force[section] += gravity[part.permutation]
    displacement, gradient, contact_force, energy, records = system.solve_supported(
        guides,
        np.zeros(len(guides)),
        args.force_steps,
        initial_displacement=displacement,
        coordinate_map=coordinate_map,
        applied_force=target_force,
        initial_force=starting_force,
        phase="free_body_force_control",
        failure_path=args.output / "force_failure.npz",
    )
    reduced_residual = np.linalg.norm(coordinate_map.T @ gradient)
    body_residual = np.linalg.norm(gradient[:body_count])
    assert reduced_residual <= 1e-9
    assert body_residual <= 1e-9
    vector = np.empty(body_count)
    vector[system.parts[0].permutation] = displacement[:body_count]
    tensor, vm = body.corner_stress(vector)
    gradients = body.corner_basis.interpolate(vector).grad.transpose(2, 3, 0, 1)
    elastic = body.align_rigid_gauge(vector)
    peak_element, peak_corner = np.unravel_index(np.argmax(vm), vm.shape)
    result = {
        "scope": "Independent full3D deformable body plus compliant shoes, free body and force-controlled backplates",
        "geometry": args.geometry,
        "mesh_sha256": hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
        "material": {
            "body_young_Pa": 1e10,
            "body_poisson": 0.3,
            "body_density_kg_m3": 2000,
            "pad_young_Pa": args.pad_young,
            "pad_poisson": 0,
            "pad_density_kg_m3": 1000,
        },
        "actuator_force_per_shoe_N": args.force,
        "friction": args.friction,
        "dhat_m": args.dhat,
        "pad_thickness_m": args.pad_thickness,
        "edge_lift_m": args.edge_lift,
        "complete_constrained_residual_N": float(reduced_residual),
        "complete_body_residual_N": float(body_residual),
        "raw_body_peak_Pa": float(vm.max()),
        "raw_body_peak_position_m": body.corner_positions[peak_element, peak_corner].tolist(),
        "maximum_body_displacement_gradient": float(np.linalg.norm(gradients, axis=(-2, -1)).max()),
        "minimum_body_deformation_jacobian": float(np.linalg.det(gradients + np.eye(3)).min()),
        "maximum_body_elastic_displacement_m": float(np.linalg.norm(elastic[body.component_dofs], axis=1).max()),
        "total_potential_J": float(energy),
        "accepted_steps": [asdict(row) for row in system.accepted_steps],
        "iterations": [asdict(row) for row in records],
        "protocol": "Temporary initialization supports removed; final contact forces and jaw translations solved from force actuation and gravity",
        "limitations": "Pilot configuration; mesh/contact convergence, actual Franka match and native replay must still be verified",
    }
    np.savez_compressed(
        args.output / "fields.npz",
        rest_positions=system.rest,
        surface_triangles=system.faces,
        displacement=displacement,
        contact_force_N=contact_force,
        total_gradient_N=gradient,
        applied_force_N=target_force,
        body_tensor_Pa=tensor,
        body_vm_Pa=vm,
        body_component_permutation=system.parts[0].permutation,
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "accepted_steps"}), flush=True)


if __name__ == "__main__":
    main()
