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
from .neohookean_reference import NeoHookeanAllContactSystem, NeoHookeanPadContactSystem
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
    parser.add_argument("--pad-material", choices=("linear", "neo_hookean"), default="linear")
    parser.add_argument("--body-material", choices=("linear", "neo_hookean"), default="linear")
    parser.add_argument("--force", type=float, default=0.05)
    parser.add_argument("--dhat", type=float, default=2.5e-6)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--initial-gap", type=float, default=0.001)
    parser.add_argument("--pad-thickness", type=float, default=0.001)
    parser.add_argument("--edge-lift", type=float, default=1e-4)
    parser.add_argument("--force-steps", type=int, default=64)
    parser.add_argument("--gravity", nargs=3, type=float, default=(0, 0, -9.81))
    parser.add_argument("--body-rigid-coordinates", action="store_true")
    parser.add_argument("--updated-friction", action="store_true")
    parser.add_argument("--friction-least-squares", action="store_true")
    parser.add_argument("--reuse-empty-factor", action="store_true")
    parser.add_argument("--centre-gauge", action="store_true")
    parser.add_argument(
        "--preloaded-state",
        type=Path,
        help="Use a retained force-step's previous displacement as an explicit initial pose",
    )
    parser.add_argument(
        "--resume-failed-step",
        action="store_true",
        help="Resume both the failed iterate and its unchanged friction-step history",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_failed_step and (args.preloaded_state is None or not args.updated_friction):
        parser.error("--resume-failed-step requires --preloaded-state and --updated-friction")
    if args.friction_least_squares and not args.updated_friction:
        parser.error("--friction-least-squares requires --updated-friction")
    if args.centre_gauge and not args.body_rigid_coordinates:
        parser.error("--centre-gauge requires --body-rigid-coordinates")
    if args.body_material == "neo_hookean" and args.pad_material != "neo_hookean":
        parser.error("--body-material neo_hookean currently requires --pad-material neo_hookean")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "configuration.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    with np.load(args.mesh) as saved:
        body = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
    left, left_back, left_drive = compliant_shoe(
        args.geometry, -1, args.pad_resolution, args.initial_gap, args.pad_thickness, args.edge_lift, args.pad_young
    )
    right, right_back, right_drive = compliant_shoe(
        args.geometry, 1, args.pad_resolution, args.initial_gap, args.pad_thickness, args.edge_lift, args.pad_young
    )
    system_type = NeoHookeanPadContactSystem if args.pad_material == "neo_hookean" else IndependentContactSystem
    if args.body_material == "neo_hookean":
        system_type = NeoHookeanAllContactSystem
    system = system_type(
        (("body", body), ("left_shoe", left), ("right_shoe", right)),
        args.dhat,
        1e6,
        args.friction,
        1e-8,
        reuse_empty_factor=args.reuse_empty_factor,
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
    if args.preloaded_state is None:
        displacement, _, _, _, _ = system.solve_supported(
            initialization_fixed,
            approach_fraction * values,
            12,
            initial_displacement=(approach_fraction * rigid_target).reshape(-1),
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
    else:
        with np.load(args.preloaded_state) as saved:
            np.testing.assert_array_equal(saved["rest_positions"], system.rest)
            np.testing.assert_array_equal(saved["surface_triangles"], system.faces)
            displacement = saved["displacement" if args.resume_failed_step else "previous_displacement"].copy()
            resumed_previous = saved["previous_displacement"].copy()
        positions = system.mesh.vertices(system.rest + displacement.reshape(-1, 3))
        if ipctk.has_intersections(system.mesh, positions):
            raise ValueError("The declared preloaded initial pose has mesh intersections")
        gradient = np.zeros_like(displacement)
    guides = np.concatenate([np.concatenate((back + 1, back + 2)) for back in backs])
    controlled = np.concatenate(backs)
    independent = np.setdiff1d(np.arange(len(displacement)), np.concatenate((guides, controlled)))
    if args.body_rigid_coordinates:
        # Six rigid coordinates restore the omitted elastic coordinate entries.
        # This is an invertible change of basis, with every body motion still free.
        independent = independent[~np.isin(independent, body_hold)]
        modes = body.rigid_modes[system.parts[0].permutation].copy()
        radius = np.linalg.norm(body.coordinates - body.com, axis=1).max()
        modes[:, 3:] /= radius
        assert np.linalg.matrix_rank(modes[body_hold]) == 6
        rigid_rows, rigid_columns = np.nonzero(modes)
        extra_coordinates = 6
    else:
        extra_coordinates = 0
    rows = np.concatenate((independent, *backs))
    columns = np.concatenate(
        (
            np.arange(len(independent)),
            *[np.full(len(back), len(independent) + extra_coordinates + index) for index, back in enumerate(backs)],
        )
    )
    values = np.ones(len(rows))
    if args.body_rigid_coordinates:
        rows = np.concatenate((rows, rigid_rows))
        columns = np.concatenate((columns, len(independent) + rigid_columns))
        values = np.concatenate((values, modes[rigid_rows, rigid_columns]))
    coordinate_map = sparse.csr_matrix(
        (values, (rows, columns)), shape=(len(displacement), len(independent) + extra_coordinates + 2)
    )
    sparse.save_npz(args.output / "coordinate_map.npz", coordinate_map)
    target_force = np.zeros_like(displacement)
    starting_force = np.zeros_like(displacement)
    for back, drive in zip(backs, (left_drive, right_drive)):
        target_force[back] = drive * args.force / len(back)
        starting_force[back] = gradient[back].sum() / len(back)
    # Stabilize both contacts before releasing the initialization fixture and applying gravity.
    held_columns = np.concatenate(
        (
            np.flatnonzero(~np.isin(independent, body_hold)),
            [len(independent) + extra_coordinates, len(independent) + extra_coordinates + 1],
        )
    )
    held_map = coordinate_map[:, held_columns]
    held_guides = np.concatenate((guides, body_hold))
    normal_actuation = target_force.copy()
    if args.preloaded_state is None:
        displacement, _, _, _, _ = system.solve_supported(
            held_guides,
            np.zeros(len(held_guides)),
            args.force_steps,
            initial_displacement=displacement,
            coordinate_map=held_map,
            applied_force=normal_actuation,
            initial_force=starting_force,
            phase="held_body_force_control",
            failure_path=args.output / "held_force_failure.npz",
        )
    starting_force = normal_actuation
    for part in system.parts:
        acceleration = np.zeros(part.reference.basis.N)
        acceleration[part.reference.component_dofs] = args.gravity
        gravity = part.reference.mass @ acceleration
        section = slice(part.first_node * 3, part.first_node * 3 + part.reference.basis.N)
        target_force[section] += gravity[part.permutation]
    original_coordinate_map = coordinate_map
    centre_shift = 0.0
    if args.centre_gauge:
        if abs(target_force.reshape(-1, 3)[:, 0].sum()) > 1e-13:
            raise ValueError("Common-x gauge requires zero total applied x force")
        centre_shift = (displacement[backs[0]].mean() + displacement[backs[1]].mean()) / 2
        displacement[0::3] -= centre_shift
        if args.resume_failed_step:
            resumed_previous[0::3] -= centre_shift
        count = coordinate_map.shape[1]
        transform = sparse.coo_matrix(
            (
                np.r_[np.ones(count - 2), -1, 1],
                (np.r_[np.arange(count - 2), count - 2, count - 1], np.r_[np.arange(count - 2), count - 2, count - 2]),
            ),
            shape=(count, count - 1),
        ).tocsr()
        coordinate_map = (coordinate_map @ transform).tocsr()
        sparse.save_npz(args.output / "original_coordinate_map.npz", original_coordinate_map)
        sparse.save_npz(args.output / "coordinate_map.npz", coordinate_map)
    force_solver = system.solve_updated_friction if args.updated_friction else system.solve_supported
    resumed_arguments = {"initial_previous_displacement": resumed_previous} if args.resume_failed_step else {}
    if args.updated_friction:
        resumed_arguments["least_squares"] = args.friction_least_squares
    displacement, gradient, contact_force, energy, records = force_solver(
        guides,
        np.zeros(len(guides)),
        args.force_steps,
        initial_displacement=displacement,
        coordinate_map=coordinate_map,
        applied_force=target_force,
        initial_force=starting_force,
        phase="free_body_force_control",
        failure_path=args.output / "force_failure.npz",
        **resumed_arguments,
    )
    reduced_residual = np.linalg.norm(original_coordinate_map.T @ gradient)
    body_residual = np.linalg.norm(gradient[:body_count])
    assert reduced_residual <= 1e-9
    assert body_residual <= 1e-9
    vector = np.empty(body_count)
    vector[system.parts[0].permutation] = displacement[:body_count]
    tensor, vm = body.corner_stress(vector)
    nonlinear_upper = None
    if args.body_material == "neo_hookean":
        tensor, vm, nonlinear_upper = system.body_bulk.stress_bounds(vector)
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
            "body_material_model": args.body_material,
            "pad_young_Pa": args.pad_young,
            "pad_poisson": 0,
            "pad_density_kg_m3": 1000,
            "pad_material_model": args.pad_material,
        },
        "actuator_force_per_shoe_N": args.force,
        "friction": args.friction,
        "gravity_m_per_s2": args.gravity,
        "body_rigid_coordinates": args.body_rigid_coordinates,
        "centre_gauge": args.centre_gauge,
        "initial_common_x_shift_m": float(centre_shift),
        "supported_factorizations": system.supported_factorizations,
        "empty_factor_reuses": system.empty_factor_reuses,
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
        "protocol": "Temporary support retained until both force-controlled contacts are established, then removed before gravity; final jaw translations and contact forces are solved outputs",
        "friction_solve": "Updated-force Jacobian and constrained-residual Newton"
        if args.updated_friction
        else "Lagged friction fixed point",
        "friction_newton_least_squares": args.friction_least_squares,
        "preloaded_initial_state": None
        if args.preloaded_state is None
        else {
            "path": str(args.preloaded_state.resolve()),
            "sha256": hashlib.sha256(args.preloaded_state.read_bytes()).hexdigest(),
            "selected_field": "displacement with previous_displacement history"
            if args.resume_failed_step
            else "previous_displacement",
            "interpretation": "Explicit saved initial pose; no convergence claim for its failed subsequent state",
        },
        "limitations": "Pilot configuration; mesh/contact convergence, actual Franka match and native replay must still be verified",
    }
    if nonlinear_upper is not None:
        lower = float(vm.max())
        upper = float(nonlinear_upper.max())
        result["nonlinear_body_peak_bounds"] = {
            "sampled_corner_lower_Pa": lower,
            "certified_global_upper_Pa": upper,
            "relative_interval_width": (upper - lower) / upper,
            "definition": "Actual unsmoothed Cauchy corner lower bound and Bernstein numerator/determinant whole-volume upper bound; no exact-corner maximum claim",
        }
        result["kinematic_diagnostics_definition"] = (
            "Displacement gradients, symmetric strains and infinitesimal mass-gauge displacements remain linear-coordinate diagnostics; nonlinear body energy and Cauchy stress use the complete finite F. Objective pose/deformation comparisons require a subsequent finite rigid-frame fit."
        )
    if args.pad_material == "neo_hookean":
        diagnostics = []
        for part, bulk in system.pad_bulk:
            count = part.reference.basis.N
            section = slice(part.first_node * 3, part.first_node * 3 + count)
            pad_vector = np.empty(count)
            pad_vector[part.permutation] = displacement[section]
            lower = bulk.determinant_lower_bound(pad_vector)
            assert lower > 0
            diagnostics.append(
                {
                    "part": part.name,
                    "quadrature_order": bulk.quadrature_order,
                    "whole_volume_determinant_lower_bound": lower,
                }
            )
        result["nonlinear_pad_diagnostics"] = diagnostics
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
