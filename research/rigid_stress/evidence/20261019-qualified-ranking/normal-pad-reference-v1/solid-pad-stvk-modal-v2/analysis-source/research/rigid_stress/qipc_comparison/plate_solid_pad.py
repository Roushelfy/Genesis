"""Independent continuous P2 solid reference with free motion and physical pads."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from .artifacts import file_id, runtime_identity, write_json
from .compliant_plate_reference import pad_quadrature
from .condensed_pad_dynamics import CondensedPadDynamics
from .cvxopt_factor import CvxoptCholeskyFactor
from .matrix_free_pad_dynamics import MatrixFreePadDynamics
from .modal_pad_dynamics import ModalPadDynamics
from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum
from .p2_stvk import P2StVK
from .plates import equilibrium, force_history, parser
from .solid_pad_step import advance


class OrderedFactor:
    def __init__(self, operator, order):
        assert len(np.unique(order)) == operator.shape[0] == len(order)
        self.order = order
        self.factor = CvxoptCholeskyFactor(operator[order][:, order].tocsc())

    def solve(self, rhs):
        answer = np.empty_like(rhs)
        answer[self.order] = self.factor.solve(rhs[self.order])
        return answer


def step_checks(row):
    return {
        "complete_equation": row["equation_residual_relative"] <= 1e-7,
        "actual_pressure": row["normal_law_relative_error"] <= 1e-7,
        "actual_friction": row["friction_law_relative_error"] <= 1e-7,
        "point_velocity_prediction": row["point_velocity_prediction_error_m_per_s"] <= 1e-9,
        "complete_boundary": row["minimum_omitted_gap_m"] >= -1e-12,
        "current_moment": row["current_moment_balance_relative_error"] <= 1e-7,
    }


def main():
    cli = parser()
    cli.add_argument("--friction", type=float, choices=(0.0, 0.3), default=0.0)
    cli.add_argument("--gravity-x", type=float, choices=(0.0, 1.0), default=0.0)
    cli.add_argument("--gravity-start", type=float, default=0.2)
    cli.add_argument("--gravity-ramp", type=float, default=0.05)
    cli.add_argument("--candidate-radius", type=float, default=0.003)
    cli.add_argument("--quadrature-order", type=int, default=0)
    cli.add_argument("--contact-solver", choices=("condensed", "matrix-free", "modal"), default="condensed")
    cli.add_argument("--elasticity", choices=("linear", "stvk"), default="linear")
    cli.add_argument("--warm-contact", action="store_true")
    args = cli.parse_args()
    assert not args.empty and not args.shear_distance and args.plate_offset_x == 0
    assert args.record_stride == 1 and (not args.gravity_x or args.friction == 0.3)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    started = perf_counter()
    with np.load(args.mesh) as saved:
        vertices, tets, outer, faces = (saved[k] for k in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    write_json(args.output / "building.json", {"identity": runtime_identity(), "mesh": file_id(args.mesh)})
    model = FreeElasticP2(
        vertices, tets, 1e10, 0.3, 2000, column_faces=faces, through_wall_ordering=True, factorize=False
    )
    reference = model.reference
    nonlinear = P2StVK(reference) if args.elasticity == "stvk" else None
    np.testing.assert_array_equal(reference.component_dofs.reshape(-1), np.arange(reference.basis.N))
    nodes, interpolation, points, area, candidate_faces = pad_quadrature(
        model, len(outer), args.candidate_radius, args.quadrature_order, sparse_interpolation=True
    )
    n_body = int(reference.basis.N)
    selector = sparse.coo_matrix(
        (np.ones(len(nodes)), (np.arange(len(nodes)), nodes)), shape=(len(nodes), len(model.vertices))
    ).tocsr()
    point_map = interpolation @ selector
    top = points[:, 2] > 0
    sign = np.where(top, -1.0, 1.0)
    plate_map = np.zeros(3 * len(points))
    plate_map[2::3] = top
    jacobian = sparse.hstack(
        (sparse.kron(point_map, sparse.eye(3)), sparse.csr_matrix(plate_map[:, None])), format="csr"
    )
    mass, stiffness = reference.mass, reference.stiffness
    operator = sparse.block_diag(
        (mass + args.dt**2 * stiffness, sparse.csr_matrix([[0.05 + args.dt * args.damping]])), format="csc"
    )
    order = np.r_[model.free, reference.gauge_dofs, n_body]
    factor = OrderedFactor(operator, order)
    factor_seconds = perf_counter() - started
    if args.contact_solver == "condensed":
        compliance = np.empty((jacobian.shape[0], jacobian.shape[0]))
        with (args.output / "condensation.jsonl").open("w", buffering=1) as journal:
            for first in range(0, len(compliance), 16):
                selected = slice(first, first + 16)
                compliance[:, selected] = jacobian @ factor.solve(jacobian[selected].T.toarray())
                journal.write(
                    json.dumps(
                        {"columns": min(first + 16, len(compliance)), "elapsed_seconds": perf_counter() - started}
                    )
                    + "\n"
                )
        contact = CondensedPadDynamics(compliance, sign, area, warm_start=args.warm_contact)
    elif args.contact_solver == "matrix-free":
        contact = MatrixFreePadDynamics(operator, factor, jacobian, sign, area, warm_start=args.warm_contact)
    else:
        modes = np.zeros((n_body + 1, 7))
        modes[:-1, :6] = reference.rigid_modes
        modes[-1, -1] = 1
        contact = ModalPadDynamics(operator, factor, jacobian, modes, sign, area, warm_start=args.warm_contact)
    _, _, quadratic_faces = model.surface(faces)
    selected_keys = {tuple(sorted(face)) for face in candidate_faces}
    omitted = np.array([tuple(sorted(face)) not in selected_keys for face in quadratic_faces[:, :3]])
    lower, upper = outer[:, 2].min() - args.clearance, outer[:, 2].max() + args.clearance
    initial_gap = np.where(top, upper - points[:, 2], points[:, 2] - lower)
    displacement = np.zeros(n_body + 1)
    velocity = np.zeros_like(displacement)
    previous_normal = np.zeros(len(points))
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mass_kg": reference.total_mass,
            "body_dofs": n_body,
            "dynamic_dofs": len(displacement),
            "fixed_dynamic_dofs": 0,
            "contact_points": len(points),
            "factor_seconds": factor_seconds,
            "setup_seconds": perf_counter() - started,
            "compliance_reciprocity_error": contact.reciprocity_error if args.contact_solver == "condensed" else None,
            "scope": "Independent affine-geometry P2 solid with explicit linear/StVK constitutive choice, full consistent inertia and free rigid modes; fixed reference-area physical pad, lagged smooth Coulomb friction. Spatial/kinematic qualification remains separate.",
        },
    )
    rows, completed = [], False
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, command, endpoint) in enumerate(force_history(args)):
                gravity = args.gravity_x * min(
                    max(((step + 1) * args.dt - args.gravity_start) / args.gravity_ramp, 0), 1
                )
                body = reference.mass_modes[:, 0] * gravity
                rhs = np.r_[
                    mass @ velocity[:-1] - args.dt * (stiffness @ displacement[:-1]) + args.dt * body,
                    0.05 * velocity[-1] + args.dt * command,
                ]
                old_gap = initial_gap + sign * (jacobian @ displacement).reshape(-1, 3)[:, 2]
                result, generalized_force, new_velocity, displacement, internal, constitutive_rounds = advance(
                    factor,
                    jacobian,
                    contact,
                    args.contact_solver,
                    rhs,
                    displacement,
                    old_gap,
                    previous_normal,
                    args,
                    command,
                    reference,
                    nonlinear,
                )
                force = result["force"]
                actual_point_velocity = (jacobian @ new_velocity).reshape(-1, 3)
                actual_gap = initial_gap + sign * (jacobian @ displacement).reshape(-1, 3)[:, 2]
                normal = 1e9 * area * np.maximum(-actual_gap, 0)
                speed = np.linalg.norm(actual_point_velocity[:, :2], axis=1)
                ratio = np.minimum(speed / 1e-4, 1)
                tangent = (
                    -(args.friction * previous_normal * (2 * ratio - ratio**2))[:, None]
                    * actual_point_velocity[:, :2]
                    / np.maximum(speed[:, None], 1e-30)
                )
                inertial = mass @ ((new_velocity[:-1] - velocity[:-1]) / args.dt)
                residual = inertial + internal - body - generalized_force[:-1]
                current_vertices = model.vertices + displacement[:-1].reshape(-1, 3)
                physical_moment = np.cross(
                    current_vertices, (generalized_force[:-1] + body - inertial).reshape(-1, 3)
                ).sum(axis=0)
                plate_residual = (
                    0.05 * (new_velocity[-1] - velocity[-1]) / args.dt
                    + args.damping * new_velocity[-1]
                    - command
                    - generalized_force[-1]
                )
                world_z = model.vertices[:, 2] + displacement[:-1].reshape(-1, 3)[:, 2]
                minimum_gap = min(
                    quadratic_triangle_minimum((world_z - lower)[quadratic_faces[omitted]]).min(),
                    quadratic_triangle_minimum((upper - displacement[-1] - world_z)[quadratic_faces[omitted]]).min(),
                )
                mode_velocity = np.linalg.solve(reference.gram, reference.mass_modes.T @ new_velocity[:-1])
                scale = max(np.linalg.norm(force), 1e-12)
                row = {
                    "step": step + 1,
                    "time_s": (step + 1) * args.dt,
                    "level": level,
                    "command_force_n": command,
                    "compression_coordinate_m": float(displacement[-1]),
                    "plate_speed_m_per_s": float(new_velocity[-1]),
                    "upper_reaction_n": float(-force[top, 2].sum()),
                    "lower_reaction_n": float(force[~top, 2].sum()),
                    "tangential_resultant_n": force[:, :2].sum(axis=0).tolist(),
                    "body_force_n": body.reshape(-1, 3).sum(axis=0).tolist(),
                    "egg_inertia_resultant_norm_n": float(np.linalg.norm(inertial.reshape(-1, 3).sum(axis=0))),
                    "egg_kinetic_j": float(0.5 * new_velocity[:-1] @ (mass @ new_velocity[:-1])),
                    "equation_residual_relative": float(
                        np.linalg.norm(np.r_[residual, plate_residual]) / max(command, 1e-3)
                    ),
                    "normal_law_relative_error": float(np.linalg.norm(sign * force[:, 2] - normal) / scale),
                    "friction_law_relative_error": float(np.linalg.norm(force[:, :2] - tangent) / scale),
                    "point_velocity_prediction_error_m_per_s": float(
                        np.max(np.abs(actual_point_velocity - result["velocity"]))
                    ),
                    "minimum_omitted_gap_m": float(minimum_gap),
                    "rigid_mode_velocity": mode_velocity.tolist(),
                    "newton_iterations": result["iterations"],
                    "linear_iterations": result.get("linear_iterations"),
                    "fixed_point_iterations": result.get("fixed_point_iterations"),
                    "constitutive_rounds": constitutive_rounds,
                    "current_moment_balance_relative_error": float(
                        np.linalg.norm(physical_moment) / (max(command, 1e-3) * 0.03)
                    ),
                    "whitened_momentum_residual": result["whitened_momentum_residual"],
                }
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if endpoint:
                    stress = (
                        reference.corner_stress(displacement[:-1])[0]
                        if nonlinear is None
                        else nonlinear.corner_stress(displacement[:-1])
                    )
                    extra = (
                        {}
                        if nonlinear is None
                        else {
                            "stvk_gradient": reference.corner_basis.interpolate(displacement[:-1]).grad.transpose(
                                2, 3, 0, 1
                            ),
                            "young_pa": reference.young,
                            "poisson": reference.poisson,
                        }
                    )
                    if nonlinear is not None:
                        scalar_mass = mass[0::3, 0::3]
                        weights = np.asarray(scalar_mass.sum(axis=1)).ravel()
                        current_center = weights @ current_vertices / reference.total_mass
                        covariance = (model.vertices - reference.com).T @ (
                            scalar_mass @ (current_vertices - current_center)
                        )
                        left, _, right = np.linalg.svd(covariance)
                        rotation = right.T @ left.T
                        assert np.linalg.det(rotation) > 0
                        extra["rotation_material_to_world"] = rotation
                    np.savez_compressed(
                        args.output / "frames" / f"level{level}.npz",
                        vertices_m=model.vertices,
                        tetrahedra=tets,
                        stress_pa=stress,
                        load_n=command,
                        displacement_m=displacement[:-1].reshape(-1, 3),
                        velocity_m_per_s=new_velocity[:-1].reshape(-1, 3),
                        source_points_m=points,
                        source_forces_n=force,
                        projected_area_m2=area,
                        forces_n=generalized_force[:-1].reshape(-1, 3),
                        body_forces_n=body.reshape(-1, 3),
                        inertial_forces_n=inertial.reshape(-1, 3),
                        equation_residual_n=residual.reshape(-1, 3),
                        internal_forces_n=internal.reshape(-1, 3),
                        source_points_current_m=points + point_map @ displacement[:-1].reshape(-1, 3),
                        **extra,
                    )
                previous_normal = sign * force[:, 2]
                velocity = new_velocity
            completed = True
        finally:
            checks = [step_checks(row) for row in rows]
            windows = []
            for level in sorted({row["level"] for row in rows}):
                data = [row for row in rows if row["level"] == level]
                window = [row for row in data if row["time_s"] >= data[-1]["time_s"] - 0.05 - 1e-10]
                gates = []
                for row in window:
                    gate = equilibrium(row, data[-1]["command_force_n"])
                    gate.update(step_checks(row))
                    if row["body_force_n"][0] != 0:
                        gate["sustained_shear"] = abs(
                            row["tangential_resultant_n"][0] + row["body_force_n"][0]
                        ) <= 0.01 * abs(row["body_force_n"][0])
                    gates.append(gate)
                windows.append(
                    {
                        "level": level,
                        "samples": len(window),
                        "checks": {key: all(g[key] for g in gates) for key in gates[0]},
                        "mechanical_window_passed": bool(
                            completed
                            and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
                            and all(all(g.values()) for g in gates)
                        ),
                    }
                )
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "steps": len(rows),
                    "elapsed_seconds": perf_counter() - started,
                    "history_checks": {k: all(c[k] for c in checks) for k in checks[0]} if checks else {},
                    "mechanical_history_passed": bool(completed and checks and all(all(c.values()) for c in checks)),
                    "windows": windows,
                    "physical_reference_qualified": False,
                },
            )


if __name__ == "__main__":
    main()
