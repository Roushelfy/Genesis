"""Solve free-body linear elastic frictionless plate contact by a condensed convex complementarity problem."""

import argparse
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import linprog, minimize

from .artifacts import file_id, runtime_identity, write_json
from .p1_elastic import FreeElasticP1
from .contact_qp import solve_contact_qp


@dataclass
class ContactMinimum:
    force: np.ndarray
    coordinates: np.ndarray
    active: np.ndarray
    rank: int
    iterations: int
    converged: bool


def active_contact_minimum(matrix, gap, rigid, target, initial):
    """Minimize the convex complementary energy with feasible primal active-set steps."""
    alpha = initial.copy()
    if alpha.min() < 0 or np.linalg.norm(rigid.T @ alpha - target) > 1e-10:
        feasible = linprog(np.zeros(len(gap)), A_eq=rigid.T, b_eq=target, bounds=(0, None), method="highs")
        if not feasible.success:
            raise ValueError(f"No nonnegative balanced contact load: {feasible.message}")
        alpha = feasible.x
    active = alpha > 0
    coordinate = np.zeros(len(target))
    converged = False
    for iteration in range(5 * len(gap) + 20):
        reduced = np.block([[matrix[np.ix_(active, active)], rigid[active]], [rigid[active].T, np.zeros((4, 4))]])
        solution, _, rank, _ = lstsq(reduced, np.r_[-gap[active], target], cond=1e-13, lapack_driver="gelsy")
        candidate = np.zeros_like(alpha)
        candidate[active] = solution[:-4]
        coordinate = solution[-4:]
        negative = active & (candidate < -1e-12)
        if negative.any():
            indices = np.flatnonzero(negative)
            fractions = alpha[indices] / (alpha[indices] - candidate[indices])
            leaving = indices[fractions.argmin()]
            fraction = fractions.min()
            alpha += fraction * (candidate - alpha)
            alpha[leaving] = 0
            active[leaving] = False
            continue
        alpha = candidate
        dual = matrix @ alpha + gap + rigid @ coordinate
        inactive = np.flatnonzero(~active)
        if not len(inactive) or dual[inactive].min() >= -1e-10:
            converged = True
            break
        active[inactive[dual[inactive].argmin()]] = True
    return ContactMinimum(alpha, coordinate, active, rank, iteration + 1, converged)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--loads", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    cli.add_argument("--candidate-radius", type=float, default=0.0002)
    cli.add_argument("--column-ordering", action="store_true")
    cli.add_argument("--element-order", type=int, choices=(1, 2), default=1)
    cli.add_argument("--continuous-quadratic-contact", action="store_true")
    cli.add_argument("--contact-operator", type=Path)
    cli.add_argument("--contact-certificate", type=Path)
    cli.add_argument("--qp-solver-path", type=Path)
    cli.add_argument("--qp-rounds", type=int, default=40)
    cli.add_argument("--cholespy-path", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert not args.continuous_quadratic_contact or args.element_order == 2
    assert args.cholespy_path is None or args.element_order == 2
    assert args.qp_solver_path is None or (args.continuous_quadratic_contact and args.contact_certificate is None)
    independent = args.contact_certificate is not None or args.qp_solver_path is not None
    certificate = None
    if args.contact_certificate is not None:
        assert args.continuous_quadratic_contact and args.contact_operator is not None and len(args.loads) == 1
        certificate = json.loads((args.contact_certificate / "result.json").read_text())
        assert certificate["candidate_face_certificate_passed"] and certificate["status"] == "Solved"
        assert certificate["operator"]["sha256"] == file_id(args.contact_operator)["sha256"]
        original = Path(certificate["field_input"]["path"])
        assert file_id(original)["sha256"] == certificate["field_input"]["sha256"]
        previous = json.loads((original.parent / "result.json").read_text())
        assert any(
            row["load_n"] == args.loads[0] and row["field"]["sha256"] == file_id(original)["sha256"]
            for row in previous["cases"]
        )
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with np.load(args.mesh) as saved:
        vertices, tets, outer, faces = (saved[key] for key in ("vertices", "tetrahedra", "outer_vertices", "mid_faces"))
    center = vertices.mean(axis=0)
    model_class = FreeElasticP1
    if args.element_order == 2:
        from .p2_elastic import FreeElasticP2, quadratic_triangle_minimum

        model_class = FreeElasticP2
    write_json(args.output / "assembly.json", {"element_order": args.element_order, "mesh": file_id(args.mesh)})
    if args.element_order == 2:
        model = model_class(
            vertices,
            tets,
            1e10,
            0.3,
            2000.0,
            column_faces=faces if args.column_ordering else None,
            cholespy_path=args.cholespy_path,
        )
    else:
        model = model_class(vertices, tets, 1e10, 0.3, 2000.0, column_faces=faces if args.column_ordering else None)
    outer_nodes = np.arange(len(outer))
    if args.element_order == 2:
        outer_nodes, faces, quadratic_faces = model.surface(faces)
        vertices = model.vertices
        outer = vertices[outer_nodes]

    radius = np.linalg.norm(outer[:, :2], axis=1)
    nodes = np.flatnonzero(radius <= args.candidate_radius)
    sign = np.where(outer[nodes, 2] < 0, 1.0, -1.0)
    top = sign < 0
    assert top.any() and (~top).any()
    initial_gap = np.where(top, outer[:, 2].max() - outer[nodes, 2], outer[nodes, 2] - outer[:, 2].min())
    # These three rigid coordinates are physical unknowns. The other rigid
    # motions do not affect frictionless horizontal-plate contact at linear order.
    lever = 0.03
    rigid = np.column_stack(
        (sign, sign * (outer[nodes, 1] - center[1]) / lever, -sign * (outer[nodes, 0] - center[0]) / lever, top)
    )
    target = np.array([0.0, 0.0, 0.0, 1.0])
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "candidate_nodes": len(nodes),
            "tetrahedra": len(tets),
            "loads_n": args.loads,
            "element_order": args.element_order,
            "qp_solver_path": str(args.qp_solver_path) if args.qp_solver_path is not None else None,
            "cholespy_path": str(args.cholespy_path) if args.cholespy_path is not None else None,
        },
    )
    factor_seconds = time.perf_counter() - start
    compliance = np.empty((len(nodes), len(nodes)))
    if args.contact_operator is not None:
        previous = json.loads((args.contact_operator.parent / "start.json").read_text())
        assert previous["mesh"]["sha256"] == file_id(args.mesh)["sha256"]
        assert previous.get("element_order", 1) == args.element_order
        with np.load(args.contact_operator) as saved:
            selected = np.searchsorted(saved["nodes"], nodes)
            assert (selected < len(saved["nodes"])).all(), "Cached operator does not cover the requested candidate set"
            for key, expected in (("nodes", nodes), ("sign", sign), ("rigid", rigid), ("initial_gap_m", initial_gap)):
                np.testing.assert_array_equal(saved[key][selected], expected)
            compliance[:] = saved["compliance_m_per_n"][np.ix_(selected, selected)]
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        for first in range(0, len(nodes), 16) if args.contact_operator is None else ():
            selected = slice(first, first + 16)
            force = np.zeros((3 * len(vertices), len(nodes[selected])))
            force[3 * outer_nodes[nodes[selected]] + 2, np.arange(len(nodes[selected]))] = sign[selected]
            corrected = force - model.mass_modes @ np.linalg.solve(model.gram, model.rigid_modes.T @ force)
            displacement = np.zeros_like(force)
            displacement[model.free] = model.factor.solve(corrected[model.free] / model.young)
            displacement -= model.rigid_modes @ np.linalg.solve(model.gram, model.mass_modes.T @ displacement)
            compliance[:, selected] = sign[:, None] * displacement[3 * outer_nodes[nodes] + 2]
            journal.write(
                json.dumps(
                    {"condensed_columns": min(first + 16, len(nodes)), "elapsed_seconds": time.perf_counter() - start}
                )
                + "\n"
            )
        reciprocity = np.linalg.norm(compliance - compliance.T) / np.linalg.norm(compliance)
        assert reciprocity < 1e-7, "Free-body compliance fails reciprocity"
        compliance = (compliance + compliance.T) / 2
        np.linalg.cholesky(compliance)
        np.savez_compressed(
            args.output / "contact-operator.npz",
            nodes=nodes,
            sign=sign,
            compliance_m_per_n=compliance,
            rigid=rigid,
            initial_gap_m=initial_gap,
        )
        rows = []
        initial = np.where(top, 1 / top.sum(), 1 / (~top).sum())
        for load in args.loads:
            solve_start = time.perf_counter()
            displacement_scale = load / (model.young * 5e-5)
            matrix = compliance * load / displacement_scale
            gap = initial_gap / displacement_scale
            fitted = (
                None
                if independent
                else minimize(
                    lambda x: (0.5 * x @ matrix @ x + gap @ x, matrix @ x + gap),
                    initial,
                    jac=True,
                    method="SLSQP",
                    bounds=[(0, None)] * len(nodes),
                    constraints={"type": "eq", "fun": lambda x: rigid.T @ x - target, "jac": lambda x: rigid.T},
                    options={"ftol": 1e-12, "maxiter": 1000},
                )
            )
            interpolation = np.eye(len(nodes))
            constraint_rounds, continuous_converged = 0, True
            qp = None
            if args.continuous_quadratic_contact:
                candidate_map = np.full(len(vertices), -1, dtype=int)
                candidate_map[outer_nodes[nodes]] = np.arange(len(nodes))
                candidate_faces = candidate_map[quadratic_faces]
                candidate_faces = candidate_faces[(candidate_faces >= 0).all(axis=1)]
            if certificate is not None:
                with np.load(args.contact_operator) as saved:
                    np.testing.assert_array_equal(saved["nodes"], nodes)
                with np.load(args.contact_certificate / "solution.npz") as saved:
                    interpolation = saved["interpolation"]
                    multipliers = saved["multipliers"]
                    coordinates = saved["coordinates_scaled"]
                assert interpolation.shape == (len(multipliers), len(nodes))
                np.testing.assert_allclose(interpolation.sum(axis=1), 1, rtol=0, atol=1e-12)
                condensed_alpha = interpolation.T @ multipliers
                condensed_gap = interpolation @ (matrix @ condensed_alpha + gap + rigid @ coordinates)
                kkt_passed = bool(
                    np.linalg.norm(rigid.T @ condensed_alpha - target) < 1e-8
                    and multipliers.min() >= -1e-10
                    and condensed_gap.min() >= -1e-10
                    and np.max(np.abs(multipliers * condensed_gap)) < 1e-8
                )
                constraint_rounds = len(certificate["constraint_rounds"])
                minimum = ContactMinimum(multipliers, coordinates, multipliers > 0, -1, constraint_rounds, kkt_passed)
                continuous_converged = certificate["candidate_face_certificate_passed"]
            elif args.qp_solver_path is not None:
                qp = solve_contact_qp(
                    matrix,
                    gap,
                    rigid,
                    target,
                    candidate_faces,
                    args.qp_solver_path,
                    round_limit=args.qp_rounds,
                    journal=journal,
                )
                interpolation = qp.interpolation
                constraint_rounds = len(qp.rounds)
                minimum = ContactMinimum(
                    qp.multipliers, qp.coordinates, qp.multipliers > 0, -1, constraint_rounds, qp.fixed_passed
                )
                continuous_converged = qp.candidate_passed
            elif args.continuous_quadratic_contact:
                from .quadratic_contact import quadratic_contact_minimum

                continuous = quadratic_contact_minimum(
                    matrix, gap, rigid, target, fitted.x, candidate_faces, active_contact_minimum, journal
                )
                minimum, interpolation = continuous.minimum, continuous.interpolation
                constraint_rounds, continuous_converged = continuous.rounds, continuous.converged
            else:
                minimum = active_contact_minimum(matrix, gap, rigid, target, fitted.x)
            alpha = interpolation.T @ minimum.force
            active = alpha > 0 if args.continuous_quadratic_contact else minimum.active
            rank = minimum.rank
            coordinate = minimum.coordinates * displacement_scale
            forces = np.zeros_like(vertices)
            forces[outer_nodes[nodes], 2] = alpha * load * sign
            tensor, elastic, correction, residual = model.solve(forces)
            displacement = elastic.copy()
            displacement[:, 2] += coordinate[0]
            displacement += np.cross(np.array([coordinate[1] / lever, coordinate[2] / lever, 0.0]), vertices - center)
            all_bottom_gap = outer[:, 2] + displacement[outer_nodes, 2] - outer[:, 2].min()
            all_top_gap = outer[:, 2].max() + coordinate[3] - outer[:, 2] - displacement[outer_nodes, 2]
            actual_gap = np.where(top, all_top_gap[nodes], all_bottom_gap[nodes])
            eig = np.linalg.eigvalsh(tensor)
            dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            mises = np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))
            gradient = model.displacement_gradient(displacement)
            reaction_error = np.linalg.norm(rigid.T @ alpha - target)
            equation_error = np.linalg.norm(residual) / np.linalg.norm(forces)
            balance_correction = np.linalg.norm(correction) / np.linalg.norm(forces)
            minimum_gap = min(all_bottom_gap.min(), all_top_gap.min())
            if args.element_order == 2:
                local_faces = np.searchsorted(outer_nodes, quadratic_faces)
                minimum_gap = min(
                    quadratic_triangle_minimum(all_bottom_gap[local_faces]).min(),
                    quadratic_triangle_minimum(all_top_gap[local_faces]).min(),
                )
            contact_error = np.abs((interpolation @ actual_gap)[minimum.active]).max() / displacement_scale
            complementarity = np.max(np.abs(minimum.force * (interpolation @ actual_gap))) / displacement_scale
            contact_optimality = (
                rank == minimum.active.sum() + 4 and contact_error < 1e-7 if not independent else complementarity < 1e-8
            )
            admitted = bool(
                minimum.converged
                and continuous_converged
                and contact_optimality
                and reaction_error < 1e-8
                and equation_error < 1e-7
                and balance_correction < 1e-8
                and minimum.force.min() >= -1e-10
                and minimum_gap >= -1e-7 * displacement_scale
            )
            path = args.output / f"load-{load:g}.npz"
            np.savez_compressed(
                path,
                vertices_m=vertices,
                outer_nodes=outer_nodes,
                outer_subtriangles=faces,
                element_order=args.element_order,
                tetrahedra=tets,
                displacement_m=displacement,
                elastic_displacement_m=elastic,
                stress_pa=tensor,
                contact_force_n=forces,
                bottom_gap_m=all_bottom_gap,
                top_gap_m=all_top_gap,
                balance_correction_n=correction,
                equation_residual_n=residual,
                candidate_nodes=nodes,
                contact_alpha=alpha,
                contact_constraint_interpolation=interpolation,
                contact_compressive_multipliers=minimum.force,
                contact_constraint_points_m=interpolation @ outer[nodes],
                rigid_coordinates_m=coordinate,
            )
            footprints = []
            for upper in (False, True):
                chosen = (top == upper) & active
                selected_faces = np.isin(faces, nodes[chosen]).all(axis=1)
                triangles = outer[faces[selected_faces]]
                area = (
                    np.linalg.norm(
                        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
                    ).sum()
                    / 2
                )
                footprint = {
                    "upper": upper,
                    "positive_force_nodes": int((chosen & (alpha > 0)).sum()),
                    "maximum_radius_m": float(radius[nodes[chosen]].max()),
                    "fully_active_face_area_m2": float(area),
                }
                if args.continuous_quadratic_contact:
                    # P2 shape functions can give negative consistent nodal forces;
                    # their sign is not the physical multiplier's sign or support.
                    footprint["fully_positive_nodal_force_face_area_m2"] = footprint.pop("fully_active_face_area_m2")
                    point = interpolation @ outer[nodes]
                    point_top = point[:, 2] > center[2]
                    carried = (point_top == upper) & (minimum.force > 0)
                    footprint["compressive_constraint_points"] = int(carried.sum())
                    footprint["maximum_constraint_radius_m"] = float(np.linalg.norm(point[carried, :2], axis=1).max())
                    gaps = (all_top_gap if upper else all_bottom_gap)[local_faces]
                    closed = -quadratic_triangle_minimum(-gaps) <= 1e-7 * displacement_scale
                    triangle = vertices[quadratic_faces[:, :3]]
                    closed &= (triangle[:, :, 2].mean(axis=1) > center[2]) == upper
                    footprint["fully_closed_quadratic_face_area_m2"] = float(
                        np.linalg.norm(
                            np.cross(
                                triangle[closed, 1] - triangle[closed, 0], triangle[closed, 2] - triangle[closed, 0]
                            ),
                            axis=1,
                        ).sum()
                        / 2
                    )
                footprints.append(footprint)
            row = {
                "load_n": load,
                "linear_contact_equations_admitted": admitted,
                "physical_reference_qualified": False,
                "optimizer_success": bool(fitted.success) if fitted is not None else None,
                "final_solver": "Independent primal-dual contact with freshly solved elasticity, global quadratic gaps, force/moment balance and force-weighted complementarity. Multiplier uniqueness is not required."
                if independent
                else "Feasible primal active-set KKT solve with blocking steps and inactive-gap admission; SLSQP supplies a feasible initial point when available.",
                "contact_solver_converged": minimum.converged,
                "active_set_converged": minimum.converged if not independent else None,
                "active_set_iterations": minimum.iterations if not independent else None,
                "optimizer_message": str(fitted.message) if fitted is not None else None,
                "optimizer_iterations": fitted.nit if fitted is not None else None,
                "active_kkt_rank": int(rank) if not independent else None,
                "active_kkt_dimension": int(minimum.active.sum() + 4) if not independent else None,
                "qp_certificate": {
                    "status": qp.status,
                    "solver_version": qp.solver_version,
                    "solver_module": qp.solver_module,
                    "rounds": qp.rounds,
                    "stationarity_scaled_l2": qp.stationarity,
                    "fixed_qp_certificate_passed": qp.fixed_passed,
                    "candidate_face_certificate_passed": qp.candidate_passed,
                }
                if qp is not None
                else None,
                "continuous_constraint_rounds": constraint_rounds,
                "continuous_constraints_converged": continuous_converged,
                "force_moment_constraint_relative_error": float(reaction_error),
                "equation_relative_residual": float(equation_error),
                "balance_correction_relative": float(balance_correction),
                "minimum_compression_fraction": float(minimum.force.min()),
                "minimum_global_gap_m": float(minimum_gap),
                "active_contact_gap_relative": float(contact_error) if not independent else None,
                "positive_multiplier_gap_relative": float(contact_error) if independent else None,
                "force_weighted_complementarity_scaled": float(complementarity),
                "plate_travel_m": float(-coordinate[3]),
                "raw_maximum_tension_pa": float(eig[..., -1].max()),
                "raw_maximum_mises_pa": float(mises.max()),
                "maximum_displacement_gradient": float(np.linalg.norm(gradient, axis=(-2, -1)).max()),
                "deformed_total_moment_nm": np.cross(vertices + displacement, forces).sum(axis=0).tolist(),
                "footprints": footprints,
                "solve_seconds": time.perf_counter() - solve_start,
                "field": file_id(path),
            }
            rows.append(row)
            journal.write(json.dumps(row) + "\n")
            initial = alpha
    write_json(
        args.output / "result.json",
        {
            "mesh": file_id(args.mesh),
            "identity": runtime_identity(),
            "factor_seconds": factor_seconds,
            "reused_contact_operator": file_id(args.contact_operator) if args.contact_operator is not None else None,
            "contact_certificate": file_id(args.contact_certificate / "result.json")
            if certificate is not None
            else None,
            "contact_certificate_solution": file_id(args.contact_certificate / "solution.npz")
            if certificate is not None
            else None,
            "total_seconds": time.perf_counter() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "compliance_reciprocity_error": float(reciprocity),
            "cases": rows,
            "element_order": args.element_order,
            "continuous_quadratic_contact": args.continuous_quadratic_contact,
            "scope": "Independent small-strain affine-tetrahedron elasticity, frictionless nodal complementarity control. "
            "P1 stress is constant per tetrahedron; P2 stress is linear and retained at all four corners, "
            "including raw extrema. P2 contact includes original vertices and affine edge midpoints; "
            "its gap admission checks exact quadratic minima across every outer triangle. "
            "No physical egg vertex pins: factorization gauges are removed, balance corrections "
            "are measured, vertical translation and two tilts are solved as physical unknowns. "
            "With continuous P2 constraints, consistent nodal load signs are separate from the nonnegative "
            "physical point multipliers; positive-nodal-force and fully-closed-face areas are separate diagnostics. "
            "Full-surface gap is checked beyond the candidate patch. This is not yet a qualified "
            "finite-deformation or continuum contact reference. Footprint area counts only faces "
            "with all three nodes active, not an interpolated pressure-support assertion.",
        },
    )


if __name__ == "__main__":
    main()
