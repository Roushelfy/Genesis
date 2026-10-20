"""Attribute retained SCR incompatibility using constructive cone-load contrasts.

Positive LP results are verified surface point-force constructions. Negative LP
results are unresolved sampling results, except for the unchanged baseline whose
larger continuous-support infeasibility certificate is already independently bound.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import cvxopt
from cvxopt import solvers
from scipy import sparse
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components, dijkstra

from .artifacts import file_id, write_json
from .audit_native_rejection import triangle_distance
from .contact_fem_exact_columns import surface_quadrature


def surface_graph(vertices, faces):
    centers = vertices[faces].mean(axis=1)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    assert np.all(counts == 2), "The captured stress boundary must be closed and manifold"
    order = np.argsort(inverse).reshape(-1, 2)
    owners = np.tile(np.arange(len(faces)), 3)[order]
    midpoint = vertices[unique].mean(axis=1)
    length = np.linalg.norm(centers[owners[:, 0]] - midpoint, axis=1)
    length += np.linalg.norm(centers[owners[:, 1]] - midpoint, axis=1)
    graph = sparse.coo_matrix((length, (owners[:, 0], owners[:, 1])), shape=(len(faces), len(faces))).tocsr()
    return graph + graph.T, centers


def generators(positions, normals, friction, anchor, radius, force_directions):
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    first = np.cross(normals, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(normals, first)
    angles = np.arange(16) * (2 * np.pi / 16)
    tangent = np.cos(angles)[None, :, None] * first[:, None] + np.sin(angles)[None, :, None] * second[:, None]
    # Exact saturated source directions must be available: a sixteen-sided
    # inscribed cone alone can falsely reject a direction between its rays.
    aligned = force_directions[None] - np.einsum("si,qi->qs", force_directions, normals)[..., None] * normals[:, None]
    aligned /= np.maximum(np.linalg.norm(aligned, axis=-1)[..., None], np.finfo(float).tiny)
    tangent = np.concatenate((tangent, aligned), axis=1)
    directions = np.concatenate((normals[:, None], normals[:, None] + friction[:, None, None] * tangent), axis=1)
    p = np.einsum("qai,qi->qa", directions, normals)
    t = np.linalg.norm(directions - p[..., None] * normals[:, None], axis=-1)
    assert np.all(p >= 0) and np.all(t <= friction[:, None] * p + 64 * np.finfo(float).eps)
    moment = np.cross((positions - anchor)[:, None] / radius, directions)
    return np.concatenate((directions, moment), axis=-1).reshape(-1, 6).T


def continuous_force_certificate(normals, friction, target, force_scale):
    """Search in three dimensions, then independently verify strict facet polar bounds."""
    axis = np.eye(3)[np.argmin(np.abs(normals), axis=1)]
    first = np.cross(normals, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(normals, first)
    blocks = np.stack((normals, -friction[:, None] * first, -friction[:, None] * second), axis=1).reshape(-1, 3)
    matrix = np.concatenate((blocks, np.vstack((np.zeros((1, 3)), -np.eye(3)))))
    h = np.r_[np.zeros(len(blocks)), 1.0, 0, 0, 0]
    previous = dict(solvers.options)
    try:
        solvers.options.update(show_progress=False, abstol=1e-12, reltol=1e-10, feastol=1e-12, maxiters=100)
        solution = solvers.conelp(cvxopt.matrix(-target / force_scale), cvxopt.matrix(matrix), cvxopt.matrix(h),
                                 dims={"l": 0, "q": [3] * len(normals) + [4], "s": []})
    finally:
        solvers.options.clear()
        solvers.options.update(previous)
    raw = np.array(solution["x"]).reshape(3)
    interior = -normals.sum(axis=0)
    interior /= max(np.linalg.norm(interior), np.finfo(float).tiny)
    attempts = []
    for mixing in (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2):
        ray = raw + mixing * interior
        ray /= max(np.linalg.norm(ray), np.finfo(float).tiny)
        compression = normals @ ray
        polar = compression + friction * np.linalg.norm(ray - compression[:, None] * normals, axis=1)
        rounding = 1024 * np.finfo(float).eps * (1 + friction) * (1 + np.linalg.norm(ray))
        upper = np.max(polar + rounding)
        projected = target @ ray
        budget = 1e-8 * force_scale * np.linalg.norm(ray)
        allowance = 1024 * np.finfo(float).eps * force_scale
        passed = bool(upper < 0 and projected > budget + allowance)
        attempts.append({"ray": ray.tolist(), "maximum_rounded_polar_bound": upper.item(),
            "target_force_projection_n": projected.item(), "force_acceptance_projection_budget_n": budget.item(),
            "roundoff_projection_allowance_n": allowance.item(), "strict_infeasibility_proven": passed})
        if passed:
            return {"solver_status": solution["status"], "independent_force_infeasibility_proven": True,
                    "proof": attempts[-1], "candidate_facet_cones": len(normals)}
    return {"solver_status": solution["status"], "independent_force_infeasibility_proven": False,
            "best_target_projection_n": max(p["target_force_projection_n"] for p in attempts),
            "candidate_facet_cones": len(normals)}


def fit(matrix, target, components, scale):
    selected = matrix[:components]
    result = linprog(np.ones(matrix.shape[1]), A_eq=selected, b_eq=target[:components] / scale,
                     bounds=(0, None), method="highs",
                     options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
    row = {"solver_status": int(result.status), "message": result.message,
           "constructively_feasible": False, "continuous_infeasibility_claim": False}
    if result.success:
        residual = np.linalg.norm(selected @ result.x - target[:components] / scale)
        assert result.x.min() >= -1e-12 and residual <= 1e-8
        row.update(constructively_feasible=True, normalized_equality_residual=residual.item(),
                   nonzero_generators=int(np.count_nonzero(result.x)),
                   unsigned_generator_coefficient_n=(scale * result.x.sum()).item())
    return row


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--certificate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bad = continuous_force_certificate(np.array([[0.0, 0, 1]]), np.array([0.3]), np.array([1.0, 0, 0]), 1.0)
    good = continuous_force_certificate(np.array([[0.0, 0, 1]]), np.array([0.3]), np.array([0.0, 0, 1]), 1.0)
    assert bad["independent_force_infeasibility_proven"] and not good["independent_force_infeasibility_proven"]
    state = args.run / "rejected-state.npz"
    receipt = json.loads((args.run / "rejected-step.json").read_text())
    start = json.loads((args.run / "start.json").read_text())
    certificate = json.loads(args.certificate.read_text())
    assert certificate["passed"] and start["args"]["check_every_step"]
    actual = file_id(state)
    assert all(actual[k] == certificate["state"][k] == receipt["state"][k] for k in ("bytes", "sha256"))
    with np.load(state) as saved:
        data = {key: saved[key] for key in saved.files}
    vertices, face_nodes = data["mesh.vertices"], data["mesh.surface_nodes"][:, :3]
    triangles = vertices[face_nodes]
    opposite = vertices[data["mesh.surface_opposite_vertices"]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals *= np.where(np.einsum("fi,fi->f", normals, opposite - triangles[:, 0]) >= 0, 1, -1)[:, None]
    graph, centers = surface_graph(vertices, face_nodes)
    component_count, components = connected_components(graph, directed=False)
    barycentric, _ = surface_quadrature(25)
    samples = np.einsum("qc,fci->fqi", barycentric, triangles)
    center_distance = np.linalg.norm(samples - centers[:, None], axis=2)
    rows = []
    for env in np.flatnonzero(data["failure.status"]):
        proof = next(row for row in certificate["rows"] if row["environment"] == env)
        assert proof["independent_infeasibility_proven"]
        active = data["contacts.valid"][:, env].astype(bool)
        force, point, anchor_point, radius, mu, source_normal, anchor_face = (
            data[key][active, env] for key in ("contacts.force", "contacts.position", "contacts.center",
                "contacts.radius", "contacts.friction", "contacts.normal", "contacts.anchor_face"))
        assert np.all(anchor_face >= 0) and np.all(anchor_face < len(triangles))
        assert len(force) == data["failure.group_count"][env]
        for center, face in zip(anchor_point, anchor_face):
            assert triangle_distance(center, triangles[face]) < 1e-12
        free = data["sources.free_moment"][active, env]
        origin, length = data["failure.position"][env], data["failure.radius"][env]
        total_force = force.sum(axis=0)
        total_moment = (np.cross(point - origin, force) + free).sum(axis=0)
        anchor_moment = (np.cross(anchor_point - origin, force) + free).sum(axis=0)
        np.testing.assert_allclose(total_force, data["failure.target_force"][env], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(total_moment, data["failure.target_moment"][env], rtol=1e-12, atol=1e-15)
        scale = np.linalg.norm(force, axis=1).sum()
        target = np.r_[total_force, total_moment / length]
        shifted_target = np.r_[total_force, anchor_moment / length]
        paths = dijkstra(graph, directed=False, indices=anchor_face)
        source_normals = source_normal / np.linalg.norm(source_normal, axis=1)[:, None]
        p = np.einsum("ni,ni->n", force, source_normals)
        t = np.linalg.norm(force - p[:, None] * source_normals, axis=1)
        source_cone_excess = t - mu * p
        # Every intrinsic region lies inside its Euclidean ball on the anchor's
        # connected component. A P1 facet normal is constant, so force-cone polar
        # bounds need every intersecting facet, without spatial sampling.
        force_normals, force_friction = [], []
        for center, support, coefficient, face in zip(anchor_point, radius, mu, anchor_face):
            allowance = 512 * np.finfo(float).eps * (np.linalg.norm(center) + support + np.linalg.norm(vertices, axis=1).max())
            selected = (components == components[face]) & np.array([triangle_distance(center, tri) <= support + allowance for tri in triangles])
            force_normals.append(normals[selected])
            force_friction.append(np.full(selected.sum(), coefficient))
        force_budget_scale = np.maximum(np.linalg.norm(force, axis=1),
            np.linalg.norm(data["sources.anchor_moment"][active, env], axis=1) / radius).sum()
        force_proof = continuous_force_certificate(np.concatenate(force_normals), np.concatenate(force_friction), total_force, force_budget_scale)
        contrasts = []
        for factor in (1.0, 1.5):
            positions, material_normals, flat_normals, friction = [], [], [], []
            for i, (center, face, support, coefficient) in enumerate(zip(anchor_point, anchor_face, radius, mu)):
                # An explicit path from anchor to its face centroid, along face
                # centroids through shared edge midpoints, then to the sample
                # lies on the surface. Its length is a geodesic upper bound.
                upper = paths[i, :, None] + np.linalg.norm(center - centers[face]) + center_distance
                upper[face] = np.linalg.norm(samples[face] - center, axis=1)
                selected = (components[:, None] == components[face]) & (upper < factor * support * (1 - 1e-10))
                selected_points = samples[selected]
                selected_normals = np.broadcast_to(normals[:, None], samples.shape)[selected]
                selected_points = np.concatenate((selected_points, center[None]))
                selected_normals = np.concatenate((selected_normals, normals[face, None]))
                positions.append(selected_points)
                material_normals.append(selected_normals)
                flat_normals.append(np.broadcast_to(source_normals[i], selected_normals.shape))
                friction.append(np.full(len(selected_points), coefficient))
            positions, material_normals, flat_normals, friction = map(np.concatenate,
                (positions, material_normals, flat_normals, friction))
            directions = np.concatenate((force, total_force[None]))
            physical = generators(positions, material_normals, friction, origin, length, directions)
            source_frame = generators(positions, flat_normals, friction, origin, length, directions)
            contrasts.append({"radius_multiplier": factor, "surface_points": len(positions),
                "physical_normals_force_only": fit(physical, target, 3, scale),
                "physical_normals_full_wrench": fit(physical, target, 6, scale),
                "physical_normals_anchor_application_wrench": fit(physical, shifted_target, 6, scale),
                "source_normals_full_wrench": fit(source_frame, target, 6, scale),
                "source_normals_anchor_application_wrench": fit(source_frame, shifted_target, 6, scale),
                "diagnostic_variant_only": factor != 1.0})
        assert not contrasts[0]["physical_normals_full_wrench"]["constructively_feasible"], "Positive construction contradicts the retained continuous certificate"
        rows.append({"environment": int(env), "native_status": int(data["failure.status"][env]),
            "original_continuous_certificate_retained": True,
            "source_radii_m": radius.tolist(), "source_friction": mu.tolist(),
            "source_application_to_anchor_distance_m": np.linalg.norm(point - anchor_point, axis=1).tolist(),
            "source_to_anchor_face_normal_angle_deg": np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", source_normals, normals[anchor_face]), -1, 1))).tolist(),
            "maximum_raw_source_cone_excess_n": source_cone_excess.max().item(),
            "independent_continuous_force_certificate": force_proof,
            "application_to_anchor_wrench_change_nm": (total_moment - anchor_moment).tolist(),
            "contrasts": contrasts})
        write_json(args.output / "progress.json", {"rows": rows})
    write_json(args.output / "result.json", {"passed": bool(rows), "state": actual,
        "original_receipt": file_id(args.run / "rejected-step.json"), "certificate": file_id(args.certificate),
        "original_start": file_id(args.run / "start.json"),
        "boundary_connected_components": int(component_count), "rows": rows,
        "analytic_force_certificate_positive_and_negative_controls_passed": True,
        "whole_goal_complete": False, "SCR_repair_admitted": False,
        "scope": "Force-only/full-wrench and geometry-normal/application/radius diagnostic contrasts on the exact captured first-failure states. Every physical candidate has an explicit surface path shorter than the stated radius and stays on the anchor's connected component; separate inner/outer boundaries are not mixed. LP generators satisfy actual circular Coulomb cones, so a verified positive equality is constructive on these sampled points. A failed LP alone never proves continuous infeasibility; only the unchanged original full-wrench case reuses its independently proven larger-support certificate. Source-normal, moved-application and1.5x-radius variants change the model and cannot admit SCR physical accuracy. No source, original failure, coefficient, solver tolerance or production engine is changed."})


if __name__ == "__main__":
    main()
