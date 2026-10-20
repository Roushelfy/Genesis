"""Measure separately budgeted nearby SCR loads on immutable first-failure states.

The reference is an explicitly declared, unconstrained, source-prior surface
transfer. It preserves each original wrench but need not satisfy the cones.
Neither this transfer nor the sampled optimization is a physical reference.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from .artifacts import file_id, runtime_identity, write_json
from .attribute_native_wrench_conflict import surface_graph
from .compare_distributed_pad_transfer import field_norm
from .contact_fem_exact_columns import surface_quadrature
from .p2_elastic import FreeElasticP2, quadratic_triangle_weights


@dataclass
class Source:
    position: np.ndarray
    normal: np.ndarray
    nodes: np.ndarray
    shape: np.ndarray
    weight: np.ndarray
    wrench: np.ndarray
    target: np.ndarray
    baseline: np.ndarray
    force_scale: float
    radius: float
    friction: float
    anchor: np.ndarray
    path_upper: np.ndarray


def source_transfer(vertices, triangles, normals, nodes, graph, centers, components,
                    anchor, face, force, point, free_moment, radius, friction, count):
    bary, quadrature = surface_quadrature(count)
    samples = np.einsum("qc,fci->fqi", bary, triangles)
    upper = dijkstra(graph, directed=False, indices=face)[:, None]
    upper = upper + np.linalg.norm(anchor - centers[face]) + np.linalg.norm(samples - centers[:, None], axis=-1)
    upper[face] = np.linalg.norm(samples[face] - anchor, axis=1)
    selected = (components[:, None] == components[face]) & (upper < radius * (1 - 1e-10))
    facet, sample = np.nonzero(selected)
    assert len(facet) >= 3, "Insufficient declared sampling; do not silently enlarge support"
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    distance = upper[selected] / radius
    weight = area[facet] * quadrature[sample] * np.exp(-0.5 * (distance / 0.45)**2) * (1 - distance**2)**2
    weight /= weight.sum()
    position = samples[selected]
    arm = (position - anchor) / radius
    skew = np.cross(arm[:, :, None].transpose(0, 2, 1), np.eye(3)[None]).transpose(0, 2, 1)
    wrench = np.concatenate((np.tile(np.eye(3), (1, len(facet))), skew.transpose(1, 0, 2).reshape(3, -1)))
    scale = np.linalg.norm(force)
    assert scale > 0
    target = np.r_[force, (np.cross(point - anchor, force) + free_moment) / radius] / scale
    prior = (weight[:, None] * force / scale).reshape(-1)
    weighted = wrench * np.repeat(weight, 3)[None]
    baseline = prior + weighted.T @ np.linalg.solve(weighted @ wrench.T, target - wrench @ prior)
    np.testing.assert_allclose(wrench @ baseline, target, rtol=0, atol=1e-11)
    shape = quadratic_triangle_weights(bary[sample])
    np.testing.assert_allclose(np.einsum("qc,qci->qi", shape, vertices[nodes[facet]]), position, rtol=0, atol=1e-13)
    return Source(position, normals[facet], nodes[facet], shape, weight, wrench, target,
                  baseline.reshape(-1, 3), scale, radius, friction, anchor, upper[selected])


def nearby(source, budget, clarabel, *, minimax=False):
    root = np.repeat(np.sqrt(source.weight), 3)
    target = source.baseline.reshape(-1) / root
    n = len(target)
    axis = np.eye(3)[np.argmin(np.abs(source.normal), axis=1)]
    first = np.cross(source.normal, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    second = np.cross(source.normal, first)
    blocks = np.stack((source.friction * source.normal, first, second), axis=1)
    cones = [clarabel.SecondOrderConeT(3) for _ in source.weight]
    cone_matrix = sparse.block_diag(list(-blocks), format="csc") @ sparse.diags(root)
    wrench = source.wrench * root[None]
    matrices, right = [cone_matrix], [np.zeros(n)]
    if minimax:
        matrices[0] = sparse.hstack((cone_matrix, sparse.csc_matrix((n, 1))), format="csc")
    if budget == 0 and not minimax:
        matrices.append(sparse.csc_matrix(wrench))
        right.append(source.target)
        cones.append(clarabel.ZeroConeT(6))
    else:
        for start in (0, 3):
            matrix = np.vstack((np.zeros(n), -wrench[start:start + 3]))
            if minimax:
                matrix = np.column_stack((matrix, [-1., 0, 0, 0]))
            matrices.append(sparse.csc_matrix(matrix))
            right.append(np.r_[0 if minimax else budget, -source.target[start:start + 3]])
            cones.append(clarabel.SecondOrderConeT(4))
    a = sparse.vstack(matrices, format="csc")
    b = np.concatenate(right)
    p = sparse.csc_matrix((n + 1, n + 1)) if minimax else sparse.eye(n, format="csc")
    q = np.r_[np.zeros(n), 1.] if minimax else -target
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = 200
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-11
    settings.iterative_refinement_abstol = settings.iterative_refinement_reltol = 1e-13
    started = perf_counter()
    solution = clarabel.DefaultSolver(p, q, a, b, cones, settings).solve()
    elapsed = perf_counter() - started
    row = {"status": str(solution.status), "iterations": solution.iterations, "seconds": elapsed,
           "accepted": False, "continuous_infeasibility_claim": False}
    if solution.status != clarabel.SolverStatus.Solved:
        return None, row
    x, z, slack = np.array(solution.x), np.array(solution.z), np.array(solution.s)
    primal = np.linalg.norm(a @ x + slack - b) / (1 + np.linalg.norm(b))
    dual = np.linalg.norm(p @ x + q + a.T @ z) / (1 + np.linalg.norm(q))
    gap = abs(solution.obj_val - solution.obj_val_dual) / (1 + abs(solution.obj_val))
    row.update(primal_residual=primal.item(), dual_residual=dual.item(), relative_gap=float(gap))
    if max(primal, dual, gap) > 1e-8:
        return None, row
    force = (root * x[:n]).reshape(-1, 3)
    # Construct actual compressive cone loads, removing only numerical cone
    # violations, then independently remeasure the actual corrected wrenches.
    compression = np.maximum(np.einsum("qi,qi->q", force, source.normal), 0)
    tangent = force - np.einsum("qi,qi->q", force, source.normal)[:, None] * source.normal
    magnitude = np.linalg.norm(tangent, axis=1)
    cap = source.friction * compression * (1 - 32 * np.finfo(float).eps)
    tangent *= np.minimum(1, np.divide(cap, magnitude, out=np.ones_like(cap), where=magnitude > 0))[:, None]
    projected = compression[:, None] * source.normal + tangent
    cone_projection = np.linalg.norm(projected - force)
    force = projected
    delta = source.wrench @ force.reshape(-1) - source.target
    correction = max(np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:]))
    allowed = x[-1] if minimax else budget
    pressure = np.einsum("qi,qi->q", force, source.normal)
    excess = np.linalg.norm(force - pressure[:, None] * source.normal, axis=1) - source.friction * pressure
    accepted = max(excess.max(), -pressure.min()) <= 1e-13 and correction <= allowed + 1e-9 and cone_projection <= 1e-8
    row.update(accepted=bool(accepted), sampled_minimax_budget=float(x[-1]) if minimax else None,
               actual_normalized_force_change=np.linalg.norm(delta[:3]).item(),
               actual_normalized_moment_change=np.linalg.norm(delta[3:]).item(),
               force_change_n=(source.force_scale * delta[:3]).tolist(),
               anchor_moment_change_nm=(source.force_scale * source.radius * delta[3:]).tolist(),
               numerical_cone_projection_n=float(cone_projection * source.force_scale),
               minimum_compression_n=float(pressure.min() * source.force_scale),
               maximum_cone_excess_n=float(excess.max() * source.force_scale),
               unsigned_point_force_change_n=float(source.force_scale * np.linalg.norm(force - source.baseline, axis=1).sum()),
               weighted_point_change_relative=float(np.linalg.norm((force - source.baseline) / np.sqrt(source.weight[:, None]))))
    return (force * source.force_scale if accepted else None), row


def scatter(model, sources, forces):
    load = np.zeros_like(model.vertices)
    for source, force in zip(sources, forces):
        np.add.at(load, source.nodes.reshape(-1), (source.shape[:, :, None] * force[:, None]).reshape(-1, 3))
    return load


def raw_peaks(stress, vertices, tets):
    deviator = stress - np.trace(stress, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    invariants = (("mises", np.sqrt(1.5 * (deviator**2).sum(axis=(-2, -1)))),
                  ("tension", np.linalg.eigvalsh(stress)[..., -1]))
    result = {}
    for name, values in invariants:
        element, corner = np.unravel_index(values.argmax(), values.shape)
        result[name] = {"pa": values[element, corner].item(), "position_m": vertices[tets[element, corner]].tolist()}
    result["spectral_norm_max_pa"] = np.abs(np.linalg.eigvalsh(stress)).max().item()
    return result


def resultants(stresses, vertices, tets, faces, count=9):
    """Exact line integration on actual tetrahedra; no assumed layer/tet ordering."""
    columns = len(vertices) // 3
    assert 3 * columns == len(vertices)
    layered = vertices.reshape(3, columns, 3)
    radial = np.linalg.norm(layered, axis=2)
    outer = layered[radial.argmax(axis=0), np.arange(columns)]
    inner = layered[radial.argmin(axis=0), np.arange(columns)]
    faces = faces % columns
    bary, weights = surface_quadrature(count)
    low = np.einsum("qc,fci->fqi", bary, inner[faces]).reshape(-1, 3)
    high = np.einsum("qc,fci->fqi", bary, outer[faces]).reshape(-1, 3)
    vector = high - low
    thickness = np.linalg.norm(vector, axis=1)
    normal = vector / thickness[:, None]
    first = np.cross(normal, np.eye(3)[np.abs(normal).argmin(axis=1)])
    first /= np.linalg.norm(first, axis=1)[:, None]
    basis = np.stack((first, np.cross(normal, first)), axis=-1)
    tri = ((outer + inner) / 2)[faces]
    area = (np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)[:, None] / 2 * weights).reshape(-1)
    inv = np.linalg.inv((vertices[tets[:, 1:]] - vertices[tets[:, :1]]).transpose(0, 2, 1))
    a3 = np.einsum("eij,qej->qei", inv, low[:, None] - vertices[tets[:, 0]][None])
    b3 = np.einsum("eij,qj->qei", inv, vector)
    a = np.concatenate((1 - a3.sum(axis=-1, keepdims=True), a3), axis=-1)
    b = np.concatenate((-b3.sum(axis=-1, keepdims=True), b3), axis=-1)
    crossing = np.divide(-a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-12)
    lower = np.maximum(np.where(b > 1e-12, crossing, -np.inf).max(axis=-1), 0)
    upper = np.minimum(np.where(b < -1e-12, crossing, np.inf).min(axis=-1), 1)
    valid = (upper > lower + 1e-12) & ~((np.abs(b) <= 1e-12) & (a < -1e-10)).any(axis=-1)
    width = np.where(valid, upper - lower, 0)
    np.testing.assert_allclose(width.sum(axis=1), 1, rtol=0, atol=1e-9)
    ray, element = np.nonzero(valid)
    mid, half = (lower[valid] + upper[valid]) / 2, width[valid] / 2
    membrane = np.zeros((len(stresses), len(low), 2, 2))
    bending = np.zeros_like(membrane)
    for offset in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        depth = mid + offset * half
        shape = a[ray, element] + b[ray, element] * depth[:, None]
        tensor = np.einsum("qc,sqcij->sqij", shape, stresses[:, element])
        tangent = basis[ray].transpose(0, 2, 1)[None] @ tensor @ basis[ray][None]
        for index in range(len(stresses)):
            np.add.at(membrane[index], ray, tangent[index] * (half * thickness[ray])[:, None, None])
            np.add.at(bending[index], ray, tangent[index] * (half * thickness[ray]**2 * (0.5 - depth))[:, None, None])
    return membrane, bending, area


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--certificate", type=Path, required=True)
    cli.add_argument("--qp-solver-path", type=Path, required=True)
    cli.add_argument("--quadrature", type=int, choices=(9, 25), required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.qp_solver_path))
    import clarabel

    assert clarabel.__version__ == "0.11.1"
    receipt = json.loads((args.run / "rejected-step.json").read_text())
    proof = json.loads(args.certificate.read_text())
    state = file_id(args.run / "rejected-state.npz")
    assert proof["passed"] and all(state[k] == receipt["state"][k] == proof["state"][k] for k in ("bytes", "sha256"))
    options = receipt["stress_options"]
    mesh = Path(options["mesh"])
    budgets = [0., 0.001, 0.01, 0.1]
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "state": state, "certificate": file_id(args.certificate), "mesh": file_id(mesh),
        "quadrature": args.quadrature, "budgets": budgets, "solver_version": clarabel.__version__,
        "budget_scope": "Per source: norm(delta F)/norm(F) and norm(delta M_anchor)/(R norm(F)); no radius/friction changes.",
        "numerical_acceptance": {"kkt": 1e-8, "actual_budget_allowance": 1e-9, "cone_rounding": 1e-13},
        "inertia_scope": "Captured omega/applied loads are unavailable. Complete correction response is exact for this linear transfer at any unchanged inertial state, including the changed consistent-mass equilibrium projection. Baseline/candidate absolute stresses are explicitly contact-only free-link responses, not the actual dynamic total stress."})
    with np.load(mesh) as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    with np.load(args.run / "rejected-state.npz") as saved:
        data = {k: saved[k] for k in saved.files}
    setup = perf_counter()
    model = FreeElasticP2(vertices, tets, options["young"], options["poisson"], options["density"], cvxopt_factor=True)
    setup_seconds = perf_counter() - setup
    distance, permutation = cKDTree(model.vertices).query(data["mesh.vertices"])
    assert distance.max() < 1e-13 and len(np.unique(permutation)) == len(model.vertices)
    nodes = permutation[data["mesh.surface_nodes"]]
    faces = nodes[:, :3]
    assert faces.max() < len(vertices)
    triangles = model.vertices[faces]
    opposite = data["mesh.vertices"][data["mesh.surface_opposite_vertices"]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    normals *= np.where(np.einsum("fi,fi->f", normals, opposite - triangles[:, 0]) >= 0, 1, -1)[:, None]
    graph, centers = surface_graph(model.vertices, faces)
    _, components = connected_components(graph, directed=False)
    signed_volumes = np.linalg.det(vertices[tets[:, 1:]] - vertices[tets[:, :1]]) / 6
    # Original Franka assets retain mixed tetrahedral orientations. Physical
    # integration uses |det J|; preserve vertex ordering used by stress samples.
    volumes = np.abs(signed_volumes)
    assert volumes.min() > 0
    rows = []
    for env in np.flatnonzero(data["failure.status"]):
        assert next(r for r in proof["rows"] if r["environment"] == env)["independent_infeasibility_proven"]
        active = data["contacts.valid"][:, env].astype(bool)
        source_indices = np.flatnonzero(active)
        sources = [source_transfer(model.vertices, triangles, normals, nodes, graph, centers, components,
            data["contacts.center"][i, env], data["contacts.anchor_face"][i, env],
            data["contacts.force"][i, env], data["contacts.position"][i, env], data["sources.free_moment"][i, env],
            data["contacts.radius"][i, env], data["contacts.friction"][i, env], args.quadrature) for i in source_indices]
        minimax = [nearby(s, None, clarabel, minimax=True)[1] for s in sources]
        baseline = scatter(model, sources, [s.baseline * s.force_scale for s in sources])
        expected_force = sum(s.force_scale * s.target[:3] for s in sources)
        expected_moment = sum(s.force_scale * (s.radius * s.target[3:] + np.cross(s.anchor, s.target[:3])) for s in sources)
        scale = sum(s.force_scale for s in sources)
        np.testing.assert_allclose(baseline.sum(axis=0), expected_force, rtol=0, atol=1e-11 * scale)
        np.testing.assert_allclose(np.cross(model.vertices, baseline).sum(axis=0), expected_moment,
                                   rtol=0, atol=1e-11 * scale * np.linalg.norm(vertices, axis=1).max())
        base_stress, base_displacement, base_correction, base_residual = model.solve(baseline)
        base_norm = field_norm(base_stress, volumes)
        fields, names = [base_stress], ["baseline"]
        source_rows = []
        for i, source in zip(source_indices, sources):
            source_rows.append({"source_index": int(i), "force_scale_n": float(source.force_scale),
                "radius_m": float(source.radius), "friction": float(source.friction),
                "points": len(source.weight), "maximum_surface_path_m": float(source.path_upper.max()),
                "minimum_prior_weight": float(source.weight.min()),
                "original_force_n": (source.force_scale * source.target[:3]).tolist(),
                "original_anchor_moment_nm": (source.force_scale * source.radius * source.target[3:]).tolist()})
        budget_rows = []
        for budget in budgets:
            pairs = [nearby(s, budget, clarabel) for s in sources]
            row = {"budget": budget, "sources": [pair[1] for pair in pairs], "accepted": all(pair[1]["accepted"] for pair in pairs)}
            if row["accepted"]:
                assert budget > 0, "A strict constructive result contradicts the retained continuous certificate"
                load = scatter(model, sources, [pair[0] for pair in pairs])
                stress, displacement, correction, residual = model.solve(load)
                delta, _, delta_correction, delta_residual = model.solve(load - baseline)
                np.testing.assert_allclose(correction - base_correction, delta_correction, rtol=1e-8, atol=1e-13)
                linear_error = field_norm(stress - base_stress - delta, volumes) / base_norm
                equation = np.linalg.norm(residual) / np.linalg.norm(load - correction)
                delta_equation = np.linalg.norm(delta_residual) / np.linalg.norm(load - baseline - delta_correction)
                assert max(linear_error, equation, delta_equation) <= 1e-7
                path = args.output / f"env{env}-budget{budget:g}.npz"
                np.savez_compressed(path, vertices=vertices, tetrahedra=tets, stress_pa=stress,
                    baseline_stress_pa=base_stress, delta_stress_pa=delta, forces_n=load,
                    baseline_forces_n=baseline, equilibrium_correction_n=correction,
                    delta_equilibrium_correction_n=delta_correction, displacement_p2_m=displacement,
                    source_point_forces_n=np.concatenate([pair[0] for pair in pairs]))
                delta_load = load - baseline
                row.update(field=file_id(path), complete_stress_rms_change_pa=float(field_norm(delta, volumes) / np.sqrt(volumes.sum())),
                    complete_stress_relative_change=float(field_norm(delta, volumes) / base_norm),
                    total_force_change_n=delta_load.sum(axis=0).tolist(),
                    total_origin_moment_change_nm=np.cross(model.vertices, delta_load).sum(axis=0).tolist(),
                    equilibrium_correction_norm_n=float(np.linalg.norm(delta_correction)),
                    linear_superposition_error=float(linear_error), equation_relative_residual=float(equation),
                    delta_equation_relative_residual=float(delta_equation), contact_only_raw_peaks=raw_peaks(stress, vertices, tets),
                    delta_raw_peaks=raw_peaks(delta, vertices, tets))
                fields.append(stress)
                names.append(f"budget{budget:g}")
            budget_rows.append(row)
            write_json(args.output / "progress.json", {"environment": int(env), "minimax": minimax, "budgets": budget_rows})
        membrane, bending, area = resultants(np.stack(fields), vertices, tets, faces)
        resultant_rows = []
        for label, values, unit in (("membrane", membrane, "N/m"), ("bending", bending, "N")):
            norm = np.sqrt(np.einsum("qij,qij,q->", values[0], values[0], area) / area.sum())
            for i in range(1, len(fields)):
                error = np.sqrt(np.einsum("qij,qij,q->", values[i] - values[0], values[i] - values[0], area) / area.sum())
                resultant_rows.append({"variant": names[i], "metric": label, "unit": unit,
                    "absolute_rms_change": float(error), "baseline_rms": float(norm), "relative_change": float(error / norm)})
        path = args.output / f"env{env}-baseline.npz"
        np.savez_compressed(path, vertices=vertices, tetrahedra=tets, stress_pa=base_stress, forces_n=baseline,
            equilibrium_correction_n=base_correction, displacement_p2_m=base_displacement,
            membrane=membrane, bending=bending, area_m2=area, variant_names=np.array(names),
            source_offsets=np.r_[0, np.cumsum([len(s.weight) for s in sources])],
            source_point_forces_n=np.concatenate([s.baseline * s.force_scale for s in sources]),
            source_positions_m=np.concatenate([s.position for s in sources]),
            source_normals=np.concatenate([s.normal for s in sources]),
            source_weights=np.concatenate([s.weight for s in sources]))
        base_equation = np.linalg.norm(base_residual) / np.linalg.norm(baseline - base_correction)
        assert base_equation <= 1e-7
        rows.append({"environment": int(env), "sources": source_rows, "sampled_minimax": minimax,
            "baseline": file_id(path), "baseline_contact_only_raw_peaks": raw_peaks(base_stress, vertices, tets),
            "baseline_equation_relative_residual": float(base_equation),
            "baseline_complete_stress_rms_pa": float(base_norm / np.sqrt(volumes.sum())),
            "budgets": budget_rows, "resultants": resultant_rows})
    write_json(args.output / "result.json", {"passed": bool(rows), "rows": rows, "setup_seconds": setup_seconds,
        "negative_orientation_tetrahedra": int(np.count_nonzero(signed_volumes < 0)),
        "whole_goal_complete": False, "production_approximation_admitted": False,
        "scope": "Sampled circular-cone nearest-prior loads on original intrinsic supports. Strict input remains rejected. Minimax is a sampled diagnostic, not a continuous minimum. Budget grid is predeclared sensitivity hypotheses, not recommended defaults. Full linear correction includes consistent-mass equilibrium correction; absolute baseline/candidate peaks omit unchanged unrecorded inertial loads. Delta VM maximum and delta spectral maximum bound any fixed-state change of global VM and tensile maxima respectively, but do not bound the change of peak location or continuum discretization error."})


if __name__ == "__main__":
    main()
