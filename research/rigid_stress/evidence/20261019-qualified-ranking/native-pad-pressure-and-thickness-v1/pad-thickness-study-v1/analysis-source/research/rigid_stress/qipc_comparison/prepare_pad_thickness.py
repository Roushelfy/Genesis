"""Rebuild inward wall thickness and its independent free-link contact compliance."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .compliant_plate_reference import condense, pad_quadrature
from .contact_qp import solve_contact_qp
from .p2_elastic import FreeElasticP2
from .prepare_inputs import make_mesh, mass_properties


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--thickness", type=float, required=True)
    cli.add_argument("--qp-solver-path", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert 0 < args.thickness <= 0.001
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "base_mesh": file_id(args.mesh), "thickness_m": args.thickness,
        "scope": "Fixed outer geometry, surface triangulation and through-wall layer count. Rebuild the inward normal offset, mass, stiffness and full regional compliance for this physical thickness."})
    with np.load(args.mesh) as saved:
        outer, inner, faces, unit = (saved[key] for key in ("outer_vertices", "inner_vertices", "mid_faces", "material_unit"))
        count = len(saved["vertices"])
    normal = (outer - inner) / 0.0005
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, rtol=0, atol=1e-12)
    assert count % len(outer) == 0
    layers = count // len(outer) - 1
    inner = outer - args.thickness * normal
    vertices, tetrahedra, boundary, volumes = make_mesh(outer, inner, faces, layers)
    assert volumes.min() > 0
    mesh_path = args.output / "mesh.npz"
    np.savez_compressed(mesh_path, vertices=vertices, tetrahedra=tetrahedra, surface_triangles=boundary,
        outer_vertices=outer, inner_vertices=inner, mid_vertices=(outer + inner) / 2,
        mid_faces=faces, material_unit=unit)
    model = FreeElasticP2(vertices, tetrahedra, 1e10, 0.3, 2000, column_faces=faces,
                         through_wall_ordering=True, cvxopt_factor=True)
    factor_seconds = perf_counter() - started
    nodes, interpolation, points, area, _ = pad_quadrature(model, len(outer), 0.003, 0)
    center = vertices.mean(axis=0)
    top = model.vertices[nodes, 2] > center[2]
    sign = np.where(top, -1.0, 1.0)
    gap = np.where(top, outer[:, 2].max() - model.vertices[nodes, 2],
                   model.vertices[nodes, 2] - outer[:, 2].min())
    rigid = np.column_stack((sign, sign * (model.vertices[nodes, 1] - center[1]) / 0.03,
                             -sign * (model.vertices[nodes, 0] - center[0]) / 0.03, top))
    before = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        compliance, reciprocity = condense(model, nodes, sign, journal)
    condensation_seconds = perf_counter() - before
    operator_path = args.output / "contact-operator.npz"
    np.savez_compressed(operator_path, nodes=nodes, compliance_m_per_n=compliance)
    load, stiffness = 0.1, 1e9
    scale = load / (stiffness * np.pi * 0.002**2)
    qp = solve_contact_qp(compliance * load / scale, gap / scale, rigid, np.array([0.0, 0.0, 0.0, 1.0]),
        np.empty((0, 6), dtype=int), args.qp_solver_path, interpolation=interpolation,
        round_limit=1, constraint_compliance=load / (scale * stiffness * area))
    assert qp.candidate_passed
    force = np.zeros_like(model.vertices)
    force[nodes, 2] = sign * qp.nodal_force * load
    stress, displacement, correction, residual = model.solve(force)
    equation_error = np.linalg.norm(residual) / np.linalg.norm(force)
    balance_error = np.linalg.norm(correction) / np.linalg.norm(force)
    assert equation_error <= 1e-7 and balance_error <= 1e-8
    baseline_path = args.output / "baseline.npz"
    np.savez_compressed(baseline_path, stress_pa=stress, forces_n=force, displacement_m=displacement,
        pressure_pa=qp.multipliers * load / area, points_m=points, area_m2=area,
        point_force_n=qp.multipliers * load, vertices_m=model.vertices, tetrahedra=tetrahedra)
    write_json(args.output / "result.json", {"passed": True, "thickness_m": args.thickness,
        "layers": layers, "mesh": file_id(mesh_path), "operator": file_id(operator_path),
        "baseline": file_id(baseline_path), "base_mesh": file_id(args.mesh),
        "factor_and_mesh_seconds": factor_seconds, "condensation_seconds": condensation_seconds,
        "reciprocity_relative_error": reciprocity, "equation_relative_error": equation_error.item(),
        "balance_correction_relative": balance_error.item(),
        "mass_properties": mass_properties(vertices, tetrahedra, volumes),
        "total_seconds": perf_counter() - started, "physical_reference_qualified": False,
        "whole_goal_complete": False})


if __name__ == "__main__":
    main()
