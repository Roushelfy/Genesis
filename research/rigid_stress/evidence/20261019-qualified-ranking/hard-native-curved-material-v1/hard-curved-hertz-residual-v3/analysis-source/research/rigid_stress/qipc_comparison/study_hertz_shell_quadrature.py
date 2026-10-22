"""Compare focused near-wall integration with uniform rules on saved shell loads."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .bilinear_normal_stress import point_tensor
from .check_native_hertz_recovery import matrix
from .p2_elastic import FreeElasticP2


def closest_barycentric(triangle, point):
    edges = triangle[1:] - triangle[0]
    local = np.linalg.solve(edges @ edges.T, edges @ (point - triangle[0]))
    bary = np.r_[1 - local.sum(), local]
    if bary.min() < 0:
        starts, ends = np.arange(3), np.roll(np.arange(3), -1)
        direction = triangle[ends] - triangle[starts]
        fraction = np.clip(np.einsum("ij,ij->i", point - triangle[starts], direction)
                           / np.einsum("ij,ij->i", direction, direction), 0, 1)
        candidates = triangle[starts] + fraction[:, None] * direction
        edge = int(np.argmin(np.linalg.norm(candidates - point, axis=1)))
        bary = np.zeros(3)
        bary[starts[edge]], bary[ends[edge]] = 1 - fraction[edge], fraction[edge]
    return bary


def integrate(cpu, center, frame, force, order, focused, selected=None, subdivisions=1):
    mesh = cpu.reference.mesh
    facets = mesh.boundary_facets()
    if selected is not None:
        keys = {tuple(sorted(face)) for face in selected}
        facets = facets[[tuple(sorted(face)) in keys for face in mesh.facets[:, facets].T]]
    faces = mesh.facets[:, facets].T
    triangles = mesh.p.T[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    owner_center = mesh.p.T[mesh.t.T[mesh.f2t[0, facets]]].mean(axis=1)
    normals[np.einsum("fi,fi->f", normals, owner_center - triangles[:, 0]) > 0] *= -1
    edge_indices = {tuple(sorted(pair)): index for index, pair in enumerate(mesh.edges.T)}
    scalar = cpu.reference.scalar_basis
    pairs = [(0, 1), (0, 2), (1, 2)]
    edge_nodes = np.array([[scalar.edge_dofs[0, edge_indices[tuple(sorted(face[[a, b]]))]]
                            for a, b in pairs] for face in faces])
    face_nodes = np.concatenate([scalar.nodal_dofs[0, faces], edge_nodes], axis=1)
    knots, weights = np.polynomial.legendre.leggauss(order)
    knots, weights = (knots + 1) / 2, weights / 2
    u, v = np.meshgrid(knots, knots, indexing="ij")
    quadrature_weights = weights[:, None] * weights[None]
    load = np.zeros_like(cpu.vertices)
    sample_count = focused_faces = 0
    for face, triangle, normal in zip(face_nodes, triangles, normals):
        if np.max(np.abs((triangle - center) @ frame[:, 2])) <= 1e-12:
            continue
        closest = closest_barycentric(triangle, center)
        distance = np.linalg.norm(closest @ triangle - center)
        diameter = max(np.linalg.norm(triangle[a] - triangle[b]) for a, b in pairs)
        rules = []
        if focused and distance < diameter / 4:
            focused_faces += 1
            for edge in range(3):
                endpoints = np.eye(3)[[edge, (edge + 1) % 3]]
                first, second = (endpoints - closest) @ triangle
                area = np.linalg.norm(np.cross(first, second))
                if area <= np.finfo(float).tiny:
                    continue
                for i_u in range(subdivisions):
                    for i_v in range(subdivisions):
                        radial, angular = (u + i_u) / subdivisions, (v + i_v) / subdivisions
                        length = np.linalg.norm((1 - angular)[..., None] * first + angular[..., None] * second, axis=2)
                        denominator = length * (1 - radial) + distance * radial
                        radius = distance * radial / denominator
                        derivative = distance * length / denominator**2
                        edge_bary = (1 - angular)[..., None] * endpoints[0] + angular[..., None] * endpoints[1]
                        bary = closest + radius[..., None] * (edge_bary - closest)
                        rules.append((bary, area * radius * derivative * quadrature_weights / subdivisions**2))
        else:
            bary = np.stack([1 - u, u * (1 - v), u * v], axis=2)
            area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]))
            rules.append((bary, area * u * quadrature_weights))
        for bary, measure in rules:
            bary, measure = bary.reshape(-1, 3), measure.reshape(-1)
            positions = bary @ triangle
            local = (positions - center) @ frame
            assert local[:, 2].min() >= -1e-12
            local[:, 2] = np.maximum(local[:, 2], 0)
            tensor = frame @ (force * point_tensor(local)) @ frame.T
            traction = -np.einsum("qij,j->qi", tensor, normal)
            shape = np.column_stack([bary * (2 * bary - 1),
                                     4 * bary[:, 0] * bary[:, 1],
                                     4 * bary[:, 0] * bary[:, 2],
                                     4 * bary[:, 1] * bary[:, 2]])
            load[face] += shape.T @ (measure[:, None] * traction)
            sample_count += len(measure)
    return load, {"samples": sample_count, "focused_faces": focused_faces}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fields", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"wrench": 1e-4, "integration_field": 1e-3, "reference_field_change": 1e-5}
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
               "fields": file_id(args.fields), "mesh": file_id(args.mesh), "targets": targets,
               "scope": "Independent CPU quadrature comparison of the existing point-traction boundary correction. Shared prescribed Hertz particulars and unchanged shell volume. Count evaluations and field changes before a native port; CPU times are not production throughput."})
    with np.load(args.fields) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(args.mesh) as saved:
        cpu = FreeElasticP2(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        selected = saved["surface_triangles"]
    permutation = cpu.reference.native_dof_permutation(data["material_nodes_m"]).reshape(-1, 3)[:, 0] // 3
    rows, fields, loads = [], [], []
    for env in range(4):
        native_load = np.empty_like(data["regular_force_n"][env])
        native_load[permutation] = data["regular_force_n"][env]
        native_regular, _, _, _ = cpu.solve(native_load)
        particular = matrix(data["corner_tensor_pa"][env]) - native_regular
        current = []
        for name, order, focused in [("focused32", 32, True), ("focused24", 24, True),
                                     ("selected16", 16, False), ("uniform16", 16, False),
                                     ("focused12", 12, True), ("focused16x2", 16, True)]:
            center, frame, force = data["centers_m"][env], data["frames"][env], data["force_n"][env]
            load, counts = integrate(cpu, center, frame, force, order, focused,
                                     selected=selected if name == "selected16" else None,
                                     subdivisions=2 if name == "focused16x2" else 1)
            regular, _, _, _ = cpu.solve(load)
            combined = regular + particular
            force_error = np.linalg.norm(load.sum(axis=0) - force * frame[:, 2]) / force
            source_moment_error = np.linalg.norm(np.cross(cpu.vertices - center, load).sum(axis=0)) / (force * data["axes_m"][env, 0])
            moment_error = np.linalg.norm(np.cross(cpu.vertices, load).sum(axis=0)
                                          - np.cross(center, force * frame[:, 2])) / (force * data["axes_m"][env, 0])
            fields.append(combined)
            loads.append(load)
            if name == "focused32":
                reference = combined
            difference = float(np.linalg.norm(combined - reference) / np.linalg.norm(reference))
            row = {"env": env, "rule": name, **counts, "wrench_error": float(max(force_error, moment_error)),
                   "source_center_moment_error": float(source_moment_error),
                   "corner_field_relative": difference,
                   "native16_load_relative": float(np.linalg.norm(load - native_load) / np.linalg.norm(native_load))}
            current.append(row)
        rows.extend(current)
        write_json(args.output / "progress.json", {"rows": rows})
    np.savez_compressed(args.output / "comparison.npz", corners_pa=fields, nodal_force_n=loads)
    report = {"passed": False, "rows": rows, "targets": targets,
              "asset_selected_faces": len(selected),
              "complete_material_boundary_faces": len(cpu.reference.mesh.boundary_facets()),
              "whole_goal_complete": False}
    write_json(args.output / "result.json", report)
    assert all(row["corner_field_relative"] < targets["reference_field_change"] for row in rows if row["rule"] == "focused24")
    assert all(row["wrench_error"] < 1e-6 for row in rows if row["rule"] == "focused32")
    assert all(row["native16_load_relative"] < 1e-7 for row in rows if row["rule"] == "selected16")
    report["passed"] = True
    report["candidates"] = {name: all(row["wrench_error"] < targets["wrench"] and
                                        row["corner_field_relative"] < targets["integration_field"]
                                        for row in rows if row["rule"] == name)
                            for name in ["focused12", "focused16x2"]}
    write_json(args.output / "result.json", report)


if __name__ == "__main__":
    main()
