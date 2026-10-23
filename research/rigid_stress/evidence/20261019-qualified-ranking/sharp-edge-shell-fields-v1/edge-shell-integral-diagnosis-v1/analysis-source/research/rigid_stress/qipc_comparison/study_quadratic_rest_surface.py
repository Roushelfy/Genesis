"""Test normal-constrained quadratic rest geometry before changing any native model."""

import argparse
from itertools import combinations, product
from pathlib import Path

import numpy as np

from ..mechanics import shell_mesh
from .artifacts import file_id, runtime_identity, write_json


def egg_geometry(unit):
    scale = 1 - 0.18 * unit[:, 2]
    return unit * np.column_stack([0.022 * scale, 0.022 * scale, np.full(len(unit), 0.03)])


def implicit(points, shape):
    if shape in ["sphere", "ellipsoid"]:
        axes = np.array([0.02] * 3) if shape == "sphere" else np.array([0.012, 0.015, 0.009])
        gradient = 2 * points / axes**2
        hessian = np.broadcast_to(np.diag(2 / axes**2), (len(points), 3, 3)).copy()
        value = (points**2 / axes**2).sum(axis=1) - 1
    else:
        x, y, z = points.T
        scale = 1 - 0.18 * z / 0.03
        radial = x*x + y*y
        value = radial / (0.022 * scale)**2 + (z / 0.03)**2 - 1
        gradient = np.column_stack([2*x / (0.022*scale)**2, 2*y / (0.022*scale)**2,
                                   2*0.18*radial / (0.022**2*0.03*scale**3) + 2*z/0.03**2])
        hessian = np.zeros((len(points), 3, 3))
        hessian[:, 0, 0] = hessian[:, 1, 1] = 2 / (0.022*scale)**2
        hessian[:, 0, 2] = hessian[:, 2, 0] = 4*0.18*x / (0.022**2*0.03*scale**3)
        hessian[:, 1, 2] = hessian[:, 2, 1] = 4*0.18*y / (0.022**2*0.03*scale**3)
        hessian[:, 2, 2] = 6*0.18**2*radial / (0.022**2*0.03**2*scale**4) + 2/0.03**2
    return value, gradient, hessian


def exact_surface(points, shape):
    parameter = np.ones(len(points))
    for _ in range(8):
        value, gradient, _ = implicit(parameter[:, None] * points, shape)
        parameter -= value / np.einsum("ni,ni->n", gradient, points)
    projected = parameter[:, None] * points
    value, gradient, hessian = implicit(projected, shape)
    assert np.max(np.abs(value)) < 1e-12
    magnitude = np.linalg.norm(gradient, axis=1)
    normal = gradient / magnitude[:, None]
    projection = np.eye(3) - normal[:, :, None] * normal[:, None, :]
    tensor = projection @ hessian @ projection / magnitude[:, None, None]
    return projected, normal, tensor


def quadratic_basis(bary):
    dimension = bary.shape[1] - 1
    derivatives = np.vstack([-np.ones(dimension), np.eye(dimension)])
    edges = list(combinations(range(dimension + 1), 2))
    values = [bary * (2*bary - 1)]
    gradients = [(4*bary - 1)[:, :, None] * derivatives]
    hessian = [4 * np.einsum("ni,nj->nij", derivatives, derivatives)]
    values.append(np.column_stack([4*bary[:, i]*bary[:, j] for i, j in edges]))
    gradients.append(np.stack([4*(bary[:, i, None]*derivatives[j] + bary[:, j, None]*derivatives[i])
                               for i, j in edges], axis=1))
    hessian.append(np.stack([4*(np.outer(derivatives[i], derivatives[j]) + np.outer(derivatives[j], derivatives[i]))
                             for i, j in edges]))
    return np.concatenate(values, axis=1), np.concatenate(gradients, axis=1), np.concatenate(hessian)


def edge_midpoints(vertices, edges, normals):
    first, second = vertices[edges[:, 0]], vertices[edges[:, 1]]
    tangent = second - first
    constraints = normals[edges]
    rhs = np.column_stack([-np.einsum("ni,ni->n", constraints[:, 0], tangent),
                            np.einsum("ni,ni->n", constraints[:, 1], tangent)]) / 4
    correction = np.einsum("nij,nj->ni", np.linalg.pinv(constraints, rcond=1e-12), rhs)
    residual = np.einsum("nij,nj->ni", constraints, correction) - rhs
    assert np.max(np.linalg.norm(residual, axis=1) / np.linalg.norm(tangent, axis=1)) < 1e-10
    return (first + second) / 2 + correction


def surface_field(nodes, bary):
    values, gradients, hessian = quadratic_basis(bary)
    points = np.einsum("qn,fni->fqi", values, nodes)
    tangent = np.einsum("qna,fni->fqia", gradients, nodes)
    area = np.cross(tangent[..., 0], tangent[..., 1])
    normals = area / np.linalg.norm(area, axis=2)[..., None]
    second = np.einsum("nab,fni->fiab", hessian, nodes)
    curvature = -np.einsum("fqi,fiab->fqab", normals, second)
    inverse = np.linalg.inv(tangent.swapaxes(-1, -2) @ tangent)
    tensor = tangent @ inverse @ curvature @ inverse @ tangent.swapaxes(-1, -2)
    return points, normals, tensor, np.linalg.norm(area, axis=2)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"normal_degrees": 1.0, "curvature_rms_relative": 0.05, "edge_normal_jump_degrees": 1.0,
               "sampled_jacobian_ratio": 0.2}
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False),
        "frame": file_id(args.frame), "targets": targets,
        "scope": "Geometry-only candidate: shared quadratic edge nodes constrained by endpoint normals. Compare exact sphere/ellipsoid/egg and estimated normals, preserve original vertices, sample through-wall curved tetrahedron Jacobians. No native geometry change, elastic solve, pressure qualification or collision surface equivalence is asserted. Normal/curvature errors and derivative jumps determine whether this reconstruction merits a mechanical test."})
    bary = np.array([[i, j, 8-i-j] for i in range(9) for j in range(9-i)]) / 8
    vertex_bary = np.eye(3)
    rows, arrays = [], {}
    with np.load(args.frame) as saved:
        center, inward = saved["hertz_centers_m"][0], saved["hertz_frames"][0, :, 2]
    for level in [1, 2, 3]:
        xyz, tets, faces, metadata = shell_mesh(level, 2, 0.0005)
        count = metadata.outer_surface_vertices
        original = xyz[:count]
        unit = np.column_stack([original[:, :2] / (0.022*(1-0.18*original[:, 2:3]/0.03)), original[:, 2]/0.03])
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1, rtol=1e-12)
        edges, inverse = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1),
                                   axis=0, return_inverse=True)
        for shape in ["sphere", "ellipsoid", "egg"]:
            vertices = (unit*0.02 if shape == "sphere" else unit*np.array([0.012, 0.015, 0.009])
                        if shape == "ellipsoid" else egg_geometry(unit))
            _, analytic, _ = exact_surface(vertices, shape)
            triangles = vertices[faces]
            area = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            assert np.all(np.einsum("ni,ni->n", area, triangles.mean(axis=1)) > 0)
            estimated = np.zeros_like(vertices)
            np.add.at(estimated, faces.reshape(-1), np.repeat(area, 3, axis=0))
            estimated /= np.linalg.norm(estimated, axis=1)[:, None]
            for kind, normal in [("analytic", analytic), ("estimated", estimated)]:
                midpoints = edge_midpoints(vertices, edges, normal)
                nodes = np.concatenate([vertices[faces], midpoints[inverse.reshape(-1, 3)]], axis=1)
                positions, normals, tensors, jacobian = surface_field(nodes, bary)
                projected, exact_normal, exact_tensor = exact_surface(positions.reshape(-1, 3), shape)
                normal_angle = np.degrees(np.arccos(np.clip(np.einsum("ni,ni->n", normals.reshape(-1, 3), exact_normal), -1, 1)))
                curvature_error = np.linalg.norm(tensors.reshape(-1, 3, 3)-exact_tensor, axis=(1, 2))
                expected_scale = np.linalg.norm(exact_tensor, axis=(1, 2))
                _, at_vertices, _, _ = surface_field(nodes, vertex_bary)
                constraint_error = np.linalg.norm(at_vertices-normal[faces], axis=2).max()
                assert constraint_error < 1e-9
                edge_bary = np.zeros((3, 5, 3))
                for edge, (first, second) in enumerate(combinations(range(3), 2)):
                    edge_bary[edge, :, first] = 1 - np.linspace(0, 1, 5)
                    edge_bary[edge, :, second] = np.linspace(0, 1, 5)
                _, edge_normal, _, _ = surface_field(nodes, edge_bary.reshape(-1, 3))
                edge_normal = edge_normal.reshape(len(faces), 3, 5, 3)
                for edge, (first, second) in enumerate(combinations(range(3), 2)):
                    reverse = faces[:, first] > faces[:, second]
                    edge_normal[reverse, edge] = edge_normal[reverse, edge, ::-1]
                order = np.argsort(inverse, kind="stable")
                paired = edge_normal.reshape(-1, 5, 3)[order].reshape(len(edges), 2, 5, 3)
                jumps = np.degrees(np.arccos(np.clip(np.einsum("eqi,eqi->eq", paired[:, 0], paired[:, 1]), -1, 1)))
                row = {"shape": shape, "level": level, "normal_input": kind, "faces": len(faces),
                       "maximum_normal_angle_degrees": float(normal_angle.max()),
                       "rms_normal_angle_degrees": float(np.sqrt(np.mean(normal_angle**2))),
                       "curvature_rms_relative": float(np.linalg.norm(curvature_error)/np.linalg.norm(expected_scale)),
                       "maximum_position_error_m": float(np.linalg.norm(positions.reshape(-1, 3)-projected, axis=1).max()),
                       "vertex_normal_constraint": float(constraint_error),
                       "maximum_edge_normal_jump_degrees": float(jumps.max()),
                       "minimum_surface_jacobian_ratio": float((jacobian / np.linalg.norm(area, axis=1)[:, None]).min())}
                if shape == "egg":
                    tetra_edges = np.array(list(combinations(range(4), 2)))
                    volume_edges = tets[:, tetra_edges]
                    columns = volume_edges % count
                    edge_lookup = {tuple(pair): index for index, pair in enumerate(edges)}
                    mean_depth = (volume_edges // count).mean(axis=2) * 0.00025
                    outer_mid = np.empty((*mean_depth.shape, 3))
                    for i, j in product(range(len(tets)), range(6)):
                        pair = tuple(sorted(columns[i, j]))
                        outer_mid[i, j] = vertices[pair[0]] if pair[0] == pair[1] else midpoints[edge_lookup[pair]]
                    # Preserve the supplied wall columns and interpolate their
                    # actual offsets. Only the common outer-edge correction changes.
                    original_mid = xyz[volume_edges].mean(axis=2)
                    outer_linear = vertices[columns].mean(axis=2)
                    volume_nodes = np.concatenate([xyz[tets], original_mid + outer_mid - outer_linear], axis=1)
                    samples = np.array([[i, j, k, 4-i-j-k] for i in range(5) for j in range(5-i)
                                        for k in range(5-i-j)]) / 4
                    _, gradients, _ = quadratic_basis(samples)
                    jac = np.einsum("qna,tni->tqia", gradients, volume_nodes)
                    original_det = np.linalg.det(xyz[tets[:, 1:]]-xyz[tets[:, :1]])
                    ratio = np.linalg.det(jac) / original_det[:, None]
                    row["minimum_volume_jacobian_ratio"] = float(ratio.min())
                    row["nonpositive_jacobians"] = int(np.count_nonzero(ratio <= 0))
                    if level == 1:
                        vertex = int(np.argmin(np.linalg.norm(vertices-center, axis=1)))
                        assert np.linalg.norm(vertices[vertex]-center) < 1e-12
                        row["captured_source_normal_difference_degrees"] = float(np.degrees(np.arccos(np.clip(-normal[vertex]@inward, -1, 1))))
                        arrays[kind+"_volume_nodes_m"] = volume_nodes
                row["geometry_targets_met"] = bool(row["maximum_normal_angle_degrees"] < targets["normal_degrees"]
                    and row["curvature_rms_relative"] < targets["curvature_rms_relative"]
                    and row["maximum_edge_normal_jump_degrees"] < targets["edge_normal_jump_degrees"]
                    and row.get("minimum_volume_jacobian_ratio", 1) > targets["sampled_jacobian_ratio"])
                rows.append(row)
                write_json(args.output / "progress.json", {"rows": rows})
    path = args.output / "geometry.npz"
    np.savez_compressed(path, **arrays)
    write_json(args.output / "result.json", {"passed": True, "targets": targets, "rows": rows,
        "geometry": file_id(path), "mechanical_accuracy_qualified": False, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
