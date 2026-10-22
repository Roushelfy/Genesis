"""Compare shape-only vertex curvature fits and their captured contact-pressure error."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from ..mechanics import shell_mesh
from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure, halfspace_stress
from .study_quadratic_rest_surface import egg_geometry, exact_surface


def vertex_fits(vertices, faces, normals):
    triangles = vertices[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    area = np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1)
    face_normal = normals[faces].mean(axis=1)
    face_normal /= np.linalg.norm(face_normal, axis=1)[:, None]
    projection = np.eye(3) - face_normal[:, :, None] * face_normal[:, None, :]
    tangent = projection @ edges.transpose(0, 2, 1)
    changes = (normals[faces[:, 1:]] - normals[faces[:, :1]]).transpose(0, 2, 1)
    derivative = projection @ changes @ np.linalg.inv(tangent.transpose(0, 2, 1) @ tangent) @ tangent.transpose(0, 2, 1)
    tensors = (derivative + derivative.transpose(0, 2, 1)) / 2
    baseline = np.zeros((len(vertices), 3, 3))
    weights = np.zeros(len(vertices))
    np.add.at(baseline, faces.reshape(-1), np.repeat(tensors * area[:, None, None], 3, axis=0))
    np.add.at(weights, faces.reshape(-1), np.repeat(area, 3))
    projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
    baseline = projection @ baseline @ projection / weights[:, None, None]
    pairs = np.unique(np.sort(faces[:, [[0, 1], [0, 2], [1, 2]]].reshape(-1, 2), axis=1), axis=0)
    symmetric = np.zeros((6, 3, 3))
    for k, (i, j) in enumerate([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]):
        symmetric[k, i, j] = symmetric[k, j, i] = 1
    direct, quadric, conditions, residuals = np.zeros_like(baseline), np.zeros_like(baseline), [], []
    for index, (origin, normal, projector) in enumerate(zip(vertices, normals, projection)):
        adjacent = pairs[(pairs == index).any(axis=1)]
        neighbors = adjacent[adjacent != index]
        edge = vertices[neighbors] - origin
        lengths = np.linalg.norm(edge, axis=1)
        tangent = edge @ projector
        change = (normals[neighbors] - normal) @ projector
        derivative = np.linalg.lstsq(tangent, change, rcond=1e-12)[0].T
        direct[index] = projector @ (derivative + derivative.T) @ projector / 2
        scale = lengths.mean()
        edge /= scale
        lengths /= scale
        neighbor_projection = np.eye(3) - normals[neighbors, :, None] * normals[neighbors, None, :]
        position_rows = 0.5 * np.einsum("ni,kij,nj->nk", edge, symmetric, edge) / lengths[:, None]
        normal_rows = np.einsum("nij,kjl,nl->nik", neighbor_projection, symmetric, edge)
        matrix = np.concatenate([position_rows, normal_rows.reshape(-1, 6)])
        rhs = np.concatenate([-edge @ normal / lengths, -(neighbor_projection @ normal).reshape(-1)])
        coefficients, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=1e-12)
        assert rank == 6
        quadric[index] = projector @ np.einsum("k,kij->ij", coefficients, symmetric) @ projector / scale
        conditions.append(singular[0] / singular[-1])
        residuals.append(np.linalg.norm(matrix @ coefficients - rhs) / np.linalg.norm(rhs))
    return {"face_average": baseline, "vertex_derivative": direct, "implicit_quadric": quadric}, conditions, residuals


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"curvature_rms_relative": 0.05, "pressure_relative": 0.05, "sample_tensor_relative": 0.05}
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "frame": file_id(args.frame),
            "reference": file_id(args.reference),
            "targets": targets,
            "scope": "Fixed-rest one-ring fits on sphere, ellipsoid and tapered egg with supplied analytic, area-weighted or mesh-library normals. Reuse the independent actual-frame smooth-gap contact reference. This compares curvature and local half-space pressure only; it does not admit curved shell recovery or validate a finite-shell stress field.",
        },
    )
    with np.load(args.frame) as saved:
        center, basis = saved["hertz_centers_m"][0], saved["hertz_frames"][0]
        direction = saved["law_direction"][0]
        basis[:, :2] = basis[:, :2] @ np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
        load, native_axes = saved["law_force_n"][0], saved["law_axes_m"][0]
    with np.load(args.reference) as saved:
        points, reference_force = saved["positions_m"], saved["force_n"]
        observations, reference_field = saved["observation_points_m"], saved["tensor_pa"]
    width = np.unique(points[:, 0])[1] - np.unique(points[:, 0])[0]
    gauss, weights = np.polynomial.legendre.leggauss(16)
    offsets = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width / 2
    weights = np.outer(weights, weights).reshape(-1) / 4
    samples = (points[:, None] + offsets[None]).reshape(-1, 2)
    rows, contact_rows, arrays = [], [], {}
    for level in [1, 2, 3]:
        xyz, _, faces, metadata = shell_mesh(level, 2, 0.0005)
        original = xyz[: metadata.outer_surface_vertices]
        unit = np.column_stack(
            [original[:, :2] / (0.022 * (1 - 0.18 * original[:, 2:3] / 0.03)), original[:, 2] / 0.03]
        )
        for shape in ["sphere", "ellipsoid", "egg"]:
            vertices = (
                unit * 0.02
                if shape == "sphere"
                else unit * np.array([0.012, 0.015, 0.009])
                if shape == "ellipsoid"
                else egg_geometry(unit)
            )
            _, analytic, exact_tensor = exact_surface(vertices, shape)
            triangles = vertices[faces]
            area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            estimated = np.zeros_like(vertices)
            np.add.at(estimated, faces.reshape(-1), np.repeat(area, 3, axis=0))
            estimated /= np.linalg.norm(estimated, axis=1)[:, None]
            mesh_normals = trimesh.Trimesh(vertices=vertices, faces=faces, process=False).vertex_normals
            for kind, normals in [("analytic", analytic), ("area", estimated), ("mesh", mesh_normals)]:
                fits, condition, residual = vertex_fits(vertices, faces, normals)
                for method, tensor in fits.items():
                    error = np.linalg.norm(tensor - exact_tensor) / np.linalg.norm(exact_tensor)
                    rows.append(
                        {
                            "level": level,
                            "shape": shape,
                            "normal_input": kind,
                            "method": method,
                            "curvature_rms_relative": float(error),
                            "target_met": bool(error < targets["curvature_rms_relative"]),
                            "quadric_max_condition": float(max(condition)),
                            "quadric_max_residual": float(max(residual)),
                        }
                    )
                    if kind == "analytic" and (
                        (method == "vertex_derivative" and shape == "sphere")
                        or (method == "implicit_quadric" and shape in ["sphere", "ellipsoid"])
                    ):
                        assert error < 1e-10, rows[-1]
                    if level == 1 and shape == "egg":
                        index = np.argmin(np.linalg.norm(vertices - center, axis=1))
                        assert np.linalg.norm(vertices[index] - center) < 1e-12
                        curvature = basis[:, :2].T @ tensor[index] @ basis[:, :2]
                        axes, direction, _, peak = ellipse(load, 1e10 / 0.91, curvature)
                        force = (
                            ellipse_pressure(samples, axes, direction, peak).reshape(len(points), -1)
                            @ weights
                            * width**2
                        )
                        mass_error = abs(force.sum() / load - 1)
                        assert mass_error < 1e-3
                        force *= load / force.sum()
                        field = halfspace_stress(observations, points, force)
                        pressure_error = np.linalg.norm(force - reference_force) / np.linalg.norm(reference_force)
                        tensor_error = np.linalg.norm(field - reference_field) / np.linalg.norm(reference_field)
                        contact_rows.append(
                            {
                                "normals": kind,
                                "method": method,
                                "axes_m": axes.tolist(),
                                "pressure_relative": float(pressure_error),
                                "tensor_relative": float(tensor_error),
                                "quadrature_force_relative": float(mass_error),
                                "native_axis_relative": float(
                                    np.linalg.norm(axes - native_axes) / np.linalg.norm(native_axes)
                                ),
                                "targets_met": bool(
                                    pressure_error < targets["pressure_relative"]
                                    and tensor_error < targets["sample_tensor_relative"]
                                ),
                            }
                        )
                        arrays[kind + "_" + method + "_force_n"] = force
                        arrays[kind + "_" + method + "_tensor_pa"] = field
                write_json(args.output / "progress.json", {"rows": rows, "contact": contact_rows})
    np.savez_compressed(args.output / "fields.npz", **arrays)
    write_json(
        args.output / "result.json",
        {"passed": True, "rows": rows, "contact": contact_rows, "targets": targets, "whole_goal_complete": False},
    )


if __name__ == "__main__":
    main()
