"""Intersect captured rigid load lines with the actual quadratic material boundary."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse
from .study_quadratic_rest_surface import quadratic_basis


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--boundaries", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.frame), file_id(args.boundaries)],
            "scope": "Read-only intersections of a captured single force with curved material faces. The original region refusal is preserved. Contact positions/forces are material coordinates; region centers and boundary frames are world coordinates. A line intersection preserves that force's wrench but does not independently validate any pressure model or admit the original rejected observation.",
        },
    )
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.boundaries) as saved:
        bounds = {key: saved[key] for key in saved.files}
    nodes = frame["material_nodes_m"][frame["material_surface_nodes"].T]
    assert nodes.shape[1:] == (6, 3)
    is_loaded = frame["contact_valid"] & (np.linalg.norm(frame["contact_force_n"], axis=2) > 0)
    rows = []
    for i_b, i_c in np.argwhere(is_loaded & (bounds["count"] > 0)):
        point, force = frame["contact_position_m"][i_b, i_c], frame["contact_force_n"][i_b, i_c]
        direction = force / np.linalg.norm(force)
        unit = np.eye(3)[np.argmin(np.abs(direction))]
        tangent = np.cross(direction, unit)
        tangent /= np.linalg.norm(tangent)
        plane = np.array([tangent, np.cross(direction, tangent)])
        chord = (nodes[:, 1:3] - nodes[:, :1]).transpose(0, 2, 1)
        chart = plane @ chord
        usable = np.abs(np.linalg.det(chart)) > 1e-12 * np.linalg.norm(chart, axis=(1, 2)) ** 2
        faces = np.flatnonzero(usable)
        local_nodes = nodes[faces]
        local = np.linalg.solve(chart[faces], ((point - local_nodes[:, 0]) @ plane.T)[..., None])[..., 0]
        for _ in range(12):
            bary = np.column_stack((1 - local.sum(axis=1), local))
            values, gradient, hessian = quadratic_basis(bary)
            positions = np.einsum("fn,fni->fi", values, local_nodes)
            tangents = np.einsum("fna,fni->fia", gradient, local_nodes)
            chart = plane @ tangents
            usable = np.abs(np.linalg.det(chart)) > 1e-12 * np.linalg.norm(chart, axis=(1, 2)) ** 2
            faces, local_nodes, local = faces[usable], local_nodes[usable], local[usable]
            local -= np.linalg.solve(chart[usable], ((positions[usable] - point) @ plane.T)[..., None])[..., 0]
        bary = np.column_stack((1 - local.sum(axis=1), local))
        values, gradient, hessian = quadratic_basis(bary)
        positions = np.einsum("fn,fni->fi", values, local_nodes)
        tangents = np.einsum("fna,fni->fia", gradient, local_nodes)
        area = np.cross(tangents[:, :, 0], tangents[:, :, 1])
        area *= np.sign(np.einsum("fi,fi->f", area, frame["material_face_normal"][faces]))[:, None]
        normals = area / np.linalg.norm(area, axis=1)[:, None]
        residuals = np.linalg.norm((positions - point) @ plane.T, axis=1)
        inside = (bary.min(axis=1) >= -1e-10) & (bary.max(axis=1) <= 1 + 1e-10)
        candidates = np.flatnonzero(inside & (normals @ direction < 0) & (residuals < 1e-12))
        assert candidates.size > 0
        witness = frame["source_witness_m"][i_b, i_c]
        selected = candidates[np.argmin(np.linalg.norm(positions[candidates] - witness, axis=1))]
        center, normal = positions[selected], normals[selected]
        quaternion = frame["frame_quaternion"][i_b]
        rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]])
        world_center = rotation.apply(center) + frame["frame_position_m"][i_b]
        region_normal, region_tangent = bounds["normal"][i_b, i_c], bounds["tangent"][i_b, i_c]
        region_basis = np.column_stack((region_tangent, np.cross(region_normal, region_tangent)))
        offset = (world_center - frame["region_center_m"][i_b, i_c]) @ region_basis
        planes = bounds["planes"][i_b, i_c]
        margins = (planes[:, 2] - planes[:, :2] @ offset) / np.linalg.norm(planes[:, :2], axis=1)
        local_tangent = np.cross(normal, np.eye(3)[np.argmin(np.abs(normal))])
        local_tangent /= np.linalg.norm(local_tangent)
        basis = np.column_stack((local_tangent, np.cross(normal, local_tangent)))
        second = np.einsum("nab,ni->iab", hessian, local_nodes[selected])
        metric_inverse = np.linalg.inv(tangents[selected].T @ tangents[selected])
        tensor = (
            -tangents[selected]
            @ metric_inverse
            @ np.einsum("i,iab->ab", normal, second)
            @ metric_inverse
            @ tangents[selected].T
        )
        curvature = basis.T @ tensor @ basis
        axes, principal, _, _ = ellipse(-force @ normal, 1e10 / 0.91, curvature)
        principal_basis = basis @ np.array([[principal[0], -principal[1]], [principal[1], principal[0]]])
        directions = planes[:, :2] @ region_basis.T @ rotation.as_matrix() @ principal_basis
        widths = np.linalg.norm(directions * axes, axis=1) / np.linalg.norm(planes[:, :2], axis=1)
        moment = np.cross(point - center, force)
        edge_norms = np.linalg.norm(planes[:, :2], axis=1)
        root_margins = planes[:, 2] / edge_norms
        world_force = rotation.apply(force)
        root_axes, root_principal, _, _ = ellipse(
            -world_force @ region_normal, 1e10 / 0.91, frame["region_curvature_per_m"][i_b, i_c]
        )
        root_rotation = np.array(
            [[root_principal[0], -root_principal[1]], [root_principal[1], root_principal[0]]]
        )
        root_widths = np.linalg.norm((planes[:, :2] @ root_rotation) * root_axes, axis=1) / edge_norms
        world_point = rotation.apply(point) + frame["frame_position_m"][i_b]
        contact_offset = (world_point - frame["region_center_m"][i_b, i_c]) @ region_basis
        contact_margins = (planes[:, 2] - planes[:, :2] @ contact_offset) / edge_norms
        world_collision_normal = rotation.apply(frame["contact_normal"][i_b, i_c])
        world_collision_normal /= np.linalg.norm(world_collision_normal)
        box_axes = np.vstack((region_normal, planes[[0, 2], :2] @ region_basis.T))
        box_axes /= np.linalg.norm(box_axes, axis=1)[:, None]
        collision_axis_cosines = box_axes @ world_collision_normal
        rows.append(
            {
                "normal_root_finite_face_margins_m": root_margins.tolist(),
                "normal_root_hertz_axes_m": root_axes.tolist(),
                "normal_root_hertz_edge_widths_m": root_widths.tolist(),
                "normal_root_margin_over_hertz_width": (root_margins / root_widths).tolist(),
                "collision_point_finite_face_margins_m": contact_margins.tolist(),
                "collision_normal_to_root_normal_degrees": float(
                    np.rad2deg(np.arccos(np.clip(world_collision_normal @ region_normal, -1, 1)))
                ),
                "collision_normal_box_axis_cosines": collision_axis_cosines.tolist(),
                "collision_normal_dominant_axis_is_selected_face": bool(
                    np.argmax(np.abs(collision_axis_cosines)) == 0
                ),
                "environment": int(i_b),
                "contact": int(i_c),
                "region_status": int(frame["region_status"][i_b, i_c]),
                "source_status": int(frame["source_status"][i_b, i_c]),
                "incoming_intersections": int(candidates.size),
                "selected_face": int(faces[selected]),
                "barycentric": bary[selected].tolist(),
                "position_material_m": center.tolist(),
                "position_world_m": world_center.tolist(),
                "witness_distance_m": float(np.linalg.norm(center - witness)),
                "force_line_residual_m": float(residuals[selected]),
                "wrench_moment_error_nm": float(np.linalg.norm(moment)),
                "normal_change_degrees": float(
                    np.rad2deg(np.arccos(np.clip(rotation.apply(normal) @ region_normal, -1, 1)))
                ),
                "tangential_force_over_normal": float(
                    np.linalg.norm(force - (force @ normal) * normal) / (-force @ normal)
                ),
                "tangent_offset_from_normal_root_m": offset.tolist(),
                "finite_face_margins_m": margins.tolist(),
                "local_curvature_per_m": curvature.tolist(),
                "local_hertz_axes_m": axes.tolist(),
                "local_hertz_edge_widths_m": widths.tolist(),
                "margin_over_local_hertz_width": (margins / widths).tolist(),
                "footprint_contained_in_face": bool(np.all(margins > widths)),
            }
        )
    assert rows
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
