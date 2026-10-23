"""Inspect captured region moments against a planar sharp-edge pressure support."""

import argparse
from pathlib import Path

import numpy as np
from scipy.special import roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales
from .study_far_edge_contact import projected_surface


def support_extrema(coefficient, width, along):
    """Bound a linear function over 0 <= x <= width*(1-y*y/along**2)."""
    cx, cy = coefficient
    values = [-cy * along, cy * along]
    if cx != 0:
        stationary = cy * along**2 / (2 * cx * width)
        if abs(stationary) <= along:
            values.append(cx * width * (1 - (stationary / along) ** 2) + cy * stationary)
    return np.array([min(values), max(values)])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--environment", type=int, default=0)
    cli.add_argument("--order", type=int, default=48)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(args.hard), file_id(args.frame)],
        },
    )
    with np.load(args.hard) as saved:
        hard = {key: saved[key] for key in saved.files}
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    env = args.environment
    batch = hard["profile"].shape[0]
    regions = np.flatnonzero(hard["source_count"][env] > 0)
    rows, arrays = [], {}
    for region in regions:
        patch = region * batch + env
        assert hard["profile"][env, region] == 2
        basis, center = hard["basis"][patch], hard["center_m"][patch]
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(basis), 1, rtol=0, atol=1e-12)
        force = hard["allocation_original"][env, region]
        moment = hard["original_moment_nm"][env, region] - np.cross(center, force)
        local_force, local_moment = force @ basis, moment @ basis
        along, width = hard["axes_m"][patch]
        geometry_basis = hard["geometry_basis"][env, region]
        geometry_compression = -force @ geometry_basis[:, 2]
        model = edge_scales(
            geometry_compression,
            hard["edge_modulus"][patch],
            hard["edge_slope"][patch],
            hard["edge_curvature"][patch],
            force @ geometry_basis[:, 0] / geometry_compression,
            hard["edge_poisson"][patch],
        )
        expected_width = model.width / (basis[:, 0] @ geometry_basis[:, 0])
        np.testing.assert_allclose([along, width], [model.along, expected_width], rtol=1e-10, atol=1e-16)
        exponent = model.exponent
        friction = hard["allocation_friction"][env, region]
        compression = local_force[2]
        assert compression > 0 and width > 0 and along > 0 and 0 < exponent < 1
        mean = 2 * (1 - exponent) * width / 5
        required_center = np.array([-local_moment[1], local_moment[0]]) / compression
        absolute_center = required_center + np.array([mean, 0])
        shape_coordinate = absolute_center[0] / width + (absolute_center[1] / along) ** 2
        inside = absolute_center[0] >= 0 and shape_coordinate <= 1

        x, wx = roots_jacobi(args.order, exponent, -exponent)
        y, wy = roots_jacobi(args.order, 1, 1)
        positions = np.stack(
            np.broadcast_arrays(width * (1 - y[:, None] ** 2) * (x[None] + 1) / 2, along * y[:, None]),
            axis=-1,
        ).reshape(-1, 2)
        weights = (wy[:, None] * wx[None]).reshape(-1)
        weights /= weights.sum()
        centered = positions - np.array([mean, 0])
        variance = np.array([4 * (1 - exponent) * (3 + 2 * exponent) * width**2 / 175, along**2 / 5])
        np.testing.assert_allclose(weights @ centered, 0, rtol=0, atol=1e-12 * along)
        np.testing.assert_allclose(centered.T @ (weights[:, None] * centered), np.diag(variance), rtol=1e-10, atol=1e-25)
        coefficient = required_center / variance
        tilted_weights = weights * (1 + centered @ coefficient)
        np.testing.assert_allclose(tilted_weights.sum(), 1, rtol=0, atol=1e-11)
        np.testing.assert_allclose(tilted_weights @ centered, required_center, rtol=1e-10, atol=1e-15)
        tilt_bounds = 1 + support_extrema(coefficient, width, along) - coefficient[0] * mean

        contacts = np.flatnonzero(
            (hard["leader"][env] == region)
            & frame["contact_valid"][env]
            & (np.linalg.norm(frame["contact_force_n"][env], axis=1) > 0)
        )
        local_positions = (frame["contact_position_m"][env, contacts] - center) @ basis
        local_forces = frame["contact_force_n"][env, contacts] @ basis
        np.testing.assert_allclose(local_forces.sum(axis=0), local_force, rtol=1e-12, atol=1e-15)
        targets = center + centered @ basis[:, :2].T
        curved_positions, outward, faces = projected_surface(
            frame["material_nodes_m"][frame["material_surface_nodes"].T],
            frame["material_face_normal"],
            targets,
            basis[:, :2].T,
            -basis[:, 2],
            center,
            return_geometry=True,
        )
        inward = -outward
        transverse = basis[:, 0] - (inward @ basis[:, 0])[:, None] * inward
        transverse /= np.linalg.norm(transverse, axis=1, keepdims=True)
        longitudinal = np.cross(inward, transverse)
        surface_weights = weights / (inward @ basis[:, 2])
        field = np.stack([inward, transverse, longitudinal], axis=2)
        resultant_matrix = np.einsum("i,ijk->jk", surface_weights, field)
        amplitudes = np.linalg.solve(resultant_matrix, force)
        curved_forces = surface_weights[:, None] * (field @ amplitudes)
        np.testing.assert_allclose(curved_forces.sum(axis=0), force, rtol=1e-12, atol=1e-15)
        curved_moment = np.cross(curved_positions - center, curved_forces).sum(axis=0) @ basis
        curved_local = (curved_positions - center) @ basis
        arrays[f"region{region}_positions_m"] = positions
        arrays[f"region{region}_weights"] = weights
        arrays[f"region{region}_tilted_weights"] = tilted_weights
        arrays[f"region{region}_curved_positions_m"] = curved_positions
        arrays[f"region{region}_curved_forces_n"] = curved_forces
        arrays[f"region{region}_curved_faces"] = faces
        rows.append(
            {
                "region": int(region),
                "contacts": contacts.tolist(),
                "local_force_n": local_force.tolist(),
                "local_free_moment_nm": local_moment.tolist(),
                "local_collision_positions_m": local_positions.tolist(),
                "local_collision_forces_n": local_forces.tolist(),
                "width_m": float(width),
                "half_length_m": float(along),
                "exponent": float(exponent),
                "captured_terminal_edge_status": int(hard["edge_status"][patch]),
                "captured_terminal_exponent": float(hard["edge_exponent"][patch]),
                "reconstructed_pressure_scales_match_retained_axes": True,
                "friction": float(friction),
                "tangential_force_ratio": float(np.linalg.norm(local_force[:2]) / compression),
                "pressure_mean_x_m": float(mean),
                "required_center_shift_m": required_center.tolist(),
                "required_center_relative_to_minimum_m": absolute_center.tolist(),
                "required_center_parabolic_coordinate": float(shape_coordinate),
                "positive_pressure_centroid_inside_planar_support": bool(inside),
                "linear_pressure_tilt_extrema": tilt_bounds.tolist(),
                "linear_pressure_tilt_nonnegative": bool(tilt_bounds[0] >= 0),
                "curved_proportional_shear_ratio": float(np.linalg.norm(amplitudes[1:]) / amplitudes[0]),
                "curved_proportional_cone_admitted": bool(
                    amplitudes[0] > 0 and np.linalg.norm(amplitudes[1:]) <= friction * amplitudes[0]
                ),
                "curved_proportional_free_moment_nm": curved_moment.tolist(),
                "curved_proportional_moment_defect_nm": (curved_moment - local_moment).tolist(),
                "sampled_curved_height_extrema_m": [float(curved_local[:, 2].min()), float(curved_local[:, 2].max())],
                "minimum_sampled_normal_alignment": float((inward @ basis[:, 2]).min()),
                "planar_contact_geometry_scope": "Fixed flat normal with zero traction height; nonnegative pressure centroid must belong to the convex parabolic support. Curved positions and normals require separate checks.",
            }
        )
    np.savez_compressed(args.output / "moments.npz", **arrays)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "environment": env,
            "quadrature_order": args.order,
            "rows": rows,
            "scope": "Captured force/moment replay and analytic necessary pressure-centroid bounds. Gauss-Jacobi moments independently check pressure normalization, centroid and variance; signed affine tilts are diagnostics until positivity holds. No production acceptance, pressure law or stress result is changed.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
