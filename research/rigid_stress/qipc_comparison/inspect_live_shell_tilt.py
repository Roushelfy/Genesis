"""Measure smooth shell displacement and rotation from an admitted native capture."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .check_curved_shell_operators import locate_curved
from .inspect_edge_shell_tilt import point_displacement
from .study_quadratic_rest_surface import quadratic_basis


def particular(points, center, basis, force, axes, exponent, profile, use_moments, poisson, young):
    """Evaluate point or four-source displacement with matched pressure second moments."""
    if not use_moments:
        return point_displacement(points, center, basis, force, poisson, young)
    spread = axes / np.sqrt(5)
    if profile == 2:
        spread = np.array([axes[1] * np.sqrt(4 * (1 - exponent) * (3 + 2 * exponent) / 175), axes[0] / np.sqrt(5)])
    result = np.zeros_like(points)
    for sign in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        source = center + basis[:, :2] @ (spread * sign)
        result += point_displacement(points, source, basis, force / 4, poisson, young)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--hard", type=Path, required=True)
    cli.add_argument("--elastic", type=Path, required=True)
    cli.add_argument("--environment", type=int, default=0)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in [args.frame, args.hard, args.elastic]],
        },
    )
    with np.load(args.frame) as saved:
        frame = {key: saved[key] for key in saved.files}
    with np.load(args.hard) as saved:
        hard = {key: saved[key] for key in saved.files}
    with np.load(args.elastic) as saved:
        elastic = {key: saved[key] for key in saved.files}
    env = args.environment
    assert frame["step_valid"][env] and elastic["step_valid"][env] and elastic["valid"][env]
    assert hard["status"][env] == 0 and elastic["recovery_status"][env] == 0
    regions = np.flatnonzero(hard["source_count"][env] > 0)
    batch = hard["profile"].shape[0]
    patches = regions * batch + env
    centers, bases, forces = hard["center_m"][patches], hard["basis"][patches], hard["force_n"][env, regions]
    direction = hard["direction"][patches]
    rotations = np.broadcast_to(np.eye(3), (len(regions), 3, 3)).copy()
    rotations[:, 0, 0], rotations[:, 1, 0] = direction[:, 0], direction[:, 1]
    rotations[:, 0, 1], rotations[:, 1, 1] = -direction[:, 1], direction[:, 0]
    bases = bases @ rotations
    local_forces = np.einsum("ni,nij->nj", forces, bases)
    axes, exponents, profiles = hard["axes_m"][patches], hard["edge_exponent"][patches], hard["profile"][env, regions]
    vertices, elements = frame["material_nodes_m"], elastic["elements"].T
    nodes = vertices[elements]
    owners, bary, valid, _ = locate_curved(nodes, centers)
    assert valid.all()
    shape, gradient, _ = quadratic_basis(bary)
    jacobian = np.einsum("qna,qni->qia", gradient, nodes[owners] - nodes[owners, :1])
    gradient = gradient @ np.linalg.inv(jacobian)
    rows, arrays = [], {}
    for use_moments in [False, True]:
        nodal = np.zeros_like(vertices)
        for i in range(len(regions)):
            nodal += particular(
                vertices,
                centers[i],
                bases[i],
                local_forces[i],
                axes[i],
                exponents[i],
                profiles[i],
                use_moments,
                elastic["poisson"],
                elastic["young_pa"],
            )
        raw = elastic["displacement_m"][env]
        mass_modes, modes = elastic["mass_modes"], elastic["modes"]
        coefficients = elastic["gram_inverse"] @ np.einsum("nic,ni->c", mass_modes, raw + nodal)
        rigid = np.einsum("nic,c->ni", modes, coefficients)
        corrected = raw - rigid
        residual = np.linalg.norm(np.einsum("nic,ni->c", mass_modes, corrected + nodal))
        scale = np.linalg.norm(np.einsum("nic,ni->c", mass_modes, raw + nodal))
        assert residual < 1e-10 * scale
        derivative = np.einsum("qni,qnj->qij", corrected[elements[owners]] - corrected[elements[owners, :1]], gradient)
        value = np.einsum("qn,qni->qi", shape, corrected[elements[owners]])
        minimum_node_distance = np.min(np.linalg.norm(vertices[None] - centers[:, None], axis=2), axis=1)
        for i, region in enumerate(regions):
            far_value, far_gradient = np.zeros(3), np.zeros((3, 3))
            distances = []
            for j in range(len(regions)):
                if j == i:
                    continue
                distances.append(np.linalg.norm(centers[i] - centers[j]) / axes[j].max())
                far_value += particular(
                    centers[i : i + 1],
                    centers[j],
                    bases[j],
                    local_forces[j],
                    axes[j],
                    exponents[j],
                    profiles[j],
                    use_moments,
                    elastic["poisson"],
                    elastic["young_pa"],
                )[0]
                for coordinate in range(3):
                    point = centers[i : i + 1].astype(complex)
                    point[:, coordinate] += 1e-30j
                    far_gradient[:, coordinate] += (
                        particular(
                            point,
                            centers[j],
                            bases[j],
                            local_forces[j],
                            axes[j],
                            exponents[j],
                            profiles[j],
                            use_moments,
                            elastic["poisson"],
                            elastic["young_pa"],
                        )[0].imag
                        / 1e-30
                    )
            total = derivative[i] + far_gradient
            local = bases[i].T @ total @ bases[i]
            normal_tilt = -local[2, :2]
            row = {
                "region": int(region),
                "profile": int(profiles[i]),
                "particular_nodal_model": "four_moments" if use_moments else "point",
                "native_force_n": forces[i].tolist(),
                "pressure_center_m": centers[i].tolist(),
                "smooth_displacement_local_m": ((value[i] + far_value) @ bases[i]).tolist(),
                "smooth_gradient_local": local.tolist(),
                "first_order_normal_tilt_local_rad": normal_tilt.tolist(),
                "first_order_normal_rotation_degrees": float(np.degrees(np.linalg.norm(normal_tilt))),
                "nearest_node_distance_over_major_radius": float(minimum_node_distance[i] / axes[i].max()),
                "minimum_other_source_distance_over_major_radius": min(distances) if distances else None,
                "mass_rigid_projection_relative_residual": float(residual / scale),
            }
            if profiles[i] == 2:
                curvature = hard["edge_curvature"][patches[i]]
                row.update(
                    edge_curvature_per_m=float(curvature),
                    edge_slope=float(hard["edge_slope"][patches[i]]),
                    smooth_along_center_shift_m=float(local[2, 1] / curvature),
                )
            rows.append(row)
        arrays["corrected_regular_moments" if use_moments else "corrected_regular_point"] = corrected
    np.savez_compressed(args.output / "displacement.npz", **arrays)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "environment": env,
            "rows": rows,
            "scope": "An admitted native regular P2 field plus other-source far particular displacements. Remove the mass-weighted rigid part of regular plus nodal particular displacement. Compare point and pressure-second-moment nodal/far approximations. The self contact displacement is excluded, so these are smooth shell corrections rather than the complete deformed contact normal. No load or source changes.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
