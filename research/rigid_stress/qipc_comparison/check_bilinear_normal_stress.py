"""Validate radial pressure-stress integrals before using shell reference fields."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .artifacts import runtime_identity, write_json
from .bilinear_normal_stress import BilinearNormalStress, point_tensor
from .hertz_reference import halfspace_stress


def quadrature_sources(axes, pressure, order):
    node, weight = np.polynomial.legendre.leggauss(order)
    widths = [axis[1] - axis[0] for axis in axes]
    positions = [(axis[:-1, None] + (node + 1) * width / 2).reshape(-1) for axis, width in zip(axes, widths)]
    weights = [np.tile(weight * width / 2, len(axis) - 1) for axis, width in zip(axes, widths)]
    points = np.stack(np.meshgrid(*positions, indexing="ij"), axis=-1).reshape(-1, 2)
    values = RegularGridInterpolator(axes, pressure)(points)
    return points, values * np.outer(*weights).reshape(-1)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    length, scale, poisson = 7e-5, 2.67e7, 0.3
    targets = {
        "point_relative": 1e-10,
        "positive_depth_tensor_over_pressure": 1e-3,
        "direct_rule_change_over_pressure": 1e-6,
        "surface_gaussian_over_pressure": 1e-3,
        "surface_limit_over_pressure": 1e-3,
        "far_tensor_over_pressure": 1e-5,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "scope": "Independent complex-step Boussinesq quadrature, analytical Gaussian surface tensors, positive-depth surface limit, angular refinement and far-source controls. Numerical pressure-stress integration only; no physical shell qualification.",
        },
    )
    rng = np.random.default_rng(7321)
    points = rng.normal(size=(31, 3)) * length
    points[:, 2] = np.abs(points[:, 2]) + 0.1 * length
    old = halfspace_stress(points, np.zeros((1, 2)), np.ones(1), poisson)
    new = point_tensor(points, poisson)
    error = float(np.linalg.norm(new - old) / np.linalg.norm(old))
    assert error < targets["point_relative"], error
    axes = [np.linspace(-1.3 * length, 1.3 * length, 17), np.linspace(-length, length, 17)]
    x, y = np.meshgrid(*axes, indexing="ij")
    pressure = scale * (1 - (x / (1.3 * length)) ** 2) ** 2 * (1 - (y / length) ** 2) ** 2 * (1 + 0.15 * x / length)
    model = BilinearNormalStress(axes, pressure, length, scale, poisson)
    query = np.array([[x, y, z] for x in [-1.0, 0.0, 0.7] for y in [-0.5, 0.4] for z in [0.3, 0.7, 2.0]]) * length
    direct = []
    for order in [8, 12]:
        source, force = quadrature_sources(axes, pressure, order)
        direct.append(halfspace_stress(query, source, force, poisson))
    begun = perf_counter()
    ray128, ray256 = model.near(query, 128), model.near(query, 256)
    seconds = perf_counter() - begun
    rule_error = float(np.max(np.abs(direct[0] - direct[1])) / scale)
    ray_error = float(np.max(np.abs(ray256 - direct[1])) / scale)
    angular_change = float(np.max(np.abs(ray128 - ray256)) / scale)
    np.savez_compressed(
        args.output / "positive-depth.npz",
        points_m=query,
        direct_q8_pa=direct[0],
        direct_q12_pa=direct[1],
        ray128_pa=ray128,
        ray256_pa=ray256,
    )
    write_json(
        args.output / "positive-depth.json",
        {
            "direct_rule_change_over_pressure": rule_error,
            "ray_tensor_error_over_pressure": ray_error,
            "angular128_256_change_over_pressure": angular_change,
        },
    )
    assert rule_error < targets["direct_rule_change_over_pressure"], rule_error
    assert ray_error < targets["positive_depth_tensor_over_pressure"], ray_error
    gaussian_axes = [np.linspace(-4 * length, 4 * length, 257)] * 2
    x, y = np.meshgrid(*gaussian_axes, indexing="ij")
    gaussian = scale * np.exp(-(x * x + y * y) / length**2)
    gaussian_model = BilinearNormalStress(gaussian_axes, gaussian, length, scale, poisson)
    radii = np.array([0, 0.3, 0.9, 1.7, 2.8])
    angle = np.array([0, 0.37, 0.9])
    query = np.array([[r * np.cos(theta), r * np.sin(theta), 0] for r in radii for theta in angle]) * length
    analytic = np.empty((len(query), 3, 3))
    for i, point in enumerate(query):
        radius = np.linalg.norm(point[:2]) / length
        value = np.exp(-(radius**2))
        integral = -np.expm1(-(radius**2)) / (2 * radius**2) if radius else 0.5
        rr = -value + (1 - 2 * poisson) * integral
        tt = -2 * poisson * value - (1 - 2 * poisson) * integral
        radial = point[:2] / np.linalg.norm(point[:2]) if radius else np.array([1, 0])
        tensor = np.zeros((3, 3))
        tensor[:2, :2] = tt * np.eye(2) + (rr - tt) * np.outer(radial, radial)
        tensor[2, 2] = -value
        analytic[i] = tensor * scale
    surface = gaussian_model.near(query, 256)
    positive = query.copy()
    positive[:, 2] = 1e-6 * length
    near_surface = gaussian_model.near(positive, 256)
    surface_error = float(np.max(np.abs(surface - analytic)) / scale)
    limit_error = float(np.max(np.abs(near_surface - surface)) / scale)
    np.savez_compressed(
        args.output / "surface.npz",
        points_m=query,
        analytic_pa=analytic,
        surface_pa=surface,
        positive_depth_pa=near_surface,
    )
    write_json(
        args.output / "surface.json",
        {"gaussian_tensor_error_over_pressure": surface_error, "surface_limit_change_over_pressure": limit_error},
    )
    assert surface_error < targets["surface_gaussian_over_pressure"], surface_error
    assert limit_error < targets["surface_limit_over_pressure"], limit_error
    far_query = np.array([[x, y, z] for x in [4, 7] for y in [-1, 2] for z in [0, 0.8, 3]]) * length
    far_ray = model.near(far_query, 512)
    far9, far17 = model.far(far_query, 9), model.far(far_query, 17)
    far_error = float(np.max(np.abs(far9 - far_ray)) / scale)
    far_rule = float(np.max(np.abs(far17 - far9)) / scale)
    np.savez_compressed(args.output / "far.npz", points_m=far_query, ray_pa=far_ray, source9_pa=far9, source17_pa=far17)
    assert max(far_error, far_rule) < targets["far_tensor_over_pressure"], (far_error, far_rule)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "point_tensor_relative_error": error,
            "positive_depth_tensor_error_over_pressure": ray_error,
            "direct_rule_change_over_pressure": rule_error,
            "angular_change_over_pressure": angular_change,
            "surface_gaussian_tensor_error_over_pressure": surface_error,
            "surface_limit_change_over_pressure": limit_error,
            "far_tensor_error_over_pressure": far_error,
            "far_source9_source17_change_over_pressure": far_rule,
            "positive_depth_control_seconds": seconds,
            "targets": targets,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
