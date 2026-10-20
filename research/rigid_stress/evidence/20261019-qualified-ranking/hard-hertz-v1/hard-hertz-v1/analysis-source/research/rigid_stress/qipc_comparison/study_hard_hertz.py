"""Compare cheap hard-contact estimates with independently solved unilateral half-space contact."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, ellipse_pressure, halfspace_stress, solve_hard_contact, square_compliance


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    load, modulus = 0.1, 1e10 / (1 - 0.3**2)
    # Independent point-load stress identities fix sign and constitutive factors.
    point = halfspace_stress(np.array([[0.0, 0, 0.001]]), np.zeros((1, 2)), np.array([load]))[0]
    expected = np.diag([0.4, 0.4, -6.0]) * load / (4 * np.pi * 0.001**2)
    np.testing.assert_allclose(point, expected, rtol=1e-13, atol=1e-10)
    rows = []
    for name, values, angle in (
        ("circle", [50.0, 50.0], 0.0),
        ("ellipse", [20.0, 80.0], 0.43),
        ("two_regions", [50.0, 50.0], 0.0),
    ):
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        curvature = rotation @ np.diag(values) @ rotation.T
        local_load = load / 2 if name == "two_regions" else load
        axes, direction, depth, peak = ellipse(local_load, modulus, curvature)
        centers = np.array([[-1.5 * axes[0], 0], [1.5 * axes[0], 0]]) if name == "two_regions" else np.zeros((1, 2))
        extent = (3.1 if name == "two_regions" else 1.6) * axes[0]
        observation_axis = np.linspace(-extent, extent, 15)
        # Fixed positive depths across both BEM grids distinguish subsurface
        # tensor errors; these are not continuum global or surface peak bounds.
        obs = (
            np.array(
                np.meshgrid(
                    observation_axis, observation_axis, np.geomspace(0.4 * axes[0], 2 * axes[0], 9), indexing="ij"
                )
            )
            .reshape(3, -1)
            .T
        )
        for count in (36, 54):
            width = 2 * extent / count
            axis = (np.arange(count) + 0.5) * width - extent
            points = np.array(np.meshgrid(axis, axis, indexing="ij")).reshape(2, -1).T
            relative = points[:, None] - centers[None]
            gap = np.min(0.5 * np.einsum("nki,ij,nkj->nk", relative, curvature, relative), axis=1)
            begin = perf_counter()
            matrix = square_compliance(points, width, modulus)
            preparation = perf_counter() - begin
            begin = perf_counter()
            solved = solve_hard_contact(matrix, gap, load)
            solve_seconds = perf_counter() - begin
            assert not np.any(solved.force.reshape(count, count)[[0, -1]] > 1e-10 * load)
            assert not np.any(solved.force.reshape(count, count)[:, [0, -1]] > 1e-10 * load)
            reference_stress = halfspace_stress(obs, points, solved.force)
            gauss, weights = np.polynomial.legendre.leggauss(8)
            offsets = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width / 2
            weights = (weights[:, None] * weights[None, :]).reshape(-1) / 4
            sample = (points[:, None] + offsets[None]).reshape(-1, 2)
            variants, arrays = {}, {}
            circular_radius = (3 * local_load / (4 * modulus * np.sqrt(np.linalg.det(curvature)))) ** (1 / 3)
            for method in ("elliptic_hertz", "circular_geometric_curvature"):
                a = axes if method == "elliptic_hertz" else np.repeat(circular_radius, 2)
                p0 = 3 * local_load / (2 * np.pi * a.prod())
                pressure = sum(ellipse_pressure(sample - center, a, direction, p0) for center in centers)
                force = pressure.reshape(len(points), -1) @ weights * width**2
                raw_force_error = abs(force.sum() / load - 1)
                # Correct quadrature mass only; no fitting to reference pressure.
                force *= load / force.sum()
                stress = halfspace_stress(obs, points, force)
                pressure_error = np.linalg.norm(force - solved.force) / np.linalg.norm(solved.force)
                stress_error = np.linalg.norm(stress - reference_stress) / np.linalg.norm(reference_stress)
                eigen = np.linalg.eigvalsh(stress)
                ref_eigen = np.linalg.eigvalsh(reference_stress)
                dev = stress - np.trace(stress, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
                ref_dev = (
                    reference_stress - np.trace(reference_stress, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
                )
                mises, ref_mises = (
                    np.sqrt(1.5 * (dev * dev).sum(axis=(1, 2))),
                    np.sqrt(1.5 * (ref_dev * ref_dev).sum(axis=(1, 2))),
                )
                variants[method] = {
                    "axes_m": a.tolist(),
                    "pressure_area_l2_relative": float(pressure_error),
                    "quadrature_force_relative_before_normalization": float(raw_force_error),
                    "subsurface_sample_tensor_relative": float(stress_error),
                    "subsurface_sample_von_mises_peak_relative": float(abs(mises.max() / ref_mises.max() - 1)),
                    "subsurface_sample_tensile_peak_relative": float(
                        abs(eigen[:, -1].max() / ref_eigen[:, -1].max() - 1)
                    ),
                    "subsurface_sample_peak_position_m": obs[mises.argmax()].tolist(),
                }
                arrays[method + "_force_n"], arrays[method + "_stress_pa"] = force, stress
            path = args.output / f"{name}-{count}.npz"
            np.savez_compressed(
                path,
                points_m=points,
                reference_force_n=solved.force,
                reference_gap_m=solved.gap,
                observation_points_m=obs,
                reference_stress_pa=reference_stress,
                **arrays,
            )
            covariance = (points * solved.force[:, None]).T @ points / load
            rows.append(
                {
                    "case": name,
                    "grid": count,
                    "load_n": load,
                    "curvature_per_m": curvature.tolist(),
                    "independent_indentation_m": solved.indentation,
                    "local_hertz_indentation_m": depth,
                    "independent_active_cells": int((solved.force > 1e-10 * load).sum()),
                    "force_weighted_covariance_m2": covariance.tolist(),
                    "iterations": solved.iterations,
                    "preparation_seconds": preparation,
                    "solve_seconds": solve_seconds,
                    "variants": variants,
                    "arrays": file_id(path),
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "point_load_stress_identity": True,
            "scope": "Independent constant-cell Boussinesq half-space unilateral contact with force equality. Two spatial grids, one-sided rigid quadratic obstacles, circular/rotated elliptical/nonconvex two-region cases. Subsurface tensors and sampled peaks exclude the surface and are not global damage maxima. No hollow-shell, friction, actual collider geometry or scene-throughput qualification.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
