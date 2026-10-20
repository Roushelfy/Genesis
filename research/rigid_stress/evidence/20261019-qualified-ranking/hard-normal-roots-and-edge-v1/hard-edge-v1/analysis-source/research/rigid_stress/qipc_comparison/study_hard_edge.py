"""Discriminate finite-face pressure estimates with an independent hard-edge solve."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import halfspace_stress, solve_hard_contact, square_compliance


def metrics(force, reference, stress, reference_stress, points, load):
    deviation = stress - np.trace(stress, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    ref_deviation = reference_stress - np.trace(reference_stress, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    vm = np.sqrt(1.5 * np.sum(deviation**2, axis=(1, 2)))
    ref_vm = np.sqrt(1.5 * np.sum(ref_deviation**2, axis=(1, 2)))
    pressure_error = np.linalg.norm(force - reference) / np.linalg.norm(reference)
    stress_error = np.linalg.norm(stress - reference_stress) / np.linalg.norm(reference_stress)
    peak_error = abs(vm.max() / ref_vm.max() - 1)
    return {
        "pressure_cell_l2_relative": float(pressure_error),
        "sample_tensor_relative": float(stress_error),
        "sample_vm_peak_relative": float(peak_error),
        "sample_tensile_peak_pa": float(np.linalg.eigvalsh(stress)[:, -1].max()),
        "pressure_max_cell_ratio": float(force.max() / reference.max()),
        "centroid_difference_m": ((points.T @ (force - reference)) / load).tolist(),
        "engineering_targets_met": bool(pressure_error < 0.15 and stress_error < 0.10 and peak_error < 0.15),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": {
                "pressure_cell_l2_relative": 0.15,
                "sample_tensor_relative": 0.10,
                "sample_vm_peak_relative": 0.15,
            },
        },
    )
    load, modulus, radius = 0.1, 1e10 / (1 - 0.3**2), 0.02
    a0 = (3 * load * radius / (4 * modulus)) ** (1 / 3)
    gauss, weights = np.polynomial.legendre.leggauss(10)
    weights = (weights[:, None] * weights[None]).reshape(-1) / 4
    observations = (
        np.array(
            np.meshgrid(np.linspace(-1.2, 1.8, 17), np.linspace(-1.8, 1.8, 17), np.geomspace(0.25, 2, 9), indexing="ij")
        )
        .reshape(3, -1)
        .T
        * a0
    )
    rows = []
    for count in (36, 54):
        width = 3.6 * a0 / count
        xs = (np.arange(count) + 0.5) * width
        ys = xs - 1.8 * a0
        points = np.array(np.meshgrid(xs, ys, indexing="ij")).reshape(2, -1).T
        offsets = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width / 2
        sample = points[:, None] + offsets[None]
        matrix = square_compliance(points, width, modulus)
        for fraction in (-0.5, 0.0, 0.5, 2.0):
            offset = fraction * a0
            gap = ((points[:, 0] + offset) ** 2 + points[:, 1] ** 2) / (2 * radius)
            begun = perf_counter()
            reference = solve_hard_contact(matrix, gap, load)
            reference_seconds = perf_counter() - begun
            grid_force = reference.force.reshape(count, count)
            # x=0 is the physical edge. Only the three remote search boundaries
            # must be unloaded for this finite-face reference.
            assert max(grid_force[-1].max(), grid_force[:, 0].max(), grid_force[:, -1].max()) < load * 1e-10
            reference_stress = halfspace_stress(observations, points, reference.force)
            cap_radius = brentq(
                lambda value: modulus / radius * (value - offset) ** 2 * (2 * value + offset) / 3 - load,
                abs(offset) * (1 + 1e-12) + a0 * 1e-10,
                abs(offset) + 3 * a0,
                xtol=a0 * 1e-12,
            )
            distance2 = (sample[..., 0] + offset) ** 2 + sample[..., 1] ** 2
            cap_pressure = 2 * modulus / (np.pi * radius) * np.sqrt(np.maximum(cap_radius**2 - distance2, 0))
            cap_force = cap_pressure @ weights * width**2
            cap_raw_force_error = abs(cap_force.sum() / load - 1)
            cap_force *= load / cap_force.sum()

            def trial_force(value):
                pressure = np.sqrt(np.maximum((value * a0) ** 2 - distance2, 0) / sample[..., 0])
                force = pressure @ weights * width**2
                return force * (load / force.sum())

            def complementary_energy(value):
                force = trial_force(value)
                return (0.5 * force @ matrix @ force + gap @ force) / (load * a0**2 / radius)

            lower = abs(fraction) + 0.03
            upper = np.sqrt(fraction**2 + 1.65**2)
            begun = perf_counter()
            optimized = minimize_scalar(
                complementary_energy, bounds=(lower, upper), method="bounded", options={"xatol": 1e-6}
            )
            assert optimized.success
            candidate_seconds = perf_counter() - begun
            edge_force = trial_force(optimized.x)
            variants, arrays = {}, {}
            for name, force in (("load_consistent_hertz_cap", cap_force), ("edge_ritz", edge_force)):
                stress = halfspace_stress(observations, points, force)
                variants[name] = metrics(force, reference.force, stress, reference_stress, points, load)
                variants[name]["energy_relative_to_reference"] = float(
                    (0.5 * force @ matrix @ force + gap @ force)
                    / (0.5 * reference.force @ matrix @ reference.force + gap @ reference.force)
                    - 1
                )
                arrays[name + "_force_n"], arrays[name + "_stress_pa"] = force, stress
            path = args.output / f"offset{fraction:g}-grid{count}.npz"
            np.savez_compressed(
                path,
                points_m=points,
                reference_force_n=reference.force,
                reference_gap_m=reference.gap,
                observations_m=observations,
                reference_stress_pa=reference_stress,
                **arrays,
            )
            rows.append(
                {
                    "offset_over_hertz_radius": fraction,
                    "grid": count,
                    "cell_width_m": width,
                    "reference_iterations": reference.iterations,
                    "reference_seconds": reference_seconds,
                    "reference_active_cells": int(np.count_nonzero(reference.force > 1e-10 * load)),
                    "reference_max_cell_pressure_pa": float(reference.force.max() / width**2),
                    "reference_indentation_m": reference.indentation,
                    "cap_radius_m": cap_radius,
                    "cap_quadrature_force_relative_before_normalization": cap_raw_force_error,
                    "ritz_radius_m": optimized.x * a0,
                    "ritz_seconds": candidate_seconds,
                    "ritz_objective_evaluations": optimized.nfev,
                    "ritz_at_search_boundary": bool(min(optimized.x - lower, upper - optimized.x) < 1e-3),
                    "variants": variants,
                    "arrays": file_id(path),
                }
            )
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "hertz_radius_m": a0,
            "force_n": load,
            "reduced_modulus_pa": modulus,
            "radius_m": radius,
            "rows": rows,
            "scope": "Rigid finite face x>=0 against a spherical quadratic elastic half-space. Independent force-controlled unilateral BEM determines pressure and support. Two spatial grids retain cell pressures and tensors sampled at positive depths >=0.25a. The Hertz cap enforces force analytically; the single-radius edge trial minimizes complementary energy with the same independent Green operator. Targets concern cell pressure and sampled subsurface fields, not sharp-edge continuum maxima, a complete shell field, friction, or native runtime cost. Trial fits and reference costs remain offline.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
