"""Compare an unfitted sharp-edge scale law with the saved coupled pressure reference."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad
from scipy.special import digamma, roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales
from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress
from .study_far_edge_contact import edge_trial
from .study_franka_contact_gap import von_mises


def displacement_difference(aspect, exponent, shear_ratio, poisson):
    """Integrate exact center/end surface displacements after removing the logarithm."""
    beta = (1 - 2 * poisson) / (2 * (1 - poisson))
    gauss, weights = roots_jacobi(48, exponent, -exponent)
    u, weights = (gauss + 1) / 2, weights / weights.sum()
    remainder = []
    for value in aspect * u:

        def normal_integrand(v):
            return 2 * (1 - v * v) / np.sqrt(value * value * (1 - v * v) ** 2 + v * v)

        def shear_integrand(v):
            return -2 * value * (1 - v * v) ** 2 / (value * value * (1 - v * v) ** 2 + v * v)

        breaks = [value, min(10 * value, 0.9)]
        normal_center = quad(normal_integrand, 0, 1, points=breaks, epsabs=1e-10, epsrel=1e-10)[0]
        shear_center = quad(shear_integrand, 0, 1, points=breaks, epsabs=1e-10, epsrel=1e-10)[0]
        normal_end = 4 / (np.sqrt(1 + 4 * value * value) + 1)
        shear_end = (
            -(2 * value - np.arctan(2 * value)) / value**2
            if value > 1e-3
            else -8 * value / 3 + 32 * value**3 / 5 - 128 * value**5 / 7
        )
        remainder.append(
            normal_center
            - (2 * np.log(2 / value) - 1)
            - (normal_end - 2)
            + beta * shear_ratio * (shear_center + np.pi - shear_end)
        )
    asymptotic = 0.75 * (
        2 * np.log(2 / aspect) - 2 * digamma(1 - exponent) + 2 * digamma(2) - 3 - np.pi * beta * shear_ratio
    )
    exact = asymptotic + 0.75 * weights @ remainder
    return float(asymptotic), float(exact)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--coupling", type=Path, required=True)
    cli.add_argument("--grids", nargs="+", type=int, default=[48, 72])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = json.loads((args.reference / "result.json").read_text())
    coupling = json.loads((args.coupling / "result.json").read_text())
    assert reference["contact"] == coupling["contact"] and coupling["passed"]
    inputs = [args.reference / "result.json", args.coupling / "result.json"]
    inputs.extend(
        args.coupling / f"grid{count}-direction{direction}.npz" for count in args.grids for direction in (1, -1)
    )
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(path) for path in inputs],
            "targets": {"pressure_cell_l2": 0.15, "sample_tensor_relative": 0.10, "sample_vm_peak_relative": 0.15},
            "scope": "Unfitted slender sharp-edge pressure law, using only normal load, modulus, physical edge gap slope/curvature and prescribed transverse shear ratio. A scalar logarithmic scale equation replaces offline energy fitting. The saved independent coupled BEM supplies validation only. Stick/slip history, shell fields, surface/global peak and actual scene admission remain open.",
        },
    )
    load, modulus, poisson = reference["normal_load_n"], 1e10 / 0.91, 0.3
    slope, curvature = reference["edge_slope"], reference["edge_along_curvature_per_m"]
    ratio = np.array(coupling["tangent_over_normal_components"])
    rows = []
    for direction in (1, -1):
        begun = perf_counter()
        model = edge_scales(load, modulus, slope, curvature, direction * ratio[0], poisson)
        scale_seconds = perf_counter() - begun
        scaled = edge_scales(4 * load, modulus, slope, curvature / 2, direction * ratio[0], poisson)
        np.testing.assert_allclose([scaled.width, scaled.along], [2 * model.width, 2 * model.along], rtol=1e-11)
        scaled = edge_scales(3 * load, 3 * modulus, slope, curvature, direction * ratio[0], poisson)
        np.testing.assert_allclose([scaled.width, scaled.along], [model.width, model.along], rtol=1e-11)
        resultant = (
            model.amplitude
            * model.width
            * model.along
            * 4
            / 3
            * np.pi
            * model.exponent
            / np.sin(np.pi * model.exponent)
        )
        np.testing.assert_allclose(resultant, load, rtol=1e-12)
        asymptotic, exact = displacement_difference(
            model.width / model.along, model.exponent, direction * ratio[0], poisson
        )
        for count in args.grids:
            with np.load(args.coupling / f"grid{count}-direction{direction}.npz") as saved:
                points, observations = saved["points_m"], saved["observations_m"]
                force, tensor = saved["coupled_force_n"], saved["coupled_tensor_pa"]
            widths = np.array(next(row["cell_widths_m"] for row in coupling["rows"] if row["grid"] == count))
            trial = edge_trial(points, widths, model.width, model.along, 0.0, 48, edge_exponent=model.exponent)
            trial *= load / trial.sum()
            active = trial > 0
            trial_tensor = halfspace_stress(observations, points[active], trial[active]) + cerruti_stress(
                observations, points[active], trial[active, None] * direction * ratio
            )
            pressure_error = np.linalg.norm(trial - force) / np.linalg.norm(force)
            field_error = np.linalg.norm(trial_tensor - tensor) / np.linalg.norm(tensor)
            peak_error = abs(von_mises(trial_tensor).max() - von_mises(tensor).max()) / von_mises(tensor).max()
            rows.append(
                {
                    "grid": count,
                    "shear_direction": direction,
                    "width_m": model.width,
                    "along_m": model.along,
                    "edge_exponent": model.exponent,
                    "amplitude_pa": model.amplitude,
                    "scale_seconds": scale_seconds,
                    "exact_trial_displacement_difference": exact,
                    "asymptotic_trial_displacement_difference": asymptotic,
                    "compatibility_approximation_relative": abs(exact - asymptotic) / abs(exact),
                    "pressure_cell_l2_relative": float(pressure_error),
                    "sample_tensor_relative": float(field_error),
                    "sample_vm_peak_relative": float(peak_error),
                    "trial_targets_met": bool(pressure_error < 0.15 and field_error < 0.10 and peak_error < 0.15),
                }
            )
            np.savez_compressed(
                args.output / f"grid{count}-direction{direction}.npz", trial_force_n=trial, trial_tensor_pa=trial_tensor
            )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "contact": reference["contact"],
            "dimensional_controls_passed": True,
            "rows": rows,
            "native_scene_admitted": False,
            "global_surface_peak_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
