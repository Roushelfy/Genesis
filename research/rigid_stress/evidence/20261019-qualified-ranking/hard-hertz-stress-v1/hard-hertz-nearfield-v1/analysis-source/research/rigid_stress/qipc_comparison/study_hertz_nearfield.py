"""Validate cheap elliptic near-field stress and measure the retained coarse-shell scale gap."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse, halfspace_stress
from .hertz_stress import elliptic_stress


def stress_measures(tensor):
    eigen = np.linalg.eigvalsh(tensor)
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    deviator = tensor - trace[..., None, None] * np.eye(3) / 3
    return eigen[..., -1], np.sqrt(1.5 * (deviator * deviator).sum(axis=(-1, -2)))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--transfer", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    rows, checks = [], {}
    depths = np.r_[0, np.geomspace(1e-5, 20, 200)]
    axis = np.column_stack((np.zeros((len(depths), 2)), depths))
    radius = np.r_[0, np.linspace(0.01, 0.99, 80), 1, np.geomspace(1.0001, 20, 80)]
    surface = np.column_stack((radius, np.zeros((len(radius), 2))))
    for poisson in (0.0, 0.3, 0.49):
        vertical = -1 / (1 + depths * depths)
        radial = -(1 + poisson) * (1 - depths * np.arctan2(1, depths)) - vertical / 2
        expected = np.zeros((len(axis), 3, 3))
        expected[:, 0, 0] = expected[:, 1, 1] = radial
        expected[:, 2, 2] = vertical
        actual = elliptic_stress(axis, np.ones(2), poisson=poisson, order=96)
        np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=1e-7)
        pressure = np.sqrt(np.maximum(1 - radius * radius, 0))
        ratio = np.full(len(radius), 1.5)
        offaxis = radius > 0
        ratio[offaxis] = (1 - pressure[offaxis] ** 3) / radius[offaxis] ** 2
        expected = np.zeros((len(surface), 3, 3))
        expected[:, 0, 0] = (1 - 2 * poisson) * ratio / 3 - pressure
        expected[:, 1, 1] = -(1 - 2 * poisson) * ratio / 3 - 2 * poisson * pressure
        expected[:, 2, 2] = -pressure
        actual = elliptic_stress(surface, np.ones(2), poisson=poisson, order=96)
        np.testing.assert_allclose(actual, expected, atol=5e-9, rtol=1e-6)
    checks["circular_axis_and_surface_three_poisson_values"] = True
    rng = np.random.default_rng(4201)
    for ratio in (1.0, 0.73, 0.25, 0.05):
        axes = np.array([1.0, ratio])
        # Includes true surface, exact edge, small positive depth and far field.
        angle = np.linspace(0, np.pi / 2, 65)
        edge = np.column_stack((np.cos(angle), ratio * np.sin(angle), np.zeros(len(angle))))
        points = np.concatenate(
            (
                edge,
                np.column_stack((np.zeros((161, 2)), np.r_[0, np.geomspace(1e-5 * ratio, 20, 160)])),
                rng.uniform([0, 0, 0], [2, 2 * ratio, 3], (256, 3)),
            )
        )
        reference = elliptic_stress(points, axes, order=192)
        orders = {}
        for order in (16, 32, 48, 96):
            tensor = elliptic_stress(points, axes, order=order)
            orders[str(order)] = {
                "tensor_relative_l2": float(np.linalg.norm(tensor - reference) / np.linalg.norm(reference)),
                "maximum_tensor_absolute_over_peak_pressure": float(
                    np.max(np.linalg.norm(tensor - reference, axis=(1, 2)))
                ),
            }
        # Positive-depth independent Boussinesq displacement differentiation,
        # integrated over the actual ellipse with a separate pressure rule.
        observations = np.array(
            [
                [0, 0, 0.2],
                [0, 0, 0.5],
                [0.4, 0.2, 0.2],
                [0.9, 0, 0.3],
                [0, 0.9, 0.4],
                [1.2, 0.3, 0.6],
                [0.2, 0.1, 1.5],
                [2, 1, 2],
            ]
        )
        observations[:, 1:] *= ratio
        integrated, integration_errors = [], []
        exact = elliptic_stress(observations, axes, order=192)
        for radial_order in (64, 128):
            t, w = np.polynomial.legendre.leggauss(radial_order)
            t, w = (t + 1) / 2, w / 2
            theta = 2 * np.pi * (np.arange(radial_order * 16) + 0.5) / (radial_order * 16)
            source = (
                np.sqrt(1 - t[:, None] ** 2)[..., None] * np.column_stack((np.cos(theta), ratio * np.sin(theta)))[None]
            ).reshape(-1, 2)
            force = np.repeat(3 * t * t * w / len(theta), len(theta)) * (2 * np.pi * ratio / 3)
            values = halfspace_stress(observations, source, force)
            integrated.append(values)
            integration_errors.append(float(np.linalg.norm(values - exact) / np.linalg.norm(exact)))
        assert integration_errors[-1] < 3e-4, (ratio, integration_errors)
        z = observations[:, 2]
        axial = elliptic_stress(np.column_stack((np.zeros((len(z), 2)), z)), axes)
        np.testing.assert_allclose(axial[:, 2, 2], -ratio / np.sqrt((1 + z * z) * (ratio * ratio + z * z)), rtol=1e-12)
        # Search the symmetry axis and complete contact edge; retain positions.
        critical = minimize_scalar(
            lambda depth: -stress_measures(elliptic_stress(np.array([[0, 0, depth]]), axes, order=96))[1][0],
            bounds=(0, 3),
            method="bounded",
            options={"xatol": 1e-11},
        )
        tension, _ = stress_measures(reference[: len(edge)])
        rows.append(
            {
                "axis_ratio": ratio,
                "orders": orders,
                "independent_pressure_integration_relative_errors": integration_errors,
                "independent_integration_change": float(
                    np.linalg.norm(integrated[1] - integrated[0]) / np.linalg.norm(exact)
                ),
                "searched_axis_von_mises_over_peak_pressure": float(-critical.fun),
                "searched_axis_von_mises_depth_over_major_axis": float(critical.x),
                "edge_tensile_over_peak_pressure": float(tension.max()),
                "edge_tensile_location_over_major_axis": edge[tension.argmax()].tolist(),
            }
        )
        path = args.output / f"ratio-{ratio}.npz"
        np.savez_compressed(
            path,
            points=points,
            stress=reference,
            independent_points=observations,
            independent_stress=integrated[-1],
            potential_stress=exact,
        )
        rows[-1]["arrays"] = file_id(path)
        write_json(args.output / "progress.json", {"rows": rows, "checks": checks})
    transfer = json.loads((args.transfer / "result.json").read_text())
    saved = next(row for row in transfer["rows"] if row["case"] == "egg")
    assert file_id(Path(saved["arrays"]["path"])) == saved["arrays"]
    with np.load(saved["arrays"]["path"]) as arrays:
        coarse = arrays["native_stress_pa"][1]
    axes, _, _, peak = ellipse(0.2, 1e10 / (1 - 0.3**2), np.diag([50.0, 80.0]))
    critical = minimize_scalar(
        lambda depth: -stress_measures(elliptic_stress(np.array([[0, 0, depth]]), axes, peak=peak, order=96))[1][0],
        bounds=(0, 3 * axes[0]),
        method="bounded",
        options={"xatol": 1e-14},
    )
    angle = np.linspace(0, 2 * np.pi, 257)
    edge = np.column_stack((axes[0] * np.cos(angle), axes[1] * np.sin(angle), np.zeros(len(angle))))
    edge_stress = elliptic_stress(edge, axes, peak=peak, order=96)
    edge_tension, _ = stress_measures(edge_stress)
    coarse_tension, coarse_vm = stress_measures(coarse)
    gap = {
        "source": file_id(args.transfer / "result.json"),
        "normal_load_n": 0.2,
        "axes_m": axes.tolist(),
        "peak_pressure_pa": peak,
        "halfspace_axis_von_mises_pa": float(-critical.fun),
        "halfspace_axis_von_mises_depth_m": float(critical.x),
        "halfspace_edge_tensile_pa": float(edge_tension.max()),
        "coarse_full_von_mises_max_pa": float(coarse_vm.max()),
        "coarse_full_tensile_max_pa": float(coarse_tension.max()),
        "coarse_over_halfspace_von_mises": float(coarse_vm.max() / -critical.fun),
        "coarse_over_halfspace_tensile": float(coarse_tension.max() / edge_tension.max()),
        "scope": "Scale comparison for the retained supplied Hertz ellipse. Half-space local candidates and a coarse faceted mass-relieved shell are different elastic domains; this is not a qualified shell error or an additive correction.",
    }
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "checks": checks,
            "rows": rows,
            "coarse_scale_gap": gap,
            "scope": "Normal elliptic half-space tensor including exact surface traction. Independent circular closed forms and positive-depth pressure integration; axis/edge searched peaks do not certify global peaks under arbitrary superposed loading. No shell, tangential-history or production lifecycle admission.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
