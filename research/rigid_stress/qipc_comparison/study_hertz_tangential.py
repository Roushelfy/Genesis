"""Check elliptic tangential tensors and measure a proportional-traction approximation."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_stress import elliptic_stress
from .hertz_tangential import cerruti_stress, elliptic_tangential_stress
from .study_hertz_nearfield import stress_measures


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    rows, fields = [], {}
    tangent = np.array([0.8, -0.6])
    for ratio in (1.0, 0.4, 0.05):
        axes = np.array([1.0, ratio])
        points = np.array([[0, 0, 0.25], [0.4, 0.2, 0.3], [0.9, 0, 0.4], [0, 0.9, 0.4], [1.2, 0.3, 0.6], [2, 1, 2]])
        points[:, 1] *= ratio
        points[:, 2] *= np.sqrt(ratio)
        reference = elliptic_tangential_stress(points, axes, tangent, order=192)
        integrated = []
        for order in (64, 128):
            t, weight = np.polynomial.legendre.leggauss(order)
            t, weight = (t + 1) / 2, weight / 2
            angle_count = 1024 if ratio < 0.1 else 8 * order
            angle = 2 * np.pi * (np.arange(angle_count) + 0.5) / angle_count
            sources = (
                np.sqrt(1 - t[:, None] ** 2)[..., None] * np.column_stack((np.cos(angle), ratio * np.sin(angle)))[None]
            ).reshape(-1, 2)
            force = np.repeat(t**2 * weight * 2 * np.pi * ratio / angle_count, angle_count)[:, None] * tangent
            integrated.append(np.concatenate([cerruti_stress(point[None], sources, force) for point in points]))
        difference = np.linalg.norm(reference - integrated[-1]) / np.linalg.norm(integrated[-1])
        refinement = np.linalg.norm(integrated[-1] - integrated[-2]) / np.linalg.norm(integrated[-1])
        row = {
            "ratio": ratio,
            "independent_cerruti_relative": float(difference),
            "cerruti_refinement_relative": float(refinement),
        }
        rows.append(row)
        write_json(args.output / "progress.json", {"rows": rows})
        assert difference < 3e-4 and refinement < 3e-4, row
        angle = np.linspace(0, 2 * np.pi, 33)
        surface = np.column_stack((np.cos(angle), ratio * np.sin(angle), np.zeros(len(angle))))
        axial = np.column_stack((np.zeros((40, 2)), np.geomspace(1e-7, 20, 40)))
        exterior = np.array([[1.00001, 0, 0], [1.00001, 0, 1e-8], [2, 1, 0], [20, 5, 20], [200, 50, 200]])
        exterior[:, 1] *= ratio
        probes = np.concatenate((surface, axial, exterior, points))
        high = elliptic_tangential_stress(probes, axes, tangent, order=192)
        row["orders"] = {
            str(order): float(
                np.linalg.norm(elliptic_tangential_stress(probes, axes, tangent, order=order) - high, axis=(1, 2)).max()
            )
            for order in (16, 32, 48, 96)
        }
        assert np.isfinite(high).all()
        fields[f"ratio{ratio}_probes"] = probes
        fields[f"ratio{ratio}_tensors"] = high
        fields[f"ratio{ratio}_cerruti"] = integrated[-1]
        write_json(args.output / "progress.json", {"rows": rows})

    # Constant normal load and a collinear loading/reversal cycle in the
    # elastically decoupled Cattaneo--Mindlin approximation. The reference
    # traction has memory; it is not a coupled rigid-indenter contact reference.
    points = (
        np.array(
            np.meshgrid(
                np.linspace(-1.5, 1.5, 21),
                np.linspace(-1.5, 1.5, 21),
                np.r_[0, np.geomspace(0.03, 2, 10)],
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )
    normal = elliptic_stress(points, np.ones(2), order=96)
    unit_tangent = np.array([1.0, 0.0])
    proportional = elliptic_tangential_stress(points, np.ones(2), unit_tangent, order=96)
    mu, maximum = 0.4, 0.6
    stick = (1 - maximum) ** (1 / 3)
    history = []
    for phase, fraction in [
        ("loading", 0.15),
        ("loading", 0.6),
        ("unloading", 0.3),
        ("unloading", 0.0),
        ("unloading", -0.3),
        ("unloading", -0.6),
        ("reloading", 0.0),
        ("reloading", 0.6),
    ]:
        if phase == "loading":
            terms = [(1.0, 1.0), ((1 - fraction) ** (1 / 3), -1.0)]
        elif phase == "unloading":
            terms = [(1.0, -1.0), (stick, -1.0), ((1 - (maximum - fraction) / 2) ** (1 / 3), 2.0)]
        else:
            terms = [(1.0, 1.0), (stick, 1.0), ((1 - (fraction + maximum) / 2) ** (1 / 3), -2.0)]
        np.testing.assert_allclose(sum(coefficient * radius**3 for radius, coefficient in terms), fraction, atol=1e-14)
        remembered = mu * sum(
            coefficient * elliptic_tangential_stress(points, np.full(2, radius), unit_tangent, peak=radius, order=96)
            for radius, coefficient in terms
        )
        approximation = mu * fraction * proportional
        reference, estimate = normal + remembered, normal + approximation
        tension, mises = stress_measures(reference)
        predicted_tension, predicted_mises = stress_measures(estimate)
        history.append(
            {
                "phase": phase,
                "force_over_mu_normal": fraction,
                "total_sample_tensor_relative": float(np.linalg.norm(estimate - reference) / np.linalg.norm(reference)),
                "tangential_sample_tensor_relative": float(
                    np.linalg.norm(approximation - remembered) / np.linalg.norm(remembered)
                ),
                "sample_tensile_peak_relative": float(abs(predicted_tension.max() / tension.max() - 1)),
                "sample_vm_peak_relative": float(abs(predicted_mises.max() / mises.max() - 1)),
                "reference_sample_tensile_peak_over_normal_peak": float(tension.max()),
                "reference_sample_vm_peak_over_normal_peak": float(mises.max()),
                "reference_tensile_peak_position": points[tension.argmax()].tolist(),
                "reference_vm_peak_position": points[mises.argmax()].tolist(),
            }
        )
        fields[f"{phase}{fraction}_reference"] = reference
        fields[f"{phase}{fraction}_proportional"] = estimate
    fields["history_points"] = points
    path = args.output / "fields.npz"
    np.savez_compressed(path, **fields)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "elliptic_controls": rows,
            "history": history,
            "arrays": file_id(path),
            "scope": "Independent positive-depth Cerruti pressure integration checks the elliptic tangential stress evaluator. Surface/edge/axis quadrature sensitivity is separate. Proportional traction is compared with the constant-normal-load collinear Cattaneo-Mindlin loading/reversal approximation on a fixed sample grid. This does not qualify coupled friction contact, complete-volume or continuum maxima, a finite shell, arbitrary history, or native/scene throughput.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
