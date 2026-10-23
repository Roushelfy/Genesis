"""Resolve sharp-edge near fields before complete-shell/source-shift integration."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import roots_jacobi

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import edge_scales
from .edge_fields import composite_tensor, continuous_tensor, strip_tensor
from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress
from .study_franka_contact_gap import von_mises


def strip_controls():
    """Compare the new closed form with the independent 3D displacement kernels."""
    width, amplitude, exponent = 0.7, 1.3, 0.56
    points = np.array([[0.13, 0, 0.2], [-0.3, 0, 0.4], [1.1, 0, 0.3], [0.6, 0, 0.8]])
    tangent = np.array([0.65, -0.4])
    x, wx = roots_jacobi(128, exponent, -exponent)
    x, wx = width * (x + 1) / 2, wx / wx.sum()
    theta, wy = np.polynomial.legendre.leggauss(192)
    y = np.tan(np.pi * theta / 2)
    wy = wy * np.pi / 2 * (1 + y * y)
    source = np.column_stack((np.tile(x, len(y)), np.repeat(y, len(x))))
    line_load = np.pi * amplitude * width * exponent / np.sin(np.pi * exponent)
    force = (line_load * wy[:, None] * wx).reshape(-1)
    numerical = halfspace_stress(points, source, force) + cerruti_stress(points, source, force[:, None] * tangent)
    analytical = strip_tensor(points, width, exponent, amplitude, tangent)
    error = np.linalg.norm(numerical - analytical) / np.linalg.norm(numerical)
    np.testing.assert_allclose(numerical, analytical, rtol=1e-7, atol=1e-8)
    surface = np.array([[0.1, 0, 0], [0.3, 0, 0], [0.6, 0, 0]])
    tensor = strip_tensor(surface, width, exponent, amplitude, tangent)
    pressure = amplitude * ((width - surface[:, 0]) / surface[:, 0]) ** exponent
    np.testing.assert_allclose(tensor[:, :, 2], -pressure[:, None] * [*tangent, 1], rtol=1e-12, atol=1e-12)
    return {"independent_3d_strip_relative": float(error), "surface_traction_checked": True}


def field_metrics(value, reference):
    norms = np.linalg.norm(reference, axis=(1, 2))
    return {
        "tensor_relative": float(np.linalg.norm(value - reference) / np.linalg.norm(reference)),
        "maximum_tensor_error_over_reference_maximum": float(
            np.linalg.norm(value - reference, axis=(1, 2)).max() / norms.max()
        ),
        "vm_peak_relative": float(abs(von_mises(value).max() / von_mises(reference).max() - 1)),
        "tensile_peak_relative": float(
            abs(np.linalg.eigvalsh(value)[:, -1].max() / np.linalg.eigvalsh(reference)[:, -1].max() - 1)
        ),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path)
    cli.add_argument("--coupling", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {"reference_tensor": 0.005, "reference_peak": 0.01, "composite_tensor": 0.05, "composite_peak": 0.10}
    inputs = [] if args.reference is None else [args.reference / "result.json", args.coupling / "result.json"]
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [file_id(p) for p in inputs],
            "targets": targets,
            "scope": "Continuous prescribed sharp-edge pressure, positive-depth samples down to 0.1 transverse width and out to four half-lengths. Independent Boussinesq/Cerruti displacement gradients integrate both pressure coordinates. A local strip plus parabolic centroid-line correction is tested without fitting stresses. This isolates pressure-to-stress approximation, not physical contact pressure, friction history, complete shell fields or global surface peaks.",
        },
    )
    checks = strip_controls()
    if args.reference is None:
        write_json(args.output / "result.json", {"passed": True, "controls": checks, "whole_goal_complete": False})
        return
    reference, coupling = [json.loads(p.read_text()) for p in inputs]
    load, modulus = reference["normal_load_n"], 1e10 / 0.91
    ratio = np.array(coupling["tangent_over_normal_components"])
    rows = []
    for direction in [1, -1]:
        model = edge_scales(
            load, modulus, reference["edge_slope"], reference["edge_along_curvature_per_m"], direction * ratio[0]
        )
        depths = np.r_[model.width * np.array([0.1, 0.3, 1, 3]), model.along * np.array([0.1, 0.25, 1, 4])]
        points = np.stack(
            np.meshgrid(
                model.width * np.array([-1, 0, 0.25, 0.75, 1.5]),
                model.along * np.array([0, 0.5, 0.9, 1.1]),
                depths,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        far = np.stack(
            np.meshgrid(
                model.along * np.array([-1, 0.5, 2]),
                model.along * np.array([0, 1, 2]),
                model.along * np.array([0.1, 1, 4]),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        points = np.concatenate([points, far])
        started = perf_counter()
        low = continuous_tensor(points, model, load, direction * ratio, 64, 48)
        high = continuous_tensor(points, model, load, direction * ratio, 96, 72)
        seconds = perf_counter() - started
        arrays = {"points_m": points, "reference_low_pa": low, "reference_high_pa": high}
        reference_change = field_metrics(low, high)
        candidates = []
        for order in [16, 32, 64]:
            started = perf_counter()
            value = composite_tensor(points, model, load, direction * ratio, order)
            arrays[f"composite{order}_pa"] = value
            candidate = {
                "order": order,
                "seconds": perf_counter() - started,
                **field_metrics(value, high),
                "depth_groups": [],
            }
            for depth in depths:
                selected = points[:, 2] == depth
                candidate["depth_groups"].append(
                    {
                        "depth_m": float(depth),
                        "depth_over_width": float(depth / model.width),
                        **field_metrics(value[selected], high[selected]),
                    }
                )
            candidates.append(candidate)
        np.savez_compressed(args.output / f"direction{direction}.npz", **arrays)
        rows.append(
            {
                "direction": direction,
                "reference_seconds": seconds,
                "reference_change": reference_change,
                "candidates": candidates,
            }
        )
        write_json(args.output / "progress.json", {"rows": rows})
    reference_passed = all(
        r["reference_change"]["tensor_relative"] < targets["reference_tensor"]
        and max(r["reference_change"]["vm_peak_relative"], r["reference_change"]["tensile_peak_relative"])
        < targets["reference_peak"]
        for r in rows
    )
    candidate_passed = all(
        r["candidates"][1]["tensor_relative"] < targets["composite_tensor"]
        and max(r["candidates"][1]["vm_peak_relative"], r["candidates"][1]["tensile_peak_relative"])
        < targets["composite_peak"]
        for r in rows
    )
    write_json(
        args.output / "result.json",
        {
            "passed": reference_passed,
            "controls": checks,
            "rows": rows,
            "composite32_targets_met": candidate_passed,
            "whole_goal_complete": False,
        },
    )
    assert reference_passed


if __name__ == "__main__":
    main()
