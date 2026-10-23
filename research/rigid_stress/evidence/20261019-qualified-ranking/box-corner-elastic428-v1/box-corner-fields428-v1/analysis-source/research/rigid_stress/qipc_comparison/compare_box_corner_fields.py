"""Compare an effective slender-edge traction field with the two-face elastic control."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .edge_asymptotic import EdgeScales
from .edge_fields import continuous_tensor
from .hertz_reference import halfspace_stress
from .hertz_tangential import cerruti_stress
from .study_franka_contact_gap import von_mises


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--source-order", type=int, default=8)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [
                file_id(p) for p in [args.reference / "result.json", args.reference / "fields.npz", args.moments]
            ],
            "source_order": args.source_order,
        },
    )
    reference = json.loads((args.reference / "result.json").read_text())
    assert reference["passed"] and reference["finite_face_membership_passed"]
    assert reference["remote_boundary_force_fraction"] < 1e-10
    shape = json.loads(args.moments.read_text())["rows"][0]
    with np.load(args.reference / "fields.npz") as saved:
        active = np.linalg.norm(saved["vector_force_n"], axis=1) > 0
        points, widths, forces = saved["points_m"][active], saved["widths_m"][active], saved["vector_force_n"][active]
    total = forces.sum(axis=0)
    poisson, slope = reference["poisson"], reference["slope"]
    width, along, exponent = shape["width_m"] * np.sqrt(1 + slope**2), shape["half_length_m"], shape["exponent"]
    pressure = EdgeScales(
        width, along, exponent, 3 * total[2] * np.sin(np.pi * exponent) / (4 * np.pi * exponent * width * along)
    )
    knots, weights = np.polynomial.legendre.leggauss(args.source_order)
    offsets = np.array(np.meshgrid(knots / 2, knots / 2, indexing="ij")).reshape(2, -1).T
    weights = np.outer(weights, weights).reshape(-1) / 4
    sources = (points[:, None] + widths[:, None] * offsets).reshape(-1, 2)
    source_forces = (forces[:, None] * weights[None, :, None]).reshape(-1, 3)
    depths = np.array([0.1, 0.25, 0.5, 1.0]) * along
    probes = (
        np.array(
            np.meshgrid(np.linspace(-0.75, 0.75, 11) * along, np.linspace(-1.5, 1.5, 13) * along, depths, indexing="ij")
        )
        .reshape(3, -1)
        .T
    )
    tensor = halfspace_stress(probes, sources, source_forces[:, 2], poisson) + cerruti_stress(
        probes, sources, source_forces[:, :2], poisson
    )
    estimate = continuous_tensor(
        probes, pressure, total[2], total[:2] / total[2], order_x=48, order_y=48, poisson=poisson
    )
    rows = []
    for depth in depths:
        selected = np.isclose(probes[:, 2], depth, rtol=1e-12, atol=0)
        exact, trial = tensor[selected], estimate[selected]
        vm, trial_vm = von_mises(exact), von_mises(trial)
        tension, trial_tension = np.linalg.eigvalsh(exact)[:, -1].max(), np.linalg.eigvalsh(trial)[:, -1].max()
        error = np.linalg.norm(trial - exact) / np.linalg.norm(exact)
        peak_error = abs(trial_vm.max() - vm.max()) / vm.max()
        rows.append(
            {
                "depth_m": float(depth),
                "tensor_relative_l2": float(error),
                "reference_sample_vm_peak_pa": float(vm.max()),
                "estimate_sample_vm_peak_pa": float(trial_vm.max()),
                "sample_vm_peak_relative_error": float(peak_error),
                "reference_sample_tensile_peak_pa": float(tension),
                "estimate_sample_tensile_peak_pa": float(trial_tension),
                "tensor_5pct_vm_peak_10pct_met": bool(error < 0.05 and peak_error < 0.10),
            }
        )
    np.savez_compressed(
        args.output / "fields.npz", probes_m=probes, reference_tensor_pa=tensor, estimate_tensor_pa=estimate
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "total_force_material_n": total.tolist(),
            "effective_shape": {"width_m": width, "half_length_m": along, "exponent": exponent},
            "scope": "Same total force compares the native-shaped effective traction distribution with an independently solved frictionless two-face local elastic control. Both stresses use the undeformed flat material Green tensor. Pressure shape is transferred without fitting the reference forces. Positive-depth probes and discrete reference/source quadrature convergence limit this comparison. Frictional history, surface/global peaks and complete curved-shell physical accuracy remain separate requirements.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
