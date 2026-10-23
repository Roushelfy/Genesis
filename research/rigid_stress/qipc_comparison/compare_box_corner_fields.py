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


def stress_errors(probes, reference, estimate):
    """Measure tensor and sampled peak errors on identical positive-depth probes."""
    vm, trial_vm = von_mises(reference), von_mises(estimate)
    tensile, trial_tensile = np.linalg.eigvalsh(reference)[:, -1], np.linalg.eigvalsh(estimate)[:, -1]
    tension, trial_tension = tensile.max(), trial_tensile.max()
    assert tension > 0
    error = np.linalg.norm(estimate - reference) / np.linalg.norm(reference)
    peak_error = abs(trial_vm.max() - vm.max()) / vm.max()
    tensile_error = abs(trial_tension - tension) / tension
    return {
        "tensor_relative_l2": float(error),
        "reference_sample_vm_peak_pa": float(vm.max()),
        "estimate_sample_vm_peak_pa": float(trial_vm.max()),
        "sample_vm_peak_relative_error": float(peak_error),
        "reference_sample_tensile_peak_pa": float(tension),
        "estimate_sample_tensile_peak_pa": float(trial_tension),
        "sample_tensile_peak_relative_error": float(tensile_error),
        "sample_vm_peak_location_difference_m": float(np.linalg.norm(probes[vm.argmax()] - probes[trial_vm.argmax()])),
        "sample_tensile_peak_location_difference_m": float(
            np.linalg.norm(probes[tensile.argmax()] - probes[trial_tensile.argmax()])
        ),
        "tensor_5pct_peaks_10pct_met": bool(error < 0.05 and max(peak_error, tensile_error) < 0.10),
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--moments", type=Path, required=True)
    cli.add_argument("--source-order", type=int, default=8)
    cli.add_argument("--diagnose-face-profiles", action="store_true")
    cli.add_argument("--baseline-fields", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "inputs": [
                file_id(p)
                for p in [
                    args.reference / "result.json",
                    args.reference / "fields.npz",
                    args.moments,
                    args.baseline_fields,
                ]
                if p is not None
            ],
            "source_order": args.source_order,
            "diagnose_face_profiles": args.diagnose_face_profiles,
        },
    )
    reference = json.loads((args.reference / "result.json").read_text())
    assert reference["passed"] and reference["finite_face_membership_passed"]
    assert reference["remote_boundary_force_fraction"] < 1e-10
    shape = json.loads(args.moments.read_text())["rows"][0]
    with np.load(args.reference / "fields.npz") as saved:
        active = np.linalg.norm(saved["vector_force_n"], axis=1) > 0
        points, widths, forces = saved["points_m"][active], saved["widths_m"][active], saved["vector_force_n"][active]
        labels, compression = saved["labels"][active], saved["normal_force_n"][active]
    face_moments = []
    for face in [0, 1]:
        selected = labels == face
        weight = compression[selected] / compression[selected].sum()
        center = weight @ points[selected]
        y = points[selected, 1] - center[1]
        dy = widths[selected, 1]
        second = weight @ (y**2 + dy**2 / 12)
        fourth = weight @ (y**4 + y**2 * dy**2 / 2 + dy**4 / 80)
        face_moments.append(
            {
                "face": face,
                "normal_force_n": float(compression[selected].sum()),
                "centroid_m": center.tolist(),
                "longitudinal_variance_m2": float(second),
                "parabolic_half_length_from_second_moment_m": float(np.sqrt(5 * second)),
                "longitudinal_fourth_over_second_squared": float(fourth / second**2),
                "scope": "Reference-derived force moments diagnose longitudinal profile structure. These parameters are measured from the reference, rather than independently predicted runtime inputs.",
            }
        )
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
    profile_tensor = np.zeros_like(tensor)
    if args.diagnose_face_profiles:
        knots, weights = np.polynomial.legendre.leggauss(128)
        normals = np.array([[slope, 0, 1], [-1, 0, slope]]) / np.sqrt(1 + slope**2)
        for face, moment in enumerate(face_moments):
            positions = np.tile(moment["centroid_m"], (len(knots), 1))
            positions[:, 1] += moment["parabolic_half_length_from_second_moment_m"] * knots
            traction = (0.75 * (1 - knots**2) * weights * moment["normal_force_n"])[:, None] * normals[face]
            np.testing.assert_allclose(traction.sum(axis=0), moment["normal_force_n"] * normals[face], atol=1e-15)
            profile_tensor += halfspace_stress(probes, positions, traction[:, 2], poisson) + cerruti_stress(
                probes, positions, traction[:, :2], poisson
            )
    baseline = None
    if args.baseline_fields is not None:
        with np.load(args.baseline_fields) as saved:
            np.testing.assert_array_equal(saved["probes_m"], probes)
            baseline = saved["reference_tensor_pa"]
    rows = []
    for depth in depths:
        selected = np.isclose(probes[:, 2], depth, rtol=1e-12, atol=0)
        exact, trial = tensor[selected], estimate[selected]
        rows.append({"depth_m": float(depth), **stress_errors(probes[selected], exact, trial)})
        if args.diagnose_face_profiles:
            rows[-1]["two_face_moment_fit_diagnostic"] = stress_errors(
                probes[selected], exact, profile_tensor[selected]
            )
        if baseline is not None:
            old = baseline[selected]
            old_error = stress_errors(probes[selected], exact, old)
            rows[-1]["reference_refinement"] = {
                "tensor_relative_l2": old_error["tensor_relative_l2"],
                "sample_vm_peak_relative_change": old_error["sample_vm_peak_relative_error"],
                "sample_tensile_peak_relative_change": old_error["sample_tensile_peak_relative_error"],
            }
    np.savez_compressed(
        args.output / "fields.npz", probes_m=probes, reference_tensor_pa=tensor, estimate_tensor_pa=estimate
    )
    if args.diagnose_face_profiles:
        np.savez_compressed(args.output / "moment-fit-diagnostic.npz", probes_m=probes, tensor_pa=profile_tensor)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "total_force_material_n": total.tolist(),
            "reference_face_pressure_moments": face_moments,
            "effective_shape": {"width_m": width, "half_length_m": along, "exponent": exponent},
            "moment_fit_scope": "The optional two-face parabolic line diagnostic fits each reference centroid and longitudinal second moment. It isolates profile-shape sensitivity with measured inputs. Independent runtime prediction and accuracy qualification require separate evidence.",
            "scope": "Same total force compares the native-shaped effective traction distribution with an independently solved frictionless two-face local elastic control. Both stresses use the undeformed flat material Green tensor. Pressure shape is transferred without fitting the reference forces. Positive-depth probes and discrete reference/source quadrature convergence limit this comparison. Frictional history, surface/global peaks and complete curved-shell physical accuracy remain separate requirements.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
