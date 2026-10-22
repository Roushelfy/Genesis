"""Compare native curvature variants using one retained independent smooth-gap contact reference."""

import argparse
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_reference import ellipse_pressure, halfspace_stress
from .study_franka_contact_gap import von_mises


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, required=True)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "baseline": file_id(args.baseline),
            "candidate": file_id(args.candidate),
            "reference": file_id(args.reference),
            "targets": {"pressure_relative": 0.05, "sample_tensor_relative": 0.05, "sample_peak_relative": 0.10},
            "scope": "Actual first-loaded native Franka frame with matched rigid/contact inputs. Pressure and positive-depth half-space stress use the retained independent smooth-egg gap reference. The original flat-source rejection remains visible; no accepted shell update or Franka stress throughput is claimed.",
        },
    )
    with np.load(args.baseline) as data:
        baseline = {key: data[key] for key in data.files}
    with np.load(args.candidate) as data:
        candidate = {key: data[key] for key in data.files}
    matched = [
        "contact_force_n",
        "contact_valid",
        "contact_position_m",
        "contact_normal",
        "source_witness_m",
        "source_geom",
        "source_pair",
        "hertz_centers_m",
        "hertz_frames",
        "law_force_n",
        "law_status",
        "position_m",
        "quaternion",
        "recovery_status",
        "step_valid",
    ]
    for key in matched:
        np.testing.assert_array_equal(candidate[key], baseline[key], err_msg=key)
    assert not candidate["step_valid"].any()
    with np.load(args.reference) as data:
        points, reference_force = data["positions_m"], data["force_n"]
        observations, reference_tensor = data["observation_points_m"], data["tensor_pa"]
    width = np.diff(np.unique(points[:, 0]))[0]
    gauss, weights = np.polynomial.legendre.leggauss(16)
    offsets = np.array(np.meshgrid(gauss, gauss, indexing="ij")).reshape(2, -1).T * width / 2
    weights = np.outer(weights, weights).reshape(-1) / 4
    samples = (points[:, None] + offsets[None]).reshape(-1, 2)
    direction = baseline["law_direction"][0]
    rotation = np.array([[direction[0], -direction[1]], [direction[1], direction[0]]])
    rows, arrays = [], {}
    for name, data in [("baseline", baseline), ("quadric", candidate)]:
        force = (
            ellipse_pressure(
                samples, data["law_axes_m"][0], rotation.T @ data["law_direction"][0], data["law_pressure_pa"][0]
            ).reshape(len(points), -1)
            @ weights
            * width**2
        )
        mass_error = abs(force.sum() / data["law_force_n"][0] - 1)
        assert mass_error < 1e-3
        force *= data["law_force_n"][0] / force.sum()
        tensor = halfspace_stress(observations, points, force)
        pressure_error = np.linalg.norm(force - reference_force) / np.linalg.norm(reference_force)
        tensor_error = np.linalg.norm(tensor - reference_tensor) / np.linalg.norm(reference_tensor)
        vm_error = abs(von_mises(tensor).max() / von_mises(reference_tensor).max() - 1)
        tension_error = abs(
            np.linalg.eigvalsh(tensor)[:, -1].max() / np.linalg.eigvalsh(reference_tensor)[:, -1].max() - 1
        )
        rows.append(
            {
                "variant": name,
                "axes_m": data["law_axes_m"][0].tolist(),
                "pressure_relative": float(pressure_error),
                "sample_tensor_relative": float(tensor_error),
                "sample_vm_peak_relative": float(vm_error),
                "sample_tensile_peak_relative": float(tension_error),
                "pressure_quadrature_force_relative": float(mass_error),
                "targets_met": bool(
                    pressure_error < 0.05 and tensor_error < 0.05 and max(vm_error, tension_error) < 0.10
                ),
                "accepted_loaded_environments": int(data["step_valid"].sum()),
                "recovery_status": data["recovery_status"].tolist(),
            }
        )
        arrays[name + "_pressure_force_n"] = force
        arrays[name + "_tensor_pa"] = tensor
    assert rows[-1]["targets_met"]
    np.savez_compressed(args.output / "fields.npz", **arrays)
    write_json(
        args.output / "result.json",
        {"passed": True, "matched_arrays": matched, "rows": rows, "whole_goal_complete": False},
    )


if __name__ == "__main__":
    main()
