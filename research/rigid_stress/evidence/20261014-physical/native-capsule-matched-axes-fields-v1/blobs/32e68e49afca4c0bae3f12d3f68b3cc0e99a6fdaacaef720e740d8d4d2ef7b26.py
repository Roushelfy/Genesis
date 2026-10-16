"""Audit converged friction controls against independently meshed bonded sticking elasticity."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contact_reference import block_reference


def source_closure(run):
    assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    for line in (run / "source.sha256").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((run / "source" / name).read_bytes()).hexdigest() == expected, (run, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--bonded-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_closure(args.contact_run)
    source_closure(args.bonded_run)
    rows = []
    for result_path in sorted(args.contact_run.glob("ipc-*/result.json")):
        result = json.loads(result_path.read_text())
        assert result["complete_free_residual_N"] <= 1e-9
        accepted = result["accepted_steps"]
        assert len(accepted) == result["approach_steps"] + result["load_steps"] + result["shear_steps"]
        assert max(step["complete_free_residual_n"] for step in accepted) <= 1e-9
        for iteration in result["iterations"]:
            assert 0 < iteration["accepted_step"] <= iteration["collision_free_step"] <= 1
        resolution = result["resolution"]
        bonded_path = args.bonded_run / f"bonded-n{resolution}-s5e-6"
        bonded = json.loads((bonded_path / "result.json").read_text())
        assert bonded["minimum_both_sides_corner_coulomb_margin_Pa"] > 0
        with np.load(bonded_path / "fields.npz") as field:
            probes = field["probes_m"]
            expected_tensor = field["probe_tensor_Pa"]
            expected_vm = field["probe_vm_Pa"]
        actual_tensor = np.empty_like(expected_tensor)
        elastic_energy, raw_peaks, raw_peak_positions = 0.0, [], []
        gap = result["gap_m"]
        with np.load(result_path.parent / "fields.npz") as field:
            rest = field["rest_positions"]
            constrained = np.isclose(rest[:, 2], -0.005) | np.isclose(rest[:, 2], gap + 0.005)
            residual = np.linalg.norm(field["reactions_N"].reshape(-1, 3)[~constrained])
            np.testing.assert_allclose(residual, result["complete_free_residual_N"], rtol=1e-12, atol=1e-18)
            offset = 0
            for name, bounds in (
                ("lower", ((-0.005, -0.005, -0.005), (0.005, 0.005, 0))),
                ("upper", ((-0.005, -0.005, gap), (0.005, 0.005, gap + 0.005))),
            ):
                reference = block_reference(*bounds, resolution, result["young_Pa"], result["poisson"])
                permutation = reference.component_dofs.reshape(-1)
                vector = np.empty(reference.basis.N)
                vector[permutation] = field["displacement"][offset : offset + reference.basis.N]
                offset += reference.basis.N
                tensor, vm = reference.corner_stress(vector)
                np.testing.assert_allclose(tensor, field[f"{name}_tensor_Pa"], rtol=1e-12, atol=1e-8)
                np.testing.assert_allclose(vm, field[f"{name}_vm_Pa"], rtol=1e-12, atol=1e-8)
                elastic = reference.align_rigid_gauge(vector)
                elastic_energy += 0.5 * elastic @ (reference.stiffness @ elastic)
                selected = probes[:, 2] < 0 if name == "lower" else probes[:, 2] > 0
                coordinates = probes[selected].copy()
                if name == "upper":
                    coordinates[:, 2] += gap
                actual_tensor[selected], _ = reference.stress_at_points(vector, coordinates)
                element, corner = np.unravel_index(np.argmax(vm), vm.shape)
                raw_peaks.append(float(vm[element, corner]))
                raw_peak_positions.append(reference.corner_positions[element, corner].tolist())
        deviator = actual_tensor - np.trace(actual_tensor, axis1=-2, axis2=-1)[:, None, None] * np.eye(3) / 3
        actual_vm = np.sqrt(1.5 * np.sum(deviator * deviator, axis=(-2, -1)))
        force = np.array(result["parts"][0]["contact_force_N"])
        expected_force = np.array(bonded["interface_force_N"])
        peak = max(raw_peaks)
        row = {
            "case": result_path.parent.name,
            "resolution": resolution,
            "dhat_m": result["dhat_m"],
            "contact_load_increment_m": result["contact_load_increment_m"],
            "contact_force_N": force.tolist(),
            "bonded_force_N": expected_force.tolist(),
            "normal_force_relative_error": float(abs(force[2] / expected_force[2] - 1)),
            "shear_force_relative_error": float(abs(force[0] / expected_force[0] - 1)),
            "probe_tensor_relative_error": float(
                np.linalg.norm(actual_tensor - expected_tensor) / np.linalg.norm(expected_tensor)
            ),
            "probe_vm_relative_error": float(np.linalg.norm(actual_vm - expected_vm) / np.linalg.norm(expected_vm)),
            "elastic_energy_J": float(elastic_energy),
            "elastic_energy_relative_error": float(abs(elastic_energy / bonded["strain_energy_J"] - 1)),
            "raw_global_peak_Pa": peak,
            "raw_global_peak_position_m": raw_peak_positions[np.argmax(raw_peaks)],
            "bonded_raw_global_peak_Pa": bonded["raw_global_peak_Pa"],
            "raw_peak_relative_error": float(abs(peak / bonded["raw_global_peak_Pa"] - 1)),
            "complete_free_residual_N": float(residual),
            "maximum_accepted_step_residual_N": max(step["complete_free_residual_n"] for step in accepted),
            "accepted_step_count": len(accepted),
            "minimum_bonded_corner_cone_margin_Pa": bonded["minimum_both_sides_corner_coulomb_margin_Pa"],
            "maximum_displacement_gradient": max(part["maximum_displacement_gradient"] for part in result["parts"]),
            "minimum_deformation_jacobian": min(part["minimum_deformation_jacobian"] for part in result["parts"]),
        }
        np.savez_compressed(
            args.output / (result_path.parent.name + ".npz"),
            probes_m=probes,
            actual_tensor_Pa=actual_tensor,
            expected_tensor_Pa=expected_tensor,
            actual_vm_Pa=actual_vm,
            expected_vm_Pa=expected_vm,
        )
        rows.append(row)
    assert len(rows) == 12
    (args.output / "summary.json").write_text(
        json.dumps(
            {
                "scope": "Supported full3D sticking controls; full egg/free-grasp and sliding acceptance remain unverified",
                "contact_scan_returncode": int((args.contact_run / "returncode.txt").read_text()),
                "note": "The earlier scan job fails its invalid strong-stress-flux equality assertion for bonded shear; all 12 IPC solves converge. Corrected bonded controls distinguish consistent nodal reactions from discretization error in stress-flux integration.",
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"cases": len(rows), "maximum_final_residual_N": max(row["complete_free_residual_N"] for row in rows)}
        )
    )


if __name__ == "__main__":
    main()
