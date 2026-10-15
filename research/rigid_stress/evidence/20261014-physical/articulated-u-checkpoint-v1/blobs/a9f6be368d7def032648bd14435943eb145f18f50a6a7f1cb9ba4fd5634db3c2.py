"""Reconstruct three resolved U fields and retain their unresolved spatial tails."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .sample_load_fields import common_physical_probes, locate_probes, sample_field
from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) >= 3
    args.output.mkdir(parents=True, exist_ok=False)
    probes = common_physical_probes("u")
    levels, fields, hashes = [], [], {}
    baseline = None
    for index, run in enumerate(args.run):
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
        config = json.loads((run / "solution/configuration.json").read_text())
        result = json.loads((run / "solution/result.json").read_text())
        physical = {
            k: config[k]
            for k in [
                "geometry",
                "pad_young",
                "force",
                "dhat",
                "friction",
                "initial_gap",
                "pad_thickness",
                "edge_lift",
                "force_steps",
                "body_rigid_coordinates",
                "updated_friction",
            ]
        }
        if baseline is None:
            baseline = physical
        assert physical == baseline
        assert result["gravity_m_per_s2"] == [0, -9.81, 0]
        assert result["complete_constrained_residual_N"] <= 1e-9
        for row in result["accepted_steps"]:
            assert row["complete_free_residual_n"] <= 1e-9
        mesh = Path(config["mesh"])
        assert hashlib.sha256(mesh.read_bytes()).hexdigest() == result["mesh_sha256"]
        with np.load(mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        with np.load(run / "solution/fields.npz") as saved:
            field = {k: saved[k].copy() for k in saved.files}
        permutation = field["body_component_permutation"]
        count = reference.basis.N
        vector = np.empty(count)
        vector[permutation] = field["displacement"][:count]
        tensor, vm = reference.corner_stress(vector)
        np.testing.assert_allclose(tensor, field["body_tensor_Pa"], rtol=1e-10, atol=1e-6)
        np.testing.assert_allclose(vm, field["body_vm_Pa"], rtol=1e-10, atol=1e-6)
        gradient = (reference.stiffness @ vector)[permutation]
        residual = gradient - field["applied_force_N"][:count] - field["contact_force_N"][:count]
        assert np.linalg.norm(residual) <= 1e-9
        np.testing.assert_allclose(residual, field["total_gradient_N"][:count], rtol=0, atol=1e-12)
        elastic = reference.align_rigid_gauge(vector)
        elements, valid, barycentric = locate_probes(reference.mesh, probes)
        sampled_tensor, sampled_vm, sampled_u = sample_field(
            reference.basis,
            elements,
            valid,
            barycentric,
            probes,
            {"tensor_Pa": tensor, "displacement": elastic},
        )
        observation = {"valid": valid, "tensor_Pa": sampled_tensor, "vm_Pa": sampled_vm, "displacement_m": sampled_u}
        fields.append(observation)
        nodes = reference.component_dofs
        contact = field["contact_force_N"][:count].reshape(-1, 3)
        positions = field["rest_positions"][: count // 3] + field["displacement"][:count].reshape(-1, 3)
        peak_cell, peak_corner = np.unravel_index(np.argmax(vm), vm.shape)
        peak_position = reference.mesh.p[:, reference.mesh.t[peak_corner, peak_cell]]
        np.testing.assert_allclose(peak_position, result["raw_body_peak_position_m"], rtol=0, atol=1e-14)
        row = {
            "run": str(run),
            "tetrahedra": reference.mesh.nelements,
            "pad_resolution": config["pad_resolution"],
            "raw_peak_Pa": float(vm.max()),
            "raw_peak_position_m": peak_position.tolist(),
            "body_energy_J": float(0.5 * elastic @ (reference.stiffness @ elastic)),
            "body_force_residual_N": float(np.linalg.norm(residual)),
            "accepted_steps": len(result["accepted_steps"]),
            "max_original_accepted_residual_N": max(r["complete_free_residual_n"] for r in result["accepted_steps"]),
            "contact_force_N": contact.sum(axis=0).tolist(),
            "contact_moment_about_reference_com_Nm": np.cross(positions - reference.com, contact).sum(axis=0).tolist(),
            "elastic_nodal_displacement_max_m": float(np.linalg.norm(elastic[nodes], axis=1).max()),
            "outside_probe_indices": np.flatnonzero(~valid).tolist(),
            "centre_gauge": config.get("centre_gauge", False),
            "initial_common_x_shift_m": result.get("initial_common_x_shift_m", 0),
        }
        levels.append(row)
        np.savez_compressed(args.output / f"level{index}.npz", points_m=probes, **observation, body_residual_N=residual)
        for path in [mesh, *(run / "solution" / name for name in ["fields.npz", "configuration.json", "result.json"])]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    common = np.logical_and.reduce([f["valid"] for f in fields])
    assert common.any()
    changes = []
    for index in range(1, len(fields)):
        change = {"coarse": index - 1, "fine": index, "common_valid_probes": int(common.sum())}
        for key in ["tensor_Pa", "vm_Pa", "displacement_m"]:
            relative, absolute = relative_change(fields[index - 1][key][common], fields[index][key][common])
            change[key] = {"relative_change": relative, "absolute_change": absolute}
        for key in ["raw_peak_Pa", "body_energy_J"]:
            change[key] = abs(levels[index - 1][key] / levels[index][key] - 1)
        changes.append(change)
    report = {
        "scope": __doc__,
        "physical_configuration": baseline,
        "levels": levels,
        "adjacent_changes": changes,
        "input_sha256": hashes,
        "goal_completion": False,
        "limitations": "Body geometry and pad discretization change together. The final-level common-x gauge/cache is an explicitly qualified reference numerical option, not a changed physical load. Original linear body/pad laws remain: finite-rotation/nonlinear compliance, separated geometry/contact axes, regularization/history tails and native feedback/model/accuracy-cost acceptance remain required. No tail passes merely from three levels or a stable resultant. Raw global peaks and outside probes remain visible.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"levels": len(levels), "adjacent_changes": changes}), flush=True)


if __name__ == "__main__":
    main()
