"""Rebuild every original/new finite-load field and compare its unchanged probes, raw peak and energy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .extend_finite_load_reference import assemble_case, sample
from .sample_load_fields import common_physical_probes
from .skfem_reference import SkfemElasticReference
from .solve_continuous_traction_axis import check_field
from .summarize_fine_references import CASES, closed_source, relative_change


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--run", type=Path, nargs=2, required=True)
    parser.add_argument("--diagnostic-target", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.diagnostic_target == 0.03
    inputs = {}

    def retain(path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == digest
        inputs[str(path)] = digest
        return digest

    for run in [args.qualification, *args.run]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    qualification_path = args.qualification / "results/summary.json"
    retain(qualification_path)
    qualification = json.loads(qualification_path.read_text())
    assert qualification["qualify_only"] and qualification["all_original_five_loads_full_fields_and_probes_qualified"]
    config_path = args.qualification / "results/configuration.json"
    retain(config_path)
    config = json.loads(config_path.read_text())
    geometry = qualification["geometry"]
    baseline, baseline_probes = Path(config["baseline_run"]), Path(config["baseline_probes"])
    for run in [baseline, baseline_probes]:
        closed_source(run)
        assert int((run / "returncode.txt").read_text()) == 0
    runs = [baseline, *args.run]
    meshes = [Path(config["baseline_mesh"])]
    for run in args.run:
        path = run / "results/configuration.json"
        retain(path)
        child = json.loads(path.read_text())
        assert child["geometry"] == geometry and Path(child["qualification"]).resolve() == args.qualification.resolve()
        meshes.append(Path(child["mesh"]))
    points = common_physical_probes(geometry)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, changes = [], []
    previous = {}
    baseline_snapshots = {}
    counts = []
    for level, (run, mesh) in enumerate(zip(runs, meshes, strict=True)):
        mesh_hash = retain(mesh)
        with np.load(mesh) as saved:
            reference = SkfemElasticReference(saved["vertices"], saved["tetrahedra"], 1e10, 0.3, 2000)
        counts.append(reference.mesh.nelements)
        assert len(counts) < 2 or counts[-1] > counts[-2]
        for case in CASES:
            field_path = run / f"results/{case}/reference_field.npz"
            snapshot_path = run / f"results/{case}/snapshot.json"
            probe_path = (
                baseline_probes / f"results/{case}/reference_field.npz"
                if level == 0
                else run / f"results/{case}/probe_field.npz"
            )
            for path in [field_path, snapshot_path, probe_path]:
                retain(path)
            snapshot = json.loads(snapshot_path.read_text())
            assert snapshot["mesh_sha256"] == mesh_hash and snapshot["geometry"] == geometry
            if level == 0:
                baseline_snapshots[case] = snapshot
            for key in ["material", "pose", "omega_rad_s", "patch_radius_m", "reference_quadrature_level"]:
                assert snapshot[key] == baseline_snapshots[case][key]
            for key in ["physical_anchor_m", "friction", "name"]:
                assert [patch[key] for patch in snapshot["patches"]] == [
                    patch[key] for patch in baseline_snapshots[case]["patches"]
                ]
            np.testing.assert_allclose(reference.total_mass, snapshot["mass_kg"], rtol=1e-12, atol=1e-18)
            force, patches = assemble_case(
                reference, geometry, case, snapshot["patch_radius_m"], snapshot["reference_quadrature_level"]
            )
            with np.load(field_path) as saved:
                u, original_force, rhs, tensor, vm = [
                    saved[name].copy() for name in ["displacement", "force_N", "rhs_N", "tensor_Pa", "vm_Pa"]
                ]
            np.testing.assert_allclose(force, original_force, rtol=1e-11, atol=1e-13)
            checked = check_field(reference, u, force, rhs, tensor, vm, snapshot["omega_rad_s"])
            probe = sample(reference, u, tensor, points)
            with np.load(probe_path) as saved:
                np.testing.assert_array_equal(saved["points_m"], points)
                np.testing.assert_array_equal(saved["valid"], probe["valid"])
                for name, absolute in [("tensor_Pa", 1e-6), ("vm_Pa", 1e-6), ("displacement_m", 1e-13)]:
                    np.testing.assert_allclose(
                        probe[name][probe["valid"]], saved[name][probe["valid"]], rtol=1e-10, atol=absolute
                    )
            energy = float(0.5 * u @ (reference.stiffness @ u))
            np.testing.assert_allclose(energy, snapshot["reference_energy_J"], rtol=1e-10, atol=1e-20)
            cell, corner = np.unravel_index(np.argmax(vm), vm.shape)
            position = reference.corner_positions[cell, corner]
            np.testing.assert_allclose(vm.max(), snapshot["reference_peak_Pa"], rtol=1e-10, atol=1e-6)
            np.testing.assert_allclose(position, snapshot["reference_peak_position_m"], rtol=0, atol=1e-14)
            row = {
                "geometry": geometry,
                "case": case,
                "level": level,
                "run": str(run),
                "mesh_sha256": mesh_hash,
                "tetrahedra": reference.mesh.nelements,
                "raw_global_peak_Pa": float(vm.max()),
                "raw_global_peak_position_m": position.tolist(),
                "energy_J": energy,
                "mass_kg": float(reference.total_mass),
                "com_m": reference.com.tolist(),
                "original_probe_count": len(points),
                "outside_probe_indices": np.flatnonzero(~probe["valid"]).tolist(),
                "full_nodal_load_reintegration_maximum_error_N": float(abs(force - original_force).max()),
                "patch_count": len(patches),
                **checked,
            }
            rows.append(row)
            if level:
                old = previous[case]
                common = probe["valid"] & old["probe"]["valid"]
                assert common.any()
                change = {
                    "geometry": geometry,
                    "case": case,
                    "from_level": level - 1,
                    "to_level": level,
                    "from_tetrahedra": counts[-2],
                    "to_tetrahedra": counts[-1],
                    "original_probe_count": len(points),
                    "common_valid_probes": int(common.sum()),
                    "excluded_pair_probe_indices": np.flatnonzero(~common).tolist(),
                    "raw_peak_relative_change": abs(vm.max() - old["row"]["raw_global_peak_Pa"]) / float(vm.max()),
                    "raw_peak_absolute_change_Pa": abs(float(vm.max()) - old["row"]["raw_global_peak_Pa"]),
                    "raw_peak_location_shift_m": float(
                        np.linalg.norm(position - old["row"]["raw_global_peak_position_m"])
                    ),
                    "energy_relative_change": abs(energy - old["row"]["energy_J"]) / abs(energy),
                    "energy_absolute_change_J": abs(energy - old["row"]["energy_J"]),
                }
                for name, label in [("tensor_Pa", "tensor"), ("vm_Pa", "vm"), ("displacement_m", "displacement")]:
                    relative, absolute = relative_change(old["probe"][name][common], probe[name][common])
                    change[f"probe_{label}_relative_change"] = relative
                    change[f"probe_{label}_absolute_change"] = absolute
                changes.append(change)
            previous[case] = {"probe": probe, "row": row}
            np.savez_compressed(
                args.output / f"level{level}-{case}.npz",
                complete_force_reintegration_difference_N=force - original_force,
                complete_equilibrium_residual_N=reference.stiffness @ u - rhs,
                **probe,
            )
            print(json.dumps(row), flush=True)
    assert len(rows) == 15 and len(changes) == 10
    tails = []
    for case in CASES:
        selected = [row for row in changes if row["case"] == case]
        keys = [
            "probe_tensor_relative_change",
            "probe_vm_relative_change",
            "probe_displacement_relative_change",
            "raw_peak_relative_change",
            "energy_relative_change",
        ]
        tails.append(
            {
                "case": case,
                "both_latest_pairs_below_initial_diagnostic": all(
                    row[key] is not None and row[key] < args.diagnostic_target for row in selected for key in keys
                ),
                "latest_pair_changes": {key: selected[-1][key] for key in keys},
                "physical_convergence_accepted": False,
            }
        )
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected for name, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "geometry": geometry,
        "tetrahedra": counts,
        "rows": rows,
        "adjacent_changes": changes,
        "initial_target": args.diagnostic_target,
        "tails": tails,
        "all_original_and_new_fields_freshly_audited": True,
        "new_equilibrium_states_solved": 0,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "Fresh stiffness/mass/RHS, independent current boundary loads, every corner/quadrature tensor, all original probe coordinates/validity, energies and raw whole-volume extrema cover every one of the15 fields. Three levels extend the original last mesh by two independently generated CAD/volume levels; these axes remain coupled. Both latest diagnostics are reported without replacing earlier six-level curves, omitted near-interface regions or unresolved geometry/volume/integration/native model/feedback/contact/application/cost requirements. Passing adjacent diagnostics alone is not continuum certification.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
