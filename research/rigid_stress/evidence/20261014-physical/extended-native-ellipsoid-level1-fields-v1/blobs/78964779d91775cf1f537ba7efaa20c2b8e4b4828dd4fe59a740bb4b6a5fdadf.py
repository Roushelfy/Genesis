"""Compare all five audited dynamic fields while varying only contact distance on one fixed CAD mesh."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_articulated_contact import quadrature_fields
from .audit_objective_frame import finite_frame
from .audit_original_history_fields import wide_volume_bounds
from .qualify_articulated_precision import tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static


def tensor_change(first, second, weights):
    numerator = float(np.sum(weights * np.sum((second - first) ** 2, axis=(-2, -1))))
    denominator = float(np.sum(weights * np.sum(second**2, axis=(-2, -1))))
    return {
        "absolute_volume_L2_change_Pa_m_3over2": float(np.sqrt(numerator)),
        "relative_volume_L2_change": float(np.sqrt(numerator / denominator)) if denominator else None,
    }


def scalar_change(first, second):
    return {"absolute_change": abs(second - first), "relative_change": abs(first / second - 1) if second else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.run) == len(args.audit) >= 3
    inputs = {}

    def retain(path):
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        assert str(path) not in inputs or inputs[str(path)] == value
        inputs[str(path)] = value

    static, first_config = restore_static(args.run[0])
    assert first_config["mode"] == "cad" and first_config["cold_closure"]
    assert not first_config["freeze_contact"] and first_config["steps"] == 5
    body = static.body
    bulk = WidePiolaBulk(body)
    permutation = static.parts[0].permutation
    args.output.mkdir(parents=True, exist_ok=False)
    fixed_config = {key: value for key, value in first_config.items() if key not in ["closure_run", "output"]}
    closure_physical, distances, rows, comparisons, previous = None, [], [], [], {}
    for index, (run, audit_run) in enumerate(zip(args.run, args.audit, strict=True)):
        for closed in [run, audit_run]:
            closed_source(closed)
            assert int((closed / "returncode.txt").read_text()) == 0
        paths = [run / "results/configuration.json", audit_run / "results/summary.json"]
        config, audit = [json.loads(path.read_text()) for path in paths]
        for path in paths:
            retain(path)
        assert {key: value for key, value in config.items() if key not in ["closure_run", "output"]} == fixed_config
        assert Path(audit["run"]).resolve() == run.resolve()
        assert audit["states_checked"] == len(audit["rows"]) == config["steps"] == 5
        assert [row["step"] for row in audit["rows"]] == list(range(1, 6))
        closure_path = Path(config["closure_run"]) / "results/configuration.json"
        retain(closure_path)
        closure = json.loads(closure_path.read_text())
        # Extra configuration-input labels are metadata, never solved states.
        physical = {key: closure[key] for key in ["capture", "control_steps", "gravity_steps"]}
        if closure_physical is None:
            closure_physical = physical
        assert physical == closure_physical
        assert closure["dhat"] > 0 and (not distances or closure["dhat"] < distances[-1])
        distances.append(closure["dhat"])
        # Check every stored geometry/map array, including actual source groups,
        # initialization coordinates and map values. No remeshing is permitted.
        for filename in ["reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz"]:
            path = run / "results" / filename
            retain(path)
            with np.load(path) as actual, np.load(args.run[0] / "results" / filename) as expected:
                assert set(actual.files) == set(expected.files)
                for key in actual.files:
                    np.testing.assert_array_equal(actual[key], expected[key])
        for qualified in audit["rows"]:
            step = qualified["step"]
            assert qualified["body_tetrahedra"] == body.mesh.nelements
            assert (
                max(qualified["complete_body_dynamic_residual_N"], qualified["complete_mapped_dynamic_residual_N"])
                <= 1e-9
            )
            path = run / f"results/step{step:04d}.npz"
            retain(path)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["body_component_permutation"], permutation)
                u, force = saved["displacement"].copy(), saved["contact_force_N"].copy()
            vector = np.empty(body.basis.N, dtype=np.longdouble)
            vector[permutation] = u[: body.basis.N]
            rotation, translation, objective = finite_frame(body, np.asarray(vector, dtype=float))
            deformation = bulk.corner_deformation_gradients(vector)
            corner = tensor_from_f(deformation, body.lame)
            deviator = corner - np.trace(corner, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
            vm = np.sqrt(1.5 * np.sum(deviator**2, axis=(-2, -1)))
            lower, upper = wide_volume_bounds(deformation, bulk.mu)
            peak = float(vm.max())
            np.testing.assert_allclose(peak, qualified["raw_global_peak_lower_Pa"], rtol=1e-12, atol=1e-5)
            np.testing.assert_allclose(upper, qualified["raw_global_peak_upper_Pa"], rtol=1e-12, atol=1e-5)
            assert lower > 0
            quadrature, energy = quadrature_fields(bulk, np.asarray(vector, dtype=float))
            objective_quad = rotation.T @ quadrature @ rotation
            objective_corner = rotation.T @ corner @ rotation
            assert np.isfinite(objective_quad).all() and energy >= 0
            row = {
                "run": str(run),
                "step": step,
                "time_s": qualified["time_s"],
                "tetrahedra": body.mesh.nelements,
                "barrier_distance_m": closure["dhat"],
                "raw_peak_lower_Pa": peak,
                "raw_peak_upper_Pa": upper,
                "raw_peak_reference_position_m": qualified["raw_peak_reference_position_m"],
                "whole_volume_J_lower_bound": lower,
                "nonlinear_energy_J": energy,
                "original_body_residual_N": qualified["complete_body_dynamic_residual_N"],
                "original_mapped_residual_N": qualified["complete_mapped_dynamic_residual_N"],
            }
            if step in previous:
                old = previous[step]
                delta = objective - old["objective"]
                numerator = max(0.0, float(delta @ (body.mass @ delta)))
                denominator = max(0.0, float(objective @ (body.mass @ objective)))
                corner_change = objective_corner - old["corner"]
                comparisons.append(
                    {
                        "step": step,
                        "time_s": qualified["time_s"],
                        "coarse_distance_m": old["row"]["barrier_distance_m"],
                        "fine_distance_m": closure["dhat"],
                        "complete_quadrature_tensor": tensor_change(old["quadrature"], objective_quad, bulk.basis.dx),
                        "complete_corner_tensor_absolute_L2_change_Pa": float(np.linalg.norm(corner_change)),
                        "complete_corner_tensor_relative_L2_change": float(
                            np.linalg.norm(corner_change) / np.linalg.norm(objective_corner)
                        ),
                        "raw_global_peak_lower_Pa": scalar_change(old["row"]["raw_peak_lower_Pa"], peak),
                        "nonlinear_energy_J": scalar_change(old["row"]["nonlinear_energy_J"], energy),
                        "consistent_mass_objective_displacement_absolute_L2_change_m_sqrt_kg": float(
                            np.sqrt(numerator)
                        ),
                        "consistent_mass_objective_displacement_relative_L2_change": float(
                            np.sqrt(numerator / denominator)
                        )
                        if denominator
                        else None,
                        "complete_nodal_contact_force_absolute_L2_change_N": float(
                            np.linalg.norm(force - old["force"])
                        ),
                        "complete_nodal_contact_force_relative_L2_change": float(
                            np.linalg.norm(force - old["force"]) / np.linalg.norm(force)
                        )
                        if np.linalg.norm(force)
                        else None,
                    }
                )
            previous[step] = {
                "row": row,
                "corner": objective_corner,
                "quadrature": objective_quad,
                "objective": objective,
                "force": force,
            }
            rows.append(row)
            np.savez_compressed(
                args.output / f"case{index:02d}-step{step:04d}.npz",
                raw_corner_tensor_Pa=corner,
                raw_corner_vm_Pa=vm,
                objective_corner_tensor_Pa=objective_corner,
                objective_quadrature_tensor_Pa=objective_quad,
                quadrature_weights_m3=bulk.basis.dx,
                proper_rotation=rotation,
                translation_m=translation,
                objective_displacement_m=objective,
                full_nodal_contact_force_N=force,
            )
            print(json.dumps(row), flush=True)
    assert len(rows) == 5 * len(args.run) and len(comparisons) == 5 * (len(args.run) - 1)
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected for path, expected in inputs.items())
    summary = {
        "scope": __doc__,
        "rows": rows,
        "comparisons": comparisons,
        "body_mass_kg": float(body.total_mass),
        "fixed_configuration": fixed_config,
        "fixed_original_capture_and_control_counts": closure_physical,
        "all_geometry_and_contact_map_arrays_exactly_equal": True,
        "every_original_five_state_force_audit_and_raw_bounds_matched": True,
        "input_sha256": inputs,
        "goal_completion": False,
        "limitations": "This is a numerical contact-distance sensitivity study on identical body/finger geometry, material, mass, contact trace, source mapping, original64 cold closure controls and five dynamic steps. Complete material-frame quadrature/corner tensors and raw whole-volume bounds remain explicit at every physical time. Initializer full fields are not saved at all64 controls and are not claimed independently audited. Three distances at one fixed mesh do not establish a continuum hard-contact limit, spatial/pressure convergence, load-model accuracy, full application feedback or warmed cost. Source-wrench/traction diagnostics are separate qualified outputs.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
