"""Qualify and continue a terminated spatial reference from its retained physical initializer."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from .audit_original_history_fields import wide_volume_bounds
from .audit_precise_spatial_state import assemble_state
from .franka_rigid_mode_coordinates import coordinate_map
from .interrupted_source_verification import verify_interrupted_source
from .precise_stencil_spatial import PreciseStencilSpatial
from .qualify_articulated_precision import peak, tensor_from_f
from .summarize_fine_references import closed_source
from .wide_coordinate_franka import WidePiolaBulk
from .wide_spatial_franka import restore_static

GEOMETRY_FILES = ("closure_state.npz", "reference_geometry.npz", "contact_geometry.npz", "contact_node_map.npz")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_state_arrays(original, candidate):
    assert original.files == candidate.files
    changes = {}
    for key in original.files:
        assert original[key].shape == candidate[key].shape
        delta = np.array(candidate[key], dtype=np.longdouble) - original[key]
        error, scale = np.linalg.norm(delta), np.linalg.norm(original[key])
        assert error <= 1e-5 * scale + 1e-12, (key, error, scale)
        changes[key] = {
            "maximum_absolute_difference": float(abs(delta).max(initial=0)),
            "relative_L2_difference": float(error / max(scale, 1e-12)),
        }
    for key in ["body_component_permutation", "applied_force_N"]:
        np.testing.assert_array_equal(original[key], candidate[key])
    return changes


def solve(args):
    observation = verify_interrupted_source(args.source_run, args.source_observation)
    static, config = restore_static(args.source_run, args.source_observation)
    original_steps = config["steps"]
    original_prefix = sorted((args.source_run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert original_prefix and 0 < args.steps <= original_steps
    assert [path.name for path in original_prefix] == [
        f"step{step:04d}.npz" for step in range(1, len(original_prefix) + 1)
    ]
    if args.qualification is None:
        assert args.steps == len(original_prefix), "First resolve and qualify every retained source state."
    else:
        closed_source(args.qualification)
        assert int((args.qualification / "returncode.txt").read_text()) == 0
        qualification = json.loads((args.qualification / "results/summary.json").read_text())
        assert Path(qualification["source_run"]).resolve() == args.source_run.resolve()
        assert qualification["source_observation_sha256"] == sha256(args.source_observation)
        assert qualification["all_original_saved_states_numerically_qualified"]
        assert qualification["states_checked"] == len(original_prefix)
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = []
    for name in GEOMETRY_FILES:
        source, destination = args.source_run / "results" / name, args.output / name
        shutil.copy2(source, destination)
        assert sha256(source) == sha256(destination)
        inputs.append({"source": str(source), "sha256": sha256(source)})
    source_configuration_sha256 = sha256(args.source_run / "results/configuration.json")
    config.update({"source_requested_steps": original_steps, "steps": args.steps, "output": str(args.output)})
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    provenance = {
        "source_run": str(args.source_run),
        "source_scheduler_state": observation["scheduler_state"],
        "source_observation": str(args.source_observation),
        "source_observation_sha256": sha256(args.source_observation),
        "source_configuration_sha256": source_configuration_sha256,
        "initial_state_and_meshes": inputs,
        "original_saved_steps": len(original_prefix),
        "original_saved_state_sha256": {path.name: sha256(path) for path in original_prefix},
        "qualification": str(args.qualification) if args.qualification else None,
        "copied_qualified_prefix": [],
    }
    system = PreciseStencilSpatial(static, config["dt"])
    system.q_target[:] = 0.018
    with np.load(args.output / "closure_state.npz") as saved:
        u = np.array(saved["displacement"], dtype=np.longdouble)
    velocity = np.zeros_like(u)
    rows, resume_step = [], 0
    if args.qualification is not None:
        qualified_run = Path(qualification["run"])
        closed_source(qualified_run)
        assert int((qualified_run / "returncode.txt").read_text()) == 0
        qualified_provenance = json.loads((qualified_run / "results/resume_provenance.json").read_text())
        assert qualified_provenance["initial_state_and_meshes"] == inputs
        assert qualified_provenance["source_configuration_sha256"] == source_configuration_sha256
        assert qualified_provenance["original_saved_state_sha256"] == provenance["original_saved_state_sha256"]
        report = json.loads((qualified_run / "results/summary.json").read_text())
        rows, resume_step = report["steps"], qualification["states_checked"]
        assert [row["step"] for row in rows] == list(range(1, resume_step + 1))
        assert resume_step < args.steps
        for step in range(1, resume_step + 1):
            source, destination = qualified_run / f"results/step{step:04d}.npz", args.output / f"step{step:04d}.npz"
            shutil.copy2(source, destination)
            assert sha256(source) == sha256(destination)
            provenance["copied_qualified_prefix"].append({"source": str(source), "sha256": sha256(source)})
        with np.load(args.output / f"step{resume_step:04d}.npz") as saved:
            u, velocity = saved["displacement"].copy(), saved["velocity"].copy()
    (args.output / "resume_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    mapping, external = coordinate_map(system), system.gravity_force([0, 0, -9.81])
    for step in range(resume_step + 1, args.steps + 1):
        previous, old_velocity = u.copy(), velocity.copy()
        system.predicted_displacement = previous + config["dt"] * old_velocity
        u, gradient, force, _energy, _iterations = system.solve_updated_friction(
            np.array([], dtype=int),
            np.array([]),
            1,
            tolerance=1e-10,
            initial_displacement=previous,
            coordinate_map=mapping,
            applied_force=external,
            initial_force=external,
            phase=f"retained_initializer_step{step}",
            failure_path=args.output / "failure.npz",
        )
        velocity = (u - previous) / config["dt"]
        acceleration = (velocity - old_velocity) / config["dt"]
        complete, body = (
            float(np.linalg.norm(mapping.T @ gradient)),
            float(np.linalg.norm(gradient[: system.body.basis.N])),
        )
        assert complete <= 1e-9 and body <= 1e-9
        row = {
            "step": step,
            "time_s": step * config["dt"],
            "complete_dynamic_residual_N": complete,
            "complete_body_dynamic_residual_N": body,
        }
        rows.append(row)
        np.savez_compressed(
            args.output / f"step{step:04d}.npz",
            displacement=u,
            velocity=velocity,
            acceleration=acceleration,
            contact_force_N=force,
            applied_force_N=external,
            body_component_permutation=system.parts[0].permutation,
        )
        print(json.dumps(row), flush=True)
    result = {
        "scope": __doc__,
        "source_run": str(args.source_run),
        "body_tetrahedra": system.body.mesh.nelements,
        "steps": rows,
        "requested_steps": args.steps,
        "source_requested_steps": original_steps,
        "complete_source_requested_history": len(rows) == original_steps,
        "goal_completion": False,
        "limitations": "The saved original held initializer is retained byte-for-byte, with zero initial velocity. "
        "The complete saved prefix is first resolved and independently qualified before further steps. "
        "No producer EXIT record is invented for the interrupted source. Centered-stencil arithmetic "
        "targets1e-10N and retains original1e-9N physical gates. This fixed-arm/two-jaw control does "
        "not establish full-arm or physical contact/geometry/model/feedback/cost acceptance.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


def audit(args):
    closed_source(args.run)
    assert int((args.run / "returncode.txt").read_text()) == 0
    provenance = json.loads((args.run / "results/resume_provenance.json").read_text())
    source, observation = Path(provenance["source_run"]), Path(provenance["source_observation"])
    verify_interrupted_source(source, observation)
    assert sha256(observation) == provenance["source_observation_sha256"]
    assert sha256(source / "results/configuration.json") == provenance["source_configuration_sha256"]
    for name, expected in provenance["original_saved_state_sha256"].items():
        assert sha256(source / "results" / name) == expected
    static, original_config = restore_static(source, observation)
    config = json.loads((args.run / "results/configuration.json").read_text())
    assert {key: value for key, value in original_config.items() if key not in {"steps", "output"}} == {
        key: value for key, value in config.items() if key not in {"steps", "output", "source_requested_steps"}
    }
    for item in provenance["initial_state_and_meshes"]:
        path = Path(item["source"])
        assert sha256(path) == item["sha256"] == sha256(args.run / "results" / path.name)
    args.output.mkdir(parents=True, exist_ok=False)
    bulk, n = WidePiolaBulk(static.body), static.body.basis.N
    with np.load(args.run / "results/closure_state.npz") as saved:
        previous = np.array(saved["displacement"], dtype=np.longdouble)
    old_velocity = np.zeros_like(previous)
    rows, comparisons = [], []
    files = sorted((args.run / "results").glob("step[0-9][0-9][0-9][0-9].npz"))
    assert [path.name for path in files] == [f"step{step:04d}.npz" for step in range(1, config["steps"] + 1)]
    for step, path in enumerate(files, 1):
        with np.load(path) as saved:
            u, velocity, acceleration, expected = [
                saved[key].copy() for key in ["displacement", "velocity", "acceleration", "contact_force_N"]
            ]
            np.testing.assert_array_equal(saved["applied_force_N"], static.gravity_force([0, 0, -9.81]))
            np.testing.assert_array_equal(saved["body_component_permutation"], static.parts[0].permutation)
            np.testing.assert_array_equal(velocity, (u - previous) / config["dt"])
            np.testing.assert_array_equal(acceleration, (velocity - old_velocity) / config["dt"])
            if step <= provenance["original_saved_steps"]:
                with np.load(source / f"results/step{step:04d}.npz") as original:
                    changes = compare_state_arrays(original, saved)
                    original_u = original["displacement"].copy()
        metrics, fields = assemble_state(static, config, u, previous, old_velocity, expected, bulk)
        vector = np.empty(n, dtype=np.longdouble)
        vector[static.parts[0].permutation] = u[:n]
        f = bulk.corner_deformation_gradients(vector)
        tensor = tensor_from_f(f, static.body.lame)
        j_lower, upper = wide_volume_bounds(f, bulk.mu)
        raw_peak = peak(tensor)
        assert raw_peak <= upper and j_lower > 0
        if step <= provenance["original_saved_steps"]:
            original_vector = np.empty(n, dtype=np.longdouble)
            original_vector[static.parts[0].permutation] = original_u[:n]
            original_tensor = tensor_from_f(bulk.corner_deformation_gradients(original_vector), static.body.lame)
            quads = [
                tensor_from_f(bulk.deformation_gradients(v).transpose(2, 3, 0, 1), static.body.lame)
                for v in [original_vector, vector]
            ]
            corner_change = float(np.linalg.norm(tensor - original_tensor) / np.linalg.norm(original_tensor))
            quad_change = float(np.linalg.norm(quads[1] - quads[0]) / np.linalg.norm(quads[0]))
            peak_change = abs(raw_peak / peak(original_tensor) - 1)
            assert max(corner_change, quad_change, peak_change) <= 1e-6
            comparisons.append(
                {
                    "step": step,
                    "all_saved_arrays": changes,
                    "full_corner_tensor_relative_change": corner_change,
                    "full_quadrature_tensor_relative_change": quad_change,
                    "raw_peak_relative_change": peak_change,
                }
            )
        row = {
            "step": step,
            **metrics,
            "raw_global_peak_lower_Pa": raw_peak,
            "raw_global_peak_upper_Pa": upper,
            "whole_volume_J_lower_bound": j_lower,
        }
        rows.append(row)
        np.savez_compressed(args.output / path.name, **fields)
        previous, old_velocity = u, velocity
    assert len(comparisons) == provenance["original_saved_steps"]
    result = {
        "scope": "Independent balances, unsmoothed global fields and complete interrupted-prefix equivalence",
        "run": str(args.run),
        "source_run": str(source),
        "states_checked": len(rows),
        "source_observation_sha256": sha256(observation),
        "rows": rows,
        "all_original_saved_states_numerically_qualified": True,
        "original_saved_state_comparisons": comparisons,
        "declared_numerical_comparison_budgets": {
            "state_relative": 1e-5,
            "state_absolute": 1e-12,
            "full_corner_quadrature_and_peak_relative": 1e-6,
        },
        "complete_source_requested_history": len(rows) == original_config["steps"],
        "goal_completion": False,
        "limitations": "This preserves the retained held initializer and qualifies every original saved array and "
        "full corner/quadrature field within the established numerical-comparison budgets. All new "
        "states pass independently reconstructed unchanged1e-9N body/mapped gates. The terminated "
        "producer remains incomplete. Physical spatial/contact/geometry convergence and full "
        "application/model/feedback/accuracy-cost acceptance remain separate requirements.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"states_checked": len(rows), "original_saved_states_compared": len(comparisons)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["solve", "audit"])
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--source-observation", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "solve":
        assert args.source_run is not None and args.source_observation is not None
        solve(args)
    else:
        assert args.run is not None
        audit(args)


if __name__ == "__main__":
    main()
