"""Check a cone-preserving radial memory model against retained friction cases."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .hertz_history_cone import ConeFrictionHistory
from .hertz_stress import elliptic_stress
from .hertz_tangential import elliptic_tangential_stress
from .study_hertz_nearfield import stress_measures


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--reference", type=Path, required=True)
    cli.add_argument("--baseline-result", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "force_residual": 1e-9,
        "slope_bound_roundoff": 1e-10,
        "tensor_relative": 0.05,
        "sample_peak_relative": 0.05,
        "frame_line_relative": 1e-6,
    }
    baseline = json.loads(args.baseline_result.read_text())
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "reference": file_id(args.reference),
            "baseline": file_id(args.baseline_result),
            "targets": targets,
            "scope": "Small offline cone-constrained line-energy projection. Constant-normal collinear cases retain their independent Cattaneo-Mindlin reference. Changing-load/vector controls test force and a sufficient continuous Coulomb bound, not independent friction physics or native cost. Original line-clipping failures remain untouched.",
        },
    )
    with np.load(args.reference) as data:
        points = data["history_points"]
        reference = {key: data[key] for key in data.files if key.endswith("_reference") and "ratio" not in key}
    normal = elliptic_stress(points, np.ones(2), order=96)
    basis = np.stack(
        [
            elliptic_tangential_stress(points, np.full(2, radius), np.array([1.0, 0.0]), peak=radius, order=96)
            for radius in np.linspace(0, 1, 33)[1:]
        ]
    )
    phases = [
        ("loading", 0.15),
        ("loading", 0.6),
        ("unloading", 0.3),
        ("unloading", 0.0),
        ("unloading", -0.3),
        ("unloading", -0.6),
        ("reloading", 0.0),
        ("reloading", 0.6),
    ]
    mu, load = 0.4, 2 * np.pi / 3
    radial = np.linspace(0, 1, 2001)[:-1]
    pressure = np.sqrt(1 - radial**2)
    rows, arrays = [], {"sample_points": points, "radial_queries": radial}
    for count in [16, 32]:
        model = ConeFrictionHistory(count)
        for phase, fraction in phases:
            started = perf_counter()
            model.update(1.0, load, mu, np.array([mu * load * fraction, 0.0]))
            seconds = perf_counter() - started
            assert np.max(np.abs(model.coefficients[:, 1])) < 1e-9
            actual = normal + mu * np.einsum(
                "k,kpij->pij", model.coefficients[:, 0], basis[32 // count - 1 :: 32 // count]
            )
            expected = reference[f"{phase}{fraction}_reference"]
            tension, vm = stress_measures(actual)
            exact_tension, exact_vm = stress_measures(expected)
            error = float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
            tensile_error = float(abs(tension.max() / exact_tension.max() - 1))
            vm_error = float(abs(vm.max() / exact_vm.max() - 1))
            traction = model.traction(radial)
            ratio = float(np.max(np.linalg.norm(traction, axis=1) / pressure))
            assert ratio <= 1 + targets["slope_bound_roundoff"], ratio
            row = {
                "intervals": count,
                "phase": phase,
                "force_over_mu_normal": fraction,
                "total_sample_tensor_relative": error,
                "sample_tensile_peak_relative": tensile_error,
                "sample_vm_peak_relative": vm_error,
                "force_residual_over_mu_normal": model.residual,
                "continuous_slope_bound": model.traction_bound,
                "sampled_surface_traction_ratio_max": ratio,
                "solve_seconds": seconds,
                "solver": model.solver,
                "targets_met": bool(
                    error <= targets["tensor_relative"]
                    and max(tensile_error, vm_error) <= targets["sample_peak_relative"]
                ),
            }
            rows.append(row)
            arrays[f"n{count}_{phase}{fraction}_tensor"] = actual
            arrays[f"n{count}_{phase}{fraction}_line"] = model.line.copy()
            arrays[f"n{count}_{phase}{fraction}_coefficients"] = model.coefficients.copy()
        write_json(args.output / "cycle-progress.json", {"rows": rows})
    variations = []
    for changing_direction in [False, True]:
        models = {count: ConeFrictionHistory(count) for count in [16, 32, 64]}
        traction_history = {count: [] for count in models}
        lines = {count: [] for count in models}
        for step, time in enumerate(np.linspace(0, 4 * np.pi, 97)):
            normal_load = load * (1 + 0.35 * np.sin(time))
            radius = (normal_load / load) ** (1 / 3)
            relative_force = np.array(
                [0.55 * np.sin(1.1 * time), 0.45 * np.sin(0.7 * time) if changing_direction else 0.0]
            )
            for count, model in models.items():
                started = perf_counter()
                model.update(radius, normal_load, mu, mu * normal_load * relative_force)
                seconds = perf_counter() - started
                traction = model.traction(radial)
                ratio = float(np.max(np.linalg.norm(traction, axis=1) / pressure))
                assert ratio <= 1 + targets["slope_bound_roundoff"], ratio
                traction_history[count].append(traction)
                lines[count].append(model.line.copy())
                variations.append(
                    {
                        "changing_direction": changing_direction,
                        "step": step,
                        "intervals": count,
                        "normal_load": normal_load,
                        "radius": radius,
                        "relative_force": relative_force.tolist(),
                        "force_residual_over_mu_normal": model.residual,
                        "continuous_slope_bound": model.traction_bound,
                        "sampled_surface_traction_ratio_max": ratio,
                        "solve_seconds": seconds,
                        "solver": model.solver,
                    }
                )
            write_json(
                args.output / "variation-progress.json",
                {"completed_direction": changing_direction, "step": step, "rows": variations},
            )
        for count, history in traction_history.items():
            arrays[f"direction{changing_direction}_n{count}_traction"] = np.stack(history)
            arrays[f"direction{changing_direction}_n{count}_line"] = np.stack(lines[count])
            reference_traction = np.stack(traction_history[64])
            relative = np.linalg.norm(np.stack(history) - reference_traction, axis=(1, 2)) / np.maximum(
                np.linalg.norm(reference_traction, axis=(1, 2)), 1e-12
            )
            for row, error in zip(
                [r for r in variations if r["changing_direction"] == changing_direction and r["intervals"] == count],
                relative,
            ):
                row["traction_relative_to_64_intervals"] = float(error)
    base, rotated = ConeFrictionHistory(16), ConeFrictionHistory(16)
    frame_errors = []
    previous_angle = 0.0
    for step in range(13):
        angle = 0.17 * step
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        force = mu * load * np.array([0.3 * np.sin(step / 3), 0.2 * np.cos(step / 4)])
        base.update(1, load, mu, force)
        rotated.update(1, load, mu, rotation @ force, angle - previous_angle)
        error = float(np.linalg.norm(rotated.line - base.line @ rotation.T) / max(np.linalg.norm(base.line), 1e-12))
        assert error <= targets["frame_line_relative"], error
        frame_errors.append(error)
        previous_angle = angle
    rotated.update(0, 0, mu, np.zeros(2))
    assert np.array_equal(rotated.line, np.zeros_like(rotated.line))
    rotated.update(1, load, mu, mu * load * np.array([0.2, 0.1]))
    fresh = ConeFrictionHistory(16)
    fresh.update(1, load, mu, mu * load * np.array([0.2, 0.1]))
    np.testing.assert_allclose(rotated.line, fresh.line, rtol=1e-9, atol=1e-11)
    arrays_path = args.output / "fields.npz"
    np.savez_compressed(arrays_path, **arrays)
    baseline_max = max(row["sampled_surface_traction_ratio_max"] for row in baseline["variations"])
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "cycle": rows,
            "variations": variations,
            "frame_errors": frame_errors,
            "arrays": file_id(arrays_path),
            "baseline_sampled_cone_violation_max": baseline_max,
            "new_sampled_cone_ratio_max": max(row["sampled_surface_traction_ratio_max"] for row in variations),
            "constant_load_sample_stress_targets_met": bool(all(row["targets_met"] for row in rows)),
            "force_and_continuous_cone_gates_met": True,
            "targets": targets,
            "native_throughput_measured": False,
            "independent_variable_load_vector_physics_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
