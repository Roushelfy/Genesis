"""Compare a candidate's bounded fields and complete-grasp traces with frozen inputs.

Run under Slurm. Field/stress bounds retain the existing source-prior checks;
the recorded model/radius bands remain reference uncertainty, not tolerances.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def differences(first, second):
    assert first.shape == second.shape
    assert np.isfinite(first).all() and np.isfinite(second).all()
    delta = first - second
    return {
        "maximum_absolute_difference": float(np.abs(delta).max()),
        "relative_L2_difference": float(np.linalg.norm(delta) / max(np.linalg.norm(second), np.finfo(float).tiny)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-fields", type=Path, required=True)
    parser.add_argument("--candidate-reward-low02", type=Path, required=True)
    parser.add_argument("--candidate-reward-low1", type=Path, required=True)
    parser.add_argument("--phase-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    inputs = {}
    runs = [args.candidate_fields, args.candidate_reward_low02, args.candidate_reward_low1]
    for run in runs:
        for name in ("returncode.txt", "program-returncode.txt"):
            assert (run / name).read_text().strip() == "0", run
        assert (run / "native-snapshot-before.txt").read_bytes() == (run / "native-snapshot-after.txt").read_bytes()
        assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
    baseline_fields = args.baseline_root / "practical-franka-four-snapshot-reconstruction-v1"
    for run in (args.candidate_fields, baseline_fields):
        path = run / "results/summary.json"
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    candidate_summary = json.loads((args.candidate_fields / "results/summary.json").read_text())
    baseline_summary = json.loads((baseline_fields / "results/summary.json").read_text())
    assert candidate_summary["input_sha256"] == baseline_summary["input_sha256"]
    field_checks = []
    for row in candidate_summary["rows"]:
        assert row["model"] == "body_source_prior" and row["radius_m"] == 0.006
        assert row["errno"] == 0 and row["native_complete_budget_fraction"] <= 1
        filename = f"step{row['step']:05d}-r6000-body_source_prior.npz"
        paths = [run / "results" / filename for run in (args.candidate_fields, baseline_fields)]
        for path in paths:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        fields = {}
        with np.load(paths[0]) as candidate, np.load(paths[1]) as baseline:
            candidate_keys = np.stack([candidate[key] for key in ("packet", "physical_source", "faces")], axis=1)
            baseline_keys = np.stack([baseline[key] for key in ("packet", "physical_source", "faces")], axis=1)
            np.testing.assert_array_equal(np.unique(candidate_keys, axis=0), np.unique(baseline_keys, axis=0))
            permutation = np.empty(len(candidate_keys), dtype=np.intp)
            for key in np.unique(candidate_keys, axis=0):
                first = np.flatnonzero(np.all(candidate_keys == key, axis=1))
                second = np.flatnonzero(np.all(baseline_keys == key, axis=1))
                assert len(first) == len(second)
                delta = (
                    candidate["traction_positions_m"][first, None, :]
                    - baseline["traction_positions_m"][None, second, :]
                )
                nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
                assert len(np.unique(nearest)) == len(first), "Traction sample correspondence must be bijective."
                permutation[first] = second[nearest]
            # This saved RHS includes reconstructed contact loads; the frozen
            # inputs are identified by input_sha256 above, not this output.
            # Use the existing source-prior nodal-load comparison bounds.
            np.testing.assert_allclose(
                candidate["full_inertia_rhs_N"], baseline["full_inertia_rhs_N"], rtol=1e-4, atol=1e-10
            )
            for key in (
                "native_tensor_Pa",
                "displacement_m",
                "mapped_contact_force_N",
                "full_inertia_rhs_N",
                "corner_tensor_Pa",
                "corner_vm_Pa",
                "quadrature_tensor_Pa",
                "traction_positions_m",
                "traction_normals",
                "traction_forces_N",
                "traction_areas_m2",
            ):
                expected = baseline[key][permutation] if key.startswith("traction_") else baseline[key]
                fields[key] = differences(candidate[key], expected)
                if key.endswith("Pa"):
                    np.testing.assert_allclose(candidate[key], expected, rtol=1e-4, atol=1e-3)
            assert fields["traction_positions_m"]["maximum_absolute_difference"] <= 1e-15
            np.testing.assert_allclose(
                candidate["mapped_contact_force_N"], baseline["mapped_contact_force_N"], rtol=1e-4, atol=1e-10
            )
        field_checks.append(
            {
                "step": row["step"],
                "complete_budget_fraction": row["native_complete_budget_fraction"],
                "fields": fields,
                "sample_matching": {
                    "samples": len(permutation),
                    "bijection_by_packet_source_face_position": True,
                    "rows_reordered": not np.array_equal(permutation, np.arange(len(permutation))),
                },
            }
        )
    assert len(field_checks) == 4
    audit = json.loads(args.phase_audit.read_text())
    inputs[str(args.phase_audit)] = hashlib.sha256(args.phase_audit.read_bytes()).hexdigest()
    reward_checks = []
    reference_ensembles = []
    for candidate_run, family, low_limit in (
        (args.candidate_reward_low02, "low-force-choice", 0.2),
        (args.candidate_reward_low1, "grip-choice", 1),
    ):
        baseline_run = args.baseline_root / f"practical-franka-{family}-body_source_prior-v1"
        for run in (candidate_run, baseline_run):
            path = run / "results/summary.json"
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            summary = json.loads(path.read_text())
            if "requested_strong_limits_N" in summary:
                assert summary["requested_strong_limits_N"] == [low_limit, 4] and summary["radius_scale"] == 1
            else:
                # The original 1N driver predates configurable force/radius.
                # Verify its exact fixed-1N, nominal-radius source, not a default.
                assert run == baseline_run and low_limit == 1
                driver = run / "source/research/rigid_stress/practical_reward_proxy.py"
                driver_hash = hashlib.sha256(driver.read_bytes()).hexdigest()
                assert driver_hash == "f9485b32cb79682f4a8629806836ccc336b9338156b1eac58b81cd50c78627ea"
                inputs[str(driver)] = driver_hash
                assert all(row["strong_limit_N"] == [1, 4] for row in summary["paired_choices"])
            assert summary["model"] == "body_source_prior"
            assert summary["warmup"] == 900 and summary["steps_per_repetition"] == 1200 and summary["repetitions"] == 3
            assert summary["invalid_stress_steps"] == 0 and summary["reset_and_unselected_peak_checks_passed"]
            assert summary["lift_window_phase"] == [0.3, 0.6]
        paths = [run / "results/raw_trace.npz" for run in (candidate_run, baseline_run)]
        for path in paths:
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0]) as candidate, np.load(paths[1]) as baseline:
            for key in ("phase", "strong_limit_N", "condition_index"):
                np.testing.assert_array_equal(candidate[key], baseline[key])
            peak = candidate["maximum_stress_Pa"]
            assert peak.shape == (3, 1200, 16)
            np.testing.assert_allclose(peak, baseline["maximum_stress_Pa"], rtol=1e-4, atol=1e-3)
            windows = {"whole_episode": peak.max(axis=1), "lift_window": candidate["lift_window_raw_peak_Pa"]}
            reference = next(row for row in audit["families"] if row["low_limit_N"] == low_limit)
            rows = []
            for condition, original in enumerate(reference["rows"]):
                result = {"condition": condition}
                for scope, values in windows.items():
                    difference = values[:, condition + 8] - values[:, condition]
                    stable = bool(np.all(difference > 0) or np.all(difference < 0))
                    band_key = (
                        "whole_episode_absolute_model_radius_band_Pa"
                        if scope == "whole_episode"
                        else "absolute_peak_model_radius_band_Pa"
                    )
                    if scope == "whole_episode":
                        band = original[band_key]
                    else:
                        analysis = "low-force-radius" if low_limit == 0.2 else "reward-radius"
                        path = args.baseline_root / f"practical-franka-{analysis}-analysis-v1/results/summary.json"
                        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                        band = json.loads(path.read_text())["rows"][condition][band_key]
                    result[scope] = {
                        "raw_peak_Pa": values[:, [condition, condition + 8]].tolist(),
                        "reference_model_radius_band_Pa": band,
                        "exceeds_reference_band": stable and float(np.abs(difference).min()) > band,
                        "lower_peak_choice": (f"{low_limit:g}N" if difference[0] > 0 else "4N")
                        if stable
                        else "unresolved",
                    }
                rows.append(result)
            reward_checks.append(
                {
                    "low_limit_N": low_limit,
                    "observed_environment_steps": int(peak.size),
                    "raw_stress_difference": differences(peak, baseline["maximum_stress_Pa"]),
                    "height_difference_m": differences(candidate["height_m"], baseline["height_m"]),
                    "rows": rows,
                }
            )
        scope_values = {"whole_episode": [], "lift_window": []}
        for model in ("pressure", "body_source_prior"):
            for half_radius in (False, True):
                suffix = "-half-radius" if half_radius else ""
                run = args.baseline_root / f"practical-franka-{family}-{model}{suffix}-v1"
                if model == "body_source_prior" and not half_radius:
                    run = candidate_run
                path = run / "results/raw_trace.npz"
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                with np.load(path) as trace:
                    scope_values["whole_episode"].extend(trace["maximum_stress_Pa"].max(axis=1))
                    scope_values["lift_window"].extend(trace["lift_window_raw_peak_Pa"])
        ensemble_rows = []
        for condition in range(8):
            result = {"condition": condition}
            for scope, values in scope_values.items():
                values = np.stack(values)
                low, high = values[:, condition], values[:, condition + 8]
                difference = high - low
                stable = bool(np.all(difference > 0) or np.all(difference < 0))
                band = float(max(np.ptp(low), np.ptp(high)))
                gap = float(np.abs(difference).min())
                result[scope] = {
                    "rank_stable": stable,
                    "minimum_absolute_separation_Pa": gap,
                    "observed_model_radius_band_Pa": band,
                    "exceeds_observed_band": stable and gap > band,
                    "lower_peak_choice": (f"{low_limit:g}N" if difference[0] > 0 else "4N") if stable else "unresolved",
                }
            ensemble_rows.append(result)
        reference_ensembles.append(
            {
                "low_limit_N": low_limit,
                "rows": ensemble_rows,
                "scope": "Retained four-model/radius reference ensemble with only nominal source-prior traces replaced by the candidate. This is not qualification of all four alternatives on candidate code.",
            }
        )
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "field_checks": field_checks,
        "reward_checks": reward_checks,
        "reference_ensembles_with_nominal_candidate": reference_ensembles,
        "input_sha256": inputs,
        "stress_comparison_bounds": {
            "rtol": 1e-4,
            "atol_Pa": 1e-3,
            "source": "Existing test_stress_source_priors full-field/peak checks.",
        },
        "limitations": "The candidate repeats the nominal-radius source-prior cases. Bands are preserved observations from the prior model/radius campaign, not newly calibrated uncertainty or fracture thresholds. Whole-episode and lift decisions remain separate; this is not throughput evidence or full RL training.",
        "passed": True,
        "goal_completion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fields": len(field_checks), "reward_families": len(reward_checks), "passed": True}))


if __name__ == "__main__":
    main()
