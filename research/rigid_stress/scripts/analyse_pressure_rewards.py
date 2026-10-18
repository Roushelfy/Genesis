"""Audit complete raw-stress traces and phase-dependent grip-choice uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    inputs = {}
    records = []
    for model in ("pressure", "body_source_prior"):
        for radius in ("1", "0.5"):
            run = args.root / f"reward-{model}-r{radius}-v1"
            assert (run / "returncode.txt").read_text().strip() == "0"
            assert (run / "program-returncode.txt").read_text().strip() == "0"
            assert (run / "source.sha256").read_bytes() == (run / "source-after.sha256").read_bytes()
            assert (run / "input-source.sha256").read_bytes() == (run / "input-source-after.sha256").read_bytes()
            assert (run / "native-snapshot-before.txt").read_bytes() == (run / "native-snapshot-after.txt").read_bytes()
            summary_path = run / "results/summary.json"
            summary = json.loads(summary_path.read_text())
            raw_path = run / "results/raw_trace.npz"
            inputs[str(summary_path)] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
            inputs[str(raw_path)] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            assert summary["history_size"] == 0 and summary["invalid_stress_steps"] == 0
            assert (
                summary["condition_bank_sha256"] == "1d40197808823971cb0f5d76e51da266aaaae6cb28453467392a3e591e4bacfa"
            )
            assert summary["envs"] == 16 and summary["warmup"] == 900
            assert summary["steps_per_repetition"] == 1200 and summary["repetitions"] == 3
            assert summary["reset_and_unselected_peak_checks_passed"]
            with np.load(raw_path) as raw:
                peak, height, phase = raw["maximum_stress_Pa"], raw["height_m"], raw["phase"]
                assert peak.shape == height.shape == phase.shape == (3, 1200, 16)
                assert np.isfinite(peak).all() and np.isfinite(height).all() and np.isfinite(phase).all()
                np.testing.assert_allclose(raw["strong_limit_N"], [0.2] * 8 + [4] * 8, rtol=1e-7, atol=0)
                lift = (phase >= 0.3) & (phase <= 0.6)
                whole_peak = peak.max(axis=1)
                lift_peak = np.where(lift, peak, 0).max(axis=1)
                np.testing.assert_array_equal(lift_peak, raw["lift_window_raw_peak_Pa"])
                exposure = np.where(lift, peak, 0).sum(axis=1) * 0.01
                np.testing.assert_allclose(exposure, raw["lift_window_stress_exposure_Pa_s"], rtol=1e-12, atol=1e-12)
                records.append(
                    {
                        "model": model,
                        "radius_scale": float(radius),
                        "whole_episode_peak_Pa": whole_peak.tolist(),
                        "lift_window_peak_Pa": lift_peak.tolist(),
                        "lift_window_exposure_Pa_s": exposure.tolist(),
                        "maximum_height_m": height.max(axis=1).tolist(),
                        "lift_window_maximum_height_m": np.where(lift, height, -np.inf).max(axis=1).tolist(),
                        "partial_resets": summary["partial_resets_per_repetition"],
                    }
                )
    rows = []
    for condition in range(8):
        item = {"condition": condition}
        low_height = np.array(
            [value for record in records for value in np.array(record["lift_window_maximum_height_m"])[:, condition]]
        )
        high_height = np.array(
            [
                value
                for record in records
                for value in np.array(record["lift_window_maximum_height_m"])[:, condition + 8]
            ]
        )
        item["low_force_lift_height_range_m"] = [float(low_height.min()), float(low_height.max())]
        item["high_force_lift_height_range_m"] = [float(high_height.min()), float(high_height.max())]

        for scope in ("whole_episode", "lift_window"):
            low, high = [], []
            shifts = []
            for record in records:
                values = np.array(record[scope + "_peak_Pa"])
                first, second = values[:, condition], values[:, condition + 8]
                low.extend(first.tolist())
                high.extend(second.tolist())
                shifts.append(
                    {
                        "model": record["model"],
                        "radius_scale": record["radius_scale"],
                        "high_minus_low_peak_Pa": (second - first).tolist(),
                    }
                )
            low, high = np.array(low), np.array(high)
            difference = high - low
            stable = bool((difference > 0).all() or (difference < 0).all())
            gap = float(abs(difference).min())
            band = float(max(np.ptp(low), np.ptp(high)))
            item[scope] = {
                "raw_differences": shifts,
                "rank_stable": stable,
                "lower_peak_choice": ("0.2N" if difference[0] > 0 else "4N") if stable else "unresolved",
                "minimum_absolute_separation_Pa": gap,
                "observed_model_radius_band_Pa": band,
                "separation_exceeds_observed_band": stable and gap > band,
                "low_force_threshold_disagreement_interval_Pa": [float(low.min()), float(low.max())],
                "high_force_threshold_disagreement_interval_Pa": [float(high.min()), float(high.max())],
            }
        item["ranking_changes_with_phase"] = (
            item["whole_episode"]["lower_peak_choice"] != item["lift_window"]["lower_peak_choice"]
        )
        rows.append(item)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    result = {
        "scope": __doc__,
        "recorded_environment_steps": 4 * 3 * 1200 * 16,
        "per_configuration_repeats": 3,
        "warmup_steps_per_configuration": 900,
        "records": records,
        "conditions": rows,
        "input_sha256": inputs,
        "limitations": "Observed model/radius ranges on unchanged complete trajectories are sensitivity bands, not physical confidence bounds. A threshold inside an observed peak range can change the counterfactual episode-level stop decision; no early termination was executed and no physical fracture threshold is calibrated. Source-prior is another load model, not ground truth. Existing contact/feedback/reference and raw-peak uncertainty remain.",
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "conditions": 8,
                "recorded_environment_steps": result["recorded_environment_steps"],
                "whole_episode_resolved": sum(x["whole_episode"]["separation_exceeds_observed_band"] for x in rows),
                "lift_window_resolved": sum(x["lift_window"]["separation_exceeds_observed_band"] for x in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
