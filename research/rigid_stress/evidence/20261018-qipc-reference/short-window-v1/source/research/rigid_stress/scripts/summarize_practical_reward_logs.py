"""Aggregate already-computed whole-episode peak scalars from immutable native logs.

This is stdlib artifact/JSON aggregation, with no simulation, array analysis or
scientific backend. Keep lift-window and whole-episode decisions distinct.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    inputs = {}

    def read(suffix):
        matching = [row for row in manifest["files"] if row["original"].endswith(suffix)]
        assert len(matching) == 1, suffix
        row = matching[0]
        stored = (args.manifest.parent / row["artifact"]).read_bytes()
        assert hashlib.sha256(stored).hexdigest() == row["stored_sha256"]
        assert row["encoding"] in {"gzip", "identity"}
        content = gzip.decompress(stored) if row["encoding"] == "gzip" else stored
        assert len(content) == row["bytes"] and hashlib.sha256(content).hexdigest() == row["sha256"]
        inputs[row["original"]] = row["sha256"]
        return content.decode()

    families = []
    for family, low_limit, analysis in [
        ("grip-choice", 1, "practical-franka-reward-radius-analysis-v1"),
        ("low-force-choice", 0.2, "practical-franka-low-force-radius-analysis-v1"),
    ]:
        snapshots = []
        for model in ["pressure", "body_source_prior"]:
            for half in [False, True]:
                name = f"practical-franka-{family}-{model}" + ("-half-radius" if half else "") + "-v1"
                log = read(f"/{name}/reward-proxy.log")
                repeats = [json.loads(line) for line in log.splitlines() if line.startswith('{"repetition":')]
                assert len(repeats) == 3 and all(len(row["raw_peak_Pa"]) == 16 for row in repeats)
                snapshots.extend(repeats)
        lift = json.loads(read(f"/{analysis}/results/summary.json"))
        rows = []
        for condition in range(8):
            low = [row["raw_peak_Pa"][condition] for row in snapshots]
            high = [row["raw_peak_Pa"][condition + 8] for row in snapshots]
            difference = [second - first for first, second in zip(low, high, strict=True)]
            stable = all(value > 0 for value in difference) or all(value < 0 for value in difference)
            gap = min(abs(value) for value in difference)
            band = max(max(low) - min(low), max(high) - min(high))
            choice = (f"{low_limit:g}N" if difference[0] > 0 else "4N") if stable else "unresolved"
            lift_row = lift["rows"][condition]
            rows.append(
                {
                    "condition": condition,
                    "whole_episode_low_limit_peak_range_Pa": [min(low), max(low)],
                    "whole_episode_high_limit_peak_range_Pa": [min(high), max(high)],
                    "whole_episode_high_minus_low_range_Pa": [min(difference), max(difference)],
                    "whole_episode_rank_stable": stable,
                    "whole_episode_lower_peak_choice": choice,
                    "whole_episode_minimum_absolute_separation_Pa": gap,
                    "whole_episode_absolute_model_radius_band_Pa": band,
                    "whole_episode_separation_exceeds_observed_band": stable and gap > band,
                    "lift_window_lower_peak_choice": lift_row["lower_peak_choice"],
                    "lift_window_separation_exceeds_observed_band": lift_row[
                        "separation_exceeds_observed_absolute_peak_band"
                    ],
                    "lower_peak_choice_changes_with_phase_scope": choice != lift_row["lower_peak_choice"]
                    if lift_row["lower_peak_choice"] != "unresolved"
                    else None,
                }
            )
        families.append({"low_limit_N": low_limit, "high_limit_N": 4, "rows": rows})
    result = {
        "scope": __doc__,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "families": families,
        "input_sha256": inputs,
        "limitations": "Whole-episode peak scalars were already computed from every raw global step observation in the actual native jobs. Lift-window discrimination does not establish lower whole-episode peak or safe termination. Unresolved bands and observed phase-dependent reversals remain; no physical fracture accuracy or statistical confidence bound is claimed.",
        "goal_completion": False,
    }
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "families": [
                    {
                        "low_limit_N": family["low_limit_N"],
                        "whole_episode_rank_stable_conditions": sum(
                            row["whole_episode_rank_stable"] for row in family["rows"]
                        ),
                        "whole_episode_conditions_exceeding_observed_band": sum(
                            row["whole_episode_separation_exceeds_observed_band"] for row in family["rows"]
                        ),
                    }
                    for family in families
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
