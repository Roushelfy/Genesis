"""Independently reread headline timing/configuration records using stdlib only.

This checks recorded rates and the frozen workload, not physical field accuracy.
Source/scheduler/wrapper qualification and numerical evidence remain separate.
"""

import argparse
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    summary = json.loads(args.summary.read_text())
    contract = json.loads(args.contract.read_text())
    assert hashlib.sha256(args.contract.read_bytes()).hexdigest() == summary["baseline_contract_sha256"]
    points = [row for row in summary["points"] if row["eligible_primary_headline"]]
    assert points
    reference = next(row for row in points if row["name"] == "baseline-four-scopes-b256-v1" and row["scope"] == "live")
    reference = json.loads(Path(reference["path"]).read_text())
    expected = [
        {key: value for key, value in row.items() if key != "mesh"}
        for row in contract["prior_exact_stress_configuration"]
    ]
    policy = {key: value for key, value in reference["policy"].items() if key != "warmup_reference_max_error_rad"}
    assert policy["control_error_budget_rad"] == 1e-9
    checks = []
    for point in points:
        path = Path(point["path"])
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == point["sha256"], path
        row = json.loads(content)
        assert row["envs"] == point["width"] and row["scope"] == point["scope"], path
        assert row["precision"] == "64" and row["dt"] == 0.01 and row["substeps"] == 1, path
        assert row["level"] == 1 and row["quadrature"] == 10, path
        assert row["warmup_steps"] == 900 and len(row["repeats"]) == 3, path
        assert row["conditions_sha256"] == contract["conditions_sha256"], path
        assert row["egg_geometry"] == "mesh64" and row["collision_method"] == "mpr" and row["contact_patch"] == "off", (
            path
        )
        assert row["native_controller"] and row["seed"] == 623016 and row["varied"], path
        assert row["output_mode"] == "max" and row["history"] == 4, path
        assert row["contact_model"] == "intrinsic" and row["intrinsic_load_partition"] == "body_source_prior", path
        assert row["gpu"] == reference["gpu"] and row["gpu_total_bytes"] == reference["gpu_total_bytes"], path
        assert row["quadrants"] == reference["quadrants"] and row["torch"] == reference["torch"], path
        assert {
            key: value for key, value in row["policy"].items() if key != "warmup_reference_max_error_rad"
        } == policy, path
        error = row["policy"]["warmup_reference_max_error_rad"]
        assert math.isfinite(error) and 0 <= error <= policy["control_error_budget_rad"], path
        if row["scope"] in ("live", "policy"):
            assert row["tetrahedra"] == 480 and row["dofs"] == 2430, path
            actual = [
                {key: value for key, value in item.items() if key != "mesh"} for item in row["stress_configurations"]
            ]
            assert actual == expected, path
        assert row["arguments"]["minimum_seconds"] == 0, path
        rates = []
        for repeat in row["repeats"]:
            assert repeat["steps"] == 2400 and repeat["seconds"] > 0, path
            assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0, path
            failures = repeat["failed_steps_per_environment"]
            if failures is not None:
                assert len(failures) == row["envs"] and all(value == 0 for value in failures), path
            rate = (repeat["steps"] * row["envs"] - repeat["invalid_environment_steps"]) / repeat["seconds"]
            assert math.isfinite(rate) and math.isclose(rate, repeat["transitions_per_second"], rel_tol=1e-12), path
            rates.append(rate)
        assert all(
            math.isclose(first, second, rel_tol=1e-12)
            for first, second in zip(rates, point["repeat_rates_valid_env_steps_per_second"], strict=True)
        ), path
        median = statistics.median(rates)
        assert math.isclose(median, point["median_rate"], rel_tol=1e-12), path
        assert math.isclose(row["envs"] / median * 1000, point["median_batch_latency_ms"], rel_tol=1e-12), path
        checks.append(
            {
                "path": str(path),
                "sha256": point["sha256"],
                "scope": row["scope"],
                "width": row["envs"],
                "recomputed_median_rate": median,
                "policy_warmup_reference_max_error_rad": error,
                "passed": True,
            }
        )
    result = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scope": __doc__,
        "summary_sha256": hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        "contract_sha256": hashlib.sha256(args.contract.read_bytes()).hexdigest(),
        "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "eligible_points": len(checks),
        "checks": checks,
        "configuration": expected,
        "passed": True,
        "goal_completion": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"eligible_points": len(checks), "passed": True}))


if __name__ == "__main__":
    main()
