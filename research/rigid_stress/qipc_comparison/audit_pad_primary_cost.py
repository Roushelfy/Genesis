"""Admit primary whole-workflow timings using each exact run's mechanical evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .audit_pad_batch import peak


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, receipts = [], []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    for run in args.runs:
        cost, result, start = [read(run / name) for name in ("cost.json", "result.json", "start.json")]
        config = start["args"]
        assert cost["primary_timing"] and cost["envs"] == 1
        method = cost["method"]
        sustained_friction = bool(config["gravity_x"])
        assert (config["friction"], config["gravity_x"]) == ((0.3, 1.0) if sustained_friction else (0.0, 0.0))
        audit = read(run.parent / (method + "-audit") / "result.json")
        audited_cases = audit["runs"] if method == "shell" else audit["cases"]
        assert len(audited_cases) == 1
        audited = audited_cases[0]["result" if method == "shell" else "run"]
        assert audited == file_id(run / "result.json"), "Mechanical audit belongs to another result"
        journal = run / "steps.jsonl"
        receipts.append(file_id(journal))
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        admitted = (
            audit["passed"]
            if method != "shell"
            else all(
                r["completed"] and all(w["mechanical_window_passed"] for w in r["equilibrium_windows"])
                for r in audit["runs"]
            )
        )
        if method == "shell":
            admitted &= all(
                not any(r["solver_failure_flags"])
                and r["equation_residual_relative"] <= 1e-3
                and r["physical_pad"]["normal_law_and_transfer_passed"]
                for r in rows
            )
            if sustained_friction:
                admitted &= all(r["physical_pad"]["friction"]["lagged_law_passed"] for r in rows)
        measured = [s for s in cost["samples"] if not s["warmup"]]
        elapsed = result.get("elapsed_seconds", result.get("elapsed_seconds_including_observation_IO"))
        intervals = sum(s["wall_seconds"] for s in cost["samples"])
        count = round((config["ramp"] + config["hold"]) / config["dt"])
        checks = {
            "own_complete_history_admitted": bool(result["completed"] and cost["completed"] and admitted),
            "warm_repeats": len(measured) >= 3 and all(s["steps"] == count for s in measured),
            "all_requested_observations_valid": all(
                s["all_stress_observations_valid"] and s["stress_observations"] == count // config["stress_stride"]
                for s in cost["samples"]
            ),
            "phase_timers_disabled": all(
                r["timing"]["physics_seconds"] == 0
                and ("stress" not in r or r["stress"]["phase_timing_enabled"] is False)
                for r in rows
            ),
            "whole_interval_accounting": intervals <= elapsed * (1 + 1e-6)
            and elapsed - intervals <= max(1.0, elapsed * 0.01),
            "expected_whole_history": len(rows) == count * len(config["loads"]),
        }
        if sustained_friction:
            # Exclude both initial normal loading and the tangential-load ramp
            # from the measured warm repeated blocks.
            assert config["gravity_start"] == 0.2 and config["gravity_ramp"] == 0.05
            checks["measured_blocks_after_friction_warmup"] = config["cost_warm_levels"] >= 2 and all(
                sample["level"] * count * config["dt"] >= config["gravity_start"] + config["gravity_ramp"]
                for sample in measured
            )
        group = 0 if method == "dpr" else 1
        if sustained_friction:
            baseline_run = args.data_root / (
                "compliant-pad-friction250-history-v1/group0/fine"
                if method == "shell"
                else f"native-pad-full-source-stability-v1/group{group + 2}/native"
            )
        else:
            baseline_run = args.data_root / (
                "compliant-pad-fine-coarse-history-v1/group0/cycle"
                if method == "shell"
                else f"native-pad-full-source-stability-v1/group{group}/native"
            )
        key = "cauchy_world_pa" if method == "shell" else "stress_pa"
        fields = []
        targets = {}
        for level in range(len(config["loads"])):
            assert config["loads"][level] == 0.1
            baseline_level = min(level, 1) if sustained_friction else 0
            if baseline_level not in targets:
                name = f"level{baseline_level}.npz" if method == "shell" else f"stress-level{baseline_level}.npz"
                baseline = baseline_run / "frames" / name
                receipts.append(file_id(baseline))
                with np.load(baseline) as saved:
                    targets[baseline_level] = saved[key]
            target = targets[baseline_level]
            target_peak = peak(target)
            endpoint = rows[(level + 1) * count - 1]
            peak_error = abs(endpoint["stress"]["raw_vm_peak_pa"] / target_peak - 1)
            error = None
            if config["stress_output"] == "full":
                path = run / "frames" / (f"level{level}.npz" if method == "shell" else f"stress-level{level}.npz")
                receipts.append(file_id(path))
                with np.load(path) as saved:
                    actual = saved[key]
                assert actual.shape == target.shape
                error = float(np.linalg.norm(actual - target) / np.linalg.norm(target))
            fields.append(
                {
                    "level": level,
                    "baseline_level": baseline_level,
                    "raw_peak_relative_difference": peak_error,
                    "same_mesh_tensor_relative_l2": error,
                }
            )
        checks["admitted_configuration_equilibrium_agreement"] = all(
            f["raw_peak_relative_difference"] <= 0.01
            and (f["same_mesh_tensor_relative_l2"] is None or f["same_mesh_tensor_relative_l2"] <= 0.005)
            for f in fields
        )
        cases.append(
            {
                "run": str(run),
                "method": method,
                "sustained_friction": sustained_friction,
                "checks": checks,
                "cost": cost,
                "fields": fields,
                "setup_seconds": start["setup_seconds"],
                "run_identity": start["identity"],
                "wall_seconds_outside_level_intervals": elapsed - intervals,
            }
        )
    passed = all(all(c["checks"].values()) for c in cases)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": passed,
            "cases": cases,
            "receipts": receipts,
            "scope": "B1 primary timings of the exact admitted complete normal or sustained-friction physical-pad workflow, with3 or more consecutive warm repeated levels. Friction timing excludes both the initial normal level and the tangential ramp level; each endpoint compares with its matching admitted normal/friction same-mesh baseline. Each primary run supplies its own history/field evidence. Full frames and max-only peaks retain0.5% field/1% peak sensitivity checks. Cross-run bitwise equality is not an observer identity test: the separate same-frozen-state control retains its stricter1e-8 threshold. Includes journal, endpoint output and mechanical checking, so not pure production-solver latency. Batched primary cost, whole-device memory peaks and independent process repeats remain separately reported.",
        },
    )
    assert passed, "Primary cost run failed admission; do not replace it with a passing profile"


if __name__ == "__main__":
    main()
