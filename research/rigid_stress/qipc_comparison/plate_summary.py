"""Summarize plate throughput, intrusive profiles and native loading/unloading evidence."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    costs, profiles, stability = [], [], []
    for campaign in ("native-cost-v1", "reference-cost-v1"):
        for path in sorted((args.root / campaign).glob("group*/*/cost.json")):
            data = json.loads(path.read_text())
            result = path.parent / "result.json"
            complete = result.is_file() and json.loads(result.read_text())["completed"]
            samples = [row for row in data["samples"] if not row["warmup"]]
            rates = [row["aggregate_env_steps_per_second"] for row in samples]
            admitted = [
                row.get("all_environment_load_and_velocity_passed", row.get("all_observed_equilibria_passed", False))
                for row in samples
            ]
            costs.append(
                {
                    "input": file_id(path),
                    "method": data["method"],
                    "envs": data["envs"],
                    "complete": complete,
                    "repeat_rates": rates,
                    "median_env_steps_per_second": median(rates) if rates else None,
                    "all_observed_mechanical_checks_passed": bool(complete and len(samples) == 3 and all(admitted)),
                    "physical_accuracy_qualified": False,
                }
            )
    for path in sorted((args.root / "native-profile-v1").glob("group*/*/profile.json")):
        data = json.loads(path.read_text())
        groups = {}
        for name, row in data["kernel_totals"].items():
            group = "unclassified"
            if "rigid_stress_post" in name:
                group = "fused_stress_reconstruction_and_field"
            elif "substep_collision" in name:
                group = "collision_and_contact_assembly"
            elif any(token in name for token in ("solve_graph", "solve_init", "island")):
                group = "rigid_constraint_solve"
            elif any(token in name for token in ("substep_pre", "substep_post", "forward_kinematics")):
                group = "mechanics_and_kinematics"
            total = groups.setdefault(group, {"launches": 0, "gpu_duration_sum_ms": 0.0})
            total["launches"] += row["count"]
            total["gpu_duration_sum_ms"] += row["total_us"] / 1000
        receipt = json.loads((path.parent / "result.json").read_text())
        profiles.append(
            {
                "input": file_id(path),
                "method": path.parent.name,
                "envs": data["envs"],
                "steps": data["steps"],
                "kernel_name_groups": groups,
                "process_peak_mib": receipt["process_memory"]["peak_mib"],
                "scope": "Intrusive separate profile; kernel duration sums can overlap and are not exclusive wall-clock stages. Unclassified names remain in the source JSON.",
            }
        )
    for path in sorted((args.root / "native-stability-v1").glob("group*/*/start.json")):
        start = json.loads(path.read_text())
        run = path.parent
        result = run / "result.json"
        receipt = json.loads(result.read_text()) if result.exists() else {}
        history = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        endpoints = [row for row in history if "raw_vm_peak_pa" in row]
        repeated = []
        for index, row in enumerate(endpoints):
            previous = next(
                (old for old in endpoints[:index] if old["command_force_n"] == row["command_force_n"]), None
            )
            if previous is None:
                continue
            with (
                np.load(run / "frames" / f"level{previous['level']}.npz") as first,
                np.load(run / "frames" / f"level{row['level']}.npz") as last,
            ):
                expected, actual = first["stress_voigt_link_pa"], last["stress_voigt_link_pa"]
                repeated.append(
                    {
                        "command_force_n": row["command_force_n"],
                        "first_level": previous["level"],
                        "return_level": row["level"],
                        "corner_voigt_array_relative_change": (
                            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
                        ).item(),
                        "plate_coordinate_return_difference_m": row["compression_coordinate_m"]
                        - previous["compression_coordinate_m"],
                        "scope": "Identical native mesh/array ordering; array norm is a repeatability diagnostic, not a volume-weighted continuum stress error.",
                    }
                )
        stability.append(
            {
                "run": str(run),
                "start": file_id(path),
                "completed": receipt.get("completed", False),
                "contact_radius_m": start["args"]["contact_radius"],
                "endpoints": endpoints,
                "same_load_return_contrasts": repeated,
                "zero_load_scope": "Raw residual/loads retained; the loaded relative equation criterion is not silently reinterpreted as an absolute zero-load criterion.",
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "costs": costs,
            "profiles": profiles,
            "native_stability": stability,
            "reference_qualified": False,
            "scope": "Numerically/mechanically screened compute and stability diagnostics, separate from the unresolved contact-field accuracy qualification.",
        },
    )


if __name__ == "__main__":
    main()
