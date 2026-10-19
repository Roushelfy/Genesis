"""Audit actual batched histories, source isolation and same-mesh single-env fields."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .plates import force_history


def peak(tensor):
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    return float(np.sqrt(1.5 * np.sum(dev * dev, axis=(-2, -1))).max())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runs", type=Path, nargs="+", required=True)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases, receipts = [], []
    for run in args.runs:
        for name in ("start.json", "result.json", "steps.jsonl", "cost.json"):
            receipts.append(file_id(run / name))
        start = json.loads((run / "start.json").read_text())
        config = start["args"]
        method = config.get("method", "shell")
        assert method in ("dpr", "scr", "shell") and config["friction"] == config["gravity_x"] == 0
        assert config["stress_output"] in ("full", "max")
        result = json.loads((run / "result.json").read_text())
        cost = json.loads((run / "cost.json").read_text())
        steps = [json.loads(line) for line in (run / "steps.jsonl").read_text().splitlines()]
        envs = config["envs"]
        count = round((config["ramp"] + config["hold"]) / config["dt"])
        scales = np.ones(envs) if config["env_load_scales"] is None else np.array(config["env_load_scales"])
        checks = {
            "complete_admitted_history": result["completed"] and result["mechanically_admitted"],
            "expected_step_count": len(steps) == count * len(config["loads"]),
            "every_environment_present": all(
                [r["env"] for r in row["environments"]] == list(range(envs)) for row in steps
            ),
            "all_observations_valid": all(
                r["stress"].get("native_valid", r["stress"].get("valid", False))
                for row in steps
                for r in row["environments"]
                if "stress" in r
            ),
        }
        expected = list(force_history(SimpleNamespace(**config)))
        checks["independent_commands_and_cadence"] = len(steps) == len(expected) and all(
            row["step"] == step + 1
            and all(
                abs(record["command_force_n"] - command * scales[env]) <= 1e-14
                and ("stress" in record) == (endpoint or (step + 1) % config["stress_stride"] == 0)
                for env, record in enumerate(row["environments"])
            )
            for step, (row, (_, _, command, endpoint)) in enumerate(zip(steps, expected))
        )
        fields = []
        for env in range(envs):
            for level, base_load in enumerate(config["loads"]):
                load = base_load * scales[env]
                baseline_level = int(np.argmin(np.abs(np.array([0.1, 0.3, 1.0]) - load)))
                assert abs([0.1, 0.3, 1.0][baseline_level] - load) < 1e-12
                endpoint = steps[(level + 1) * count - 1]["environments"][env]
                assert abs(endpoint["command_force_n"] - load) < 1e-12
                if method == "shell":
                    observed = run / f"frames/env{env}-level{level}.npz"
                    baseline = (
                        args.data_root
                        / f"compliant-pad-fine-coarse-history-v1/group0/cycle/frames/level{baseline_level}.npz"
                    )
                    key = "cauchy_world_pa"
                else:
                    group = 0 if method == "dpr" else 1
                    observed = run / f"frames/env{env}-stress-level{level}.npz"
                    baseline = (
                        args.data_root
                        / f"native-pad-full-source-stability-v1/group{group}/native/frames/stress-level{baseline_level}.npz"
                    )
                    key = "stress_pa"
                receipts.extend((file_id(observed), file_id(baseline)))
                with np.load(observed) as saved:
                    tensor = saved[key] if config["stress_output"] == "full" else None
                    source = (
                        {
                            k: saved[k]
                            for k in (
                                "original_contact_forces_n",
                                "source_forces_n",
                                "source_points_m",
                                "original_contact_points_m",
                            )
                        }
                        if method != "shell"
                        else {}
                    )
                with np.load(baseline) as saved:
                    target = saved[key]
                field_error = (
                    float(np.linalg.norm(tensor - target) / np.linalg.norm(target)) if tensor is not None else None
                )
                peak_error = abs(endpoint["stress"]["raw_vm_peak_pa"] / peak(target) - 1)
                source_error, wrench_error = None, None
                if method != "shell":
                    motion = run / f"frames/env{env}-level{level}.npz"
                    receipts.append(file_id(motion))
                    with np.load(motion) as saved:
                        force = saved["contact_force_n"] @ saved["rotation"]
                    scale = max(np.linalg.norm(force), 1e-30)
                    source_error = float(np.linalg.norm(force - source["original_contact_forces_n"]) / scale)
                    force_difference = source["source_forces_n"].sum(axis=0) - force.sum(axis=0)
                    moment_difference = np.cross(source["source_points_m"], source["source_forces_n"]).sum(
                        axis=0
                    ) - np.cross(source["original_contact_points_m"], force).sum(axis=0)
                    wrench_error = float(np.linalg.norm(np.r_[force_difference, moment_difference / 0.03]) / scale)
                fields.append(
                    {
                        "env": env,
                        "level": level,
                        "load_n": float(load),
                        "same_mesh_tensor_relative_l2": field_error,
                        "raw_peak_relative_difference": peak_error,
                        "actual_motion_source_relative_error": source_error,
                        "source_wrench_relative_error": wrench_error,
                    }
                )
        checks["all_single_env_field_controls"] = all(
            (f["same_mesh_tensor_relative_l2"] is None or f["same_mesh_tensor_relative_l2"] <= 0.005)
            and f["raw_peak_relative_difference"] <= 0.01
            for f in fields
        )
        checks["actual_source_isolation"] = all(
            f["actual_motion_source_relative_error"] is None
            or max(f["actual_motion_source_relative_error"], f["source_wrench_relative_error"]) <= 1e-10
            for f in fields
        )
        intervals = sum(s["wall_seconds"] for s in cost["samples"])
        checks["whole_interval_accounting"] = intervals <= result["elapsed_seconds"] * (1 + 1e-6) and result[
            "elapsed_seconds"
        ] - intervals <= max(1.0, result["elapsed_seconds"] * 0.01)
        cases.append(
            {
                "run": str(run),
                "method": method,
                "checks": checks,
                "fields": fields,
                "cost": cost,
                "run_identity": start["identity"],
                "setup_seconds": start["setup_seconds"],
            }
        )
    passed = all(all(case["checks"].values()) for case in cases)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": passed,
            "cases": cases,
            "receipts": receipts,
            "scope": "Independent batched trajectories and complete endpoint tensors compared with admitted same-mesh B1 normal equilibria at0.1/0.3/1N. Actual per-environment motion forces are checked against recovery input packets; fixed0.5% field/1% peak numerical-sensitivity gates apply. This audits batch conformance and measured-history admission, not continuum stress accuracy or a sufficient number of performance repeats.",
        },
    )
    assert passed, "Batched control failed; do not admit its numerical or timing claim"


if __name__ == "__main__":
    main()
