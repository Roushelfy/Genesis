"""Actual batched native motion and recovery with independently solved pad states."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

import genesis as gs

from .artifacts import file_id, runtime_identity, write_json
from .native_pad_batch_mechanics import NativeBatchPad
from .native_pad_batch_recovery import sample_batch
from .native_pad_recovery import NativePadRecovery
from .pad_primary_cost import PrimaryCost
from .plate_native_pad import arguments, history_checks
from .plates import equilibrium, force_history


def main():
    cli = arguments()
    cli.add_argument("--envs", type=int, choices=(1, 2, 4, 8), required=True)
    cli.add_argument("--env-load-scales", type=float, nargs="+")
    args = cli.parse_args()
    assert args.primary_cost and args.record_stride == 1 and args.source_reduction == "none"
    assert args.integrator == "Euler" and not args.empty and not args.shear_distance and args.plate_offset_x == 0
    assert not args.gravity_x or args.friction == 0.3
    scales = np.ones(args.envs) if args.env_load_scales is None else np.array(args.env_load_scales)
    assert scales.shape == (args.envs,) and np.all(scales > 0)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "frames").mkdir()
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    setup = perf_counter()
    mechanics = NativeBatchPad(args)
    recovery = (
        None
        if args.method == "rigid"
        else NativePadRecovery.allocate(args, mechanics.outer, mechanics.faces, mechanics.points, mechanics.xml)
    )
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "mesh": file_id(args.mesh),
            "assets": file_id(args.assets / "result.json"),
            "mass_kg_per_env": mechanics.mass_kg,
            "setup_seconds": perf_counter() - setup,
            "scope": "Each environment owns its actual pad geometry/history and seven-velocity contact solve. One native batched scene step advances all environments with their independently computed wrenches. A separate observer performs one batched call to the existing native Quadrants recovery. No single trajectory or source packet is broadcast as simulated batch work. The per-environment pad adapter runs serially and its cost is included.",
        },
    )
    primary = PrimaryCost(args)
    histories, clock_rows, completed = [[] for _ in range(args.envs)], [], False
    started = perf_counter()
    with (args.output / "steps.jsonl").open("w", buffering=1) as journal:
        try:
            for step, (level, within, command, endpoint) in enumerate(force_history(args)):
                primary.begin(level, step)
                commands = command * scales
                rows, forces, rotations, angular, frames = mechanics.advance(commands, step + 1)
                observe = endpoint or (step + 1) % args.stress_stride == 0
                if observe and recovery is not None:
                    paths = [
                        args.output / "frames" / f"env{env}-stress-level{level}.npz" if endpoint else None
                        for env in range(args.envs)
                    ]
                    observed, _ = sample_batch(
                        recovery, forces, rotations, angular, np.full(args.envs, args.friction), commands, paths
                    )
                    for row, stress in zip(rows, observed):
                        row["stress"] = stress
                for env, row in enumerate(rows):
                    row["level"] = level
                    histories[env].append(row)
                    if endpoint:
                        np.savez_compressed(args.output / "frames" / f"env{env}-level{level}.npz", **frames[env])
                journal.write(
                    json.dumps({"step": step + 1, "level": level, "environments": rows}, allow_nan=False) + "\n"
                )
                clock_row = {"level": level}
                if observe and recovery is not None:
                    clock_row["stress"] = {
                        "native_valid": all(r["stress"]["native_valid"] for r in rows),
                        "field_host_bytes": sum(r["stress"]["field_host_bytes"] for r in rows),
                    }
                    assert clock_row["stress"]["native_valid"], (
                        "A batched native observation failed; preserve all environment records"
                    )
                clock_rows.append(clock_row)
                primary.end(level, step, endpoint)
            completed = True
        finally:
            environments = []
            for env, rows in enumerate(histories):
                history = [history_checks(row) for row in rows]
                windows = []
                for level in sorted({r["level"] for r in rows}):
                    selected = [r for r in rows if r["level"] == level]
                    window = [r for r in selected if r["time_s"] >= selected[-1]["time_s"] - 0.05 - 1e-10]
                    checks = []
                    for row in window:
                        check = {**equilibrium(row, selected[-1]["command_force_n"]), **history_checks(row)}
                        body = np.array(row["body_force_n"])
                        if np.linalg.norm(body) > 0:
                            check["sustained_tangential_balance"] = bool(
                                np.linalg.norm(np.array(row["tangential_resultant_n"]) + body[:2])
                                <= 0.01 * np.linalg.norm(body)
                            )
                        checks.append(check)
                    windows.append(
                        {
                            "level": level,
                            "samples": len(window),
                            "checks": {k: all(c[k] for c in checks) for k in checks[0]},
                            "mechanical_window_passed": bool(
                                completed
                                and window[-1]["time_s"] - window[0]["time_s"] >= 0.05 - 1e-10
                                and all(all(c.values()) for c in checks)
                            ),
                        }
                    )
                environments.append(
                    {
                        "env": env,
                        "steps": len(rows),
                        "history_failed_steps": [r["step"] for r, c in zip(rows, history) if not all(c.values())],
                        "mechanical_history_passed": bool(
                            completed and history and all(all(c.values()) for c in history)
                        ),
                        "windows": windows,
                    }
                )
            admitted = bool(
                completed
                and all(
                    e["mechanical_history_passed"] and all(w["mechanical_window_passed"] for w in e["windows"])
                    for e in environments
                )
            )
            write_json(
                args.output / "result.json",
                {
                    "completed": completed,
                    "mechanically_admitted": admitted,
                    "envs": args.envs,
                    "environments": environments,
                    "elapsed_seconds": perf_counter() - started,
                    "physical_accuracy_qualified": False,
                },
            )
            primary.write(clock_rows, completed, admitted)
    assert admitted, "Batched motion failed mechanical admission; do not use its throughput"


if __name__ == "__main__":
    main()
