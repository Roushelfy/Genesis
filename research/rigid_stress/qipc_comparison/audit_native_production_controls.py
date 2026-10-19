"""Compare complete native scene trajectories, output modes and condition-matched batches."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json


MODES = ("rigid-max", "dpr-max", "dpr-full", "scr-max", "scr-full")
MOTION = (
    ("position_m", 0, 3, 1e-10),
    ("quaternion_up_to_sign", 3, 7, 1e-10),
    ("velocity_m_per_s", 7, 10, 1e-9),
    ("angular_velocity_rad_per_s", 10, 13, 1e-9),
    ("robot_qpos_rad", 13, 22, 1e-10),
    ("robot_qvel_rad_per_s", 22, 31, 1e-9),
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--baseline-campaign", type=Path, help="Also compare stress outputs to the original pinned cohort")
    cli.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    cli.add_argument("--methods", nargs="+", choices=("rigid", "dpr", "scr"), default=["rigid", "dpr", "scr"])
    cli.add_argument(
        "--require-diverse", action="store_true", help="Require distinct authored conditions for every environment"
    )
    cli.add_argument(
        "--group-indices", type=int, nargs="+", help="Explicit source groups for a separately pinned variant"
    )
    args = cli.parse_args()
    assert "rigid" in args.methods and len(args.methods) == len(set(args.methods))
    modes = tuple(mode for mode in MODES if mode.split("-")[0] in args.methods)
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, cases, diversity = [], [], {}, []

    def check(label, passed, **details):
        checks.append({"name": label, "passed": bool(passed), **details})

    def compare_motion(label, actual, reference):
        for name, first, last, tolerance in MOTION:
            a, b = actual[..., first:last], reference[..., first:last]
            if name == "quaternion_up_to_sign":
                a = np.where((a * b).sum(axis=-1, keepdims=True) < 0, -a, a)
            delta = np.abs(a - b)
            error = float(delta.max())
            per_step = delta.max(axis=(1, 2))
            nonzero = np.flatnonzero(per_step > 0)
            exceeded = np.flatnonzero(per_step > tolerance)
            check(
                label + "/" + name,
                error <= tolerance,
                maximum_absolute_error=error,
                target=tolerance,
                first_nonzero_step=int(nonzero[0] + 1) if len(nonzero) else None,
                first_threshold_exceeding_step=int(exceeded[0] + 1) if len(exceeded) else None,
                maximum_error_index=[int(i) for i in np.unravel_index(delta.argmax(), delta.shape)],
            )

    def compare_peak(label, actual, reference):
        delta = np.abs(actual - reference)
        budget = 1e-3 + 1e-8 * np.abs(reference)
        check(
            label,
            (delta <= budget).all(),
            maximum_absolute_error_pa=float(delta.max()),
            maximum_budget_ratio=float((delta / budget).max()),
            relative_target=1e-8,
            absolute_target_pa=1e-3,
        )

    assert len(args.batches) == len(set(args.batches)) and min(args.batches) > 0
    assert args.group_indices is None or len(args.group_indices) == len(args.batches)
    if args.group_indices is None:
        assert all(batch in (1, 8, 32) for batch in args.batches), "Other batches require explicit source groups"
    group_indices = (
        [(1, 8, 32).index(batch) for batch in args.batches] if args.group_indices is None else args.group_indices
    )
    for group, batch in zip(group_indices, args.batches):
        for mode in modes:
            root = args.campaign / f"group{group}" / mode
            start = json.loads((root / "start.json").read_text())
            receipts.append(file_id(root / "start.json"))
            assert start["args"]["envs"] == batch and start["args"]["steps"] == 1200 and start["args"]["warmup"] == 900
            method = mode.split("-")[0]
            assert start["args"]["method"] == method
            path = root / "trajectory.npz"
            receipts.append(file_id(path))
            with np.load(path) as saved:
                data = {name: saved[name] for name in saved.files}
            assert data["motion"].shape == (1200, batch, 31)
            label = f"B{batch}/{mode}"
            check(label + "/finite_motion", np.isfinite(data["motion"]).all())
            check(label + "/errno", (data["errno"] == 0).all())
            check(
                label + "/unit_quaternion",
                np.max(np.abs(np.linalg.norm(data["motion"][..., 3:7], axis=-1) - 1)) <= 1e-12,
            )
            if method != "rigid":
                check(label + "/every_step_valid", data["valid"].all())
                check(
                    label + "/peaks_finite_nonnegative",
                    np.isfinite(data["peak_pa"]).all() and (data["peak_pa"] >= 0).all(),
                )
            if mode.endswith("full"):
                stress, vm = data["stress_pa"], data["mises_pa"]
                ticks = data["field_ticks"]
                assert stress.shape == (len(ticks), batch, 480, 4, 6) and len(ticks) == 11
                assert np.array_equal(ticks, [0, *range(119, 1200, 120)])
                xx, yy, zz, xy, xz, yz = np.moveaxis(stress, -1, 0)
                independent_vm = np.sqrt(
                    0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + xz**2 + yz**2)
                )
                check(label + "/full_finite", np.isfinite(stress).all() and np.isfinite(vm).all())
                check(label + "/independent_mises", np.allclose(vm, independent_vm, rtol=1e-10, atol=1e-7))
                check(
                    label + "/full_max_reduction",
                    np.allclose(data["peak_pa"][ticks], vm.max(axis=(-2, -1)), rtol=1e-10, atol=1e-7),
                )
            cases[batch, mode] = data
        baseline = cases[batch, "rigid-max"]
        bank = Path(start["conditions"]["path"])
        assert file_id(bank) == start["conditions"]
        receipts.append(file_id(bank))
        with np.load(bank) as saved:
            authored = np.column_stack((saved["position"], saved["quaternion"], saved["radius"], saved["friction"]))
        selected = authored[np.arange(batch) % len(authored)]
        distinct = len(np.unique(selected, axis=0))
        diversity.append({"batch": batch, "bank_conditions": len(authored), "distinct_authored_conditions": distinct})
        if args.require_diverse:
            check(f"B{batch}/distinct_authored_conditions", distinct == batch, actual=distinct, required=batch)
        for mode in modes[1:]:
            data = cases[batch, mode]
            compare_motion(f"B{batch}/{mode}-vs-rigid", data["motion"], baseline["motion"])
            check(
                f"B{batch}/{mode}/same_contact_counts",
                np.array_equal(data["contact_counts"], baseline["contact_counts"]),
            )
        for method in ("dpr", "scr"):
            if method not in args.methods:
                continue
            compare_peak(
                f"B{batch}/{method}/max-vs-full",
                cases[batch, method + "-max"]["peak_pa"],
                cases[batch, method + "-full"]["peak_pa"],
            )
    for reference_batch, batch in zip(sorted(args.batches), sorted(args.batches)[1:]):
        # The frozen8-condition bank repeats by environment index. B1 covers
        # only condition0; compare that condition in B8, not the other seven.
        ids = np.array([0]) if reference_batch == 1 else np.arange(batch)
        reference_ids = ids % reference_batch
        for mode in modes:
            a, b = cases[batch, mode], cases[reference_batch, mode]
            label = f"B{batch}-vs-B{reference_batch}/{mode}"
            compare_motion(label, a["motion"][:, ids], b["motion"][:, reference_ids])
            check(
                label + "/same_contact_counts",
                np.array_equal(a["contact_counts"][:, ids], b["contact_counts"][:, reference_ids]),
            )
            if mode != "rigid-max":
                compare_peak(label + "/peak", a["peak_pa"][:, ids], b["peak_pa"][:, reference_ids])
            if mode.endswith("full"):
                delta = np.abs(a["stress_pa"][:, ids] - b["stress_pa"][:, reference_ids])
                budget = 1e-3 + 1e-8 * np.abs(b["stress_pa"][:, reference_ids])
                check(
                    label + "/full_tensor",
                    (delta <= budget).all(),
                    maximum_absolute_error_pa=float(delta.max()),
                    maximum_budget_ratio=float((delta / budget).max()),
                )
    if args.baseline_campaign is not None:
        assert all(batch in (1, 8, 32) for batch in args.batches), "Original baseline covers only B1/B8/B32"
        for batch in args.batches:
            group = (1, 8, 32).index(batch)
            for mode in modes[1:]:
                path = args.baseline_campaign / f"group{group}" / mode / "trajectory.npz"
                receipts.append(file_id(path))
                with np.load(path) as saved:
                    original = {name: saved[name] for name in saved.files}
                actual = cases[batch, mode]
                label = f"B{batch}/{mode}/original-stress-baseline"
                compare_motion(label, actual["motion"], original["motion"])
                compare_peak(label + "/peak", actual["peak_pa"], original["peak_pa"])
                if mode.endswith("full"):
                    delta = np.abs(actual["stress_pa"] - original["stress_pa"])
                    budget = 1e-3 + 1e-8 * np.abs(original["stress_pa"])
                    check(
                        label + "/tensor",
                        (delta <= budget).all(),
                        maximum_absolute_error_pa=float(delta.max()),
                        maximum_budget_ratio=float((delta / budget).max()),
                    )
    passed = all(row["passed"] for row in checks)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "receipts": receipts,
            "recorded_scene_steps_per_case": 1200,
            "case_count": len(cases),
            "batches": args.batches,
            "methods": args.methods,
            "unaudited_methods": [method for method in ("dpr", "scr") if method not in args.methods],
            "condition_diversity": diversity,
            "replay_timing_admitted": False,
            "physical_reference_qualified": False,
            "scope": "Complete motion/errno/peak paths and11 full-field snapshots per full-output run. Condition-matched batches and pure-rigid comparisons use fixed predeclared thresholds. Replay timing and physical continuum accuracy are separate claims.",
        },
    )
    if not passed:
        raise SystemExit("Native motion/output conformance failed; retained detailed errors.")


if __name__ == "__main__":
    main()
