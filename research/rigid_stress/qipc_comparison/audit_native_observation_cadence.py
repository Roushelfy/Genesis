"""Independently audit complete sparse histories, partial resets and matched timings."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .audit_native_production_controls import MOTION


def modes(strides):
    return [
        ("rigid", "max", 1),
        *[(method, output, stride) for method in ("dpr", "scr") for stride in strides for output in ("max", "full")],
    ]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--campaign", type=Path, required=True)
    cli.add_argument("--batches", type=int, nargs="+", required=True)
    cli.add_argument("--strides", type=int, nargs="+", default=[1, 10, 50])
    cli.add_argument("--timing-campaign", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(set(args.batches)) == len(args.batches) and min(args.batches) > 0
    assert len(set(args.strides)) == len(args.strides) and min(args.strides) == 1
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, cases, results, timing_rows = [], [], {}, {}, []

    def read(path):
        receipts.append(file_id(path))
        return json.loads(path.read_text())

    def check(name, passed, **details):
        checks.append({"name": name, "passed": bool(passed), **details})

    def numeric(name, actual, reference, rtol, atol):
        delta = np.abs(actual - reference)
        budget = atol + rtol * np.abs(reference)
        check(
            name,
            np.isfinite(actual).all() and (delta <= budget).all(),
            maximum_absolute_error=float(delta.max()),
            maximum_budget_ratio=float((delta / budget).max()),
            rtol=rtol,
            atol=atol,
        )

    def motion(name, actual, reference):
        for label, first, last, tolerance in MOTION:
            a, b = actual[..., first:last], reference[..., first:last]
            if label == "quaternion_up_to_sign":
                a = np.where((a * b).sum(axis=-1, keepdims=True) < 0, -a, a)
            numeric(name + "/" + label, a, b, 0, tolerance)

    source = read(args.campaign / "source.json")
    for row in source["files"]:
        actual = file_id(Path(source["source"]) / row["path"])
        assert actual["bytes"] == row["bytes"] and actual["sha256"] == row["sha256"], row["path"]
    for group, batch in enumerate(args.batches):
        for method, output, stride in modes(args.strides):
            name = f"{method}-{output}-s{stride}"
            root = args.campaign / f"group{group}" / name
            start, result = read(root / "start.json"), read(root / "result.json")
            assert result["passed"] and all(result["checks"].values())
            spec = start["args"]
            assert (spec["method"], spec["output_mode"], spec["stress_stride"], spec["envs"]) == (
                method,
                output,
                stride,
                batch,
            )
            assert spec["steps"] == 1200 and spec["warmup"] == 900 and spec["field_stride"] == 100
            assert file_id(start["conditions"]["path"]) == start["conditions"]
            receipts.append(start["conditions"])
            path = root / "trajectory.npz"
            assert file_id(path) == result["trajectory"]
            receipts.append(file_id(path))
            with np.load(path) as archive:
                data = {key: archive[key] for key in archive.files}
            assert data["motion"].shape == (1200, batch, 31)
            label = f"B{batch}/{name}"
            check(label + "/motion_finite_errno_zero", np.isfinite(data["motion"]).all() and (data["errno"] == 0).all())
            check(
                label + "/quaternion_normalized",
                np.abs(np.linalg.norm(data["motion"][..., 3:7], axis=-1) - 1).max() <= 1e-12,
            )
            expected_mask = ((np.arange(1200) + 1) % stride == 0) & (method != "rigid")
            check(label + "/freshness_mask", np.array_equal(data["fresh_observation_mask"], expected_mask))
            if method != "rigid":
                check(
                    label + "/requested_valid_peaks",
                    data["valid"][expected_mask].all()
                    and np.isfinite(data["peak_pa"][expected_mask]).all()
                    and (data["peak_pa"][expected_mask] >= 0).all(),
                )
                check(
                    label + "/callback_request_count",
                    result["manual_requests_issued"] == (2100 // stride if stride > 1 else 0),
                )
                previous = np.zeros(batch)
                held = True
                partial_resets = 0
                for tick in range(1200):
                    reset = (tick > data["controller_delays"]) & ((tick - data["controller_delays"]) % 600 == 0)
                    partial_resets += int(reset.sum())
                    previous[reset] = 0
                    if expected_mask[tick]:
                        previous = data["peak_pa"][tick].copy()
                    else:
                        held = held and np.array_equal(data["peak_pa"][tick], previous)
                check(
                    label + "/held_peaks_and_partial_resets",
                    held and partial_resets > 0,
                    reset_environment_count=partial_resets,
                )
            if output == "full":
                ticks = data["field_ticks"]
                stress, vm = data["stress_pa"], data["mises_pa"]
                assert stress.shape == (len(ticks), batch, 480, 4, 6)
                assert np.array_equal(np.intersect1d(ticks, np.arange(99, 1200, 100)), np.arange(99, 1200, 100))
                check(label + "/full_fields_are_fresh", expected_mask[ticks].all() and np.isfinite(stress).all())
                xx, yy, zz, xy, xz, yz = np.moveaxis(stress, -1, 0)
                independent_vm = np.sqrt(
                    0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + xz**2 + yz**2)
                )
                numeric(label + "/independent_vm", vm, independent_vm, 1e-10, 1e-7)
                numeric(label + "/global_max_reduction", data["peak_pa"][ticks], vm.max(axis=(-2, -1)), 1e-10, 1e-7)
                if stride > 1:
                    check(
                        label + "/all_unsampled_full_fields_held_or_reset",
                        result["held_field_checks"] == 1200 - 1200 // stride
                        and result["checks"]["all_unsampled_fields_held_or_partial_reset"],
                    )
            cases[batch, method, output, stride] = data
            results[batch, method, output, stride] = (start, result)
        rigid = cases[batch, "rigid", "max", 1]
        for method, output, stride in modes(args.strides)[1:]:
            data = cases[batch, method, output, stride]
            label = f"B{batch}/{method}-{output}-s{stride}"
            motion(label + "/vs-rigid", data["motion"], rigid["motion"])
            check(label + "/same_contact_counts", np.array_equal(data["contact_counts"], rigid["contact_counts"]))
            baseline = cases[batch, method, output, 1]
            mask = data["fresh_observation_mask"]
            numeric(label + "/sampled_peak_vs_every_step", data["peak_pa"][mask], baseline["peak_pa"][mask], 1e-4, 1e-3)
            if output == "full":
                common, a, b = np.intersect1d(data["field_ticks"], baseline["field_ticks"], return_indices=True)
                assert len(common) >= 12
                numeric(
                    label + "/full_tensor_vs_every_step", data["stress_pa"][a], baseline["stress_pa"][b], 1e-4, 1e-3
                )
            if output == "max":
                numeric(
                    label + "/max_vs_full",
                    data["peak_pa"][mask],
                    cases[batch, method, "full", stride]["peak_pa"][mask],
                    1e-8,
                    1e-3,
                )
    if len(args.batches) > 1:
        reference_batch = min(args.batches)
        for batch in args.batches:
            if batch == reference_batch:
                continue
            ids = np.arange(batch) % reference_batch
            for method, output, stride in modes(args.strides):
                a, b = cases[batch, method, output, stride], cases[reference_batch, method, output, stride]
                label = f"B{batch}-vs-B{reference_batch}/{method}-{output}-s{stride}"
                motion(label, a["motion"], b["motion"][:, ids])
                check(label + "/contact_counts", np.array_equal(a["contact_counts"], b["contact_counts"][:, ids]))
                if method != "rigid":
                    numeric(label + "/peaks", a["peak_pa"], b["peak_pa"][:, ids], 1e-8, 1e-3)
                if output == "full":
                    numeric(label + "/full_fields", a["stress_pa"], b["stress_pa"][:, ids], 1e-8, 1e-3)
    if args.timing_campaign is not None:
        timed_source = read(args.timing_campaign / "source.json")
        assert timed_source == source
        for group, batch in enumerate(args.batches):
            gpu_ids = set()
            batch_rows = []
            for method, output, stride in modes(args.strides):
                name = f"{method}-{output}-s{stride}"
                root = args.timing_campaign / f"group{group}" / name
                start, result = read(root / "start.json"), read(root / "result.json")
                reference_start, reference_result = results[batch, method, output, stride]
                assert result["passed"] and result["conditions"] == reference_start["conditions"]
                assert result["stress_configuration"] == reference_result["stress_configuration"]
                for key in ("envs", "method", "output_mode", "stress_stride", "warmup", "steps"):
                    assert start["args"][key] == reference_start["args"][key]
                assert result["dt_s"] == 0.01 and result["substeps"] == 1 and len(result["repeats"]) == 3
                gpu_ids.add(result["gpu_uuid"])
                for row in result["repeats"]:
                    assert row["scene_steps"] == 1200 and row["invalid_environment_updates"] == 0 and row["seconds"] > 0
                    assert row["requested_updates_per_env"] == (1200 // stride if method != "rigid" else 0)
                    assert np.isclose(row["env_scene_steps_per_second"], batch * 1200 / row["seconds"], rtol=1e-14)
                rates = [row["env_scene_steps_per_second"] for row in result["repeats"]]
                batch_rows.append(
                    {
                        "batch": batch,
                        "method": method,
                        "output": output,
                        "stride": stride,
                        "median_env_scene_steps_per_second": float(np.median(rates)),
                        "minimum": min(rates),
                        "maximum": max(rates),
                        "gpu_uuid": result["gpu_uuid"],
                    }
                )
            check(f"B{batch}/paired_timing_same_gpu", len(gpu_ids) == 1)
            baseline_rate = batch_rows[0]["median_env_scene_steps_per_second"]
            for row in batch_rows:
                row["added_scene_latency_fraction"] = baseline_rate / row["median_env_scene_steps_per_second"] - 1
            timing_rows.extend(batch_rows)
        with (args.output / "timing.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
    passed = all(row["passed"] for row in checks)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "receipts": receipts,
            "source_files_rehashed": len(source["files"]),
            "case_count": len(cases),
            "timing_rows": timing_rows,
            "physical_reference_qualified": False,
            "scope": "Complete1200-step motion/errno/peaks, requested freshness, held/reset outputs,12 common full-field snapshots, max/full and condition-matched batch consistency. Sparse results do not certify unsampled loads or replace known diverse/every-step failures. Warm in-process timings have no policy or host field readback; collider physical error remains pending.",
        },
    )
    if not passed:
        raise SystemExit("Sparse observation conformance failed; all original observations retained")


if __name__ == "__main__":
    main()
