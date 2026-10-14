"""Validate final-source workload records and summarize the nonconvex milestone."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from research.rigid_stress.scripts.verify_nonconvex_publication import digest


def validate_run(path, batch, scope, seed, output_mode, revision):
    assert (path / "returncode.txt").read_text().strip() == "0", path
    assert (path / "source.sha256").read_bytes() == (path / "source-after.sha256").read_bytes(), path
    assert (path / "conditions-before.sha256").read_bytes() == (path / "conditions-after.sha256").read_bytes(), path
    source_count = 0
    for line in (path / "source.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        snapshot = path / "source" / relative.lstrip("/")
        assert digest(snapshot)[0] == expected, snapshot
        if relative.startswith("genesis/"):
            reference = subprocess.check_output(["git", "show", f"{revision}:{relative}"])
            assert hashlib.sha256(reference).hexdigest() == expected, relative
            source_count += 1
    assert source_count >= 30, (path, source_count)
    value = json.loads((path / "result.json").read_text())
    assert (value["envs"], value["scope"], value["seed"], value["output_mode"]) == (batch, scope, seed, output_mode)
    assert (value["dt"], value["substeps"], value["precision"], value["quadrature"]) == (0.01, 1, "64", 10)
    assert value["intrinsic_load_partition"] == "body_source_prior"
    assert value["contact_model"] == "intrinsic" and value["condition_count"] <= batch
    bank_hash = digest(path / "conditions.npz")[0]
    assert value["conditions_sha256"] == bank_hash
    assert len(value["repeats"]) == 3
    for repeat in value["repeats"]:
        assert repeat["steps"] == 1280 and repeat["seconds"] >= 10
        assert repeat["invalid_environment_steps"] == repeat["scatter_overflow_calls"] == 0
        assert len(repeat["failed_steps_per_environment"]) == batch
        assert max(repeat["failed_steps_per_environment"]) == 0
    budgets = [
        np.max(np.array(row["residual_N"]) / np.maximum(1e-11, 1e-8 * np.array(row["rhs_N"])))
        for row in value["warmup_quality"]
        if "residual_N" in row
    ]
    assert budgets and max(budgets) <= 1
    metrics = value["trajectory_metrics"]
    assert len(metrics["repeats"]) == 3
    for row in metrics["repeats"]:
        totals = row["totals"]
        assert totals["observed_steps"] == batch * 1280
        assert (
            totals["invalid_stress_steps"]
            == totals["completed_lift_failure_cycles"]
            == totals["lift_grasp_fail_steps"]
            == 0
        )
        assert totals["completed_lift_success_cycles"] > 0
    telemetry = list(csv.DictReader((path / "device-telemetry.csv").read_text().splitlines(), skipinitialspace=True))
    assert len(telemetry) > 100 and all(row["uuid"] == value["gpu_uuid"] for row in telemetry)
    seconds = sum(row["seconds"] for row in value["repeats"])
    summary = {
        "envs": batch,
        "scope": scope,
        "seed": seed,
        "output_mode": output_mode,
        "conditions_sha256": bank_hash,
        "gpu_uuid": value["gpu_uuid"],
        "effective_env_steps_per_second": batch * 1280 * 3 / seconds,
        "batch_steps_per_second": 1280 * 3 / seconds,
        "repeat_seconds": [row["seconds"] for row in value["repeats"]],
        "native_buffer_bytes": value["native_buffer_bytes"],
        "sampled_device_peak_MiB_including_setup_and_trace": max(
            int(row["memory.used [MiB]"].split()[0]) for row in telemetry
        ),
        "warmup_complete_residual_budget_fraction_max": float(max(budgets)),
        "trajectory_totals": [row["totals"] for row in metrics["repeats"]],
        "verified_numerical_source_files": source_count,
        "condition_count": value["condition_count"],
    }
    return summary, value


def rigid_signature(value):
    keys = (
        "observed_steps",
        "lift_window_steps",
        "lift_grasp_pass_steps",
        "lift_grasp_fail_steps",
        "completed_lift_success_cycles",
        "completed_lift_failure_cycles",
        "weak_grip_steps",
        "weak_grip_descents",
        "release_steps",
        "release_descents",
        "two_finger_contact_steps",
    )
    rows = [{key: row["per_environment"][key] for key in keys} for row in value["trajectory_metrics"]["repeats"]]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def output_pair(parent, batch, seed, scope, revision):
    summaries, signatures, positions, peaks = {}, {}, {}, {}
    field_bytes = {}
    trace_stages = {}
    for route in ("max-before", "full", "max-after"):
        path = parent / scope / route
        mode = "full" if route == "full" else "max"
        summary, value = validate_run(path, batch, scope, seed, mode, revision)
        summaries[route] = summary
        signatures[route] = rigid_signature(value)
        positions[route] = np.array(value["egg_position_m"])
        peaks[route] = np.array(value["maximum_stress_Pa"])
        field_bytes[route] = sum(
            row["bytes"] for row in value["native_buffers"] if row["name"].endswith((".stress_tensor", ".von_mises"))
        )
        assert field_bytes[route] == (batch * 107520 if mode == "full" else 0)
        trace_path = path / "result.trace.json"
        if trace_path.exists():
            events = json.loads(trace_path.read_text())["traceEvents"]
            selected = [
                event for event in events if event.get("cat") == "kernel" and "peak" in event.get("name", "").lower()
            ]
            assert selected, trace_path
            trace_stages[route] = {
                "peak_kernel_names": sorted({event["name"] for event in selected}),
                "peak_kernel_calls": len(selected),
                "peak_kernel_mean_ms_per_trace_step": sum(event["dur"] for event in selected) / 50 / 1000,
                "scope": "Separate intrusive 50-step post-timing trace; includes global peak/output kernels.",
            }
        del value
    assert len({row["gpu_uuid"] for row in summaries.values()}) == 1
    assert len({row["conditions_sha256"] for row in summaries.values()}) == 1
    assert len(set(signatures.values())) == 1, parent
    position_control = float(np.max(np.abs(positions["max-after"] - positions["max-before"])))
    full_position_difference = float(np.max(np.abs(positions["full"] - positions["max-before"])))
    transform_roundoff = 32 * np.finfo(np.float64).eps * float(np.max(np.abs(positions["max-before"])))
    # Closed-loop policy observations may amplify independent CUDA reduction
    # roundoff. Compare the mode effect with the retained same-mode control;
    # the original physical wrench/residual/stress gates stay unchanged.
    assert full_position_difference <= position_control + transform_roundoff, parent
    for route in ("full", "max-after"):
        np.testing.assert_allclose(peaks[route], peaks["max-before"], rtol=1e-4, atol=1e-3)
    pooled_max_seconds = sum(
        summaries[route]["repeat_seconds"][i] for route in ("max-before", "max-after") for i in range(3)
    )
    max_rate = batch * 1280 * 6 / pooled_max_seconds
    full_rate = summaries["full"]["effective_env_steps_per_second"]
    return {
        "batch": batch,
        "seed": seed,
        "scope": scope,
        "routes": summaries,
        "pooled_max_env_steps_per_second": max_rate,
        "full_env_steps_per_second": full_rate,
        "full_over_max_rate_ratio": full_rate / max_rate,
        "full_throughput_cost_percent": 100 * (1 - full_rate / max_rate),
        "field_bytes": field_bytes,
        "identical_per_environment_rigid_grasp_contact_descent_counters": True,
        "full_position_difference_m_max": full_position_difference,
        "same_mode_position_control_difference_m_max": position_control,
        "position_comparison": "Full-mode difference is bounded by the measured independent max/max control plus transform roundoff; exact rigid equality is not claimed for policy rollout.",
        "trace_stages": trace_stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing-only", action="store_true")
    args = parser.parse_args()
    scopes = []
    for batch, seed in (
        (1024, 623101),
        (2048, 623102),
        (4096, 623103),
        (8192, 623104),
        (16384, 991909),
        (32768, 623102),
    ):
        manifest = args.evidence / f"native-final-matched-scopes-b{batch}-{seed}-checkpoint-v1/manifest.json"
        summary = json.loads(manifest.read_text())["summary"]
        assert summary["numerical_source_revision"] == args.revision
        assert summary["identical_per_environment_rigid_grasp_contact_descent_counters"]
        assert summary["rigid_live_position_error_m"] == 0
        for name, row in summary["scopes"].items():
            assert name in ("rigid", "recovery", "live", "policy")
            assert math.isfinite(row["effective_env_steps_per_second"]) and row["effective_env_steps_per_second"] > 0
            assert row["warmup_full_residual_budget_fraction_max"] <= 1
        scopes.append({"manifest": str(manifest), "summary": summary})
    copied = json.loads((args.evidence / "native-final-copied-curve-checkpoint-v1/manifest.json").read_text())[
        "summary"
    ]
    assert copied["source_revision"] == args.revision and copied["same_gpu_and_complete_rigid_counters"]
    assert [row["batch"] for row in copied["widths"]] == [8192, 16384, 24576, 32768]
    edge_parent = args.runtime_root / "20261011-nonconvex/native-final-copied-edge-v1/b36864-623102"
    edge = []
    for scope in ("live", "policy"):
        summary, value = validate_run(edge_parent / scope, 36864, scope, 623102, "max", args.revision)
        assert summary["conditions_sha256"] == copied["conditions_sha256"]
        assert summary["condition_count"] == 2048
        edge.append(summary)
        del value
    result = {
        "numerical_source_revision": args.revision,
        "matched_scopes": scopes,
        "copied_curve": copied,
        "edge_b36864": edge,
    }
    if not args.existing_only:
        current = args.runtime_root / "20261014-closeout"
        result["output_pairs"] = [
            output_pair(current / f"output-modes-b{batch}", batch, seed, scope, args.revision)
            for batch, seed in ((1024, 623101), (8192, 623104))
            for scope in ("live", "policy")
        ]
        reports = {}
        for backend in ("cpu", "cuda"):
            path = current / f"regression-v2-{backend}"
            assert (path / "returncode.txt").read_text().strip() == "0"
            report = ET.parse(path / "report.xml").getroot()
            suite = report.find("testsuite")
            assert int(suite.attrib["errors"]) == int(suite.attrib["failures"]) == 0
            assert int(suite.attrib["tests"]) >= 135
            if backend == "cuda":
                assert int(suite.attrib["skipped"]) == 0
            assert str(Path.cwd() / "genesis/__init__.py") in (path / "import.txt").read_text()
            reports[backend] = {
                "suite": suite.attrib,
                "cases": [
                    {"name": case.attrib["name"], "skipped": case.find("skipped") is not None}
                    for case in suite.findall("testcase")
                ],
            }
        result["final_regression"] = reports
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "matched_widths": len(scopes),
                "edge_rates": [row["effective_env_steps_per_second"] for row in edge],
                "output_pairs": len(result.get("output_pairs", [])),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
