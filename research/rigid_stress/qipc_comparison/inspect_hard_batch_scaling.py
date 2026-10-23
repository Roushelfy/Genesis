"""Check saved moving-scene observations across batch sizes and a fresh memory run."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .time_hard_franka import compare


def read_run(path, inputs):
    start, result = [json.loads((path / name).read_text()) for name in ["start.json", "result.json"]]
    assert result["passed"]
    inputs.extend(file_id(path / name) for name in ["start.json", "result.json"])
    for receipt in start["inputs"] + start["collision_assets"]:
        assert file_id(receipt["path"]) == receipt
    assert start["start_step"] == 100 and start["steps"] == 30
    assert len(start["batches"]) == 1
    batch = start["batches"][0]
    audits = {}
    for row in result["rows"]:
        mode = row["mode"]
        assert row["batch"] == batch and row["audit_environment_steps"] == 130 * batch
        assert len(row["seconds_per_scene_step"]) == start["repeats"]
        source = path / f"{mode}-b{batch}-audit.npz"
        inputs.append(file_id(source))
        with np.load(source) as saved:
            data = {key: saved[key] for key in saved.files}
        assert data["position_m"].shape == (130, batch, 3)
        if mode != "rigid":
            assert data["valid"].all() and (data["input_status"] == 0).all()
            assert (data["recovery_status"] <= 1).all()
            assert row["admitted_timed_environment_steps"] == 30 * batch * start["repeats"]
            assert row["surface_quadrature"] == 16
            np.testing.assert_array_equal(np.isposinf(data["vm_pa"]), data["sharp_edge"])
            for field, label in [("loaded", "loaded"), ("sharp_edge", "sharp")]:
                window = int(data[field][100:].sum())
                assert row[label + "_window_environment_steps"] == window
                assert row[label + "_timed_environment_steps"] == window * start["repeats"]
        audits[mode] = data
    return start, result, audits


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--timings", type=Path, nargs=2, required=True)
    cli.add_argument("--memory", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = []
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=False)})
    runs = [read_run(path, inputs) for path in args.timings]
    first, second = runs
    assert first[0]["identity"] == second[0]["identity"], "Primary batches must share the GPU allocation."
    assert first[0]["inputs"] == second[0]["inputs"]
    assert first[0]["collision_assets"] == second[0]["collision_assets"]
    assert first[0]["repeats"] == second[0]["repeats"] == 3
    sizes = [run[0]["batches"][0] for run in runs]
    assert sizes[1] > sizes[0] and sizes[1] % sizes[0] == 0
    condition_file = Path(first[0]["inputs"][2]["path"])
    with np.load(condition_file) as bank:
        count = len(bank["position"])
    assert all(batch % count == 0 for batch in sizes)
    rows = []
    for mode in ["rigid", "max", "full"]:
        original, enlarged = first[2][mode], second[2][mode]
        # Check every replicated environment, not just the first condition bank.
        for _, _, audits in runs:
            actual = audits[mode]
            indices = np.arange(actual["position_m"].shape[1]) % count
            expected = {key: values[:, indices] for key, values in actual.items()}
            compare(expected, actual)
        indices = np.arange(sizes[1]) % sizes[0]
        compare({key: value[:, indices] for key, value in original.items()}, enlarged)
        for _, _, audits in runs:
            compare(audits["rigid"], audits[mode])
            if mode == "full":
                compare(audits["max"], audits["full"])
        timings = [next(row for row in run[1]["rows"] if row["mode"] == mode) for run in runs]
        rows.append(
            {
                "mode": mode,
                "timings": timings,
                "aggregate_rate_ratio": timings[1]["environment_steps_per_second"]
                / timings[0]["environment_steps_per_second"],
                "every_frame_and_environment_observations_agree": True,
            }
        )
    memory = json.loads((args.memory / "result.json").read_text())
    inputs.append(file_id(args.memory / "result.json"))
    assert memory["passed"] and all(memory["checks"].values()) and not memory["primary_timing"]
    assert memory["driver_error"] is None and memory["memory_error"] is None
    mem_start, mem_result, mem_audits = read_run(args.memory / "driver", inputs)
    assert mem_start["inputs"] == second[0]["inputs"]
    assert mem_start["collision_assets"] == second[0]["collision_assets"]
    assert mem_start["batches"] == [sizes[1]] and mem_start["repeats"] == 1
    assert list(mem_audits) == ["full"]
    compare(second[2]["full"], mem_audits["full"])
    polling = memory["polling_audit"]
    assert polling["collector_returncode"] == 0 and polling["target_pid"] == memory["profile_pid"]
    assert polling["poll_count"] >= polling["visible_poll_count"] > 0
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "passed": True,
            "inputs": inputs,
            "batches": sizes,
            "independent_initial_conditions": count,
            "primary_rows": rows,
            "sampled_gpu_peak_mib": max(sample["process_mib"] for sample in memory["memory"]["samples"]),
            "process_max_rss_kib": memory["process_max_rss_kib"],
            "memory_driver_row": mem_result["rows"][0],
            "memory_polling": polling,
            "scope": "Same-allocation complete rigid/max/full primary timings, three continuous100..129 windows. "
            "All130 audited frames and environments retain the existing position/stress/status checks across "
            "batch sizes, within tiled condition banks and against the separate fresh-process full memory run. "
            "The bank is repeated, not an independently diverse population. Sampled memory is not a continuous "
            "peak or capacity bound. No best-batch, full-cycle or physical-accuracy qualification.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
