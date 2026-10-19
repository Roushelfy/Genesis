"""Join admitted whole-workflow timings and separate process-memory profiles."""

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from .artifacts import file_id, runtime_identity, write_json


def physical_config(config):
    excluded = {
        "output",
        "primary_cost",
        "intrusive_memory_profile",
        "intrusive_phase_profile",
        "stress_stride",
        "stress_output",
        "cost_warm_levels",
        "envs",
        "env_load_scales",
        "loads",
    }
    return {k: v for k, v in config.items() if k not in excluded}


def gpu_configuration(identity):
    rows = list(csv.reader(identity["nvidia_smi_visible_inventory"].splitlines()))[1:]
    configurations = {(r[0].strip(), r[3].strip(), r[4].strip()) for r in rows}
    assert len(configurations) == 1, "Ambiguous GPU model/driver/capacity inventory"
    return next(iter(configurations))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--batch-sizes", type=int, choices=(1, 4), nargs="+", default=[1, 4])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert len(set(args.batch_sizes)) == len(args.batch_sizes)
    args.output.mkdir(parents=True, exist_ok=False)
    identities = {}
    receipts = []

    def identify(path):
        path = Path(path)
        if str(path) not in identities:
            identities[str(path)] = file_id(path)
        return identities[str(path)]

    def read(path):
        receipts.append(identify(path))
        return json.loads(Path(path).read_text())

    def admitted_case(audit_path, run):
        audit = read(audit_path)
        assert audit["passed"]
        for receipt in audit["receipts"]:
            assert identify(receipt["path"]) == receipt, "An audit input changed after admission"
        case = next(c for c in audit["cases"] if c["run"] == str(run))
        assert all(case["checks"].values())
        for name in ("start.json", "result.json", "cost.json", "steps.jsonl"):
            assert identify(run / name) in audit["receipts"], "Audit does not bind this exact run"
        return case

    timings, memory = [], []
    configs, gpu_configs = {}, set()
    for envs in args.batch_sizes:
        campaign = "pad-primary-cost-v1" if envs == 1 else "pad-primary-batch4-v1"
        for group, (stride, mode) in enumerate(((1, "full"), (1, "max"), (20, "full"), (20, "max"))):
            for method in ("dpr", "scr", "shell"):
                run = args.data_root / campaign / f"group{group}/{method}"
                audit_path = (
                    args.data_root
                    / (
                        "pad-primary-every-audit-v1/group0/every-step/result.json"
                        if stride == 1
                        else "pad-primary-sparse-audit-v1/group0/sparse/result.json"
                    )
                    if envs == 1
                    else run.with_name(method + "-audit") / "result.json"
                )
                case = admitted_case(audit_path, run)
                start, cost = [read(run / name) for name in ("start.json", "cost.json")]
                config = start["args"]
                assert cost == case["cost"]
                assert cost["primary_timing"] and not cost.get("intrusive_memory_profile", False)
                assert cost["completed"] and cost["envs"] == config.get("envs", 1) == envs
                assert cost["stress_stride"] == config["stress_stride"] == stride
                assert cost["output_mode"] == config["stress_output"] == mode
                assert config["loads"] == [0.1] * 4 and config["friction"] == config["gravity_x"] == 0
                assert config.get("env_load_scales") is None
                gpu_configs.add(gpu_configuration(start["identity"]))
                baseline = configs.setdefault(method, physical_config(config))
                assert physical_config(config) == baseline, "Configuration changed across timing modes/batches"
                samples = [s for s in cost["samples"] if not s["warmup"]]
                assert len(samples) == cost["measured_repeats"] == 3
                assert all(s["steps"] == 200 and s["all_stress_observations_valid"] for s in samples)
                latencies = [1000 * s["wall_seconds"] / s["steps"] for s in samples]
                rates = [envs * s["steps"] / s["wall_seconds"] for s in samples]
                assert median(latencies) == cost["median_latency_ms"]
                for sample, rate in zip(samples, rates):
                    assert sample["aggregate_env_steps_per_second"] == rate
                timings.append(
                    {
                        "method": method,
                        "envs": envs,
                        "output_mode": mode,
                        "stress_stride": stride,
                        "median_batch_step_ms": median(latencies),
                        "minimum_batch_step_ms": min(latencies),
                        "maximum_batch_step_ms": max(latencies),
                        "median_aggregate_env_steps_per_second": median(rates),
                        "warm_repeats": len(samples),
                        "setup_seconds": start["setup_seconds"],
                        "median_observation_host_bytes_per_batch_step": median(
                            s["field_host_bytes"] / s["steps"] for s in samples
                        ),
                        "maximum_endpoint_peak_relative_difference": max(
                            f["raw_peak_relative_difference"] for f in case["fields"]
                        ),
                        "run": str(run),
                        "admission_audit": str(audit_path),
                    }
                )
        for mode_index, mode in enumerate(("full", "max")):
            group = mode_index + (2 if envs == 4 else 0)
            for method in ("dpr", "scr", "shell"):
                wrapper = args.data_root / f"pad-memory-profile-v1/group{group}/{method}"
                run = wrapper / "run"
                profile = read(wrapper / "result.json")
                assert profile["passed"] and all(profile["checks"].values()) and profile["primary_timing"] is False
                for receipt in profile["receipts"]:
                    assert identify(receipt["path"]) == receipt
                case = admitted_case(wrapper.with_name(method + "-audit") / "result.json", run)
                start, cost = [read(run / name) for name in ("start.json", "cost.json")]
                config = start["args"]
                assert config["envs"] == envs and config["stress_output"] == mode and config["stress_stride"] == 1
                assert config["loads"] == [0.1] and config.get("env_load_scales") is None
                assert physical_config(config) == configs[method]
                assert cost == case["cost"] and cost["primary_timing"] is False and cost["intrusive_memory_profile"]
                gpu_configs.add(gpu_configuration(start["identity"]))
                samples = profile["memory"]["samples"]
                assert len(samples) > 1 and len({s["gpu_uuid"] for s in samples}) == 1
                assert max(s["process_mib"] for s in samples) == profile["memory"]["peak_mib"]
                gaps = [b["seconds"] - a["seconds"] for a, b in zip(samples, samples[1:])]
                memory.append(
                    {
                        "method": method,
                        "envs": envs,
                        "output_mode": mode,
                        "stress_stride": 1,
                        "sampled_gpu_process_peak_mib": profile["memory"]["peak_mib"],
                        "cpu_process_max_rss_mib": profile["process_max_rss_kib"] / 1024,
                        "sample_count": len(samples),
                        "median_sample_interval_seconds": median(gaps),
                        "maximum_sample_interval_seconds": max(gaps),
                        "run": str(run),
                        "admission_audit": str(wrapper.with_name(method + "-audit") / "result.json"),
                    }
                )
    assert len(gpu_configs) == 1, "Hardware/driver differs across the compared runs"
    assert len(timings) == 12 * len(args.batch_sizes) and len(memory) == 6 * len(args.batch_sizes)
    for name, rows in (("timings", timings), ("memory", memory)):
        with (args.output / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "passed": True,
            "batch_sizes": args.batch_sizes,
            "full_B1_B4_matrix": set(args.batch_sizes) == {1, 4},
            "gpu_model_driver_capacity": list(next(iter(gpu_configs))),
            "timings": timings,
            "memory": memory,
            "physical_configurations": configs,
            "receipts": receipts,
            "verified_audit_inputs": list(identities.values()),
            "whole_goal_complete": False,
            "scope": "Only the explicitly selected batch sizes. Every timing/profile is bound to its own immutable complete-history and endpoint conformance audit. Timings include the declared research mechanics/recovery/transfers/admission/journal workflow, with consecutive warm blocks rather than independent-process statistics. Native B4 pad mechanics is serial per environment. Separate fresh-process memory profiles use every-step observation and include build/warm/rollout; sampled peaks can miss short transients and are not sparse-cadence memory measurements. Endpoint PyTorch allocation is not substituted for whole-process GPU memory. Normal0.1N configurations only; friction accuracy and final cross-case conclusions remain separate.",
        },
    )


if __name__ == "__main__":
    main()
