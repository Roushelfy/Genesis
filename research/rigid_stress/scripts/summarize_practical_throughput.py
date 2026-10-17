"""Aggregate recorded timing/profiling metadata without running scientific code.

Read a frozen sacct export alongside actual wrapper exits and source closures.
Partial results remain observations until the enclosing Slurm job completes.
Native buffer accounting is not a measurement of peak process VRAM.
"""

import argparse
import csv
import datetime
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exit_code(path):
    return int(path.read_text()) if path.exists() else None


def source_closure(run):
    manifest = run / "snapshot.json"
    inventory = run / "snapshot-inventory-after.json"
    if not manifest.exists() or not inventory.exists():
        return False
    rows = json.loads(manifest.read_text())["files"]
    expected = [row["path"] + ": OK" for row in rows]
    checks = [run / f"snapshot-{stage}-verification.txt" for stage in ("before", "after")]
    return all(path.exists() and path.read_text().splitlines() == expected for path in checks) and json.loads(
        inventory.read_text()
    ) == {"extra_files": [], "missing_files": [], "passed": True}


def warmup_summary(rows):
    result = {"samples": len(rows), "sampling_interval_steps": 10}
    for key in (
        "contacts",
        "contact_fit_iterations_max",
        "corrections",
        "fallbacks",
        "contact_integration_retries",
        "history_hits",
    ):
        counts = Counter(value for row in rows if key in row for value in row[key])
        count = sum(counts.values())
        result[key] = {
            "count": count,
            "unobserved_sample_steps": sum(key not in row for row in rows),
            "histogram": dict(sorted(counts.items())),
            "mean": sum(value * frequency for value, frequency in counts.items()) / count if count else None,
        }
    for key in ("contact_wrench_force_error_N", "contact_wrench_moment_error_Nm"):
        result[key] = max((value for row in rows if key in row for value in row[key]), default=None)
    return result


def normalized_configuration(configuration):
    return [{key: value for key, value in row.items() if key != "mesh"} for row in configuration]


def rollout_work(metrics, load_partition):
    if metrics is None:
        return None
    rows = []
    for repeat in metrics["repeats"]:
        totals = repeat["totals"]
        observed = totals["observed_steps"]
        assert observed > 0
        per_environment = repeat["per_environment"]
        rows.append(
            {
                "observed_environment_steps": observed,
                "valid_stress_contact_fraction": totals["loaded_steps"] / observed,
                "two_finger_contact_fraction": totals["two_finger_contact_steps"] / observed,
                "history_prediction_hit_fraction": (
                    totals["history_prediction_hit_steps"] / observed
                    if "history_prediction_hit_steps" in totals
                    else None
                ),
                "recorded_fit_evaluations": totals["fit_evaluation_sum"],
                "fit_evaluations_per_observed_environment_step": totals["fit_evaluation_sum"] / observed,
                "maximum_fit_evaluations_in_one_step": max(per_environment["fit_evaluation_max"]),
                "maximum_samples_per_source": max(per_environment["sample_count_max"]),
                "maximum_nonzero_egg_contacts": max(per_environment["egg_contact_count_max"]),
                "integration_retry_fraction": totals["integration_retry_frames"] / observed,
                "elastic_corrections_per_environment_step": totals["elastic_corrections"] / observed,
                "elastic_fallbacks_per_environment_step": totals["elastic_fallbacks"] / observed,
                "invalid_stress_steps": totals["invalid_stress_steps"],
                "work_percentiles": repeat["work_percentiles"],
            }
        )
    return {
        "repeats": rows,
        "counter_contract": metrics["repeats"][0]["contract"],
        "evaluation_interpretation": (
            "The body-union leader carries cumulative dual evaluations. Inactive member constraints retain zero evaluations; only wrench errors/status propagate to members. The sum counts evaluations, not independent solves or GPU time."
            if load_partition == "body_source_prior"
            else "Evaluation counts sum the recorded active contact constraints; they are not GPU time."
        ),
        "contact_fraction_interpretation": "Any valid stress-contact witness, including a valid zero-force witness. Nonzero rigid contacts use the separately recorded threshold.",
    }


def sampled_memory(run, gpu_uuid, filename="process-memory-samples.csv"):
    path = run / filename
    if not path.exists():
        return None
    samples = Counter()
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            if (row.get("gpu_uuid") or "").strip() != gpu_uuid:
                continue
            value = (row.get("used_gpu_memory [MiB]") or "").strip().removesuffix(" MiB").strip()
            if value.isdigit():
                samples[row["timestamp"]] += int(value) * 1024**2
    peak = max(samples.values(), default=None)
    return {
        "path": str(path),
        "sha256": digest(path),
        "sampling_interval_ms": 100,
        "samples_with_compute_processes": len(samples),
        "maximum_observed_compute_process_bytes": peak,
        "samples_at_observed_maximum": sum(value == peak for value in samples.values()),
        "scope": "Sum of compute-process memory on the benchmark GPU UUID. A sampled lower bound on peak VRAM; includes process contexts and excludes graphics/driver allocations.",
    }


def kernel_stage(name):
    for stage, prefixes in (
        ("controller", ("kernel_inputs_",)),
        ("rigid_dynamics", ("kernel_substep_dynamics_",)),
        ("collision", ("kernel_substep_collision_",)),
        ("rigid_constraint_solve", ("func_solve_init_", "_kernel_solve_graph_", "_qd_graph_do_while_cond")),
        ("rigid_integration", ("kernel_step_2_",)),
        ("reset", ("kernel_reset_", "kernel_forward_kinematics_links_geoms_")),
        (
            "contact_mapping",
            (
                "kernel_associate_",
                "kernel_resolve_stress_contacts_",
                "kernel_capture_rigid_sources_",
                "kernel_pack_intrinsic_contacts_",
                "kernel_load_intrinsic_contacts_",
                "kernel_bind_source_vertices_",
                "kernel_resolve_sources_",
            ),
        ),
        ("path_queries", ("kernel_footprint_ports_",)),
        ("surface_sampling", ("kernel_fan_frontier_", "kernel_surface_fan_profiles_")),
        (
            "source_priors",
            ("kernel_bind_traction_support_", "kernel_snapshot_source_priors_", "kernel_partition_source_priors_"),
        ),
        ("traction_fit", ("kernel_fit_tractions_impl_",)),
        (
            "traction_refinement",
            ("kernel_refine_failed_", "kernel_select_failed_fans_", "kernel_refresh_refined_source_priors_"),
        ),
        (
            "wrench_acceptance",
            ("kernel_search_wrench_certificate_", "kernel_certify_continuous_", "kernel_diagnose_wrenches_"),
        ),
        (
            "load_assembly_and_diagnostics",
            (
                "kernel_scatter_cached_tractions_",
                "kernel_scatter_tractions_",
                "kernel_commit_intrinsic_contacts_",
                "kernel_commit_body_partition_",
                "kernel_capture_contact_failure_",
            ),
        ),
        ("inertia_relief", ("kernel_balance_impl_",)),
        ("history_prediction", ("kernel_predict_", "kernel_hits_", "kernel_append_")),
        ("elastic_solve", ("kernel_apply_inverse_", "kernel_solve_cooperative_", "kernel_direct_init_")),
        ("complete_residual", ("kernel_full_residual_",)),
        ("global_peak_and_full_output", ("kernel_peak_", "kernel_full_field_")),
        ("step_lifecycle", ("kernel_begin_step_", "kernel_accept_")),
        ("torch_auxiliary", ("void at::",)),
    ):
        if name.startswith(prefixes):
            return stage
    return "unclassified"


def trace_summary(path):
    events = json.loads(path.read_bytes())["traceEvents"]
    timed = [event for event in events if event.get("ph") == "X" and "dur" in event]
    kernels = Counter()
    runtime_counts, runtime_us, copies = Counter(), Counter(), Counter()
    stage_us, stage_counts, resources = Counter(), Counter(), {}
    step_count, step_wall_us = 0, 0.0
    for event in timed:
        category, name = event.get("cat"), event["name"]
        if category == "kernel":
            kernels[name] += event["dur"]
            stage = kernel_stage(name)
            stage_us[stage] += event["dur"]
            stage_counts[stage] += 1
            resources[name] = {
                key: value
                for key, value in event.get("args", {}).items()
                if key in ("registers per thread", "shared memory", "grid", "block", "est. achieved occupancy %")
            }
        elif category in ("cuda_runtime", "cuda_driver"):
            runtime_counts[name] += 1
            runtime_us[name] += event["dur"]
        elif category in ("gpu_memcpy", "gpu_memset"):
            copies[name] += 1
        elif category == "user_annotation" and name == "rigid_stress_trace_step":
            step_count += 1
            step_wall_us += event["dur"]
    return {
        "path": str(path),
        "sha256": digest(path),
        "step_count": step_count,
        "step_wall_sum_ms": step_wall_us / 1000,
        "kernel_count": sum(event.get("cat") == "kernel" for event in timed),
        "kernel_duration_sum_ms": sum(kernels.values()) / 1000,
        "top_kernel_duration_sum_ms": [[name, duration / 1000] for name, duration in kernels.most_common(20)],
        "top_kernel_resource_metadata": {name: resources[name] for name, _ in kernels.most_common(20)},
        "kernel_stage_duration_sum_ms": {name: value / 1000 for name, value in stage_us.items()},
        "kernel_stage_counts": dict(stage_counts),
        "unclassified_kernel_names": [name for name in kernels if kernel_stage(name) == "unclassified"],
        "runtime_counts": dict(runtime_counts),
        "runtime_duration_sum_ms": {name: value / 1000 for name, value in runtime_us.items()},
        "copy_counts": dict(copies),
        "scope": "Intrusive moving steps after the timed rollout; not the complete trajectory or headline timing. Nested CPU/runtime times overlap GPU work and must not be added to kernel sums.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--sacct", type=Path, required=True, help="sacct --format=JobID,State,ExitCode,Elapsed,NodeList -P"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.root / "meta/baseline-contract-v1.json"
    contract = json.loads(contract_path.read_text())
    scheduler = {}
    for line in args.sacct.read_text().splitlines()[1:]:
        values = line.split("|")
        if len(values) >= 5 and "." not in values[0]:
            scheduler[values[0].split("_")[0]] = dict(zip(("job_id", "state", "exit", "elapsed", "node"), values))
    points, jobs, profiles, pending_reads = [], [], [], []
    for journal_path in sorted((args.root / "meta").glob("*-command.json")):
        journal = json.loads(journal_path.read_text())
        name = journal_path.name.removesuffix("-command.json")
        run = args.root / name
        job_id = journal["stdout"].strip().split(";")[0]
        state = scheduler.get(job_id)
        scheduler_passed = state is not None and state["state"] == "COMPLETED" and state["exit"] == "0:0"
        jobs.append(
            {
                "name": name,
                "submission_sha256": digest(journal_path),
                "scheduler": state,
                "wrapper_exit": exit_code(run / "returncode.txt"),
            }
        )
        candidates = [run / "results/benchmark.json", *sorted(run.glob("*/results/benchmark.json"))]
        for path in candidates:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                if scheduler_passed:
                    raise
                pending_reads.append({"path": str(path), "reason": "Nonterminal job has an incomplete JSON write."})
                continue
            child = path.parent.parent
            closed = source_closure(child)
            passed = scheduler_passed and closed and exit_code(child / "returncode.txt") == 0
            passed = passed and exit_code(child / "program-returncode.txt") == 0
            condition_pair_closed = True
            if child != run:
                passed = passed and exit_code(run / "returncode.txt") == 0
                before, after = (run / f"conditions-{stage}.sha256" for stage in ("before", "after"))
                condition_pair_closed = before.exists() and after.exists() and before.read_bytes() == after.read_bytes()
            common = {
                "name": name,
                "trial": child.name,
                "path": str(path),
                "sha256": digest(path),
                "scope": data["scope"],
                "width": data["envs"],
                "job_id": job_id,
                "source_closed": closed,
                "job_and_wrapper_passed": passed,
                "condition_pair_closed": condition_pair_closed,
                "gpu": data["gpu"],
                "gpu_uuid": data["gpu_uuid"],
                "gpu_total_bytes": data["gpu_total_bytes"],
                "snapshot_manifest_sha256": digest(child / "snapshot-files.sha256"),
            }
            if data["scope"] == "profile":
                histogram = Counter(data["contacts_per_environment"])
                profiles.append(
                    {
                        **common,
                        "stage_wall_ms": data["stage_wall_ms"],
                        "contact_histogram": dict(sorted(histogram.items())),
                        "scope_note": "Frozen recovery snapshot; repeated stages include synchronization and do not represent mean moving-step cost.",
                    }
                )
                continue
            repeats = data["repeats"]
            rates = [row["transitions_per_second"] for row in repeats]
            invalid = sum(row["invalid_environment_steps"] for row in repeats)
            overflow = sum(row["scatter_overflow_calls"] for row in repeats)
            arguments = data["arguments"]
            same_protocol = all(
                arguments[key] == value
                for key, value in {
                    "warmup": 900,
                    "steps": 2400,
                    "repetitions": 3,
                    "minimum_seconds": 0,
                    "varied": True,
                    "seed": 623016,
                    "level": 1,
                }.items()
            )
            intrusive = (
                bool(arguments["trace"])
                or data["trajectory_metrics"] is not None
                or journal.get("intrusive_memory_capacity_diagnostic", False)
            )
            same_physics = data["conditions_sha256"] == contract["conditions_sha256"]
            same_physics = same_physics and data["dt"] == 0.01 and data["substeps"] == 1
            if data["scope"] in ("live", "policy"):
                same_physics = same_physics and data["tetrahedra"] == 480 and data["dofs"] == 2430
                same_physics = same_physics and normalized_configuration(
                    data["stress_configurations"]
                ) == normalized_configuration(contract["prior_exact_stress_configuration"])
            point = {
                **common,
                "repeat_rates_valid_env_steps_per_second": rates,
                "median_rate": statistics.median(rates),
                "median_batch_latency_ms": statistics.median(row["seconds"] / row["steps"] * 1000 for row in repeats),
                "invalid_environment_steps": invalid,
                "scatter_overflow_calls": overflow,
                "intrusive_diagnostic": intrusive,
                "same_timing_protocol": same_protocol,
                "same_primary_physics_and_options": same_physics,
                "eligible_primary_headline": passed
                and condition_pair_closed
                and same_protocol
                and same_physics
                and not intrusive
                and invalid == 0
                and overflow == 0,
                "native_buffer_bytes_at_end": data.get("native_buffer_bytes"),
                "torch_peak_allocated_bytes": data["torch_peak_allocated_bytes"],
                "device_free_bytes_at_end": data["device_free_bytes"],
                "effective_contact_block_size": data["effective_contact_block_size"],
                "warmup": warmup_summary(data["warmup_quality"]),
                "rollout_work": rollout_work(data["trajectory_metrics"], data["intrinsic_load_partition"]),
                "sampled_process_memory": sampled_memory(run, data["gpu_uuid"])
                or sampled_memory(run, data["gpu_uuid"], f"process-memory-{child.name}.csv"),
            }
            if "trace" in data:
                trace = path.with_suffix(".trace.json")
                assert trace.exists(), trace
                point["trace"] = trace_summary(trace)
                point["trace_metadata"] = data.get("trace_metadata")
            points.append(point)
    comparisons = []
    for name in sorted({point["name"] for point in points}):
        group = {point["scope"]: point for point in points if point["name"] == name}
        if all(
            scope in group and group[scope]["eligible_primary_headline"]
            for scope in ("rigid", "live", "policy-rigid", "policy")
        ):
            comparisons.append(
                {
                    "name": name,
                    "width": group["live"]["width"],
                    "live_incremental_stress_ms": group["live"]["median_batch_latency_ms"]
                    - group["rigid"]["median_batch_latency_ms"],
                    "policy_incremental_stress_ms": group["policy"]["median_batch_latency_ms"]
                    - group["policy-rigid"]["median_batch_latency_ms"],
                }
            )
    pairs = []
    for name in sorted({point["name"] for point in points if "paired-live" in point["name"]}):
        group = {point["trial"]: point for point in points if point["name"] == name}
        trials = ("baseline-a1", "candidate-b1", "candidate-b2", "baseline-a2")
        if all(trial in group for trial in trials):
            baseline = [
                value
                for trial in ("baseline-a1", "baseline-a2")
                for value in group[trial]["repeat_rates_valid_env_steps_per_second"]
            ]
            candidate = [
                value
                for trial in ("candidate-b1", "candidate-b2")
                for value in group[trial]["repeat_rates_valid_env_steps_per_second"]
            ]
            same_gpu = len({group[trial]["gpu_uuid"] for trial in trials}) == 1
            pairs.append(
                {
                    "name": name,
                    "same_gpu": same_gpu,
                    "all_trials_eligible": same_gpu
                    and all(group[trial]["eligible_primary_headline"] for trial in trials),
                    "baseline_repeat_rates": baseline,
                    "candidate_repeat_rates": candidate,
                    "candidate_median_rate_ratio": statistics.median(candidate) / statistics.median(baseline),
                    "candidate_minimum_over_baseline_maximum": min(candidate) / max(baseline),
                    "baseline_trial_median_drift_ratio": group["baseline-a2"]["median_rate"]
                    / group["baseline-a1"]["median_rate"],
                    "note": "Observed repeat range, not a statistical confidence interval. ABBA uses separate warmed processes on one GPU.",
                }
            )
    result = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "aggregation_source_sha256": digest(Path(__file__)),
        "baseline_contract_sha256": digest(contract_path),
        "scheduler_export_sha256": digest(args.sacct),
        "jobs": jobs,
        "pending_artifact_reads": pending_reads,
        "points": points,
        "frozen_profiles": profiles,
        "four_scope_comparisons": comparisons,
        "same_gpu_paired_live_comparisons": pairs,
        "memory_note": "Native buffers and Torch allocator peaks cover different allocators. Neither the end-of-run free memory nor native byte count establishes peak process VRAM.",
        "goal_completion": False,
    }
    assert not args.output.exists(), args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "points": len(points),
                "eligible_primary_headline": sum(point["eligible_primary_headline"] for point in points),
                "frozen_profiles": len(profiles),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
