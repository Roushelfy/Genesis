"""Summarize separate moving traces without treating intrusive timings as rates."""

import argparse
import hashlib
import json
from pathlib import Path

GROUPS = {
    "pressure_anchor": ("kernel_anchor_", "kernel_refine_contacts_", "kernel_pack_contacts_"),
    "pressure_sample_and_fit": ("kernel_pressure_warp_", "kernel_pressure_correct_warp_"),
    "intrinsic_support_and_samples": (
        "kernel_fan_frontier_",
        "kernel_refine_failed_",
        "kernel_select_failed_",
        "kernel_refresh_refined_",
        "kernel_bind_traction_support_",
    ),
    "intrinsic_fit": ("kernel_fit_tractions_impl_",),
    "intrinsic_contact_pack": ("kernel_pack_intrinsic_contacts_", "kernel_commit_intrinsic_contacts_"),
    "scatter": ("kernel_scatter_faces_impl_", "kernel_scatter_cached_tractions_", "kernel_load_intrinsic_contacts_"),
    "elastic_recovery": (
        "kernel_surface_pack_",
        "kernel_surface_apply_packed_",
        "kernel_direct_init_",
        "kernel_apply_inverse_",
        "kernel_solve_cooperative_",
    ),
    "balance_residual_acceptance": ("kernel_balance_impl_", "kernel_full_residual_impl_", "kernel_accept_impl_"),
    "peak": ("kernel_peak_impl_",),
    "history": ("kernel_history_",),
    "fused_rigid_stress_post_unseparated": ("kernel_rigid_stress_post",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--summaries", default="trace-summaries-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    records, inputs = [], {}
    for case in ("pressure-default-live", "pressure-scatter128-live", "intrinsic-live", "pressure-unfused-live"):
        cohort = "trace-pressure-unfused-v1" if case == "pressure-unfused-live" else "trace-b2048-v1"
        trace_path = args.root / args.summaries / (case + ".json")
        benchmark_path = args.root / cohort / case / "results/benchmark.json"
        for path in (trace_path, benchmark_path):
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        trace = json.loads(trace_path.read_text())
        benchmark = json.loads(benchmark_path.read_text())
        assert trace["step_count"] == 600 and benchmark["envs"] == 2048
        assert benchmark["history"] == 0 and benchmark["quadrature"] == 10
        groups = {name: 0.0 for name in GROUPS}
        matched = {name: [] for name in GROUPS}
        other = 0.0
        for kernel, duration in trace["kernel_total_ms"]:
            names = [name for name, prefixes in GROUPS.items() if kernel.startswith(prefixes)]
            assert len(names) <= 1, (kernel, names)
            if names:
                groups[names[0]] += duration
                matched[names[0]].append(kernel)
            else:
                other += duration
        assert abs(sum(groups.values()) + other - trace["kernel_sum_ms"]) <= 1e-8 * trace["kernel_sum_ms"]
        largest = sorted(benchmark["native_buffers"], key=lambda row: row["bytes"], reverse=True)[:12]
        records.append(
            {
                "case": case,
                "steps": 600,
                "gpu_uuid": benchmark["gpu_uuid"],
                "cpu_step_annotation_ms_per_step": trace["step_wall_sum_ms"] / 600,
                "gpu_kernel_sum_ms_per_step": trace["kernel_sum_ms"] / 600,
                "gpu_interval_union_ms_per_step": trace["kernel_interval_union_ms"] / 600,
                "grouped_kernel_ms_per_step": {name: value / 600 for name, value in groups.items()},
                "other_rigid_controller_framework_kernel_ms_per_step": other / 600,
                "matched_kernel_names": matched,
                "runtime_counts": trace["runtime_counts"],
                "runtime_duration_sum_ms": trace["runtime_duration_sum_ms"],
                "runtime_memcpy_fraction_overlapping_gpu_kernels": trace[
                    "runtime_memcpy_fraction_overlapping_gpu_kernels"
                ],
                "copy_or_memset_sum_ms": trace["copy_or_memset_sum_ms"],
                "explicit_copy_or_memset_bytes": trace.get("explicit_copy_or_memset_bytes"),
                "native_buffer_bytes": benchmark["native_buffer_bytes"],
                "native_logical_buffer_bytes": benchmark["native_logical_buffer_bytes"],
                "largest_native_buffers": largest,
                "setup_seconds_including_build_JIT": benchmark["setup_seconds"],
                "trace_sha256": trace["trace_sha256"],
                "scope": "intrusive moving-contact trace, excluded from headline rates",
            }
        )
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in inputs.items())
    result = {
        "scope": __doc__,
        "records": records,
        "group_prefixes": GROUPS,
        "input_sha256": inputs,
        "limitations": [
            "Pressure fused post kernels cannot be reliably split into scientific stages from generated names.",
            "Pressure support quadrature and fitting share a kernel; report their combined cost.",
            "The unfused pressure diagnostic changes scheduling and is not a decomposition of the default wall time.",
            "Kernel categories are disjoint; CPU/API spans overlap GPU work and must not be added to GPU durations.",
            "History0 dispatches no history prediction/maintenance; history4 has a separate repeated rate experiment.",
            "Logical/native allocation bytes and copy durations are not DRAM traffic or hardware bandwidth counters.",
            "No device DRAM counter was collected; no measured bandwidth bottleneck is claimed.",
        ],
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(records), "passed": True}))


if __name__ == "__main__":
    main()
