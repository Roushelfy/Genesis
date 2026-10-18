"""Build concise report tables from the closed benchmark metadata using stdlib."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    data = json.loads(args.metadata.read_text())
    assert not data["cohorts_still_incomplete"] and data["passed_metadata_checks"]
    records = data["records"]

    def row(cohort, profile, scope="live"):
        matches = [r for r in records if r["cohort"] == cohort and r["profile"] == profile and r["scope"] == scope]
        assert len(matches) == 1
        return matches[0]

    lines = []

    def table(title, columns, values):
        lines.extend([title, "", "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"])
        lines.extend("| " + " | ".join(str(v) for v in values_row) + " |" for values_row in values)
        lines.append("")

    widths = [32, 2048, 8192, 16384, 32768]
    names = ["preflight-b32-v1", "core-b2048-v1", "wide-b8192-v1", "core-b16384-v1", "wide-b32768-v1"]
    rate_rows, cost_rows, route_rows = [], [], []
    for width, cohort in zip(widths, names):
        scopes = [
            ("shared", "rigid"),
            ("pressure-default", "live"),
            ("intrinsic", "live"),
            ("shared", "policy-rigid"),
            ("pressure-default", "policy"),
            ("intrinsic", "policy"),
        ]
        selected = [row(cohort, p, s) for p, s in scopes]
        assert all(r["headline_eligible"] for r in selected)
        rate_rows.append([f"{width:,}", *[f"{r['median_valid_envstep_s']:,.0f}" for r in selected]])
        for profile, label in [("pressure-default", "Pressure"), ("intrinsic", "Intrinsic")]:
            live, policy = row(cohort, profile), row(cohort, profile, "policy")
            cost_rows.append(
                [
                    f"{width:,}",
                    label,
                    f"{live['median_step_ms']:.3f} / {live['incremental_stress_step_ms']:.3f}",
                    f"{live['step_time_ratio']:.3f} / {100 * live['throughput_loss_fraction']:.2f}%",
                    f"{policy['median_step_ms']:.3f} / {policy['incremental_stress_step_ms']:.3f}",
                    f"{policy['step_time_ratio']:.3f} / {100 * policy['throughput_loss_fraction']:.2f}%",
                ]
            )
            settings = live["requested_effective"]
            route_rows.append(
                [
                    f"{width:,}",
                    label,
                    settings["method_selected"],
                    str(settings["effective_packed_block_sizes"]),
                    str(settings["effective_rigid_post_fusion"]),
                    str(settings["effective_contact_wrench_reuse"]),
                    str(settings["effective_contact_moment_reuse"]),
                ]
            )
    table(
        "Median valid aggregate env-step/s, n=3 per cell:",
        ["B", "Rigid", "Pressure live", "Intrinsic live", "Policy rigid", "Pressure policy", "Intrinsic policy"],
        rate_rows,
    )
    table(
        "Batch / incremental stress latency in ms; time ratio / throughput loss:",
        ["B", "Model", "Live ms / extra ms", "Live ratio / loss", "Policy ms / extra ms", "Policy ratio / loss"],
        cost_rows,
    )
    memory_rows = []
    for width in [32, 2048, 16384, 32768]:
        selected = [row(f"memory-b{width}-v1", p, s) for p, s in scopes]
        assert all(r["intrusive"] for r in selected)
        memory_rows.append([f"{width:,}", *[f"{r['memory']['sampled_peak_GiB']:.3f}" for r in selected]])
    table(
        "Separate sampled process peak GiB,100ms lower bounds:",
        ["B", "Rigid", "Pressure live", "Intrinsic live", "Policy rigid", "Pressure policy", "Intrinsic policy"],
        memory_rows,
    )
    screen_rows = []
    base = row("tuning-b2048-v1", "pressure-default")["median_valid_envstep_s"]
    for profile in [
        "pressure-default",
        "pressure-scatter128",
        "pressure-scatter256",
        "pressure-fast",
        "pressure-tuned",
        "pressure-history4",
    ]:
        r = row("tuning-b2048-v1", profile)
        assert len(set(r["scatter_overflow_calls"])) == 1
        screen_rows.append(
            [
                profile.removeprefix("pressure-"),
                f"{r['median_valid_envstep_s']:,.0f}",
                f"{100 * (r['median_valid_envstep_s'] / base - 1):+.2f}%",
                r["scatter_overflow_calls"][0],
            ]
        )
    table(
        "Same-GPU B2048 live pressure screen:",
        ["Setting", "env-step/s", "Change vs default", "Full fallback passes/repeat"],
        screen_rows,
    )
    confirm = []
    for width in (2048, 16384):
        values = []
        for scope in ("live", "policy"):
            default, tuned = [
                row(f"tuned-confirm-b{width}-v1", profile, scope) for profile in ("pressure-default", "pressure-tuned")
            ]
            values.extend(
                [
                    f"{default['median_valid_envstep_s']:,.0f} / {tuned['median_valid_envstep_s']:,.0f}",
                    f"{100 * (tuned['median_valid_envstep_s'] / default['median_valid_envstep_s'] - 1):+.2f}%",
                ]
            )
        confirm.append([f"{width:,}", *values])
    table(
        "Separate same-allocation default/tuned confirmation:",
        ["B", "Default / tuned live", "Live gain", "Default / tuned policy", "Policy gain"],
        confirm,
    )
    full_rows = []
    for width in (32, 16384):
        for profile, label in [("pressure-default", "Pressure"), ("intrinsic", "Intrinsic")]:
            maximum, full = [row(f"output-mode-b{width}-{mode}-v1", profile) for mode in ("max", "full")]
            assert maximum["gpu_uuid"] == full["gpu_uuid"]
            full_rows.append(
                [
                    f"{width:,}",
                    label,
                    f"{maximum['median_valid_envstep_s']:,.0f} / {full['median_valid_envstep_s']:,.0f}",
                    f"{full['median_step_ms'] - maximum['median_step_ms']:+.3f}",
                    f"{100 * (full['median_step_ms'] / maximum['median_step_ms'] - 1):+.2f}%",
                    f"{(full['native_buffer_bytes'] - maximum['native_buffer_bytes']) / 1024**2:+.2f}",
                ]
            )
    table(
        "Matched max/full, n=3, native production/storage cost:",
        ["B", "Model", "Max / full env-step/s", "Extra ms/batch", "Time change", "Native MiB change"],
        full_rows,
    )
    table(
        "Actual native routes for normal max/history0 (requested auto settings preserved in JSON):",
        ["B", "Model", "Solve", "Packed block", "Rigid-post fused", "Wrench reuse", "Moment reuse"],
        route_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
