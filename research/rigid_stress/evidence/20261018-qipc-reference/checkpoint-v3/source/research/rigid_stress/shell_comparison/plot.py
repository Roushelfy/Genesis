"""Render the predeclared shell comparison figures from measured fields and tables."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .common import write_json
from .nature_mpl_style import WONG_PALETTE, apply_nature_style, figure_size, rendered_text_qa, style_axes

P = WONG_PALETTE
COLOURS = [P[k] for k in ("blue", "bluish_green", "reddish_purple", "vermillion", "orange", "black")]


def vm(tensor):
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * (dev * dev).sum(axis=(-1, -2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root, output = args.runs, args.output
    d = json.loads(args.summary.read_text())
    apply_nature_style(mpl)
    mpl.rcParams.update({"savefig.bbox": None, "font.sans-serif": ["DejaVu Sans"]})
    qa = []

    def grid(rows, cols, height):
        fig, axes = plt.subplots(rows, cols, figsize=figure_size("double", height))
        fig.subplots_adjust(
            left=0.08, right=0.92 if cols == 3 else 0.98, bottom=0.10, top=0.94, wspace=0.48, hspace=0.55
        )
        for i, ax in enumerate(axes.flat):
            style_axes(ax)
            ax.text(-0.19, 1.12, chr(97 + i), transform=ax.transAxes, fontsize=8, weight="bold", va="top")
        return fig, axes.flat

    def save(fig, stem):
        issues = rendered_text_qa(fig)
        qa.append({"figure": stem, "text_issues": issues, "size_mm": (fig.get_size_inches() * 25.4).tolist()})
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=450)
        fig.savefig(output / f"{stem}_preview.png", dpi=150)
        plt.close(fig)

    fig, axes = grid(2, 3, 140)
    a, b, c, e, f, g = axes
    with np.load(root / "prepare-loads-v2/group0/inputs/level2.npz") as z:
        cloud = z["cloud_points_m"]
        traction = np.linalg.norm(z["cloud_forces_n"], axis=1) / z["cloud_areas_m2"]
    image = a.scatter(
        np.arctan2(cloud[:, 1], cloud[:, 0]),
        cloud[:, 2] * 1000,
        c=traction / 1000,
        cmap="viridis",
        s=2,
        linewidths=0,
        rasterized=False,
    )
    fig.colorbar(image, ax=a, pad=0.02, fraction=0.06).set_label("Traction (kPa)")
    a.set(title="Common finite compression", xlabel="Azimuth (rad)", ylabel="Material z (mm)")
    with np.load(root / "analyse-prescribed-v1/group0/audit/level2-common-fields.npz") as z:
        positions = z["positions_m"]
        actual, expected = vm(z["shell_tensor_pa"][:, 1]), vm(z["reference_tensor_pa"][:, 1])
    norm = Normalize(0, max(actual.max(), expected.max()) / 1000)
    for ax, values, title in ((b, expected, "Volume FEM, tangent projection"), (c, actual, "Shell, tangent fibers")):
        image = ax.scatter(
            np.arctan2(positions[:, 1], positions[:, 0]),
            positions[:, 2] * 1000,
            c=values / 1000,
            cmap="viridis",
            norm=norm,
            s=6,
            linewidths=0,
        )
        fig.colorbar(image, ax=ax, pad=0.02, fraction=0.06).set_label("Tangent VM (kPa)")
        ax.set(title=title, xlabel="Azimuth (rad)", ylabel="Material z (mm)")
    for i, (key, name) in enumerate(
        (
            ("same_position_tangent_tensor_relative_error", "Tensor"),
            ("membrane_resultant_relative_error", "Membrane"),
            ("bending_resultant_relative_error", "Bending"),
        )
    ):
        e.plot(
            [1, 2],
            [x[key] * 100 for x in d["compatible_fields"]],
            marker=("o", "s", "^")[i],
            color=COLOURS[i],
            label=name,
        )
    e.set(
        title="Same-load shell discrepancy",
        xlabel="Mesh level",
        ylabel="Area-weighted relative error (%)",
        xticks=[1, 2],
        ylim=(0, 105),
    )
    e.legend(loc="lower right")
    for key, name, colour, marker in (
        ("independent_lame_corner_relative_error", "Volume FEM / Lame", COLOURS[0], "o"),
        ("shell_thin_tangent_tensor_relative_error", "Shell / thin sphere", COLOURS[3], "s"),
    ):
        f.plot([1, 2], [x[key] * 100 for x in d["sphere"]], color=colour, marker=marker, label=name)
    f.set(title="Curved analytic controls", xlabel="Mesh level", ylabel="Relative tensor error (%)", xticks=[1, 2])
    f.legend()
    for level in (1, 2):
        native = next(x for x in d["static_cost"] if x["configuration"] == f"native-level{level}")
        shell = next(x for x in d["static_cost"] if x["configuration"] == f"shell-level{level}")
        error = d["compatible_fields"][level - 1]["same_position_tangent_tensor_relative_error"]
        g.scatter(native["median_ms"], native["native_same_mesh_full_tensor_error"], marker="o", color=COLOURS[0])
        g.scatter(shell["median_ms"], error, marker="s", color=COLOURS[3])
        g.annotate(
            f"N{level}",
            (native["median_ms"], native["native_same_mesh_full_tensor_error"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=5,
        )
        g.annotate(
            f"S{level}",
            (shell["median_ms"], error),
            xytext=(4, -9 if level == 1 else 4),
            textcoords="offset points",
            fontsize=5,
        )
    tuned = next(x for x in d["static_cost"] if x["configuration"] == "inverse-cap1g")
    g.axvline(tuned["median_ms"], ls=":", color=COLOURS[1], label="N2, 1 GiB inverse cap")
    g.set(
        xscale="log",
        yscale="log",
        xlabel="Prescribed full field (ms)",
        ylabel="Same-mesh tensor discrepancy",
        title="Cost / discrepancy (different norms)",
    )
    g.legend(loc="center right")
    save(fig, "mechanics")

    fig, axes = grid(2, 3, 140)
    a, b, c, e, f, g = axes
    audit = json.loads((root / "contact-audit-v1/group0/audit/result.json").read_text())
    for env, label, colour, marker in (
        (0, "Interior", COLOURS[3], "o"),
        (1, "Vertex", COLOURS[0], "s"),
        (2, "Edge", COLOURS[4], "^"),
    ):
        samples = [s for s in audit["geometry"][0]["samples"] if s["env"] == env]
        a.plot(
            [(s["step"] + 1) for s in samples],
            [s["finite_thickness_overlap_m"] * 1000 for s in samples],
            color=colour,
            marker=marker,
            markersize=2,
            label=label,
        )
    a.axhline(0, color="0.5", lw=0.5, ls=":")
    a.set(title="Deformed coarse-sheet geometry", xlabel="Time (ms)", ylabel="Finite-thickness overlap (mm)")
    a.legend()
    with np.load(root / "diagnostic-controls-v2/group0/drop-n2/trajectory.npz") as z:
        impulse = z["impulses"].sum(axis=1)
        for env, label, colour in ((0, "Interior", COLOURS[3]), (1, "Vertex", COLOURS[0]), (2, "Edge", COLOURS[4])):
            b.plot(np.arange(200) + 1, np.linalg.norm(impulse[:, env], axis=1) * 1e3, label=label, color=colour)
    b.set(title="Actual coarse contact impulses", xlabel="Time (ms)", ylabel="Impulse magnitude (mN s)")
    b.legend()
    for path, label, colour, ls in (
        ("group0/drop-n16", "1 substep / cap50", COLOURS[3], "-"),
        ("group0/drop-n16-pcg200", "1 substep / cap200", COLOURS[4], "--"),
        ("group1/drop-n16-sub4", "4 substeps / cap50", COLOURS[0], "-"),
    ):
        with np.load(root / f"diagnostic-controls-v2/{path}/trajectory.npz") as z:
            c.plot(np.arange(200) + 1, z["peak_pa"].max(axis=1) / 1e6, label=label, color=colour, ls=ls)
            e.plot(np.arange(200) + 1, z["residuals"][..., 1].max(axis=1) / 1e-4, color=colour, ls=ls)
    c.set(title="Fine-sheet raw stress", xlabel="Time (ms)", ylabel="Raw fiber peak (MPa)", yscale="log")
    c.legend()
    e.axhline(1, ls=":", color="0.3")
    e.set(title="True linear-system residual", xlabel="Time (ms)", ylabel="Block-Jacobi norm / threshold")
    with np.load(root / "contact-audit-v1/group0/audit/impulse-energy-ledger.npz") as z:
        for env in range(3):
            f.plot(np.arange(200) + 1, z["total_energy_j"][:, env], color=COLOURS[env], label=f"Offset {env}")
        f.axhline(z["initial_total_energy_j"][0], color="0.3", ls=":")
    f.set(
        title="Free drop: mechanical energy",
        xlabel="Time (ms)",
        ylabel="Elastic + kinetic + gravity (J)",
        yscale="symlog",
        ylim=(-0.01, 2000),
    )
    f.set_yscale("symlog", linthresh=0.001)
    f.legend()
    for label, colour in (("press-mu0", COLOURS[0]), ("press-mu06", COLOURS[3])):
        with np.load(root / f"press-controls-v3/group0/{label}/trajectory.npz") as z:
            g.plot(
                np.arange(200) + 1,
                z["impulses"].sum(axis=(1, 2))[:, 0] / 0.001,
                color=colour,
                label="mu = 0" if label == "press-mu0" else "mu = 0.6",
            )
    g.axvspan(150, 200, color="0.93", zorder=0)
    g.set(title="Prescribed press / slide / release", xlabel="Time (ms)", ylabel="Total shell x impulse / dt (N)")
    g.legend()
    save(fig, "contact")

    fig, axes = grid(2, 2, 138)
    a, b, c, e = axes
    configs = [
        ("full-shell-history-v1/group0/default32-sub1", "S32, 1 substep", COLOURS[3], ":"),
        ("peak-traces-v2/group1/shell32-sub10", "S32, 10 substeps", COLOURS[4], "-"),
        ("peak-traces-v2/group1/shell64-sub10-pcg200", "S64, 10 substeps, cap200", COLOURS[5], "--"),
        ("peak-traces-v2/group0/pressure", "Pressure", COLOURS[0], "-"),
        ("peak-traces-v2/group0/intrinsic", "Source prior", COLOURS[2], "-."),
    ]
    for path, label, colour, ls in configs:
        with np.load(root / path / "trajectory-repeat0.npz") as z:
            a.plot(
                (np.arange(900) + 1) * 0.01,
                z["peaks_pa"].max(axis=1),
                color=colour,
                ls=ls,
                label=label,
            )
    a.set(
        title="Complete grasp history, raw output",
        xlabel="Time (s)",
        ylabel="Maximum of 8 full-field peaks (Pa)",
        yscale="log",
    )
    a.legend(loc="upper left", ncol=2)
    for path, label, colour in (
        ("full-shell-history-v1/group0/default32-sub10", "Initial process", COLOURS[4]),
        ("fresh-restarts-v1/group0/fresh-restart1", "Fresh process 1", COLOURS[0]),
        ("fresh-restarts-v1/group0/fresh-restart2", "Fresh process 2", COLOURS[3]),
    ):
        with np.load(root / path / "trajectory-repeat0.npz") as z:
            b.plot((np.arange(900) + 1) * 0.01, z["peaks_pa"].max(axis=1) / 1e9, color=colour, label=label)
    b.set(title="Identical inputs, S32 / 10 substeps", xlabel="Time (s)", ylabel="Raw fiber peak (GPa)")
    b.legend()
    modes = ["public", "shell-dv-only", "shell-scratch"]
    for i, mode in enumerate(modes):
        values = [r["tensor_relative_difference"] for r in d["cache_ablation"] if r["mode"] == mode]
        c.scatter(np.full(len(values), i), values, marker=("o", "s", "^")[i], color=COLOURS[i])
    c.set(
        title="One-step restore ablation (FP64)",
        ylabel="Tensor relative difference",
        yscale="log",
        xticks=range(3),
        xticklabels=["Public state", "+ shell dv", "+ all shell scratch"],
        xlim=(-0.5, 2.5),
    )
    for config, label, colour, marker in (
        ("pressure", "Pressure", COLOURS[0], "o"),
        ("intrinsic", "Source prior", COLOURS[2], "^"),
        ("shell32-sub10", "S32 / 10", COLOURS[4], "s"),
        ("shell64-sub10-pcg200", "S64 / 10 / cap200", COLOURS[5], "d"),
    ):
        values = [r["raw_peak_pa"] for r in d["raw_peaks"] if r["configuration"] == config]
        e.plot(np.arange(8), values, marker=marker, color=colour, label=label)
    e.set(
        title="Retained raw maxima and peak snapshots",
        xlabel="Condition index",
        ylabel="Raw peak (Pa)",
        yscale="log",
        xticks=range(8),
    )
    e.legend(ncol=2)
    save(fig, "consistency")

    fig, axes = grid(2, 2, 144)
    a, b, c, e = axes
    batches = [8, 128, 1024, 4096]
    for i, (config, label, cohort) in enumerate(
        (
            ("native-pressure", "Pressure", "validated-native-v2"),
            ("native-pressure-tuned", "Pressure tuned", "validated-native-v2"),
            ("native-intrinsic", "Source prior", "validated-native-v2"),
            ("external-shell-fp32", "Shell32 diagnostic", "paired-normal-v1"),
            ("external-shell-fp64", "Shell64 diagnostic", "paired-normal-v1"),
        )
    ):
        values = [
            [
                x["env_step_s"]
                for x in d["rates"]
                if x["configuration"] == config and x["batch"] == batch and x["cohort"] == cohort
            ]
            for batch in batches
        ]
        median = np.median(values, axis=1)
        a.errorbar(
            batches,
            median,
            yerr=[median - np.min(values, axis=1), np.max(values, axis=1) - median],
            color=COLOURS[i],
            marker=("o", "s", "^", "d", "v")[i],
            ls="--" if i > 2 else "-",
            label=label,
        )
    a.set(
        xscale="log",
        yscale="log",
        xlabel="Batch environments",
        ylabel="Aggregate env-step/s",
        title="Complete full-field scene steps",
    )
    a.legend()
    for i, config in enumerate(("native-pressure", "native-pressure-tuned", "native-intrinsic")):
        overhead = []
        for batch in batches:
            values = [
                x["batch_step_ms"]
                for x in d["rates"]
                if x["configuration"] == config and x["batch"] == batch and x["cohort"] == "validated-native-v2"
            ]
            base = [
                x["batch_step_ms"]
                for x in d["rates"]
                if x["configuration"] == "native-rigid" and x["batch"] == batch and x["cohort"] == "validated-native-v2"
            ]
            overhead.append(np.median(values) - np.median(base))
        b.plot(
            batches,
            overhead,
            color=COLOURS[i],
            marker=("o", "s", "^")[i],
            label=("Pressure", "Pressure tuned", "Source prior")[i],
        )
    b.set(
        xscale="log",
        xlabel="Batch environments",
        ylabel="Increment over paired rigid (ms)",
        title="Native full-field marginal latency",
    )
    b.legend()
    mem_order = [
        "memory-external-rigid-fp64",
        "memory-external-shell-fp32",
        "memory-external-shell-fp64",
        "memory-native-rigid",
        "memory-native-pressure",
        "memory-native-pressure-tuned",
        "memory-native-intrinsic",
    ]
    # The exact directory prefix is retained by each executed task.
    memory_by_suffix = {row["configuration"].removeprefix("memory-"): row for row in d["memory"]}
    labels = ["Ext R64", "S32", "S64", "Nat R", "P", "Pt", "I"]
    memory_values = [memory_by_suffix[name.removeprefix("memory-")]["peak_mib"] / 1024 for name in mem_order]
    c.bar(np.arange(7), memory_values, color=["0.5", COLOURS[3], COLOURS[4], "0.7", *COLOURS[:3]])
    c.set(title="B4096 sampled process peak", ylabel="GPU memory (GiB)", xticks=range(7), xticklabels=labels)
    selected = [
        "trace-external-shell-fp32-summary",
        "trace-external-shell-fp64-summary",
        "trace-native-rigid-summary",
        "trace-native-pressure-summary",
        "trace-native-pressure-tuned-summary",
        "trace-native-intrinsic-summary",
    ]
    profile_by_name = {row["configuration"]: row for row in d["profiles"]}
    bottom = np.zeros(len(selected))
    groups = [
        "rigid/control/other",
        "shell Hessian products",
        "shell PCG/preconditioner",
        "shell coarse",
        "shell forces",
        "coupling",
        "shell integrate",
        "shell full output",
        "native stress/mapping",
    ]
    colours = ["0.7", COLOURS[3], COLOURS[4], COLOURS[2], P["sky_blue"], COLOURS[1], "0.4", P["black"], COLOURS[0]]
    for group, colour in zip(groups, colours, strict=True):
        values = [profile_by_name[name]["gpu_groups"].get(group, 0) for name in selected]
        e.bar(np.arange(len(selected)), values, bottom=bottom, color=colour, label=group)
        bottom += values
    e.set(
        title="B128 moving trace, GPU kernel sums",
        ylabel="Kernel sum (ms / scene step)",
        xticks=range(6),
        xticklabels=["S32", "S64", "R", "P", "Pt", "I"],
    )
    e.legend(loc="upper right", fontsize=5, ncol=1)
    save(fig, "performance")
    write_json(output / "rendered-text-qa.json", qa)


if __name__ == "__main__":
    main()
