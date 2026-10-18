"""Render the predeclared comparison figures from actual new worker results."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm
from scipy.spatial.transform import Rotation

from ..shell_comparison.nature_mpl_style import (
    WONG_PALETTE,
    apply_nature_style,
    figure_size,
    rendered_text_qa,
    style_axes,
)
from .artifacts import file_id, runtime_identity, write_json


P = WONG_PALETTE
COLOURS = {"reference": P["black"], "dpr": P["blue"], "scr": P["bluish_green"], "shell": P["vermillion"]}
LABELS = {"reference": "QIPC candidate", "dpr": "DPR", "scr": "SCR", "shell": "Shell stock"}


def journal(folder):
    return [json.loads(line) for line in (folder / "steps.jsonl").read_text().splitlines()]


def vm(tensor):
    dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * (dev * dev).sum(axis=(-2, -1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    apply_nature_style(mpl)
    mpl.rcParams.update({"savefig.bbox": None, "font.sans-serif": ["DejaVu Sans"]})
    root, output = args.runs, args.output
    sources = [file_id(args.cost), file_id(args.comparison)]
    qa = []
    figure_data = {}

    def grid(rows=2, cols=2, height=132):
        fig, axes = plt.subplots(rows, cols, figsize=figure_size("double", height), squeeze=False)
        fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.94, wspace=0.4, hspace=0.58)
        for i, ax in enumerate(axes.flat):
            style_axes(ax)
            ax.text(-0.20, 1.15, chr(97 + i), transform=ax.transAxes, weight="bold", fontsize=8, va="top")
        return fig, axes

    def save(fig, stem):
        for axis_index, ax in enumerate(fig.axes):
            for line_index, line in enumerate(ax.lines):
                figure_data[f"{stem}_axis{axis_index}_line{line_index}_xy"] = line.get_xydata()
            for collection_index, collection in enumerate(ax.collections):
                if collection.get_array() is not None:
                    figure_data[f"{stem}_axis{axis_index}_collection{collection_index}_values"] = np.asarray(
                        collection.get_array()
                    )
                    figure_data[f"{stem}_axis{axis_index}_collection{collection_index}_points"] = np.asarray(
                        collection.get_offsets()
                    )
        qa.append(
            {"figure": stem, "text_issues": rendered_text_qa(fig), "size_mm": (fig.get_size_inches() * 25.4).tolist()}
        )
        for suffix in ("svg", "pdf", "png"):
            fig.savefig(output / f"{stem}.{suffix}", dpi=450)
        fig.savefig(output / f"{stem}_preview.png", dpi=150)
        plt.close(fig)

    cost = json.loads(args.cost.read_text())
    rates = [
        row for row in cost["configurations"] if row.get("scope") == "normal-full-field" and row.get("raw_repeats")
    ]
    fig, axes = grid()
    for substeps, ax in zip((1, 10), axes[0]):
        for model, label, color, marker in (
            ("rigid", "Rigid baseline", P["black"], "s"),
            ("pressure", "DPR", P["blue"], "o"),
            ("pressure-tuned", "DPR tuned", P["reddish_purple"], "^"),
            ("intrinsic", "SCR", P["bluish_green"], "D"),
        ):
            selected = sorted(
                [
                    row
                    for row in rates
                    if row["args"]["model"] == model
                    and row["args"]["substeps"] == substeps
                    and row["args"]["output_mode"] == "full"
                ],
                key=lambda row: row["args"]["envs"],
            )
            b = [row["args"]["envs"] for row in selected]
            y = np.array([row["median_aggregate_env_steps_per_second"] for row in selected])
            if not len(y):
                continue
            err = np.array(
                [
                    [row["minimum_aggregate_env_steps_per_second"], row["maximum_aggregate_env_steps_per_second"]]
                    for row in selected
                ]
            )
            ax.errorbar(
                b, y, yerr=np.stack((y - err[:, 0], err[:, 1] - y)), color=color, marker=marker, label=label, capsize=2
            )
        for name, label, color, marker, steps in (
            ("default-full", "Shell default", P["vermillion"], "v", 1),
            ("fine-tuned-full", "Shell fine", P["orange"], "x", 10),
        ):
            if substeps != steps:
                continue
            selected = sorted(
                [row for row in rates if row["configuration"].startswith(name)], key=lambda row: row["args"]["envs"]
            )
            ax.plot(
                [row["args"]["envs"] for row in selected],
                [row["median_aggregate_env_steps_per_second"] for row in selected],
                color=color,
                marker=marker,
                linestyle="--",
                label=label,
            )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="Environments (B)",
            ylabel="Aggregate env-step/s",
            title=f"Full fields; {10 // substeps} ms physical step",
        )
        ax.legend(loc="upper left", ncol=2, columnspacing=0.7, handlelength=1.4)
    ax = axes[1, 0]
    for model, label, color in (
        ("pressure", "DPR", P["blue"]),
        ("pressure-tuned", "DPR tuned", P["reddish_purple"]),
        ("intrinsic", "SCR", P["bluish_green"]),
    ):
        selected = sorted(
            [
                row
                for row in rates
                if row["args"]["model"] == model
                and row["args"]["substeps"] == 10
                and row["args"]["output_mode"] == "full"
            ],
            key=lambda row: row["args"]["envs"],
        )
        values = []
        for row in selected:
            baseline = next(
                value
                for value in rates
                if value["args"]["model"] == "rigid"
                and value["args"]["substeps"] == 10
                and value["args"]["envs"] == row["args"]["envs"]
            )
            values.append(row["median_batch_step_ms"] - baseline["median_batch_step_ms"])
        ax.plot([row["args"]["envs"] for row in selected], values, "o-", color=color, label=label)
    ax.set(
        xscale="log",
        xlabel="Environments (B)",
        ylabel="Marginal scene latency (ms)",
        title="Full fields; ten physical substeps",
    )
    ax.legend()
    ax = axes[1, 1]
    memory = [
        row
        for row in cost["configurations"]
        if row["campaign"].endswith("fp64-memory-v1") and row.get("process_memory")
    ]
    for model, label, color in (
        ("pressure", "DPR", P["blue"]),
        ("intrinsic", "SCR", P["bluish_green"]),
        ("shell", "Shell fine", P["orange"]),
    ):
        selected = sorted(
            [
                row
                for row in memory
                if row["args"]["model"] == model and (model != "shell" or row["configuration"].startswith("fine-tuned"))
            ],
            key=lambda row: row["args"]["envs"],
        )
        if selected:
            ax.plot(
                [row["args"]["envs"] for row in selected],
                [row["process_memory"]["peak_mib"] / 1024 for row in selected],
                "o-",
                color=color,
                label=label,
            )
    ax.set(
        xscale="log",
        xlabel="Environments (B)",
        ylabel="Sampled process memory (GiB)",
        title="Intrusive memory runs; full fields",
    )
    ax.legend()
    save(fig, "new-full-field-cost")

    folders = {
        "reference": root / "roundoff-prescribed-v1/group0/snh",
        "dpr": root / "native-prescribed-v1/group0/dpr-1e10",
        "scr": root / "native-prescribed-v1/group2/scr-1e10",
        "shell": root / "shell-prescribed-v1/group0/stock",
    }
    comparison = json.loads(args.comparison.read_text())
    fig, axes = grid()
    for name, folder in folders.items():
        sources.extend([file_id(folder / "result.json"), file_id(folder / "steps.jsonl")])
        history = journal(folder)
        times = np.array([row["time_s"] for row in history])
        key = "raw_vm_peak_last_physical_substep_pa" if name in ("dpr", "scr") else "raw_vm_peak_pa"
        peak = np.array([row[key] if row[key] is not None else np.nan for row in history]) / 1e6
        axes[0, 1].plot(times, np.where(peak > 0, peak, np.nan), color=COLOURS[name], label=LABELS[name], linewidth=0.7)
        with np.load(folder / "rest.npz") as rest:
            data = {key: rest[key] for key in rest.files}
        if name in ("dpr", "scr"):
            start = json.loads((folder / "start.json").read_text())
            urdf = Path(start["elastic_mesh"]["path"]).parent / "egg_shell.urdf"
            origin = ElementTree.parse(urdf).find("./link/inertial/origin")
            native_com = np.fromstring(origin.attrib["xyz"], sep=" ")
            assert native_com.shape == (3,)
            sources.append(file_id(urdf))
        cm_times, cm = [], []
        for path in sorted((folder / "frames").glob("step*.npz")):
            with np.load(path) as frame:
                if name == "reference":
                    position = data["masses_kg"] @ frame["positions_m"] / data["masses_kg"].sum()
                elif name == "shell":
                    if "vertex_masses_kg" in data:
                        masses = data["vertex_masses_kg"]
                    else:
                        tri = data["vertices_m"][data["faces"]]
                        area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2
                        masses = np.bincount(
                            data["faces"].reshape(-1), weights=np.repeat(area / 3, 3), minlength=len(data["vertices_m"])
                        )
                    position = masses @ frame["vertices_world_m"] / masses.sum()
                else:
                    position = frame["link_position_m"] + Rotation.from_quat(
                        frame["link_quaternion_wxyz"][[1, 2, 3, 0]]
                    ).apply(native_com)
                cm_times.append(frame["time_s"].item())
                cm.append(position[2] * 1000)
        axes[0, 0].plot(cm_times, cm, color=COLOURS[name], label=LABELS[name], linewidth=0.8)
    axes[0, 0].set(xlabel="Time (s)", ylabel="Mass-weighted COM height (mm)")
    axes[0, 0].legend(ncol=2)
    axes[0, 1].set(yscale="log", xlabel="Time (s)", ylabel="Raw global VM peak (MPa)")
    for name in ("dpr", "scr", "shell"):
        for key, ax, linestyle in (
            ("tangent_field", axes[1, 0], "-"),
            ("inner_surface_tangent", axes[1, 1], "--"),
            ("outer_surface_tangent", axes[1, 1], "-"),
        ):
            times, values = [], []
            for row in comparison["history"]:
                if not row["all_fields_finite"]:
                    continue
                entry = next(value for value in row["errors"] if value["method"] == name)
                times.append(row["time_s"])
                values.append(entry[key]["relative_tensor_error"])
            ax.plot(
                times,
                [value if value is not None and value > 0 else np.nan for value in values],
                color=COLOURS[name],
                linestyle=linestyle,
                linewidth=0.8,
                label=LABELS[name] if key != "inner_surface_tangent" else None,
            )
    axes[1, 0].set(yscale="log", xlabel="Time (s)", ylabel="Relative tangent tensor discrepancy")
    axes[1, 1].set(
        yscale="log", xlabel="Time (s)", ylabel="Relative surface tensor discrepancy", title="Inner dashed; outer solid"
    )
    axes[1, 0].legend(ncol=3)
    save(fig, "complete-prescribed-diagnostic")

    selected = [args.comparison.parent / f"common-step{tick:06d}.npz" for tick in (119, 419)]
    snapshots = []
    for path in selected:
        with np.load(path) as saved:
            snapshots.append({name: saved[name] for name in saved.files})
        sources.append(file_id(path))
    vmax = max(vm(saved[f"{name}_surface_tangent_pa"]).max() / 1000 for saved in snapshots for name in folders)
    fig, axes = grid(rows=2, cols=4, height=104)
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.15, top=0.91, wspace=0.38, hspace=0.5)
    norm = SymLogNorm(linthresh=0.001, vmin=0, vmax=vmax)
    for row, saved in enumerate(snapshots):
        mask = saved["surface_side"] == 1
        position = saved["surface_points_m"][mask]
        for col, name in enumerate(folders):
            ax = axes[row, col]
            image = ax.scatter(
                np.arctan2(position[:, 1], position[:, 0]),
                position[:, 2] * 1000,
                c=vm(saved[f"{name}_surface_tangent_pa"][mask]) / 1000,
                cmap="viridis",
                norm=norm,
                s=2,
                linewidths=0,
            )
            ax.set(xlim=(-np.pi, np.pi), xticks=(-3, 0, 3))
            if row == 0:
                ax.set_title(LABELS[name])
            if row == 1:
                ax.set_xlabel("Azimuth (rad)")
            if col == 0:
                ax.set_ylabel(f"Material z (mm)\n{saved['time_s'].item():.1f} s")
            else:
                ax.set_yticklabels([])
    cax = fig.add_axes((0.93, 0.20, 0.015, 0.59))
    fig.colorbar(image, cax=cax).set_label("Outer tangent VM (kPa); symlog")
    save(fig, "shared-outer-tangent-maps")
    np.savez_compressed(output / "figure-data.npz", **figure_data)
    write_json(output / "rendered-text-qa.json", qa)
    write_json(
        output / "result.json",
        {
            "kind": "predeclared-qipc-comparison-figures",
            "identity": runtime_identity(),
            "sources": sources,
            "reference_qualified": False,
            "scope": "Actual full histories and measured new cost. Common maps retain all displayed values on one shared symmetric-log scale; raw global peaks remain separate. Diagnostic discrepancies are not an accuracy ranking. Missing completed cost outputs are not imputed.",
        },
    )


if __name__ == "__main__":
    main()
