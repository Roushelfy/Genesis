"""Render qualified control curves and matched shell fields, preserving failed cases."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from .spherical_shell_validation import analytic_shell


def file_identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_style(path):
    spec = importlib.util.spec_from_file_location("physical_figure_style", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_nature_style(matplotlib)
    return module


def export(fig, stem, style):
    preview = style.save_preview(fig, stem.with_name(stem.name + "_preview").with_suffix(".png"))
    issues = style.rendered_text_qa(fig)
    if issues:
        raise RuntimeError(f"Rendered text QA failed for{preview}: {issues}")
    paths = style.save_nature_figure(fig, stem)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shell-run", type=Path, required=True)
    parser.add_argument("--contact-run", type=Path, required=True)
    parser.add_argument("--style-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    style = load_style(args.style_module)
    inputs = [file_identity(args.style_module)]
    shell_summary = args.shell_run / "sweep/summary.json"
    shell = json.loads(shell_summary.read_text())["results"]
    inputs.append(file_identity(shell_summary))
    assert len({r["mesh"]["mesh_sha256"] for r in shell}) >= 3
    contacts, failures = [], []
    for returncode in sorted(args.contact_run.glob("n*-d*.returncode")):
        inputs.append(file_identity(returncode))
        name = returncode.stem
        resolution, dhat = name.split("-d")
        if int(returncode.read_text()) != 0:
            failures.append({"name": name, "resolution": int(resolution[1:]), "dhat_m": float(dhat)})
            continue
        path = args.contact_run / name / "result.json"
        row = json.loads(path.read_text())
        assert row["complete_free_residual_N"] <= 1e-9
        row["force_error_percent"] = 100 * abs(
            row["parts"][0]["contact_force_N"][2] / row["analytic_frictionless_zero_gap_force_N"] + 1
        )
        row["raw_peak_Pa"] = max(part["peak_Pa"] for part in row["parts"])
        row["peak_error_percent"] = 100 * abs(row["raw_peak_Pa"] / 5000 - 1)
        contacts.append(row)
        inputs.append(file_identity(path))
    assert len(contacts) >= 3
    curves = args.output / "curves.csv"
    with curves.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "resolution",
                "tetrahedra",
                "dhat_m",
                "raw_peak_Pa",
                "force_error_percent",
                "peak_error_percent",
                "energy_error_percent",
                "tensor_probe_error_percent",
            )
        )
        for row in shell:
            writer.writerow(
                (
                    "shell",
                    row["mesh"]["mesh_size"],
                    row["mesh"]["tetrahedra"],
                    "",
                    row["peak_Pa"],
                    "",
                    100 * row["peak_relative_error"],
                    100 * row["energy_relative_error"],
                    100 * row["common_probe_tensor_relative_error"],
                )
            )
        for row in contacts:
            writer.writerow(
                (
                    "blocks",
                    row["resolution"],
                    row["parts"][0]["tetrahedra"],
                    row["dhat_m"],
                    row["raw_peak_Pa"],
                    row["force_error_percent"],
                    row["peak_error_percent"],
                    "",
                    "",
                )
            )

    fig, axes = plt.subplots(1, 3, figsize=style.figure_size("double", 68))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.83, wspace=0.48)
    colours = (style.WONG_PALETTE["blue"], style.WONG_PALETTE["orange"], style.WONG_PALETTE["bluish_green"])
    for key, label, colour, marker in zip(
        ("peak_relative_error", "energy_relative_error", "common_probe_tensor_relative_error"),
        ("Raw global peak", "Energy", "Full tensor probes"),
        colours,
        ("o", "s", "^"),
    ):
        axes[0].plot(
            [r["mesh"]["tetrahedra"] for r in shell],
            [100 * r[key] for r in shell],
            marker=marker,
            color=colour,
            label=label,
        )
    axes[0].set_xscale("log")
    counts = np.array([row["mesh"]["tetrahedra"] for row in shell])
    ticks = counts[[0, len(counts) // 2, -1]]
    axes[0].set_xticks(ticks, labels=[f"{value / 1000:.2g}k" for value in ticks])
    axes[0].xaxis.set_minor_locator(NullLocator())
    axes[0].set_ylim(0, 26)
    axes[0].set_yticks((0, 5, 10, 15, 20, 25))
    axes[0].set_xlabel("Tetrahedra")
    axes[0].set_ylabel("Shell error vs. analytic (%)")
    axes[0].legend(loc="upper right")
    for resolution, colour, marker in zip(sorted({r["resolution"] for r in contacts}), colours, ("o", "s", "^")):
        rows = sorted((r for r in contacts if r["resolution"] == resolution), key=lambda r: r["dhat_m"], reverse=True)
        for ax, quantity in zip(axes[1:], ("force_error_percent", "peak_error_percent")):
            ax.plot(
                [r["dhat_m"] * 1e6 for r in rows],
                [r[quantity] for r in rows],
                color=colour,
                marker=marker,
                label=f"{resolution} nodes/axis",
            )
    for ax in axes[1:]:
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks((1.25, 2.5, 5, 10), labels=("1.25", "2.5", "5", "10"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylim(0, 24)
        ax.set_yticks((0, 5, 10, 15, 20))
        ax.set_xlabel("Contact activation distance (µm)")
        for failure in failures:
            ax.plot(
                failure["dhat_m"] * 1e6,
                0.025,
                marker="x",
                color=style.WONG_PALETTE["vermillion"],
                linestyle="none",
                transform=ax.get_xaxis_transform(),
                clip_on=False,
            )
        if failures:
            ax.plot([], [], marker="x", color=style.WONG_PALETTE["vermillion"], linestyle="none", label="Failed solve")
        ax.legend(loc="upper right")
    axes[1].set_ylabel("Normal force error (%)")
    axes[2].set_ylabel("Raw global peak error (%)")
    for i, ax in enumerate(axes):
        ax.axhline(3, color=style.WONG_PALETTE["grey"], linestyle="--", linewidth=0.65)
        style.style_axes(ax)
        style.add_panel_label(ax, "abc"[i], x=-0.2, y=1.17)
    paths = {"controls": export(fig, args.output / "physical_controls", style)}

    finest = shell[-1]
    case = args.shell_run / "sweep" / f"h{finest['mesh']['mesh_size']:g}"
    with np.load(case / "mesh.npz") as saved:
        vertices, tets = saved["vertices"], saved["tetrahedra"]
    with np.load(case / "field.npz") as saved:
        tensor = saved["tensor_Pa"]
    assert hashlib.sha256((case / "field.npz").read_bytes()).hexdigest() == finest["field_sha256"]
    centres = vertices[tets].mean(axis=1)
    section = abs(centres[:, 1]) < 0.0006
    positions = centres[section]
    sampled = tensor[section].mean(axis=1)
    dev = sampled - np.trace(sampled, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    actual = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    expected = analytic_shell(positions, 0.008, 0.012, 1e4, 1e10, 0.3)[2]
    np.savez_compressed(
        args.output / "shell_slice.npz",
        positions_m=positions,
        actual_vm_Pa=actual,
        analytic_vm_Pa=expected,
        raw_global_peak_Pa=finest["peak_Pa"],
        raw_peak_position_m=finest["peak_position_m"],
    )
    fig, axes = plt.subplots(1, 2, figsize=style.figure_size("double", 82))
    fig.subplots_adjust(left=0.08, right=0.83, bottom=0.18, top=0.83, wspace=0.32)
    limit = np.ceil(max(finest["peak_Pa"], finest["exact_peak_Pa"]) / 1000)
    for i, (ax, values, title) in enumerate(zip(axes, (expected, actual), ("Analytic Lamé", "Independent P2 FEM"))):
        plotted = ax.scatter(
            positions[:, 0] * 1000,
            positions[:, 2] * 1000,
            c=values / 1000,
            s=5,
            marker="s",
            cmap="cividis",
            vmin=0,
            vmax=limit,
            linewidths=0,
        )
        ax.set(xlabel="x (mm)", ylabel="z (mm)", title=title, xlim=(-12.5, 12.5), ylim=(-12.5, 12.5))
        ax.set_aspect("equal")
        ax.set_xticks((-10, 0, 10))
        ax.set_yticks((-10, 0, 10))
        style.style_axes(ax)
        style.add_panel_label(ax, "ab"[i], x=-0.17, y=1.16)
    peak = np.asarray(finest["peak_position_m"]) * 1000
    axes[1].scatter(peak[0], peak[2], marker="*", s=30, facecolors="none", edgecolors="black", linewidths=0.65)
    cax = fig.add_axes((0.875, 0.24, 0.022, 0.5))
    fig.colorbar(plotted, cax=cax, label="von Mises stress (kPa)", ticks=np.arange(0, limit + 1, 5))
    paths["matched_shell_slice"] = export(fig, args.output / "shell_matched_slice", style)
    inputs.extend(file_identity(case / name) for name in ("mesh.npz", "field.npz", "result.json"))
    report = {
        "scope": "Initial physical controls; full load-model and deformable-grasp acceptance remains pending",
        "inputs": inputs,
        "exports": paths,
        "failed_contact_cases": failures,
        "figure_contract": {
            "controls_claim": "Scalar peak convergence can precede full tensor convergence; contact regularization changes force",
            "controls_panels": [
                "Shell peak, energy and full-tensor probe errors",
                "Block force errors",
                "Block raw peak errors",
            ],
            "slice_claim": "Matched physical stress sampling with identical colour limits retains visible spatial differences",
            "slice_sampling": "Exact per-element linear stress at tetrahedral centroids in fixed |y|<0.6mm slab; no cross-element smoothing",
            "peak_marker": "Projected location of the full3D unaveraged corner maximum; may lie outside the displayed slab",
        },
        "captions": {
            "controls": "Full3D independent scikit-fem P2 shell versus Lamé, and supported deformable-block IPC controls. "
            "Dashed lines show the initially declared3% target. Red crosses at the activation-distance axis mark "
            "failed solves with no accepted error value. Contact force/peak errors use0.5N/5000Pa analytic zero-gap values.",
            "slice": f"Common centroid samples within |y|<0.6mm at h={finest['mesh']['mesh_size']:g}; units and colour limits match. "
            f"The raw full3D global corner peak is{finest['peak_Pa'] / 1000:.5g}kPa, versus{finest['exact_peak_Pa'] / 1000:.5g}kPa analytic. "
            "The open star projects its location; slice colours do not replace that unsmoothed maximum.",
        },
        "alt_text": "Three convergence curves compare independent shell and contact errors. A paired shell slice compares analytic "
        "and FEM von Mises stress with the same colour scale and a marked projection of the global raw FEM hotspot.",
    }
    (args.output / "figure_contract.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
