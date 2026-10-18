"""Plot complete reference sensitivity and signed diagnostics from explicit source receipts."""

import argparse
import csv
import json
from pathlib import Path
from xml.etree import ElementTree

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ..shell_comparison.nature_mpl_style import (
    WONG_PALETTE,
    apply_nature_style,
    figure_size,
    rendered_text_qa,
    style_axes,
)
from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--fixed-wall", type=Path, required=True)
    parser.add_argument("--friction", type=Path, required=True)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = []

    def read(path):
        sources.append(file_id(path))
        return json.loads(path.read_text())

    pairs = read(args.pairs)["comparisons"]
    fixed = read(args.fixed_wall)["comparisons"][0]
    friction = read(args.friction)["comparisons"][0]
    reactions = read(args.reactions)["comparisons"]
    audits = read(args.diagnostics)["histories"]

    def select(rows, key, ending):
        found = [row for row in rows if row[key].endswith(ending)]
        assert len(found) == 1, (key, ending, len(found))
        return found[0]

    contrasts = [
        ("Wall2 / wall4", select(pairs, "candidate_folder", "base1-surface1-wall2")),
        ("Surface2 / surface1", select(pairs, "candidate_folder", "base1-surface2-wall4")),
        ("Half timestep", select(pairs, "candidate_folder", "dt-half")),
        (r"Friction $\epsilon_v/2$", friction),
        ("Surface3/2 (wall2)", fixed),
    ]
    field = np.array(
        [row["complete_time_integrated_common_metrics"]["common_tangent"]["relative_error"] for _, row in contrasts]
    )
    for _, row in contrasts:
        assert row["common_times"] == 600 and not row["missing_candidate_ticks"]
    reaction_suffixes = ["base1-surface1-wall2", "base1-surface2-wall4", "dt-half", "friction-half"]
    reaction_labels = [label for label, _ in contrasts[:4]]
    reaction_values = []
    for suffix in reaction_suffixes:
        row = next(row for row in reactions if Path(row["source"]["path"]).parent.name == suffix)
        assert not row["missing_candidate_ticks"]
        metrics = row["complete_time_integrated_finger_metrics"]
        reaction_values.append([metrics[key]["relative_error"] for key in ("finger_force", "finger_moment")])
    reaction_values = np.array(reaction_values)

    apply_nature_style(mpl)
    mpl.rcParams.update({"savefig.bbox": None, "font.sans-serif": ["DejaVu Sans"]})
    fig, axes = plt.subplots(2, 2, figsize=figure_size("double", 148))
    fig.subplots_adjust(left=0.115, right=0.98, bottom=0.09, top=0.94, wspace=0.50, hspace=0.53)
    palette = WONG_PALETTE
    for index, ax in enumerate(axes.flat):
        style_axes(ax)
        ax.text(-0.24, 1.12, chr(97 + index), transform=ax.transAxes, fontsize=8, weight="bold", va="top")
    ax = axes[0, 0]
    positions = np.arange(len(contrasts))
    ax.barh(positions, field * 100, height=0.56, color=palette["blue"])
    ax.set_yticks(positions, [label for label, _ in contrasts], fontsize=5)
    ax.invert_yaxis()
    ax.axvline(5, color=palette["black"], linestyle=":", linewidth=0.8, label="5% field target")
    ax.set(xlabel="Tangential field contrast (%)", title="Complete reference-field sensitivity", xlim=(0, 50))
    ax.legend(loc="lower right", frameon=False)

    ax = axes[0, 1]
    positions = np.arange(len(reaction_labels))
    for index, (label, color, marker) in enumerate(
        (("Finger force", palette["blue"], "o"), ("Finger moment", palette["vermillion"], "s"))
    ):
        ax.plot(
            reaction_values[:, index] * 100,
            positions + (index - 0.5) * 0.16,
            marker=marker,
            color=color,
            linestyle="none",
            label=label,
        )
    ax.set_yticks(positions, reaction_labels, fontsize=5)
    ax.invert_yaxis()
    ax.axvline(2, color=palette["black"], linestyle=":", linewidth=0.8, label="2% reaction target")
    ax.set(xlabel="Per-finger vector contrast (%)", title="Opposing reactions stay separate", xlim=(0, 42))
    ax.legend(loc="lower right", frameon=False)

    series = (
        ("roundoff-prescribed-v1/group0/snh", "Coarse", palette["black"], "-"),
        ("prescribed-mesh-uncertainty-v1/group1/base1-surface2-wall4", "Surface2 / wall4", palette["blue"], "--"),
        (
            "prescribed-surface-uncertainty-v1/group1/base1-surface3-wall2",
            "Surface3 / wall2",
            palette["bluish_green"],
            "-.",
        ),
        ("prescribed-numerical-uncertainty-v1/group0/dt-half", "Half timestep", palette["vermillion"], ":"),
    )
    curves = []
    for suffix, label, color, linestyle in series:
        audit = select(audits, "folder", suffix)
        assert audit["complete_history_established"], suffix
        path = Path(audit["folder"]) / "steps.jsonl"
        sources.append(file_id(path))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        times = np.array([row["time_s"] for row in rows])
        peaks = np.array([row["raw_vm_peak_pa"] for row in rows])
        energy = np.array([row["egg_energy_change_minus_contact_work_j"] for row in rows])
        assert np.isfinite(peaks).all() and np.isfinite(energy).all() and abs(times[-1] - 6) < 1e-12
        axes[1, 0].plot(times, peaks / 1e6, color=color, linestyle=linestyle, linewidth=0.8, label=label)
        axes[1, 1].plot(times, energy * 1000, color=color, linestyle=linestyle, linewidth=0.8, label=label)
        curves.extend((label, t, p, e) for t, p, e in zip(times, peaks, energy))
    axes[1, 0].set(xlabel="Time (s)", ylabel="Raw global VM (MPa)", title="Every physical-step peak", xlim=(0, 6))
    axes[1, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, frameon=False, columnspacing=0.8)
    axes[1, 0].set_title("Every physical-step peak", pad=24)
    axes[1, 1].axhline(0, color=palette["grey"], linestyle=":", linewidth=0.6)
    axes[1, 1].set(
        xlabel="Time (s)", ylabel=r"$\Delta E-W_{\rm contact}$ (mJ)", title="Signed egg work ledger", xlim=(0, 6)
    )

    for filename, headers, rows in (
        (
            "field-contrasts.csv",
            ["contrast", "relative_tangent_contrast"],
            [(label, value) for (label, _), value in zip(contrasts, field)],
        ),
        (
            "reaction-contrasts.csv",
            ["contrast", "relative_finger_force", "relative_finger_moment"],
            [(label, *value) for label, value in zip(reaction_labels, reaction_values)],
        ),
        (
            "physical-step-curves.csv",
            ["configuration", "time_s", "raw_global_vm_pa", "egg_energy_minus_recorded_contact_work_j"],
            curves,
        ),
    ):
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
    issues = rendered_text_qa(fig)
    write_json(args.output / "rendered-text-qa.json", [{"figure": "reference-qualification", "text_issues": issues}])
    stem = args.output / "reference-qualification"
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=450)
    fig.savefig(args.output / "reference-qualification_preview.png", dpi=150)
    actual_mm = (fig.get_size_inches() * 25.4).tolist()
    plt.close(fig)
    svg = ElementTree.parse(stem.with_suffix(".svg"))
    editable_text = len(svg.findall(".//{http://www.w3.org/2000/svg}text"))
    assert editable_text > 0
    write_json(
        args.output / "result.json",
        {
            "kind": "complete-reference-qualification-figure",
            "identity": runtime_identity(),
            "claim_anchor": "qipc-reference-uncertainty-20261018",
            "sources": sources,
            "figure_size_mm": actual_mm,
            "rendered_text_issues": issues,
            "editable_svg_text_elements": editable_text,
            "reference_qualified": False,
            "scope": "Separate complete-history contrasts and full physical-step signed/raw diagnostics. Fixed-wall surface3/surface2 has surface2 as denominator; other field/reaction contrasts use coarse surface1/wall4. No smoothing, phase cut, clipped values, confidence interval, additive error budget or accuracy-qualified winner is implied. Signed energy is the recorded egg ledger, not a complete conservation proof.",
        },
    )


if __name__ == "__main__":
    main()
