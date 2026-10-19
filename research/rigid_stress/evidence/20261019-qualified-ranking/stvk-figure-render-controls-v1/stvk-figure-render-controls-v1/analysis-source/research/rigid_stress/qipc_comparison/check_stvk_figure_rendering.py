"""Check rendering on qualified normal fields and mechanically admitted coarse friction data.

This fixed control never produces final fine-reference friction comparisons.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .stvk_figure_render import (
    history_series,
    load_style,
    render_response,
    render_surface,
    render_traction,
    surface_data,
    traction_data,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, figures = {}, {}, {}

    def record(path):
        key = str(path)
        if key not in receipts:
            receipts[key] = file_id(path)
        return receipts[key]

    def read(path):
        record(path)
        return json.loads(path.read_text())

    def verify(receipt):
        assert record(Path(receipt["path"])) == receipt, "Changed audited input: " + receipt["path"]

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    normal = read(args.data_root / "pad-normal-qualification-v2/group0/qualification/result.json")
    check(
        "existing_normal_reference_qualification",
        normal["normal_reference_change_targets_passed"] and all(normal["checks"].values()),
    )
    for receipt in normal["receipts"]:
        verify(receipt)
    directories = {}
    for group, method in enumerate(("dpr", "scr", "shell")):
        directory = args.data_root / f"pad-normal-surface-fields-v1/group{group}/{method}-0"
        result = read(directory / "result.json")
        check(
            method + "_normal_centroid_display",
            result["surface_quadrature_points_per_face"] == 1 and result["physical_volume_weighting"],
        )
        assert len(result["cases"]) == 1
        for raw in result["cases"][0]["raw"]:
            verify(raw["mesh"])
            verify(raw["field"])
        directories[method] = directory
    geometry, values = surface_data(
        args.data_root / "compliant-pad-joint-refinement-mesh-v1/group0/meshes/land-contact-0.npz",
        directories,
        record,
    )
    check("complete_normal_surface_tensors_and_identical_material_positions", len(values) == 8)

    data = args.data_root / "friction-plot-data-cpu-control-v1/group0/data"
    control = read(data / "result.json")
    check("coarse_physical_data_reader_admitted", control["passed"] and all(control["checks"].values()))
    check(
        "coarse_data_not_fine_spatial_admission",
        not control["spatial_reference_qualified"] and not control["whole_goal_complete"],
    )
    for receipt in (*control["receipts"], *(row["output"] for row in control["traction"]), control["force_history"]):
        verify(receipt)
    samples = traction_data(data, record)
    check("all_eight_traction_arrays_recomputed_and_native_arrays_agree", len(control["traction"]) == 8)
    history_path = Path(control["force_history"]["path"])
    with history_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    series = history_series(rows)
    check("all1600_history_rows_and_complete_native_mechanics_agree", len(rows) == 1600)
    for name, bad in (
        ("missing_physical_step", rows[:-1]),
        ("repeated_step", [rows[1], *rows[1:]]),
        ("nonfinite_force", [{**rows[0], "upper_reaction_n": "nan"}, *rows[1:]]),
        ("changed_native_actual_mass", [{**rows[0], "mass_kg": "1"}, *rows[1:]]),
    ):
        try:
            history_series(bad)
        except (ValueError, AssertionError):
            check(name + "_rejected", True)
        else:
            raise AssertionError(name + " was silently accepted")
    record(args.style)
    style = load_style(args.style)
    for source in (
        Path(__file__),
        Path(__file__).with_name("stvk_figure_render.py"),
        Path(__file__).with_name("pad_spatial_figure.py"),
    ):
        record(source)
        shutil.copy2(source, args.output / source.name)
    shutil.copy2(args.style, args.output / "nature_mpl_style.py")
    shutil.copy2(history_path, args.output / "control-force-history.csv")
    for path in data.glob("traction-*-level*.npz"):
        shutil.copy2(path, args.output / path.name)

    write_json(
        args.output / "figure-contract.json",
        {
            "claim_anchor": "stvk-renderer-actual-data-controls",
            "main_claim": "The renderer preserves complete surface coverage, every physical force sample and the full actual-mass mechanical trajectories in checked editable exports.",
            "target": "English repository supplement; renderer control, not final friction accuracy artwork.",
            "material_passport": list(receipts.values()),
            "panel_specification": {
                "normal-render-control-surface": "Eight complete normal-only material maps: outer/inner by DPR, SCR, repaired250um Shell and qualified125um P2 reference.",
                "coarse-control-normal-traction": "Upper/lower actual physical normal load density for shared native mechanics, repaired Shell and mechanically admitted500um P2.",
                "coarse-control-tangent-traction": "Same actual points, both force components retained; density magnitude and net-force direction keys use shared scales.",
                "coarse-control-force-response": "Full upper/lower normal reactions against clearance-corrected closure; opposing tangential resultant and each actual body force against time.",
            },
            "evidence_hierarchy": "Existing normal reference qualification and own physical-history checks, then exact q1 surface/force exports. The coarse friction reference has no fine spatial admission.",
            "uncertainty": "No new accuracy interval, rank or statistical confidence is inferred from these renderer controls.",
            "colour_roles": "Continuous magnitudes: viridis with shared untrimmed scales. Method curves: blue/circle native, vermillion/square Shell, green/triangle P2; plate/reaction channels additionally use solid/dotted lines.",
            "limits": "Angular charts are not equal-area. Centroid displays are not numerical peak definitions. Point markers do not represent physical sample areas. Densities use reference projected area and physical pad force, not reconstructed native boundary traction. Net arrows are summed forces, not local vector fields. Final friction graphics still require qualified fine reference, final method-error admission and matching traction extraction.",
        },
    )
    figures["normal-render-control-surface"] = render_surface(
        geometry, values, args.output / "normal-render-control-surface", style, "P2, 125 µm"
    )
    figures["coarse-control-normal-traction"] = render_traction(
        samples, args.output / "coarse-control-normal-traction", style, "P2, 500 µm"
    )
    figures["coarse-control-tangent-traction"] = render_traction(
        samples, args.output / "coarse-control-tangent-traction", style, "P2, 500 µm", tangent=True
    )
    figures["coarse-control-force-response"] = render_response(
        series, args.output / "coarse-control-force-response", style, "P2, 500 µm"
    )
    check("four_complete_editable_export_sets", len(figures) == 4)
    check("all_text_geometry_checks", all(not value["export"]["text_issues"] for value in figures.values()))
    captions = (
        "Claim anchor: stvk-renderer-actual-data-controls.\n\n"
        "Renderer control using the qualified normal0.1N fields: complete outer/inner material maps share one untrimmed tangent-plane Mises scale, with identical centroids on the125µm P2 grid. Both poles and all azimuths are retained. Each face has one display value; numerical error/peak tables use their own higher-order integration.\n\n"
        "Coarse physical friction controls show every nonzero-force sample on both pads, without pressure interpolation, smoothing, percentile clipping or a force threshold. Native applied-point positions are used; Shell/P2 use recorded current points. Physical point forces are divided by projected reference areas. Both tangential components remain in the source arrays. Equal-length direction keys represent equal net force; they do not represent a local vector field. All contact panels share complete support bounds and one scale per density quantity. Marker sizes are visual sampling aids.\n\n"
        "Force-response controls retain all400 steps for each method, including the initial approach and normal-to-friction transition. Closure already subtracts both initial clearances. The tangential reaction is−ΣFx; each body-force curve uses its actual finite-mesh mass. DPR/SCR mechanical curves merge only after all recorded values agree. Requested native equilibrium fields do not admit the failed every-step stress mode.\n\n"
        "Place these controls with the rendering/data provenance notes, separately from the final friction error/ranking figures. The500µm P2 history has passed its own mechanics, but these controls do not qualify its spatial stress error. No final friction-method order is inferred. Labels and axes remain editable; dense data artists are rasterized at450dpi, with600dpi PNG and150dpi previews.\n"
    )
    (args.output / "captions.md").write_text(captions)
    (args.output / "alt-text.md").write_text(
        "Eight angular maps cover both material surfaces and both poles for the existing normal comparison. Six-panel maps compare upper and lower physical pad loads for native, repaired Shell and a coarse solid reference, preserving the full force-bearing extent. Arrow keys give both signed net tangential components with a common force scale. Two trajectory panels show normal compliance and tangential support throughout the full recorded mechanics. Coarse controls do not establish fine-reference friction accuracy.\n"
    )
    (args.output / "body-handoff.md").write_text(
        "These are data and renderer controls. The complete maps and trajectories verify coverage and export readability; numerical maxima printed in their profiles are display/sample maxima, rather than continuum bounds. Interpret fine friction field concentrations, method separation or uncertainty-supported ties only after the qualified-reference ranking packet exists. Actual masses, all force-bearing points and the full approach/hold history remain visible; no nominal common body-force curve replaces the measured curves.\n"
    )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": all(checks.values()),
            "checks": checks,
            "figures": figures,
            "receipts": list(receipts.values()),
            "final_friction_accuracy_figures": False,
            "fine_friction_reference_qualified": False,
            "submission_ready": False,
            "preview_inspected": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
