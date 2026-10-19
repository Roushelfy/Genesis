"""Render friction figures only from matching qualified fine fields and admitted rankings."""

import argparse
import csv
import json
import shutil
from pathlib import Path

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
    cli.add_argument("--reference-qualification", type=Path, required=True)
    cli.add_argument("--ranking", type=Path, required=True)
    cli.add_argument("--force-data", type=Path, required=True)
    cli.add_argument("--style", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    receipts, checks, figures, directories = {}, {}, {}, {}
    error = None

    def record(path):
        key = str(path)
        if key not in receipts:
            receipts[key] = file_id(path)
        return receipts[key]

    def read(path):
        record(path)
        return json.loads(path.read_text())

    def verify(receipt):
        assert record(Path(receipt["path"])) == receipt, "Changed admitted input: " + receipt["path"]

    def check(name, passed):
        checks[name] = bool(passed)
        assert passed, name

    try:
        qualified = read(args.reference_qualification)
        check(
            "explicit_fine_friction_reference_qualified",
            qualified.get("passed", False)
            and qualified.get("fine_reference_qualified", False)
            and bool(qualified.get("checks"))
            and all(qualified["checks"].values()),
        )
        for receipt in (*qualified["receipts"], *qualified["verified_inputs"]):
            verify(receipt)
        rank = read(args.ranking)
        check(
            "complete_method_errors_admitted",
            rank["passed"] and rank["errors_admitted"] and all(rank["checks"].values()) and len(rank["rows"]) == 54,
        )
        check("eighteen_conditional_metric_rankings", len(rank["conditional_rankings"]) == 18)
        ranked_inputs = {row["path"]: row for row in rank["receipts"]}
        check(
            "ranking_binds_same_qualification",
            ranked_inputs[str(args.reference_qualification)] == record(args.reference_qualification),
        )
        for receipt in rank["receipts"]:
            verify(receipt)
        accepted = qualified["accepted_reference_frames"]
        check("two_qualified_reference_endpoints", len(accepted) == 2 and accepted[0] != accepted[1])
        details = {row["method"]: row for row in rank["details"] if row["level"] == 1}
        check("three_friction_candidate_details", set(details) == {"dpr", "scr", "shell"})
        reference_mesh = None
        for group, method in enumerate(("dpr", "scr", "shell")):
            directory = args.data_root / f"canonical-stvk-method-fields-v2/group{3 + group}/{method}-level1-q1"
            result = read(directory / "result.json")
            check(
                method + "/centroid_display_only",
                result["surface_quadrature_points_per_face"] == 1 and result["physical_volume_weighting"],
            )
            check(method + "/single_case", len(result["cases"]) == 1)
            case = result["cases"][0]
            check(
                method + "/same_ranked_frames_and_meshes",
                all(
                    case["raw"][index][key] == details[method]["raw"][index][key]
                    for index in range(2)
                    for key in ("mesh", "field")
                ),
            )
            check(method + "/fine_friction_endpoint", case["raw"][1]["field"]["path"] == accepted[1])
            for raw in case["raw"]:
                verify(raw["mesh"])
                verify(raw["field"])
            if reference_mesh is None:
                reference_mesh = case["raw"][1]["mesh"]
            check(method + "/same_reference_mesh", reference_mesh == case["raw"][1]["mesh"])
            directories[method] = directory
        data = read(args.force_data)
        check(
            "own_complete_physical_force_histories",
            data["passed"] and all(data["checks"].values()) and data["trajectory_rows"] == 1600,
        )
        # The extractor does not grant spatial admission. Bind it to the
        # independently qualified reference and to the ranking's own histories.
        extracted_inputs = {row["path"]: row for row in data["receipts"]}
        for method, candidate in rank["candidates"].items():
            for filename in ("start.json", "result.json", "steps.jsonl"):
                path = str(Path(candidate["run"]) / filename)
                check(method + "/same_extracted_" + filename, extracted_inputs[path] == ranked_inputs[path])
        for receipt in (*data["receipts"], *(row["output"] for row in data["traction"]), data["force_history"]):
            verify(receipt)
        for level in range(2):
            reference = [
                row["input"] for row in data["traction"] if row["method"] == "reference" and row["level"] == level
            ]
            check(
                f"reference/{level}/matching_qualified_traction_frame",
                len(reference) == 1 and reference[0]["path"] == accepted[level],
            )
            for method in ("dpr", "scr", "shell"):
                actual = [row["input"] for row in data["traction"] if row["method"] == method and row["level"] == level]
                expected = Path(rank["candidates"][method]["run"]) / f"frames/level{level}.npz"
                check(
                    f"{method}/{level}/matching_own_traction_frame", len(actual) == 1 and actual[0] == record(expected)
                )
        geometry, values = surface_data(Path(reference_mesh["path"]), directories, record)
        samples = traction_data(args.force_data.parent, record)
        with Path(data["force_history"]["path"]).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        series = history_series(rows)
        check("all_complete_field_force_and_history_arrays", len(values) == 8 and len(series) == 3)
        record(args.style)
        for source in (
            Path(__file__),
            Path(__file__).with_name("stvk_figure_render.py"),
            Path(__file__).with_name("pad_spatial_figure.py"),
        ):
            record(source)
            shutil.copy2(source, args.output / source.name)
        shutil.copy2(args.style, args.output / "nature_mpl_style.py")
        shutil.copy2(Path(data["force_history"]["path"]), args.output / "force-history.csv")
        for path in args.force_data.parent.glob("traction-*-level*.npz"):
            shutil.copy2(path, args.output / path.name)
        write_json(
            args.output / "figure-contract.json",
            {
                "claim_anchor": "qualified-sustained-friction-complete-fields-and-physical-loads",
                "main_claim": "Matching own-admitted fields and loads show complete spatial stress, physical contact support and actual-mass mechanical response for the qualified0.1N sustained-friction case.",
                "material_passport": list(receipts.values()),
                "panel_specification": {
                    "friction-complete-surface-stress": "Outer/inner by DPR, SCR, repaired250um Shell and qualified125um P2; complete angular material domain and shared tangent-Mises scale.",
                    "friction-normal-traction": "Upper/lower by shared native mechanics, Shell and fine P2; every actual nonzero-force sample and one density scale.",
                    "friction-tangent-traction": "Same complete support; magnitude density and signed net Fx/Fy keys at a common force scale.",
                    "friction-force-response": "Normal compliance and complete sustained tangential support; each actual mass and both normal reactions retained.",
                },
                "error_and_uncertainty_source": record(args.ranking),
                "limits": "q1 is for display; q9/q25 define numerical errors/peaks. Reference refinement envelopes are conditional, not statistical confidence or certified continuum bounds. Contact maps are physical pad loads per projected reference area, not reconstructed native traction or deformed-area Cauchy traction. No smoothing, percentile clipping, polar removal or force threshold. Native endpoint admission does not admit its failed every-step mode. Curves retain complete mechanics through0.4s; warm cost/production throughput require matching configurations before comparison.",
            },
        )
        style = load_style(args.style)
        figures["friction-complete-surface-stress"] = render_surface(
            geometry, values, args.output / "friction-complete-surface-stress", style, "P2, 125 µm"
        )
        figures["friction-normal-traction"] = render_traction(
            samples, args.output / "friction-normal-traction", style, "P2, 125 µm"
        )
        figures["friction-tangent-traction"] = render_traction(
            samples, args.output / "friction-tangent-traction", style, "P2, 125 µm", tangent=True
        )
        figures["friction-force-response"] = render_response(
            series, args.output / "friction-force-response", style, "P2, 125 µm"
        )
        check("all_export_text_geometry_checks", all(not item["export"]["text_issues"] for item in figures.values()))
        (args.output / "caption.md").write_text(
            "Claim anchor: qualified-sustained-friction-complete-fields-and-physical-loads.\n\n"
            "At the qualified0.1N sustained-friction endpoint, all outer/inner material centroids and both poles are displayed under a single tangent-Mises scale. The angular chart is not equal-area; error integration and unsmoothed raw/common numerical peaks use the independent q9/q25 tables.\n\n"
            "Actual upper/lower physical pad loads retain every nonzero-force sample, without interpolation, smoothing or clipping. Densities divide point forces by their projected reference areas. Native points are the recorded applied wrench locations. Net tangential arrows retain both signed components under one force scale; they are not local vector fields. Marker size is a display aid. These physical loads are distinct from reconstructed native boundary traction.\n\n"
            "Complete400-step trajectories show both plate reactions against clearance-corrected closure and opposing tangential reaction−ΣFx alongside each actual-mass body force. DPR/SCR mechanics merge after independent agreement checks. Requested native equilibrium fields do not qualify its separately failed every-step stress mode. Shell denotes the repaired250µm variant. Conditional ranks/uncertainty remain in the accompanying exact error table; no universal method order follows from image colour.\n\n"
            "Place with the dimensional error/ranking table. Editable SVG/PDF text,450dpi dense artists,600dpi PNG and150dpi previews are retained with exact inputs and plotted arrays. All previews require visual inspection and a separate quantitative body-text interpretation before publication.\n"
        )
    except (OSError, KeyError, ValueError, AssertionError) as failure:
        error = str(failure)
        checks["complete_qualified_figure_contract"] = False
    passed = bool(checks) and all(checks.values())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "passed": passed,
            "checks": checks,
            "error": error,
            "figures": figures,
            "receipts": list(receipts.values()),
            "fine_reference_and_method_errors_admitted": passed,
            "preview_inspected": False,
            "submission_ready": False,
            "whole_goal_complete": False,
        },
    )
    if not passed:
        raise SystemExit("Qualified figure contract failed; no final artwork is admitted")


if __name__ == "__main__":
    main()
