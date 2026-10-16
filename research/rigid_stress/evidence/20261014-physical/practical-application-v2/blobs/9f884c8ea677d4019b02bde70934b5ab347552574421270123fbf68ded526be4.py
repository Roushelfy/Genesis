"""Audit compact application evidence, paired costs and reward traces; export unsmoothed maps."""

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.colors import Normalize
from nature_mpl_style import apply_nature_style, figure_size

from .skfem_reference import SkfemElasticReference
from .summarize_fine_references import closed_source


def section(vertices, tetrahedra, tensors):
    """Intersect each original affine tetrahedron independently with material x=0."""
    points, values, triangles = [], [], []
    for cell, tensor in zip(vertices[tetrahedra], tensors, strict=True):
        if cell[:, 0].min() > 0 or cell[:, 0].max() < 0:
            continue
        local = []
        for first, second in combinations(range(4), 2):
            a, b = cell[first, 0], cell[second, 0]
            if a == b:
                continue
            weight = -a / (b - a)
            if 0 <= weight <= 1:
                point = cell[first] + weight * (cell[second] - cell[first])
                stress = tensor[first] + weight * (tensor[second] - tensor[first])
                if not any(np.linalg.norm(point - old[0]) < 1e-13 for old in local):
                    local.append((point, stress))
        if len(local) < 3:
            continue
        coordinates = np.array([item[0][1:] for item in local])
        centre = coordinates.mean(axis=0)
        order = np.argsort(np.arctan2(coordinates[:, 1] - centre[1], coordinates[:, 0] - centre[0]))
        start = len(points)
        for index in order:
            point, tensor_value = local[index]
            deviator = tensor_value - np.eye(3) * np.trace(tensor_value) / 3
            points.append(point[1:])
            values.append(np.sqrt(1.5 * np.sum(deviator**2)))
        triangles.extend((start, start + index, start + index + 1) for index in range(1, len(local) - 1))
    points = np.array(points)
    return mtri.Triangulation(points[:, 0] * 1000, points[:, 1] * 1000, triangles), np.array(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--pressure-cost", type=Path, required=True)
    parser.add_argument("--source-prior-cost", type=Path, required=True)
    parser.add_argument("--pressure-reward", type=Path, required=True)
    parser.add_argument("--source-prior-reward", type=Path, required=True)
    parser.add_argument("--pressure-reward-half-radius", type=Path)
    parser.add_argument("--source-prior-reward-half-radius", type=Path)
    parser.add_argument("--reward-radii-only", action="store_true")
    parser.add_argument("--wrench-diagnostics-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def read(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    def arrays(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            return {key: saved[key].copy() for key in saved.files}

    def closed(run):
        assert (run / "returncode.txt").read_text().strip() == "0"
        closed_source(run)
        for name in ["source.sha256", "source-after.sha256", "native-snapshot-before.txt", "native-snapshot-after.txt"]:
            file = run / name
            inputs[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        assert (run / "native-snapshot-before.txt").read_bytes() == (run / "native-snapshot-after.txt").read_bytes()

    if args.reward_radii_only:
        assert args.pressure_reward_half_radius is not None and args.source_prior_reward_half_radius is not None
        runs = [
            args.pressure_reward,
            args.source_prior_reward,
            args.pressure_reward_half_radius,
            args.source_prior_reward_half_radius,
        ]
        traces, reports = [], []
        for run in runs:
            closed(run)
            reports.append(read(run / "results/summary.json"))
            trace = arrays(run / "results/raw_trace.npz")
            assert np.isfinite(trace["maximum_stress_Pa"]).all()
            traces.append(trace)
        rows = []
        limits = traces[0]["strong_limit_N"][[0, 8]]
        for trace in traces:
            np.testing.assert_array_equal(trace["strong_limit_N"][[0, 8]], limits)
        for condition in range(8):
            differences = np.concatenate(
                [
                    trace["lift_window_raw_peak_Pa"][:, condition + 8] - trace["lift_window_raw_peak_Pa"][:, condition]
                    for trace in traces
                ]
            )
            stable = bool((differences > 0).all() or (differences < 0).all())
            low = np.concatenate([trace["lift_window_raw_peak_Pa"][:, condition] for trace in traces])
            high = np.concatenate([trace["lift_window_raw_peak_Pa"][:, condition + 8] for trace in traces])
            absolute_model_band = max(float(np.ptp(low)), float(np.ptp(high)))
            minimum_gap = float(np.abs(differences).min())
            rows.append(
                {
                    "condition": condition,
                    "paired_difference_high_minus_low_range_Pa": [float(differences.min()), float(differences.max())],
                    "low_limit_peak_range_Pa": [float(low.min()), float(low.max())],
                    "high_limit_peak_range_Pa": [float(high.min()), float(high.max())],
                    "ranking_stable_across_tested_models_and_radii": stable,
                    "lower_peak_choice": f"{limits[0] if differences[0] > 0 else limits[1]:g}N"
                    if stable
                    else "unresolved",
                    "minimum_absolute_paired_separation_Pa": minimum_gap,
                    "absolute_peak_model_radius_band_Pa": absolute_model_band,
                    "separation_exceeds_observed_absolute_peak_band": stable and minimum_gap > absolute_model_band,
                }
            )
        assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
        (args.output / "summary.json").write_text(
            json.dumps(
                {
                    "scope": "Paired low/high grip-force reward-proxy repeatability and footprint sensitivity across two production models and two declared physical radius schedules, with all choices retained.",
                    "actual_commanded_strong_limits_N": limits.tolist(),
                    "reports": reports,
                    "rows": rows,
                    "observed_environment_steps": sum(report["observed_environment_steps"] for report in reports),
                    "invalid_stress_steps": sum(report["invalid_stress_steps"] for report in reports),
                    "input_sha256": inputs,
                    "goal_completion": False,
                    "limitations": "Twelve paired observations per condition (three restarts times two models times two radius schedules), not twelve independent physical experiments. Stable ordering in this tested set is weaker than separation exceeding an absolute model/radius band. Neither validates continuum or experimental stress, a fracture threshold, or decisions inside unresolved uncertainty.",
                },
                indent=2,
            )
            + "\n"
        )
        return

    closed(args.reconstruction)
    reconstruction = read(args.reconstruction / "results/summary.json")
    accepted = [row for row in reconstruction["rows"] if row["classification"] == "accepted_conditional_snapshot"]
    assert len(accepted) == 32
    packets_run = Path(reconstruction["packets"])
    packet_summary = read(packets_run / "results/summary.json")
    physical_checks = []
    for row in accepted:
        label = f"step{row['step']:05d}-r{round(row['radius_m'] * 1e6)}-{row['model']}"
        data = arrays(args.reconstruction / f"results/{label}.npz")
        packets = read(packets_run / f"results/step{row['step']:05d}-packets.json")
        target_force = np.array([item["force_N"] for item in packets])
        target_moment = np.cross(np.array([item["point_m"] for item in packets]), target_force) + np.array(
            [item["free_moment_Nm"] for item in packets]
        )
        actual_force = data["traction_forces_N"]
        actual_moment = np.cross(data["traction_positions_m"], actual_force)
        force_sum = np.array([math.fsum(actual_force[:, component]) for component in range(3)])
        moment_sum = np.array([math.fsum(actual_moment[:, component]) for component in range(3)])
        force_target = np.array([math.fsum(target_force[:, component]) for component in range(3)])
        moment_target = np.array([math.fsum(target_moment[:, component]) for component in range(3)])
        scale = np.linalg.norm(force_target)
        if not scale:
            scale = np.linalg.norm(target_force, axis=1).sum()
        force_error = float(np.linalg.norm(force_sum - force_target))
        moment_error = float(np.linalg.norm(moment_sum - moment_target))
        area, normals = data["traction_areas_m2"], data["traction_normals"]
        assert np.all(area > 0)
        density = actual_force / area[:, None]
        pressure = np.einsum("ij,ij->i", density, normals)
        shear = density - pressure[:, None] * normals
        violation = np.maximum(np.linalg.norm(shear, axis=1) - packet_summary["friction"] * pressure, -pressure)
        physical_checks.append(
            {
                "case": label,
                "compensated_force_error_N": force_error,
                "global_net_scaled_diagnostic_force_budget_N": 1e-8 * scale,
                "compensated_moment_error_Nm": moment_error,
                "global_net_scaled_diagnostic_moment_budget_Nm": 1e-8 * scale * row["radius_m"],
                "maximum_local_coulomb_violation_Pa": float(max(0, violation.max())),
                "global_net_force_N": force_target.tolist(),
                "sum_individual_force_norms_N": float(np.linalg.norm(target_force, axis=1).sum()),
                "native_reported_force_errors_N": row["reported_force_error_N"],
                "native_reported_moment_errors_Nm": row["reported_moment_error_Nm"],
                "global_net_scaled_force_gate": bool(force_error <= 1e-8 * scale),
                "global_net_scaled_moment_gate": bool(moment_error <= 1e-8 * scale * row["radius_m"]),
                "packet_checks": [],
            }
        )
        for index, packet in enumerate(packets):
            chosen = data["packet"] == index
            forces = actual_force[chosen]
            moments = np.cross(data["traction_positions_m"][chosen] - packet["point_m"], forces)
            net = np.array([math.fsum(forces[:, component]) for component in range(3)])
            moment = np.array([math.fsum(moments[:, component]) for component in range(3)])
            packet_scale = float(np.linalg.norm(target_force[index]))
            physical_checks[-1]["packet_checks"].append(
                {
                    "packet": index,
                    "force_error_N": float(np.linalg.norm(net - target_force[index])),
                    "moment_about_actual_contact_point_error_Nm": float(
                        np.linalg.norm(moment - packet["free_moment_Nm"])
                    ),
                    "force_budget_N": 1e-8 * packet_scale,
                    "moment_budget_Nm": 1e-8 * packet_scale * row["radius_m"],
                }
            )
        if row["model"] == "body_source_prior":
            # This model constrains the aggregate and intentionally redistributes
            # local packet wrenches. The original native aggregate budget applies.
            contract_passed = (
                physical_checks[-1]["global_net_scaled_force_gate"]
                and physical_checks[-1]["global_net_scaled_moment_gate"]
                and max(row["reported_force_error_N"]) <= 1e-8 * scale
                and max(row["reported_moment_error_Nm"]) <= 1e-8 * scale * row["radius_m"]
            )
            contract = "body aggregate wrench,1e-8 of aggregate force scale; packet redistribution is measured"
        else:
            # Independent packet constraints have individual force scales and
            # moments about their own application points. A cancellation-scaled
            # global diagnostic is a distinct stricter statistic, retained above.
            contract_passed = all(
                packet["force_error_N"] <= packet["force_budget_N"]
                and packet["moment_about_actual_contact_point_error_Nm"] <= packet["moment_budget_Nm"]
                for packet in physical_checks[-1]["packet_checks"]
            )
            contract = "each packet wrench,1e-8 of its own force scale and physical radius"
        physical_checks[-1].update(native_contract=contract, native_contract_passed=bool(contract_passed))

    (args.output / "wrench-diagnostics.json").write_text(json.dumps(physical_checks, indent=2) + "\n")
    if args.wrench_diagnostics_only:
        print("Recorded independent native cloud wrench diagnostics; no acceptance reclassification.", flush=True)
        return
    assert all(row["native_contract_passed"] for row in physical_checks), "See preserved wrench-diagnostics.json"

    costs = []
    sampled_invariance = []
    for name, run in [("pressure", args.pressure_cost), ("body_source_prior", args.source_prior_cost)]:
        assert (run / "returncode.txt").read_text().strip() == "0"
        scopes = {}
        for scope in ["rigid", "live", "policy-rigid", "policy"]:
            current = run / scope
            assert (current / "returncode.txt").read_text().strip() == "0"
            assert read(current / "snapshot-inventory-after.json")["passed"]
            assert (current / "snapshot-before-verification.txt").read_bytes() == (
                current / "snapshot-after-verification.txt"
            ).read_bytes()
            result = read(current / "results/benchmark.json")
            assert all(row["steps"] == 2400 and row["invalid_environment_steps"] == 0 for row in result["repeats"])
            assert result["envs"] == 32 and result["conditions_sha256"]
            scopes[scope] = result
            rates = [row["transitions_per_second"] for row in result["repeats"]]
            costs.append(
                {
                    "model": name,
                    "scope": scope,
                    "median_valid_envsteps_per_s": float(np.median(rates)),
                    "min_valid_envsteps_per_s": min(rates),
                    "max_valid_envsteps_per_s": max(rates),
                    "median_batch_step_seconds": float(
                        np.median([row["seconds"] / row["steps"] for row in result["repeats"]])
                    ),
                    "native_buffer_bytes": result.get("native_buffer_bytes", 0),
                    "torch_peak_allocated_bytes": result["torch_peak_allocated_bytes"],
                    "gpu_uuid": result["gpu_uuid"],
                    "resets_per_repeat": [row["resets"] for row in result["repeats"]],
                }
            )
        assert len({value["gpu_uuid"] for value in scopes.values()}) == 1
        assert len({value["conditions_sha256"] for value in scopes.values()}) == 1
        rigid, live = scopes["rigid"], scopes["live"]
        assert len(rigid["warmup_quality"]) == len(live["warmup_quality"])
        height_difference = max(
            float(np.max(np.abs(np.array(first["height_m"]) - second["height_m"])))
            for first, second in zip(rigid["warmup_quality"], live["warmup_quality"], strict=True)
        )
        final_difference = float(np.max(np.abs(np.array(rigid["egg_position_m"]) - live["egg_position_m"])))
        sampled_invariance.append(
            {
                "model": name,
                "scripted_warmup_maximum_height_difference_m": height_difference,
                "scripted_final_maximum_coordinate_difference_m": final_difference,
                "policy_scope_has_stress_dependent_actions": True,
                "limits": "Sampled warmup and final states, not every timed substep. Scripted control does not read stress; policy inference does.",
            }
        )
    for row in costs:
        baseline = next(item for item in costs if item["model"] == row["model"] and item["scope"] == "rigid")
        if row["scope"] == "live":
            row["incremental_scripted_seconds_per_batch_step"] = (
                row["median_batch_step_seconds"] - baseline["median_batch_step_seconds"]
            )

    rewards = []
    traces = []
    for name, run in [("pressure", args.pressure_reward), ("body_source_prior", args.source_prior_reward)]:
        closed(run)
        report = read(run / "results/summary.json")
        data = arrays(run / "results/raw_trace.npz")
        assert report["invalid_stress_steps"] == 0 and np.isfinite(data["maximum_stress_Pa"]).all()
        window = (data["phase"] >= 0.3) & (data["phase"] <= 0.6)
        peak = np.where(window, data["maximum_stress_Pa"], 0).max(axis=1)
        np.testing.assert_array_equal(peak, data["lift_window_raw_peak_Pa"])
        rewards.append({"model": name, "report": report})
        traces.append(data)
    stable = []
    first, second = (data["lift_window_raw_peak_Pa"] for data in traces)
    for index in range(8):
        differences = np.concatenate([value[:, index + 8] - value[:, index] for value in [first, second]])
        same_sign = bool((differences > 0).all() or (differences < 0).all())
        gap = float(np.abs(differences).min())
        variability = max(float(np.ptp(value[:, column])) for value in [first, second] for column in [index, index + 8])
        model_difference = max(
            float(np.abs(first[:, column] - second[:, column]).max()) for column in [index, index + 8]
        )
        stable.append(
            {
                "condition": index,
                "paired_4N_minus_1N_peak_range_Pa": [float(differences.min()), float(differences.max())],
                "lower_peak_choice": ("1N" if differences[0] > 0 else "4N") if same_sign else "unresolved",
                "minimum_absolute_paired_separation_Pa": gap,
                "maximum_repeat_span_Pa": variability,
                "maximum_cross_model_peak_difference_Pa": model_difference,
                "discrimination_exceeds_observed_repeat_and_model_difference": same_sign
                and gap > variability + model_difference,
            }
        )

    apply_nature_style(matplotlib)
    geometry = arrays(packets_run / "results/body_mesh.npz")
    reference = SkfemElasticReference(geometry["vertices"], geometry["tetrahedra"], 1e10, 0.3, 2000)
    selected_step = 866
    resolved = arrays(packets_run / f"results/step{selected_step:05d}.npz")
    tensor_l, _ = reference.corner_stress(resolved["resolved_linear_displacement_m"])
    selected = [
        ("Resolved L", None),
        ("Pressure, 3 mm", "r3000-pressure"),
        ("Pressure, 6 mm", "r6000-pressure"),
        ("Source prior, 6 mm", "r6000-body_source_prior"),
    ]
    datasets = [
        None if label is None else arrays(args.reconstruction / f"results/step{selected_step:05d}-{label}.npz")
        for _, label in selected
    ]
    sections = [
        section(geometry["vertices"], geometry["tetrahedra"], tensor_l if data is None else data["corner_tensor_Pa"])
        for data in datasets
    ]
    stress_scale = Normalize(0, max(values.max() for _, values in sections) / 1e6)
    figure, axes = plt.subplots(1, 4, figsize=figure_size("double", 66), sharex=True, sharey=True)
    for index, (axis, (title, _), (triangulation, values)) in enumerate(zip(axes, selected, sections, strict=True)):
        artist = axis.tripcolor(triangulation, values / 1e6, shading="gouraud", cmap="cividis", norm=stress_scale)
        axis.set_aspect("equal")
        axis.set_xlabel("Material y (mm)")
        axis.set_title(title)
        axis.text(0, 1.1, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.set_xlim(-24, 24)
        axis.set_ylim(-33, 37)
    axes[0].set_ylabel("Material z (mm)")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.19, top=0.80, wspace=0.15)
    colour_axis = figure.add_axes([0.91, 0.23, 0.014, 0.50])
    figure.colorbar(artist, cax=colour_axis, label="Section von Mises (MPa)")
    export = []
    for suffix in ["svg", "pdf", "png"]:
        path = args.output / f"franka-stress-section.{suffix}"
        figure.savefig(path, dpi=1000, bbox_inches=None)
        export.append(str(path))
    figure.savefig(args.output / "franka-stress-section-preview.png", dpi=180, bbox_inches=None)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=figure_size("double", 70), sharex=True, sharey=True)
    plotted = datasets[1:]
    densities = [
        np.einsum("ij,ij->i", data["traction_forces_N"], data["traction_normals"]) / data["traction_areas_m2"] / 1000
        for data in plotted
    ]
    traction_scale = Normalize(0, max(value.max() for value in densities))
    for index, (axis, (title, _), data, density) in enumerate(zip(axes, selected[1:], plotted, densities, strict=True)):
        for source, marker in [(1, "o"), (2, "s"), (3, "^")]:
            mask = data["physical_source"] == source
            if mask.any():
                artist = axis.scatter(
                    data["traction_positions_m"][mask, 1] * 1000,
                    data["traction_positions_m"][mask, 2] * 1000,
                    c=density[mask],
                    s=1.3,
                    marker=marker,
                    cmap="viridis",
                    norm=traction_scale,
                    linewidths=0,
                    label=f"Source {source}",
                )
        axis.set_aspect("equal")
        axis.set_xlabel("Material y (mm)")
        axis.set_title(title)
        axis.text(0, 1.1, chr(97 + index), transform=axis.transAxes, fontsize=8, fontweight="bold")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, markerscale=3)
    axes[0].set_ylabel("Material z (mm)")
    figure.subplots_adjust(left=0.08, right=0.87, bottom=0.30, top=0.80, wspace=0.17)
    colour_axis = figure.add_axes([0.91, 0.31, 0.014, 0.40])
    figure.colorbar(artist, cax=colour_axis, label="Normal traction (kPa)")
    for suffix in ["svg", "pdf", "png"]:
        path = args.output / f"franka-traction-projection.{suffix}"
        figure.savefig(path, dpi=1000, bbox_inches=None)
        export.append(str(path))
    figure.savefig(args.output / "franka-traction-projection-preview.png", dpi=180, bbox_inches=None)
    plt.close(figure)
    assert all("<text" in Path(path).read_text() for path in export if path.endswith(".svg"))
    for filename, data in [
        ("costs.csv", costs),
        ("reconstruction-wrench-checks.csv", physical_checks),
        ("reward-discrimination.csv", stable),
    ]:
        with (args.output / filename).open("w", newline="") as stream:
            keys = list(dict.fromkeys(key for row in data for key in row))
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
    assert inputs == {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in inputs}
    report = {
        "scope": __doc__,
        "reconstruction_rows": len(accepted),
        "physical_checks": physical_checks,
        "costs": costs,
        "sampled_scripted_auxiliary_invariance": sampled_invariance,
        "reward_proxy": rewards,
        "reward_discrimination": stable,
        "cross_model_maximum_height_difference_m": float(np.max(np.abs(traces[0]["height_m"] - traces[1]["height_m"]))),
        "input_sha256": inputs,
        "figure_contract": {
            "claim_anchor": "franka-footprint-assumptions-change-stress",
            "material_passport": {
                "source": str(args.reconstruction),
                "status": "32 numerically accepted conditional snapshots; not continuum truth",
            },
            "section": "Material x=0, independently intersect each affine tetrahedron. Exact tensor interpolation to section vertices, then Gouraud display of their von Mises values. No cell merging or smoothing; raw global maxima are retained separately. Shared full section scale, no percentile clipping.",
            "traction": "All native positive-area samples, material y/z projection, actual normal traction and physical source marker identities. Projection can overlap opposite-side samples; not a continuum island segmentation.",
            "size_mm": [183, 66],
            "exports": export,
            "preview_visually_inspected": False,
            "remaining_QA": "Inspect both rendered previews before calling figures final.",
        },
        "goal_completion": False,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
