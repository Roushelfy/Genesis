"""Derive bounded comparison tables from immutable executed receipts and full fields."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ..skfem_reference import SkfemElasticReference
from .common import file_identity, write_json


def unpack(value):
    result = np.zeros((*value.shape[:-1], 3, 3))
    for index, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        result[..., i, j] = result[..., j, i] = value[..., index]
    return result


def relative(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def table(output, name, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{name}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root, out = args.runs, args.output
    sources = {}

    def load(path):
        path = root / path
        sources[str(path.relative_to(root))] = file_identity(path)
        return json.loads(path.read_text())

    def arrays(path):
        path = root / path
        sources[str(path.relative_to(root))] = file_identity(path)
        return np.load(path)

    rates = []
    for campaign in ("paired-normal-v1", "validated-native-v2"):
        for path in sorted(root.joinpath(campaign).glob("group*/*/result.json")):
            d = load(path.relative_to(root))
            for repeat in d["repeats"]:
                assert repeat["completed_requested_history"] and repeat["final_output_finite_checked_after_timer"]
                valid = repeat.get("invalid_environment_steps")
                qualified = campaign == "validated-native-v2" and d["args"]["model"] != "rigid"
                if qualified:
                    assert valid == 0 and d["warmup_invalid_environment_steps"] == 0
                rates.append(
                    {
                        "cohort": campaign,
                        "configuration": path.parent.name.rsplit("-b", 1)[0],
                        "batch": d["args"]["envs"],
                        "repeat": repeat["repeat"],
                        "env_step_s": repeat["aggregate_env_steps_per_second"],
                        "batch_step_ms": repeat["batch_step_ms"],
                        "numerically_qualified": qualified,
                        "invalid_env_steps": valid,
                        "physical_accuracy_qualified": False,
                        "build_s": d["build_seconds"],
                        "warmup_s": d["warmup_seconds"],
                        "host": d["identity"]["host"],
                        "gpu": d["identity"]["gpu"],
                        "rigid_dofs": d.get("rigid_dofs"),
                        "receipt": str(path.relative_to(root)),
                    }
                )
    table(out, "rates", rates)
    rate_summary = []
    for cohort, configuration, batch in sorted({(r["cohort"], r["configuration"], r["batch"]) for r in rates}):
        rows = [r for r in rates if (r["cohort"], r["configuration"], r["batch"]) == (cohort, configuration, batch)]
        rate_summary.append(
            {
                "cohort": cohort,
                "configuration": configuration,
                "batch": batch,
                "median_env_step_s": float(np.median([r["env_step_s"] for r in rows])),
                "min_env_step_s": min(r["env_step_s"] for r in rows),
                "max_env_step_s": max(r["env_step_s"] for r in rows),
                "median_batch_ms": float(np.median([r["batch_step_ms"] for r in rows])),
            }
        )
    table(out, "rate-summary", rate_summary)

    memory = []
    for path in sorted(root.joinpath("paired-diagnostics-v1/group0").glob("*/result.json")):
        d = load(path.relative_to(root))
        memory.append(
            {
                "configuration": path.parent.name,
                "batch": d["args"]["envs"],
                "peak_mib": d["process_memory"]["peak_mib"],
                "build_s": d["build_seconds"],
                "warmup_s": d["warmup_seconds"],
                "rigid_dofs": d["rigid_dofs"],
            }
        )
    table(out, "memory", memory)

    profiles = []
    kernel_rows = []
    for path in sorted(root.joinpath("profiles-v1/group0").glob("*-summary.json")):
        d = load(path.relative_to(root))
        groups = {}
        for name, milliseconds in d["kernel_total_ms"]:
            if "shell_system_product" in name:
                group = "shell Hessian products"
            elif "shell_pcg" in name:
                group = "shell PCG/preconditioner"
            elif "shell_coarse" in name:
                group = "shell coarse"
            elif "shell_compute_forces" in name:
                group = "shell forces"
            elif "shell_integrate" in name or "shell_apply_dv" in name:
                group = "shell integrate"
            elif "coupl" in name:
                group = "coupling"
            elif "recover_fields" in name or "locate_peaks" in name:
                group = "shell full output"
            elif any(
                token in name for token in ("stress", "traction", "pressure", "surface_fan", "elastic", "inverse")
            ):
                group = "native stress/mapping"
            else:
                group = "rigid/control/other"
            groups[group] = groups.get(group, 0.0) + milliseconds / d["step_count"]
            kernel_rows.append(
                {
                    "configuration": path.stem,
                    "kernel": name,
                    "group": group,
                    "gpu_ms_per_step": milliseconds / d["step_count"],
                }
            )
        assert np.isclose(sum(groups.values()), d["kernel_sum_ms"] / d["step_count"])
        profiles.append(
            {
                "configuration": path.stem,
                "cpu_annotation_ms": d["step_wall_sum_ms"] / d["step_count"],
                "gpu_kernel_sum_ms": d["kernel_sum_ms"] / d["step_count"],
                "kernels_per_step": d["kernel_count"] / d["step_count"],
                "gpu_groups": groups,
                "copies": d["explicit_copy_or_memset_bytes"],
                "runtime_calls": d["runtime_counts"],
                "runtime_ms": d["runtime_duration_sum_ms"],
            }
        )
    table(out, "kernel-groups", kernel_rows)

    dynamics = []
    peaks = []
    for campaign in ("full-shell-history-v1", "full-native-history-v1", "peak-traces-v2", "fresh-restarts-v1"):
        for path in sorted(root.joinpath(campaign).glob("group*/*/result.json")):
            d = load(path.relative_to(root))
            with arrays(path.parent.relative_to(root) / "trajectory-repeat0.npz") as z:
                row = {
                    "campaign": campaign,
                    "configuration": path.parent.name,
                    "peak_pa": float(z["peaks_pa"].max()),
                    "recorded_steps": len(z["peaks_pa"]),
                    "position_z_min_m": float(z["positions"][..., 2].min()),
                    "position_z_max_m": float(z["positions"][..., 2].max()),
                    "position_definition": "unweighted shell vertex centroid"
                    if d["args"]["model"] == "shell"
                    else "native authored link origin",
                }
                if z["caps"].size:
                    residual = z["residuals"]
                    row.update(
                        cap_exhausted_substeps=int(z["caps"].sum()),
                        recursive_budget_control_env_pass_fraction=float((residual[..., 2] <= 1).mean()),
                        true_block_jacobi_control_env_pass_fraction=float(
                            (residual[..., 1] <= d["args"]["tolerance"]).mean()
                        ),
                        max_true_l2_relative=float(residual[..., 0].max()),
                        max_true_block_jacobi_relative=float(residual[..., 1].max()),
                        max_recursive_squared_budget=float(residual[..., 2].max()),
                        max_active_iterations_per_control=int(z["iterations"].max()),
                    )
                dynamics.append(row)
                if "raw_peak_pa" in z:
                    np.testing.assert_array_equal(z["raw_peak_pa"], z["peaks_pa"].max(axis=0))
                    for env in range(len(z["raw_peak_pa"])):
                        field = z[f"peak_field_env{env}"]
                        tensor = field if field.shape[-2:] == (3, 3) else unpack(field)
                        dev = tensor - np.trace(tensor, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
                        measured = np.sqrt(1.5 * (dev * dev).sum(axis=(-1, -2))).max()
                        np.testing.assert_allclose(measured, z["raw_peak_pa"][env], rtol=3e-6, atol=1e-4)
                        peaks.append(
                            {
                                "configuration": path.parent.name,
                                "environment": env,
                                **{
                                    key: z[key][env].item()
                                    for key in (
                                        "raw_peak_pa",
                                        "raw_peak_sample_step",
                                        "raw_peak_time_s",
                                        "raw_peak_phase",
                                        "raw_peak_primitive",
                                    )
                                },
                                **{
                                    f"{frame}_{axis}_m": float(z[f"raw_peak_{frame}_position_m"][env, i])
                                    for frame in ("world", "material")
                                    for i, axis in enumerate("xyz")
                                },
                            }
                        )
    table(out, "dynamics", dynamics)
    table(out, "raw-peaks", peaks)

    old = "full-shell-history-v1/group0/default32-sub10/trajectory-repeat0.npz"
    with arrays(old) as z:
        baseline_positions, baseline_peaks, baseline_phases = (
            z["positions"].copy(),
            z["peaks_pa"].copy(),
            z["phases"].copy(),
        )
    restarts = []
    for path in sorted(root.joinpath("fresh-restarts-v1").glob("group*/*/trajectory-repeat0.npz")):
        with arrays(path.relative_to(root)) as z:
            assert np.array_equal(z["phases"], baseline_phases)
            restarts.append(
                {
                    "run": path.parent.name,
                    "max_position_difference_m": float(np.abs(z["positions"] - baseline_positions).max()),
                    "peak_history_relative_difference": relative(z["peaks_pa"], baseline_peaks),
                    "maximum_peak_pa": float(z["peaks_pa"].max()),
                }
            )

    controls = []
    for campaign in ("diagnostic-controls-v2", "press-controls-v3"):
        for path in sorted(root.joinpath(campaign).glob("group*/*/result.json")):
            d = load(path.relative_to(root))
            with arrays(path.parent.relative_to(root) / "trajectory.npz") as z:
                residual = z["residuals"]
                controls.append(
                    {
                        "configuration": path.parent.name,
                        "peak_pa": float(z["peak_pa"].max()),
                        "cap_exhausted_substeps": int(z["caps"].sum()),
                        "max_true_block_jacobi_relative": float(residual[..., 1].max()),
                        "recursive_budget_pass_fraction": float((residual[..., 2] <= 1).mean()),
                        "max_active_iterations_per_control": int(z["iterations"].max()),
                        "total_vertex_impulse_n_s": z["impulses"].sum(axis=(0, 1, 2)).tolist(),
                        "final_sphere_x_displacement_m": (
                            z["sphere_positions"][-1, :, 0] - z["sphere_positions"][0, :, 0]
                        ).tolist(),
                    }
                )
    contact = load("contact-audit-v1/group0/audit/result.json")
    geometry = []
    for audit in contact["geometry"]:
        for env in sorted({sample["env"] for sample in audit["samples"]}):
            rows = [sample for sample in audit["samples"] if sample["env"] == env and "geometry_failure" not in sample]
            misses = [
                s
                for s in rows
                if s["finite_thickness_overlap_m"] > 1e-5
                and s["vertex_candidates_at_recorded_pose"] == 0
                and s["measured_nonzero_impulse_vertices"] == 0
            ]
            geometry.append(
                {
                    "source": audit["source"]["path"],
                    "env": env,
                    "sampled_overlap_without_vertex_candidate_or_impulse_count": len(misses),
                    "max_overlap_without_candidate_m": max(
                        (s["finite_thickness_overlap_m"] for s in misses), default=0
                    ),
                    "minimum_oriented_closest_side_m": min(s["oriented_closest_side_distance_m"] for s in rows),
                    "maximum_finite_thickness_overlap_m": max(s["finite_thickness_overlap_m"] for s in rows),
                }
            )
    table(out, "contact-geometry", geometry)
    table(out, "contact-controls", controls)

    sphere = []
    static = []
    for level in (1, 2):
        group = level - 1
        with arrays(f"prepare-sphere-v1/group0/inputs/level{level}.npz") as z:
            independent, analytic = z["independent_corner_tensor_pa"], z["analytic_corner_tensor_pa"]
            target = float(z["thin_tangent_sigma_pa"])
        with arrays(f"analytic-sphere-v1/group{group}/native-level{level}/fields.npz") as z:
            native = unpack(z["tensor_pa"])
        with arrays(f"analytic-sphere-v1/group{group}/shell-level{level}/fields.npz") as z:
            triangles = z["rest_vertices"][z["faces"]]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            area = np.linalg.norm(normals, axis=1) / 2
            normals /= 2 * area[:, None]
            projection = np.eye(3) - normals[:, :, None] * normals[:, None, :]
            tangent = np.einsum("fij,fjk,fkl->fil", projection, z["shell_tensor_pa"][:, 1], projection)
            mean_hoop = np.trace(tangent, axis1=1, axis2=2) / 2
            sphere.append(
                {
                    "level": level,
                    "native_vs_independent_relative_error": relative(native, independent),
                    "native_lame_corner_relative_error": relative(native, analytic),
                    "independent_lame_corner_relative_error": relative(independent, analytic),
                    "shell_thin_tangent_tensor_relative_error": float(
                        np.sqrt(
                            np.sum(area[:, None, None] * (tangent - target * projection) ** 2)
                            / np.sum(area[:, None, None] * (target * projection) ** 2)
                        )
                    ),
                    "shell_area_mean_hoop_pa": float(area @ mean_hoop / area.sum()),
                    "thin_target_pa": target,
                }
            )

        mesh_path = f"prepare-inputs-v2/group0/matched-inputs/egg-level{level}.npz"
        with arrays(mesh_path) as z:
            reference = SkfemElasticReference(z["vertices"], z["tetrahedra"], 1e10, 0.3, 2000.0)
        with arrays(f"prepare-loads-v2/group0/inputs/level{level}.npz") as z:
            solution = reference.solve(z["volume_force_n"])
        assert solution.residual_budget_fraction <= 1
        with arrays(f"static-cost-v1/group{group}/native-level{level}/fields.npz") as z:
            native_static_error = relative(unpack(z["tensor_pa"]), solution.tensor_pa)
        for model in ("native", "shell"):
            d = load(f"static-cost-v1/group{group}/{model}-level{level}/result.json")
            costs = [1000 * v["seconds"] / v["problems"] for v in d["repeats"]]
            static.append(
                {
                    "configuration": f"{model}-level{level}",
                    "level": level,
                    "method": d["method"]["method"],
                    "median_ms": float(np.median(costs)),
                    "min_ms": min(costs),
                    "max_ms": max(costs),
                    "native_same_mesh_full_tensor_error": native_static_error if model == "native" else None,
                }
            )
    for label in ("default-direct", "inverse-cap1g"):
        d = load(f"static-inverse-tuning-v1/group0/{label}/result.json")
        costs = [1000 * v["seconds"] / v["problems"] for v in d["repeats"]]
        static.append(
            {
                "configuration": label,
                "level": 2,
                "method": d["method"]["method"],
                "median_ms": float(np.median(costs)),
                "min_ms": min(costs),
                "max_ms": max(costs),
                "setup_s": d["method"].get("setup_seconds_in_this_process"),
                "logical_inverse_payload_bytes": d["method"].get("logical_full_inverse_payload_bytes"),
            }
        )
    table(out, "sphere", sphere)
    table(out, "static-cost", static)
    summary = {
        "kind": "executed-shell-comparison",
        "rates": rates,
        "rate_summary": rate_summary,
        "memory": memory,
        "profiles": profiles,
        "dynamics": dynamics,
        "raw_peaks": peaks,
        "fresh_restarts": restarts,
        "controls": controls,
        "contact_geometry": geometry,
        "contact_ledger": contact["ledger"],
        "sphere": sphere,
        "static_cost": static,
        "compatible_fields": load("analyse-prescribed-v1/group0/audit/result.json")["cases"],
        "cache_ablation": load("cache-ablation-v2/group0/ablation/result.json")["checks"],
        "sources": sources,
    }
    # Compact, explicitly derived plotting inputs. Original full arrays remain
    # immutable on data and are covered by sources above; these are not raw copies.
    plot_paths = [
        ("prepare-loads-v2/group0/inputs/level2.npz", ["cloud_points_m", "cloud_forces_n", "cloud_areas_m2"]),
        (
            "analyse-prescribed-v1/group0/audit/level2-common-fields.npz",
            ["positions_m", "shell_tensor_pa", "reference_tensor_pa"],
        ),
        ("contact-audit-v1/group0/audit/impulse-energy-ledger.npz", ["total_energy_j", "initial_total_energy_j"]),
    ]
    for path in root.glob("diagnostic-controls-v2/group*/*/trajectory.npz"):
        plot_paths.append((str(path.relative_to(root)), ["peak_pa", "residuals", "impulses"]))
    for path in root.glob("press-controls-v3/group*/*/trajectory.npz"):
        plot_paths.append((str(path.relative_to(root)), ["impulses"]))
    for campaign in ("full-shell-history-v1", "fresh-restarts-v1", "peak-traces-v2"):
        for path in root.joinpath(campaign).glob("group*/*/trajectory-repeat0.npz"):
            plot_paths.append((str(path.relative_to(root)), ["peaks_pa"]))
    for path, keys in plot_paths:
        target = out / "plot-inputs" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with arrays(path) as z:
            selected = {key: z[key] for key in keys}
            if "impulses" in selected:
                # Collapse vertices only; preserve environment and vector axes.
                selected["impulses"] = selected["impulses"].sum(axis=1, keepdims=True)
            np.savez_compressed(target, **selected)
    target = out / "plot-inputs/contact-audit-v1/group0/audit/result.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, contact)
    write_json(out / "result.json", summary)


if __name__ == "__main__":
    main()
