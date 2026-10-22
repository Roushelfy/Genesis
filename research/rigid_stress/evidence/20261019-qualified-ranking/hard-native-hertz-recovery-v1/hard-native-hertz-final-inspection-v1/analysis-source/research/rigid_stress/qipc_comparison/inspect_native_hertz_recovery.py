"""Inspect complete native arrays, independent P2 peaks and paired timings."""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import material_points, remote
from .p2_elastic import FreeElasticP2
from .study_hertz_nearfield import stress_measures
from .study_hertz_residual import elliptic_stress, field_difference


def read(path):
    return json.loads(path.read_text())


def snapshot(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def independent_peaks(control, output, targets):
    data = snapshot(control / "fields.npz")
    mesh = snapshot(control / "mesh.npz")
    cpu = FreeElasticP2(mesh["vertices"], mesh["tetrahedra"], 1e10, 0.3, 2000)
    positions, weights = data["positions_m"], data["volume_weights_m3"]
    nv = len(weights)
    axes, pressure = data["hertz_axes_m"], data["hertz_pressure_pa"]
    forces = 2 * np.pi * axes.prod(axis=1) * pressure / 3
    active = np.flatnonzero(forces > 0)
    block = (len(positions) - nv) // len(active)
    assert block == 90 and len(positions) - nv == block * len(active)
    frames = data["frames"].copy()
    for j, i in enumerate(active):
        vector = positions[nv + block * j] - data["centers_m"][i]
        projected = data["frames"][i].T @ vector
        angle = np.arctan2(projected[1], projected[0])
        q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        frames[i] = data["frames"][i] @ q
        theta = np.arange(32) * 2 * np.pi / 32
        predicted = (
            np.column_stack([axes[i, 0] * np.cos(theta), axes[i, 1] * np.sin(theta), np.zeros(32)]) @ frames[i].T
            + data["centers_m"][i]
        )
        assert np.max(np.abs(predicted - positions[nv + j * block : nv + j * block + 32])) < 1e-12
    fields, corners, rows = [], [], []
    for env in range(4):
        load = np.zeros_like(cpu.vertices)
        for i in [env, env + 4]:
            if forces[i] > 0:
                load += remote(cpu, data["centers_m"][i], frames[i], forces[i], axes[i], pressure[i], True, 12)
        regular, u, _, residual = cpu.solve(load)
        field = np.einsum("nc,ncij->nij", data["query_barycentric"], regular[data["query_elements"]])
        corner = regular.copy()
        for i in [env, env + 4]:
            if forces[i] > 0:
                particular = elliptic_stress(
                    material_points(positions, data["centers_m"][i], frames[i]), axes[i], peak=pressure[i], order=96
                )
                field += frames[i] @ particular @ frames[i].T
                particular = elliptic_stress(
                    material_points(cpu.reference.corner_positions.reshape(-1, 3), data["centers_m"][i], frames[i]),
                    axes[i],
                    peak=pressure[i],
                    order=96,
                ).reshape(corner.shape)
                corner += frames[i] @ particular @ frames[i].T
        fields.append(field)
        corners.append(corner)
        error = np.linalg.norm(data["tensor_pa"][env] - field) / max(np.linalg.norm(field), 1)
        volume = (
            field_difference(data["tensor_pa"][env, :nv], field[:nv], weights) if forces[[env, env + 4]].sum() else 0.0
        )
        tensile, vm = stress_measures(np.concatenate([corner.reshape(-1, 3, 3), field]))
        vm_error = abs(data["native_vm_peak_pa"][env] - vm.max()) / max(abs(vm.max()), 1)
        tensile_error = abs(data["native_tensile_peak_pa"][env] - tensile.max()) / max(abs(tensile.max()), 1)
        row = {
            "env": env,
            "higher_rule_complete_field_relative": float(error),
            "higher_rule_volume_relative": float(volume),
            "higher_rule_vm_peak_relative": float(vm_error),
            "higher_rule_tensile_peak_relative": float(tensile_error),
            "reference_vm_pa": float(vm.max()),
            "reference_tensile_pa": float(tensile.max()),
            "independent_elastic_relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(load), 1e-15)),
        }
        assert error < targets["reference_field"] and volume < targets["reference_field"]
        assert vm_error < targets["reference_peak"] and tensile_error < targets["reference_peak"], row
        rows.append(row)
    path = output / (control.parent.name + "-" + control.name + "-independent-fields.npz")
    np.savez_compressed(path, tensor_pa=np.stack(fields), corner_tensor_pa=np.stack(corners))
    return {"control": str(control), "rows": rows, "saved_reference": file_id(path)}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--controls", type=Path, required=True)
    cli.add_argument("--paired", type=Path, required=True)
    cli.add_argument("--kernel", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    targets = {
        "same_algorithm": 1e-7,
        "reference_field": 1e-3,
        "reference_peak": 1e-2,
        "far_model_change": 1e-3,
        "cpu_gpu": 1e-7,
    }
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "targets": targets,
            "scope": "Read-only complete-array/reduction/covariance and every-frame paired audit. Additional independent prescribed-Hertz P2 reference uses96-point tensor and12-order boundary integration, with shared coarse material geometry; this is not continuum or independent unknown-pressure qualification.",
        },
    )
    for campaign in [args.controls, args.paired, args.kernel]:
        for p in campaign.glob("group*/returncode.txt"):
            assert p.read_text().strip() == "0"
    bindings = []
    reductions = []
    for group in [0, 1]:
        for name in ["exact", "far", "rotated-far"]:
            root = args.controls / f"group{group}" / name
            report = read(root / "result.json")
            assert report["passed"]
            data = snapshot(root / "fields.npz")
            reference = snapshot(root / "reference-fields.npz")
            weights = data["volume_weights_m3"]
            for env in range(4):
                observed = data["tensor_pa"][env]
                expected = reference["expected_tensor_pa"][env]
                error = np.linalg.norm(observed - expected) / max(np.linalg.norm(expected), 1)
                assert error < targets["same_algorithm"]
                tension, vm = stress_measures(
                    np.concatenate([data["corner_tensor_pa"][env].reshape(-1, 3, 3), observed])
                )
                np.testing.assert_allclose(data["native_vm_peak_pa"][env], vm.max(), rtol=1e-12, atol=1e-8)
                np.testing.assert_allclose(data["native_tensile_peak_pa"][env], tension.max(), rtol=1e-12, atol=1e-8)
            assert np.array_equal(data["native_status"], [0, 0, 0, 1]) and not np.any(data["native_errno"])
            assert np.all(data["native_elastic_valid"]) and np.all(data["native_step_valid"])
            bindings.extend(file_id(root / name) for name in ["fields.npz", "reference-fields.npz", "result.json"])
            reductions.append(independent_peaks(root, args.output, targets))
            write_json(args.output / "progress.json", {"independent_prescribed_load_checks": reductions})
    changes = []
    for group in [0, 1]:
        exact = snapshot(args.controls / f"group{group}/exact/fields.npz")
        fast = snapshot(args.controls / f"group{group}/far/fields.npz")
        for env in range(3):
            nv = len(exact["volume_weights_m3"])
            volume = field_difference(
                fast["tensor_pa"][env, :nv], exact["tensor_pa"][env, :nv], exact["volume_weights_m3"]
            )
            vm = abs(fast["native_vm_peak_pa"][env] / exact["native_vm_peak_pa"][env] - 1)
            tensile = abs(fast["native_tensile_peak_pa"][env] / exact["native_tensile_peak_pa"][env] - 1)
            assert max(volume, vm, tensile) < targets["far_model_change"]
            changes.append(
                {
                    "backend_group": group,
                    "env": env,
                    "volume_change": float(volume),
                    "vm_change": float(vm),
                    "tensile_change": float(tensile),
                }
            )
    cpu_gpu = []
    for name in ["exact", "far", "rotated-far"]:
        first = snapshot(args.controls / f"group0/{name}/fields.npz")
        second = snapshot(args.controls / f"group1/{name}/fields.npz")
        same = first["query_elements"] == second["query_elements"]
        error = np.linalg.norm(first["tensor_pa"][:, same] - second["tensor_pa"][:, same]) / max(
            np.linalg.norm(first["tensor_pa"][:, same]), 1
        )
        raw = np.linalg.norm(first["tensor_pa"] - second["tensor_pa"]) / max(np.linalg.norm(first["tensor_pa"]), 1)
        corner = np.linalg.norm(first["corner_tensor_pa"] - second["corner_tensor_pa"]) / max(
            np.linalg.norm(first["corner_tensor_pa"]), 1
        )
        assert error < targets["cpu_gpu"]
        assert corner < targets["cpu_gpu"]
        cpu_gpu.append(
            {
                "case": name,
                "same_element_relative": float(error),
                "corner_relative": float(corner),
                "raw_all_query_relative": float(raw),
                "different_element_probe_count": int(np.count_nonzero(~same)),
            }
        )
    rotation_rows = []
    for group in [0, 1]:
        base = snapshot(args.controls / f"group{group}/far/fields.npz")
        moved = snapshot(args.controls / f"group{group}/rotated-far/fields.npz")
        rotation = moved["frames"][0] @ base["frames"][0].T
        np.testing.assert_array_equal(base["query_elements"], moved["query_elements"])
        expected = rotation @ base["tensor_pa"] @ rotation.T
        error = np.linalg.norm(moved["tensor_pa"] - expected) / max(np.linalg.norm(expected), 1)
        assert error < targets["same_algorithm"]
        rotation_rows.append({"backend_group": group, "complete_tensor_rotation_relative": float(error)})
    exact_root, fast_root = [args.paired / "group0" / name for name in ["exact", "far"]]
    assert read(exact_root / "start.json")["identity"] == read(fast_root / "start.json")["identity"]
    timing = []
    for first, second in zip(read(exact_root / "result.json")["rows"], read(fast_root / "result.json")["rows"]):
        batch, mode = first["batch"], first["mode"]
        assert (batch, mode) == (second["batch"], second["mode"])
        for root, row in [(exact_root, first), (fast_root, second)]:
            path = root / f"{mode}-b{batch}-audit.npz"
            data = snapshot(path)
            assert data["status"].shape == (batch, 24) and np.all(data["status"] <= 1) and np.all(data["elastic_valid"])
            assert (
                np.isfinite(data["vm_pa"]).all()
                and np.isfinite(data["tensile_pa"]).all()
                and np.max(data["wrench_error"]) < 1e-4
            )
            expected = (np.arange(batch)[:, None] % 4 != 3) & (
                (np.arange(24)[None] + np.arange(batch)[:, None]) % 13 != 0
            )
            np.testing.assert_array_equal(data["loaded"], expected)
            np.testing.assert_array_equal(data["status"], np.where(expected, 0, 1))
            assert (
                row["admitted_loaded_audit_frames"] == int(expected.sum())
                and row["admitted_all_audit_frames"] == 24 * batch
            )
            assert row["median_seconds"] == float(np.median(row["seconds_per_update"]))
            bindings.append(file_id(path))
        timing.append(
            {
                "batch": batch,
                "mode": mode,
                "exact_seconds": first["median_seconds"],
                "far_seconds": second["median_seconds"],
                "reduction_fraction": 1 - second["median_seconds"] / first["median_seconds"],
                "repeat_ranges_separate": max(second["seconds_per_update"]) < min(first["seconds_per_update"]),
                "admitted_all_audit_frames": second["admitted_all_audit_frames"],
                "admitted_loaded_audit_frames": second["admitted_loaded_audit_frames"],
            }
        )
    for path in args.kernel.glob("group*/check/result.json"):
        assert read(path)["passed"]
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "bindings": bindings,
            "targets": targets,
            "independent_prescribed_load_checks": reductions,
            "far_model_changes": changes,
            "cpu_gpu_changes": cpu_gpu,
            "rotation_covariance": rotation_rows,
            "same_allocation_timings": timing,
            "general_pressure_friction_or_scene_qualified": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
