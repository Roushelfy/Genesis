"""Replay actual rigid inputs through native grouped sharp-edge pressure and force updates."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd
import torch
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import StressContactState, create_contacts
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.hard_binding import StressHardBinding
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions, StressHertzRegionState
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import create_sources, kernel_resolve_sources
from genesis.engine.solvers.rigid.stress.surface import (
    StressSurface,
    StressSurfaceInfo,
    func_surface_geometry,
    func_surface_height_derivatives,
)
from genesis.engine.solvers.rigid.stress.surface_walk import func_projected_surface_walk
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils import geom
from genesis.utils.array_class import V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array
from .check_live_curved_hertz_franka import hard_snapshot as snapshot


@qd.kernel
def kernel_smooth_geometry_diagnostic(
    output: qd.Tensor,
    contacts: StressContactState,
    regions: StressHertzRegionState,
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i_c, i_b in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1]):
        if contacts.valid[i_c, i_b] and regions.opponent_type[i_c, i_b] == gs.GEOM_TYPE.PLANE:
            rotation = contacts.frame_quaternion[i_b]
            root = geom.qd_inv_transform_by_quat(regions.center[i_c, i_b] - contacts.frame_position[i_b], rotation)
            normal = geom.qd_inv_transform_by_quat(regions.normal[i_c, i_b], rotation)
            tangent = geom.qd_inv_transform_by_quat(regions.tangent[i_c, i_b], rotation)
            plane = qd.Matrix.rows([tangent, normal.cross(tangent)])
            current, point = contacts.anchor_face[i_c, i_b], contacts.center[i_c, i_b]
            offset = root - point
            is_active = True
            for iteration in range(9):
                if is_active:
                    current, bary, point = func_projected_surface_walk(
                        current, point, offset, normal, info, surface, epsilon
                    )
                    output[i_c, i_b][3] = current
                    if current < 0:
                        is_active = False
                    else:
                        located, derivative, area = func_surface_geometry(current, bary, info)
                        gradient, hessian, regular = func_surface_height_derivatives(
                            current, bary, plane, normal, info, epsilon
                        )
                        error = (area.normalized() - normal).norm()
                        if iteration == 0:
                            output[i_c, i_b][0], output[i_c, i_b][1], output[i_c, i_b][2] = (
                                current,
                                error,
                                gradient.norm(),
                            )
                        output[i_c, i_b][4], output[i_c, i_b][5], output[i_c, i_b][6] = (
                            error,
                            gradient.norm(),
                            (point - root).norm(),
                        )
                        output[i_c, i_b][7], output[i_c, i_b][8], output[i_c, i_b][9] = (
                            iteration,
                            hessian.trace(),
                            hessian.determinant(),
                        )
                        output[i_c, i_b][10], output[i_c, i_b][11] = (
                            contacts.anchor_face[i_c, i_b],
                            regions.status[i_c, i_b],
                        )
                        is_active = (
                            regular and hessian.determinant() > 0 and hessian.trace() > 0 and gradient.norm() > 1e-10
                        )
                        if is_active:
                            offset = -plane.transpose() @ (hessian.inverse() @ gradient)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--duplicate-active", action="store_true")
    cli.add_argument("--transform-world", action="store_true")
    cli.add_argument("--geometry-only", action="store_true")
    cli.add_argument("--require-admission", action="store_true")
    cli.add_argument("--timing-steps", type=int, default=0)
    cli.add_argument("--timing-repeats", type=int, default=5)
    cli.add_argument("--profile", action="store_true")
    for name in ["frame", "boundaries", "geometry", "raw-contacts", "mesh", "reference", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "duplicate_active_environment_for_isolation": args.duplicate_active,
            "transformed_world_frames": args.transform_world,
            "geometry_diagnostic_only": args.geometry_only,
            "inputs": [
                file_id(p)
                for p in [args.frame, args.boundaries, args.geometry, args.raw_contacts, args.mesh, args.reference]
            ],
            "targets": {"center_m": 1e-9, "force_relative": 1e-5, "support_relative": 5e-4, "wrench_relative": 1e-8},
            "scope": "Captured actual source provenance, current finite-face roots and native material anchoring feed grouped pressure laws and three native force/support feedback updates. Compare the independent grouped offline result in environment0; retain every environment status. Merged collision-sample invariance, original total wrench, friction cones, unload/isolation and restoration are checked. This is input conformance, not recovered stress or accepted full-scene throughput.",
        },
    )
    data = []
    for path in [args.frame, args.boundaries, args.geometry, args.raw_contacts]:
        with np.load(path) as saved:
            data.append({key: saved[key] for key in saved.files})
    frame, bounds, geometry, raw = data
    batch, count = frame["contact_valid"].shape
    if args.duplicate_active:
        for key in [
            "contact_valid",
            "contact_force_n",
            "contact_position_m",
            "contact_normal",
            "source_witness_m",
            "source_geom",
            "source_pair",
            "frame_position_m",
            "frame_quaternion",
            "region_center_m",
            "region_curvature_per_m",
            "region_status",
        ]:
            frame[key] = np.repeat(frame[key][:1], batch, axis=0)
        for key in ["normal", "tangent", "planes", "count"]:
            bounds[key] = np.repeat(bounds[key][:1], batch, axis=0)
        for key in ["sort_idx", "geom_a", "geom_b", "friction"]:
            raw[key] = np.repeat(raw[key][:1], batch, axis=0)
    if args.transform_world:
        assert args.duplicate_active and batch == 4
        rotations = Rotation.from_rotvec([[0, 0, 0], [0.3, -0.5, 0.2], [-0.7, 0.2, 1.1], [1.3, -0.8, 0.6]])
        translations = np.array([[0, 0, 0], [0.17, -0.23, 0.41], [-0.08, 0.32, 0.55], [0.73, -0.42, -0.16]])
        current = Rotation.from_quat(frame["frame_quaternion"][:, [1, 2, 3, 0]])
        frame["frame_quaternion"] = (rotations * current).as_quat()[:, [3, 0, 1, 2]]
        frame["frame_position_m"] = rotations.apply(frame["frame_position_m"]) + translations
        frame["region_center_m"] = np.einsum("bij,bcj->bci", rotations.as_matrix(), frame["region_center_m"])
        frame["region_center_m"] += translations[:, None]
        for key in ["normal", "tangent"]:
            bounds[key] = np.einsum("bij,bcj->bci", rotations.as_matrix(), bounds[key])
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    surface = StressSurface(4, model.info)
    np.testing.assert_array_equal(array(model.info.surface_nodes), frame["material_surface_nodes"])
    contacts, sources = create_contacts(count, batch, 0.001), create_sources(count, batch)
    for field, key in [
        (contacts.force, "contact_force_n"),
        (contacts.position, "contact_position_m"),
        (contacts.normal, "contact_normal"),
        (sources.witness, "source_witness_m"),
    ]:
        field.from_numpy(frame[key].transpose(1, 0, 2))
    for field, key in [(contacts.valid, "contact_valid"), (sources.geom, "source_geom"), (sources.pair, "source_pair")]:
        field.from_numpy(frame[key].T)
    contacts.frame_position.from_numpy(frame["frame_position_m"])
    contacts.frame_quaternion.from_numpy(frame["frame_quaternion"])
    indices = raw["sort_idx"][:, :count]
    envs = np.arange(batch)[:, None]
    first, second = raw["geom_a"][envs, indices], raw["geom_b"][envs, indices]
    other = np.where(first == frame["source_geom"], second, first)
    opponent_type = geometry["geom_type"][np.maximum(other, 0)]
    opponent_type[~frame["contact_valid"]] = -1
    contacts.friction.from_numpy(raw["friction"][envs, indices].T)
    region = StressHertzRegions(contacts).state
    for field, values in [
        (region.center, frame["region_center_m"]),
        (region.normal, bounds["normal"]),
        (region.tangent, bounds["tangent"]),
        (region.curvature, frame["region_curvature_per_m"]),
        (region.boundary_planes, bounds["planes"]),
    ]:
        field.from_numpy(np.swapaxes(values, 0, 1))
    region.status.from_numpy(frame["region_status"].T)
    region.n_boundaries.from_numpy(bounds["count"].T)
    region.opponent_type.from_numpy(opponent_type.T)
    epsilon = float(np.finfo(gs.np_float).eps)
    kernel_resolve_sources(1e-4, epsilon, contacts, sources, model.info, surface.info)
    module = StressHardBinding(contacts, model.info)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    result = snapshot(module)
    if args.timing_steps:
        assert result["status"][0] == 0 and args.timing_steps > 0 and args.timing_repeats > 0
        if args.profile:
            qd.profiler.clear_kernel_profiler_info()
        samples = []
        for repeat in range(args.timing_repeats):
            qd.sync()
            begun = perf_counter()
            for step in range(args.timing_steps):
                module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
            qd.sync()
            samples.append((perf_counter() - begun) / args.timing_steps)
        if args.profile:
            qd.profiler.print_kernel_profiler_info()
        repeated = snapshot(module)
        for key in result:
            np.testing.assert_allclose(repeated[key], result[key], rtol=1e-9, atol=1e-12, equal_nan=True)
        write_json(
            args.output / "timing.json",
            {
                "seconds_per_update": samples,
                "median_seconds": float(np.median(samples)),
                "batch": batch,
                "slots_per_environment": count,
                "steps": args.timing_steps,
                "repeats": args.timing_repeats,
                "instrumented": args.profile,
                "status": result["status"].tolist(),
                "scope": "Warmed resident hard-input update, including dispatch, geometry, grouping, pressure, force allocation and native acceptance; input snapshot and final readbacks outside timing. Precomputed collision regions/source resolution excluded. Repeated fixed captured inputs, not moving-scene or complete stress throughput.",
            },
        )
    np.savez_compressed(args.output / "binding.npz", **result)
    write_json(
        args.output / "numerical.json",
        {
            key: result[key].tolist()
            for key in [
                "active",
                "status",
                "geometry_status",
                "profile",
                "leader",
                "source_count",
                "force_error",
                "moment_error",
                "cone_margin",
                "allocation_status",
                "maximum_change",
            ]
        },
    )
    if args.geometry_only:
        if args.require_admission:
            assert result["status"][0] == 0, result["status"]
        diagnostic = V_VEC(12, dtype=gs.qd_float, shape=(count, batch))
        diagnostic.fill(float("nan"))
        kernel_smooth_geometry_diagnostic(diagnostic, contacts, region, model.info, surface.info, epsilon)
        values = array(diagnostic)
        np.savez_compressed(args.output / "smooth-geometry.npz", diagnostic=values)
        selected = np.isfinite(values[:, :, 3])
        write_json(
            args.output / "result.json",
            {
                "passed": True,
                "diagnostic_only": True,
                "native_status": result["status"].tolist(),
                "columns": [
                    "initial_anchor",
                    "initial_normal_error",
                    "initial_gradient",
                    "final_anchor",
                    "final_normal_error",
                    "final_gradient",
                    "root_shift_m",
                    "iterations",
                    "hessian_trace",
                    "hessian_determinant",
                    "source_anchor",
                    "region_status",
                ],
                "slots": np.argwhere(selected).tolist(),
                "diagnostics": np.where(np.isfinite(values[selected]), values[selected], None).tolist(),
                "whole_goal_complete": False,
            },
        )
        return
    assert result["status"][0] == 0, result["status"]
    references = [row for row in json.loads(args.reference.read_text())["cases"] if row["kind"] == "allocated"]
    expected_leaders = sorted(row["contact"] for row in references)
    np.testing.assert_array_equal(np.flatnonzero(result["source_count"][0]), expected_leaders)
    if args.transform_world:
        for env in range(1, batch):
            np.testing.assert_array_equal(result["source_count"][env], result["source_count"][0])
            np.testing.assert_allclose(result["force_n"][env], result["force_n"][0], rtol=1e-8, atol=1e-12)
            for key in ["center_m", "basis", "axes_m", "amplitude_pa"]:
                first = np.array(expected_leaders) * batch
                np.testing.assert_allclose(result[key][first + env], result[key][first], rtol=1e-6, atol=1e-12)
    rows = []
    for row in references:
        contact = row["contact"]
        index = contact * batch
        np.testing.assert_array_equal(np.flatnonzero(result["leader"][0] == contact), row.get("contacts", [contact]))
        reference_force = np.array(row["basis"]) @ row["force"]
        force_error = np.linalg.norm(result["force_n"][0, contact] - reference_force) / np.linalg.norm(reference_force)
        center_error = np.linalg.norm(result["center_m"][index] - row["center"])
        axes_error = np.max(np.abs(result["axes_m"][index] / [row["along"], row["width"]] - 1))
        amplitude_error = abs(result["amplitude_pa"][index] / row["amplitude"] - 1)
        rows.append(
            {
                "contact": contact,
                "center_error_m": float(center_error),
                "force_relative": float(force_error),
                "axes_relative": float(axes_error),
                "amplitude_relative": float(amplitude_error),
            }
        )
    write_json(args.output / "comparison.json", {"rows": rows})
    assert max(row["center_error_m"] for row in rows) < 1e-9, rows
    assert max(row["force_relative"] for row in rows) < 1e-5, rows
    assert max(max(row["axes_relative"], row["amplitude_relative"]) for row in rows) < 5e-4, rows
    wrench_rows = []
    for env in np.flatnonzero(result["status"] == 0):
        leaders = np.flatnonzero(result["source_count"][env] > 0)
        selected = leaders * batch + env
        forces = result["force_n"][env, leaders]
        raw_force, raw_position = frame["contact_force_n"][env], frame["contact_position_m"][env]
        active = frame["contact_valid"][env] & (np.linalg.norm(raw_force, axis=1) > 0)
        force_error = (
            np.linalg.norm(forces.sum(axis=0) - raw_force[active].sum(axis=0))
            / np.linalg.norm(raw_force[active], axis=1).sum()
        )
        moment = np.cross(result["center_m"][selected], forces).sum(axis=0)
        original = np.cross(raw_position[active], raw_force[active]).sum(axis=0)
        moment_error = np.linalg.norm(moment - original) / np.sum(
            result["normal_force_n"][selected] * result["axes_m"][selected, 0]
        )
        assert max(force_error, moment_error) < 1e-8
        wrench_rows.append(
            {"environment": int(env), "force_error": float(force_error), "moment_error": float(moment_error)}
        )
    # Merge the two samples of each refined gripper region and retain their free couple.
    merged_force, merged_valid = frame["contact_force_n"].copy(), frame["contact_valid"].copy()
    couples = np.zeros_like(merged_force)
    for row in references:
        members = row.get("contacts", [row["contact"]])
        leader = row["contact"]
        force = frame["contact_force_n"][0, members]
        position = frame["contact_position_m"][0, members]
        merged_force[0, leader] = force.sum(axis=0)
        couples[0, leader] = np.cross(position - frame["contact_position_m"][0, leader], force).sum(axis=0)
        for contact in members[1:]:
            merged_force[0, contact], merged_valid[0, contact] = 0, False
    contacts.force.from_numpy(merged_force.transpose(1, 0, 2))
    contacts.valid.from_numpy(merged_valid.T)
    sources.free_moment.from_numpy(couples.transpose(1, 0, 2))
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    merged = snapshot(module)
    np.testing.assert_array_equal(merged["status"], result["status"])
    np.testing.assert_allclose(
        merged["force_n"][0, expected_leaders], result["force_n"][0, expected_leaders], rtol=1e-9, atol=1e-13
    )
    for key in ["center_m", "axes_m", "amplitude_pa"]:
        np.testing.assert_allclose(
            merged[key][np.array(expected_leaders) * batch],
            result[key][np.array(expected_leaders) * batch],
            rtol=1e-9,
            atol=1e-13,
        )
    merged_valid[0] = False
    contacts.valid.from_numpy(merged_valid.T)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    unloaded = snapshot(module)
    assert unloaded["status"][0] == 1 and not unloaded["active"][0]
    np.testing.assert_array_equal(unloaded["source_count"][0], 0)
    np.testing.assert_array_equal(unloaded["force_n"][1:], result["force_n"][1:])
    contacts.valid.from_numpy(frame["contact_valid"].T)
    contacts.force.from_numpy(frame["contact_force_n"].transpose(1, 0, 2))
    sources.free_moment.fill(0)
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    restored = snapshot(module)
    np.testing.assert_array_equal(restored["status"], result["status"])
    np.testing.assert_array_equal(restored["force_n"], result["force_n"])
    module.reset(torch.tensor([0], dtype=gs.tc_int, device=gs.device))
    reset = snapshot(module)
    assert reset["status"][0] == 1 and not reset["active"][0]
    np.testing.assert_array_equal(reset["force_n"][0], 0)
    np.testing.assert_array_equal(reset["force_n"][1:], result["force_n"][1:])
    np.testing.assert_array_equal(reset["status"][1:], result["status"][1:])
    module.update(contacts, sources, region, model.info, surface.info, 1e10 / 0.91, 0.3, epsilon)
    np.testing.assert_array_equal(array(module.state.force), result["force_n"])
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "comparison": rows,
            "wrenches": wrench_rows,
            "environment_status": result["status"].tolist(),
            "merged_source_invariance": True,
            "unload_isolation_and_restore": True,
            "partial_reset_and_reload": True,
            "world_frame_invariance": args.transform_world,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
