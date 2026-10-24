"""Exercise public manual/radius-only/Hertz-profile DPR on one unchanged Franka bank."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from examples.speed_benchmark.stress_stages import kernel_restore_geometry
from genesis.engine.solvers.rigid.stress.automatic_radius import func_estimate_radius, func_witness_curvature
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.utils import array_class
from genesis.utils.array_class import V, V_MAT
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def kernel_reestimate_radius(
    selection: qd.types.ndarray(),
    contacts: StressContactState,
    collider: array_class.ColliderState,
    dyn: array_class.DynState,
    info: qd.template(),
):
    for row in range(selection.shape[0]):
        i_b, slot = selection[row, 0], selection[row, 1]
        source = collider.contact_sort_idx[slot, i_b]
        func_estimate_radius(slot, i_b, source, contacts, collider, dyn, info)


@qd.kernel
def kernel_failed_curvature(
    selection: qd.types.ndarray(),
    tensor: qd.Tensor,
    status: qd.Tensor,
    collider: array_class.ColliderState,
    dyn: array_class.DynState,
    info: qd.template(),
):
    for row, side in qd.ndrange(selection.shape[0], 2):
        i_b, slot = selection[row, 0], selection[row, 1]
        source = collider.contact_sort_idx[slot, i_b]
        i_g = collider.contact_data.geom_a[source, i_b]
        i_v = collider.contact_data.witness.vertex_a[source, i_b]
        point = collider.contact_data.witness.point_a[source, i_b]
        if side == 1:
            i_g = collider.contact_data.geom_b[source, i_b]
            i_v = collider.contact_data.witness.vertex_b[source, i_b]
            point = collider.contact_data.witness.point_b[source, i_b]
        value, code = func_witness_curvature(i_g, i_v, i_b, point, dyn, info)
        tensor[row, side] = value
        status[row, side] = code


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--mode", choices=["manual", "radius", "hertz"], required=True)
    cli.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    cli.add_argument("--batch", type=int, default=4)
    cli.add_argument("--steps", type=int, default=130)
    cli.add_argument("--quadrature", type=int, default=10)
    cli.add_argument("--stress-backend", choices=["volume", "shell"], default="volume")
    cli.add_argument("--shell-gauge", choices=["translation", "legacy"], default="translation")
    cli.add_argument("--box-edge-policy", choices=("unsupported", "planar"), default="unsupported")
    cli.add_argument("--unfused", action="store_true")
    cli.add_argument("--contact-moment-reuse", choices=("auto", "on", "off"), default="auto")
    cli.add_argument("--partial-reset-at", type=int)
    cli.add_argument("--field-steps", type=int, nargs="*", default=[])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "conditions": file_id(args.conditions),
            "mode": args.mode,
            "batch": args.batch,
            "steps": args.steps,
            "quadrature": args.quadrature,
            "stress_backend": args.stress_backend,
            "shell_gauge": args.shell_gauge,
            "box_edge_policy": args.box_edge_policy,
            "unfused": args.unfused,
            "contact_moment_reuse": args.contact_moment_reuse,
            "partial_reset_at": args.partial_reset_at,
            "field_steps": args.field_steps,
            "scope": "Actual public DPR path, original elastic mesh and rigid dynamics. Automatic radius and pressure "
            "profile are separate controls. Every environment/frame remains in the audit. Preparation and diagnostic "
            "reads are not throughput measurements. No analytic enrichment or physical accuracy claim.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    begun = perf_counter()
    workload = FrankaEgg(
        args.batch,
        conditions=args.conditions,
        output_mode="full",
        radius_model="manual" if args.mode == "manual" else "hertz",
        pressure_profile="hertz" if args.mode == "hertz" else "dpr",
        quadrature=args.quadrature,
        stress_backend=args.stress_backend,
        shell_gauge=args.shell_gauge,
        box_edge_policy=args.box_edge_policy,
        fused_pipeline=not args.unfused,
        fused_rigid_post=False if args.unfused else None,
        contact_moment_reuse=None if args.contact_moment_reuse == "auto" else args.contact_moment_reuse == "on",
    )
    qd.sync()
    preparation = perf_counter() - begun
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    if (
        args.stress_backend == "volume"
        and args.mode == "hertz"
        and args.backend == "gpu"
        and args.contact_moment_reuse == "auto"
    ):
        assert entry.reuse_contact_moments
    rows, peaks, positions = [], [], []
    failure = None
    reset_checked = False
    for step in range(args.steps):
        if step == args.partial_reset_at:
            previous_peak = tensor_to_array(workload.link.get_max_stress()).copy()
            previous_position = tensor_to_array(workload.egg.get_pos()).copy()
            workload.reset(np.array([0]))
            peak = tensor_to_array(workload.link.get_max_stress())
            position = tensor_to_array(workload.egg.get_pos())
            np.testing.assert_array_equal(peak[1:], previous_peak[1:])
            np.testing.assert_array_equal(position[1:], previous_position[1:])
            assert peak[0] == 0
            reset_checked = True
        error = None
        solver = workload.scene.rigid_solver
        solver.update_forward_pos()
        geometry_position = np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True, copy=True))
        geometry_quaternion = np.ascontiguousarray(qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True, copy=True))
        try:
            workload.step()
            workload.scene.rigid_solver.check_errno()
        except gs.GenesisException as exception:
            error = str(exception)
        fields = {
            name: qd_to_numpy(vars(entry.contacts)[name], transpose=True, copy=True)
            for name in [
                "valid",
                "status",
                "radius_status",
                "radius",
                "force",
                "normal",
                "friction",
                "force_error",
                "moment_error",
                "position",
                "center",
                "anchor_face",
                "is_refined",
                "weight_sum",
                "coefficient",
            ]
        }
        active = fields["valid"] & (np.linalg.norm(fields["force"], axis=-1) > 1e-30)
        finite_radius = active & np.isfinite(fields["radius"])
        row = {
            "step": step,
            "active_contacts": int(active.sum()),
            "radius_status_counts": dict(
                zip(*[a.tolist() for a in np.unique(fields["radius_status"][active], return_counts=True)])
            ),
            "contact_status_counts": dict(
                zip(*[a.tolist() for a in np.unique(fields["status"][active], return_counts=True)])
            ),
            "invalid_radius_contacts": int((active & ~finite_radius).sum()),
            "radius_range_m": [
                float(np.min(fields["radius"][finite_radius])),
                float(np.max(fields["radius"][finite_radius])),
            ]
            if finite_radius.any()
            else None,
            "error": error,
        }
        rows.append(row)
        write_json(args.output / "progress.json", {"rows": rows})
        if error is not None:
            failure = row
            solver = workload.scene.rigid_solver
            collider = solver.collider.collider_state
            fields.update(
                {
                    "source_" + name: qd_to_numpy(vars(collider.contact_data)[name], transpose=True, copy=True)
                    for name in ["geom_a", "geom_b", "normal", "pos", "force"]
                }
            )
            fields.update(
                {
                    "witness_" + name: qd_to_numpy(vars(collider.contact_data.witness)[name], transpose=True, copy=True)
                    for name in ["point_a", "point_b", "vertex_a", "vertex_b"]
                }
            )
            fields["contact_sort_idx"] = qd_to_numpy(collider.contact_sort_idx, transpose=True, copy=True)
            fields["geoms_type"] = qd_to_numpy(solver.dyn_info.geoms.type, copy=True)
            fields["geoms_data"] = qd_to_numpy(solver.dyn_info.geoms.data, copy=True)
            fields["geoms_pos"] = qd_to_numpy(solver.dyn_state.geoms.pos, transpose=True, copy=True)
            fields["geoms_quat"] = qd_to_numpy(solver.dyn_state.geoms.quat, transpose=True, copy=True)
            fields["preintegration_geoms_pos"] = geometry_position
            fields["preintegration_geoms_quat"] = geometry_quaternion
            np.savez_compressed(args.output / "failure.npz", **fields)
            if entry.automatic_radius is not None:
                kernel_restore_geometry(geometry_position, geometry_quaternion, solver.dyn_state)
                selected = np.ascontiguousarray(np.argwhere(active & (fields["status"] != 0)), dtype=gs.np_int)
                curvature = V_MAT(3, 3, dtype=gs.qd_float, shape=(len(selected), 2))
                curvature_status = V(dtype=gs.qd_int, shape=(len(selected), 2))
                kernel_failed_curvature(
                    selected, curvature, curvature_status, collider, solver.dyn_state, entry.automatic_radius
                )
                fields["failed_selection"] = selected
                fields["failed_curvature"] = qd_to_numpy(curvature, copy=True)
                fields["failed_curvature_status"] = qd_to_numpy(curvature_status, copy=True)
                kernel_reestimate_radius(selected, entry.contacts, collider, solver.dyn_state, entry.automatic_radius)
                repeated_radius = qd_to_numpy(entry.contacts.radius, transpose=True, copy=True)
                repeated_status = qd_to_numpy(entry.contacts.radius_status, transpose=True, copy=True)
                diagnostics = []
                for i, (env, slot) in enumerate(selected):
                    source = fields["contact_sort_idx"][env, slot]
                    direction = fields["source_normal"][env, source]
                    direction = direction / np.linalg.norm(direction)
                    projection = np.eye(3) - np.outer(direction, direction)
                    relative = projection @ fields["failed_curvature"][i].sum(axis=0) @ projection
                    diagnostics.append(
                        {
                            "environment": int(env),
                            "contact": int(slot),
                            "frame": "preintegration",
                            "original_radius_status": int(fields["radius_status"][env, slot]),
                            "repeated_radius_status": int(repeated_status[env, slot]),
                            "repeated_radius_m": float(repeated_radius[env, slot])
                            if np.isfinite(repeated_radius[env, slot])
                            else None,
                            "curvature_status": fields["failed_curvature_status"][i].tolist(),
                            "projected_curvature_eigenvalues_per_m": np.linalg.eigvalsh(relative).tolist()
                            if np.isfinite(relative).all()
                            else None,
                        }
                    )
                write_json(args.output / "curvature-diagnostics.json", {"rows": diagnostics})
            np.savez_compressed(args.output / "failure.npz", **fields)
            break
        peak = tensor_to_array(workload.link.get_max_stress()).copy()
        tensor, vm = workload.link.get_stress_field()
        assert np.isfinite(tensor_to_array(tensor)).all()
        np.testing.assert_allclose(
            peak, tensor_to_array(vm).reshape((args.batch, -1)).max(axis=1), rtol=1e-12, atol=1e-8
        )
        if args.stress_backend == "shell":
            resultants = dict(zip(("membrane", "bending", "transverse"), workload.link.get_shell_resultants()))
            for label, value in resultants.items():
                assert np.isfinite(tensor_to_array(value)).all(), label
        if step in args.field_steps:
            if args.stress_backend == "shell":
                fields.update({"shell_" + label: tensor_to_array(value) for label, value in resultants.items()})
            np.savez_compressed(
                args.output / f"field-{step:04d}.npz",
                stress_tensor=tensor_to_array(tensor),
                von_mises=tensor_to_array(vm),
                nodal_force=qd_to_numpy(entry.state.force, transpose=True, copy=True),
                angular_velocity=qd_to_numpy(entry.omega, transpose=True, copy=True),
                displacement=qd_to_numpy(entry.state.displacement, transpose=True, copy=True),
                **fields,
            )
        peaks.append(peak)
        positions.append(tensor_to_array(workload.egg.get_pos()).copy())
    np.savez_compressed(args.output / "observations.npz", peaks=np.array(peaks), positions=np.array(positions))
    write_json(
        args.output / "result.json",
        {
            "passed": failure is None,
            "genesis_module": gs.__file__,
            "effective_contact_moment_reuse": entry.reuse_contact_moments,
            "preparation_seconds": preparation,
            "accepted_frames": len(peaks),
            "failure": failure,
            "partial_reset_checked": reset_checked,
            "rows": rows,
            "whole_goal_complete": False,
        },
    )
    if failure is not None:
        raise RuntimeError("Complete DPR scene admission failed; the original refusal is retained.")


if __name__ == "__main__":
    main()
