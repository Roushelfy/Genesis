"""Check actual Franka admission with an explicitly supplied curved material asset."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.contact import StressContactState
from genesis.engine.solvers.rigid.stress.curvature import StressCurvatureInfo, func_surface_curvature
from genesis.engine.solvers.rigid.stress.pair_geometry import func_opponent_boundary
from genesis.engine.solvers.rigid.stress.provenance import StressSourceState
from genesis.utils import array_class, geom

from .artifacts import file_id, runtime_identity, write_json
from .check_live_hertz_franka import capture
from .check_native_hertz_recovery import array


@qd.kernel
def kernel_boundary_diagnostic(
    output: qd.types.ndarray(),
    contacts: StressContactState,
    sources: StressSourceState,
    dyn_state: array_class.DynState,
    collider: array_class.ColliderState,
    dyn_info: array_class.DynInfo,
    curvature: StressCurvatureInfo,
    epsilon: float,
):
    for i_c, i_b, side in qd.ndrange(contacts.valid.shape[0], contacts.valid.shape[1], 2):
        if contacts.valid[i_c, i_b] and contacts.force[i_c, i_b].norm() > 0:
            i_contact = collider.contact_sort_idx[i_c, i_b]
            i_g = sources.geom[i_c, i_b]
            if side == 1:
                other = collider.contact_data.geom_a[i_contact, i_b]
                if other == i_g:
                    other = collider.contact_data.geom_b[i_contact, i_b]
                i_g = other
            seed = geom.qd_transform_by_trans_quat(
                sources.witness[i_c, i_b], contacts.frame_position[i_b], contacts.frame_quaternion[i_b]
            )
            point, normal, gap, face, boundary_status = func_opponent_boundary(
                i_g, i_b, seed, dyn_state, dyn_info, epsilon
            )
            direction, operator, curvature_status = func_surface_curvature(
                i_g, face, i_b, point, dyn_state, dyn_info, curvature, epsilon
            )
            output[i_b, i_c, side, 0] = i_g
            output[i_b, i_c, side, 1] = dyn_info.geoms.type[i_g]
            output[i_b, i_c, side, 2] = face
            output[i_b, i_c, side, 3] = boundary_status
            output[i_b, i_c, side, 4] = curvature_status
            output[i_b, i_c, side, 5] = gap
            for j in qd.static(range(3)):
                output[i_b, i_c, side, 6 + j] = point[j]
                output[i_b, i_c, side, 9 + j] = normal[j]
                output[i_b, i_c, side, 12 + j] = direction[j]


def diagnose(workload, output, step, is_boundary_capture):
    solver = workload.scene.rigid_solver
    entry = solver.stress_recovery.links[0]
    regions = entry.hertz.regions.state
    initial = np.full((4, entry.contacts.valid.shape[0], 2, 15), np.nan)
    kernel_boundary_diagnostic(
        initial,
        entry.contacts,
        entry.hertz.sources,
        solver.dyn_state,
        solver.collider.collider_state,
        solver.dyn_info,
        solver.stress_recovery.curvature.info,
        solver.stress_recovery.source_epsilon,
    )
    np.savez_compressed(
        output / f"geometry{step:04d}.npz",
        initial=initial,
        region_normal=array(regions.normal),
        region_tangent=array(regions.tangent),
        region_gap=array(regions.gap),
        region_residual=array(regions.normal_residual),
        region_geometry_kind=array(regions.geometry_kind),
        region_opponent_link=array(regions.opponent_link),
        geom_type=array(solver.dyn_info.geoms.type),
        geom_link=array(solver.dyn_info.geoms.link_idx),
        geom_data=array(solver.dyn_info.geoms.data),
        geom_position=array(solver.dyn_state.geoms.pos),
        geom_quaternion=array(solver.dyn_state.geoms.quat),
        geom_face_start=array(solver.dyn_info.geoms.face_start),
        geom_face_end=array(solver.dyn_info.geoms.face_end),
        faces=array(solver.dyn_info.faces.verts_idx),
        vertices=array(solver.dyn_info.verts.init_pos),
        normals=array(solver.dyn_info.verts.init_normal),
        curvature_status=array(solver.stress_recovery.curvature.info.status),
        curvature_weight=array(solver.stress_recovery.curvature.info.vertices_weight),
    )
    if is_boundary_capture:
        np.savez_compressed(
            output / f"boundaries{step:04d}.npz",
            planes=array(regions.boundary_planes),
            count=array(regions.n_boundaries),
            normal=array(regions.normal),
            tangent=array(regions.tangent),
        )
    active = array(entry.contacts.valid) & (np.linalg.norm(array(entry.contacts.force), axis=2) > 0)
    write_json(
        output / "geometry.json",
        {
            "step": step,
            "columns": [
                "geom",
                "type",
                "face",
                "boundary_status",
                "curvature_status",
                "gap",
                "point_x",
                "point_y",
                "point_z",
                "normal_x",
                "normal_y",
                "normal_z",
                "smooth_normal_x",
                "smooth_normal_y",
                "smooth_normal_z",
            ],
            "scope": "Initial boundary query at the captured material witness; full region status remains authoritative. No production state is changed.",
            "active": [
                {
                    "environment": int(b),
                    "contact": int(c),
                    "sides": np.where(np.isfinite(initial[b, c]), initial[b, c], None).tolist(),
                }
                for b, c in zip(*np.nonzero(active))
            ],
        },
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stress-mesh", type=Path, required=True)
    cli.add_argument("--steps", type=int, default=30)
    cli.add_argument("--diagnose", action="store_true")
    cli.add_argument("--boundary-planes", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.stress_mesh),
            "steps": args.steps,
            "environments": 4,
            "scope": "Actual moving Franka, original rigid collision surface and controller, supplied quadratic material geometry and native Hertz/friction. Retain every step and stop on the unchanged production error. Loaded rejection does not count as useful throughput, and success alone is not physical stress qualification.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", seed=623016, logging_level="warning")
    workload = FrankaEgg(
        4,
        level=1,
        seed=623016,
        varied=True,
        contact_model="hertz",
        output_mode="full",
        egg_geometry="mesh64",
        use_gjk_collision=False,
        enable_contact_patch=False,
        fused_rigid_post=False,
        stress_mesh=args.stress_mesh,
    )
    workload.stress = False
    workload.save_conditions(args.output / "conditions.npz")
    rows = []
    for step in range(args.steps):
        try:
            workload.step()
        except gs.GenesisException:
            rows.append(capture(workload, args.output, step))
            if args.diagnose:
                write_json(args.output / "result.json", {"passed": False, "rows": rows, "whole_goal_complete": False})
                diagnose(workload, args.output, step, args.boundary_planes)
            raise
        rows.append(capture(workload, args.output, step))
        write_json(args.output / "progress.json", {"rows": rows})
        if args.diagnose and not all(rows[-1]["accepted"]):
            write_json(args.output / "result.json", {"passed": False, "rows": rows, "whole_goal_complete": False})
            diagnose(workload, args.output, step, args.boundary_planes)
            raise AssertionError(f"First invalid stress observation at step {step}")
    assert any(any(row["loaded"]) for row in rows)
    assert all(all(row["accepted"]) for row in rows)
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
