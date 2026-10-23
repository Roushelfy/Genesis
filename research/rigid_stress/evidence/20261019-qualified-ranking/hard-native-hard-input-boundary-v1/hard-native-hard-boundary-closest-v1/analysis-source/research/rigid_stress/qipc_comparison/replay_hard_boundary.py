"""Measure boundary quadrature convergence on the actual rejected hard-contact frame."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.edge import StressEdge
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.hertz_recovery import (
    StressHertzFrames,
    StressHertzRecovery,
    kernel_hertz_halfspace,
    kernel_hertz_residual_accept,
    kernel_hertz_residual_begin,
    kernel_hertz_residual_traction,
)
from genesis.engine.solvers.rigid.stress.hertz_tangential import StressHertzShear
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.provenance import func_closest_triangle, func_material_source_point
from genesis.engine.solvers.rigid.stress.surface import (
    StressSurface,
    StressSurfaceInfo,
    func_surface_geometry,
    func_surface_projection,
)
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


@qd.kernel
def kernel_boundary_geometry(
    output: qd.Tensor,
    frames: StressHertzFrames,
    info: StressInfo,
    surface: StressSurfaceInfo,
):
    for i_f, i_p in qd.ndrange(info.surface_nodes.shape[0], frames.center.shape[0]):
        center, basis = frames.center[i_p], frames.basis[i_p]
        closest, bary = func_closest_triangle(i_f, center, surface)
        curved, tangent, area = func_surface_geometry(i_f, bary, info)
        projected, projected_bary, projected_area, valid = func_surface_projection(
            i_f, center, basis[:, :2].transpose(), info
        )
        true_closest, true_bary = func_material_source_point(i_f, center, bary, info, surface, 1e-12, False)
        output[i_f, i_p] = qd.Vector(
            [
                (curved - center).norm(),
                (true_closest - center).norm(),
                (projected - center).norm(),
                bary[0],
                bary[1],
                bary[2],
                true_bary[0],
                true_bary[1],
                true_bary[2],
                projected_bary[0],
                projected_bary[1],
                projected_bary[2],
                projected_area.normalized().dot(basis[:, 2]),
                valid,
            ]
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["frame", "hard", "mesh", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--orders", type=int, nargs="+", default=[8, 10, 12, 16, 24, 32])
    cli.add_argument("--refined", action="store_true")
    cli.add_argument("--separate-sources", action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "inputs": [file_id(path) for path in [args.frame, args.hard, args.mesh]],
            "orders": args.orders,
            "refined": args.refined,
            "separate_sources": args.separate_sources,
            "scope": "Actual environment zero, with the exact accepted hard-binding forces, frames and support. Boundary integration only; unchanged 1e-4 wrench gate. No recovered-field or physical accuracy claim.",
        },
    )
    with np.load(args.frame) as archive:
        frame = {key: archive[key] for key in archive.files}
    with np.load(args.hard) as archive:
        hard = {key: archive[key] for key in archive.files}
    batch = hard["source_count"].shape[0]
    assert hard["active"][0] and hard["status"][0] == 0
    selected = np.flatnonzero((frame["law_status"] == 0) & (np.arange(len(frame["law_status"])) % batch == 0))
    assert len(selected) > 1
    count = len(selected)
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    law = StressHertz(count)
    for field, key in [
        (law.state.force, "law_force_n"),
        (law.state.axes, "law_axes_m"),
        (law.state.direction, "law_direction"),
        (law.state.peak_pressure, "law_pressure_pa"),
        (law.state.status, "law_status"),
    ]:
        field.from_numpy(frame[key][selected].copy())
    shear = StressHertzShear(
        friction=V(dtype=gs.qd_float, shape=(count,)),
        force=V_VEC(2, dtype=gs.qd_float, shape=(count,)),
        coefficients=V_VEC(2, dtype=gs.qd_float, shape=(16, count)),
    )
    basis = frame["hertz_frames"][selected]
    direction = frame["law_direction"][selected]
    rotation = np.zeros((count, 3, 3))
    rotation[:, 0, 0], rotation[:, 1, 0] = direction[:, 0], direction[:, 1]
    rotation[:, 0, 1], rotation[:, 1, 1] = -direction[:, 1], direction[:, 0]
    rotation[:, 2, 2] = 1
    force = hard["force_n"][0, selected // batch]
    local = np.einsum("ni,nij->nj", force, basis @ rotation)
    np.testing.assert_allclose(local[:, 2], frame["law_force_n"][selected], rtol=1e-12, atol=1e-14)
    shear.force.from_numpy(local[:, :2].copy())
    edge = StressEdge(count)
    edge.state.status.from_numpy(hard["edge_status"][selected].copy())
    replay_batch = count if args.separate_sources else 1
    module = StressHertzRecovery(
        replay_batch, count // replay_batch, output_mode="max", moving_probes=False, shear=shear, edge=edge.state
    )
    for field, key in [
        (module.frames.center, "hertz_centers_m"),
        (module.frames.basis, "hertz_frames"),
        (module.frames.anchor, "hertz_anchors"),
    ]:
        field.from_numpy(frame[key][selected].copy())
    state = model.create_state(replay_batch, "max")
    errno = V(dtype=gs.qd_int, shape=(replay_batch,))
    geometry = V_VEC(14, dtype=gs.qd_float, shape=(model.info.surface_nodes.shape[0], count))
    rows, forces = [], []
    for order in args.orders:
        surface = StressSurface(order, model.info)
        kernel_begin_step(state)
        errno.fill(0)
        kernel_hertz_residual_begin(
            1e-12, module.frames, law.state, state, module.state, model.info, surface.info, True
        )
        kernel_hertz_halfspace(1e-12, module.frames, law.state, module.state, model.info)
        kernel_hertz_residual_traction(
            0.3, 1e-12, module.frames, law.state, shear, state, module.state, model.info, surface.info, True, True
        )
        force_n = array(state.force)
        pre_accept_status = array(module.state.status)
        kernel_hertz_residual_accept(module.frames, law.state, shear, state, module.state, model.info, True, errno)
        error = array(module.state.wrench_error)
        if order == 10 and not args.separate_sources:
            if args.refined:
                assert error[0] < 1e-4, error
            else:
                np.testing.assert_allclose(error[0], frame["recovery_wrench_error"][0], rtol=1e-8, atol=1e-12)
        rows.append(
            {
                "order": order,
                "status": array(module.state.status).tolist(),
                "wrench_error": error.tolist(),
                "pre_accept_status": pre_accept_status.tolist(),
            }
        )
        forces.append(force_n)
        write_json(args.output / "progress.json", {"rows": rows})
    for row, raw_force in zip(rows, forces):
        row["raw_force_vector_relative_to_highest_order"] = (
            np.linalg.norm(raw_force - forces[-1], axis=(1, 2)) / np.linalg.norm(forces[-1], axis=(1, 2))
        ).tolist()
    kernel_boundary_geometry(geometry, module.frames, model.info, surface.info)
    np.savez_compressed(
        args.output / "forces.npz",
        orders=args.orders,
        force_n=np.stack(forces),
        selected=selected,
        geometry=array(geometry),
        centers_m=frame["hertz_centers_m"][selected],
        anchors=frame["hertz_anchors"][selected],
        vertices_m=array(model.info.vertices),
        surface_nodes=array(model.info.surface_nodes),
        expected_force_n=force,
    )
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
