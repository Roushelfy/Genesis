"""Measure boundary quadrature convergence on the actual rejected hard-contact frame."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.engine.solvers.rigid.stress.hertz_tangential import StressHertzShear
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ["frame", "hard", "mesh", "output"]:
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--orders", type=int, nargs="+", default=[8, 10, 12, 16, 24, 32])
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(),
            "inputs": [file_id(path) for path in [args.frame, args.hard, args.mesh]],
            "orders": args.orders,
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
    module = StressHertzRecovery(1, count, 1, output_mode="max", moving_probes=False, shear=shear)
    for field, key in [
        (module.frames.center, "hertz_centers_m"),
        (module.frames.basis, "hertz_frames"),
        (module.frames.anchor, "hertz_anchors"),
    ]:
        field.from_numpy(frame[key][selected].copy())
    state = model.create_state(1, "max")
    errno = V(dtype=gs.qd_int, shape=(1,))
    rows, forces = [], []
    for order in args.orders:
        surface = StressSurface(order, model.info)
        kernel_begin_step(state)
        errno.fill(0)
        module.scatter(law.state, state, model.info, surface.info, 0.3, 1e-12, errno)
        error = float(array(module.state.wrench_error)[0])
        if order == 10:
            np.testing.assert_allclose(error, frame["recovery_wrench_error"][0], rtol=1e-8, atol=1e-12)
        rows.append({"order": order, "status": int(array(module.state.status)[0]), "wrench_error": error})
        forces.append(array(state.force)[0])
        write_json(args.output / "progress.json", {"rows": rows})
    for row, force in zip(rows, forces):
        row["force_vector_relative_to_highest_order"] = float(
            np.linalg.norm(force - forces[-1]) / np.linalg.norm(forces[-1])
        )
    np.savez_compressed(args.output / "forces.npz", orders=args.orders, force_n=np.stack(forces), selected=selected)
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
