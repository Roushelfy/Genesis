"""Refine only the original DPR quadrature for retained finite-radius contact failures."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts, kernel_anchor, kernel_pressure, kernel_scatter
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.utils.misc import qd_to_numpy

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--failure", type=Path, required=True)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--orders", type=int, nargs="+", default=[10, 16, 24])
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=False),
            "failure": file_id(args.failure),
            "mesh": file_id(args.mesh),
            "orders": args.orders,
        },
    )
    with np.load(args.failure) as saved:
        data = {name: saved[name] for name in saved.files}
    selected = np.argwhere(data["valid"] & (data["status"] == 3) & np.isfinite(data["radius"]))
    assert len(selected)
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    model = StressModel(gs.options.RigidStressOptions.dpr(mesh=args.mesh, method="direct"))
    rows = []
    for order in args.orders:
        surface = StressSurface(order, model.info)
        for env, slot in selected:
            contacts = create_contacts(1, 1, float(data["radius"][env, slot]))
            contacts.valid.fill(True)
            contacts.friction.fill(float(data["friction"][env, slot]))
            for name in ["position", "force", "normal"]:
                vars(contacts)[name].from_numpy(data[name][env, slot][None, None].copy())
            state = model.create_state(1)
            kernel_anchor(float(np.finfo(float).eps), contacts, surface.info)
            kernel_pressure(contacts, surface.info)
            kernel_scatter(contacts, state, model.info, surface.info)
            fields = {
                name: qd_to_numpy(vars(contacts)[name], transpose=True, copy=True)
                for name in [
                    "status",
                    "candidate_count",
                    "evaluations",
                    "coefficient",
                    "weight_sum",
                    "center",
                    "anchor_face",
                    "force_error",
                    "moment_error",
                ]
            }
            np.savez_compressed(args.output / f"q{order}-e{env}-c{slot}.npz", **fields)
            row = {
                "quadrature": order,
                "environment": int(env),
                "contact": int(slot),
                "status": int(fields["status"][0, 0]),
                "candidates": int(fields["candidate_count"][0, 0]),
                "evaluations": int(fields["evaluations"][0, 0]),
                "radius_m": float(data["radius"][env, slot]),
                "anchor_difference_m": float(np.linalg.norm(fields["center"][0, 0] - data["center"][env, slot])),
            }
            rows.append(row)
            write_json(args.output / "progress.json", {"rows": rows})
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "rows": rows,
            "scope": "Quadrature-only diagnosis of the existing scalar-direction DPR fit. No radius, source force, source point or pressure-solver tolerance is changed. Friction is copied from the captured source.",
        },
    )


if __name__ == "__main__":
    main()
