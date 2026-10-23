"""Exercise public manual/radius-only/Hertz-profile DPR on one unchanged Franka bank."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--conditions", type=Path, required=True)
    cli.add_argument("--mode", choices=["manual", "radius", "hertz"], required=True)
    cli.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    cli.add_argument("--batch", type=int, default=4)
    cli.add_argument("--steps", type=int, default=130)
    cli.add_argument("--unfused", action="store_true")
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
            "unfused": args.unfused,
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
        fused_pipeline=not args.unfused,
        fused_rigid_post=False if args.unfused else None,
    )
    qd.sync()
    preparation = perf_counter() - begun
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    rows, peaks, positions = [], [], []
    failure = None
    for step in range(args.steps):
        error = None
        try:
            workload.step()
        except gs.GenesisException as exception:
            error = str(exception)
        fields = {
            name: qd_to_numpy(vars(entry.contacts)[name], transpose=True, copy=True)
            for name in ["valid", "status", "radius_status", "radius", "force", "normal", "force_error", "moment_error"]
        }
        active = fields["valid"] & (np.linalg.norm(fields["force"], axis=-1) > 1e-30)
        row = {
            "step": step,
            "active_contacts": int(active.sum()),
            "radius_status_counts": dict(
                zip(*[a.tolist() for a in np.unique(fields["radius_status"][active], return_counts=True)])
            ),
            "contact_status_counts": dict(
                zip(*[a.tolist() for a in np.unique(fields["status"][active], return_counts=True)])
            ),
            "radius_range_m": [float(np.min(fields["radius"][active])), float(np.max(fields["radius"][active]))]
            if active.any()
            else None,
            "error": error,
        }
        rows.append(row)
        write_json(args.output / "progress.json", {"rows": rows})
        if error is not None:
            failure = row
            np.savez_compressed(args.output / "failure.npz", **fields)
            break
        peak = tensor_to_array(workload.link.get_max_stress()).copy()
        tensor, vm = workload.link.get_stress_field()
        assert np.isfinite(tensor_to_array(tensor)).all()
        np.testing.assert_allclose(peak, tensor_to_array(vm).max(axis=(1, 2)), rtol=1e-12, atol=1e-8)
        peaks.append(peak)
        positions.append(tensor_to_array(workload.egg.get_pos()).copy())
    np.savez_compressed(args.output / "observations.npz", peaks=np.array(peaks), positions=np.array(positions))
    write_json(
        args.output / "result.json",
        {
            "passed": failure is None,
            "preparation_seconds": preparation,
            "accepted_frames": len(peaks),
            "failure": failure,
            "rows": rows,
            "whole_goal_complete": False,
        },
    )
    if failure is not None:
        raise RuntimeError("Complete DPR scene admission failed; the original refusal is retained.")


if __name__ == "__main__":
    main()
