"""Measure warmed native material walks with device-resident query arrays."""

import argparse
from pathlib import Path
import time

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import file_id, runtime_identity, write_json
from .check_curved_surface_walk import kernel_walk
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--controls", type=Path, required=True)
    cli.add_argument("--curved", type=Path, required=True)
    cli.add_argument("--affine", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=True),
            "inputs": [file_id(args.curved), file_id(args.affine)]
            + [file_id(args.controls / (kind + "-rotated0.npz")) for kind in ["affine", "curved"]],
            "scope": "Warmed synchronized inclusive wall time for one Quadrants projected-walk dispatch over repeated geometry-control queries. Inputs/outputs stay on GPU during timing; numerical audits are outside. These are geometry component costs, not input-pipeline, friction, full stress or accepted scene throughput.",
        },
    )
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    rows = []
    for kind, asset in [("affine", args.affine), ("curved", args.curved)]:
        started = time.perf_counter()
        model = StressModel(RigidStressOptions(mesh=asset, contact_model="hertz", method="direct"))
        surface = StressSurface(4, model.info)
        qd.sync()
        preparation = time.perf_counter() - started
        with np.load(args.controls / (kind + "-rotated0.npz")) as saved:
            controls = {key: saved[key] for key in ["anchors", "centers", "targets", "normals"]}
        for repeat in [1, 32, 1024]:
            inputs = [
                torch.as_tensor(np.tile(controls[key], (repeat,) + (1,) * (controls[key].ndim - 1)), device=gs.device)
                for key in controls
            ]
            count = len(inputs[0])
            faces, points, bary = (
                V(dtype=gs.qd_int, shape=(count,)),
                V_VEC(3, dtype=gs.qd_float, shape=(count,)),
                V_VEC(3, dtype=gs.qd_float, shape=(count,)),
            )
            for _ in range(3):
                kernel_walk(*inputs, faces, points, bary, model.info, surface.info, float(np.finfo(gs.np_float).eps))
            qd.sync()
            timings = []
            for _ in range(5):
                started = time.perf_counter()
                for _ in range(10):
                    kernel_walk(
                        *inputs, faces, points, bary, model.info, surface.info, float(np.finfo(gs.np_float).eps)
                    )
                qd.sync()
                timings.append((time.perf_counter() - started) / 10)
            error = np.linalg.norm(array(points) - np.tile(controls["targets"], (repeat, 1)), axis=1)
            assert np.all(array(faces) >= 0) and error.max() < 1e-12
            median = float(np.median(timings))
            rows.append(
                {
                    "kind": kind,
                    "queries": count,
                    "preparation_seconds_including_cold_compilation": preparation,
                    "inclusive_seconds_per_dispatch": timings,
                    "median_seconds_per_dispatch": median,
                    "queries_per_second": count / median,
                    "maximum_position_error_m": float(error.max()),
                    "rejected_queries": 0,
                }
            )
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
