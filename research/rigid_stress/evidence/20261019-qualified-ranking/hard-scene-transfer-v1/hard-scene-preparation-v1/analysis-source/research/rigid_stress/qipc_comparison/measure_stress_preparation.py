"""Measure cold and warmed shared-operator construction plus declared native tensor storage."""

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import DataKind, iter_data
from genesis.utils.misc import qd_to_torch

from .artifacts import file_id, runtime_identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(), "mesh": file_id(args.mesh)})
    gs.init(backend=gs.cuda, precision="64", logging_level="warning")
    models, rows = [], []
    for repeat in range(4):
        qd.sync()
        started = perf_counter()
        model = StressModel(RigidStressOptions(mesh=args.mesh))
        qd.sync()
        rows.append(
            {
                "repeat": repeat,
                "cold": repeat == 0,
                "seconds": perf_counter() - started,
                "phases": asdict(model.build_timings),
            }
        )
        models.append(model)
    sizes = {}
    for name, info in [
        ("operators", model.info),
        ("factor", model.factor.info),
        ("inverse", model.inverse.info),
        ("surface_inverse", model.surface_inverse.info),
    ]:
        for item in iter_data(info, name, DataKind.CONSTANT):
            view = qd_to_torch(item.value, transpose=True)
            sizes[item.name] = view.numel() * view.element_size()
    write_json(
        args.output / "result.json",
        {
            "execution_completed": True,
            "rows": rows,
            "warm_median_seconds": float(np.median([row["seconds"] for row in rows[1:]])),
            "warm_phase_median_seconds": {
                key: float(np.median([row["phases"][key] for row in rows[1:]])) for key in rows[0]["phases"]
            },
            "shared_logical_bytes_by_field": sizes,
            "shared_logical_bytes": sum(sizes.values()),
            "nodes": model.info.vertices.shape[0],
            "elements": model.info.elements.shape[0],
            "method": model.selected_method,
            "scope": "One first construction and three warmed independent model constructions on the same mesh/material. Inclusive construction includes topology processing, Quadrants operators and factors; imports and gs.init are excluded. Byte counts cover declared shared tensors, not temporary construction state, allocator overhead or process peak memory.",
        },
    )


if __name__ == "__main__":
    main()
