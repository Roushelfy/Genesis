"""Profile exact nonlinear force evaluation on a saved admitted P2 displacement."""

import argparse
import cProfile
import io
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np

from .artifacts import file_id, runtime_identity, write_json
from .p2_elastic import FreeElasticP2
from .p2_stvk import P2StVK


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--frame", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.mesh) as saved:
        vertices, tetrahedra, faces = (saved[k] for k in ("vertices", "tetrahedra", "mid_faces"))
    model = FreeElasticP2(
        vertices, tetrahedra, 1e10, 0.3, 2000, column_faces=faces, through_wall_ordering=True, factorize=False
    )
    nonlinear = P2StVK(model.reference)
    with np.load(args.frame) as saved:
        displacement = saved["displacement_m"].reshape(-1)
    assert displacement.shape == (model.reference.basis.N,)
    force, _ = nonlinear.force_energy(displacement)
    correction = nonlinear.force_correction(displacement)
    relative = np.linalg.norm(correction - (model.reference.stiffness @ displacement - force)) / np.linalg.norm(force)
    assert relative < 1e-7
    timings = {}
    profiler = cProfile.Profile()
    profiler.enable()
    for name, operation in (
        ("gradient_interpolation", lambda: model.reference.basis.interpolate(displacement).grad),
        ("complete_force_energy", lambda: nonlinear.force_energy(displacement)),
        ("direct_correction", lambda: nonlinear.force_correction(displacement)),
    ):
        samples = []
        for _ in range(5):
            start = perf_counter()
            operation()
            samples.append(perf_counter() - start)
        timings[name] = samples
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(35)
    (args.output / "profile.txt").write_text(stream.getvalue())
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "mesh": file_id(args.mesh),
            "frame": file_id(args.frame),
            "direct_correction_force_relative_difference": float(relative),
            "timings_seconds": timings,
            "scope": "Intrusive CPU profile on an admitted saved state; no simulation changes and no primary trajectory timing claim.",
        },
    )


if __name__ == "__main__":
    main()
