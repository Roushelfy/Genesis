"""Replay the retained vertex-adjacent probe failure on pinned old/new sources."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz import StressHertz
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery, kernel_hertz_moving_probes
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.options.rigid_stress import RigidStressOptions

from .artifacts import file_id, write_json
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--control", type=Path, required=True)
    cli.add_argument("--expect-failure", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.control / "frame7.npz"
    with np.load(path) as saved:
        data = {key: saved[key] for key in saved}
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    batch = len(data["vm_pa"])
    patches = len(data["law_force_n"])
    model = StressModel(RigidStressOptions(mesh=args.control / "mesh.npz", method="pcg"))
    law = StressHertz(patches)
    for target, key in [
        (law.state.force, "law_force_n"),
        (law.state.axes, "law_axes_m"),
        (law.state.direction, "law_direction"),
        (law.state.peak_pressure, "law_pressure_pa"),
        (law.state.status, "law_status"),
    ]:
        target.from_numpy(data[key])
    module = StressHertzRecovery(batch, patches // batch, output_mode="max", moving_probes=True)
    module.frames.center.from_numpy(data["hertz_centers_m"])
    module.frames.basis.from_numpy(data["hertz_frames"])
    module.frames.anchor.from_numpy(data["hertz_anchors"])
    module.probe_offsets.from_numpy(data["probe_offsets"])
    module.state.status.fill(0)
    kernel_hertz_moving_probes(
        module.probe_offsets, module.frames, law.state, module.state, module.probes, model.info, 64
    )
    elements, points, bary = (
        array(module.probes.element),
        array(module.probes.position),
        array(module.probes.barycentric),
    )
    loaded = np.flatnonzero(data["law_force_n"] > 0)
    invalid = {int(patch): np.flatnonzero(elements[patch] < 0).tolist() for patch in loaded}
    failed = any(invalid.values())
    if args.expect_failure:
        assert failed and invalid[3], invalid
    else:
        assert not failed, invalid
        native_elements, vertices = array(model.info.elements), array(model.info.vertices)
        located = np.einsum("pqc,pqci->pqi", bary[loaded], vertices[native_elements[elements[loaded], :4]])
        np.testing.assert_allclose(located, points[loaded], rtol=0, atol=2e-16)
        assert bary[loaded].min() >= -1e-10 and bary[loaded].max() <= 1 + 1e-10
    np.savez_compressed(args.output / "probe-locations.npz", positions_m=points, elements=elements, barycentric=bary)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "input": file_id(path),
            "expected_original_failure_observed": bool(args.expect_failure and failed),
            "original_failed_estimator_admitted": False,
            "all_loaded_probes_located": not failed,
            "invalid_probes": invalid,
            "unchanged_barycentric_tolerance": 1e-10,
            "walk_steps": 64,
            "global_search_used": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
