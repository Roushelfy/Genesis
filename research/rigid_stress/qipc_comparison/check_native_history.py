"""Audit sustained native prediction, current-load fallback and partial lifecycle changes."""

import argparse
from pathlib import Path

import numpy as np

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.misc import qd_to_numpy

from .artifacts import runtime_identity, write_json
from .check_native_atlas import cpu_control, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, required=True)
    parser.add_argument("--stress-backend", choices=("shell", "volume"), required=True)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lifecycle", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(identity=runtime_identity(), budget=1e-5, complete=False, rows=[], lifecycle=[])
    write_json(args.output / "result.json", result)
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    workload = FrankaEgg(
        args.envs,
        history_size=2,
        conditions=args.conditions,
        output_mode="full",
        radius_model="hertz",
        pressure_profile="hertz",
        quadrature=16,
        box_edge_policy="planar",
        stress_backend=args.stress_backend,
    )
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    state, history = entry.state, entry.history.state
    reference_options = entry.options.model_copy(update={"history_size": 0})
    fields = (("displacement", state.displacement), ("stress_tensor", state.stress_tensor), ("peak", state.peak))
    if args.stress_backend == "shell":
        fields += (
            ("membrane_resultant", state.membrane_resultant),
            ("bending_resultant", state.bending_resultant),
            ("transverse_resultant", state.transverse_resultant),
        )
    checkpoint = None
    for step in range(args.steps):
        if args.lifecycle and step == 350:
            checkpoint = workload.scene.get_state()
        if args.lifecycle and step in (250, 400):
            assert args.envs > 1
            before = qd_to_numpy(history.displacements)
            counts = qd_to_numpy(history.count)
            if step == 250:
                workload.reset(np.array([1]))
            else:
                workload.scene.reset(state=checkpoint, envs_idx=[1])
            after = qd_to_numpy(history.displacements)
            others = np.arange(args.envs) != 1
            np.testing.assert_array_equal(before[:, others], after[:, others])
            np.testing.assert_array_equal(counts[others], qd_to_numpy(history.count)[others])
            assert qd_to_numpy(history.count)[1] == 0
            result["lifecycle"].append(
                dict(step=step, kind="partial_reset" if step == 250 else "checkpoint", passed=True)
            )
        workload.step()
        candidate = {name: qd_to_numpy(array, transpose=True) for name, array in fields}
        # Includes predicted hits: the accepted state must actually feed future prediction.
        np.testing.assert_array_equal(
            qd_to_numpy(history.displacements)[:, :, 0].transpose(1, 0, 2), candidate["displacement"]
        )
        force = qd_to_numpy(state.force, transpose=True)
        loaded = np.linalg.norm(force, axis=(1, 2)) > 1e-30
        hit = qd_to_numpy(history.hit)
        assert qd_to_numpy(state.valid).all()
        row = dict(
            step=step,
            loaded=int(np.count_nonzero(loaded)),
            hits=int(np.count_nonzero(hit)),
            loaded_hits=int(np.count_nonzero(hit & loaded)),
            exact_environments=int(np.count_nonzero(~hit)),
            corrections=qd_to_numpy(state.corrections).tolist(),
            fallbacks=qd_to_numpy(state.fallbacks).tolist(),
        )
        entry.model.recover(entry.omega, state, reference_options, surface_load=True)
        row["errors"] = {name: relative(candidate[name], qd_to_numpy(array, transpose=True)) for name, array in fields}
        result["rows"].append(row)
        write_json(args.output / "result.json", result)
        assert qd_to_numpy(state.valid).all()
        assert max(row["errors"].values()) < 1e-5
        if step == min(129, args.steps - 1):
            result["cpu_control"] = cpu_control(entry)
            np.savez_compressed(args.output / "fields.npz", **candidate)
    result["complete"] = True
    write_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
