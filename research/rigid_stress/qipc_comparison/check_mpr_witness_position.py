"""Manufacture optional MPR scratch states to reproduce the witness-roundoff counterexample.

The repaired production factory always retains portal points; disabling them
here is an explicit diagnostic construction, not a production option.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from genesis.engine.solvers.rigid.collider.mpr import mpr_find_pos
from genesis.utils import array_class
from genesis.utils.misc import tensor_to_array

from .artifacts import file_id, runtime_identity, write_json


@qd.kernel
def evaluate(
    first: qd.types.ndarray(),
    second: qd.types.ndarray(),
    state: array_class.MPRState,
    info: array_class.ColliderInfo,
    config: qd.template(),
    output: qd.types.ndarray(),
):
    for i_b in range(first.shape[0]):
        for corner in qd.static(range(4)):
            a = qd.Vector([first[i_b, corner, k] for k in qd.static(range(3))])
            b = qd.Vector([second[i_b, corner, k] for k in qd.static(range(3))])
            state.simplex_support.v1[corner, i_b] = a
            state.simplex_support.v2[corner, i_b] = b
            state.simplex_support.v[corner, i_b] = a - b
        position = mpr_find_pos(0, 1, i_b, state, info, config)
        for k in qd.static(range(3)):
            output[i_b, k] = position[k]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.gpu, precision="64", seed=623018, logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.5), size=(0.1, 0.1, 0.1)))
    scene.build(n_envs=1)
    solver = scene.rigid_solver
    rng = np.random.default_rng(623018)
    count = 384
    scales = np.repeat([0.001, 1.0, 1000.0], count // 3)
    angles = rng.uniform(-np.pi, np.pi, count)[:, None] + np.arange(3)[None] * (2 * np.pi / 3)
    radii = rng.uniform(0.005, 0.03, (count, 3)) * scales[:, None]
    difference = np.zeros((count, 4, 3))
    difference[:, 1:, 0] = radii * np.cos(angles)
    difference[:, 1:, 1] = radii * np.sin(angles)
    difference[:, 1:, 2] = rng.uniform(1e-8, 1e-4, count)[:, None] * scales[:, None]
    second = rng.normal(size=difference.shape) * (0.005 * scales[:, None, None])
    second += np.array([0.65, -0.01, 0.03]) * scales[:, None, None]
    first = second + difference
    a, b = (torch.as_tensor(x, dtype=gs.tc_float, device=gs.device) for x in (first, second))
    outputs = []
    for enabled in (False, True):
        # Manufactured optional scratch states reproduce the counterexample.
        # The repaired production factory always retains these portal points.
        state = array_class.get_mpr_state(count)
        state = replace(
            state,
            witness_a=array_class.V_VEC(3, dtype=gs.qd_float, shape=(count,) if enabled else ()),
            witness_b=array_class.V_VEC(3, dtype=gs.qd_float, shape=(count,) if enabled else ()),
        )
        result = torch.empty((count, 3), dtype=gs.tc_float, device=gs.device)
        evaluate(a, b, state, solver.collider.collider_info, solver.rigid_config, result)
        qd.sync()
        outputs.append(tensor_to_array(result))
    delta = np.abs(outputs[0] - outputs[1])
    assert np.isfinite(outputs).all()
    np.savez_compressed(
        args.output / "portals.npz", first=first, second=second, witness_off=outputs[0], witness_on=outputs[1]
    )
    rows = []
    for scale in (0.001, 1.0, 1000.0):
        mask = scales == scale
        rows.append(
            {
                "geometry_scale": scale,
                "cases": int(mask.sum()),
                "different_positions": int((delta[mask] > 0).any(axis=1).sum()),
                "maximum_absolute_difference_m": float(delta[mask].max()),
            }
        )
    write_json(
        args.output / "result.json",
        {
            "identity": runtime_identity(),
            "check_completed": True,
            "manufactured_optional_scratch_states": True,
            "bitwise_identical": bool((delta == 0).all()),
            "cases": rows,
            "portals": file_id(args.output / "portals.npz"),
            "scope": "Same raw simplex/support arrays and actual MPR position routine with optional witness buffers off/on. Synthetic nondegenerate portals exercise three geometry scales; this local control does not alone establish the cause of complete-scene trajectory differences.",
        },
    )


if __name__ == "__main__":
    main()
