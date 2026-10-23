"""Check sharp-edge infinity, finite sampling and observation lifecycle through public getters."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.engine.solvers.rigid.stress.hertz_contacts import kernel_hertz_contact_begin, kernel_hertz_contact_peak
from genesis.engine.solvers.rigid.stress.hertz_recovery import StressHertzRecovery
from genesis.utils.misc import tensor_to_array

from .artifacts import runtime_identity, write_json
from .check_live_hertz_normal import build_fixture
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "scope": "Inject known accepted/rejected recovery outputs to test peak aggregation and public getters. Sharp-edge field physics is tested separately; these finite fixture values are not physical stresses.",
        },
    )
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    smooth = StressHertzRecovery(4, 1, moving_probes=True)
    assert not smooth.has_edge and smooth.edge.force.shape == ()
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    fixture = build_fixture(path)
    entry, link = fixture.entry, fixture.target.base_link
    assert entry is not None and entry.hertz is not None
    native = entry.hertz
    assert np.isnan(tensor_to_array(link.get_max_sampled_stress())).all()
    count = native.law.state.force.shape[0]
    loads = np.zeros(count)
    loads[:4] = 1
    native.law.state.force.from_numpy(loads)
    rows = []
    cases = [
        (1, [10, 20, 30, 40], [0, 1], [True] * 4, [np.inf, np.inf, 30, 40], [10, 20, 30, 40]),
        (1, [11, 19, 31, 39], [2], [True, False, True, True], [np.inf, np.nan, np.inf, 40], [11, np.nan, 31, 40]),
        (2, [8, 9, 10, 11], [], [True] * 4, [8, 9, 10, 11], [8, 9, 10, 11]),
    ]
    for step, peaks, edge_slots, valid, expected_global, expected_sampled in cases:
        kernel_hertz_contact_begin(step, native.state, entry.state)
        peaks, valid = np.array(peaks, dtype=gs.np_float), np.array(valid)
        entry.state.peak.from_numpy(peaks)
        entry.state.step_valid.from_numpy(valid)
        # This is the output contract of the preceding recovery acceptance pass.
        previous = array(entry.state.step_peak)
        entry.state.step_peak.from_numpy(np.where(valid, np.maximum(previous, peaks), np.nan))
        edge_status = np.ones(count, dtype=gs.np_int)
        edge_status[edge_slots] = 0
        native.hard.edge.state.status.from_numpy(edge_status)
        native.recovery.state.tensile_peak.from_numpy(peaks * 0.5)
        kernel_hertz_contact_peak(
            native.recovery.state.tensile_peak, native.law.state, native.hard.edge.state, entry.state, native.state
        )
        np.testing.assert_equal(tensor_to_array(link.get_max_stress()), expected_global)
        np.testing.assert_equal(tensor_to_array(link.get_max_sampled_stress()), expected_sampled)
        np.testing.assert_equal(
            tensor_to_array(link.get_max_sampled_stress(envs_idx=[0, 2])), np.array(expected_sampled)[[0, 2]]
        )
        sampled = link.get_max_sampled_stress()
        sampled.fill_(-1)
        np.testing.assert_equal(tensor_to_array(link.get_max_sampled_stress()), expected_sampled)
        rows.append(
            {
                "step": step,
                "finite_sampled": np.isfinite(expected_sampled).tolist(),
                "infinite_global": np.isposinf(expected_global).tolist(),
            }
        )
    native.reset(fixture.scene._sanitize_envs_idx([2]), entry.contacts)
    np.testing.assert_equal(tensor_to_array(link.get_max_sampled_stress()), [8, 9, np.nan, 11])
    write_json(
        args.output / "result.json", {"passed": True, "rows": rows, "partial_reset": True, "whole_goal_complete": False}
    )


if __name__ == "__main__":
    main()
