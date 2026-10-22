"""Check observation aggregation against actual public physical-substep callbacks."""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTet

import genesis as gs
from genesis.utils.misc import tensor_to_array

from .artifacts import write_json
from .check_live_hertz_normal import build_fixture
from .check_native_hertz_recovery import array


class ClockAudit:
    def __init__(self, fixture):
        self.fixture = fixture
        self.peaks = []
        self.clocks = []

    def __call__(self, i_substep: int):
        entry = self.fixture.entry
        assert entry is not None
        sim = self.fixture.scene.sim
        self.peaks.append(array(entry.state.peak))
        self.clocks.append([i_substep, sim.cur_substep_global, sim.cur_substep_local])


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--route", choices=["hertz", "dpr"], required=True)
    cli.add_argument("--expect-last-only", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gs.init(backend=gs.cuda if args.backend == "gpu" else gs.cpu, precision="64", logging_level="warning")
    mesh = MeshTet.init_tensor(np.array([-0.004, 0, 0.004]), np.array([-0.004, 0, 0.004]), np.array([-0.002, 0, 0.002]))
    path = args.output / "mesh.npz"
    np.savez_compressed(
        path, vertices=mesh.p.T, tetrahedra=mesh.t.T, surface_triangles=mesh.facets[:, mesh.boundary_facets()].T
    )
    fixture = build_fixture(path, dt=2e-4, substeps=2, route=args.route)
    fixture.target.set_dofs_velocity(np.tile([0, 0, -1e-4, 0, 0, 0], (4, 1)))
    audit = ClockAudit(fixture)
    fixture.scene.register_post_substep_callback(audit)
    fixture.scene.step()
    observed = tensor_to_array(fixture.target.base_link.get_max_stress())
    raw = np.stack(audit.peaks)
    clocks = np.array(audit.clocks)
    expected = raw.max(axis=0)
    np.savez_compressed(
        args.output / "clock-and-peaks.npz", raw=raw, observed=observed, expected=expected, clocks=clocks
    )
    assert len(raw) == 2 and (clocks[:, 0] == [0, 1]).all() and (clocks[:, 2] == 0).all()
    assert np.any(raw[0] > raw[1] + 1e-7 * np.maximum(1, np.abs(raw[0])))
    if args.expect_last_only:
        np.testing.assert_allclose(observed, raw[-1], rtol=1e-7)
        assert not np.allclose(observed, expected, rtol=1e-7)
    else:
        np.testing.assert_allclose(observed, expected, rtol=1e-7)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "route": args.route,
            "clocks": clocks.tolist(),
            "source_expected_last_only": args.expect_last_only,
            "original_last_only_behavior_admitted": False,
            "complete_physical_substep_aggregation": not args.expect_last_only,
            "maximum_relative_miss": float(np.max(np.abs(observed - expected) / np.maximum(np.abs(expected), 1))),
            "public_post_substep_callback_audit": True,
            "production_kernels_replaced": False,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
