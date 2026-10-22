"""Refuse a segment that exits a curved face even when both endpoints are inside."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

import genesis as gs
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC

from .artifacts import runtime_identity, write_json
from .check_curved_surface_walk import kernel_walk
from .check_native_hertz_recovery import array


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--expect-baseline-failure", action="store_true")
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    vertices = 0.01 * np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    midpoints = vertices[edges].mean(axis=1)
    midpoints[0, 1] += 0.0015
    asset = args.output / "mesh.npz"
    np.savez_compressed(
        asset,
        vertices=vertices,
        tetrahedra=np.array([[0, 1, 2, 3]]),
        surface_triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
        edge_positions=midpoints,
    )
    model = StressModel(RigidStressOptions(mesh=asset, contact_model="hertz", method="direct"))
    surface = StressSurface(4, model.info)
    indices = array(model.info.surface_nodes).T
    face = np.flatnonzero(np.all(np.sort(indices[:, :3], axis=1) == [0, 1, 2], axis=1))[0]
    disconnected = V(dtype=gs.qd_int, shape=model.info.surface_neighbors.shape)
    disconnected.fill(-1)
    info = replace(model.info, surface_neighbors=disconnected)
    # The projected lower edge is y = 0.6*x*(1-x); its inward bulge makes the face nonconvex.
    local = np.array([[0.1, 0.02], [0.8, 0.02], [0.2, 0.3], [0.3, 0.3]])
    points = 0.01 * np.column_stack(
        (local[:, 0], local[:, 1] + 0.6 * local[:, 0] * (1 - local.sum(axis=1)), np.zeros(4))
    )
    centers, targets = points[[0, 1, 2]], points[[1, 0, 3]]
    middle = 0.5 * (points[0] + points[1]) / 0.01
    assert middle[1] < 0.6 * middle[0] * (1 - middle[0])
    anchors = np.full(3, face, dtype=gs.np_int)
    normals = np.tile([0.0, 0.0, 1.0], (3, 1))
    located, positions, bary = (
        V(dtype=gs.qd_int, shape=(3,)),
        V_VEC(3, dtype=gs.qd_float, shape=(3,)),
        V_VEC(3, dtype=gs.qd_float, shape=(3,)),
    )
    kernel_walk(
        anchors,
        centers,
        targets,
        normals,
        located,
        positions,
        bary,
        info,
        surface.info,
        float(np.finfo(gs.np_float).eps),
    )
    result = array(located)
    if args.expect_baseline_failure:
        np.testing.assert_equal(result, [face, face, face])
    else:
        np.testing.assert_equal(result, [-1, -1, face])
    np.testing.assert_allclose(array(positions)[2], targets[2], rtol=0, atol=1e-14)
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "faces": result.tolist(),
            "expected_baseline_shortcut_observed": args.expect_baseline_failure,
            "minimum_projected_jacobian_coefficient": 0.4,
            "scope": "A regular P2 face has positive projected Jacobian but a nonconvex projected boundary. Both endpoints lie inside; the segment leaves and returns. Disconnected adjacency requires refusing both directions while preserving an interior segment.",
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
