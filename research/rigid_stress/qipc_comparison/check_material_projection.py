"""Compare material-face projections over every curved face and its edge neighborhoods."""

import argparse
from pathlib import Path

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.engine.solvers.rigid.stress.data import StressInfo
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.provenance import func_material_source_point
from genesis.engine.solvers.rigid.stress.surface import StressSurface, StressSurfaceInfo
from genesis.options.rigid_stress import RigidStressOptions

from .artifacts import file_id, runtime_identity, write_json
from .check_native_hertz_recovery import array


@qd.kernel
def kernel_project(
    faces: qd.types.ndarray(),
    points: qd.types.ndarray(),
    known_barycentric: qd.types.ndarray(),
    bound: qd.types.ndarray(),
    positions: qd.types.ndarray(),
    barycentric: qd.types.ndarray(),
    info: StressInfo,
    surface: StressSurfaceInfo,
    epsilon: float,
):
    for i in range(faces.shape[0]):
        i_face = faces[i]
        point = qd.Vector([points[i, j] for j in qd.static(range(3))])
        known = qd.Vector([known_barycentric[i, j] for j in qd.static(range(3))])
        position, bary = func_material_source_point(i_face, point, known, info, surface, epsilon, bound[i] != 0)
        for j in qd.static(range(3)):
            positions[i, j], barycentric[i, j] = position[j], bary[j]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--precision", choices=["32", "64"], default="64")
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--baseline", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "start.json",
        {
            "identity": runtime_identity(require_gpu_inventory=args.backend == "gpu"),
            "inputs": [file_id(args.mesh)] + ([file_id(args.baseline)] if args.baseline else []),
            "precision": args.precision,
            "scope": "Every quadratic material face receives bound/unbound corner, edge, interior, exterior and signed-normal-offset queries. Same inputs compare the executed production projection against a frozen unpruned source. Position and barycentric budgets are128 machine epsilons, with material-coordinate scale for position. This checks implementation equivalence, not a new closest-point oracle or contact-pressure accuracy.",
        },
    )
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision=args.precision, logging_level="warning")
    model = StressModel(RigidStressOptions.hertz(mesh=args.mesh, method="direct"))
    surface = StressSurface(4, model.info)
    vertices = array(model.info.vertices)
    nodes = vertices[array(model.info.surface_nodes).T]
    epsilon = float(np.finfo(gs.np_float).eps)
    bary = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.2, 0.3, 0.5],
            [0.5, 0.2, 0.3],
            [0.3, 0.5, 0.2],
            [8 * epsilon, 0.4, 0.6 - 8 * epsilon],
            [-0.3, 0.65, 0.65],
            [1.4, -0.2, -0.2],
        ],
        dtype=gs.np_float,
    )
    shape = np.column_stack(
        [bary * (2 * bary - 1), 4 * bary[:, 0] * bary[:, 1], 4 * bary[:, 0] * bary[:, 2], 4 * bary[:, 1] * bary[:, 2]]
    )
    positions = np.einsum("qn,fni->fqi", shape, nodes)
    normal = array(surface.info.face_normal)
    diameter = np.linalg.norm(nodes[:, :3] - np.roll(nodes[:, :3], 1, axis=1), axis=2).max(axis=1)
    offsets = np.array([0, 1e-7, -0.05, 0.05, -0.5, 0.5, 2], dtype=gs.np_float)
    points = (
        positions[:, :, None] + diameter[:, None, None, None] * normal[:, None, None] * offsets[None, None, :, None]
    )
    faces = np.repeat(np.arange(len(nodes), dtype=gs.np_int), len(bary) * len(offsets))
    known = np.broadcast_to(bary[None, :, None], points.shape).reshape(-1, 3).copy()
    points = points.reshape(-1, 3)
    bound_points = positions[:, :10].reshape(-1, 3)
    bound_faces = np.repeat(np.arange(len(nodes), dtype=gs.np_int), 10)
    bound_known = np.broadcast_to(bary[None, :10], (len(nodes), 10, 3)).reshape(-1, 3)
    bound = np.r_[np.zeros(len(points), dtype=gs.np_int), np.ones(len(bound_points), dtype=gs.np_int)]
    faces, points, known = [
        np.concatenate(pair) for pair in [(faces, bound_faces), (points, bound_points), (known, bound_known)]
    ]
    projected, projected_bary = np.empty_like(points), np.empty_like(points)
    kernel_project(faces, points, known, bound, projected, projected_bary, model.info, surface.info, epsilon)
    assert np.isfinite(projected).all() and np.isfinite(projected_bary).all()
    np.testing.assert_allclose(projected_bary.sum(axis=1), 1, rtol=0, atol=128 * epsilon)
    comparison = None
    if args.baseline:
        with np.load(args.baseline) as original:
            for name, value in [("faces", faces), ("points_m", points), ("known_barycentric", known), ("bound", bound)]:
                np.testing.assert_array_equal(value, original[name], err_msg=name)
            position_budget = 128 * epsilon * np.abs(vertices).max()
            np.testing.assert_allclose(projected, original["projected_m"], rtol=0, atol=position_budget)
            np.testing.assert_allclose(projected_bary, original["projected_barycentric"], rtol=0, atol=128 * epsilon)
            comparison = {
                "position_exact": bool(np.array_equal(projected, original["projected_m"])),
                "barycentric_exact": bool(np.array_equal(projected_bary, original["projected_barycentric"])),
                "maximum_position_difference_m": float(abs(projected - original["projected_m"]).max()),
                "maximum_barycentric_difference": float(abs(projected_bary - original["projected_barycentric"]).max()),
                "position_absolute_budget_m": float(position_budget),
                "barycentric_absolute_budget": 128 * epsilon,
            }
    np.savez_compressed(
        args.output / "fields.npz",
        faces=faces,
        points_m=points,
        known_barycentric=known,
        bound=bound,
        projected_m=projected,
        projected_barycentric=projected_bary,
    )
    write_json(
        args.output / "result.json",
        {
            "passed": True,
            "precision": args.precision,
            "faces": len(nodes),
            "queries": len(points),
            "bound_queries": int(bound.sum()),
            "comparison": comparison,
            "whole_goal_complete": False,
        },
    )


if __name__ == "__main__":
    main()
