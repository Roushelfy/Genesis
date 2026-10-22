"""Check same-pose finite-face half-planes, rotation and environment-local release."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.curvature import StressCurvature
from genesis.engine.solvers.rigid.stress.hertz_regions import StressHertzRegions
from genesis.engine.solvers.rigid.stress.provenance import create_sources

from .artifacts import runtime_identity, write_json
from .check_hertz_regions import array, kernel_sources


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "start.json", {"identity": runtime_identity(require_gpu_inventory=args.backend == "gpu")})
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.cuda, precision="64", logging_level="warning")
    scene = gs.Scene(show_viewer=False)
    sphere = scene.add_entity(gs.morphs.Sphere(radius=0.01, pos=(0.002, 0.001, 0.01)))
    box = scene.add_entity(gs.morphs.Box(size=(0.006, 0.008, 0.004), pos=(0, 0, -0.002), fixed=True))
    scene.build(n_envs=3)
    solver = scene.rigid_solver
    epsilon = float(np.finfo(gs.np_float).eps)
    cache = StressCurvature(solver.dyn_info, epsilon)
    contacts = create_contacts(1, 3, 1.0, cooperative=False, cooperative_scatter=False)
    sources = create_sources(1, 3)
    regions = StressHertzRegions(contacts)
    own = np.array([sphere.base_link.geoms[0].idx], dtype=gs.np_int)
    local_centers = np.array([[0.002, 0.001, 0.01], [0.003002, 0.001, 0.01], [0.002, 0.001, 0.01]])
    corners = np.array([[x, y, 0] for x, y in product([-0.003, 0.003], [-0.004, 0.004])])
    rows = []
    for angle in [np.zeros(3), np.array([0.31, -0.48, 0.27])]:
        rotation = Rotation.from_rotvec(angle)
        quaternion = rotation.as_quat()[[3, 0, 1, 2]]
        translation = np.array([0.04, -0.03, 0.02])
        sphere.set_pos(rotation.apply(local_centers) + translation)
        sphere.set_quat(quaternion)
        box.set_pos(rotation.apply([0, 0, -0.002]) + translation)
        box.set_quat(quaternion)
        for phase in ["loaded", "release", "restore"]:
            forces = np.zeros((3, 1, 3))
            if phase != "release":
                forces[:2, 0, 2] = [0.1, 0.05]
            kernel_sources(
                own,
                np.array([[0.002, 0.001, 0]]),
                forces,
                translation,
                quaternion,
                contacts,
                sources,
                solver.collider.collider_state,
                sphere.base_link.idx,
                box.base_link.idx,
                box.base_link.geoms[0].idx,
                bad_pair=False,
                bad_moment=False,
                bad_position=False,
                swapped=False,
            )
            regions.update(
                sphere.base_link.idx,
                contacts,
                sources,
                solver.dyn_state,
                solver.collider.collider_state,
                solver.dyn_info,
                cache.info,
                epsilon,
            )
            planes, count = array(regions.state.boundary_planes), array(regions.state.n_boundaries)
            status, center = array(regions.state.status), array(regions.state.center)
            normal, tangent = array(regions.state.normal), array(regions.state.tangent)
            np.testing.assert_array_equal(count[:, 0], [0, 0, 0] if phase == "release" else [4, 4, 0])
            np.testing.assert_array_equal(status[:, 0], [1, 1, 1] if phase == "release" else [0, 2, 1])
            assert np.count_nonzero(planes[count == 0]) == 0
            if phase != "release":
                np.testing.assert_allclose(
                    normal[:2, 0], np.tile(rotation.apply([0, 0, -1]), (2, 1)), atol=1e-13, rtol=0
                )
                for i_b in range(2):
                    basis = np.column_stack((tangent[i_b, 0], np.cross(normal[i_b, 0], tangent[i_b, 0])))
                    vertices = (rotation.apply(corners) + translation - center[i_b, 0]) @ basis
                    residual = vertices @ planes[i_b, 0, :, :2].T - planes[i_b, 0, :, 2]
                    assert residual.max() < 1e-13
                    np.testing.assert_array_equal(np.sum(np.abs(residual) < 1e-13, axis=0), 2)
                    np.testing.assert_allclose(np.linalg.norm(planes[i_b, 0, :, :2], axis=1), 1, atol=1e-13, rtol=0)
                np.testing.assert_allclose(planes[1, 0, :, 2].min(), -2e-6, atol=1e-13, rtol=0)
            rows.append(
                {
                    "angle": angle.tolist(),
                    "phase": phase,
                    "status": status[:, 0].tolist(),
                    "count": count[:, 0].tolist(),
                }
            )
    write_json(args.output / "result.json", {"passed": True, "rows": rows, "whole_goal_complete": False})


if __name__ == "__main__":
    main()
