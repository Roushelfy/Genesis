import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.engine.solvers.rigid.stress.curvature import (
    CurvatureStatus,
    StressCurvature,
    kernel_vertex_curvature,
)
from genesis.utils.misc import qd_to_numpy


@pytest.mark.required
@pytest.mark.precision("64")
def test_vertex_curvature_reproducible_area_average(asset_tmp_path):
    path = asset_tmp_path / "curvature-sphere.obj"
    trimesh.creation.icosphere(subdivisions=2, radius=0.01).export(path)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Mesh(file=str(path), fixed=True))
    scene.build()
    geometry = scene.rigid_solver.dyn_info
    curvature = StressCurvature(geometry, float(np.finfo(np.float64).eps)).info

    def read(value):
        return qd_to_numpy(value, transpose=True, copy=True)

    positions, normals = read(geometry.verts.init_pos), read(geometry.verts.init_normal)
    faces, tensor = read(geometry.faces.verts_idx), read(curvature.tensor)
    selected = read(curvature.status) == CurvatureStatus.SMOOTH_NORMAL_ESTIMATE
    assert selected.any()
    faces, tensor = faces[selected], tensor[selected]
    area = np.linalg.norm(
        np.cross(positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]]),
        axis=1,
    )
    weights = np.zeros(len(positions))
    reference = np.zeros((len(positions), 3, 3))
    np.add.at(weights, faces.reshape(-1), np.repeat(area, 3))
    np.add.at(reference, faces.reshape(-1), np.repeat(area[:, None, None] * tensor, 3, axis=0))
    supported = weights > 0
    normal = normals[supported] / np.linalg.norm(normals[supported], axis=1)[:, None]
    projection = np.eye(3) - normal[:, :, None] * normal[:, None, :]
    reference[supported] = projection @ reference[supported] @ projection / weights[supported, None, None]
    baseline = read(curvature.vertices_tensor), read(curvature.vertices_weight)
    assert np.linalg.norm(baseline[0] - reference) <= 2e-14 * np.linalg.norm(reference)
    assert np.linalg.norm(baseline[1] - weights) <= 2e-14 * np.linalg.norm(weights)
    for _ in range(8):
        kernel_vertex_curvature(geometry, curvature)
        assert np.array_equal(read(curvature.vertices_tensor), baseline[0])
        assert np.array_equal(read(curvature.vertices_weight), baseline[1])
