from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import genesis as gs
from genesis.engine.solvers.rigid.stress.contact import create_contacts
from genesis.engine.solvers.rigid.stress.intrinsic import IntrinsicContacts
from genesis.engine.solvers.rigid.stress.model import StressModel
from genesis.engine.solvers.rigid.stress.paths import StressSurfacePaths
from genesis.engine.solvers.rigid.stress.recovery import kernel_begin_step
from genesis.engine.solvers.rigid.stress.surface import StressSurface
from genesis.options.rigid_stress import RigidStressOptions
from genesis.utils.array_class import V, V_VEC
from genesis.utils.misc import qd_to_numpy
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("case", ("normal", "shear"))
def test_balanced_source_field_covariance_and_load_scale(case, tmp_path, record_property):
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    with np.load(assets / "elastic.npz") as saved:
        vertices, tets, faces = (saved[key] for key in ("vertices", "tetrahedra", "surface_triangles"))
    with np.load(Path(__file__).with_name("assets") / f"balanced_plate_{case}.npz") as saved:
        anchors, points, forces, radii, friction, moments = (
            saved[key] for key in ("anchor", "point", "force", "radius", "friction", "moment")
        )
    scales = np.array([1e-6, 1.0, 1e6])
    scene = gs.Scene(show_viewer=False)
    entity = scene.add_entity(
        gs.morphs.URDF(
            file=assets / "egg_shell.urdf", align=False, convexify=True, decimate=False, preserve_mesh_precision=True
        )
    )
    scene.build(n_envs=len(scales))
    rotations = [
        np.eye(3),
        Rotation.from_rotvec([0.4, -0.7, 0.3]).as_matrix(),
        Rotation.from_rotvec([-0.2, 0.6, 0.9]).as_matrix(),
    ]
    shifts = [np.zeros(3), np.array([0.7, -0.4, 1.1]), np.array([-1.2, 0.8, -0.3])]
    reference = None
    for index, (rotation, shift, length) in enumerate(zip(rotations, shifts, [1.0, 0.1, 10.0])):
        mesh = tmp_path / f"transformed-{index}.npz"
        np.savez(mesh, vertices=length * vertices @ rotation.T + shift, tetrahedra=tets, surface_triangles=faces)
        options = RigidStressOptions(
            mesh=mesh, method="direct", contact_model="intrinsic", intrinsic_load_partition="body_source_prior"
        )
        model = StressModel(options)
        surface = StressSurface(10, model.info)
        paths = StressSurfacePaths(model.info, subdivisions=8, precompute=True)
        raw = create_contacts(len(forces), len(scales), length * 0.006)
        raw.valid.fill(True)
        transformed_points = length * points @ rotation.T + shift
        transformed_anchors = length * anchors @ rotation.T + shift
        transformed_forces = (forces @ rotation.T)[:, None] * scales[None, :, None]
        transformed_moments = length * (moments @ rotation.T)[:, None] * scales[None, :, None]
        raw.position.from_numpy(np.repeat(transformed_points[:, None], len(scales), axis=1))
        raw.force.from_numpy(transformed_forces)
        raw.radius.from_numpy(np.repeat((length * radii)[:, None], len(scales), axis=1))
        raw.friction.from_numpy(np.repeat(friction[:, None], len(scales), axis=1))
        mapped = IntrinsicContacts(entity.base_link, raw, model.info, surface.info, options, paths)
        mapped.raw_sources.geom.fill(entity.base_link.geoms[0].idx)
        mapped.raw_sources.pair.from_numpy(
            np.repeat(np.arange(len(forces), dtype=gs.np_int)[:, None], len(scales), axis=1)
        )
        mapped.raw_sources.witness.from_numpy(np.repeat(transformed_anchors[:, None], len(scales), axis=1))
        mapped.raw_sources.free_moment.from_numpy(
            transformed_moments - np.cross((transformed_points - transformed_anchors)[:, None], transformed_forces)
        )
        state = model.create_state(len(scales), "full")
        errno = V(dtype=gs.qd_int, shape=(len(scales),))
        omega = V_VEC(3, dtype=gs.qd_float, shape=(len(scales),))
        errno.fill(0)
        omega.fill(0)
        kernel_begin_step(state)
        mapped.scatter(raw, state, model.info, surface.info, options, float(np.finfo(gs.np_float).eps), errno)
        assert_equal(qd_to_numpy(errno), 0)
        assert_equal(qd_to_numpy(raw.status, transpose=True), 0)
        model.recover(omega, state)
        assert qd_to_numpy(state.valid, transpose=True).all()
        packed = qd_to_numpy(state.stress_tensor, transpose=True)
        tensor = packed[..., [[0, 3, 4], [3, 1, 5], [4, 5, 2]]]
        tensor = rotation.T @ tensor @ rotation
        tensor = tensor * length**2 / scales.reshape((-1,) + (1,) * (tensor.ndim - 1))
        peak = qd_to_numpy(state.peak, transpose=True) * length**2 / scales
        for env in (0, 2):
            assert np.linalg.norm(tensor[env] - tensor[1]) / np.linalg.norm(tensor[1]) < 1e-4
        if reference is None:
            reference = tensor[1].copy()
            reference_peak = peak[1]
        error = np.linalg.norm(tensor - reference, axis=(-2, -1))
        assert np.linalg.norm(error) / (np.sqrt(len(scales)) * np.linalg.norm(reference)) < 1e-4
        assert_allclose(peak, reference_peak, rtol=1e-4, atol=0)
        record_property(
            f"geometry_{index}_relative_field_error",
            float(np.linalg.norm(error) / (np.sqrt(len(scales)) * np.linalg.norm(reference))),
        )
        record_property(f"geometry_{index}_normalized_peaks", peak.tolist())
