from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from tests.utils.assertions import assert_allclose, assert_equal


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("output_mode", ("max", "full"))
def test_native_intrinsic_observations_partial_reset_and_checkpoint(show_viewer, output_mode):
    assets = Path(__file__).resolve().parents[2] / "examples/rigid/assets/hollow_egg/level1"
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            enable_contact_patch=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.16, 0.12, 0.08),
            camera_lookat=(0.0, 0.0, 0.025),
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Plane(),
    )
    egg = scene.add_entity(
        morph=gs.morphs.URDF(
            file=assets / "egg_shell.urdf",
            pos=(0.0, 0.0, 0.031),
            align=False,
            convexify=True,
            decimate=False,
            preserve_mesh_precision=True,
        ),
    )
    link = egg.base_link
    link.configure_stress_recovery(
        gs.options.RigidStressOptions.scr(
            mesh=assets / "elastic.npz",
            output_mode=output_mode,
            contact_block_size=1,
        )
    )
    scene.build(n_envs=3)
    for _ in range(8):
        scene.step()
    scene.rigid_solver.check_errno()
    entry = scene.rigid_solver.stress_recovery.links[0]
    peak = tensor_to_array(link.get_max_stress(copy=True))
    assert np.isfinite(peak).all() and (peak > 0).all()
    assert qd_to_numpy(entry.state.step_valid, transpose=True).all()
    fields = None
    if output_mode == "full":
        fields = tuple(tensor_to_array(value).copy() for value in link.get_stress_field(copy=True))
        assert_equal(fields[1].max(axis=(1, 2)), peak)
    checkpoint = scene.rigid_solver.__getstate__()
    scene.step()
    next_peak = tensor_to_array(link.get_max_stress(copy=True))
    position = tensor_to_array(egg.get_pos()).copy()
    scene.step()
    following_peak = tensor_to_array(link.get_max_stress(copy=True))
    following_position = tensor_to_array(egg.get_pos()).copy()
    scene.rigid_solver.__setstate__(checkpoint)
    assert_equal(tensor_to_array(link.get_max_stress()), 0)
    if fields is not None:
        assert all(np.isnan(tensor_to_array(value)).all() for value in link.get_stress_field())
    scene.step()
    scene.rigid_solver.check_errno()
    assert_allclose(tensor_to_array(egg.get_pos()), position, rtol=0, atol=1e-15)
    assert_allclose(tensor_to_array(link.get_max_stress()), next_peak, rtol=1e-4, atol=1e-3)
    displacement = qd_to_numpy(entry.state.displacement, transpose=True, copy=True)
    peak = tensor_to_array(link.get_max_stress(copy=True))
    if fields is not None:
        fields = tuple(tensor_to_array(value).copy() for value in link.get_stress_field(copy=True))
    scene.reset(envs_idx=np.array([0]))
    assert_equal(qd_to_numpy(entry.state.displacement, transpose=True)[1:], displacement[1:])
    assert_equal(qd_to_numpy(entry.state.displacement, transpose=True)[0], 0)
    assert_equal(tensor_to_array(link.get_max_stress())[1:], peak[1:])
    assert_equal(tensor_to_array(link.get_max_stress())[0], 0)
    if fields is not None:
        for actual, expected in zip(link.get_stress_field(), fields):
            actual = tensor_to_array(actual)
            assert_equal(actual[1:], expected[1:])
            assert np.isnan(actual[0]).all()
    scene.step()
    scene.rigid_solver.check_errno()
    assert qd_to_numpy(entry.state.step_valid, transpose=True).all()
    assert_allclose(tensor_to_array(egg.get_pos())[1:], following_position[1:], rtol=0, atol=1e-15)
    assert_allclose(tensor_to_array(link.get_max_stress())[1:], following_peak[1:], rtol=1e-4, atol=1e-3)
