"""Independent shell controls and public rigid-observation lifecycle coverage."""

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import qd_to_numpy, tensor_to_array
from research.rigid_stress.qipc_comparison.check_native_shell_contacts import control as contact_control
from research.rigid_stress.qipc_comparison.check_native_shell_operators import control as operator_control
from research.rigid_stress.shell_cpu import geometry


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("curved", (False, True))
def test_native_shell_operators_and_complete_fields(tmp_path, curved):
    vertices, quads = geometry.egg_mesh(2) if curved else geometry.plate_mesh(2, Lx=0.04)
    operator_control(tmp_path, "shell", vertices, quads, "direct", "translation", spinning=True)


@pytest.mark.required
@pytest.mark.precision("64")
def test_native_shell_nonconvex_contact_and_release(tmp_path):
    vertices, quads = geometry.plate_mesh(2, Lx=0.04)
    # The material below this open shell's valley is nonconvex; the support crosses its ridge chart.
    vertices[:, 2] = 0.15 * np.abs(vertices[:, 0])
    contact_control(tmp_path, "valley", vertices, quads, 16, "faces")


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize(
    "output_mode, update_mode", (("max", "every_step"), ("full", "every_step"), ("full", "manual"))
)
@pytest.mark.parametrize("method", ("direct", "inverse"))
def test_native_shell_public_lifecycle(tmp_path, output_mode, update_mode, method):
    vertices, quads = geometry.sphere_mesh(2)
    asset = tmp_path / "shell.npz"
    np.savez(asset, vertices=vertices, quads=quads)
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.001), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    body = scene.add_entity(gs.morphs.Sphere(radius=0.02, pos=(0.0, 0.0, 0.0195)))
    link = body.base_link
    link.configure_stress_recovery(
        gs.options.RigidStressOptions.dpr(
            mesh=asset,
            backend="shell",
            method=method,
            output_mode=output_mode,
            update_mode=update_mode,
            contact_radius=8e-5,
            quadrature=16,
        )
    )
    scene.build(n_envs=3)
    entry = scene.rigid_solver.stress_recovery.links[0]
    if method == "inverse" and gs.backend == gs.cuda:
        assert entry.model.is_fused_recovery_enabled(entry.options)
    initial = np.array([[0.0, 0.0, 0.0195], [0.0, 0.0, 0.03], [0.0, 0.0, 0.0197]])
    body.set_pos(initial)
    if update_mode == "manual":
        scene.step()
        np.testing.assert_array_equal(tensor_to_array(link.get_max_stress()), 0.0)
        link.request_stress_update()
    scene.step()
    scene.rigid_solver.check_errno()
    before_peak = tensor_to_array(link.get_max_stress()).copy()
    assert np.isfinite(before_peak).all() and before_peak[0] > 0.0
    np.testing.assert_array_equal(before_peak[1], 0.0)
    before_u = qd_to_numpy(entry.state.displacement, transpose=True, copy=True)
    if output_mode == "full":
        tensor, vm = map(tensor_to_array, link.get_stress_field())
        assert tensor.shape == (3, len(quads), 2, 4, 6)
        assert vm.shape == (3, len(quads), 2, 4)
        np.testing.assert_allclose(vm.reshape((3, -1)).max(axis=1), before_peak, rtol=2e-14, atol=0.0)
        resultants = [tensor_to_array(value).copy() for value in link.get_shell_resultants()]
        assert [value.shape for value in resultants] == [
            (3, len(quads), 4, 6),
            (3, len(quads), 4, 6),
            (3, len(quads), 4, 3),
        ]
        assert all(np.isfinite(value).all() for value in resultants)
        for value in resultants:
            np.testing.assert_array_equal(value[1], 0.0)
    else:
        with pytest.raises(gs.GenesisException, match="output_mode='full'"):
            link.get_shell_resultants()
    checkpoint = scene.rigid_solver.__getstate__()
    assert checkpoint.configs["stress.0.options"].backend == "shell"
    assert any(item.name == "stress.0.shell.quads" for item in scene.rigid_solver.data)
    if output_mode == "full":
        assert "stress.0.state.membrane_resultant" in checkpoint.arrays
    scene.reset(envs_idx=np.array([0]))
    after_peak = tensor_to_array(link.get_max_stress())
    np.testing.assert_array_equal(after_peak[1:], before_peak[1:])
    np.testing.assert_array_equal(after_peak[0], 0.0)
    np.testing.assert_array_equal(qd_to_numpy(entry.state.displacement, transpose=True)[1:], before_u[1:])
    if output_mode == "full":
        for actual, expected in zip(link.get_shell_resultants(), resultants):
            actual = tensor_to_array(actual)
            assert np.isnan(actual[0]).all()
            np.testing.assert_array_equal(actual[1:], expected[1:])
    scene.rigid_solver.__setstate__(checkpoint)
    np.testing.assert_array_equal(tensor_to_array(link.get_max_stress()), 0.0)
    if output_mode == "full":
        assert all(np.isnan(tensor_to_array(value)).all() for value in link.get_shell_resultants())
    if update_mode == "manual":
        link.request_stress_update()
    scene.step()
    scene.rigid_solver.check_errno()
    assert np.isfinite(tensor_to_array(link.get_max_stress())).all()
