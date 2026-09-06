import numpy as np
import pytest
import torch

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs

_TABLE_TOP = 0.4
_BOX_HALF = 0.025
_DROP_CLEARANCE = 0.06


@pytest.mark.required
@pytest.mark.precision("64")
def test_reset_restores_released_link_constraints(tmp_path):
    path = tmp_path / "sheet.obj"
    path.write_text("v -0.1 -0.1 0\nv 0.1 -0.1 0\nv 0.1 0.1 0\nv -0.1 0.1 0\nf 1 2 3\nf 1 3 4\n")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=False,
    )
    tool = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.05, 0.05, 0.05)),
        material=gs.materials.Rigid(rho=1000.0),
    )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(path), pos=(0.0, 0.0, 0.5)),
        material=gs.materials.FEM.Cloth(E=1e5, nu=0.3, rho=1000.0, thickness=1e-3, bending_stiffness=0.0),
    )
    coupler = scene.sim.coupler
    coupler.enable_soft_transform(tool, strength=(1e6, 1e6))
    coupler.enable_fem_position_dbc(cloth, stiffness=1e8, fix_tol=1e-5)
    scene.build()
    initial = cloth.get_state().pos.clone()
    cloth.set_vertex_constraints([0, 1, 2, 3], link=tool.links[0], is_soft_constraint=True, stiffness=1e8)
    coupler.capture_reset_state()

    for displacement in (0.04, -0.04):
        cloth.remove_vertex_constraints()
        scene.reset()
        coupler.set_soft_transform_target(tool, [displacement, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
        for _ in range(20):
            scene.step()
        actual = (cloth.get_state().pos - initial).cpu().numpy()
        np.testing.assert_allclose(actual, np.broadcast_to([displacement, 0.0, 0.0], actual.shape), atol=1e-4)


def _scene_with_dropped_boxes():
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=1.0e-3,
            init_collision_pair_capacity=100_000,
            contact_friction=0.5,
            contact_resistance=1.0e7,
        ),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.5 * _TABLE_TOP), size=(0.6, 0.6, _TABLE_TOP), fixed=True),
        material=gs.materials.Rigid(coup_friction=0.5, contact_resistance=1.0e7),
    )
    boxes = [
        scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, y, _TABLE_TOP + _BOX_HALF + _DROP_CLEARANCE),
                size=(2.0 * _BOX_HALF,) * 3,
                fixed=False,
            ),
            material=gs.materials.Rigid(
                rho=500.0,
                coup_friction=0.5,
                contact_resistance=1.0e7,
                qipc_rigid_body=rigid_body,
            ),
        )
        for y, rigid_body in ((-0.1, False), (0.1, True))
    ]
    scene.build()
    return scene, boxes


def _heights(boxes) -> torch.Tensor:
    return torch.stack([box.get_pos().reshape(-1)[2].detach().cpu() for box in boxes])


@pytest.mark.required
@pytest.mark.precision("64")
def test_capture_reset_state_makes_the_settled_pose_the_reset_pose():
    scene, boxes = _scene_with_dropped_boxes()
    authored = _heights(boxes)
    for _ in range(60):
        scene.step()
    landed = _heights(boxes)
    assert float((authored - landed).min()) > 0.5 * _DROP_CLEARANCE

    scene.sim.coupler.capture_reset_state()
    for _ in range(20):
        scene.step()
    scene.reset()

    torch.testing.assert_close(_heights(boxes), landed, rtol=0.0, atol=1e-4)
    for _ in range(5):
        scene.step()
    torch.testing.assert_close(_heights(boxes), landed, rtol=0.0, atol=1e-3)


@pytest.mark.required
@pytest.mark.precision("64")
def test_reset_without_capture_returns_to_the_authored_pose():
    scene, boxes = _scene_with_dropped_boxes()
    authored = _heights(boxes)
    for _ in range(60):
        scene.step()
    scene.reset()

    torch.testing.assert_close(_heights(boxes), authored, rtol=0.0, atol=1e-6)
