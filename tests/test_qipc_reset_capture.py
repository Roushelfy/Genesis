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
