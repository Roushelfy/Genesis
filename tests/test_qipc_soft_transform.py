"""QIPC coupler kinematic-driving tests: SoftTransformConstraint + per-entity d_hat.

(design: docs/adhesion_tape_design.md, A5.3)
"""

import numpy as np
import pytest

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs

DT = 0.01


@pytest.mark.required
def test_soft_transform_tracks_target(show_viewer):
    """A free ABD box driven by an STC follows a moving pose target."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500.0),
    )
    scene.sim.coupler.enable_soft_transform(box, strength=(1e4, 1e4))
    scene.build()

    start = np.array([0.0, 0.0, 0.3])
    target = start.copy()
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    # engage at the current pose, then translate +x over 100 steps
    scene.sim.coupler.set_soft_transform_target(box, start, quat, enabled=True)
    for _ in range(5):
        scene.step()
    for i in range(100):
        target = start + np.array([0.2 * (i + 1) / 100.0, 0.0, 0.0])
        scene.sim.coupler.set_soft_transform_target(box, target, quat)
        scene.step()
    for _ in range(30):  # settle at the final target
        scene.sim.coupler.set_soft_transform_target(box, target, quat)
        scene.step()

    pos = box.get_pos().reshape(-1)[:3].cpu().numpy()
    err = np.linalg.norm(pos - target)
    assert err < 0.02, f"STC-driven box did not track target (err={err:.4f}, pos={pos}, target={target})"


@pytest.mark.required
def test_soft_transform_requires_enable(show_viewer):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500.0),
    )
    scene.build()
    with pytest.raises(Exception, match="enable_soft_transform"):
        scene.sim.coupler.set_soft_transform_target(box, (0.0, 0.0, 0.3), (1.0, 0.0, 0.0, 0.0))


def test_solver_options_passthrough(show_viewer):
    """solver_* options land in the QIPC scene config; unset ones keep defaults."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            solver_newton_max_iter=300,
            solver_newton_velocity_tol=5e-3,
            solver_line_search_max_iter=16,
            contact_ccd_partition=False,
        ),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500.0),
    )
    scene.build()
    config = scene.sim.coupler._scene.config
    assert int(config["newton/max_iter"]) == 300
    assert float(config["newton/velocity_tol"]) == pytest.approx(5e-3)
    assert int(config["line_search/max_iter"]) == 16
    assert int(config["contact/ccd_partition"]) == 0
    # untouched knob keeps the QIPC default
    assert int(config["linear_system/max_iter"]) == 1024


def test_qipc_d_hat_stamped(show_viewer):
    """Rigid.qipc_d_hat lands as a per-geometry d_hat meta override."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=True, contact_d_hat=1e-4),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500.0, qipc_d_hat=1e-3),
    )
    scene.build()

    stamped = []
    for slot in scene.sim.coupler._scene.geometries:
        geo = slot.geometry
        if "d_hat" in geo.meta:
            stamped.append(float(geo.meta["d_hat"].cpu()[0]))
    assert stamped == [pytest.approx(1e-3)]
