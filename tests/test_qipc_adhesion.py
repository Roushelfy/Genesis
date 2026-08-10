"""QIPC coupler adhesion tests: Phase-1 soft adhesion + Phase-2 distance bonds.

Scene shape (mirrors cgq's test_bond_abd_fem_mixed): a FEM cube hangs beneath a
fixed ABD slab with a small initial gap (< d_hat). Without adhesion it free
falls; with soft adhesion (beta0=1) or distance bonds it holds. No ground plane
is needed, which keeps the scenes minimal. One scene build per test (the
autouse initialize_genesis fixture gives each test a fresh Genesis session).
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
D_HAT = 0.01
GAP = 0.005  # initial slab-underside to cube-top gap: inside the contact band

SLAB_SIZE = 0.2
SLAB_CENTER_Z = 0.5
CUBE_SIZE = 0.1
CUBE_CENTER_Z = SLAB_CENTER_Z - SLAB_SIZE / 2 - GAP - CUBE_SIZE / 2

N_STEPS = 60
FREE_FALL_DROP = 0.5 * 9.8 * (N_STEPS * DT) ** 2  # ~1.76 m

BOND_OPTS = dict(
    adhesion_bond_distance_lock=True,
    adhesion_bond_distance_lock_ratio=1.5,
    adhesion_bond_max_bonds=4096,
    adhesion_bond_kappa=1e8,
)


def _build_hanging_cube_scene(show_viewer, coupler_kwargs=None, adhesion=None):
    """Fixed ABD slab + FEM cube hanging beneath it with a GAP-wide air gap."""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=True,
            contact_d_hat=D_HAT,
            init_collision_pair_capacity=20000,
            **(coupler_kwargs or {}),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -1.0, 0.6),
            camera_lookat=(0.0, 0.0, 0.4),
        ),
        show_viewer=show_viewer,
    )
    slab = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, SLAB_CENTER_Z), size=(SLAB_SIZE, SLAB_SIZE, SLAB_SIZE), fixed=True),
        material=gs.materials.Rigid(rho=500.0, coup_friction=0.3),
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, CUBE_CENTER_Z), size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)),
        material=gs.materials.FEM.Elastic(E=1e5, nu=0.3, rho=1000.0, friction_mu=0.3, model="stable_neohookean"),
    )
    if adhesion is not None:
        scene.sim.coupler.add_adhesion(cube, slab, **adhesion)
    scene.build()
    return scene, slab, cube


def _run_and_measure_drop(scene, cube, n_steps=N_STEPS):
    z0 = float(cube.get_state().pos[0][:, 2].mean())
    for _ in range(n_steps):
        scene.step()
    pos = cube.get_state().pos[0].cpu().numpy()
    assert np.isfinite(pos).all()
    return z0 - float(pos[:, 2].mean())


@pytest.mark.required
def test_soft_adhesion_holds(show_viewer):
    """beta0=1 soft adhesion carries the hanging cube."""
    scene, _slab, cube = _build_hanging_cube_scene(
        show_viewer,
        adhesion=dict(Cn=1e6, Ct=0.0, W=1.0, eta=1.0, bonding_rate=0.0, beta0=1.0),
    )
    # auto constitution selection: adhesion declared -> adhesive_ipc
    assert scene.sim.coupler._scene.config["contact/constitution"] == "adhesive_ipc"

    drop = _run_and_measure_drop(scene, cube)
    assert drop < 0.05, f"cube fell {drop:.3f} m despite soft adhesion"

    # beta state is populated and saturated (beta0=1 seeded pairs)
    keys, betas = scene.sim.coupler.adhesion.dump_adhesion_state()
    assert keys.shape[0] > 0
    assert betas.max() > 0.9


@pytest.mark.required
def test_no_adhesion_control_free_falls(show_viewer):
    """Without adhesion the same cube free-falls (and auto keeps consistent_ipc)."""
    scene, _slab, cube = _build_hanging_cube_scene(show_viewer)
    constitution_names = {cls.__name__ for cls in scene.sim.coupler._scene.sim_systems}
    assert "ConsistentIPCContactConstitution" in constitution_names
    assert scene.sim.coupler._scene.contact_tabular.at(0, 0).bond is None

    drop = _run_and_measure_drop(scene, cube)
    assert drop > 0.5 * FREE_FALL_DROP, f"control cube should free-fall, dropped only {drop:.3f} m"


@pytest.mark.required
def test_distance_bond_holds(show_viewer):
    """Pure Phase-2 lock (no soft adhesion) carries the cube; bonds are queryable."""
    scene, _slab, cube = _build_hanging_cube_scene(show_viewer, coupler_kwargs=BOND_OPTS)
    default_model = scene.sim.coupler._scene.contact_tabular.at(0, 0)
    assert default_model.adhesion is None
    assert default_model.bond is not None
    assert default_model.bond.ratio == BOND_OPTS["adhesion_bond_distance_lock_ratio"]
    assert default_model.bond.kappa == BOND_OPTS["adhesion_bond_kappa"]
    drop = _run_and_measure_drop(scene, cube)
    assert drop < 0.1, f"cube fell {drop:.3f} m despite distance bonds"

    adhesion = scene.sim.coupler.adhesion
    assert adhesion.get_bond_count() > 0
    topos = adhesion.get_bond_topos()
    assert topos.ndim == 2 and topos.shape[1] == 4
    assert (topos >= 0).all()
    assert adhesion.fem_global_vertex_offset() > 0


@pytest.mark.required
def test_distance_bond_releases_under_force(show_viewer):
    """A tiny per-scene release_force lets the bonds go and the cube falls."""
    scene, _slab, cube = _build_hanging_cube_scene(
        show_viewer,
        coupler_kwargs=dict(**BOND_OPTS, adhesion_bond_release_force=1e-3),
    )
    drop = _run_and_measure_drop(scene, cube)
    assert drop > 0.3, f"bonds should release under a 1e-3 N threshold, dropped only {drop:.3f} m"


def test_adhesion_validation(show_viewer):
    """Host-side validation gives readable errors instead of silent no-ops.

    All failure paths trigger before qipc scene.init(), so several failed
    builds can share one Genesis session.
    """
    # Bad params raise immediately (no build needed)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.FEM.Elastic(E=1e5, nu=0.3, rho=1000.0),
    )
    with pytest.raises(Exception, match="non-negative"):
        scene.sim.coupler.add_adhesion(cube, None, Cn=-1.0)
    with pytest.raises(Exception, match="beta0"):
        scene.sim.coupler.add_adhesion(cube, None, Cn=1.0, beta0=2.0)

    # Plane target rejected at build (before qipc scene.init)
    plane = scene.add_entity(gs.morphs.Plane())
    scene.sim.coupler.add_adhesion(cube, plane, Cn=1.0)
    with pytest.raises(Exception, match="Plane/ground"):
        scene.build()

    # Lock without capacity rejected
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(adhesion_bond_distance_lock=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.FEM.Elastic(E=1e5, nu=0.3, rho=1000.0),
    )
    with pytest.raises(Exception, match="adhesion_bond_max_bonds"):
        scene.build()

    # Bonds in a FEM-less scene rejected
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(adhesion_bond_distance_lock=True, adhesion_bond_max_bonds=1024),
        show_viewer=show_viewer,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.Rigid(rho=500.0),
    )
    with pytest.raises(Exception, match="FEM"):
        scene.build()

    # add_adhesion with explicit 'consistent' constitution rejected
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT),
        coupler_options=gs.options.QIPCCouplerOptions(contact_constitution="consistent"),
        show_viewer=show_viewer,
    )
    cube = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.3), size=(0.1, 0.1, 0.1)),
        material=gs.materials.FEM.Elastic(E=1e5, nu=0.3, rho=1000.0),
    )
    scene.sim.coupler.add_adhesion(cube, None, Cn=1.0)
    with pytest.raises(Exception, match="consistent"):
        scene.build()
