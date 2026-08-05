"""Genesis-to-QIPC sealed-volume gas integration tests."""

from pathlib import Path

import numpy as np
import pytest
import trimesh

try:
    import quadrants as qd  # noqa: F401
    from qipc import Scene as QIPCScene  # noqa: F401
    from qipc.inflation.authoring import pillow_bag
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs


DT = 0.01


def _write_pillow(path: Path, *, width: float = 0.20, open_mesh: bool = False) -> None:
    positions, triangles = pillow_bag(width=width, depth=0.13, bulge=0.025, nx=8, nz=6)
    if open_mesh:
        triangles = triangles[:-1]
    trimesh.Trimesh(vertices=positions, faces=triangles, process=False).export(path)


def _write_disconnected_pillows(path: Path) -> None:
    positions, triangles = pillow_bag(width=0.16, depth=0.13, bulge=0.025, nx=8, nz=6)
    shifted = positions + np.array([0.4, 0.0, 0.0])
    combined_positions = np.concatenate((positions, shifted))
    combined_triangles = np.concatenate((triangles, triangles + len(positions)))
    trimesh.Trimesh(vertices=combined_positions, faces=combined_triangles, process=False).export(path)


def _add_bag(scene, path: Path, *, name: str, x: float, p_gauge0: float):
    return scene.add_entity(
        name=name,
        morph=gs.morphs.Mesh(file=str(path), pos=(x, 0.0, 0.0)),
        material=gs.materials.FEM.SealedGasShell(
            E=1e7,
            nu=0.3,
            rho=910.0,
            thickness=1e-4,
            bending_stiffness=1e4,
            membrane="stvk",
            p_gauge0=p_gauge0,
        ),
    )


def _volume(entity) -> float:
    positions = entity.get_state().pos[0].cpu().numpy()
    triangles = np.asarray(entity.surface_triangles).reshape(-1, 3)
    return _mesh_volume(positions, triangles)


def _mesh_volume(positions: np.ndarray, triangles: np.ndarray) -> float:
    center = positions.mean(axis=0)
    a, b, c = (positions[triangles[:, axis]] - center for axis in range(3))
    return abs(float(np.einsum("ij,ij->i", np.cross(a, b), c).sum() / 6.0))


def test_sealed_gas_material_validation():
    material = gs.materials.FEM.SealedGasShell(p_gauge0=500.0)
    assert material.p_atm == 101325.0
    assert material.p_atm + material.p_gauge0 == 101825.0

    with pytest.raises(gs.GenesisException, match="p_atm.*p_gauge0.*positive"):
        gs.materials.FEM.SealedGasShell(p_atm=100.0, p_gauge0=-100.0)
    with pytest.raises(gs.GenesisException, match="p_atm.*p_gauge0.*finite.*positive"):
        gs.materials.FEM.SealedGasShell(p_atm=1e308, p_gauge0=1e308)
    with pytest.raises(gs.GenesisException, match="v_min_rel"):
        gs.materials.FEM.SealedGasShell(v_min_rel=1.0)


def test_sealed_gas_requires_qipc(tmp_path, show_viewer):
    path = tmp_path / "bag.obj"
    _write_pillow(path)
    scene = gs.Scene(show_viewer=show_viewer)
    with pytest.raises(gs.GenesisException, match="requires.*QIPCCouplerOptions"):
        _add_bag(scene, path, name="bag", x=0.0, p_gauge0=0.0)


def test_sealed_gas_runtime_api_requires_build(tmp_path, show_viewer):
    path = tmp_path / "bag.obj"
    _write_pillow(path)
    scene = gs.Scene(
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    bag = _add_bag(scene, path, name="bag", x=0.0, p_gauge0=0.0)

    with pytest.raises(gs.GenesisException, match="FEMEntity is not built yet"):
        bag.get_gas_state()
    with pytest.raises(gs.GenesisException, match="FEMEntity is not built yet"):
        bag.set_gas_state(enabled=False)


def test_sealed_gas_rejects_open_mesh(tmp_path, show_viewer):
    path = tmp_path / "open_bag.obj"
    _write_pillow(path, open_mesh=True)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    _add_bag(scene, path, name="open_bag", x=0.0, p_gauge0=0.0)
    with pytest.raises(gs.GenesisException, match="open_bag.*closed.*consistently wound"):
        scene.build()


def test_sealed_gas_rejects_multiple_shells_in_one_entity(tmp_path, show_viewer):
    path = tmp_path / "two_bags.obj"
    _write_disconnected_pillows(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    _add_bag(scene, path, name="two_bags", x=0.0, p_gauge0=0.0)
    with pytest.raises(gs.GenesisException, match="two_bags.*2 disconnected.*one FEM entity per gas bag"):
        scene.build()


@pytest.mark.required
def test_sealed_gas_accepts_linear_pcg(tmp_path, show_viewer):
    path = tmp_path / "bag.obj"
    _write_pillow(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(
            contact_enable=False,
            solver_linear_solver="linear_pcg",
        ),
        show_viewer=show_viewer,
    )
    bag = _add_bag(scene, path, name="linear_pcg_bag", x=0.0, p_gauge0=0.0)
    scene.build()
    scene.step()

    assert bag.get_gas_state().enabled is True
    assert scene.sim.coupler._scene.config["linear_system/solver"] == "linear_pcg"
    assert np.isfinite(bag.get_state().pos[0].cpu().numpy()).all()


@pytest.mark.required
def test_sealed_gas_reference_volume_uses_initial_geometry(tmp_path, show_viewer):
    path = tmp_path / "bag.obj"
    _write_pillow(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    bag = _add_bag(scene, path, name="prestressed_bag", x=0.0, p_gauge0=0.0)
    initial_positions = bag.init_positions.cpu().numpy()
    triangles = np.asarray(bag.surface_triangles).reshape(-1, 3)
    center = initial_positions.mean(axis=0)
    rest_positions = center + 0.5 * (initial_positions - center)
    initial_volume = _mesh_volume(initial_positions, triangles)
    rest_volume = _mesh_volume(rest_positions, triangles)

    scene.sim.coupler.set_fem_rest_positions(bag, rest_positions)
    scene.build()

    coupler = scene.sim.coupler
    entry = coupler._fem_entry_by_entity[bag]
    entity_slice = slice(entry.offset, entry.offset + entry.n_verts)
    finite_element = coupler._scene.finite_element
    assert initial_volume == pytest.approx(8.0 * rest_volume)
    np.testing.assert_allclose(finite_element.x[entity_slice].cpu().numpy(), initial_positions, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(finite_element.x_bar[entity_slice].cpu().numpy(), rest_positions, rtol=0.0, atol=1e-12)
    assert bag.get_gas_state().v0 == pytest.approx(initial_volume)


@pytest.mark.required
def test_sealed_gas_runtime_state_is_entity_scoped_and_resettable(tmp_path, show_viewer):
    small_path = tmp_path / "small_bag.obj"
    large_path = tmp_path / "large_bag.obj"
    _write_pillow(small_path, width=0.16)
    _write_pillow(large_path, width=0.24)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    small = _add_bag(scene, small_path, name="small_bag", x=-0.3, p_gauge0=100.0)
    ordinary_cloth = scene.add_entity(
        name="ordinary_cloth",
        morph=gs.morphs.Mesh(file=str(small_path), pos=(0.0, 0.0, 0.0)),
        material=gs.materials.FEM.Cloth(
            E=1e7,
            nu=0.3,
            rho=910.0,
            thickness=1e-4,
            bending_stiffness=1e4,
            membrane="stvk",
        ),
    )
    large = _add_bag(scene, large_path, name="large_bag", x=0.3, p_gauge0=900.0)
    scene.build()

    small_initial = small.get_gas_state()
    large_initial = large.get_gas_state()
    assert small_initial.p0 == pytest.approx(101425.0)
    assert large_initial.p0 == pytest.approx(102225.0)
    assert small_initial.v0 < large_initial.v0
    with pytest.raises(gs.GenesisException, match="only available.*SealedGasShell"):
        ordinary_cloth.get_gas_state()

    new_v0 = 1.25 * small_initial.v0
    small.set_gas_state(p0=120000.0, v0=new_v0, enabled=False)
    assert small.get_gas_state().p0 == pytest.approx(120000.0)
    assert small.get_gas_state().v0 == pytest.approx(new_v0)
    assert small.get_gas_state().enabled is False
    assert large.get_gas_state() == large_initial

    coupler = scene.sim.coupler
    small_bag = coupler._sealed_gas_bag_by_entity[small]
    gas_data = coupler._sealed_gas_data()
    initial_v_min = small_initial.v0 * small.material.v_min_rel
    assert float(gas_data["v_min"][small_bag]) == pytest.approx(initial_v_min)

    with pytest.raises(gs.GenesisException, match="p0 must be finite and positive"):
        small.set_gas_state(p0=0.0)
    with pytest.raises(gs.GenesisException, match="v0 must be finite and positive"):
        small.set_gas_state(p0=130000.0, v0=0.0)
    assert small.get_gas_state().p0 == pytest.approx(120000.0)
    with pytest.raises(gs.GenesisException, match="enabled must be a bool"):
        small.set_gas_state(enabled=1)

    native_scene = coupler._scene
    reset_topology_version = native_scene._reset_topo_version
    native_scene._reset_topo_version = -1
    state_before_refused_reset = small.get_gas_state()
    with pytest.raises(RuntimeError, match="topology changes"):
        scene.reset()
    assert small.get_gas_state() == state_before_refused_reset
    native_scene._reset_topo_version = reset_topology_version

    scene.reset()
    assert small.get_gas_state() == small_initial
    assert large.get_gas_state() == large_initial
    assert float(coupler._sealed_gas_data()["v_min"][small_bag]) == pytest.approx(initial_v_min)


@pytest.mark.required
def test_sealed_gas_overpressure_inflates_vs_disabled_control(tmp_path, show_viewer):
    path = tmp_path / "bag.obj"
    _write_pillow(path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=DT, gravity=(0.0, 0.0, 0.0)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    active = _add_bag(scene, path, name="active_bag", x=-0.2, p_gauge0=500.0)
    control = _add_bag(scene, path, name="disabled_bag", x=0.2, p_gauge0=500.0)
    scene.build()
    control.set_gas_state(enabled=False)

    active_v0 = _volume(active)
    control_v0 = _volume(control)
    for _ in range(30):
        scene.step()

    active_ratio = _volume(active) / active_v0
    control_ratio = _volume(control) / control_v0
    assert active_ratio > control_ratio + 0.001
    assert abs(control_ratio - 1.0) < 5e-5
    assert scene.sim.coupler.get_solver_statistics().max_line_search_iters < 12
