import numpy as np
import pytest
import torch

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs
import genesis.utils.geom as gu


def _module():
    from genesis.engine.couplers.qipc_coupler import tape_dispenser

    return tape_dispenser


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int32).reshape(-1, 4)
    order = np.lexsort(tuple(rows[:, column] for column in reversed(range(4))))
    return rows[order]


def _affine_tilt_degrees(current: torch.Tensor, initial: torch.Tensor) -> float:
    current_affine = current[3:12].detach().cpu().numpy().reshape(3, 3)
    initial_affine = initial[3:12].detach().cpu().numpy().reshape(3, 3)
    relative = current_affine @ np.linalg.inv(initial_affine)
    u, _, vt = np.linalg.svd(relative)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


@pytest.mark.required
@pytest.mark.precision("64")
def test_add_tape_dispenser_machine_is_ringless_urdf_only():
    module = _module()
    position = np.array([0.5, 0.0, 0.86661028])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_machine_coupler_options()),
        show_viewer=False,
    )
    machine = module.add_tape_dispenser_machine(
        scene,
        pos=position,
        euler=(0.0, 0.0, 180.0),
    )
    scene.build()

    coupler = scene.sim.coupler
    native = coupler._scene
    assert native.config["linear_system/solver"] == "partition_pcg"
    assert native.config["linear_system/preconditioner"] == "mas"
    assert native.config["linear_system/abd_preconditioner"] == "tree"
    assert native.affine_body.n_bodies == 4
    assert native.affine_body.n_verts == 16008
    assert native.finite_element.n_verts == 0
    assert float(native.affine_body.q_v.abs().max()) == 0.0
    assert float(native.joint_system.theta.abs().max()) == 0.0
    assert float(native.joint_system.kp.abs().max()) == 0.0
    assert float(native.joint_system.kv.abs().max()) == 0.0

    geometries = {slot.name: slot.geometry for slot in native.geometries}
    assert geometries["tape_cutter"].vertices.size == 10646
    assert geometries["Cylinder"].vertices.size == 512
    assert geometries["blade"].vertices.size == 767
    assert geometries["tape_wheel"].vertices.size == 4083
    wheel_link = next(link for link in machine.links if link.name == "tape_wheel")
    assert not any(
        str(vgeom.metadata.get("mesh_path", "")).endswith("/scotch3850_ring.glb") for vgeom in wheel_link.vgeoms
    )

    q_initial = native.affine_body.q.detach().clone()
    theta_initial = native.joint_system.theta.detach().clone()
    scene.step()
    assert torch.isfinite(native.affine_body.q).all()
    scene.reset()
    torch.testing.assert_close(native.affine_body.q, q_initial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native.joint_system.theta, theta_initial, rtol=0.0, atol=0.0)
    assert float(native.affine_body.q_v.abs().max()) == 0.0


@pytest.mark.required
@pytest.mark.precision("64")
def test_add_tape_dispenser_machine_remains_upright_on_table():
    module = _module()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_machine_coupler_options()),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(0.597, 0.0, 0.38),
            size=(0.85, 1.5, 0.76),
            fixed=True,
            collision=True,
        ),
        material=gs.materials.Rigid(coup_friction=0.8),
    )
    machine = module.add_tape_dispenser_machine(
        scene,
        pos=(0.5, 0.0, 0.86661028),
        euler=(0.0, 0.0, 180.0),
    )
    scene.build()

    native = scene.sim.coupler._scene
    geometries = {slot.name: slot.geometry for slot in native.geometries}
    root_body = int(np.asarray(geometries["tape_cutter"].meta["abd_body_offset"].cpu()).reshape(-1)[0])
    q_initial = native.affine_body.q.detach().clone()
    root_initial = q_initial[root_body]
    table_top = float(table.get_AABB()[1, 2])
    minimum_machine_z = float(machine.get_AABB()[0, 2])
    maximum_tilt = 0.0

    for _ in range(200):
        scene.step()
        assert torch.isfinite(native.affine_body.q).all()
        maximum_tilt = max(
            maximum_tilt,
            _affine_tilt_degrees(native.affine_body.q[root_body], root_initial),
        )
        minimum_machine_z = min(minimum_machine_z, float(machine.get_AABB()[0, 2]))

    final_tilt = _affine_tilt_degrees(native.affine_body.q[root_body], root_initial)
    assert final_tilt < 5.0
    assert maximum_tilt < 5.0
    assert minimum_machine_z >= table_top - 0.005

    scene.reset()
    torch.testing.assert_close(native.affine_body.q, q_initial, rtol=0.0, atol=0.0)
    assert float(native.affine_body.q_v.abs().max()) == 0.0


@pytest.mark.required
@pytest.mark.precision("64")
def test_add_tape_dispenser_matches_f249_contact_and_reset():
    module = _module()
    asset = module.TapeDispenserAsset.packaged()
    position = np.array([0.3, -0.2, 0.5])
    euler = np.array([0.0, 0.0, 17.0])
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_coupler_options()),
        show_viewer=False,
    )
    component = module.add_tape_dispenser(scene, asset, pos=position, euler=euler)
    scene.build()

    coupler = scene.sim.coupler
    native = coupler._scene
    assert native.config["linear_system/solver"] == "partition_pcg"
    assert native.config["linear_system/preconditioner"] == "mas"
    assert native.config["linear_system/abd_preconditioner"] == "tree"
    assert native.affine_body.n_bodies == 4
    assert native.affine_body.n_verts == 16200
    assert native.finite_element.n_verts == 1936
    assert coupler.adhesion.get_bond_count() == 969
    assert coupler.adhesion.get_bond_seed_result(component.tape) == (969, 0)
    assert float(native.affine_body.q_v.abs().max()) == 0.0
    assert float(native.finite_element.velocities.abs().max()) == 0.0
    assert float(native.joint_system.kp.abs().max()) == 0.0
    assert float(native.joint_system.kv.abs().max()) == 0.0

    world_rotation = gu.quat_to_R(gu.xyz_to_quat(euler, degrees=True)) @ module._REFERENCE_TO_GENESIS
    expected_tape = (asset.tape_positions - module._REFERENCE_BASE_POSITION) @ world_rotation.T + position
    tape_entry = coupler._fem_entry(component.tape)
    tape_slice = slice(tape_entry.offset, tape_entry.offset + tape_entry.n_verts)
    torch.testing.assert_close(
        native.finite_element.x[tape_slice],
        torch.as_tensor(expected_tape, dtype=torch.float64, device="cuda"),
        rtol=0.0,
        atol=2e-9,
    )

    expected_body_q = []
    for source_q in asset.body_q:
        transformed = np.empty(12)
        transformed[:3] = world_rotation @ (source_q[:3] - module._REFERENCE_BASE_POSITION) + position
        transformed[3:] = (world_rotation @ source_q[3:].reshape(3, 3)).reshape(-1)
        expected_body_q.append(transformed)
    actual_q = native.affine_body.q.detach().cpu().numpy()
    for expected in expected_body_q:
        assert np.min(np.max(np.abs(actual_q - expected), axis=1)) < 2e-12

    expected_joint_by_name = dict(zip(asset.joint_names, asset.joint_theta, strict=True))
    actual_joint_by_name = dict(
        zip(coupler._jc.joint_names, native.joint_system.theta.detach().cpu().numpy(), strict=True)
    )
    assert actual_joint_by_name.keys() == expected_joint_by_name.keys()
    for name, expected in expected_joint_by_name.items():
        assert actual_joint_by_name[name] == pytest.approx(expected, abs=1e-14)

    geometries = {slot.name: slot.geometry for slot in native.geometries}
    assert geometries["tape_cutter"].vertices.size == 10646
    assert geometries["Cylinder"].vertices.size == 512
    assert geometries["blade"].vertices.size == 767
    assert geometries["tape_wheel"].vertices.size == 4275
    wheel_link = next(link for link in component.machine.links if link.name == "tape_wheel")
    ring_visuals = [
        vgeom
        for vgeom in wheel_link.vgeoms
        if str(vgeom.metadata.get("mesh_path", "")).endswith("/scotch3850_ring.glb")
    ]
    assert len(ring_visuals) == 1
    ring_visual = ring_visuals[0]
    np.testing.assert_allclose(ring_visual.init_vverts, asset.ring_positions, rtol=0.0, atol=2e-9)
    np.testing.assert_array_equal(ring_visual.init_vfaces, asset.ring_triangles)
    np.testing.assert_array_equal(ring_visual.init_pos, np.zeros(3))
    np.testing.assert_array_equal(ring_visual.init_quat, np.array([1.0, 0.0, 0.0, 0.0]))
    ring_element = coupler._contact_elem_by_entity[component.ring_region][0]
    wheel_element_ids = np.asarray(geometries["tape_wheel"].vertices["contact_element_id"].cpu(), dtype=np.int32)
    assert np.count_nonzero(wheel_element_ids == ring_element.id) == 192
    assert np.all(wheel_element_ids[-192:] == ring_element.id)

    tab = native.contact_tabular
    info = coupler._contact_elem_by_entity

    def model(first, second):
        return tab.at(info[first][0].id, info[second][0].id)

    tape_cylinder = model(component.tape, component.cylinder_region)
    tape_blade = model(component.tape, component.blade_region)
    tape_sharp = model(component.tape, component.sharp_region)
    tape_machine = model(component.tape, component.machine)
    tape_ring = model(component.tape, component.ring_region)
    tape_self = model(component.tape, component.tape)
    tape_default = tab.at(info[component.tape][0].id, tab.default_element().id)
    assert tape_cylinder.enable and tape_cylinder.adhesion is None and tape_cylinder.bond is None
    assert tape_blade.enable and tape_blade.adhesion is None and tape_blade.bond is None
    assert not tape_sharp.enable and tape_sharp.adhesion is None and tape_sharp.bond is None
    assert not tape_machine.enable and tape_machine.adhesion is None and tape_machine.bond is None
    assert tape_ring.enable and tape_ring.adhesion is not None and tape_ring.bond is not None
    assert tape_self.enable and tape_self.adhesion is not None and tape_self.bond is not None
    assert tape_ring.bond.rest_snap is False
    assert tape_self.bond.rest_snap is False
    assert tape_default.enable and tape_default.adhesion is None and tape_default.bond is None

    q_initial = native.affine_body.q.detach().clone()
    x_initial = native.finite_element.x.detach().clone()
    theta_initial = native.joint_system.theta.detach().clone()
    bonds_initial = _sorted_rows(coupler.adhesion.get_bond_topos())
    with pytest.raises(gs.GenesisException, match="cannot be combined with a frozen BondSystem state"):
        coupler.adhesion.seed_bonds(bonds_initial[:1], asset.roll.thick)

    scene.step()
    assert torch.isfinite(native.affine_body.q).all()
    assert torch.isfinite(native.finite_element.x).all()
    assert float((native.finite_element.x - x_initial).norm(dim=1).max()) < 1.5e-3

    scene.reset()
    torch.testing.assert_close(native.affine_body.q, q_initial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native.finite_element.x, x_initial, rtol=0.0, atol=0.0)
    torch.testing.assert_close(native.joint_system.theta, theta_initial, rtol=0.0, atol=0.0)
    assert float(native.affine_body.q_v.abs().max()) == 0.0
    assert float(native.finite_element.velocities.abs().max()) == 0.0
    np.testing.assert_array_equal(_sorted_rows(coupler.adhesion.get_bond_topos()), bonds_initial)


@pytest.mark.required
@pytest.mark.precision("64")
def test_add_tape_dispenser_cluster_uses_wheel_proxy_and_replays_membership():
    module = _module()
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_coupler_options()),
        show_viewer=False,
    )
    component = module.add_tape_dispenser(
        scene,
        pos=(0.5, 0.0, 0.86661028),
        euler=(0.0, 0.0, 180.0),
        rigid_cluster=True,
    )
    scene.build()
    assert component.lifecycle is not None
    component.lifecycle.initialize()

    coupler = scene.sim.coupler
    native = coupler._scene
    wheel_geometry = next(slot.geometry for slot in native.geometries if slot.name == "tape_wheel")
    wheel_body = int(np.asarray(wheel_geometry.meta["abd_body_offset"].cpu()).reshape(-1)[0])
    assert component.lifecycle._cluster.proxy_body_index == wheel_body
    assert component.lifecycle.initial_member_count == 2720
    assert component.lifecycle.member_count == 2720
    assert coupler.adhesion.get_bond_count() == 969

    scene.step()
    component.lifecycle.before_step()
    assert component.lifecycle.member_count == 2720
    scene.reset()
    component.lifecycle.reset()
    assert component.lifecycle.member_count == 2720
    assert component.lifecycle.released_total == 0
    assert component.lifecycle.melted_total == 0
    assert coupler.adhesion.get_bond_count() == 969
