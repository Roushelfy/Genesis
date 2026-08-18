import numpy as np
import pytest
import torch

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs


def _module():
    from genesis.engine.couplers.qipc_coupler import tape_table

    return tape_table


def _sorted_rows(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.int32).reshape(-1, 4)
    order = np.lexsort(tuple(rows[:, column] for column in reversed(range(4))))
    return rows[order]


@pytest.mark.required
@pytest.mark.precision("64")
def test_tape_table_component_restores_frame_zero_and_replays_two_resets():
    module = _module()
    asset = module.packaged_asset(3)
    table_top = 0.4
    coupler_options = module.recommended_coupler_options(asset)
    assert coupler_options["solver_newton_velocity_tol"] == 0.05
    assert coupler_options["solver_newton_max_iter"] == 300
    assert coupler_options["solver_linear_max_iter"] == 800
    assert coupler_options["solver_linear_tol_rate"] == 1.0e-4
    assert coupler_options["solver_line_search_max_iter"] == 16
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(**coupler_options),
        show_viewer=False,
    )
    scene.add_entity(
        morph=gs.morphs.Box(pos=(0.7, 0.7, 0.7), size=(0.1, 0.1, 0.1), fixed=True),
        material=gs.materials.Rigid(coup_friction=0.5),
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(
            pos=(-0.1, 0.0, 0.5 * table_top),
            size=(0.6, 0.4, table_top),
            fixed=True,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            contact_resistance=1.0e7,
            qipc_d_hat=asset.d_hat,
        ),
    )
    transform = module.placement_transform_for_hub(asset, hub_xy=(-0.12, 0.03), table_top=table_top)
    component = module.add_tape_table_component(
        scene,
        table,
        asset,
        transform=transform,
    )
    scene.build()

    coupler = scene.sim.coupler
    native = coupler._scene
    adhesion = coupler.adhesion
    assert scene.sim.cur_step_global == 0
    assert component.internal_bonds.name == "internal"
    assert component.table_bonds.name == "table"
    assert adhesion.get_bond_state_result(component.internal_bonds) == (870, 0)
    assert adhesion.get_bond_state_result(component.table_bonds) == (187, 0)
    assert adhesion.get_bond_count() == 1057
    assert adhesion._bond_seed_batch is None
    assert len(adhesion._bond_state_batches) == 1
    restored_release_force = adhesion._bond_state_batches[0]["release_force"]
    np.testing.assert_array_equal(restored_release_force[:870], np.full(870, 1.0e30))
    np.testing.assert_array_equal(restored_release_force[870:], np.full(187, 0.5))
    assert coupler._affine_clusters._requests == []

    internal_topologies = adhesion.get_bond_state_topologies(component.internal_bonds)
    table_topologies = adhesion.get_bond_state_topologies(component.table_bonds)
    assert internal_topologies is not None and internal_topologies.shape == (870, 4)
    assert table_topologies is not None and table_topologies.shape == (187, 4)
    assert set(map(tuple, internal_topologies)).isdisjoint(map(tuple, table_topologies))

    placed = asset.placed(transform)
    tape_entry = coupler._fem_entry(component.tape)
    tape_slice = slice(tape_entry.offset, tape_entry.offset + tape_entry.n_verts)
    torch.testing.assert_close(
        native.finite_element.x[tape_slice],
        torch.as_tensor(placed.tape_positions.copy(), dtype=torch.float64, device="cuda"),
        rtol=0.0,
        atol=2.0e-9,
    )
    hub_body = next(entry.body_offset for entry in coupler._free_base_entries if entry.entity is component.hub)
    expected_hub_q = np.concatenate((placed.hub_transform[:3, 3], placed.hub_transform[:3, :3].reshape(-1)))
    np.testing.assert_allclose(placed.hub_positions.mean(axis=0)[:2], (-0.12, 0.03), rtol=0.0, atol=2.0e-16)
    np.testing.assert_allclose(expected_hub_q[:2], (-0.12, 0.03), rtol=0.0, atol=2.0e-16)
    torch.testing.assert_close(
        native.affine_body.q[hub_body],
        torch.as_tensor(expected_hub_q, dtype=torch.float64, device="cuda"),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.isfinite(native.affine_body.q).all()
    assert torch.isfinite(native.finite_element.x).all()
    assert float(native.affine_body.q_v.abs().max()) == 0.0
    assert float(native.finite_element.velocities.abs().max()) == 0.0

    info = coupler._contact_elem_by_entity
    tabular = native.contact_tabular

    def model(first, second):
        return tabular.at(info[first][0].id, info[second][0].id)

    for first, second in (
        (component.tape, component.tape),
        (component.tape, component.tail_region),
        (component.tail_region, component.tail_region),
        (component.tape, component.hub),
        (component.tail_region, component.hub),
    ):
        pair = model(first, second)
        assert pair.enable and pair.adhesion is not None and pair.bond is not None
        assert pair.adhesion.Cn == pair.adhesion.Ct == pair.adhesion.W == 0.0
        assert pair.adhesion.sticky == (0, 0, 0, 0)
        assert pair.bond.rest_snap is False
        assert pair.bond.release.force == 1.0e30
    tail_pad_pair = model(component.tail_region, component.table_pad_region)
    assert tail_pad_pair.enable and tail_pad_pair.adhesion is not None and tail_pad_pair.bond is not None
    assert tail_pad_pair.adhesion.Cn == tail_pad_pair.adhesion.Ct == tail_pad_pair.adhesion.W == 0.0
    assert tail_pad_pair.adhesion.sticky == (0, 0, 0, 0)
    assert tail_pad_pair.bond.rest_snap is False
    assert tail_pad_pair.bond.release.force == 0.5
    tail_mask = np.zeros(len(asset.tape_positions), dtype=bool)
    tail_mask[asset.tail_vertex_ids] = True
    triangle_tail_count = tail_mask[asset.tape_tris].sum(axis=1)
    assert int(((triangle_tail_count > 0) & (triangle_tail_count < 3)).sum()) == 20
    for first, second in (
        (component.tape, component.table_pad_region),
        (component.hub, component.table_pad_region),
    ):
        pair = model(first, second)
        assert pair.enable and pair.adhesion is None and pair.bond is None
    for first, second in (
        (component.tape, table),
        (component.tail_region, table),
        (component.hub, table),
        (table, component.table_pad_region),
        (component.table_pad_region, component.table_pad_region),
    ):
        pair = model(first, second)
        assert not pair.enable and pair.adhesion is None and pair.bond is None

    q_initial = native.affine_body.q.detach().clone()
    x_initial = native.finite_element.x.detach().clone()
    q_v_initial = native.affine_body.q_v.detach().clone()
    velocities_initial = native.finite_element.velocities.detach().clone()
    bonds_initial = _sorted_rows(adhesion.get_bond_topos())
    fem_global_offset = adhesion.fem_global_vertex_offset()
    table_vertices = np.unique(table_topologies[table_topologies < fem_global_offset])
    assert len(table_vertices) > 0

    def assert_reset_state():
        assert adhesion.get_bond_count() == 1057
        torch.testing.assert_close(native.affine_body.q, q_initial, rtol=0.0, atol=0.0)
        torch.testing.assert_close(native.finite_element.x, x_initial, rtol=0.0, atol=0.0)
        torch.testing.assert_close(native.affine_body.q_v, q_v_initial, rtol=0.0, atol=0.0)
        torch.testing.assert_close(native.finite_element.velocities, velocities_initial, rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(_sorted_rows(adhesion.get_bond_topos()), bonds_initial)
        assert adhesion.get_bond_state_result(component.internal_bonds) == (870, 0)
        assert adhesion.get_bond_state_result(component.table_bonds) == (187, 0)

    scene.step()
    assert torch.isfinite(native.affine_body.q).all()
    assert torch.isfinite(native.finite_element.x).all()
    bond_count = adhesion.get_bond_count()
    max_displacement = float((native.finite_element.x[tape_slice] - x_initial[tape_slice]).norm(dim=1).max())
    gs.logger.info(f"Tape-table CUDA check: bonds={bond_count}, max displacement={max_displacement:.9g} m.")
    live_topologies = set(map(tuple, adhesion.get_bond_topos()))
    assert set(map(tuple, internal_topologies)).issubset(live_topologies)
    assert 870 <= bond_count < 1057
    assert max_displacement < 1.5e-3
    scene.reset()
    assert_reset_state()

    adhesion.release_bonds_by_vertices(table_vertices, require_all=False)
    assert adhesion.get_bond_count() == 870
    scene.reset()
    assert_reset_state()
