"""CUDA integration tests for Genesis queued QIPC affine clusters."""

from pathlib import Path

import numpy as np
import pytest

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)

import genesis as gs


def _write_square(path: Path) -> None:
    path.write_text("v -0.1 -0.1 0.0\nv 0.1 -0.1 0.0\nv 0.1 0.1 0.0\nv -0.1 0.1 0.0\nf 1 2 3\nf 1 3 4\n")


def _cloth_scene(tmp_path: Path, show_viewer: bool, *, with_proxy: bool, rigid: bool = False):
    mesh_path = tmp_path / "cluster_square.obj"
    _write_square(mesh_path)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.8)),
        coupler_options=gs.options.QIPCCouplerOptions(contact_enable=False),
        show_viewer=show_viewer,
    )
    proxy = None
    if with_proxy:
        proxy = scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.25),
                size=(0.05, 0.05, 0.05),
                fixed=True,
            ),
            material=gs.materials.Rigid(rho=1000.0),
        )
    cloth = scene.add_entity(
        morph=gs.morphs.Mesh(file=str(mesh_path), pos=(0.0, 0.0, 0.5)),
        material=gs.materials.FEM.Cloth(
            E=1e5,
            nu=0.3,
            rho=1000.0,
            thickness=1e-3,
            bending_stiffness=0.0,
            membrane="stvk",
        ),
    )
    initial_tris = np.arange(len(cloth.surface_triangles), dtype=np.int32)
    if rigid:
        handle = scene.sim.coupler.add_rigid_cluster(cloth, initial_tris=initial_tris)
    else:
        handle = scene.sim.coupler.add_affine_cluster(
            cloth,
            proxy_entity=proxy,
            proxy_link=None if proxy is None else proxy.links[0].name,
            kappa=1e8,
            initial_tris=initial_tris,
        )
    scene.build()
    return scene, cloth, proxy, handle


@pytest.mark.required
def test_ghost_cluster_runtime_membership_and_reset(tmp_path, show_viewer):
    scene, cloth, _proxy, handle = _cloth_scene(tmp_path, show_viewer, with_proxy=False)
    n_triangles = len(cloth.surface_triangles)

    assert handle.fem_vertex_range == range(cloth.n_vertices)
    assert handle.member_count == n_triangles
    proxy_body_index = handle.proxy_body_index

    handle.detach(tris=[0])
    assert handle.member_count == n_triangles - 1
    handle.join(tris=[0])
    assert handle.member_count == n_triangles

    scene.step()
    scene.reset()
    assert handle.member_count == n_triangles
    assert handle.proxy_body_index == proxy_body_index


@pytest.mark.required
def test_rigid_ghost_cluster_falls_as_one_body_and_refuses_reset(tmp_path, show_viewer):
    scene, cloth, _proxy, handle = _cloth_scene(tmp_path, show_viewer, with_proxy=False, rigid=True)
    n_triangles = len(cloth.surface_triangles)
    initial = cloth.get_state().pos.clone()

    assert handle.member_count == n_triangles
    assert handle.proxy_body_index == 0
    handle.detach(tris=[0])
    assert handle.member_count == n_triangles - 1
    handle.join(tris=[0])
    assert handle.member_count == n_triangles

    for _ in range(3):
        scene.step()
    # An exact rigid proxy in free fall: every riding vertex drops by the same amount.
    drop = (initial - cloth.get_state().pos)[..., 2].cpu().numpy()
    assert (drop > 0.0).all()
    np.testing.assert_allclose(drop, drop.mean(), atol=1e-7, rtol=0.0)

    with pytest.raises(RuntimeError, match="rigid FEM clusters"):
        scene.reset()


@pytest.mark.required
def test_rigid_link_proxy_uses_existing_fixed_body(tmp_path, show_viewer):
    scene, cloth, proxy, handle = _cloth_scene(tmp_path, show_viewer, with_proxy=True)
    coupler = scene.sim.coupler
    entry = coupler._fem_entry(cloth)
    initial = cloth.get_state().pos.clone()

    assert handle.fem_vertex_range == range(entry.offset, entry.offset + entry.n_verts)
    assert handle.proxy_body_index in coupler._body_indices_t.cpu().tolist()
    assert coupler._scene.affine_clusters[0].proxy_slot_id != coupler._scene.affine_clusters[0].fem_slot_ids[0]
    assert proxy.links[0].is_fixed

    for _ in range(3):
        scene.step()
    np.testing.assert_allclose(cloth.get_state().pos.cpu().numpy(), initial.cpu().numpy(), atol=1e-7, rtol=0.0)
