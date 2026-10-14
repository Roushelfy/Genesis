from pathlib import Path

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import tensor_to_array
from tests.utils.assertions import assert_allclose


@pytest.mark.required
@pytest.mark.precision("64")
@pytest.mark.parametrize("contact_patch", (False, True))
def test_shallow_convex_contact_matches_independent_separating_translation(contact_patch, asset_tmp_path):
    with np.load(Path(__file__).with_name("assets") / "nonconvex_epa_shallow_bound.npz") as asset:
        saved = {key: asset[key].copy() for key in asset.files}
    mesh = asset_tmp_path / "convex_egg.obj"
    trimesh.Trimesh(vertices=saved["mesh_vertices"], faces=saved["mesh_faces"], process=False).export(
        mesh, file_type="obj", digits=17
    )
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            gravity=(0.0, 0.0, 0.0),
        ),
        rigid_options=gs.options.RigidOptions(
            use_gjk_collision=True,
            enable_contact_patch=contact_patch,
        ),
        show_viewer=False,
    )
    box = scene.add_entity(
        morph=gs.morphs.Box(
            pos=saved["box_positions"][0],
            quat=saved["box_quaternions"][0],
            size=saved["box_size"],
            fixed=True,
        ),
        vis_mode="collision",
    )
    egg = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=mesh,
            pos=saved["mesh_position"],
            quat=saved["mesh_quaternion"],
            decimate=False,
        ),
        vis_mode="collision",
    )
    scene.build(n_envs=2)
    box.set_pos(saved["box_positions"])
    box.set_quat(saved["box_quaternions"])
    scene.step()
    contacts = box.get_contacts(with_entity=egg)
    normals = tensor_to_array(contacts["normal"])
    geoms = tensor_to_array(contacts["geom_a"])
    valid = tensor_to_array(contacts["valid_mask"])
    depth = tensor_to_array(contacts["penetration"])
    for environment in range(2):
        assert valid[environment].any()
        direction = normals[environment, valid[environment]]
        direction = np.where(
            (geoms[environment, valid[environment]] == box.geom_start)[:, None], direction, -direction
        )
        assert (direction @ saved["reference_normals"][environment] > 0.99).all()
        assert_allclose(
            depth[environment, valid[environment]].max(),
            saved["reference_penetration"][environment],
            rtol=1e-7,
            atol=1e-11,
        )
