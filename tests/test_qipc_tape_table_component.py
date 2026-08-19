import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import genesis as gs
from genesis.utils.misc import get_assets_dir


def _module():
    # Importing the coupler compiles Quadrants functions whose annotations are
    # initialized by the per-test Genesis fixture.
    from genesis.engine.couplers.qipc_coupler import tape_table_component

    return tape_table_component


def _builder_module():
    from genesis.engine.couplers.qipc_coupler import tape_table

    return tape_table


def _json_bytes(value):
    return np.frombuffer(json.dumps(value, separators=(",", ":")).encode("utf-8"), dtype=np.uint8)


def _contact_policy():
    return {
        "schema": "qipc_contact_policy_v1",
        "enable": True,
        "friction_rate": 0.5,
        "resistance": 1.0e7,
        "adhesion": None,
    }


def _bond_policy(*, calibratable, release_force):
    return {
        "schema": "qipc_bond_policy_v1",
        "kappa": 3.0e7,
        "ratio": 1.5,
        "margin": 0.0,
        "floor_ratio": 0.5,
        "rest_snap": False,
        "occlusion": False,
        "release": {
            "force": release_force,
            "strain": 1.0e30,
            "gap": 1.0e30,
            "slip": 1.0e30,
        },
        "release_force_calibratable": calibratable,
    }


def _state_payload(prefix, count, *, release_force):
    Dm_inv = np.repeat(
        np.array(
            [
                [2.0, 0.5, 0.0],
                [0.0, 3.0, 0.25],
                [0.75, 0.0, 4.0],
            ],
            dtype=np.float64,
        )[None],
        count,
        axis=0,
    )
    V0 = abs(1.0 / np.linalg.det(Dm_inv[0])) / 6.0
    return {
        f"{prefix}_bond_state_Dm_inv": Dm_inv,
        f"{prefix}_bond_state_V0": np.full(count, V0, dtype=np.float64),
        f"{prefix}_bond_state_d_rest": np.zeros(count, dtype=np.float64),
        f"{prefix}_bond_state_kappa": np.full(count, 3.0e7, dtype=np.float64),
        f"{prefix}_bond_state_release_force": np.full(count, release_force, dtype=np.float64),
        f"{prefix}_bond_state_release_strain": np.full(count, 1.0e30, dtype=np.float64),
        f"{prefix}_bond_state_release_gap": np.full(count, 1.0e30, dtype=np.float64),
        f"{prefix}_bond_state_release_slip": np.full(count, 1.0e30, dtype=np.float64),
        f"{prefix}_bond_state_age": np.zeros(count, dtype=np.int32),
    }


def _component_payload():
    tape_positions = np.array(
        [
            [0.0, 0.01, -0.5],
            [0.0, 0.01, 0.5],
            [1.0, 0.01, -0.5],
            [1.0, 0.01, 0.5],
        ],
        dtype=np.float64,
    )
    hub_positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    hub_tris = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    table_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    table_tets = np.array([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32)
    table_surface_tris = np.array(
        [[0, 1, 3], [0, 3, 2], [1, 2, 3], [0, 2, 4], [0, 4, 1], [2, 1, 4]],
        dtype=np.int32,
    )
    internal_topologies = np.array([[4, 0, 1, 2], [5, 0, 1, 3]], dtype=np.int32)
    table_topologies = np.array(
        [[5, 0, 1, 3], [6, 0, 3, 2], [7, 1, 2, 3], [8, 0, 2, 4]],
        dtype=np.int32,
    )
    body_roles = {
        "schema": "qipc_component_body_roles_v1",
        "bodies": {
            "tape": {
                "role": "deformable_tape",
                "representation": "triangle_shell",
                "fixed": False,
                "initial_velocity": "zero",
            },
            "hub": {
                "role": "tape_hub",
                "representation": "affine_triangle_mesh",
                "fixed": False,
                "initial_velocity": "zero",
            },
            "table_attachment_pad": {
                "role": "table_attachment_pad",
                "representation": "affine_tetrahedral_mesh",
                "fixed": True,
                "initial_velocity": "zero",
            },
        },
    }
    bond_state_manifest = {
        "schema": "qipc_bond_state_manifest_v1",
        "mode": "grouped_frozen_state_v2",
        "portable_exact_restore": True,
        "dm_inv_transform": "right_multiply_placement_rotation_transpose",
        "groups": {
            "internal": {"owner": "hub", "fem": "tape", "topology_space": "hub_tape_local_v2"},
            "table": {
                "owner": "table_attachment_pad",
                "fem": "tape",
                "topology_space": "table_tape_local_v2",
            },
        },
        "captured_fields": [
            "Dm_inv",
            "V0",
            "d_rest",
            "kappa",
            "release_force",
            "release_strain",
            "release_gap",
            "release_slip",
            "age",
        ],
        "omitted_restore_fields": ["slot_alive", "pair_key", "dead_slot_capacity"],
        "omission_reason": (
            "pair_key is recomputed after topology remap; slot_alive and dead_slot_capacity "
            "belong to the runtime slot layout."
        ),
    }
    payload = {
        "component_schema": np.array(b"qipc_tape_table_component"),
        "component_schema_version": np.int32(2),
        "component_schema_abi": np.array(b"qipc_tape_table_component_v2"),
        "units": np.array(b"m"),
        "up_axis": np.array(b"+Y"),
        "handedness": np.array(b"right"),
        "component_frame": np.array(b"authoring_table_top_world"),
        "component_origin": np.zeros(3, dtype=np.float64),
        "component_transform": np.eye(4, dtype=np.float64),
        "source_qipc_commit": np.array(b"0123456789abcdef0123456789abcdef01234567"),
        "source_asset_name": np.frombuffer(b"scotch3850_wound.npz", dtype=np.uint8),
        "source_asset_sha256": np.array(b"a" * 64),
        "params_json": _json_bytes({"RCC_RELEASE_FORCE": 0.5}),
        "authoring_params_json": _json_bytes({"attached_length": 0.5}),
        "body_roles_json": _json_bytes(body_roles),
        "bond_state_manifest_json": _json_bytes(bond_state_manifest),
        "internal_contact_policy_json": _json_bytes(_contact_policy()),
        "internal_bond_policy_json": _json_bytes(_bond_policy(calibratable=False, release_force=1.0e30)),
        "table_contact_policy_json": _json_bytes(_contact_policy()),
        "table_bond_policy_json": _json_bytes(_bond_policy(calibratable=True, release_force=0.5)),
        "tape_positions": tape_positions,
        "tape_rest_positions": tape_positions.copy(),
        "tape_velocities": np.zeros_like(tape_positions),
        "tape_tris": np.array([[0, 2, 3], [0, 3, 1]], dtype=np.int32),
        "tape_is_fixed": np.zeros(4, dtype=np.int32),
        "hub_positions": hub_positions,
        "hub_tris": hub_tris,
        "hub_transform": np.array(
            [
                [1.0, 1.0e-3, 0.0, 0.2],
                [0.0, 0.999, 0.0, 0.3],
                [0.0, 0.0, 1.0, 0.4],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "hub_affine_velocity": np.zeros(12, dtype=np.float64),
        "hub_is_fixed": np.int32(0),
        "table_positions": table_positions,
        "table_tets": table_tets,
        "table_surface_tris": table_surface_tris,
        "table_transform": np.eye(4, dtype=np.float64),
        "table_affine_velocity": np.zeros(12, dtype=np.float64),
        "table_is_fixed": np.int32(1),
        "internal_bond_topologies": internal_topologies,
        "internal_bond_topology_space": np.array(b"hub_tape_local_v2"),
        "internal_bond_fem_offset": np.int32(4),
        "internal_bond_seed_rest_height": np.float64(0.0),
        "table_bond_topologies": table_topologies,
        "table_bond_topology_space": np.array(b"table_tape_local_v2"),
        "table_bond_fem_offset": np.int32(5),
        "table_bond_seed_rest_height": np.float64(0.0),
        "bond_state_mode": np.array(b"grouped_frozen_state_v2"),
        "bond_frozen_state_restore_supported": np.int32(1),
        "tail_vertex_ids": np.array([0, 1, 2, 3], dtype=np.int32),
        "tail_rows": np.array([0, 1], dtype=np.int32),
        "terminal_row_ids": np.array([2, 3], dtype=np.int32),
        "attached_length": np.float64(0.5),
        "nx": np.int32(1),
        "nz": np.int32(1),
        "tape_length": np.float64(1.0),
        "width": np.float64(1.0),
        "hub_r_outer": np.float64(0.4),
        "hub_r_inner": np.float64(0.3),
        "hub_height": np.float64(0.5),
        "thick": np.float64(4.0e-5),
        "d_hat": np.float64(8.0e-5),
        "youngs": np.float64(4.4e9),
        "poisson": np.float64(0.42),
        "density": np.float64(1850.0),
        "bending_e": np.float64(6.5e4),
    }
    payload.update(_state_payload("internal", len(internal_topologies), release_force=1.0e30))
    payload.update(_state_payload("table", len(table_topologies), release_force=1.0e30))
    return payload


def _save(path, payload):
    np.savez_compressed(path, **payload)
    return path


def test_tape_table_component_v2_loads_without_pickle_and_preserves_ids(tmp_path):
    module = _module()
    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "component.npz", _component_payload()))

    assert asset.source_asset_name == "scotch3850_wound.npz"
    assert asset.nx == 1
    assert asset.nz == 1
    np.testing.assert_array_equal(asset.tail_vertex_ids, [0, 1, 2, 3])
    np.testing.assert_array_equal(
        asset.table_bonds.topologies,
        [[5, 0, 1, 3], [6, 0, 3, 2], [7, 1, 2, 3], [8, 0, 2, 4]],
    )
    assert asset.internal_bonds.seed_rest_height == 0.0
    assert asset.internal_bonds.bond_policy["release"]["force"] == 1.0e30
    np.testing.assert_array_equal(
        asset.internal_bonds.frozen_state.release_force,
        [1.0e30, 1.0e30],
    )
    assert asset.table_bonds.bond_policy["release_force_calibratable"] is True
    assert asset.table_bonds.bond_policy["release"]["force"] == 0.5
    np.testing.assert_array_equal(
        asset.table_bonds.frozen_state.release_force,
        [1.0e30, 1.0e30, 1.0e30, 1.0e30],
    )
    assert asset.table_surface_tris.flags.writeable is False
    assert not np.allclose(asset.hub_transform[:3, :3].T @ asset.hub_transform[:3, :3], np.eye(3))


def test_tape_table_component_placement_is_explicit_and_keeps_source_unchanged(tmp_path):
    module = _module()
    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "component.npz", _component_payload()))
    transform = module.qipc_y_up_to_genesis_z_up_transform()
    transform[:3, 3] = [2.0, 3.0, 4.0]

    placed = asset.placed(transform)

    assert transform.flags.writeable
    np.testing.assert_allclose(placed.tape_positions, asset.tape_positions @ transform[:3, :3].T + transform[:3, 3])
    np.testing.assert_allclose(placed.hub_transform, transform @ asset.hub_transform)
    for batch, frozen_state in (
        (asset.internal_bonds, placed.internal_frozen_state),
        (asset.table_bonds, placed.table_frozen_state),
    ):
        source_state = batch.frozen_state
        expected_Dm_inv = (source_state.Dm_inv.reshape(-1, 3, 3) @ transform[:3, :3].T).reshape(-1, 9)
        np.testing.assert_allclose(frozen_state.Dm_inv, expected_Dm_inv)
        for field in (
            "V0",
            "d_rest",
            "kappa",
            "release_force",
            "release_strain",
            "release_gap",
            "release_slip",
            "age",
        ):
            np.testing.assert_array_equal(getattr(frozen_state, field), getattr(source_state, field))
        assert all(
            not getattr(frozen_state, field).flags.writeable
            for field in (
                "Dm_inv",
                "V0",
                "d_rest",
                "kappa",
                "release_force",
                "release_strain",
                "release_gap",
                "release_slip",
                "age",
            )
        )
    np.testing.assert_array_equal(asset.tape_positions, _component_payload()["tape_positions"])
    np.testing.assert_array_equal(
        asset.internal_bonds.frozen_state.Dm_inv,
        _component_payload()["internal_bond_state_Dm_inv"].reshape(-1, 9),
    )
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)


def test_tape_table_component_placement_does_not_freeze_the_callers_transform(tmp_path):
    module = _module()
    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "component.npz", _component_payload()))
    transform = np.eye(4, dtype=np.float64)

    placed = asset.placed(transform)

    assert transform.flags.writeable
    transform[0, 3] = 3.0
    assert placed.transform[0, 3] == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mode", "topology_seed_v2", "must declare portable grouped_frozen_state_v2"),
        ("portable_exact_restore", False, "must declare portable grouped_frozen_state_v2"),
        (
            "dm_inv_transform",
            "left_multiply_placement_rotation",
            "dm_inv_transform must be 'right_multiply_placement_rotation_transpose'",
        ),
    ],
)
def test_tape_table_component_rejects_nonportable_frozen_state_manifest(tmp_path, field, value, message):
    module = _module()
    payload = _component_payload()
    manifest = json.loads(payload["bond_state_manifest_json"].tobytes())
    manifest[field] = value
    payload["bond_state_manifest_json"] = _json_bytes(manifest)

    with pytest.raises(gs.GenesisException, match=message):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "nonportable-manifest.npz", payload))


def test_tape_table_component_rejects_disabled_frozen_state_restore(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["bond_frozen_state_restore_supported"] = np.int32(0)

    with pytest.raises(gs.GenesisException, match="must be 1 for portable exact restore"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "restore-disabled.npz", payload))


def test_tape_table_component_rejects_unicode_json_and_legacy_pickle(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["params_json"] = np.array('{"RCC_RELEASE_FORCE":0.5}')
    with pytest.raises(gs.GenesisException, match="params_json must be a non-empty uint8 byte vector"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "unicode.npz", payload))

    payload = _component_payload()
    payload["params"] = np.array([{"RCC_RELEASE_FORCE": 0.5}], dtype=object)
    with pytest.raises(gs.GenesisException, match="legacy object field 'params' is forbidden"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "pickle.npz", payload))


def test_tape_table_component_rejects_internal_table_face(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["table_surface_tris"] = np.vstack(
        (payload["table_surface_tris"][:-1], np.array([[0, 1, 2]], dtype=np.int32))
    )

    with pytest.raises(gs.GenesisException, match="exactly the tetrahedral boundary"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "internal-face.npz", payload))


def test_tape_table_component_rejects_mixed_owner_internal_bond_triangle(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_topologies"][0] = [4, 0, 1, 5]

    with pytest.raises(gs.GenesisException, match="triangle must belong entirely to the hub or entirely to the tape"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "mixed-owner-internal-bond.npz", payload))


def test_tape_table_component_rejects_internal_bond_triangle_not_on_source_surface(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_topologies"][0] = [0, 4, 5, 6]

    with pytest.raises(gs.GenesisException, match="tape PT triangles must reference tape_tris surface faces"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "non-surface-internal-bond.npz", payload))


def test_tape_table_component_accepts_distinct_pt_point_roles_over_the_same_four_vertices(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_topologies"] = np.array(
        [[5, 4, 6, 7], [6, 4, 7, 5]],
        dtype=np.int32,
    )

    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "distinct-point-roles.npz", payload))

    np.testing.assert_array_equal(asset.internal_bonds.topologies, payload["internal_bond_topologies"])


def test_tape_table_component_rejects_degenerate_settled_tape_triangle(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["tape_positions"][3] = payload["tape_positions"][2]

    with pytest.raises(gs.GenesisException, match="settled tape mesh contains a zero-area triangle"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "degenerate-settled-tape.npz", payload))


def test_tape_table_component_rejects_inconsistent_frozen_bond_volume(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_state_V0"][0] *= 2.0

    with pytest.raises(gs.GenesisException, match=r"V0 must equal abs\(1 / det\(Dm_inv\)\) / 6"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "inconsistent-volume.npz", payload))


def test_tape_table_component_rejects_missing_or_duplicate_tail_bond(tmp_path):
    module = _module()
    payload = _component_payload()
    topologies = payload["table_bond_topologies"].copy()
    topologies[-1, 0] = topologies[-2, 0]
    payload["table_bond_topologies"] = topologies

    with pytest.raises(gs.GenesisException, match="exactly one PT bond for every declared tail vertex"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "duplicate-tail.npz", payload))


@pytest.mark.parametrize(
    ("policy_release_force", "active_release_forces"),
    [
        (2.0e30, [1.0e30, 1.0e30]),
        (1.0e30, [1.0e30, 0.5]),
    ],
)
def test_tape_table_component_rejects_inconsistent_internal_release_force(
    tmp_path,
    policy_release_force,
    active_release_forces,
):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_policy_json"] = _json_bytes(
        _bond_policy(calibratable=False, release_force=policy_release_force)
    )
    payload["internal_bond_state_release_force"] = np.asarray(
        active_release_forces,
        dtype=np.float64,
    )

    with pytest.raises(
        gs.GenesisException,
        match="release.force must equal the unique value.*internal_bond_state_release_force",
    ):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "inconsistent-internal-release.npz", payload))


def test_tape_table_component_accepts_releasable_winding_and_skips_the_state_match(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_policy_json"] = _json_bytes(_bond_policy(calibratable=True, release_force=0.5))

    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "releasable-winding.npz", payload))

    assert asset.internal_bonds.bond_policy["release_force_calibratable"] is True
    assert asset.internal_bonds.bond_policy["release"]["force"] == 0.5
    np.testing.assert_array_equal(
        asset.internal_bonds.frozen_state.release_force,
        np.full(len(asset.internal_bonds.topologies), 1.0e30),
    )


def test_tape_table_component_rejects_uncoupled_release_calibration_flag(tmp_path):
    module = _module()
    payload = _component_payload()
    payload["internal_bond_policy_json"] = _json_bytes(_bond_policy(calibratable=True, release_force=1.0e30))

    with pytest.raises(
        gs.GenesisException,
        match="release_force_calibratable must be set exactly when the release force is a runtime value",
    ):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "uncoupled-calibration.npz", payload))


def test_tape_table_component_rejects_authoring_table_release_force(tmp_path):
    module = _module()
    payload = _component_payload()
    policy = _bond_policy(calibratable=True, release_force=0.7)
    payload["table_bond_policy_json"] = _json_bytes(policy)

    with pytest.raises(gs.GenesisException, match="must use params_json.RCC_RELEASE_FORCE"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "authoring-force.npz", payload))


def test_tape_table_component_rejects_sentinel_table_release_force(tmp_path):
    module = _module()
    payload = _component_payload()
    policy = _bond_policy(calibratable=True, release_force=1.0e30)
    payload["table_bond_policy_json"] = _json_bytes(policy)

    with pytest.raises(
        gs.GenesisException,
        match="release_force_calibratable must be set exactly when the release force is a runtime value",
    ):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "sentinel-table-force.npz", payload))


def test_packaged_tape_table_components_match_the_authoring_certificate():
    module = _module()
    directory = Path(get_assets_dir()) / "qipc" / "tape_table_component_v2"
    manifest = json.loads((directory / "manifest.json").read_text())
    filenames = [f"scotch3850_table_{inches}in_component.npz" for inches in (3, 4, 5, 6)]
    assets = [module.TapeTableComponentAsset.from_npz(directory / filename) for filename in filenames]

    assert [
        (
            asset.attached_length,
            len(asset.internal_bonds.topologies),
            len(asset.table_bonds.topologies),
            len(asset.table_surface_tris),
            len(asset.tail_vertex_ids),
        )
        for asset in assets
    ] == [
        (0.0762, 870, 187, 5180, 187),
        (0.1016, 761, 253, 5612, 253),
        (0.127, 682, 308, 5972, 308),
        (0.1524, 612, 363, 6332, 363),
    ]
    assert {asset.source_qipc_commit for asset in assets} == {"e17bb95974c56fedcbb1912c3ef86199df1465ed"}
    assert manifest["source"]["commit"] == "e17bb95974c56fedcbb1912c3ef86199df1465ed"
    assert {filename: hashlib.sha256((directory / filename).read_bytes()).hexdigest() for filename in filenames} == {
        filename: manifest["assets"][filename]["sha256"] for filename in filenames
    }


def test_packaged_tape_table_builder_verifies_and_loads_manifest_asset():
    asset = _builder_module().packaged_asset(3)

    assert asset.path.name == "scotch3850_table_3in_component.npz"
    assert len(asset.internal_bonds.topologies) == 870
    assert len(asset.table_bonds.topologies) == 187


def test_packaged_releasable_tape_table_components_carry_the_unwind_force():
    module = _builder_module()
    directory = Path(get_assets_dir()) / "qipc" / "tape_table_component_v2"
    manifest = json.loads((directory / "manifest.json").read_text())

    counts = {3: (870, 187), 4: (761, 253), 5: (682, 308), 6: (612, 363)}
    for inches, (internal_count, table_count) in counts.items():
        asset = module.packaged_asset(inches, winding="releasable")
        filename = f"scotch3850_table_{inches}in_unwind_component.npz"
        assert asset.path.name == filename
        assert manifest["assets"][filename]["winding"] == "releasable"
        assert hashlib.sha256(asset.path.read_bytes()).hexdigest() == manifest["assets"][filename]["sha256"]
        assert len(asset.internal_bonds.topologies) == internal_count
        assert len(asset.table_bonds.topologies) == table_count
        assert asset.internal_bonds.bond_policy["release_force_calibratable"] is True
        assert asset.internal_bonds.bond_policy["release"]["force"] == 0.5
        np.testing.assert_array_equal(
            asset.internal_bonds.frozen_state.release_force,
            np.full(internal_count, 1.0e30),
        )

    with pytest.raises(gs.GenesisException, match="winding must be 'locked' or 'releasable'"):
        module.packaged_asset(3, winding="unwound")


def test_tape_table_builder_restamps_a_calibratable_internal_batch():
    module = _builder_module()
    asset = module.packaged_asset(3, winding="releasable")
    scene = gs.Scene(
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_coupler_options(asset)),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.2), size=(0.8, 0.4, 0.4), fixed=True),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            contact_resistance=1.0e7,
            qipc_d_hat=asset.d_hat,
        ),
    )

    component = module.add_tape_table_component(
        scene,
        table,
        asset,
        transform=module.placement_transform_for_hub(asset, hub_xy=(-0.15, 0.0), table_top=0.4),
    )

    requests = {request.handle.name: request for request in scene.sim.coupler.adhesion._bond_state_requests}
    assert requests.keys() == {component.internal_bonds.name, component.table_bonds.name}
    np.testing.assert_array_equal(requests["internal"].release_force, np.full(870, 0.5))
    np.testing.assert_array_equal(requests["table"].release_force, np.full(187, 0.5))


def test_tape_table_builder_rejects_invalid_release_before_mutating_the_scene():
    module = _builder_module()
    asset = module.packaged_asset(3)
    scene = gs.Scene(
        coupler_options=gs.options.QIPCCouplerOptions(**module.recommended_coupler_options(asset)),
        show_viewer=False,
    )
    table = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.0, 0.0, 0.2), size=(0.6, 0.4, 0.4), fixed=True),
        material=gs.materials.Rigid(
            coup_friction=0.5,
            contact_resistance=1.0e7,
            qipc_d_hat=asset.d_hat,
        ),
    )
    entity_count = len(scene.entities)

    with pytest.raises(gs.GenesisException, match="table_bond_release_force must be finite and non-negative"):
        module.add_tape_table_component(
            scene,
            table,
            asset,
            transform=module.placement_transform_for_hub(asset, hub_xy=(0.0, 0.0), table_top=0.4),
            table_bond_release_force=-1.0,
        )

    assert len(scene.entities) == entity_count
