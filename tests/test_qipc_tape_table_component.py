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


def _bond_policy(*, calibratable):
    return {
        "schema": "qipc_bond_policy_v1",
        "kappa": 3.0e7,
        "ratio": 1.5,
        "margin": 0.0,
        "floor_ratio": 0.5,
        "rest_snap": False,
        "occlusion": False,
        "release": {"force": 0.5, "strain": 1.0e30, "gap": 1.0e30, "slip": 1.0e30},
        "release_force_calibratable": calibratable,
    }


def _state_payload(prefix, count):
    return {
        f"{prefix}_bond_state_Dm_inv": np.repeat(np.eye(3, dtype=np.float64)[None], count, axis=0),
        f"{prefix}_bond_state_V0": np.ones(count, dtype=np.float64),
        f"{prefix}_bond_state_d_rest": np.zeros(count, dtype=np.float64),
        f"{prefix}_bond_state_kappa": np.full(count, 3.0e7, dtype=np.float64),
        f"{prefix}_bond_state_release_force": np.full(count, 0.5, dtype=np.float64),
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
    internal_topologies = np.array([[4, 0, 1, 2]], dtype=np.int32)
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
        "mode": "topology_seed_v2",
        "portable_exact_restore": False,
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
        "omission_reason": "Native slot ownership and dead capacity are scene-layout dependent.",
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
        "internal_bond_policy_json": _json_bytes(_bond_policy(calibratable=False)),
        "table_contact_policy_json": _json_bytes(_contact_policy()),
        "table_bond_policy_json": _json_bytes(_bond_policy(calibratable=True)),
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
        "bond_state_mode": np.array(b"topology_seed_v2"),
        "bond_frozen_state_restore_supported": np.int32(0),
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
    payload.update(_state_payload("internal", len(internal_topologies)))
    payload.update(_state_payload("table", len(table_topologies)))
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
    assert asset.table_bonds.bond_policy["release_force_calibratable"] is True
    assert asset.table_surface_tris.flags.writeable is False
    assert not np.allclose(asset.hub_transform[:3, :3].T @ asset.hub_transform[:3, :3], np.eye(3))


def test_tape_table_component_placement_is_explicit_and_keeps_source_unchanged(tmp_path):
    module = _module()
    asset = module.TapeTableComponentAsset.from_npz(_save(tmp_path / "component.npz", _component_payload()))
    transform = module.qipc_y_up_to_genesis_z_up_transform()
    transform[:3, 3] = [2.0, 3.0, 4.0]

    placed = asset.placed(transform)

    np.testing.assert_allclose(placed.tape_positions, asset.tape_positions @ transform[:3, :3].T + transform[:3, 3])
    np.testing.assert_allclose(placed.hub_transform, transform @ asset.hub_transform)
    np.testing.assert_array_equal(asset.tape_positions, _component_payload()["tape_positions"])
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)


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


def test_tape_table_component_rejects_missing_or_duplicate_tail_bond(tmp_path):
    module = _module()
    payload = _component_payload()
    topologies = payload["table_bond_topologies"].copy()
    topologies[-1, 0] = topologies[-2, 0]
    payload["table_bond_topologies"] = topologies

    with pytest.raises(gs.GenesisException, match="exactly one PT bond for every declared tail vertex"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "duplicate-tail.npz", payload))


def test_tape_table_component_rejects_authoring_table_release_force(tmp_path):
    module = _module()
    payload = _component_payload()
    policy = _bond_policy(calibratable=True)
    policy["release"]["force"] = 1.0e30
    payload["table_bond_policy_json"] = _json_bytes(policy)

    with pytest.raises(gs.GenesisException, match="must use params_json.RCC_RELEASE_FORCE"):
        module.TapeTableComponentAsset.from_npz(_save(tmp_path / "authoring-force.npz", payload))


def test_packaged_tape_table_components_match_the_authoring_certificate():
    module = _module()
    directory = Path(get_assets_dir()) / "qipc" / "tape_table_component_v2"
    assets = [
        module.TapeTableComponentAsset.from_npz(directory / f"scotch3850_table_{inches}in_component.npz")
        for inches in (3, 4, 5, 6)
    ]

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
    assert {asset.source_qipc_commit for asset in assets} == {"36f2816b52b414882db07acce51c3c30c31ba341"}
