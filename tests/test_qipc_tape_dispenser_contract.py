import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

import genesis as gs

try:
    import quadrants as qd
    from qipc import Scene as QIPCScene
except ImportError:
    pytest.skip("QIPC coupler requires 'quadrants' and 'qipc' packages.", allow_module_level=True)


def _module():
    from genesis.engine.couplers.qipc_coupler import tape_dispenser

    return tape_dispenser


def _copy_packaged_asset(module, destination):
    shutil.copytree(module._ASSET_DIRECTORY, destination)
    return destination


def _rewrite_npz(path, updates):
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload.update(updates)
    with path.open("wb") as output:
        np.savez_compressed(output, **payload)


def _refresh_hash(directory, relative):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"][relative] = hashlib.sha256((directory / relative).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


@pytest.mark.required
def test_packaged_tape_dispenser_snapshot_contract():
    module = _module()
    asset = module.TapeDispenserAsset.packaged()

    assert asset.body_names == ("tape_cutter", "Cylinder", "blade", "tape_wheel")
    assert asset.joint_names == ("Cylinder_axle", "blade_hinge", "tape_wheel_axle")
    assert asset.tape_positions.shape == (1936, 3)
    assert asset.tape_triangles.shape == (3500, 3)
    assert asset.ring_positions.shape == (192, 3)
    assert asset.ring_triangles.shape == (384, 3)
    assert asset.body_q.shape == (4, 12)
    assert asset.bond_topologies.shape == (969, 4)
    assert asset.bond_Dm_inv.shape == (969, 9)
    assert int(asset.bond_topologies.min()) == 0
    assert int(asset.bond_topologies.max()) < 192 + 1936
    np.testing.assert_allclose(
        asset.joint_theta,
        [-3.769009435060002, 0.10000000003512168, 2.258796965468916],
        rtol=0.0,
        atol=1e-14,
    )

    options = module.recommended_coupler_options()
    assert options["contact_d_hat"] == 8.0e-5
    assert options["adhesion_bond_default"] is False
    assert options["solver_linear_tol_rate"] == 1.0e-3
    assert options["solver_linear_solver"] == "linear_pcg"
    assert options["solver_linear_preconditioner"] == "diag"

    machine_options = module.recommended_machine_coupler_options()
    assert machine_options["contact_d_hat"] == 8.0e-5
    assert machine_options["contact_constitution"] == "consistent"
    assert "adhesion_bond_distance_lock" not in machine_options

    with np.load(asset.directory / "scotch3850_wound.npz", allow_pickle=False) as roll:
        assert "params_json" in roll.files
        assert "params" not in roll.files
        assert json.loads(bytes(roll["params_json"]).decode("utf-8"))["LOCK"] == 1

    wheel = next(
        link
        for link in ET.parse(asset.directory / "tape_dispenser.urdf").getroot().findall("link")
        if link.get("name") == "tape_wheel"
    )
    wheel_visuals = wheel.findall("visual")
    assert [mesh.get("filename") for mesh in wheel.findall("./visual/geometry/mesh")] == [
        "meshes/tape_wheel.glb",
        "meshes/scotch3850_ring.glb",
    ]
    ring_origin = wheel_visuals[1].find("origin")
    assert ring_origin is not None
    assert ring_origin.get("xyz") == "0 0 0"
    assert ring_origin.get("rpy") == "0 0 0"
    assert [mesh.get("filename") for mesh in wheel.findall("./collision/geometry/mesh")] == ["meshes/tape_wheel.glb"]

    ring_visual = trimesh.load(
        asset.directory / "meshes/scotch3850_ring.glb",
        force="mesh",
        process=False,
    )
    np.testing.assert_allclose(ring_visual.vertices, asset.ring_positions, rtol=0.0, atol=2e-9)
    np.testing.assert_array_equal(ring_visual.faces, asset.ring_triangles)

    machine_urdf = asset.directory / "tape_dispenser_machine.urdf"
    assert hashlib.sha256(machine_urdf.read_bytes()).hexdigest() == (
        "57bd75c90c0b5335aa0d3fed7ce536b0d676bae9020841387ff0248ce762aadc"
    )
    machine_root = ET.parse(machine_urdf).getroot()
    assert {link.get("name") for link in machine_root.findall("link")} == {
        "Cylinder",
        "blade",
        "sharp",
        "tape_cutter",
        "tape_dispenser_lower_mount",
        "tape_wheel",
    }
    assert {mesh.get("filename") for mesh in machine_root.findall(".//mesh")} == {
        "meshes/Cube.glb",
        "meshes/blade.glb",
        "meshes/cylinder.glb",
        "meshes/sharp.glb",
        "meshes/tape_cutter.glb",
        "meshes/tape_wheel.glb",
    }
    machine_wheel = next(link for link in machine_root.findall("link") if link.get("name") == "tape_wheel")
    assert [mesh.get("filename") for mesh in machine_wheel.findall("./visual/geometry/mesh")] == [
        "meshes/tape_wheel.glb"
    ]


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"format": "wrong", "version": 1, "files_sha256": {}}, "expected manifest format"),
        (
            {"format": "genesis.qipc.tape_dispenser", "version": 2, "files_sha256": {}},
            "expected manifest format",
        ),
        (
            {"format": "genesis.qipc.tape_dispenser", "version": 1, "files_sha256": {}},
            "missing required hashes",
        ),
        (
            {
                "format": "genesis.qipc.tape_dispenser",
                "version": 1,
                "files_sha256": {"../escape.glb": "0" * 64},
            },
            "unsafe manifest file path",
        ),
    ],
)
def test_tape_dispenser_rejects_invalid_manifest_contract(tmp_path, manifest, message):
    module = _module()
    directory = tmp_path / "asset"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(gs.GenesisException, match=message):
        module.TapeDispenserAsset.from_directory(directory)


def test_tape_dispenser_rejects_hash_mismatch(tmp_path):
    module = _module()
    directory = _copy_packaged_asset(module, tmp_path / "asset")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files_sha256"]["scotch3850_wound.npz"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(gs.GenesisException, match="SHA256 mismatch for 'scotch3850_wound.npz'"):
        module.TapeDispenserAsset.from_directory(directory)


def test_tape_dispenser_rejects_legacy_pickled_roll_params(tmp_path):
    module = _module()
    directory = _copy_packaged_asset(module, tmp_path / "asset")
    roll_path = directory / "scotch3850_wound.npz"
    _rewrite_npz(roll_path, {"params": np.array([{"must_not_execute": True}], dtype=object)})
    _refresh_hash(directory, "scotch3850_wound.npz")

    with pytest.raises(gs.GenesisException, match="legacy pickled roll params are not allowed"):
        module.TapeDispenserAsset.from_directory(directory)


def test_tape_dispenser_rejects_negative_tape_local_bond_id(tmp_path):
    module = _module()
    directory = _copy_packaged_asset(module, tmp_path / "asset")
    state_path = directory / "post_f249_static.npz"
    with np.load(state_path, allow_pickle=False) as state:
        local = state["bond_topo_local"].copy()
        assert state["bond_topo_owner"][0, 0] == 0
    local[0, 0] = -1
    _rewrite_npz(state_path, {"bond_topo_local": local})
    _refresh_hash(directory, "post_f249_static.npz")

    with pytest.raises(gs.GenesisException, match="bond topology local IDs must be non-negative"):
        module.TapeDispenserAsset.from_directory(directory)


def test_tape_dispenser_rejects_nonfinite_body_state(tmp_path):
    module = _module()
    directory = _copy_packaged_asset(module, tmp_path / "asset")
    state_path = directory / "post_f249_static.npz"
    with np.load(state_path, allow_pickle=False) as state:
        body_q = state["abd_q"].copy()
    body_q[0, 0] = np.nan
    _rewrite_npz(state_path, {"abd_q": body_q})
    _refresh_hash(directory, "post_f249_static.npz")

    with pytest.raises(gs.GenesisException, match="abd_q must contain only finite values"):
        module.TapeDispenserAsset.from_directory(directory)
