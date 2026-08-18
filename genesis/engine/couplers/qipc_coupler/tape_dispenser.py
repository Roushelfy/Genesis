"""Faithful post-f249 Scotch 3850 tape-dispenser component import.

The packaged snapshot is the canonicalized state immediately after reference
frames 0 through 249 have each executed ``pre_step`` and one QIPC step. The
dispenser is upside down, its tape end has been pulled 110 mm, all velocities
are zeroed, root/tape-end constraints are released, and tape contact with the
Cylinder and blade is enabled. Dynamic cutting is intentionally outside this
component: the 32 parked cut-spare vertices are omitted, while blade collision
is preserved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import get_assets_dir, tensor_to_array

from .contact import QIPCContactRegion
from .rigid_attachment import QIPCRigidAttachment
from .tape import TapeAsset, TapeBondClusterController, _verify_same_vertex_order, _write_obj, add_tape_bond_cluster

_ASSET_DIRECTORY = Path(get_assets_dir()) / "qipc" / "tape_dispenser_v2"
_ASSET_FORMAT = "genesis.qipc.tape_dispenser"
_ASSET_VERSION = 1
_STATE_FORMAT = "qipc.tape_dispenser_component_state"
_STATE_VERSION = 1
_RING_FORMAT = "qipc.tape_dispenser_ring_local"
_RING_VERSION = 1
_BODY_NAMES = ("tape_cutter", "Cylinder", "blade", "tape_wheel")
_JOINT_NAMES = ("Cylinder_axle", "blade_hinge", "tape_wheel_axle")
_MACHINE_URDF = "tape_dispenser_machine.urdf"
_MACHINE_PROXY_URDF = "tape_dispenser_machine_proxy.urdf"
_FULL_URDF = "tape_dispenser.urdf"
_FULL_PROXY_URDF = "tape_dispenser_proxy.urdf"
_MACHINE_MESH_FILES = frozenset(
    {
        "meshes/Cube.glb",
        "meshes/blade.glb",
        "meshes/cylinder.glb",
        "meshes/sharp.glb",
        "meshes/tape_cutter.glb",
        "meshes/tape_wheel.glb",
    }
)
_MESH_FILES = _MACHINE_MESH_FILES | {"meshes/scotch3850_ring.glb"}
_PROXY_MESH_FILES = frozenset(
    {
        "meshes/collision_proxies/Cube.glb",
        "meshes/collision_proxies/blade.glb",
        "meshes/collision_proxies/cylinder.glb",
        "meshes/collision_proxies/sharp.glb",
        "meshes/collision_proxies/tape_cutter.glb",
        "meshes/collision_proxies/tape_wheel.glb",
    }
)
_URDF_MESH_FILES = {
    _FULL_URDF: _MESH_FILES,
    _MACHINE_URDF: _MACHINE_MESH_FILES,
    _FULL_PROXY_URDF: _MESH_FILES | _PROXY_MESH_FILES,
    _MACHINE_PROXY_URDF: _MACHINE_MESH_FILES | _PROXY_MESH_FILES,
}
_REQUIRED_FILES = frozenset(
    {
        *_URDF_MESH_FILES,
        "scotch3850_wound.npz",
        "post_f249_static.npz",
        "ring_local.npz",
        *_MESH_FILES,
        *_PROXY_MESH_FILES,
    }
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REFERENCE_BASE_POSITION = np.array([0.1119, 0.10661028, -0.0485], dtype=np.float64)
_REFERENCE_TO_GENESIS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_array(array: np.ndarray, *, label: str) -> dict:
    try:
        value = json.loads(bytes(array).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        gs.raise_exception(f"TapeDispenserAsset: invalid {label}: {error}.")
    if not isinstance(value, dict):
        gs.raise_exception(f"TapeDispenserAsset: {label} must contain a JSON object.")
    return value


def _asset_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        gs.raise_exception("TapeDispenserAsset: manifest file paths must be non-empty POSIX relative paths.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or str(relative) != value or any(part in ("", ".", "..") for part in relative.parts):
        gs.raise_exception(f"TapeDispenserAsset: unsafe manifest file path '{value}'.")
    return value


def _asset_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            gs.raise_exception(f"TapeDispenserAsset: symlinks are not allowed in asset path '{relative}'.")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        gs.raise_exception(f"TapeDispenserAsset: missing packaged file '{relative}': {error}.")
    if not resolved.is_relative_to(root) or not resolved.is_file():
        gs.raise_exception(f"TapeDispenserAsset: asset path '{relative}' is not a regular file inside the package.")
    return resolved


def _validated_package(directory: str | os.PathLike[str]) -> tuple[Path, dict[str, str]]:
    try:
        root = Path(directory).expanduser().resolve(strict=True)
    except OSError as error:
        gs.raise_exception(f"TapeDispenserAsset: asset directory is unavailable: {error}.")
    if not root.is_dir():
        gs.raise_exception(f"TapeDispenserAsset: asset root '{root}' is not a directory.")

    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        gs.raise_exception(f"TapeDispenserAsset: missing regular manifest '{manifest_path}'.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        gs.raise_exception(f"TapeDispenserAsset: invalid manifest '{manifest_path}': {error}.")
    if not isinstance(manifest, dict):
        gs.raise_exception("TapeDispenserAsset: manifest must contain a JSON object.")
    if manifest.get("format") != _ASSET_FORMAT or manifest.get("version") != _ASSET_VERSION:
        gs.raise_exception(f"TapeDispenserAsset: expected manifest format '{_ASSET_FORMAT}' version {_ASSET_VERSION}.")
    hashes = manifest.get("files_sha256")
    if not isinstance(hashes, dict):
        gs.raise_exception("TapeDispenserAsset: manifest files_sha256 must be an object.")

    normalized: dict[str, str] = {}
    for raw_relative, expected in hashes.items():
        relative = _asset_relative_path(raw_relative)
        if not isinstance(expected, str) or _SHA256_PATTERN.fullmatch(expected) is None:
            gs.raise_exception(f"TapeDispenserAsset: invalid SHA256 for '{relative}'.")
        normalized[relative] = expected
    missing = sorted(_REQUIRED_FILES - normalized.keys())
    if missing:
        gs.raise_exception("TapeDispenserAsset: manifest is missing required hashes: " + ", ".join(missing) + ".")

    for relative, expected in normalized.items():
        path = _asset_file(root, relative)
        actual = _sha256(path)
        if actual != expected:
            gs.raise_exception(f"TapeDispenserAsset: SHA256 mismatch for '{relative}': {actual}, expected {expected}.")

    for urdf_relative, expected_meshes in _URDF_MESH_FILES.items():
        urdf_path = _asset_file(root, urdf_relative)
        try:
            urdf_root = ET.parse(urdf_path).getroot()
        except (OSError, ET.ParseError) as error:
            gs.raise_exception(f"TapeDispenserAsset: invalid {urdf_relative}: {error}.")
        referenced_meshes: set[str] = set()
        for mesh in urdf_root.findall(".//mesh"):
            relative = _asset_relative_path(mesh.get("filename"))
            if relative not in normalized:
                gs.raise_exception(
                    f"TapeDispenserAsset: {urdf_relative} mesh '{relative}' is not covered by the manifest."
                )
            _asset_file(root, relative)
            referenced_meshes.add(relative)
        if referenced_meshes != expected_meshes:
            gs.raise_exception(
                f"TapeDispenserAsset: {urdf_relative} mesh set does not match the version-1 dispenser contract."
            )
    return root, normalized


def _require_npz_keys(archive, required: set[str], *, label: str) -> None:
    missing = sorted(required - set(archive.files))
    if missing:
        gs.raise_exception(f"TapeDispenserAsset: {label} is missing keys: {', '.join(missing)}.")


def _float_array(value: np.ndarray, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.floating) or value.shape != shape:
        gs.raise_exception(f"TapeDispenserAsset: {name} must be a floating array with shape {shape}.")
    array = np.ascontiguousarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        gs.raise_exception(f"TapeDispenserAsset: {name} must contain only finite values.")
    return array


def _integer_array(value: np.ndarray, shape: tuple[int, ...], *, name: str, dtype) -> np.ndarray:
    if not np.issubdtype(value.dtype, np.integer) or value.shape != shape:
        gs.raise_exception(f"TapeDispenserAsset: {name} must be an integer array with shape {shape}.")
    return np.ascontiguousarray(value, dtype=dtype)


@dataclass(frozen=True)
class TapeDispenserAsset:
    """Validated packaged geometry and canonical post-f249 state."""

    directory: Path
    roll: TapeAsset
    tape_positions: np.ndarray
    tape_triangles: np.ndarray
    body_names: tuple[str, ...]
    body_q: np.ndarray
    joint_names: tuple[str, ...]
    joint_theta: np.ndarray
    ring_positions: np.ndarray
    ring_triangles: np.ndarray
    bond_topologies: np.ndarray
    bond_Dm_inv: np.ndarray
    bond_V0: np.ndarray
    bond_d_rest: np.ndarray
    bond_kappa: np.ndarray
    bond_release_force: np.ndarray
    bond_release_strain: np.ndarray
    bond_release_gap: np.ndarray
    bond_release_slip: np.ndarray
    bond_age: np.ndarray

    @property
    def bond_topos(self) -> np.ndarray:
        """Frozen bond topology in compact `[ring | tape]` vertex space."""
        return self.bond_topologies

    @property
    def bond_topos_space(self) -> str:
        return "global"

    @property
    def bond_fem_gvo(self) -> int:
        return len(self.ring_positions)

    @property
    def tape_tris(self) -> np.ndarray:
        return self.tape_triangles

    @classmethod
    def packaged(cls) -> TapeDispenserAsset:
        return cls.from_directory(_ASSET_DIRECTORY)

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str]) -> TapeDispenserAsset:
        root, _hashes = _validated_package(directory)
        state_required = {
            "abd_q",
            "tape_x",
            "joint_theta",
            "bond_topo_owner",
            "bond_topo_local",
            "bond_Dm_inv",
            "bond_V0",
            "bond_d_rest",
            "bond_kappa",
            "bond_release_force",
            "bond_release_strain",
            "bond_release_gap",
            "bond_release_slip",
            "bond_age",
            "manifest_json",
        }
        try:
            with np.load(root / "post_f249_static.npz", allow_pickle=False) as state:
                _require_npz_keys(state, state_required, label="post_f249_static.npz")
                state_manifest = _json_array(state["manifest_json"], label="state manifest_json")
                if state_manifest.get("format") != _STATE_FORMAT or state_manifest.get("version") != _STATE_VERSION:
                    gs.raise_exception(
                        f"TapeDispenserAsset: expected state format '{_STATE_FORMAT}' version {_STATE_VERSION}."
                    )
                layout = state_manifest.get("layout")
                if not isinstance(layout, dict):
                    gs.raise_exception("TapeDispenserAsset: state layout must be an object.")
                layout_required = {
                    "active_bonds",
                    "body_names",
                    "joint_names",
                    "n_cut_spare_vertices",
                    "n_original_tape_vertices",
                    "n_ring_vertices",
                    "n_tape_vertices",
                }
                missing_layout = sorted(layout_required - layout.keys())
                if missing_layout:
                    gs.raise_exception(
                        "TapeDispenserAsset: state layout is missing fields: " + ", ".join(missing_layout) + "."
                    )
                body_names = tuple(layout["body_names"])
                joint_names = tuple(layout["joint_names"])
                if body_names != _BODY_NAMES or joint_names != _JOINT_NAMES:
                    gs.raise_exception(
                        "TapeDispenserAsset: state body/joint names do not match the version-1 contract."
                    )
                count_values = (
                    layout["active_bonds"],
                    layout["n_cut_spare_vertices"],
                    layout["n_original_tape_vertices"],
                    layout["n_ring_vertices"],
                    layout["n_tape_vertices"],
                )
                if any(not isinstance(value, int) or isinstance(value, bool) for value in count_values):
                    gs.raise_exception("TapeDispenserAsset: state layout counts must be integers.")
                n_bonds = layout["active_bonds"]
                n_spares = layout["n_cut_spare_vertices"]
                n_tape = layout["n_original_tape_vertices"]
                n_ring = layout["n_ring_vertices"]
                n_tape_total = layout["n_tape_vertices"]
                if n_bonds <= 0 or n_spares < 0 or n_tape <= 0 or n_ring <= 0 or n_tape_total != n_tape + n_spares:
                    gs.raise_exception("TapeDispenserAsset: state layout counts are inconsistent.")

                tape_all = _float_array(state["tape_x"], (n_tape_total, 3), name="tape_x")
                tape_positions = np.ascontiguousarray(tape_all[:n_tape])
                body_q = _float_array(state["abd_q"], (len(body_names), 12), name="abd_q")
                joint_theta = _float_array(
                    state["joint_theta"],
                    (len(joint_names),),
                    name="joint_theta",
                )
                owners = _integer_array(
                    state["bond_topo_owner"],
                    (n_bonds, 4),
                    name="bond_topo_owner",
                    dtype=np.uint8,
                )
                local = _integer_array(
                    state["bond_topo_local"],
                    (n_bonds, 4),
                    name="bond_topo_local",
                    dtype=np.int32,
                )
                if not np.isin(owners, (0, 1)).all():
                    gs.raise_exception("TapeDispenserAsset: bond topology owner must be tape=0 or ring=1.")
                if np.any(local < 0):
                    gs.raise_exception("TapeDispenserAsset: bond topology local IDs must be non-negative.")
                if np.any((owners == 0) & (local >= n_tape)) or np.any((owners == 1) & (local >= n_ring)):
                    gs.raise_exception("TapeDispenserAsset: bond topology references omitted or out-of-range vertices.")
                bond_topologies = np.where(owners == 1, local, local + n_ring).astype(np.int32)

                state_values = {
                    "body_q": body_q,
                    "joint_theta": joint_theta,
                    "bond_Dm_inv": _float_array(
                        state["bond_Dm_inv"],
                        (n_bonds, 9),
                        name="bond_Dm_inv",
                    ),
                    "bond_V0": _float_array(state["bond_V0"], (n_bonds,), name="bond_V0"),
                    "bond_d_rest": _float_array(
                        state["bond_d_rest"],
                        (n_bonds,),
                        name="bond_d_rest",
                    ),
                    "bond_kappa": _float_array(state["bond_kappa"], (n_bonds,), name="bond_kappa"),
                    "bond_release_force": _float_array(
                        state["bond_release_force"],
                        (n_bonds,),
                        name="bond_release_force",
                    ),
                    "bond_release_strain": _float_array(
                        state["bond_release_strain"],
                        (n_bonds,),
                        name="bond_release_strain",
                    ),
                    "bond_release_gap": _float_array(
                        state["bond_release_gap"],
                        (n_bonds,),
                        name="bond_release_gap",
                    ),
                    "bond_release_slip": _float_array(
                        state["bond_release_slip"],
                        (n_bonds,),
                        name="bond_release_slip",
                    ),
                    "bond_age": _integer_array(
                        state["bond_age"],
                        (n_bonds,),
                        name="bond_age",
                        dtype=np.int32,
                    ),
                }
        except (OSError, ValueError, KeyError, TypeError) as error:
            gs.raise_exception(f"TapeDispenserAsset: invalid post_f249_static.npz: {error}.")

        if np.any(state_values["bond_V0"] <= 0.0) or np.any(state_values["bond_kappa"] <= 0.0):
            gs.raise_exception("TapeDispenserAsset: bond V0 and kappa must be positive.")
        for name in ("bond_release_force", "bond_release_strain", "bond_release_gap", "bond_release_slip"):
            if np.any(state_values[name] < 0.0):
                gs.raise_exception(f"TapeDispenserAsset: {name} must be non-negative.")
        if np.any(state_values["bond_age"] < 0):
            gs.raise_exception("TapeDispenserAsset: bond_age must be non-negative.")

        ring_required = {"ring_local_positions", "ring_triangles", "manifest_json"}
        try:
            with np.load(root / "ring_local.npz", allow_pickle=False) as ring:
                _require_npz_keys(ring, ring_required, label="ring_local.npz")
                ring_manifest = _json_array(ring["manifest_json"], label="ring manifest_json")
                if ring_manifest.get("format") != _RING_FORMAT or ring_manifest.get("version") != _RING_VERSION:
                    gs.raise_exception(
                        f"TapeDispenserAsset: expected ring format '{_RING_FORMAT}' version {_RING_VERSION}."
                    )
                if ring_manifest.get("ring_vertex_count") != n_ring:
                    gs.raise_exception("TapeDispenserAsset: ring manifest vertex count does not match state layout.")
                n_ring_triangles = ring_manifest.get("ring_triangle_count")
                if not isinstance(n_ring_triangles, int) or isinstance(n_ring_triangles, bool) or n_ring_triangles <= 0:
                    gs.raise_exception("TapeDispenserAsset: ring triangle count must be a positive integer.")
                ring_positions = _float_array(
                    ring["ring_local_positions"],
                    (n_ring, 3),
                    name="ring_local_positions",
                )
                ring_triangles = _integer_array(
                    ring["ring_triangles"],
                    (n_ring_triangles, 3),
                    name="ring_triangles",
                    dtype=np.int32,
                )
        except (OSError, ValueError, KeyError, TypeError) as error:
            gs.raise_exception(f"TapeDispenserAsset: invalid ring_local.npz: {error}.")
        if int(ring_triangles.min()) < 0 or int(ring_triangles.max()) >= n_ring:
            gs.raise_exception("TapeDispenserAsset: ring triangle indices are out of range.")

        roll_path = root / "scotch3850_wound.npz"
        try:
            with np.load(roll_path, allow_pickle=False) as roll_archive:
                _require_npz_keys(
                    roll_archive,
                    {"params_json", "tape_positions", "tape_tris"},
                    label="scotch3850_wound.npz",
                )
                if "params" in roll_archive.files:
                    gs.raise_exception("TapeDispenserAsset: legacy pickled roll params are not allowed.")
                _float_array(roll_archive["tape_positions"], (n_tape, 3), name="roll tape_positions")
                raw_roll_triangles = roll_archive["tape_tris"]
                if not np.issubdtype(raw_roll_triangles.dtype, np.integer):
                    gs.raise_exception("TapeDispenserAsset: roll tape_tris must use an integer dtype.")
            roll = TapeAsset.from_npz(str(roll_path), allow_legacy_pickle=False)
        except (OSError, ValueError, KeyError, TypeError) as error:
            gs.raise_exception(f"TapeDispenserAsset: invalid scotch3850_wound.npz: {error}.")
        if roll.tape_positions.shape != tape_positions.shape:
            gs.raise_exception(
                "TapeDispenserAsset: active post-f249 tape layout does not match the wound tape topology."
            )
        if roll.tape_tris.ndim != 2 or roll.tape_tris.shape[1] != 3 or len(roll.tape_tris) == 0:
            gs.raise_exception("TapeDispenserAsset: roll tape triangles must have shape (n, 3).")
        if int(roll.tape_tris.min()) < 0 or int(roll.tape_tris.max()) >= n_tape:
            gs.raise_exception("TapeDispenserAsset: roll tape triangle indices are out of range.")
        positive_roll_values = (
            roll.nx,
            roll.nz,
            roll.thick,
            roll.tape_length,
            roll.width,
            roll.youngs,
            roll.density,
            roll.hub_r_outer,
            roll.hub_r_inner,
            roll.hub_height,
            roll.d_hat,
        )
        if any(not np.isfinite(value) or value <= 0 for value in positive_roll_values):
            gs.raise_exception("TapeDispenserAsset: roll geometry and material values must be finite and positive.")
        if not np.isfinite(roll.bending_e) or roll.bending_e < 0.0 or not (-1.0 < roll.poisson < 0.5):
            gs.raise_exception("TapeDispenserAsset: roll bending modulus or Poisson ratio is invalid.")
        if roll.hub_r_inner >= roll.hub_r_outer:
            gs.raise_exception("TapeDispenserAsset: roll hub inner radius must be smaller than its outer radius.")

        return cls(
            directory=root,
            roll=roll,
            tape_positions=tape_positions,
            tape_triangles=roll.tape_tris,
            body_names=body_names,
            body_q=state_values["body_q"],
            joint_names=joint_names,
            joint_theta=state_values["joint_theta"],
            ring_positions=ring_positions,
            ring_triangles=ring_triangles,
            bond_topologies=bond_topologies,
            bond_Dm_inv=state_values["bond_Dm_inv"],
            bond_V0=state_values["bond_V0"],
            bond_d_rest=state_values["bond_d_rest"],
            bond_kappa=state_values["bond_kappa"],
            bond_release_force=state_values["bond_release_force"],
            bond_release_strain=state_values["bond_release_strain"],
            bond_release_gap=state_values["bond_release_gap"],
            bond_release_slip=state_values["bond_release_slip"],
            bond_age=state_values["bond_age"],
        )


@dataclass(frozen=True)
class TapeDispenser:
    """Genesis entities and stable QIPC handles created by the importer."""

    machine: object
    tape: object
    ring: QIPCRigidAttachment
    cylinder_region: QIPCContactRegion
    blade_region: QIPCContactRegion
    sharp_region: QIPCContactRegion
    ring_region: QIPCContactRegion
    asset: TapeDispenserAsset
    lifecycle: TapeBondClusterController | None


def recommended_machine_coupler_options() -> dict[str, object]:
    """QIPCCouplerOptions for the rigid dispenser without tape or hub."""
    return {
        "contact_enable": True,
        "contact_d_hat": 8.0e-5,
        "init_collision_pair_capacity": 100_000,
        "contact_friction": 0.2,
        "contact_resistance": 1.0e5,
        "contact_constitution": "consistent",
        "solver_newton_velocity_tol": 1.0e-2,
        "solver_newton_max_iter": 128,
        "solver_linear_max_iter": 800,
        "solver_linear_tol_rate": 1.0e-3,
        "solver_linear_preconditioner": "mas",
        "solver_abd_preconditioner": "tree",
        "solver_linear_solver": "partition_pcg",
        "solver_line_search_max_iter": 16,
    }


def recommended_coupler_options() -> dict[str, object]:
    """QIPCCouplerOptions matching the frozen reference scene."""
    options = recommended_machine_coupler_options()
    options.update(
        {
            "contact_constitution": "adhesive",
            "adhesion_bond_distance_lock": True,
            "adhesion_bond_distance_lock_ratio": 1.5,
            "adhesion_bond_max_bonds": 1,
            "adhesion_bond_default": False,
            "adhesion_bond_kappa": 3.0e7,
            "adhesion_bond_release_force": 0.5,
            "adhesion_bond_lock_floor_ratio": 0.5,
        }
    )
    return options


def _transformed_machine_state(asset: TapeDispenserAsset, pos, euler):
    position = np.asarray(pos, dtype=np.float64).reshape(3)
    extra_rotation = gu.quat_to_R(gu.xyz_to_quat(np.asarray(euler, dtype=np.float64), degrees=True))
    world_rotation = extra_rotation @ _REFERENCE_TO_GENESIS

    body_q: dict[str, np.ndarray] = {}
    for name, source_q in zip(asset.body_names, asset.body_q, strict=True):
        transformed = np.empty(12, dtype=np.float64)
        transformed[:3] = world_rotation @ (source_q[:3] - _REFERENCE_BASE_POSITION) + position
        transformed[3:] = (world_rotation @ source_q[3:].reshape(3, 3)).reshape(-1)
        body_q[name] = transformed
    return position, world_rotation, body_q


def _add_machine_entity(scene, asset, urdf_file, position, root_rotation, machine_surface):
    return scene.add_entity(
        morph=gs.morphs.URDF(
            file=str(asset.directory / urdf_file),
            pos=tuple(position),
            quat=tuple(gu.R_to_quat(root_rotation)),
            fixed=False,
            align=False,
            merge_fixed_links=False,
            convexify=False,
            decimate=False,
            watertighten=None,
            preserve_collision_mesh_topology=True,
            file_meshes_are_zup=True,
            requires_jac_and_IK=False,
        ),
        material=gs.materials.Rigid(
            coup_friction=0.2,
            contact_resistance=1.0e5,
            qipc_abd_kappa=1.0e8,
            qipc_kappa_pivot=1.0e7,
            qipc_kappa_axis=1.0e7,
            qipc_default_kp=0.0,
            qipc_default_kv=0.0,
            qipc_d_hat=asset.roll.d_hat,
            qipc_self_contact=False,
        ),
        surface=machine_surface,
    )


def add_tape_dispenser_machine(
    scene,
    asset: TapeDispenserAsset | None = None,
    *,
    pos=(0.0, 0.0, 0.0),
    euler=(0.0, 0.0, 0.0),
    machine_surface=None,
    collision_proxies: bool = True,
):
    """Add only the ringless rigid dispenser URDF before scene build.

    ``pos`` is the desired ``tape_cutter`` root origin in Genesis Z-up and
    ``euler`` is an additional rotation applied to the canonical upside-down
    orientation. The three articulated joints start at their URDF zero values.
    No FEM tape, hub attachment, contact region, adhesion, or bond is created.
    The root is free and all joints are unactuated.
    """
    asset = TapeDispenserAsset.packaged() if asset is None else asset
    position, _world_rotation, body_q = _transformed_machine_state(asset, pos, euler)
    root_rotation = body_q["tape_cutter"][3:].reshape(3, 3)
    return _add_machine_entity(
        scene,
        asset,
        _MACHINE_PROXY_URDF if collision_proxies else _MACHINE_URDF,
        position,
        root_rotation,
        machine_surface,
    )


def add_tape_dispenser(
    scene,
    asset: TapeDispenserAsset | None = None,
    *,
    pos=(0.0, 0.0, 0.0),
    euler=(0.0, 0.0, 0.0),
    machine_surface=None,
    tape_surface=None,
    rigid_cluster: bool = False,
    cluster_kappa: float = 1.0e8,
    cluster_collar: int = 3,
    cluster_detach_displacement_ratio: float = 5.0,
    collision_proxies: bool = True,
) -> TapeDispenser:
    """Add the upside-down canonical post-f249 dispenser before scene build.

    ``pos`` is the desired ``tape_cutter`` root origin in Genesis Z-up.
    ``euler`` is an additional proper rotation applied around that origin. The
    returned machine has a free root and unactuated joints; "static" means zero
    initial velocity, not permanently fixed geometry.
    """
    coupler = scene.sim.coupler
    required = (
        "add_rigid_attachment",
        "add_contact_region",
        "assign_contact_region",
        "set_contact_pair",
        "set_rigid_initial_state",
    )
    missing = [name for name in required if not hasattr(coupler, name)]
    if missing:
        gs.raise_exception(
            "add_tape_dispenser requires the QIPC coupler component APIs (missing: " + ", ".join(missing) + ")."
        )
    asset = TapeDispenserAsset.packaged() if asset is None else asset

    position, world_rotation, body_q = _transformed_machine_state(asset, pos, euler)
    root_rotation = body_q["tape_cutter"][3:].reshape(3, 3)

    machine = _add_machine_entity(
        scene,
        asset,
        _FULL_PROXY_URDF if collision_proxies else _FULL_URDF,
        position,
        root_rotation,
        machine_surface,
    )
    coupler.set_rigid_initial_state(
        machine,
        body_q=body_q,
        joint_theta=dict(zip(asset.joint_names, asset.joint_theta.tolist(), strict=True)),
    )

    ring = coupler.add_rigid_attachment(
        machine,
        link="tape_wheel",
        name="scotch3850_ring",
        vertices=asset.ring_positions,
        triangles=asset.ring_triangles,
    )

    tape_world = (asset.tape_positions - _REFERENCE_BASE_POSITION) @ world_rotation.T + position
    temporary_directory = tempfile.mkdtemp(prefix="qipc_tape_dispenser_")
    tape_obj = os.path.join(temporary_directory, "post_f249_tape.obj")
    _write_obj(tape_obj, tape_world, asset.tape_triangles)
    tape = scene.add_entity(
        morph=gs.morphs.Mesh(file=tape_obj, scale=1.0),
        material=gs.materials.FEM.Cloth(
            E=asset.roll.youngs,
            nu=asset.roll.poisson,
            rho=asset.roll.density,
            thickness=asset.roll.thick,
            bending_stiffness=asset.roll.bending_e,
            membrane="stvk",
            bending_model="hinge",
            strain_limit_multiplier=0.0,
            friction_mu=0.5,
            contact_resistance=1.0e5,
        ),
        surface=tape_surface,
    )
    _verify_same_vertex_order(tape_world, tensor_to_array(tape.init_positions).astype(np.float64))
    coupler.set_fem_rest_positions(tape, asset.roll.flat_rest_positions())

    cylinder = coupler.add_contact_region("tape_dispenser_Cylinder", friction=0.2, resistance=1.0e5)
    blade = coupler.add_contact_region("tape_dispenser_blade", friction=0.2, resistance=1.0e5)
    sharp = coupler.add_contact_region("tape_dispenser_sharp", friction=0.2, resistance=1.0e5)
    ring_region = coupler.add_contact_region("scotch3850_ring_hub", friction=0.5, resistance=1.0e5)
    coupler.assign_contact_region(cylinder, machine, link="Cylinder")
    coupler.assign_contact_region(blade, machine, link="blade")
    coupler.assign_contact_region(sharp, machine, link="sharp")
    coupler.assign_contact_region(ring_region, machine, attachment=ring)

    coupler.set_contact_pair(tape, machine, enabled=False, friction=0.0, resistance=1.0e5)
    coupler.set_contact_pair(tape, sharp, enabled=False, friction=0.2, resistance=1.0e5)
    coupler.set_contact_pair(tape, cylinder, enabled=True, friction=0.2, resistance=1.0e5)
    coupler.set_contact_pair(tape, blade, enabled=True, friction=0.2, resistance=1.0e5)

    machine_regions = (machine, cylinder, blade, sharp)
    for index, first in enumerate(machine_regions):
        coupler.set_contact_pair(ring_region, first, enabled=False, friction=0.0, resistance=1.0e5)
        for second in machine_regions[index:]:
            coupler.set_contact_pair(first, second, enabled=False, friction=0.0, resistance=1.0e5)

    coupler.add_adhesion(
        tape,
        tape,
        Cn=0.0,
        Ct=0.0,
        W=0.0,
        eta=1.0,
        sticky=(0, 1, 1, 1),
        friction=0.5,
        resistance=1.0e5,
        distance_lock=True,
        distance_lock_ratio=1.5,
        distance_lock_rest_snap=False,
        release_force=0.5,
    )
    coupler.add_adhesion(
        tape,
        ring_region,
        Cn=0.0,
        Ct=0.0,
        W=0.0,
        eta=1.0,
        sticky=(0, 0, 1, 1),
        friction=0.5,
        resistance=1.0e5,
        distance_lock=True,
        distance_lock_ratio=1.5,
        distance_lock_rest_snap=False,
        release_force=0.5,
    )

    transformed_Dm_inv = (asset.bond_Dm_inv.reshape(-1, 3, 3) @ world_rotation.T).reshape(-1, 9)
    coupler.adhesion.add_bond_state_request(
        tape,
        rigid_source=ring,
        topologies=asset.bond_topologies,
        source_fem_global_offset=ring.n_vertices,
        Dm_inv=transformed_Dm_inv,
        V0=asset.bond_V0,
        d_rest=asset.bond_d_rest,
        kappa=asset.bond_kappa,
        release_force=asset.bond_release_force,
        release_strain=asset.bond_release_strain,
        release_gap=asset.bond_release_gap,
        release_slip=asset.bond_release_slip,
        age=asset.bond_age,
    )

    lifecycle = None
    if rigid_cluster:
        never_member = np.zeros(len(asset.tape_positions), dtype=bool)
        row_width = int(asset.roll.nz) + 1
        never_member[-row_width:] = True
        lifecycle = add_tape_bond_cluster(
            scene,
            tape,
            asset,
            kappa=cluster_kappa,
            collar=cluster_collar,
            detach_displacement=cluster_detach_displacement_ratio * asset.roll.d_hat,
            proxy_entity=machine,
            proxy_link="tape_wheel",
            structured_row_width=row_width,
            never_member=never_member,
        )

    return TapeDispenser(
        machine=machine,
        tape=tape,
        ring=ring,
        cylinder_region=cylinder,
        blade_region=blade,
        sharp_region=sharp,
        ring_region=ring_region,
        asset=asset,
        lifecycle=lifecycle,
    )
