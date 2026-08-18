"""Validated QIPC tape-to-table component assets.

The component is authored in QIPC's right-handed, Y-up world frame with the
table top at ``y=0``. Loading preserves that frame and every vertex ID. Scene
builders must explicitly request a placement transform; this keeps topology
certificates and geometry in one well-defined source namespace.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import genesis as gs

_COMPONENT_SCHEMA = "qipc_tape_table_component"
_COMPONENT_SCHEMA_VERSION = 2
_COMPONENT_SCHEMA_ABI = "qipc_tape_table_component_v2"
_INTERNAL_TOPOLOGY_SPACE = "hub_tape_local_v2"
_TABLE_TOPOLOGY_SPACE = "table_tape_local_v2"
_BOND_STATE_MODE = "grouped_frozen_state_v2"
_DM_INV_TRANSFORM = "right_multiply_placement_rotation_transpose"
_HEX_40 = re.compile(r"[0-9a-f]{40}")
_HEX_64 = re.compile(r"[0-9a-f]{64}")

_JSON_FIELDS = frozenset(
    {
        "params_json",
        "authoring_params_json",
        "body_roles_json",
        "bond_state_manifest_json",
        "internal_contact_policy_json",
        "internal_bond_policy_json",
        "table_contact_policy_json",
        "table_bond_policy_json",
    }
)

_STATE_FIELDS = (
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

_REQUIRED_FIELDS = frozenset(
    {
        "component_schema",
        "component_schema_version",
        "component_schema_abi",
        "units",
        "up_axis",
        "handedness",
        "component_frame",
        "component_origin",
        "component_transform",
        "source_qipc_commit",
        "source_asset_name",
        "source_asset_sha256",
        *_JSON_FIELDS,
        "tape_positions",
        "tape_rest_positions",
        "tape_velocities",
        "tape_tris",
        "tape_is_fixed",
        "hub_positions",
        "hub_tris",
        "hub_transform",
        "hub_affine_velocity",
        "hub_is_fixed",
        "table_positions",
        "table_tets",
        "table_surface_tris",
        "table_transform",
        "table_affine_velocity",
        "table_is_fixed",
        "internal_bond_topologies",
        "internal_bond_topology_space",
        "internal_bond_fem_offset",
        "internal_bond_seed_rest_height",
        "table_bond_topologies",
        "table_bond_topology_space",
        "table_bond_fem_offset",
        "table_bond_seed_rest_height",
        "bond_state_mode",
        "bond_frozen_state_restore_supported",
        *(f"{group}_bond_state_{field}" for group in ("internal", "table") for field in _STATE_FIELDS),
        "tail_vertex_ids",
        "tail_rows",
        "terminal_row_ids",
        "attached_length",
        "nx",
        "nz",
        "tape_length",
        "width",
        "hub_r_outer",
        "hub_r_inner",
        "hub_height",
        "thick",
        "d_hat",
        "youngs",
        "poisson",
        "density",
        "bending_e",
    }
)

_EXPECTED_BODY_ROLES: dict[str, object] = {
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


def _fail(message: str) -> None:
    gs.raise_exception(f"TapeTableComponentAsset: {message}")


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    try:
        return archive[name]
    except ValueError as error:
        _fail(f"field '{name}' cannot be loaded without pickle: {error}.")


def _bytes_scalar(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = _array(archive, name)
    if value.shape != () or value.dtype.kind != "S":
        _fail(f"{name} must be a scalar byte string.")
    try:
        return value.item().decode("ascii")
    except UnicodeDecodeError as error:
        _fail(f"{name} must contain ASCII: {error}.")


def _utf8_bytes(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.uint8) or value.ndim != 1 or not len(value):
        _fail(f"{name} must be a non-empty uint8 byte vector.")
    try:
        return value.tobytes().decode("utf-8")
    except UnicodeDecodeError as error:
        _fail(f"{name} must contain UTF-8: {error}.")


def _json_object(archive: np.lib.npyio.NpzFile, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            _utf8_bytes(archive, name),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, ValueError) as error:
        _fail(f"invalid {name}: {error}.")
    if not isinstance(value, dict):
        _fail(f"{name} must contain a JSON object.")
    return value


def _int_scalar(archive: np.lib.npyio.NpzFile, name: str) -> int:
    value = _array(archive, name)
    if value.shape != () or value.dtype != np.dtype(np.int32):
        _fail(f"{name} must be an int32 scalar.")
    return int(value)


def _float_scalar(archive: np.lib.npyio.NpzFile, name: str) -> float:
    value = _array(archive, name)
    if value.shape != () or value.dtype != np.dtype(np.float64):
        _fail(f"{name} must be a float64 scalar.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite.")
    return result


def _float_array(archive: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.float64) or value.shape != shape:
        _fail(f"{name} must be a float64 array with shape {shape}.")
    result = np.ascontiguousarray(value)
    if not np.isfinite(result).all():
        _fail(f"{name} must contain only finite values.")
    return _readonly(result)


def _int_array(archive: np.lib.npyio.NpzFile, name: str, shape: tuple[int, ...]) -> np.ndarray:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.int32) or value.shape != shape:
        _fail(f"{name} must be an int32 array with shape {shape}.")
    return _readonly(np.ascontiguousarray(value))


def _point_array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.float64) or value.ndim != 2 or value.shape[1] != 3 or not len(value):
        _fail(f"{name} must be a non-empty float64 array with shape (n, 3).")
    result = np.ascontiguousarray(value)
    if not np.isfinite(result).all():
        _fail(f"{name} must contain only finite values.")
    return _readonly(result)


def _index_array(archive: np.lib.npyio.NpzFile, name: str, width: int, *, allow_empty: bool = False) -> np.ndarray:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.int32) or value.ndim != 2 or value.shape[1] != width:
        _fail(f"{name} must be an int32 array with shape (n, {width}).")
    if not allow_empty and not len(value):
        _fail(f"{name} must not be empty.")
    return _readonly(np.ascontiguousarray(value))


def _id_array(archive: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    value = _array(archive, name)
    if value.dtype != np.dtype(np.int32) or value.ndim != 1 or not len(value):
        _fail(f"{name} must be a non-empty int32 vector.")
    return _readonly(np.ascontiguousarray(value))


def _validate_rigid_transform(transform: np.ndarray, *, name: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        _fail(f"{name} must be a finite 4x4 transform.")
    if not np.array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
        _fail(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12):
        _fail(f"{name} rotation must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _fail(f"{name} rotation must be right-handed.")


def _validate_proper_affine_transform(transform: np.ndarray, *, name: str) -> None:
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        _fail(f"{name} must be a finite 4x4 transform.")
    if not np.array_equal(transform[3], np.array([0.0, 0.0, 0.0, 1.0])):
        _fail(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    determinant = float(np.linalg.det(transform[:3, :3]))
    if not math.isfinite(determinant) or determinant <= 0.0:
        _fail(f"{name} linear block must be invertible and orientation-preserving.")


def _validate_indices(indices: np.ndarray, vertex_count: int, *, name: str) -> None:
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= vertex_count):
        _fail(f"{name} contains a vertex outside [0, {vertex_count}).")
    if any(len(set(row.tolist())) != indices.shape[1] for row in indices):
        _fail(f"{name} contains a degenerate element with repeated vertex IDs.")
    canonical = np.sort(indices, axis=1)
    if len(np.unique(canonical, axis=0)) != len(indices):
        _fail(f"{name} contains duplicate elements.")


def _validate_pt_indices(indices: np.ndarray, vertex_count: int, *, name: str) -> None:
    if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= vertex_count):
        _fail(f"{name} contains a vertex outside [0, {vertex_count}).")
    if any(len(set(row.tolist())) != indices.shape[1] for row in indices):
        _fail(f"{name} contains a degenerate element with repeated vertex IDs.")
    canonical = np.concatenate((indices[:, :1], np.sort(indices[:, 1:], axis=1)), axis=1)
    if len(np.unique(canonical, axis=0)) != len(indices):
        _fail(f"{name} contains duplicate point-triangle elements.")


def _validate_triangle_geometry(positions: np.ndarray, triangles: np.ndarray, *, name: str) -> None:
    edges_0 = positions[triangles[:, 1]] - positions[triangles[:, 0]]
    edges_1 = positions[triangles[:, 2]] - positions[triangles[:, 0]]
    if np.any(np.linalg.norm(np.cross(edges_0, edges_1), axis=1) == 0.0):
        _fail(f"{name} contains a zero-area triangle.")


def _validate_closed_surface(triangles: np.ndarray, *, name: str) -> None:
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        for first, second in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count != 2 for count in edge_counts.values()):
        _fail(f"{name} must be a closed two-manifold triangle surface.")


def _tet_boundary_owners(tetrahedra: np.ndarray) -> dict[tuple[int, int, int], tuple[int, int]]:
    owners: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for tet_index, (a, b, c, d) in enumerate(tetrahedra.tolist()):
        for face, opposite in (((a, b, c), d), ((a, b, d), c), ((a, c, d), b), ((b, c, d), a)):
            owners.setdefault(tuple(sorted(face)), []).append((tet_index, opposite))
    nonmanifold = [face for face, face_owners in owners.items() if len(face_owners) > 2]
    if nonmanifold:
        _fail("table_tets contains a non-manifold face shared by more than two tetrahedra.")
    return {face: face_owners[0] for face, face_owners in owners.items() if len(face_owners) == 1}


def _validate_table_mesh(positions: np.ndarray, tetrahedra: np.ndarray, surface: np.ndarray) -> None:
    _validate_indices(tetrahedra, len(positions), name="table_tets")
    _validate_indices(surface, len(positions), name="table_surface_tris")
    edges_1 = positions[tetrahedra[:, 1]] - positions[tetrahedra[:, 0]]
    edges_2 = positions[tetrahedra[:, 2]] - positions[tetrahedra[:, 0]]
    edges_3 = positions[tetrahedra[:, 3]] - positions[tetrahedra[:, 0]]
    signed_six_volumes = np.einsum("ij,ij->i", np.cross(edges_1, edges_2), edges_3)
    if np.any(signed_six_volumes <= 0.0):
        _fail("table_tets must have strictly positive signed volumes.")

    boundary = _tet_boundary_owners(tetrahedra)
    surface_keys = [tuple(sorted(triangle.tolist())) for triangle in surface]
    if set(surface_keys) != set(boundary) or len(surface_keys) != len(boundary):
        _fail("table_surface_tris must be exactly the tetrahedral boundary, with no internal or missing faces.")
    for triangle, key in zip(surface, surface_keys, strict=True):
        _, opposite = boundary[key]
        first, second, third = (int(value) for value in triangle)
        normal = np.cross(positions[second] - positions[first], positions[third] - positions[first])
        if float(np.dot(normal, positions[opposite] - positions[first])) >= 0.0:
            _fail("table_surface_tris must be wound outward.")


def _number(value: object, *, name: str, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be a JSON number.")
    result = float(value)
    if not math.isfinite(result):
        _fail(f"{name} must be finite.")
    if positive and result <= 0.0:
        _fail(f"{name} must be positive.")
    if nonnegative and result < 0.0:
        _fail(f"{name} must be non-negative.")
    return result


def _exact_keys(value: dict[str, Any], expected: set[str], *, name: str) -> None:
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        _fail(f"{name} keys do not match the v2 contract; missing={missing}, unexpected={unexpected}.")


def _validate_contact_policy(policy: dict[str, Any], *, name: str) -> None:
    _exact_keys(policy, {"schema", "enable", "friction_rate", "resistance", "adhesion"}, name=name)
    if policy["schema"] != "qipc_contact_policy_v1":
        _fail(f"{name}.schema must be 'qipc_contact_policy_v1'.")
    if not isinstance(policy["enable"], bool):
        _fail(f"{name}.enable must be a boolean.")
    _number(policy["friction_rate"], name=f"{name}.friction_rate", nonnegative=True)
    _number(policy["resistance"], name=f"{name}.resistance", positive=True)
    if policy["adhesion"] is not None:
        _fail(f"{name}.adhesion must be null for explicit authored topology.")


def _validate_bond_policy(policy: dict[str, Any], *, name: str, calibratable: bool) -> None:
    _exact_keys(
        policy,
        {
            "schema",
            "kappa",
            "ratio",
            "margin",
            "floor_ratio",
            "rest_snap",
            "occlusion",
            "release",
            "release_force_calibratable",
        },
        name=name,
    )
    if policy["schema"] != "qipc_bond_policy_v1":
        _fail(f"{name}.schema must be 'qipc_bond_policy_v1'.")
    _number(policy["kappa"], name=f"{name}.kappa", positive=True)
    _number(policy["ratio"], name=f"{name}.ratio", positive=True)
    _number(policy["margin"], name=f"{name}.margin", nonnegative=True)
    _number(policy["floor_ratio"], name=f"{name}.floor_ratio", nonnegative=True)
    if policy["rest_snap"] is not False:
        _fail(f"{name}.rest_snap must be false for the authored current geometry.")
    if not isinstance(policy["occlusion"], bool):
        _fail(f"{name}.occlusion must be a boolean.")
    release = policy["release"]
    if not isinstance(release, dict):
        _fail(f"{name}.release must be an object.")
    _exact_keys(release, {"force", "strain", "gap", "slip"}, name=f"{name}.release")
    for field in ("force", "strain", "gap", "slip"):
        _number(release[field], name=f"{name}.release.{field}", positive=True)
    if policy["release_force_calibratable"] is not calibratable:
        _fail(f"{name}.release_force_calibratable must be {str(calibratable).lower()}.")


def _validate_body_roles(value: dict[str, Any]) -> None:
    if value != _EXPECTED_BODY_ROLES:
        _fail("body_roles_json does not match the qipc_component_body_roles_v1 contract.")


def _validate_bond_state_manifest(value: dict[str, Any]) -> None:
    _exact_keys(
        value,
        {
            "schema",
            "mode",
            "portable_exact_restore",
            "dm_inv_transform",
            "groups",
            "captured_fields",
            "omitted_restore_fields",
            "omission_reason",
        },
        name="bond_state_manifest_json",
    )
    expected_groups = {
        "internal": {"owner": "hub", "fem": "tape", "topology_space": _INTERNAL_TOPOLOGY_SPACE},
        "table": {"owner": "table_attachment_pad", "fem": "tape", "topology_space": _TABLE_TOPOLOGY_SPACE},
    }
    if value["schema"] != "qipc_bond_state_manifest_v1":
        _fail("bond_state_manifest_json.schema must be 'qipc_bond_state_manifest_v1'.")
    if value["mode"] != _BOND_STATE_MODE or value["portable_exact_restore"] is not True:
        _fail("bond_state_manifest_json must declare portable grouped_frozen_state_v2 state.")
    if value["dm_inv_transform"] != _DM_INV_TRANSFORM:
        _fail("bond_state_manifest_json.dm_inv_transform must be 'right_multiply_placement_rotation_transpose'.")
    if value["groups"] != expected_groups:
        _fail("bond_state_manifest_json.groups does not match the v2 topology namespaces.")
    if value["captured_fields"] != list(_STATE_FIELDS):
        _fail("bond_state_manifest_json.captured_fields does not match the frozen state arrays.")
    if value["omitted_restore_fields"] != ["slot_alive", "pair_key", "dead_slot_capacity"]:
        _fail("bond_state_manifest_json.omitted_restore_fields does not match the v2 contract.")
    if not isinstance(value["omission_reason"], str) or not value["omission_reason"].strip():
        _fail("bond_state_manifest_json.omission_reason must be a non-empty string.")


@dataclass(frozen=True)
class TapeTableBondFrozenState:
    """Portable live bond rows for exact restoration after topology remapping."""

    Dm_inv: np.ndarray
    V0: np.ndarray
    d_rest: np.ndarray
    kappa: np.ndarray
    release_force: np.ndarray
    release_strain: np.ndarray
    release_gap: np.ndarray
    release_slip: np.ndarray
    age: np.ndarray


@dataclass(frozen=True)
class TapeTableBondBatch:
    """One component-local ``[rigid | tape]`` frame-zero seed batch."""

    topologies: np.ndarray
    topology_space: str
    fem_offset: int
    seed_rest_height: float
    contact_policy: dict[str, Any]
    bond_policy: dict[str, Any]
    frozen_state: TapeTableBondFrozenState


@dataclass(frozen=True)
class TapeTableComponentPlacement:
    """Component geometry and frozen bond state after one world placement."""

    transform: np.ndarray
    tape_positions: np.ndarray
    tape_rest_positions: np.ndarray
    hub_positions: np.ndarray
    hub_transform: np.ndarray
    table_positions: np.ndarray
    table_transform: np.ndarray
    internal_frozen_state: TapeTableBondFrozenState
    table_frozen_state: TapeTableBondFrozenState


def _placed_frozen_state(
    state: TapeTableBondFrozenState,
    rotation: np.ndarray,
) -> TapeTableBondFrozenState:
    transformed_Dm_inv = (state.Dm_inv.reshape(-1, 3, 3) @ rotation.T).reshape(-1, 9)
    return TapeTableBondFrozenState(
        Dm_inv=_readonly(np.ascontiguousarray(transformed_Dm_inv)),
        V0=state.V0,
        d_rest=state.d_rest,
        kappa=state.kappa,
        release_force=state.release_force,
        release_strain=state.release_strain,
        release_gap=state.release_gap,
        release_slip=state.release_slip,
        age=state.age,
    )


@dataclass(frozen=True)
class TapeTableComponentAsset:
    """Strictly validated ``qipc_tape_table_component_v2`` data."""

    path: Path
    source_qipc_commit: str
    source_asset_name: str
    source_asset_sha256: str
    params: dict[str, Any]
    authoring_params: dict[str, Any]
    body_roles: dict[str, Any]
    bond_state_manifest: dict[str, Any]
    tape_positions: np.ndarray
    tape_rest_positions: np.ndarray
    tape_velocities: np.ndarray
    tape_tris: np.ndarray
    tape_is_fixed: np.ndarray
    hub_positions: np.ndarray
    hub_tris: np.ndarray
    hub_transform: np.ndarray
    hub_affine_velocity: np.ndarray
    table_positions: np.ndarray
    table_tets: np.ndarray
    table_surface_tris: np.ndarray
    table_transform: np.ndarray
    table_affine_velocity: np.ndarray
    internal_bonds: TapeTableBondBatch
    table_bonds: TapeTableBondBatch
    tail_vertex_ids: np.ndarray
    tail_rows: np.ndarray
    terminal_row_ids: np.ndarray
    attached_length: float
    nx: int
    nz: int
    tape_length: float
    width: float
    hub_r_outer: float
    hub_r_inner: float
    hub_height: float
    thick: float
    d_hat: float
    youngs: float
    poisson: float
    density: float
    bending_e: float

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> TapeTableComponentAsset:
        """Load an untrusted component without enabling NumPy pickle."""
        asset_path = Path(path).expanduser()
        try:
            archive_context = np.load(asset_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            _fail(f"cannot open '{asset_path}': {error}.")
        with archive_context as archive:
            missing = sorted(_REQUIRED_FIELDS - set(archive.files))
            if missing:
                _fail(f"missing v2 fields in '{asset_path}': {', '.join(missing)}.")
            if "params" in archive.files:
                _fail("legacy object field 'params' is forbidden; use params_json UTF-8 bytes.")

            if _bytes_scalar(archive, "component_schema") != _COMPONENT_SCHEMA:
                _fail(f"component_schema must be '{_COMPONENT_SCHEMA}'.")
            if _int_scalar(archive, "component_schema_version") != _COMPONENT_SCHEMA_VERSION:
                _fail(f"component_schema_version must be {_COMPONENT_SCHEMA_VERSION}.")
            if _bytes_scalar(archive, "component_schema_abi") != _COMPONENT_SCHEMA_ABI:
                _fail(f"component_schema_abi must be '{_COMPONENT_SCHEMA_ABI}'.")
            for name, expected in (
                ("units", "m"),
                ("up_axis", "+Y"),
                ("handedness", "right"),
                ("component_frame", "authoring_table_top_world"),
            ):
                if _bytes_scalar(archive, name) != expected:
                    _fail(f"{name} must be {expected!r}.")

            component_origin = _float_array(archive, "component_origin", (3,))
            component_transform = _float_array(archive, "component_transform", (4, 4))
            if not np.array_equal(component_origin, np.zeros(3)):
                _fail("component_origin must be [0, 0, 0].")
            if not np.array_equal(component_transform, np.eye(4)):
                _fail("component_transform must be identity in the authoring table-top world frame.")

            source_qipc_commit = _bytes_scalar(archive, "source_qipc_commit")
            source_asset_sha256 = _bytes_scalar(archive, "source_asset_sha256")
            if _HEX_40.fullmatch(source_qipc_commit) is None:
                _fail("source_qipc_commit must be a lowercase 40-character hexadecimal commit.")
            if _HEX_64.fullmatch(source_asset_sha256) is None:
                _fail("source_asset_sha256 must be a lowercase 64-character hexadecimal digest.")
            source_asset_name = _utf8_bytes(archive, "source_asset_name")
            if (
                not source_asset_name.strip()
                or "\x00" in source_asset_name
                or "/" in source_asset_name
                or "\\" in source_asset_name
            ):
                _fail("source_asset_name must be a non-empty portable UTF-8 basename.")

            json_values = {name: _json_object(archive, name) for name in _JSON_FIELDS}
            _validate_body_roles(json_values["body_roles_json"])
            _validate_bond_state_manifest(json_values["bond_state_manifest_json"])
            _validate_contact_policy(json_values["internal_contact_policy_json"], name="internal_contact_policy_json")
            _validate_contact_policy(json_values["table_contact_policy_json"], name="table_contact_policy_json")
            _validate_bond_policy(
                json_values["internal_bond_policy_json"],
                name="internal_bond_policy_json",
                calibratable=False,
            )
            _validate_bond_policy(
                json_values["table_bond_policy_json"], name="table_bond_policy_json", calibratable=True
            )

            nx = _int_scalar(archive, "nx")
            nz = _int_scalar(archive, "nz")
            if nx <= 0 or nz <= 0:
                _fail("nx and nz must be positive.")
            tape_vertex_count = (nx + 1) * (nz + 1)
            tape_positions = _float_array(archive, "tape_positions", (tape_vertex_count, 3))
            tape_rest_positions = _float_array(archive, "tape_rest_positions", (tape_vertex_count, 3))
            tape_velocities = _float_array(archive, "tape_velocities", (tape_vertex_count, 3))
            tape_is_fixed = _int_array(archive, "tape_is_fixed", (tape_vertex_count,))
            if np.any(tape_velocities != 0.0) or np.any(tape_is_fixed != 0):
                _fail("tape must start with zero velocity and no fixed vertices.")
            tape_tris = _index_array(archive, "tape_tris", 3)
            if len(tape_tris) != 2 * nx * nz:
                _fail("tape_tris count must equal 2*nx*nz.")
            _validate_indices(tape_tris, tape_vertex_count, name="tape_tris")
            _validate_triangle_geometry(tape_rest_positions, tape_tris, name="rest tape mesh")
            _validate_triangle_geometry(tape_positions, tape_tris, name="settled tape mesh")

            hub_positions = _point_array(archive, "hub_positions")
            hub_tris = _index_array(archive, "hub_tris", 3)
            _validate_indices(hub_tris, len(hub_positions), name="hub_tris")
            _validate_triangle_geometry(hub_positions, hub_tris, name="hub mesh")
            _validate_closed_surface(hub_tris, name="hub_tris")
            hub_transform = _float_array(archive, "hub_transform", (4, 4))
            _validate_proper_affine_transform(hub_transform, name="hub_transform")
            hub_affine_velocity = _float_array(archive, "hub_affine_velocity", (12,))
            if np.any(hub_affine_velocity != 0.0) or _int_scalar(archive, "hub_is_fixed") != 0:
                _fail("hub must be dynamic with zero affine velocity.")

            table_positions = _point_array(archive, "table_positions")
            table_tets = _index_array(archive, "table_tets", 4)
            table_surface_tris = _index_array(archive, "table_surface_tris", 3)
            _validate_table_mesh(table_positions, table_tets, table_surface_tris)
            table_transform = _float_array(archive, "table_transform", (4, 4))
            _validate_rigid_transform(table_transform, name="table_transform")
            if not np.array_equal(table_transform, np.eye(4)):
                _fail("table_transform must be identity in the authoring frame.")
            table_affine_velocity = _float_array(archive, "table_affine_velocity", (12,))
            if np.any(table_affine_velocity != 0.0) or _int_scalar(archive, "table_is_fixed") != 1:
                _fail("table attachment pad must be fixed with zero affine velocity.")

            internal_bonds = _load_bond_batch(
                archive,
                group="internal",
                expected_space=_INTERNAL_TOPOLOGY_SPACE,
                expected_offset=len(hub_positions),
                rigid_vertex_count=len(hub_positions),
                tape_vertex_count=tape_vertex_count,
                contact_policy=json_values["internal_contact_policy_json"],
                bond_policy=json_values["internal_bond_policy_json"],
            )
            _validate_internal_bond_faces(
                internal_bonds.topologies,
                hub_tris,
                tape_tris,
                hub_vertex_count=len(hub_positions),
            )
            _validate_internal_release_force(internal_bonds)
            table_bonds = _load_bond_batch(
                archive,
                group="table",
                expected_space=_TABLE_TOPOLOGY_SPACE,
                expected_offset=len(table_positions),
                rigid_vertex_count=len(table_positions),
                tape_vertex_count=tape_vertex_count,
                contact_policy=json_values["table_contact_policy_json"],
                bond_policy=json_values["table_bond_policy_json"],
            )
            _validate_table_bond_faces(
                table_bonds.topologies,
                table_surface_tris,
                table_vertex_count=len(table_positions),
            )
            if _bytes_scalar(archive, "bond_state_mode") != _BOND_STATE_MODE:
                _fail(f"bond_state_mode must be '{_BOND_STATE_MODE}'.")
            if _int_scalar(archive, "bond_frozen_state_restore_supported") != 1:
                _fail("bond_frozen_state_restore_supported must be 1 for portable exact restore.")

            tail_vertex_ids = _id_array(archive, "tail_vertex_ids")
            tail_rows = _id_array(archive, "tail_rows")
            terminal_row_ids = _id_array(archive, "terminal_row_ids")
            attached_length = _float_scalar(archive, "attached_length")
            tape_length = _float_scalar(archive, "tape_length")
            width = _float_scalar(archive, "width")
            hub_r_outer = _float_scalar(archive, "hub_r_outer")
            hub_r_inner = _float_scalar(archive, "hub_r_inner")
            hub_height = _float_scalar(archive, "hub_height")
            thick = _float_scalar(archive, "thick")
            d_hat = _float_scalar(archive, "d_hat")
            youngs = _float_scalar(archive, "youngs")
            poisson = _float_scalar(archive, "poisson")
            density = _float_scalar(archive, "density")
            bending_e = _float_scalar(archive, "bending_e")
            _validate_physical_parameters(
                attached_length=attached_length,
                tape_length=tape_length,
                width=width,
                hub_r_outer=hub_r_outer,
                hub_r_inner=hub_r_inner,
                hub_height=hub_height,
                thick=thick,
                d_hat=d_hat,
                youngs=youngs,
                poisson=poisson,
                density=density,
                bending_e=bending_e,
            )
            _validate_tail_contract(
                nx=nx,
                nz=nz,
                tape_length=tape_length,
                attached_length=attached_length,
                tail_vertex_ids=tail_vertex_ids,
                tail_rows=tail_rows,
                terminal_row_ids=terminal_row_ids,
                table_bond_topologies=table_bonds.topologies,
                table_vertex_count=len(table_positions),
            )

            params = json_values["params_json"]
            source_release_force = _number(
                params.get("RCC_RELEASE_FORCE"), name="params_json.RCC_RELEASE_FORCE", positive=True
            )
            table_release = table_bonds.bond_policy["release"]
            assert isinstance(table_release, dict)
            if float(table_release["force"]) != source_release_force:
                _fail(
                    "table_bond_policy_json.release.force must use params_json.RCC_RELEASE_FORCE, not authoring force."
                )

            return cls(
                path=asset_path,
                source_qipc_commit=source_qipc_commit,
                source_asset_name=source_asset_name,
                source_asset_sha256=source_asset_sha256,
                params=params,
                authoring_params=json_values["authoring_params_json"],
                body_roles=json_values["body_roles_json"],
                bond_state_manifest=json_values["bond_state_manifest_json"],
                tape_positions=tape_positions,
                tape_rest_positions=tape_rest_positions,
                tape_velocities=tape_velocities,
                tape_tris=tape_tris,
                tape_is_fixed=tape_is_fixed,
                hub_positions=hub_positions,
                hub_tris=hub_tris,
                hub_transform=hub_transform,
                hub_affine_velocity=hub_affine_velocity,
                table_positions=table_positions,
                table_tets=table_tets,
                table_surface_tris=table_surface_tris,
                table_transform=table_transform,
                table_affine_velocity=table_affine_velocity,
                internal_bonds=internal_bonds,
                table_bonds=table_bonds,
                tail_vertex_ids=tail_vertex_ids,
                tail_rows=tail_rows,
                terminal_row_ids=terminal_row_ids,
                attached_length=attached_length,
                nx=nx,
                nz=nz,
                tape_length=tape_length,
                width=width,
                hub_r_outer=hub_r_outer,
                hub_r_inner=hub_r_inner,
                hub_height=hub_height,
                thick=thick,
                d_hat=d_hat,
                youngs=youngs,
                poisson=poisson,
                density=density,
                bending_e=bending_e,
            )

    def placed(self, transform: np.ndarray) -> TapeTableComponentPlacement:
        """Apply one explicit rigid transform without changing any vertex ID."""
        transform = np.array(transform, dtype=np.float64, order="C", copy=True)
        _validate_rigid_transform(transform, name="placement transform")
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        def points(values: np.ndarray) -> np.ndarray:
            return _readonly(np.ascontiguousarray(values @ rotation.T + translation))

        return TapeTableComponentPlacement(
            transform=_readonly(transform),
            tape_positions=points(self.tape_positions),
            tape_rest_positions=points(self.tape_rest_positions),
            hub_positions=points(self.hub_positions),
            hub_transform=_readonly(np.ascontiguousarray(transform @ self.hub_transform)),
            table_positions=points(self.table_positions),
            table_transform=_readonly(np.ascontiguousarray(transform @ self.table_transform)),
            internal_frozen_state=_placed_frozen_state(self.internal_bonds.frozen_state, rotation),
            table_frozen_state=_placed_frozen_state(self.table_bonds.frozen_state, rotation),
        )


def _load_bond_batch(
    archive: np.lib.npyio.NpzFile,
    *,
    group: str,
    expected_space: str,
    expected_offset: int,
    rigid_vertex_count: int,
    tape_vertex_count: int,
    contact_policy: dict[str, Any],
    bond_policy: dict[str, Any],
) -> TapeTableBondBatch:
    topologies = _index_array(archive, f"{group}_bond_topologies", 4)
    topology_space = _bytes_scalar(archive, f"{group}_bond_topology_space")
    fem_offset = _int_scalar(archive, f"{group}_bond_fem_offset")
    seed_rest_height = _float_scalar(archive, f"{group}_bond_seed_rest_height")
    if topology_space != expected_space:
        _fail(f"{group}_bond_topology_space must be '{expected_space}'.")
    if fem_offset != expected_offset or fem_offset != rigid_vertex_count:
        _fail(f"{group}_bond_fem_offset must equal the rigid vertex count {rigid_vertex_count}.")
    if seed_rest_height != 0.0:
        _fail(f"{group}_bond_seed_rest_height must be 0 to preserve the authored rest state.")
    _validate_pt_indices(topologies, rigid_vertex_count + tape_vertex_count, name=f"{group}_bond_topologies")
    tape_mask = topologies >= fem_offset
    if np.any(tape_mask.sum(axis=1) == 0):
        _fail(f"every {group} bond must reference at least one tape vertex.")
    triangle_tape_count = tape_mask[:, 1:].sum(axis=1)
    if group == "internal" and np.any((triangle_tape_count != 0) & (triangle_tape_count != 3)):
        _fail("every internal bond triangle must belong entirely to the hub or entirely to the tape.")
    if group == "table" and (
        np.any(~tape_mask[:, 0]) or np.any(tape_mask[:, 1:]) or np.any(tape_mask.sum(axis=1) != 1)
    ):
        _fail("every table bond must preserve [tape point, table triangle] PT role ordering.")

    count = len(topologies)
    state_values: dict[str, np.ndarray] = {}
    for field in _STATE_FIELDS:
        name = f"{group}_bond_state_{field}"
        if field == "Dm_inv":
            matrices = _float_array(archive, name, (count, 3, 3))
            if np.any(np.linalg.det(matrices) == 0.0):
                _fail(f"{group}_bond_state_Dm_inv must be invertible.")
            state_values[field] = _readonly(np.ascontiguousarray(matrices.reshape(count, 9)))
        elif field == "age":
            state_values[field] = _int_array(archive, name, (count,))
        else:
            state_values[field] = _float_array(archive, name, (count,))
    if np.any(state_values["V0"] <= 0.0):
        _fail(f"{group}_bond_state_V0 must be positive.")
    expected_V0 = np.abs(1.0 / np.linalg.det(state_values["Dm_inv"].reshape(-1, 3, 3))) / 6.0
    if not np.allclose(state_values["V0"], expected_V0, rtol=1.0e-10, atol=0.0):
        _fail(f"{group}_bond_state_V0 must equal abs(1 / det(Dm_inv)) / 6.")
    if np.any(state_values["kappa"] <= 0.0):
        _fail(f"{group}_bond_state_kappa must be positive.")
    for field in ("release_force", "release_strain", "release_gap", "release_slip"):
        if np.any(state_values[field] <= 0.0):
            _fail(f"{group}_bond_state_{field} must be positive.")
    if np.any(state_values["age"] < 0):
        _fail(f"{group}_bond_state_age must be non-negative.")
    frozen_state = TapeTableBondFrozenState(**state_values)
    return TapeTableBondBatch(
        topologies=topologies,
        topology_space=topology_space,
        fem_offset=fem_offset,
        seed_rest_height=seed_rest_height,
        contact_policy=contact_policy,
        bond_policy=bond_policy,
        frozen_state=frozen_state,
    )


def _validate_internal_release_force(batch: TapeTableBondBatch) -> None:
    release = batch.bond_policy["release"]
    assert isinstance(release, dict)
    policy_force = float(release["force"])
    active_forces = np.unique(batch.frozen_state.release_force)
    if active_forces.shape != (1,) or float(active_forces[0]) != policy_force:
        _fail(
            "internal_bond_policy_json.release.force must equal the unique value across all "
            "active internal_bond_state_release_force rows."
        )


def _validate_internal_bond_faces(
    topologies: np.ndarray,
    hub_triangles: np.ndarray,
    tape_triangles: np.ndarray,
    *,
    hub_vertex_count: int,
) -> None:
    hub_faces = {tuple(sorted(triangle.tolist())) for triangle in hub_triangles}
    tape_faces = {tuple(sorted(triangle.tolist())) for triangle in tape_triangles}
    for topology in topologies:
        triangle = topology[1:]
        if np.all(triangle < hub_vertex_count):
            if tuple(sorted(triangle.tolist())) not in hub_faces:
                _fail("internal bond hub PT triangles must reference hub_tris surface faces.")
        else:
            local_triangle = triangle - hub_vertex_count
            if tuple(sorted(local_triangle.tolist())) not in tape_faces:
                _fail("internal bond tape PT triangles must reference tape_tris surface faces.")


def _validate_table_bond_faces(
    topologies: np.ndarray,
    surface_triangles: np.ndarray,
    *,
    table_vertex_count: int,
) -> None:
    surface_faces = {tuple(sorted(triangle.tolist())) for triangle in surface_triangles}
    for topology in topologies:
        face = tuple(sorted(topology[1:].tolist()))
        if face not in surface_faces:
            _fail("table bond PT triangles must reference table_surface_tris boundary faces.")
        if int(topology[0]) < table_vertex_count:
            _fail("table bond PT point must be a tape vertex in [table | tape] space.")


def _validate_physical_parameters(
    *,
    attached_length: float,
    tape_length: float,
    width: float,
    hub_r_outer: float,
    hub_r_inner: float,
    hub_height: float,
    thick: float,
    d_hat: float,
    youngs: float,
    poisson: float,
    density: float,
    bending_e: float,
) -> None:
    positive = {
        "attached_length": attached_length,
        "tape_length": tape_length,
        "width": width,
        "hub_r_outer": hub_r_outer,
        "hub_r_inner": hub_r_inner,
        "hub_height": hub_height,
        "thick": thick,
        "d_hat": d_hat,
        "youngs": youngs,
        "density": density,
        "bending_e": bending_e,
    }
    for name, value in positive.items():
        if value <= 0.0:
            _fail(f"{name} must be positive.")
    if attached_length > tape_length:
        _fail("attached_length must not exceed tape_length.")
    if hub_r_inner >= hub_r_outer:
        _fail("hub_r_inner must be smaller than hub_r_outer.")
    if not -1.0 < poisson < 0.5:
        _fail("poisson must be in (-1, 0.5).")


def _validate_tail_contract(
    *,
    nx: int,
    nz: int,
    tape_length: float,
    attached_length: float,
    tail_vertex_ids: np.ndarray,
    tail_rows: np.ndarray,
    terminal_row_ids: np.ndarray,
    table_bond_topologies: np.ndarray,
    table_vertex_count: int,
) -> None:
    attached_cells = min(nx, math.ceil(attached_length / (tape_length / nx)))
    expected_rows = np.arange(nx - attached_cells, nx + 1, dtype=np.int32)
    expected_tail = (expected_rows[:, None] * (nz + 1) + np.arange(nz + 1, dtype=np.int32)[None, :]).reshape(-1)
    expected_terminal = np.arange(nx * (nz + 1), (nx + 1) * (nz + 1), dtype=np.int32)
    if not np.array_equal(tail_rows, expected_rows):
        _fail("tail_rows must be the contiguous quantized attachment rows ending at nx.")
    if not np.array_equal(tail_vertex_ids, expected_tail):
        _fail("tail_vertex_ids must contain every vertex of tail_rows in structured tape order.")
    if not np.array_equal(terminal_row_ids, expected_terminal):
        _fail("terminal_row_ids must be the complete final structured tape row.")

    tape_global_ids = table_bond_topologies[table_bond_topologies >= table_vertex_count]
    tape_local_ids = tape_global_ids - table_vertex_count
    unique, counts = np.unique(tape_local_ids, return_counts=True)
    if not np.array_equal(unique, tail_vertex_ids) or np.any(counts != 1):
        _fail("table bonds must contain exactly one PT bond for every declared tail vertex.")


def qipc_y_up_to_genesis_z_up_transform() -> np.ndarray:
    """Return the explicit right-handed Y-up to Genesis Z-up rotation."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
