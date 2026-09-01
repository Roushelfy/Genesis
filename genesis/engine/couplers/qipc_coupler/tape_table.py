"""Compose an authored tape-roll/table attachment component in Genesis."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.engine.entities import FEMEntity, RigidEntity
from genesis.utils.misc import get_assets_dir, tensor_to_array

from .adhesion import BondStateHandle
from .contact import QIPCContactRegion
from .rigid_attachment import QIPCRigidAttachment
from .tape import _verify_same_vertex_order, _write_obj, solver_cfg_to_options
from .tape_table_component import (
    TapeTableBondBatch,
    TapeTableBondFrozenState,
    TapeTableComponentAsset,
    qipc_y_up_to_genesis_z_up_transform,
)

_ASSET_DIRECTORY = Path(get_assets_dir()) / "qipc" / "tape_table_component_v2"
_ASSET_MANIFEST = _ASSET_DIRECTORY / "manifest.json"
_PACKAGED_LENGTHS = frozenset({3, 4, 5, 6})
_PACKAGED_WINDINGS = frozenset({"locked", "releasable"})
_QIPC_DEFAULT_LINEAR_TOL_RATE = 1.0e-4


@dataclass(frozen=True)
class TapeTableComponent:
    """Genesis entities and stable QIPC handles created by the builder."""

    tape: FEMEntity
    hub: RigidEntity
    table_pad: QIPCRigidAttachment
    tail_region: QIPCContactRegion
    table_pad_region: QIPCContactRegion
    internal_bonds: BondStateHandle
    table_bonds: BondStateHandle | None
    """`None` when the tape/table interface is cohesive: there is no per-bond state."""
    asset: TapeTableComponentAsset


def packaged_asset(attached_inches: int = 3, winding: str = "locked") -> TapeTableComponentAsset:
    """Load one of the packaged 3, 4, 5, or 6 inch component assets.

    `winding` selects the batch: `locked` pins the wound core permanently while
    `releasable` carries the product unwind release force for the loader.
    """
    if isinstance(attached_inches, bool) or attached_inches not in _PACKAGED_LENGTHS:
        gs.raise_exception("packaged tape-table attached_inches must be one of 3, 4, 5, or 6.")
    if winding not in _PACKAGED_WINDINGS:
        gs.raise_exception("packaged tape-table winding must be 'locked' or 'releasable'.")
    suffix = "" if winding == "locked" else "_unwind"
    filename = f"scotch3850_table_{attached_inches}in{suffix}_component.npz"
    asset_path = _ASSET_DIRECTORY / filename
    expected_sha256 = _packaged_asset_sha256(filename)
    try:
        asset_bytes = asset_path.read_bytes()
    except OSError as error:
        gs.raise_exception(f"Cannot read packaged tape-table asset '{asset_path}': {error}.")
    actual_sha256 = hashlib.sha256(asset_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        gs.raise_exception(
            f"Packaged tape-table asset '{filename}' does not match manifest SHA-256 "
            f"(expected {expected_sha256}, got {actual_sha256})."
        )
    return TapeTableComponentAsset.from_npz(asset_path)


def _packaged_asset_sha256(filename: str) -> str:
    try:
        manifest = json.loads(_ASSET_MANIFEST.read_text(encoding="utf-8"))
        expected = manifest["assets"][filename]["sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        gs.raise_exception(f"Invalid packaged tape-table manifest '{_ASSET_MANIFEST}': {error}.")
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "genesis.qipc.tape_table_component"
        or manifest.get("version") != 2
        or manifest.get("schema_abi") != "qipc_tape_table_component_v2"
    ):
        gs.raise_exception("Packaged tape-table manifest identity does not match component v2.")
    if not isinstance(expected, str) or len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        gs.raise_exception(f"Packaged tape-table manifest SHA-256 for '{filename}' is invalid.")
    return expected


def placement_transform(table_top_origin) -> np.ndarray:
    """Place the asset's Y-up table-top origin at one Genesis Z-up point."""
    origin = np.asarray(table_top_origin, dtype=np.float64).reshape(-1)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        gs.raise_exception("tape-table placement origin must contain three finite values.")
    transform = qipc_y_up_to_genesis_z_up_transform()
    transform[:3, 3] = origin
    return transform


def placement_transform_for_hub(
    asset: TapeTableComponentAsset,
    *,
    hub_xy,
    table_top: float,
    yaw_degrees: float = 0.0,
) -> np.ndarray:
    """Place the hub center at `hub_xy` while keeping the table at `table_top`.

    `yaw_degrees` spins the component about the vertical axis through the hub
    center; at zero the authored free tail points toward `+x`.
    """
    target_xy = np.asarray(hub_xy, dtype=np.float64).reshape(-1)
    if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
        gs.raise_exception("tape-table hub_xy must contain two finite values.")
    table_top = float(table_top)
    if not math.isfinite(table_top):
        gs.raise_exception("tape-table table_top must be finite.")
    yaw = math.radians(float(yaw_degrees))
    if not math.isfinite(yaw):
        gs.raise_exception("tape-table yaw_degrees must be finite.")
    transform = qipc_y_up_to_genesis_z_up_transform()
    yaw_rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform[:3, :3] = yaw_rotation @ transform[:3, :3]
    rotated_hub_center = transform[:3, :3] @ asset.hub_positions.mean(axis=0)
    transform[:2, 3] = target_xy - rotated_hub_center[:2]
    transform[2, 3] = table_top
    return transform


def recommended_coupler_options(asset: TapeTableComponentAsset) -> dict[str, object]:
    """Return QIPC options matching the authored component state and policies."""
    internal = asset.internal_bonds.bond_policy
    table = asset.table_bonds.bond_policy
    internal_release = _release_policy(asset.internal_bonds)
    table_release = _release_policy(asset.table_bonds)
    for name in ("kappa", "margin", "occlusion"):
        if internal[name] != table[name]:
            gs.raise_exception(f"Tape-table component requires one shared {name} policy across bond batches.")
    for name in ("strain", "gap", "slip"):
        if internal_release[name] != table_release[name]:
            gs.raise_exception(f"Tape-table component requires one shared release {name} across bond batches.")
    contact = asset.internal_bonds.contact_policy
    solver_config = asset.params.get("SOLVER_CFG")
    if not isinstance(solver_config, dict):
        gs.raise_exception("Tape-table component params.SOLVER_CFG must be an object.")
    solver_options = solver_cfg_to_options(solver_config)
    # The authoring script inherits QIPC's default when SOLVER_CFG omits this
    # field. Spell it out so a component stays reproducible if that default
    # changes and does not inherit the looser generic tape-roll preset.
    solver_options.setdefault("solver_linear_tol_rate", _QIPC_DEFAULT_LINEAR_TOL_RATE)
    options: dict[str, object] = {
        "contact_enable": True,
        "contact_d_hat": asset.d_hat,
        "init_collision_pair_capacity": 100_000,
        "contact_friction": float(contact["friction_rate"]),
        "contact_resistance": float(contact["resistance"]),
        "contact_constitution": "adhesive",
        "adhesion_bond_distance_lock": True,
        "adhesion_bond_distance_lock_ratio": float(internal["ratio"]),
        "adhesion_bond_max_bonds": max(
            16_384, len(asset.internal_bonds.topologies) + len(asset.table_bonds.topologies)
        ),
        "adhesion_bond_default": False,
        "adhesion_bond_kappa": float(internal["kappa"]),
        "adhesion_bond_lock_margin": float(internal["margin"]),
        "adhesion_bond_lock_floor_ratio": float(internal["floor_ratio"]),
        "adhesion_occlusion": bool(internal["occlusion"]),
        "adhesion_bond_release_force": float(internal_release["force"]),
        "adhesion_bond_release_strain": float(internal_release["strain"]),
        "adhesion_bond_release_gap": float(internal_release["gap"]),
        "adhesion_bond_release_slip": float(internal_release["slip"]),
        **solver_options,
    }
    return options


def add_tape_table_component(
    scene,
    table: RigidEntity,
    asset: TapeTableComponentAsset,
    *,
    transform: np.ndarray,
    table_bond_release_force: float | None = None,
    tape_surface=None,
    hub_surface=None,
) -> TapeTableComponent:
    """Add a settled roll whose authored tail is bonded to a fixed table.

    The caller supplies an explicit rigid transform from the asset's Y-up,
    table-top frame to the Genesis world. `placement_transform_for_hub` covers
    the common hub-centered Z-up placement. The subdivided authoring table is
    appended as massless collision geometry on `table` so its exact vertex IDs
    can own the table bond batch without introducing a duplicate rigid body.
    It is also the sole table collision surface for the tape and hub, so their
    motion must remain over the authored pad's working area. Other scene bodies
    should collide with the coarse main table and opt out of the pad. The
    builder reserves one set of component contact-region names and therefore
    supports one tape-table component per scene.
    """
    coupler = scene.sim.coupler
    required = (
        "add_adhesion",
        "add_rigid_attachment",
        "add_contact_region",
        "assign_contact_region",
        "assign_fem_contact_region",
        "set_contact_pair",
        "set_fem_rest_positions",
        "set_rigid_initial_state",
    )
    missing = [name for name in required if not hasattr(coupler, name)]
    if not hasattr(coupler, "adhesion") or not hasattr(coupler.adhesion, "add_bond_state_request"):
        missing.append("adhesion.add_bond_state_request")
    if missing:
        gs.raise_exception(
            "add_tape_table_component requires the QIPC component APIs (missing: " + ", ".join(missing) + ")."
        )
    _validate_coupler_options(coupler, asset)
    if len(table.links) != 1 or not table.links[0].is_fixed:
        gs.raise_exception("add_tape_table_component requires a fixed, single-link main table entity.")

    internal_kwargs = _bond_row_kwargs(asset.internal_bonds)
    release_force = (
        float(_release_policy(asset.table_bonds)["force"])
        if table_bond_release_force is None
        else float(table_bond_release_force)
    )
    if not math.isfinite(release_force) or release_force < 0.0:
        gs.raise_exception("table_bond_release_force must be finite and non-negative.")
    cohesive_table = asset.table_adhesion is not None
    table_kwargs = (
        _adhesion_row_kwargs(asset.table_adhesion, asset.table_bonds)
        if cohesive_table
        else _bond_row_kwargs(asset.table_bonds, release_force=release_force)
    )

    placed = asset.placed(transform)
    friction = float(asset.internal_bonds.contact_policy["friction_rate"])
    resistance = float(asset.internal_bonds.contact_policy["resistance"])
    pad_friction = float(asset.table_bonds.contact_policy["friction_rate"])
    pad_resistance = float(asset.table_bonds.contact_policy["resistance"])
    temporary_directory = tempfile.mkdtemp(prefix="qipc_tape_table_")

    tape_path = os.path.join(temporary_directory, "tape.obj")
    _write_obj(tape_path, placed.tape_positions, asset.tape_tris)
    tape = scene.add_entity(
        morph=gs.morphs.Mesh(file=tape_path, scale=1.0),
        material=gs.materials.FEM.Cloth(
            E=asset.youngs,
            nu=asset.poisson,
            rho=asset.density,
            thickness=asset.thick,
            bending_stiffness=asset.bending_e,
            membrane="stvk",
            bending_model="hinge",
            strain_limit_multiplier=0.0,
            friction_mu=friction,
            contact_resistance=resistance,
        ),
        surface=tape_surface,
    )
    _verify_same_vertex_order(
        placed.tape_positions,
        tensor_to_array(tape.init_positions).astype(np.float64),
    )
    coupler.set_fem_rest_positions(tape, placed.tape_rest_positions)

    hub_local = _inverse_affine_points(asset.hub_positions, asset.hub_transform)
    hub_path = os.path.join(temporary_directory, "hub.obj")
    _write_obj(hub_path, hub_local, asset.hub_tris)
    hub = scene.add_entity(
        morph=gs.morphs.Mesh(
            file=hub_path,
            scale=1.0,
            fixed=False,
            align=False,
            decimate=False,
            convexify=False,
            watertighten=None,
            file_meshes_are_zup=True,
        ),
        material=gs.materials.Rigid(
            rho=1000.0,
            coup_friction=friction,
            contact_resistance=resistance,
            qipc_abd_kappa=1.0e8,
            qipc_default_kp=0.0,
            qipc_default_kv=0.0,
            qipc_d_hat=asset.d_hat,
            qipc_self_contact=False,
        ),
        surface=hub_surface,
    )
    hub_collision_vertices = np.concatenate([geom.init_verts for geom in hub.links[0].geoms], axis=0)
    _verify_same_vertex_order(hub_local, hub_collision_vertices)
    hub_q = np.concatenate((placed.hub_transform[:3, 3], placed.hub_transform[:3, :3].reshape(-1)))
    coupler.set_rigid_initial_state(
        hub,
        body_q={hub.links[0].name: hub_q},
        joint_theta={},
    )

    table_transform = _root_transform(table)
    table_local = _inverse_affine_points(placed.table_positions, table_transform)
    table_pad = coupler.add_rigid_attachment(
        table,
        link=table.links[0],
        name="tape_table_subdivided_pad",
        vertices=table_local,
        triangles=asset.table_surface_tris,
    )

    tail_region = coupler.add_contact_region(
        "tape_table_tail",
        friction=float(asset.table_bonds.contact_policy["friction_rate"]),
        resistance=float(asset.table_bonds.contact_policy["resistance"]),
    )
    table_pad_region = coupler.add_contact_region(
        "tape_table_subdivided_pad",
        friction=float(asset.table_bonds.contact_policy["friction_rate"]),
        resistance=float(asset.table_bonds.contact_policy["resistance"]),
    )
    coupler.assign_fem_contact_region(tail_region, tape, verts_idx_local=asset.tail_vertex_ids)
    coupler.assign_contact_region(table_pad_region, table, attachment=table_pad)

    # The fine pad is the tape component's one continuous table surface. Routing
    # the tail to the pad and the body to the coarse table would disable mixed
    # primitives at their shared vertex seam because QIPC masks them with AND.
    coupler.set_contact_pair(tape, table, enabled=False, friction=friction, resistance=resistance)
    coupler.set_contact_pair(tail_region, table, enabled=False, friction=friction, resistance=resistance)
    coupler.set_contact_pair(hub, table, enabled=False, friction=friction, resistance=resistance)
    coupler.set_contact_pair(
        tape,
        table_pad_region,
        enabled=True,
        friction=pad_friction,
        resistance=pad_resistance,
    )
    coupler.set_contact_pair(
        hub,
        table_pad_region,
        enabled=True,
        friction=pad_friction,
        resistance=pad_resistance,
    )
    coupler.set_contact_pair(table, table_pad_region, enabled=False, friction=friction, resistance=resistance)
    coupler.set_contact_pair(
        table_pad_region,
        table_pad_region,
        enabled=False,
        friction=friction,
        resistance=resistance,
    )

    for first, second in (
        (tape, tape),
        (tape, tail_region),
        (tail_region, tail_region),
        (tape, hub),
        (tail_region, hub),
    ):
        coupler.add_adhesion(first, second, **internal_kwargs)

    coupler.add_adhesion(
        tail_region,
        table_pad_region,
        **table_kwargs,
    )

    # A calibratable batch stores the authoring sentinel in its frozen rows and
    # publishes the runtime force through its policy, so restamp it on restore.
    internal_release_force = (
        float(_release_policy(asset.internal_bonds)["force"])
        if asset.internal_bonds.bond_policy["release_force_calibratable"] is True
        else None
    )
    internal_bonds = _add_frozen_state_request(
        coupler,
        tape,
        rigid_source=hub,
        batch=asset.internal_bonds,
        state=placed.internal_frozen_state,
        name="internal",
        release_force=internal_release_force,
    )
    # A cohesive interface has no frozen per-bond state to restore.
    table_bonds = (
        None
        if cohesive_table
        else _add_frozen_state_request(
            coupler,
            tape,
            rigid_source=table_pad,
            batch=asset.table_bonds,
            state=placed.table_frozen_state,
            name="table",
            release_force=release_force,
        )
    )

    return TapeTableComponent(
        tape=tape,
        hub=hub,
        table_pad=table_pad,
        tail_region=tail_region,
        table_pad_region=table_pad_region,
        internal_bonds=internal_bonds,
        table_bonds=table_bonds,
        asset=asset,
    )


def _release_policy(batch: TapeTableBondBatch) -> dict[str, object]:
    release = batch.bond_policy["release"]
    assert isinstance(release, dict)
    return release


def _add_frozen_state_request(
    coupler,
    tape: FEMEntity,
    *,
    rigid_source,
    batch: TapeTableBondBatch,
    state: TapeTableBondFrozenState,
    name: str,
    release_force: float | None = None,
) -> BondStateHandle:
    restored_release_force = (
        state.release_force
        if release_force is None
        else np.full_like(state.release_force, release_force, dtype=np.float64)
    )
    return coupler.adhesion.add_bond_state_request(
        tape,
        rigid_source=rigid_source,
        name=name,
        topologies=batch.topologies,
        source_fem_global_offset=batch.fem_offset,
        Dm_inv=state.Dm_inv,
        V0=state.V0,
        d_rest=state.d_rest,
        kappa=state.kappa,
        release_force=restored_release_force,
        release_strain=state.release_strain,
        release_gap=state.release_gap,
        release_slip=state.release_slip,
        age=state.age,
    )


def _bond_row_kwargs(batch: TapeTableBondBatch, *, release_force: float | None = None) -> dict[str, object]:
    contact = batch.contact_policy
    bond = batch.bond_policy
    if contact["enable"] is not True or contact["adhesion"] is not None:
        gs.raise_exception("Tape-table runtime bond rows require an enabled barrier and no soft adhesion.")
    release = _release_policy(batch)
    return {
        "Cn": 0.0,
        "Ct": 0.0,
        "W": 0.0,
        "eta": 1.0,
        "bonding_rate": 0.0,
        "beta0": 0.0,
        "sticky": (0, 0, 0, 0),
        # BondSystem uses Adhesion.sticky as its independent lock-creation
        # gate. A zero-energy Adhesion row with no sticky side preserves the
        # authored bonds while preventing un-authored locks from appearing.
        "enabled": True,
        "friction": float(contact["friction_rate"]),
        "resistance": float(contact["resistance"]),
        "distance_lock": True,
        "distance_lock_ratio": float(bond["ratio"]),
        "distance_lock_floor_ratio": float(bond["floor_ratio"]),
        "distance_lock_rest_snap": bool(bond["rest_snap"]),
        "release_force": float(release["force"]) if release_force is None else release_force,
    }


def _adhesion_row_kwargs(policy: dict, batch: TapeTableBondBatch) -> dict[str, object]:
    """A cohesive tape/table row: the adhesion law carries the interface, no locks.

    Unlike `_bond_row_kwargs` this leaves the barrier alone -- a distance lock is
    source-removed from the contact broad-phase and has to stand in for the barrier it
    displaces, which forces it to be near-rigid; a cohesive row coexists with the barrier,
    so its stiffness answers only to the peel force it is calibrated against.
    """
    contact = batch.contact_policy
    return {
        "Cn": float(policy["Cn"]),
        "Ct": float(policy["Ct"]),
        "W": float(policy["W"]),
        "eta": float(policy["eta"]),
        "bonding_rate": float(policy.get("bonding_rate", 1.0)),
        "beta0": float(policy.get("beta0", 1.0)),
        "sticky": tuple(int(flag) for flag in policy["sticky"]),
        "enabled": True,
        "friction": float(contact["friction_rate"]),
        "resistance": float(contact["resistance"]),
        "distance_lock": False,
    }


def _validate_coupler_options(coupler, asset: TapeTableComponentAsset) -> None:
    options = coupler.options
    expected = recommended_coupler_options(asset)
    critical = (
        "contact_enable",
        "contact_d_hat",
        "contact_constitution",
        "adhesion_bond_distance_lock",
        "adhesion_bond_default",
        "adhesion_bond_kappa",
        "adhesion_bond_lock_margin",
        "adhesion_occlusion",
    )
    mismatched = [name for name in critical if getattr(options, name) != expected[name]]
    if options.adhesion_bond_max_bonds <= 0:
        mismatched.append("adhesion_bond_max_bonds")
    if mismatched:
        gs.raise_exception(
            "add_tape_table_component requires recommended_coupler_options(asset); mismatched: "
            + ", ".join(mismatched)
            + "."
        )


def _root_transform(entity: RigidEntity) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(entity.morph.pos, dtype=np.float64)
    transform[:3, :3] = gu.quat_to_R(np.asarray(entity.morph.quat, dtype=np.float64))
    return transform


def _inverse_affine_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    affine = transform[:3, :3]
    translation = transform[:3, 3]
    return np.ascontiguousarray((points - translation) @ np.linalg.inv(affine).T, dtype=np.float64)
