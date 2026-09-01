"""Adhesion support for the QIPC coupler (design doc: docs/adhesion_tape_design.md).

Wraps QIPC's two-layer adhesion stack:

- Phase-1 soft adhesion: per-pair beta-state RCC potential, configured through
  per-pair `Adhesion` contact-table groups.
- Phase-2 distance bonds: locked virtual-tet bonds configured through per-pair
  `Bond` contact-table groups.

Everything is declarative-before-build: QIPC reads the contact tables exactly
once at scene init. The only runtime-mutable state is the per-pair beta table
(dump/load) and bond seeding. Seeded frame-zero bonds are replayed after reset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import geometric_mean, harmonic_mean

if TYPE_CHECKING:
    from genesis.options.solvers import QIPCCouplerOptions


def _contact_pt_keys(topologies: np.ndarray) -> np.ndarray:
    """QIPC's positive FNV-1a key for canonical point-triangle rows."""
    rows = np.asarray(topologies, dtype=np.int32).reshape(-1, 4)
    keys = np.empty(len(rows), dtype=np.int64)
    uint64_mask = (1 << 64) - 1
    positive_mask = (1 << 63) - 1
    for index, row in enumerate(rows):
        point = int(row[0]) & 0xFFFFFFFF
        triangle = sorted(int(value) & 0xFFFFFFFF for value in row[1:])
        value = 1469598103934665603
        for vertex in (point, *triangle):
            value ^= vertex
            value = (value * 1099511628211) & uint64_mask
        keys[index] = value & positive_mask
    return keys


def _require_current_qipc_api() -> None:
    try:
        from qipc import Scene as QIPCScene
        from qipc.contact import Adhesion, Bond, Release

        required = (Adhesion, Bond, Release, QIPCScene.reset)
    except (AttributeError, ImportError):
        gs.raise_exception(
            "QIPCCoupler requires a current cuda-graph-qipc build with the grouped "
            "contact API and reset support. Install the Genesis 'qipc' extra in its "
            "supported Python 3.12 environment; legacy flat-contact QIPC builds are not supported."
        )
    if any(value is None for value in required):
        gs.raise_exception("QIPCCoupler requires callable grouped contact and reset APIs from cuda-graph-qipc.")


class AdhesionRequest(NamedTuple):
    """One add_adhesion() call, resolved into contact-tabular rows at build."""

    source: object  # RigidEntity | FEMEntity
    targets: tuple  # entities; may include source (self-adhesion)
    Cn: float
    Ct: float
    W: float
    eta: float
    bonding_rate: float
    p0: float
    beta0: float
    sticky: tuple[int, int, int, int]
    enabled: bool
    friction: float | None
    resistance: float | None
    distance_lock: bool | None
    distance_lock_ratio: float | None
    distance_lock_floor_ratio: float | None
    distance_lock_rest_snap: bool | None
    release_force: float | None
    release_strain: float | None
    release_gap: float | None
    release_slip: float | None


@dataclass(frozen=True, eq=False)
class BondSeedHandle:
    """Stable handle and provenance for one authored bond batch."""

    key: int
    name: str
    fem_entity: object = field(repr=False)
    rigid_entity: object | None = field(repr=False)
    source_fem_global_offset: int | None
    rest_height: float
    strict_rigid_mapping: bool
    _owner: object = field(repr=False)


class BondSeedRequest(NamedTuple):
    """Authored distance bonds resolved into this scene after QIPC init."""

    handle: BondSeedHandle
    topologies: np.ndarray


@dataclass(frozen=True, eq=False)
class BondStateHandle:
    """Stable handle and provenance for one frozen bond-state batch."""

    key: int
    name: str
    fem_entity: object = field(repr=False)
    rigid_entity: object | None = field(repr=False)
    source_fem_global_offset: int
    strict_rigid_mapping: bool
    _owner: object = field(repr=False)


class BondStateRequest(NamedTuple):
    """Frozen bond slots resolved from component-local to scene-global IDs."""

    handle: BondStateHandle
    topologies: np.ndarray
    Dm_inv: np.ndarray
    V0: np.ndarray
    d_rest: np.ndarray
    kappa: np.ndarray
    release_force: np.ndarray
    release_strain: np.ndarray
    release_gap: np.ndarray
    release_slip: np.ndarray
    age: np.ndarray


class QIPCAdhesionManager:
    """Owns adhesion declarations and runtime adhesion/bond state access.

    The coupler wires it into three build hooks (request queue -> tabular rows,
    constitution selection) and exposes it as ``coupler.adhesion``.
    """

    def __init__(self, options: QIPCCouplerOptions) -> None:
        self._options = options
        self._requests: list[AdhesionRequest] = []
        self._bond_seed_requests: list[BondSeedRequest] = []
        self._bond_state_requests: list[BondStateRequest] = []
        self._bond_seed_batch: tuple[np.ndarray, float] | None = None
        self._bond_state_batches: list[dict[str, np.ndarray]] = []
        self._bond_seed_handles_by_entity: dict[object, list[BondSeedHandle]] = {}
        self._bond_state_handles_by_entity: dict[object, list[BondStateHandle]] = {}
        self._bond_seed_results: dict[BondSeedHandle, tuple[int, int]] = {}
        self._bond_seed_topologies_by_handle: dict[BondSeedHandle, np.ndarray] = {}
        self._bond_state_results: dict[BondStateHandle, tuple[int, int]] = {}
        self._bond_state_topologies_by_handle: dict[BondStateHandle, np.ndarray] = {}
        self._bond_state_captured_in_reset = False
        self._bond_dump_topologies: torch.Tensor | None = None
        self._bond_dump_count: torch.Tensor | None = None
        self._scene = None  # qipc Scene, set by the coupler at build

    # -------------------------------------------------------------------------
    # Declaration (pre-build)
    # -------------------------------------------------------------------------

    def add_request(
        self,
        source_entity,
        target_entities=None,
        *,
        Cn: float,
        Ct: float = 0.0,
        W: float = 1.0,
        eta: float = 1.0,
        bonding_rate: float = 0.0,
        p0: float = 0.0,
        beta0: float = 0.0,
        sticky: tuple[int, int, int, int] = (1, 1, 1, 1),
        enabled: bool = True,
        friction: float | None = None,
        resistance: float | None = None,
        distance_lock: bool | None = None,
        distance_lock_ratio: float | None = None,
        distance_lock_floor_ratio: float | None = None,
        distance_lock_rest_snap: bool | None = None,
        release_force: float | None = None,
        release_strain: float | None = None,
        release_gap: float | None = None,
        release_slip: float | None = None,
    ) -> None:
        """Queue an adhesion declaration (see QIPCCoupler.add_adhesion)."""
        if self._scene is not None:
            gs.raise_exception("QIPCCoupler.add_adhesion must be called before scene.build().")
        if target_entities is None:
            targets: tuple = (source_entity,)
        elif isinstance(target_entities, (list, tuple, set, frozenset)):
            targets = tuple(target_entities)
        else:
            targets = (target_entities,)
        if not targets:
            gs.raise_exception("QIPCCoupler.add_adhesion requires at least one target entity.")
        for value, name in (
            (Cn, "Cn"),
            (Ct, "Ct"),
            (W, "W"),
            (bonding_rate, "bonding_rate"),
            (p0, "p0"),
        ):
            if value < 0:
                gs.raise_exception(f"QIPCCoupler.add_adhesion: {name} must be non-negative.")
        if eta <= 0:
            gs.raise_exception("QIPCCoupler.add_adhesion: eta must be positive.")
        if not (0.0 <= beta0 <= 1.0):
            gs.raise_exception("QIPCCoupler.add_adhesion: beta0 must be in [0, 1].")
        sticky = tuple(sticky)
        if len(sticky) != 4 or any(value not in (0, 1) for value in sticky):
            gs.raise_exception("QIPCCoupler.add_adhesion: sticky must be a 4-tuple containing only 0 or 1.")
        if distance_lock_ratio is not None and distance_lock_ratio <= 0:
            gs.raise_exception("QIPCCoupler.add_adhesion: distance_lock_ratio must be positive.")
        if distance_lock_floor_ratio is not None and distance_lock_floor_ratio < 0:
            gs.raise_exception("QIPCCoupler.add_adhesion: distance_lock_floor_ratio must be non-negative.")
        for value, name in (
            (release_force, "release_force"),
            (release_strain, "release_strain"),
            (release_gap, "release_gap"),
            (release_slip, "release_slip"),
        ):
            if value is not None and value < 0:
                gs.raise_exception(f"QIPCCoupler.add_adhesion: {name} must be non-negative.")

        self._requests.append(
            AdhesionRequest(
                source=source_entity,
                targets=targets,
                Cn=float(Cn),
                Ct=float(Ct),
                W=float(W),
                eta=float(eta),
                bonding_rate=float(bonding_rate),
                p0=float(p0),
                beta0=float(beta0),
                sticky=sticky,
                enabled=bool(enabled),
                friction=friction,
                resistance=resistance,
                distance_lock=distance_lock,
                distance_lock_ratio=distance_lock_ratio,
                distance_lock_floor_ratio=distance_lock_floor_ratio,
                distance_lock_rest_snap=distance_lock_rest_snap,
                release_force=release_force,
                release_strain=release_strain,
                release_gap=release_gap,
                release_slip=release_slip,
            )
        )

    def add_bond_seed_request(
        self,
        fem_entity,
        *,
        topologies: np.ndarray,
        source_fem_global_offset: int | None,
        rest_height: float,
        rigid_entity=None,
        name: str = "default",
        strict_rigid_mapping: bool = False,
    ) -> BondSeedHandle:
        """Queue one named authored bond batch for post-init global-id resolution."""
        if self._scene is not None:
            gs.raise_exception("QIPCCoupler.add_bond_seed_request must be called before scene.build().")
        if self._bond_state_requests:
            gs.raise_exception("QIPCCoupler.add_bond_seed_request cannot be combined with a frozen bond state request.")
        if not isinstance(name, str) or not name:
            gs.raise_exception("QIPCCoupler.add_bond_seed_request: name must be a non-empty string.")
        if not isinstance(strict_rigid_mapping, bool):
            gs.raise_exception("QIPCCoupler.add_bond_seed_request: strict_rigid_mapping must be a bool.")
        entity_handles = self._bond_seed_handles_by_entity.setdefault(fem_entity, [])
        if any(handle.name == name for handle in entity_handles):
            gs.raise_exception(
                f"QIPCCoupler: authored bond seed name '{name}' is already registered for this FEM entity."
            )

        topos = np.ascontiguousarray(topologies, dtype=np.int32).reshape(-1, 4)
        if topos.size and int(topos.min()) < 0:
            gs.raise_exception("QIPCCoupler: authored bond topology vertex ids must be non-negative.")
        if source_fem_global_offset is not None and source_fem_global_offset < 0:
            gs.raise_exception("QIPCCoupler: source_fem_global_offset must be non-negative.")
        if not np.isfinite(rest_height) or rest_height < 0.0:
            gs.raise_exception("QIPCCoupler: authored bond rest_height must be finite and non-negative.")

        handle = BondSeedHandle(
            key=len(self._bond_seed_requests),
            name=name,
            fem_entity=fem_entity,
            rigid_entity=rigid_entity,
            source_fem_global_offset=source_fem_global_offset,
            rest_height=float(rest_height),
            strict_rigid_mapping=strict_rigid_mapping,
            _owner=self,
        )
        entity_handles.append(handle)
        self._bond_seed_requests.append(
            BondSeedRequest(
                handle=handle,
                topologies=topos.copy(),
            )
        )
        return handle

    def add_bond_state_request(
        self,
        fem_entity,
        *,
        rigid_source,
        name: str = "default",
        topologies: np.ndarray,
        source_fem_global_offset: int,
        Dm_inv: np.ndarray,
        V0: np.ndarray,
        d_rest: np.ndarray,
        kappa: np.ndarray,
        release_force: np.ndarray,
        release_strain: np.ndarray,
        release_gap: np.ndarray,
        release_slip: np.ndarray,
        age: np.ndarray,
    ) -> BondStateHandle:
        """Queue a complete frozen BondSystem state for post-init remapping.

        Unlike ``add_bond_seed_request``, this preserves every per-slot rest,
        stiffness, release, and age field. Named requests are aggregated into
        one native replacement transaction and cannot mix with ordinary
        authored bond seeds.
        """
        if self._scene is not None:
            gs.raise_exception("QIPCCoupler.add_bond_state_request must be called before scene.build().")
        if self._bond_seed_requests:
            gs.raise_exception("QIPCCoupler.add_bond_state_request cannot be combined with a seeded bond request.")
        if not isinstance(name, str) or not name:
            gs.raise_exception("QIPCCoupler.add_bond_state_request: name must be a non-empty string.")
        entity_handles = self._bond_state_handles_by_entity.get(fem_entity, ())
        if any(handle.name == name for handle in entity_handles):
            gs.raise_exception(
                f"QIPCCoupler: frozen bond state name '{name}' is already registered for this FEM entity."
            )
        topos = np.ascontiguousarray(topologies, dtype=np.int32)
        if topos.ndim != 2 or topos.shape[1] != 4:
            gs.raise_exception("QIPCCoupler.add_bond_state_request: topologies must have shape (n, 4).")
        if topos.size and int(topos.min()) < 0:
            gs.raise_exception("QIPCCoupler.add_bond_state_request: topology IDs must be non-negative.")
        if isinstance(source_fem_global_offset, bool) or not isinstance(source_fem_global_offset, (int, np.integer)):
            gs.raise_exception("QIPCCoupler.add_bond_state_request: source_fem_global_offset must be an integer.")
        source_fem_global_offset = int(source_fem_global_offset)
        if source_fem_global_offset <= 0:
            gs.raise_exception("QIPCCoupler.add_bond_state_request: source_fem_global_offset must be positive.")

        n_bonds = len(topos)
        if n_bonds == 0:
            gs.raise_exception("QIPCCoupler.add_bond_state_request: topologies must not be empty.")
        is_fem = topos >= source_fem_global_offset
        if np.any(is_fem.sum(axis=1) == 0):
            gs.raise_exception("QIPCCoupler.add_bond_state_request: every topology must reference a FEM vertex.")
        triangle_fem_count = is_fem[:, 1:].sum(axis=1)
        if np.any((triangle_fem_count != 0) & (triangle_fem_count != 3)):
            gs.raise_exception(
                "QIPCCoupler.add_bond_state_request: each PT triangle must belong entirely to FEM or rigid vertices."
            )

        def vector(
            values,
            name: str,
            dtype,
            *,
            positive: bool = False,
            non_negative: bool = False,
        ) -> np.ndarray:
            array = np.ascontiguousarray(values, dtype=dtype)
            if array.shape != (n_bonds,):
                gs.raise_exception(f"QIPCCoupler.add_bond_state_request: {name} must have shape ({n_bonds},).")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                gs.raise_exception(f"QIPCCoupler.add_bond_state_request: {name} must be finite.")
            if positive and np.any(array <= 0):
                gs.raise_exception(f"QIPCCoupler.add_bond_state_request: {name} must be positive.")
            if non_negative and np.any(array < 0):
                gs.raise_exception(f"QIPCCoupler.add_bond_state_request: {name} must be non-negative.")
            return array.copy()

        Dm_inv_array = np.ascontiguousarray(Dm_inv, dtype=np.float64)
        if Dm_inv_array.shape != (n_bonds, 9) or not np.isfinite(Dm_inv_array).all():
            gs.raise_exception(f"QIPCCoupler.add_bond_state_request: Dm_inv must be finite with shape ({n_bonds}, 9).")
        if np.any(np.linalg.matrix_rank(Dm_inv_array.reshape(-1, 3, 3)) < 3):
            gs.raise_exception("QIPCCoupler.add_bond_state_request: every Dm_inv matrix must be invertible.")
        request_topologies = topos.copy()
        request_Dm_inv = Dm_inv_array.copy()
        request_V0 = vector(V0, "V0", np.float64, positive=True)
        expected_V0 = np.abs(1.0 / np.linalg.det(request_Dm_inv.reshape(-1, 3, 3))) / 6.0
        if not np.allclose(request_V0, expected_V0, rtol=1.0e-10, atol=0.0):
            gs.raise_exception("QIPCCoupler.add_bond_state_request: V0 must equal abs(1 / det(Dm_inv)) / 6.")
        request_d_rest = vector(d_rest, "d_rest", np.float64)
        request_kappa = vector(kappa, "kappa", np.float64, positive=True)
        request_release_force = vector(release_force, "release_force", np.float64, non_negative=True)
        request_release_strain = vector(release_strain, "release_strain", np.float64, non_negative=True)
        request_release_gap = vector(release_gap, "release_gap", np.float64, non_negative=True)
        request_release_slip = vector(release_slip, "release_slip", np.float64, non_negative=True)
        request_age = vector(age, "age", np.int32, non_negative=True)
        handle = BondStateHandle(
            key=len(self._bond_state_requests),
            name=name,
            fem_entity=fem_entity,
            rigid_entity=rigid_source,
            source_fem_global_offset=source_fem_global_offset,
            strict_rigid_mapping=True,
            _owner=self,
        )
        request = BondStateRequest(
            handle=handle,
            topologies=request_topologies,
            Dm_inv=request_Dm_inv,
            V0=request_V0,
            d_rest=request_d_rest,
            kappa=request_kappa,
            release_force=request_release_force,
            release_strain=request_release_strain,
            release_gap=request_release_gap,
            release_slip=request_release_slip,
            age=request_age,
        )
        self._bond_state_handles_by_entity.setdefault(fem_entity, []).append(handle)
        self._bond_state_requests.append(request)
        return handle

    # -------------------------------------------------------------------------
    # Build hooks (called by the coupler)
    # -------------------------------------------------------------------------

    def enabled(self) -> bool:
        """Whether the adhesive contact constitution should be selected."""
        opt = self._options
        if opt.contact_constitution == "adhesive":
            return True
        if opt.contact_constitution == "consistent":
            return False
        return bool(self._requests) or opt.adhesion_bond_distance_lock or opt.adhesion_bond_max_bonds > 0

    def bonds_enabled(self) -> bool:
        opt = self._options
        return self.enabled() and opt.adhesion_bond_distance_lock and opt.adhesion_bond_max_bonds > 0

    def has_bond_seed_requests(self) -> bool:
        return bool(self._bond_seed_requests or self._bond_state_requests)

    def has_frozen_bond_state(self) -> bool:
        return bool(self._bond_state_requests)

    def apply_constitution(self, scene, *, has_fem_entities: bool) -> None:
        """Validate and apply AdhesiveIPCContact (must run before scene.init())."""
        self._scene = scene
        opt = self._options

        if not self.enabled():
            if opt.adhesion_bond_distance_lock or opt.adhesion_bond_max_bonds > 0:
                gs.raise_exception(
                    "QIPCCoupler: adhesion_bond_* options are set but contact_constitution='consistent'."
                )
            if self._requests:
                gs.raise_exception("QIPCCoupler: add_adhesion() was called but contact_constitution='consistent'.")
            return

        _require_current_qipc_api()

        if opt.adhesion_bond_distance_lock and opt.adhesion_bond_max_bonds <= 0:
            gs.raise_exception(
                "QIPCCoupler: adhesion_bond_distance_lock=True requires adhesion_bond_max_bonds > 0 "
                "(the Genesis option remains the distance-bond enable guard)."
            )
        if self.bonds_enabled() and opt.adhesion_bond_distance_lock_ratio <= 0:
            gs.raise_exception("QIPCCoupler: adhesion_bond_distance_lock_ratio must be positive when bonds are on.")
        if self.bonds_enabled() and not has_fem_entities:
            gs.raise_exception(
                "QIPCCoupler: distance bonds require at least one FEM entity in the scene "
                "(QIPC's BondSystem needs FEM vertices; pure rigid-rigid bonding is unsupported)."
            )

        from qipc.constitution import AdhesiveIPCContact

        if self.bonds_enabled() and opt.adhesion_bond_default:
            scene.contact_tabular.default_model(
                friction_rate=opt.contact_friction,
                resistance=opt.contact_resistance,
                bond=self._make_bond(),
            )
        AdhesiveIPCContact().apply_to(scene)
        gs.logger.debug(
            "QIPCCoupler: AdhesiveIPCContact applied (bonds " + ("on" if self.bonds_enabled() else "off") + ")."
        )

    def insert_tabular_rows(self, tab, elem_by_entity: dict) -> None:
        """Write adhesion request rows into the contact tabular.

        Must run AFTER the coupler's plain friction/resistance pair rows
        (insert is an upsert, so these rows override).
        """
        _require_current_qipc_api()

        from qipc.contact import Adhesion

        for req in self._requests:
            src = elem_by_entity.get(req.source)
            if src is None:
                gs.raise_exception(
                    "QIPCCoupler.add_adhesion: source entity is not a coupled rigid/FEM entity "
                    "(Plane/ground entities cannot participate in adhesion)."
                )
            elem_s, mu_s, res_s = src
            for target in req.targets:
                tgt = elem_by_entity.get(target)
                if tgt is None:
                    gs.raise_exception(
                        "QIPCCoupler.add_adhesion: target entity is not a coupled rigid/FEM entity "
                        "(Plane/ground entities cannot participate in adhesion: half-plane contacts "
                        "never enter the point-triangle adhesion kernel)."
                    )
                elem_t, mu_t, res_t = tgt
                friction = req.friction if req.friction is not None else float(geometric_mean(mu_s, mu_t))
                resistance = req.resistance if req.resistance is not None else float(harmonic_mean(res_s, res_t))
                adhesion = (
                    Adhesion(
                        Cn=req.Cn,
                        Ct=req.Ct,
                        W=req.W,
                        eta=req.eta,
                        bonding_rate=req.bonding_rate,
                        p0=req.p0,
                        beta0=req.beta0,
                        sticky=req.sticky,
                    )
                    if req.enabled
                    else None
                )
                bond = None
                if self.bonds_enabled() and req.distance_lock is not False:
                    bond = self._make_bond(
                        ratio=req.distance_lock_ratio,
                        floor_ratio=req.distance_lock_floor_ratio,
                        rest_snap=req.distance_lock_rest_snap,
                        release_force=req.release_force,
                        release_strain=req.release_strain,
                        release_gap=req.release_gap,
                        release_slip=req.release_slip,
                    )
                tab.insert(
                    elem_s,
                    elem_t,
                    friction_rate=float(friction),
                    resistance=float(resistance),
                    adhesion=adhesion,
                    bond=bond,
                )

    def _make_bond(
        self,
        *,
        ratio: float | None = None,
        floor_ratio: float | None = None,
        rest_snap: bool | None = None,
        release_force: float | None = None,
        release_strain: float | None = None,
        release_gap: float | None = None,
        release_slip: float | None = None,
    ):
        from qipc.contact import Bond, Release

        opt = self._options
        return Bond(
            kappa=opt.adhesion_bond_kappa,
            ratio=opt.adhesion_bond_distance_lock_ratio if ratio is None else float(ratio),
            margin=opt.adhesion_bond_lock_margin,
            floor_ratio=opt.adhesion_bond_lock_floor_ratio if floor_ratio is None else float(floor_ratio),
            rest_snap=True if rest_snap is None else bool(rest_snap),
            occlusion=opt.adhesion_occlusion,
            release=Release(
                strain=opt.adhesion_bond_release_strain if release_strain is None else float(release_strain),
                force=opt.adhesion_bond_release_force if release_force is None else float(release_force),
                gap=opt.adhesion_bond_release_gap if release_gap is None else float(release_gap),
                slip=opt.adhesion_bond_release_slip if release_slip is None else float(release_slip),
            ),
        )

    # -------------------------------------------------------------------------
    # Runtime state (post-build)
    # -------------------------------------------------------------------------

    def _require_scene(self):
        if self._scene is None:
            gs.raise_exception("QIPCCoupler: adhesion runtime state is only available after scene.build().")
        return self._scene

    def _bond_system(self):
        scene = self._require_scene()
        bond_system = scene._bond_system
        if bond_system is None or bond_system.max_bonds <= 0:
            gs.raise_exception(
                "QIPCCoupler: distance bonds are not enabled "
                "(set adhesion_bond_distance_lock=True and adhesion_bond_max_bonds > 0)."
            )
        return bond_system

    def _adhesion_system(self):
        scene = self._require_scene()
        for cls, system in scene.sim_systems.items():
            if cls.__name__ == "AdhesiveIPCContactConstitution":
                return system
        gs.raise_exception("QIPCCoupler: the adhesive contact constitution is not active in this scene.")

    def get_contact_info(self) -> tuple[int, int, int]:
        """(n_pairs_pt, n_pairs_ee, n_active) from the QIPC contact system."""
        return self._require_scene().solver.get_contact_info()

    def fem_global_vertex_offset(self) -> int:
        """Global vertex id at which the FEM block starts.

        QIPC lays global vertices out as ``[ABD | RIGID | FEM | halfplane]``,
        so the FEM block starts after both the affine and pure-rigid vertices.
        """
        scene = self._require_scene()
        n_abd = int(scene.affine_body.n_verts)
        n_rigid = int(scene.rigid_body.n_verts)
        return n_abd + n_rigid

    def get_bond_topos(self) -> np.ndarray:
        """Alive distance-bond topologies as (n, 4) GLOBAL vertex ids.

        FEM vertex global id = fem_global_vertex_offset() + geometry
        fem_vert_offset + local index; ids below the offset are non-FEM prefix vertices.
        """
        return self._dump_bond_topologies(is_released=False)

    def get_released_bond_topos(self) -> np.ndarray:
        """Distance bonds released by the preceding QIPC step.

        The returned ``(n, 4)`` array uses GLOBAL vertex ids and remains valid
        until the next QIPC step begins. This feed lets application policies
        advance a peel or fracture front without scanning historical bonds.
        """
        return self._dump_bond_topologies(is_released=True)

    def _dump_bond_topologies(self, *, is_released: bool) -> np.ndarray:
        bond_system = self._bond_system()
        required = bond_system.max_bonds * 4
        if self._bond_dump_topologies is None or self._bond_dump_topologies.numel() < required:
            self._bond_dump_topologies = torch.empty(required, dtype=torch.int32, device="cuda")
        if self._bond_dump_count is None:
            self._bond_dump_count = torch.zeros(1, dtype=torch.int32, device="cuda")
        out = self._bond_dump_topologies
        cnt = self._bond_dump_count
        cnt.zero_()
        if is_released:
            bond_system.dump_released_topos(out, cnt)
        else:
            bond_system.dump_lock_topos(out, cnt)
        torch.cuda.synchronize()
        n = int(cnt.item())
        return out[: n * 4].reshape(n, 4).cpu().numpy()

    def get_bond_count(self) -> int:
        return int(self.get_bond_topos().shape[0])

    def release_bonds_by_vertices(self, vertex_ids: np.ndarray, *, require_all: bool) -> None:
        """Release locks selected by GLOBAL vertex ids.

        With ``require_all=False`` a lock is removed when any stencil vertex is
        selected; with ``True`` every stencil vertex must be selected. The
        former is appropriate when a detached FEM region still shares bonds
        with the cluster interior.
        """
        vertices = np.asarray(vertex_ids, dtype=np.int64).reshape(-1)
        if vertices.size == 0:
            return
        scene = self._require_scene()
        n_global = self.fem_global_vertex_offset() + int(scene.finite_element.n_verts)
        if int(vertices.min()) < 0 or int(vertices.max()) >= n_global:
            gs.raise_exception(f"QIPCCoupler.release_bonds_by_vertices: global vertex ids must lie in [0, {n_global}).")
        mask = torch.zeros(n_global, dtype=torch.int32, device="cuda")
        mask[torch.as_tensor(np.unique(vertices), dtype=torch.int64, device="cuda")] = 1
        self._bond_system().release_locks_by_verts(mask, n_global, require_all=bool(require_all))
        torch.cuda.synchronize()

    def seed_bonds(self, topos: np.ndarray, rest_height: float) -> None:
        """Seed distance bonds from (n, 4) GLOBAL vertex-id topologies.

        Call after `scene.build()` and before the first step. QIPC stamps each
        seed's stiffness and release thresholds from its resolved per-pair
        `Bond` model. The batch is replayed after subsequent scene resets.
        """
        topologies = np.ascontiguousarray(topos, dtype=np.int32).reshape(-1, 4)
        if topologies.size == 0:
            return
        if self._bond_state_requests:
            gs.raise_exception("QIPCCoupler.seed_bonds cannot be combined with a frozen BondSystem state.")
        if int(topologies.min()) < 0:
            gs.raise_exception("QIPCCoupler.seed_bonds: topology vertex ids must be non-negative.")
        if not np.isfinite(rest_height) or rest_height < 0.0:
            gs.raise_exception("QIPCCoupler.seed_bonds: rest_height must be finite and non-negative.")
        if self._bond_seed_batch is not None:
            gs.raise_exception(
                "QIPCCoupler.seed_bonds: a frame-zero seed transaction is already installed; "
                "queue named add_bond_seed_request batches before scene.build() to aggregate multiple sources."
            )

        height = float(rest_height)
        self._require_nonoverlapping_bond_topologies((("manual", topologies),), kind="seed")
        self._seed_bond_batch(topologies, height)
        self._bond_seed_batch = (topologies.copy(), height)
        self._bond_state_captured_in_reset = False

    def apply_bond_seed_requests(
        self,
        fem_layout: dict[object, tuple[int, int]],
        rigid_vertex_ids: dict[object, np.ndarray],
    ) -> None:
        """Resolve all authored batches, then install one native transaction."""
        if not self._bond_seed_requests and not self._bond_state_requests:
            return

        fem_global_offset = self.fem_global_vertex_offset()
        resolved: list[tuple[BondSeedRequest, np.ndarray, int]] = []
        for request in self._bond_seed_requests:
            handle = request.handle
            layout = fem_layout.get(handle.fem_entity)
            if layout is None:
                gs.raise_exception("QIPCCoupler: authored bond seed FEM entity is not coupled in this scene.")
            fem_offset, n_fem_vertices = layout
            mapped, n_dropped = self._map_bond_seed_request(
                request,
                fem_global_offset=fem_global_offset,
                fem_offset=fem_offset,
                n_fem_vertices=n_fem_vertices,
                rigid_vertex_ids=rigid_vertex_ids.get(handle.rigid_entity),
            )
            resolved.append((request, mapped, n_dropped))

        non_empty = [(request, mapped) for request, mapped, _ in resolved if len(mapped)]
        if non_empty:
            rest_height = non_empty[0][0].handle.rest_height
            for request, _ in non_empty[1:]:
                self._require_compatible_rest_height(
                    rest_height,
                    request.handle.rest_height,
                    first_name=non_empty[0][0].handle.name,
                    second_name=request.handle.name,
                )
            self._require_nonoverlapping_bond_topologies(
                tuple((request.handle.name, mapped) for request, mapped in non_empty),
                kind="seed",
            )
            combined = np.concatenate([mapped for _, mapped in non_empty], axis=0)
            self._seed_bond_batch(combined, rest_height)
            self._bond_seed_batch = (combined.copy(), rest_height)
            self._bond_state_captured_in_reset = False

        for request, mapped, n_dropped in resolved:
            handle = request.handle
            self._bond_seed_topologies_by_handle[handle] = mapped.copy()
            result = (len(mapped), n_dropped)
            self._bond_seed_results[handle] = result
            gs.logger.info(
                f"QIPCCoupler: mapped {result[0]} authored distance bonds for batch '{handle.name}' "
                f"({result[1]} rows dropped because their rigid-side ids could not be mapped)."
            )

        resolved_state: list[tuple[BondStateRequest, np.ndarray]] = []
        for request in self._bond_state_requests:
            handle = request.handle
            layout = fem_layout.get(handle.fem_entity)
            if layout is None:
                gs.raise_exception("QIPCCoupler: frozen bond state FEM entity is not coupled in this scene.")
            fem_offset, n_fem_vertices = layout
            mapped, n_dropped = self._map_bond_seed_request(
                request,
                fem_global_offset=fem_global_offset,
                fem_offset=fem_offset,
                n_fem_vertices=n_fem_vertices,
                rigid_vertex_ids=rigid_vertex_ids.get(handle.rigid_entity),
            )
            if n_dropped:
                gs.raise_exception("QIPCCoupler: frozen bond state could not map its rigid attachment vertices.")
            resolved_state.append((request, mapped))

        if resolved_state:
            self._require_nonoverlapping_bond_topologies(
                tuple((request.handle.name, mapped) for request, mapped in resolved_state),
                kind="frozen-state",
            )
            state_fields = (
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
            batch = {
                "topo": np.ascontiguousarray(np.concatenate([mapped for _, mapped in resolved_state], axis=0)),
                **{
                    name: np.ascontiguousarray(
                        np.concatenate([request[field_index] for request, _ in resolved_state], axis=0)
                    )
                    for field_index, name in enumerate(state_fields, start=2)
                },
            }
            batch["pair_key"] = _contact_pt_keys(batch["topo"])
            self._restore_bond_state_batch(batch)
            self._bond_state_batches.append(batch)
            self._bond_state_captured_in_reset = False

        for request, mapped in resolved_state:
            handle = request.handle
            self._bond_state_topologies_by_handle[handle] = mapped.copy()
            self._bond_state_results[handle] = (len(mapped), 0)
            gs.logger.info(f"QIPCCoupler: mapped {len(mapped)} frozen distance bonds for batch '{handle.name}'.")

    def restore_seeded_bonds(self) -> None:
        """Replay authored/manual frame-zero bond batches after `Scene.reset`."""
        if self._bond_state_captured_in_reset:
            return
        for batch in self._bond_state_batches:
            self._restore_bond_state_batch(batch)
        if self._bond_seed_batch is not None:
            topologies, rest_height = self._bond_seed_batch
            self._seed_bond_batch(topologies, rest_height)

    def mark_bond_state_captured_in_reset(self) -> None:
        """Record that QIPC's current reset snapshot already owns all authored bonds."""
        self._bond_state_captured_in_reset = True

    def get_bond_seed_handle(self, fem_entity, *, name: str) -> BondSeedHandle | None:
        """Return a named pre-build handle for one FEM entity."""
        handles = self._bond_seed_handles_by_entity.get(fem_entity, ())
        return next((handle for handle in handles if handle.name == name), None)

    def get_bond_state_handle(self, fem_entity, *, name: str) -> BondStateHandle | None:
        """Return a named frozen-state handle for one FEM entity."""
        handles = self._bond_state_handles_by_entity.get(fem_entity, ())
        return next((handle for handle in handles if handle.name == name), None)

    def get_bond_seed_result(
        self,
        seed: BondSeedHandle | BondStateHandle | object,
    ) -> tuple[int, int] | None:
        """Return `(restored, dropped_rigid_rows)` for a handle or legacy FEM query."""
        if isinstance(seed, BondStateHandle):
            return self.get_bond_state_result(seed)
        handle = self._resolve_bond_seed_handle(seed)
        if handle is not None:
            return self._bond_seed_results.get(handle)
        return self.get_bond_state_result(seed)

    def get_bond_seed_topologies(
        self,
        seed: BondSeedHandle | BondStateHandle | object,
    ) -> np.ndarray | None:
        """Return mapped authored rows for one handle or an unambiguous legacy FEM query."""
        if isinstance(seed, BondStateHandle):
            return self.get_bond_state_topologies(seed)
        handle = self._resolve_bond_seed_handle(seed)
        if handle is None:
            return self.get_bond_state_topologies(seed)
        topologies = self._bond_seed_topologies_by_handle.get(handle)
        return None if topologies is None else topologies.copy()

    def get_bond_state_result(self, state: BondStateHandle | object) -> tuple[int, int] | None:
        """Return `(restored, dropped_rigid_rows)` for one frozen-state batch."""
        handle = self._resolve_bond_state_handle(state)
        return None if handle is None else self._bond_state_results.get(handle)

    def get_bond_state_topologies(self, state: BondStateHandle | object) -> np.ndarray | None:
        """Return mapped authored rows for one frozen-state batch."""
        handle = self._resolve_bond_state_handle(state)
        topologies = None if handle is None else self._bond_state_topologies_by_handle.get(handle)
        return None if topologies is None else topologies.copy()

    def _resolve_bond_seed_handle(self, seed: BondSeedHandle | object) -> BondSeedHandle | None:
        if isinstance(seed, BondSeedHandle):
            if seed._owner is not self:
                gs.raise_exception("QIPCCoupler: bond seed handle belongs to a different adhesion manager.")
            return seed
        handles = self._bond_seed_handles_by_entity.get(seed, ())
        if len(handles) > 1:
            names = ", ".join(repr(handle.name) for handle in handles)
            gs.raise_exception(
                "QIPCCoupler: bond seed query by FEM entity is ambiguous across batches "
                f"{names}; pass the BondSeedHandle returned by add_bond_seed_request."
            )
        return handles[0] if handles else None

    def _resolve_bond_state_handle(self, state: BondStateHandle | object) -> BondStateHandle | None:
        if isinstance(state, BondStateHandle):
            if state._owner is not self:
                gs.raise_exception("QIPCCoupler: frozen bond state handle belongs to a different adhesion manager.")
            return state
        handles = self._bond_state_handles_by_entity.get(state, ())
        if len(handles) > 1:
            names = ", ".join(repr(handle.name) for handle in handles)
            gs.raise_exception(
                "QIPCCoupler: frozen bond state query by FEM entity is ambiguous across batches "
                f"{names}; pass the BondStateHandle returned by add_bond_state_request."
            )
        return handles[0] if handles else None

    @staticmethod
    def _require_compatible_rest_height(
        first: float,
        second: float,
        *,
        first_name: str | None = None,
        second_name: str | None = None,
    ) -> None:
        if first == second:
            return
        batches = ""
        if first_name is not None and second_name is not None:
            batches = f" for batches '{first_name}' and '{second_name}'"
        gs.raise_exception(
            "QIPCCoupler: all distance-bond seeds must use one compatible rest_height"
            f"{batches}; QIPC seed_locks accepts one scalar for the aggregated slot transaction "
            f"(got {first} and {second})."
        )

    @staticmethod
    def _require_nonoverlapping_bond_topologies(
        batches: tuple[tuple[str, np.ndarray], ...],
        *,
        kind: str,
    ) -> None:
        seen: dict[tuple[int, int, int, int], str] = {}
        for name, topologies in batches:
            for row in topologies:
                key = (int(row[0]), *sorted(int(value) for value in row[1:]))
                previous = seen.get(key)
                if previous is not None:
                    gs.raise_exception(
                        "QIPCCoupler: authored distance-bond topology occurs more than once "
                        f"across {kind} batches '{previous}' and '{name}': {key}."
                    )
                seen[key] = name

    def _seed_bond_batch(self, topologies: np.ndarray, rest_height: float) -> None:
        bond_system = self._bond_system()
        topos_t = torch.as_tensor(topologies.reshape(-1), device="cuda")
        bond_system.seed_locks(topos_t, rest_height)

    def _restore_bond_state_batch(self, batch: dict[str, np.ndarray]) -> None:
        bond_system = self._bond_system()
        n_bonds = len(batch["topo"])

        def tensor(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.as_tensor(batch[name], dtype=dtype, device="cuda").reshape(-1).contiguous()

        bond_system.restore_slots(
            torch.ones(n_bonds, dtype=torch.int32, device="cuda"),
            tensor("topo", torch.int32),
            tensor("Dm_inv", torch.float64),
            tensor("V0", torch.float64),
            tensor("d_rest", torch.float64),
            tensor("kappa", torch.float64),
            tensor("release_force", torch.float64),
            tensor("release_strain", torch.float64),
            tensor("release_gap", torch.float64),
            tensor("release_slip", torch.float64),
            tensor("age", torch.int32),
            tensor("pair_key", torch.int64),
        )

    @staticmethod
    def _map_bond_seed_request(
        request: BondSeedRequest | BondStateRequest,
        *,
        fem_global_offset: int,
        fem_offset: int,
        n_fem_vertices: int,
        rigid_vertex_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        source = request.topologies
        if isinstance(request, BondSeedRequest):
            handle = request.handle
            source_fem_offset = handle.source_fem_global_offset
            strict_rigid_mapping = handle.strict_rigid_mapping
            batch_name = handle.name
        else:
            handle = request.handle
            source_fem_offset = handle.source_fem_global_offset
            strict_rigid_mapping = handle.strict_rigid_mapping
            batch_name = handle.name
        destination_fem_offset = fem_global_offset + fem_offset

        if source_fem_offset is None:
            if source.size and int(source.max()) >= n_fem_vertices:
                gs.raise_exception("QIPCCoupler: authored FEM-local bond topology exceeds the target FEM vertex count.")
            return np.ascontiguousarray(source + destination_fem_offset, dtype=np.int32), 0

        is_fem = source >= source_fem_offset
        fem_local = source - source_fem_offset
        if fem_local[is_fem].size and int(fem_local[is_fem].max()) >= n_fem_vertices:
            gs.raise_exception("QIPCCoupler: authored global bond topology exceeds the target FEM vertex count.")

        can_map_rigid = rigid_vertex_ids is not None and len(rigid_vertex_ids) == source_fem_offset
        if not can_map_rigid:
            keep = is_fem.all(axis=1)
            if strict_rigid_mapping and not keep.all():
                actual_count = "missing" if rigid_vertex_ids is None else str(len(rigid_vertex_ids))
                gs.raise_exception(
                    f"QIPCCoupler: authored bond batch '{batch_name}' requires exact rigid vertex mapping "
                    f"for {source_fem_offset} source vertices, but the target provides {actual_count}."
                )
            mapped = source[keep] - source_fem_offset + destination_fem_offset
            return np.ascontiguousarray(mapped, dtype=np.int32), int((~keep).sum())

        rigid_vertex_ids = np.asarray(rigid_vertex_ids, dtype=np.int64)
        mapped = np.empty_like(source, dtype=np.int64)
        mapped[is_fem] = fem_local[is_fem] + destination_fem_offset
        mapped[~is_fem] = rigid_vertex_ids[source[~is_fem]]
        return np.ascontiguousarray(mapped, dtype=np.int32), 0

    def dump_adhesion_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-pair adhesion beta state as (keys int64, betas float64) arrays."""
        system = self._adhesion_system()
        cap = int(system.adhesion_pair_cap)
        keys = torch.zeros(cap, dtype=torch.int64, device="cuda")
        betas = torch.zeros(cap, dtype=torch.float64, device="cuda")
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        system.dump_adhesion_pair_state(keys, betas, cnt)
        torch.cuda.synchronize()
        n = int(cnt.item())
        return keys[:n].cpu().numpy(), betas[:n].cpu().numpy()

    def load_adhesion_state(self, keys: np.ndarray, betas: np.ndarray) -> None:
        """Claim-and-set per-pair adhesion beta state.

        Only valid within the same built scene: keys hash GLOBAL vertex ids and
        do not transfer across different vertex layouts.
        """
        system = self._adhesion_system()
        keys_t = torch.as_tensor(np.asarray(keys, dtype=np.int64), device="cuda")
        betas_t = torch.as_tensor(np.asarray(betas, dtype=np.float64), device="cuda")
        system.load_adhesion_pair_state(keys_t, betas_t)
