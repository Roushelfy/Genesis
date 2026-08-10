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

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import geometric_mean, harmonic_mean

if TYPE_CHECKING:
    from genesis.options.solvers import QIPCCouplerOptions


def _require_current_qipc_api() -> None:
    from qipc import Scene as QIPCScene
    from qipc import contact

    missing = [name for name in ("Adhesion", "Bond", "Release") if not hasattr(contact, name)]
    if not hasattr(QIPCScene, "reset"):
        missing.append("Scene.reset")
    if missing:
        gs.raise_exception(
            "QIPCCoupler requires a current cuda-graph-qipc build with the grouped "
            f"contact API and reset support (missing: {', '.join(missing)}). Install "
            "the Genesis 'qipc' extra in its supported Python 3.12 environment; "
            "legacy flat-contact QIPC builds are not supported."
        )


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
    enabled: bool
    friction: float | None
    resistance: float | None
    distance_lock: bool | None
    distance_lock_ratio: float | None
    release_force: float | None


class BondSeedRequest(NamedTuple):
    """Authored distance bonds resolved into this scene after QIPC init."""

    fem_entity: object
    rigid_entity: object | None
    topologies: np.ndarray
    source_fem_global_offset: int | None
    rest_height: float


class QIPCAdhesionManager:
    """Owns adhesion declarations and runtime adhesion/bond state access.

    The coupler wires it into three build hooks (request queue -> tabular rows,
    constitution selection) and exposes it as ``coupler.adhesion``.
    """

    def __init__(self, options: "QIPCCouplerOptions") -> None:
        self._options = options
        self._requests: list[AdhesionRequest] = []
        self._bond_seed_requests: list[BondSeedRequest] = []
        self._bond_seed_batches: list[tuple[np.ndarray, float]] = []
        self._bond_seed_results: dict[object, tuple[int, int]] = {}
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
        enabled: bool = True,
        friction: float | None = None,
        resistance: float | None = None,
        distance_lock: bool | None = None,
        distance_lock_ratio: float | None = None,
        release_force: float | None = None,
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
        if distance_lock_ratio is not None and distance_lock_ratio <= 0:
            gs.raise_exception("QIPCCoupler.add_adhesion: distance_lock_ratio must be positive.")
        if release_force is not None and release_force < 0:
            gs.raise_exception("QIPCCoupler.add_adhesion: release_force must be non-negative.")

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
                enabled=bool(enabled),
                friction=friction,
                resistance=resistance,
                distance_lock=distance_lock,
                distance_lock_ratio=distance_lock_ratio,
                release_force=release_force,
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
    ) -> None:
        """Queue authored bond topologies for post-init global-id resolution."""
        if self._scene is not None:
            gs.raise_exception("QIPCCoupler.add_bond_seed_request must be called before scene.build().")
        if any(request.fem_entity is fem_entity for request in self._bond_seed_requests):
            gs.raise_exception("QIPCCoupler: an authored bond seed is already registered for this FEM entity.")

        topos = np.ascontiguousarray(topologies, dtype=np.int32).reshape(-1, 4)
        if topos.size and int(topos.min()) < 0:
            gs.raise_exception("QIPCCoupler: authored bond topology vertex ids must be non-negative.")
        if source_fem_global_offset is not None and source_fem_global_offset < 0:
            gs.raise_exception("QIPCCoupler: source_fem_global_offset must be non-negative.")
        if not np.isfinite(rest_height) or rest_height <= 0.0:
            gs.raise_exception("QIPCCoupler: authored bond rest_height must be finite and positive.")

        self._bond_seed_requests.append(
            BondSeedRequest(
                fem_entity=fem_entity,
                rigid_entity=rigid_entity,
                topologies=topos.copy(),
                source_fem_global_offset=source_fem_global_offset,
                rest_height=float(rest_height),
            )
        )

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
        return bool(self._bond_seed_requests)

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

        if self.bonds_enabled():
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
                    )
                    if req.enabled
                    else None
                )
                bond = None
                if self.bonds_enabled() and req.distance_lock is not False:
                    bond = self._make_bond(
                        ratio=req.distance_lock_ratio,
                        release_force=req.release_force,
                    )
                tab.insert(
                    elem_s,
                    elem_t,
                    friction_rate=float(friction),
                    resistance=float(resistance),
                    adhesion=adhesion,
                    bond=bond,
                )

    def _make_bond(self, *, ratio: float | None = None, release_force: float | None = None):
        from qipc.contact import Bond, Release

        opt = self._options
        return Bond(
            kappa=opt.adhesion_bond_kappa,
            ratio=opt.adhesion_bond_distance_lock_ratio if ratio is None else float(ratio),
            margin=opt.adhesion_bond_lock_margin,
            occlusion=opt.adhesion_occlusion,
            release=Release(
                strain=opt.adhesion_bond_release_strain,
                force=opt.adhesion_bond_release_force if release_force is None else float(release_force),
                gap=opt.adhesion_bond_release_gap,
                slip=opt.adhesion_bond_release_slip,
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
        bond_system = getattr(scene, "_bond_system", None)
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
        """Global vertex id at which the FEM block starts (== number of ABD verts)."""
        return int(getattr(self._require_scene().affine_body, "n_verts", 0) or 0)

    def get_bond_topos(self) -> np.ndarray:
        """Alive distance-bond topologies as (n, 4) GLOBAL vertex ids.

        FEM vertex global id = fem_global_vertex_offset() + geometry
        fem_vert_offset + local index; ids below the offset are ABD vertices.
        """
        bond_system = self._bond_system()
        out = torch.zeros(bond_system.max_bonds * 4, dtype=torch.int32, device="cuda")
        cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
        bond_system.dump_lock_topos(out, cnt)
        torch.cuda.synchronize()
        n = int(cnt.item())
        return out[: n * 4].reshape(n, 4).cpu().numpy()

    def get_bond_count(self) -> int:
        return int(self.get_bond_topos().shape[0])

    def seed_bonds(self, topos: np.ndarray, rest_height: float) -> None:
        """Seed distance bonds from (n, 4) GLOBAL vertex-id topologies.

        Call after `scene.build()` and before the first step. QIPC stamps each
        seed's stiffness and release thresholds from its resolved per-pair
        `Bond` model. The batch is replayed after subsequent scene resets.
        """
        topologies = np.ascontiguousarray(topos, dtype=np.int32).reshape(-1, 4)
        if topologies.size == 0:
            return
        if int(topologies.min()) < 0:
            gs.raise_exception("QIPCCoupler.seed_bonds: topology vertex ids must be non-negative.")
        if not np.isfinite(rest_height) or rest_height <= 0.0:
            gs.raise_exception("QIPCCoupler.seed_bonds: rest_height must be finite and positive.")

        self._seed_bond_batch(topologies, float(rest_height))
        self._bond_seed_batches.append((topologies.copy(), float(rest_height)))

    def apply_bond_seed_requests(
        self,
        fem_layout: dict[object, tuple[int, int]],
        rigid_vertex_ids: dict[object, np.ndarray],
    ) -> None:
        """Resolve and seed all authored batches after QIPC assigns global ids."""
        if not self._bond_seed_requests:
            return

        fem_global_offset = self.fem_global_vertex_offset()
        for request in self._bond_seed_requests:
            layout = fem_layout.get(request.fem_entity)
            if layout is None:
                gs.raise_exception("QIPCCoupler: authored bond seed FEM entity is not coupled in this scene.")
            fem_offset, n_fem_vertices = layout
            mapped, n_dropped = self._map_bond_seed_request(
                request,
                fem_global_offset=fem_global_offset,
                fem_offset=fem_offset,
                n_fem_vertices=n_fem_vertices,
                rigid_vertex_ids=rigid_vertex_ids.get(request.rigid_entity),
            )
            self.seed_bonds(mapped, request.rest_height)
            result = (len(mapped), n_dropped)
            self._bond_seed_results[request.fem_entity] = result
            gs.logger.info(
                f"QIPCCoupler: seeded {result[0]} authored distance bonds "
                f"({result[1]} rows dropped because their rigid-side ids could not be mapped)."
            )

    def restore_seeded_bonds(self) -> None:
        """Replay authored/manual frame-zero bond batches after `Scene.reset`."""
        for topologies, rest_height in self._bond_seed_batches:
            self._seed_bond_batch(topologies, rest_height)

    def get_bond_seed_result(self, fem_entity) -> tuple[int, int] | None:
        """Return `(seeded, dropped_rigid_rows)` for an automatic asset seed."""
        return self._bond_seed_results.get(fem_entity)

    def _seed_bond_batch(self, topologies: np.ndarray, rest_height: float) -> None:
        bond_system = self._bond_system()
        topos_t = torch.as_tensor(topologies.reshape(-1), device="cuda")
        bond_system.seed_locks(topos_t, rest_height)

    @staticmethod
    def _map_bond_seed_request(
        request: BondSeedRequest,
        *,
        fem_global_offset: int,
        fem_offset: int,
        n_fem_vertices: int,
        rigid_vertex_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, int]:
        source = request.topologies
        source_fem_offset = request.source_fem_global_offset
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
