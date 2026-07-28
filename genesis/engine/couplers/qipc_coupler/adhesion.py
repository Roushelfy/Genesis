"""Adhesion support for the QIPC coupler (design doc: docs/adhesion_tape_design.md).

Wraps QIPC's two-layer adhesion stack:

- Phase-1 soft adhesion: per-pair beta-state RCC potential, configured through
  the contact tabular columns (Cn/Ct/W/eta/bonding_rate/p0/beta0/adhesion_enabled).
- Phase-2 distance bonds: locked virtual-tet bonds (AdhesiveIPCContact bond_*
  scene config + per-pair distance_lock/distance_lock_ratio/release_force).

Everything is declarative-before-build: QIPC reads the contact tables exactly
once at scene init. The only runtime-mutable state is the per-pair beta table
(dump/load) and one-shot bond seeding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import torch

import genesis as gs
from genesis.utils.misc import geometric_mean, harmonic_mean

if TYPE_CHECKING:
    from genesis.options.solvers import QIPCCouplerOptions


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


class QIPCAdhesionManager:
    """Owns adhesion declarations and runtime adhesion/bond state access.

    The coupler wires it into three build hooks (request queue -> tabular rows,
    constitution selection) and exposes it as ``coupler.adhesion``.
    """

    def __init__(self, options: "QIPCCouplerOptions") -> None:
        self._options = options
        self._requests: list[AdhesionRequest] = []
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
            (eta, "eta"),
            (bonding_rate, "bonding_rate"),
            (p0, "p0"),
        ):
            if value < 0:
                gs.raise_exception(f"QIPCCoupler.add_adhesion: {name} must be non-negative.")
        if not (0.0 <= beta0 <= 1.0):
            gs.raise_exception("QIPCCoupler.add_adhesion: beta0 must be in [0, 1].")

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
                gs.raise_exception(
                    "QIPCCoupler: add_adhesion() was called but contact_constitution='consistent'."
                )
            return

        if opt.adhesion_bond_distance_lock and opt.adhesion_bond_max_bonds <= 0:
            gs.raise_exception(
                "QIPCCoupler: adhesion_bond_distance_lock=True requires adhesion_bond_max_bonds > 0 "
                "(QIPC silently keeps the bond system inert otherwise)."
            )
        if self.bonds_enabled() and not has_fem_entities:
            gs.raise_exception(
                "QIPCCoupler: distance bonds require at least one FEM entity in the scene "
                "(QIPC's BondSystem needs FEM vertices; pure rigid-rigid bonding is unsupported)."
            )

        from qipc.constitution import AdhesiveIPCContact

        AdhesiveIPCContact().apply_to(
            scene,
            bond_distance_lock=opt.adhesion_bond_distance_lock,
            bond_distance_lock_ratio=opt.adhesion_bond_distance_lock_ratio,
            bond_max_bonds=opt.adhesion_bond_max_bonds,
            bond_kappa=opt.adhesion_bond_kappa,
            bond_lock_margin=opt.adhesion_bond_lock_margin,
            bond_release_strain=opt.adhesion_bond_release_strain,
            bond_release_force=opt.adhesion_bond_release_force,
            bond_release_gap=opt.adhesion_bond_release_gap,
            bond_release_slip=opt.adhesion_bond_release_slip,
            occlusion=opt.adhesion_occlusion,
        )
        gs.logger.debug(
            "QIPCCoupler: AdhesiveIPCContact applied (bonds " + ("on" if self.bonds_enabled() else "off") + ")."
        )

    def insert_tabular_rows(self, tab, elem_by_entity: dict) -> None:
        """Write adhesion request rows into the contact tabular.

        Must run AFTER the coupler's plain friction/resistance pair rows
        (insert is an upsert, so these rows override).
        """
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
                tab.insert(
                    elem_s,
                    elem_t,
                    friction_rate=float(friction),
                    resistance=float(resistance),
                    Cn=req.Cn,
                    Ct=req.Ct,
                    W=req.W,
                    eta=req.eta,
                    bonding_rate=req.bonding_rate,
                    p0=req.p0,
                    beta0=req.beta0,
                    adhesion_enabled=req.enabled,
                    distance_lock=True if req.distance_lock is None else bool(req.distance_lock),
                    distance_lock_ratio=(
                        -1.0 if req.distance_lock_ratio is None else float(req.distance_lock_ratio)
                    ),
                    release_force=-1.0 if req.release_force is None else float(req.release_force),
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

        Call once after scene.build() and before the first step (QIPC's
        seed_locks contract). Seeded bonds use the scene-global
        kappa/release_force.
        """
        bond_system = self._bond_system()
        topos_t = torch.as_tensor(np.asarray(topos, dtype=np.int32).reshape(-1), device="cuda")
        bond_system.seed_locks(topos_t, float(rest_height))

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
