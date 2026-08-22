"""Tape roll import for the QIPC coupler (design: docs/adhesion_tape_design.md, §6).

A tape roll asset is the output of cuda-graph-qipc's guided-winding stage
(``examples/adhesive_tape_wind.py --save roll.npz``): a wound shell strip plus
the ring-hub parameters. Importing it into Genesis means

- a Cloth FEM entity whose INITIAL positions are the wound coil and whose REST
  geometry is the flat strip (the stored membrane + bending strain is the
  prestress that makes the coil springy),
- an ABD ring-hub rigid entity,
- graded adhesive contact pairs (tape-tape self adhesion with ``beta0=1`` so the
  imported coil holds from frame 0, and a stronger tape-hub glue band).

The scene must use the adhesive contact constitution with distance bonds for
peel-force semantics; ``recommended_coupler_options(asset)`` returns the
matching ``QIPCCouplerOptions`` fields.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

import genesis as gs
import genesis.utils.geom as gu

from .coupler import QIPCCoupler


@dataclass
class TapeAsset:
    """Parsed tape-roll asset (cgq adhesive_tape_wind npz format)."""

    tape_positions: np.ndarray  # (n_verts, 3) wound coil, hub axis +z
    tape_tris: np.ndarray  # (n_tris, 3) int32
    nx: int
    nz: int
    thick: float
    tape_length: float
    width: float
    youngs: float
    poisson: float
    density: float
    bending_e: float
    hub_r_outer: float
    hub_r_inner: float
    hub_height: float
    d_hat: float
    params: dict
    # Wind-saved distance-lock topologies ((n, 4) vertex ids), present for
    # locked winds. bond_topos_space "global" means ids live in the WIND
    # scene's [ABD hub | FEM tape] global layout with bond_fem_gvo = its hub
    # vertex count; legacy dumps are FEM-local (space None, gvo 0).
    bond_topos: np.ndarray | None = None
    bond_topos_space: str | None = None
    bond_fem_gvo: int = 0

    @classmethod
    def from_npz(cls, path: str, *, allow_legacy_pickle: bool = True) -> TapeAsset:
        """Load a tape asset.

        New assets should store parameters as UTF-8 JSON in `params_json`.
        `allow_legacy_pickle` exists only for trusted winding outputs that still
        contain the historical object-array `params` field; untrusted component
        importers must disable it.
        """
        data = np.load(os.path.expanduser(path), allow_pickle=allow_legacy_pickle)
        required = (
            "tape_positions",
            "tape_tris",
            "nx",
            "nz",
            "thick",
            "tape_length",
            "width",
            "youngs",
            "poisson",
            "density",
            "bending_e",
            "hub_r_outer",
            "hub_r_inner",
            "hub_height",
            "d_hat",
        )
        missing = [key for key in required if key not in data]
        if missing:
            gs.raise_exception(f"TapeAsset.from_npz: missing keys {missing} in '{path}'.")
        params = {}
        if "params_json" in data:
            try:
                params = json.loads(bytes(data["params_json"]).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                gs.raise_exception(f"TapeAsset.from_npz: invalid params_json in '{path}': {error}.")
            if not isinstance(params, dict):
                gs.raise_exception(f"TapeAsset.from_npz: params_json must contain an object in '{path}'.")
        elif "params" in data:
            if not allow_legacy_pickle:
                gs.raise_exception(
                    f"TapeAsset.from_npz: legacy pickled params are disabled for untrusted asset '{path}'."
                )
            raw = data["params"]
            try:
                params = dict(raw.reshape(-1)[0])
            except (TypeError, ValueError):
                params = {}
        asset = cls(
            tape_positions=np.ascontiguousarray(data["tape_positions"], dtype=np.float64),
            tape_tris=np.ascontiguousarray(data["tape_tris"], dtype=np.int32),
            nx=int(data["nx"]),
            nz=int(data["nz"]),
            thick=float(data["thick"]),
            tape_length=float(data["tape_length"]),
            width=float(data["width"]),
            youngs=float(data["youngs"]),
            poisson=float(data["poisson"]),
            density=float(data["density"]),
            bending_e=float(data["bending_e"]),
            hub_r_outer=float(data["hub_r_outer"]),
            hub_r_inner=float(data["hub_r_inner"]),
            hub_height=float(data["hub_height"]),
            d_hat=float(data["d_hat"]),
            params=params,
            bond_topos=(
                np.ascontiguousarray(data["bond_topos"], dtype=np.int64).reshape(-1, 4)
                if "bond_topos" in data
                else None
            ),
            bond_topos_space=(str(data["bond_topos_space"]) if "bond_topos_space" in data else None),
            bond_fem_gvo=(int(data["bond_fem_gvo"]) if "bond_fem_gvo" in data else 0),
        )
        n_expected = (asset.nx + 1) * (asset.nz + 1)
        if asset.tape_positions.shape != (n_expected, 3):
            gs.raise_exception(
                f"TapeAsset.from_npz: tape_positions shape {asset.tape_positions.shape} does not match "
                f"(nx+1)*(nz+1)={n_expected} vertices."
            )
        return asset

    def flat_rest_positions(self) -> np.ndarray:
        """The flat (unrolled) strip with the SAME vertex layout as the coil.

        vid(i, j) = i * (nz + 1) + j, long axis +y, width along z (matches cgq's
        make_tangent_tape). Absolute placement is irrelevant: only the rest
        metric and the (flat -> zero) dihedral angles are consumed.
        """
        dy = self.tape_length / self.nx
        dz = self.width / self.nz
        verts = np.empty(((self.nx + 1) * (self.nz + 1), 3), dtype=np.float64)
        for i in range(self.nx + 1):
            for j in range(self.nz + 1):
                verts[i * (self.nz + 1) + j] = (0.0, i * dy, j * dz - 0.5 * self.width)
        return verts


class _TapeClusterAsset(Protocol):
    tape_positions: np.ndarray
    tape_tris: np.ndarray
    bond_topos: np.ndarray | None
    bond_topos_space: str | None
    bond_fem_gvo: int


def make_ring_hub(r_out: float, r_in: float, height: float, n_sides: int = 48):
    """Closed hollow-ring (annulus prism) trimesh, axis +z, centered.

    Port of cgq's make_ring_hub: 4 vertex blocks of n_sides (outer-front,
    outer-back, inner-front, inner-back) with outward-consistent normals so the
    ABD divergence-theorem mass/inertia are correct.
    """
    th = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)
    zf, zb = 0.5 * height, -0.5 * height
    verts = []
    for radius in (r_out, r_in):
        for z in (zf, zb):
            verts += [[radius * np.cos(t), radius * np.sin(t), z] for t in th]
    verts = np.array(verts, dtype=np.float64)
    n = n_sides
    of, ob, if_, ib = 0, n, 2 * n, 3 * n
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append([of + i, ob + i, ob + j])
        tris.append([of + i, ob + j, of + j])  # outer wall (+r)
        tris.append([if_ + i, ib + j, ib + i])
        tris.append([if_ + i, if_ + j, ib + j])  # inner wall (-r)
        tris.append([of + i, of + j, if_ + j])
        tris.append([of + i, if_ + j, if_ + i])  # front cap (+z)
        tris.append([ob + i, ib + j, ob + j])
        tris.append([ob + i, ib + i, ib + j])  # back cap (-z)
    return verts, np.array(tris, dtype=np.int32)


_SOLVER_CFG_TO_OPTION = {
    "newton/velocity_tol": ("solver_newton_velocity_tol", float),
    "newton/max_iter": ("solver_newton_max_iter", int),
    "linear_system/max_iter": ("solver_linear_max_iter", int),
    "linear_system/tol_rate": ("solver_linear_tol_rate", float),
    "linear_system/preconditioner": ("solver_linear_preconditioner", str),
    "linear_system/solver": ("solver_linear_solver", str),
    "line_search/max_iter": ("solver_line_search_max_iter", int),
    "contact/ccd_partition": ("contact_ccd_partition", lambda v: bool(int(v))),
}


def solver_cfg_to_options(solver_cfg: dict) -> dict:
    """Translate a qipc SOLVER_CFG dict onto the coupler's solver_* fields.

    Opt-in only: recommended_coupler_options deliberately does NOT apply the
    asset's wind-time solver configuration. Empirically the QIPC defaults
    solve the imported-roll scene in ~2 Newton iterations / 10 ms per step,
    while the wind's own values are pathological here — in particular
    `line_search/max_iter=16` (vs default 12) lets the line search accept
    ever-smaller steps until Newton stagnates at its iteration cap
    (~85 s per step on the same scene).
    """
    options: dict = {}
    for config_key, (field, cast) in _SOLVER_CFG_TO_OPTION.items():
        if config_key in solver_cfg:
            options[field] = cast(solver_cfg[config_key])
    return options


# Solver profile for imported tape rolls, measured on the lift/sway/release
# timeline (examples/qipc/tape_lift_drop.py) rather than taken from the wind's
# own SOLVER_CFG:
#   velocity_tol  qipc's default 0.05 makes Newton's absolute tolerance
#                 (tol*dt) 0.5mm per iteration, which a lock-stiffened coil's
#                 free-fall correction stays under -- the solve reports
#                 convergence with an airborne spool floating in place. 3.8e-3
#                 is what the lift/sway/release timeline needs to both land the
#                 released spool AND track the native reference (pull-end 0.1201
#                 vs 0.120, mid-sway 0.1269 vs 0.127) at the tol_rate below; the
#                 looser 0.01 lands too on a tight linear solve but leaves the
#                 spool floating once tol_rate is relaxed.
#   *_max_iter    cgq's tape values. The caps are hit on lock-heavy frames
#                 either way, and the lower linear cap is the cheaper place
#                 to stop.
#   tol_rate      3e-3 instead of qipc's 1e-4. Wherever PCG is tolerance-limited
#                 rather than cap-limited this halves its iterations for free
#                 (median 739 -> 458 on the dexhand scene). Paired with the
#                 velocity_tol above it costs the roll timeline nothing: 145s vs
#                 142s at (0.01, 1e-4), same landing, closer to the reference.
# Scenes wanting interactive rate over tape fidelity should loosen velocity_tol
# rather than tol_rate: on the bimanual dexhand scene 0.01 gives a 50ms median
# step where 3.8e-3 gives 228ms (its PCG goes back to running into the cap).
# line_search/max_iter is deliberately NOT set from the wind's 16: it stalls
# Newton at its cap on imported rolls (see solver_cfg_to_options).
_TAPE_SOLVER_PROFILE = {
    "solver_newton_velocity_tol": 3.8e-3,
    "solver_newton_max_iter": 300,
    "solver_linear_max_iter": 800,
    "solver_linear_tol_rate": 3e-3,
}


def recommended_coupler_options(asset: TapeAsset) -> dict:
    """QIPCCouplerOptions fields for importing this asset.

    Contact/adhesion values come from the asset's wind-time parameters; the
    solver fields are the measured tape profile (_TAPE_SOLVER_PROFILE), which
    is what makes released rolls actually fall instead of hovering. Override
    any of them by updating the returned dict.
    """
    params = asset.params
    # Respect the wind-time adhesion mode: LOCK=1 -> Phase-2 distance bonds
    # (the coil holds itself, peel force = RCC_RELEASE_FORCE); LOCK=0 -> pure
    # soft adhesion (the coil relies on beta-state stickiness alone and will
    # gradually unroll, matching cgq's --no-lock behavior).
    lock = bool(int(params.get("LOCK", 1)))
    options = dict(
        contact_enable=True,
        contact_d_hat=asset.d_hat,
        contact_friction=float(params.get("MU", 0.5)),
        contact_resistance=1e7,
        init_collision_pair_capacity=600000,
        adhesion_bond_distance_lock=lock,
        adhesion_bond_distance_lock_ratio=float(params.get("DISTANCE_LOCK_RATIO", 1.0)),
        adhesion_bond_max_bonds=16384 if lock else 0,
        adhesion_bond_kappa=float(params.get("RCC_KAPPA", 1e6)),
        adhesion_bond_release_force=float(params.get("RCC_RELEASE_FORCE", 0.5)),
        **_TAPE_SOLVER_PROFILE,
    )
    return options


def seed_asset_locks(scene, tape_entity, asset: TapeAsset) -> tuple[int, int]:
    """Return or create the wind-saved bond topology seed for an imported roll.

    ``add_tape_roll`` registers authored locks for automatic seeding during
    ``scene.build()``. Calling this after build remains supported for callers
    that used an older/manual import path; for an automatically seeded entity it
    returns the recorded ``(n_seeded, n_dropped_hub_rows)`` without duplicating
    bonds.

    Saved ids live in the wind scene's global layout ``[ABD hub | FEM tape]``.
    The automatic path rebases tape ids and maps hub ids through the imported
    hub entity's actual QIPC vertex range, so unrelated preceding rigid bodies
    do not invalidate the seed. It only accepts a hub with the authored vertex
    count; otherwise hub rows are dropped and the innermost turn re-bonds
    dynamically (`beta0=1`). The manual compatibility fallback below can only
    retain hub rows when the whole ABD layout still matches the wind scene.
    Returns `(n_seeded, n_dropped_hub_rows)`.
    """
    if asset.bond_topos is None:
        return 0, 0
    coupler = scene.sim.coupler
    adhesion = coupler._adhesion
    automatic_result = adhesion.get_bond_seed_result(tape_entity)
    if automatic_result is not None:
        return automatic_result
    entry = coupler._fem_entry(tape_entity)
    fem_gvo = adhesion.fem_global_vertex_offset()
    our_base = fem_gvo + entry.offset

    topos = asset.bond_topos.astype(np.int64, copy=True)
    dropped = 0
    if asset.bond_topos_space == "global":
        src_gvo = int(asset.bond_fem_gvo)
        n_affine_verts = int(coupler._scene.affine_body.n_verts)
        is_fem = topos >= src_gvo
        # Affine body dynamics (ABD) hub ids remain valid when the authored hub exactly occupies the
        # current affine-body prefix. FEM ids rebase across the full non-FEM prefix.
        if src_gvo == n_affine_verts and entry.offset == 0:
            topos = np.where(is_fem, topos - src_gvo + our_base, topos)
        else:
            keep = is_fem.all(axis=1)
            dropped = int((~keep).sum())
            if dropped:
                gs.logger.debug(
                    f"seed_asset_locks: dropping {dropped} hub-side lock rows (wind ABD layout has "
                    f"{src_gvo} vertices, this scene has {n_affine_verts}); those pairs re-bond dynamically."
                )
            topos = topos[keep] - src_gvo + our_base
    else:  # legacy dumps: FEM-local tape ids
        topos = topos + our_base
    if topos.size == 0:
        return 0, dropped
    n_global = fem_gvo + coupler._scene.finite_element.n_verts
    if topos.min() < 0 or topos.max() >= n_global:
        gs.raise_exception(
            f"seed_asset_locks: bond topologies index outside this scene's global vertex range "
            f"[0, {n_global}) (got [{topos.min()}, {topos.max()}])."
        )

    # Band-edge rest height, matching cgq's drop: xi(pair) + ratio * d_hat with
    # xi = 2 * thick (point + max triangle thickness).
    rest_height = _bond_rest_height(asset)
    adhesion.seed_bonds(topos, rest_height)
    return int(topos.shape[0]), dropped


def _bond_rest_height(asset: TapeAsset) -> float:
    ratio = float(asset.params.get("DISTANCE_LOCK_RATIO", 1.0))
    return 2.0 * asset.thick + ratio * asset.d_hat


def bond_cluster_member_triangles(asset: _TapeClusterAsset, collar: int) -> np.ndarray:
    """Select the bonded roll interior that may be represented by one cluster.

    The wind-authored distance bonds form the rigid certificate. The largest
    connected unbonded component is the payout front; smaller unbonded islands
    enclosed by the certificate are filled. ``collar`` graph rings adjacent to
    that front remain deformable, and a triangle joins only when all three of
    its vertices lie deeper than the collar.

    Returns entity-local triangle indices suitable for
    ``QIPCCoupler.add_affine_cluster(..., tris=...)``.
    """
    if isinstance(collar, bool) or not isinstance(collar, (int, np.integer)) or collar < 0:
        gs.raise_exception("bond_cluster_member_triangles: collar must be a non-negative integer.")
    triangles, bonded, adjacency = _bond_cluster_certificate(asset)
    member = _bond_cluster_target(triangles, bonded, ~bonded, adjacency, int(collar))
    return np.flatnonzero(member).astype(np.int32)


def _bond_cluster_inputs(asset: _TapeClusterAsset) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    if asset.bond_topos is None or asset.bond_topos.size == 0:
        gs.raise_exception("bond_cluster_member_triangles requires wind-authored distance bonds.")

    n_vertices = int(asset.tape_positions.shape[0])
    topologies = np.asarray(asset.bond_topos, dtype=np.int64).reshape(-1)
    if asset.bond_topos_space == "global":
        topologies = topologies[topologies >= asset.bond_fem_gvo] - asset.bond_fem_gvo
    tape_vertices = topologies[(topologies >= 0) & (topologies < n_vertices)]
    if tape_vertices.size == 0:
        gs.raise_exception("bond_cluster_member_triangles: authored bonds contain no tape vertices.")

    bonded = np.zeros(n_vertices, dtype=bool)
    bonded[np.unique(tape_vertices)] = True
    adjacency: list[set[int]] = [set() for _ in range(n_vertices)]
    triangles = np.asarray(asset.tape_tris, dtype=np.int64)
    for a, b, c in triangles:
        for left, right in ((a, b), (b, c), (c, a)):
            adjacency[int(left)].add(int(right))
            adjacency[int(right)].add(int(left))

    return triangles, bonded, adjacency


def _bond_cluster_certificate(
    asset: _TapeClusterAsset,
    never_member: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    triangles, bonded, adjacency = _bond_cluster_inputs(asset)
    if never_member is not None:
        # Mask BEFORE hole-fill, mirroring the packaged driver's certificate
        # (qipc.cluster_release_driver) so adopt_membership verification holds.
        bonded &= ~np.asarray(never_member, dtype=bool).reshape(-1)
    free = ~bonded
    n_vertices = len(bonded)
    component = np.full(n_vertices, -1, dtype=np.int32)
    component_sizes: list[int] = []
    for seed in np.flatnonzero(free):
        if component[seed] >= 0:
            continue
        component_id = len(component_sizes)
        frontier = [int(seed)]
        component[seed] = component_id
        size = 0
        while frontier:
            vertex = frontier.pop()
            size += 1
            for neighbor in adjacency[vertex]:
                if free[neighbor] and component[neighbor] < 0:
                    component[neighbor] = component_id
                    frontier.append(neighbor)
        component_sizes.append(size)
    if component_sizes:
        exterior_component = int(np.argmax(component_sizes))
        bonded |= free & (component != exterior_component)
    return triangles, bonded, adjacency


def _structured_bond_cluster_certificate(
    asset: _TapeClusterAsset,
    *,
    row_width: int,
    never_member: np.ndarray,
    free_run: int = 4,
    max_bonded_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Build one continuous bonded prefix for a row-major wound tape."""
    triangles, raw_bonded, adjacency = _bond_cluster_inputs(asset)
    n_vertices = len(raw_bonded)
    if row_width <= 0 or n_vertices % row_width:
        gs.raise_exception("structured tape row_width must be positive and divide the vertex count.")
    never = np.asarray(never_member, dtype=bool).reshape(-1)
    if len(never) != n_vertices:
        gs.raise_exception("structured tape never_member mask must match the tape vertex count.")
    # Mask BEFORE the row scan, mirroring the packaged driver's
    # freeze/adopt structured path (qipc.cluster_release_driver).
    raw_bonded = raw_bonded & ~never
    row_fraction = raw_bonded.reshape(-1, row_width).mean(axis=1)
    free_row = len(row_fraction)
    for row in range(len(row_fraction) - free_run + 1):
        if np.all(row_fraction[row : row + free_run] <= max_bonded_fraction):
            free_row = row
            break
    if free_row == len(row_fraction):
        never_rows = np.flatnonzero(never.reshape(-1, row_width).any(axis=1))
        if len(never_rows):
            free_row = int(never_rows.min())

    bonded = np.zeros(n_vertices, dtype=bool)
    bonded[: free_row * row_width] = True
    bonded &= ~never
    return triangles, bonded, adjacency


def _bond_cluster_target(
    triangles: np.ndarray,
    bonded: np.ndarray,
    freed: np.ndarray,
    adjacency: list[set[int]],
    collar: int,
) -> np.ndarray:
    distance = np.full(len(bonded), np.iinfo(np.int32).max, dtype=np.int64)
    frontier = deque(int(vertex) for vertex in np.flatnonzero(freed))
    distance[freed] = 0
    while frontier:
        vertex = frontier.popleft()
        next_distance = distance[vertex] + 1
        for neighbor in adjacency[vertex]:
            if distance[neighbor] > next_distance:
                distance[neighbor] = next_distance
                frontier.append(neighbor)

    deep = bonded & (distance >= collar)
    return deep[triangles].all(axis=1)


def _normalize_tape_cluster_collars(collar: int, activation_collar: int | None) -> tuple[int, int]:
    """Validate and normalize the runtime and initial-membership collars."""
    if isinstance(collar, (bool, np.bool_)) or not isinstance(collar, (int, np.integer)) or collar <= 0:
        gs.raise_exception("TapeBondClusterController: collar must be a positive integer.")
    collar = int(collar)

    if activation_collar is None:
        activation = collar
    else:
        if isinstance(activation_collar, (bool, np.bool_)) or not isinstance(activation_collar, (int, np.integer)):
            gs.raise_exception("TapeBondClusterController: activation_collar must be a positive integer.")
        activation = int(activation_collar)
        if activation <= 0:
            gs.raise_exception("TapeBondClusterController: activation_collar must be a positive integer.")

    if activation < collar:
        gs.raise_exception("TapeBondClusterController: activation_collar must be at least collar.")
    return collar, activation


class TapeBondClusterController:
    """Runtime peel policy for a queued distance-bond tape cluster.

    The affine cluster is a mechanics optimization, not a second fracture
    model. Wind-authored bonds certify the initial rigid interior; the release
    policy itself is qipc's packaged ``ClusterReleaseDriver``
    (``qipc.cluster_release_driver``): exact released-pair events with
    organic-relock cancellation, live-bond support deferral, connected
    monotone front advance, an optional structured per-column sloped front,
    and a euclidean/radial peel-motion gate. ``collar`` graph rings behind the
    front remain deformable. Membership can only shrink during an episode.

    Genesis queues the initial membership before ``scene.build`` so authored
    reset replay works; the driver therefore ADOPTS that membership at
    ``initialize``/``reset`` (``ClusterReleaseDriver.adopt_membership``)
    instead of joining it at activation the way the qipc examples do.
    """

    def __init__(
        self,
        coupler,
        cluster,
        tape_entity,
        asset: _TapeClusterAsset,
        *,
        collar: int,
        detach_displacement: float,
        bond_seed_handle=None,
        certificate: tuple[np.ndarray, np.ndarray, list[set[int]]] | None = None,
        activation_collar: int | None = None,
        detach_gate: str = "euclidean",
        radial_center: np.ndarray | tuple[float, float, float] | None = None,
        radial_axis: np.ndarray | tuple[float, float, float] | None = None,
        release_front_band: int | None = None,
        clear_bonds_on_detach: bool = True,
        structured_row_width: int | None = None,
        structured_max_front_slope: int | None = None,
        never_member: np.ndarray | None = None,
    ) -> None:
        collar, activation = _normalize_tape_cluster_collars(collar, activation_collar)
        if not np.isfinite(detach_displacement) or detach_displacement <= 0.0:
            gs.raise_exception("TapeBondClusterController: detach_displacement must be finite and positive.")
        if detach_gate not in ("euclidean", "radial"):
            gs.raise_exception("TapeBondClusterController: detach_gate must be 'euclidean' or 'radial'.")

        triangles, bonded, adjacency = (
            _bond_cluster_certificate(asset, never_member) if certificate is None else certificate
        )
        n_vertices = len(bonded)
        if never_member is None:
            never = np.zeros(n_vertices, dtype=bool)
        else:
            never = np.asarray(never_member, dtype=bool).reshape(-1)
            if len(never) != n_vertices:
                gs.raise_exception("TapeBondClusterController: never_member mask must match the tape vertex count.")
        initial_member = _bond_cluster_target(triangles, bonded, ~bonded, adjacency, activation)

        self._coupler = coupler
        self._cluster = cluster
        self._tape_entity = tape_entity
        self._bond_seed_handle = bond_seed_handle
        self._triangles = triangles
        self._never = never
        self._collar = collar
        self._activation_collar = activation
        self._detach_displacement = float(detach_displacement)
        self._detach_gate = str(detach_gate)
        self._radial_center = radial_center
        self._radial_axis = radial_axis
        self._release_front_band = release_front_band
        self._clear_bonds_on_detach = bool(clear_bonds_on_detach)
        self._structured_row_width = structured_row_width
        self._structured_max_front_slope = structured_max_front_slope
        self._initial_member = initial_member
        self._initialized = False
        self._driver = None

    @property
    def driver(self):
        """The bound ``qipc.cluster_release_driver.ClusterReleaseDriver`` (post-initialize)."""
        return self._driver

    @property
    def initial_member_count(self) -> int:
        return int(self._initial_member.sum())

    @property
    def member_count(self) -> int:
        if self._driver is None:
            return int(self._initial_member.sum())
        return int(self._driver.member.sum())

    @property
    def released_total(self) -> int:
        return 0 if self._driver is None else int(self._driver.released_total)

    @property
    def melted_total(self) -> int:
        return 0 if self._driver is None else int(self._driver.melted_total)

    @property
    def cleared_bonds_total(self) -> int:
        return 0 if self._driver is None else int(self._driver.cleared_bonds_total)

    def initialize(self) -> None:
        """Bind runtime QIPC rows after ``scene.build()``."""
        vertex_range = self._cluster.fem_vertex_range
        if len(vertex_range) != len(self._never):
            gs.raise_exception("TapeBondClusterController: queued cluster vertex range does not match the tape asset.")
        self._initialized = True
        self.reset()

    def reset(self) -> None:
        """Rebuild the driver against the replayed authored membership."""
        self._require_initialized()
        driver = self._build_driver()
        adopted = driver.adopt_membership(
            self._initial_member,
            structured_row_width=self._structured_row_width,
            structured_max_front_slope=self._structured_max_front_slope,
        )
        member_count = self._cluster.member_count
        if member_count != adopted:
            gs.raise_exception(
                "TapeBondClusterController: QIPC membership replay produced "
                f"{member_count} elements, expected {adopted}."
            )
        self._driver = driver

    def before_step(self) -> int:
        """Advance the monotone peel front before the next QIPC step."""
        self._require_initialized()
        return self._driver.step()

    def _build_driver(self):
        from qipc.cluster_release_driver import ClusterReleaseDriver

        scene = self._coupler._scene
        collection = self._cluster.qipc_collection
        vertex_range = self._cluster.fem_vertex_range
        return ClusterReleaseDriver(
            scene,
            collection,
            np.asarray(self._triangles, dtype=np.int32),
            n_verts=len(self._never),
            collar=self._collar,
            activation_collar=self._activation_collar,
            never_member=self._never,
            detach_disp=self._detach_displacement,
            detach_gate=self._detach_gate,
            radial_center=self._radial_center,
            radial_axis=self._radial_axis,
            release_front_band=self._release_front_band,
            clear_bonds_on_detach=self._clear_bonds_on_detach,
            fem_vertex_offset=vertex_range.start,
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            gs.raise_exception("TapeBondClusterController.initialize must be called after scene.build().")


def add_tape_bond_cluster(
    scene,
    tape_entity,
    asset: _TapeClusterAsset,
    *,
    kappa: float,
    collar: int,
    detach_displacement: float,
    proxy_entity=None,
    proxy_link=None,
    structured_row_width: int | None = None,
    never_member: np.ndarray | None = None,
    bond_seed_handle=None,
    activation_collar: int | None = None,
    detach_gate: str = "euclidean",
    radial_center: np.ndarray | tuple[float, float, float] | None = None,
    radial_axis: np.ndarray | tuple[float, float, float] | None = None,
    release_front_band: int | None = None,
    clear_bonds_on_detach: bool = True,
    structured_max_front_slope: int | None = None,
) -> TapeBondClusterController:
    """Queue a releasable tape cluster and return its runtime peel policy."""
    collar, activation = _normalize_tape_cluster_collars(collar, activation_collar)
    coupler = scene.sim.coupler
    if coupler._options.adhesion_bond_lock_floor_ratio <= 0.0:
        gs.raise_exception(
            "add_tape_bond_cluster requires adhesion_bond_lock_floor_ratio > 0 "
            "so cleared near-barrier bonds cannot immediately re-lock."
        )
    certificate = None
    if structured_row_width is not None:
        if never_member is None:
            gs.raise_exception("structured tape clusters require a never_member mask.")
        certificate = _structured_bond_cluster_certificate(
            asset,
            row_width=structured_row_width,
            never_member=never_member,
        )
        triangles, bonded, adjacency = certificate
        initial_member = _bond_cluster_target(triangles, bonded, ~bonded, adjacency, activation)
        member_triangles = np.flatnonzero(initial_member).astype(np.int32)
    else:
        certificate = _bond_cluster_certificate(asset, never_member)
        triangles, bonded, adjacency = certificate
        initial_member = _bond_cluster_target(triangles, bonded, ~bonded, adjacency, activation)
        member_triangles = np.flatnonzero(initial_member).astype(np.int32)
    cluster = coupler.add_affine_cluster(
        tape_entity,
        proxy_entity=proxy_entity,
        proxy_link=proxy_link,
        kappa=kappa,
        initial_tris=member_triangles,
    )
    if bond_seed_handle is None:
        bond_seed_handle = coupler.adhesion.get_bond_seed_handle(tape_entity, name="internal")
    return TapeBondClusterController(
        coupler,
        cluster,
        tape_entity,
        asset,
        collar=collar,
        detach_displacement=detach_displacement,
        bond_seed_handle=bond_seed_handle,
        certificate=certificate,
        activation_collar=activation,
        detach_gate=detach_gate,
        radial_center=radial_center,
        radial_axis=radial_axis,
        release_front_band=release_front_band,
        clear_bonds_on_detach=clear_bonds_on_detach,
        structured_row_width=structured_row_width,
        structured_max_front_slope=structured_max_front_slope,
        never_member=never_member,
    )


def _write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as fh:
        fh.writelines(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n" for v in verts)
        fh.writelines(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n" for f in faces)


def _kabsch(source: np.ndarray, target: np.ndarray):
    """Best-fit rigid transform (R, t) mapping source -> target (same order)."""
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    H = (source - cs).T @ (target - ct)
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    R = Vt.T @ D @ U.T
    return R, ct - R @ cs


def _verify_same_vertex_order(source: np.ndarray, actual: np.ndarray) -> None:
    """Assert the entity kept the source's vertex ORDER (any rigid placement).

    Mesh loading may place the mesh with its own transform conventions, but the
    rest-geometry mapping only needs the vertex order to survive. Fit the rigid
    transform assuming identical order; a tiny residual proves the order (and
    that no welding/splitting happened).
    """
    if source.shape != actual.shape:
        gs.raise_exception(
            f"Tape import: entity has {actual.shape[0]} vertices but the source has "
            f"{source.shape[0]} (mesh processing must not merge/split vertices)."
        )
    R, t = _kabsch(source, actual)
    residual = float(np.abs(source @ R.T + t - actual).max())
    diag = float(np.linalg.norm(source.max(axis=0) - source.min(axis=0)))
    if residual > max(1e-6, 1e-4 * diag):
        gs.raise_exception(
            f"Tape import: the entity's vertices do not match the source order "
            f"(rigid-fit residual {residual:.2e}); mesh processing reordered or deformed them."
        )


def add_tape_roll(
    scene,
    asset: TapeAsset,
    *,
    pos=(0.0, 0.0, 0.0),
    euler=(90.0, 0.0, 0.0),
    with_hub: bool = True,
    hub_fixed: bool = False,
    friction: float | None = None,
    tape_tape_adhesion: dict | None = None,
    tape_hub_adhesion: dict | None = None,
    tape_surface=None,
    hub_surface=None,
):
    """Add a wound tape roll (cloth strip + optional ring hub) to a Genesis scene.

    Call BEFORE scene.build(). The default ``euler=(90, 0, 0)`` stands the roll
    up (asset hub axis +z -> world +y), matching cgq's tape-drop loader. The
    caller should configure the coupler with ``recommended_coupler_options``
    (adhesive constitution + distance bonds + the asset's d_hat).

    Placement is applied analytically (``R(euler) @ v + pos``) by baking the
    transform into the meshes and adding both entities with an identity morph
    pose. Passing pos/euler through the morph instead would misalign the coil
    and the hub: the FEM mesh loader pivots the rotation about the vertex COM
    (fem_entity.py) while the rigid loader pivots about the file origin -- for
    this asset that is a ~7mm relative offset, which embeds the hub in the coil
    wall (the coil's free tail shifts its COM away from the hub axis).

    Returns (tape_entity, hub_entity_or_None).
    """
    coupler = scene.sim.coupler
    if not isinstance(coupler, QIPCCoupler):
        gs.raise_exception("add_tape_roll requires the QIPC coupler (QIPCCouplerOptions).")

    R_place = gu.quat_to_R(gu.xyz_to_quat(np.asarray(euler, dtype=np.float64), degrees=True))
    t_place = np.asarray(pos, dtype=np.float64)

    # Adhesion/friction values follow the asset's wind-time params (env > preset
    # resolution already happened at wind time); explicit overrides win.
    params = asset.params
    if friction is None:
        friction = float(params.get("MU", 0.5))
    cn = float(params.get("CN", 1.0))
    asset_adhesion = {
        "Cn": cn,
        "Ct": float(params.get("CT", cn)),
        "W": float(params.get("ADH_W", 1.0)),
        "eta": float(params.get("ETA", 100.0)),
        "bonding_rate": float(params.get("BONDING_RATE", 1.0)),
        "beta0": 1.0,  # imported coil holds from frame 0 (re-bond instead of state transfer)
        "friction": friction,
    }
    tmp_dir = tempfile.mkdtemp(prefix="qipc_tape_")
    tape_obj = os.path.join(tmp_dir, "tape_roll.obj")
    tape_world = asset.tape_positions @ R_place.T + t_place
    _write_obj(tape_obj, tape_world, asset.tape_tris)

    tape = scene.add_entity(
        morph=gs.morphs.Mesh(file=tape_obj, scale=1.0),
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
        ),
        surface=tape_surface,
    )

    # The flat rest strip maps 1:1 by vertex ORDER; verify the loader kept it
    # (its placement conventions do not matter — rest is frame-invariant).
    from genesis.utils.misc import tensor_to_array

    actual = tensor_to_array(tape.init_positions).astype(np.float64)
    _verify_same_vertex_order(tape_world, actual)
    coupler.set_fem_rest_positions(tape, asset.flat_rest_positions())

    hub = None
    hub_vertex_count = 0
    if with_hub:
        hub_obj = os.path.join(tmp_dir, "tape_hub.obj")
        hub_verts, hub_tris = make_ring_hub(asset.hub_r_outer, asset.hub_r_inner, asset.hub_height)
        hub_vertex_count = len(hub_verts)
        # Unlike the tape, the hub keeps morph-level placement: the ring is
        # COM-centered at the file origin by construction, so the rigid
        # loader's pivot (file origin unaligned, COM frame under align=True
        # auto-reframing for free bodies) coincides with the analytic
        # R(euler) @ v + pos either way. Baking world coordinates instead
        # would CANCEL under align=True (morph pos places the COM frame).
        _write_obj(hub_obj, hub_verts, hub_tris)
        hub = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=hub_obj,
                pos=tuple(pos),
                euler=tuple(euler),
                scale=1.0,
                fixed=hub_fixed,
                # Keep the ring EXACTLY as authored. Genesis's default rigid
                # processing targets its own collision system: convexification
                # decomposes the annulus into 8 hulls that fill the bore (160
                # verts, mass +43%), and the vertex count/order change breaks
                # the wind-time vertex ids that seed_asset_locks transfers.
                # QIPC takes arbitrary triangle meshes, so pass the ring
                # through: 192 verts in source order, geometrically exact
                # mass. (The ring is already watertight, so the wrap that
                # watertighten=None disables is a no-op here -- it is set to
                # make "do not touch this mesh" explicit.)
                convexify=False,
                watertighten=None,
            ),
            material=gs.materials.Rigid(
                rho=1000.0,
                coup_friction=friction,
            ),
            surface=hub_surface,
        )

    # cgq's wind/drop use ONE adhesion row for tape-tape and tape-hub alike; the
    # rod-wind demos grade them (hub 10x, mandrel 100x) -- do that via overrides.
    tape_tape = dict(asset_adhesion)
    tape_tape.update(tape_tape_adhesion or {})
    coupler.add_adhesion(tape, None, **tape_tape)

    if hub is not None:
        tape_hub = dict(asset_adhesion)
        tape_hub.update(tape_hub_adhesion or {})
        coupler.add_adhesion(tape, hub, **tape_hub)

    lock_asset = bool(int(params.get("LOCK", 1)))
    if asset.bond_topos is not None and lock_asset and coupler.adhesion.bonds_enabled():
        source_fem_global_offset = asset.bond_fem_gvo if asset.bond_topos_space == "global" else None
        rigid_seed_entity = None
        if source_fem_global_offset is not None and hub is not None:
            if hub_vertex_count == source_fem_global_offset:
                rigid_seed_entity = hub
            else:
                gs.logger.warning(
                    f"Tape asset records {source_fem_global_offset} rigid-side vertices, but the imported "
                    f"hub has {hub_vertex_count}; hub-side seed rows will be dropped."
                )
        coupler.adhesion.add_bond_seed_request(
            tape,
            topologies=asset.bond_topos,
            source_fem_global_offset=source_fem_global_offset,
            rest_height=_bond_rest_height(asset),
            rigid_entity=rigid_seed_entity,
            name="internal",
        )

    return tape, hub
