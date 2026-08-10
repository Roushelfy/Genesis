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

import os
import tempfile
from dataclasses import dataclass

import numpy as np

import genesis as gs
import genesis.utils.geom as gu


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
    def from_npz(cls, path: str) -> "TapeAsset":
        data = np.load(os.path.expanduser(path), allow_pickle=True)
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
        if "params" in data:
            raw = data["params"]
            try:
                params = dict(raw.reshape(-1)[0])
            except Exception:
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
_TAPE_SOLVER_PROFILE = dict(
    solver_newton_velocity_tol=3.8e-3,
    solver_newton_max_iter=300,
    solver_linear_max_iter=800,
    solver_linear_tol_rate=3e-3,
)


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
        is_fem = topos >= src_gvo
        # Hub (ABD) ids pass through only if the ABD range is identical: same
        # vertex count AND the hub sitting at ABD offset 0, i.e. this scene's
        # FEM global offset equals the wind's.
        if src_gvo == fem_gvo and entry.offset == 0:
            topos = np.where(is_fem, topos - src_gvo + our_base, topos)
        else:
            keep = is_fem.all(axis=1)
            dropped = int((~keep).sum())
            if dropped:
                gs.logger.debug(
                    f"seed_asset_locks: dropping {dropped} hub-side lock rows (wind ABD layout has "
                    f"{src_gvo} vertices, this scene has {fem_gvo}); those pairs re-bond dynamically."
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


def _write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w") as fh:
        for v in verts:
            fh.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for f in faces:
            fh.write(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}\n")


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
    if not hasattr(coupler, "add_adhesion"):
        gs.raise_exception("add_tape_roll requires the QIPC coupler (QIPCCouplerOptions).")

    R_place = gu.quat_to_R(gu.xyz_to_quat(np.asarray(euler, dtype=np.float64), degrees=True))
    t_place = np.asarray(pos, dtype=np.float64)

    # Adhesion/friction values follow the asset's wind-time params (env > preset
    # resolution already happened at wind time); explicit overrides win.
    params = asset.params
    if friction is None:
        friction = float(params.get("MU", 0.5))
    cn = float(params.get("CN", 1.0))
    asset_adhesion = dict(
        Cn=cn,
        Ct=float(params.get("CT", cn)),
        W=float(params.get("ADH_W", 1.0)),
        eta=float(params.get("ETA", 100.0)),
        bonding_rate=float(params.get("BONDING_RATE", 1.0)),
        beta0=1.0,  # imported coil holds from frame 0 (re-bond instead of state transfer)
        friction=friction,
    )

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
            material=gs.materials.Rigid(rho=1000.0, coup_friction=friction),
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
        )

    return tape, hub
