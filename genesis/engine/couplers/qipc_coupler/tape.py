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

    @classmethod
    def from_npz(cls, path: str) -> "TapeAsset":
        data = np.load(os.path.expanduser(path), allow_pickle=True)
        required = (
            "tape_positions", "tape_tris", "nx", "nz", "thick", "tape_length", "width",
            "youngs", "poisson", "density", "bending_e",
            "hub_r_outer", "hub_r_inner", "hub_height", "d_hat",
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


def recommended_coupler_options(asset: TapeAsset) -> dict:
    """QIPCCouplerOptions fields matching the asset's wind-time configuration.

    Includes the wind's solver configuration (SOLVER_CFG baked into the npz)
    translated onto the coupler's solver_* passthrough fields.
    """
    params = asset.params
    options = dict(
        contact_enable=True,
        contact_d_hat=asset.d_hat,
        contact_friction=float(params.get("MU", 0.5)),
        contact_resistance=1e7,
        init_collision_pair_capacity=600000,
        adhesion_bond_distance_lock=True,
        adhesion_bond_distance_lock_ratio=float(params.get("DISTANCE_LOCK_RATIO", 1.0)),
        adhesion_bond_max_bonds=16384,
        adhesion_bond_kappa=float(params.get("RCC_KAPPA", 1e6)),
        adhesion_bond_release_force=float(params.get("RCC_RELEASE_FORCE", 0.5)),
    )
    solver_cfg = params.get("SOLVER_CFG") or {}
    for config_key, (field, cast) in _SOLVER_CFG_TO_OPTION.items():
        if config_key in solver_cfg:
            options[field] = cast(solver_cfg[config_key])
    return options


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

    Returns (tape_entity, hub_entity_or_None).
    """
    coupler = scene.sim.coupler
    if not hasattr(coupler, "add_adhesion"):
        gs.raise_exception("add_tape_roll requires the QIPC coupler (QIPCCouplerOptions).")

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
    _write_obj(tape_obj, asset.tape_positions, asset.tape_tris)

    tape = scene.add_entity(
        morph=gs.morphs.Mesh(file=tape_obj, pos=tuple(pos), euler=tuple(euler), scale=1.0),
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
    _verify_same_vertex_order(asset.tape_positions, actual)
    coupler.set_fem_rest_positions(tape, asset.flat_rest_positions())

    hub = None
    if with_hub:
        hub_obj = os.path.join(tmp_dir, "tape_hub.obj")
        hub_verts, hub_tris = make_ring_hub(asset.hub_r_outer, asset.hub_r_inner, asset.hub_height)
        _write_obj(hub_obj, hub_verts, hub_tris)
        hub = scene.add_entity(
            morph=gs.morphs.Mesh(file=hub_obj, pos=tuple(pos), euler=tuple(euler), scale=1.0, fixed=hub_fixed),
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

    return tape, hub
