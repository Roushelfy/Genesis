# QIPCCoupler Adhesion + Tape Import — Design

Status: A5.1–A5.3 implemented (2026-07-28); A5.4 (bond/beta state transfer)
deferred. Companion to [fem_design.md](fem_design.md); current milestone status
lives in [roadmap.md](roadmap.md). Last audited against cuda-graph-qipc @
`9d04b459`.

> Implementation map: adhesion manager in `../adhesion.py` (options
> `contact_constitution` + `adhesion_bond_*`, `coupler.add_adhesion`,
> `coupler.adhesion` runtime state); tape import in `../tape.py`
> (`TapeAsset.from_npz`, `add_tape_roll`, `recommended_coupler_options`) plus
> `coupler.set_fem_rest_positions` and the `FEM.Cloth`
> membrane/bending/strain-limit fields; kinematic driving via
> `coupler.enable_soft_transform` / `set_soft_transform_target` and
> `Rigid.qipc_d_hat`. Tests: `tests/test_qipc_adhesion.py`,
> `tests/test_qipc_tape.py`, `tests/test_qipc_soft_transform.py`.

## 0. Scope and summary

Two deliverables:

1. **Adhesion**: expose qipc's two-layer adhesion stack through the coupler —
   *Phase-1 soft adhesion* (per-pair β-state RCC potential: `Cn/Ct/W/eta/
   bonding_rate/p0/beta0/adhesion_enabled` on the contact tabular) and
   *Phase-2 distance bonds* (locked virtual-tet bonds with per-pair
   `distance_lock/distance_lock_ratio/release_force` and scene-global
   create/release config on `AdhesiveIPCContact`).
2. **Tape import**: bring a wound tape roll into a Genesis scene as a first-class
   pair of entities (prestressed shell strip + ABD ring hub), loadable from the
   qipc tape asset format (`.npz` produced by cgq's `adhesive_tape_wind.py`).

Key facts the design is built on (from the cgq source, references in §9):

- Everything adhesion-related is **frozen at `scene.init()`** (tables read once
  by `Solver._wire_contact_tabular`); the only runtime mutables are per-pair β
  (`dump/load_adhesion_pair_state`) and bond seeding (`seed_locks`). So the
  coupler API is declarative-before-build, mirroring `IPCCoupler.add_rcc_adhesion`
  ("register before finalize").
- `AdhesiveIPCContact` with all-zero adhesion params is **bit-identical** to
  `ConsistentIPCContact` (asserted upstream), at the cost of one extra
  lagged-position snapshot per step.
- A tape roll is **not authored geometrically**: cgq generates a flat strip and
  *simulates* the winding (SPC-guided, zero-g), then persists the wound state to
  `.npz`. Importing tape = wound initial positions + **flat rest geometry**
  (prestress: membrane + bending energy stored in the coil) + graded adhesive
  contact pairs. No rods anywhere — the "rod" in the rod-wind examples is a
  fixed ABD cylinder mandrel; rod entities are explicitly out of scope.

## 1. Options (`QIPCCouplerOptions`)

```python
# Contact constitution selection
contact_constitution: Literal["auto", "consistent", "adhesive"] = "auto"
#   auto  -> "adhesive" iff any add_adhesion() request or bond option is set,
#            else "consistent" (avoids the lagged-position cost when unused).

# Phase-2 distance-bond globals (all map 1:1 onto AdhesiveIPCContact.apply_to)
adhesion_bond_distance_lock: StrictBool = False
adhesion_bond_distance_lock_ratio: NonNegativeFloat = 0.5   # band = xi + c*d_hat
adhesion_bond_max_bonds: NonNegativeInt = 0                 # 0 = bonds inert
adhesion_bond_kappa: PositiveFloat = 1e8
adhesion_bond_lock_margin: NonNegativeFloat = 0.0
adhesion_bond_release_strain: PositiveFloat = 1e30
adhesion_bond_release_force: PositiveFloat = 1e30           # Newtons, dt-independent
adhesion_bond_release_gap: PositiveFloat = 1e30             # metres
adhesion_bond_release_slip: PositiveFloat = 1e30            # metres
adhesion_occlusion: StrictBool = False                      # bond-create gate only
```

Build-time validation in the coupler (all host-side, readable errors):
- `adhesion_bond_distance_lock and adhesion_bond_max_bonds == 0` → raise
  (upstream this is *silently* inert).
- bonds requested but no FEM entities in the scene → raise (`BondSystem`
  requires FEM vertices; ABD-only scenes cannot bond).
- `add_adhesion` targeting a Plane entity → raise (half-plane contacts never
  run the PT adhesion kernel; ground is inherently non-sticky).
- `Ct > 0` requires the friction machinery, which is wired iff the *default
  model's* `friction_rate > 0` — guaranteed already because
  `contact_friction: PositiveFloat` cannot be 0. Documented, no check needed.

## 2. Per-pair API: `coupler.add_adhesion(...)`

Aligned with the eval-branch `IPCCoupler.add_rcc_adhesion` (source → targets,
queued requests, applied before finalize), adapted to qipc semantics:

```python
def add_adhesion(
    self,
    source_entity,                    # RigidEntity | FEMEntity
    target_entities=None,             # entity | sequence | None (None = self-adhesion)
    *,
    Cn: float,                        # normal stiffness (area-lumped! see §7)
    Ct: float = 0.0,                  # tangential stiffness (rides friction kernels)
    W: float = 1.0,                   # normalized debond work
    eta: float = 1.0,                 # debond viscosity (larger = slower debond)
    bonding_rate: float = 0.0,        # beta growth under compression (0 = frozen)
    p0: float = 0.0,                  # compression saturation pressure
    beta0: float = 0.0,               # seed beta; 1.0 = pre-bonded on first contact
    enabled: bool = True,             # per-pair adhesion_enabled mask
    friction: float | None = None,    # pair friction override (None = geometric mean)
    resistance: float | None = None,  # pair resistance override (None = harmonic mean)
    distance_lock: bool | None = None,        # None = qipc default (True when bonds on)
    distance_lock_ratio: float | None = None, # None = inherit global (sentinel -1)
    release_force: float | None = None,       # None = inherit global (sentinel -1)
) -> None
```

- Must be called **before `scene.build()`**; requests are queued and merged into
  `_setup_contact_tabular` (which already creates per-entity `ContactElement`s
  and inserts the full upper-triangle pair matrix — the merge only *adds
  columns* to rows it already writes).
- `target_entities=None` or `== source` → self-adhesion pair `(elem, elem)`
  (tape–tape).
- Sentinel translation: `None → -1` only for `distance_lock_ratio` /
  `release_force`; the Phase-1 columns have no sentinels (0 means 0).
- **No `sticky_side`** (present in the libuipc RCC API): qipc PT adhesion is
  side-symmetric. One-sided tape is approximated by geometry (the coil's
  winding means only the inward face ever comes into PT range) — same approach
  as cgq's own tape demos. A future per-vertex `contact_element_id` "sticky
  region" channel could narrow this; deferred.
- Pairs *not* named in any request keep pure friction/resistance rows; note the
  qipc per-pair `distance_lock` default is **True** (opt-out), so when the
  global lock is on, unnamed pairs may bond — matching upstream semantics.
  `add_adhesion(..., distance_lock=False)` opts a pair out.

## 3. Runtime APIs (post-build)

```python
coupler.get_contact_info() -> (n_pairs_pt, n_pairs_ee, n_active)
coupler.get_bond_topos() -> np.ndarray (n_bonds, 4) int32   # GLOBAL vertex ids
coupler.get_bond_count() -> int
coupler.dump_adhesion_state() -> (keys, betas)               # device tensors
coupler.load_adhesion_state(keys, betas) -> None             # e.g. re-bond mid-run
coupler.seed_bonds(topos, rest_height) -> None               # once, before 1st step
```

- Bond topos are in qipc *global* vertex-id space; the coupler translates FEM
  ids back to `(entity, local_vertex)` using the recorded `fem_vert_offset`
  ranges (ABD ids reported raw with the owning entity looked up from geometry
  slot metadata).
- No bond clearing / per-bond mutation exists upstream; not exposed.
- β keys hash global vertex ids → `load_adhesion_state` is only valid within
  the same built scene (documented; the tape importer re-forms bonds through
  `beta0=1` instead of transferring state, see §5).

## 4. Cloth material extensions (needed by tape, generally useful)

`gs.materials.FEM.Cloth` grows four fields, passed straight through the qipc
`Cloth` preset (which already accepts them):

```python
membrane: Literal["baraff_witkin", "stvk"] = "baraff_witkin"
shear_modulus: PositiveFloat | None = None          # BW only; default E/(2(1+nu))
strain_limit_multiplier: NonNegativeFloat | None = None  # None = constitution default
bending_model: Literal["quadratic", "hinge"] = "quadratic"
```

Mapping notes:
- tape (wind/drop class) = `membrane="stvk"` + `bending_model="hinge"`
  (DiscreteShellBending) + `strain_limit_multiplier=0`;
  rod-wind class = BW membrane (`stretch 1e7 / shear 1e5`) + hinge bending +
  strain limit 100. Both expressible with these fields.
- `bending_stiffness` keeps its current meaning (`bending_youngs_modulus`);
  qipc hinge bending uses `k = E_b h^3 / (24 (1 - nu^2))`.

## 5. Prestress: `coupler.set_fem_rest_positions(entity, rest_verts)`

The single load-bearing gap for tape. Declarative pre-build API:

```python
coupler.set_fem_rest_positions(fem_entity, rest_verts)  # (n_verts, 3), same topology
```

`_build_fem_entities` then calls
`scene.geometries.create(name, geo, rest_geometry=trimesh(rest_verts, faces))`.
qipc derives the rest metric (`Dm_inv_2d`), rest areas, **lumped masses**, and
`theta_bar` from the rest mesh while the simulation starts at the (wound)
initial positions — the coil carries stored membrane + bending energy exactly
like cgq's `adhesive_tape_drop`. Validation: vertex count/topology must match;
warn if rest area differs wildly from initial area (mass comes from rest).
Genesis-side rendering/writeback is untouched (positions flow back as today).

## 6. Tape import

### 6.1 Asset (`genesis/engine/couplers/qipc_coupler/tape.py`)

```python
@dataclass
class TapeAsset:
    tape_positions: np.ndarray   # wound coil verts (n, 3), hub axis +z
    tape_tris: np.ndarray        # (m, 3) int32
    nx: int; nz: int
    thick: float; tape_length: float; width: float
    youngs: float; poisson: float; density: float; bending_e: float
    hub_r_outer: float; hub_r_inner: float; hub_height: float
    d_hat: float
    params: dict                 # full effective preset (pickled dict in npz)

    @classmethod
    def from_npz(cls, path) -> "TapeAsset"
    def flat_rest_positions(self) -> np.ndarray   # straight strip, same (nx+1)*(nz+1) layout
```

Reads cgq's wind output (`_pc0716_winner.npz`-style). The flat rest strip is
regenerated analytically from `nx/nz/tape_length/width` (same vertex layout as
`make_tangent_tape`; absolute placement of the rest strip is irrelevant — only
the rest metric and dihedral angles enter). Optional npz keys
(`bond_topos`, `pair_state_*`) are parsed but **ignored in v1** (§8 A5.4).

### 6.2 Scene builder

```python
def add_tape_roll(
    scene, coupler, asset, *,
    pos=(0, 0, 0), euler=(90, 0, 0),      # default stands the roll up (hub axis +z -> world +y)
    with_hub=True, hub_fixed=False,
    tape_surface=None, hub_surface=None,
    adhesion_overrides: dict | None = None,
) -> (cloth_entity, hub_entity | None)
```

Steps (all on existing or §1–§5 plumbing):
1. Transform `tape_positions` by (euler, pos); write a temp OBJ; add a FEM
   entity with `FEM.Cloth(E=youngs, nu=poisson, rho=density,
   thickness=thick, bending_stiffness=bending_e, membrane="stvk",
   bending_model="hinge", strain_limit_multiplier=0, friction_mu=0.5)`.
2. `coupler.set_fem_rest_positions(tape, asset.flat_rest_positions())`.
3. If `with_hub`: generate the ring-hub trimesh (`make_ring_hub` port: annulus
   prism, outward-consistent normals), temp OBJ → `Rigid` entity
   (`fixed=hub_fixed`), transformed by the same (euler, pos).
4. Register adhesion (defaults from the asset's preset, overridable):
   - `coupler.add_adhesion(tape, tape, Cn=1.0, Ct=ct, W=1.0, eta=100.0,
     bonding_rate=1.0, beta0=1.0, friction=0.5)` — **`beta0=1` makes the
     imported coil hold from frame 0** without any bond-state transfer;
   - `coupler.add_adhesion(tape, hub, Cn=10.0, ...)` (glue tab band);
5. Caller enables Phase-2 in options for peel-force semantics
   (`adhesion_bond_distance_lock=True, ratio=1.0, max_bonds=16384,
   kappa=1e6, release_force=0.5`), matching the cgq production preset.

Ground contact needs no adhesion handling (half-plane pairs are inherently
inert for adhesion) — but the roll must clear the ground per the existing
`_preflight_ground_clearance`.

### 6.3 Where the asset comes from

- v1: consume a cgq-produced npz directly (`adhesive_tape_wind.py --preset ...
  --save-roll`), copied into the Genesis assets dir or given by path.
- v2 (planned): `examples/qipc/tape_wind.py` — a Genesis-side port of the wind
  stage using soft `set_vertex_constraints` / `update_constraint_targets` for
  SPC row-band guidance, writing the same npz format. This remains gated on
  working removal and explicit soft-constraint reset semantics; the current
  runtime hard-constraint path writes QIPC's init-only `is_fixed` and is not a
  usable fallback.

## 7. Kinematic rigid driving (wind/orbit class) — phase A5.3

Rod-wind-class demos additionally drive ABD bodies kinematically via
`SoftTransformConstraint` + per-frame `aim_q` writes. Two small APIs:

```python
# pre-build (queued; applied to the merged body owning `link`, default base link)
coupler.enable_soft_transform(entity, link=None, strength=(1e3, 1e3))
# post-build, per step (graph-safe device writes)
coupler.set_soft_transform_target(entity, pos, quat, enabled=True, link=None)
```

Plus a per-entity broad-phase band override for coarse robot meshes meeting
thin tape: `gs.materials.Rigid.qipc_d_hat: float | None = None` → stamped as
`geo.meta["d_hat"]` on the entity's ABD geometries (upstream supports this
per-geometry override; the finger demo uses 1e-3 against 1.8e-4 tape).

Not needed for the drop-class demo; implement after A5.1/A5.2.

## 8. Implementation phases

| phase | content | validation |
|---|---|---|
| **A5.1 adhesion core** | §1 options, §2 `add_adhesion`, constitution auto-select, §3 runtime APIs, host validation | port `adhesive_cloth_peel` (stick vs. no-stick assert); port `test_distance_bond` stick/hold/release-force thresholds; ABD+FEM mixed bond test |
| **A5.2 tape import (drop class)** | §4 Cloth fields, §5 rest_geometry channel, §6 TapeAsset + `add_tape_roll` | prestress unit test (wound strip springs open without adhesion; holds with `beta0=1`); tape-drop port: lift roll by free tail → hub carried (the cgq drop diagnostic), rendered video |
| **A5.3 kinematic driving** | §7 STC + `aim_q`, `qipc_d_hat` | orbit/pull demo; optional Genesis-side wind (§6.3 v2) |
| **A5.4 state transfer** | `seed_bonds` + β import from npz (bit-exact pre-bonded roll) | deferred; `beta0=1` re-bonding covers the practical need |

## 9. Constraints & gotchas carried into the design (upstream facts)

1. All tabular/constitution config is read once at `scene.init()`; post-init
   edits are silently ignored (or worse for re-wiring) → hence the
   declare-before-build API shape.
2. `Ct` lives inside the friction kernels; wired only when the default model's
   `friction_rate > 0` (already guaranteed by `contact_friction`).
3. One contact constitution per scene; "adhesive with zero params" == CIPC
   bit-for-bit, so `auto` never changes physics, only the CP0 snapshot cost.
4. **`Cn`/`Ct` are area-lumped** (not area-weighted like the barrier): the same
   value gives different net stick force at different mesh resolutions.
   Documented on `add_adhesion`; tape defaults come from the asset preset,
   which was tuned at the asset's own resolution.
5. Adhesion reach is `xi + d_hat` with `d* = xi + 0.5*d_hat` hardcoded —
   `contact_d_hat` is effectively the adhesion-range knob.
6. Phase-2 formation is contact-driven only (`bonding_rate`/`beta0` play no
   role); locks suppress the PT barrier while alive; release conditions:
   strain/force (tension-gated), gap, slip, flip, degenerate.
7. Bond formation requires the closest feature to be face-interior and rejects
   ABD-ABD pairs (P and triangle can't both be ABD) — pure rigid-rigid bonding
   is not available; at least one side must be FEM.
8. One-step engagement lag (locks form at CP4, act next frame) — tests budget
   settle frames before asserting.
9. `eta=0` silently disables debonding; `p0=0` everywhere upstream; keep both
   defaults but document.
10. Occlusion gates bond creation only (Phase-1 unaffected at current HEAD) and
    is O(n_tris) per candidate — off by default.
11. **Airborne locked coils hover at the default `newton/velocity_tol=0.05` —
    a convergence artifact, not lock physics** (2026-07-29). Newton's absolute
    tolerance is `velocity_tol * dt` = 0.5 mm/iteration at dt=0.01; for a
    lock-stiffened spool the free-fall correction stays under that and the
    solve terminates with the spool floating (`newton=2`, "converged").
    Reproduced with cgq's own `adhesive_tape_drop --lock` elevated +2 cm on
    the stock wheel (hovers indefinitely at the shipped SOLVER_CFG, which
    also uses 0.05); the demo's own FREEFALL release phase silently no-ops the
    same way. Tightening `velocity_tol` to 0.01 makes the native elevated coil
    land within 80 frames. Historical Genesis measurements matched that result,
    but the current `tape_lift_drop` port no longer reaches freefall because
    `remove_vertex_constraints` writes QIPC's init-only `is_fixed` buffer and
    raises. Treat the tolerance result as native evidence until soft-constraint
    removal is repaired; the Genesis release example is not a current gate.
    The lock energy is translation-invariant as documented; an earlier "locks
    resist rigid translation" reading is superseded. Imports keep the seated
    spawn as the cheap default; use `solver_newton_velocity_tol=0.01` once
    airborne locked-coil release is restored.
12. **Tape placement must be baked into the mesh** (`add_tape_roll` does this):
    Genesis's FEM Mesh loader pivots `euler` about the vertex COM
    (`fem_entity.py`), and the wound coil's COM is pulled off-axis by the free
    tail (~9mm for the default preset), so morph-level pos/euler shifted the
    coil vs the hub by ~7.4mm — the hub wall spawned EMBEDDED in the coil
    (119 verts inside the bore), producing mutual pass-through and heavy
    solver churn. The tape OBJ is therefore written in world coordinates with
    an identity morph pose. The HUB deliberately keeps morph placement: the
    ring is COM-centered at the file origin, so origin-pivot == COM-pivot
    under either rigid `align` semantics — whereas baked world coordinates
    would CANCEL under `align=True` auto-reframing (free simple bodies get a
    COM frame and morph pos places that frame: a baked hub lands at the world
    origin). `test_tape_roll_hub_concentric` (free hub) is the regression
    guard on the actual hub geometry.
13. **Rigid meshes reach QIPC through Genesis's collision-asset pipeline, so
    pass `convexify=False`** for anything QIPC should see exactly. The ABD
    build reads `link.geoms[*].init_verts/init_faces`, i.e. post-processing
    geometry: convexification decomposed the ring hub into 8 hulls that fill
    its bore (192 → 160 verts, mass 5.28e-3 → 7.56e-3 kg, +43%) and renumbered
    its vertices, which broke both the contact surface and the wind-time
    vertex ids. With `convexify=False` (`add_tape_roll` sets it, plus
    `watertighten=None` to make the intent explicit — the closed ring makes
    the wrap a no-op anyway) the ring arrives with its 192 vertices in source
    order and Genesis's own inertial is then geometrically exact (the 48-gon's
    inscribed-area ratio 0.99715 × the analytic annulus). No coupler-side
    override API is needed. Payoff measured on the lift-drop timeline: all 454
    saved locks transfer instead of 265, the trajectory lands within ~2mm of
    the native reference everywhere, and the run got 2.6x FASTER (269s → 102s;
    PCG stops pegging its cap) — the bore-filling hulls had been fighting the
    coil. Guard: `test_tape_hub_mesh_is_exact`.
14. The assembly-buffer capacity heuristic oscillates on this scene (resize ↔
    post-frame padding shrink every few steps, `sim_engine.cu`
    `handle_count_overflow`): the resize predicts `counted × fric_factor(2) ×
    1.2` but the post-frame shrink floors padding at `1.2 × actual` where
    actual ≈ counted — below the next prediction. Each early resize can cost a
    CUDA-graph rebuild (first steps 0.7–50 s); steady state is unaffected
    (~10 ms/step). Engine-side; worth reporting with the tape scene as repro.
