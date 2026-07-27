# QIPCCoupler FEM Support — Design

Status: P1+P2 implemented on `feat/qipc-coupler-fem` (2026-07-27). Companion to
[roadmap.md](roadmap.md). Based on a survey of the upstream `IPCCoupler` FEM path
(identical on `origin/main` and this branch), the cuda-graph-qipc (`qipc`) FEM
frontend, and the Genesis `FEMSolver` internals.

> **Implementation notes (P1/P2, 2026-07-27):**
> - Volumetric SNH + Cloth preset + per-entity contact tabular + zero-copy pos/vel
>   writeback + `substep_post_coupling` skip are in `coupler.py`; runtime vertex
>   constraints (resident SoftPositionConstraint / `fe.is_fixed`), teleport
>   (`x`+`x_prev`), and `external_acc` are wired through `FEMEntity`.
> - **Backend bug found:** any scene mixing a qipc *half-plane* ground + ABD + FEM
>   with contact enabled crashes at `SimEngine::init()` (CUDA launch failure) —
>   ABD+FEM, halfplane+ABD, and halfplane+FEM pairs all work individually.
>   Minimal repro: `~/workspace/qipc-test/pure_matrix.py halfplane` (pure qipc, no
>   Genesis; reproduces on cgq origin/main c56de72, sm_120). The coupler works
>   around it by emulating Plane entities with a large fixed ABD slab whenever FEM
>   entities are present (`_create_ground_abd_slab`).
> - Two smaller fixes: world-anchored FIXED joints (e.g. `Box(fixed=True)`) no
>   longer break merge-group construction; degenerate link inertials fall back to
>   the mesh-density ABD mass path instead of producing a singular 12x12 mass.
> - Validation: `examples/qipc/fem_smoke.py` (box_drop / cloth_drape / cloth_pin)
>   and `examples/qipc/robot_cloth_video.py` (Franka pinch-grasps a hanging cloth).

## 0. Summary

The qipc backend's FEM feature set already exceeds the subset libuipc's coupler actually
uses (StableNeoHookean volumetric + Baraff-Witkin cloth), and its frontend API is
structurally isomorphic to libuipc (`constitution.apply_to(geo)` + ContactTabular +
animator), so the `IPCCoupler` FEM code can be ported almost section-by-section. The
design opportunities are fixing three structural weaknesses of the IPC FEM path:

1. **Writeback round-trips through the host** (SceneVisitor over all geometries +
   numpy stack per substep). qipc exposes `scene.finite_element.x` / `.velocities` as
   zero-copy CUDA float64 torch views — feed them straight into a quadrants kernel.
2. **Velocities are never written back** (`entity.get_state().vel` is always zero under
   IPC). qipc maintains `fe.velocities` natively; write them back alongside positions.
3. **Vertex constraints/BCs unsupported** (`set_vertex_constraints` raises under IPC).
   qipc's `aim_position` / `is_constrained` / `is_fixed` are CUDA-graph-safe live device
   writes, enabling a *stronger* runtime constraint API than the libuipc coupler has.

Plasticity, `SealedVolumeGas`, and `AdhesiveIPCContact` all follow qipc's
"declare at build, tune via live views at runtime" pattern and slot into the same frame.

## 1. Architecture and data flow

### Build (`QIPCCoupler.build()` extension)

Entity classification adds a FEM branch to the existing plane/ABD split:

```
plane → qipc ground                                   (existing)
rigid → ABD + joints                                  (existing)
FEM   → isinstance(material, Cloth) ? trimesh + shell
                                    : tetmesh + volumetric   (new)
```

Per FEM entity (single env, see §8):

```python
# volumetric: geo = tetmesh(entity.init_positions → f64, entity.elems → i32)
# cloth:      geo = trimesh(entity.init_positions → f64, entity.surface_triangles → i32)
# 1) constitution from the material registry (§2)
# 2) per-entity ContactElement.apply_to(geo) (§3)
# 3) resident lazy SoftPositionConstraint (is_constrained=0) for runtime BCs (§5)
slot = scene.geometries.create(f"fem_{i_e}", geo)
# after scene.init(): record (entity.v_start, int(geo.meta["fem_vert_offset"]), entity.n_vertices)
```

Key constraint: **the qipc Scene freezes at `init()`** — geometry, constitutions, the
contact tabular, and adhesion pairs are uploaded once. All *feature declarations* must
converge at build; runtime may only touch the documented live device buffers. This
dictates the API split throughout: **declaration via materials/options, tuning via
entity runtime APIs**.

### Per step

```
preprocess(f):  rigid control forwarding (existing) + FEM runtime target flush:
                _tgt set_position/set_velocity → fe.x[s] + fe.x_prev[s] + fe.velocities[s]  (teleport semantics)
                constraint targets → geo.vertices["aim_position"].gpu()[:]                  (graph-safe)
                gas state → scene._constitution_data[SealedVolumeGas]["p0"|"v0"|"enabled"]
couple(f):      substep accumulation as today → scene.step() → unified writeback (§4)
```

### Required upstream change (fem_solver.py)

`FEMSolver.substep_pre_coupling` already early-returns for `QIPCCoupler`
(`fem_solver.py:962-963`), but **`substep_post_coupling` (`:986-990`) has no matching
skip** — `compute_pos(f)` would overwrite our `f+1` frame with
`pos[f] + dt·vel[f+1]`. Add the same `isinstance(coupler, QIPCCoupler): return`.
(IPC survives without it by writing frame 0 and letting `compute_pos` propagate — at
the cost of zeroed velocities. We do not take that path.)

## 2. Material mapping (registry pattern)

Implement as an explicit registry (`dict[type[Material], ConstitutionBuilder]`) so new
materials add one builder without touching the coupler body:

| Genesis material (fields) | qipc constitution | Notes |
|---|---|---|
| `FEM.Elastic(model="stable_neohookean")` E/nu/rho | `StableNeoHookean(youngs_modulus, poissons_ratio, mass_density)` | parity with IPC |
| `FEM.Elastic(model="stvk_hencky")` **(new enum value)** | `StvkHencky(...)` | true Lamé; prerequisite for plasticity |
| `FEM.Elastic(model="linear"/"linear_corotated")` | **warn + fall back to SNH** (or raise) | IPC silently ignores `model`; do better |
| **new** `FEM.Plastic(Elastic)`: `yield_stress`, `hardening` ("none"/"linear"/"voce"), `hardening_coeff`, `yield_stress_sat` | `StvkHencky` + `VonMisesPlasticity` modifier | plasticity is a *modifier*, not a constitution; per-tet params let multiple plastic bodies share one backend |
| `FEM.Cloth`: E, nu, rho, thickness, bending_stiffness + **new** `membrane` ("baraff_witkin"/"stvk"), `strain_limit_multiplier`, `contact_thickness`, `bending` ("quadratic"/"hinge"/None) | `Cloth` preset (`youngs_modulus, shear_modulus/poissons_ratio, thickness, bending_youngs_modulus, ...`) | two semantic traps: ① the BW membrane takes `shear_modulus`, not nu (only `membrane="stvk"` takes `poissons_ratio`); for BW derive `G = E/(2(1+nu))`. ② qipc decouples `thickness` (mass/elasticity) from `contact_thickness` (IPC gap) — garments starting in contact need `contact_thickness=0`; Genesis `Cloth` needs this field |
| **new** `FEM.SealedGasShell(Cloth)`: `p_gauge0`, `p_atm`, `gamma`, `v_min_rel`, `auto_flip` | membrane constitution + `SealedVolumeGas().apply_to(geo, ...)` (extra/stacked constitution) | §6 |
| `FEM.Muscle` | **raise explicitly** | qipc has no muscle actuation; IPC silently degrades it to plain SNH, which is worse |
| `friction_mu` / `contact_resistance` | ContactTabular per-pair | §3 |

`gs.materials.Rigid` has a `qipc_*` field-prefix precedent, but the fields above are
physical (not coupler tuning), so they should be first-class — the libuipc coupler can
consume them later too.

## 3. Contact tabular: reuse IPCCoupler's formulas + fix the qipc gating trap

- Per-entity `ContactElement` (`fem_contact_{i}` / `cloth_contact_{i}` / `abd_contact_{i}` /
  ground); upper triangle *including* self-pairs (FEM self-contact on by default).
- Pairwise formulas as in `ipc_coupler/coupler.py:675-691`:
  `mu_ij = sqrt(mu_i * mu_j)`, `kappa_ij = 2*k_i*k_j/(k_i+k_j)`; ground pairs likewise.
- **Trap**: qipc only wires the friction machinery when the *default* model's
  `friction_rate > 0` (`qipc/solver/solver.py:485-491`); per-pair mu on top of a zero
  default silently does nothing. Set
  `default_model(friction_rate=max(min_mu, eps), resistance=options.contact_resistance)`.
- The table freezes at init: `enable_rigid_rigid_contact`-style switches get baked in at
  build, exactly as IPC does.

This step also completes roadmap Priority 4 (rigid per-pair contact) — same mechanism.

## 4. State writeback (the core upgrade over IPC)

```python
# once after build (both sides keep creation order → pure offset mapping):
#   per entity: (gs_v_start, qipc_fem_vert_offset, n_verts) → flat index arrays
# in couple():
fe = self._scene.finite_element
self._kernel_fem_writeback(fe.x, fe.velocities, self._map_gs2qipc, f + 1)
```

- `fe.x` / `fe.velocities` are DLPack torch views over native-owned CUDA float64
  buffers; quadrants kernels accept `qd.types.ndarray` directly → **zero-copy, no host
  round trip** (IPC walks every scene geometry + numpy-stacks per substep).
- **Write frame `f+1`, both pos and vel**: `save_ckpt`'s `copy_frame(1, 0)` promotes it
  to the next step's frame 0, so rendering (`get_state_render(cur_substep_local)`) and
  `entity.get_state()` are automatically correct — FEM rendering is pull-based, there
  is no analogue of the rigid `geoms_state` staleness problem.
- Velocities come straight from qipc's `fe.velocities`; no finite differencing.
- Precision: qipc is float64-only; cast to the `gs.init` precision inside the kernel
  (QIPC examples already require `precision="64"`).

## 5. Boundary conditions / constraints (leapfrogging the libuipc coupler)

The libuipc coupler raises on `set_vertex_constraints`; qipc constraint targets are
graph-safe runtime device writes, so the full runtime API is supportable:

| Genesis API | qipc mechanism |
|---|---|
| `set_vertex_constraints(soft=True, stiffness)` | resident `SoftPositionConstraint` (applied to all FEM geoms at build with `is_constrained=0`); at runtime write `is_constrained` / `strength_ratio` / `aim_position` via `.gpu()[:]` |
| `set_vertex_constraints(soft=False)` | `fe.is_fixed[idx] = 1` (live int32 view; note `is_fixed` ignores the contact barrier) or `PositionDBC` (kinematic, CCD-safe — right for mocap-driven verts) |
| `update_constraint_targets` / `link=` following a rigid link | compute targets from the link pose in `preprocess` and write `aim_position` (equivalent of `_kernel_update_linked_vertex_constraints`) |
| `set_position` (teleport) | write `fe.x` **and** `fe.x_prev` (the predictor reads only `x_prev`/`velocities`; writing `x` alone is physically undone — qipc's canonical teleport idiom) |
| `set_velocity` | `fe.velocities[slice] = v` |
| external forces (uniform force etc.) | `fe.external_acc` (live, enters the predictor) |

Cost caveat: the resident SPC hangs a zero-strength energy term on every FEM vertex —
confirm on the cgq side that `is_constrained=0` vertices are skipped in the kernel
(intended); otherwise fall back to apply-at-build-on-demand and document that
constraints must be declared before `scene.build()`.

## 6. SealedVolumeGas

- Declaration: `FEM.SealedGasShell` material → membrane (StVKShell/Cloth) +
  `SealedVolumeGas().apply_to(geo, p_gauge0, p_atm, gamma, v_min_rel, auto_flip)`
  stacked as an extra constitution.
- **Build-time validation**: requires a watertight, consistently wound trimesh (qipc's
  `closed_surface_orientation` raises `MeshValidationError` — catch and re-raise with
  the Genesis entity name). `V0` is taken from `geo.positions` (initial state), not
  `rest_geometry`.
- **Hard requirement**: `linear_system/solver == "partition_pcg"` (qipc default). If
  `QIPCCouplerOptions` ever exposes the solver choice, validate the combination.
- Runtime: `scene._constitution_data[SealedVolumeGas]` exposes live `p0`/`v0`/`enabled`
  views → expose `entity.set_gas_state(p0=..., v0=..., enabled=...)` for
  inflate/deflate (the `chip_bag_press` example pattern).

## 7. AdhesiveIPCContact

- qipc's contact constitution is a **scene-level single choice** (consistent / gipc /
  adhesive; a second `apply_to` raises) →
  `QIPCCouplerOptions.contact_constitution: Literal["consistent","gipc","adhesive"] = "consistent"`
  plus global adhesion params (`bond_distance_lock`, `bond_distance_lock_ratio`,
  `bond_max_bonds`, `bond_kappa`, `bond_release_{force,strain,gap,slip}`, `occlusion`).
- Per-pair adhesion params (`Cn, Ct, W, eta, bonding_rate, beta0, adhesion_enabled,
  distance_lock, distance_lock_ratio, release_force`) go through ContactTabular insert →
  provide **`coupler.add_adhesion(entity_a, entity_b, **params)` (called before
  `scene.build()`)**, with naming aligned to the eval-branch IPCCoupler's
  `add_rcc_adhesion` / `rcc_bonded_pt_distance_lock_*` API so both couplers present a
  consistent user surface.
- With all adhesion params zero, `AdhesiveIPCContact` reproduces C-IPC bit-for-bit —
  defaulting to it would be low-risk, but keep it opt-in initially.
- Mixed ABD–FEM adhesion is covered by cgq's `test_bond_abd_fem_mixed.py`; gripper–soft
  scenarios work out of the box.

## 8. reset, n_envs, substeps

- **reset** (roadmap P3): pass `workspace=` to the Scene, `scene.dump()` at end of
  build (frame-0 checkpoint, mirroring IPC's `_finalize_ipc`), `reset()` →
  `scene.recover(0)` + immediate writeback. dump/recover includes plastic state.
  Restrict to `envs_idx is None`.
- **n_envs > 1**: qipc has no batch/subscene concept (zero hits repo-wide). Emulating
  env isolation with per-pair disables explodes the dense N×N tabular. **Phase 1 keeps
  `n_envs <= 1`**; the long-term fix is a subscene/batch feature on the cgq side, not a
  coupler-level workaround.
- **substeps**: keep the existing rigid pattern (accumulate substeps, one qipc step per
  Genesis step) + the `_is_first_step` guard for the build-time warm-up step.

## 9. Phased plan

1. **P1 core volumetric + cloth**: classification/geometry, SNH + Cloth mapping,
   per-entity contact tabular (completes roadmap P4 as a side effect), zero-copy
   pos+vel writeback, `substep_post_coupling` skip → port `tests/ipc/test_deformable.py`
   cloth corner-drag and biaxial-stretch (the latter is the only numeric validation of
   the E/nu mapping).
2. **P2 constraints/BC**: SPC/PositionDBC/is_fixed wiring, the full
   `set_vertex_constraints` family + teleport + `external_acc`.
3. **P3 plasticity**: `stvk_hencky` enum + `FEM.Plastic` + plastic-strain readback,
   validated against cgq `test_tet_plasticity.py`.
4. **P4 SealedVolumeGas**: material + validation + runtime inflate/deflate.
5. **P5 adhesion**: `contact_constitution` option + `add_adhesion` + bond params.
6. **P6 reset** (dump/recover).

## 10. Risks and incidental findings

Risks:

1. Contact-pair buffer regrow crash (deterministic `dynamic_exclusive_sum` failure
   reproduced 2026-07-16 on cgq 022e621). cgq docs claim overflow yields to host and
   reallocs; re-verify on current cgq main — this is the biggest stability risk for
   contact-heavy scenes.
2. qipc mandates Python ≥3.13 and float64; batched RL workloads are out of scope for now.
3. Resident-SPC overhead needs measurement (§5 caveat).

Incidental Genesis bugs found during the survey (worth reporting independently):

1. `FEMSolver._kernel_get_el2v` indexes its per-entity output array with the *global*
   element index — out-of-bounds for any FEM entity with `el_start > 0`
   (`fem_solver.py:1339-1349`).
2. `_kernel_add_cloth` never sets vertex `mass_inv`; `FEM.Cloth` under a non-IPC coupler
   reads uninitialized data, and nothing validates the combination.
3. `FEMSolver.save_ckpt` allocates fresh `(B, n, 3)` torch buffers under a new
   `ckpt_name` every step, so `_ckpt` grows unboundedly even in non-grad mode — a
   memory leak for long FEM runs.
