# IPC-Authoritative Coupling Mode — Architecture

Stable design for the `ipc_authoritative` coupling type. See
[roadmap.md](roadmap.md) for status and [conventions.md](conventions.md) for the
enforceable rules.

## 1. Engine topology

`genesis.engine.couplers.ipc_coupler.coupler` imports the libuipc Python binding
(`import uipc`, [coupler.py:35](../coupler.py)). The coupler **delegates all
physics to libuipc `World.advance()`** — there is no Taichi re-implementation of
IPC.

- Engine build: local fork **`Roushelfy/libuipc`**, branch **`RCC-Adhesion`**
  (follow head), **cuda backend only** (`UIPC_WITH_CUDA_BACKEND=ON`,
  `UIPC_WITH_CUDA_MIXED_BACKEND=OFF`).
- Target runtime = **gs-gym-internal** (the consumer): Python **3.10**, torch
  **cu129**, nvcc **12.8**, `CUDA_ARCHITECTURES=native`. gs-gym-internal vendors
  this Genesis branch (`third_party/Genesis`) and the fork
  (`third_party/libuipc`) as editable installs, and already drives robots with
  `external_articulation` inside cloth/tape/RCC scenes.
- Public PyPI `pyuipc` (0.0.24+) also exposes the joint/contact API used here;
  the fork's delta is RCC adhesion + SOCU/mixed backends, **not** articulation.
  We default to the fork because the consumer needs RCC coexistence.

## 2. Where it hooks in the step pipeline

The rigid solver already has an IPC split-step. Per substep:

```
RigidSolver.substep_pre_coupling(f)         # rigid_solver.py:1185
  if IPCCoupler and has_any_rigid_coupling:
      kernel_step_1(...)                     # forces / acc
      _func_constraint_force()               # Genesis collision+constraint  <-- DISABLED in this mode
      cache_pre_prediction_transforms()      # push set_qpos teleports to IPC
      kernel_predict_integrate(...)          # Genesis integration            <-- SKIPPED in this mode
      kernel_forward_kinematics_links_geoms()# predicted link transforms
      return
IPCCoupler.couple(f)                          # coupler.py:1878
  _store_gs_rigid_states()
  _pre_advance_write_ipc_attributes()         # <-- push per-joint TORQUES here (new)
  _ipc_world.advance(); _ipc_world.retrieve()
  _retrieve_ipc_fem_states(); _retrieve_ipc_rigid_states()   # coupler.py:2546
  _post_advance_write_qpos(); _sync_rigid_fk()  # IPC state -> qpos -> FK -> geoms_state
RigidSolver.substep_post_coupling(f)          # rigid_solver.py:1420 -> step_2 write-through
Visualizer.update_visual_states()             # reads geoms_state -> render
```

`ipc_authoritative` is a **specialization of this existing seam**, not a new loop.

## 3. Per-step data flow (`ipc_authoritative`)

```
        ┌─────────────────────────── Genesis ───────────────────────────┐
policy → control_dofs_position/velocity/force(+kp/kv)  (sets dofs_state.ctrl_*)
        │                                                                 │
        │  get_dofs_control_force()      # abd/accessor.py kernel:        │
        │    FORCE     : τ = ctrl_force                                   │
        │    VELOCITY  : τ = kv·(ctrl_vel − vel)                          │
        │    POSITION  : τ = kp·(ctrl_pos − pos) + kv·(ctrl_vel − vel)    │
        │    clamp to force_range                                         │
        │           per-DOF τ  ──map──▶ per-joint scalar torque[]         │
        └─────────────────────────────┬──────────────────────────────────┘
                                       ▼  AffineBodyRevoluteJointExternalForce.apply_to(sc, torque[])
        ┌─────────────────────────── libuipc ──────────────────────────┐
        │  World.advance()   # integrates ABD dynamics + joints + contact│
        │  AffineBodyStateAccessorFeature.copy_to(state_geo)             │
        │           builtin.transform (4×4 per body)                     │
        └─────────────────────────────┬──────────────────────────────────┘
                                       ▼
        ┌─────────────────────────── Genesis ───────────────────────────┐
        │  reconstruct joint angle = signed-angle(parent,child,axis) − offset
        │  joint vel = (qᵗ − qᵗ⁻¹)/dt                                    │
        │  write dofs_state.pos/vel, qpos, links_state                   │
        │  FK → geoms_state → render / getters                           │
        └─────────────────────────────────────────────────────────────────┘
```

Note the loop is closed: `get_dofs_control_force` reads `dofs_state.pos/vel`,
which we populate from the previous step's IPC readback. So the order each step
is **readback → reconstruct → compute torque → push → advance**.

## 4. What is reused vs new

**Reused as-is** (already exercised by `external_articulation` / `ipc_only`):
- ABD body construction per link, watertight→`AffineBodyConstitution`
  ([coupler.py](../coupler.py) `_add_articulation_entities_to_ipc`, ~:1283).
- `AffineBodyStateAccessorFeature` readback (`_retrieve_ipc_rigid_states`, :2546)
  — `copy_from`/`copy_to` genuinely read/write the backend `body_id_to_q` (see
  journal §M0).
- `qpos`/`links_state` writeback + FK (`_post_advance_write_qpos`, `_sync_rigid_fk`).
- Render path (`Visualizer.update_visual_states` reads `geoms_state`).
- `World.recover()` full-scene reset (`coupler.reset()`, :1986).
- Fixed-joint merge (`utils.find_abd_merge_target`).
- Teleport sync (`mark_abd_updated` → `cache_pre_prediction_transforms`).

**New work:**
1. `COUPLING_TYPE.IPC_AUTHORITATIVE` + config plumbing.
2. Robot build branch: ABD links + `AffineBodyRevoluteJoint` (with per-joint
   `init_angle` = reference offset) + `AffineBodyRevoluteJointExternalForce`
   constitution. (Existing ext_art uses `ExternalArticulationConstraint`; this
   mode does **not**.)
3. `_push_joint_torques()`: per-DOF τ from `get_dofs_control_force` → per-joint
   scalar torque array → `AffineBodyRevoluteJointExternalForce.apply_to`.
4. `_reconstruct_joint_positions()`: signed-angle-from-transforms + offset.
5. Skip `kernel_predict_integrate`; force collision/constraint off.

## 5. Robot-in-IPC construction (fixed base, revolute+fixed)

- Each non-fixed link → one affine body (`AffineBodyConstitution`); fixed-child
  links folded into ancestor body via the existing merge.
- Base link is `is_fixed` (v1 invariant).
- Each revolute DOF → one `AffineBodyRevoluteJoint` edge with attributes
  `l_geo_id/r_geo_id/l_inst_id/r_inst_id/strength_ratio/init_angle` and
  world/local axis endpoints (the fork constitution tracks the joint angle from
  `init_angle`). `init_angle` is the **per-joint zero offset** (C4): set so the
  reconstructed angle equals Genesis `qpos` at the loader pose.
- One `AffineBodyRevoluteJointExternalForce` over the joint geometry; per-step
  `apply_to(sc, torque[])` writes the actuation. The fork converts scalar joint
  torque → affine body moment internally (no manual skew like sim2sim 0.0.23).

## 6. Per-file change map (Genesis)

| File | Change |
|---|---|
| `ipc_coupler/data.py` | add `COUPLING_TYPE.IPC_AUTHORITATIVE`; `ArticulatedEntityData` fields for torque/offset arrays |
| `genesis/options/solvers.py` | `IPCCouplerOptions.actuation_mode` (default keeps current behavior) |
| `ipc_coupler/coupler.py` `_setup_coupling_config` (~:604) | recognize new type; validate fixed base + revolute/fixed; assert/auto-set `enable_collision=False`, `disable_constraint=True`; assert BDF1 |
| `ipc_coupler/coupler.py` build (~:1283) | `_add_ipc_authoritative_entities`: ABD + `AffineBodyRevoluteJoint(init_angle)` + `AffineBodyRevoluteJointExternalForce` |
| `ipc_coupler/coupler.py` `couple()` (~:1878) | new branch: push torques (pre-advance), reconstruct joints (post-advance) |
| `ipc_coupler/coupler.py` new methods | `_push_joint_torques()`, `_reconstruct_joint_positions()` |
| `rigid_solver.py` `substep_pre_coupling` (:1185) | for ipc_authoritative entities: run torque eval but **skip** `kernel_predict_integrate` |
| `rigid_entity.py` | `coup_type='ipc_authoritative'` acceptance in `control_dofs_*` guards |

## 7. Why BDF1 is required

`copy_from(transform)` writes the backend generalized coordinate
`m_abd.m_impl.body_id_to_q` (== `qs()`), and the **BDF1** predictor re-derives
`q_prev = qs(i)` at the start of each `advance()`
([abd_bdf1_time_integrator.cu:37]) so a pre-advance write is consistent with zero
spurious velocity. BDF2 keeps an extra velocity-history buffer (`q_v_n_1s`) that
`copy_from` does not touch, so teleports/state writes can carry a one-step
artifact. Default scene integrator is `bdf1`
([scene_default_config.cpp:15] in the fork). This mode locks BDF1.
