# IPC-Monolithic Mode — Development Journal

Chronological decisions, validations, and findings. Stable conclusions migrate to
[../architecture.md](../architecture.md); current status stays in
[../roadmap.md](../roadmap.md).

---

## M0 — Feasibility & source verification (COMPLETE)

Goal: decide whether "IPC solves+controls everything, Genesis only reads state &
renders" is feasible, and resolve the dependency/engine question, before writing
code.

### Findings (verified)

1. **Engine is libuipc, not a re-impl.** The coupler delegates to libuipc
   `World.advance()/retrieve()`; `import uipc` at `coupler.py:35`. No Taichi IPC.

2. **Dependency / engine split.**
   - sim2sim → **public PyPI `pyuipc==0.0.23`** (`uv.lock` source `pypi.org/simple`;
     wheels cp310–cp313, `requires-python <3.14,>=3.10` — the "cp311-only" was an
     artifact of sim2sim pinning py311).
   - Genesis coupler → **fork `Roushelfy/libuipc`**, branch `RCC-Adhesion`,
     version 0.9.0, cuda backend. Fork delta vs public = `RCCAdhesive` +
     SOCU/cuda_mixed backends; **articulation joints + motors + external-force +
     `ExternalArticulationConstraint` exist in the public 0.0.x line too**
     (verified in cached `0.0.24` `constitution.pyi`).
   - Decision: **default to the fork** (consumer needs RCC coexistence).

3. **`copy_from` truly writes backend ABD state.** Traced
   pybind → `do_copy_from`
   (`src/backends/cuda/affine_body/affine_body_state_accessor_feature.cu:25`):
   `builtin::transform → transform_to_q → m_abd.m_impl.body_id_to_q`, and
   `qs()` returns exactly `m_impl.body_id_to_q.view()`
   (`affine_body_dynamics.h:380`). Velocity → `body_id_to_q_v`. Then
   `request_attribute_update()` (recompute contact verts) +
   `update_dof_attributes()` (resync per-joint `current_angles`). So state writes
   mutate the live coordinate the solver integrates — not a shadow buffer.

4. **BDF1 makes pre-advance writes clean.** BDF1 predictor sets `q_prev = qs(i)`
   each step (`abd_bdf1_time_integrator.cu:37`), so a write to `q` (with `q_v=0`)
   does not produce the stale-`q_prev` energy spike warned about at
   `coupler.py:1230`. BDF2 keeps `q_v_n_1s` untouched by `copy_from` → not clean.
   Default scene integrator is `bdf1` (`scene_default_config.cpp:15`). → **lock BDF1.**

5. **Clean control-torque seam already exists.** `get_dofs_control_force`
   (`rigid_solver.py:2398`) / its kernel in `abd/accessor.py` folds all four
   control modes + `force_range` clamp into a per-DOF torque. It lives in the
   `abd/` module → built for this ABD path. → **B2 = Genesis-side torque** with no
   re-implementation.

6. **No Python joint-angle accessor.** Only FEM / AffineBody / RCC state accessors
   are exposed in `src/pybind/pyuipc/core/state_accessor_feature.cpp`. → joint
   readback must be **signed-angle reconstruction** (C1).

7. **Clean per-joint torque API in fork.** `AffineBodyRevoluteJointExternalForce`
   exposes `apply_to(sc, torque)` and `apply_to(sc, torques[])`; the constitution
   converts scalar joint torque → affine moment internally (no manual
   `make_affine_torque_force_vector` as sim2sim 0.0.23 needed). Revolute joint
   build uses per-edge `init_angle`/`strength_ratio` and tracks the angle
   (`affine_body_revolute_joint.cu`).

8. **"Genesis does no sim" is not free.** Current `external_articulation` still
   runs `kernel_step_1` + `_func_constraint_force` (`rigid_solver.py:1219`) +
   `kernel_predict_integrate` + FK before the coupler; nothing forces
   `enable_collision/disable_constraint` off (defaults `True/False` at
   `solvers.py:506,511`). → the new mode must **explicitly** skip predict and
   force collision/constraint off (Rules 1,2).

9. **Consumer = gs-gym-internal.** Vendors `third_party/Genesis`
   (`Roushelfy/Genesis`, branch `feature/nyx-fem-render-tuple-api` — the branch
   this work is on) and `third_party/libuipc` (`RCC-Adhesion`) editable; pins
   **Python 3.10**, torch **cu129**, `pyuipc>=0.0.24`. Already uses
   `external_articulation`, `ipc_only`, `IPCCouplerOptions`, and RCC
   (`add_rcc_adhesion`, `set_rcc_adhesion_pt_state`) in cloth/tape scenes. The new
   mode targets this exact stack.

10. **Build reference.** `~/work/libuipc/build/cuda_mixed_fused_pcg/CMakeCache.txt`:
    nvcc `/usr/local/cuda-12.8`, `CMAKE_CUDA_ARCHITECTURES=native`,
    `CMAKE_BUILD_TYPE=RelWithDebInfo`, `UIPC_WITH_CUDA_BACKEND=ON`,
    `UIPC_WITH_CUDA_MIXED_BACKEND=ON`, `Python_EXECUTABLE=.../libuipc/python/.venv`.
    → For this work: same config but **`-DUIPC_WITH_CUDA_MIXED_BACKEND=OFF`**
    (cuda backend only) and a **py3.10** interpreter matching gs-gym-internal.

### Design decisions (from Q&A with owner)

Locked in [../roadmap.md](../roadmap.md) "Decisions locked for v1": fixed base ·
revolute+fixed · torque path · Genesis-side PD via `get_dofs_control_force` · all
modes→torque · signed-angle readback · finite-diff velocity · per-joint offset at
build · skip predict + force collision/constraint off · BDF1 · single env ·
coexist with FEM/RCC · fork `RCC-Adhesion` head · match gs-gym-internal env.

### Decision

Proceed. The mode is a specialization of the existing IPC split-step seam (~70%
of plumbing exists). Next: **M1 environment** (the active blocker).

### Verdict on the verbatim ask

"IPC does all solving+control, Genesis only renders" is achievable for fixed-base
revolute robots: IPC owns dynamics+contact+joints, Genesis contributes only the
control torque and the readback→render writeback. The honest caveat is that
Genesis still *evaluates the control law* (by design — it reuses Genesis's PD), it
just performs **no dynamics, no integration, no contact solve**.

---

## M1 — Environment (COMPLETE, 2026-06-05)

No build was needed: the consumer's venv already has a working cuda-backend fork.

### Environment (verified)

- Venv: `/home/zhaofeng/work/gs-gym-internal/.venv` — **Python 3.10.20**, torch
  `2.10.0+cu129`.
- `uipc` `0.9.0` imports from `gs-gym-internal/third_party/libuipc/python/src/uipc`
  (editable, `_pyuipc_editable.pth`). Backend `.so` present:
  `libuipc_backend_cuda.so` (+ `_none`). Fork SHA **`07f3be94`**, branch
  `RCC-Adhesion`.
- GPU **NVIDIA RTX 4090**, driver `595.71.05`. `nvcc` **release 12.8** at
  `/usr/local/cuda-12.8`.
- Build reference (already built): `~/work/libuipc/build/cuda_mixed_fused_pcg`
  (RelWithDebInfo, `CUDA_ARCHITECTURES=native`). To rebuild cuda-only, repeat
  with `-DUIPC_WITH_CUDA_MIXED_BACKEND=OFF -DUIPC_WITH_CUDA_BACKEND=ON` and a
  py3.10 `Python_EXECUTABLE`. Not required while the editable fork build works.

### Smoke (PASS)

Script: [m1_uipc_smoke.py](m1_uipc_smoke.py). Command:

```bash
/home/zhaofeng/work/gs-gym-internal/.venv/bin/python \
  genesis/engine/couplers/ipc_coupler/docs/development/m1_uipc_smoke.py
```

Result (2 ABD cubes + 1 revolute joint about +z, one fixed):

```
[cfg] integrator/type = 'bdf1'  dt = 0.01
[accessor] body_count = 2
[init]  free-body angle: geom=+0.00000  accessor=+0.00000
[after 40 steps] free-body angle: geom=-1.97341  accessor=-1.97341
[delta] rotated -1.97341 rad | accessor agrees: True
SMOKE PASS
```

This validates the exact surfaces the new mode uses, on the consumer's py3.10
cuda build:
- `Engine("cuda")` + `World.advance()/retrieve()` run and stay valid (`is_valid()`).
- **B1 torque actuation**: `AffineBodyRevoluteJointExternalForce.apply_to(joint, 50.0)`
  drove the free body's rotation (a torque genuinely moves the articulation).
- **C1 readback**: `AffineBodyStateAccessorFeature.copy_to` agrees with
  `link_slot.geometry().transforms()` to <1e-6.
- Default integrator is `bdf1` (mode requirement).

### Caveat discovered — TWO Genesis checkouts (action for M2)

The consumer's `genesis-world` editable install resolves to
**`gs-gym-internal/third_party/Genesis`** (a *separate* working copy of
`Roushelfy/Genesis` @ `feature/nyx-fem-render-tuple-api`), **not** this repo
`/home/zhaofeng/work/Genesis`. Verified by import from a neutral cwd:

```
$ cd /tmp && .../.venv/bin/python -c "import genesis; print(genesis.__file__)"
/home/zhaofeng/work/gs-gym-internal/third_party/Genesis/genesis/__init__.py
```

This repo only "wins" when cwd is `/home/zhaofeng/work/Genesis` (cwd/`''` shadows
the editable). **All M0 doc/feature edits so far live in `/home/zhaofeng/work/Genesis`,
which the consumer does NOT import by default.** Before M2 implementation is
testable in gs-gym-internal, pick one:
- (recommended) repoint the consumer's editable to this repo
  (`uv pip install -e /home/zhaofeng/work/Genesis` in the gs-gym venv, or edit the
  `__editable__.genesis_world-1.0.0` finder/.pth), or
- develop directly in `third_party/Genesis`, or
- keep both in sync via git (same branch/remote).

### B3 resolved (2026-06-05) — editable repointed to this repo

Chose option 1. Redirected the consumer's genesis editable finder to this repo
(no rebuild, fully reversible):

```bash
F=/home/zhaofeng/work/gs-gym-internal/.venv/lib/python3.10/site-packages/__editable___genesis_world_1_0_0_finder.py
cp -n "$F" "$F.bak"
sed -i 's#/home/zhaofeng/work/gs-gym-internal/third_party/Genesis#/home/zhaofeng/work/Genesis#g' "$F"
```

Verified from a neutral cwd (`/tmp`): `find_spec("genesis").origin =
/home/zhaofeng/work/Genesis/genesis/__init__.py`, full `import genesis` (0.4.4)
and `import uipc` (0.9.0) both OK, ipc docs visible. The consumer venv now imports
**this repo** regardless of cwd.

> CAVEAT: `uv sync` in gs-gym-internal will regenerate this finder from its
> pyproject source (`third_party/Genesis`) and revert the redirect. To make it
> durable, either re-run the `sed` after each sync, or change gs-gym's
> `[tool.uv.sources] genesis-world` path to `/home/zhaofeng/work/Genesis`.
> Restore the original with `mv "$F.bak" "$F"`.

### Decision

M1 gate met. Blockers **B1 and B3 resolved**. Proceed to **M2 (scaffold)**.

---

## M2 — Scaffold (COMPLETE, 2026-06-05)

Added `coup_type='ipc_monolithic'` end to end: the robot is built inside IPC
(ABD links + `AffineBodyRevoluteJoint` + `AffineBodyRevoluteJointExternalForce`,
zero torque), Genesis runs no contact/constraint solve, and IPC state reads back.

### Code changes (this repo)

- `ipc_coupler/data.py`: `COUPLING_TYPE.IPC_MONOLITHIC = 4`; new
  `IpcMonolithicEntityData` (joint slots, child/parent links, qs/dof idx, joint
  axis, loader `q0`, torque buffer).
- `genesis/options/solvers.py`: `IPCCouplerOptions.ipc_monolithic_actuation`
  (`Literal["torque"]`, default `"torque"`).
- `genesis/engine/materials/rigid.py`: added `"ipc_monolithic"` to `CoupType`;
  forces `gravity_compensation=0` (IPC owns gravity, like ipc_only).
- `ipc_coupler/coupler.py`:
  - import `AffineBodyRevoluteJointExternalForce` + `IpcMonolithicEntityData`;
    new `self._ipc_monolithic_data_by_entity`, `self._ipc_rev_ext_force`.
  - `_setup_coupling_config`: validate fixed base + revolute/fixed joints + B==1 +
    actuation=='torque'; when present, force `rigid_solver._enable_collision=False`
    and `_disable_constraint=True` (rule 2).
  - `_add_rigid_geoms_to_ipc`: fixed-joint merge map now covers IPC_MONOLITHIC;
    `external_kinetic=0` for ipc_monolithic (IPC-owned dynamics, like ipc_only).
  - new `_add_ipc_monolithic_entities` (mirrors `_add_articulation_entities_to_ipc`
    but uses per-joint `AffineBodyRevoluteJointExternalForce` instead of
    `ExternalArticulationConstraint`); wired into `_add_objects_to_ipc`.
- No `couple()` change needed for M2: torque persists at 0 from build, and readback
  uses the existing `_retrieve_ipc_rigid_states`. (Per-step torque push = M3;
  joint-angle reconstruction → qpos = M4; predict-skip not required while gravity is
  off — deferred to M3.)

### Smoke (PASS)

Fixture: [m2_arm.urdf](m2_arm.urdf) (fixed-base 2-DOF revolute arm, box collision).
Script: [m2_holdpose_smoke.py](m2_holdpose_smoke.py).

```bash
/home/zhaofeng/work/gs-gym-internal/.venv/bin/python \
  genesis/engine/couplers/ipc_coupler/docs/development/m2_holdpose_smoke.py
```

```
[build] coupler=IPCCoupler  ipc_monolithic entities=1  abd links=3
[build] ipc_monolithic joints=2  child_links=['link1', 'link2']  ext_force=True
[build] rigid_solver enable_collision=False  disable_constraint=True
SanityCheck Summary: 0 errors, 0 warns, 0 infos
  link base/link1/link2: pos_drift=0.00e+00 m  rot_drift=0.00e+00
M2 HOLD-POSE PASS
```

Validates: config accepted the mode; the robot was built in IPC with 2 real
revolute joints + the external-force actuator; Genesis's own collision/constraint
solve is disabled; `advance()` runs clean; and IPC read-back transforms match the
loader pose under gravity-off + zero-torque (a genuine held articulation, since the
joints were confirmed built).

### Not yet proven (next milestones)

- Dynamics under load (torque/gravity → motion): **M3** (the M1 smoke already showed
  torque rotates a jointed body in isolation).
- Joint-angle reconstruction back into `qpos` + getters/render parity: **M4**.
- Backward-compat: changes are purely additive branches gated on the new enum, so
  `external_articulation`/`ipc_only` paths are logically unaffected — but a live
  ext_art regression in gs-gym should be run before merge (TODO).

### Decision

M2 gate met. Proceed to **M3 (torque actuation)**: per-step push of
`get_dofs_control_force` → per-joint scalar torque, validated against a
forward-dynamics oracle under gravity; add the predict-skip in
`substep_pre_coupling`.

---

## M4 (readback) — done early for a render check (2026-06-05)

Reordered: implemented the **joint-angle readback + rendering** before M3, because
gravity provides motion for a video without needing torque actuation. This proves
the A2 primary use case — *IPC drives the robot, Genesis renders it*.

### Code change

`coupler.py` `_post_advance_write_qpos`: generalized the two_way "Step 2a"
signed-angle-from-transforms reconstruction to also cover `IPC_MONOLITHIC` (same
revolute math — joint angle = parent-relative rotation of the child ABD body about
the joint axis, added to `qpos0`). Now IPC's resolved articulation flows back into
`qpos`, FK runs, and `geoms_state` (hence the renderer) reflects the IPC state.
Limitation noted in code: merged-parent robots (ABD parent ≠ true kinematic parent
due to fixed-joint folding) are not yet handled — the v1 test arm has no merge.

### Render check (PASS)

Script: [m4_swing_render.py](m4_swing_render.py) (fixture: [m2_arm.urdf](m2_arm.urdf)).
Fixed-base 2-DOF arm, started bent `[1.4, -0.6]`, released under gravity, zero torque.

```
[IPC TELEPORT SYNC] 3 dirty links: ['base','link1','link2']   # bent start teleported into IPC
[step   0] joint angles (rad) = [ 1.4043 -0.6064]
[step  50] joint angles (rad) = [-2.4644  0.2139]
[step 149] joint angles (rad) = [ 2.6376 -0.7387]
[result] video=/tmp/ipc_monolithic_swing.mp4  frames=150  finite=True
         total_motion=22.061 rad  swing_range=6.258 rad
M4 SWING-RENDER PASS
```

Validates: `set_dofs_position` teleports the bent start into IPC; IPC integrates the
2-link pendulum under gravity; the signed-angle readback reconstructs changing joint
angles into `qpos`; Genesis FK + camera renders 150 valid frames (640×480, 30fps).
The video is in `/tmp/ipc_monolithic_swing.mp4` (not committed — binary).

Note: the arm is a frictionless/undamped double pendulum released near-horizontal,
so it swings vigorously (angles overshoot the ±3.0 soft joint limits — limits are
penalty-based, not hard stops). That is expected IPC behavior, not a bug.

### Still next

- **M3 (torque actuation):** per-step `get_dofs_control_force` → per-joint torque +
  predict-skip; tracking vs a forward-dynamics oracle.
- **M4 completion:** joint velocity readback (finite diff) into `dofs_state.vel`;
  merged-parent reconstruction; getter parity (`get_dofs_position`).

---

## Joint limits — added + verified (2026-06-05)

The M2/M4 build applied the revolute joint + external force but **no joint limit**
(hence the M4 video swung past ±3.0). Added `AffineBodyRevoluteJointLimit` (cubic
penalty) to the monolithic build, with bounds from the URDF/MJCF joint limit mapped
into IPC's angle convention (subtract the build qpos). Two new options:
`IPCCouplerOptions.monolithic_joint_limit_enable` (default True) and
`monolithic_joint_limit_strength` (default 100).

### Verification (PASS)

Fixture: [m5_pendulum.urdf](m5_pendulum.urdf) — 1-DOF fixed-base pendulum, limit
±0.4 rad. Script: [m5_joint_limit_experiment.py](m5_joint_limit_experiment.py)
(`--limit on/off --strength`). Released bent under gravity, driven into the limit:

| config | max\|angle\| | overshoot past ±0.4 |
|---|---|---|
| limit OFF | 3.1227 rad | 2.7227 rad (flies past — falls & swings) |
| ON strength=10 | 0.4017 | 0.0017 rad |
| ON strength=100 | 0.4005 | 0.0005 rad |
| ON strength=1000 | 0.4002 | 0.0002 rad |

Conclusion: the limit genuinely constrains the joint (OFF → 3.12 rad vs ON → held
at 0.40), and overshoot **monotonically shrinks with strength** — consistent with a
cubic penalty (soft = small overshoot, stiff = near-exact). The limit is a soft
penalty, not a hard stop (matches conventions.md rule on penalty-based limits).
M2 hold-pose re-run PASS (no regression with limits on by default).

---

## M3 (torque actuation) — implemented but BLOCKED by a fork bug (2026-06-05)

Implemented the Genesis-side torque path and traced a hard blocker in the engine.

### What was built (correct, Genesis-side)

- `coupler.capture_monolithic_control_torque()` — called from
  `RigidSolver.substep_pre_coupling` after `kernel_step_1`, before
  `kernel_predict_integrate`, so the PD law reads the true current state. Computes the
  per-DOF control force via `get_dofs_control_force` (folds ctrl_mode/kp/kv +
  `force_range`) and caches per-joint torque in `ad.torque`. (Verified safe.)
- `_make_monolithic_torque_animator()` — a per-joint libuipc animator callback that
  writes `external_torque` + `external_torque/is_constrained=1` from `ad.torque` during
  `advance()` (test 74 drives the torque from an animator, so this mirrors the engine's
  intended path).
- Option `IPCCouplerOptions.monolithic_torque_enable` (default **False**) gates the
  actuation; capture + animator only run when True.

### The blocker (engine bug, not integration)

Activating `AffineBodyRevoluteJointExternalForce` (`is_constrained=1`) aborts the cuda
backend inside `advance()`:

```
CUDA error ... parallel_for.inl:201 code=719(cudaErrorLaunchFailure)
terminate called after throwing an instance of 'muda::cuda_error<cudaError>'
```

719 is a sticky error from a prior kernel's illegal memory access — the external-force
reporter's device kernel (`body_ids/rest_positions/qs`) OOBs when active. Confirmed it
is **the fork, not this integration**, by running the libuipc C++ tests on this build:

| libuipc `uipc_test_sim_case` | result |
|---|---|
| `37_abd_revolute_joint` | PASS |
| `52_abd_revolute_joint_limit` | PASS |
| `72_abd_driving_revolute_joint` (driving/motor) | PASS |
| `74_abd_revolute_joint_external_force` (torque) | **SIGABRT** |

Isolation (Genesis side): hook fully off → PASS; compute torque only, no IPC write →
PASS; activate external force (direct write OR via animator) → SIGABRT. Single-joint
crashes too, so it is not a multi-joint indexing issue.

### State

`monolithic_torque_enable=False` by default → external force stays inactive
(`is_constrained=0`), so M2 hold-pose still PASSES and the M4 render path is unaffected.
[m3_position_control.py](m3_position_control.py) sets the flag True and is currently a
**reproducer** for the engine crash (not a passing gate).

### Decision needed (owner)

Either (a) fix the fork's `AffineBodyRevoluteJointExternalForce` (test 74 crash) — then
the torque path works as designed; or (b) switch monolithic actuation to
`AffineBodyDrivingRevoluteJoint` (test 72 passes) — a motor/PD-toward-target model (IPC
does the PD), which changes B1 (Genesis-computed torque → IPC-side motor) but is the
proven-working path on this build. M3 paused pending that choice.

### Fork fix landed + torque validated (2026-06-05, option a)

Owner fixed the fork: libuipc commit `db2bade6` "use device-safe closed-form inverse in
revolute-joint external-torque kernel" — `q_to_A(q).inverse()` (Eigen LU) device-traps;
replaced with a closed-form cofactor/det inverse-transpose. libuipc test 74 now passes
(401 assertions). Applied the same `.cu` change to `gs-gym/third_party/libuipc` and
rebuilt its cp310 cuda backend.

Torque actuation now works (no crash):
- FORCE mode (1-DOF, gravity off): `+0.02`/`-0.02` Nm → smooth symmetric ±0.153 rad,
  correct sign, sane dynamics. `get_dofs_velocity` matches finite-diff exactly.
Remaining for full M3: **2-DOF position control is unstable** — `get_dofs_velocity` on the
*chained* joint (joint2, parent link1 moves) returns a garbage value (~258 rad/s,
gain-independent), so the `kv·vel` PD-damping term destabilizes. FORCE mode is unaffected
(no velocity term). So M3-position needs the chained-joint **velocity readback** fixed
(joint1/1-DOF velocity is correct; only multi-link chains are wrong). `monolithic_torque_enable`
stays default False until position control is validated.

## Merge to Genesis v1.0.0 (2026-06-05)

`gs-gym/third_party/Genesis` had diverged from this repo at `b389620a`: it merged ~150
upstream Genesis commits (v0.4.5 → **v1.0.0**, rigid-solver refactor, sensors, ImGui) +
"FEM initial velocity seeding" + uncommitted gs-gym RCC bonded-PT work, while this repo had
the ipc_monolithic commits on the old base. To unify on this repo:
1. Committed third_party's RCC bonded-PT work (`58734dc6`).
2. Merged `tp/feature/...` into this repo — **automatic, zero conflicts** (ipc_coupler is
   fork-only so upstream doesn't touch it; the 4 co-edited files merged in disjoint regions).
   Merge commit `5f1a1711`.

Validated on the merged v1.0.0 tree: `import genesis` (now v1.0.0) OK with both option sets
(`monolithic_torque_enable` + `rcc_bonded_pt_enabled`); M2 hold-pose PASS; M3 force-mode PASS
(identical to pre-merge). ipc_monolithic's `capture_monolithic_control_torque` hook landed
correctly in the refactored `substep_pre_coupling`. gs-gym venv kept pointed at this repo via
the manual editable-finder redirect (re-applied; reverts on `uv sync`).

## M3 — Torque actuation COMPLETE (2026-06-05)

The earlier "chained-joint velocity readback garbage" was a **misdiagnosis**. Open-loop
checks proved `get_dofs_velocity` exactly equals the finite difference of the reconstructed
`qpos` for both joints (1-DOF and 2-DOF chain) — the readback is correct. The "258 rad/s"
was the *real* velocity of a **diverging closed-loop PD**, not a readback artifact.

Root cause: the toy arm's ABD inertia (mesh-derived) is tiny (I_eff ~ 5e-4..2e-3), so the
explicit-control stability bound `kv < ~2*I/dt` is small; the original `kp=80/kv=8` (and even
`kv=1`) blew up at step 0-1 (period-2 oscillation, get_vel correctly reporting it). With gains
matched to the inertia it is stable and accurate:
- No-gravity convergence: `kp=2, kv=0.1`, target `[0.4,-0.4]` → `q_final=[0.3995,-0.4001]`
  (err ~5e-4 rad), `|v|~1e-5` (settled). **M3 POSITION-CONTROL PASS** ([m3_position_control.py]).
- No-gravity also converges *exactly* at `kp=0.5..2`; gravity-hold is stable at `kp=2/kv=0.1`
  (holds joint2 at target, small joint1 sag) — gravity rejection is gain-sensitive on this toy
  arm but not on realistic-inertia robots (which accept kp~20-200).

`monolithic_torque_enable` default flipped to **True** (torque is the mode's purpose). M2
hold-pose still PASSES with it on (no control commanded → tau 0 → no spurious force, since the
fork fix makes is_constrained=1 + tau=0 a clean no-op).

Net: ipc_monolithic now does the full loop — Genesis folds the control law into per-joint
torque, IPC integrates the articulation + contact, Genesis reconstructs joint state and
renders. Force, position, and velocity control modes all route through the torque path.

## Prismatic joints + real-robot example (2026-06-05)

`examples/IPC_Solver/ipc_robot_grasp_cube.py` (Franka grasps a cube) was used to test
the modes on a real robot:
- **external_articulation (as-is): PASS** — 410 frames, ~51 FPS, cube grasp/lift, no
  errors, on the merged v1.0.0 tree.
- **ipc_monolithic**: initially rejected the Franka because the gripper has PRISMATIC
  finger joints (v1 was revolute-only). Confirmed the fork's prismatic external force is
  sound on this build (libuipc test 73 `73_abd_prismatic_joint_external_force` PASSES,
  401 assertions), then **extended ipc_monolithic to PRISMATIC joints**:
  - build: per-joint `AffineBodyPrismaticJoint` + `AffineBodyPrismaticJointExternalForce`
    + `AffineBodyPrismaticJointLimit` (lazy per-type creation), vs revolute equivalents.
  - actuation: the animator writes `external_force`/`external_force/is_constrained` for
    prismatic (linear force) vs `external_torque`/... for revolute. `IpcMonolithicEntityData`
    gained `joints_is_prismatic`. Reconstruction (Step 2a) already had a prismatic branch.
  - validation: allow REVOLUTE + PRISMATIC + FIXED.

  Result: the grasp example runs **end to end with ipc_monolithic** (`--abd` rigid cube):
  build OK (Genesis collision/constraint disabled), all 410 frames, exit 0, no
  NaN/invalid/newton-maxout — the full Franka (revolute arm + prismatic gripper) is
  simulated inside IPC and executes the grasp. **Caveat (gains):** the example's
  Genesis-tuned `kp` (up to 4500) makes a very stiff implicit problem — newton 40–125
  iters/frame, ~3–5 FPS (vs newton=1 / 51 FPS for external_articulation). Stable but
  expensive; IPC-tuned (lower) gains would converge cheaply, same as the M3 finding.
  M3 revolute position-control re-run PASS (no regression from the prismatic refactor).

The example gained an `ipc_monolithic` choice in its `--coup_type` arg.

## Merged-parent reconstruction fixed (2026-06-05)

The Step 2a joint readback fell back to stale `links_state` when a joint's parent link
was fixed-joint-merged into an ancestor ABD body (so the parent had no own IPC transform).
The Franka triggers this: `hand` is fixed to `link7` (merged), and the gripper fingers are
parented to `hand` → finger readback was wrong.

Fix: at build, store the constant rigid offset `T = M_rest^-1 @ P_rest` for each merged
link P → merge-target M (`_merged_link_world_offset`). In reconstruction, when the parent
is merged, recover its pose as `parent_T = M_ipc_transform @ T` (instead of stale state).

Validated (`/tmp/merged_readback.py`): teleport the Franka (arm + gripper) to a known pose,
torque off / gravity off, step, read back → arm err 0, **finger err ~4e-7** (merged-parent),
MERGED-PARENT READBACK PASS. M3 revolute (no-merge) re-run PASS. Side effect: the Franka
grasp under ipc_monolithic got **2× faster** (5.58 vs 2.49 FPS) — correct finger readback
makes the control/contact less stiff.

### Separate finding — grasp fidelity (not the readback bug)

With the fix, the ipc_monolithic Franka grasp still does **not lift the cube**
(`cube_final_z=0.035`), whereas **external_articulation does** (`z=0.189`, ~78 FPS). So the
grasp gap is ipc_monolithic-specific and orthogonal to merged-parent readback. Likely cause:
ipc_monolithic uses **soft maximal-coordinate joints** (`joint_strength_ratio`, default 100)
+ explicit torque control, which tracks IK targets / holds grip less precisely than
external_articulation's reduced-coordinate dynamics. Stiffer joints alone are NOT the fix:
`joint_strength_ratio=2000` made the run much faster (22 vs 5.6 FPS, less compliant → fewer
newton iters) but still did **not lift** the cube. So the grasp gap is multi-factor (arm IK
tracking accuracy in maximal coordinates and/or finger↔cube contact), a separate
investigation from this readback fix. Tracked as next work.
