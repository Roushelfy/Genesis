# Implicit-PD Joints — Development Journal

Chronological log for `AffineBodyPDRevoluteJoint` / `AffineBodyPDPrismaticJoint`
(roadmap §M6). Stable design: [../pd_joints.md](../pd_joints.md).

## 2026-06-09 — Diagnosis that motivated M6

**Symptom (user):** gs-gym-internal `pick-deformable-toy-pony` (and marvin gripper
tasks) run with `--robot-coupling ipc_monolithic` show **whole-arm jitter**;
default coupling (`external_articulation`/`two_way`) is smooth.

**Integration audit:** the gs-gym wiring is correct — same `/home/zhaofeng/work/Genesis`
coupler, `n_envs==1` gate, fixed base, correct material kwargs, no `gravity_compensation`
for monolithic, dt/substeps unchanged. Not a wiring bug.

**Latent bug found (separate):** `ipc_monolithic` jitters with `substeps>1` —
`build_ipc_scene_config` ([utils.py:26]) sets the IPC world `dt` to the **full** dt but
`couple()→advance()` runs once **per substep**, so the world over-integrates. Envs force
`substeps=1` so it is not the pony issue, but it is real. (Tracked in memory
`ipc-monolithic-substeps-bug`.)

**Root cause (reproduced, no policy).** Held the reset pose in the real pony env and
logged arm `|q̇|`:

| config | arm `|q̇|max` by step ~50 | |
|---|---|---|
| `external_articulation`, kp=7200/kv=600 | → 0.0001 (settles) | ✅ |
| `ipc_monolithic`, kp=7200/kv=600 | → 7.7 (grows) | ❌ diverges |
| `ipc_monolithic`, kp=720/kv=600 (only kp↓) | → 2.4 | ❌ still diverges |
| `ipc_monolithic`, kp=7200/kv=60 (only kv↓) | → 58 | ❌ worse |
| `ipc_monolithic`, kp=720/kv=60 (both↓) | → 0.0006 | ✅ |

Probe of the monolithic torque solve confirmed `mass_mat` *is* augmented
(`diag(m_aug)`=7.31 at the shoulder, `m_crb`=1.31 positive — no negative inertia).
So it is **not** a mass-matrix sign bug. The instability is **explicit PD on light
per-body affine inertia**: the constant injected wrench makes both `kp` and `kv`
explicit, bounded by `kp·dt²/I_body<4` and `kv·dt/I_body<2`, with `I_body` the light
single-link affine inertia (~5e-3) rather than the composite `M_crb` (~1.3) the torque
is computed with. external integrates the PD implicitly in joint space (`M_aug≈7.3`), so
its `kp·dt²/M_aug≈0.1` is fine. Franka is stable only because its links are heavy.
Lowering kp alone does not fix it (kv also explicit) → both gains must drop → loses the
stiff tracking the 10× gains gave.

**Decision (owner):** implement faithful **implicit** PD as IPC constitutions
(`AffineBodyPDRevoluteJoint`/`Prismatic`), not lower gains and not the stiffness-only
`AffineBodyDrivingRevoluteJoint` (which is mass-scaled, has no `kv`, no clamp — would
change the controller). Support `v_des≠0`. Math + ladder: [../pd_joints.md](../pd_joints.md).

**Fork capability confirmed.** `Roushelfy/libuipc` already exposes
`AffineBodyDrivingRevoluteJoint`/`AffineBodyDrivingPrismaticJoint` to Python (verified
import), with an implicit `½·κ·(θ−aim)²` energy (`κ=strength·body_mass`), `aim_angle`
per-edge attribute, `compute_current_angles()`, and a `JointDofReporter`/`PredictDof`
system — passing libuipc test `72_abd_driving_revolute_joint`. That is the scaffolding
the PD constitutions extend (add the `(kv/dt)(θ−θₙ−v_des·dt)²` damping term + real kp/kv
units + optional clamp). History: monolithic deliberately chose the explicit-torque path
(option a) on 2026-06-05 after fixing a fork crash (`db2bade6`); §M6 revisits that for
stiff-gain light-link robots.

### Next: P0 prototype (coupler-only, no rebuild)
Drive the existing `AffineBodyDrivingRevoluteJoint` with `κ_eff=kp+kv/dt`,
`aim_eff=(kp·q_des+(kv/dt)(θₙ+v_des·dt))/κ_eff` to validate that implicit PD (a) fixes the
pony/marvin divergence and (b) matches Genesis `integrator=implicit` `q(t)` on a 1-DOF
hinge — before writing the C++ constitution. Gate in [../pd_joints.md](../pd_joints.md) §P0.

## 2026-06-09 — P0 prototype: IMPLEMENTED + PASSED (coupler-only, no fork rebuild)

Wired `ipc_monolithic_actuation="pd"` (opt-in; default stays `"torque"`):
- `solvers.py`: option extended to `Literal["torque","pd"]`.
- `coupler.py`: import `AffineBodyDrivingRevoluteJoint`; in `_add_ipc_monolithic_entities`
  compose it onto each revolute joint edge (build placeholder strength); new
  `_write_monolithic_pd_drive()` (called from `_pre_advance_write_ipc_attributes`) writes per
  step `aim_angle = aim_eff`, `driving/strength_ratio = κ_eff/m_parent`,
  `driving/is_constrained = 1`. `capture_monolithic_control_torque` already early-returns for
  non-`torque` actuation, so `ad.torque` stays 0 → no explicit wrench. Prismatic stays on torque.
- `ipc_monolithic_1dof_debug.py`: `--actuation` arg.

**Results (gate PASS):**

| test | torque (explicit) | pd (implicit) |
|---|---|---|
| 1-DOF hinge kp=100/kv=5, SHM amp 0.8 | RMS 0.024, max\|q\| 0.80 | RMS 0.024, max\|q\| 0.80 (tracks) |
| 1-DOF hinge kp=7200/kv=600 (stiff) | RMS 0.119 (sluggish, bounded\*) | RMS **0.027**, max\|q\| 0.79 (stable) |
| **pony hold-pose, arm \|q̇\| @ stiff gains** | **0.02 → 7.7 → diverges** | **0.10 → 0.0002 (settles)** ✅ |

\*the single-body 1-DOF doesn't reproduce the divergence (its M_crb/M_aug attenuation shrinks
the effective explicit kp); the divergence is a multi-body affine/penalty-joint effect, which
the pony reproduces and pd fixes. Implicit PD also tracks *better* (0.027 vs 0.119).

**Caveats (prototype, not the shipped path):** stiffness scaled by `body_mass` (driving-joint
convention) and the symbolic energy's exact κ→joint-stiffness factor are not calibrated, so a
bit-exact match to `integrator=implicit` is deferred to P1 (the real `AffineBodyPDRevoluteJoint`
uses real kp/kv units). `force_range` clamp not yet applied. P0 confirms the **mechanism**:
making the PD implicit removes the divergence and tracks faithfully. → proceed to P1.

## 2026-06-09 — Hardened to exact PD; dedicated constitution DROPPED (M6 done)

**Insight (owner):** since the PD energy collapses to the driving joint's *exact*
`½K(θ−θ̃)²`, the original `AffineBodyDrivingRevoluteJoint` can realize faithful implicit PD
directly — **no new constitution needed.** Read the libuipc spec: `E=(K/2)(θ−θ̃)²` with
`K=γ(m_i+m_j)` (SUM of both affine-body masses) and the same raw `atan2` angle — so the combine
is exact, and the only P0 imperfection was using `m_parent` instead of `m_i+m_j`. The "symbolic
κ-factor" worry was unfounded.

**Hardened P0 → exact PD:** `_write_monolithic_pd_drive` now sets
`strength_ratio = κ_eff/(m_parent+m_child)` (was `/m_parent`). Default stays `"torque"`.

**Re-validated (exact m_i+m_j):**
- 1-DOF driver (pd vs `external_articulation`): cross-RMS **0.0095 rad @ kp=2000/kv=40**
  (identical max\|q\|); @ kp=100/kv=5 cross-RMS 0.050 but pd tracks *tighter* (RMS 0.023 vs
  0.073) — implicit kp removes lag. **Genesis has no full implicit-kp integrator** (only
  `Euler`/`implicitfast`/`approximate_implicitfast`, all kp-explicit), so there is no bit-exact
  reference; pd is the faithful implicit extension (same steady state, no lag, stable).
- pony hold-pose @ kp=7200/kv=600: settles **0.10→0.0001** (torque → 7.7 diverges).

**Decision:** the driving-joint folding **is** the shipped M6 actuation. The dedicated
`AffineBodyPD{Revolute,Prismatic}Joint` constitution is **DEFERRED** — only worth building for
a `force_range` torque clamp or a kp/kv-direct API (coupler not needing link masses). The
libuipc spec `affine_body_pd_revolute_joint.md` (UID 31) is kept as that design. Docs/roadmap/
conventions re-scoped accordingly. **No CUDA build was needed.**

## 2026-06-09 — Per-DOF routing + force_range clamp + made default ("pd"); A: doc cleanup

**B — routing + clamp + default.** Made `"pd"` the **default** and added **per-DOF routing** so
it's a safe drop-in:
- `capture_monolithic_control_torque` no longer early-returns for non-`torque`; it runs in both
  modes and writes torque only for **prismatic + FORCE-mode** DOFs (PD revolute DOFs get
  `ad.torque=0`). Previously `"pd"` left the gripper (prismatic, `type="slide"`) **unactuated**.
- `_write_monolithic_pd_drive` clamps the implied start-force `κ_eff·(aim_eff−q_n)` into the
  joint's `force_range` (cap `aim_angle`). **Why:** the Franka grasp crashed IPC at frame 2 —
  the start-of-step PD force is `kp·(q_des−q_n)` regardless of kv, and `kp=4500` × a large initial
  IK reach error → unbounded stiff implicit spring → Newton/line-search non-convergence. The
  torque path survives because `get_dofs_control_force` clamps to `force_range` (Franka: ±87/±12);
  the clamp gives the PD path the same bound (faithful to Genesis). (`set_dofs_kp` does NOT zero
  kv — arm kv stayed 450 from the MJCF; the failure was the unclamped force, not kv.)
- Renamed `"pd_prototype" → "pd"`.

**Validated:**
- **Franka grasp+lift regression** (the hard case: contact + gripper + big IK jumps): default
  `pd` lifts the cube to **0.177** (torque 0.188), Newton converges, arm stable. Without the clamp
  it crashed at frame 2.
- pony hold-pose (default pd, routing + clamp): settles **0.10→0.0000** (gripper now actuated via
  torque routing).

**A — doc cleanup.** Deleted the deferred libuipc spec
`docs/specification/constitutions/affine_body_pd_revolute_joint.md` (unbuilt constitution; design
folded into the "Deferred" section of pd_joints.md). Trimmed pd_joints.md's deferred section to a
short pointer. Renamed `pd_prototype→pd` across docs. Updated roadmap/conventions to default=`pd`,
routing, and the now-shipped force_range clamp. M6 closed.
