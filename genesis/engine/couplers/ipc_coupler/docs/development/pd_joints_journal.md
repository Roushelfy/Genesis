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

## 2026-06-09 — Extended implicit PD to prismatic (slide) joints

Slides now use `AffineBodyDrivingPrismaticJoint` (was: prismatic stayed on torque). So in `"pd"`
mode **all position/velocity DOFs** (revolute + prismatic) are implicit-PD; only FORCE-mode DOFs
remain on the torque path.
- Build: apply `AffineBodyDrivingPrismaticJoint` to prismatic joints (alongside the base
  `AffineBodyPrismaticJoint`); revolute → `AffineBodyDrivingRevoluteJoint` as before.
- `_write_monolithic_pd_drive`: handles both — prismatic writes `aim_distance` with
  `strength_ratio = κ_eff/(2(m_i+m_j))` (the prismatic energy is a **two-term** sum → joint
  stiffness `2K`, vs revolute's single term `K`, per the constitution spec); revolute writes
  `aim_angle` with `κ_eff/(m_i+m_j)`. Same `aim_eff`, force_range clamp, q_n-from-Genesis-qpos.
- `capture` routing now zeros torque for **all** ctrl_mode≤VELOCITY DOFs (dropped the
  `not prismatic` guard); only FORCE-mode keeps torque.

Validated: Franka grasp (gripper = prismatic, now driving-prismatic) lifts cube to **0.196**
(torque 0.188, revolute-only-PD 0.177) — gripper closes correctly, so `aim_distance` frame is
consistent with Genesis qpos. Pony hold-pose (arm+gripper both PD) settles **0.10→0.0000**.

This closes the deferred "C": torque path now serves only FORCE mode. (Follow-up still possible:
the coupled implicit-damping solve in `capture` exists for kv — now only FORCE-mode DOFs use it,
where kv=0, so it could be simplified to a plain force pass-through. Left as-is for now.)

## 2026-06-09 — pd FAILS under fast motion; reverted default to "torque"

The `ipc_coupling_perf_bench` (Franka, large fast sinusoid, max|q̇|~12 rad/s) exposed a hard pd
failure: at the fast part of the motion the IPC Newton solve **diverges** —
`Line Search Exits Max Iteration 8` every iter, `Newton Iteration Exits Max Iteration 1024`,
residual GROWS (0.6→2.1), `advance ≈ 72 s/step`, then SIGABRT. Not coupler overhead
(`pre_coup=0.4 ms`); it's the IPC solve.

**Cause:** the implicit-damping folding makes the driving-joint energy very stiff —
`κ_eff = kp + kv/dt` is dominated by `kv/dt` (2000 + 100/0.01 = 12000). The spring pulls θ toward
`aim_eff ≈ θ_n` with that stiffness. Under fast motion the inertial prediction `x̃ = x_n + dt·v_n`
moves far while the stiff spring resists → ill-conditioned incremental potential → IPC's
CCD-limited line search can't progress → Newton diverges. Slow/hold tests (grasp, pony) never hit
it (spring stays near equilibrium). **Same energy stiffness exists in the deferred dedicated
constitution** (the `(kv/dt)(θ−θ_n)²` term) — it's inherent to encoding velocity damping as a
position-increment quadratic, not a folding artifact.

**Confirmed contrast (same fast motion, Franka):**
- torque: `newton=5`, `advance=20 ms/step`, no maxout — stable + fast.
- pd: `newton=1024` (maxout), `advance≈72 s/step`, diverges → abort.

So pd is **not** a strict improvement: it fixes light-link stiff-gain *stability* (pony, slow) but
**regresses fast motion** (which torque handles). No monolithic actuation is universally best:
- heavy links / fast motion → **torque** (fast, stable);
- light links + stiff gains + slow manipulation → **pd** (torque diverges);
- light links + fast motion, or max robustness → **external_articulation** (robust, slower).

**Action:** reverted the default to `"torque"` (restores the validated fast-motion behavior; pd is
opt-in). Open question: a kp-only driving joint (`K=kp`, drop the `kv/dt` stiffness, rely on
backward-Euler numerical damping) might converge under fast motion at the cost of kv fidelity —
not yet tried.

## 2026-06-09 — δθ incremental angle: pd_eac prototype PROVES the fix (no build), then dedicated constitution

**Insight (owner):** the absolute-angle ±π branch cut that makes `pd` diverge under fast motion is
exactly what `ExternalArticulationConstraint` (EAC) avoids — it optimizes the *incremental* angle
`δθ = atan2(sinθ·cosθᵗ − cosθ·sinθᵗ, …)` against the previous-step frame, so δθ stays near 0 and
never hits the cut. Mathematically equivalent to the absolute form wherever both are well-defined.

**Prototype `ipc_monolithic_actuation="pd_eac"` (coupler-only, NO fork build):** deliver the implicit
PD through EAC with a DIAGONAL mass = `diag(κ_eff)` and `delta_theta_tilde = aim_eff − q_n`. Bodies
stay `external_kinetic=0` (IPC keeps their inertia), so the EAC term is a pure control-stiffness
penalty — no double-count. `ref_dof_prev` pinned to the start-of-step state.

**Validated (the whole point):**
| test (Franka kp2000/kv100) | torque | pd (abs θ) | pd_eac (δθ) |
|---|---|---|---|
| fast motion @ peak vel (frame 51) | newton 3–8 ✅ | NEWTON_MAXOUT 1024, 72 s → SIGABRT ❌ | newton=3, 20 ms ✅ |
| pony hold @ stiff kp×10/kv×10 | →6 rad/s jitter ❌ | 0.0000 ✅ | **0.0000 ✅** |
| hold q=0 vs gravity (max\|q\|) | 0.057 | 0.0008 | 0.0004 |

`pd_eac` is the **only mode stable on BOTH the pony (light-link stiff gains) and fast motion** — the
δθ idea is fully validated, reusing the already-compiled EAC constitution.

**Tracking lag — investigated:** 1-DOF (no clamp) RMS ≈ 3× pd's (e.g. 0.069 vs 0.022 @ 0.5 Hz),
*frequency-independent ratio*. NOT the reference (pinning `ref_dof_prev` → identical RMS to 5 digits;
`q_prevs` was already correct), NOT damping (kv÷3 barely moved it). Theory-fit vs the implicit-PD
model: pd_eac matches the model but at an **inertia that inflates ~4× at high gains** (kp2000: I_fit
0.20 vs pd's 0.05), while matching pd exactly at low gains. Root mechanism (energy/gradient should be
identical at convergence) remains analytically unexplained — empirically pd_eac is "heavier" at high
κ_eff. Modest in practice: RMS/amp ~3–5 %, and the pony still holds at 0.0000.

**Decision (owner):** build the dedicated **`AffineBodyPDRevoluteJoint` (UID 33) / `AffineBodyPDPrismaticJoint`
(UID 34)** — a clean per-edge `½·κ_eff·(δθ − aim_increment)²` constraint reusing EAC's δθ kernels, with
a **Gauss-Newton-only Hessian** (`κ_eff·J·Jᵀ`, no second-order `make_spd` term). Goals: (a) clean
kp/kv semantics, no EAC mass-as-stiffness reuse; (b) self-contained branch-cut robustness in
monolithic; (c) the GN Hessian is the lever to test whether it closes the high-gain inertia inflation.

Committed: pd_eac path `16666e65` + ref_dof_prev `ad6a41c0`; gs-gym env knob `27befcb`
(`GS_GYM_IPC_MONO_ACTUATION`).
