# Implicit-PD Joint Constitutions — Design

`AffineBodyPDRevoluteJoint` / `AffineBodyPDPrismaticJoint`: new libuipc-fork
constitutions that evaluate Genesis's PD control law **implicitly inside IPC's
Newton solve**, so `ipc_monolithic` actuation is both **faithful** (exact PD law,
real `kp`/`kv` units, velocity target) and **unconditionally stable** (no explicit
`kp·dt²/I` or `kv·dt/I` bound).

Status/phases live in [roadmap.md](roadmap.md) (§M6). Enforceable rules in
[conventions.md](conventions.md). Chronological log in
[development/pd_joints_journal.md](development/pd_joints_journal.md).

## Why (the problem this solves)

The shipped `ipc_monolithic` actuation injects Genesis's control torque as a
**constant per-body wrench** (`AffineBodyRevoluteJointExternalForce`) that IPC
holds fixed across its backward-Euler solve. That makes **both** the position
gain `kp` and the velocity gain `kv` **explicit**, applied to the **light
per-body affine inertia** (a marvin arm link ≈ 5e-3 kg·m², vs the joint-space
composite `M_crb` ≈ 1.3 the torque is computed with). So it carries
`kp·dt²/I_body < 4` and `kv·dt/I_body < 2`. The Franka (heavy links) stays inside
those bounds; the marvin gripper (light links, gains bumped 10× to kp=7200/kv=600)
diverges — whole-arm jitter, reproduced free of any policy (see journal). `external_articulation`
is immune because Genesis integrates the PD **implicitly in joint space** (effective
inertia `M_aug = M_crb + dt·diag(kv+damping) ≈ 7.3`, so `kp·dt²/M_aug ≈ 0.1`).

The fix is to make IPC evaluate the PD **implicitly** too — then there is no
explicit-gain stability bound, and the per-body-vs-composite inertia mismatch is
solved automatically because the Newton solve uses the true coupled Hessian.

## North-star principle

> The PD constitution adds a term to IPC's incremental potential whose gradient,
> at the converged state, is **exactly** Genesis's PD torque
> `kp·(q_des − q) + kv·(v_des − q̇)` evaluated at the *new* state `(qₙ₊₁, q̇ₙ₊₁)`.
> Faithfulness = the control **law** is unchanged (same `kp`, `kv`, `q_des`,
> `v_des`, units). Stability = it enters as a **convex quadratic** (SPD Hessian),
> so it cannot destabilize the Newton solve at any gain.

## Status & decision (2026-06-09): shipped via the existing driving joint — no new constitution

The implicit-PD energy below is **algebraically exactly** the existing
`AffineBodyDrivingRevoluteJoint`'s `½K(θ−θ̃)²` penalty (its spec confirms `E = (K/2)(θ−θ̃)²`,
`K = γ(m_i+m_j)`, with the same raw `atan2` angle). Two PD quadratics in the same `θ` combine
into one, so **the existing driving joint realizes faithful implicit PD exactly** when the
coupler sets, per step:

```
γ (driving/strength_ratio) = (kp + kv/dt) / (m_i + m_j)   # cancels the mass scaling → K = kp + kv/dt
θ̃ (aim_angle)             = (kp·q_des + (kv/dt)(θ_n + v_des·dt)) / (kp + kv/dt)
```

> ⚠️ **pd is OPT-IN, not the default** — it **diverges under fast motion** (the stiff
> `κ_eff=kp+kv/dt` energy makes IPC's Newton solve hit NEWTON_MAXOUT ≈ 70 s/step; see journal
> 2026-06-09). Default is `"torque"`. Use pd only for light-link **slow** manipulation where
> torque diverges; for light-link **fast** motion use `external_articulation`.

So **M6 ships as a coupler-only path** (`ipc_monolithic_actuation="pd"`, opt-in;
`_write_monolithic_pd_drive`) — no new libuipc constitution, no CUDA rebuild — reusing tested
code (libuipc test 72). **Per-DOF routing:** all position/velocity DOFs → PD folding — revolute
via `AffineBodyDrivingRevoluteJoint` (`aim_angle`), prismatic via `AffineBodyDrivingPrismaticJoint`
(`aim_distance`, `γ` halved since its energy is a two-term sum). **Only FORCE-mode DOFs use the
torque path** (`capture_monolithic_control_torque` runs in both modes, writing torque only for
FORCE-mode DOFs). The implied PD start-force is **clamped to the
joint's `force_range`** (cap `aim_angle`): faithful to Genesis's torque clamp, and required for
the Newton solve to converge on large position errors (a stiff `kp` × big initial reach error is
otherwise unbounded). A **dedicated `AffineBodyPD{Revolute,Prismatic}Joint` constitution is
deferred** (design sketched below) — it would only add a kp/kv-direct API (coupler not needing
link masses); the folding already covers force_range + v_des, so it is not needed.

**Validated (2026-06-09, exact `m_i+m_j` calibration):**
- 1-DOF hinge: pd path tracks (RMS 0.008–0.023, max\|q\|≈amp) and **agrees with Genesis PD** —
  cross-RMS vs `external_articulation` = **0.0095 rad** at kp=2000 (identical max\|q\|); at
  moderate gains pd tracks *tighter* (RMS 0.023 vs 0.073) because its kp is implicit, not worse.
  Genesis has **no full implicit-kp integrator** (only `Euler`/`implicitfast`/
  `approximate_implicitfast`, all kp-explicit), so there is no bit-exact reference — the pd path
  is the faithful implicit extension (same steady state, no lag, unconditionally stable).
- **pony hold-pose @ stiff kp=7200/kv=600**: explicit torque diverges (\|q̇\| 0.02→7.7); the pd
  path **settles** (0.10→0.0001). Whole-arm jitter fixed.

The math, the deferred-constitution design, and the test ladder below remain the reference.

## The math

Let `θ(x)` be the joint coordinate as a function of the two affine bodies'
12-DOF transforms `x` (revolute: signed angle about the axis; prismatic: signed
displacement along the axis). For an `ipc_monolithic` robot `θ` *is* the Genesis
DOF `q` for that joint (same joint, calibrated zero offset). Per driven joint,
add to IPC's per-step incremental potential:

```
E_PD(x) = ½·kp·( θ(x) − q_des )²            ← position stiffness (kp)
        + ½·(kv/dt)·( θ(x) − θₙ − v_des·dt )²   ← Rayleigh damping (kv), velocity-target aware
```

- `q_des`, `v_des` — per-DOF position / velocity targets from the control law
  (`v_des` defaults to 0 but is fully supported and need not be 0).
- `θₙ` — joint coordinate at the **start** of the substep (= Genesis `qₙ`).
- `dt` — substep dt. `kp`, `kv` — the Genesis PD gains for this DOF.

The damping term is the discrete Rayleigh dissipation `½·kv·dt·(q̇ − v_des)²`
with backward-Euler `q̇ₙ₊₁ = (θₙ₊₁ − θₙ)/dt`, expanded to a quadratic in `θ`.

**Gradient (the generalized force IPC applies), at the Newton solution θ = θₙ₊₁:**

```
f = −∂E_PD/∂θ = kp·(q_des − θₙ₊₁) − (kv/dt)·(θₙ₊₁ − θₙ − v_des·dt)
              = kp·(q_des − qₙ₊₁) + kv·(v_des − q̇ₙ₊₁)
```

— **exactly Genesis's PD law, evaluated implicitly at the new state.** (Gravity /
Coriolis are not in this term; IPC produces them from the affine-body dynamics,
same as today. The explicit external-force torque path is dropped for PD-driven
DOFs.)

**Hessian:** `∂²E_PD/∂θ² = kp + kv/dt > 0` — constant, SPD. Chained through
`∂θ/∂x` it contributes SPD blocks to the global Hessian, so the Newton system
stays SPD for **any** `kp, kv ≥ 0`. No gain-dependent stability bound.

### Relationship to Genesis integrators (fidelity)

Making both `kp` and `kv` implicit is the full-implicit-PD form. **Genesis ships no
full implicit-kp integrator** — only `Euler`, `implicitfast`, `approximate_implicitfast`,
all of which treat `kp` **explicitly** (`implicitfast`/`approximate_implicitfast` fold only
`kv`). So there is no bit-exact Genesis reference. What holds:

- **Steady state is integrator-independent**: `f = 0 ⇔ q = q_des ∧ q̇ = v_des`. Identical
  equilibrium/tracking to any Genesis integrator (verified: identical max\|q\|).
- **In the regime where Genesis is stable** (moderate gains) the pd path agrees with
  `external_articulation` closely (cross-RMS 0.0095 rad at kp=2000; tighter tracking at low
  gains because implicit `kp` removes lag).
- **Beyond that regime** (stiff gains on light bodies) the explicit-kp Genesis path / explicit
  torque diverges while the pd path stays stable — this is the whole point.

## Deferred: a dedicated `AffineBodyPD{Revolute,Prismatic}Joint` constitution

Not implemented (the folding above is exact and covers force_range + v_des). It would only add a
**kp/kv-direct API** so the coupler need not fold gains/masses: a constitution composing onto the
base revolute/prismatic joint (like `AffineBodyDrivingRevoluteJoint`) with edge attributes
`pd/kp, pd/kv, pd/aim_angle, pd/aim_velocity, pd/is_constrained`, energy `½K(θ−θ̃)²` reusing the
driving joint's symbolic `E/G/H` with `K=kp+kv/dt`, `θₙ` from the base `current_angles`, `dt`
from `info.dt()`. Build it only if folding-in-the-coupler becomes a maintenance burden.

## Shipped path: implicit PD via the existing driving joint (coupler-only)

| | joint coord `θ(x)` | reuses |
|---|---|---|
| `AffineBodyPDRevoluteJoint` (hinge) | signed angle about axis | `AffineBodyRevoluteJoint` connectivity, `current_angles`, `l_basis/r_basis`, `F01` symbolic block |
| `AffineBodyPDPrismaticJoint` (slide) | signed displacement along axis | `AffineBodyPrismaticJoint` connectivity + displacement coord |

The PD constitution **composes with** the existing connectivity joint (which keeps
the links attached + enforces the axis), exactly as `AffineBodyDrivingRevoluteJoint`
does — it only adds the actuation energy. Per-joint device attributes:
`pd/kp`, `pd/kv`, `pd/q_des`, `pd/v_des`, `pd/is_constrained`; `θₙ` is read from the
joint's `current_angles`.

## De-risk prototype (Phase 0 — coupler-only, no fork rebuild)

Two quadratics in `θ` combine into one:

```
½·kp·(θ−q_des)² + ½·(kv/dt)·(θ−θₙ−v_des·dt)²  =  ½·κ_eff·(θ − aim_eff)² + const
    κ_eff   = kp + kv/dt
    aim_eff = ( kp·q_des + (kv/dt)·(θₙ + v_des·dt) ) / κ_eff
```

`AffineBodyDrivingRevoluteJoint` computes `½·K·(θ−θ̃)²` with `K = γ·(m_i+m_j)` (the SUM of the
two affine-body masses — per its spec). So the **exact PD gradient** is reproduced with no C++
change by setting, each step (`coupler._write_monolithic_pd_drive`, gated by
`ipc_monolithic_actuation="pd"`):

```
γ (driving/strength_ratio) = κ_eff / (m_i + m_j)     # cancels the mass scaling → K = κ_eff
aim_angle                  = aim_eff
driving/is_constrained     = 1
```

reading `θₙ` from Genesis `qₙ`, `dt` from `substep_dt`. Prismatic (slide) DOFs use the same scheme
via `AffineBodyDrivingPrismaticJoint` (`aim_distance`, `γ = κ_eff/(2(m_i+m_j))` — its energy is a
two-term sum → joint stiffness `2K`). Only FORCE-mode DOFs keep the torque path. This is **exact**
(the merge is algebraic, not approximate) and is the **shipped** M6 actuation — the earlier "m_parent" form
was a calibration bug, now `m_i+m_j`. Remaining gap vs a dedicated constitution: no `force_range`
torque clamp, and the coupler must know the two link masses (folded into `γ`).

## `force_range` clamp (open)

Genesis clamps τ to `[−f, f]`. In an energy framework this is a potential whose
gradient saturates (quadratic core + linear tails beyond the force limit; a
Huber-like potential). Plan: ship unclamped first (a well-tuned PD rarely
saturates in free motion), add the saturating potential as a follow-up. Tracked
as an open question.

## Validation ladder (rule → test)

| Level | Test | Enforces |
|---|---|---|
| **Unit (CPU oracle)** | combine-quadratic algebra: `κ_eff`/`aim_eff` reproduces `kp·(q_des−θ)+kv·(v_des−θ̇)` over sampled `(θ,q_des,v_des,kp,kv,θₙ,dt)` | PD-law correctness |
| **Unit (symbolic)** | constitution `E/G/H` vs finite-difference of `E_PD(x)` on random transform pairs (incl. `v_des≠0`) | gradient/Hessian correctness |
| **Contract** | building a PD joint creates `pd/*` attributes; FORCE-mode DOFs get **no** PD joint (stay on torque); SPD Hessian (positive `kp+kv/dt`) | API + composition |
| **Live, GPU (shipped path) ✅** | 1-DOF hinge: pd path tracks (RMS 0.008–0.023, max\|q\|≈amp) + agrees with Genesis PD (cross-RMS 0.0095 @ kp=2000); pony hold-pose settles (0.10→0.0001) where torque diverges (→7.7) | DONE 2026-06-09 |
| **Live, GPU (deferred constitution)** | dedicated `AffineBodyPD*`: 2-DOF (hinge+slide) + `v_des≠0` ramp + Franka EE parity | only if built |
| **Scene (live, GPU)** | pony / hang_cloth full policy run: robot stable, deformable contact intact | follow-up |
| **Source-scan** | PD-driven pos/vel DOFs route through `AffineBodyDriving{Revolute,Prismatic}Joint` folding; only FORCE-mode DOFs stay on `…ExternalForce` | actuation routing |

## Phase plan

- **P0 — Implicit PD via driving-joint folding (coupler-only): ✅ DONE / SHIPPED (2026-06-09).**
  `ipc_monolithic_actuation="pd"` + `_write_monolithic_pd_drive` (revolute, exact
  `γ = κ_eff/(m_i+m_j)`). Gate PASS: 1-DOF stable + tracks + agrees with Genesis PD
  (cross-RMS 0.0095 at kp=2000); pony hold-pose settles (0.10→0.0001) where torque diverges.
  This **is** the M6 solution — no new constitution, no CUDA build.
- **P1–P4 — Dedicated `AffineBodyPD{Revolute,Prismatic}Joint` constitution: DEFERRED.** Only
  needed if a `force_range` torque clamp or a `kp/kv`-direct API (coupler not knowing link
  masses) becomes necessary. Would be: revolute constitution (C++/CUDA + pybind, reusing the
  driving-joint symbolic `½K(θ−θ̃)²`) → prismatic → coupler routing → clamp. The libuipc spec
  `affine_body_pd_revolute_joint.md` (UID 31) is written; implementation not started.

Remaining polish on the shipped path (not blocking): expose `pd` through gs-gym's
SceneArgs so envs can opt in without patching defaults; optional `force_range` handling for the
PD DOFs (currently unclamped — fine for tracking, FORCE-mode keeps the clamped torque path).

## Open questions

1. `force_range` clamp formulation (Huber potential vs post-hoc gradient cap).
2. Per-step vs build-time `kp/kv` (RL uses fixed gains → build-time fine; expose a
   per-step update path if needed).
3. Confirm no gravity/Coriolis double-count when the explicit torque is dropped for
   PD DOFs (IPC already applies body gravity; the old path's `qf_bias` bookkeeping
   is removed for these DOFs).
4. `body_mass` normalization in the prototype — which body's mass the existing
   driving joint scales by (affects `κ_eff` calibration); irrelevant once the real
   constitution uses `kp` directly.
