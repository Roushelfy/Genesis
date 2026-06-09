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

### Relationship to Genesis integrators (fidelity contract)

Making both `kp` and `kv` implicit = Genesis's **`gs.integrator.implicit`**
semantics (the default `approximate_implicitfast` has `kv` implicit but `kp`
**explicit**). Consequences:

- **Steady state is integrator-independent**: `f = 0 ⇔ q = q_des ∧ q̇ = v_des`.
  Identical equilibrium/tracking to any Genesis integrator.
- **Transient differs only in `kp` damping** vs `approximate_implicitfast`
  (implicit `kp` is slightly more damped). To compare bit-for-bit, run the
  `external_articulation` reference with `integrator=gs.integrator.implicit` —
  then monolithic-PD and external share the same controller exactly.

## The two constitutions

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

The shipped `AffineBodyDrivingRevoluteJoint` already computes `½·κ·(θ−aim)²` with
`κ = strength_ratio · body_mass`. So we can reproduce the **exact PD gradient**
without any C++ change by, each step, setting
`strength_ratio = κ_eff / body_mass` and `aim_angle = aim_eff` (reading `θₙ` from
Genesis `qₙ`). This validates the *math* (does implicit PD fix the marvin and
match `integrator=implicit`?) before investing in the constitution.
**Prototype only** — caveats: no `force_range` clamp, `body_mass` normalization
bookkeeping, conflated kp/kv semantics. Not the shipped path.

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
| **Prototype (live, GPU, coupler-only)** | 1-DOF hinge at stiff `kp/kv`: stable + tracks Genesis `integrator=implicit` `q(t)`; marvin/pony hold-pose no longer diverges | the math fixes it |
| **Integration (live, GPU)** | real constitution: 1-DOF + 2-DOF (hinge+slide) match `external@implicit`; `v_des≠0` ramp tracks; Franka EE trajectory parity vs external | constitutions correct |
| **Scene (live, GPU)** | pony / hang_cloth full run: robot stable, deformable contact intact, no whole-arm jitter | end-to-end |
| **Source-scan** | PD-driven DOFs route through the PD constitution; FORCE-mode DOFs still through `…ExternalForce`; no explicit per-body PD wrench on PD DOFs | actuation routing |

## Phase plan

- **P0 Prototype (coupler-only):** drive the existing `AffineBodyDrivingRevoluteJoint`
  with `κ_eff`/`aim_eff`. Gate: stiff-gain 1-DOF stable + matches `integrator=implicit`;
  marvin/pony hold-pose stops diverging.
- **P1 Revolute constitution:** `AffineBodyPDRevoluteJoint` (C++/CUDA + regenerated
  symbolic `.inl` with the damping term + pybind + python). Gate: unit symbolic test +
  1-DOF live parity.
- **P2 Prismatic constitution:** `AffineBodyPDPrismaticJoint`. Gate: 2-DOF hinge+slide
  parity (incl. `v_des≠0`).
- **P3 Coupler integration:** route monolithic position/velocity DOFs to the PD joints;
  keep FORCE-mode on the torque path. Gate: Franka EE parity vs `external@implicit`,
  no regression on the existing grasp example.
- **P4 Clamp + polish:** saturating-potential `force_range`; update conventions Rules 3–4
  + source-scan; pony/marvin scene gate; commit.

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
