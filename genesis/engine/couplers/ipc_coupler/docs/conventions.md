# IPC-Monolithic Coupling Mode — Conventions & Validation Ladder

Enforceable rules for the `ipc_monolithic` coupling type, and the tests that
check them. See [architecture.md](architecture.md) for design and
[roadmap.md](roadmap.md) for status.

## Rules (each must be checkable in code or a test)

1. **No Genesis integration.** `kernel_predict_integrate` is not called for any
   `ipc_monolithic` entity's DOFs. Genesis produces control torque only; IPC
   integrates dynamics.
2. **No Genesis contact/constraint solve.** When any `ipc_monolithic` entity
   exists, the rigid solver runs with `enable_collision=False` **and**
   `disable_constraint=True`. The coupler asserts this at build (or sets it and
   logs once) — otherwise `_func_constraint_force` ([rigid_solver.py:1219])
   double-solves contacts against IPC.
3. **Actuation channel by control mode (revised M6).** Two opt-in modes via
   `IPCCouplerOptions.ipc_monolithic_actuation`:
   - `"torque"` (default): per-joint scalar torque via `AffineBodyRevoluteJointExternalForce`
     / prismatic variant (kp/kv **explicit** — diverges on light links at stiff gains).
   - `"pd_prototype"`: **position/velocity revolute DOFs** are driven by the existing
     `AffineBodyDrivingRevoluteJoint` as an *implicit* PD servo (the coupler folds
     `kp,kv,q_des,v_des,θₙ,dt` into `γ=(kp+kv/dt)/(m_i+m_j)`, `aim_angle=θ̃`, each step in
     `_write_monolithic_pd_drive`; `ad.torque` stays 0 so no wrench is written). **FORCE-mode
     and prismatic DOFs keep the torque path** (low/stable gains; no kp/kv to fold).
   - Still forbidden for these entities: `ExternalArticulationConstraint`,
     `SoftTransformConstraint`. (Pre-M6: torque was the *only* channel; it held for the Franka
     but diverges for light-link robots at stiff gains — see §M6 / [pd_joints.md](pd_joints.md).)
   - A dedicated `AffineBodyPD{Revolute,Prismatic}Joint` constitution is **deferred** (would
     only add `force_range` clamp / a kp/kv-direct API); the driving-joint folding is exact.
4. **The PD law stays Genesis's; only its evaluation moves.** Gains (`kp`,`kv`), targets
   (`q_des`,`v_des`), `ctrl_mode` come from Genesis (`act_bias` / `dofs_state.ctrl_pos/ctrl_vel`).
   In `"pd_prototype"` the folded `γ`/`aim_angle` reproduce `kp·(q_des−q)+kv·(v_des−q̇)` exactly
   at the converged state — the same controller, evaluated implicitly. (`force_range` is not
   yet clamped on PD DOFs; FORCE-mode DOFs retain the clamped torque path.)
5. **Readback is reconstruction.** Joint angle = signed-angle-about-axis from
   parent/child body transforms minus the per-joint build offset; joint velocity
   = finite difference `(qᵗ − qᵗ⁻¹)/dt` (zero on first step). No native joint
   accessor is used (none is exposed to Python).
6. **Integrator is BDF1.** The mode rejects (asserts) a non-BDF1 scene
   integrator.
7. **Fixed base only (v1).** A non-fixed base routed to `ipc_monolithic` is a
   build-time error (mirrors the existing `external_articulation` check at
   [coupler.py:622], which forces free-base robots elsewhere).
8. **Revolute + fixed joints only (v1).** Any other joint type on an
   `ipc_monolithic` entity is a build-time error.
9. **Single env (v1).** `B>1` with `ipc_monolithic` is rejected until the
   per-env push path is validated.
10. **Zero-offset calibration.** Each revolute joint's `init_angle` is set so the
    reconstructed angle equals Genesis `qpos` at the loader pose; a build-time
    assert checks reconstruction(loader pose) ≈ qpos within 1e-4 rad.

## Naming

- Enum: `COUPLING_TYPE.IPC_MONOLITHIC` (`data.py`).
- User selector: `RigidMaterial(coup_type="ipc_monolithic")`.
- Option: `IPCCouplerOptions.ipc_monolithic_actuation` (string).
- New coupler methods: `_add_ipc_monolithic_entities`, `_push_joint_torques`,
  `_reconstruct_joint_positions`.

## Validation ladder (rule → test)

Tests are wired into the Genesis test path (`pytest`); live-runtime tests that
need a GPU + the built fork are marked and excluded from the default gate (mirror
the `genesis_ref_runtime` style used in sim2sim).

| Level | Test | Enforces |
|---|---|---|
| **Unit (CPU oracle)** | `get_dofs_control_force` for each ctrl mode equals a NumPy PD oracle (kp/kv/force_range) | Rule 4 |
| **Unit** | signed-angle reconstruction round-trips a synthetic parent/child transform pair to a known angle; degenerate axis (‖e‖≤1e-8) raises | Rule 5 |
| **Contract** | building `ipc_monolithic` with free base / non-revolute joint / BDF2 / B>1 raises; building it flips `enable_collision`/`disable_constraint` off | Rules 2,6,7,8,9 |
| **Source-scan** | the `ipc_monolithic` path contains no `kernel_predict_integrate` and no `ExternalArticulationConstraint`; in `pd_prototype` mode PD revolute DOFs route through `AffineBodyDrivingRevoluteJoint` (folded), FORCE-mode + prismatic DOFs through `…ExternalForce` | Rules 1,3 |
| **Integration (live, GPU)** | 1-DOF revolute pendulum under constant torque: IPC-integrated angle vs an independent forward-dynamics oracle within tol; step-position target converges to setpoint | Rules 3,4; M3 gate |
| **Integration (live, GPU)** | `get_dofs_position` after a step equals reconstruction of `copy_to` transforms; loader-pose offset calibration within 1e-4 | Rules 5,10; M4 gate |
| **Scene (live, GPU)** | fixed-base arm + cloth/RCC in one scene runs N steps and renders the arm | M5 gate |

## Performance note (not a perf feature)

Primary goal is rendering coexistence, not throughput. Record only: `couple()`
wall time/step and IPC Newton iters/step (already logged at
[coupler.py:1971]). Single-env. B>1 throughput is explicitly out of scope for v1;
do not make speed claims.
