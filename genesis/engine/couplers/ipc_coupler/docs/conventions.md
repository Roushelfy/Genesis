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
3. **Torque is the only actuation channel.** Control enters IPC exclusively via
   `AffineBodyRevoluteJointExternalForce` (and the prismatic variant later) as a
   per-joint scalar torque. No `ExternalArticulationConstraint`, no
   `AffineBodyDrivingRevoluteJoint`/motor, no `SoftTransformConstraint` for these
   entities.
4. **Torque is computed Genesis-side.** The per-DOF torque comes from
   `get_dofs_control_force` ([abd/accessor.py] kernel), which folds FORCE /
   VELOCITY / POSITION / position_velocity and clamps to `force_range`. The
   coupler must not re-implement the PD law.
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
| **Source-scan** | the `ipc_monolithic` path contains no `kernel_predict_integrate`, no `ExternalArticulationConstraint`, no motor/driving-joint constitution | Rules 1,3 |
| **Integration (live, GPU)** | 1-DOF revolute pendulum under constant torque: IPC-integrated angle vs an independent forward-dynamics oracle within tol; step-position target converges to setpoint | Rules 3,4; M3 gate |
| **Integration (live, GPU)** | `get_dofs_position` after a step equals reconstruction of `copy_to` transforms; loader-pose offset calibration within 1e-4 | Rules 5,10; M4 gate |
| **Scene (live, GPU)** | fixed-base arm + cloth/RCC in one scene runs N steps and renders the arm | M5 gate |

## Performance note (not a perf feature)

Primary goal is rendering coexistence, not throughput. Record only: `couple()`
wall time/step and IPC Newton iters/step (already logged at
[coupler.py:1971]). Single-env. B>1 throughput is explicitly out of scope for v1;
do not make speed claims.
