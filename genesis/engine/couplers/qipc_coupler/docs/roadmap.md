# QIPCCoupler Roadmap

## Core principle

Every Genesis feature mapped to QIPC must preserve a single authoritative
native state while providing deterministic entity ownership, validated runtime
controls, and explicit lifecycle behavior.

Non-negotiable rules:

1. Geometry, constitutions, contact rows, and constraints are declared before
   QIPC `Scene.init()`.
2. Runtime APIs write only documented live views and validate units and shape
   before the first write.
3. Native compact rows are resolved from QIPC ownership tables, never guessed
   from Genesis creation order.
4. Genesis reads positions and velocities from QIPC and does not integrate a
   second competing state.
5. A runtime mutation is marked complete only after its reset contract has an
   integration test.

## Current state (2026-08-04)

### Rigid and articulation (complete)

- [x] Multiple ABD entities and half-plane ground contact
- [x] Fixed-joint merging and free-base writeback
- [x] Revolute and prismatic joints
- [x] Position, velocity, and direct-force control forwarding
- [x] Per-entity gains, force ranges, and contact parameters
- [x] Scene-level solver-option passthrough
- [x] Soft transform constraints

### FEM and cloth (partial)

- [x] Stable Neo-Hookean volumetric FEM
- [x] Cloth membrane, bending, strain limiting, and contact thickness
- [x] Zero-copy native float64 reads with device-side Genesis writeback/cast
- [x] Soft vertex constraint creation/target updates, link following, and teleport
- [ ] Repair soft vertex-constraint removal without writing init-only `is_fixed`
- [ ] Replace the broken runtime `is_fixed` hard-constraint path with QIPC `PositionDBC`
- [ ] Define and test soft/hard constraint state restoration on `scene.reset()`
- [ ] Expose and test external acceleration through a public `FEMEntity` API
- [x] Distinct elastic rest geometry for prestress
- [x] Per-entity contact tabular rows

### Adhesion (complete)

- [x] Soft adhesion and distance-bond declarations
- [x] Runtime adhesion/bond state queries
- [x] Authored tape bond seeding with current-scene id mapping and reset replay

### Affine clusters (complete)

- [x] Pre-build FEM/proxy declaration queue with rigid-link resolution
- [x] Stable runtime handle with entity-local join/detach and native row ranges
- [x] Build-time QIPC membership-reset capability gate
- [x] Authored membership replay after seeded bonds on reset

### Lifecycle (partial)

- [x] QIPC native reset with immediate Genesis writeback
- [ ] Restore or explicitly preserve every coupler-owned live constraint state

### Sealed-volume gas (complete)

- [x] `FEM.SealedGasShell` material and parameter validation
- [x] QIPCCoupler-only backend guard
- [x] Closed, consistently wound, single-component mesh preflight with entity-scoped errors
- [x] QIPC Cloth + `SealedVolumeGas` composition before scene init
- [x] Robust entity-to-bag mapping through native `vert_index`/`vert_bag`
- [x] Typed runtime state snapshot and validated `p0`/`v0`/`enabled` writes
- [x] Runtime `v0` control with a fixed authored-geometry `v_min` safety floor
- [x] Genesis reset restoration for gas state omitted by raw QIPC reset
- [x] Default `partition_pcg` and alternative `linear_pcg` coverage
- [x] Elastic rest geometry and gas reference-volume separation
- [x] Multi-bag isolation and overpressure-versus-disabled physics gate

The stable contract is documented in [fem_design.md](fem_design.md).

## Next milestones

### FEM plasticity

- [ ] Add an explicit plastic FEM material/configuration rather than extending
      `Elastic` with nullable plastic fields.
- [ ] Map StVK-Hencky elasticity and QIPC's plasticity modifier.
- [ ] Expose plastic-strain diagnostics without leaking native buffers.
- [ ] Add deterministic parameter, return-map, and reset integration tests.

### Batch environments

- [ ] Define a QIPC-native subscene or batch abstraction.
- [ ] Preserve contact isolation without an O(N^2) coupler-side disable table.
- [ ] Add full and partial reset semantics for `n_envs > 1`.

### Remaining compatibility coverage

- [ ] Add systematic joint-type-by-base-type tests.
- [ ] Expand observation parity for entity position, orientation, and velocity.
- [ ] Add a long-running contact-capacity churn scene once a reproducible
      benchmark protocol is agreed.

## Known limitations

- One QIPCCoupler scene supports one Genesis environment.
- QIPC performs native FEM/ABD computations in float64; Genesis writeback is
  cast to the configured Genesis precision.
- `SealedGasShell` validates topology and connectivity, not self-intersection or
  mesh conditioning.
- Re-enabling disabled gas after shell collapse is not guaranteed safe because
  disabling also removes the gas volume barrier.
- QIPC topology changes cannot currently be crossed by Genesis reset.
- Runtime hard FEM constraints currently fail because QIPC `is_fixed` is an
  init-only buffer; use soft constraints until the path is migrated to
  `PositionDBC`.
- Removing a soft FEM constraint currently fails because the removal path also
  writes init-only `is_fixed`.
- Soft FEM constraint masks and targets currently survive `scene.reset()`.

## Validation commands

Run from the Genesis repository in a development environment where QIPC is
installed and an NVIDIA GPU is available.

| Gate | Command | Required result |
| --- | --- | --- |
| Sealed gas | `pytest tests/test_qipc_sealed_gas.py --backend gpu -n 0 -x` | all material, lifecycle, multi-bag, solver, and physics tests pass |
| Core QIPC alignment | `pytest tests/test_qipc.py --backend gpu -n 0 -x` | standalone-QIPC alignment and control tests pass |
| Adhesion | `pytest tests/test_qipc_adhesion.py --backend gpu -n 0 -x` | soft adhesion and distance-bond tests pass |
| Affine clusters | `pytest tests/test_qipc_affine_cluster.py --backend gpu -n 0 -x` | ghost and rigid-link proxy lifecycle/reset tests pass |
| Affine-cluster contracts | `pytest tests/test_qipc_affine_cluster_contract.py --backend cpu -n 0 -x` | declaration, validation, capability, and reset-order contracts pass |
| Tape/prestress | `pytest tests/test_qipc_tape.py --backend gpu -n 0 -x` | prestress, asset, and tape lifecycle tests pass or asset-marked cases skip explicitly |

## Planned gates

These commands do not exist yet and must not be treated as current validation.

| Gate | Target command | Missing piece |
| --- | --- | --- |
| FEM plasticity | `pytest tests/test_qipc_plasticity.py --backend gpu -n 0 -x` | material mapping and test module |
| Batch isolation | `pytest tests/qipc/test_batch.py --backend gpu -n 0 -x` | QIPC batch/subscene support |
| Contact churn benchmark | `pytest tests/benchmarks/test_qipc_contact_churn.py --backend gpu -n 0 -m benchmarks` | fixed workload, warmup, timers, and baseline record |
