# QIPCCoupler FEM Architecture

Status: volumetric FEM, cloth, soft-constraint creation/target updates,
per-entity contact, adhesion, prestress, native scene reset, sealed-volume gas,
and affine clusters are implemented. Soft-constraint removal/reset, runtime hard vertex
constraints, a public external-acceleration API, plasticity, and
multi-environment simulation remain open; see [roadmap.md](roadmap.md).

## Core principle

Every FEM feature must be fully declared before QIPC `Scene.init()`. Runtime
control may only mutate documented live device state, through a Genesis API
that validates ownership, units, and reset behavior.

Consequences:

1. Genesis owns material configuration and entity identity.
2. QIPC owns solver buffers and the authoritative simulated state.
3. A Genesis entity never infers a native row from creation order.
4. Live QIPC views are reacquired on every API call; they are never cached.
5. A runtime feature is complete only when its reset behavior is explicit and
   tested. Sealed gas meets that requirement; remaining constraint gaps are
   tracked in the roadmap.

## Architecture and data flow

During `QIPCCoupler.build()` entities are classified and converted before the
QIPC scene freezes:

```text
Plane       -> QIPC half-plane
Rigid       -> affine body dynamics geometry + joints
FEM.Elastic -> tetmesh + StableNeoHookean
FEM.Cloth   -> trimesh + QIPC Cloth preset
SealedGasShell
            -> trimesh + QIPC Cloth preset + SealedVolumeGas
```

Every FEM geometry also receives an initially inert
`SoftPositionConstraint`. Contact elements and pairwise contact models are
inserted before `Scene.init()`. After init, Genesis records each geometry's
native FEM vertex offset and writes QIPC positions and velocities back into the
corresponding `FEMEntity` after every QIPC step.

Runtime teleports write `x`, `x_prev`, and velocity consistently. Soft vertex
constraint creation and target updates write QIPC's live device views, but the
current removal path also touches init-only `is_fixed` and fails. The hard-
constraint path has the same incompatibility and must be migrated to
`PositionDBC`. Sealed-gas controls write the gas constitution's live per-bag
arrays directly between simulation steps.

## Affine clusters

`QIPCCoupler.add_affine_cluster` queues a pairing between one FEM entity and
either an implicit QIPC proxy or a selected merged body of a Genesis rigid
entity. The queue resolves in two phases required by QIPC:

1. after rigid/FEM geometry conversion and before `Scene.init()`, declare the
   proxy and FEM geometry with QIPC;
2. after init assigns native offsets, validate the entity-local edge/triangle/
   tet selections and apply authored initial membership.

The returned `QIPCAffineCluster` remains stable across both phases and across
reset. Its runtime `join`/`detach` methods accept entity-local primitive
indices. `fem_vertex_range` identifies the entity in QIPC's FEM-local vertex
buffer, `proxy_body_index` identifies its ABD row, and `member_count` reports
the current number of clustered elements.

Raw QIPC reset restores the snapshot captured at init, where membership is
empty and post-init authored bonds do not yet exist. Genesis therefore restores
authored state in strict dependency order:

```text
QIPC Scene.reset -> restore seeded bonds -> replay initial membership
```

Runtime membership changes are intentionally not preserved by reset; reset
returns to the selection passed as `initial_edges`/`initial_tris`/
`initial_tets`. The build-time capability gate requires QIPC's cluster API and
its membership reset/restore state surface, so an older native extension fails
with a host-side error rather than silently losing membership.

## Material mapping

| Genesis material | QIPC primary constitution | Additional behavior |
| --- | --- | --- |
| `FEM.Elastic(model="stable_neohookean")` | `StableNeoHookean` | E, nu, and rho are forwarded |
| `FEM.Elastic` with another model | `StableNeoHookean` | Emits a fallback warning |
| `FEM.Cloth` | `Cloth` preset | Membrane, bending, strain limit, and contact thickness are forwarded |
| `FEM.SealedGasShell` | `Cloth` preset | Stacks `SealedVolumeGas` |
| `FEM.Muscle` | unsupported | Build raises instead of silently degrading |

`SealedGasShell` subclasses `Cloth` so the existing surface-triangle sampling
path remains the single shell implementation. It is QIPCCoupler-only; using it
with another coupler raises while the entity is added.

## State and boundary conditions

QIPC's native float64 FEM views are the source of truth. Genesis reads those
views on-device and writes them back at the current Genesis precision. Current
runtime status is:

| Genesis operation | Status and QIPC state |
| --- | --- |
| soft constraint create/update | supported through `aim_position`, `strength_ratio`, `is_constrained` |
| soft constraint removal | broken; the current path also writes init-only `is_fixed` |
| hard vertex constraint | unsupported at runtime; current `is_fixed` path is init-only |
| entity teleport | `x`, `x_prev`, `velocities=0` |
| entity velocity | `velocities` |
| external acceleration | private coupler hook exists for `external_acc`; no public `FEMEntity` API or gate |
| sealed-gas control | per-bag `p0`, `v0`, and `enabled`; build-time `v_min` is fixed |

QIPC currently has no environment-batch abstraction, so the coupler rejects
`n_envs > 1` and partial reset.

## SealedVolumeGas

### Physical model

For current enclosed volume `V`, reference volume `v0`, reference absolute
pressure `p0`, and polytropic exponent `gamma`, QIPC uses

```text
p_in(V) = p0 * (v0 / V) ** gamma
delta_p(V) = p_in(V) - p_atm
grad(E) = (p_atm - p_in(V)) * grad(V)
```

inside an implicit energy with an exact gradient and a matrix-free rank-1 gas
Hessian. The shell loading is the internal-minus-ambient pressure `delta_p`, not
unopposed absolute pressure. `gamma=1` is isothermal. `p0` is the pressure at
`V=v0`; it is not a measurement of the current pressure after the bag deforms.

The membrane remains the primary shell constitution. Gas is an extra
constitution on the same triangle geometry.

### User-facing material

```python
bag = scene.add_entity(
    name="bag",
    morph=gs.morphs.Mesh(file="closed_bag.obj"),
    material=gs.materials.FEM.SealedGasShell(
        E=1e7,
        nu=0.3,
        rho=910.0,
        thickness=1e-4,
        membrane="stvk",
        p_gauge0=500.0,
        p_atm=101325.0,
        gamma=1.0,
        v_min_rel=1e-4,
        auto_flip=True,
    ),
)
```

| Field | Unit and contract |
| --- | --- |
| `p_gauge0` | Pa, finite; initial `p0 = p_atm + p_gauge0` must be finite and positive |
| `p_atm` | Pa absolute, positive |
| `gamma` | positive; 1 is isothermal |
| `v_min_rel` | dimensionless and strictly between 0 and 1 |
| `auto_flip` | accepts globally inward winding by storing sign -1; topology is not rewritten |

### Build order and mesh contract

The coupler performs these steps for each `SealedGasShell`:

1. Create a QIPC trimesh from the Genesis initial positions and surface faces.
2. Apply the configured QIPC Cloth preset.
3. Run a host preflight with `closed_surface_orientation` so errors include the
   Genesis entity name.
4. Require exactly one edge-connected closed shell. Multiple bags must be
   separate Genesis entities.
5. Apply `SealedVolumeGas` with keyword arguments from the material.
6. Apply the resident position constraint and create the geometry slot.
7. Let QIPC validate the sealed geometry again during `Scene.init()`.

The preflight requires every directed triangle edge to appear once with its
reverse, consistent global winding, nonzero signed volume, and one connected
triangle component. With `auto_flip=True`, inward winding records a native sign
coefficient rather than modifying the shared membrane/contact topology.

This validation does not prove that the surface is free of self-intersection or
that every triangle and enclosed volume is numerically well-conditioned. Those
remain asset-quality requirements.

### Initial geometry versus elastic rest geometry

The authored `geometry.positions` define the initial simulated configuration
and the gas reference volume `v0`. An optional `rest_geometry` defines elastic
rest areas, masses, and bending metrics only. Therefore
`set_fem_rest_positions()` can create membrane/bending prestress but does not
change how much gas was sealed into the authored bag.

Only the simulated geometry is stamped with `SealedVolumeGas`; the rest geometry
is not.

### Entity-to-bag ownership

QIPC stores gas parameters in compact per-bag arrays. A bag row is neither the
Genesis FEM entity index nor its FEM vertex offset. After `Scene.init()`, the
coupler reads the live `vert_index` and `vert_bag` tables, intersects them with
each entity's native FEM vertex range, and requires exactly one unique bag row.
It also rejects two entities resolving to the same row.

The resulting `FEMEntity -> bag row` map is the only mapping used by runtime
APIs. Each access reacquires `LiveConstitutionData` because native buffers can
be reallocated and an older DLPack tensor may become stale.

### Runtime API

`FEMEntity.get_gas_state()` returns a scalar snapshot:

```python
state = bag.get_gas_state()
# state.p0: reference absolute pressure [Pa]
# state.v0: reference volume [m^3]
# state.enabled: whether the gas energy is active
```

The getter synchronizes device state and is intended for control and
diagnostics, not per-frame high-rate telemetry.

`FEMEntity.set_gas_state()` changes any subset of the live controls:

```python
bag.set_gas_state(p0=102000.0)
bag.set_gas_state(v0=1.1 * state.v0)
bag.set_gas_state(enabled=False)
```

- `p0` is an absolute reference pressure in Pa and must be finite and positive.
- `v0` is a gas-law reference volume in m^3 and must be finite and positive. It
  can model adding or removing gas without moving the mesh.
- `v_min` remains the fixed collapse floor
  `material.v_min_rel * authored_initial_volume`. It is a geometric line-search
  safeguard, not a gas-quantity control, so changing `v0` does not move it.
- `enabled` must be a bool and is written as exactly 0 or 1. Fractional values
  are forbidden because QIPC's energy and derivative paths do not give them
  matching semantics.
- `enabled=False` disables the complete gas energy, including its collapse
  barrier. It models opening/removing the virtual gas chamber, not gradual
  leakage. Re-enable only while the shell remains closed, consistently oriented,
  and above its volume floor.
- Calls must occur between scene steps, not concurrently with a QIPC solve.

`p_atm`, `gamma`, orientation sign, and connectivity are build-time state and
are not exposed as runtime controls.

### Reset semantics

Raw QIPC `Scene.reset()`, dump, and recover currently do not checkpoint the gas
live arrays. Genesis therefore snapshots each managed bag's initial `p0`, `v0`,
`v_min`, and `enabled` after init. `QIPCCoupler.reset()` reacquires the current
views, restores those values, and then invokes QIPC reset. If native reset is
refused before it changes the scene, Genesis rolls the gas arrays back to their
pre-call runtime values. A successful Genesis reset returns gas control state to
the material-defined build-time baseline.

### Linear solver compatibility

The gas Hessian requires a PCG matvec hook. Both QIPC solvers exposed by Genesis
are supported:

- `partition_pcg` (default)
- `linear_pcg`

There is no sealed-gas-specific reason to force the partitioned solver.

### Validation ladder

`tests/test_qipc_sealed_gas.py` provides the Genesis integration oracles:

| Layer | Contract |
| --- | --- |
| material | pressure and volume-floor parameter guards |
| build validation | open and multi-component shells fail with the entity name |
| solver contract | `linear_pcg` builds and steps |
| rest geometry | elastic prestress does not replace the authored gas reference volume |
| multi-bag integration | entity-scoped rows do not cross-write across an intervening ordinary cloth entity |
| runtime/reset | `p0`, `v0`, and `enabled` update; fixed `v_min` and all controls reset correctly |
| physics scene | overpressure inflates while a disabled control remains static |

Run the gate in a Genesis development environment with a CUDA QIPC build:

```bash
pytest tests/test_qipc_sealed_gas.py --backend gpu -n 0 -x
```

QIPC's own `tests/test_sealed_volume_gas.py` remains the oracle for the native
energy, gradient, Hessian, winding, and squeeze physics. Genesis tests focus on
configuration, ownership, lifecycle, and end-to-end wiring.

## Related subsystems

Adhesion and tape-specific prestress are documented separately in
[adhesion_tape_design.md](adhesion_tape_design.md). Current milestone status,
known limitations, and runnable repository gates live in
[roadmap.md](roadmap.md).
