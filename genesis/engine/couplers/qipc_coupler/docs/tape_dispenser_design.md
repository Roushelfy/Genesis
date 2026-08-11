# QIPC Tape Dispenser Import Design

Status: Genesis components, validation gates, and gs-core scene composition
implemented. Last audited against cuda-graph-qipc commit
`c66c312e682cdde1cbad885ff4774f274b48d02c`.

> Implementation: `../tape_dispenser.py`; frozen assets:
> `genesis/assets/qipc/tape_dispenser_v2/`; tests:
> `tests/test_qipc_tape_dispenser_contract.py` and
> `tests/test_qipc_tape_dispenser.py`.

## 1. Scope and invariant

`add_tape_dispenser` imports the settled tape-dispenser state from
`cuda-graph-qipc/examples/tape_dispenser/run_sim.py` as one reusable Genesis
QIPC component. The frozen artifact is the authority: construction must not
re-run the settling simulation or infer native IDs from scene insertion order.

The component contains:

- four articulated ABD bodies: `tape_cutter`, `Cylinder`, `blade`, and
  `tape_wheel`;
- the named state of `Cylinder_axle`, `blade_hinge`, and `tape_wheel_axle`;
- 1,936 active tape vertices and 3,500 tape triangles;
- a massless 192-vertex collision ring attached to the existing wheel ABD,
  with the same topology rendered as a visual-only cardboard core;
- all 969 active distance-bond slots, including their rest, stiffness,
  release, and age fields;
- tape collision with the Cylinder and blade; and
- tape self-bonding and tape-to-ring bonding.

The importer does not add a table, carton, ground, robot, camera, or
controller. Those belong to scene composition. Dynamic cutting is also out of
scope.

## 2. Meaning of post-f249 and static

The source scene executed:

```text
for frame in 0..249 inclusive:
    pre_step(frame)
    scene.step()
```

This is 250 solver steps and produces the state *after* frame 249. The export
then released the root and tape-end DBCs, zeroed all ABD/FEM velocities, and
did not take another physics step.

`post_f249_static.npz` therefore means a canonical zero-velocity snapshot. It
does not mean fixed geometry or mechanical equilibrium. The imported root is
free, the joints are unactuated, and gravity/contact can move the system on its
first subsequent step. A permanently anchored variant needs an explicit root
constraint rather than joint `kp`/`kv` gains.

Independent 250-frame source runs are not bitwise reproducible because CUDA
contact and bond creation use unordered atomics. The checked-in artifact and
its hashes, rather than regeneration, are therefore the regression oracle.

## 3. Public API

```python
from genesis.engine.couplers.qipc_coupler.tape_dispenser import (
    TapeDispenserAsset,
    add_tape_dispenser,
    add_tape_dispenser_machine,
    recommended_coupler_options,
    recommended_machine_coupler_options,
)

gs.init(precision="64")

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=0.01,
        gravity=(0.0, 0.0, -9.8),
    ),
    coupler_options=gs.options.QIPCCouplerOptions(
        **recommended_coupler_options()
    ),
)
component = add_tape_dispenser(
    scene,
    pos=(0.3, -0.2, 0.5),
    euler=(0.0, 0.0, 0.0),
)
scene.build()
```

The rigid-only ablation uses the canonical ringless URDF and the non-adhesive
contact recommendation:

```python
scene = gs.Scene(
    coupler_options=gs.options.QIPCCouplerOptions(
        **recommended_machine_coupler_options()
    ),
)
machine = add_tape_dispenser_machine(scene, pos=(0.3, -0.2, 0.5))
scene.build()
```

`add_tape_dispenser_machine` creates only the four-body rigid URDF. It keeps
the native `tape_wheel` mechanical link but does not create the Scotch ring
visual or attachment, FEM tape, contact regions, adhesion, or bonds. Its three
articulated joints start at their URDF zero values; the root remains free and
the joints remain unactuated.

The call must occur before `scene.build()` and requires the QIPC coupler.
`TapeDispenserAsset.packaged()` validates the bundled asset, while
`TapeDispenserAsset.from_directory(path)` accepts the same versioned layout at
another location. Every file listed in `manifest.json` is checked by SHA-256
before use. Required files cannot be omitted from the hash table; absolute,
parent-relative, escaped, and symlinked paths are rejected, and every URDF mesh
must remain inside the package. Roll parameters use UTF-8 `params_json`, so the
untrusted-directory path never enables NumPy pickle loading.

`pos` is the desired Genesis-world position of the `tape_cutter` root origin,
not the tape center or bottom contact point. `euler` is an additional XYZ Euler
rotation in degrees about that origin. The default already preserves the
expected upside-down orientation; callers should not add another 180-degree
correction.

The returned `TapeDispenser` retains the machine and tape entities plus stable
ring, Cylinder, blade, sharp, and ring-contact handles. The recommended options
match the source scene but do not mutate or validate an already-created scene.

## 4. Coordinates and placement

The source-to-Genesis proper rotation is:

```text
(x, y, z)source -> (x, -z, y)Genesis
```

For caller rotation `R_extra`, `R = R_extra R_reference` and placement is:

```text
p_body' = R (p_body - p_reference_root) + pos
A_body' = R A_body
x_tape' = R (x_tape - p_reference_root) + pos
Dm_inv' = Dm_inv R^T
```

Subtracting the stored source root gives `pos` stable component-origin
semantics. Body and joint overlays resolve by names because native ABD body
order is not the same as URDF link order.

## 5. Geometry and state construction

Both machine URDF variants disable fixed-link merging, convexification, decimation,
watertightening, alignment, and collision-mesh post-processing. The general
`URDF(preserve_collision_mesh_topology=True)` contract retains the exact
urdfpy collision vertices and faces. Exact topology is required because the
frozen attachment, contact-region, and bond mappings depend on vertex identity.

The ringless `tape_dispenser_machine.urdf` is the upstream canonical six-mesh
URDF. The full `tape_dispenser.urdf` additionally references the ring's pure
visual mesh. The ring uses the general queued rigid-attachment API. It contributes
collision triangles to the `tape_wheel` geometry, shares that body's affine
transform, and contributes no mass or fifth rigid body.

The URDF also references a visual-only GLB generated from the same 192 vertices
and 384 triangles. It follows the `tape_wheel` link but is deliberately absent
from the URDF collision list: QIPC's queued attachment remains the sole physical
ring representation, so visualization cannot duplicate collision vertices,
mass, contact regions, or frozen bond IDs.

Only the 1,936 active tape vertices are imported. The source cuttable mesh has
32 additional parked vertices, but they are deliberately omitted because this
component has no topology-changing cutting constitution or DBC lifecycle.

The full four-body affine state is stored instead of reconstructing it from
the root and joint angles. The settled wheel contains a small non-rigid affine
deformation that forward kinematics cannot reproduce exactly. Machine joint
gains are zero: `kp=0` and `kv=0` mean unactuated, not fixed.

## 6. Contact policy

| Pair | Collision | Adhesion / bond |
| --- | ---: | --- |
| tape - tape | enabled | frozen distance bonds |
| tape - ring | enabled | frozen distance bonds |
| tape - Cylinder | enabled | none |
| tape - blade | enabled | none |
| tape - sharp | disabled | none |
| tape - generic machine | disabled | none |
| ring - machine parts | disabled | none |
| machine internal pairs | disabled | none |

The Cylinder and blade part rows override the disabled generic tape-machine
row. This component sets `adhesion_bond_default=False`: only the two rows shown
with frozen bonds carry a Bond model. External scene pairs stay collision-
enabled under normal QIPC contact composition but cannot inherit a distance
bond accidentally.

Blade collision means the blade geometry can contact and repel tape. It does
not split the FEM mesh. There is no cut detector, triangle split, spare-vertex
activation, or blade-triggered topology mutation. A distance bond releasing
under its threshold is debonding, not cutting.

## 7. Frozen bonds and reset

The importer restores slots through QIPC `BondSystem.restore_slots`; it does
not recreate topology through `seed_locks`. Preserved fields are:

- topology and scene-remapped pair key;
- `Dm_inv`, `V0`, `d_rest`, and `kappa`;
- force, strain, gap, and slip release thresholds; and
- bond age.

Tape and ring IDs are component-local in the artifact and are remapped only
after the composed scene assigns final global offsets. Pair keys are then
recomputed from those IDs.

After ABD, joint, FEM, attachment, and bond overlays, the coupler rebuilds
derived global positions and contact candidates without advancing physics,
clears lagged/adhesive pair history, and promotes the composed state to the
QIPC reset baseline. `scene.reset()` restores body transforms, tape positions,
joint angles, zero velocities, and the frozen bond slots.

One scene currently accepts at most one frozen bond-state request. It cannot be
mixed with authored or manual `seed_bonds` batches because restoring slots
replaces the whole native `BondSystem` state.

## 8. Validation ladder

| Layer | Oracle | Assertions |
| --- | --- | --- |
| Asset contract | checked-in manifest and NPZ metadata | hashes, safe paths/no pickle, stable names, counts, finite arrays, joint values, solver options |
| Parser contract | synthetic multi-node GLB | exact URDF collision vertices/faces and incompatible-option rejection |
| GPU integration | frozen f249 artifact and ringless URDF | full transforms/tape/969 bonds plus machine-only 142,181 rigid vertices, zero FEM, one finite step, exact reset |
| gs-core scenes | composed robot and table | placement, external pairs, Pika/Wuji compatibility, teleop reset |

Run the implemented gates from the Genesis checkout:

```bash
.venv/bin/python -m pytest \
  tests/test_qipc_tape_dispenser_contract.py \
  --backend cpu -n 0 -x -q

.venv/bin/python -m pytest \
  tests/parsers/test_mesh.py::test_urdf_preserve_collision_mesh_topology \
  tests/parsers/test_mesh.py::test_urdf_preserve_collision_mesh_topology_rejects_processing \
  --backend cpu -n 0 -x -q

.venv/bin/python -m pytest \
  tests/test_qipc_tape_dispenser.py \
  --backend gpu -n 0 -x -q
```

The integration test explicitly requests Genesis precision 64. A separate
32-bit smoke test may be added later, but it must not weaken this fidelity
oracle's tolerance. The scene step asserts finite state and bounded initial
motion; it does not incorrectly assert static equilibrium.

## 9. Known limitations

- Dynamic cutting and the source's 32 parked cut-spare vertices are absent.
- The root is free and unactuated; no permanent anchoring policy is included.
- Runtime replay starts at the frozen snapshot instead of simulating frames
  0-249.
- Partial QIPC environment reset is unsupported.
- Strict snapshot comparisons require Genesis precision 64.
- The importer neither selects a robot nor adds surrounding scene objects.

## 10. gs-core composition

The follow-up gs-core scene should remain independent of robot choice:

```text
--robot-type marvin_pika|marvin_wuji
--scene-type carton_tape_dispenser_qipc
```

Its scene builder should add the table and fixed carton, construct QIPC with
the recommended options, call `add_tape_dispenser` before build, and place the
`tape_cutter` root on the right. No dispenser controller or 250-frame warm-up
is needed. Both Pika and Wuji should reuse this scene through compatibility
validation rather than introducing combined environment types.
