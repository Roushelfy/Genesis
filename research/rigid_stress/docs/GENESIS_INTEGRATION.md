# Genesis integration audit and scene work

Audit baseline: `e9e1214d192914ddec85ca28014c459cfbc6c860` from upstream main.
The root package is Genesis World 1.4.3 plus later main commits. It uses
Quadrants 1.3.3, not the older Taichi-only API. Re-audit if the base changes.

## Current code locations

| Purpose | Path / API |
|---|---|
| Current per-contact public read | `genesis/engine/entities/rigid_entity/rigid_entity.py`, `RigidEntity.get_contacts(with_entity=None, exclude_self_contact=False, is_padded=False)` |
| Contact extraction/cache/padding | `genesis/engine/solvers/rigid/collider/collider.py`, `Collider.get_contacts` |
| Final normal + friction force | `genesis/engine/solvers/rigid/constraint/solver.py`, force reconstruction near lines 5790–5845 |
| Contact storage/frames | `genesis/utils/array_class.py`, contact state structs; collider `contact.py` |
| Existing Franka grasp | `examples/rigid/franka_cube.py` |
| Parallel Franka setup | `examples/tutorials/parallel_simulation.py`, `examples/rigid/ik_franka_batched.py` |
| Benchmark example | `examples/speed_benchmark/franka.py` (does not validate this stress task) |
| Classical linear/angular acceleration | `RigidEntity.get_links_acc`, `get_links_acc_ang` |
| Authored pose / angular velocity | `get_pos(relative=True)`, `get_quat(relative=True)`, `get_ang` |
| Substep interval | `scene.sim.substep_dt`, `scene.sim.substeps` |

Inspect these files rather than relying on an old snippet. Upstream links are
permalinks under
`https://github.com/Genesis-Embodied-AI/genesis-world/blob/e9e1214d192914ddec85ca28014c459cfbc6c860/`.

## Force semantics

`egg.get_contacts()` returns `geom_a`, `geom_b`, `link_a`, `link_b`,
`position`, `force_a`, `force_b`, and a parallel-scene `valid_mask`. The caller
may be on either side. The code explicitly sets:

```text
force_a = -contact_data.force
force_b = +contact_data.force
```

Select the force **on the egg** by checking its geom interval; do not assume
the egg is always A or B. Use its global link/geom IDs after scene build. A
net contact-force accessor loses contact locations and cannot recover local
stress loads.

The reconstructed Cartesian force contains normal and tangential components.
Elliptic friction combines normal/tangent rows; pyramidal friction combines
four cone directions. Spin/rolling rows contain additional moments and are
not fully represented by the linear `force` vector. Baseline scene explicitly
disables spin/rolling while keeping translational Coulomb friction. Supporting
those optional modes requires a tested vectorized contact-wrench getter.

The API describes contact forces, not impulses. Trace the reconstruction and
validate units with a resting shell: sum of support forces should approximate
its weight. Do not divide the reported force by dt again. Compare contact
wrench with net force/torque and rigid acceleration for matching substeps.

## Padding and host synchronization

Use `is_padded=True` on the target zero-copy CUDA path; `valid_mask` distinguishes
live slots. Padded outputs avoid the live-count `.max().item()` synchronization
in the current zero-copy getter. Padding behavior is backend-dependent: the
fallback getter can still query counts on host. Do not promise no host sync
without profiling the actual path.

When pruning/sorting changes contact order, the getter applies the permutation
with a gather. Physical slots may be stale outside `valid_mask`. Zero-copy
views can be overwritten by the next rigid step. Consume or explicitly copy
them before advancing; make CUDA stream dependencies explicit. Never infer
the next frame's contact IDs from storage order.

Start with the public batched getter. If profiling proves extraction a material
bottleneck, add one vectorized **solver getter** for positions, egg-applied
forces, optional moments, masks and link association. Follow root code style;
avoid bespoke private field accesses duplicated per consumer/env.

## Exact sampling point

Public contacts describe the most recent `scene.step()`, i.e. the final rigid
solve when multiple substeps are used. Step-end maximum is different from the
maximum over every physical substep. Start with substeps=1 in the seed so its
sampling contract is explicit. The production path must choose and label:

1. step-end peak; or
2. peak across all rigid substeps (default desired verification mode).

For the latter, integrate after each contact-force reconstruction and before
contact buffers are overwritten, accumulate max for the policy step, and pass
the correct pose/omega/acceleration at that time. Do not average contact forces
first: VM of an average can miss the transient maximum. Variable solver dt
options can change the common simulator substep count; read resolved values.

Benchmark all variants at the same stated dt and sampling contract. A peak-only
callback that secretly skips active substeps is a fidelity change.

## Rigid and recovery mesh/mass alignment

`assets.py` generates an exterior OBJ for rigid collision plus URDF mass/COM/
inertia from the FEM wall's consistent M and G. The volume enclosed by the
exterior is not the wall's physical mass. Do not accept Genesis's solid-envelope
mass estimate without replacing it with shell moments. Verify public mass and
COM/inertial orientation after loading. The authored mesh frame and the
reference recovery frame must match, including any URDF inertial alignment,
`offset_pos`/`offset_quat`, morph scale and convexification.

The current scene requests `align=False`, `convexify=True`, `decimate=False`.
Convexified collision can differ from the FEM exterior. Record and bound that
difference; positions must be projected/associated consistently and test the
finite footprint on the **recovery** surface. A nearest-face lookup alone must
not silently move the rigid contact wrench.

## Scene seed and work still required

`franka_egg.py` is a public-API starting point: plane, Panda, generated hollow
shell inertia, 0.6 translational friction, force-limited fingers, approach,
clamp, lift, hold and release commands. It includes per-env lateral offsets,
all egg contacts (not only Franka), world-to-authored transforms and a CPU
debug adapter. The seed has not been executed in the present environment.

Its finite-patch mapping does not exactly reproduce the input point moment;
its inferred acceleration follows the mapped loads, so this difference must
be audited against rigid physics. Fix/validate the traction model before
acceptance. A fixed global demo radius does not establish physical footprint
fidelity. Include varied radii and a documented physical patch law or sensor
input in the acceptance suite. Do not claim this issue is resolved merely by
the six-mode force balance.

Agent work:

- Validate actual grasp/lift/hold/release using COM height, grasp contacts,
  tangential force and slip diagnostics. Tune IK/PD/force limits using real
  scene observations, with no initial-state teleport counted as grasp.
- Randomize egg pose, grasp offsets/yaw and friction; no contact symmetry
  assumption is allowed in the recovery algorithm.
- Add a modest deliberate lateral motion to exercise sliding; report actual
  tangential force and relative velocity, not only `mu>0` in configuration.
- Add partial environment resets and phase offsets; histories must be local.
- Compare measured-acceleration and inferred-inertia-relief recovery under
  complete matching loads, separating contact-map/wrench error.
- Add a tested public stress observation and device-resident scalar maxima.
- Keep visualization optional; render/video runs separate from throughput.

Only then label the scene an accepted end-to-end benchmark. A scripted grasp
is enough to establish simulator throughput; policy-inference throughput needs
the separate matched inference benchmark described in `VALIDATION.md`.
