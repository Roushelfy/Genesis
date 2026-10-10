# Native rigid stress observation

The production implementation lives in `genesis/engine/solvers/rigid/stress/`.
The normal example and benchmark import no research solver. All physical
assembly, contact mapping, factor/inverse construction, recovery, residuals
and peak reduction run in Quadrants. NumPy handles mesh topology at build time;
the SciPy/Numba research implementation is an independent test oracle.

## Usage

Configure a link before `scene.build()`:

```python
gs.init(backend=gs.gpu, precision="64")
egg.base_link.configure_stress_recovery(
    gs.options.RigidStressOptions(mesh="elastic.npz", contact_radius=0.006)
)
scene.build(n_envs=8)
scene.step()
peak_pa = egg.base_link.get_max_stress(copy=False)
egg.base_link.set_stress_contact_radius(radii[:, None])
scene.reset(envs_idx=[1, 4])
```

The NPZ contains finite `vertices` (n, 3), positive affine `tetrahedra`
(m, 4), and exterior `surface_triangles` (s, 3), in the **authored link
frame**. P2 edge nodes are constructed natively. The elastic geometry and
declared E, Poisson ratio and density are fixed. Declare compatible rigid
shell mass/COM/inertia in the rigid asset; the example URDF does so explicitly.
Recovery observes a rigid trajectory and does not change collision or motion.

The observation is the unaveraged **global** maximum von Mises stress over all
P2 tetrahedra and all four corners, maximized over every physical substep of
the scene step. With affine P2 displacement, stress is affine and its von
Mises norm attains a maximum at a corner. Units are Pa. A batched scene returns
`[B]` on the scene device; an unbatched scene returns a scalar. `copy=False`
aliases the observation; treat that view as read-only.

The radius setter broadcasts to `[selected environments, sorted contact
capacity]`. Pass `[B, 1]` for independently changing per-environment radii;
pass a scalar for a common radius. Columns refer to current sorted contact
slots, whose identities may change. Persistent contact-specific pad metadata
should be supplied again after such changes. Radii are INFO and persist
through reset; displacement and observed stress are invalidated only for
selected environments. Reset and state mutations use native solver notices.

Solver checkpoint export includes stress configuration, contact radii, warm
displacement and derived observations. Restoring rigid state triggers the
ordinary geometry/dynamics notices: stress peaks, displacement and history
are invalidated for all restored environments. The next physical step
recomputes the observation. A saved peak is not a valid restored observation.

## Build-time output selection

Set `RigidStressOptions(output_mode="full", ...)` before `scene.build()` to
enable `tensor_pa, von_mises_pa = link.get_stress_field(copy=False)`.
The default `output_mode="max"` keeps the existing maximum observation,
allocates no full-field storage and compiles out full-field writes.
Reconfiguring a built link or changing its output mode requires rebuilding.
Links with different output modes can share the same immutable operators.

The returned layouts are `[B, n_tets, 4, 6]` and `[B, n_tets, 4]`, or
`[n_tets, 4, 6]` and `[n_tets, 4]` for an unbatched scene. Tensor components
are `xx, yy, zz, xy, xz, yz` in Pa, in the **authored link frame**. Tetrahedra
and their four corners follow the NPZ connectivity order. These are all
unaveraged affine-P2 corner stresses, determining the entire affine tensor
field in each element; duplicated shared corners intentionally retain their
element stresses. Device views may be strided; `copy=False` views are read-only
and change when the next recovery writes their buffers.

The field describes the latest physical recovery substep. Its von Mises
maximum equals the maximum-only result for that recovered state. With
multiple substeps, `get_max_stress()` retains the temporal maximum over the
scene step, which may exceed the latest field maximum. Before first recovery,
after reset/restoration or following an invalid recovery, full fields contain
NaNs. A partial reset invalidates only its selected environments. Observations
are recomputed on the next physical step; checkpointed fields do not become
valid merely by restoring a checkpoint. The level-1 FP64 tensor plus von
Mises storage is 107,520 bytes per environment, versus zero in max mode.

The same corner scan computes both modes in Quadrants. Full mode changes
observation writes, not loads, displacement, equilibrium budgets or rigid
motion. See [OUTPUT_MODES.md](OUTPUT_MODES.md) for acceptance requirements.

## Model and errors

Each solved point contact includes its complete normal and tangential force.
A force-line anchor moves the footprint centre on the exterior without
changing its point wrench. Positive Duffy/Gauss quadrature samples a compact
Gaussian/bump pad profile. A nonnegative constant-ratio vector traction
preserves force and the two independent moment constraints perpendicular to
the force; nodal P2 loads may be signed even though physical pressure is
nonnegative. The law is in [PAD_LAW.md](PAD_LAW.md).

A sampled fit failure triggers native local Q10 integration on the anchor
face. A central triangle whose centroid is the anchor plus six surrounding
triangles partitions that face. Parent-face shape functions transfer the new
samples, replacing the original face samples. Retry and cumulative fit work
are included in live timing. See [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md).

Inferred rigid acceleration relief removes resultant force and torque using
consistent mass. Uniform gravity cancels under this model. Six fixed fields
represent consistent-mass centrifugal loads from the same substep angular
velocity. Six independent displacement pins fix a gauge; acceptance checks
**all rows**, including those pins, against
`max(absolute_tolerance, tolerance * ||rhs||)`.

Finite pressure cannot represent every point wrench on every sampled
footprint. A failed input, anchor, pressure solve, wrench check, unsupported
applied/coupling load, or contact-pool overflow produces NaN and a rigid errno;
`scene.rigid_solver.check_errno()` raises. The implementation neither clips
friction forces nor silently expands a footprint. The original B=2048 apex
counterexample is a successful local-integration regression with independent
CPU wrench, complete residual and peak comparisons.

Currently supported stress links are independent free roots with the standard
rigid contact coupler, without autodiff, joint supports, rolling or torsional
contact moments. Enabling those unsupported configurations raises at build.
Mass, inertia and COM setters, and external wrench application to a stressed
link, raise because they would change the declared model or omit a required
surface traction. Multiple configured links can share immutable operators
when geometry/material/mass and method agree; their loads and states remain
independent. Disabled scenes retain the original fused rigid step.

## Numerical methods

`cooperative_pressure=True` packs active contact/environment pairs on CUDA
with a device counter, then assigns one warp to each pair. Warp reductions
integrate the pressure Gram and constrained Newton/line-search evaluations.
There is no host count read or candidate truncation. CPU and
`cooperative_pressure=False` use the serial native pressure kernels with
the same law and wrench checks.

`method="auto"` uses a full pinned inverse if its shared storage fits
`inverse_max_bytes` (64 MiB default); otherwise it selects sparse direct.
The full inverse maps **every** nodal load, rather than a fixed contact-response
basis. Level 1 requires 47,239,200 bytes in FP64. Explicit `method="inverse"`
rejects an insufficient budget; it never allocates an unbounded dense inverse.

`surface_inverse=True` additionally constructs exact responses for every
exterior P2 node/force component and six centrifugal fields, within the
shared inverse storage budget. It retains the complete displacement,
equilibrium check and global stress scan. Arbitrary interior load validation
and residual corrections use the full inverse. Level-1 extra storage is
9,565,128 bytes. Disable the option for a full-load application ablation.

`packed_surface_loads=True` refreshes each environment's complete list of
strictly nonzero boundary force vectors in Quadrants on CUDA. It preserves
the original ascending node order, includes every nonzero component without
a threshold, and evaluates all centrifugal terms. No contact or nonzero load
is discarded. Mutable lists belong to each link's scratch state and are
rebuilt on every recovery, including after reset or checkpoint restoration.
CPU keeps the dense operator path. `--dense-surface-loads` measures its dense
CUDA ablation, with the same model and acceptance budgets.

`fused_pipeline=True` captures the serial association, pressure, scatter,
complete solve/residual/peak and acceptance stages in one native CUDA graph
when cooperative exterior inverse recovery is available without load history.
Both output modes use it. Other configurations retain the independently
callable native passes. `--unfused-pipeline` supplies the matched ablation.

`cooperative_scatter=True` reduces integration-point loads per face before
node accumulation on CUDA, preserving all eligible Q10 and local retry
samples. CPU and the disabled option use scalar scatter. The benchmark
exposes `--full-inverse` and `--serial-scatter`.

`face_parallel_scatter=True` additionally assigns one CUDA warp to each
eligible contact/face pair. A bounded device task buffer is refreshed on
every recovery, with `scatter_tasks_per_env=32` as its build-time capacity
per environment. Its task and wrench storage adds 1,792 bytes per environment
plus two eight-byte counters. All contacts and conservative face bounds are
examined. If either active-contact storage or task capacity is insufficient,
the complete contact-warp scatter runs in Quadrants on the device. Final
force and moment checks retain their original budgets. There is no host
count read in recovery. CPU, scalar scatter and the disabled option retain
their existing paths. `--contact-warp-scatter` selects the CUDA ablation,
and `--scatter-tasks-per-env` measures buffer/fallback tradeoffs. Ordinary
benchmark repeats report every full-batch scatter overflow call.

`cached_peak=True` reuses immutable P2 corner gradients while scanning every
tetrahedron's four corners on CUDA. The cache contains four vertex and three
incident-edge gradients per corner; the other three edge gradients are
analytically zero at that corner. Its level-1 shared storage, including node
indices, is 376,320 bytes. CPU retains the original ten-term evaluation order.
`cached_face_bounds=True` stores conservative triangle bounds for the
complete face scatter, adding 3,840 shared bytes. The benchmark exposes
`--uncached-peak` and `--uncached-face-bounds` for matching ablations.

`cooperative_balance=True` computes the complete six-component rigid-mode
wrench using CUDA node/environment tiles, including every nodal contact
load and all centrifugal terms. It applies the shared immutable projection
`mass_modes @ gram_inverse` to remove inferred acceleration. The projection
adds 116640 shared bytes on the low mesh; it depends on the declared geometry
and mass, rather than contact locations or histories. CPU uses the original
reduction/projection order. `--serial-balance` selects the CUDA comparison.
Both paths keep the original full-equilibrium acceptance, including gauge
rows. Odd batches and partial changes to force/angular velocity are tested
against independent CPU FP64 recovery.

Contact rejection reports the affected link, environment, contact slot,
original radius and actual friction coefficient. The diagnostic distinguishes
invalid input/friction, an absent force-line anchor, an unresolved sampled
fit, and failure to preserve the contact force/moment. An unresolved fit does
not establish physical infeasibility. Diagnostic host transfers occur only
when the rigid solver reports an error.

The native block LDL factor is shared across environments. Its symbolic
minimum-degree ordering is Python topology work; every numeric factor value
is computed in Quadrants. CUDA solves on at most 1,024 P2 nodes use a warp
per environment and shared work storage when `cooperative_solve=True`.
Other meshes/backends use the serial native triangular solve. This fallback
is correct but has no large-mesh performance claim in this milestone.

`inverse_precision="32"` halves inverse storage while retaining scene
arithmetic. Residual failures receive masked inverse corrections followed by
the native factor fallback if necessary. Default scene-precision inverse
storage is the conservative configuration selected by measurement.

`method="pcg"` supports scalar-diagonal or block preconditioning, independent
environment convergence and a strict iteration limit. `warm_start=True`
reuses the same environment's displacement. Thin-shell conditioning made
the simple PCG path slower and unresolved at 4,000 iterations in the probe;
it is retained as an explicit option with residual rejection, not the default.

`history_size=4` enables four corrected load/displacement columns per
environment. A stable cached native basis predicts displacement; the complete
residual rejects unsuccessful predictions and the original solve handles
them. Only corrected accepted environments append. Geometry/dynamics changes
and reset invalidate the selected environments. This cost/memory tradeoff
regressed the measured changing-contact workload, so `history_size=0` is the
default and selected benchmark configuration. CPU/CUDA FP64 scene precision
and FP32 mixed inverse storage are validated in this milestone.

Use [RUNNING.md](RUNNING.md) for reproducible commands and the native results
report for measurements. These low-resolution checks establish numerical
consistency and performance; physical mesh/quadrature convergence is deferred.
