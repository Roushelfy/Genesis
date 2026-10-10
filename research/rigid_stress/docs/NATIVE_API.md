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

`cooperative_scatter=True` reduces integration-point loads per face before
node accumulation on CUDA, preserving all eligible Q10 and local retry
samples. CPU and the disabled option use scalar scatter. The benchmark
exposes `--full-inverse` and `--serial-scatter`.

`cached_peak=True` reuses immutable P2 corner gradients while scanning every
tetrahedron's four corners on CUDA. The cache contains four vertex and three
incident-edge gradients per corner; the other three edge gradients are
analytically zero at that corner. Its level-1 shared storage, including node
indices, is 376,320 bytes. CPU retains the original ten-term evaluation order.
`cached_face_bounds=True` stores conservative triangle bounds for the
complete face scatter, adding 3,840 shared bytes. The benchmark exposes
`--uncached-peak` and `--uncached-face-bounds` for matching ablations.

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
