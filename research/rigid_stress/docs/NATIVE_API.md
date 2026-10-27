# Native rigid stress observation

The production implementation lives in `genesis/engine/solvers/rigid/stress/`.
The normal example and benchmark import no research solver. All physical
assembly, contact mapping, factor/inverse construction, recovery, residuals
and peak reduction run in Quadrants. NumPy handles mesh topology at build time;
the SciPy/Numba research implementation is an independent test oracle.

`backend="volume"` remains the default. `backend="shell"` selects the native
curved Q4 shell with an authored midsurface NPZ, thickness and optional directors.
It supports DPR finite-pressure patch loading, including eccentric friction;
SCR/intrinsic and resolved-sample transfer are not shell routes. Its maximum
is sampled in-plane membrane-plus-bending stress. Full stress has shape
`[B,n_quads,2,4,6]`; `get_shell_resultants()` returns separate membrane, bending
and transverse resultants. See the [shell API/running guide](RUNNING_NATIVE_SHELL_DPR.md)
for units, frames and lifecycle, and the [measured report](SHELL_DPR_NATIVE_20261024.md)
for configuration recommendations and physical limits. The volume-specific
load-route qualifications below do not establish a shell contact-peak reference.

Shell DPR can use `shell_loading="planar"` with
`loading_subdivisions=4`, `8`, `16` or `32`. These triangles refine the loading
surface and its consistent displacement/load map; the original Q4 elasticity,
mass, gauge, response and stress sampling remain unchanged. This route requires
the Hertz pressure profile and patch transfer. Original geometry determines
anchors and automatic radii, and the original-point force/friction moment
contract remains active. Unsupported proxy work uses the qualified original
material-component route; uncertified original connectivity remains a rejection.

The public loading default is `"q4"`. The opt-in subdivision8/capacity32 defaults
are separate from the complete four/64-condition Franka accuracy configuration:
subdivision32, `quadrature=10`, `scatter_tasks_per_env=128`, FP64 simulation and
inverse, and `history_size=0`. That configuration uses process-local
`QD_GRAPH=0` set before importing Quadrants/Genesis. See the
[planar loading report](PLANAR_LOADING_RESULTS_20261027.md) for the full recipe,
accuracy, throughput, preparation and memory decisions. Qualification compares
the same discrete Q4 elastic model; it does not establish continuum contact peaks.

For small finite-pressure footprints near a face edge, the optional
`pressure_quadrature="angle"` uses angular fan integration while preserving
the physical radius, pressure profile and original acceptance checks.
`quadrature=128` passes the retained 64-condition heterogeneous volume bank;
lower degrees do not satisfy that bank's independent load-error budget.
The default `"edge"` mode remains unchanged. See the
[native candidate report](NATIVE_CANDIDATE_RESULTS_20261025.md) for accuracy,
cost and the preserved failed configurations. This is a quadrature repair,
not a change to the declared pressure model or its physical qualifications.

The [practical application report](PRACTICAL_APPLICATION_RESULTS_20261016.md)
gives measured settings, costs and operating limits. Prefer the intrinsic
source-prior configuration below when requiring local friction-admissible
tractions. Its aggregate constraint permits measured local load redistribution.
The faster pressure law has diagnosed local-cone violations in the actual egg
reference and is only a declared simplified proxy there. A point wrench does
not identify the physical footprint, and neither option calibrates a fracture
threshold. Keep native load rejections explicit.

## Usage

Configure a link before `scene.build()`:

Choose a complete route with `RigidStressOptions.dpr(...)` (Directional Patch
Recovery) or `RigidStressOptions.scr(...)` (Surface-Constrained Recovery).
DPR includes the supplied tangential force. SCR combines intrinsic surface
support and `body_source_prior`. Both accept the usual numerical/output keyword
options and return ordinary `RigidStressOptions` with all effective fields.

| Constructor | Exact serialized load configuration |
| --- | --- |
| `.dpr(mesh=...)` | `contact_model="pressure"` |
| `.dpr(mesh=..., load_transfer="resolved")` | Pressure-mode samples transferred conservatively to their witnessed boundary |
| `.scr(mesh=...)` | `contact_model="intrinsic", intrinsic_load_partition="body_source_prior"` |

The preset constructors fix those identifying fields. Use the main
`RigidStressOptions(...)` constructor for other explicit load-law variants.
The presets retain the existing solver and defaults. Historical identifiers in
receipts remain unchanged, and the names do not express an accuracy ranking.

Use `load_transfer="resolved"` only when the producer has already integrated
the physical pressure/traction samples. Native Quadrants source association
retains geom/pair witnesses, material face and barycentric coordinates, including
inner and nonconvex boundaries. Signed P2 shape functions preserve virtual work,
force and original-point moment without a second disk. The supplied normal and
friction input contract, boundary ambiguity/projection checks, original-wrench
budget and complete elastic acceptance remain active. A free moment that this
point sample cannot represent is rejected.

This option does not determine support or pressure from sparse collider points.
It has no reconstruction radius; `set_stress_contact_radius()` rejects changes
for this route. The default `load_transfer="patch"` retains the manual finite
footprint. Resolved transfer uses the ordinary max/full observation, update,
checkpoint and reset lifecycle; physical automatic pressure prediction and its
material/pad inputs remain a separate integration requirement.

```python
gs.init(backend=gs.gpu, precision="64")
egg.base_link.configure_stress_recovery(
    gs.options.RigidStressOptions.scr(
        mesh="elastic.npz",
        contact_radius=0.006,
    )
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

For occasional keyframes, also set `update_mode="manual"` before building.
After applying that step's controller/state inputs, call
`link.request_stress_update()` and then `scene.step()`. This observes that link
across all environments and all physical substeps of the requested scene step.
Unrequested steps advance ordinary rigid mechanics while skipping that link's
auxiliary stress recovery. The default `update_mode="every_step"` observes
every scene step.

In manual mode, getters retain the last sampled field and sampled-step maximum
between requests. Record the requested times when exporting them. These held
values represent their sampled step rather than the current unsampled state.
State changes clear the affected observations as described below and cancel
pending requests for that link. Request after setting the step's inputs.
Requests made after the first physical substep of a scene step raise an error.
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
observed scene step, which may exceed the latest field maximum. Before first recovery,
after reset/restoration or following an invalid recovery, full fields contain
NaNs. A partial reset invalidates only its selected environments. Observations
are recomputed on the next observed physical step; checkpointed fields do not become
valid merely by restoring a checkpoint. The level-1 FP64 tensor plus von
Mises storage is 107,520 bytes per environment, versus zero in max mode.

The same corner scan computes both modes in Quadrants. Full mode changes
observation writes, not loads, displacement, equilibrium budgets or rigid
motion. See [OUTPUT_MODES.md](OUTPUT_MODES.md) for acceptance requirements.

## Model and errors

`contact_model="intrinsic"` retains the collision geom/pair witness, reference
boundary face and barycentric coordinates. Material normals point away from
the owning tetrahedron, including cavity faces. A finite footprint follows
actual surface adjacency from its witnessed anchor; nearby disconnected
sheets are never merged. Verified covered planar segments and adjacent-face
unfolding supplement the shared-edge port graph. Its distances can
overestimate continuous intrinsic distance; `surface_edge_subdivisions=8`
declares the current approximation. Q10 integration uses a compact Gaussian
profile of width `0.45 * contact_radius` and a squared taper. Local fan/disk
refinement retains the physical radius and every eligible sample.

The original force application point is preserved. Projection to an anchor
adds `(point - anchor) × force` to its free moment. Vector tractions obey
the true local material-normal Coulomb cones, and signed P2 nodal scatter
preserves all six resultant wrench components. The default
`intrinsic_load_partition="componentwise"` conserves each point wrench.
`intrinsic_support="pair_union"` explicitly lets each component use its
collision pair's finite disks, retaining each destination's radius and friction.
`intrinsic_load_partition="body_source_prior"` instead conserves the complete
link wrench, using every original contact force as its own local traction
prior. This declared partition model supports separate contact islands and
self-balanced compression; individual disk resultants may differ from their
priors. See [BODY_SOURCE_PRIOR_MODEL.md](BODY_SOURCE_PRIOR_MODEL.md) for the
objective, constraints and original-point acceptance budgets.

The intrinsic workspace grows to the observed complete source/sample count.
Contact birth/death, changes to position/radius/friction, release and partial
reset invalidate or verify reuse. No source, sample or hard environment is
discarded. CUDA uses cooperative fitting and warp-aggregated sampler counters;
CPU retains its canonical native numerical route. Physical loads, geometry,
residuals and stress use FP64 in the qualified profile; warm Newton Hessians
use FP32, with FP64 gradients, cold retries and unchanged final acceptance.

The retained `contact_model="pressure"` selects the earlier pad law.
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
are included in live timing. See [CONTACT_REPAIR_20261010.md](https://github.com/Roushelfy/Genesis/blob/1606ee3a9e9628a935ad5e446984f920af3adc66/research/rigid_stress/docs/CONTACT_REPAIR_20261010.md).

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

For `contact_model="pressure"`, `fused_pipeline=True` captures the serial association, pressure, scatter,
complete solve/residual/peak and acceptance stages in one native CUDA graph
when cooperative exterior inverse recovery is available without load history.
Both output modes use it. Other configurations retain the independently
callable native passes. `--unfused-pipeline` supplies the matched ablation.
Intrinsic recovery retains its dynamic source/sample growth and native
dispatch; requesting `fused_pipeline` does not enable the pressure graph for
that contact model. The benchmark records the effective selection.

`fused_rigid_post=None` additionally selects a serial graph containing the
unchanged rigid contact-force postprocessing, full stress pipeline and rigid
integration. The current automatic selection uses it below B=8192 when one
configured stress link is eligible for the fused pipeline. `True` requests
it at any batch size; `False` keeps those three dispatches separate. Multiple
stress links and configurations without the fused stress pipeline retain
their ordinary execution order. This scheduling option changes no load,
solve, acceptance or rigid arithmetic. Standalone recovery remains available
for pre-integration snapshot measurement.

The benchmark exposes `--rigid-post-fusion auto|on|off` and records both the
requested and effective selection. The example exposes
`--unfused-rigid-post`. Paired changing-contact trajectories validate exact
rigid/phase agreement and unchanged nodal, full-field and peak budgets through
partial resets and restored states.

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

Four build-time controls select measured large-batch scheduling. Their `None`
defaults use the original routes below 8192 environments and the following
routes at B >= 8192; explicit settings support matched ablations:

- `contact_wrench_reuse`: project the actual integrated contact wrench rather
  than reduce it again over every node. A shared 6x6 centrifugal wrench map
  adds 288 bytes. Complete task overflow uses the original full nodal reduction.
- `contact_moment_reuse`: integrate the current footprint's pressure Gram and
  six P2 load moments together. Strictly positive affine pressure contracts
  these moments; clipped, locally refined and overflowing cases perform the
  complete original integration. This assumes no temporal contact identity
  and adds 217 bytes per bounded face-task capacity (plus a one-byte disabled
  backend placeholder). It requires CUDA cooperative pressure/face scatter.
- `packed_block_size`: use 512-thread inverse blocks. Allowed explicit values
  are 0 (original), 256 and 512. Boundary node order and every nonzero column
  remain unchanged.
- `coalesced_residual`: change only the node/environment scheduling of the
  complete residual product and reduction. All six gauge rows remain checked.

Equal declared geometry, material and mass share only immutable operators;
contact caches, loads, fields and history remain private to each link and
environment. Link-filtered pose notices invalidate the affected stress link.
Existing unbounded dynamics notices conservatively invalidate every watched
link in the selected environments. A single native mutation emitting both
geometry and dynamics clears each affected observation/history once.
All configured options are immutable
after build; mutation raises with a rebuild diagnostic from recovery/data/
observation access. Dynamic radii remain supplied through their ordinary setter.

The Franka workload computes its phase, joint targets and radius schedule in
one Quadrants configuration kernel on CUDA, retaining public control/range/
radius setters and ordinary scene reset. `--torch-controller` selects its
original input scheduling for ablation. This changes controller dispatch,
not stress arithmetic, dynamics, contact law or final acceptance.

Contact rejection reports the affected link, environment, contact slot,
original radius and actual friction coefficient. The diagnostic distinguishes
invalid input/friction, ambiguous or unresolved surface provenance,
unresolved integration, numerical fitting failure, and independently certified
incompatible force/wrench cones. An unresolved fit does not establish physical
infeasibility. The first failure's original point, radius, force, free moment
and aggregate target are retained until the error is cleared. Diagnostic host
transfers occur only when the rigid solver reports an error.

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
