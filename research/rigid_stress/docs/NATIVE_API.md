# Native rigid stress observation

The production implementation lives in `genesis/engine/solvers/rigid/stress/`.
The normal example and benchmark import no research solver. All physical
assembly, contact mapping, factor/inverse construction, recovery, residuals
and peak reduction run in Quadrants. NumPy handles mesh topology at build time;
the SciPy/Numba research implementation is an independent test oracle.

## Usage

Configure a link before `scene.build()`:

```python
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

## Model and errors

Each solved point contact includes its complete normal and tangential force.
A force-line anchor moves the footprint centre on the exterior without
changing its point wrench. Positive Duffy/Gauss quadrature samples a compact
Gaussian/bump pad profile. A nonnegative constant-ratio vector traction
preserves force and the two independent moment constraints perpendicular to
the force; nodal P2 loads may be signed even though physical pressure is
nonnegative. The law is in [PAD_LAW.md](PAD_LAW.md).

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
friction forces nor silently expands a footprint. The B=2048 counterexample
and independent CPU rejection are archived in the native evidence.

Currently supported stress links are independent free roots with the standard
rigid contact coupler, without autodiff, joint supports, rolling or torsional
contact moments. Enabling those unsupported configurations raises at build.
Mass, inertia and COM setters, and external wrench application to a stressed
link, raise because they would change the declared model or omit a required
surface traction. Multiple configured links can share immutable operators
when geometry/material/mass and method agree; their loads and states remain
independent. Disabled scenes retain the original fused rigid step.

## Numerical methods

`method="auto"` uses a full pinned inverse if its shared storage fits
`inverse_max_bytes` (64 MiB default); otherwise it selects sparse direct.
The full inverse maps **every** nodal load, rather than a fixed contact-response
basis. Level 1 requires 47,239,200 bytes in FP64. Explicit `method="inverse"`
rejects an insufficient budget; it never allocates an unbounded dense inverse.

The native block LDL factor is shared across environments. Its symbolic
minimum-degree ordering is Python topology work; every numeric factor value
is computed in Quadrants. CUDA solves on at most 1,024 P2 nodes use a warp
per environment and shared work storage when `cooperative_solve=True`.
Other meshes/backends use the serial native triangular solve. This fallback
is correct but has no large-mesh performance claim in this milestone.

`inverse_precision="32"` halves inverse storage while retaining scene
arithmetic. Residual failures receive masked inverse corrections followed by
the native factor fallback. Two corrections alone were insufficient on
arbitrary asymmetric loads; the fallback is essential. Default scene-precision
inverse storage is the conservative configuration selected by measurement.

`method="pcg"` supports scalar-diagonal or block preconditioning, independent
environment convergence and a strict iteration limit. `warm_start=True`
reuses the same environment's displacement. Thin-shell conditioning made
the simple PCG path slower and unresolved at 4,000 iterations in the probe;
it is retained as an explicit option with residual rejection, not the default.

Use [RUNNING.md](RUNNING.md) for reproducible commands and the native results
report for measurements. These low-resolution checks establish numerical
consistency and performance; physical mesh/quadrature convergence is deferred.
