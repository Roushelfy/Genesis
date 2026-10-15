# Independent contact refinement and reference Newton qualification

The complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. This checkpoint separates collision-surface discretization from
body discretization in the actual compound-finger fixture. It exposes an
unresolved traction concentration despite nearly unchanged coarse-body stress.
It also qualifies a reference-only linear-solve change needed to make finer
contact studies affordable. Neither result establishes full-arm application
matching or a physical accuracy/cost operating range.

## Fixed bulk, independently refined contact surface

`refined_franka_contact.py` builds a sparse scalar interpolation G from original
P2 body nodes and actual rigid finger nodes to a separate collision surface.
The three levels retain480 body tetrahedra and2970 complete source unknowns.
Only body collision triangles are refined. All12 original compound finger
geometries, fixed polyhedral rest geometry, material, mass, friction, barrier,
64 cold PD targets and80 dynamic steps at1.25ms remain fixed.

Collision vertices follow exact quadratic P2 traces; negative interpolation
weights are retained. IPC pulls forces and tangents back through G, preserving
the source unknowns rather than introducing a new mechanical discretization.
The installed IPC1.6.0 constructor expects a scalar node map and expands it to
coordinate DOFs. Its `displace_vertices` accepts original source displacements;
`vertices` first subtracts collision-rest positions and is incompatible with a
different source-node count. See the pinned primary
[implementation](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/collision_mesh.cpp)
and [interface](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/src/ipc/collision_mesh.hpp).
The isolated runner patches copied sources and records both input and executable
before/after hashes; canonical sources used by live references stay fixed.

Initial API qualifications714718/714721/714729/714736 fail and remain archived.
The first invalid map shape reaches a native signal139. Qualification714743
uses a smooth planar fixture with strictly interior projections. Force pullback
agrees exactly; the maximum Hessian pullback difference is1.819e-12, within
the declared sparse-product roundoff budget. Decreasing finite-difference steps
give final work/tangent errors9.58e-10/3.82e-10. These checks concern the map,
not resolved physical contact.

Actual geometry qualifications714754 fail because the contact-only diagnostic
omits `dhat`; repaired714767_0/1/2 pass. P2 quadratic-trace error is at most
1.085e-19m2. Direct collision-coordinate differentiation and source-coordinate
pullback agree exactly for force and current-friction tangent at all levels.
Finite-difference tangent error decreases to6.691e-5 under the declared1e-4
gate. Actual active rows do not exercise negative interpolation weights.
Manufactured patch714768 activates nine such rows: force/tangent relative
differences are1.986e-15/1.252e-14, and the final current-friction derivative
error is1.958e-9. Its finite-gap friction torque is retained and is separate
from conservation under the mapping.

## Three solved contact levels do not establish a contact tail

All714782_0/1/2 solve the complete original protocol at unchanged mapped/body
dynamic force budgets1e-9N. Level0 reproduces every npy payload in all83
original geometry/state NPZ archives byte for byte. Comparison714801 uses the
qualified2400 common material probes and10 exact physical times.

| Contact level | Collision vertices / triangles | Final raw peak lower (Pa) | Largest mapped residual (N) |
|---|---:|---:|---:|
|0|502 /948|156447.016857|2.5904e-10|
|1|1462 /2868|156443.278545|2.5807e-10|
|2|5302 /10548|156435.487345|2.8087e-10|

Full objective tensor changes are0.002419% and0.005042%; body energy changes
are0.004842% and0.010093%. Separate contact force resultant changes are
0.002392% and0.004986%. These are small changes at one coarse bulk setting,
with no decreasing stress tail established. The original unaveraged nonlinear
peak lower/whole-volume upper bounds remain recorded. All lower peaks retain
the same physical reference vertex as the earlier480/3840 comparison.

Traction audit714800 reconstructs current normal and friction forces independently
from saved states. Pullback matches all saved source contact loads exactly.
All three final states retain two normal stencils, two active body nodes and
two separate active-node islands at the fixed1e-8N activity threshold.

| Level | Active lumped current area (mm2) | Maximum normal force/area (kPa) | Normal density L2 (Pa m) |
|---|---:|---:|---:|
|0|96.51359|8.28321|81.3653|
|1|24.12836|33.13207|162.7267|
|2|6.03209|132.52172|325.4372|

These densities divide each unsmoothed collision-vertex force by its current
barycentric lumped triangle area. The area is a quadrature diagnostic, not an
independently resolved physical patch boundary. Support area quarters while
maximum normal/shear densities approximately quadruple and their integrated
L2 norms approximately double. Each island remains a single active node;
its full six-component wrench is retained in the report and raw surface arrays.
The resulting evidence does not support contact-pressure convergence. It also
does not prove a singularity: coupled local volume/contact refinement and
separate geometry/regularization studies are still needed. Stable aggregate
loads or coarse-body stress cannot replace those checks.

The original common-field comparison JSON retains a generic limitations string
describing joint volume/contact refinement; its explicit
`declared_discretization_axis=contact_refinement`, configurations and fixed
tetrahedron counts define this experiment. This report corrects that wording
without rewriting immutable runtime evidence.

## Cached factorization preconditions the exact current Newton equation

Separated comparison714775 also closes the fixed-seed-contact volume study714550.
The equal480-tet constructors agree exactly. With original collision triangles
held fixed,480 to3840 tetrahedra changes common tensors56.22253%, energy35.15838%
and the raw peak30.81419%. At fixed3840 tetrahedra, switching from seed to refined
contact triangles changes tensors only0.0024223% and energy0.0048504%. Thus this
first spatial discrepancy is dominated by volume discretization. A fixed nodal
contact law can itself retain concentration, so the result does not establish
physical convergence or replace joint local refinement. Fine frozen-contact
maximum complete mapped/body residuals are3.874e-10/3.754e-10N.

`cached_reference_newton.py` keeps the current assembled Newton matrix in every
GMRES product and reuses an earlier LU only as a preconditioner. A failed
iteration or unacceptable true linear residual falls back to a fresh current
LU. Phase/dimension changes invalidate reuse. The original outer nonlinear
force budgets, line search, collision guard and physical equations remain.
This changes independent research references, not production Quadrants code.

Coarse control714804 completes the original480-tet64-target/80-step history.
It performs344 recorded linear solves with three fresh factorizations and1660
GMRES iterations. These counts do not establish a controlled speedup.
Independent equivalence714815 checks all80 saved states and all unaveraged
corner Cauchy tensors against the original solved cold case. Maximum relative
tensor difference is3.2943e-11; contact-force difference is1.1498e-10 and body
displacement difference1.5468e-11. Both runs satisfy all original complete
mapped/body1e-9N gates. A3840-tet cached control714827 is scheduled for the same
all-state comparison714839. The30720-tet continuation714840 depends on successful
completion of both jobs, before this change qualifies finer physical results.

The updated common-field reporting also passes714841: both comparisons retain
exactly the earlier physical observables while explicitly recording
`varying_discretizations=[contact_refinement]`. It sorts contact studies by the
declared level and replaces the earlier generic joint-refinement wording.

## Reproduction and publication

The immutable [checkpoint manifest](../evidence/20261014-physical/contact-discretization-checkpoint-v1/manifest.json)
contains21 terminal runs:14 successful and seven failed diagnostics, with1339
exact original files and571 independently checked source identities. Its SHA256
is `063744bdbfcb8a772c7926f00462cd8b19a178a3458fce308d616844758bcae7`.
The adjacent verification checks stored bytes against every original. Included
`reproduction.json` gives closed-source commands for each original run;
`slurm-status.txt` preserves scheduler outcomes independently of wrapper codes.
The [report-metadata checkpoint](../evidence/20261014-physical/contact-report-metadata-checkpoint-v1/manifest.json)
adds56 exact files for714841 and the exact dependent Slurm submission commands.
Its SHA256 is `4c64ea62771954a6bd903ca73b67bf2ee3cc3700cbe8927828bbbf94dfc91d68`.

For fresh directories under the established data run root, invoke
`scripts/run_refined_contact_franka.sh RUN solve --closure-run CLOSURE
--body-refinement 0 --cold-closure --control-steps 64 --dt 0.00125 --steps 80
--contact-refinement LEVEL` for LEVEL0/1/2. Use its `audit --run RUN0 RUN1 RUN2`
mode for traction/island fields and `scripts/compare_spatial_franka.sh AUDIT
--axis contact_refinement --run RUN0 RUN1 RUN2` for common material fields.
Use `scripts/run_cached_reference.sh RUN dynamic` with the same original
protocol and `--body-refinement LEVEL`, followed by its `qualify --run ORIGINAL
CACHED` mode. Run inside the pinned physical-validation environment through
the project cluster wrapper; material, controls and force gates stay fixed.

## Next acceptance evidence

1. Complete the fixed-contact volume comparison and a third useful body level.
   Preserve the existing56.2% first spatial tensor change until stronger
   evidence explains its origin and tail.
2. Jointly resolve the contact neighborhood and volume while monitoring raw
   traction concentration, per-island wrenches, common-point tensors and global
   peak location. Keep geometry, contact regularization and history as separate
   axes. Declare a singular outcome only with independent supporting evidence.
3. Complete converged full hollow egg/Franka and interior-hotspot U references,
   then distinguish resolved-load replay error from rigid/deformable feedback.
4. Finish applicable identical-input load-model comparisons, physical operating
   ranges, matched maps and warmed costs at verified accuracy.

Original source/configuration, complete successful and failed runs, environment,
hardware, commands and immutable artifact hashes accompany this checkpoint.
No scheduled job counts as completed evidence. The A/B/C matrix and every
original physical-validation completion requirement remain active.
