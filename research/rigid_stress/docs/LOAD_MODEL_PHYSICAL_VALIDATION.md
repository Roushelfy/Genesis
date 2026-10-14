# Load-model validation and physical convergence

User direction, 2026-10-14. This is the active next stage for
`Roushelfy/Genesis:rigid-stress-recovery`, following the completed low-mesh
[nonconvex performance milestone](NATIVE_NONCONVEX_RESULTS_20261014.md).
The starting branch is `7a7f25aa2f4699032603b17f997509c3316c3bca`.

## Goal

Determine how accurately the rigid stress feature predicts stress for declared
finite contact loads, and how its load-distribution models affect that prediction.
Validate against an independent open-source finite element method (FEM), analytic
benchmarks where applicable, and resolved deformable-contact references.
Establish mesh, geometry, contact-integration and intrinsic-distance convergence
on several convex and nonconvex shapes. The result must identify a defensible
accuracy/cost operating range and concrete limitations, rather than only prove
that two implementations solve the same assembled system.

The previous performance-first gate has been satisfied for its documented
workload. Physical convergence studies are now explicitly authorized. Choose
a useful, affordable progression from small central-processing-unit (CPU)
cases to representative native graphics-processing-unit (GPU) checks. Do not
repeat the completed large-batch performance campaign before doing this work.

Keep the production feature inside the existing Genesis rigid solver, with
Quadrants for production numerics and rigid dynamics unaffected by recovery.
Independent external FEM/contact software is a validation reference, not a
replacement production backend. Geometry, linear material and mass remain
fixed within each case. Contact positions, counts, radii, friction, directions,
symmetry and hotspots are unrestricted. Keep full/max outputs and global
unaveraged maximum stress. Preserve both existing contact-model options.

## Questions to answer

1. Does the native elastic recovery agree with independently assembled FEM
   for the same physical problem, including load transfer and inertia relief?
2. How much do `pressure`, intrinsic `componentwise`, `pair_union` support
   and `body_source_prior` change pressure, shear, internal stress and peak
   location for identical inputs? Distinguish incompatible model inputs from
   a failed numerical solve.
3. When does preserving only aggregate wrench move load between real contact
   patches or islands enough to bias stress? Does the local force prior retain
   the correct response under self-balanced gripping and bending?
4. Which mesh and contact-discretization settings provide stable stress for
   each shape, load and finite footprint? What remains singular or unresolved?
5. Against a deformable-contact reference, when is rigid-plus-linear-elastic
   recovery a useful approximation, and when does omitted contact/deformation
   feedback materially change loads or stress?

## Separate three levels of evidence

### A. Independent elasticity and boundary-load verification

Use at least one established open-source FEM package, such as scikit-fem or
DOLFINx, with its own assembly, quadrature, boundary-load integration and stress
evaluation. Pin versions, element conventions and source identity. Reusing
the research `P2Shell` or a different sparse solver alone is not this check.

First compare equivalent discretizations with identical prescribed nodal
loads to isolate elastic algebra. Independently assemble stiffness, consistent
mass and rigid modes. Verify complete equilibrium, energy, displacement modulo
rigid gauge, full stress and peak. Compare loads from the same continuous
traction field using each package's boundary integration separately.
Then compare independently generated meshes at common physical points or
regions. Stress smoothing must not hide elementwise differences.

Use simple analytic or manufactured checks with all assumptions stated:
uniform strain/stress, bending or torsion in an appropriate smooth geometry,
and a sphere/spherical-shell load with a known solution if suitable.
Use the boundary conditions required by that solution. A fixed support for an
analytic test is not equivalent to the feature's free-body inertia relief.
Offline supported tests may diagnose mechanics, but native acceptance must
also cover its actual supported free-link problem.

### B. Load-model sensitivity with identical rigid inputs

Freeze reproducible contact snapshots, reference geometry, material, mass,
pose, angular velocity and original force points/wrenches. Hold the physical
footprint radii and friction fixed when comparing models. Audit provenance and
collision/reference normal compatibility before interpreting a fit failure.

Compare the applicable models above against an independently constructed
finite traction reference. Include equal-and-opposite gripping, force couples,
asymmetric contacts, near-sliding friction, changing counts and multiple islands.
A net wrench near zero is not evidence of small internal stress.

Record each source/patch/island resultant and moment as well as the total.
For `body_source_prior`, quantify how far each recovered local wrench moves
from its prior and how that movement changes the interior stress. Aggregate
wrench conservation and a small residual cannot establish local load accuracy.
Assess supported area, pressure/shear concentration, integrated traction norms
and strain energy, including unstable concentration in tiny-weight samples.
Do not declare a model better merely because it accepts more rigid contacts.

Study physical radius/profile sensitivity separately, with each choice recorded
as a different load model rather than a refinement of the same problem.

A point wrench does not uniquely identify pressure. Use independently specified
physical traction profiles, documented pad compliance or solved contact to
supply references. Make modeling choices explicit. If models legitimately
impose different constraints, report their differences rather than call them
numerical error. Invalid or unsupported cases remain visible.

### C. Deformable-contact reference

Build representative open-source FEM contact references that solve the contact
loads and deformation together. Begin with controlled simple geometry, then
include the full hollow egg grasp and a nonconvex case with an interior hotspot.
Use a documented, converged unilateral/frictional contact formulation.
Independent contact can use another package or backend. Record its equations,
contact regularization, stopping criteria and convergence evidence.
Standard FEM elasticity alone does not validate a contact algorithm.

Match physical geometry, material, friction, pad/gripper compliance, supports
and loading protocol. State whether control is force or displacement based.
Do not simultaneously impose an arbitrary contact force and jaw displacement.
Compare corresponding actuation/forces, patch locations, pressure/shear,
per-island wrench, internal stress and peak position. Displacement-controlled
rigid and deformable scenes can produce different contact forces. Separate
that feedback error from load-mapping error by replaying the resolved reference
loads through native recovery.

Use small-strain, quasi-static cases first. Compare inertia relief or actual
supports consistently. Show that deformation is small relative to relevant
thickness, gap and footprint scales, not only to the object diameter.
If large rotation, buckling, dynamics or failure becomes important, identify
the limitation and use a suitable reference only where it answers the question.
This stage need not become a new deformable production solver or full RL study.

## Shape and load coverage

Use a compact representative collection, extending it when evidence exposes
a distinct failure. The table guides coverage rather than fixes asset filenames.

| Geometry | Main purpose | Representative loading |
|---|---|---|
| Solid sphere/ellipsoid | Smooth convex control | Opposing finite patches, off-axis grip, friction |
| Full hollow sphere/egg | Curvature, thickness, Franka workload | Narrow/broad pads, asymmetric grip, slip, supports |
| Rounded block/capsule | Smooth flat/curved transitions | Edge-near grip, normal plus shear, multiple pads |
| Rounded U/C-shaped link | Concavity and bending with an interior hotspot | Inner-wall contacts and asymmetric arms |
| Rounded thin-neck/dumbbell | Interior hotspot | Opposing end loads, offset bending, torsion |
| Cavity/narrow-gap branches | Internal surfaces, separate islands | Inner/outer contacts, same-facing sheets, motion |

Use full three-dimensional models without relying on symmetry. A symmetry
reduction can be an additional cross-check. A deliberately sharp reentrant
corner or point load can serve as a singularity control, labeled separately.
Document thickness, neck/gap width, fillet radii and mesh quality. Keep the
Franka/full hollow egg as the principal application. Do not search only near
contacts for stress maxima.

## Convergence protocol

Prescribe a fixed continuum geometry and load law for each sweep. Keep physical
patch size, profile, force, friction and geometric rounding fixed as the mesh
changes. Carry anchors in physical/reference coordinates, not persistent face
indices across remeshing. Regenerate nodal loads consistently.
A fixed total force smeared over a shrinking nodal neighborhood is a different
load problem and cannot establish finite-footprint convergence.

Separate and then combine:
- Volume/surface mesh refinement, including through-thickness resolution.
- Geometry approximation error, especially shell curvature and fillets.
- Contact area quadrature, including small disks and boundary-crossing patches.
- Intrinsic-distance approximation and port/unfolding resolution.
- Contact reference regularization and spatial discretization where applicable.
- Linear/contact solve and arithmetic error, kept below discretization changes.

Use enough levels to show a credible trend and tail. Three or more useful
levels is a starting point, not a pass merely because two adjacent values match.
Include local mesh refinement where contact or a neck dominates.
Prefer an independently refined reference and check more than a scalar maximum.

Track unaveraged global von Mises maximum, its physical location, full tensor
and von Mises errors on common probes/regions, energy and displacement,
per-patch wrench, integrated pressure/shear and complete residuals.
For affine P2 tetrahedra, audit the existing unaveraged four-corner maximum
against the independently reconstructed linear stress field. Do not substitute
a quadrature-only or smoothed nodal maximum. If reference geometry/elements
differ, verify their field extrema with appropriate sampling or optimization.
Report observed refinement changes with absolute errors near zero.
Choose and declare practical accuracy criteria before interpreting the curves,
and distinguish discretization convergence from model accuracy.
A suggested initial engineering target is a few percent for a smooth finite
load, to be justified per observable rather than imposed on every geometry.

Finite-area loading does not guarantee a bounded peak at a sharp reentrant
corner, abrupt boundary-condition transition or contact edge. Diagnose persistent
peak growth using hotspot location, local fields, geometry/loading regularity
and independent FEM. Keep the original raw maximum in every report.
Fixed physical-region averages or quantiles may be supplemental damage metrics,
explicitly defined in physical units and never substituted silently.
Rounded geometry is a different declared physical case. Do not tune material,
radius, friction, averaging region or solve tolerance to manufacture convergence.

## Deliverables and completion

Produce reusable meshes/loads/contact snapshots and opt-in scripts in this
repository. Keep long fine-mesh studies outside routine unit tests.
Record complete successful and failed cases, immutable configuration and
source hashes, CPU/FEM package versions and hardware. Use the existing data
root and publication conventions without duplicating unnecessary artifacts.
Commit and push useful progress as it becomes available.

Provide a concise report with convergence curves and side-by-side stress maps
using matched units/color limits and marked contact patches/interior hotspots.
Include traction maps and per-island redistribution for load-model comparisons.
Give exact commands and a compact table of verified settings and limitations.

Complete this stage when the representative convex/nonconvex matrix has:
independent FEM agreement, explained convergence or singularity outcomes,
quantified model sensitivity, and meaningful deformable-contact comparisons
for the egg and an interior-hotspot case. Unsupported/unavailable reference
checks are explicitly unverified, not counted as completed evidence.
If an algorithm defect or unjustified load-model bias is found, diagnose and
fix it within the native feature, or state the remaining modeling limitation
with measured evidence. Preserve the qualified fast path and quantify material
accuracy/cost tradeoffs at representative resolutions after any selected change.
A full RL training campaign and experimental fracture calibration are separate.

Choose reference packages, mesh generators, sweep sizes and numerical methods
with engineering judgment. Existing historical convergence and scikit-fem
experiments are useful starting points, not acceptance of the current law.

## Starting references

- [Current load-partition law](BODY_SOURCE_PRIOR_MODEL.md), [native API](NATIVE_API.md)
  and [completed numerical/performance evidence](NATIVE_NONCONVEX_CLOSEOUT_20261014.md).
- [scikit-fem documentation](https://scikit-fem.readthedocs.io/en/stable/) and
  [examples](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html).
- [DOLFINx elasticity demo](https://docs.fenicsproject.org/dolfinx/v0.10.0.post5/python/demos/demo_elasticity.html).
  Pin the actual installed version. This is an elasticity reference, not a
  ready-made frictional grasp-contact benchmark.
