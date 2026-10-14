# Physical-validation execution protocol

The active requirements are [LOAD_MODEL_PHYSICAL_VALIDATION.md](LOAD_MODEL_PHYSICAL_VALIDATION.md).
Starting source is `12cdceef`, with numerical engine `76537c2f`. This protocol
declares observables and initial criteria before inspecting new sweep results.
It is an execution record, and establishes no acceptance by itself.

## Independent reference and numerical gates

Use scikit-fem's vector quadratic tetrahedral element with independent
stiffness, consistent-mass, rigid-mode and boundary-load assembly. Map degrees
of freedom by physical node coordinates and component, retaining element
stress without smoothing. Recover the full four-corner affine stress field
using scikit-fem basis gradients. Test prescribed nodal loads before separately
integrated continuum tractions. Free-link reference solves use independently
assembled consistent-mass inertia relief and a rigid gauge. Supported analytic
tests declare their own boundary conditions separately.

Native complete equilibrium retains `max(1e-11 N, 1e-8 ||rhs||)`, including
all six gauge rows. Independent same-discretization stress comparison retains
RTOL1e-4/ATOL1e-3 Pa and original wrench tolerance1e-8. Independently assembled
stiffness/mass and manufactured fields must agree at a tighter arithmetic
scale before being used as a physical reference. Record energy, displacement
after rigid-gauge alignment, tensor/von Mises and peak location as well as
the scalar peak. Include nonzero angular velocity and asymmetric original
loads in native checks.

## Initial physical criteria

For smooth finite loads, start with3% tail changes in global peak, energy and
common-probe stress norms over at least three useful refinement levels.
Require a stable hotspot region and a trend toward an independently refined
reference. A single small adjacent change does not establish convergence.
Report absolute errors for near-zero components. Extend any sweep whose trend
or contact/geometry resolution remains ambiguous. Persistent raw-peak growth
requires a documented singularity diagnosis or further work.

Hold continuum geometry, rounding, material, physical radius, friction and
traction profile fixed within each sweep. Separate surface geometry, volume/
through-thickness mesh, boundary quadrature and intrinsic-distance refinement.
Use physical anchors across remeshing. Radius/profile changes are separate
load models. Always retain the raw unaveraged global maximum and location.
Common physical-region metrics supplement that maximum.

For model comparisons, hold rigid geometry/pose/angular velocity and each
original application point, force, moment, radius and friction identical.
Compare pressure, intrinsic componentwise, pair-union and body-source-prior
where their declared constraints admit the inputs. Retain rejected cases.
Record per-source/island forces and moments, pressure/shear, area, traction
concentration, strain energy and interior stress. Compare each local wrench
against its physical traction reference and against its prior. Aggregate
conservation alone supplies no local-accuracy acceptance.

## Representative coverage and contact reference

The required collection covers a smooth solid sphere/ellipsoid, full hollow
sphere/egg, rounded flat/curved control, rounded U/C, rounded thin-neck and
cavity/narrow-gap branches. Include opposed gripping, bending/torsion couples,
asymmetric contacts, friction near slip, multiple islands and changing counts.
Use complete3D geometry with marked contact and interior hotspot regions.

Resolved deformable contact must jointly solve deformation and unilateral/
frictional loads for a controlled convex benchmark, the full hollow egg and a
nonconvex interior-hotspot case. Pin and document the reference formulation,
compliance, actuation, regularization, solve tolerances and convergence.
Replay resolved loads through native recovery to separate mapping error from
contact-feedback error. Compare displacement with wall/neck/gap/patch scales.
Contact formulation verification and resolution/regularization convergence
are mandatory before using those fields as physical truth.

## Execution, artifacts and report

Long studies remain opt-in commands, with small numerical tests in the normal
test path. Runtime data, environments, caches and logs remain under
`RIGID_STRESS_DATA_ROOT`, primarily `runs/20261014-physical-validation/`.
Record failed runs, source before/after hashes, configurations, mesh quality,
versions, hardware, residuals and timing. Preserve the completed performance
evidence. Reuse existing data rather than copying whole historical campaigns.

The final report must include convergence curves, matched-unit/color-limit
stress maps, traction maps and per-island redistribution. Every figure traces
to archived data and reproducible code. Completion requires the full matrix,
independent FEM checks, explained physical convergence/singularities, model
sensitivity and meaningful egg/interior-hotspot deformable-contact comparison.
Unverified cells remain visible and do not count as completed evidence.
