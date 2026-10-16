# Fixed-boundary thickness controls and complete initializer qualification

The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains
active. This checkpoint prepares two separated physical-error studies and
closes the recording/independent-audit qualification needed for complete cold
initializer coverage. Geometry and load-only assets are not accepted stress
states. Pending studies below retain every original physical parameter and gate.

## Volume thickness changes while the actual contact trace is preserved

The [near-surface tensor diagnostic](PHYSICAL_CONTACT_DISTANCE_DEPTH_20261016.md)
finds unresolved local spatial changes at 1–50 µm depth. All three preceding
contact meshes retain two layers across the 500 µm wall. Job 719593 constructs
separate 2/4/8-layer controls from the original covered 31668-tet geometry:

|Thickness layers|Tetrahedra|Vertices|Actual compact IPC surface vertices / triangles|
|---|---:|---:|---:|
|2|31668|7923|21294 / 42532|
|4|63336|13205|21294 / 42532|
|8|126672|23769|21294 / 42532|

All original vertices, including the middle layer, remain byte-preserved.
New internal vertices interpolate within the original radial intervals; there
is no CAD reprojection, rounding or boundary relocation. Both original boundary
triangle sets and their canonical P2 coordinates match exactly. The two-layer
NPZ is the original asset byte-for-byte. Internal faces are manifold and have
opposite incident orientations. Volume, area, mass, COM and inertia match at
the original geometry tolerances. Represented volume remains
3.7120804080153644e-6 m³ and mass remains 0.007424160816030729 kg. The unchanged
polyhedron's approximately −0.845% mass error relative to the declared continuum
egg is retained; radial refinement does not repair that geometry error.

All six original physical initializer anchors remain within 1e-12 m of P2
nodes. Job 719606 then independently reconstructs the **actual existing** P2
boundary subdivision, compact surface and IPC interpolation for each asset.
Vertex correspondence is bijective, retains topologically distinct coincident
finger vertices, and preserves every triangle, group and lumped area. Maximum
surface-position differences are 0, 8.674e-19 and 8.941e-19 m. Zero/affine IPC
source-displacement checks pass and initial surfaces have no intersections.
These are measured coordinate differences, not a claim of bit-identical
reconstructed IPC coordinates at the finer levels. No stiffness, contact force
or equilibrium state is solved by either geometry job. The new internal
simplices are not claimed to be per-parent nested partitions.

## Record every original initializer control without changing its solution

The opt-in `record_paired_initializer.patch` adds output only inside isolated
reference snapshots. It retains the actual held coordinate map, constraints,
body component permutation, applied gravity, jaw target and all 64 accepted
displacements, previous friction displacements, full gradients and contact
forces. The original cold protocol and five dynamic steps are unchanged.

Producer 719626 repeats the original 1920-tet Schur/METIS case. Dynamic auditor
719627 freshly passes all original 1e-9 N force gates. Comparator 719628 then
checks every original saved array and every corner/quadrature stress component:
all measured differences are zero, all geometry/map NPZ files match, and final
initializer bytes match. The comparison does not enable the historical
resolved-initializer allowance. This qualifies recording only, without inferring
physical accuracy from numerical agreement.

Independent initializer auditor 719655 reconstructs all 64 original held
equations. It uses each actual predecessor and static 1e-8 m friction mollifier,
constant gravity and the original capped proportional jaw effort. It introduces
no dynamic inertia, velocity damping or timestep-scaled friction into these
static equations. The complete original held map is reconstructed exactly;
six constrained body reactions remain explicit. Every corner and integration
point stress, raw whole-volume peak interval and determinant bound is retained.

The maximum independent held residual is 5.312475e-11 N against the unchanged
1e-9 N gate. Maximum full-gradient reconstruction difference is 3.790164e-13 N.
The strict per-component saved-contact-force comparison has zero maximum
difference. Maximum full quadrature/corner-deformation tensor discrepancy is
4.656741e-8 Pa at the unchanged 1e-10 relative / 1e-6 Pa absolute gate. All 64
controls retain 16588800 explicit quadrature tensor components. The largest
raw lower peak over the initializer is 357431.390 Pa; corresponding whole-volume
upper bounds remain explicit, rather than replacing this maximum with a smooth
or sampled statistic.

Both original audit failures remain visible with all partial outputs. Job 719642
fails the tensor gate: directly interpolating F also interpolates the identity
with rounded barycentric weights, creating a roughly 2e-6 Pa artificial stress.
The correction interpolates F−I and restores the identity exactly. Job 719647
passes 53 controls before the next strict force comparison fails: the auditor
had used a different ordering for current-position and relative-displacement
mapping than the original IPC contract. Job 719655 reconstructs each stencil
with the actual IPC current/relative-displacement arithmetic. A separately
centered evaluation of those same current coordinates remains a roundoff
diagnostic, with maximum normal-force difference 1.065815e-14 N. No tolerance,
material parameter, load or physical control is relaxed to accept either fix.

## Original continuous traction integration is a separate axis

Load-only job 719549 first reproduces all five original 502230-tet finite-load
vectors exactly using their unchanged original quadrature-level-3 source.
On the fixed 554574-tet mesh it then integrates the same continuum law at
levels 2/3/4, holding every physical center, 3 mm radius, compact profile,
strength, normal/shear rule and friction fixed. All complete nodal vectors,
physical sample positions, normals, weights and tractions remain retained,
with independent resultant/moment and quadratic virtual-work checks.

Last adjacent full nodal vector changes are 2.939e-9 to 3.929e-9 relative,
or 1.532e-11 to 2.671e-11 N absolute. In contrast, the current-facet continuum
integration differs from the previously frozen physical-sample functional by
approximately 0.02114–0.02453% in full nodal-vector norm. The two functionals
remain explicitly distinguished. Neither small changes nor preserved wrenches
certify stress accuracy. This job solves zero equilibrium states.

## Scheduled full solves and reproducibility

Jobs 719713/719716/719719 solve the 2/4/8-layer contact cases with the same
original 64-control cold history and five dynamic states. Their launch guards
require both the complete original dynamic-field equivalence and successful
64-control initializer qualification, plus actual IPC trace qualification.
Dependent dynamic auditors are 719714/719717/719720; complete initializer
auditors are 719715/719718/719721. These jobs are scheduled work, not evidence
that any new radial contact field or physical tail has passed.

Original launch attempts 719657/719660/719663 terminate with exit 13 before the
reference wrapper starts: `execve(): /mnt/home/zhaofeng/.local/bin/env: Permission denied`.
They create zero runtime reference directories or equilibrium states. Their six
unstarted dependent audits are explicitly cancelled. Retrying uses `bash` as the
gs-srun command and sets qualification variables inside the container shell;
no physical source changes. Original errors, scheduler states and exact retry
commands are retained in the [launch follow-up](../evidence/20261014-physical/separated-axes-launch-followup-v1/manifest.json).

Jobs 719652/719653/719654 perform the five complete stress solves at each
continuous-traction quadrature level on the fixed 554574-tet mesh. They must
freshly qualify all five original frozen fields and original force/corner/
quadrature gates before solving, then reintegrate every new full nodal load.
Dependent auditor 719656 separately rebuilds stiffness and consistent mass,
checks all 20 frozen/new complete fields, every original 36-point probe field,
raw maximum and energy, and retains ten adjacent plus 15 frozen-functional
stress comparisons. Its exact tensor-volume formula is the independently
manufactured-qualified original. Pending jobs provide no quadrature/stress pass.

The [immutable qualification publication](../evidence/20261014-physical/separated-axes-qualification-v1/manifest.json)
retains all nine closed jobs, source/configuration closures, actual scheduler
commands, both failed audits and their partial fields, full 64-state successful
audit arrays, original recording states and complete load-only quadrature
arrays. Parent manifests preserve unchanged large original input fields.
New runtime data, logs and caches stay under the established data root.
The qualification manifest SHA256 is
`c2618ffca0a419400e0bb81aa6c611efafde0093f97400efdbf3e785185b3f18`;
all 932 archive entries verify against 930 actual runtime originals, including
one exactly reconstructed large original. The archived plan preserves its
then-scheduled first launch IDs; the follow-up records their actual failures
and replacement IDs rather than rewriting that immutable history.

Entry points are `scripts/refine_contact_egg_thickness.sh`,
`scripts/audit_contact_thickness_trace.sh`,
`scripts/record_schur_paired_reference.sh`,
`scripts/audit_paired_initializer_history.sh`,
`scripts/audit_continuous_traction_load_axis.sh`,
`scripts/solve_continuous_traction_axis.sh` and
`scripts/audit_continuous_traction_fields.sh`.
Exact commands are retained in the manifest's `meta/*commands-v*.json` inputs.
All original A/B/C physical/model/local-island, full egg/Franka and interior-hotspot
application/history/feedback and physically verified warmed native-cost
requirements remain active. No overall completion is claimed.
