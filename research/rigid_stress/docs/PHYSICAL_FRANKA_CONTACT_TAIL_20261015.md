# Finer contact tail, full-arm inputs and fixed-polyhedron reference refinement

This continues [contact regularization and inertial validation](PHYSICAL_CONTACT_REGULARIZATION_20261015.md)
under the complete [physical-validation goal](LOAD_MODEL_PHYSICAL_VALIDATION.md).
Completed input/control/constructor checks are separated from live reference
work and from physical stress/contact convergence.

## Contact-only tail with recorded fixed initializer compliance

Three finer fixed-hand controls714426_0/1/2 complete the same64-target/52s
protocol at0.3125ms integration interval. Contact time constants are2.5,1.25
and0.625ms. All actual weld parameters remain
[0.0025,1,0.9,0.95,0.001,0.5,2]. Geometry, material, mass, friction, actuation,
control cadence and duration are unchanged. Every case retains4608 distinct
same-substep snapshots, including final runtime geometry.

The public equality setter indexes the static equality count, zero in this
runtime-weld-only scene. The opt-in research driver therefore updates only
the time-constant entry of active weld slot0/env0 through its existing field.
Full-array readback verifies the requested value, the other six parameters
and all other slots. No production API/default or shared configuration changes.

Audit714448 repeats the finite box-face projection and raw penetration checks.
It verifies fixed actual weld/contact metadata and compares80 exact common
released-phase times without interpolation.

| Contact time constant | Final penetration, contacts16/23 (um) | Final normal forces16/23 (N) |
|---|---:|---:|
|2.5ms|42.8405 /42.8299|0.395136 /0.395038|
|1.25ms|10.8209 /10.8183|0.398386 /0.398288|
|0.625ms|2.71126 /2.71060|0.399221 /0.399123|

Common-time normal-force L2 changes decrease from0.81023% to0.20642%, with
maximum sampled changes0.003250N and0.000835N. These contact-law force
diagnostics are below the original3% target and show a decreasing tail.
They do not establish stress, patch-area/shear, angular settling or spatial
accuracy. The finest geometric overlap is still about0.542% of the0.5mm wall;
it remains a solver-regularization quantity rather than material deformation.

## Corresponding full seven-arm-joint application inputs

Two original `FrankaEgg` application captures714431_0/1 retain the original
approach/grip/lift plus the declared stationary hold at the same0.3125ms
integration interval. All seven arm joints and both fingers remain dynamic.
Only per-geom contact time constants differ:10ms and0.625ms. Joint/equality
parameters are unchanged. Each capture has576 distinct pose/force/geometry
frames, including320 consecutive physical substeps over the final0.1s hold.

These inputs now retain all nine joint positions, velocities and control-force
readouts, actual arm/finger gains, generalized armatures, passive damping,
hand linear/angular velocity, mixed contact solver parameters and penetration.
Arm gains are4500/4500/3500/3500/2000/2000/2000, with velocity gains one tenth
of those values; finger gains remain100/10 with4N hold caps. Actual generalized
armature readouts are0.1 for all nine coordinates. Units differ for arm rotations
and prismatic fingers; those readouts are not relabeled as one force unit.

Strict input audit714443 verifies all12 finger geoms and runtime mass/COM.
All480 hold frames per case retain the original1e-10m slider guard; largest
reconstruction errors are1.875e-16/2.227e-16m. Full-control audit714454 verifies
identical saved initial/control conditions, identical geometry/material inputs
and controller settings, plus320 exact paired physical times/source identities.

| Full application contact constant | Final normal forces16/23 (N) | Final penetration16/23 (um) | Angular body speed (rad/s) |
|---|---:|---:|---:|
|10ms|0.353191 /0.353093|461.527 /461.448|0.094570|
|0.625ms|0.399607 /0.399509|2.71596 /2.71529|0.086127|

Common-time normal-force L2 change is11.6166%. Hand speed is below1.35e-9m/s
and hand angular speed below3.82e-9rad/s in the final samples, while the object
continues to rotate. These are actual full-arm application observations, not
a deformable full-arm pass or a two-level convergence tail. The fine contact
setting changes physical solver outputs materially even when the hand is
nearly stationary; its effect cannot be assigned to elastic deformation.

## Spatial references keep geometry and actual held points fixed

`refine_franka_reference.py` uniformly refines the same declared polyhedral
egg:480,3840 and30720 tetrahedra. It preserves geometry, material, total mass,
COM, all actual rigid finger vertices/triangles and the same six physical
initializer points/components. A nearest-coordinate guard verifies those
points at1e-12m. Selecting new QR pivots is avoided. Surface contact triangles
refine with the volume mesh, so this is a combined volume/contact-discretization
axis at fixed polyhedral geometry, not a volume-only or CAD-curvature sweep.

Qualifications714444_0/1 complete body mass-mode, actual jaw inertia, virtual
work and no-contact derivative checks. Total mass remains0.0065298482104kg;
energy/gradient differences remain below3.49e-7/2.54e-7, and gradient/Hessian
differences below2.84e-11/1.74e-11. The30720-tet original qualification is
terminalTIMEOUT at30min and produces no qualification result. Its wrapper
returncode file says0, so allocator status and missing results take precedence;
it does not count as accepted evidence. The isolated sparse study remains live.

The cold480-tet case714449_0 repeats all64 supported actuation increments,
then80 inertial steps at1.25ms. Its final nonlinear lower/upper peaks are
156447.0168565/156447.0171845Pa, identical to the previous same-resolution
pilot. The3840-tet cold study remains live. Cold configuration and current
target/fixed-point records are saved before potentially failing increments.
These qualifications/one-level results do not establish the spatial tail.

## Sparse mass constructor preserves every tested equation

The original inertial constructor assigns a sparse body mass into a LIL
subblock. Inspection of the actual SciPy implementation used by714459 records
its `x.toarray()` fallback. That path introduces an unnecessary quadratic-memory
temporary. `SparseInertialRigidFingerContactSystem` instead uses sparse block
assembly and18 scalar COM-block entries. Physical inertia, armature, gravity,
PD clipping/passive damping, contact, constitutive response and force budgets
are unchanged.

Qualification714459 finds exactly zero mass-entry, active-force and tangent-entry
differences against the original constructor on the solved active-contact pilot.
The sparse cold480-tet run714462_0 repeats the complete closure and80-step
release. Every npy payload in all83 matched NPZ archives is byte-identical to
714449_0, including reference geometry, displacement/velocity/acceleration,
contact/applied loads and full body tensor/peak fields. ZIP metadata is excluded
from this array-payload comparison. This verifies numerical equivalence; no
uncontrolled wall-time or high-resolution memory-speedup claim is made.

The original long studies retain their captured canonical source. A reusable
isolated runner copies inputs, applies the one-call constructor patch only in
the executable snapshot, asserts module origin and records input/executable
before/after hashes separately. The30720-tet sparse study714462_1 is live.
Existing unpatched studies are kept as their original attempts, not silently
replaced or counted as physical evidence.

## Full-egg initialization failure is retained

The longer full-egg reference714003_3 is terminalFAILED after2h23m, at supported
initialization increment384. Lagged friction cycles around1e-6N residuals and
does not meet the original budget. Its original configuration, previous/current
states, contacts, complete iteration log and source closure are retained.

`recover_egg_initialization.py` reconstructs the exact failed geometry/materials,
all fixed rows/target values and previous friction state, then differentiates
current friction through geometry and relative displacement at the unchanged
1e-9N budget. Job714465 completes the identical failed increment: its initial
current-force residual is2.4182e-9N, and one additional Newton update reaches
4.1085e-12N with all fixed values unchanged and no mesh intersections. This
recovers only that increment. A copied-driver patch verifies its original
boundary target at increment384 and continues every remaining initialization
increment, supported force-control ramp and free gravity stage. Original
continuation714486 fails its resumed equilibrium guard. The v2 copied driver
restores the saved six body-hold rows explicitly, retains the pad rows, checks
the original target at increment384 and records the guard values. Job714492
is terminalFAILED: its complete free residual is1.00925e-6N, above the original
1e-9N budget. Exact npy payload hashes show that both regenerated shoe meshes
match the original saved vertices and tetrahedra, so mesh topology differences
do not explain the discrepancy. Same-state operator audit714502 completes:
all three part stiffness matrices, mass modes, rigid modes and Gram matrices
have exactly zero differences. Both reconstructed systems give4.1085e-12N
free residual, with4.6085e-18N gradient difference. This rules out the tested
bulk assembly/projection difference, but does not reproduce or explain714492's
earlier residual. The cause remains unverified; no full-egg acceptance is claimed.
The initial audit launch714501 fails before entering the driver because the
Slurm wrap uses `/bin/sh` while the shared environment requires Bash; its log
is retained. The corrected submission explicitly invokes Bash.
Continuation714503 requalifies the identical saved increment with its original
preceding friction state, fixed values and1e-9N budget before advancing the
remaining original targets. Its saved-step guard passes at4.1085e-12N with zero
additional Newton updates in that process, and it advances into the remaining
initialization. This differs from714492's observed initial residual; the
earlier discrepancy remains unexplained by the independent audit. It is running and remains
unqualified for complete initialization/contact acceptance.
The original third-level U714003_0 remains live on its original handle/source.

## Reproduction and remaining full goal

Finer controls use `capture_franka_fixture.sh` with `--substeps 32`,
`--weld-timeconst 0.0025` and the recorded contact time constant. Full-arm
captures use `capture_franka_reference.sh` with `--substeps 32`,
`--contact-timeconst` and `--dense-hold-tail-steps 10`.
`audit_full_franka_controls.sh` checks actual shared conditions/controller inputs.
`dynamic_franka_reference.sh` adds `--body-refinement`, `--cold-closure` and
the original64 control increments. `run_sparse_franka.sh` executes the isolated
constructor replacement; `recover_egg_initialization.sh` diagnoses the failed
same-step equilibrium. Exact Slurm submissions and fresh runtime directories
are retained under the established physical data root.

Full solved deformable application matching, reference spatial/quadrature/
regularization/history convergence, per-source/island traction/model comparisons,
explained global extrema, matched stress maps and representative verified
physical accuracy/cost remain required. This checkpoint keeps every original
A/B/C, geometry and deliverable requirement active.

## Immutable publication identity

The following manifests retain exact complete originals, with zero prefix-only
verifications. Runtime configuration, source closures, commands and both accepted
and failed cases remain distinguishable.

| Checkpoint | Verified original entries | Manifest SHA-256 |
|---|---:|---|
| `franka-tail-checkpoint-v1` |17268|`b6f4c121abf03c1bf67c7a27a7a72b2edb7b8652a2bcc0e49984da52c86c9ce2`|
| `egg-init-recovery-checkpoint-v1` |19|`1a327c92a4e557477bf122bb77a4284df6aebe2e581d6c41fc2f8e3ccdb79e5d`|
| `reference-failures-checkpoint-v1` |58|`85b3a3e96a3b5d64cc5a559c725842bf6a66f106f71360442d8371ab169318c5`|
| `egg-resume-failure-checkpoint-v1` |37|`25616b098346ea679d242a8c1f86509f20f08e718e80024670e0e9e993c80fef`|
| `egg-operator-audit-checkpoint-v1` |19|`8c12ef247c516132b1d9ce060f51ad7d0c4e8e71a95e3570fd4be83912f522c6`|

These immutable manifests record state at publication time. Their earlier
"live" observations do not override subsequent authoritative Slurm terminal
status. In particular,30720-tet original qualification714444_2 isTIMEOUT despite
its wrapper returncode0, and resumed-egg jobs714486/714492 areFAILED. A failure
archive or a successful publication verifier is not physical acceptance.
