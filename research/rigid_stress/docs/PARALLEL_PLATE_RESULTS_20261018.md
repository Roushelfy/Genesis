# Parallel-plate stress comparison: measured results and unsupported quantities

The remote task specification is commit`914a558df`,
[PARALLEL_PLATE_COMPARISON_GOAL.md](PARALLEL_PLATE_COMPARISON_GOAL.md).
The previous Franka qualification campaigns remain stopped. This report closes
the bounded plate comparison under the specification's explicitly diagnosed
unsupported-quantity clause. It does **not** establish a physical accuracy
ranking. The [requirement audit](PARALLEL_PLATE_AUDIT.md) distinguishes measured
results, unsupported quantities and optional work that was not run.

The simple scene now reaches verified QIPC and DPR mechanical equilibria.
It exposes three separate limitations: unresolved QIPC point-contact spatial
fields, SCR rejection near a cancelling body wrench, and original PR22 Shell's
contact projection outside its implicit elastic solve. These limitations must
remain visible in the comparison; a successful subset cannot replace
the rejected original baselines.

## Scene and admission

**QIPC variant label:** all QIPC plate results in this report use the separately
pinned roundoff diagnostic inherited from the
[preceding reference study](QIPC_REFERENCE_RESULTS_20261018.md), not an
unmodified-main result. On original revision`fe252bd30`, one non-KKT line-search
comparison permits`16 * double_epsilon * abs(E0)` beyond E0; energy,
derivatives and CCD are unchanged. The captured patch SHA256 is
`c9ecba4791247396c4bad5cf2937a9bb23973e9182296cdcb4a94da0ee1c32ea`.
The original-main stagnation evidence remains in that earlier study. “QIPC”
in the compact tables/figures means this diagnostic candidate throughout;
passing results are not attributed to an unmodified upstream binary.

Full free hollow egg, E=10GPa, nu=0.3, density2000kg/m3, gravity0. Two120mm
square plates are10mm thick; the lower is fixed and the upper50g plate has
one axial slide with50Ns/m damping. No egg vertices or surface symmetry
constraints are imposed. The common normal force ramps over0.1s; separation
is measured. Native mechanics needs2s settling holds, whereas the short QIPC
pilot evaluates0.3s endpoints. They are compared as candidate equilibria,
not as identical transient histories. The separate friction/tangential control
uses the same2.5s history for all methods, as described below.

Corrected QIPC setup uses its original finite-stiffness guide, explicit root
world position, collision group3 and authored plate inertials. Initial actual
plate vertex bounds are asserted. Empty QIPC/native plate motion matches the
backward-Euler mass/damping recurrence to the declared floating-point bounds;
native default armature must explicitly be0. Earlier adapter failures and
the initially misplaced/intersecting QIPC campaign are preserved and excluded.

Selected contact-entry diagnostic:20um clearance per plate, equal to QIPC
d_hat; translate the large plate mesh3mm along its plane. This preserves the
local flat surface while moving a triangulation diagonal away from the pole.
The original zero-offset/22um cases remain available. The observed reduction
in free lateral drift suggests sensitivity to contact features; it is not a
proved general cure. Native-equilibria-v2 uses the same20um/3mm geometry.

Native exact mu=0 is unavailable: material and collider enforce mu>=0.01.
QIPC mu=0 and mu=0.01 controls explicitly bound the normal-loading difference.
Shell uses original PR22 kernels, nominal500um wall and its own mass/contact
envelope; these differences are recorded rather than adjusted to match fields.
DPR is`pressure`; SCR is`intrinsic` +`body_source_prior`; radius6mm is declared
before comparison and is not fitted to the FEM reference.

The solid rest volume is3.2649241052e-6m3 and mass6.5298482104g; the native
rigid URDF authors exactly that mass. Original Shell has6.9462501265g nodal
mass,6.38% more, with500um nominal thickness and2mm contact softness. The
common projected solid wall is431.135–485.849um. These original-baseline
differences are disclosed, not corrected by fitting. QIPC uses stable
Neo-Hookean elasticity; native reconstruction is linear elastic and Shell
uses its thin-wall constitutive law. Same-load CPU checks below bound the
small-strain solid constitutive/numerical difference on the selected mesh.

Predeclared equilibrium tests: each actual reaction within1% of command
(1mN scale floor), plate speed<=1um/s, egg inertial resultant<=1% of command,
egg kinetic energy<=1e-10J, and endpoint nodal equation residual<=1e-3 of
contact norm. A stable final50ms window is checked separately. Native uses
a conservative rotational-plus-translational kinetic upper bound; its full
elastic field is evaluated only at endpoints. Intermediate unobserved elastic
residuals are not claimed to have been measured.

## Newton and PCG tolerances are different criteria

The primary short pilot uses Newton velocity tolerance1e-8 and PCG
`tol_rate=1e-4`. In the pinned QIPC linear-PCG source,
`pcg_compute_conv_threshold_kernel` sets the threshold to
`tol_rate * abs(r0.T @ M_inv @ r0)`; termination compares the new preconditioned
residual quadratic form to that threshold. Thus1e-6 is **not** a relative
Euclidean residual norm of1e-6. With an SPD fixed preconditioner it corresponds
to a preconditioned norm ratio1e-3. Newton's outer criterion uses a maximum
displacement correction scaled by`velocity_tol * dt`, not a direct force
residual criterion. Independent force/elastic-equation closure remains necessary.

| Off-diagonal0.1N run | Newton velocity tol | PCG tol_rate | Endpoint equation relative residual | Endpoint result |
| --- | ---: | ---: | ---: | --- |
| Original | 1e-8 | 1e-4 | 1.949e-6 | Pass |
| Repeat | 1e-8 | 1e-4 | 1.482e-5 | Pass |
| Newton tighter | 1e-9 | 1e-4 | 2.863e-3 | Reject equation closure |
| PCG tighter | 1e-8 | 1e-6 | 3.281e-3 | Reject equation closure |
| PCG looser | 1e-8 | 1e-2 | 3.539e-3 | Reject equation closure |
| dt/2 | 1e-8 | 1e-4 | 9.355e-6 | Final50ms window and endpoint pass |
| mu=0.01 | 1e-8 | 1e-4 | 1.876e-6 | Final50ms window and endpoint pass |

The tighter runs repeatedly reach the4096 PCG cap, including their last step,
yet return tiny Newton corrections. Their fields differ from the original by
about0.44%/0.47% in the common tangent metric and fail the stricter equation
gate. Neither their smaller input tolerances nor finite completion qualifies
them. The looser-inner run is faster but also fails equation closure and is
not an accepted speedup. Do not replace the1e-4 baseline by1e-6 without measured
benefit. Total early PCG cap counts are retained; the original's final50ms
window has no observed cap, while the repeat has three sampled cap events.
Neither observation establishes that every unrecorded inner solve converged.
The stored strict solver-exit flag still rejects any run with an early PCG
cap; it has not been relaxed. The separately named mechanical-window and
endpoint-equation gates describe the narrower passing observations. Tighter
tolerances are not inherently worse: these measured runs are cap-sensitive,
and the same nominal configuration can enter different iteration regimes.
Overall field accuracy also depends on discretization, contact and timestep;
neither Newton nor PCG tolerance alone certifies it.

## Discrete elasticity passes; contact-field refinement does not

The original off-diagonal endpoint has actual reactions0.09999999839 and
0.09999999813N, kinetic energy1.15e-16J and no detected edge/triangle
intersection. All final50ms mechanical and equation checks pass. Independent
CPU linear Lamé FEM on the **same measured nodal traction and mesh** differs
by2.42e-6 in complete volume-weighted tensor norm. Its equation residual is
1.71e-12N. The six-mode rigid gauge's balancing correction is reported
(4.83e-10N L2); it supplies no hidden unreported support.

The surface2/wall4 endpoint also reaches mechanical/equation equilibrium,
and its same-traction CPU difference is5.96e-6. However, both surface grids
carry each plate's entire normal load on **one polar FEM vertex**; measured
nodal footprint radius is zero. This is not a resolved pressure footprint.

| Spatial diagnostic at0.1N | Raw von Mises peak | Common tangent change vs surface1/wall4 | Status |
| --- | ---: | ---: | --- |
| surface1/wall4 | 25.574kPa | Baseline | Discrete endpoint admitted; continuum field unresolved |
| surface2/wall4 | 53.012kPa | 61.27% | Discrete endpoint admitted; strong spatial sensitivity |
| surface1/wall2,0.3s | 25.238kPa | 11.52% | Still moving; not a thickness-convergence result |
| surface2/wall8 | 52.839kPa | Not used as an admitted contrast | Equation residual4.65%; not admitted refinement |

The surface1->2 membrane change is59.25% and bending change146.26% on the
shared material columns. Raw peaks are neither clipped nor averaged. These
observations diagnose unresolved polar contact fields; they do not prove an
asymptotic continuum singularity or justify excluding polar regions. No
global mesh sweep or fitted reconstruction radius is used to conceal this.
Extending the wall2 hold to1s did not recover equilibrium: both contacts were
lost and kinetic energy remained about1.29e-6J. It is preserved as a failed
settling control, not included in a thickness-convergence estimate.

Halving dt changes the admitted common tangent field by9.12e-6 relative;
changing QIPC mu from0 to0.01 changes it by4.58e-7. Halving d_hat from20um
to10um while retaining the20um initial clearance gives an endpoint equation
residual1.88e-6 and raw peak25.574kPa. It changes plate travel from about
0.181um to20.129um, exposing the contact-band contribution to gap response.
Its final50ms mechanical/equation window passes, with one sampled PCG cap;
the common tangent field changes6.56e-7 relative, or0.00262Pa RMS.
These bounded controls do not resolve the one-node pressure support.

## DPR equilibria and SCR rejection

Native-equilibria-v1 passes all three final50ms mechanical windows and endpoint
elastic residual checks. The geometry-aligned v2 repeats its fields; v2
endpoints are below. Plate travel includes contact regularization and clearance;
it must not be read as pure elastic egg indentation.

| Command | Actual upper reaction | Upper plate travel | DPR raw VM peak | Elastic equation residual |
| ---: | ---: | ---: | ---: | ---: |
| 0.1N | 0.09999904N | 376.668um | 22.679kPa | 3.45e-11 |
| 0.3N | 0.299999999N | 872.369um | 68.038kPa | 3.31e-11 |
| 1N | 1.000000000N | 1703.447um | 226.793kPa | 3.17e-11 |

The original0.1N DPR/common-QIPC tangent contrast is42.25%, or1687Pa RMS.
The two actual reactions agree within1%, but QIPC's spatial change is61.27%.
Consequently42.25% is a **diagnostic discrete-field contrast**, not a qualified
physical error. The field adapter also exports inner/outer tensors, membrane,
bending, full3D solid-only contrasts and raw peak locations in compatible
material frames. Missing SCR fields remain missing.

The continuous three-load QIPC run passes all three final50ms mechanical
windows and endpoint equation checks, as does geometry-aligned DPR v2.
The following are **discrete diagnostic contrasts**, not physical errors:

| Load | QIPC raw VM | Tangent relative / absolute RMS | Membrane relative / absolute RMS | Bending relative / absolute RMS |
| ---: | ---: | ---: | ---: | ---: |
| 0.1N | 25.574kPa | 42.254% /1687.31Pa | 41.212% /0.759074N/m | 76.551% /1.07497e-5N |
| 0.3N | 76.723kPa | 42.254% /5061.98Pa | 41.212% /2.27724N/m | 76.552% /3.22495e-5N |
| 1N | 255.747kPa | 42.254% /16873.25Pa | 41.211% /7.59080N/m | 76.551% /1.07498e-4N |

Inner/outer tangent contrasts are42.422%/41.422% at all three levels;
their0.1N absolute RMS values are1621.22/1725.99Pa and scale approximately
with force. The separately reported solid-only full3D contrast is57.288%
(2705.27Pa RMS at0.1N). Shell's unavailable transverse components are not
scored as zero. Complete tensors, absolute metrics and all per-level values
are in the field receipts.

QIPC raw VM maxima occur at the reported P1 tetrahedron centroid
(1.1360,-1.8839,28.2915)mm; native maxima occur at a native corner sample
(0,0,30)mm. Maximum-principal locations and values are retained separately.
These sampling definitions differ and are not presented as a matched-point
peak-location error. The QIPC plate travels are0.1810/0.3825/1.0014um at
0.1/0.3/1N, with measured upper reactions0.100000031/0.300000021/0.999999326N.

SCR rejects its first settled endpoint at step2100 in both geometry variants.
The full v2 rejected state is saved without relaxing engine acceptance.
The local priors are approximately+0.09999904156/-0.09999904121N. Body-source
partitioning retains these priors while replacing auxiliary constraints with
their aggregate wrench. Its net force is only3.5474e-10N, although its retained
group force scale is0.199998N. In`cooperative_scatter.py`, final acceptance
uses1e-8 times the **nonzero net force**, falling back to the larger scale only
when the net force is exactly zero. The resulting moment budget is2.1285e-20Nm;
measured cancellation error is3.3982e-20Nm. Force and local friction checks
pass, but moment acceptance fails with status40 and propagates to both loads.
This is a diagnosed cancellation-sensitive admission failure. It is not
evidence of a large physical friction/wrench violation. The original baseline
remains rejected; no successful field is fabricated and no engine fix is
included in this comparison.

## Original Shell fails the coupled equilibrium test

The zero-load control remains near its initial3.3e-6Pa stress and negligible
kinetic energy. At0.1N command, dt1ms,0.3s, original PCG50/tol1e-4 yields
only about0.02746N actual reaction and113.22MPa raw VM stress. Tightening to
PCG300/tol1e-8 produces essentially the same response with no PCG caps and
true linear residual near1e-8. Half dt changes reaction to0.07146N and peak
to75.20MPa; this is not settled equal-force convergence.

Read-only diagnostics isolate the splitting: the unconstrained implicit
elastic step uses A=M+dt²K, then contact projects velocity by delta_v and
reports impulse M delta_v. The frozen linearized coupled equation instead
requires A delta_v; the measured missing term is A delta_v-M delta_v.

| dt | Contact impulse L2 | Frozen elastic/contact defect L2 | Defect/impulse |
| ---: | ---: | ---: | ---: |
| 1ms | 3.880e-5Ns | 1.441Ns | 37154 |
| 0.5ms | 5.052e-5Ns | 0.4813Ns | 9526 |

This is a frozen linearized balance diagnostic, not an exact nonlinear
reaction reconstruction. It explains why an accurate unconstrained PCG solve
does not certify coupled contact equilibrium. Shell fields at these times
remain diagnostic and cannot enter the common-force accuracy ranking.
The diagnostic reads the original solve's cached membrane F/stress/stiffness
and hinge gradients/stiffness into owned output buffers. Integration does not
refresh those caches, and plasticity/fracture are disabled here; the post-step
field observer also leaves them unchanged. Thus the measured operator is the
frozen solve operator, not a stiffness reevaluated at the displaced geometry.

## Warmed native sparse-field cost

The plate workload continues at a held0.1N after mechanical settling and
field compilation. Each1000-step repeat requests and transfers the complete
stress/VM fields every100steps, retaining actual engine admission and finite
checks. No frame files are written inside timing. All environments' end-window
reactions and plate speeds are checked, including the pure-rigid control;
no-contact runs cannot enter this table. Three warmed repeats on RTX PRO6000
Blackwell give the following median aggregate env-step/s (raw repeats remain
in the packet). Duplicated environments test batching cost, not random-scene
robustness or a larger range of contact conditions.

| Batch | Pure rigid | DPR with sparse complete-field transfers |
| ---: | ---: | ---: |
| 1 | 725 | 697 |
| 8 | 5470 | 5359 |
| 64 | 43511 | 41804 |

Separate intrusive B64 build/warm/profile jobs sample process peaks2130MiB
for rigid and2268MiB for DPR. Their200-step CPU/CUDA traces retain every kernel
name and duration; the largest GPU kernel is collision, not the sparse stress
observation. Profile/kernel sums and memory-instrumented runs are not primary
wall-clock rates. SCR has no admitted baseline field here; original Shell has
no admitted coupled equilibrium. These native rates therefore support an
engineering cost statement, not an accuracy-qualified cross-method speedup.

The independent primary QIPC cost run measures4.1750/4.1287/4.1561 env-step/s
over three300-step repeats after100 warm steps, with a full-field observation
and transfer every100steps. However, its observed equation checks fail and
288/295/286 steps reach the PCG cap. The median4.1561 is a **rejected raw cost**,
not an accepted reference rate. A separate intrusive profile run reaches a
different, passing endpoint/low-cap regime; it is not substituted for the
failed three-repeat primary run. Timing is sensitive to the solver state.
QIPC process memory peaks at5282MiB in that separate build/warm/profile run.

| Separate intrusive profile | Window | GPU kernel duration sums by group |
| --- | ---: | --- |
| Native DPR, B64 | 200steps | Collision/contact118.302ms; constraint solve68.199ms; mechanics34.759ms; fused reconstruction/field1.980ms; unclassified36.322ms |
| Pure rigid, B64 | 200steps | Collision/contact113.133ms; constraint solve68.124ms; mechanics35.122ms; unclassified36.353ms |
| QIPC, B1 | 2steps | Contact/barrier/CCD3.720ms; elastic assembly2.067ms; linear solve/preconditioner0.658ms; guide/prediction0.849ms; unclassified1.320ms |

All use RTX PRO6000 Blackwell Server GPUs and the same container tag; device
UUIDs and environments remain in receipts. Kernel-name groups retain all raw
names and unclassified work. Sums can overlap and are not exclusive wall-time
stages; the different profile windows and solver regimes are explicit. Native
field requests fuse reconstruction and field work, so no fictitious exclusive
reconstruction/observation split is reported. Sparse host-transfer costs are
recorded separately in cost receipts. No valid error-versus-cost frontier or
accuracy-qualified speedup exists while the physical reference is unresolved.

## Stability and matched-history friction control

DPR completes the continuous0.1/0.3/1/0.3/0.1/0N load/unload sequence with
all six mechanical windows and endpoint equation checks passing. Returning
to0.3N changes the identical corner tensor array by6.36e-9 relative and
plate coordinate by4.02e-12m; returning to0.1N gives1.95e-5 and6.86e-9m.
These are repeatability diagnostics, not volume-weighted physical errors.
At zero command the actual upper force is2.19uN and raw peak0.496Pa, within
the predeclared10uN force bound derived from the1mN scale floor. Neither is
reported as exactly zero.

Changing the declared DPR radius from6mm to3/9mm gives raw peaks
31.916/17.188kPa instead of22.679kPa at the same actual0.1N load. Common
tangent contrasts to the unresolved reference are45.04%/42.49%, compared
with42.25% at6mm. No radius is fitted or selected from those errors. The
peak's radius dependence limits physical failure-threshold interpretation.

The bounded friction control uses common mu=0.3,0.1N normal load, a0.1s
normal ramp, and10um lateral plate motion from2.1 to2.3s, followed by a hold
to2.5s. The egg remains free. All backends use the same declared history,
50g upper plate,1g noncontact serial carriage, lateral kp=1e5N/m and
150Ns/m damping. QIPC retains its original friction regularization
(eps_v=1e-4m/s), so equal coefficients do not imply identical friction laws.
Its serial finite-stiffness guide uses1e10, explicitly separated from the
normal baseline's1e8, after an unchanged-bound empty-motion check passed.
The earlier1e8 empty test missed its1e-10m/s velocity-error bound; its
failure is retained. Native and original Shell empty tests also pass.

| Method at2.5s | Actual upper normal force | Actual lateral travel | Upper tangential force norm | Raw VM peak | Admission |
| --- | ---: | ---: | ---: | ---: | --- |
| QIPC | 0.100000003N | 10.00000055um | 4.97e-8N | 25.575kPa | Mechanical/equation final50ms window passes; PCG caps persist |
| DPR | 0.099999831N | 10.00016776um | 1.68e-5N | 22.679kPa | Mechanical window and endpoint equation pass |
| SCR | Near0.1N before field request | Same prescribed target | Failure state retained | Missing | Traction fit rejected at step2500 |
| Original Shell | 0.0388656N | 9.95246444um | 0.00835N | 155.014MPa | Not mechanically settled |

The small final tangential load reflects relaxation of a free egg and is not
a sustained-shear validation. During the imposed motion, QIPC at2.2s has
about-0.00380N upper x reaction and equation residual0.0445; the trajectory
contains522 PCG-cap steps and larger residuals during sliding. These events
are retained, so a passing final endpoint cannot qualify the entire motion.
All six final-window observations still contain an inner PCG cap, despite
passing the measured equation gate. The strict solver-exit flag remains false.
The settled DPR/QIPC tangent contrast is42.255% (1687.33Pa RMS), membrane
41.212%, and bending76.552%; these remain diagnostic discrete comparisons.
Shell still moves at1.135mm/s with kinetic energy2.45e-5J at2.5s and has a
frozen split defect/impulse ratio about48019. A longer common history did
not establish its coupled equilibrium.

SCR's shear failure is status33, unresolved vector traction fit, with
fit status3 and normalized force/moment fit errors3.49e-7/3.60e-8. The
retained opposing priors have group scale0.199999665N and net force only
6.06e-11N. This is an earlier fit-stage rejection, not the normal case's
final status40 moment test. Zero reconstructed loads in this failure dump
are uncommitted failure buffers, not a recovered zero-stress solution. The
diagnosis does not claim that changing the final moment budget alone would
repair this different rejection.

An additional oriented-egg case was not run: the optional control would not
resolve the already localized normal-reference support, native admission or
Shell coupling limitations. General off-axis accuracy remains unqualified;
existing arbitrary-contact engine requirements have not been narrowed.

## Practical interpretation

- DPR provides an admitted, inexpensive conditional elastic reconstruction
  for this rigid-contact workload, with repeatable load/unload behavior.
  Its selected pressure radius and contact mechanics remain modeling inputs;
  the reported stress is not a validated absolute physical failure threshold.
- SCR is unsupported for the tested nearly cancelling balanced plate loads.
  The normal acceptance-scale defect and the separate shear fit rejection
  are reproducible. A future engine repair needs both cases as regressions;
  this comparison preserves the original implementation and its failures.
- Original PR22 Shell does not supply a settled common-force reference at
  the tested practical timesteps. Tightening its inner PCG does not fix the
  measured contact/implicit-elastic splitting defect.
- QIPC agrees with independent same-load discrete elasticity but does not
  resolve the pressure footprint or spatial stress field. Contact-localized
  resolution/regularization qualification is needed before physical peak
  errors, three-method rankings or accuracy-qualified speedups are supported.

## Execution and evidence

All scientific execution uses rtx-mid Slurm and GENESIS_IMAGE_VER=1_26 gs-srun.
New runs, logs and task JSON live under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates/`.
The old user-authorized home fallback is retained without new writes.
Every submission captures backend revision/patch/file identities, analysis
source, exact argv, runtime/GPU identity and private cache configuration.
The [closeout packet](../evidence/20261018-parallel-plates/closeout-v1/README.md)
contains final receipts, both original SCR failure states and reviewed
color/grayscale figures with editable SVG/PDF exports. The immutable
[earlier checkpoint](../evidence/20261018-parallel-plates/checkpoint-v1/README.md)
retains pilot and tolerance evidence. Captions retain differing actual loads,
discrete-force rather than pressure units, and missing-field status.

Useful entrypoints are`plate_reference`, `plate_native`, `plate_shell`,
`plate_audit`, `plate_failure`, `plate_fields` and `plate_figures` under
`research.rigid_stress.qipc_comparison`. Submit with the established
`research.rigid_stress.shell_comparison.submit` helper, explicit backend and
environment paths, a fresh data output, `--tasks` and `--node-local-cache`.
The [running guide](RUNNING_PARALLEL_PLATES.md) gives concrete commands and
backend/environment mappings. Saved task JSON lists exact runnable arguments;
never hot-edit a captured worker or reuse an existing output directory.
The [requirement audit](PARALLEL_PLATE_AUDIT.md) records the supported closure
and remaining unsupported scientific quantities. Existing Franka throughput
remains separate engineering evidence; no full-Franka qualification sweep
was restarted.
