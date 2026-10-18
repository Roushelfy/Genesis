# QIPC reference comparison execution status

The [full goal](QIPC_REFERENCE_COMPARISON_GOAL.md) is active. Previous shell and
same-mesh FEM evidence does not qualify a common three-method contact reference.

## Source inspection and initial build

2026-10-18 UTC: task branch fast-forwarded to goal commit918ccf43f. Remote
`Roushelfy/cuda-graph-qipc:main` is still
`fe252bd30acc6c717507b74add31512812f48761`. A detached original checkout is at
`../qipc-stress-reference-original`. Four required submodules are pinned to
their authored commits; no tracked QIPC source is patched.

The revolute virtual-work torque fix `ee1673f74` is already in main. The
separate theta-unwrap commit `c8b27b451` and research integration641459b88/later
FP64/PCG-quality branches were inspected as selection/qualification context;
their binaries are not silently loaded into this main checkout. Existing
research quality is task-specific, with recorded cap/initial-gap failures; it
does not qualify the new hollow-egg scene.

The actual SNH energy includes `-mu/2 log(Ic+1)` and a shifted volume term;
its stored parameters are `mu_snh=4mu/3`, `lambda_snh=lambda+5mu/6`. Stress
recovery must differentiate that exact energy, rather than a textbook
neo-Hookean formula or projected Hessian. StVK-Hencky uses another energy.

The first native build is prepared with runtime checks, full FP64-capable
source, Release, sm120, no viewer/stub generation, and an unchanged existing
Python3.12/Torch2.11 environment. CMake, objects, runtime package, CCCL, caches,
logs and exact commands live under the new data run. The source checkout's
generated CCCL path is a compatibility link into that data root. Dependency
source caches are reused read-only; no environment is upgraded or overwritten.
The reusable submitter now captures QIPC/CMake/vendored sources and submodule
identities, plus the new comparison package. Its original Genesis defaults
remain available.

Actual build726135 completed all453compile/link steps (Release solver image
62422704B, SHA256`cebd257411c97a01e5d0c78ca8d5fb6271955c9b3b5c20ce3e53dab95e175cf1`),
then failed its import because this main's top-level package imports the viewer
even for headless use. The failure/log/runtime remain immutable. Build726150
adds the viewer through the same completed incremental build cache, with fresh
logs/runtime output and no tracked QIPC source patch. No solver rebuild is
inferred without inspecting the actual build log.

Controls726147 completed. A host C++ probe compiled against the **unmodified
generated SNH energy/gradient header** agrees with the CUDA Cauchy/Piola observer
to7.22e-16 at E=1. Rotation covariance and small-strain E/nu controls pass.
SNH and Hencky finite-strain differences remain distinct. Fixed-domain
refinement keeps the level1 mass at6.5298482104g while lumped inertia error
decreases9.871% to0.6175%; the separately rounder level2 geometry has a different
volume/mass and is not mislabeled numerical refinement. These checks do not
qualify contact dynamics or a complete grasp.

Public `RigidStressOptions.dpr(...)` / `.scr(...)` constructors are implemented
as complete option presets with the old explicit identifiers. All other defaults
and numerical kernels are unchanged. Full-field/max shared-operator tests now
exercise DPR and verify complete SCR serialized equivalence. CPU/CUDA execution
of that focused test passed (job726151; one test per backend).

Build726150 completed/imported successfully. Its eight compile/link steps build
the viewer; the solver SHA above is unchanged. The first failed runtime and
import log remain preserved. Successful runtime extensions are explicitly
linked into the isolated original checkout.

## First actual dynamics and numerical controls

Original-main hollow-egg/plane workflow726153 completed20 steps of1ms,
including actual barrier/friction contact gradients, unprojected elastic
Cauchy fields and momentum/energy diagnostics. Its final local force residual
was1.98% of nodal contact-force norm, with15 linear-cap steps. Successful
process completion does not establish physical-reference qualification.

Bounded controls726158 retain the same3840-tet input and unmodified main.
With diagnostics enabled, default float MAS reaches4096 PCG iterations on15/20
steps; its final recursive residual increases in the last Newton system.
Double MAS has0/20 such cap steps and final local force residual4.45e-6
relative to contact. Standard versus partitioned PCG under double MAS gives
13294.9801 versus13294.9806Pa final raw VM peak. Actual sparse systems and
residual histories are retained. This is a precision diagnosis, not spatial/
contact/friction convergence or grasp qualification.

The same job ran independent CPU P1 isotropic-stiffness/lumped-mass implicit
Euler controls against actual QIPC FEM dynamics, for SNH/Hencky and two load
scales. Final volume-weighted tensor discrepancies range7.0e-7 to4.23e-4.
The tight1e-8m/s SNH/full-load point hits its256-Newton cap after line-search
stagnation; retain that failure and inspect stopping/energy precision before
admission. Constitutive and dynamic differences are recorded separately.

Source audit finds original main does not import MJCF damping/armature; drive
kv also lies outside its proportional-force cap. These are material mismatches
for the original Franka task. Integration641459b8826bb061e1bba01dfe9eba39793c3855
contains authored passive terms and hard minimal-coordinate joints, including
commit a9206cd11. Its previous native build has an exact matching-source receipt
and SHA8763018e8ca49e2b7c156cc5cdcf42318b03dfd0363618c3b824daaca39e1213.
An isolated detached candidate explicitly reuses that image. It is a different
candidate from fe252bd30 main; new egg and robot controls must qualify it before
use. Old task quality is not transferred to this egg.

No three-way stress-error ranking, complete-grasp qualification or new rate is
claimed. Exact receipts live under `runs/20261018-qipc-reference/` on data,
including `linear-controls-v1` and `meta/integration-native-reuse-v1`.

## Franka controller and first coupled attempt

No-contact controls now use the original XML collision shapes/inertials and all
nine authored armatures0.1 and passive dampers1. The generated XML only selects
the accepted initial bank keyframe and resolves asset paths. Asset bytes are
hashed. Native's capped start-frame PD and implicit damping Jacobian are mapped
onto QIPC's original force/kv API; numerical kernels remain unchanged.

The XML also couples the two fingers. The initial uncoupled candidate differs
by about1mm within20ms and is retained as an explicit failed mapping. A source-
derived increment mapping of native equality impedance/inverse weight uses
QIPC's public linear coupling plus paired predictor force. It does not fit
held-out stress, contact forces or reconstruction radii. The actual first-step
joint positions match native; after20ms the largest coordinate difference is
about1.2e-8 (job726168 versus726165). The state-dependent equality-inertia
coefficient differs from its authored dmin approximation by at most0.136% in
that control. Preserve and budget this approximation in later grasps.

Gravity100ms controls at1ms/0.5ms also completed on both engines
(726171/726173). Their complete trajectories and effective actuator forces are
retained; derived trajectory/error admission is still required. Neither these
controls nor the small-strain controls qualify coupled contact.

First combined egg/Franka attempt726170 fails before its first accepted step:
`forest preconditioner: non-SPD edge pivot1 = -4.68478931963709545e+09`.
The native CUDA failure prevents subsequent device-state recovery; the input,
initial rest packet, generated robot asset, empty step journal and original
traceback remain intact. No complete or partial rate is assigned. A separate
read-only initial contact audit and native collision-filter inventory are
submitted (726175/726176), to distinguish geometry/filter mapping from an
actual reference solver defect. Do not suppress the assertion or rank this
candidate as reference-qualified.

### Initial finger geometry diagnosis

Initial read-only contact audits726175/726177 find400 active PT and1000 EE
pairs between the supposedly open fingers, with near-zero distances. The rigid
vertex publication agrees with world poses to1.25e-16m; this is not a publication
or observer-frame error. The original reader obtains inherited primitive type,
but passes a literal0.05 size and does not expand inherited geom position.
The five authored small fingertip boxes are therefore imported incorrectly.

The generated input now explicitly materializes inherited geom attributes.
QIPC source/native image and original authored XML/meshes stay unchanged. Actual
MuJoCo3.15 control726179 compiles original and expanded inputs: all compared
geom type/size/local pose, mesh arrays, link inertials, joint passive values,
equality and actuator arrays are bitwise equal. World geom poses at common
initial q are also exactly equal. The generated input changes only keyframe
selection, meshdir resolution and explicit representation of existing defaults.
New contact audit726180 and corrected coupled pilot follow; neither earlier
failed input nor zero-contact control evidence is relabeled.

## Common observables and actual three-way histories

Canonical-input contact audit726180 has no active initial finger pairs, with
rigid publication error below1.25e-16m. Corrected200ms QIPC pilots complete all
200 steps with no caps/nonfinite states. Tightening Newton velocity tolerance
from1e-5 to1e-7m/s is essential: the loose point permits about46% maximum local
force residual during impact, despite successful Newton exit. With1e-7,
linear1e-7 versus1e-5 agrees to1.05e-6 in reactions, and local residuals stay
below1.07e-4 relative to nodal contact norm (726188/726190). Raw small unloaded
peaks still show floating-point sensitivity; retain absolute errors/floors.

The new common metric uses1280 fixed-geometry material face-centroid columns,
40960 Gauss points on the union of native/QIPC tet intersections. It compares
corotated tangent Cauchy tensors, membrane and bending resultants on one
explicit weighted domain. Analytic constant-stress/linear-bending integrals,
affine interpolation and rigid-rotation checks pass in726208. The first sampler
attempt726207 incorrectly assumed native tet winding; the corrected locator
preserves original corner order and accepts either nondegenerate orientation.
This metric is not an exact full-volume integral or a global-peak search.

All three200ms candidate output pilots pass (726192/726194). New full900-control-
tick case0 DPR/SCR and stock-shell runs complete (726201/726204), with additional
cases2/7 and a finer diagnosed shell still running at this checkpoint. Fields,
forces/impulses, frames and full-history journals are retained. These are new
intrusive error/stability datasets, not primary throughput measurements.
Shell coupling friction is now explicit per single-case bank, through a helper
option whose old default0.6 is preserved. The nominal500um shell mass is6.94625g
versus6.52985g solid mass: retain the6.38% area-thickness approximation mismatch;
do not silently call it matched mass.

The first complete QIPC candidate726195 stops after1086 accepted steps at
1.086s, with a256-Newton KKT proxy stationarity/feasibility exception on the next
step. Prior accepted local force residuals and all raw fields remain available.
The whole attempted history is unqualified. Wider100um contact band and Hencky
controls726198 also stop near the same first-grasp stage; the half-step case
continues. A standard-PCG/tighter-linear candidate726219 tests whether the
partitioned solve is responsible. No increased iteration count, different
energy or changed band alone is claimed to solve this failure.

Actual three-way diagnostic comparisons726214 now cover108 common times through
1.08s. At that final available time, compatible tangent errors are about0.499
for DPR/SCR and2.81e4 for stock shell. This is **not an accuracy ranking**:
reference spatial/dt/contact uncertainty and the complete grasp remain
unqualified, and the models' feedback/mass/contact differences are included.
Source/frame/sampling identities and exact absolute errors are recorded. Full
reference qualification, replay separation, both protocols and new primary
speed evidence are still required.

## Prescribed motion, independent replay and current numerical limits

All six capped DPR/SCR case0/2/7 histories726201 and both stock/fine-diagnostic
shell histories726204 now complete900 ticks. Completion/finite output is not
physical qualification. The half-step QIPC control726198 also fails near1.088s;
standard PCG726219 fails after881 accepted steps, and the smoother-friction
control726224 fails after798. Friction-off726224 is a separate causal diagnostic,
never a replacement for the original friction task.

The frozen shared motion726222 uses the first complete6s of case0's rigid
trajectory, joint-linear interpolation and exact MuJoCo FK at0.25ms. It is fixed
without QIPC stress or footprint fitting. Native/Shell use finite implicit
servos with caps overridden; QIPC uses its original per-link rigid pose DBC and
a free3D hollow solid. Native physics-step time is an optional example argument
whose default remains10ms. A new frozen native source isad6b3e2da; the previous
validation checkout9bd1086ca remains unchanged.

The1.2s tracking audit726282 measures maximum authored-link position errors
7.11e-12m for damping1e10 and7.11e-10m for1e8, with the same bound in the shell
pilot. This is measured enforcement, not an assertion that actuator gains equal
the capped protocol. All three6s native prescribed histories726276 and stock,
coarse-diagnostic and fine-diagnostic shell histories726279 now complete. Raw
VM/principal peak states are retained in the new workers. Complete grasp force,
field and gain-sensitivity audits remain required.

Original-main prescribed QIPC726263 reaches1.087s, then hits256 Newton iterations.
Its preceding726227 attempt was an observer-only API error: original main lacks
the integration branch's body-offset binding. The corrected observer verifies
original main's rigid ownership prefix against its public owner array and the
absence of ABD/cluster topology changes. DBC/tolerance726264 and Hencky726271
controls preserve their distinct failed/partial histories. Neither the finite
pose error nor a small residual relative to hundreds of newtons of squeeze
establishes a small error in the much smaller net egg wrench. Net balance and
motion outcomes must be checked separately.

Independent actual-load replay726290 now passes an affine elastic control and
replays measured spatial contact minus actual acceleration/gravity loads into
independent P1 and P2 elasticity. P2 virtual-work transfer preserves force and
undeformed moment, and records the deformed-coordinate moment difference and
the small rigid-balance correction. A complete material locator resolves the
first scikit-fem boundary-point lookup failure726282: maximum selected-tet
projection distance is1.49e-16m, with no coordinate clipping in the transfer.
At1.16s of the failed Hencky source, independent P1/P2 common tangent errors are
0.01449/0.74389. This diagnoses constitutive/discretization response under the
actual resolved loads; it is not a qualified contact ranking or point-wrench
reconstruction comparison.

One bounded numerical diagnostic is isolated from the unchanged QIPC baseline:
`line-search-roundoff-diagnostic.patch` adds16 double epsilons times absolute
current energy to the non-KKT line-search comparison. Original physical energy,
gradient, Hessian, CCD and all other native objects are unchanged. Job726291
recompiles that one translation unit and performs fresh device/host links into
owned files, hashing reused objects and verifying the original binary unchanged.
The variant library SHA256 is
`d4b90d5cbcbee44fb2c92b9406d5ca15f70e4dcda82469c5fb9865bc37526211`.
Its SNH candidate726298 uses velocity1e-9/linear1e-8 and advances beyond the
old first-grasp failure with measured force residuals; the Hencky candidate
still exhausts Newton/PCG at1.103s. This variant is not yet qualified. Bounded
complete-history mesh/thickness and dt/stopping/friction/band studies are
registered separately; their failures and unresolved peaks remain evidence.

New throughput work distinguishes scopes explicitly. The historically named
`primary-*-cost-v1` campaigns enable process-memory polling and are intrusive
diagnostic cost/memory runs. Their FP32 native DPR warmup rejects the contact
wrench preservation budget atB1/64/4096; retain these failures without relaxing
the check. The separate `full-field-*-timing-v1` campaigns useFP64, no memory
polling, field export or solver instrumentation inside the timer, with900 warmup
and three900-tick repeats. Native rigid/DPR/tuned-DPR/SCR full output and separate
max output, substeps1/10 andB1/64/4096 are registered; shell default/fine-tuned
full output usesB1/32/256. The cohort repeats the same case0 bank and matching
shell friction, so it must be labeled a duplicate-case scaling cohort rather
than a varied RL bank. These are new measurements; no old rates are substituted.
The submitter can retain individual task exit codes while continuing independent
configurations, and still returns a failing campaign status if any task fails.

See [reproduction instructions](RUNNING_QIPC_REFERENCE.md). Full reference
qualification, actual point-wrench replay, lifecycle studies, profiles and the
final shared-scale figures/report remain unfinished.

## Complete sensitivity history and resolved-versus-point load layers

Original-main Hencky/DBC1e4 candidate726271 completes all6000 prescribed steps.
The complete-history audit726335 measures maximum COM tracking7.50e-9m and
rotation1.09e-6rad; the egg escapes rather than following the higher-strength
held branch. This is a real full-history sensitivity result, not a qualified
unique grasp prediction. Actual three-way comparisons now cover600 common
observations in726335, including one-sided inner/outer surface tangent stresses,
bulk-column tangent fields and membrane/bending resultants. The surface metric
checks2560 exact boundary positions, selecting the cell entered from within
the wall. Display samples remain distinct from raw global extrema.

Read-only original normal-barrier queries726312 use the saved actual robot/FEM
poses and original contact kernels. They do not advance or replace friction
history. Recorded total contact/inertial forces remain authoritative. A
nonnegative linear program compresses each connected active mesh-support
component while preserving its physical force/moment. Its radius comes only
from the pre-existing condition-bank schedule. Aggregated nodal normals and
provider-local support ids are explicitly derived packets, not fabricated
observations from the native collider.

Native replay726321 runs116 observation times through actual production
Quadrants kernels. Resolved-load core replay has0 invalid frames. At1.16s its
common tangent difference from independent P2 is1.74e-11 relative, while its
difference from the failed nonlinear P1 source remains0.74389. This verifies
the elastic core for that replay; it does not establish P1 contact/mesh truth.
Compressed point-only packets with measured inertia have4 DPR and6 SCR rejected
frames; preserve those source/wrench checks. Rigid-body-inertia replays are
separate ablations. The first replay attempt726316 had an observer-only static
connectivity-axis error; remaining affected tasks were cancelled with a recorded
reason, then rerun fresh. No failed attempt was relabeled.

The shared M1 build directory had advanced after the earlier641459 artifact.
The new clean40275f0a2 source and copied matching native are recorded separately.
Its only QIPC code difference from641459 is a5-line dense-BVH stack barrier
placement change. Original main already has per-child-round barriers; this
change is not evidence that original main misses contacts. Baseline726310 still
fails after1086 accepted steps. A separately pinned
`kkt-roundoff-diagnostic.patch` changes its fixed incremental-energy allowance
to16 double epsilons times absolute energy. Fresh native build726309 snapshots
project includes, reuses hashed immutable objects, recompiles only the declared
pipeline unit and performs fresh links/import checks. Candidate726314 then
completes1300 capped-control steps, passing the former first-contact failure
with0 invalid steps, but6 recorded PCG-cap steps. It remains unqualified pending
full histories and uncertainty. New9s cases0/2/7 and case0 dt/mesh/stopping controls
are submitted separately. Their workers also record passive, equality and
complete rigid-plus-FEM contact work, using public per-vertex link owners rather
than KKT proxy ids; these are observation changes, not a controller alteration.

Geometry-only shell wall inputs726335 match6.529848g rest mass by using each
wall-prism volume divided by its original midpoint area, without changing
density or fitting stiffness/footprints. Vertex-lumped inertia differences are
9.87%,2.46%,0.602% at surface0/1/2. A separately labeled research input adapter
binds face mass before build and existing thickness state before stepping/reset
capture. It changes no PR22 kernel. Nominal500um stock remains intact. Actual
per-face thickness and vertex masses enter the common observer for this variant;
director/contact representation differences remain. Its physical pilots must
pass before full geometry-consistent histories are admitted.

FP64 primary full/max timing now completesB1/64 native configurations; larger
native/shell batches continue. MatchingFP64 memory/capacity and moving loaded
profiles are separate campaigns726324/726327. Memory polling and profiler traces
are not used as primary throughput samples. Additional surface refinement is
registered because the existing prescribed force/field histories still show
material spatial uncertainty. All completion gates below remain open as stated.

## Complete six-second controls and new audits

The16-epsilon original-main SNH diagnostic726298 completes6000/6000 physical
steps at1ms, nonlinear velocity tolerance1e-9 and linear tolerance1e-8. Original
unmodified-main Hencky variants726273 atDBC1e6/1e7 and looser nonlinear stopping
also complete6s. Neither completion closes the reference gate. The same-surface
wall2/wall4 comparison spans600 observations and reaches232% common-column
discrepancy; the1e-8/1e-9 nonlinear tolerance contrast reaches0.921%. The dynamic
Hencky contrast also changes stopping and is not a pure constitutive ablation.
`reference-pairs-v1`726380 retains exact union-column integrals, boundary fields,
net wrench differences and identical-deformation constitutive diagnostics.
Independent surface2/surface3 wall2 runs726365 continue alongside the existing
surface2 wall4/wall8, dt, contact-band and friction-regularization histories.

All600 prescribed observations now have actual three-way common tensor,
membrane/bending and inner/outer comparisons in`strict-complete-fields-v1`
726378. The maps keep raw global peaks separate. Full spatial load replay726377,
normal query/compressed packets726376 and production Quadrants replay726392
cover the complete6s history. The core has0 rejected frames, while the derived
DPR/SCR point-only interfaces reject15/397 of600 frames; these are interface
ablations, not native-collider rollout failure rates. The116-frame unit control
has normal-force discrepancy6.93e-11 relative; physical force/moment compression
defects are3.29e-12N/6.52e-15Nm. Actual robot-link reactions are now retained from
recorded global gradients, with physical points/moments rather than inferred
counterbody assignment. Replays continue to label their source unqualified.

New near-slip lifecycle726370 uses tick427 and61-tick continuation, three public
restores, partial restores with the selected controller clock aligned, B1/B2
and duplicate/fresh processes. DPR/SCR continuation position changes are about
2.08e-10m and stress changes0.00434Pa; immediate native full-field invalidation
is intentional and continuation restores finite output. Stock shell shows
approximately0.68mm restore-continuation position and574MPa tensor differences;
its duplicated inputs differ by13.2cm. Geometry-only wall inputs remain separate,
including their own lifecycle record. Capped QIPC public dump/recover and full
reset controls726402 use actual native checkpoints plus explicit Python servo
state, with no trajectory-snapshot reconstruction of friction history.

Mass-matched shell pilots726350 and all three complete6s histories726362 finish.
New per-case shell submission726387 exits before physics because a legacy
friction assertion preceded the explicit bank-coefficient materialization;
`wall-capped-v1/correction.json` preserves the error. The check is now after
construction, and fresh20-tick pilots726407 pass. Read-only impulse observers
also accumulate physical moment of impulse and exact contact-phase kinetic work,
instead of multiplying an accumulated impulse by a final point. Stock thickness
and physics remain a separate baseline, and finite wall variants are not thereby
reference-qualified.

The physical-step observer pilot726405/726416 keeps10ms capped targets and1ms
physics. Native DPR/SCR match130 common observations within the declared pose/
field tolerances; shell fails that equivalence and stays a separate variant.
Full native physical-step histories726425 retain the actual interior raw peak
fields and control-interval average forces/moments, pending full equivalence.
A new isolated PR22 substep-peak observer adds only a callback after its unchanged
integration. Its combined[read-only patch](../shell_comparison/pr22-substep-peaks.patch)
is separate from the original diagnostic patch. Pilot726429 must record all200
physical steps before full capped shell peak histories use this source.

New cost audit726397 preserves primary, memory and trace scopes and failures.
AtB4096,1ms physics/10substeps, full DPR/tuned DPR/SCR currently measure
27417/30558/10937 aggregate10ms control env-step/s. These are FP64/common-host-
controller measurements, not the earlier FP32/native-controller workload.
Fresh group/cache activation726419 separately measures initialization, build
and first/second step; normal timed configurations can reuse campaign caches.
Three figures726400 follow the[figure contracts](QIPC_REFERENCE_FIGURE_SPEC.md),
retain editable SVG/PDF,450dpi PNG, data and150dpi previews. All pass rendered
text checks, and export QA726404 verifies183mm widths, embedded non-Type3 fonts
and editable SVG text. The actual previews were visually inspected. They remain
explicitly diagnostic where reference qualification is open.

## Complete-history metrics and additional completed controls

The [metric contract](QIPC_REFERENCE_METRICS.md) defines one space/time L2 norm
on the same common tangential domain for all methods. It includes all600 finite
observations, including low-load intervals, without per-time relative-error
masking. Independent full3D solid-only metrics remain supplementary. The exact
executed observer/metric source hashes match the current files for726486 and
726487. Relative discrepancies from the coarse strict SNH candidate are:

| Complete6s contrast | Common tangent | Inner | Outer | Membrane | Bending |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPR versus candidate | 0.999500 | 0.999488 | 0.999503 | 0.999497 | 0.999145 |
| SCR versus candidate | 0.999497 | 0.999485 | 0.999500 | 0.999494 | 0.999133 |
| Stock Shell versus candidate | 4.033453 | 4.118496 | 3.920339 | 4.066672 | 1.324473 |
| Same surface, wall2 versus wall4 | 0.124305 | 0.007386 | 0.006009 | 0.012414 | 0.233756 |
| Surface2/wall4 versus surface1/wall4 | 0.410273 | 0.383388 | 0.398641 | 0.391720 | 0.810806 |
| Half timestep versus1ms | 0.229989 | 0.225615 | 0.234044 | 0.228355 | 0.356276 |
| Nonlinear1e-8 versus1e-9 | 8.791e-8 | 8.660e-8 | 8.937e-8 | 8.700e-8 | 9.606e-8 |

These are measured discrepancies, not qualified physical accuracy scores. The
spatial and timestep contrasts exceed the declared5% common-field target.
Per-time, raw peak and reaction gates are not replaced by this history norm.
Hencky with different stopping/DBC settings changes the history norm0.3376%;
it remains a combined contrast rather than a pure constitutive test.

Surface2/wall2 and surface3/wall2 both complete6s in726365. Friction smoothing
halved also completes6s in726304. Shell narrow-cutoff prescribed/capped histories
726473/726475, strict stopping/half-step histories726480/726478 and matched
full-field cost726460 all complete. Their finite completion does not qualify
their physics or suppress reported penetration/cap/lifecycle defects.

The actual-contact probe726488 completes native prescribed6s and capped9s,
retaining reported penetration, counterpart public-link identity and uncapped
friction utilization with1e-9N normal denominator floor. Outcome audit726482
uses actual mass COMs and authored hand kinematics. First lift/hold maximum
hand-relative COM drift is0.07351mm for the coarse reference candidate,
2.823mm for DPR/SCR and12.54mm for stock Shell. This macroscopic observable is
not a local stick/slip or friction-cone certificate.

Wall-geometry audit726477 uses PR22's actual first fan face at each vertex,
superseding the retained union-envelope assumption in audit-v1. Nominal500um
has about21% normal second-moment discrepancy; volume/area thickness reduces
it to0.344–0.368%, while sampled support gaps remain as large as30.16um.
These direction samples are not an all-direction Hausdorff bound. The actual
default2mm coupling softness adds4.605mm cutoff padding beyond half thickness.
Separate8.686um softness inputs set that cutoff to20um; they retain PR22's
velocity/impulse contact law and are not an IPC substitution or force fit.

Source inspection finds QIPC's implemented barrier pair weight is
`min(Aa,Ab)*dhat/4` and half-plane weight is `Avertex*dhat`, with the solver's
dt-squared scaling. Its friction uses lagged normal force and C1 norm smoothing
at `epsilon_velocity*dt`. Area weighting is present; the name Consistent IPC
alone does not establish convergence. Source formulas and remaining mappings
are in the [working report](QIPC_REFERENCE_RESULTS_20261018.md).

The61440-tet observer initially triggers a cuSOLVER batched-eigensolver API
error even on a known-finite analytic input. Independent NumPy stress/eigenvalue
controls726471 pass with bounded16384-matrix observer chunks. This changes no
elastic energy, native solver or physical state. The corrected refinement
retry726472 records2.343s before the storage incident below; it is incomplete.

## Shared storage incident,2026-10-18 UTC

At approximately11:46UTC several long jobs exit1 near-simultaneously. Their
final result/returncode files are absent; their last saved states are finite
and do not report an invalid step. At11:58UTC a subsequent task-list write
directly raises `OSError122: Disk quota exceeded`; `/mnt/data` reports800TB
used and zero available. A separate write/fsync probe at12:02UTC also fails122.
The missing terminal logs cannot prove the individual exception in every
interrupted worker, so those exits are classified as incomplete during a
confirmed storage outage, not counted as demonstrated physical solver failures.

No task GPU writer remains active. Only private regenerable CUDA/Numba/
Quadrants/RTC caches of explicitly successful terminal groups were removed.
Genesis precomputations, native images/checkpoints, raw arrays, profiler traces,
logs, source receipts and historical evidence remain unchanged. The data mount
still reports no free space after that cleanup. No runtime dataset is diverted
into home. Exact deletions are journaled in
[the storage incident record](QIPC_STORAGE_INCIDENT_20261018.json).

[The affected-run inventory](QIPC_STORAGE_AFFECTED_RUNS_20261018.json) records
last valid journal rows, completion receipts and checkpoint SHA256 checks.
Checkpointed capped case0/half-dt/surface2/strict runs stop at2.338/2.252/
2.416/2.287s. Each has hash-verified actual public native checkpoints through2s.
This verifies files, not cross-process scientific continuation; resume is not
yet implemented or validated. The older non-checkpointed case0/case2 stop at
5.270/8.613s and cannot be resumed from trajectory frames. Case7 does complete
all9000 steps, one protocol reset,0 invalid steps and24 recorded PCG-cap steps.
Its reference qualification remains false.

The uncompleted prescribed wall8, half-contact-band and corrected surface3/wall4
runs reach2.883/5.788/2.343s respectively. These prefixes do not replace complete
refinement histories. Shell max-only observer work is prepared but has not been
executed; there is no Shell max-only timing claim. Final uncertainty/reaction/
energy audits, capped comparisons, secondary Shell timing, figures, compact
publication and requirement closeout still need a writable persistent data
location. The goal stays active and all completion gates remain required.

The12:13UTC write probe confirms the same errno122 storage block. Source review
finds that the old history audit cannot describe missing/corrupt terminal
receipts, and its absent cap flags can appear as zero. Prepared changes now
retain structurally valid journal prefixes with explicit unreadable byte ranges,
unknown counters, process receipts and configured caps. New synthetic metadata
controls cover those admission failures without claiming physical validation.
The prepared Shell max control derives VM independently from complete Cauchy
tensors after an actual loaded prefix and checks selected-environment nonfinite
isolation. The [task contract](QIPC_PENDING_VALIDATION_20261018.md) is preserved.

After a further12:31UTC probe fails122, an explicitly disclosed bounded temporary
project output root and private node-local caches permit actual Slurm execution.
Job726497 completes all eight groups/nine tasks. The12 metadata-admission controls
and independent max-only200-observation control pass; full/max observer storage
is53784/352B atB2, and maximum independent peak difference2.98e-8Pa. Rebuilt
surface2 metric comparisons and per-finger histories still fail the reference
uncertainty targets. Matched Shell timing726506 and fresh capped cases0/2 in
726509 are running. Resolved Shell static-load pilots726513/726515 preserve
source wrenches in the undeformed configuration, admitting3/7 at both6/24 outer
iterations; deformed moment drift and full600-time replay remain explicit.
The [fallback record](QIPC_REFERENCE_FALLBACK_20261018.md) contains exact paths,
completed values, scope qualifications and subsequent job identities. The goal
and all completion gates remain unchanged.

## Completion gates

1. Exact-energy stress/parameter and independent small-control validation.
2. Matched 3D hollow egg/robot inputs and both prescribed-motion and capped
   control protocols, with dynamic histories and declared contact mismatches.
3. Reference mesh/thickness/contact, dt, solve and friction regularization
   uncertainty on controls and a complete representative grasp.
4. Common-position/time tensor/resultant/peak errors for DPR/SCR/shell,
   reference-load replay versus point-wrench reconstruction and full feedback.
5. Lifecycle, batch/restart and failure evidence; separate stock/diagnostic
   variants and retain unresolved quantities.
6. Full-field primary/max-only secondary cost, useful tuned settings, matched
   hardware/memory/profiles, figures, detailed report/reproduction/source closure.
7. Complete requirement audit, focused checks, final commit/push.

All gates remain required; they are not reduced to the inexpensive first checks.


## Post-outage validated checkpoint, 2026-10-18

The disclosed fallback retains original data/builds in place and uses private
node-local caches. Its initial1GB persistent budget was explicitly revised to2GB
before exceeding it; no environment or compatibility link was migrated.

- Validation726497: all eight groups/nine tasks return0, including12 interrupted
  evidence controls and independent loaded full-Cauchy/max-only/NaN isolation.
- Shell paired726506: all groups return0; stock full/max358.142/356.189 and
  volume-surface2 full/max65.4668/65.3310 diagnostic env-step/s. Cost audit726552
  retains112 configurations, eight cold records and six full profile tables.
- Full nominal Shell static replay726518 admits114/600. The supposed volume task
  omitted its wall argument and is a preserved nominal duplicate. Corrected
  volume726545 admits113/600; audit726554 uses only the valid two sources and
  verifies actual mass/thickness. Large deformed moment drift and rejected
  statics prevent pure material-error interpretation.
- Adaptive construction726528 admits145344-tet h0.5mm/wall4, while h0.25mm exceeds
  its face bound. Pilot726533 completes the requested1.16s, but696 PCG-cap steps
  and606.5-weight maximum net residual reject numerical qualification. Its
 14-frame reaction audit726557 emits no full-history norm; field comparison
  fails common-column coverage. Diagnostic726564 finds the first query inside
  an actual tet by independent float64 and80-digit Decimal solves. Coverage
  admission is unchanged; no field score or physical hole is inferred.
- Expanded outcomes/friction726534 and diagnostic audit726532 complete. All
  numerical/refinement/pose scopes and negative outcomes remain explicit.
- Uncertainty figure726561 and export QA726562 pass; actual PNG and independent
  PDF preview are inspected. Earlier clipped/overlapping drafts are retained.

[Checkpointv2](../evidence/20261018-qipc-reference/checkpoint-v2/README.md)
selects existing derived values, exact executed sources/receipts and the new
figure. No scientific analysis runs on the login node. Fresh original-state
9s capped cases0/2 in726509 are still running; neither is admitted as complete.
The goal and unresolved qualification gates remain unchanged and active.


## Material-query correction and complete capped comparison, 2026-10-18

Job726565 validates a direct-coordinate fallback for interval preselection:
actual recorded inside point accepted, true outside point rejected, exact
one-sided boundary interpolation retained and all21 legacy arrays exactly
unchanged. Six adaptive fallbacks recover14 prefix comparisons; at1.13s
common tangent/membrane/bending contrasts are0.628771/0.578531/2.407296.
This repairs observation coverage only;696 PCG-cap steps and606.5-weight
net residual still reject that adaptive pilot's numerical qualification.

Cost figure726571 and export QA726572 pass at183x160mm, with editable SVG,
embedded PDF fonts and450dpi PNG. Both actual PNG and independent PDF preview
are viewed. Full/max, sampled memory, n1 cold activation and overlapping
intrusive kernel durations retain their separate meanings.

The same726572 also completes both case7 common-field comparisons over all900
times/9s. DPR/SCR/stock Shell tangent discrepancies are7.14440/7.17043/18398.8;
volume-surface2 Shell gives1.36409e10. All use the same1511.70Pa reference
RMS over time and1Pa denominator floor. They are diagnostic, not qualified
accuracy scores or evidence that lifting visually improves stress prediction.

Six full original-control case7 sensitivity groups are submitted as726567 with
concurrency3,240-minute limits and actual0.5s checkpoints. The disclosed fallback
budget becomes4GB before crossing2GB, based on measured existing output sizes.
New derived snapshot selection can limit duplicate field exports without
changing any complete-time metric or deleting source raw/peak fields.
Fresh complete cases0/2 in726509 remain live and are not restarted.


## ET failure correction, complete case2 and full9s controls

The current selected evidence is [checkpointv3](../evidence/20261018-qipc-reference/checkpoint-v3/README.md).
It preserves complete selected JSON bytes as lossless gzip plus compact value
selections, exact sources/variants/tasks/patches/return codes and the inspected
full/max-memory-cold-profile figure. The goal remains incomplete.

-726585 strengthens the material-query controls: opposite interior points choose
 distinct one-sided cells at the same exact boundary. All21 old arrays stay
 bitwise equal.726614 adds complete600/900-grid and sparse/missing/shifted-grid
 controls; sparse adaptive pilots emit no whole-history norms.
-726576 removes all PCG caps by increasing only the adaptive pilot's budget to
 16384, but retained logs show2 ET forced exits, maximum net residual119.496N
 (1865.45 egg weights), and raw VM589.948MPa. The4k pilot has6 ET events.
 Raising the budget does not cure early native Newton termination.
-726640 passes15 admission controls and audits25 histories. Missing historical
 per-step ET counters remain unknown; original completed flags are not rewritten.
-726639 validates the new runner gate on a fresh actual failure:1102 recorded
 steps, first ET event1.102s, last invalid true, requested completion false,
 process/group1, no Newton/PCG caps. Static topology, failure frame and actual
 native checkpoint are retained. No continuation through the state occurs.
-726679 checks the exact reported pair with80-digit Decimal and24 analytic
 controls. It is outside the triangle despite the source predicate reporting
 intersection in its approximate-coplanar branch. All5 vertices belong to the
 egg, and global/FEM/active-point saved geometry agrees exactly. This is one
 detector false positive; native forced termination still rejects the solve.
 Earlier forensic versions preserve a mistaken floor-inclusive owner assertion
 and its explicit correction rather than claiming solver-state inconsistency.
-726579/726582 complete unloaded original bank schedules0/2/7 for9s each;
726587 matches all9000 commands/caps/phases/reset epochs and passes the declared
 arm/finger/hand bands. No failed QIPC status/PCG caps. Maximum hand-position
 difference is about10.52um, still material relative to the20um contact band.
-726606 captures immutable initial rigid-frame bindings.726609/726654 verify all
900 recorded actual public body poses for capped7/2 against authored FK at
 actual joint states, with maximum COM errors1.543e-15/1.414e-15m. These are
 geometry controls, not loaded contact qualification.
-726509 case2 completes all9000 steps/9s/one reset.726654 completes both full
 common-field comparisons: tangent DPR/SCR/stock Shell0.988975/0.990957/3439.938;
 volume-surface2 Shell1.981454e7. All900 times use the same reference RMS8971.386Pa.
 Case0 in the same campaign remains running; it is not restarted on a poll.
-726567 wall2 and strict finish full9s.726654/726679 compare all900 observations:
 tangent contrasts1.374548/1.323348 and finger-force contrasts1.0/1.010078.
 Strict changes stopping and budget together and retains1 PCG cap; wall2 has0.
 These failures of the5%/2% targets remain visible despite finite completion.
-726647 completes three matched B1 full-field repeats for both stock/fine Shell.
 These supply the correct batch scope for later B1 error/cost points; B32
 timing cannot stand in for B1 when Shell batch equivalence fails.

The user explicitly authorized continued writes to the existing home fallback
following the storage-policy review. The4GB persistent-output bound and private
node-local caches remain; new submissions record `--allow-home-output`. Original
data, environments/builds and compatibility links remain in place. No unrelated
jobs are stopped. Full raw/source closure and the remaining capped, local-contact,
reference-qualification and final figure/report requirements remain open.


## User-directed execution change: sparse keyframes, profiling first

The user-update section in `QIPC_REFERENCE_COMPARISON_GOAL.md` supersedes dense
stress-observation requirements. Full now means spatial fields at a few
predeclared keyframes. Whole-history stress norms and temporal maxima are no
longer required. Published dense packets remain unchanged as supplementary
evidence. This update is acknowledged before further long submissions.

Next execution order:

1. Continuous original-state prescribed contact window through1.16s, profiled
   over(1.05,1.16]s, including onset and loaded contact. Predeclared diagnostic
   keyframes are1.10/1.13/1.16s, selected from motion/contact stages rather than
   candidate stress errors. Separate controller, solver, native/geometry checks,
   stress/contact observer, scalar/full-field transfer, compression/I/O and cold
   setup. Explicit synchronization makes this an intrusive profile, not final
   uninstrumented throughput. A later matched timing pass must verify the choice.
2. Bounded numerical contrasts around the existing original-main/16eps source:
   baseline, Newton stopping only, linear stopping only and linear budget only.
   A baseline dense-compute/sparse-export control isolates observation cost.
   Keep material/contact physics unchanged. Inspect actual iterations, caps,
   ET/LS/Newton exits, force balance, reactions and common keyframe fields.
3. Choose settings only after measured residual/observable and cost checks;
   qualify capped coupling separately before a new full-length sparse campaign.
   Preserve continuous contact/friction/controller history. Sparse mode must
   avoid `SolidObserver.read_step` on unselected steps, rather than merely
   suppressing files. Native failures, finite states and geometry-only det(F)
   checks remain on every physical step; rejected states retain native dumps.

Existing jobs were assessed without hot edits or cancellation. Fresh case0
726509_0 has passed8.66/9s and remains useful; case7 surface/dt/band controls
726567_1/2/4 continue with their immutable sources. The not-yet-started final
friction task726567_5 is reversibly held (`JobHeldUser`) until short-window
results justify further long work. Existing dense observations can later be
sampled at the declared frames; they do not prescribe new dense computation.
The user's temporary-home authorization and4GB output bound still apply.

## Measured short-window checkpoint and PCG stopping semantics

Jobs726699/726705 execute8 original-state contact-window configurations;
726709 adds3 fresh boundary-timing repeats. Analyses726704/726719 verify all3
predeclared fields/reactions and actual numerical bands. The
[short-window packet](../evidence/20261018-qipc-reference/short-window-v1/README.md)
retains all33 full-spatial NPZs, exact journals/timing data and source variants.
Sparse computation makes3 stress/contact observer calls rather than1160,
with strict same-setting keyframe equivalence. Native solver calls dominate:
66.857s of the baseline67.206s window, versus0.152s observation. The baseline
uses1,102,522 PCG iterations in110 physical steps. No n1 dense/sparse speed ratio
or full-grasp throughput follows from these intrusive spans.

Linear1e-6 at fixed Newton1e-9 passes all3 provisional keyframe bands; the
whole-window repetitions measure59.292/62.896/63.783s, with separately measured
initial read-only elastic activation and deferred compression/I/O. Newton1e-7
alone fails actual net force balance at1.13s despite tiny field contrasts.
Partition-PCG gate off/on fails common field/reaction bands; the enabled gate
has4892 attempts and0 accepts. It is an entry gate, not residual-based adaptive
inexact Newton stopping. Target-binding audits726722/726724 independently
verify robot poses against shared targets and retain actual free-egg COM/tilt
at the keys. Small keyframe target errors do not explain path differences.

The user questioned whether PCG1e-6 is needed when Newton controls nonlinear
precision. Local source confirms that `tol_rate` multiplies the initial
`abs(r^T M^-1 r)`, so its square root is the relative preconditioned residual
norm for a fixed SPD preconditioner. The configuration's short residual/rhs
description is imprecise. Newton compares the mapped vertex direction before
CCD with `velocity_tol * dt`; direct force residuals remain necessary.
The1e-6 setting is provisional and has not been shown necessary. Predeclared
job726725 fixes Newton1e-9 and compares PCG1e-6/1e-4/1e-3/1e-2 with identical
continuous physics,3 keys, instrumentation and actual per-step robot-tracking
checks. All4 fresh processes share an allocation; each tolerance has one run.
Results are pending, outside the frozen11-run packet. No native binary change
or adaptive residual forcing policy is included in this comparison.

Existing case0 completes9000 steps/9s with one reset,0 invalid steps and175
PCG-cap steps; those caps remain visible. Case7 surface2 and half-band also
complete. Half-dt726567_2 is still running and friction726567_5 remains held.
At the latest storage check, the authorized fallback occupies3.721GB of its
4GB bound; the four-run comparison has a20MB output estimate and private
node-local caches. Original data/environments/symlinks remain unchanged.

Job726725 and the four analyses in726726 have now completed with true return
code0. PCG1e-6/1e-4/1e-3/1e-2 all pass the predeclared three-keyframe bands;
window times are59.8877/50.5423/47.4926/34.9529s. Each is a fresh-process n1
contrast in one allocation. Window PCG totals are1008832/810215/729291/500367;
Newton totals1285/1740/2172/2221. PCG1e-4 has1 cap step, the others0, with no
native invalid-step/ET/Newton/LS forced exit. At1e-2 the maximum common tangent
contrast is4.949e-5, finger-force contrast2.606e-5, net residual/weight1.313e-5.
The [companion packet](../evidence/20261018-qipc-reference/short-window-pcg-v1/README.md)
retains all12 new full-spatial fields, exact numerical/timing results, native
stopping source, logs and source reconstruction identities. Prior shared
reference/target/input states remain in the frozen11-run packet. These results
support the user's concern about inner oversolving:1e-6 is unnecessary for the
declared window bands. The next candidate is1e-2 with unchanged Newton/physical
checks, still requiring repeats, capped coupling and complete sparse-grasp
qualification. No full-goal completion or physical accuracy ranking is claimed.
