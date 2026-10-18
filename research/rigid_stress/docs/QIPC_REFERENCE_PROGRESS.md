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
Its only QIPC code difference from641459 is the5-line dense-BVH stack barrier
fix. Baseline726310 still fails after1086 accepted steps. A separately pinned
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
