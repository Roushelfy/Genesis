# Native rigid stress recovery

The feature reconstructs elastic stress from changing rigid contacts inside
Genesis's existing rigid solver, entirely in Quadrants. It supports normal and
friction loads, raw global maximum/full fields, batch isolation and partial resets.
Use `RigidStressOptions.dpr(...)` for pressure and `.scr(...)` for
intrinsic/body_source_prior. Manual sampling uses `update_mode="manual"` and
`link.request_stress_update()`; every-step sampling remains the default.
See the [API](docs/NATIVE_API.md), [usage](docs/RUNNING.md) and
[source-prior model](docs/BODY_SOURCE_PRIOR_MODEL.md).

## Current target: automatic contact radius for the existing DPR

**Current priority, 2026-10-23:** [maximize this native path's throughput](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md#immediate-goal-maximize-native-automatic-dpr-throughput),
especially small-footprint integration, geometry/moment reuse and batch tuning.
Keep the original DPR/FEM and validate max/full complete-scene improvements;
physical convergence campaigns do not block this performance phase.

**User correction, 2026-10-23:** keep the mature DPR and prefactorized FEM;
add the cheapest useful Hertz/other contact-radius estimate. See
[the corrected goal](docs/DPR_AUTOMATIC_RADIUS_GOAL_20261023.md).
Feed finite Hertz/contact pressure through the existing DPR load transfer into
that same FEM. Keep radius-only and pressure-profile effects separately measured.
Analytic stress enrichment, boundary-traction correction and combined fields
are stopped as the active direction. Earlier results below remain reproducible historical research.

The [native automatic-radius DPR option](docs/AUTOMATIC_DPR_IMPLEMENTATION_20261023.md)
is available on this branch. CPU/GPU controls, the 130-frame sharp-box scene
window, partial reset and unfused execution pass with an explicit planar
edge-footprint approximation and quadrature 16. Matched B64 full scene costs
4.564 ms/step for automatic radius with Hertz pressure, 2.671 ms for manual DPR
and 1.734 ms for pure rigid. The report retains a failed bitwise replay audit
and quantified coarse-mesh physical discrepancies; neither full-cycle grasp
qualification nor sharp-edge peak accuracy is claimed. The original DPR/FEM,
friction, max/full outputs and numerical acceptance checks remain in use.

## Historical research and verified scopes

The [independent single-edge hollow-shell control](docs/SHARP_EDGE_SHELL_CONTROL_20261023.md)
now retains complete-domain fields, exact through-wall integration and refined
positive-depth peaks. At5mN, the unfitted law has4.185% cell-pressure error.
Exact regular-stress integration gives coarse-candidate membrane/bending
errors0.687% /0.815%, with observed variation sums1.301% /1.714%; both pass
the declared conditional checks. Cell-aligned volume quadrature now passes
its observed integration checks, yielding5.618% against the discrete reference.
However, pressure-grid change remains3.055%, above2%; the volume physical-error
gate remains false. The original3.260% estimate is retained as an unconverged readout.
Refined tensile/VM errors are3.761% /1.635% beyond0.707um depth and
2.670% /1.988% beyond7.687um, passing the declared conditional variation
checks. The0.177um tensile peak fails pressure-grid convergence. Original
local140-point results, failed readouts and all numerical gates are retained.
The ideal surface maximum remains unbounded; global continuum peaks and
matching native-scene physical accuracy remain open.

The [actual sharp-edge main pipeline](docs/SHARP_EDGE_PRESSURE_20261022.md#near-saturation-friction-solve)
now admits all130 frames0--129 in both max/full modes. A bounded native block
Newton initialization repairs the former frame110 friction failure without
changing its model or numerical acceptance gates. CPU/GPU exact replay,
48-step histories and16-step sharp/smooth transition controls pass. Valid fixed
friction replay improves from3.07ms to2.09ms; complete scene costs are separate.
The [input optimization](docs/SHARP_EDGE_PRESSURE_20261022.md#loaded-region-allocation-and-moving-boundary-diagnosis)
reduces matched B4 input replay from47.78ms to5.65ms and B64 from50.39ms to5.85ms,
with all25 captured output fields checked against the baseline. These are fixed
input component measurements, not complete scene throughput. Independent physical
errors and full-cycle scene admission remain open. The
[admitted paired closing-window timer](docs/SHARP_EDGE_PRESSURE_20261022.md#admitted-paired-scene-cost)
retains the original pure-rigid/max/full2.64/254.06/254.25ms atB4 and
2.85/286.20/286.32ms atB64. The latest
[ordered geometry search](docs/SHARP_EDGE_PRESSURE_20261022.md#ordered-geometry-search-acceleration)
reduces complete max/full costs to51.03/51.21ms atB4 and72.46/72.35ms atB64
(four initial conditions tiled16 times), approximately5x/4x the original rates.
CPU/GPU disconnected/nonconvex and threshold-boundary controls pass; actual
0--129 audits retain admission and public observation conformance. Raw field
roundoff differences and the initial failed pruning attempt remain documented.
The latest [boundary quadrature scheduling](docs/SHARP_EDGE_PRESSURE_20261022.md#boundary-quadrature-scheduling)
reduces max/full scene costs further to40.69/40.36ms atB4 and53.91/54.06ms atB64,
with unchanged quadrature and public conformance checks. Boundary traction drops
from17.71ms to7.37ms; planar geometry dominates that device profile. The
[cooperative planar root search](docs/SHARP_EDGE_PRESSURE_20261022.md#cooperative-planar-normal-root-search)
now reduces matched max/full scene costs from41.28/41.08ms to31.25/31.07ms atB4
and54.93/55.02ms to50.62/50.68ms atB64. Each source pair shares one allocation.
CPU/GPU geometry controls and every-step public-output checks pass; stricter
raw field failures remain recorded. The planar kernel alone drops13.20ms to
3.10ms in a separate profile. These improvements retain the short100--129
closing window. Earlier complete-cycle audits of the baseline and candidates
all reject max mode at frame428.
An independent CPU/GPU certificate proves that the single region's point force
cannot represent the required wrench (normalized residual lower bound1.004e-3).
The [continuous traction check](docs/SHARP_EDGE_PRESSURE_20261022.md#single-region-pressure-support-and-traction-limits)
also excludes the original force over the predicted undeformed support, even
with arbitrary nonnegative pressure and Coulomb traction. The required planar
pressure centroid lies about75 widths from the sharp edge; the collider's
approximately50micrometre midpoint offset is much larger than the0.60micrometre
predicted width. Cheap load/geometry corrections need independent stress-error
evidence; extra region-moment variables alone cannot repair this model.
The [contact-frame and displacement checks](docs/SHARP_EDGE_PRESSURE_20261022.md#contact-frames-and-shell-displacement)
find that using the opposing face normal increases the force ratio to0.844,
above friction0.719. At admitted frame427, the smooth along-edge correction is
only0.0067micrometre. Conditional local displacement at a declared positive
depth converges to0.154micrometre, far below the collider's application offset.
Adjacent-face contact or a separately qualified nearby-load estimate needs a
physical comparison before changing production behavior.
The [corrected opposing-plane and two-face check](docs/SHARP_EDGE_PRESSURE_20261022.md#adjacent-face-wrench-limits)
locates both raw opponent witnesses on the actual sharp edge. The original
moment requires at least11.8micrometres of side-face depth. Moving each force
to its true witness removes that transverse couple, while the predicted
longitudinal support still fails a continuous wrench-cone certificate.
Ideal sharp edges remain singular; finite sampled peaks retain their probe depths.
The [frictionless two-face elastic control](docs/SHARP_EDGE_PRESSURE_20261022.md#two-face-elastic-reference-and-field-sensitivity)
now satisfies deformed cell-mean face membership with unloaded remote boundaries.
The effective single profile has11.94%/10.35% tensor and37.85%/32.48% tensile-peak
errors at6.595/16.488micrometre depths. Spatial reference uncertainty remains
material. Separate reference-fitted face profiles reduce these errors, motivating
independent two-profile prediction; the fit supplies no runtime qualification.
Hertz numerical and target/source quadrature controls are retained separately.
The subsequent [independent-profile and surface audit](docs/SHARP_EDGE_PRESSURE_20261022.md#independent-face-scales-and-surface-ordering)
finds that a geometric cap improves conditional field errors but still violates
finite-face geometry. The finest linear reference also has8 reversed adjacent
loaded-cell pairs, persistent under refined displacement integration, with local
gradients about1.94. Its field comparisons remain diagnostics; they supply no
continuum two-face physical qualification or admitted runtime correction.

The [contact-motion and sliding control](docs/SHARP_EDGE_PRESSURE_20261022.md#contact-motion-and-prescribed-sliding-control)
captures pre-solve geometry with final rigid velocities and reproduces the
original frame428 refusal on the latest engine. Circular friction cones give
an attaining force-only side-load lower bound of2.119mN, compared with the
frictionless21.729mN assignment. Measured saturated-sliding directions reproduce
a nearby resultant with0.1204% error. Their small local contact solve passes
normal complementarity and cell-mean face checks, but transverse edge refinement
reveals4 reversed loaded-cell pairs and large surface-metric strains. Both
persist as limitations despite improved conditional finite-depth fields.
The original frictionless default arrays remain exactly unchanged. This optional
control supplies no new runtime admission or qualified physical stress error.

The [fixed surface-basis specialization](docs/SHARP_EDGE_PRESSURE_20261022.md#fixed-quadratic-surface-basis-specialization)
retains the quadratic geometry formula while specializing its six nodes and
vertex/edge branches. CPU/GPU curved-field controls and14 source/provenance
regressions pass after repairing a stale test fixture. Source-paired max/full
scene medians decrease0.97%/0.30% atB4 and1.03%/1.24% atB64; B4 repeat ranges
overlap, so this remains a modest measured candidate. Public observations pass,
stricter raw field differences remain recorded, and frame428 still rejects.
Primary campaigns now isolate Genesis/Quadrants caches by source checkout even
inside a shared allocation. The prior cooperative-root performance result and
physical-reference limitations remain the main established evidence.

The [quadratic-edge projection pruning](docs/SHARP_EDGE_PRESSURE_20261022.md#bounded-quadratic-edge-projection)
uses conservative curve bounds to skip unnecessary edge iterations. All45,120
CPU64/CUDA64/CUDA32 projection comparisons match exactly, and existing curved
source controls plus14 source/nonconvex regressions pass. The separately
profiled source kernel decreases3.964ms to3.506ms. Paired full-scene max/full
medians decrease1.36%/0.94% atB4 and0.43%/0.43% atB64, with overlapping B4
full-mode repeats. Public output checks pass; strict raw barycentric/corner
differences and the original frame428 refusal remain recorded. The complete
scene benefit remains modest. The task runner also reads its task plan before creating
output, preserving failed preflight attempts without leaving new partial
directories for missing plans.

The [ordered cooperative source search](docs/SHARP_EDGE_PRESSURE_20261022.md#ordered-cooperative-source-projection)
evaluates candidate faces in CUDA warps and retains the serial face-order,
distance-bound and ambiguity decisions. All45,120 complete-source queries
match exactly across CPU64/CUDA64/CUDA32, and14 source/nonconvex regressions
pass. Source-kernel profiling decreases3.548ms to0.569ms. Paired profiler-off
max/full scene costs decrease9.81%/10.93% atB4 and2.77%/3.53% atB64; the
component speedup therefore differs from complete-scene and batch scaling.
Public observations pass, while strict raw barycentric/corner differences and
the original frame428 refusal remain recorded. Physical qualification remains open.

The [larger-batch study](docs/SHARP_EDGE_PRESSURE_20261022.md#larger-batch-throughput-and-process-memory)
measures B256/B1024 with separate process-memory runs and passing public output
comparisons. B1024 max/full reach 2420 / 2412 env-step/s;
fresh-process sampled GPU peaks are 11.55 / 14.71 GiB.
The batches tile four starting conditions and cover the admitted 100--129 window;
they do not establish the best batch, diverse-population or full-cycle throughput.
Loaded/sharp timed counters now include all repeats; older timing and admission
values remain valid and archived unchanged. Engine code and physical limits are unchanged.

The [boundary traction contraction](docs/SHARP_EDGE_PRESSURE_20261022.md#boundary-traction-vector-contractions)
follows the B1024 profile that places 61.85% of GPU kernel time in boundary traction. A
single native expression uses three matrix-vector products. CPU64/CUDA64 complete
recovery and public scene checks pass. Measured B1024 max/full costs decrease by
2.33% / 2.21%. Strict raw field differences remain recorded, and frame 428 still refuses. The
profile, primary timing and independent physical-error requirements retain their
separate scopes.

The unchanged engine's [same-GPU B1024/B2048 check](docs/SHARP_EDGE_PRESSURE_20261022.md#b2048-throughput-and-sampled-memory)
measures B2048 max/full at2529 /2520 env-step/s, only1.386% /1.366% above
the paired B1024 rates. Fresh-process full-mode sampled GPU memory is27.12GiB.
All130-frame public observations pass across batches, replicated conditions
and the memory run. These remain four-condition tiled windows, with no new
best-batch, full-cycle or physical-accuracy qualification.

The preceding [contact-application study](docs/SHARP_EDGE_PRESSURE_20261022.md#collision-application-points-and-updated-support)
identifies the ground's184micrometre offset as the collider's half-penetration
application point. Refined collision geometry and grouped sources reduce the
updated-load volume/near-field sensitivity to2.68%/1.79%, within the5% budget.
This is a captured-frame model comparison. The earlier
[native geometry/allocation measurements](docs/SHARP_EDGE_PRESSURE_20261022.md#native-edge-geometry-and-force-allocation)
retain component costs for the allocator before backtracking; they exclude
input processing and complete scene recovery.

The latest [remote-edge quadrature and load-allocation study](docs/SHARP_EDGE_PRESSURE_20261022.md#remote-field-quadrature-and-paired-component-cost)
reduces65,536 native64-bit particular-tensor queries from2.83ms to1.25ms in a
paired GPU allocation. Independent prescribed-pressure point error stays below
0.027%; the remote rule also fixes a retained32-bit cancellation failure.
Reconstructing all three physical pressure centers permits a wrench/cone
compatible force allocation, but its fixed-support near-field change is5.67%,
above the5% budget. Keep that load approximation unadmitted. Complete physical
accuracy, actual source binding and accepted max/full scene throughput remain open.

The new [sharp-edge pressure component](docs/SHARP_EDGE_PRESSURE_20261022.md)
uses load, edge slope/curvature and transverse shear without fitting contact
widths. On two captured gaps and reversed shear, independent coupled-reference
pressure errors are3.4--5.8% and positive-depth tensor errors1.3--1.6%. Native
CPU/CUDA32/64-bit controls pass; resident scale cost is measured separately.
Actual source/wrench compatibility, complete shell recovery, edge-peak
interpretation and accepted max/full grasp throughput remain open.

The same [sharp-edge study](docs/SHARP_EDGE_PRESSURE_20261022.md#sharp-edge-particular-stress)
now adds native near-field tensors, using an exact strip and finite line
correction. Continuous prescribed-pressure errors are0.164--0.524% on the
declared samples; CPU/CUDA32/64-bit controls pass. Coupling to the curved-shell
point correction shows that translating pressure to the captured rigid-force
line changes common near-probe tensors by about141%, despite small peak-value
changes. The two-source smooth shell tilt does not explain the displacement.
This rejects that source-placement shortcut; actual scene binding, qualified
complete physical fields and accepted scene throughput remain active work.

The latest user direction on2026-10-22 is to **proceed with sharp edges first**;
the earlier fillet/chamfer dimension wait is superseded. The latest
[force-line repair and sharp-edge diagnostic](docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#curved-force-line-projection-and-edge-geometry)
preserves a curved contact's resultant moment and passes CPU/CUDA controls.
Original/refined Franka still reject at108/106. Sharp-edge reference pressure
and positive-depth stress checks remain diagnostic; pressure/wrench agreement,
complete-field accuracy and edge-peak convergence remain open.

The latest [curved material walk and friction binding](docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#curved-material-walks-and-friction-binding)
uses actual quadratic faces and first-edge crossings. CPU/CUDA geometry,
nonconvex path and moving radial-history controls pass; old-source negative
controls retain their failures. Component timings are measured separately.
Actual Franka still first rejects at108(original collision) or106(refined):
selected box faces match collision normals, but refined normal roots lie outside
by25–34 local Hertz widths. Wrench-compatible curved local frames, finite-edge
pressure/global-peak accuracy and accepted max/full grasp throughput remain open.

**Latest user priority:** implement
[cheap Hertz/thin-shell hard-contact estimates](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
and maximize native throughput. Do not assume or extend a deformable gripper pad.
Coupled automatic contact solves remain optional references unless they prove
very accurate physical stress and a worthwhile measured cost. The pad studies
below are historical evidence, not the current implementation target.

The immediate performance priority is the
[contact-input optimization plan](docs/AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md#immediate-priority-reduce-contact-input-overhead):
profile added work, reuse collision features and continuous-contact caches,
precompute fixed geometry, fuse native work, improve grouping and tune useful
batch sizes. The B4 input pilot includes repeated geometry searches and audits,
not just curvature lookup; its added1.777ms is not an inherent Hertz cost.
Report matched validity/numerics and input/module/full-scene measurements.

The [profiled input optimization](docs/HARD_CONTACT_ESTIMATES_20261020.md#profiled-contact-input-optimization)
now measures matched B4/B64 scene and phase costs. Parallel clearing/active-count
grouping and shared barycentric preprocessing retain every-step validity and
wrenches, exact motion, and CPU/GPU geometry controls. Research audits are
separate from primary timing. Input-only speed does not qualify rejected
contacts or replace complete max/full stress scene measurements.

The [cache ablations and batch scaling](docs/HARD_CONTACT_ESTIMATES_20261020.md#cache-ablations-and-batch-scaling)
now cover B4/B64/B256/B1024, CUDA task attribution, logical memory and exact
shared-environment isolation. The selected nearest-surface bounds reduce the
measured planar device query by about18-21%; inclusive scene gains are smaller
and paired subtraction varies. Full temporal caches remain captured experiments.
All original geometric rejections remain visible; complete stress/history and
physical shell qualification are still open.

The [moving-scene max/full pilot](docs/HARD_CONTACT_ESTIMATES_20261020.md#moving-scene-transfer-and-coarse-maxfull-output)
now connects native Hertz transfer and coarse elasticity for paired B4/B64
Franka runs. Independent CPU/GPU bridge controls and both max/full comparisons
pass after fixed-order operator preparation repairs an observed B64 cross-build
peak failure; the original failure and negative regression remain retained.
Actual boundary/source-moment and friction rejection remains
substantial; B4 admits 658/2808 loaded steps. Proportional shear, unresolved
near-field/shell accuracy and failed intervals prevent public estimator promotion.

The [larger composed scenes](docs/HARD_CONTACT_ESTIMATES_20261020.md#complete-scene-scaling-through-1024-environments)
also pass every-step max/full and trajectory comparisons at B256/B1024.
B1024 measures 15.65/15.67 ms per max/full step versus 7.25 ms pure rigid,
about 65,400 raw env-step/s; only 52.77% of loaded observations pass the discrete
gate. Traction transfer is the main measured batch-dependent addition. These
coarse-field rates retain the same unresolved physical-accuracy limitations.

The [inactive-sample bandwidth optimization](docs/HARD_CONTACT_ESTIMATES_20261020.md#avoiding-inactive-sample-geometry-stores)
reduces same-allocation B1024 max/full step time from 15.673/15.760 to
11.749/11.885 ms, about 25%/24.6%, with unchanged numerical gates and admission.
Negative face tags invalidate unused geometry, avoiding unnecessary clearing;
allocated sample capacity is unchanged. All five measured tasks finish before
preemption; the missing read-only full comparison passes in a separately bound
CPU reinspection. These are raw scene rates, with physical accuracy still open.

The [hollow-shell decomposition](docs/HERTZ_HOLLOW_SHELL_RESIDUAL_20261021.md)
now retains full tensors, material-column membrane/bending and sampled peak
locations for three meshes of one fixed prescribed-pressure egg. Corrected
internal refinement passes; the independent direct-pressure reference and
physical stress-error gates remain unqualified. No scene accuracy is inferred.

The [independent hollow-shell normal-pressure control](docs/HOLLOW_SHELL_CONTACT_PRESSURE_20261021.md)
now solves pressure/support rather than prescribing Hertz pressure. Three fixed
body meshes and a targeted 97-grid refinement pass the original pressure-change
targets. Hertz's fine-case pressure error is 1.06%, with a 2.20% observed
conditional envelope. This single normal polar case supports a cheap pressure
baseline; complete stress, friction and moving-scene accuracy remain open.

The [independent-pressure complete stress comparison](docs/HOLLOW_SHELL_PRESSURE_STRESS_20261022.md)
now gives2.188% volume tensor and5.023%/0.771% tensile/VM peak errors for a
coarse Hertz plus central finite-shell response. Observed reference-variation
envelopes are3.479% and8.007%/0.819%. This offline normal-case qualification
does not transfer to level1 native scene accuracy or general contact/friction.

The [continuous-Coulomb history candidate](docs/HERTZ_CONE_HISTORY_20261022.md)
repairs the old variable-load surface-cone failure in a declared new model.
CPU/GPU controls pass. Same-allocation B1024 history cost falls from4.411 to
0.808ms with16 intervals using a cooperative layout; these are component times.
General vector-friction physics and whole-scene integration remain required.

The [native normal recovery component](docs/NATIVE_HERTZ_RECOVERY_20261022.md)
combines Hertz fields with the existing finite-domain elastic solve. CPU/GPU,
rotation, multiple-source, max/full and failure/release controls pass. Independent
prescribed-load P2 differences are at most 0.01974% for volume tensors and
0.001842%/0.000296% for tensile/VM peaks. Four moment-matched far sources reduce
same-allocation B1024 max/full component time from 12.16/12.21 to 7.39/7.43 ms.
All audited frames pass. Live automatic contacts, moving probes, curved/inner
contacts and physical friction integration remain open; public options still
use their established path. The report pins the experimental engine source.

The [moving material probe extension](docs/NATIVE_MOVING_HERTZ_PROBES_20261022.md)
follows supplied normal contacts with 89 probes per region, locating them through
actual material-face adjacency. CPU/GPU, rotation, release, isolation and max/full
controls pass. Independent sampled corner/peak differences meet the existing
5%/10% engineering budgets; the stricter 0.1% corner target fails at 0.1282%
and remains recorded as failed. Order6 integration passes the moving/rotated
controls and costs 6.63/6.67 ms per B1024 max/full update, including resident
input motion and probe location. Faster order4 retains a rotated wrench refusal.
Automatic live inputs, general physical contacts/friction and scene integration
remain open; these are experimental normal component results.

The [actual native normal route](docs/NATIVE_LIVE_HERTZ_NORMAL_20261022.md)
now connects current rigid contacts and predicted support to public max/full,
principal and moving-probe observations in a pinned experimental engine source.
CPU/GPU independent fields, physical substeps, manual cadence, partial resets,
snapshot/view semantics and release/reload pass. An inherited checkpoint-clock
bug is repaired for both Hertz and DPR; original negative controls remain frozen.
Same-allocation B1024 two-substep max/full normal scene time is 9.02/9.16 ms
versus 2.30 ms pure rigid, with every audited loaded observation admitted.
The final helper uses measured order8. General curved/inner material response,
physical friction history and accepted Franka throughput remain active work.

The [shared source bounds](docs/NATIVE_SOURCE_BOUNDS_20261022.md) retain all
saved source/full-field outputs and seven existing tests on each CPU/CUDA
backend. Two paired source orders improve B1024 max/full scenes by about
5.2–5.5%/4.5–4.8%; B4096 improves by about3.7%. B1 is order-sensitive and
keeps its original max-mode regression. Every audited observation is admitted.

The [independent changing-force friction control](docs/TANGENTIAL_CONTACT_REFERENCE_20261022.md)
now compares radial16 history with vector Cerruti/Coulomb surface contact.
On96 positive-depth points, history's maximum combined tensor/tensile-peak
differences are0.561%/2.364%, versus6.899%/21.435% for proportional shear.
Combined spatial/time reference changes meet2%; strict tangent-only targets
remain failed. These conditional controls support native history integration;
surface/global/full-volume and hollow-egg/friction qualification remain open.

The [native vector-friction route](docs/NATIVE_LIVE_HERTZ_FRICTION_20261022.md)
now advances physical history on every substep while manual observations hold.
CPU/GPU actual force-driven and rotated scenes pass max/full, checkpoint
continuation, selected reset and pure-rigid trajectory controls. Source795805290
measures48.13/48.14ms per B1024 max/full sphere/flat scene step, with every
loaded audit admitted. This is a declared radial approximation on supported
flat material patches; curved/inner material and accepted Franka throughput
remain active work. Independent elasticity checks do not certify pressure.

The [native hollow-egg check](docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md)
locates the actual Franka refusal at a coarse vertex: its smooth contact normal
differs22.64 degrees from the material facet normal. Separate facet-centered
shell controls pass same-load CPU/GPU fields with order16 boundary integration;
order8 fails the unchanged wrench gate. A subsequent audit finds that the
asset lists only80 of160 material-boundary faces. Source2e39f5f75 now completes
the boundary and resolves near-wall integration: four normal controls pass
CPU/GPU and independent higher-order fields below2.58e-9 relative. Parallel
integration preserves those fields and improves the measured moving normal
shell module by2.41–8.51 times acrossB1–B1024. The captured Franka geometry
study finds64 of89 tangent-frame probes outside the faceted material; its
fixed-compliance contact control exposes a substantial facet/smooth-gap
discrepancy. These controls do not qualify physical hollow-shell pressure.
The normal-component buffer repair, failed controls and probe-coordinate
sensitivity remain reproducible. Accepted Franka throughput and curved-source
physical accuracy remain open. The subsequent shape-only quadric curvature
fit reduces the actual first-contact pressure error from6.264% to1.683% against
the retained smooth-gap reference, adding about0.667ms GPU initialization for
the672-vertex control. CPU/GPU fits and curved-shell operator/probe checks pass.
The coarse egg still has9.138% vertex-curvature RMS error, and the actual Franka
frame remains refused by the unchanged material-geometry checks. This is
pressure-input progress; accepted shell stress throughput remains required.

The [native curved-material implementation](docs/NATIVE_HOLLOW_EGG_ADMISSION_20261022.md#native-curved-material-and-source-integration)
now passes independent CPU/GPU stiffness, consistent-mass and complete-stress
checks, plus the existing affine and live-friction controls. A source-centered
quadrature change reduces the captured curved normal-source wrench error from
about1.4% to below7.1e-7 at the same sample count. Its prescribed-Hertz decomposition
comparison passes, while native curved source transfer, moving probes/friction
and admitted Franka throughput remain open. These checks do not qualify physical
contact pressure or shell discretization.

The [first hard-contact controls](docs/HARD_CONTACT_ESTIMATES_20261020.md)
now compare independent half-space contact with circular/elliptic Hertz and a
cheap two-region interaction correction. Ellipticity reduces the rotated-case
pressure error from58.21% to2.41%; one correction reduces the two-region error
from9.63% to2.50%. A Quadrants lookup passes101 cases per CPU/GPU backend and
costs about21us for1–8192 supplied regions on the measured GPU. These are
half-space/component results; shell/friction accuracy, automatic physical-region
inputs, public lifecycle integration and actual scene throughput remain open.

The [finite-footprint transfer](docs/HARD_CONTACT_ESTIMATES_20261020.md#finite-footprints-on-the-witnessed-mesh)
now passes seven CPU/GPU cases and independent same-load complete P2 fields.
It maps96 physical-ellipse samples through actual boundary adjacency, including
small footprints, inner cavities and coarse egg vertices. Dense-rule stress
sensitivity is at most0.199% in these controls. Estimate plus transfer costs
0.173–0.196ms on GPU for1–512 supplied regions. Coarse curved tangential loads
retain a measured moment/cone conflict; a separately measured6x6 correction
is not engineering-admitted. Live grouping/curvature and public scene integration
remain required.

The [normal near-field tensor](docs/HARD_CONTACT_ESTIMATES_20261020.md#a-cheap-normal-near-field-tensor)
now evaluates elliptical Hertz stress through the surface, edge and interior.
Independent circular/pressure-integral controls pass. Native16-point evaluation
passes2,849 tensors per CPU/GPU backend, with at most0.01805% of peak pressure
as the sampled tensor error. GPU cost is0.111–0.173ms for77 probes on1–512
regions. The retained0.2N ellipse predicts a16.5MPa local von-Mises candidate
versus0.086MPa in the coarse shell: correct transfer alone does not resolve
contact peaks. Finite-domain correction, friction history and scene integration
remain required; direct addition would duplicate part of the contact response.

The [finite-box residual control](docs/HARD_CONTACT_ESTIMATES_20261020.md#finite-box-particular-stress-and-boundary-correction)
now subtracts the particular boundary traction before elastic recovery. Coarse
versus fine corrected complete-field difference is1.575%, with sampled peak
changes below0.025%. Direct-pressure FEM still changes8.03% between its final
meshes, so this is internal decomposition evidence, not a qualified independent
physical error. Native normal-box max/full output, moving supplied centers and
curvature, release and isolation are tested separately from scene integration.
Using a closed-form point-force tensor only on the distant box boundary keeps
the full-volume difference below0.024% versus refined Hertz boundary correction
and reduces matched GPU module time by48–57%. Full updates cost1.100/1.566/3.267ms
at B1/128/512. This distance regime does not yet qualify shell inner walls.

The [current-curvature and live-input study](docs/HARD_CONTACT_ESTIMATES_20261020.md#direct-planar-normal-roots)
now directly inverts cached mesh normals for planar opponents. CPU/GPU controls
pass, and the matched moving Franka input-stage median falls from6.825ms to4.025ms
versus2.249ms for pure rigid. Only1381/2808 loaded environment steps pass geometry
checks, and large remaining source moments are incompatible with centered Hertz
pressure. These measurements do not admit usable stress throughput. A separate
[finite-edge control](docs/HARD_CONTACT_ESTIMATES_20261020.md#finite-straight-edge-pressure-candidates)
reduces fine-grid sampled stress error to1.26–5.78% with a mixed variational
pressure candidate. Its offline fitting, half-space samples and sharp-edge
limits remain distinct from native runtime and complete shell accuracy.
Friction history and useful public max/full integration remain open.

The [elliptic tangential evaluator](docs/HARD_CONTACT_ESTIMATES_20261020.md#elliptic-tangential-stress-and-the-cost-of-discarding-history)
now passes independent Cerruti integration and2923 native CPU/GPU tensors per
backend, including surface limits. Its GPU cost is0.290–0.460ms for79 probes on
1–512 supplied regions. A separate reversal control rejects a memoryless
proportional-friction law: at zero tangential resultant it loses residual local
stress and incurs47.76% sampled tensile-peak error against the declared
Cattaneo-Mindlin approximation. The tensor kernel is reusable for traction
superposition; a cheap native history law and shell/scene integration remain open.

The [first CPU contact-region study](docs/AUTOMATIC_CONTACT_PROGRESS_20261020.md)
now measures distributed-source double spreading: the retained2mm DPR baseline
places about42% of normal load on the inner boundary. Conservative outer P2
transfer preserves force/moment/work and changes complete stress by8.70% on
the same linear elastic mesh. This is a transfer contrast, with inspected
pressure maps; its physical error is now measured separately below.

The [native transfer and source-input study](docs/RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md)
adds400-step Quadrants resolved-pressure conformance and complete1000-step
DPR/SCR bounded-roundoff variants with independent mechanics/cost admission.
Original diverse SCR conflicts remain rejected; env330 additionally has a
force-only continuous certificate. The [final fixed-radius comparison](docs/QUALIFIED_FRICTION_RESULTS_20261020.md)
now includes qualified fine references,54 dimensional errors,18 conditional
rankings, matching costs and four visually inspected figures.

The [eight-case compliance study](docs/PAD_COMPLIANCE_STUDY_20261020.md) now
separates model and source-sampling error. Rigid prediction is close in the
nominal soft-pad regime; pressure error reaches24.47% for the flexible-link/stiff-pad
case, and diagonal compliance adds little accuracy. These pad results remain historical controls;
current implementation prioritizes cheap hard-contact estimates. Exact elimination
of pad variables now reproduces all16 coupled reference fields; paired CPU
solves fall from23.5–24.6s to0.060–0.093s, with fixed preprocessing separate.
Fixed/adaptive PCG controls retain the same outer/physical checks; the tested
adaptive heuristic increases total iterations. New250/1000um physical controls
show15.52% versus0.49% rigid-pressure error with the hard pad. The reduced
pressure stage now runs in Quadrants:48 CPU/GPU cases and48 independent complete
fields pass, with at most1.50e-10 stress difference from the full-QP references.
Warm repeats, release, failure isolation and partial recovery pass atB8/B2.
Subsequent native factor/map construction and cold initialization also pass all48
pressure cases and their independent complete fields. Compact native compliance
construction matches independent FEM to4.16e-14 under frame/material changes. Large physical compliances and geometry
maps remain supplied inputs; public automatic integration and controlled
module/scene throughput remain open.

The [current opposing-geometry component](docs/PAD_COMPLIANCE_STUDY_20261020.md#current-opposing-geometry-component)
passes five focused cases on each of CPU/GPU. It preserves link-pair wrenches
when collision sources are split and selects farther accessible boundaries
when the nearest surface is occluded. Analytic/mesh blockers and incident-face
constraints prevent false contact through material or artificial triangulation
edges. Current poses, velocity, environment isolation and partial resets pass.
This component is not yet connected to pressure prediction; composite interior
union distances, unsupported shapes and complete physical support remain open.

The [engineering load-sensitivity study](docs/ENGINEERING_LOAD_SENSITIVITY_20261020.md)
adds18 qualified resolved-transfer errors: corrected-boundary tangential error
is10.56%, versus7.94% for fixed-radius DPR, with metric-specific conditional
rankings. Signed tensor cross terms now demonstrate partial cancellation by
the old wrong-boundary transfer. Improved surface paths admit cooperative SCR
loads within1%/0.1% aggregate budgets, with8.76%/14.66% complete-stress changes
and measured individual-source redistribution. The earlier29.81%/33.51%
unbudgeted changes apply only to the narrower centroid-path diagnostic.
Numerical/path sensitivity, a matched source-local control and inspected maps
are retained. No production tolerance is relaxed.

Native DPR now provides `load_transfer="resolved"` for already integrated
pressure samples. It uses the witnessed material boundary, including cavities
and nonconvex faces, with conservative P2 work transfer. Nonconvex/cavity work
and isolation controls, native max/full checkpoint/reset controls and six
manual-pressure regressions pass. The retained normal/friction complete fields
agree with independent CPU resolved transfer to1.26e-11. A separate400-step
normal/friction history passes every observation and mechanical window with
explicit bounded source reconciliation; endpoint stresses agree with the earlier
resolved research adapter to3.24e-11. This removes a second
disk from a declared resolved input; automatic physical support/pressure
prediction, its material/pad configuration and matched production costs remain
open.

The [active goal](docs/QUALIFIED_STRESS_RANKING_GOAL.md) is qualified DPR/SCR/Shell
stress-error comparisons on a free full hollow egg between parallel plates.
The [accuracy/throughput protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md)
also requires independent GPU module and actual native scene measurements now.
The goal remains in progress. Full Franka physical-reference qualification is
outside this goal; its low-resolution scene remains useful for performance.

| Track | Verified result and remaining work |
| --- | --- |
| Physical normal loads | [Qualified normal fields](docs/NORMAL_PAD_QUALIFICATION_20261019.md), [common peaks](docs/COMMON_PEAK_BATCH_RESULTS_20261019.md), [81 dimensional errors](docs/NORMAL_ABSOLUTE_ERRORS_20261019.md) and [inspected spatial maps](docs/NORMAL_SPATIAL_COST_20261019.md). Ranks depend on metric and reference uncertainty. Fixed250um Shell fields are not mesh-independent thin-shell accuracy. |
| Same prescribed load | [C2 reference](docs/PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md) passes25 qualification checks; [three-method comparison](docs/PRESCRIBED_METHOD_RANKING_20261019.md) supplies27 errors,15 signed tensor splits and12 complete-resultant checks. This is static prescribed-load validation. |
| Physical load/response separation | [250um Shell controls](docs/SHELL_PAD_CONTROL_SEPARATION_20261019.md) pass151 aggregation checks. Continuous, own-pad static, pressure replay and same-source solid fields remain distinct; their signed tensor differences retain cross terms. |
| Sustained friction | [Final qualified comparison](docs/QUALIFIED_FRICTION_RESULTS_20261020.md):251 reference checks,36 comparisons/18 envelopes,218 error-audit checks for54 dimensional errors/18 conditional rankings,100 matching-cost checks and four inspected final figures. DPR/SCR complete tangent errors are about7.94%/7.99% versus25.98% repaired Shell; raw Mises peak-magnitude error favors Shell. Baseline every-step failures and separately admitted1000-step bounded-roundoff variants remain distinct. |
| Research workflow cost | [24 normal timing cases/12 process-memory profiles](docs/COMPLETE_PAD_COST_20261019.md) and [friction keyframe costs](docs/FRICTION_KEYFRAME_COST_REUSE_20261019.md) include their documented research work. [Matched normal associations](docs/MATCHED_PAD_ERROR_COST_20261019.md) pass478 checks for24 configurations/120 applicable metrics; final friction associations remain pending. These are separate from production throughput. |
| Native production/module work | The [source-bound scene summary](docs/NATIVE_PERFORMANCE_SUMMARY_20261019.md) passes14,506 checks for61 primary configurations and20 independent-process memory profiles, preserving observed sampling gaps, exact conformance and source identities. The [minimal MPR repair](docs/NATIVE_CONTACT_CONSISTENCY_20261019.md), original motion failures, diverse SCR certificates and [sparse-output checks](docs/NATIVE_OBSERVATION_CADENCE_20261019.md) remain explicit. Four current-source B8 module modes retain replay/170 binding checks; the independent collector passes14 controls and all24 profile query-gap targets. Final physical-error/throughput consolidation remains open; collider speeds inherit no physical-pad errors. |

The [completion checklist](docs/QUALIFIED_STRESS_COMPLETION_CHECKLIST.md) and
[execution record](docs/QUALIFIED_STRESS_PROGRESS_20261019.md) retain outstanding
requirements and failures. A passing endpoint, numerical-core agreement or a
new report cannot close missing physical-reference or complete-history checks.

## Reproduce and interpret measurements

Follow the [parallel-plate running guide](docs/RUNNING_PARALLEL_PLATES.md).
It links source pins, data identities, captured task recipes and audit commands.
Use the established `rtx-mid` / `GENESIS_IMAGE_VER=1_26 gs-srun` workflow and
[storage policy](../../DATA_LOCATION.md); runtime data and logs stay on data.
Keep expensive refinements opt-in and never hot-edit an active engine source.

Stress-module replay excludes physical dynamics. Native production timing
compares pure rigid with max/full stress on identical evolving scenes, retaining
required runtime validity checks and GPU output. Full host readback and policy
inference require their own accounting. Default few-collider/6mm source timings
cannot inherit the qualified full-centroid/2mm pad errors. Source, geometry,
load representation, precision, cadence and solver settings must match before
joining error and speed. See the [protocol](docs/ACCURACY_AND_THROUGHPUT_PROTOCOL.md).

## Earlier accepted work and limitations

The [pressure comparison](docs/PRESSURE_COMPARISON_RESULTS_20261018.md),
[Shell PR22 comparison](docs/SHELL_COMPARISON_RESULTS_20261018.md),
[practical throughput study](docs/PRACTICAL_THROUGHPUT_RESULTS_20261017.md) and
[application study](docs/PRACTICAL_APPLICATION_RESULTS_20261016.md) retain their
own workloads, numerical acceptance and model limits. Their different rates
are not interchangeable current accuracy/cost recommendations.
The [Franka/QIPC report](docs/QIPC_REFERENCE_RESULTS_20261018.md) and
[bounded plate report](docs/PARALLEL_PLATE_RESULTS_20261018.md) preserve failed
reference/contact qualification and useful independent controls; they are not
accepted universal physical ground truth. Neither recovery route calibrates an
absolute physical fracture threshold.

Tests and useful reference modules retain their imports. The
[cleanup report](docs/BRANCH_CLEANUP_RESULTS.md), [archive](archive/README.md)
and immutable [evidence packets](evidence/) preserve earlier source identities,
negative results and restoration recipes without reopening historical goals.
