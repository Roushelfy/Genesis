# Native rigid stress recovery

The feature reconstructs elastic stress from changing rigid contacts inside
Genesis's existing rigid solver, entirely in Quadrants. It supports normal and
friction loads, raw global maximum/full fields, batch isolation and partial resets.
Use `RigidStressOptions.dpr(...)` for pressure and `.scr(...)` for
intrinsic/body_source_prior. Manual sampling uses `update_mode="manual"` and
`link.request_stress_update()`; every-step sampling remains the default.
See the [API](docs/NATIVE_API.md), [usage](docs/RUNNING.md) and
[source-prior model](docs/BODY_SOURCE_PRIOR_MODEL.md).

## Current goal and verified scope

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
