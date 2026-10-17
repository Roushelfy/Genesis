# Practical throughput experiments, 2026-10-17

The [throughput goal](PRACTICAL_THROUGHPUT_GOAL.md) remains **open**. This follows
the [initial baseline checkpoint](PRACTICAL_THROUGHPUT_PROGRESS_20261017.md).
The [recorded summary](PRACTICAL_THROUGHPUT_EXPERIMENTS_20261017.json) preserves
95 observed scope results and52 performance-eligible points, including
superseded candidates. The [scheduler export](PRACTICAL_THROUGHPUT_EXPERIMENT_SCHEDULER_20261017.txt)
is frozen separately. Performance eligibility checks the recorded numerical
gates, source, exits and timing protocol; it does not replace independent
field/lifecycle qualification or establish final recommendation/completion.
An [independent metadata re-read](PRACTICAL_THROUGHPUT_METADATA_AUDIT_20261017.json)
checks all52 eligible points' actual three2400-step rows, rates, geometry,
controller, policy and precision. Observed policy reference errors remain below
the unchanged1e-9rad budget; they are not configuration equality fields.
A [source-label correction](PRACTICAL_THROUGHPUT_SOURCE_LABEL_20261017.json)
records job724962's actual baseline imports/snapshot after a descriptive engine
label was found incorrect in its retained original submission journal.

All runtime paths below are relative to
`runs/20261017-practical-throughput/` under the data root in `DATA_LOCATION.md`.
Immutable snapshots and exact submission commands remain there. At the frozen
main checkpoint no production candidate had been selected. Subsequent matched
capacity closures below justify retaining the barycentric cache; final operating
settings and the remaining large-batch checks are still open.

## Closed robust baseline through B16384

The original matched configuration, condition bank and900 warmup/three2400-step
repeats remain fixed. Median valid env-step/s on RTX PRO6000 Blackwell:

| Environments | Rigid | Rigid + stress | Policy + rigid | Policy + rigid + stress |
| ---: | ---: | ---: | ---: | ---: |
|32|11293.87|2615.39|9821.80|2524.08|
|256|84845.51|16803.28|75216.98|16271.65|
|2048|434466.83|45118.75|398893.63|44814.51|
|8192|763819.52|52658.16|751745.52|52509.98|
|16384|898658.33|53674.88|891844.22|53560.01|

Parents724933–724936 and724974 now completed0:0. Every repetition has zero
invalid environment steps and zero scatter overflows, and every child closes
all899 snapshot hashes and its file inventory. The live native-buffer counts
are119964854,447926614,3071620694 and12067143254 bytes respectively. These
are feature allocation counts, not peak process VRAM.

Actual baseline B32768 and B49152 trials724962/724971 failed with CUDA OOM
during profile-buffer growth; failed logs and exits remain. B65536 was cancelled
while pending, not observed to fail or pass. B24576 trial724975
and workspace-size2 B32768 trial724976 remain unfinished. B16384 is therefore
the largest closed width so far, not a demonstrated plateau or final capacity.

Separate100ms NVML memory samplers725021–725024 now complete0:0 for all four
scopes at B32/B256/B2048/B8192. Their rates are excluded from headline timings.
Maximum observed compute-process VRAM, GiB:

| Environments | Rigid | Rigid + stress | Policy + rigid | Policy + rigid + stress |
| ---: | ---: | ---: | ---: | ---: |
|32|2.1797|2.3594|2.1875|2.3672|
|256|2.3672|2.9219|2.3770|2.9316|
|2048|3.9043|7.0840|3.9180|7.7520|
|8192|9.1152|25.7949|9.1445|25.8242|

These sampled peaks include setup,900 warmup steps and one2400-step pass per
scope. They are lower bounds with GPU UUID, sampling resolution and peak
sample count retained, not confidence intervals or native-buffer sizes. CUDA
reports101975851008 total device bytes for these allocations. The B2048 policy
peak is the actually observed value, not a smoothed or interpolated estimate.

![Closed baseline throughput and sampled memory](../evidence/20261017-throughput/baseline-scaling-v1/scaling.png)

The [figure packet](../evidence/20261017-throughput/baseline-scaling-v1/README.md)
contains editable SVG, PDF,450dpi PNG, raw repeat/source CSV, exact renderer/style
dependency, compressed input and visual/source/scheduler QA. Both panels use
only complete matched widths. The stress-enabled throughput shoulder is visible;
the four sparse widths do not establish final capacity. The current goal needs
the final selected implementation's operating point and curves as well.

## Complete-period moving profile

Job724992 completed0:0 and its899 source hashes/inventory close. It uses an
otherwise unchanged baseline snapshot with only the benchmark measurement
harness updated: `baseline-complete-cycle-trace-v1`, manifest SHA256
`840fa4ea4408e15f878c10ed52d21498eeee266090e3b83fe72949d60da78407`.
The new `--trace-steps` defaults to the original50;600 records an entire
controller period without changing headline rollout steps or output frequency.
Post-trace errno and invalid-step checks qualify the observed window separately.

The trace covers ticks2400–2999, all256 environment resets and zero invalid
environment steps. It contains218824 GPU kernels,34896 HtoD copies and2463
DtoH copies. The full210598717-byte trace stays on data; the compact
[occupancy summary](PRACTICAL_THROUGHPUT_CYCLE_TRACE_20261017.json) and stage
summary are committed. Runtime API records include93056 `cuLaunchKernel`,8768
`cudaLaunchKernel` and2535 each of `cudaMemcpyAsync` and stream synchronization.
API launches and GPU kernel events have different scopes, including graph replay.

The following are kernel-duration sums divided by600 observed moving steps:

| Recorded GPU kernel stage | Mean summed duration per step, ms |
| --- | ---: |
|Traction fitting|3.662982|
|Elastic solve and initialization|2.033209|
|Load assembly and contact diagnostics|1.609265|
|Collision|1.101704|
|Contact mapping/source capture|0.921725|
|History prediction and maintenance|0.918180|
|Surface sampling|0.736010|
|Complete residual checks|0.303270|
|Global peak/full-output kernels in max mode|0.125947|

All kernel names are classified, with smaller dynamics, integration, controller,
reset, path, source-prior, refinement, acceptance and lifecycle stages retained
in JSON. Classification is by actual kernel names; it does not infer per-kernel
CPU work from a frozen microbenchmark. The summed7757.628ms of GPU durations
and9867.776ms of CPU step annotations overlap. Neither their difference nor
their sum is an unprofiled CPU/GPU timing decomposition. Kernel interval union
is reported separately, and CUPTI overhead stays outside headline claims.

Additional device work-counter runs725032/725033 observe every timed step at
B256/B8192. They retain loaded fractions, maxima/evaluation counters,
integration retries, elastic corrections/fallbacks and invalid steps. In
`body_source_prior`, the union leader carries the cumulative evaluation count;
inactive members retain zero evaluations. The commit path propagates wrench
errors/status to members. The earlier description of duplicated evaluation
counts was incorrect; the recorded values are unchanged. Evaluation counts
are distinct from independent solve counts or GPU wall time. Original
every10-step warmup samples remain a separate distribution.

The complete timed trajectories contain614400 and19660800 environment steps.
Valid stress-contact witnesses occupy97.46354% and two-finger nonzero contact
occupies56.29167% at both widths. A valid witness can have zero force; these
fractions have different meanings. Recorded fit evaluations average2.260143
and2.260089 per observed environment step, with a maximum13 in one step,
maximum1019 samples per source and maximum9 nonzero egg contacts. Integration
retries, elastic corrections/fallbacks and invalid steps are all zero in these
observations. The observed contact mix is preserved across these two widths;
the counts do not prove all workloads have the same expensive-path frequency.

The benchmark observer now additionally counts history predictions that pass
the complete residual, before recovery. Its baseline-only measurement overlay
is `baseline-work-history-metrics-v2`,899-file manifest
`11202ed1feb79f9724f31fdb72a0c2e4e4af9972525edf86ed63598ec1d6d301`.
B256 job725085 completes0:0 with149443 hits out of614400 observed environment
steps (24.32340%), zero invalid steps and closed hashes/inventory. The B8192
counterpart725086 also completes0:0 with4787110/19660800 hits (24.34850%),
zero invalid steps and closed hashes/inventory. These intrusive runs never supply headline
rates. This field is absent, rather than assumed zero, in earlier summaries.

## Bounded optimization results and remaining candidates

The packed-count bypass only activated when the entire750-slot raw capacity
fit the workspace. It never activated in the practical16-slot workload.
Its B2048 live result differed from the original by about3.4% despite taking
the same application path, illustrating allocation/run variation. It is not
evidence of a code speedup.

The merged-capacity readback candidate completes15 GPU and3 CPU field/lifecycle
checks, all passing, but adds one collection kernel per step. Its closed B32
live/policy medians2634.40/2543.06 and B2048 live45250.46 env-step/s differ
from baseline by less than1%. The avoided host fence does not establish a
material end-to-end benefit. The implementation has been removed from the
working candidate; its full source/results remain immutable.

Two numerical/storage experiments have distinct snapshots and are not mixed
into the baseline:

- `lazy-traction-hessian-v1` builds a Newton matrix only for an accepted,
  unconverged iterate. It passes23 CPU checks; GPU checks and same-GPU B2048
  ABBA trial724982 were initially pending. The now-closed same-GPU comparison
  gives a median candidate/baseline ratio0.964993 (3.50% slower). Its B256 diagnostic rate15571.52 is
  below the matched baseline diagnostic16643.53. It performs a second geometry
  traversal when a new direction is required; the measured result does not
  justify promoting the idea. Trace register counts fall242→224, so reduced
  register count alone does not predict application speed.
- `lazy-hessian-reconstructed-fan-v2` additionally reconstructs positions
  instead of caching them, retaining coalesced profile layout. It passes four
  CPU lifecycle/disk checks, but its B256 diagnostic rate14380.14 is lower
  still. Native buffers fall to347754862 bytes. GPU qualification and its
  complete B32768 four-scope capacity trial724988 remain open. Its memory
  reduction and rate penalty must both remain visible.

The current geometry-cache candidate, `barycentric-geometry-cache-v1`, restores
baseline fitting and caches two triangle coordinates per sample instead of
three world coordinates. It reuses those coordinates for both position and P2
shape queries, updates them on disk/fan refinement and grows/copies the complete
required workspace. It changes no physical footprint, force, friction, mesh or
tolerance. The existing direct reconstruction check retains its1e-15m bound.
Its immutable manifest SHA256 is
`4bc5a8871d8968e36a6853409ab2d9dd45944285d0f5402f2dd3ff0538fb6c80`.

The B256 diagnostic has zero invalid steps,16693.37 env-step/s and414536046
native bytes, versus baseline447926614 bytes. The7.45% buffer reduction is
measured. A material steady-state speedup is not demonstrated; usable capacity
gain is now closed by the allocation-matched experiment below.
Four CPU checks724996 and24 GPU cases724997 now pass, as do four CPU and four
GPU cached disk/refinement/torque cases725042/725043. Their original geometric,
wrench, full-field and gauge bounds remain unchanged. The
[targeted check record](PRACTICAL_THROUGHPUT_TARGETED_CHECKS_20261017.json)
retains actual JUnit counts and engine/test hash closures. Same-GPU B2048 ABBA
trial724999 completes0:0 with ratio0.998857: the result is effectively flat.
Candidate B32768 four-scope trial725051 remains open. B40960 workspace2
memory/capacity trial725052 has since closed; its result and allocation-matched
control are described below. The retained
[engine/test source audit](PRACTICAL_THROUGHPUT_CACHE_SOURCE_AUDIT_20261017.json)
matches the five implementation files byte-for-byte to the tested snapshot
and all three test ASTs to their qualified overlays. Final configuration
qualification remains open.

A final small scheduling ablation uses one warp per fit block instead of four,
with the original arithmetic and fallback unchanged. Its B256 diagnostic
16690.89 env-step/s also provides no material small-batch gain. Six targeted
GPU cases725007 pass. Same-GPU B2048 ABBA trial725009 closes with ratio1.003437,
below a1% gain. It is not adopted as a material improvement. Every ABBA
trial retains900 warmup steps and three2400-step repeats; incomplete pairs
are not declared winners. The summarizer records GPU UUID consistency,
baseline drift and the observed candidate/baseline repeat range, without
calling that range a statistical confidence interval.

The fixed6x6 Cholesky/triangular-solve loop-unrolling ablation
`static-traction-direction-v1` changes neither arithmetic order nor fitting
budgets. Eight CPU checks725067 and eight GPU checks725068 pass. Its B256
diagnostic725069 measures16745.02 env-step/s against16643.53 for the original
diagnostic, about0.61% higher. This single diagnostic does not establish a
large-batch improvement. The now-closed matched B2048 ABBA job725090 gives
ratio1.000022, effectively flat. The
[paired closure](PRACTICAL_THROUGHPUT_STATIC_FIT_20261017.json) checks all actual
exits, four source closures, one GPU UUID and twelve unchanged-duration repeats.
This unrolling is not retained. Its manifest
is `c67f82341e5830117d15398d5b12f786d06534f8fd1aea2f3309ceb5ce8e0d6a`.

## Bounded application checks and endpoint refinement

The geometry-cache candidate completes nominal-radius source-prior grasp runs
725093/725094 with the original0.2N/4N and1N/4N force choices,900 warmup steps
and three complete1200-step restarts each. Four fixed contact/load snapshots
725095 also complete0:0, preserving the exact original packet hashes and
numerical gates. The largest recorded complete-residual budget fraction is
0.043551 and the largest raw-peak relative change from the original four cases
is1.282e-11. The closed
[field/trajectory comparison](PRACTICAL_THROUGHPUT_APPLICATION_CHECKS_20261017.json)
725121 finds complete tensor relative L2 changes at most1.075e-10, and identical
sample positions after a bijection by packet/source/face/position. Frontier
sample export ordering differs across launches; row numbers are not physical
point identities. All four position comparisons retain the1e-15m bound and
observe zero matched-position difference. All per-step raw peaks pass the
existing1e-4-relative/1e-3Pa stress bounds; the largest absolute difference is
0.000508Pa. Heights and controller phases are identical. Original within-run
complete residual, gauge, wrench and replay checks remain enforced.

The first comparison job725098 stopped because its checker treated the saved
reconstructed RHS as an immutable input. That output includes the reconstructed
contact force and differs at about4.8e-15N. The source shows that the resolved
inertia/support input itself remains fixed. Job725099 corrects that checker
semantic error, then exposed a second checker mistake: comparing unsorted
traction rows. Version3 reaches the original fixed1N summary, whose old schema
predates configurable radius/force fields. Final checks725112/725121 match
samples bijectively and identify that fixed driver by its exact archived source
hash and actual force arrays. Cross-run nodal loads use the existing source-prior
test1e-4/1e-10 bounds; each run still enforces the stricter within-run load
replay1e-8/1e-12 bounds. These are distinct checks. Failed sources/logs remain,
and the [terminal scheduler record](PRACTICAL_THROUGHPUT_APPLICATION_SCHEDULER_20261017.txt)
retains both failures and successful checks. No production arithmetic or original
acceptance was changed to pass a comparison.

Replacing only nominal source-prior traces in the retained model/radius
reference ensemble preserves7/8 above-band lift decisions versus1/8 whole-episode
decisions for0.2N/4N;1N/4N has1/8 in both scopes. A nominal-only comparison can
give a different count and does not replace the cross-model uncertainty test.
The other three model/radius alternatives reuse original evidence; this is not
new qualification of all alternatives on candidate code. Phase-dependent choice
reversals and independent FEM/R/L/M limitations remain. These bounded checks do
not reopen the complete reference matrix.

B16384 parent724974 is closed. B24576 parent724975 remains open, although its
live child median is52731.09 env-step/s; that child is still provisional.
New matched B12288 trial725087 refines the shoulder;
separate B12288/B16384 memory trials725088/725089 supply missing process peaks.
Workspace2 B32768 and slower uncached B32768 trials724976/724988 remain open.
Every live handle is retained; none was restarted because an observation timed
out. Final settings are selected only from closed, qualified results.

## Closed B40960 live capacity observation

After the frozen main checkpoint, the candidate's workspace2 job725052 completes
0:0 with all899 child hashes/inventory closed. The complete required workspace
grows to9; no valid contacts are truncated. Its900-step warmup and one2400-step
live pass observe98304000 environment transitions, zero invalid steps/overflows,
53464.89 env-step/s and43044167278 native-buffer bytes (40.088GiB).
The separate100ms sampler observes a98383691776-byte compute-process peak
(91.627GiB), near the101975851008-byte CUDA device limit. This sampled peak
includes setup/growth and is a lower bound; end-of-run free memory is not its
substitute. Actual UUIDs, source/configuration, compact repeat records and
scheduler states are in the [capacity record](PRACTICAL_THROUGHPUT_CAPACITY_20261017.json).

This is a single-pass live capacity diagnostic, not a three-repeat four-scope
headline. The combined settings demonstrably run wider than the original block16
baseline's failed B32768. The allocation-matched original-engine workspace2
B40960 job725123 now fails with CUDA OOM in `_grow_sample_profiles`, while the
candidate completes that exact workload. Their recorded CLI files are byte
identical; both contain every required physical/numerical setting. The original
three-coordinate allocation fails where the candidate uses two coordinates.
Candidate workspace2 B43008 job725124 also fails with CUDA OOM during growth.
Both failures close899 source hashes/inventories and actual program/wrapper
exits1; neither is assigned a valid rate. Their
[boundary evidence](PRACTICAL_THROUGHPUT_CAPACITY_BOUNDARY_20261017.json) and
[terminal states](PRACTICAL_THROUGHPUT_CAPACITY_SCHEDULER_20261017.txt) establish
a tested live-capacity bracket of40960 successful and43008 failed for this
protocol/allocation. They do not claim an exact maximum independent of allocator
or workload. The code change therefore improves demonstrated usable capacity;
the observed benefit is not solely initial workspace tuning.

Candidate code-only B8192 four-scope and separate memory jobs725125/725126,
and a second four-scope memory point at B32768 (725150), complete the remaining
selected-candidate curve observations. They retain physical/numerical budgets
and remain open. B16384 full-output lifecycle job725134 now completes0:0.
Its [closed large-batch check](PRACTICAL_THROUGHPUT_LARGE_BATCH_CHECK_20261017.json)
verifies a checkpoint roundtrip, all-environment peak/displacement/history
isolation after resetting four environments, the following untouched physical
step, a complete additional controller period and frozen zero-load/recontact.
Seventeen representative environments retain seven full tensor/von-Mises
snapshots. Native complete-residual checks cover all16384 environments; the
largest sampled budget fraction is0.993147 without loosening the bound. Live
invalid steps are zero. Frozen replay's normalized force/moment errors are at
most5.576e-10/1.193e-9 against the unchanged1e-8 limits, and its largest raw-peak
difference is9.226e-9Pa. Full reset fields are NaN by API contract, while peak
resets to zero. The terminal source inventory,899 hashes, executed diagnostic
source and actual exits all close. This intrusive full-output check is not
throughput or new physical-model calibration. The
[reproduction guide](RUNNING_PRACTICAL_THROUGHPUT.md) separates normal timings,
intrusive memory/work/profile runs and numerical checks.

A bounded final fit-scheduling experiment restricts the body-prior auxiliary
fit grid to its one active constraint per environment, clearing inactive
diagnostics in the existing support-binding pass. Every physical source remains
in the support/fit integrations. It is isolated from the retained cache source,
with899-file manifest
`0705751f810b7a693b3776f3d64230f23255c95f481c674349320e6577bc568b`.
CPU job725138 and GPU job725139 each pass eight checks. The first B256
diagnostic725140 returns0 from the scientific program and preserves all899
source hashes, but its wrapper correctly rejects three prelaunch Ruff cache
sidecars in the source inventory. Its rate is ineligible. A fresh whitelist
copy preserves identical manifest source bytes and reruns that diagnostic as
725151. That clean diagnostic now completes0:0 with16651.46 env-step/s,
against16643.53 in the original diagnostic. This is effectively flat; no material
gain is demonstrated and this scheduling change is not retained. The
[closed diagnostic/test record](PRACTICAL_THROUGHPUT_LEADER_FIT_20261017.json)
and [terminal states](PRACTICAL_THROUGHPUT_LEADER_FIT_SCHEDULER_20261017.txt)
preserve both attempts. The old source, cache sidecars and failure remain;
no live source was cleaned or changed.

The selected code is now the cache-only implementation. Its final matched sweep
uses B8192/B16384/B32768 with four normal scopes and separate four-scope process
memory passes (725125/725152/725051 and725126/725153/725150). The B16384 point
matches the already observed original-code throughput shoulder. These pending
measurements complete selection and curves; the rejected scheduling variants
are not combined into the selected source.

## Delivery still required

Select improvements from closed measured results, finish/refine the practical
large-batch endpoint and peak-memory observations, validate the selected
implementation/configuration, retain a bounded complete-grasp proxy check,
and deliver scaling/memory curves plus reproducible final settings and a
measured bottleneck/plateau account. Existing independent FEM/R/L/M evidence
is reused. Full RL training, a new universal convergence matrix and fracture
calibration remain outside this throughput stage. The prior7/8 named lift
comparison and1/8 whole-episode comparison remain distinct; no force choices
or reward-reporting windows have been retuned.
