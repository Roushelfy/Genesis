# Practical throughput experiments, 2026-10-17

The [throughput goal](PRACTICAL_THROUGHPUT_GOAL.md) remains **open**. This follows
the [initial baseline checkpoint](PRACTICAL_THROUGHPUT_PROGRESS_20261017.md).
The [recorded summary](PRACTICAL_THROUGHPUT_EXPERIMENTS_20261017.json) preserves
53 observed scope results and36 performance-eligible points, including
superseded candidates. The [scheduler export](PRACTICAL_THROUGHPUT_EXPERIMENT_SCHEDULER_20261017.txt)
is frozen separately. Performance eligibility checks the recorded numerical
gates, source, exits and timing protocol; it does not replace independent
field/lifecycle qualification or establish final recommendation/completion.
An [independent metadata re-read](PRACTICAL_THROUGHPUT_METADATA_AUDIT_20261017.json)
checks all36 eligible points' actual three2400-step rows, rates, geometry,
controller, policy and precision. A [source-label correction](PRACTICAL_THROUGHPUT_SOURCE_LABEL_20261017.json)
records job724962's actual baseline imports/snapshot after a descriptive engine
label was found incorrect in its retained original submission journal.

All runtime paths below are relative to
`runs/20261017-practical-throughput/` under the data root in `DATA_LOCATION.md`.
Immutable snapshots and exact submission commands remain there. No production
candidate has been accepted for final delivery at this checkpoint.

## Closed robust baseline through B8192

The original matched configuration, condition bank and900 warmup/three2400-step
repeats remain fixed. Median valid env-step/s on RTX PRO6000 Blackwell:

| Environments | Rigid | Rigid + stress | Policy + rigid | Policy + rigid + stress |
| ---: | ---: | ---: | ---: | ---: |
|32|11293.87|2615.39|9821.80|2524.08|
|256|84845.51|16803.28|75216.98|16271.65|
|2048|434466.83|45118.75|398893.63|44814.51|
|8192|763819.52|52658.16|751745.52|52509.98|

All four parents724933–724936 now completed0:0. Every repetition has zero
invalid environment steps and zero scatter overflows, and every child closes
all899 snapshot hashes and its file inventory. The live native-buffer counts
are119964854,447926614,3071620694 and12067143254 bytes respectively. These
are feature allocation counts, not peak process VRAM.

Actual baseline B32768 and B49152 trials724962/724971 failed with CUDA OOM
during profile-buffer growth; failed logs and exits remain. B65536 was cancelled
while pending, not observed to fail or pass. B16384/B24576 trials724974/724975
and workspace-size2 B32768 trial724976 remain unfinished. B8192 is therefore
the largest closed width so far, not a demonstrated plateau or final capacity.

Separate100ms NVML memory samplers run for all four scopes at B32/B256/B2048/
B8192 (725021–725024). Their rates are excluded from headline comparisons.
The reported maximum will be a sampled compute-process VRAM lower bound,
with GPU UUID, sampling resolution and observed peak sample count retained.
Four-scope final capacity and memory curves remain delivery work.

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
`body_source_prior`, member diagnostics repeat the group's evaluation count;
their contact-summed counter is a work proxy, not a count of independent dual
solves. Original every10-step warmup samples remain a separate distribution.

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
  ABBA trial724982 are pending. Its single B256 diagnostic rate15571.52 is
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
measured; a material speedup or usable capacity gain is not yet demonstrated.
Four CPU checks pass;24 GPU cases724997 and the same-GPU B2048 ABBA trial724999
remain open. Production/test edits remain provisional until qualification and
the relevant before/after measurements close.

A final small scheduling ablation uses one warp per fit block instead of four,
with the original arithmetic and fallback unchanged. Its B256 diagnostic
16690.89 env-step/s also provides no material small-batch gain. Six targeted
GPU cases725007 and same-GPU B2048 ABBA trial725009 remain open. Every ABBA
trial retains900 warmup steps and three2400-step repeats; incomplete pairs
are not declared winners. The summarizer records GPU UUID consistency,
baseline drift and the observed candidate/baseline repeat range, without
calling that range a statistical confidence interval.

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
