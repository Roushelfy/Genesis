# Source-bound native performance summary

Scene consolidator730541 passes14,506 checks, binds61 original primary
configurations to their complete motion/output controls and rehashes4,319 files.
It replaces only the20 separate memory observations with independent-process
profiles; all61 performance/configuration/error rows are identical to730473.
Four module memory profiles have their own170-check binding audit. Actual
GIL-hold/error/cleanup controls pass14 checks and all24 profiles meet the
predeclared1s observed-query-gap target. Physical errors for these collider
configurations remain unqualified.

The [original report](NATIVE_PRODUCTION_PERFORMANCE_20261019.md) retains the
superseded engine's32/259 failed motion checks and its rates. The
[contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) retains the MPR repair,
regressions, diverse-input failures and measured tuning. Earlier730473 and
thread-memory receipts remain preserved with their original scope.

## Configuration and accounting

The canonical immutable engine is1ff39b077. All858 captured engine/workload/asset
files and submitted worker inventories are rehashed. The earliest repaired B8
control still records92692aa61 plus its actual two-file patch. Its only remaining
difference from the canonical858-file inventory is three comment lines in
array_class.py: exact reconstructed original bytes/hash and whole-module Python
AST equality establish equivalence. Its original revision/patch is preserved,
rather than silently relabeled. This does not admit the genuinely different
unrepaired92692aa61 engine or its scene rates.

All positive scene rows use the moving eight-condition Franka/egg bank, mesh64
collision geometry, level1 recovery mesh, FP64, history0, dt0.01s and one
substep. Each primary process has900 warmup steps and three1200-step intervals.
Full/max output stays on GPU. Physical integration, contact, control, requested
recovery and required native acceptance remain timed; full host dumps,
independent audits and research plotting stay outside. Different cohorts/batches
can use different physical GPUs of the same RTX PRO6000 Blackwell Server Edition
model. Each overhead uses only its own batch/cohort's paired rigid baseline.

Counts are aggregate environment scene steps, not policy transitions. The
original benchmark emits a policy descriptor and allocates its network even in
rigid/live modes, but source review shows inference occurs only in policy
scopes. These measured rigid/live scopes do not execute inference. Their reported
setup timer ends before that allocation and excludes warmup; it is not total
cold-start time. The cadence worker has a separate construction/JIT timer.
Neither those setup scopes nor module-update rates should be interchanged.

## Every-step scene throughput

Values are median aggregate env-scene-step/s. The CSV/JSON retain minimum and
maximum warm-repeat rates, batch-step latency, paired added latency, exact GPU
UUID, full normalized stress options, mesh/bank/source signatures and the
actual conformance receipt for every row.

| Batch | Rigid | DPR max | DPR full | SCR max | SCR full |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 541.84 | 327.98 | 326.67 | 108.53 | 110.52 |
| 8 | 2885.44 | 2155.58 | 2145.17 | 656.64 | 654.74 |
| 32 | 11430.27 | 8434.60 | 8435.38 | 2603.79 | 2604.72 |
| 128 | 44162.83 | 31816.51 | 31842.80 | 10703.99 | 10790.97 |
| 512 | 171797.79 | 103693.64 | 101302.98 | 35751.78 | 37315.45 |
| 2048 | 443437.69 | 207430.11 | 203983.09 | 80556.67 | 80287.69 |
| 8192 | 783736.98 | 280707.71 | 275221.63 | 110557.53 | 110004.62 |

The largest tested batch gives the highest measured aggregate rates in this
matrix, but overhead relative to its paired rigid run also grows: at B8192,
DPR max/full add179.20%/184.77% scene latency and SCR max/full608.90%/612.46%.
At B128 the corresponding additions are38.80%/38.69% and312.58%/309.26%.
B8192 is a successfully tested operating point, not a located memory limit,
diverse-input guarantee or universal RL capacity.

## Observation cadence is a separate configuration

The additional26 rows cover B8/B128, max/full and stride1/10/50 using their own
paired baselines. They retain521 complete motion/output/reset/source checks.
See the [cadence report](NATIVE_OBSERVATION_CADENCE_20261019.md) for all values.
At B128, its rigid baseline is44385.6 env-scene-step/s; DPR max is32108/42255/43562
and SCR max10833/33013/41164 across those strides. These are a separate cohort
from the B128 row above. The summary retains both instead of choosing the faster
measurement across allocations.

Use every-step observations when every physical step needs a stress value.
Stride10/50 provides observations at10/2Hz rather than100Hz for this dt;
unsampled getter values are held or reset and do not certify missed temporal
peaks. Freshness requires the explicit observation mask, not step_valid alone.
Sparse rows inherit no every-step memory profile or every-step reliability.

## Sampled process memory and its observed gaps

The current20 scene profiles are separate fresh processes with one complete
1200-step interval after900 warmup steps. Their rates remain intrusive diagnostics;
only the original primary rates appear in the throughput table. NVIDIA samples
include parent-process allocations outside Torch and the logical native inventory.
The independent collector logs all query times, even before GPU visibility.

| Batch | Rigid MiB | DPR max MiB | DPR full MiB | SCR max MiB | SCR full MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2334 | 2510 | 2510 | 2640 | 2640 |
| 512 | 2656 | 2928 | 2992 | 3538 | 3474 |
| 2048 | 4004 | 4724 | 4948 | 6422 | 7254 |
| 8192 | 9340 | 12108 | 12940 | 21902 | 22830 |

Across these20 scene and four module profiles, the largest observed query-start
and process-sample gaps are0.759928s and0.759925s. All meet the predeclared1s
query-gap target. Counts, initial visibility, query durations, complete polling
records and primary/profile GPU identities remain in the receipts. Polling at a
requested0.1s plus query/scheduling latency can still miss shorter transients;
these are sampled peaks and successfully tested jobs, not a continuous peak or
located hardware-capacity limit. Sparse configurations inherit no memory profile.

Earlier thread profiles had first process samples at12.46–21.15s and maximum
observed sample gaps0.504–34.513s. They remain preserved. The new B8192 SCR full
observation is22830MiB versus its earlier20270MiB. These are separate fresh-process
observations subject to compilation/allocator transients and sampling; preserve
both rather than infer a general storage ordering from the earlier lower full
value. The controlled GIL experiment below establishes the collector mechanism,
not a unique cause for every historical difference.

## Reliability and physical-error boundaries

The eight-condition bank repeats at larger batches. Separately authored
B128/B512 banks pass64 complete rigid/DPR checks in total; no primary timing
was measured for those passing controls. All four SCR max/full attempts fail
during warmup with their original status36 logs and nonzero return codes.
Separate after-every-step max-output captures at767/env9 and645/env330 have
independent continuous-wrench infeasibility certificates. Those exact captured
states are not assigned to full-output runs, whose own failures are retained.
These are observed attempted configurations, not a statistical failure-rate
estimate. Do not drop environments, clip loads or change radius/friction to
turn these failures into throughput.

The nominal6mm native-collider source configuration differs from the qualified
2mm full-centroid physical-pad case. Every physical-error cell in this scene
summary remains null. The [matched pad summary](MATCHED_PAD_ERROR_COST_20261019.md)
provides separate legitimate error/cost associations. Every-step native pad
friction failures remain separate from successful requested keyframes and from
this moving collider bank. Fine-friction reference/ranking/figure qualification
and final combined delivery remain required by the active goal.

## Reproduce the join and current-source module follow-up

Run native_scene_summary through the established shell_comparison.submit
rtx-mid/container path, with explicit --root RUN_ROOT and --b8-equivalence
RUN_ROOT/meta/native-performance-b8-comment-equivalence-v1.json. Reuse the
immutable task JSON and worker receipts in the evidence packet; choose a fresh
output directory. For the current join, use the paired --memory128-512/--memory-audit128-512
and --memory2048-8192/--memory-audit2048-8192 overrides in
meta/native-scene-process-memory-summary-v1-tasks.json. Omitting these keeps the
historical default thread-profile paths. The reader performs metadata/hash/configuration association,
not a new scientific trajectory integration. Original730471 fails on an absent
rigid-only stress-options field;730473 fixes that schema handling while requiring
the complete options for every stress-enabled record. The original failure and
captured source remain intact.

## Current-source resident module replay

Primary730460 completes all four modes and independent replay audit730463
passes. Current independent-process memory730526 is bound by audit730536's170
checks, rehashing858 engine/asset files and293/297 primary/profile workers.
Original thread run730461 and audit730498 remain preserved. All modes retain
1200 frames,24 typed device buffers and exact partial stress resets. Complete
replay peaks and11 full snapshots per full-output run match within unchanged
rtol1e-4/atol1e-3Pa criteria. Largest primary replay peak difference is0.000374Pa;
this numerical consistency criterion is not a physical-error bound.

| B8 mode | Median env-updates/s | Minimum–maximum | Batch-update ms | Sampled process MiB | Samples | Largest observed sample gap s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dpr-max | 8232.96 | 8222.79–8235.81 | 0.971704 | 3516 | 1364 | 0.177 |
| dpr-full | 8087.35 | 8083.33–8100.26 | 0.989199 | 3528 | 468 | 0.206 |
| scr-max | 966.09 | 963.96–966.19 | 8.280835 | 3548 | 794 | 0.211 |
| scr-full | 965.15 | 964.38–966.63 | 8.288890 | 3560 | 680 | 0.195 |

All primary modes share one GPU allocation; memory uses another GPU of the same
model. Module timing includes device restoration, dispatch and unchanged native
reconstruction/recovery/acceptance, with three warm histories. It excludes scene
dynamics, original recording, validation, compression and policy execution.
The four profiles each use one diagnostic replay and sample from import through
recording/validation/replay. Each retains1,240,128,000 bytes of input history;
the displayed process memory includes that history and construction transients,
rather than measuring recovery workspace alone. Their sample gaps and initial
visibility delays remain explicit, so these peaks carry the same sampling limits
as the scene profiles. Profile times never replace primary rates. No large-batch
module scaling or production-scene memory is inferred from these B8 measurements.

The first memory reader730464 required an absent per-task returncode from a
single-task non-keep-going audit. The corrected reader requires the successful
group returncode and exact sole declared task; original failure bytes remain.
Only the metadata reader reran. Both numerical campaigns and every original
physics/acceptance check stayed unchanged. The initial60min numerical-job limits
were reduced to30min before execution using the old job's actual8m19s as the
resource basis; new jobs complete in8m03s/7m58s. Original submissions and exact
scheduler updates remain retained. Fine-friction qualification and final combined
delivery remain open.


The [performance summary packet](../evidence/20261019-qualified-ranking/native-performance-summary-v1/README.md) retains203 compact payloads/1,464,777 bytes with exact original/encoded hashes, both failed metadata readers, source/worker variants, actual terminal receipts and CSV tables. Sixteen full replay histories/trajectories stay on data with external hashes. Live fine outputs and final friction products are excluded.

## Independent polling follow-up

The observed thread-sampling gaps motivated an instrumentation repair, without
changing the numerical drivers, primary timings or live references. A separate
stdlib-only Python process now queries only its parent PID through NVIDIA and
retains every query's start/end time, including queries before GPU visibility.
A temporary file in the explicit job TMPDIR prevents pipe capacity from blocking
the collector when its parent holds the GIL. Normal completion, query failure,
owned-child cleanup and disabled sampling have explicit behavior.

Actual integration control730520 passes14 checks. During a3s C call retaining
the parent GIL, the captured old thread has a3.06274s sample gap; the independent
collector obtains23 samples with a0.133726s maximum gap. The complete query
history's maximum start-time gap is0.169658s. The child imports no scientific
runtime, targets the same GPU parent and exits normally. A private NVIDIA-command
failure exits17 and is rejected with the child reaped; the test's PATH is restored.
This identifies a real mechanism for thread gaps, rather than establishing that
every historical gap has the same cause.

The two profile wrappers add only a PID import/output record: whole-module AST
comparison after removing those two additions matches their original captured
wrappers. Numerical runpy calls, arguments and acceptance remain unchanged.
New separate profiles730526/730531/730533 cover the same four B8 module modes
and twenty B128/B512/B2048/B8192 scene modes. Their audits require exact source,
configuration, parent/collector identities, successful child exit and complete
poll/sample consistency. Observed gaps and a predeclared1s query-gap target are
reported separately from mechanical admission. The61-row follow-up730541 changes
only separately bound memory observations; all primary performance/error rows
must stay identical. All24 fresh profiles pass their own complete-runtime and independent polling
audits. Final730541 passes14,506 checks with4,319 file identities; its61 primary
rows are unchanged except the separate memory observations. Module/scene
primary timing, physical errors and live references remain unchanged.

An independent sampler removes the parent's GIL dependency. Query latency,
scheduling and the polling interval can still miss short transients; a passing
coverage target is not a continuous peak or maximum-capacity guarantee. Original
thread profiles and all earlier scoped observations remain preserved.


The [independent-process packet](../evidence/20261019-qualified-ranking/independent-process-memory-v2/README.md)
retains331 payloads/2,775,215 bytes. Eight full replay arrays and twenty raw
benchmark JSON files remain on data with exact external hashes. The oversized
unpublished first draft is preserved on data with a verified352-file migration
journal; no frozen manifest or scientific result is rewritten. Fine-friction
qualification, final rankings/figures and complete matched delivery remain open.
