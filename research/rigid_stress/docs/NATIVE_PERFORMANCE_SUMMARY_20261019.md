# Source-bound native performance summary

Scene consolidator730473 passes14,454 checks, binds61 primary configurations
to their own complete motion/output controls and rehashes4,267 distinct files.
It also binds20 separate process-memory profiles. This is the current positive
scene table; the [original report](NATIVE_PRODUCTION_PERFORMANCE_20261019.md)
retains the superseded engine's32/259 failed motion checks and its own rates.
The [contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) retains the MPR
repair, independent regressions, diverse-input failures and measured tuning.

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

All20 profiles are separate fresh processes, with one complete1200-step
interval after900 warmup steps. Their timings are intrusive diagnostics.
NVIDIA process samples include allocations outside Torch and the logical
native-buffer inventory. B8192 sampled peaks are9340/12108/12940/21902/20270MiB
for rigid/DPR max/DPR full/SCR max/SCR full.

The new join retains actual sample counts, first/last sample times and
maximum/median observed gaps. A requested0.1s interval was not sustained
throughout setup: first process samples occur12.46–21.15s after sampler start,
and maximum observed gaps range0.504–34.513s across the profiles. For example,
B128 DPR max has367 samples and a34.513s gap; B8192 SCR full has1685 samples
and a0.504s gap. These records are sampled observations, with potentially missed
transients. They do not bound unseen peaks, establish maximum capacity, or prove
that full SCR uses less memory than max SCR. Successful full-run admission
establishes only that these exact tested jobs fit on their allocated hardware.

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
output directory. The reader performs metadata/hash/configuration association,
not a new scientific trajectory integration. Original730471 fails on an absent
rigid-only stress-options field;730473 fixes that schema handling while requiring
the complete options for every stress-enabled record. The original failure and
captured source remain intact.

## Current-source resident module replay

Primary730460 and separate memory730461 both complete all four modes; independent
replay audit730463 passes. Matching-memory audit730498 passes170 checks, rehashing
858 engine/asset files and293 captured workers per campaign. All modes retain
1200 recorded frames,24 typed device buffers and their exact partial stress
resets. Complete replay peaks and11 full snapshots per full-output run match
within the unchanged rtol1e-4/atol1e-3Pa criteria. Largest peak difference is
0.000374Pa; this numerical consistency budget is not a physical-error bound.

| B8 mode | Median env-updates/s | Minimum–maximum | Batch-update ms | Sampled process MiB | Samples | Largest observed gap s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dpr-max | 8232.96 | 8222.79–8235.81 | 0.971704 | 3514 | 561 | 14.936 |
| dpr-full | 8087.35 | 8083.33–8100.26 | 0.989199 | 3528 | 310 | 7.320 |
| scr-max | 966.09 | 963.96–966.19 | 8.280835 | 3548 | 583 | 1.344 |
| scr-full | 965.15 | 964.38–966.63 | 8.288890 | 3560 | 572 | 0.751 |

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
