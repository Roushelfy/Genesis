# Practical native rigid stress throughput, 2026-10-17

**Working report; final wide-batch curves and operating-point selection remain
open.** The [completion audit](PRACTICAL_THROUGHPUT_COMPLETION_AUDIT_20261017.md)
retains every remaining delivery requirement.

The robust full-shell workload now has a closed50.845% code improvement at
B8192 with its original history4 configuration. A separately qualified history0
setting gives another34.494% in a same-source/same-GPU comparison. The complete
history0 four-scope B8192 run measures109293.43 live and108062.87 policy valid
env-step/s. B16384 now closes116912.11 live and116695.21 policy env-step/s;
final B32768 operating measurements are still running.

## Workload and comparison scope

All comparisons retain the480 P2-tet/2430-DOF full hollow egg, original material
and fixed mass, native Franka controller and eight-condition bank, nominal6mm
radius schedule, quadrature10, ports8, intrinsic body_source_prior law, mesh64
collider, MPR/patch-off, dt0.01 and one substep. Production arithmetic remains
Quadrants FP64 for stress. The policy scope adds the frozen26/128/128/7 inference network
and rollout work; it measures inference rather than RL training.

Normal rates include their stated control/reset/policy work,900 warmup steps
and three2400-step repeats. Memory/work/profile diagnostics are separate.
All accepted points require actual program/wrapper exits, terminal successful
Slurm accounting,899 source hashes and a clean source inventory. Independent
metadata audits reread original rates/options and reject observer/sampler rows.

## Measured gains

| Paired measurement | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| B8192 original/new code, fixed history4, same-GPU A/B/B/A |52719.54|79524.92|1.50845|
| B8192 same new code, history4/history0, same GPU |78768.50|105939.20|1.34494|
| B32 cache-only/masked-recovery code, same-GPU A/B/B/A |2256.89|2317.37|1.02680|

Values are median valid env-step/s. Each A/B/B/A side pools six raw repeats;
the configuration comparison has three per setting. These are distinct
experiments; the two large ratios are not a single combined paired measurement.
The earlier unmatched low B32 result did not recur as a paired slowdown.
The [experiment ledger](PRACTICAL_THROUGHPUT_EXPERIMENTS_20261017.md) retains
the exact source identities, raw repeats, failed cases and rejected ablations.

The [closed history0 B8192 operating point](PRACTICAL_THROUGHPUT_HISTORY0_OPERATING_POINT_20261017.json)
has the following four scopes on one allocated GPU:

| Scope | Valid env-step/s | Batch latency, ms |
| --- | ---: | ---: |
| Rigid |773385.49|10.5924|
| Rigid + stress |109293.43|74.9542|
| Policy + rigid |730858.57|11.2087|
| Policy + rigid + stress |108062.87|75.8077|

The median incremental stress costs are64.3618ms/batch live and64.5990ms/batch
with policy. The aggregate rate is not an individual environment's frame rate.
The [closed B16384 measurement](PRACTICAL_THROUGHPUT_HISTORY0_WIDE_20261017.json)
has907392.82 rigid,116912.11 live,899990.35 policy-rigid and116695.21 policy
env-step/s. Doubling B8192 adds6.97% live and7.99% policy aggregate throughput;
batch latency and memory increase. These are comparable allocations rather
than a same-GPU paired batch-width experiment.
The [independent raw metadata audit](PRACTICAL_THROUGHPUT_HISTORY0_RAW_RATE_AUDIT_20261017.json)
rereads all eight closed B8192/B16384 scope records and passes their raw rates,
unchanged stress/controller/policy settings and per-environment failure counts.
The frozen original has a throughput shoulder from B8192 through B24576;
final-source shoulder/refinement and curves remain open.

## What changed and what still costs time

The geometry cache stores current fan samples as two barycentric coordinates
and reuses their geometry/P2 shapes across mapping, fit and scatter passes.
Contact changes update the samples while preserving the physical footprints. Its isolated
B2048 speed change was effectively flat; it is retained for its demonstrated
usable-capacity benefit at unchanged initial workspace2: original code OOMs at
B40960, whereas the cache completes it without contact truncation.

History misses now use the existing exact boundary-force/centrifugal response
instead of the full dense inverse. Accepted history predictions remain intact;
original complete residual, inverse correction, factor fallback and global raw
peak/full-output rules remain. In moving B8192 profiles, elastic recovery falls
from58.768 to5.281ms/step and CPU step annotation from159.717 to108.024ms.

History prediction/maintenance still costs26.050ms at only24.519% measured hits.
Disabling it in the separate configuration removes maintenance and its
preliminary prediction check; accepted output still passes full equilibrium
and gauge checks at the original tolerances every step. The history0 profile
has CPU step time77.541ms, traction fitting19.567ms, surface sampling18.435ms,
load assembly8.129ms, elastic recovery5.324ms and residual checks2.665ms.
These overlapping profile categories are not additive wall-time components.
The retained traction/sample work is now the largest measured remaining cost.

CUDA API spans overlap GPU execution:99.812% of memcpy API-span union overlaps
kernels in the new history4 trace. API duration is not independent CPU overhead.
Cold setup in the first B8192 diagnostic is110.03s before policy creation and
warmup; its model build/JIT subsets are0.311s topology,1.740s operators,
14.114s factor,0.611s inverse and0.329s surface inverse. Warmup also includes
runtime compilation and physical steps; these costs are outside headline timing.

## Memory, capacity and settings

On RTX PRO6000 Blackwell with101975851008 CUDA-visible device bytes, separate
100ms sampled process peaks are19.0449/21.4180GiB live/policy at history0 B8192,
and40.5273/40.5293GiB at B16384. The independent B32768 memory pass closes
78.7461/78.8984GiB. These include setup/growth and are sampled
lower bounds. Separate-process peaks include allocator/sampling effects and
do not isolate the policy's memory cost. Its intrusive one-repeat rates remain
excluded from headline timings; final B32768 repeated rates remain open.

With initial workspace2 growing to all9 actual contacts, history4 passes B40960
and OOMs at B43008; history0 passes B47104 and OOMs at B49152. The history0
successful capacity pass observes98098479104 peak process bytes and113049600
valid transitions. These one-pass capacity timings are intrusive and are
excluded from repeated headline rates. The brackets describe these measured
settings, not an exact universal limit.

History0 is the qualified fast-setting candidate for this changing-contact
low mesh. Final batch selection awaits the B32768 repeated measurement.
History4 remains available for workloads with useful reuse; it preserves the
same output semantics. Initial workspace2 is a separately labeled memory tuning,
with full growth rather than a cap on physical contacts. Full output remains
available for field observations; max is the measured throughput setting.

## Numerical/application scope and reproduction

CPU/GPU independent FP64 checks retain complete equilibrium including all six
gauge rows, full tensor/raw global peak and correction/fallback behavior.
B16384 lifecycle checks cover checkpoint/reset isolation, the following live
step, a complete controller period and frozen complete release/recontact.
History0 and history4 saved moving fields pass all1370880 tensor components at
the original1e-4/1e-3Pa comparison bounds, with maximum difference0.000156370Pa.
Both original force-choice families retain their complete1200-step episodes.

The retained0.2/4N model/radius ensemble has7/8 above-band lift comparisons and
1/8 whole-episode comparisons. Load reconstruction bias and reference uncertainty
remain; solver consistency does not calibrate physical damage. No force choices,
reporting windows or tolerances were retuned. Full RL training, experimental
fracture calibration and a universal new convergence campaign remain separate.

The [running guide](RUNNING_PRACTICAL_THROUGHPUT.md),
[exact source/input recipes](../evidence/20261017-throughput/source-recipes-v1/README.md)
and [baseline figure packet](../evidence/20261017-throughput/baseline-scaling-v2/README.md)
provide commands, exact committed-source reconstruction and inspected exports.
Final-source curve packets and the terminal completion decision remain open.
`scripts/plot_practical_comparison.py` prepares a direct baseline/history4,
selected/history4 and selected/history0 comparison from the two independently
audited final summaries. Its preview job725284 waits for summary/preview725282
and the remaining normal B32768 producers; no final plot is claimed yet.
