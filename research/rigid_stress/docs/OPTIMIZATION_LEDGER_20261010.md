# Native low-mesh optimization ledger

This is an active development ledger, not a completion or maximum-throughput
claim. The reference is the full hollow L1 mesh: 810 P2 nodes, 2,430 DOFs,
480 tetrahedra, 162 exterior P2 nodes, 80 exterior faces and fixed positive
Q10x10 quadrature. All production physical arithmetic uses Quadrants FP64.
The finite pad load and actual combined contact friction change are described
in [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md).

## Native current-footprint reuse checkpoint, 2026-10-11 00:30 UTC

The matched B=32768 contact-moment pair completes on GPU
`GPU-e869c013-9bdf-a217-9938-536a954a188d`: 286557.11 -> 296888.30
valid env-step/s, +3.61%, repeat standard deviations 390.93/159.50,
zero invalid transitions. These are three 2400-step actual changing-contact
repeats after 900 warmup steps. The bounded research cache is outside the
native inventory in that trial; add 227540992 bytes at B=32768 (217 bytes
per 32-per-environment task capacity). The native integration includes all
its fields in the inventory. Small B=1024 still has no clear benefit.

The native combined candidate passes 47 GPU cases, 27 CPU cases (20 CUDA
skips), and 432 independent CPU FP64 full-field oracle samples. The three
captured apex fixtures pass both moment reuse and forced complete overflow.
Illegal inputs retain diagnostic status; constrained and refined contacts
use complete integration. An empty scalar-bool ndarray allocation fails
when reuse is disabled; an unused one-byte placeholder fixes this backend
compatibility issue. The rejected import and allocation attempts, cancelled
partial test run, corrected source and successful tests remain published.
Native ordinary interaction/ablation timing is still running on two seeds.

Deriving Gram from integrated P2 moments is rejected by matched snapshot
cost. At B=1024, explicit Gram plus load moments takes 1.55215 ms versus
1.69080 ms for derived Gram; B=32768 takes 23.19926 versus 27.55953 ms.
Both preserve nodal loads, full residual and peak within the existing
budgets. The reduced sample arithmetic does not offset the added device
construction cost. It introduces no production code.

The full-rollout interval profiler is validated over 1200 unmodified
changing-contact policy steps, including partial resets, with a full
CPU/CUDA trace. CUDA intervals include host-induced gaps; nested recovery,
substep and scene intervals must not be added. These intrusive diagnostics
are not ordinary throughput. Larger profiles remain required.

An independently measured controller-input fusion computes the same
phase, arm target, FP32-default scalar grip/force limit and radius schedule
in one Quadrants kernel, retaining the existing public setters and reset
path. With resident arm indices, matched B=1024 policy timing is
72813.03 -> 74690.84 env-step/s, +2.58%, repeat standard deviations
205.49/109.32 and zero invalid steps. An earlier CPU-index variant gives
only +0.89%. The independent CPU NumPy FP64 input oracle checks 1211
environments, 22 boundary/reset/delay ticks and no/FP32/FP64 residuals:
maximum arm-target error 4.69e-15 rad, radius error 1.73e-18 m, and exact
grip/force-limit scalar values. A full-field trajectory oracle also passes
144 snapshots. A second seed, large-batch benefit and native composition
are still being evaluated; final source selection remains open.

The small B=1024/2048 two-seed live/policy completed-episode audits have
zero numerical failures and no failed completed episodes under the stated
lift/hold/release criteria. New outputs provide trajectory-only counts by
subtracting warmup, plus per-environment packed-column, admitted-face,
affine contraction and sample-visit counts. Overflow and nonlinear pressure
work are separately identified; these are work counters, not individual
GPU latency. Larger audits are running.

The publication contains 885 terminal artifacts. All 885 publication and
885 original hashes pass; manifest SHA256:
`07c1ff8189f1d174891787a697e3889171335cf85614afdb22400f1b567ab6d4`.
The Chrome trace is preserved as a 28 MiB lossless gzip. The first unpublished
checkpoint used a 582 MiB JSON summary and GitHub rejected that push. Trace
events are dictionary records, so numeric-array summarization does not reduce
them; the archive helper now compresses trace JSON directly. Every publication
and original hash was checked again after this packaging correction.
The ongoing ordinary ablations, large audits and later controller results
are excluded from this publication checkpoint.

## Validated scheduling controls, 23:57 UTC

The native integrated-wrench, packed-block, immutable-options and coalesced
complete-residual candidate passes all 41 GPU cases, 27 CPU cases (14 CUDA
skips), and 480 independent CPU FP64 full-field trajectory samples, including
forced overflow and four substeps. Its source is preserved in
`native-wrench-block-residual-v3.patch` and the separately preserved balance
module. Complete residuals include every gauge row; all final budgets remain
unchanged. Ordinary native interaction timing remains required.

The contact-moment research path passes 192 normal and 96 forced-overflow,
four-substep full-field oracle samples. At B=32768, snapshot fit plus scatter
changes 27.32849 -> 23.34712 ms. The matched B=1024 actual rollout changes
73319.00 -> 73397.59 env-step/s, +0.11%, with repeat standard deviations
358.13/930.95 and no invalid transitions. This is insufficient evidence for
a small-batch default. The large matched pair is still running. Native
integration is being tested with the three captured apex fixtures, illegal
inputs, constrained pressures, partial resets and state restoration. The
first integration attempt fails at module import because a Quadrants kernel
argument had a default; its Python-wrapper correction is being validated.
No numerical acceptance is inferred from that rejected import attempt.

The packed-traversal control isolates launch size from loop order. At
B=32768, original block 512 takes 22.66150 ms, compared with
22.59740/22.61802/22.65073 ms for environment tiles 32/64/128 at the same
block size. The best additional saving is 0.06410 ms (0.28% of the stage),
roughly 0.06% of measured whole-rollout time at this checkpoint. Retaining
the original traversal avoids padding and added tuning for this small
snapshot difference. The earlier 3.9% apparent gain mostly came from block
512; that block choice remains a separately validated candidate.

The native tiled-balance B=49152 policy inference rollout completes both
seeds with 900 warmup steps and three 2400-step repeats: 292279.47 and
291954.53 valid env-step/s (5.94644 and 5.93983 batch-step/s), repeat
standard deviations 832.04 and 831.36, zero invalid transitions. Native
buffers occupy 10875304213 bytes and the saved reset state 63700992 bytes.
These use `e6094ff9-native-balance-t8-v2`, not the newer scheduling candidate;
they do not establish a final-source throughput optimum or RL training rate.

The single-writer publication contains 824 terminal artifacts; all 824
publication hashes and all 824 original hashes pass. Manifest SHA256:
`76c595e3bf60be28ac8f0f94f4c25483d272cc6c03ad2c40f186e346867dc1ad`.
Active jobs are excluded. Historical research overrides should run against
their recorded source revision or preserved generated source; final native
benchmark flags provide the production ablations after integration.

## Research scheduling and contact-moment checkpoint, 23:21 UTC

The matched B=32768 seed-623001 integrated-wrench trial completes on GPU
`GPU-7aa59c7f-652c-7945-af70-929e66498148`: baseline 286060.20 env-step/s
(repeat standard deviation 242.18), reuse 288826.55 (556.44), +0.97%,
zero invalid transitions. The B=1024 pair remains -0.33%. Mid-batch
snapshot balance at B=2048/4096/8192/16384 changes
0.21095/0.40331/0.81676/1.62629 to
0.15269/0.24912/0.43260/0.80238 ms; full residual and original peak
budgets pass. These are scheduling diagnostics, not mid-batch rollout rates.

The full-residual layout trial completes on GPU
`GPU-eec6c32a-fe0b-5322-8219-698fede93860`: 284528.87 -> 285770.41
env-step/s, +0.44%, standard deviations 379.25/459.42 and zero invalid
transitions. It also passes 192 independent full-field oracle samples;
snapshot residual vectors are bit-identical. Native interaction measurements
are still required for both scheduling choices.

The pressure/scatter repeated-footprint candidate is different from caching
only nonlinear iterations. For an affine pressure accepted as positive on
the entire footprint, integrate `sum(weight * P2_shape * dual_coordinate)`
alongside the pressure Gram matrix, then contract these nodal moments with
the fitted coefficient. Affine geometry and P2 partition of unity preserve
the wrench; actual final wrench checks remain mandatory. Nonpositive,
constrained, locally refined and overflowing cases retain their complete
original integration. No radius, point, sample or acceptance budget changes.
The bounded research cache adds 217 bytes/task capacity and refreshes every
recovery, without contact-history assumptions. B=1024 snapshot combined
fit/scatter changes 1.76455 -> 1.55635 ms; nodal differences are below
7.91e-8 of the unchanged load budget, and full residual/peak checks pass.
Its independent trajectory oracle and large timing are running. Actual
end-to-end benefit and production selection remain open.

The native wrench/block/immutable-options integration passes 480 independent
CPU FP64 full-field samples, including forced complete overflow and four
substeps. Forty GPU cases pass; the remaining new test initially omitted
the required `cached_bounds` template argument. Its corrected invocation
passes both pressure scheduling cases. Full CPU testing is running; no
production numerical failure was inferred from that test invocation error.
Build-time options are copied per link and mutation now requires rebuilding,
preventing stale material/mass/shared-operator or workspace configurations.

The B=32768 normal every-step wrench-reuse audit covers 108134400
environment observations with zero invalid steps. It observes five local
integration retries among 327387629 nonzero contacts, a maximum 90 counted
Newton evaluations, and no correction, factor or scatter overflow. Complete
residual uses at most 0.011539 of budget; force and radius-normalized moment
errors remain below 7.77e-10 and 6.99e-10. The forced B=1024 full-field policy
audit covers 3379200 observations, executes 3296 complete scatter fallbacks
and has zero invalid steps. Fit counters count Newton evaluations; line
search work is not separately counted, so their mean is not a measured
fraction of stage time. All per-environment tails remain preserved.

The completed-episode audit passes all 114 completed episodes in each
B=32 live/policy case for both seeds. Restart-aborted and unfinished episodes
are reported separately, and weak-grip height drops remain recorded. These
are stated scripted lift/hold/release criteria, not fracture or measured
tangential-slip validation. Larger final-source episode audits remain open.

The published manifest contains 723 artifacts; all 723 publication and
723 original hashes pass. SHA256:
`72855fdecc61ea4fdd7323ecaa13c73ba182428b8ec0fa455a452eb35f4dd602`.
Later ongoing trials above have their data-side originals and will be
published when terminal. Performance plateau and final batch choice remain
unresolved while the applicable contact-moment candidate is evaluated.

## Native balance checkpoint, 23:03 UTC

The native 8-node/32-environment reduction and shared mass-mode projection
pass 27 CPU cases (14 CUDA-only skips), all 41 GPU cases, and 480 independent
CPU FP64 full-field trajectory snapshots. Two additional odd-batch tests at
B=2051/32771 exercise padding and repeated shared-storage reuse, all six
gauge rows, centrifugal loads and isolated partial load resets. A subsequent
test-name/explicit-array-shape cleanup passes those two GPU cases again.
The immutable projection costs 116640 bytes per shared L1 operator.

Matched ordinary production live measurements use 900 warmup steps and
three 2400-step repeats. At B=32768, seed 623001, on GPU
`GPU-40b7c172-fb9d-63ea-58c7-0677967a8583`, the original balance gives
284089.82 env-step/s (repeat standard deviation 361.36); native tiled
balance gives 288803.04 (91.49), a 1.66% gain with zero invalid transitions.
Matched B=1024 pairs on GPU `GPU-56ea077b-99d7-8845-1f28-8cfa2174feb6`
give 73786.42 -> 73984.84 for seed 510000 and 72835.28 -> 74144.69 for
seed 623001, gains of 0.27% and 1.80%. This implementation is retained.

Separate allocations complete all four ordinary scopes at B=1024/2048
for both seeds, again three 2400-step repeats after 900 warmup, with zero
invalid transitions. These source `e6094ff9-native-balance-t8-v2` rates
are a development checkpoint; candidate combinations and final larger-batch
selection remain required.

| B | Seed | Rigid | Recovery snapshot | Live | Policy inference rollout |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 510000 | 102209.75 | 283453.79 | 72355.73 | 72304.46 |
| 1024 | 623001 | 101728.47 | 293048.62 | 74125.26 | 73022.38 |
| 2048 | 510000 | 169433.23 | 357404.03 | 112824.23 | 111986.67 |
| 2048 | 623001 | 172834.38 | 370078.78 | 114854.78 | 114380.60 |

The eight untimed every-step live/policy audits at these batches cover
40550400 environment observations with zero invalid steps, overflow calls,
local integration retries, corrections or factor fallbacks. All environments
remain included. Mean pressure fitting uses 0.0209--0.0216 evaluations per
nonzero contact; per-contact maxima are 5--9. Actual combined friction ranges
approximately 0.42--0.84. The existing hold-step proxy remains explicitly
separate from completed-episode outcomes, for which a new audit is running.

The new contact-wrench reuse candidate saves about 1.68 ms of B=32768
snapshot balance and passes 192 normal plus 96 forced-overflow full-field
oracle samples, as well as large every-step audits. Its B=1024 actual pair
gives 73570.12 -> 73330.79 env-step/s, about -0.33%; a universal default
would not be justified. Its large actual comparison remains in progress.
The complete residual layout candidate preserves residual vectors exactly
but changes B=1024 timing from 0.35050 to 0.38672 ms; at B=32768 it changes
10.90294 to 10.30419 ms. Actual large rollout selection remains in progress.

On native tiled balance, a matched B=32768 seed-623001 research launch
ablation gives 288973.19 -> 290872.91 env-step/s for packed block 0/512
(standard deviations 26.73/1051.91), a 0.66% gain. Together with the earlier
seed-510000 +0.82% result this warrants evaluating a large-batch production
path, while retaining the original small-batch scheduling.

Conditional initialization of unused contact diagnostics saves only
0.00588/0.19084 ms at B=1024/32768. It leaves stale scratch diagnostics
unless every consumer adds validity masking. Its large endpoint ceiling is
about 0.17%, so it is rejected on cost and diagnostic-maintenance evidence.
Caching only nonlinear iterations would add work on all changing contacts
to save rare iterations. Their mean counter is about 0.021/contact, with
line-search work not separately counted; this is an amortization concern,
not a measured 2.1% stage-time bound. Reusing footprint moments across
pressure and scatter is being evaluated separately above. Final plateau,
max/full costs, completed episodes and larger four-scope runs remain open.

This checkpoint archives 660 completed artifacts, including all sixteen
ordinary small-batch scope measurements, the matched native large pair,
eight full-environment audits and native max/full stage profiles. All 660
publication and 660 original hashes pass streaming verification. Manifest
SHA256: `03db7c59b2d92080d93054a73e9b0d87c897c7f7e683049f9a56569b69b2c438`.

## Production face scheduling and relief checkpoint, 22:09 UTC

The production B=32768, seed-623001 comparison now completes on one GPU,
UUID `GPU-e3ae17a2-8c3a-b59e-e80d-ac03599051ab`. With 900 warmup steps and
three 2400-step repeats, contact scheduling gives 266958.92 env-step/s
(repeat standard deviation 107.70); bounded face scheduling gives 278477.20
(496.07), a 4.31% gain. Both have zero invalid transitions. Native storage
changes from 7213379957 to 7272100213 bytes. This is production evidence,
separate from the earlier generated research implementation's 4.18% result.

On the selected face/reset/1e-9 budget, packed application block size 512
gives 279638.54 -> 281943.87 env-step/s at B=32768, seed 510000. The repeat
standard deviations are 833.08 and 49.66; all transitions are valid. This
0.82% result is below the prior source's 1.68% result. Selection remains
pending an interaction check with the new inertia-relief reduction; small
batch microtiming was worse, so no universal block-size default is inferred.

The shared immutable relief projection alone reduces B=1024 / 32768
snapshot balance from 0.30385 / 5.12686 to 0.27863 / 4.45316 ms. Adding an
8-node, 32-environment CUDA reduction changes 0.30386 / 5.12703 to
0.12052 / 3.20779 ms. It keeps every nodal load and six wrench components.
Complete residual and peak checks pass, as do 192 independent full-field
FP64 snapshots. Its ordinary same-device B=1024 seed-623001 prototype pair
gives 73109.00 -> 74040.04 env-step/s, a 1.27% gain (standard deviations
64.25 / 226.75), with zero invalid transitions. Native integration is being
validated and measured separately before retention.

The first shared reduction lacked a barrier before grid-stride storage
reuse. Small batch tests passed, but B=32768 failed RHS consistency with
maximum difference 2.85e-5 N. That version is rejected and preserved. The
corrected version adds the barrier and passes the large comparison. At
B=32768, node tiles 4 / 8 / 16 cost 3.03884 / 3.20779 / 3.21453 ms. Only
tile 8 currently has actual rollout evidence; these isolated times do not
establish an end-to-end tile choice. The native regression includes odd
B=2051 and 32771 to exercise padding and repeated shared-storage reuse.

All eight matched ordinary max/full B=1024 measurements complete on GPU
`GPU-56ea077b-99d7-8845-1f28-8cfa2174feb6`, source `e6094ff9`. Each uses
900 warmup steps and three 2400-step repeats with no invalid transitions.
Full output costs 107520 bytes/environment, 110100480 bytes at B=1024.
Its live/policy rates are 72895.53 / 72750.94 for seed 510000 and
72622.83 / 72220.09 for seed 623001. Matched maximum-only rates are
73909.92 / 73521.04 and 73250.64 / 72625.25. The four paired costs range
from 0.56% to 1.37%. Final optimized-source cost/profile measurements and
all-scope larger-batch selection remain open; no plateau is claimed.

This checkpoint publishes 520 completed artifacts. Streaming verification
checks every publication and data-side original SHA256; all 520 + 520
checks pass. Manifest SHA256 is
`31fc905c157be0774e80b79f7718ba2e8b7e42091886b49fb9b747ad94029727`.
Rejected frontend/shared-storage trials and the baseline metadata correction
are preserved with their original sources. The generated relief probes
reproduce the pre-integration `e6094ff9` balance body; native integration
uses its own ordinary benchmark rather than those source overrides.

## Face scheduling checkpoint, 21:35 UTC

The compact research implementation completes the same-device B=32768
seed-623001 pair with 900 warmup steps and three 2400-step trajectories.
Contact scheduling gives 278,903.62 env-step/s (repeat standard deviation
83.33). Compact face scheduling gives 290,569.24 (627.03), a 4.18% gain.
Both have zero invalid transitions. This checkpoint uses the recorded
`943ffbd0-internal-1e-9-face-compact-v5` generated-source override. The
production implementation adds explicit admission guards and is undergoing
its own matched B=32768 comparison before claiming that gain for production.

Production v2 passes all 39 GPU cases, 27 CPU cases and 672 independent
FP64 full-field snapshots. Its B=1024 same-device actual gain is 2.50%.
The local-name-only v3 additionally passes all 18 captured-contact cases
and every-step audits of 67,200 observations each for normal live and
forced-overflow full-field policy. The latter executes 2096 complete
scatter fallback calls with zero invalid observations. Normal live has
zero overflow calls. The benchmark and audit expose these counters.

Production v3 diagnostic stage times at B=1024 / 32768 are pressure
1.33801 / 10.63019 ms, face scatter 0.58365 / 15.80074 ms, packed inverse
0.86262 / 22.94848 ms, complete residual 0.34474 / 10.92872 ms, global
peak 0.24884 / 6.76030 ms and inertia relief 0.30335 / 5.11516 ms.
Exact sources and raw wall times are preserved. These separate allocations
provide diagnostic profiles, not matched retention percentages.

The unchanged-final-budget block-512 comparison is still running. A
new shared inertia-relief projection trial evaluates the repeated nodal
mass-mode/rigid-mode Gram product. Its first global-wrapper prototype
has a Quadrants frontend error and no valid numerical or timing result.
The explicit tensor-parameter version is under evaluation. Final four-scope
batch selection, matched max/full ordinary timing and final tail/memory
measurements remain required. Performance plateau is not yet established.

## Error-budget and reset checkpoint, 20:58 UTC

Internal pressure termination is selected at normalized residual 1e-9,
with the final 1e-8 force/moment, full-gauge equilibrium and independent
FP64 stress checks unchanged. The newly captured B=32768 seed-623001
apex input passes serial, warp and fused-graph regression. CPU/GPU each
pass the existing 27 cases; three additional graph cases and 480 full-field
oracle snapshots pass. See the contact report for the strict-trial failure,
actual errors and unchanged acceptance budgets.

The example now saves its fully initialized scene state and uses one
`Scene.reset(state=..., envs_idx=...)` call. Both seeds pass the 2400-step
saved/legacy physics, loads, actual friction and complete-field equivalence
comparison, with 48 partial environment resets each. The previous small
batch paired trial gains about 23%. At B=32768, the matched three-repeat
2400-step seed-510000 comparison yields 259765.35 env-step/s for individual
setters and 268994.42 for the saved state (about 3.55%); all transitions
are valid. The saved state adds 42467328 bytes at that batch. Retain this
end-to-end improvement, expose `--legacy-reset` for the ablation, and
include its state storage and reset cost in final measurements.

The compact face-task trial uses 58720264 bytes at B=32768, versus the
previous 1188036612-byte workspace. It retains all fixed/local Q10 samples
and fully recomputes the original scatter on capacity overflow. Its
snapshot scatter is 21.10169 -> 15.52726 ms, with force/peak/full-residual
checks passing, including forced overflow. On the new stopping-budget and
saved-reset baseline, B=1024 actual three-repeat live rates are
71076.75 -> 72489.54 env-step/s (about 1.99%), with no invalid transitions.
Its independent 192-snapshot full-field oracle passes. The B=32768 actual
comparison remains in flight, so production face scheduling is pending.

The independent packed-application block-512 trial on the prior 7f source
at B=32768 yields 268715.93 -> 273219.47 env-step/s (about 1.68%), all valid.
Small-batch microtiming was worse. Re-evaluate its large-batch benefit on
the selected final scheduling/reset baseline before enabling it.

The old 2c B=49152 FP64 policy failure remains preserved with no valid
throughput row. A separate full-size replay and the focused replay do not
reproduce it; neither establishes repair of that particular uncaptured
input. New-source long policy measurements remain required. The owned
old failed-sweep allocation was cancelled after it stopped producing
results. Final acceptance and throughput-plateau work remain open.

## Validated changes

Production face scheduling checkpoint (`562597f1-native-face-v2`) passes
27 CPU cases (12 CUDA-only skips), all 39 CUDA cases, and 672 independent
full-field FP64 oracle snapshots. Those snapshots include two seeds,
four physical substeps and deliberately forced complete-scatter overflow.
The workspace bounds memory while its overflow path recomputes all original
contacts and integration samples on the device. Force/moment, complete
gauge residual and peak/tensor consistency assertions retain their budgets.

Same-device B=1024 seed-623001 live timing uses 900 warmup steps and three
2400-step varied trajectories. Original contact scheduling gives
70,491.40 env-step/s (repeat standard deviation 391.05); production face
scheduling gives 72,253.15 (126.48), a 2.50% gain. Both include all reset,
observation, contact-radius and fallback work, with zero invalid transitions
and zero scatter overflow calls. Default workspace adds 1,835,008 bytes at
B=1024. The saved initial-state reset cost is identical in both cases.
The actual production B=32768 matched pair and updated block-512 ablation
remain in progress. Exact tested v2 scatter source and tracked patch are
preserved separately from the subsequent local-variable naming cleanup.

The rebuilt 1e-9 stopping-budget baseline profiles actual seed-623001 Q10
contacts after 900 warmup steps, with 50 isolated stage repetitions.
At B=1024 / B=32768, pressure including retry costs 1.13326 / 10.12758 ms,
scatter 0.92149 / 20.78146 ms, packed inverse application
0.81120 / 22.38534 ms, complete residual 0.34856 / 10.92890 ms, and full
global peak 0.25094 / 6.58309 ms. The exact sources and raw stage evidence
are `internal-budget-baseline-profile-b*-v2`. Isolated synchronized stage
times diagnose bottlenecks; ordinary live timing determines retention.

| Candidate | Matched native microtiming at B=1024 | Same-device live development comparison | Decision |
|---|---:|---:|---|
| Exact complete exterior operator | 7.2782 → 1.5147 ms application | 41,310.81 → 54,137.61 env-step/s | Retain, e1fbf937 |
| Face load reduction before six node writes | 2.6750 → 1.1553 ms scatter | 53,820.81 → 58,316.33 env-step/s | Retain, e0b6714c |
| Shared P2 corner gradients | 1.1140 → 0.3183 ms global peak | 55,428.84 → 61,603.35 env-step/s | Retain, 4f45bcb0 |
| Shared immutable triangle bounds | 1.1825 → 0.8998 ms scatter | 61,799.85 → 62,531.25 env-step/s | Retain, 219a2558 |

These comparisons each use one full 1,200-step varied trajectory repeat after
900 warmup steps. They include changing actual contact friction, continuously
updated footprint radius, support, sliding, release and delayed environment
resets. They establish development decisions; repeated final comparisons are
still required. Numbers from different comparison allocations must not be
combined into a cumulative percentage. All environments, including failed
bilateral holds, remain in timing.

The peak change adds 460,800 bytes of immutable corner gradients. It retains
all four corners of every tetrahedron. Randomized independent CPU full-gauge
residual and global peak tests pass; cached and uncached native peaks are
exactly equal in the regression. Both optimized milestones pass all 18 GPU
numerical/lifecycle cases. Surface bounds add 3,840 immutable bytes, leave
the footprint unchanged and include all eligible samples.

## Evaluated alternatives

| Candidate | Evidence | Current decision |
|---|---|---|
| FP64 shared-memory operator tiles, 32 environments x 4 output nodes | Native 1.5472 ms; source tiles 8/16/32/64 give 1.5568/1.5499/1.5477/1.6345 ms | No gain in this B=1024 trial; retain reproducible probe |
| Parallelizing the four peak corners | Uncached 1.5211 ms, cached 0.3831 ms, versus cached four-corner element scan 0.3183 ms | Reject scheduling variant |
| FP32 boundary storage with FP64 arithmetic | Application 3.1095 ms, checked path 11.2833 ms; 1024/1024 fail the initial residual | Reject |
| Compensated FP32 boundary arithmetic | Application 1.0941 ms, checked path 9.2700 ms; 1024/1024 fail the initial residual | Reject after counting mandatory corrections |
| Paired FP32 boundary storage with FP64 arithmetic | All initial residuals pass, application 6.2779 ms, checked path 6.8804 ms | Reject; accurate but slower |
| Fused elastic graph, including full residual, two masked corrections, factor fallback and peak | Separate 2.6544 ms; serial 2.4969 and parallel 2.4268 ms in a snapshot | Reject serial after actual repeated measurement; parallel actual run fails |
| Fused anchor, pressure, local retry and scatter graph | 2.2753 → 2.1741 ms, nodal difference at most 5.56e-17 N | Evaluate actual end-to-end integration |

The precision trial applies complete boundary loads and arbitrary angular
velocities. Its native FP64 application/checked reference is 1.5136/2.1095 ms.
The original tolerance and six gauge rows remain mandatory; faster unchecked
application is not a valid throughput improvement. No Torch stress solve or
external numerical backend is used. CPU/SciPy computations are offline
references on allocated hardware.

The shared-memory trial reuses operator coefficients across environments but
adds synchronization and shared-memory reads. The current output layout has
contiguous environment rows and a 9.57 MB complete boundary operator, below
the measured device's 128 MB L2 cache. The negative timing is the evidence
for rejecting these tested tile sizes, not a general claim against tiling.
The installed Quadrants interface exposes no native tensor-core matrix
multiply in its tile API. An external BLAS/Torch substitution would not
satisfy this feature's production numerical contract.

The actual serial elastic graph comparison uses three 1,800-step repeats
after 900 warmup steps at B=1024 on the same allocated GPU. Native repeat
rates are 57,312.53 / 57,444.79 / 57,464.81 env-step/s; serial fused rates
are 57,746.87 / 57,316.42 / 57,402.74. The weighted mean gain is about
0.14%, within repeat variation. It does not justify a production graph
refactor. The parallel graph passes the isolated snapshot comparison but
aborts actual warmup with `CUDA_ERROR_ILLEGAL_ADDRESS`, surfaced at the
acceptance launch and stream synchronization. Its full failed log is kept;
it has no valid rollout rate and is not enabled in the feature.

## Independent actual-contact checks

`native_trajectory_oracle.py` runs 32 real Panda environments for 1,800 steps
and rotates four sampled environment rows every 50 steps. It independently
reconstructs pad tractions and nodal forces, checks raw rigid world-to-local
contact force/normal/position transforms and compares the full equilibrium
including gauge rows and the complete global peak with CPU FP64.

Seeds 510000 and 623001 each pass 144 snapshots. Actual sampled coefficients
span [0.424783,0.808543] and [0.437376,0.828128]. Maximum full-residual budget
fractions are 0.005542 and 0.005826; maximum peak errors are 5.50e-6 and
1.49e-6 Pa. These are same-mesh checks, not physical mesh convergence.

The untimed validity probe now retains per-environment contact counts,
initial-plus-retry fit evaluations, local retries, correction/fallback
counts, complete-residual budget ratios, wrench preservation errors and
bilateral lift checks across resets. Its initial 8-environment 800-step
smoke audit passes all 6,400 observations.

## Latest contact conditioning repair

The first B=16384 randomized live attempt fails at environment 14658,
contact 0, actual mu 0.813261572 and original radius 5.93127930 mm. Status 3
remains after local integration in source 219a2558. This run is not a valid
throughput point. The captured input's fixed 130 samples are LP-infeasible;
local 563 samples are feasible, with LP residual 1.10e-19. Independent CPU
pressure fitting converges in 14 evaluations, force error 8.45e-16 N and
moment error 3.45e-20 Nm. This distinguishes sampling inadequacy from a
native fitting failure after the sampling repair.

The converged active Hessian has determinant/trace^3 ratio 3.41e-15, below
the native 1e-14 guard. Diagonal normalization changes that numerical ratio
to 3.71e-7 without changing the pressure objective, sampled set, target or
wrench acceptance. Source b08e866a scales dual coordinates before testing
and inverting the active Hessian. Both original and new apex fixtures pass
serial and CUDA warp recovery: four GPU tests, including independent nodal
forces, friction, full-gauge equilibrium and full-domain peak. The new native
input uses 94 cumulative evaluations (80 fixed, 14 local), versus 160 and
failure previously. All 20 GPU numerical/lifecycle cases pass in 253.61 s.
The repaired B=16384 live trajectory completes at 208,743.68 env-step/s and
12.7407 batch steps/s in a single development repeat. Sampled whole-card
peak through setup and rollout is 20,939 MiB; native buffers occupy
3,628,907,365 bytes. The original failed run remains disclosed. Larger
trajectories and repeated selection are still being checked.
Physical infeasibility must not be inferred from numerical status 3.

The same repaired source completes B=32768 at 229,969.89 env-step/s
(about 7.0181 batch steps/s), with 7,191,804,773 native buffer bytes and
39,525 MiB sampled whole-card peak. These B=16384/32768 measurements each
use one 1,200-step repeat and the same GPU UUID; they remain development
points rather than a final throughput selection.

## Current exact load-packing candidate

An actual B=1024 Panda snapshot has 0 to 77 strictly nonzero exterior
P2 nodal force vectors per environment, mean 34.834, out of the complete
162-node space. Device-side packing of all nonzero vectors, followed by
application of their original complete-operator columns, reduces application
from 1.5463 to 0.8170 ms, including packing. Displacement and full-domain
peak are exactly equal to dense native application; complete residual is
at most 2.44e-11 N and passes the unchanged per-environment budget.
The feature implementation preserves ascending node order and refreshes
scratch lists every recovery. Full random boundary loads, changed sparse
loads and zero-load/centrifugal-only transitions are covered by regression.
Matched end-to-end repeats and larger-batch launch-layout tests are running.

Source 2c6e5ecc retains packing and compact CUDA peak geometry after the
B=1024 same-GPU three-repeat actual varied trajectory comparison (2,400
steps per repeat, 900 warmup). Dense rates are
55,782.80 / 55,724.97 / 55,699.94 env-step/s, weighted mean 55,735.88.
Packed rates are 58,066.60 / 57,892.90 / 58,053.40, weighted mean
58,004.18, a 4.07% improvement. This comparison includes packing,
continuously updated contact radius, resets, residuals and fallback dispatch.
It does not combine rates from different GPU allocations.

The same paired B=2048 comparison gives dense repeat rates
87,482.8 / 87,592.6 / 87,567.0 env-step/s, weighted mean 87,547.43;
packed rates 93,782.6 / 93,852.3 / 93,802.3, weighted mean 93,812.39.
The improvement is 7.16%; all three repeats complete without numerical
rejection. The controller, workload seed, sampling and budgets are unchanged.

Three-repeat route comparison on the same new source uses identical
2,400-step varied trajectories. Default weighted rate is 58,021.32
env-step/s; four-column history gives 37,265.43 (repeat rates
37,190.2 / 37,387.3 / 37,219.4). History is rejected again after counting
prediction, checks, rejected-load solves and basis maintenance.
The native sparse direct route gives 33,354.50 env-step/s (repeat rates
33,076.5 / 33,543.9 / 33,446.8), versus the same 58,021.32 inverse default.
The shared factor remains the validated fallback and explicit alternative,
but it is not the throughput choice on this low mesh.

The B=32768 actual snapshot has 0 to 79 nonzero exterior nodes, mean
35.169. Dense application costs 48.7374 ms; default packing plus application
costs 23.2378 ms. Displacement and complete peak are exactly equal; maximum
full-gauge residual is 3.62e-11 N and all per-environment budgets pass.
Packed thread blocks 64/128/256/512/1024 cost
23.3680/23.2222/22.8001/22.4123/22.4320 ms. At B=1024 the corresponding
times are 0.8164/0.8112/0.8156/0.8308/0.9505 ms versus default 0.8101.
The larger block trades a small large-batch benefit for worse small-batch
scheduling; end-to-end evidence is needed before changing launch selection.

The shared-memory operator trial also runs at B=32768 with randomized
complete boundary loads. Native checked application costs 48.6736 ms;
source tiles 8/16/32/64 cost 48.2142/48.0323/47.6993/49.9360 ms. All
complete residuals pass (maximum 8.65e-12 N), with equal global peaks.
The best dense tiling gain is about 2%, while exact load packing on actual
contacts removes about half the application cost. The tested dense tiles
therefore do not replace the actual-contact packed operator.

The seven-term cached corner scan passes all 20 GPU cases. CPU produces
rounding differences of about 1e-10 Pa in six cached/uncached strict equality
checks, while independent oracle budgets pass. CPU therefore keeps its
original evaluation order rather than weakening that check. The combined
current source passes all 18 CPU numerical cases in 110.87 s.
All 20 GPU numerical/lifecycle cases pass in 256.55 s. Two 2,400-step
actual Panda runs, seeds 510000/623001, pass 192 independent CPU FP64
snapshots each. Maximum full-gauge budget fractions are 0.005848/0.006117,
and maximum global peak errors are 6.30e-6/2.14e-6 Pa. These runs include
unchanged complete operator loads, actual mu variation and independent
per-environment resets; no mesh convergence is inferred.

## Further large-batch cost probes

Exact device failure compaction passes unchanged complete residuals and
displacement equality for empty, isolated, 32-environment and all-environment
displacement perturbations at B=1024/32768. Zero-failure correction plus
masked residual costs 0.0791 ms at B=1024 versus packed 0.0897 ms;
B=32768 gives 0.1749 versus 0.0872 ms. Packing is included. Current live
FP64 loads have no correction/factor fallback in the snapshots. The saved
large-batch tail is small relative to the full step; actual comparison is
still running before final selection. No host count read is introduced.

Environment chunks preserve displacement and complete peak exactly.
B=32768 packed application costs 22.4882 ms; chunks 1024/2048/4096/8192/
16384 give 23.0922/22.4477/22.2379/22.2133/22.3394 ms, including one
full packing and every dispatch. B=61440 costs 41.6825 ms whole, versus
43.1641/41.9419/41.5798/41.5602/41.7305 ms for those chunks. No material
gain appears; additional source and scheduling paths are not retained.

Source 2c6e5ecc B=1024 actual-snapshot stage wall times are: begin 0.0168,
association 0.0986, anchor 0.1244, pressure 1.1595, face scatter 0.9103,
inertia relief 0.3052, RHS reduction 0.0413, packed solve 0.8106,
complete residual 0.3474, complete peak 0.2500 and acceptance 0.0273 ms.
This is diagnostic sequential stage timing, not additive rollout throughput;
full tail dispatch and ordinary rigid/controller costs remain in live rates.

Source 2c6e5ecc, with additional declared material input coverage, passes
21 GPU cases in 346.32 s and 19 CPU cases in 123.58 s. The second assembly
input is E=3.2e9 Pa, nu=0.22 and density=1100 kg/m^3, independently compared
against CPU FP64 stiffness, consistent mass, mass modes and gauge Gram.

The actual failure-compaction pair at B=32768, 2,400 steps after 900 warmup,
gives 265,559.74 versus 265,915.49 env-step/s: about 0.13%. Together with
the small measured tail and slower B=1024 microcase, this does not justify
an additional default numerical path.

The B=61440 randomized development trajectory completes at 292,812.86
env-step/s, 4.76583 batch steps/s, with 72,093 MiB sampled whole-card peak.
B=49152 on the same allocation gives 289,520.52, a 1.14% lower throughput
with 58,139 MiB sampled peak. These single repeats suggest a platform;
two-seed, three-repeat long trajectories are running before final selection.

A separate 50-step intrusive trace has 453.214 ms kernel interval union
within 1094.074 ms of CPU step ranges for live, and 461.011/1113.492 ms
for policy. CUDA stream synchronization counts are 3334/3434; copy/memset
counts are 5584/5834. GPU range annotations mirror each CPU range on both
streams and must not be added to the CPU wall ranges. The parser now
counts the 50 original CPU annotations once. These are trace measurements,
not unprofiled utilization or training rates.

The full serial association-to-acceptance graph prototype passes three
actual B=1024 2,400-step repeats. Separate mean is 58,487.56 env-step/s
(58,354.27/58,657.03/58,452.18); fused is 59,066.35
(58,719.79/59,188.61/59,293.82), about 0.99% faster. A static typed engine
implementation is being validated in an isolated worktree; ongoing primary
measurements retain the 2c6e5ecc numerical source. Its API-preserving
wrapper refactor passes 21 GPU and 19 CPU cases, with actual paired rollout
and independent trajectory oracle checks still in progress.

Policy FP32 is evaluated separately; every stress calculation remains
Quadrants FP64. B=1024 three-repeat rates are 56,174.90 FP64 and 56,646.91
FP32 (+0.84%). B=32768 one-repeat rates are 258,626.69 and 260,512.84
(+0.73%). The same seeded network's warmup output is independently checked
against an FP64 policy copy; maximum scaled control error is 2.78e-11 rad,
below the declared 1e-9 rad policy budget. Casting observations and writing
actions are included. Further selection and lifecycle evidence remain open.

## Still required

Repair and regress any newly found normal contacts before selecting larger
batches. Evaluate graph integration and remaining measured candidates,
rebuild matching end-to-end baselines if the load integration changes, then
repeat all four timing scopes at 1024, 2048 and larger valid batches. Include
multiple seeds, longer trajectories, every-step audit, per-environment tails,
grasp outcomes, sampled whole-card setup/rollout memory, and final profiling.
The active goal remains open.

## Serial pipeline and full-field implementation checkpoint

The original-decorator pipeline comparison uses one GPU UUID
GPU-46adcad5-6c6e-1013-cf33-5b956ba93598, B=1024, 900 warmup and three
2,400-step varied repeats. Original rates are
57,400.18 / 57,540.75 / 57,460.68 env-step/s; refactored separate passes
give 57,413.48 / 57,531.76 / 57,244.94. The fused serial pipeline gives
58,210.32 / 58,346.91 / 58,395.69, approximately 1.5% above the original.
This preserves the original public passes and includes every residual,
correction, fallback, contact update and reset.

The output extension from remote commit 1d1a5ca5 is implemented as
`output_mode="max"|"full"` and `RigidLink.get_stress_field()`. Full mode
stores all four corners of all 480 low-mesh tetrahedra, tensor components
xx/yy/zz/xy/xz/yz plus von Mises, in the authored link frame. The same native
scan computes both modes. Max uses zero-length compound fields because
this Quadrants version rejects zero-length scalar ndarrays; full-mode writes
and invalidation are selected by explicit compile-time templates. Initial
scalar-allocation and template/default-argument failures remain in raw logs.

The implementation passes **25 CPU cases in 84.97 s and 25 GPU cases in
372.59 s**. Full mode covers independent FP64 tensor/von-Mises recovery,
max/full reduction on one identical recovered state, rotated actual contacts,
1/4 substeps, independent partial reset, checkpoint invalidation, unbatched
views, shared operators across different output modes and invalid solves.
Separate CUDA load reductions differ by at most 4.11e-24 m in the development
same-state test; sharing the recovered displacement correctly isolates the
output comparison without changing any production acceptance budget.

Two full-field Panda trajectories (seeds 510000/623001, B=32, 2,400 steps)
pass 192 independent CPU FP64 snapshots each, including all corner tensor
components, wrench/frame/friction, full gauge residual and peak. A four-
substep B=8, 1,200-step trajectory passes another 96 snapshots. The max
observation retains scene-step temporal reduction; fields are latest-substep
snapshots and become NaN after reset/restoration or failure.

Native diagnostic counters now count invalid environment scene steps once,
even with multiple substeps, and survive ordinary environment resets. Final
benchmark records report attempted and valid transitions and per-environment
failure counts. Their maintenance belongs to timing. Final repeated full/max
cost and large-batch measurements remain required. Face-parallel scatter and
larger packed-application block scheduling are being measured before freezing
the final throughput selection.

The isolated serial/parallel elastic graphs cost 2.4969/2.4268 ms against
separate 2.6544 ms and preserve snapshot peaks/residuals. The serial variant
improves three-repeat live throughput by only 0.14%. The parallel variant
raises a CUDA illegal-address error during actual warmup; its failed log is
retained and no throughput is claimed. The full serial pipeline above is a
separate candidate. The seven-node gradient cache is retained in 2c6e5ecc.
Bounded nested Newton/line loops reduce B=1024 pressure from 1.2794 to
1.2456 ms, but B=32768 gives only 11.6812 to 11.5923 ms. Nodal differences
stay below 8.33e-17 N; the marginal large-batch cost reduction does not justify
another default pressure scheduling path.

The current evidence manifest contains 252 completed artifacts, 33,289,689
publication bytes. Every artifact and original runtime SHA256 was checked.
Long numerical arrays remain complete in their explicitly recorded data-side
originals; compact publications preserve source/configuration and summarize
those arrays. Final large runs and pipeline selection are still pending and
are not included in this checkpoint.

## Reset and face-task checkpoint, 20:27 UTC

Source 7f5a63e5 is pushed with the validated serial pipeline and full output.
The evidence manifest now contains 307 completed artifacts, retaining earlier
provenance and adding the 25 CPU/25 GPU tests, three full-field CPU FP64
trajectory oracles, all eight B=1024/2048 every-step audits and research trials.
The latest audits contain zero numerical failures; their bilateral lift/hold
step proxy succeeds for approximately 93.68% of checked steps. Every environment
also has transient failed hold checks, which remain disclosed; this is not an
episode-level grasp certificate.

Saving the fully initialized public Scene state once and calling
`Scene.reset(state=initial_state, envs_idx=ids)` avoids a reset followed by four
separate pose/friction setters. Native reset/invalidation remains active.
Matched B=1024, seed 623001, 900 warmup, three 2400-step repeats give
58,884.93 / 58,864.45 / 58,903.76 env-step/s for the individual setters and
72,534.96 / 72,606.05 / 72,320.80 for the saved state (approximately +23.1%).
An independent full-field CPU FP64 trajectory passes 192 samples. Paired
B=16 trajectories for both seeds pass position, quaternion, qpos, mapped load,
actual coefficient, validity, tensor and peak comparisons through 48
environment resets. Large-batch selection and final memory costs are pending.

The bounded face-task trial assigns one CUDA warp to each eligible contact
face instead of looping all faces in one contact warp. It keeps every fixed
or locally partitioned Q10 sample. B=32768 snapshot scatter including task
packing and contact-wrench checking changes 21.1350 to 15.4792 ms. A task
capacity overflow skips the partial face path and completely recomputes the
original scatter on device; forced-overflow testing preserves all loads.
The initial contact-slot wrench workspace costs 1,188,036,612 bytes at that
batch. A compact active-slot workspace reduces the B=1024 workspace to
1,835,016 bytes while preserving its approximately 0.547 ms scatter and
independent residual/peak checks. The complete face pipeline passes 192
full-field CPU FP64 trajectory samples. Large live comparisons, compact
large-batch validation and production selection are still running.

The old 2c6e5ecc B=49152 policy run (seed 510000, FP64 policy) rejects a normal
contact in environment 12271, slot 1: status 3 after local integration,
radius 0.00546090089 m and actual mu 0.76912725. No valid policy rate is claimed
for it. Its raw failure remains preserved. Contact capture, LP/conditioning
diagnosis and regression repair take priority over final throughput acceptance.
The matching B=32768 face comparison's original native path also rejects a
contact with seed 623001, environment 13456, slot 0, radius 0.00411071168 m and
actual mu 0.813451501. This occurs before the face candidate is run; it cannot
be attributed to that scheduling variant. The failed comparison has no rate.
The benchmark runner now records an in-progress case before launch and
propagates a nonzero child exit code; a scheduler COMPLETED status cannot
substitute for a successfully completed measurement artifact.
