# Native low-mesh optimization ledger

This is an active development ledger, not a completion or maximum-throughput
claim. The reference is the full hollow L1 mesh: 810 P2 nodes, 2,430 DOFs,
480 tetrahedra, 162 exterior P2 nodes, 80 exterior faces and fixed positive
Q10x10 quadrature. All production physical arithmetic uses Quadrants FP64.
The finite pad load and actual combined contact friction change are described
in [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md).

## Validated changes

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
