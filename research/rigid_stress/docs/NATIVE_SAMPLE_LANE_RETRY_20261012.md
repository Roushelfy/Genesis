# Bounded sample-lane retry

The selected CUDA path replaces the first unresolved serial retry with an
eight-sample-lane cooperative reduction. A full32-thread subgroup stays active;
lanes0–7 each visit their strided samples once and the remaining lanes add
zero. The new reduction order follows normal32-lane and cold32-lane fitting.
Successful constraints remain unchanged, evaluation counts include every
attempt, and disk/fan refinement retains serial fallback and continuation.
CPU engine behavior, source priors, physical footprint, Q10/s8 sampling,
friction and final wrench/residual/stress budgets stay unchanged.

The matched B2048/seed623102 pair runs sequentially on one RTX PRO6000
Blackwell Server Edition GPU,97887MiB, driver595.71.05, container1_26,
Genesis1.4.3/Quadrants1.3.3. Low480 affine P2 tetrahedra/810 P2 nodes/2430 DOFs,
FP64, max output, mesh64/GJK patch-on and dt0.01/substep1 are fixed. Each route
has600 warmup and three1280-step changing trajectories; timed metrics,
observations, parameter updates, resets, history and every fallback remain
included.

| First unresolved fallback | Effective env-step/s |
|---|---:|
| Serial |59395.526156|
| Eight sample lanes |62061.434613|

The ratio is1.044883994x. Every repeat has zero invalid steps/overflows,
2323 selected-environment resets and3142 successful/zero failed lift cycles.
Rigid positions agree within1e-12m and every per-environment rigid grasp,
contact and descent counter agrees. This is a separate paired experiment;
its gain cannot be added arithmetically to the cold-retry experiment's gain.

The candidate passes23 CUDA cases in436.01s and24 independent CPU FP64
contact/nodal/full-field comparisons. Maximum normalized wrench error is
7.386596734e-10, complete residual budget fraction0.0453641732 and global
peak difference8.408358553e-6Pa. The production form removes its experimental
switch and passes another23 CUDA cases in434.84s. Copied production files
match that qualified source closure exactly.

A separate complete1280-step instrumented trajectory captures160 GPU fields.
Independent CPU assembly, gauge/factor, complete equilibrium, tensor/von Mises
and global maximum checks pass: maximum residual budget fraction0.0054419277
and peak difference2.682441846e-6Pa. The CPU oracle receives captured balanced
RHS; fixture oracles separately check contact fitting and nodal scatter.
Numerical failures are zero and all hard frames remain included.

At ticks468/959/1068, eight-lane retry takes81.253/80.217/81.399ms versus
approximately737–740ms for the prior diagnostic's first serial retry. Whole
instrumented steps are343.598/354.222/356.052ms. Fan refinement still takes
approximately107ms and refined serial fitting76–89ms. Fences, four pre-attempt
diagnostic exports and different allocations perturb latency; the paired
live experiment establishes the throughput gain. These remaining tails are
actionable, so this result is not a plateau.

The v1 ablation archive has248 entries/113 unique blobs/16250002 stored bytes;
manifest SHA256:
`91f940261109ab0e78197bce1c00efe91ce001698db97830bda7e410d049ecdf`.
The production-form v2 archive has189 entries/121 blobs/20811652 bytes;
manifest SHA256:
`a7be808b65728d287f6a9eace770120d3e8c27ade8ad28690a50156a0353c3b6`.
Allocated CPU publishers verify every original/stored/decompressed hash,
source closure, failure count and acceptance budget. Reproducible patches,
launchers, reports, trajectories and field captures are under
`evidence/20261011-nonconvex/native-sample-lane-retry-*`.

Compact sample storage, combined-source qualification, broader final scopes/
seeds, refinement scheduling, sample geometry reuse and a measured throughput
plateau remain under the active goal. No physical convergence or RL training
throughput is claimed.
