# Native EPA bound consistency and valid wider live recovery

This corrects the rigid collision producer identified in
[the preceding stability checkpoint](NATIVE_STABILITY_CONTINUATION_20261012.md).
The correction applies to ordinary rigid simulation with stress enabled or
disabled. Stress recovery retains its independent auxiliary role, declared
load model, public defaults and final numerical acceptance budgets. The
active nonconvex/performance goal remains incomplete.

## Cause and local correction

At the captured B4096/623103 env3092 pose, geom14/egg24 originally produces a
normal opposed to the material normal. Patch-off and patch-on detection both
reproduce it. Replacing the approximate mesh support query with an exhaustive
vertex query reproduces it too, ruling that approximation out for this input.

Read-only EPA tracing shows the nearest returned face is12, whose face-map
status is−2: the last horizon deleted it. The next active nearest face is15.
Its squared distance lower bound is1.3712401826510844e-8, against upper bound
1.3712401826510764e-8. In distance units these are0.00011709996510038269 and
0.00011709996510038235 m, differing by3.4e-19 m. The strict squared comparison
rejects this roundoff excess and returns the previous, now deleted face.
That obsolete interior face supplies the wrong witness pair and normal.

The safe EPA comparison now uses its **existing distance tolerance**, just
as its face-admission code already does:
`lower2 > upper2 + tolerance*(2*upper+tolerance)`.
An excess within the existing distance budget retains the active nearest
face. The ordinary `(upper-lower) < tolerance` stopping check then stops on
that face. No new tolerance is introduced. MuJoCo-compatible EPA is unchanged.
The correction does not flip a normal or alter forces to satisfy stress.

An independent CPU FP64 separating-axis calculation covers all convex
face normals and edge-cross axes for the exact captured discrete geoms.
Geom14/egg24 has minimum translation direction
`[0.0004481384,-0.9984089695,-0.0563855378]`, depth0.00011709996510038921 m.
Geom15/egg24 has depth0.0000952964007290124 m and its original direction.
Corrected full Franka captured-pose detection agrees in patch-off and patch-on
modes. Geom14's new depth is0.00011709996510038269 m and its normal agrees;
the second pair is preserved. This is a convex collision-geometry reference,
not physical stress convergence or a replacement for material topology.

## Regression and unchanged stress budgets

The captured geometry is reduced to a fixed small box and the42-vertex
convex collision egg in two environments. The test checks public contact
normals and depths against the independent separating-translation targets.
Both patch variants fail on old CPU code with the reversed normal; both pass
after the correction on CPU and CUDA. Standalone times are63.48 s and112.41 s.
The small asset's SHA256 is
`bfa1723bef55465873ec1c8597c3131aef1127616954153165f6c80e0f747b86`.

The expanded suite passes **17 low FP64 cases on CPU** and **all18 cases on
CUDA**, including an existing FP32 geometry-scale check. CPU has one failure
in that additional FP32 finger/lamp check. Its exact1.16215267e-6-m depth
discrepancy against1e-7 acceptance is unchanged on the old code; the old CUDA
check also passes. This pre-existing CPU issue is disclosed and preserved,
with no assertion, precision restriction or acceptance change. Full suite
times are316.76 s CPU and614.24 s CUDA. Initial selector errors are retained.

Eight source-prior field cases with three actual friction variants on each
backend give **48 independent CPU FP64 full-field comparisons**. Maximum
normalized force/moment error is7.387e-10, full residual including six gauge
rows consumes at most0.045857 of its unchanged budget, and maximum peak
difference is8.244e-6 Pa. Complete tensor/von Mises fields, Coulomb cones,
wrench/frame mapping, contact transitions, partial resets and isolation pass.
Final wrench remains1e-8; full residual remains
`max(1e-11 N,1e-8 ||rhs||)`; stress remains rtol1e-4/atol1e-3 Pa.

Ruff passes the new regression module. The existing EPA module has five
pre-existing lint findings, reproduced unchanged on its baseline. Its only
numerical edit is the bound comparison; unrelated style code is preserved.

## Actual B4096 live workload

The original varied B4096/623103 workload now completes600 warmup steps and
three1280-step live repetitions with **zero invalid environment steps,
zero scatter overflow and zero failure-episode resets**. Each repetition has
4643 ordinary resets. All4096 per-environment failure counters are zero.
Effective repeat rates are42251.954/42394.488/42368.060 env-step/s;
aggregate **42338.077 effective env-step/s,10.336444 batch step/s**.
Warmup full residual consumes at most0.007961 of budget.

The explicit application driver still performs timed post-step errno checks
and preserves strict failure semantics; its failure response is unused here.
All required contact updates, recovery work, observations/controller, history,
fallback checks and resets are within scope. The producer correction changes
contact trajectories, so the old failure-reset rate is not a matched speed
baseline and the old captured incompatible wrench is not relabelled feasible.
It remains a valid strict physical failure fixture for its original inputs.

Hardware: RTX PRO6000 Blackwell Server Edition97887 MiB, driver595.71.05,
GPU UUID `c41c23c1-ebfd-c8a1-9f2f-6b6b8db067f0`, container1_26,
Quadrants1.3.3/Genesis1.4.3. Source is`bc364325` plus the recorded EPA patch.
Low complete hollow shell has480 P2 tets/810 P2 nodes/2430 DOFs; Q10/squared
fans/eight edge subdivisions, FP64, dt0.01/one substep, native face frontier,
GJK patch on and explicit body source priors are retained. Native allocation
is12144479838 B, free device memory80605675520 B, Torch peak14940672 B.
This is live simulation throughput, not policy or RL training throughput.

## Evidence and remaining work

The [archive](../evidence/20261011-nonconvex/native-epa-bounds-checkpoint-v1/manifest.json)
contains1129 entries,199 unique blobs and15902680 stored bytes. Manifest SHA256:
`f5835362356b19ee9d63b3ac0ff5a861c2953c8bf1b6d025c1a7e541b13058e7`.
Allocated CPU publication verifies every original/stored/decompressed hash,
all48 fields, the producer/SAT proof, old/new test outcomes and full live
accounting. It preserves diagnostic compilation failures, initial selector
errors and the unchanged CPU FP32 failure alongside successful evidence.

Broader seeds and scales, explicit grasp outcomes and per-environment tail
work, current matched four-scope baselines, detailed profiling and remaining
measured optimization/plateau are still required. The contact pipeline remains
the main performance cost. Current-step port-seed reuse is under isolated
evaluation; no unmeasured gain or plateau is claimed. No high physical mesh
or physical integration convergence experiment was run.
