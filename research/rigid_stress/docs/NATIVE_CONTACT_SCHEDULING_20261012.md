# Sample scheduling and verified-circle contact integration

This checkpoint follows the [native continuous-wrench diagnosis](NATIVE_CONTINUOUS_WRENCH_20261011.md).
Its source base is `047d7de9e4650e4f31631cdd909659660f16d9e9`. The active
nonconvex goal remains incomplete. The scheduling results below use explicitly
archived experimental Quadrants kernels on that unchanged integration law.
The subsequent production circle repair changes local integration; its
performance needs a new baseline before selecting further optimizations.

The [publication manifest](../evidence/20261011-nonconvex/contact-scheduling-circle-checkpoint-v1/manifest.json)
records **1060 copied raw artifacts, 26,103,429 bytes**, with every original and
published SHA256 verified on allocated CPU compute. No selected runtime file
exceeds the publication size limits; compilation caches and temporary files are
excluded. Manifest SHA256 is
`76ac15d39ff5f3487d713c1844d34699de96f08ad59898cc8c535267ffb69f53`.

## Matched scheduling evidence

The inexpensive full hollow shell has 480 tetrahedra, 810 P2 nodes and 2430
displacement DOFs. The collision workload uses `mesh64`, GJK patch contacts,
Q10 squared surface fans, s8 paths, adjacent unfolding and declared pair-union
support. Each original component retains its own point, force, free moment and
acceptance budget; every destination retains its actual radius and friction.
Precision is FP64, timestep 0.01 s, one substep, maximum-only output, no load
history. Internal termination remains 9e-9 with 128 evaluations. Final wrench,
complete residual including gauge rows and CPU stress budgets are unchanged.

The v10 fit candidate gives a CUDA warp to one contact and distributes its
actual samples across the lanes. Warp reductions assemble the dual energy,
six gradient entries and 21 lower-triangle Hessian entries. Only lane zero
solves the small system and writes fit state. Trial acceptance, adaptive
regularization, trust-step control and original-point termination retain the
scalar algorithm. The two matrix attempts use one runtime helper call.
CPU uses one lane. The v2 commit and scatter candidates similarly distribute
samples, preserve the independently checked local cones and original wrench,
and sum every original component before signed P2 atomic accumulation.
These experimental kernels are not selected production scheduling yet.

The candidates pass **51 CPU cases in 384.79 s and 51 CUDA cases in 534.14 s**.
Complete live checks also pass B16/seed623001 and B32/seed623101, each for 1280
steps, retaining all environments. Together they include 61,440 environment
transitions, 432 independent CPU FP64 full-field comparisons and 56 saved-state
resets. Every-step force/moment errors stay below 1e-8, including release and
changing/slipping contacts. Actual combined friction varies approximately
0.4243–0.8281; it is not held at one.

Detailed B32 snapshot profiles share exactly the same condition bank,
SHA256 `156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.
They use 600 warmup steps and 20 snapshot repetitions. Scalar and cooperative
profiles use different GPUs of the same model; the repeated live comparison
below uses one physical GPU.

| Stage | Scalar | Experimental sample-parallel |
|---|---:|---:|
| Intrinsic pipeline wall time | 169.081 ms | 9.834 ms |
| Fit kernel mean | 145.052 ms | 5.384 ms |
| Original-contact commit kernel mean | 9.940 ms | 0.266 ms |
| Signed scatter kernel mean | 10.017 ms | 0.269 ms |
| Elastic solve wall time | 0.193 ms | 0.194 ms |

Kernel means come from the native Quadrants profiler; wall timings include
dispatch/synchronization and are not the sum of only the listed kernels.
The successful-frame certificate kernels are small here; this does not measure
the tail cost of unresolved environments. Fit remains the largest measured
stage. Snapshot replay changes no original point, preserves wrench acceptance
and complete equilibrium, and changes global peak by less than 9e-9 Pa.

The actual live comparison runs scalar then cooperative on the same RTX PRO
6000 Blackwell Server Edition, UUID
`GPU-e5860f82-f768-686d-acb8-a5360f1b1247`, node `rtx-209-115`, driver 595.71.05.
Each variant performs 600 warmup steps, then three complete 1280-step varied
trajectories with seed623101, including native control, current contact updates
and required resets. Both variants use the same condition bank and workload
settings.

| Variant | Effective env-step/s, three repeats | Whole-batch step/s, three repeats |
|---|---|---|
| Scalar native | 201.986, 202.131, 202.011 | 6.3121, 6.3166, 6.3128 |
| Experimental sample-parallel | 2673.836, 2668.943, 2678.520 | 83.5574, 83.4045, 83.7037 |

The mean rate gain is approximately **13.23×**. Every repetition has zero
invalid environment steps, zero scatter overflow and 37 resets. Each result
retains the 32 per-environment failure counters, all zero. Native measured
buffers occupy 167,455,666 bytes. These rates cover live rigid plus stress;
the result's policy configuration metadata does not mean policy inference was
executed in this scope. They are not policy rollout or complete RL training
rates, larger-batch acceptance, or an optimization plateau.

The v11 storage candidate retains the logical `(sample,contact,environment)`
scalar interface while making samples physically contiguous within an
environment. It passes all 51 CUDA cases; CPU takes the unchanged v10 branch.
On the same saved B32 input its single intrinsic profile is 9.702 ms, only
about 1.34% below v10. These two profiles use different physical GPUs of the
same model. This is insufficient evidence to select the layout.
Larger-batch repeated measurements remain necessary.

## Legal small-radius sampling defect

An additional fixture uses the unchanged low shell's actual face 0, physical
radius **10 micrometres**, actual friction 0.5 and tangential/normal force
ratio 0.4. Its rigid point retains the offset moment under projection. The old
Q10 squared triangle fan selects 30 points; its most negative tangential
coordinate is -1.3199032295 micrometres. The required pressure centroid is
-5.6599516147 micrometres, outside that sample hull by **4.3400483853
micrometres**.

A uniform vector traction on a positive-area subdisk centered at that centroid
with radius 1.0850120963 micrometres realizes the exact force and zero moment
about the original point. The entire subdisk lies inside the unchanged physical
radius and one real material triangle; its ratio 0.4 is strictly inside the
actual friction cone 0.5. This proves continuum feasibility independently of the
native fit and without a singular point load. The old native fit/diagnostic
remain unresolved rather than falsely establishing continuous infeasibility.
More Newton iterations cannot fix the missing sampled wrench.

## Production integration repair

For a contact whose entire physical disk is verified to lie within one actual
boundary triangle, production now uses a radius-scaled circular area rule.
The geometric guard checks anchor-plane roundoff and distance to all three
real edges against the unchanged radius plus operand-scaled FP64 allowance.
Any path leaving this convex triangle already spends at least R before its
first boundary crossing. Thus the complete intrinsic footprint in this regime
is exactly the disk; a nearby disconnected sheet or a hole cannot enter it.
Other footprints retain the existing finite triangle fans and intrinsic paths.

The rule uses the existing Q10 nodes with radial coordinate `r/R=u^2`, angular
coordinate `theta=2*pi*v` and area Jacobian `4*pi*R^2*u^3`. It retains the
declared Gaussian/taper profile, true inward normal, physical radius, original
point/force/free moment and actual friction. It integrates 100 finite-area
samples independently of the face-size/radius ratio. Geometry and signed P2
bases are reconstructed in Quadrants from shared face data and typed contact
chart metadata; no per-sample geometry cloud is added. Circle sample identifiers
use a disjoint namespace inside the existing exact profile allocation.

The standalone native candidate passes independent CPU FP64 geometry, profile,
vector fit, signed nodal loads, full equilibrium, complete tensor/von Mises and
global peak comparisons on CPU and CUDA. Both fits finish in eight evaluations.
Force/moment relative errors are below 2.3e-15. Native peaks are approximately
268079.36133 Pa and the independent CPU peak is 268079.361331804 Pa. Complete
residuals are 2.094e-11 N on CPU and 1.919e-11 N on CUDA, within the unchanged
per-input `max(1e-11 N,1e-8||rhs||)` budget.

The new production regression makes this legal fixture successful and packs
four independent radii, 1/5/10/20 micrometres, and actual coefficients
0.5/0.6/0.85/1.2 into one model. It checks independent CPU quadrature and fitted
forces, original-point sample and P2 wrenches, local cones, full residual/gauge
rows, tensor/von Mises/global peak, exact unaffected output buffers at partial
reset, release/recovery and chart invalidation at a real triangle boundary.
The metadata participates in normal SCRATCH data enumeration and is rebuilt
from current geometry each pass. The complete combined suite passes **52 CPU
cases in 428.92 s and 52 CUDA cases in 610.86 s**, with source closure intact.

## Frozen failed controls and reproduction

The first cooperative profile/live launchers stop before Python because `rg`
is absent in the container; corrected launchers use its available `grep`.
Both first circle probes reach successful native fitting and geometry/profile
checks, then fail because the harness passes eight arguments to the four-argument
CPU oracle. The same pre-validation harness also had the P2 edge ordering
wrong; v2 uses Genesis's 01/02/12 ordering and the actual oracle interface.
Those failed source epochs and logs are retained.

The first production test passes CPU but its CUDA replay-only comparison
requires 1e-12 relative/1e-8 Pa beyond the stated final acceptance. Its largest
violating difference is 6.006e-8 Pa after independently accepted initial
wrench/full residual/CPU fields. The corrected test recomputes the release
result against the independent CPU oracle, including full residual and tensor/
von Mises/global peak, using the unchanged final 1e-4/1e-3 Pa budgets. Exact
unaffected buffers at the reset itself are still checked. The first combined
CPU allocation is cancelled before source edits and retained as an interrupted
run, with source closure audited; it is not counted as a pass.

Genesis 1.4.3, Quadrants 1.3.3, container 1_26, CPU EPYC 9454, and actual RTX PRO 6000
Blackwell Server GPUs are used. Every successful source epoch records before/
after hashes. Runtime data and caches remain under the established project data
root. The publication manifest records immutable native/experimental sources,
all closed logs and saved conditions, with large snapshots verified in place.

Inside an allocated CPU/GPU node, the production combined regressions use
`bash research/rigid_stress/scripts/check_nonconvex_contact.sh cpu` or `cuda`.
The archived `run_native_circle_integration_v2.sh` includes all nine native
modules and the new fixture. Archived candidate drivers, launchers, condition
bank and exact source copies reproduce the experimental scheduling comparisons;
they must not be mistaken for a current production speedup.

## Remaining acceptance and tuning

This repairs the exhibited legal one-face sampling failure. It does not establish
universal sufficient integration for boundary-near wrenches or general intrinsic
footprints; justified local refinement remains applicable. The continuously
incompatible B1024 step602 original component still requires the outstanding
explicit source/load-model decision. No component is dropped or replaced by
an accepted pair total here.

Retaining the scheduling improvement requires validation and matched new-law
measurements after this repair, followed by batch scaling, layout decisions,
device dispatch/fusion, unresolved-environment compaction, shared geometry/
operator work and any other measured high-benefit candidates. The new-law
rigid-only/recovery/live/policy scopes, repeated 1024/2048/larger trajectories,
different seeds, grasp outcomes, tail/failed-environment cost, memory and
measured plateau remain unfinished. No high-resolution physical mesh or
integration convergence study is run in this checkpoint.
