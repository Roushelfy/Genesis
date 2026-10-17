# Next goal: maximize throughput of the practical rigid stress feature

User direction, 2026-10-17, on `Roushelfy/Genesis:rigid-stress-recovery`.
This is the active next stage after the scoped
[application delivery](PRACTICAL_APPLICATION_RESULTS_20261016.md).
The [application-first validation policy](PRACTICAL_VALIDATION_GOAL.md)
continues to govern accuracy claims and bounded validation work.

Delivery status, 2026-10-17: this stage is complete. The
[final report](PRACTICAL_THROUGHPUT_RESULTS_20261017.md) and
[requirement-by-requirement audit](PRACTICAL_THROUGHPUT_COMPLETION_AUDIT_20261017.md)
record the measured operating points, verified source/numerics and inspected
exports. The original requirements below are preserved.

## Outcome

Make the recommended robust configuration fast for many parallel Franka/full
hollow egg grasps. Find and optimize the actual end-to-end throughput limit,
explain the remaining bottlenecks, and deliver reproducible recommended
settings. Do not stop at the existing32-environment measurement or treat the
previous different-workload large-batch rates as evidence for this configuration.

Keep the feature inside Genesis's rigid solver with all production numerical
computation in Quadrants. Preserve fixed geometry/material/mass, arbitrary
changing and asymmetric contacts, normal force and friction, support loads and
enabled moments, convex/nonconvex options, global raw maximum/full outputs,
per-environment history and partial resets. Smooth contact evolution permits
reuse, but contact transitions must correctly invalidate it.

## Start from a matched practical workload

Use the delivered low-resolution full shell (480 P2 tetrahedra,2430 DOF) and
the recorded condition bank/controller as the initial performance workload.
The primary model is intrinsic `body_source_prior`, with the declared nominal
6mm radius, quadrature10, ports8, history4, max output, mesh64 collider and
recorded MPR/patch-off settings. Reproduce the exact saved configuration rather
than reconstructing undocumented defaults from this paragraph.

Existing RTX PRO6000 Blackwell B32 rates are2623.09 live and2487.41 policy
valid env-step/s for source-prior; pressure gives5840.70/5437.83.
These are starting measurements, not throughput plateaus or physically
calibrated rates. Pressure is a separately labeled simplified comparator;
its known local contact/friction-admissibility defect remains. Do not obtain
the primary speedup by silently switching the robust model to pressure.

Freeze a paired baseline before changing code. Keep timestep/substeps, contact
law, mesh/material, physical footprint, policy/control, simulated duration and
output/check frequency matched. If an approximation or different fidelity is
worth offering, report it as a separate validated accuracy/cost option.

## First measure scaling and explain cost

Sweep batch sizes from small batches into the practical GPU-capacity range,
then refine around the best valid throughput and memory limit. Choose the
widths and repeat lengths using engineering judgment; no fixed batch or FPS
target is prescribed. Retain condition diversity, changing contacts, support,
release and partial resets. Report any active-set/contact-count changes across
batch sizes rather than attributing all rate changes to GPU efficiency.

Use matched rigid, rigid+stress, policy+rigid and policy+rigid+stress scopes.
Measure aggregate valid environment transitions/s, batch latency, incremental
stress cost and peak memory. Include reset/control/policy costs in their stated
scopes. Warm compilation/caches, use repeated measurements on comparable GPU
allocations, and keep intrusive observers/profilers out of headline timings.
Distinguish cold preprocessing/JIT from amortized runtime and state the scope
of policy inference versus full RL training.

Build a useful breakdown of CPU orchestration/synchronization, rigid dynamics
and collision, contact transforms/support/path queries, traction fitting,
load assembly, prediction/history, elastic recovery, fallback/selection,
complete residual checks, global peak/full output and reset/memory traffic.
Record kernel launches and host/device copies where relevant, together with
fallback/iteration counts, active contacts and fraction of environments needing
each expensive path. Use GPU event/kernel profiling and end-to-end wall time
as appropriate; explain overlap and profiler perturbation instead of forcing
all stage timings to add up.

## Optimize the demonstrated bottlenecks

Implement and tune the applicable improvements, preserving numerical contracts.
Promising directions include shared immutable storage, device-resident contact
data and outputs, avoiding copies/synchronization, better batch layout and
fusion, batched recovery/fallback work, failed-environment compaction, smooth
history/active-set reuse, cheaper history maintenance, sparse active contact
work, support/path caching and fast paths with verified invalidation.
Investigate the traction-fit versus elastic-recovery split before choosing
where to invest. Avoid repeating optimizations already qualified unless the
new workload exposes a new bottleneck.

Mixed precision or cheaper numerical work may be explored where evidence
supports it, with same-input stress/wrench/residual checks and a reliable
fallback. Do not loosen original acceptance tolerances to create a speedup.
Preserve local friction/compression and source/aggregate wrench contracts.
Convex fast paths must remain optional and have a correct robust nonconvex
path; arbitrary contact position and symmetry cannot be assumed.

Let profiling determine algorithms and tuning. Provide matched before/after
measurements and small ablations for material improvements. A memory reduction
counts toward the goal when it improves usable batch capacity or throughput.
Report expensive ideas that did not help, without building an exhaustive
experiment matrix. Finish with an evidence-based practical plateau and an
account of the significant remaining opportunities or blockers.

## Validate what matters without restarting the convergence campaign

Reuse existing independent FEM and R/L/M evidence. For changed production
arithmetic, run targeted same-input checks of complete equilibrium/gauge,
wrenches, full tensor/raw global peak and selected physical traction samples.
Exercise environment independence, contact transitions/release, cache
invalidation, fallback and partial resets. Use small CPU/GPU cases to find
defects, then validate the selected optimized large-batch configuration.
Keep full/max output semantics even when max is the throughput default.

Retain a bounded complete-grasp check using raw per-step maxima and selected
stress/traction maps. Assess useful reward discrimination across the complete
recorded episode as well as explicitly named phases. Preserve the observed
phase-dependent reversals:7/8 above-band lift comparisons are not7/8 reliable
whole-episode decisions (the delivered whole-episode result was1/8).
Do not tune force choices or reporting windows merely to manufacture a pass.

Load reconstruction's measured peak bias and reference uncertainty remain
limitations, not solver failures to hide or claims of real damage accuracy.
Fix concrete defects exposed by tuning, but do not reopen the universal
high-mesh convergence matrix, rebuild a deformable production solver, or make
full RL training/fracture calibration prerequisites for this throughput stage.

## Delivery and completion

Commit implementation and useful progress on this branch. Deliver runnable
commands, a concise profile/optimization report, scaling and memory curves,
matched baseline/final rates, numerical checks, and recommended fast/robust
settings with their accuracy limitations. Preserve enough source/configuration,
raw summaries and failed-case evidence to reproduce conclusions; reuse compact
archives rather than repeating giant publication campaigns.

Complete this stage when the recommended robust workload has a measured
large-batch operating point/plateau, the material applicable bottlenecks have
been addressed or explained with measurements, relevant numerical/lifecycle
checks pass, and complete-grasp proxy limitations remain clear. Unavailable GPU
runs or unimplemented opportunities must be labeled honestly. The desired
result is the fastest demonstrated useful native feature, not an invented FPS
promise or another broad physical-convergence exercise.
