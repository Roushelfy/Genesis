# CPU 12-feature Atlas: matched Quadrants GPU validation

**Delivered specification, 2026-10-27.** See [the matched GPU report](CPU12_GPU_RESULTS_20261026.md), [the native backend decisions](CPU12_NATIVE_RESULTS_20261026.md) and [the requirement packet](../evidence/20261026-cpu12-gpu-parity-v1/README.md). The original scope below is preserved. Establish a faithful GPU comparison of the
published CPU 12-feature construction, then carry promising constructions
through the existing native rigid stress feature. Deliver an accuracy,
complete-scene throughput and memory decision for shell and volume separately.
Measured negative results are valid outcomes after the specified mechanism
has been implemented and tuned.

## Starting point and scope

The primary executable reference is
[`cpu_repro`](../atlas_study/cpu_repro/README.md) at
[`a10c5abe`](https://github.com/Roushelfy/Genesis/tree/a10c5abe3b5bc3c5caa57d3d4d9dadab2b277ade/research/rigid_stress/atlas_study/cpu_repro),
with its [measured report](../atlas_study/cpu_repro/REPORT_ZH.md).
It is the published reconstruction of the unavailable original `cpu_next`.
Its newly measured gains and conditional/negative findings are the reference.

The [previous native campaign](LOCAL_REFINEMENT_RESULTS_20261026.md) tested
genuine reduced spaces with 3/6-feature libraries and different coarse spaces
and mother discretizations. Its tuned complete-path negatives remain valid
for those constructions. The CPU 12-feature configuration still requires a
matched native comparison. Keep the useful internal source-grouping and
single-recovery results, the earlier H/FP32/screen/history negatives, and
the maintained pressure/fallback/angular improvements. Frozen experimental
source at `822ad285497f79d3a24f3ce2f0dd7a58ec6cd699` is available for reuse.

## First milestone: match the CPU shell construction

Port the discrete mechanism into Quadrants and compare it with the frozen CPU
oracle before changing the feature-space construction. Use the same egg,
material/thickness, Q4 MITC4 mother, bilinear coarse prolongation P, local
enrichment and halo, twelve centers, gauge, mass and inertia relief.
The reference has 5,196 raw / 5,190 free mother DOF and 3,522 raw / 3,516 free
DOF per feature. Match the actual spaces and operators, beyond their dimensions.
A validated change of basis is acceptable when its equivalence is demonstrated.

Freeze matching geometry, topology, P/B spaces, loads, selectors and outputs
with source identities and input hashes. Exercise the published 42 loaded
cases plus release, boundary/junction controls and the static performance bank.
Use the prescribed material-cap loads and their original integration for this
parity milestone. The CPU material-cap adapter and the native automatic-radius
DPR law have distinct qualifications.

Compare mother and Atlas on GPU with identical current loads, integration,
precision, stress sampling and output. Cover continuous weighting, nearest
per contact and narrow transitions, same-feature RHS aggregation, batched
solves, merged displacements and one global tensor recovery. Keep all positive
junction contributors and sum tensors before evaluating stress criteria.
Count active groups, support coverage and fallbacks.

Check load transfer, common relief/gauge, reduced operators and unscaled
feature solve residuals against the CPU reference. Separately reproduce
the combined full-field and sampled-peak errors relative to the same mother,
using the report's norms and samples. Floating-point differences get explicit
precision-appropriate tolerances. Preserve existing exact-mode contracts.
The mother residual of an approximate field remains a spatial diagnostic.
An unconditional full-mother correction would change the tested mechanism.

A small research driver may establish parity, but the reusable numerical
implementation belongs to the existing native stress feature. All production
floating numerical preparation and runtime remain in Quadrants. CPU/SciPy
matrices, factors and solutions serve as independent validation oracles outside
production and timed GPU execution.

## Second milestone: native adaptation and useful throughput

Use the parity findings to adapt and tune both native midsurface shell and
three-dimensional volume FEM. Each backend keeps its own mother, geometry,
load transfer, gauge/relief and stress outputs. Volume requires its own
numerical and spatial qualification. Label any altered shell space or native
load law as an adaptation with its own accuracy/cost results.

Prioritize a useful accuracy/work reduction. Explore a bounded, informative
range of library density, local/core/halo size, coarse-space quality and
transition width, guided by the profile and failures. Prefer actual smaller
feature operators. Compare with the optimized mother and a suitable coarse
or joint-refinement control at matching accuracy and output. Another 3/6-center
trial alone does not close the CPU 12-feature milestone.

Preserve mature automatic-radius DPR pressure and normal/friction moments,
arbitrary asymmetric moving contacts, both surface sides, convex/nonconvex
surface identity, full-object max/full output, environment isolation, release,
partial resets and checkpoints. Validate finite-footprint coverage and remote
hotspots. Use conservative validated fallback where coverage or approximation
is unqualified, and report its frequency and cost.

Profile the full path, including contact/load preparation, selection/coverage,
grouping, projection/relief, solves, lift, stress/maximum reduction, acceptance
and fallback. Optimize measured bottlenecks and retain useful combinations.
Revisit prior negative candidates only where a changed smaller-space workload
or new measured bottleneck provides a concrete reason.

Measure frozen-contact modules and complete Franka/hollow-egg RL workflows at
small and useful large batches. Include max and full modes separately,
matched pure-rigid and optimized mother baselines, required runtime checks,
warmup, paired repeats, invalid/rejected steps and reproducible source/config
identities. Report latency and valid environment-steps/s. Account for actual
controller, rigid-step fusion, transfers and scheduling in full-scene results.

Report preparation time, feature dimensions/nonzeros/factor fill, shared
inventory, per-environment buffers and peak device/process memory. Keep
numerical-oracle exports and intrusive profiling outside production timings.
CPU timing is evidence for a mechanism, not a target GPU speedup.

## Completion

Deliver runnable source, focused consistency/lifecycle checks, raw paired
measurements and a concise recommendation for each backend. Distinguish:

1. CPU/GPU equivalence within a declared feature space.
2. Spatial approximation error against the matching mother, including full
   tensor fields, sampled maxima and applicable shell resultants.
3. Incremental component gains and complete-scene accuracy/throughput/memory
   tradeoffs, including fallback.

Use low/moderate models and preserve the existing declared error budgets.
Numerical consistency and performance tuning take priority over new
high-resolution physical-convergence campaigns. Same-discretization agreement
qualifies the discrete algorithm. Physical contact-area accuracy and continuum
contact peaks retain their existing limitations.

Keep beneficial production changes. Archive tuned negative experiments with
runnable source pins and restoration instructions. Close the matched CPU
milestone and each native backend decision explicitly, with remaining gaps
identified. Documentation alone or an isolated solve speedup does not complete
this goal.
