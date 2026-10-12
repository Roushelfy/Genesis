# Agent goal: native Quadrants rigid stress recovery

## Active extension: nonconvex contact mapping, 2026-10-11

Implement the contact mapping contract in
[docs/NONCONVEX_CONTACT_REQUIREMENTS.md](docs/NONCONVEX_CONTACT_REQUIREMENTS.md).
Preserve surface provenance, orient normals from the material boundary, select
a finite footprint along surface topology, and fit vector tractions preserving
the original six-component wrench and local friction constraints. Support
concave surfaces, cavities, narrow gaps and changing contacts without convexity
or symmetry assumptions.

Integrate this into the existing rigid stress feature using Quadrants for the
complete production numerical path. Keep the Franka/full hollow egg workload,
full/max output options, fixed elastic operators and global peak calculation.
Use small nonconvex fixtures for contact checks and inexpensive meshes for
detailed profiling and aggregate throughput optimization. Validate same-mesh
CPU consistency and lifecycle semantics; physical convergence remains a later,
separate study. Choose the distance algorithm, scheduling and tuning from
evidence rather than copying the Python prototype or fixing its port count.

This is a new requirement after the milestone below. Its prior completion
record does not establish native nonconvex-contact acceptance. Record and push
validated progress, including unfinished work and unavailable device checks.

## Extension progress, 2026-10-12

The extension remains active. Numerical baseline `6a861556` contains
qualified contact repairs and selected seed/layout/port/scatter optimizations.
The subsequent [cold cooperative retry](docs/NATIVE_COOPERATIVE_RETRY_20261012.md)
is selected with a same-GPU1.036986834x live gain,23 CPU/23 CUDA production
cases and320 independent dynamic fields passing unchanged final budgets.
The subsequent [eight-sample-lane retry](docs/NATIVE_SAMPLE_LANE_RETRY_20261012.md)
has a separate paired1.044883994x live gain,23 CUDA production cases and160
dynamic independent fields passing. [Complete-sample compaction](docs/NATIVE_COMPACT_TRACTION_PROFILES_20261012.md)
passes31 CPU/31 CUDA cases and96 fields, saving15.715GB native bytes atB8192
on a separated seed cohort, with a small throughput regression. Its fused
capacity follow-up and remaining refinement/layout/reuse work are active.
[Matched B1024 four scopes](docs/NATIVE_MATCHED_SCOPES_20261012.md) include timed
grasp outcomes, resets and environment work. Contact graphs, fastcache and
standalone device-frontier maximum have no selected dynamic end-to-end benefit;
see [the ablation ledger](docs/NATIVE_CONTACT_SAMPLING_ABLATIONS_20261012.md).
[Dynamic profiling and320 independent GPU field samples](docs/NATIVE_DYNAMIC_PROFILE_20261012.md)
pass unchanged budgets and identify a0.74-1.49s serial fit tail. Cold retry
removes the largest unrefined tail; refinement tails remain. Baseline B4096
and B8192 matched four scopes now pass. Final-source wider scopes, separated
seeds, actual-sample memory compaction, remaining candidate decisions and a
throughput plateau are still required before extension completion.

## Completion record, 2026-10-11

The low-shell native feature, contact repair, unchanged-budget numerical and
lifecycle validation, measured optimization plateau and reproducible four-scope
GPU evidence are complete. See [the final report](docs/NATIVE_RESULTS_20261011.md)
and [generated measurement tables](docs/FINAL_MEASUREMENTS_20261011.md).
Production engine is `f3e92c62`, standalone timing preparation is `1f22f6a2`;
52 GPU cases, 31 CPU cases and 432 independent CPU FP64 full-field samples pass.
All 2004 publication artifacts and their originals pass verification. The
requirements below are preserved as the original brief; physical convergence
remains a separate later study.

## Internal stopping budget, 2026-10-10 follow-up

Match internal contact fitting termination to the actual error budget. The
historical 2e-12 normalized dual threshold is not an acceptance requirement.
An internal relaxation must pass unchanged final contact force/moment,
complete equilibrium including all gauge rows, and independent CPU FP64
stress consistency checks before selection. Sampling/physical infeasibility
must still be distinguished from numerical stagnation.

## Additional output requirement, 2026-10-10

Add a build-time option selecting maximum-only or complete stress-field
output, with maximum-only remaining the default. Full mode exposes the
complete device-side tensor/von Mises field and preserves the existing
maximum observation. Validate same-state numerical agreement, lifecycle
semantics and matched throughput/memory costs for both modes on the low mesh.
See [docs/OUTPUT_MODES.md](docs/OUTPUT_MODES.md) for the required contract.

## Current acceptance, 2026-10-10 follow-up

Prioritize legal-contact robustness before further optimization. The actual
B=2048 apex contact is infeasible under the Q10 constant-ratio sampled pad
law (132 eligible samples, transverse hull gap 10.9 micrometres). Improve the
declared load representation or local integration and turn that fixture into
successful recovery. Distinguish sampling, fitting failure and physical
incompatibility. Keep force, moment, frames and friction constraints explicit.
Vary actual combined contact coefficients in the live workload.

Use only the low-resolution full hollow shell in this phase. Rebuild a matched
baseline after any load-model change. Profile and evaluate applicable batch
tiling/reuse, scatter/reduction, scheduling, device conditions, failure packing,
shared operators, precision/correction, history and global peak candidates.
Record a measured result or concrete cost/profile reason for each decision.

Final GPU evidence covers effective rigid-only, recovery-only, live and policy
rates, batch steps/s, numerical failures, grasp outcomes, tail work and memory.
Use changing contacts, different seeds, partial resets and repeated trajectories
at B=1024,2048 and larger batches. Resume peak exploration after contact repair.
The earlier copied-condition scan was stopped at the user's request and is
preliminary historical evidence. Policy inference defines rollout scope only.

Complete and push only after contact, numerical and lifecycle checks pass and
measured optimization reaches a plateau with no unresolved actionable major
bottleneck. Missing resources must leave their tests explicitly unverified.

## Implementation objective

Implement auxiliary fixed-geometry linear-elastic stress recovery as an
optional, native feature of Genesis's existing rigid solver in
`Roushelfy/Genesis`, branch `rigid-stress-recovery`. Return the global maximum
von Mises stress per environment and maximize measured aggregate environment
transitions/s, including parallel scenes and matched policy inference.

Read [docs/NATIVE_QUADRANTS_PLAN.md](docs/NATIVE_QUADRANTS_PLAN.md) first, then
the relevant math/reference evidence and Genesis's development conventions.
This goal and that document replace the previous standalone GPU architecture
and convergence-first acceptance sequence. The existing research implementation
is a numerical reference and source of measurements, not the final feature.

Use Quadrants for all production numerical computation, including contact/load
mapping, elastic operators, recovery solves/preconditioners, residuals, history,
selection and maximum-stress reduction. Integrate configuration, state,
substep execution, resets and public observation into the existing rigid
solver lifecycle. Independent offline CPU FP64 oracles and ordinary Python
configuration/asset handling remain useful; CuPy, raw CUDA, Triton, Torch
numerical kernels, cuDSS and cuSPARSE do not implement the production feature.
Consult [upstream PR #3372](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3372)
for native integration, Quadrants and profiling patterns. Choose the numerical
method and API that best fit this feature; a separate Mochi solver is not needed.

Use a Franka Panda gripping a full hollow egg-shaped rigid link. Only reference
geometry is fixed relative to contacts: positions, counts, footprints,
directions and symmetry remain unrestricted. Include normal and tangential
frictional forces and all support contacts. Preserve enabled contact moments,
or explicitly reject unsupported modes. Smooth grasp evolution can help reuse;
contact changes, slip, release and partial resets must remain correct. Material
and mass are declared fixed inputs to any shared operator. Rigid dynamics stay
unchanged by this auxiliary stress calculation.

Prioritize performance work on inexpensive low-resolution full-shell meshes.
First establish same-mesh numerical consistency, obtain detailed profiling,
and iteratively remove the actual throughput and memory bottlenecks. Explore
the documented candidates and additional opportunities identified by profiling;
retain improvements based on end-to-end measurements, not isolated timings.
Keep accuracy, timestep, sampling and the load model fixed within comparisons.

Before detailed profiling and the applicable performance optimizations have
been completed and documented, do not run high-resolution physical mesh or
quadrature convergence studies. Numerical consistency checks continue throughout:
same-input FP64 comparisons, complete residual including gauge rows, global
peak error, contact wrench/frame consistency, batching and reset independence.
Low-resolution results are performance-development results, not physically
converged egg stresses. Physical convergence is a later, separate stage.

Deliver native engine code, a normal Genesis example and tests, reproducible
low-resolution benchmarks, a detailed stage profile, optimization/ablation
evidence and a concise account of remaining bottlenecks. Measure rigid-only,
recovery, live rigid-plus-stress and policy-plus-rigid-plus-stress scopes.
Report actual hardware, precision, mesh, batch size, memory, numerical errors
and aggregate transitions/s. Commit and push to this task branch. Work
autonomously; use evidence and engineering judgment to choose implementation
details. If device access is unavailable, identify the unmeasured parts.
