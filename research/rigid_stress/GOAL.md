# Agent goal: native Quadrants rigid stress recovery

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
