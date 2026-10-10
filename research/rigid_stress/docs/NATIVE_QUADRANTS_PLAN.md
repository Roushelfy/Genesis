# Native rigid stress recovery: performance-first plan

User direction, 2026-10-10. This document and `../GOAL.md` are the current
implementation brief. They supersede the earlier standalone research backend,
native CUDA-library recommendations, mandatory GPU direct-backend choice and
convergence-first acceptance sequence. Earlier code and evidence remain useful
for mathematics, numerical comparisons and bottleneck diagnosis.

## Result to build

Add optional auxiliary linear-elastic stress recovery to Genesis's existing
rigid solver. A normal Genesis scene should configure a rigid link for stress
observation, call `scene.step()`, retrieve its device-resident maximum von Mises
stress, and reset selected environments through the ordinary lifecycle. The
feature should work without importing or running `research.rigid_stress`.

Use Quadrants for all production numerical kernels: elastic assembly/operator
application, contact transforms and mapping, load balance, solve and
preconditioning, residuals, history, selection and global maximum reduction.
Python can handle options, assets, orchestration and reporting; existing
zero-copy Torch accessors can expose observations and policies can use Torch.
Neither circumstance authorizes Torch numerical work inside stress recovery.
The production feature does not delegate numerical work to CuPy, raw CUDA,
Triton, cuDSS or cuSPARSE. Independent offline NumPy/SciPy FP64 reference
calculations are allowed for validation and are outside throughput timing.

Fixed geometry permits shared preprocessing, operators, factors or
preconditioners. Select a Quadrants implementation that gives the best measured
throughput; the previous cuDSS backend is not a required algorithm. Preserve
the linear-elastic problem while changing its numerical solution strategy.

Integrate with rigid solver configuration, typed state/info, build, substeps,
error handling, observations and partial resets. Reuse existing change
notifications and state APIs. Keep configuration and storage proportional to
the built scene and enabled stress links; account for the disabled-feature
cost. Put production code, tests and examples in the normal engine layout.
The agent should choose the exact public API and module structure using sibling
conventions rather than preserve the callback-based research interface.

## Upstream integration reference

[PR #3372, "[FEATURE] add mochi solver"](https://github.com/Genesis-Embodied-AI/genesis-world/pull/3372)
was inspected as an open, unmerged proposal at head
`c7d29e08f5abd049b95ec187591f64c33553c18a`. Its integration shows options flowing
through Scene/Simulator, native typed solver state, Quadrants dense/iterative
linear kernels, component tests, examples and a benchmark with kernel profiling.
Inspect its current changes when borrowing a pattern; it is not a stable API
or a requirement to import/merge the PR. Here stress recovery augments the
existing rigid solver instead of registering another physics solver.

## Model and contact scope

The 2026-10-10 follow-up prioritizes the sampled pad model's genuine apex
infeasibility. Repair its load representation or integration before further
throughput tuning. Preserve original wrench and friction acceptance and
validate actual combined coefficients that vary. Rebuild all baselines
after a model/integration change. See [CONTACT_REPAIR_20261010.md](CONTACT_REPAIR_20261010.md).

The benchmark is a Franka Panda grasping a full hollow egg-shaped rigid link.
The shell is hollow for elastic stiffness and mass, and rigid for dynamics and
collision. Use the existing fixed-geometry, small-strain, quasi-static elastic
model, initially affine P2 tetrahedra and a global unaveraged von Mises maximum.
Do not feed elastic displacement back into collision or substitute deformable
FEM dynamics. Material, mass and discretization are explicit operator inputs.

Only reference geometry is fixed relative to contacts. Contact locations,
counts, normals, force directions, footprint areas and symmetry can vary.
Include actual normal and tangential friction forces and every support contact.
Handle pure contact moments when enabled, or clearly reject unsupported modes.
No symmetry reduction, fixed contact-response basis or preselected hotspot can
serve as the correctness path. Smooth trajectories allow reuse but contact
birth/death, slip, release, abrupt changes and partial resets remain supported.

Keep a declared finite-footprint model: a rigid point force alone does not
identify local pressure. Mesh size must not silently change that model.
Contacts, pose and angular velocity must refer to the same rigid substep.
Choose the sampling contract explicitly and hold it fixed in comparisons.

## Development order and validation budget

1. Establish a small, numerically consistent native Quadrants path and an
   inexpensive Franka/egg workload. Full-shell level 1 or 2 is a useful starting
   size; choose the smallest mesh that meaningfully exercises the stage under
   investigation. No symmetry shortcuts are needed.
2. Profile in detail and iterate on throughput, storage and execution overhead.
   Sweep parallel environments, layouts, chunks, precision and numerical methods
   where relevant. Short experiments guide development; repeat complete matched
   workloads at measurement checkpoints.
3. Finish the applicable performance work and document the remaining limiting
   stages, successful changes and rejected candidates. Then schedule physical
   mesh and quadrature convergence as a separate later study.

Before step 3, do not run the old high-resolution acceptance workloads,
level-4/5/6 physical convergence sweep, fine-mesh oracle replay or high-resolution
quadrature refinement. Disable physical convergence in the new default test
and benchmark workflow; expose it as an explicit later opt-in. The old frozen
level-6 configuration and archived GPU rates are historical references, not
the default workload for this phase. There is no need to rerun those results.

Numerical checks continue throughout optimization. Compare identical mesh,
loads, footprint/quadrature and elastic inputs with an independent FP64 oracle.
Check assembled nodal loads and resultant force/moment, complete equilibrium
including gauge rows, displacement modulo gauge and the global scalar maximum.
Include rigid-frame transforms, freefall/centrifugal cases, arbitrary asymmetric
frictional loads, no contacts, repeated and abrupt loads, batching and partial
resets. Use small cases that expose errors without long GPU runs.

Precision changes need same-mesh residual and peak-error measurements with
absolute errors near zero, plus correction/failure handling. The existing
strict/throughput numerical profiles are useful starting points; document the
chosen profile and compare speed at matching accuracy. Physical discretization
error is deferred, not certified by a low-resolution numerical check.

"Finish performance work" is an evidence-based decision, not an endless search
for every conceivable optimization. Review every material bottleneck and the
applicable candidates below, investigate further opportunities suggested by
the profile, and record measured benefit, regression or a concrete reason a
candidate does not apply. No fixed algorithm or mandatory tuning grid is imposed.

## Profiling breakdown

Do not combine almost all stress work into a single "recovery" measurement.
Separate at least the applicable stages below; subdivide a dominant stage.

| Stage | Useful measurements |
|---|---|
| Rigid control, collision and constraint solve | Matched stress-disabled baseline |
| Contact association and material-frame transforms | Time, launches, copies, contact counts |
| Surface anchor search | Time and faces/nodes visited per contact |
| Footprint candidate search | Candidate counts and spatial-index effectiveness |
| Patch weights and wrench constraints | Integration/recomputation and local small solve |
| Nodal scatter and RHS assembly | Atomics, contention and bytes moved |
| Inertia relief and centrifugal terms | Dense/sparse products and full-array passes |
| Prediction and history projection | Per-environment work and prediction success |
| Residual operator application | Operator bandwidth and arithmetic |
| Residual reduction and acceptance | Reduction passes and duplicate work |
| Failed-environment selection and packing | Count synchronization and pack/scatter cost |
| Linear solve/preconditioner | Iterations, operator/preconditioner/reduction cost |
| Correction and final residual | Work by failure fraction, fallback cost |
| Global maximum-stress scan | Gather, corner arithmetic, reduction, register use |
| History maintenance | Append, KQ/Gram updates, rank checks, reset cost |
| Observation, policy and resets | Device/host boundary and end-to-end overhead |

Record initialization, operator/factor/preconditioner construction, JIT and
graph setup separately from steady state. Also record allocation count, peak
memory, per-buffer sizes, copy bytes, launch count, synchronization, GPU idle
time and per-environment workload variation. Use Quadrants and available GPU
profilers with low-overhead wall/event timing; final throughput is measured
without intrusive profiling. Concurrent stages need critical-path analysis,
not a sum of overlapping isolated stage times.

Report recovery-only, live rigid-plus-stress, policy-plus-rigid-plus-stress and
matched rigid baselines separately. The primary rate is total counted
environment transitions divided by wall time; also report batch steps/s and
stress substeps. Record hardware, mesh/DOFs, batch/chunk size, precision,
numerical errors, sampling, contact statistics and grasp outcome. A policy
inference benchmark does not establish completed RL training throughput.

## Optimization candidates to investigate

These guide exploration; profiling should determine the order and best method.

- **Direct native data flow.** Read rigid contact/state arrays through one
  integrated vectorized interface at the correct substep. Fuse force-side
  selection and coordinate arithmetic; use zero-copy observation accessors.
  Remove actual copies, not just DLPack wrappers that already alias memory.
- **Persistent storage and layout.** Reuse buffers sized for the actual scene
  and selected algorithm. Have producers write the consumer's layout and DOF
  ordering, avoid redundant gather/scatter and full displacement expansion,
  and tune environment/node/tet tiling. Sweep resident B and recovery chunks
  independently. Profile transpose cost against consumer kernel performance.
- **Contact search and reuse.** Replace full-surface anchor scans with an exact
  spatial hierarchy. Try previous-face fast checks with a robust search
  fallback. Cache/reuse valid footprint geometry, weights and local moments;
  avoid repeated expensive weight evaluation. Match physical contacts rather
  than assume padded slots keep their identity. Measure atomic versus local
  or segmented reductions and handle capacity changes without losing loads.
- **Small patch solves.** Try the admissible all-positive closed-form small
  solve with a constrained fallback. Retain force/moment and friction checks.
  Candidate compaction and local work scheduling may outperform global padding.
- **Load algebra.** Derive fixed centrifugal fields and fuse mass-balanced RHS
  evaluation with contact-wrench reduction. Under the existing inferred-
  acceleration relief, uniform gravity cancels; verify the precise assumptions
  before exploiting this. Avoid repeated full mass products and nodal passes.
- **Quadrants linear methods.** Compare reusable decompositions or structured
  solves where useful with batched sparse/matrix-free Krylov methods. Explore
  geometry/elasticity-aware multigrid or coarse corrections, rigid near-null
  modes and thickness-aware block smoothers where they address conditioning.
  Share immutable operators/preconditioners, while maintaining independent
  per-environment convergence and correction. Small dense cases can help tune
  kernels but should not prescribe a nonscalable large-model implementation.
- **Temporal reuse.** Warm-start from prior displacement and test small history
  spaces, including h=0. Bypass prediction/history work when its measured cost
  exceeds saved solve work. Append only for corrected environments, cache Gram
  updates, remove duplicate residual evaluations and preserve reset ownership.
- **Device selection.** Compare compact, masked and chunked failed environments,
  including selection/packing/history overhead. Native iterative kernels may
  consume device counts directly. Exploit that without forcing a host roundtrip
  merely to reproduce the old external solver's dynamic RHS interface.
- **Precision.** Test FP32, scaling and mixed-precision correction, with FP64
  residual/peak comparisons as needed. Explore supported Quadrants tiled or
  Tensor Core operations only when they improve the measured path and satisfy
  numerical consistency. CPU TF32 emulation is not device performance evidence.
- **Maximum-stress kernel.** Fuse full-domain affine-P2 corner evaluation and
  reduction without materializing stress fields. Tune connectivity/geometry
  layout, locality, lane assignment, gradient recomputation versus storage,
  block reduction and register pressure. Hierarchical exclusion requires valid
  global bounds and a full-scan fallback, not a fixed hotspot assumption.
- **Scheduling.** Remove unnecessary Python/environment loops, dispatches,
  allocations and host synchronization. Measure supported Quadrants graphs
  and stream/event overlap. Use runtime flags for per-environment work and
  account for dispatch overhead, load imbalance and the tail of hard solves.

New bottlenecks may appear after each change. Keep before/after profiles and
end-to-end ablations, including all-failed frames where temporal methods lose.
Do not count a lower mesh, fewer sampled substeps, wider footprint, looser
tolerance or omitted friction as an implementation speedup.

## Completion evidence

Commit the native feature, normal example and concise numerical tests, with
reproducible low-resolution profiling/benchmark commands and raw summaries.
Explain the final configuration, gains, rejected ideas, residual/peak errors,
memory limits and remaining bottlenecks. Verify optional-feature isolation,
actual Franka contact behavior and device-resident observations. Push to the
task branch. Later physical convergence has its own explicit workload and
report; it is not a prerequisite for this performance milestone.

Consult primary interface references for the installed Quadrants version:

- [Quadrants tensors, layouts and zero-copy](https://genesis-embodied-ai.github.io/quadrants/user_guide/tensor.html)
- [Quadrants interop](https://genesis-embodied-ai.github.io/quadrants/user_guide/interop.html)
- [Quadrants performance](https://genesis-embodied-ai.github.io/quadrants/user_guide/performance.html)
- [Quadrants graphs](https://genesis-embodied-ai.github.io/quadrants/user_guide/graph.html)
- [Quadrants streams](https://genesis-embodied-ai.github.io/quadrants/user_guide/streams.html)
