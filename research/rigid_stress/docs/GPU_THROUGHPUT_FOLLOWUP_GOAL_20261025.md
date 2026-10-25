# GPU throughput follow-up for native rigid stress recovery

**Active user goal, 2026-10-25.** Evaluate and implement the applicable GPU
optimizations discussed after the CPU feature-mesh experiments. Maximize
useful multi-environment throughput inside Genesis's existing rigid stress
feature. Keep production numerical preparation and runtime in **Quadrants**.

Use engineering judgment on implementation and tuning. The candidates below
are a work list to evaluate, not a requirement to stack every algorithm or
expose a separate public option for each. Keep measured winners; document
negative results, existing equivalents and justified inapplicability.

## Starting point and priority

Read [GOAL.md](../GOAL.md), [the delivered native shell report](SHELL_DPR_NATIVE_20261024.md),
[the volume throughput baseline](DPR_THROUGHPUT_20261023.md), and
[the existing atlas CPU study](../atlas_study/README.md).

The native shell already has packed surface-load response, recovery fusion,
fixed-geometry reuse and max reduction without allocating full fields.
The current report identifies pressure integration and constrained correction
as substantial costs, especially at large batches. Establish a fresh profile
of the actual branch before choosing work. Extend existing mechanisms and
measure incremental gains rather than rebuilding them under new names.

The CPU atlas experiments motivate candidates; they do not establish native
DPR equivalence, GPU speedups or RL rates. Reproduce a small matched CPU control
when the source for a separately reported experiment is unavailable.
Keep the production support-ball/pressure law and shell surface geometry.
The projected material-sphere Hertz cap used in a CPU fixture is a different
load adapter, not a drop-in replacement.

Only reference geometry, material and thickness are fixed. Preserve arbitrary
moving/asymmetric convex and nonconvex contacts, arbitrary contact counts and
surface sides, normal forces, friction and eccentric moments. Smooth contact
motion permits reuse; correctness also covers release/recontact, shocks,
contact identity changes, partial reset and checkpoint continuation.
Retain volume and shell backends and pre-build max/full output selection.
The main performance scene remains the complete Franka/hollow-egg grasp.

Prioritize low-resolution numerical consistency, detailed profiling and
throughput work. High-resolution physical convergence campaigns remain behind
this phase. Preserve qualified results and their physical limitations.

## Candidate optimization work

### 1. Device-resident data, load preparation and batching

Share geometry, operators, factors and response data across identical links
and environments. Keep current contacts, source preparation, grouping,
recovery and output on device; avoid per-environment Python loops, host
readbacks and synchronization for scheduling decisions. Reuse the existing
native collision provenance, witness/curvature/radius and geometry/moment
work. Profile pressure integration, constrained fit and scatter explicitly.

Evaluate shared source construction, elimination of duplicate and zero work,
same-link/same-environment load aggregation, batched multiple RHS, and grouping
by feature mesh or response operator. Compare grouping cost and load balance
against the work saved. Sweep useful block/tile sizes and environment batches;
use persistent work buffers and existing graph-compatible mechanisms where
they improve measured performance.

Evaluate cheaper quadrature, analytical or contained-support integration,
closed-form surface inversion and fixed-geometry caching where still useful.
Existing native implementations count as the starting point. A faster rule
must pass same-load field, wrench and acceptance checks; pressure/support
changes belong to a separately labeled model comparison.

### 2. Few feature meshes per contact, with continuous transitions

Compare the existing continuous blend, nearest-one-per-contact, top-two
diagnostic and nearest-one with a narrow continuous overlap. For each contact,
assign only that contact's load to its selected mesh or weighted meshes.
Merge contributions sharing (mesh, environment) before solving. Combine
stress tensors in a common physical frame before computing von Mises or a
maximum; adding independent maxima is incorrect.

Two contacts will often need at most two distinct meshes. This is not a
global two-mesh cap: arbitrary additional contacts and finite patch coverage
must remain supported. Allow three or more contributors at multiple-region
junctions, or use another demonstrably continuous construction. Hard top-two
truncation does not guarantee continuity.

Use the full finite load footprint, witnessed material side and nonconvex
surface identity to establish local refinement coverage. Euclidean proximity
alone cannot select across a cavity or thin wall. Fall back to a covered
mesh/union/global route when needed, and include its cost and frequency.

Measure full-field accuracy, sampled maxima, temporal jumps, refinement
coverage, factor/response storage and preprocessing cost as the library size,
refined core/halo size and transition width vary. Keep a small useful sweep,
including the existing CPU choices where reproducible, rather than assuming
that more library meshes or fewer selected meshes always wins.
Preserve global elastic coupling and common inertia-relief/gauge semantics.

### 3. Shared recovery and output fusion

For a common-mother embedding with linear stress operator S, evaluate

    q_mother = sum_i T_i q_i
    sigma = S q_mother

instead of independently recovering every contribution. Compare this with
coarse-coordinate accumulation plus local-bubble scatter. Verify the common
embedding, output locations and frames; independent nonnested meshes require
a validated mapping before this identity applies.

Fuse source projection, recovery, stress arithmetic and output where it
reduces measured memory traffic and dispatch cost without excessive register
or shared-memory pressure. Avoid large temporary tensors and duplicate
transforms.

For max output, stream the requested local stress components into von Mises
and block/global reduction, writing only the requested results and required
validity state. Rotation-invariant criteria can use the local orthonormal
frame. Keep all existing public criteria and resultants correct. Full output
continues to expose its documented samples/components. Benchmark both modes
against the current already-fused native baseline.

### 4. Full response operator and precision

Compare prefactorized multiple-RHS recovery, the existing packed native
surface response, and a complete nodal-load-to-sampled-stress response

    H = S K_gauge^-1 R B
    sigma = H g

where B constructs the physical generalized load and R applies the common
inertia relief. Apply relief exactly once. Preserve top/bottom surface
couples, independent force/moment channels and noncontact loads such as
centrifugal loading; explicitly support or reject a channel outside the
prepared basis. The single-side CPU force basis alone does not cover these.

For H, compare regular tiled dense multi-environment products against reading
only columns touched by current sparse contact loads. Consider hybrid grouping
when it improves locality. Measure zero-work arithmetic, indirect memory
traffic, matrix reuse, output cost and batch occupancy rather than selecting
from FLOP counts alone. Keep the winning production numerical path in
Quadrants.

Measure shared operator storage, per-environment workspace/output, construction
time and scaling with mesh/library size. Full H storage grows with input
DOFs times output samples; a small CPU inventory is not a high-resolution
memory prediction. Avoid storing every patch union or redundant layouts
without a demonstrated benefit.

Separate FP32 storage, FP32 accumulation, mixed/refined solves and any
TF32-capable implementation. Start from the existing FP64 numerical reference.
The native report retains a full-FP32 plate failure and an FP32-inverse option
that also changes the algorithm/fusion path; preserve that evidence and
isolate these effects in new comparisons. CPU FP32 response agreement does
not validate GPU reductions, full FP32 solves or TF32. Use supported Quadrants
capabilities and state unavailable modes honestly. Change defaults only with
matching numerical and complete-workflow evidence.

### 5. Maximum-only screening

First compare the regular fused all-sample maximum against any screening.
For a block b of output samples and a linear H, the candidate bound is

    L[b,j] = max_(p in b) VM(H[p,j])
    U[b] = sum_j abs(g[j]) L[b,j]

using the triangle inequality for the von-Mises seminorm. Evaluate promising
blocks exactly and stop only when every remaining upper bound is below the
current lower bound, with justified floating-point safeguards. Temporal bounds
can use the previous bound plus L times abs(delta g).

All stress contributions must be included in the bound. Preserve global
coverage: a nonconvex link or thin neck may peak far from contact. The result
is the maximum over the declared sampling set, with the same physical
resolution limits as an exhaustive scan. Quantized inventory/rounded bounds
need their own conservative error treatment.

Compare full scanning with screening including bounds, sorting/bucketing,
compaction, history and temporary storage. Group or use fixed rounds where
helpful. A large percentage of skipped samples alone is not a speedup claim.

### 6. History, failed-environment compaction and state maintenance

Evaluate multi-frame prediction under current loads with the current
original-operator residual/acceptance checks. Batch the failed environments
into a multiple-RHS exact fallback and scatter results back. Compare compacted
execution with a regular masked/full batch as the failure fraction changes.
Include predictor, gate, gather/scatter, fallback and maintenance costs.

Compare copying history with rotating preallocated buffers. Evaluate cached
Kq only if the avoided operator work exceeds its added memory traffic; cache
the actual Kq, not the RHS of an approximate accepted solution. Check drift
and near-threshold decisions, and refresh/directly evaluate when needed.

Keep environment-private history and invalidate every affected state on
release, recontact, contact/feature changes, reset and checkpoint restore as
the algorithm requires. Residual tolerance is not a stress-error certificate:
report full-field and peak errors over sustained trajectories with accepted
predictions feeding subsequent history.

### 7. Condensation and reduction candidates

Retain applicable controls for both static-condensation directions, including
the exact coarse/local Schur coupling. Compare their storage, construction,
batched solve and recovery costs against direct prefactorization. Do not drop
cross-patch coupling to claim an equivalent solve.

Global POD performed poorly in the reported CPU controls. Revisit reduction
only with a concrete improved basis/error strategy and a measured need.
Reuse exact unit responses only while geometry and load shape remain fixed;
a force-dependent Hertz radius changes the spatial load shape.

## Measurement and numerical acceptance

Use a short baseline/profile phase, then prioritize the largest measured
end-to-end opportunities. The suggested order is device/load/batch efficiency,
recovery/max fusion and FP32 response alternatives, few-mesh selection, then
screening/history if still justified. Pressure-side work may outrank every
recovery candidate on the existing native branch.

For every candidate, record implemented, already present, inapplicable,
measured win/loss or unavailable, with the evidence/reason. Applicability
judgments and small failed prototypes are valid outcomes; documentation alone
does not substitute for an applicable implementation and benchmark.

Use the same geometry, shell/volume formulation, pressure law, force/friction
history, precision, quadrature, output samples and update cadence for paired
claims. Separate load/model changes, spatial approximation, prediction error
and floating-point error. Define an explicit error budget for each approximate
configuration before tuning and keep its results separate from exact numerical
equivalence. Preserve existing assertions and admission checks; retain exact
fallbacks and make any approximate behavior a clearly documented choice.

Focused controls cover:

- CPU/native same-load complete tensor fields and original-operator residuals;
- force/moment and rigid-mode invariance;
- moving finite patches, seams/junctions, distant/asymmetric contacts,
  inner/outer sides, concave/cavity geometry and a far-from-contact maximum;
- tangential loading and eccentric couples, release/reload, abrupt changes;
- batch isolation, untouched environments, partial resets and checkpoints;
- long enough predictor histories to reveal accumulated error and drift.

Measure both frozen-input stress-module cost and complete evolving Franka
scene cost against pure rigid and delivered volume/shell max/full baselines.
Include approach, closure, frictional motion and release, plus heterogeneous
per-environment poses, contact locations and phases. Report the tiled control
separately from the heterogeneous workload.

Report synchronized batch latency, admitted and attempted environment-steps/s,
peak and persistent/shared/per-environment memory, cold preparation/JIT and
steady-state costs. Sweep small and useful large batches; tune for useful
throughput, not a mandated environment count. Separate optional host exports
and policy inference. Report physical substeps and observation cadence.
Environment-steps/s is not whole-RL-training throughput.

Use warmups and paired/interleaved repeats, preserve distributions and failures,
and pin sources/settings. Profile pressure/source, grouping, solve/response,
stress/max, history/acceptance, launches, copies and synchronization. Distinguish
kernel count from host launch count, and graph-enabled timing from secondary
device profiling. Intrusive stage times are not additive headline costs.

CPU speedups, hardware peak rates and skipped-work percentages are hypotheses
for GPU tuning, not measured GPU or RL performance.

## Deliverable

Deliver the integrated winning path(s), focused regressions, runnable benchmark
commands, detailed stage profiles and paired numerical/throughput/memory
results. Recommend a small understandable set of configurations for max/full,
shell/volume and useful batches. Include the complete candidate ledger with
negative outcomes and remaining limitations, following the existing data and
reproduction policies. Preserve historical evidence and avoid unrelated
branch cleanup or reopening expensive physical-reference campaigns.
