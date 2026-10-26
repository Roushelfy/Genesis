# Native integration and tuning of stress recovery candidates

**Delivered, 2026-10-26:** see the [final report](NATIVE_CANDIDATE_RESULTS_20261025.md)
and [fourteen-cell audit](../evidence/20261025-native-candidates-v1/metadata/requirements-final.json).
The original specification below is preserved.

**User goal, 2026-10-25.** Take all seven GPU optimization candidates
through real native integration, targeted tuning and matched measurement.
Keep the paths that improve useful throughput or the measured
accuracy/resource tradeoff. Revert production paths that remain unfavorable
after reasonable tuning, retaining reproducible code and negative evidence.

This explicitly advances beyond the
[completed candidate study](GPU_THROUGHPUT_20261025.md).
Its winning pressure and fallback changes are the baseline. The earlier
[GPU candidate specification](GPU_THROUGHPUT_FOLLOWUP_GOAL_20261025.md)
continues to supply numerical and measurement requirements, with the
stronger integration/completion requirement below taking precedence.

Use engineering judgment and profiling to choose the architecture and work
order. Keep the implementation in Genesis's existing rigid stress feature.
Production numerical preparation and runtime stay entirely in Quadrants.

## Required native work

| Candidate | Required next step |
| --- | --- |
| Nearest feature mesh per contact with a narrow continuous transition | Build a usable curved-egg atlas path, including compatible geometry, load/output mappings and continuous selection. Support a complete finite footprint, multiple contacts and multiple-region junctions. |
| Same-mesh load aggregation and multiple-RHS batching | Connect selected native contacts to per-(mesh, environment) aggregation and device grouping, then reuse shared factors/responses across environments. Retain the existing single-mesh fast path. |
| Merge displacement before stress recovery | Connect common-space accumulation and one stress recovery to actual native solutions and outputs. Compare with independent recovery and coarse-plus-local accumulation. |
| Complete precomputed stress response H | Integrate actual current-load construction, all supported force/moment/spin channels, required state/acceptance work and public max/full output. Tune dense, sparse-column and useful hybrid application. |
| FP32 response inventory | Test storage and accumulation separately on the integrated response path, including residual checks, corrections/fallback and memory. Keep FP64 as the independent numerical baseline. |
| Maximum upper-bound screening | Connect globally conservative bounds, candidate evaluation and fallback to native max output. Compare with the current fused exhaustive max, including every screening cost. |
| History prediction and failed-environment compaction | Connect sustained prediction to current native loads, acceptance, compact exact fallback, output and lifecycle. Tune against history-off, the existing predictor and the integrated compaction baseline. |

## Both backends are required

Run every one of the seven candidates through native integration and tuning
in both modes:

- midsurface shell elements with physical thickness, translation/rotation
  DOFs, membrane/bending stresses and documented resultants;
- three-dimensional volume FEM of the hollow wall, with its actual
  inner/outer boundaries, through-thickness displacement gradients and 3D
  stress samples.

Use the complete Franka/hollow-egg workflow for both backends, keeping valid
matched baselines for each. Shared implementation and scheduling are welcome,
but one backend's passing result or timing does not complete the other.
Track a seven-candidate by two-backend matrix of integration, numerical,
tuning, module/scene and memory outcomes. Retention and defaults can differ.

Build each backend's own common discretization, load/output mappings,
rigid-mode gauge, inertia relief, H and numerical reference. Common-mother
accumulation happens within a backend; it does not mix shell and volume
solutions. Within each backend, all feature spaces share the mother model's
gauge and inertia relief before projection. Shell work preserves rotational
DOFs, bending and physical-side friction eccentricity. Volume refinement covers the loaded surface and the
necessary thickness interior while preserving 3D coupling.

Report precision budgets, history acceptance and screening bounds separately
for each backend's outputs. Full H storage depends on input channels times
output scalars and precision; do not infer its cost or a winner from the
backend name or displacement DOF count alone. Shell sampled in-plane maxima
and volume sampled 3D maxima have different physical scope; compare each to
its own same-model reference. Volume representation still needs adequate
resolution to qualify local contact stresses.

All fourteen candidate/backend decisions require native evidence.
Prototype-only or unimplemented items remain open in that matrix.

A planar fixture, CPU-built GPU benchmark input, H-only product or
recovery-only product is useful intermediate evidence. It cannot complete a
native integration item by itself. Required mappings, displacement/state
handling and acceptance are implementation work, not grounds for declaring
a candidate inapplicable without trying to solve them.

## Resolve the atlas integration gap

Make compatible curved geometry and common discrete mechanics part of this
task. Build a common mother discretization with Galerkin-compatible feature
spaces, or another independently validated mapping that preserves the
intended physical geometry, work transfer and output semantics. Implement
the missing preparation in the native architecture.

Use one consistent rigid-mode gauge and inertia-relief construction across
contributions. Preserve shell translation/rotation DOFs, both physical
surface sides, friction eccentricity and all supported noncontact loads.
Changing a feature mesh must not silently change mass, rigid-mode relief or
the physical load being compared. Document and measure any deliberate
spatial/geometric approximation separately.

Only reference shape, thickness and material are fixed. Contact positions,
symmetry, contact count, force directions and convexity are not assumptions.
Keep witnessed material-side identity through nonconvex regions, cavities
and thin walls. Cover the entire finite pressure footprint. A coverage
fallback must remain valid and its work/frequency must be included in timing.

Select per contact and merge only equal (mesh, environment) contributions.
Two contacts often use two meshes; additional contacts or junctions may need
more. Narrow continuous transitions must handle ties without a hard two-mesh
cap. Combine tensors in a common frame before nonlinear stress criteria.
Check a far-from-contact maximum, such as a nonconvex link's thin neck.

The native curved mapping and common relief are explicit remaining work.
"Not yet implemented" or "no validated embedding yet" is an unfinished status,
not a successful completion category for this stage.

## Integrate, tune, decide

First establish a minimal correct native path for the seven candidates,
starting with shared atlas geometry/mechanics needed by the first three.
Use small controls for numerical debugging and the real evolving native
scene early enough to expose integration costs.

Then profile and tune the complete path, addressing the actual bottleneck:
load preparation/pressure, grouping, indirect reads, batching, response/solve,
stress output, screening, residual/correction and history maintenance.
Investigate the layouts, tile/block sizes, shared reuse and batch ranges
that materially change the result. A first naive negative timing is a
starting point for diagnosis, not sufficient optimization evidence.

Test standalone changes against the current optimized baseline, then test
promising interactions such as few-mesh grouping plus shared recovery, FP32
response plus sparse/dense scheduling, and predictor plus compact fallback.
Use bounded profiling-driven sweeps rather than a full Cartesian product.
Preserve existing valid measurements; repeat them only when a changed
implementation, workload or new question justifies the cost.

For H, include every state and numerical acceptance obligation required by
the native feature. Optimize how these obligations are computed and shared.
If replacing a mechanism, demonstrate the same acceptance contract; simply
omitting original-operator checks is not an equivalent optimization.
FP32 CPU/H-only agreement does not certify the integrated GPU path.
TF32 can be explored if supported, but an unavailable TF32 API does not
close or block the required FP32 work.

Keep screening globally conservative, including roundoff allowances and
every stress contribution. Include bound construction/application,
ordering/compaction and exact candidate work in its cost. A sampled maximum
keeps the original sampling/resolution limitation.

For prediction, accepted approximate states feed subsequent history. Include
the gate, resets, gathers/scatters, exact fallback and history maintenance.
Evaluate buffer rotation or actual cached Kq where the integrated profile
supports it; existing isolated losses need not be repeated unchanged.
Residual tolerance is not a stress-error bound.

Retain a candidate in the maintained production implementation when it
provides a reproducible useful gain at the declared accuracy and scope.
A specialized regime can justify a supported internal dispatch or a small
documented option; a loss elsewhere must remain visible. A useful gain can
be lower complete-scene time, higher useful batch throughput within memory,
or a clearly measured improvement in the accuracy/cost tradeoff.
Change defaults only when the measured recommendation supports it.

When reasonable native tuning still loses, record a pinned runnable native
variant, its tuning evidence and a matched negative result, then revert the
unhelpful production path. Preserve experiments outside the active hot path.
Keep public configuration small and understandable.

## Numerical and performance evidence

Keep the current DPR pressure/radius law, current-force friction/moments,
geometry and update cadence fixed for equivalent scheduling/precision
comparisons. Preserve original numerical acceptance and lifecycle checks.
Declare separate budgets for atlas spatial approximation, FP32 arithmetic
and history prediction before using their results for selection.
Existing justified budgets/evidence can be reused. Do not widen contact
support, reduce observation frequency or drop required checks to claim an
equivalent-model speedup.

Use low-resolution numerical consistency first. Compare complete tensor
fields, sampled criteria, force/moment transfer, original-operator residual
and rigid modes, with small independent CPU controls. Cover moving patches,
seams/junctions, asymmetric/distant contacts, inner/outer sides, nonconvex
material identity, friction reversal, release/reload, contact/feature changes,
partial resets, checkpoint restoration and environment isolation.

Measure stress-module and full evolving Franka grasp cost, max/full output,
small and useful large batches, and heterogeneous per-environment poses,
contacts and phases. Include all applicable load/mapping/selection, solve,
acceptance, fallback and state costs. Separate setup/JIT, persistent shared
inventory, per-environment working/output memory and steady-state runtime.
Compare with pure rigid, current shell/volume and each candidate's matched
native baseline. Report admitted versus attempted rates, failures and repeat
distributions; use paired/interleaved comparisons where practical.
Keep policy inference and host export accounting explicit.

Preserve the heterogeneous volume frame-221 rejection and diagnose it if it
blocks a required comparison. A valid shell/native control can advance the
candidate work while the rejected volume trajectory remains separately open.
Neither skipping failing frames nor freezing contact motion creates an
admitted complete-cycle rate. A new robustness repair requires its own
numerical control and should be measured separately from scheduling gains.

High-resolution physical convergence and broad physical-reference campaigns
remain behind this integration/throughput phase. Same-model numerical
agreement does not certify local three-dimensional or continuum contact peaks.

## Completion and handoff

For each of the seven candidates in each of the two backends, deliver one of:

- a native integrated, validated and tuned path retained for a measured regime;
- a reproducible native integrated/tuned experiment with enough numerical,
  module, full-scene and memory evidence to justify reverting it.

An incomplete mapping, standalone simplified prototype, missing production
acceptance work or unmeasured integration cost remains open. Real hardware
or mathematical blockers must be reported explicitly as blockers, rather
than relabeled completed evaluation.

Deliver an itemized integration/tuning ledger, small reproducible command
sets, source pins, numerical controls, profiles, paired measurements and a
concise recommended configuration. Retain negative results, historical
artifacts and unrelated work under the existing storage/reproduction policy.
The agent should carry the work through these decisions without waiting
for high-resolution convergence or approval for routine implementation.
