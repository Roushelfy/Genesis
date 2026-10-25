# Native candidate integration ledger

Active specification: [native integration and tuning](NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md),
revision `b8e63b195`. Baseline engine: `5c44042a4`, delivered in `f8a98971f`.
All seven candidates require independent shell and volume native decisions.
Previous prototype timings are supporting evidence only.

| Candidate | Shell | Volume |
| --- | --- | --- |
| Curved feature atlas with continuous selection | Open: common-space implementation | Open: common-space implementation |
| Same-mesh contact aggregation / multiple RHS | Open | Open |
| Common displacement then one stress recovery | Open | Open |
| Complete H including state and acceptance | Open | Open |
| FP32 storage and accumulation | Open | Open |
| Conservative global maximum screening | Open | Open |
| Sustained prediction and compact fallback | Open | Open |

## Numerical budgets declared before selection

Use each backend's own FP64/q16/history-off numerical baseline, unchanged
pressure law, original residual gate (relative `1e-8`, absolute `1e-11`),
force/moment tests and lifecycle behavior. No residual tolerance is interpreted
as a stress-error certificate.

- Exact scheduling: complete generalized load relative error below `1e-10`,
  complete stress tensor below `1e-8`, original wrench tolerance `1e-8`.
- Atlas spatial approximation: common-point complete tensor error below 10%
  and sampled peak error below 20% against the covered joint mother mesh.
  Continuity, complete footprint and witnessed material identity are separate
  mandatory checks. The original native acceptance gate remains mandatory.
- FP32 inventory: tensor and sampled peak relative error below `1e-4`,
  including corrections/fallback and the unchanged acceptance gate.
- Sustained history: tensor and sampled peak relative error below `1e-5`,
  with accepted predictions fed back into history.
- Shell membrane, bending and transverse resultants are checked separately,
  using the corresponding tensor budget for each channel; mixed-unit aggregate
  norms cannot hide a failed channel. Volume comparisons include all current
  3D stress samples and through-thickness DOFs.
- Bounds must cover all loads and roundoff. Screening returns the same sampled
  maximum as exhaustive recovery, within the exact scheduling budget.

## Work order and required evidence

1. Establish backend-specific mother spaces, Galerkin-compatible feature
   maps, common gauge/inertia relief and native per-contact grouping. Use
   independent small CPU controls followed early by evolving Franka scenes.
2. Integrate common/independent recovery, full H, precision, bounds and
   sustained prediction into actual load/state/acceptance/output paths.
3. Profile and tune meaningful layouts, batching and useful interactions.
   Measure module and complete-scene max/full, small/large heterogeneous
   batches, setup/JIT, shared/per-environment and peak memory.
4. Record a retained regime or a pinned runnable native tuned negative for
   each of fourteen cells. Missing native work remains open.

The heterogeneous volume frame-221 pressure-fit rejection remains an open
robustness item; preserve it and validate any repair separately from scheduling.
Keep all earlier negative experiments and protected sources. New runtime data
and job receipts use the explicit run directory in `DATA_LOCATION.md`.
