# Native candidate integration ledger

Active specification: [native integration and tuning](NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md),
revision `b8e63b195`. Baseline engine: `5c44042a4`, delivered in `f8a98971f`.
All seven candidates require independent shell and volume native decisions.
Previous prototype timings are supporting evidence only.

| Candidate | Shell | Volume |
| --- | --- | --- |
| Curved feature atlas with continuous selection | 140-frame native agreement; geometry/lifecycle/tuning open | 140-frame native agreement; geometry/lifecycle/tuning open |
| Same-mesh contact aggregation / multiple RHS | Native grouping passes; tuning/measurement open | Native grouping passes; tuning/measurement open |
| Common displacement then one stress recovery | Merged path passes; independent/coarse comparisons open | Merged path passes; independent/coarse comparisons open |
| Complete H including state and acceptance | FP64/correction pilot passes; tuning/measurement open | FP64/correction pilot passes; tuning/measurement open |
| FP32 storage and accumulation | Both precision pilots pass; tuning/memory open | Both precision pilots pass; tuning/memory open |
| Conservative global maximum screening | Native max pilot passes; tuning/measurement open | Native max pilot passes; tuning/measurement open |
| Sustained prediction and compact fallback | 140-frame feedback passes; lifecycle/long-cycle/tuning open | 140-frame feedback passes; lifecycle/long-cycle/tuning open |

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

## Initial implementation and unresolved checks

`native-candidates-v0` (`0559287e2`) and `v1` (`1561e94eb`) preserve the
first native curved atlas and its kernel-wrapper correction. Regions live on
the original physical surface graph, without connections across disconnected
inner/outer sheets. Smooth weights interpolate graph-distance fields on the
witnessed face. Every intersecting face AABB must be covered by every selected
region; otherwise the complete contact uses the mother response.

The initial feature spaces use `U_i = K_g^{-1} R E_i`, where the mother gauge
and relief are shared before the regional boundary injection. Consequently
`U_i f_i` is a Galerkin solution in `range(U_i)`, with exact global extension
through the wall. This is a condensed response-space atlas, not independently
remeshed geometry. Equal region/environment loads aggregate during native
scatter. Device grouping applies the shared response columns, and the initial
path merges displacements before the original residual/correction and stress
passes. Spatial truncation is absent in this first control. Independent and
coarse-plus-local recovery, lifecycle coverage and tuning remain open.

The first pilot failed at import because Quadrants kernels cannot have default
arguments; `v1` moves these defaults to Python wrappers. Both original failures
are retained. No numerical success or performance decision follows from them.

The preserved volume frame-221 load reproduces status 3 on the CPU with the
unchanged q16 geometry and pressure law. Independent LP on its 1,536 positive
quadrature samples reports infeasibility; least-squares pressure fitting cannot
reach the original tolerance. This is a sampled-problem result, not a proof of
continuous physical infeasibility. Local quadrature diagnostics are running.

## Native source v6 and first passing controls

Frozen `native-candidates-v6`, source `dc6c9e4ea`, fixes explicit optional
output masks and uses an actual two-frame state container. Earlier v0-v5
import/compile/allocation failures remain on data. Small native module controls
pass for both backends, including three grouped contributions, H, predictive
feedback, release and screened max. Independent CPU displacement/response
errors are below `3.8e-11`; common projected work is below `7.1e-12`.

`native-pilot-v6` and `response-pilot-v6` validate 140 evolving Franka steps
with two environments, every-step complete fields and original acceptance:

| Path | Shell worst field/peak error | Volume worst field/peak error |
| --- | --- | --- |
| Exact grouped atlas / merged displacement | 3.41e-14 | 4.73e-13 |
| H FP64, including forced original correction | 1.23e-12 | 3.02e-11 |
| H FP32 storage / FP64 accumulation | 2.99e-8 | 2.79e-8 |
| H FP32 storage / FP32 accumulation | 5.15e-7 | 3.01e-7 |
| Conservative H max screening | 2.58e-15 | 4.80e-13 |
| Two-frame sustained prediction | 8.86e-9 | 8.85e-9 |

Shell numbers include separately normalized N/M/Q. FP32 applies to H only;
q, residual/correction and factors remain FP64. These are numerical pilots,
not completed tuning or retention decisions. Duplicate FP64 correction array
element `749767_1` was cancelled because the identical checks are retained in
`749768_0`; no failed frame was removed. Geometry controls preserve both initial
worker dependency and options-bound failures before their corrected runs.

## Separate pressure quadrature robustness result

Angular fan quadrature preserves the support radius and pressure law. On the
saved frame-221 load, angular q16, q32, q64 and q128 all pass the original fit,
wrench and full-field residual checks; edge q64/q128 also pass. Against angular
q128, angular q16 has generalized-load error `2.312e-6`, tensor error `1.213e-6`
and peak error `4.513e-7`. It fails the predeclared `1e-6` load budget despite
passing the `1e-4` field budget. Angular q32 reduces these to `9.263e-7`,
`4.857e-7` and `1.821e-7`; q64's load error is `8.861e-8`. Edge q128 agrees
with angular q128 to `1.102e-8` in load, while edge q64 is `2.628e-5` away.
Keep all refinement results; do not widen the budget to admit q16.

`heterogeneous-angle-v6` completes all 600 evolving volume frames with angular
q16. This establishes trajectory robustness only. The qualified q32 variant
still needs its full-cycle control and separate cost measurement. Equivalent
scheduling comparisons keep their original q16 edge baseline where admitted.
