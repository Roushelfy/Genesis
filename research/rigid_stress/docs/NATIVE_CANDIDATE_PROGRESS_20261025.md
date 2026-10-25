# Native candidate integration ledger

Active specification: [native integration and tuning](NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md),
revision `b8e63b195`. Baseline engine: `5c44042a4`, delivered in `f8a98971f`.
All seven candidates require independent shell and volume native decisions.
Previous prototype timings are supporting evidence only.

| Candidate | Shell | Volume |
| --- | --- | --- |
| Curved feature atlas with continuous selection | Geometry, 600-frame, module/scene and memory measured; native tuned loss, cleanup pending | Inner/outer/nonconvex and 600-frame controls pass; module gain loses in scene, cleanup pending |
| Same-mesh contact aggregation / multiple RHS | Native grouping and compact/direct layout measured; scene loss, cleanup pending | Native grouping and compact/direct layout measured; scene loss, cleanup pending |
| Common displacement then one stress recovery | Merged/independent/coarse controls and full comparisons measured; cleanup pending | Merged/independent/coarse controls and full comparisons measured; cleanup pending |
| Complete H including state and acceptance | FP64/correction and 600-frame controls pass; final input-major layout trial running | FP64/correction and 600-frame controls pass; final input-major layout trial running |
| FP32 storage and accumulation | Both precisions validated and tiled/module/memory measured; final layout trial running | Both precisions validated and tiled/module/memory measured; final layout trial running |
| Conservative global maximum screening | 600-frame/distant-peak controls and block tuning complete; final layout trial running | 600-frame/distant-peak controls and block tuning complete; final layout trial running |
| Sustained prediction and compact fallback | Stricter guard passes both 600-frame lifecycle banks; final scene bookend pending | Stricter guard passes both 600-frame lifecycle banks; final scene bookends pending |

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

The original heterogeneous volume frame-221 pressure-fit rejection is retained.
The angular/q128 repair now passes the complete-bank controls below and its
substantial cost is measured separately from scheduling.
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

## Native alternative recovery and coverage diagnosis

Frozen v7 (`28b98e896`) adds independent regional tensor recovery, with spin,
correction and predicted-state remainder included before the nonlinear criterion.
It also adds an eighteen-direction common Galerkin coarse space and exact
complementary regional response banks. Both backends pass the small native
controls for these paths. Coarse reconstruction errors are below `2.0e-16`;
the shell/volume Galerkin work errors are `2.07e-14` and `6.40e-12`.
The complementary responses retain their global mother extension; this is
not a spatially truncated local solve.

The q32 curved/valley/U controls pass load, full-field, release, reordering and
friction-reversal checks. The volume egg and both nonconvex fixtures exercise
three or more contributors. The shell-egg junction correctly fails that coverage
assertion: all contacts fall back because opposite-sheet AABBs overlap. A
conservative vertex-slab exclusion is being validated to remove this unnecessary
fallback while preserving complete finite-support coverage. The failed assertion
remains in `atlas-geometry-v7` and is not counted as a passing junction test.

Both backends' merged-atlas and H FP32-storage/FP64-accumulation checks now pass
all 600 evolving frames. History lifecycle and FP32 accumulation/screening
full-cycle controls continue. Angular q32 completes the heterogeneous volume
cycle and passes the separate saved-frame load/field budget. Initial native
module timing and actual CUDA traces are running at B64 and B8192; no candidate
has a final retention decision yet.

## Completed controls and profile-driven tuning

All v7 600-frame controls now pass. Worst full-field/peak errors are:

| Path | Shell | Volume |
| --- | --- | --- |
| Merged atlas | 3.515e-14 | 4.729e-13 |
| H FP32 storage / FP64 accumulation | 3.931e-8 | 3.847e-8 |
| H FP32 storage / FP32 accumulation | 1.059e-6 | 5.962e-7 |
| Conservative screened maximum | 7.204e-15 | 6.027e-13 |
| Two-frame prediction, including lifecycle changes | 2.286e-7 | 2.275e-7 |

Both history controls include partial reset and checkpoint restoration, accepted
prediction feedback and unaffected-environment checks. Loaded-frame hit rate is
22.979%; frozen-snapshot hit rates are not used to claim complete-scene speed.

The conservative face-slab exclusion in v8 fixes the shell junction fallback.
V9 (`274b2d62e`) additionally compacts grouped loads to authored boundary rows
and avoids clearing inactive selector scratch. All five geometry controls pass:
shell egg/valley, volume egg/U and a separate volume egg with its complete
tetrahedral exterior boundary, including explicit inner/outer loads. Junctions
have three or more contributors. Complete-field errors are below `2.0e-13`.
V9 also fixes the failed v8 H dtype compilation; all six small precision/backend
controls pass. Failed v8 runs remain available, with cancelled duplicate work
recorded in `meta/response-v8-stop.json`.

The v6 module pilot is complete on NVIDIA RTX PRO 6000 Blackwell Server Edition
GPUs (96 GB), with five repetitions and matched baseline bookends. These are
native stress-module timings at one frozen contact snapshot, not complete-scene
rates. At B8192, shell/volume baseline medians are `53.549 / 23.143 ms`; atlas
is `56.488 / 25.312`, H FP64 `79.918 / 42.636`, H FP32/FP32 `63.661 / 31.440`,
screening `54.923 / 30.752`, and two-frame prediction `52.640 / 24.432`.
The B64 paths also lose to baseline. Actual moving-step CUDA traces identify
grouped-buffer clearing and selection, H multiplication, screening evaluation,
and history append plus duplicate residual work as targets. They do not omit
load fitting or numerical acceptance from accounting.

Frozen v10 (`8956740bf`) tests bounded H output tiles (1/2/4/8), conservative
screening blocks (32/64/128), rotating two-frame buffers and exact failed-list
residual work for large batches. The original residual thresholds are unchanged.
`native-kernels-v10`, `atlas-geometry-v10`, `complete-atlas-v10` and
`module-atlas-v10` are in progress. Full evolving-scene, tuned module and memory
comparisons remain required before any of the fourteen retention decisions.

## V10 results and follow-up controls

All fourteen v10 small kernel cases pass (both backends, tiled FP64/FP32,
dense/hybrid and B1024 mixed prediction hits/misses). All five geometry cases
also pass. In the nonconvex shell and volume controls, the sampled global peak
is respectively at least `12.09` and `14.47` contact radii from the nearest
contact. Conservative screening agrees with exhaustive recovery there.
The eight complete atlas controls and fourteen complete response controls
pass all 600 frames, including heterogeneous inputs. Their volume heterogeneous
q32 evidence is same-model scheduling/precision evidence, subject to the
stronger quadrature finding below.

V10 history passes uniform 600-frame lifecycle controls but fails the
heterogeneous displacement check at frame 579: shell `1.0653e-5`, volume
`1.1914e-5`. At this frame the tensor errors are `7.974e-6` and `8.416e-6`;
the stricter complete-state control is not relabeled passing. Both failures
remain in `complete-history-v10`. V11 (`7bbfa4faa`) keeps the original residual
gate and adds a predictor-only guard at one tenth of its relative/absolute
thresholds, reusing the already computed original-operator residual. Exact
solve/correction thresholds remain unchanged. Its numerical and hit-rate
effects still need validation.

V10 module tuning brings the volume three-region atlas to `22.998 ms` against
its `23.075 ms` initial baseline bookend at B8192; this tiny difference alone
does not establish a gain. Shell atlas remains slower. Independent tensor
recovery and the coarse complement cost more than merged recovery. H FP32/FP32
tile 4 improves the shell/volume module medians to `57.033 / 25.872 ms`, still
above their matched baselines. Tile 8 and FP64-accumulation tile 4 incur worse
costs; screening blocks 64/128 also lose to block 32. Complete-scene and final
bookend results remain pending. V11 therefore additionally tests direct
common-displacement accumulation, eliminating the regional displacement
allocation for merged recovery while retaining it for independent recovery.

The broader four-keyframe quadrature control reproduces the saved q32 fields
within `1.1e-13` and rejects q32's load accuracy: errors against q128 range
from `1.496e-6` to `2.807e-6`. Q64 is below `1e-6` at these four frames, but
frame 599 is close to the limit. The q256 reference initially switches some
contacts from the clipped fan to ordinary face quadrature under the existing
spacing heuristic, causing a non-monotone `6e-5`--`7e-5` load discrepancy.
This failed reference attempt is retained. A fixed-fan q128/q256 control
confirms that switch for the first two frames (q128 refines all 8/25 active
contacts; q256 automatically refines only 6/18). Fixed-fan q128/q256 load
differences are `1.03e-7` and `6.52e-8` there. Remaining fixed-fan controls
and a complete native robustness/cost trial are still required; q32 is no
longer called qualified for the whole heterogeneous comparison.

## Complete-bank quadrature and response layout follow-up

The stronger two-frame predictor guard passes all 600 heterogeneous frames in
both backends, including partial reset and checkpoint restoration. The v10
frame-579 displacement failures remain retained. V11's direct merged atlas
passes the six small kernel controls and all five geometry controls, including
complete volume inner/outer boundaries and distant nonconvex maxima.

V12 (`b8b3ac7ed`) makes explicitly selected angular pressure quadrature keep the
clipped fan at every supported degree; ordinary edge/q16 defaults stay unchanged.
The first four conditions were insufficient to select q64: on the complete
64-condition bank, its load errors at frames 129 and 400 are `1.0452e-6` and
`1.0667e-6`, exceeding the unchanged `1e-6` budget. Those failures and the earlier
automatic q256 path-switch failure remain in the data directory.

Angular q128 completes all 600 frames for all 64 conditions. The independent
native-to-fixed-fan-q256 audit (`heterogeneous-bank128-audit-v12`, job 750167)
passes all four keyframes, with worst load `2.2989e-7`, displacement `2.2487e-8`,
tensor `8.8433e-8`, and sampled peak `9.1315e-8` errors. Mechanical inputs are
bitwise unchanged; radii satisfy the pre-existing `rtol=1e-12` contract, with
bitwise comparisons recorded separately. The earlier audit requiring bitwise
radii failed on a one-ULP difference and remains preserved. No load or field
budget was widened. Final heterogeneous volume numerical and timing controls
therefore use angle/q128, with separate repair-cost comparisons against the
unqualified q32/q64 paths. This is quadrature consistency, not continuum peak
qualification.

V11 module tuning gives volume atlas3 direct accumulation `22.568 ms` at B8192
against baseline bookends `22.786 / 22.775 ms`; the complete scene must determine
whether that small snapshot difference is useful. Shell atlas3 remains slower.
H FP32/FP32 tile4 block64/128/256 has no material block-size gain. Its measured
CUDA product uses 80 registers/thread and an estimated 50% occupancy; these
trace facts do not establish spilling. V13 (`d5641351a`) transposes H and its
bounds to input-major storage so adjacent output samples are contiguous during
screening and tiled evaluation. Numerical formulas, input order, FP64 state,
original-operator gates and correction fallback are unchanged. Small native
controls precede matched layout timings. All fourteen retention decisions
remain open pending complete evidence and final production-path cleanup.
