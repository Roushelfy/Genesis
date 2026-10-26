# Native candidate integration and tuning, 2026-10-25

This report implements the [native integration goal](NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md).
**Completed, 2026-10-26:** all fourteen candidate/backend decisions have native
integration, numerical, tuning, complete-scene and memory evidence. None of the
seven incremental candidates provides a reproducible useful gain on the measured
workloads. Their production paths are removed; the previously delivered engine
and optional angular quadrature repair remain. The
[execution ledger](NATIVE_CANDIDATE_PROGRESS_20261025.md) preserves the successive
trials and rejected configurations, and the
[requirement audit](../evidence/20261025-native-candidates-v1/metadata/requirements-final.json)
links the fourteen separate decisions to their receipts.

The baseline is the previously delivered engine `5c44042a4`. Measurements use
NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, the established Slurm image
`1_26`, FP64 scene/state arithmetic, dt=0.01 s and one physical substep. Each
backend is compared with its own mother model and outputs. Shell in-plane
sampled stress and volume three-dimensional sampled stress are different
physical quantities; their values are not ranked against one another.

## Decisions and recommended configuration

Every cell below is a **native integrated and tuned negative result**, with
its runnable implementation retained in Git. The maintained engine source is
`e67a86c18f7cf11db28ccc2df22a3ba3a5ddf03a`; the complete experimental source is
`d5641351a127f674b4a19bf3d921cc3be0200789`. No history is rewritten.

| Candidate | Shell decision | Volume decision |
|---|---|---|
| Continuous curved feature atlas | Remove: 3/6/12-region and direct layouts do not improve complete scenes | Remove: the small module gain disappears in the complete scene |
| Same-mesh aggregation / multiple RHS | Remove additional regional grouping: costs exceed the optimized one-mesh baseline | Remove additional regional grouping: no complete-scene or memory benefit |
| Common displacement before recovery | Remove atlas-specific merge: best direct merge still loses; independent/coarse controls cost more | Remove atlas-specific merge: best direct merge still loses; independent/coarse controls cost more |
| Complete H with state/acceptance | Remove: dense/sparse/hybrid and both layouts retain extra work beyond the required displacement gate | Remove: both layouts lose with required displacement, residual and correction included |
| FP32 H storage / accumulation | Remove: smaller H inventory does not improve time or memory over baseline | Remove: smaller H inventory remains additional to baseline and does not improve scene time |
| Conservative global max screening | Remove: snapshot gain does not survive the moving scene; full output costs more | Remove: max and full remain slower after block/layout tuning |
| Sustained prediction / exact compaction | Remove new two-frame path: stronger guard passes, but scene time loses | Remove new two-frame path: stronger guard passes, but scene time loses |

The earlier shared pressure preparation, fixed Q4 projection reuse and exact
failed-environment compaction remain in production. Removing the additional
regional/predictive candidates does not remove those baseline optimizations.

For the measured four-condition fixture, retain FP64 scene arithmetic,
`quadrature=16`, `pressure_quadrature="edge"` and `history_size=0`; choose shell
or volume for the intended model and output. The API's general defaults are
unchanged. For the qualified 64-condition heterogeneous volume trajectory,
use `pressure_quadrature="angle", quadrature=128`; its extra integration cost
is necessary for the declared load budget on this bank, not a universal
minimum degree for other geometry. The only new maintained public option is
that angular integration choice; the supported degree limit becomes 128.

## Native implementation and scope

The curved atlas uses a common mother discretization and condensed feature
spaces `U_i = K_g^-1 R E_i`. Here `K_g` is the same gauged mother stiffness,
`R` the same inertia relief, and `E_i` the regional boundary-load injection.
The exact globally extended response is a Galerkin solution in `range(U_i)`.
This is a response-space atlas on the actual curved geometry, rather than
independently remeshed eggs. No deliberate spatial approximation is used in
the measured native atlas. The volume extension retains interior wall DOFs
and 3D coupling; shell extension retains translations, rotations, bending and
physical-side friction moments.

Native graph distances on connected material sheets determine continuous
cubic contact weights. All contributors at a junction participate; selection
has no hard two-region limit. A conservative finite-footprint AABB and
vertex-slab test checks complete coverage, with the complete mother response
as fallback. Disconnected inner/outer sheets are not joined by proximity.
The volume geometry controls include the full tetrahedral exterior and
explicit inner/outer loads, beyond the authored outer boundary used by the
Franka fixture.

Scatter aggregates equal `(region, environment)` loads into compact boundary
rows. Device grouping applies shared response columns to multiple RHSs.
Merged recovery directly accumulates common displacement before the original
residual/correction gate and one stress recovery. The independent control
recovers regional tensors and shell N/M/Q before summation and nonlinear
criteria; its remainder carries spin, correction and prediction displacement.
The coarse/local control uses an 18-direction Galerkin coarse response and
its exact complementary regional response. Both controls preserve the same
gauge, relief and full output.

Complete H is prepared in Quadrants using every boundary force/rotational
load channel and six centrifugal channels. It stores complete sampled stress
and shell membrane/bending/transverse resultants. Required displacement,
original-operator residual, correction and fallback are still computed.
Corrected or predicted states use stress recovered from their actual accepted
displacement. H storage and multiplication can independently use FP32;
displacement and its acceptance remain FP64.

Global screening bounds combine all input contributions, a tensor-norm
roundoff allowance and conservative summation margins. A CUDA warp evaluates
the most promising blocks until their remaining bound cannot exceed the
known peak. Full output uses exhaustive H. Bounds are over the original
sampling set, with no contact-neighborhood restriction.

The two-frame predictor extrapolates the preceding accepted displacements,
feeds accepted predictions back into history, rotates buffers and compacts
exact misses. At B>=1024, post-solve residual work uses the same exact failed
list. Its additional guard uses one tenth of the original residual tolerances;
the original solve/correction thresholds remain unchanged. Partial reset and
checkpoint restore clear the relevant history and preserve other environments.

## Numerical controls

Declared budgets are `1e-10` for exact generalized loads, `1e-8` for exact
complete tensors/criteria, `1e-4` for FP32 response output and `1e-5` for
history. History controls additionally enforce the same budget on complete
displacement. Shell N/M/Q are compared separately; mixed units cannot hide a
failed channel. Original relative/absolute residual tolerances are `1e-8`
and `1e-11`, and are not treated as stress-error certificates.
For complete-field trajectory controls, the error is the per-environment
flattened L2 difference divided by the reference L2 norm (floor `1e-30`);
reported values are the worst environment and frame. Sampled maxima are
checked separately. This is not a relative-error guarantee for every nearly
zero individual tensor component.

Small independent SciPy/FP64 controls check gauged K, displacement, selected
response columns, work and rigid-mode relief. Native geometry controls cover
curved eggs, a shell valley, a volume U, complete inner/outer boundaries,
moving seams, at least three contributing regions, friction reversal,
contact-slot permutation, release/reload and distant contacts. Nonconvex
global maxima lie at least 12.09 contact radii away in the shell control and
14.47 radii away in the volume control; screening still matches exhaustive
recovery. All five geometry controls and the direct-accumulation controls pass.

Every-frame 600-step Franka controls compare actual current loads and complete
fields, including heterogeneous conditions and lifecycle operations. V10's
two-frame predictor fails the heterogeneous displacement budget at frame 579
(shell `1.0653e-5`, volume `1.1914e-5`), although tensor errors there remain
below `1e-5`. The stronger guard passes the full heterogeneous lifecycle
controls: worst displacement is `2.9591e-6` for shell and `4.9654e-6` for volume.
Loaded prediction hit fractions are approximately 31.0% and 31.7%, respectively.
Frozen-snapshot hit fractions are not complete-scene gains. V13 repeats the
600-frame tiled four-condition lifecycle control with the stronger guard:
worst displacement remains `2.2859e-7` for shell and `2.2741e-7` for volume.

The final volume angle/q128 heterogeneous controls pass all 600 frames:
atlas tensor error `6.2412e-13`, H FP32/FP32 sampled-peak error `6.4985e-7`,
screened-peak error `4.2762e-13`, and guarded-history tensor error `3.5067e-6`.
All required correction and lifecycle paths retain independent controls.

V13's eight response controls also pass all 600 frames. The FP64 case includes
deliberate response perturbation to force original-operator correction and
recovery from the corrected displacement. FP32/FP32 and screening use the full
heterogeneous bank; all eight cases include lifecycle operations.

| Response control | Shell worst tensor/criterion/resultant error | Volume worst tensor/criterion error |
|---|---:|---:|
| FP64, including correction probe | 1.226e-12 | 3.328e-11 |
| FP32 storage, FP64 accumulation | 3.464e-8 | 3.847e-8 |
| FP32 storage and accumulation | 9.969e-7 | 6.499e-7 |
| Screened sampled maximum | 7.736e-15 | 4.282e-13 |

For the heterogeneous screen controls, mean evaluated block counts are
4.65/7.14 (shell/volume), with a maximum of all 60 blocks. The method includes
that exhaustive case; it does not assume a cheap or local maximum.

## Profile-guided tuning

The native pilot includes pressure fitting, scatter, state, acceptance and
output. Moving-step CUDA traces identify atlas clearing/selection, response
products, screening work and duplicate history residual/maintenance costs.
The tested changes therefore include:

- Atlas regions 3/6/12, compact boundary loads, inactive-weight clearing
  removal, conservative slab coverage and direct displacement accumulation.
- Merged, independent tensor and coarse/complementary recovery.
- H sparse, dense and hybrid scheduling, output tiles 1/2/4/8, block sizes
  64/128/256, separate FP32 storage/accumulation and input/output-major layouts.
- Screening blocks 32/64/128 and coalesced output/bound reads.
- Two-frame buffer rotation, sustained guarded prediction and compact
  post-solve residuals, against history-off and the existing four-column path.

For example, output tile 4 reduces FP32/FP32 module time from the v6 pilot's
63.661/31.440 ms to v11's 54.979/25.856 ms (shell/volume, B8192). The respective
v11 baseline bookends are 52.299–52.459/22.775–22.786 ms, so this substantial
candidate improvement does not itself establish a baseline gain. Tile 8 and
FP64-accumulation tile 4 cost more. Block64/128/256 differences are small.
The tile4 product trace reports 80 registers/thread and 50% estimated
occupancy; it does not establish register spilling.

Input-major H reduces the measured moving-step screening kernel from the
v6 pilot's 4.352/8.834 ms to 2.263/4.781 ms (shell/volume), including bound
evaluation, ordering and exact candidates. Native load packing still costs
0.454/0.157 ms. V13's shell B8192/max module is 52.334 ms against bookends
52.619/52.715 ms; volume is 26.641 ms against 22.845/22.841 ms. This identifies
a small shell snapshot opportunity and a remaining volume loss. Full output
retains the expensive exhaustive product. These intrusive kernel observations
are not added together or substituted for complete-scene wall time.

Direct atlas3 gives a small volume module improvement at B8192 (22.568 ms
against 22.775–22.786 ms), but the complete scene takes 47.429 ms against
46.186–46.200 ms. The corresponding full-output scene takes 48.017 ms against
46.792–46.812 ms. Shell atlas3 also loses: max 73.277 ms against
69.993–70.117 ms; full 74.881 ms against 71.590–71.593 ms. Independent and
coarse recovery incur more work and memory. These are complete moving-contact
cycles, rather than conclusions drawn from the frozen module alone.

The complete B8192 v11 comparison below uses output-major H and the stricter
two-frame guard. Values are median milliseconds per evolving batch step;
baseline ranges show the two bookend medians, not statistical confidence
intervals. Every row preserves runtime acceptance and every-step outputs.

| Path | Shell max | Shell full | Volume max | Volume full |
|---|---:|---:|---:|---:|
| Baseline bookends | 70.559–70.661 | 70.349–70.495 | 45.975–46.037 | 46.713–46.777 |
| H64, tile1 | 87.991 | 113.787 | 58.381 | 58.704 |
| H32 storage, FP64 accumulation, tile1 | 110.153 | 169.073 | 72.834 | 73.351 |
| H32 storage/accumulation, tile4 | 72.132 | 76.773 | 47.788 | 48.448 |
| Screen32; exhaustive H for full | 74.151 | 113.728 | 53.890 | 58.690 |
| Guarded two-frame history | 72.032 | 71.859 | 49.336 | 50.083 |
| Existing four-column history, timing control | 82.766 | 82.668 | 67.562 | 68.381 |

The four-column predictor retains its earlier independent error-budget failure;
its runtime-valid timing is not an accuracy-qualified recommendation. The
input-major H/screen comparison remains separate from this fixed-source table.

The 64-condition bank also loses with the output-major candidates. At B1024,
the complete moving-scene medians are:

| Path | Shell max | Shell full | Volume max, angle/q128 | Volume full, angle/q128 |
|---|---:|---:|---:|---:|
| Baseline bookends | 13.997–14.017 | 14.618–14.629 | 64.768–65.148 | 64.313–64.358 |
| Direct atlas3 | 15.390 | 16.065 | 65.573 | 65.510 |
| H32 storage/accumulation, tile4 | 15.063 | 16.865 | 66.486 | 66.004 |
| Screen32; exhaustive H for full | 15.331 | 22.657 | 67.261 | 67.395 |
| Guarded two-frame history | 14.380 | 15.126 | 66.307 | 65.815 |

The B64 heterogeneous comparisons show the same direction. In particular,
the stronger history guard passes its accuracy budget but does not recover
its added maintenance/gating cost in either backend. Raw repeat distributions
and runtime admission counts are retained in the compact benchmark receipts.

## Final input-major response comparison

All 48 final complete-scene tasks pass runtime acceptance. The final paired
comparison below uses input-major H, FP32 storage/accumulation with tile4,
and FP64 conservative screening with block32. Values are median ms per
evolving batch step; both baseline bookends are shown. The four-condition
bank is tiled at B64/B8192; the 64-condition bank is tiled at B1024, with
angle/q128 used for its volume rows. Numerical qualification is supplied by
the separate complete-field controls, not inferred from runtime admission.

| Backend | Bank / B | Output | Baseline before | H32/32 | Screen32 | Baseline after |
|---|---|---|---:|---:|---:|---:|
| Shell | 4 conditions / 8192 | max | 69.506 | 71.143 | 69.865 | 69.392 |
| Shell | 4 conditions / 8192 | full | 70.855 | 78.232 | 115.400 | 70.842 |
| Volume | 4 conditions / 8192 | max | 46.375 | 48.505 | 49.857 | 46.306 |
| Volume | 4 conditions / 8192 | full | 47.468 | 49.406 | 59.455 | 47.404 |
| Shell | 4 conditions / 64 | max | 6.471 | 6.978 | 7.500 | 6.464 |
| Shell | 4 conditions / 64 | full | 6.529 | 7.060 | 7.331 | 6.505 |
| Volume | 4 conditions / 64 | max | 5.153 | 5.908 | 6.663 | 5.148 |
| Volume | 4 conditions / 64 | full | 5.232 | 5.925 | 5.983 | 5.223 |
| Shell | 64 conditions / 1024 | max | 13.754 | 15.180 | 14.796 | 13.721 |
| Shell | 64 conditions / 1024 | full | 14.374 | 17.686 | 23.162 | 14.349 |
| Volume | 64 conditions / 1024 | max | 64.745 | 66.904 | 67.308 | 64.743 |
| Volume | 64 conditions / 1024 | full | 64.784 | 66.848 | 68.283 | 64.786 |

The shell B8192/max module gain of roughly 0.5–0.7% becomes a complete-scene
loss: screening is 69.865 ms against 69.392/69.506 ms. It is not retained as
a specialized winner. The final heterogeneous comparisons also lose in both
backends. All repeat latencies and attempted/runtime-admitted rates remain in
the [benchmark receipts](../evidence/20261025-native-candidates-v1/metadata/benchmarks.json).

## Measurement contracts and memory

Complete-scene timing uses 600 warmup steps and three independently reset
600-step evolving grasp repeats. Module timing uses a real contact snapshot,
130 warmup steps and five 100-step repeats. Baseline bookends run in the same
allocation. Small/large tiled four-condition batches are B64/B8192; heterogeneous batches
are B64/B1024 with 64 distinct poses/conditions. All stress updates and
required acceptance work are included. Policy inference and research host
exports are excluded. Setup/JIT and intrusive CUDA traces are reported
separately from synchronized wall time. Pure rigid takes approximately
2.35–2.37 ms at B64 and 9.54–9.60 ms at B8192 for the tiled four-condition
bank. For the heterogeneous bank it takes 3.53–3.59 ms at B64 and
4.01–4.05 ms at B1024; different contact histories make these distinct
workloads rather than a common batch-scaling curve.

Memory receipts separate constant native inventory, state, scratch and
outputs. At B8192/full, the direct atlas3 adds about 617/335 MiB of scratch
(shell/volume). Independent recovery adds about 18.2/13.0 GiB. H64 adds about
321/43.5 MiB of constant inventory; H32 approximately halves that additional
inventory, but retains the required FP64 displacement and factors. Its input
and scheduling workspaces remain additional to baseline.

The complete H inventory has the following measured dimensions; each output
entry is a six-component vector, with transverse resultants padded to six.
Input-major storage changes ordering, not this inventory size.

| Backend | Input channels | Output vectors | H64 bytes | H32 bytes | FP64 bounds, block32 |
|---|---:|---:|---:|---:|---:|
| Shell | 1458 | 4800 | 335923200 | 167961600 | 699840 |
| Volume | 492 | 1920 | 45342720 | 22671360 | 236160 |

Thus H storage is `inputs × output vectors × 6 × bytes/component`, plus bounds.
Its cost cannot be inferred from displacement DOFs or the backend label alone.

Sampled GPU peaks (200 ms sampling, including driver/context and reset state)
are lower bounds on instantaneous peaks. Shell baseline/direct-atlas3/H64/H32/
two-frame-history are 13733/14341/14181/14021/13893 MiB. Volume values are
13093/13413/13157/13157/13381 MiB. Allocation granularity hides the smaller
volume H32 saving in sampled peaks; exact native inventories still record it.
These observations do not establish a larger usable batch than baseline.

## Heterogeneous volume pressure repair

The original edge/q16 frame-221 rejection and its sampled infeasible moment
fit remain retained. Angular fan integration preserves radius, support,
pressure profile and contact mechanics. It resolves near-edge small patches
using angle and the corresponding ray Jacobian. Explicit angular mode always
uses the clipped fan, avoiding the high-degree heuristic switch to ordinary
face quadrature. The default edge mode is unchanged.

The isolated failed contact admits q32, but broader keyframes reject its load
accuracy. The first four conditions admit q64; the entire 64-condition bank
rejects it at frames 129/400 (`1.0452e-6`/`1.0667e-6` load error). The attempted
automatic q256 reference also switches integration paths and fails; its fixed
fan counterpart is the controlled reference. All these failures are preserved.

Q128 completes every frame of the full bank and passes four independent
native-to-fixed-fan-q256 field audits. Worst errors are `2.2989e-7` load,
`2.2487e-8` displacement, `8.8433e-8` tensor and `9.1315e-8` sampled peak.
The load and field budgets are unchanged (`1e-6` and `1e-4`). Mechanical inputs
are bitwise equal; radii satisfy the already established relative `1e-12`
contract, with one-ULP bitwise differences recorded separately.

Q128's cost is substantial and is separate from scheduling changes. At B64,
matched volume/max complete-cycle time is about 10.98 ms, versus 4.96/6.18 ms
for unqualified q32/q64. These cheaper rates are attempted timing evidence,
not qualified equivalent-accuracy alternatives. Heterogeneous volume candidate
comparisons all use the same q128 repair and unchanged load/state gates.

## Maintained-path validation

The adopted engine `e67a86c18f7cf11db28ccc2df22a3ba3a5ddf03a` restores the
previous production engine and retains optional angular integration. All 32
focused regressions pass (23 volume, 9 shell, no skips). Its complete
64-condition volume trajectory passes all 600 frames, and its four saved
fields pass the same independent fixed-fan q256 audit.

The default edge/q16 full-output checks compare the original engine,
the maintained engine and the original engine again in each allocation:

| Backend / B | Original before, ms | Maintained, ms | Original after, ms |
|---|---:|---:|---:|
| Shell / 64 | 6.370 | 6.334 | 6.317 |
| Volume / 64 | 5.213 | 5.265 | 5.193 |
| Shell / 8192 | 70.715 | 70.870 | 71.125 |
| Volume / 8192 | 46.782 | 46.894 | 45.248 |

These results do not establish zero overhead from the optional quadrature
branch: the small volume case adds about 0.06 ms. The late large-volume
baseline changes during its own repeats (46.735, 45.099, 45.248 ms), so a
precise overhead estimate from that bookend would confound runtime drift.
The complete repeats are retained; no default-path speedup is claimed.

## Physical limitations

The performance conclusions apply to these tuned native implementations,
coarse discretizations, condition banks and GPU. In particular, the atlas
result concerns exact condensed feature spaces on a shared mother model;
it does not rule out other approximate local refinements or different hardware.

The gripper box retains the user-selected ideal sharp edges and existing
planar Hertz surrogate. These controls certify numerical consistency of the
specified discrete model. They do not qualify a continuum sharp-edge maximum,
resolve an unknown fillet, or establish mesh convergence of local contact
stress. Higher-resolution physical convergence remains outside this phase.
