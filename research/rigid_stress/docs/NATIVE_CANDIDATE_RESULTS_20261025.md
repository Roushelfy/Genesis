# Native candidate integration and tuning, 2026-10-25

This report implements the [native integration goal](NATIVE_CANDIDATE_INTEGRATION_GOAL_20261025.md).
**Work in progress:** the final response-layout comparison and production-path
decisions remain open. The [execution ledger](NATIVE_CANDIDATE_PROGRESS_20261025.md)
preserves the successive trials and rejected numerical configurations.

The baseline is the previously delivered engine `5c44042a4`. Measurements use
NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, the established Slurm image
`1_26`, FP64 scene/state arithmetic, dt=0.01 s and one physical substep. Each
backend is compared with its own mother model and outputs. Shell in-plane
sampled stress and volume three-dimensional sampled stress are different
physical quantities; their values are not ranked against one another.

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

Direct atlas3 gives a small volume module improvement at B8192 (22.568 ms
against 22.775–22.786 ms), but the complete scene takes 47.429 ms against
46.186–46.200 ms. The corresponding full-output scene takes 48.017 ms against
46.792–46.812 ms. Shell atlas3 also loses: max 73.277 ms against
69.993–70.117 ms; full 74.881 ms against 71.590–71.593 ms. Independent and
coarse recovery incur more work and memory. These are complete moving-contact
cycles, rather than conclusions drawn from the frozen module alone.

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

## Physical limitations

The gripper box retains the user-selected ideal sharp edges and existing
planar Hertz surrogate. These controls certify numerical consistency of the
specified discrete model. They do not qualify a continuum sharp-edge maximum,
resolve an unknown fillet, or establish mesh convergence of local contact
stress. Higher-resolution physical convergence remains outside this phase.
