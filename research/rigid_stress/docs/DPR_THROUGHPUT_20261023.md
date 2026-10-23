# Automatic DPR throughput optimization

Work in progress against remote goal `8cab17283`; the
[complete requirements](DPR_AUTOMATIC_RADIUS_GOAL_20261023.md#immediate-goal-maximize-native-automatic-dpr-throughput)
and [starting measurements](AUTOMATIC_DPR_IMPLEMENTATION_20261023.md) remain authoritative.
The baseline engine is `40775c1d0`. Runtime data use `automatic-dpr-opt-*`
under the established `20261019-qualified-ranking` data root.

## Requirements and evidence to collect

| Requirement | Evidence required | Status |
| --- | --- | --- |
| Preserve original model, force/friction, support and FEM gates | CPU/GPU disk/sector load moments, same-mesh stress, original gates | First candidate submitted |
| Contained-disk analytic moments and refined integration reuse | Independent off-plane disk moments; clipped/edge/vertex controls; overflow fallback | First candidate submitted |
| Material complete-scene speedup, max/full | Paired repeated profiler-off rigid/manual/radius/Hertz measurements | First pilot submitted |
| Useful batch optimum, beyond B64 | Batch sweep with env-step/s, memory and repeat variation | Pending |
| Diverse changing contacts beyond the four tiled conditions | Seeded position/yaw/friction bank and equivalent-input comparisons | Bank generator prepared |
| Release/recontact, reset, fusion, nonconvex identity | Focused moving-contact controls, partial reset, extended low-batch cycle and retained failures | Pending |
| Continue through measured bottlenecks | Final stage/device/launch breakdown; evaluated candidates and stopping reasons | Pending |
| Deliver integrated option and practical commands on branch | Final settings, executable commands, compact evidence and remote verification | Pending |

## Numerical method

`b9b383e52` extends the existing optional contact-moment cache to refined
supports. A contained Hertz section contributes exact degree-two Gram and
degree-three P2/affine-pressure moments. Its central covariance is
`R_section**2 / 5 * (I - n n^T)` and its mass is
`2*pi*R_section**3/(3*radius)`. Odd central moments vanish. The coefficient
acceptance bound still proves positivity over the complete support; constrained
fits and intersected/multiface supports retain their numerical integration.
Scratch is rebuilt for each update, with the original overflow fallback.

The candidate also rejects exact face misses before packing tasks and skips
zero-weight fans during scatter. An initial refined fit is not recomputed after
successful moment construction. A failed already-refined fit remains failed.
`66ec61def` validates this path on CPU/GPU and the Franka workload. `2952043cd`
then shares each face/ball section across its samples. `a9d5a51a9` enables
moment reuse by default for automatic-radius Hertz pressure on CUDA, including
small batches. The optional original route and complete overflow fallback stay
available. Manual DPR's small-batch default is unchanged.

The earlier failed bitwise audit and physical coarse-mesh discrepancies remain
in the starting evidence. Independent runs will use the established numerical
error criteria; the previous failure will not be relabelled as passing.

## Measured optimization stages

All numbers below use the unchanged four-condition bank, level-1 FEM (480
tetrahedra, 2,430 DOFs), fp64, q16 and RTX PRO 6000 Blackwell. Primary timing
excludes the profiler and diagnostic exports. Each paired pilot uses one GPU.

| B64 full live variant | Median ms/step | Campaign |
| --- | ---: | --- |
| Original `40775c1d0` | 4.6232 | `automatic-dpr-opt-pilot-v3` |
| Early-exit fit, original integration route | 4.6186 | same allocation |
| Analytical moments and refined-moment reuse, `66ec61def` | 3.3577 | same allocation |
| Repeated moment-reuse control | 3.3662 | `automatic-dpr-opt-pilot-v4` |
| Shared face sections, `c3671fe62` | 3.2630 | same allocation |

The early-exit change alone has no material scene benefit. Refined-moment reuse
provides the main gain. Face-section reuse supplies another roughly 3% scene
improvement. These rows retain three repeats each and zero invalid environment
steps; the final default configuration is checked separately.

At the loaded B64 snapshot, cached anchor/pressure/scatter wall time falls from
0.587 to 0.460 ms after sharing sections. The separate pressure/fit subdivision
falls from 0.439 to 0.312 ms; cached scatter remains 0.074 ms. There are 16
contained Hertz faces and 560 clipped faces, including 224 exactly zero-area
fans. This profile motivates the subsequent ray-reuse candidate `db84114d1`.

The original Quadrants kernel profiler omits captured graphs. Historical
`stage_device_ms` values of zero therefore mean unrecorded work, not free
execution. New records call this `stage_recorded_kernel_ms`. Secondary
`QD_GRAPH=0` profiles expose every kernel but change scheduling; they cannot be
subtracted from graph-enabled scene time to infer production launch overhead.
`automatic-dpr-opt-profile-device-v1` records that secondary experiment, with
the explicit environment in its captured workload script.

## Batch search

`automatic-dpr-opt-batches-v1` measures `66ec61def` at B4 through B4096.
`automatic-dpr-opt-batches-v2` measures section reuse at B8192 through B32768.
All available max/full repeats have zero invalid steps. The larger-batch sweep
reaches a clear plateau: B16384 max/full are 49.691/50.668 ms (329.7k/323.4k
environment-steps/s), versus B32768 at 99.215/101.340 ms (330.3k/323.3k).
Doubling the batch beyond B16384 buys essentially no throughput and doubles
per-environment storage and latency. B16384 is the useful throughput choice
for this bank; B4/B64 serve low-latency use. This is a measured workload/device
choice, not a universal optimum.

## Numerical evidence retained so far

- `automatic-dpr-opt-controls-v4` and `-v5`: independent disk/sector P2 loads,
  same-mesh direct FEM stress, CPU/GPU controls, explicit sharp-box statuses,
  cached moments and forced-overflow fallback pass. Each backend's focused
  regression selection has eight passes, including moving nonconvex contacts.
  Exact disk moments agree with independent integration to 2.91e-14 relative
  load error in the CPU control.
- `automatic-dpr-opt-scenes-v2`: radius-only/Hertz 130-frame comparisons,
  partial reset, unfused execution, B4 600-frame controller cycle and B8
  160-frame diverse bank pass. Against the original engine, saved full-stress
  relative differences are at most 9.93e-14 and per-environment nodal-load
  differences at most 3.43e-16. Rigid positions and source contact fields match.
- The 600-frame run spans approach, closing, lifting, translation and opening
  commands, with changing loaded-contact counts. It establishes numerical
  admission; it does not certify successful grasping or physical stress peaks.
- The diverse bank uses eight independent seeded positions, yaw angles,
  friction values, manual-radius controls and staggered approach delays, with
  fresh inverse-kinematics targets. No condition is filtered by stress admission.
- `automatic-dpr-opt-quadrature-v1` rechecks cheaper q8/q10/q12 rules. q8's
  independent DPR edge-load error is 1.55e-9 against the unchanged 1e-9 gate.
  Both q8 and q10 Hertz scene runs retain their step-108 fit rejection, so the
  new analytical path does not erase the earlier q10 failure. Further results
  remain pending; no tolerance is loosened.

The original approximately 86% coarse-FEM physical stress discrepancy and
failed exact-bitwise audit remain in the starting report and evidence packet.
Collision boxes remain sharp. The explicitly selected planar box-edge radius
surrogate is still an approximation; numerical conformance does not establish
a physical sharp-edge surface maximum.
