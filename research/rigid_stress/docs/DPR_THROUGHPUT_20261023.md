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

## First candidate

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
The feature remains opt-in with `contact_moment_reuse=True` until paired
measurements support a default choice. No timing or accuracy claim is made
for this candidate until its submitted checks finish.

The earlier failed bitwise audit and physical coarse-mesh discrepancies remain
in the starting evidence. Independent runs will use the established numerical
error criteria; the previous failure will not be relabelled as passing.
