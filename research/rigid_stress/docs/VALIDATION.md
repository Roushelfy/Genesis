# Acceptance and throughput measurement contract

## Separate three kinds of error

1. **Equation/precision error**: same mesh and same assembled complete RHS,
   compare optimized recovery against FP64 full direct solves.
2. **Load-model error**: force/moment consistency, finite footprint law,
   quadrature, contact pruning and collision/recovery surface differences.
3. **Mechanical discretization/model error**: full-shell mesh refinement and
   applicability of small-strain quasi-static elasticity.

A fast result in (1) does not establish (2) or (3). Always preserve the raw
force/torque diagnostics and full residual, even when only a scalar stress is
returned to the policy. Zero RHS/zero stress needs absolute error reporting.

## Mathematical tests

Required on CPU and again on the target GPU:

- Complete equilibrium including gauge rows, R^T b compatibility and K R.
- Freefall gives zero elastic stress; angular centrifugal loading does not.
- Alternate gauge invariance and world/material-frame rigid pose covariance.
- P2 affine strain patch test and all-corner scalar maximum vs full D B u.
- Random asymmetric normal+tangential loads at arbitrary exterior locations,
  with varying counts, directions, areas and disjoint/overlapping patches.
- Shared-factor serial vs batch; B=1 and multiple B; compact failure masks
  0, 1, sparse, dense, all, and permuted IDs.
- History independence, rank-deficient updates, capacity rollover, cache drift,
  reset of subsets, repeated frames, variable timestep/invalid extrapolation.
- Full FP64/FP32/scaled/refined numerical comparisons; nonfinite/stagnation
  must fail or explicitly recover with an accepted fallback.
- Optional pure contact torque: torque-only loads create nonzero stress and
  reproduce the intended resultant moment without dropping linear constraints.

`check_cpu.py` is a short seed check, not the whole acceptance suite.
`reference/output/rigid_stress_reference.py` contains additional mechanics tests.
The Torch dense-debug test verifies arithmetic on CPU, not CUDA stream/library
behavior. Verify permutations/leading dimensions with nonsquare RHS layouts.

## Actual Genesis Franka/egg tests

Implement automated checks and keep one visual rollout showing the same seed:
approach -> clamp -> lift -> hold/translate/slip -> lower/release. The egg must
be an ordinary rigid link with shell-matched inertia. Do not substitute the
prescribed synthetic contact snapshots or a deformable Genesis FEM entity.

At minimum include:

- One nominal grasp that actually lifts and holds the egg, verified by COM
  elevation and finger-contact history rather than a commanded arm target.
- At least 32 deterministic varied seeds with grasp offsets/orientations,
  friction and finite footprint parameters; record grasp success/failures and
  the nonzero frictional work/forces encountered. No symmetry filtering.
- Abrupt contact birth/death, intentionally slipping cases, a table support
  transition and partial environment resets.
- Contacts where the egg occurs as A and as B; per-env frame/pose consistency.
- Contact resultants vs link net force and moment, shell mass/COM/inertia,
  classical acceleration and the rigid Newton-Euler balance.
- Fixed physical footprint law over at least three full-model mesh levels,
  independently refined quadrature and the stated <=2% proposed criterion.
- No pruned load/contact capacity overflow silently discarded; compare pruning
  options under the same stress model.
- Correct sampling at substeps=1 and multiple substeps. A reported peak over
  substeps must not be only the last-substep stress.

Reject a mapper that changes the intended rigid contact wrench beyond the
published tolerance. First target double-precision integrated force/moment
checks <=1e-8 in normalized force and moment scales for prescribed traction
tests. Live finite-precision rigid comparisons need an explicitly justified
tolerance and units. A dimensionally mixed six-vector norm is not enough.
If the physical footprint model cannot identify the pressure distribution,
report model sensitivity instead of claiming one uniquely correct peak.

## Numerical accuracy profiles

Use `ASSUMPTIONS.md` strict and throughput profiles. On held-out live replay,
save **identical complete RHS** and compare every timed profile to same-mesh
FP64 full solves. Report max/p95/mean relative peak errors (reference >1 Pa),
absolute error near zero, full residuals, fallback/refinement counts, and
no-load frames separately. The final chosen throughput configuration must
pass its declared empirical error profile on all evaluated frames.

Split calibration from validation. Do not tune tolerances on the held-out
reference results. State that empirical validation is not a rigorous universal
output-error certificate; retain runtime full residual and fail-safe correction.

## Benchmark scopes

Report all four distinctly:

| Scope | Included |
|---|---|
| Rigid baseline | control/actions, Genesis step, reset bookkeeping; no stress |
| Recovery microbenchmark | already assembled RHS -> solve/predict/history/full peak; device timing |
| Live rigid + stress | control/actions, rigid step, extraction, mapping, body/inertia loads, all requested recoveries, max observation, reset bookkeeping |
| Policy + rigid + stress | same as live plus batched policy forward and observation/action processing |

Use a fixed small Torch MLP policy or realistic project policy, state its
architecture and observation dimension, and compare matched baseline/augmented
loops. The stress scalar must stay on GPU. A scripted action loop measures
simulator throughput, not learned policy quality or completed PPO training.
Optional PPO updates can be reported as a separate rollout+learning benchmark.

Main metric:

```text
aggregate env transitions/s = counted environment steps / elapsed wall time
batch steps/s = simulation batch steps / elapsed wall time
per-env average rate = aggregate rate / B
```

The same term FPS must not stand for all three. With a 50 Hz policy and
physics substeps, report policy transitions separately from physical substeps
and stress recoveries. Include all reset/packing/fallback/history costs. Do not
count warmup as steady-state samples or exclude slow failure frames.

Record actual GPU name (Ada vs RTX PRO/Blackwell), driver, CUDA/cuDSS/cuSPARSE,
Torch/Quadrants versions, precision, repo/base commits, CPU, B, mesh/DOFs/factor
nnz, history capacity, peak VRAM, dt/substeps, contact count/radius distribution,
success fraction, numerical error profile and sampling contract.

Warm up JIT, factor analysis, graphs and pipeline; reset histories before the
actual complete grasp measurement if warmup contaminated them. Time >=10 s
or enough repeated complete grasps, >=3 repetitions with synchronized CUDA
boundary events plus wall-time total. Publish mean and dispersion/p50/p95;
throughput is total transitions / total elapsed, not inverse median latency.
Build/factor/JIT/asset time is reported separately. No visualization in timed
runs; keep rendered examples separately.

Sweep B from {1,8,32,128,512,2048,8192} until memory/latency limits; finer
meshes may need smaller B. Sweep solve chunks, h={0,4,8}, precision and direct/
temporal/compacted paths. Freeze dt, mesh, pressure law and numerical profile
within a speed comparison. Publish strict and throughput-profile Pareto tables.

## Required ablations and final artifacts

Compare direct GPU baseline, batching, compact vs padded failed RHS, cached vs
rebuilt Gram/history, temporal vs direct, FP32/scaling/refinement vs FP64,
fused vs temporary-tensor peak, graph vs eager, and the adaptive path. Include
all-failed abrupt controls: the known CPU result shows history can slow them.

Commit runnable CLI benchmarks, configurations, raw CSV/JSON and a final
Markdown report with commands, all acceptance checks, chosen configuration,
VRAM and throughput. Include one recorded real scene rollout or reproducible
visualization command. Clearly mark unavailable hardware/library tests rather
than fabricating completion. A performance estimate is never a measured result.
