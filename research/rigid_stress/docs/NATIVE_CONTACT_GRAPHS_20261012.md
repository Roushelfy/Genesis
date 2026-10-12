# Contact graph and dispatch ablations

Complete contact graphs and per-kernel graphs pass their numerical checks but
are not selected: their snapshot profiling gains do not improve the matched
changing-contact rollout. Production remains `6a861556`.

The complete graph inlines the exact original Quadrants kernel `.fn` bodies,
including source loading/binding/resolution, port paths, frontier sampling,
source priors, constrained fitting, all integration retries and continuous
certificates, fused final traction/scatter and publication. It consumes the
current frontier maximum on device. The raw-contact maximum still crosses to
the host for exact workspace growth. No numerical algorithm, radius, friction,
Q10/s8 sampling, stopping or final acceptance budget changes. Per-kernel
variants rewrap the same original numerical bodies with graph/fastcache flags.

All measurements use the low complete hollow shell, 480 affine P2 tetrahedra,
810 P2 nodes and 2430 DOFs, FP64, `body_source_prior`, max output,
mesh64/GJK patch-on, dt0.01/substep1. RTX PRO6000 Blackwell Server Edition,
97887MiB, driver595.71.05, container1_26, Genesis1.4.3/Quadrants1.3.3.

## Frozen diagnostic profiles

Each route has600 warmup steps and128 frozen actual snapshot passes. Comparisons
inside each group are sequential on the same GPU; the two groups use different
GPUs and must not be combined into an incremental speedup.

| Per-kernel route, GPUc41c23c1 | Contact wall ms |
|---|---:|
| Original |16.908571|
| Fastcache only |16.506932|
| Graph only |14.633937|
| Graph and fastcache |14.493418|

| Whole contact route, GPU0c97d4b8 | Contact wall ms |
|---|---:|
| Original |16.280509|
| Complete graph |13.581572|

Every snapshot retains normalized force/moment errors below1e-8, nodal repeat
differences below1e-11N and peak differences below1e-3Pa. Captured graph device
work is omitted by the Quadrants kernel profiler; the table is stage wall time,
not a complete device-only accounting.

## Changing-contact end-to-end ablation

Four routes run sequentially on one GPU with B2048/seed623102,600 warmup steps
and three1280-step repeats per route. All include the timed trajectory metrics,
ordinary control/rigid steps, changing contact radius, observations needed by
metrics, reset, history and fallback work. No slow or failed environment is
removed from the denominator. Each repeat has2323 selected-environment resets.

| Route | Repeat whole step ms | Effective env-step/s | Batch step/s | Relative to original |
|---|---|---:|---:|---:|
| Original |35.1384,35.2309,35.5802|57989.897748|28.315380|1.000000|
| Per-kernel graph and fastcache |36.2674,36.4828,36.1689|56408.824204|27.543371|0.972735|
| Complete graph |35.5753,35.5449,35.4061|57675.911793|28.162066|0.994586|
| Complete graph and fastcache |35.4825,35.2438,35.4551|57863.268532|28.253549|0.997816|

All routes have zero invalid environment steps and scatter overflows. Each
repeat completes3142 successful lift cycles with zero failed cycles and
154804 passing/zero failing lift-window observations. Final rigid positions
and every per-environment grasp/contact/descent counter are identical across
routes. Stress fit evaluation counts have small floating-point ordering
differences and are preserved in the raw results; no acceptance changes.
Native buffers are5,498,093,150 bytes in every route. Setup including JIT is
69.738/70.554/71.958/71.957s respectively, separate from rollout timing.

The original route is retained. The snapshot saving provides no end-to-end
gain here. Fastcache without graphs is a separate live candidate, being
compared with original runs both before and after it. Dynamic stage/tail
profiling is also separate and pending; these numbers alone do not attribute
the graph regression to a particular device stage.

## Numerical and lifecycle evidence

CUDA23 cases pass in502.957s, including actual scene source/lifecycle paths,
workspace growth, release/rebirth, partial reset, invalid member diagnostics,
nonconvex gap/hole exclusion, and constrained/refined/hard source-prior cases.
For the24 recorded independent CPU FP64 fields, the test adapter reconstructs
fresh original physical contacts and reruns the complete graph before the
unchanged signed nodal/full-field/gauge residual/peak assertions. Maximum
normalized wrench error is7.386597564e-10, complete residual budget fraction
0.045691947, and peak difference8.150971553e-6Pa. Componentwise footprint
oracles remain part of the matrix. CPU production is unchanged.

The first B16 smoke wrapper incorrectly read `.fn` from a Python dispatcher;
the corrected wrapper unwraps its Quadrants `_impl` kernel. Both raw runs and
source closures are retained. This was an adapter startup error, not a fitting
or physical failure, and no assertion was relaxed.

Reproduce from the archived complete/per-kernel probes, profile-pair and four
live launchers, and complete-graph regression adapter. Archive
`native-contact-graphs-rejected-v1` contains953 entries/161 unique blobs/
72,282,492 stored bytes. An allocated CPU publisher verifies original,
stored and decompressed hashes, source closures, reports,24 fields, snapshot
budgets, all live rows and environment-level trajectory equality.
Manifest SHA256:
`98fd40909634ae3b494ede90d1d5e60d55b8da6783edcb54bd3a54e829efc90f`.

The active extension still needs broader matched scopes/batches/seeds,
dynamic/tail cost attribution, remaining applicable candidates and a measured
plateau. No physical convergence or RL training throughput is claimed.
