# Native nonnegative path-bound pruning

The selected point-distance route skips a target-to-port Euclidean norm when
the accumulated anchor-to-port path already reaches or exceeds the current
best complete distance. The final segment length is nonnegative, so this
comparison preserves the same minimum. It retains finite surface paths,
adjacent unfolding, planar verification, all Q10 samples/profile weights,
physical radius, friction and final acceptance budgets.

CPU23 cases qualify the bound with the unchanged CPU sampler. CUDA23 cases
qualify it together with selected seed reuse and sample-major scheduling.
Their48 recorded independent source-prior full fields have maximum normalized
wrench error7.386596988e-10, full equilibrium/gauge residual budget fraction
0.045856225 and peak difference8.084287401e-6Pa. Further footprint cases compare
complete fields against independent P2Shell oracles. Source lifecycle,
capacity changes, partial reset, environment isolation and U/cavity gap/hole
checks pass. CPU/GPU matrices take297.268/413.036s.

The B2048/623102 pairs use the same low complete hollow shell
(480P2 tetrahedra/810P2 nodes/2430 DOFs), FP64/Q10/s8, `body_source_prior`,
maximum output, mesh64/GJK patch-on, dt0.01 and one substep. Hardware is RTX
PRO6000 Blackwell Server Edition,97887MiB, driver595.71.05, container1_26,
Genesis1.4.3/Quadrants1.3.3. Both routes include the identical qualified EPA
correction and selected seed reuse. The earlier environment-major isolated
profile gives33.176090→26.676988ms contact wall. The combination with selected
sample-major scheduling gives18.233514→16.813467ms, and fan sampling drops to
3.982ms. These profile pairs execute128 frozen-snapshot passes after600 varied
warmup steps. Snapshot repeat difference is at most5.551115123e-16N in nodal
force and1.407170203e-8Pa in peak, with final normalized wrench errors below1e-8.

The combined live pair runs sequentially on GPU03ec8bc3. Each route warms600
steps then runs three1280-step repeats,2323 selected-environment resets each.
Controller, contact updates, reset/history/repair/fallback costs are timed.

| Route | Repeat effective env-step/s | Aggregate effective env-step/s | Batch step/s |
|---|---|---:|---:|
| Original complete port norms |52288.621,52456.390,52499.648|52414.728230|25.593129|
| Nonnegative bound |53986.772,54193.033,54179.707|54119.672973|26.425622|

All repeats have zero invalid steps or scatter overflows. The measured
1.032527971x end-to-end gain selects this bound. Native buffers stay
6,111,942,238 bytes. Maximum warmup full-residual budget fractions are
0.009348642 and0.009110120. Absolute rates from different GPU pairs represent
their own measurements. The preceding sample-layout pair's55763.374 rate is
not a matched baseline for this incremental comparison.

Recreate the archived `probe_port_bounds_v1.py`, original distance function,
v1/v2 measurement wrappers, combined profile/CUDA launcher and v2 live pair.
The private combined source is90b2964e plus selected seed reuse, sample-major
scheduling and this candidate. Before/after collider/stress closures are equal.
Archive `evidence/20261011-nonconvex/native-port-bounds-checkpoint-v1/manifest.json`
contains623 entries/131 unique blobs/43,602,921 stored bytes, including the
earlier isolated profile. An allocated CPU publisher verifies all raw/stored/
decompressed hashes,48 recorded fields, complete closures and every rate/
failure row. Manifest SHA256:
`8065082a6e1859c6d219ee3a7eb3425ce373a88512cf185c86c3938d3ae2961c`.

The goal remains active. Exact contact-capacity growth has a measured memory
reduction and is being evaluated with growth/history costs included. Current
four-scope/wider-batch/multiple-seed evidence, timed grasp/tail statistics,
dynamic profiling, remaining applicable candidates and a plateau remain open.
Physical convergence and RL training throughput are separate studies.
