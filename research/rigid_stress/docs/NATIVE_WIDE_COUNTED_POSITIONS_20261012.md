# Wide counted-position live probes

These exploratory runs use the qualified counted-profile/current-step-position
prototype and production `50b1b61c` eight-lane retry. They retain serial fan
refinement and precede the cleaned production combination. Both use the low
complete hollow shell, FP64/Q10/s8, `body_source_prior`, max output,
mesh64/GJK patch-on, dt0.01/substep1. The elastic mesh has480 affine P2
tetrahedra,810 P2 nodes and2430 displacement degrees of freedom.

Each allocated RTX PRO6000 Blackwell Server Edition GPU has97887MiB,
driver595.71.05/container1_26, Genesis1.4.3/Quadrants1.3.3. Each route includes
600 changing warmup steps and three1280-step changing live repeats. All
environments, observations, contact/radius updates, resets, metrics, history and
fallback are timed. Seeds and assigned GPUs differ between widths.

| B | Seed | Effective env-step/s | Batch step/s | Native stress bytes | Sampled device peak MiB |
|---|---:|---:|---:|---:|---:|
|16384|991909|107641.888928|6.569940|16138591818|43445|
|32768|623102|104753.768134|3.196831|32185196106|86555|

B16384 repeat seconds are195.239538/194.626979/194.613751. Each repeat has
18571 resets,25124 completed successful lifts and zero failed lifts, invalid
environment steps or scatter overflows. B32768 repeat seconds are
400.579231/400.331940/400.278268; each has37138 resets,50247 completed
successful lifts and the same zero failures. Every per-environment record is
published, including difficult trajectories. Native final acceptance runs at
each step. The largest recorded warmup complete residual budget fractions are
0.00898630313 and0.0103066889, including all gauge rows.

The assigned-device200ms sampled peak includes setup and trajectories, and is
not an exact transient maximum. B32768 reports19035324416 device-free bytes
at the endpoint. Native stress buffers do not include all rigid/controller/
Torch state or allocation caches; sampled device use includes those costs.
Lossless sample growth preserves all physical samples and replays the producer
before fitting. Independent same-mesh fixture qualification is the separate
31 CPU/31 CUDA/96-field position checkpoint.

The B32768 aggregate rate is lower while its batch step rate roughly halves.
These separate GPU/seed cohorts establish feasible resident widths and guide
the next sweep. They do not establish a paired scaling curve, the final-source
optimum or a throughput plateau. Clean production qualification, final matched
four scopes/separated seeds and remaining bottleneck decisions are pending.
No physical convergence or complete RL training throughput is claimed.

`native-wide-counted-positions-checkpoint-v1` preserves complete raw results,
source snapshots/before-after hashes, runners, logs, condition banks and
assigned-device telemetry. The allocated CPU publisher checks all outcomes,
budgets and original/stored/decompressed hashes. Manifest SHA256:
`322ccdf96256be1a46004b6b5d64ba1a0feda46f453f6ee303b2dc1ea3d85a37`.
