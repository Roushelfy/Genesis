# Changing-contact latency, hard environments and elasticity oracle

Current numerical source `6a861556` has a measurable long tail in the first
serial traction retry. Removing a host maximum read or selecting contact
graphs does not address that cost. A masked cooperative cold retry is now
being evaluated ahead of the existing serial fallback; no candidate has been
selected by this diagnostic alone.

The diagnostic uses B2048/seed623102 on GPU10687902, RTX PRO6000 Blackwell
Server Edition,97887MiB, driver595.71.05, container1_26,
Genesis1.4.3/Quadrants1.3.3. Low full hollow shell,480 affine P2 tetrahedra,
810 P2 nodes/2430 DOFs, FP64/Q10/s8, `body_source_prior`, max output,
mesh64/GJK patch-on, dt0.01/substep1. It builds and saves its exact condition
bank, warms up600 steps and restarts for two complete1280-step trajectories.

The first pass fences once per batch step. The second fences before/after each
measured native stage. Both include trajectory metrics in measured step
latency. Per-step work export and selected field captures happen outside that
latency. These fences perturb the schedule, so neither diagnostic is a
headline end-to-end throughput measurement. Every environment and every hard
step remains included. Stage measurements are serialized call latencies,
not a sum of overlapping GPU event intervals.

| Pass | Mean whole step ms | Median | p90 | p99 | Maximum |
|---|---:|---:|---:|---:|---:|
| Boundary fence |36.8900|36.0717|38.6534|42.2528|1567.2937|
| Stage fences |39.8075|38.7276|41.7006|45.8279|1572.0093|

Each pass contains2323 selected-environment resets and3142 successful/zero
failed lift cycles, with zero invalid environment steps. Final rigid positions
are exactly equal. Raw `(1280,2048,9)` work arrays retain contact counts, total
and maximum evaluations, total and maximum samples, integration retries,
elastic corrections/fallbacks and invalid flags. These arrays identify the
environments contributing work; they do not measure individual parallel GPU
wall times.

## Measured stage and tail costs

The largest instrumented mean call costs are cooperative fitting9.882ms,
rigid constraints5.990ms, fan samples3.697ms, first serial retry3.031ms,
final traction/scatter2.838ms, collision2.251ms, surface inverse1.405ms,
complete residual0.964ms and footprint ports0.847ms. The recovery path here
already shares its immutable boundary inverse and uses current-step packed
surface loads. The old7.29ms full-inverse hotspot is not the dominant stage
of this dynamic current-source workload.

| Tick | Hard/retry environment | Whole step ms | Cooperative fit ms | First serial retry ms | Fan refine ms | Refined fit ms |
|---|---|---:|---:|---:|---:|---:|
|468|1914|977.214|27.123|737.505|108.233|76.925|
|934|Serial retry succeeds without integration refinement|1572.009|47.833|1492.792|0.155|0.135|
|959|192|994.389|25.803|740.278|107.944|89.303|
|1068|1914|981.177|28.406|737.440|108.220|76.935|

Maximum recorded evaluations is270. Integration-refinement frames occur at
468,959,1068, but tick934 demonstrates that refinement count alone does not
identify all expensive numerical retries. The three explicitly refined frames
average978.652ms in the boundary pass versus34.678ms across the remaining
frames, which still include the expensive unrefined retry. Per-step raw work,
stage calls, the16 largest latency frames and all integration retries are
preserved. No tail clipping or discarded environment is used to obtain a rate.

This gives a concrete high-return candidate: distribute a failed cold fit
across a warp before its serial reduction path. It must retain final wrench,
cone, complete residual and stress consistency budgets, preserve successful
contacts and account for the additional masked dispatch. Serial fallback is
still required when the cooperative retry does not converge. Candidate
qualification and matched live results are pending, so performance has not
yet reached a demonstrated plateau.

## Independent FP64 elasticity checks

Seven declared trajectory ticks plus every integration-retry frame capture
up to16 environments per pass: ordinary environments and the largest current
fit work. In total320 GPU displacement/peak samples pass independent CPU
affine-P2 stiffness assembly and sparse factorization, including all equilibrium
rows. The CPU uses its own gauge; full tensor/von Mises fields computed from
GPU displacement match the independent solved fields at rtol1e-4/atol1e-3Pa.
GPU global maximum also matches. Worst global peak difference is
1.618085662e-6Pa and complete residual budget fraction0.005393628.

The independent oracle receives captured balanced RHS. This audit therefore
checks the elastic solve/complete equilibrium/global peak, and does not claim
independent reconstruction of dynamic contact fitting or inertia relief.
Those stages retain their separate unchanged-budget contact/frame/oracle
fixtures. No physical mesh/integration convergence experiment was run.

Reproduce from the archived dynamic probe, launcher and CPU oracle publisher.
Archive `native-dynamic-profile-checkpoint-v1` has99 entries/96 unique blobs/
37,856,504 stored bytes. All original/stored/decompressed hashes, immutable
source closures, full work arrays, latencies and320 fields are verified.
Manifest SHA256:
`f3512db42dc3cafb9adc9d9c43a2540a206a179ec97d99148b8e04b724a06208`.

Broader batch/seed measurements, the measured tail candidate, remaining
applicable optimization decisions and a plateau remain active goal work.
