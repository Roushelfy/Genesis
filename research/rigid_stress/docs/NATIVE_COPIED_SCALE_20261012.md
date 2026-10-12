# Current-source copied B1024 profiling and B2048 live measurement

These supplemental measurements capture numerical source **`cad0c46c`**,
including native boundary fallback and bounded GPU packing. They use the
same32-condition bank copied across environments, including controller
delays, rather than creating new contact distributions. This is valid
scaling evidence for that workload; it does not establish diverse large-batch
or policy acceptance, a throughput optimum or an optimization plateau.

## B1024 profile

The actual contact snapshot is replayed1,280 times per profiled stage after
600 live warmup steps. Quadrants' enabled kernel profiler reports:

| Kernel work | Mean device cost |
|---|---:|
| Surface-fan profile generation,main sample kernel |22.125 ms|
| Intrinsic edge-port propagation |3.174 ms|
| Cooperative traction fit |1.339 ms|
| Cooperative nodal scatter |0.872 ms|
| Source resolution |0.737 ms|
| Original-point wrench/cone commit |0.651 ms|

The additive stage wall measurement is31.733 ms for the intrinsic contact
pipeline,0.848 ms for linear recovery,0.343 ms for complete residual and
0.247 ms for global peak. These stage times include dispatch/synchronization
and applicable graphs; kernel profiler entries omit captured graph work.
They describe snapshot profiling and are not live throughput.

Native stress buffers occupy3,097,545,298 bytes. Snapshot replay preserves
original points exactly; normalized force/moment errors are8.930e-9/4.144e-9,
nodal replay difference2.220e-16 N, peak replay difference1.022e-8 Pa,
full residual maximum1.247e-11 N and complete residual budget fraction0.005326.
Gauge rows and final error budgets remain unchanged. These are replay checks,
not new independent per-step CPU oracles at B1024.

The dominant cost is now surface sampling, rather than the earlier inverse
application hotspot or traction fitting. The canonical code already
parallelizes samples. The next candidate restricts its launch grid to actual
candidate faces, retaining every original sample/profile and accounting for
its added frontier buffers and device-to-host maximum readback. This remains
under qualification and is not selected by this measurement.

## B2048 live

All2,048 environments execute three complete1,280-step repetitions after
600 warmup steps. Each repeat includes2,368 partial environment resets and
all contact updates, traction retries/diagnostics, solve, complete residual,
global peak and existing history/controller work. Measured repetition rates
are31,163.129/31,186.471/31,143.132 valid env-step/s. Aggregating by actual
elapsed time gives **31,164.234 valid env-step/s and15.2169 batch-step/s**.
Invalid environment steps, every per-environment failure count and scatter
overflow are zero. All environments remain in the accounting.

Native stress buffers occupy6,121,977,426 bytes. Recorded device free memory
is89,749,192,704 bytes; Torch peak is7,485,952 bytes and excludes native
Quadrants allocations. No policy inference or RL training is included in
the live scope. The known policy failure remains disclosed separately in
the [policy807 audit](NATIVE_SOURCE_PRIORS_20261012.md).

## Hardware and reproducibility

Both allocations use NVIDIA RTX PRO6000 Blackwell Server Edition,97,887 MiB,
driver595.71.05, image `genesis:1_26`, Quadrants1.3.3, FP64. B1024 profile
uses `GPU-c76298f5-14a5-a899-a833-a3a917a745f8`; B2048 live uses
`GPU-17819d06-c563-257f-30fe-100c9f111279`. They are different GPUs/scopes
and are not a matched batch-speedup comparison. The full low shell has480
P2 tetrahedra,810 P2 nodes and2,430 DOFs. The declared footprint/integration
law, Q10/s8, material/mass, dt0.01/one substep, GJK mesh64/contact patches
and final precision budgets remain fixed. No high mesh or physical
mesh/integration convergence experiment ran.

The [selected scaling manifest](../evidence/20261011-nonconvex/selected-scale-checkpoint-v1/manifest.json)
maps97 artifacts to51 unique blobs/5,085,031 bytes, SHA256
`5604439bc386eb1aa730822e39f6400d5fbb9c6d806d62c227466aa018502589`.
Large raw JSON/logs use explicitly labeled lossless gzip; each entry records
both original-content and stored-content hashes. Allocated CPU publication
verifies source closures, all originals, stored blobs and decompressed
content. Full warmup quality, commands, contact conditions, per-env counters,
raw profiler entries, hardware, source and launchers remain reproducible.
Condition-bank SHA256:
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.

Legal general contact/model integration, lifecycle and independent field
coverage for the source-prior alternative, complete policy throughput,
diverse seeds/larger batches and applicable measured optimization remain open.
