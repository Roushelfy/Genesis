# Exact observed contact workspace growth

An aggregate that outgrows its current native contact workspace now allocates
the observed requirement. Capacity growth preserves every contact, the
original physical footprint, Q10 samples/weights, friction, solver budget
and final acceptance. It replaces doubling that retained unused rows. This
also follows the engine instruction to allocate the exact required memory.
No fixed grasp-contact count is assumed, and later growth remains possible.

The isolated candidate contains qualified EPA, current-step seed reuse and
sample-major scheduling. Subsequent port pruning is absent. CPU23/CUDA23
cases pass in296.584/419.060s, including workspace growth, source changes,
release/rebirth, partial reset, invalid-member diagnostic restoration,
environment isolation, U/cavity gap/hole checks and independent fields.
The48 recorded source-prior fields have maximum normalized wrench error
7.386604288e-10, complete equilibrium/gauge residual budget fraction
0.045856225 and peak difference8.183189493e-6Pa. Additional footprint
tests compare complete fields against independent P2Shell oracles.

B2048/623102 uses the low complete hollow shell
(480P2 tetrahedra/810P2 nodes/2430 DOFs), FP64/Q10/s8, explicit
`body_source_prior`, maximum output, mesh64/GJK patch-on, dt0.01/substep1.
Hardware is RTX PRO6000 Blackwell Server Edition,97887MiB, driver595.71.05,
container1_26, Genesis1.4.3/Quadrants1.3.3. Both routes grow once during
600-step warmup from4 to7 required contacts; doubling allocates8 and the
candidate7. Snapshot128-pass profile is18.630598 versus18.698345ms contact
wall. Native buffers decrease6,111,942,238→5,497,390,686 bytes, saving
614,551,552 bytes. This is a memory benefit without a measured speed benefit.

The live pair executes sequentially on GPU4e7dc608. Each route runs three
1280-step repeats after warmup, with2323 selected-environment resets per
repeat. Reset, load updates, history, recovery and fallback costs are timed.

| Capacity route | Repeat valid env-step/s | Aggregate valid env-step/s |
|---|---|---:|
| Doubling |54828.100,55088.290,54272.682|54727.571022|
| Exact requirement |53919.791,54483.803,55047.639|54479.853013|

All repeats have zero invalid environment steps and scatter overflows. The
rate ratio0.995473616 does not select a throughput speedup. This workload
does not grow during timed repeats; more frequent growth and wider batches
need their actual allocation/history costs counted before drawing broader
performance conclusions. Memory savings motivate retaining exact capacity
for those measurements, subject to combined-current-source qualification.

Recreate the archived `probe_exact_contact_growth_v1.py`, measurement,
profile/live pair and regression wrappers. The disabled route recreates
doubling at the workspace allocator; the enabled route uses exact size.
All source closures match before/after. Snapshot repeat differences are
within unchanged wrench and field budgets. Archive
`evidence/20261011-nonconvex/native-exact-contact-growth-checkpoint-v1/manifest.json`
contains476 entries/118 unique blobs/29,580,913 stored bytes. An allocated CPU
publisher verifies raw/stored/decompressed hashes, all reports and48 fields,
source closures, both profile/live pairs and actual growth records.
Manifest SHA256:
`6e45ce6430a62f337e6f2892cacd0bf3134f317fa0801ad2eaf2abe1f7e8e3e2`.

The current primary combination adds selected port pruning and timed
trajectory collection. CUDA23 cases pass in416.240s, with24 recorded
independent full fields inside the unchanged numerical budgets. Three
1280-step B2048 live repeats each have2323 selected resets,3142 completed
successful lifts, zero failed lifts and zero numerical failures. Per-environment
fit tails reach270 evaluations, integration retries are recorded, and elastic
corrections/fallbacks are zero. This verifies the combination and supports
retaining its memory benefit; it is not a matched speed comparison. A separate
allocated CPU publisher verifies the full current-source records and
`native-current-capacity-checkpoint-v1` archive:176 entries/109 blobs/
9,003,469 stored bytes, manifest SHA256
`822da12f2a2659efd054f71cf3b2a83fe03b442385ff55dfaa9cdb3efbb44093`.

The goal remains active; broad four-scope/seeds/batches, dynamic profiles,
tail costs, applicable candidates and a measured plateau remain open.
Physical mesh convergence and full RL training are outside this iteration.
