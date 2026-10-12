# Wide native baseline, dynamic profiling and independent wrench oracle

The nonconvex extension remains active. These measurements describe production
`92ec3e0c658f0436f67c81661e36305977487197`, after the qualified counted sample
storage, exact current-step positions and parallel refined-fan combination.
They establish a new low-shell baseline for subsequent optimization, rather
than declaring the extension complete.

## Matched changing-contact scopes

Each row uses one saved condition bank and one assigned RTX PRO 6000 Blackwell
Server Edition GPU, 97887 MiB, driver 595.71.05, container `1_26`, Genesis 1.4.3
and Quadrants 1.3.3 (`00423caa`). Across rows the allocated GPU and seed differ;
these are not paired source-change speedups or a precise throughput optimum.
The complete hollow shell has 480 affine P2 tetrahedra, 810 P2 nodes, 2430 DOFs
and 80 exterior faces. Collision geometry has 42 vertices. `mesh64` means
FP64 collision coordinates, not a 64-vertex mesh.

Settings remain FP64, Q10, s8, `body_source_prior`, max output, GJK/contact patch
on, dt 0.01, one substep. Each scope has 600 changing warmup steps and three
1280-step repeats. Rigid/live/policy include required observations, contact
parameter updates, resets, trajectory metrics, history, fallback and every
environment. Recovery-only repeats a frozen actual contact snapshot with its
matching pre-integration frame; it does not represent changing-contact recovery
rollout. Policy is FP32 26→128→128→7 inference plus observation/action
interoperation, rigid and stress; it is not complete RL training.

Cells are **effective env-step/s / batch step/s**:

| Batch / seed | Rigid only | Recovery only | Live rigid + stress | Policy + rigid + stress |
|---|---:|---:|---:|---:|
|1024 / 623101|126909.819 / 123.935|85269.544 / 83.271|41867.152 / 40.886|41453.867 / 40.482|
|2048 / 623102|213466.725 / 104.232|118363.477 / 57.795|63738.132 / 31.122|63320.269 / 30.918|
|4096 / 623103|354267.610 / 86.491|134654.432 / 32.875|83593.961 / 20.409|84570.396 / 20.647|
|8192 / 623104|507336.480 / 61.931|140157.217 / 17.109|99667.906 / 12.166|99408.530 / 12.135|
|16384 / 991909|665201.417 / 40.601|134308.081 / 8.198|105910.718 / 6.464|105892.927 / 6.463|
|32768 / 623102|821700.989 / 25.076|122198.467 / 3.729|104316.704 / 3.183|104146.207 / 3.178|

All measured repeats have zero invalid stress steps, scatter overflow, failed
lift steps or completed failed lift cycles. Rigid/live final positions agree
exactly and all per-environment rigid, contact, grasp and descent counters
agree. Every per-environment record remains in the evidence. At B16384 each
live repeat has 18571 resets, 25124 completed successful lift cycles and 17
integration retry frames. Live repeat seconds are
198.259566 / 197.880006 / 197.894357; policy seconds are
198.202906 / 197.966197 / 197.964632. These counts describe the declared
controller workload, not a learned grasp-success probability.

Native live stress buffers are 1.016 / 1.966 / 3.859 / 7.657 / 16.139 / 32.185 GB.
Assigned-device live sampled peaks including setup are respectively
5577 / 7883 / 12477 / 23267 / 40321 / 87519 MiB. At B16384 the four-scope sampled
peaks are 24017 / 45279 / 40321 / 45219 MiB. Device telemetry samples every
200 ms and does not establish exact transient maxima. Different full sample
capacities legitimately arise from different seed cohorts.

The corresponding archives are
`native-selected-matched-scopes-b{1024,2048,4096,8192,16384,32768}-{seed}-checkpoint-v1`.
Their manifests retain source before/after closures, original and stored hashes,
complete results, banks, hardware, telemetry and per-environment outcomes.
Large JSON results are split into exact ordered 32 MiB byte ranges, compressed
individually and reconstructed against the original complete SHA256; no records
are omitted. B32768 also closes with zero failures in every scope; each live/
policy repeat includes37138 resets and50247 successful completed lifts.
Live repeat seconds are402.137447 /401.926066 /402.158648, policy
403.019963 /402.636350 /402.540548. Its four-scope sampled peaks are
45189 /84575 /87519 /87839MiB. The archived B32768 manifest SHA256 is
`f6b3a6d51a01d6420f318d9351436df266b29458bb684fcf722c8429b89015ab`.

## Same-GPU copied-condition live scaling

The frozen B2048/seed623102 bank is tiled exactly, using the same assigned GPU
and source throughout. All four widths use 600 warmup steps and three1280-step
changing live repeats, including observations, updates, resets, metrics and
fallback. Every environment's rigid/contact/grasp/descent counters match the
same2048-case block exactly, and final positions have exact zero difference.
This verifies workload identity rather than assuming that independently sampled
wide batches have equal difficulty.

| Batch | Effective env-step/s | Batch step/s | Three repeat env-step/s | Native stress GB | Sampled GPU peak MiB |
|---|---:|---:|---|---:|---:|
|8192|102177.171|12.472799|102135.854 /102197.430 /102198.254|7.644|21413|
|16384|108645.547|6.631198|108559.175 /108680.473 /108697.096|15.216|39379|
|24576|108064.410|4.397152|108044.227 /108072.201 /108076.807|22.787|57403|
|32768|106636.078|3.254275|106591.469 /106669.681 /106647.115|30.358|82097|

There are zero numerical failures, overflows or failed lifts in all repeats.
For this baseline cohort, throughput peaks atB16384, drops about0.54% atB24576
and about1.85% atB32768. The repeat ranges do not overlap, although sequential
widths are not an interleaved thermal/drift experiment. This establishes a
practical platform at these measured widths; it does not prove a universal
batch optimum or the absence of unknown optimizations.

The throughput-best width uses roughly39GiB on a96GiB-class device. Extra raw
source/certificate compaction could reduce resident memory, but its ability to
admit still wider batches alone would not improve this already flattening
curve. Its remaining per-step bandwidth benefit needs to be assessed after
valid-source initialization and diagnostic clearing, preserving full original
indices, radius API, failure identity and lifecycle. Complete global sample
CSR would require either another expensive physical producer pass or a new
global allocation/indirection path; current lossless counted storage already
admitsB32768, and coalesced position access is being evaluated without that
allocation cost. No CSR implementation or measured speedup is claimed.

Archive `native-selected-copied-curve-checkpoint-v1` has349 manifest entries;
original/stored/decompressed hashes and every bounded JSON reconstruction pass.
Manifest SHA256:
`85b73a7e1862cd6f6a2636fd9335b066b73813a20014df1709ae8064a316b8fd`.

## B16384 changing-contact diagnostic profile

This separate instrumented run uses seed 623102, GPU
`GPU-7a15f021` (full UUID in its manifest), 600 warmup steps and 1280 changing
steps. Every stage is fenced for diagnosis, so its 159.611501 ms mean step is
not headline throughput and should not be summed with unrelated stage studies.
Step percentiles [min, p50, p90, p99, max] are
27.416 / 178.094 / 181.062 / 334.266 / 359.692 ms.

| Stage | Mean ms |
|---|---:|
|Physical fan sampling|43.775718|
|Cooperative traction fitting|33.888673|
|Rigid constraints|14.539021|
|Final signed P2 scatter|11.345054|
|Blocked shared surface inverse|10.341822|
|Rigid collision|7.341449|
|Intrinsic footprint ports|6.430244|
|Complete elastic residual|5.620115|
|Global maximum stress|3.497253|
|Source resolution|3.112007|
|Contact packing and force clearing|2.196611|
|Source capture|1.861953|
|Cold cooperative retry|0.585276|
|Eight-sample-lane retry|1.363812|
|Third serial retry|1.134933|

Fan refinement wrapper mean is 0.310715 ms, with nested chart/sample means
0.130734 / 0.151918 ms. Those nested timings overlap their wrapper and must
not be added again. All difficult environments remain included: 22 steps
perform integration retries; there are zero invalid steps, 18571 resets and
25124 completed successful lifts. The archive retains per-environment work,
tail attempts, distributions and the full native allocation table.

Native stress buffers total 15,244,566,090 bytes for this cohort. Major groups
include 3,008,102,400 bytes of raw source provenance, 1,101,004,800 bytes of
wrench certificates, 5,098,111,064 bytes of traction workspace and
833,552,384 bytes of footprints. Force/RHS/displacement/residual each use
318,504,960 bytes. These measurements motivate coalesced sample-position access,
valid-source-only initialization, sparse diagnostic clearing and warm Hessian
precision experiments. Raw provenance compaction still needs a concrete
benefit/cost decision preserving original contact identities and lifecycle.

464 captured fields pass independent CPU FP64 stiffness, inertia/gauge,
factorization and complete tensor/von Mises/global-maximum checks. Worst peak
difference is 1.921639e-6 Pa; complete residual is at most 0.005654082 of the
unchanged budget, including all six gauge rows. This audit receives captured
balanced RHS, so it does not independently establish dynamic traction mapping
or inertia relief; the additional audit below covers those inputs separately.

Archive `native-selected-dynamic-checkpoint-v1`: 208 entries, 142 unique blobs,
117,340,103 stored bytes; manifest SHA256
`89c694cfa2e1dae3c29e545c6b9690ee5cae223e93ae726eca75eff450742632`.

## Independent original-point aggregate wrench and inertia relief

A separate B2048/seed623102 GPU capture preserves actual original contact
position, force, radius, free moment, angular velocity, raw nodal force,
balanced RHS, displacement and maximum at 160 field samples. The CPU oracle
independently constructs mass, COM, centrifugal load
`-omega × (omega × (x-COM))`, inertia relief, K/gauge rows, a sparse FP64 solve,
complete stress tensor, von Mises and global peak.

All samples pass unchanged budgets. Worst normalized aggregate wrench error
is 8.66059146e-9; complete residual budget fraction is 0.006073481;
global peak difference is 1.28557440e-6 Pa. CPU-versus-native relief RHS
difference is 4.93526772e-8 **of the residual budget**, not raw relative error.
Fixture traction checks remain independently responsible for local Coulomb
cones and individual wrench acceptance; this aggregate audit does not recreate
every dynamic local QP.

The initial offline check incorrectly compared aggregate moment about the
coordinate origin against only the local contact radius. Its seven false
positives and diagnostic run are preserved. A force error changes moment under
reference translation. The corrected comparison uses
`a = sum(||F_i|| p_i) / sum(||F_i||)` and
`delta M_a = delta M_0 - a × delta F`.
The sum of unchanged individual original-point bounds propagates as
`1e-8 sum(||F_i|| (R_i + ||p_i-a||) + ||M_free_i||)`.
For one contact this is exactly the original-point bound with `a=p`.
No production threshold, input load or final acceptance was changed.

Archive `native-selected-wrench-oracle-checkpoint-v2`: 205 entries, 133 unique
blobs, 20,984,848 stored bytes; manifest SHA256
`5e74808ef43f7be806a18e4f513f6da40fe97f9285563aa287921b6673995d6e`.

Remaining work is final-source wide scaling, measured decisions for the applicable
new candidates, clean combined qualification and final-source plateau evidence.
These low-mesh consistency results do not establish physical mesh/integration
convergence or complete RL training performance.
