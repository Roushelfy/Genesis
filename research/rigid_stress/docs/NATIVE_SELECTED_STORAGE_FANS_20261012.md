# Qualified production sample storage and fan scheduling

The cleaned native implementation combines lossless producer-counted sample
growth, exact current-step physical positions and CUDA parallel refined-fan
sampling. Experimental selection switches are removed. CPU retains sequential
sampling/full storage; other public integration modes keep their existing
behavior. The initial one-slot allocation is the smallest legal Quadrants
tensor. Every producer counts the complete physical sample range, grows exactly
when needed, copies valid unaffected samples and replays before fitting.
No physical sample, contact, radius or friction is dropped or altered.

CPU31 cases pass in481.245s and CUDA31 in689.686s. Both contain fresh
manual/native contact entries, nonconvex/lifecycle checks and48 independent
CPU FP64 traction/nodal/full elastic fields. Across96 recorded fields, maximum
normalized force/moment error is7.386597204e-10, complete residual budget
fraction0.0458756807 and scalar peak difference8.215225535e-6Pa. All six gauge
rows and the unchanged final budgets remain checked. Eight native CUDA fixtures
compare every selected cached position against direct chart reconstruction,
with exact zero difference. The hardest targeted CPU/CUDA smokes also pass.

A same-GPU B2048/seed623102 pair compares the qualified counted-position
prototype with sequential fans against this cleaned production combination.
Both use FP64/Q10/s8, `body_source_prior`, max output, mesh64/GJK patch-on,
dt0.01/substep1,600 changing warmup steps and three1280-step changing live
repeats. All observations, contact/radius updates, resets, metrics, history,
fallback and environments are timed on RTX PRO6000 Blackwell Server Edition,
97887MiB/driver595.71.05/container1_26, Genesis1.4.3/Quadrants1.3.3.

| Route | Effective env-step/s | Batch step/s | Repeat seconds | Native stress bytes |
|---|---:|---:|---|---:|
|Counted positions, sequential fans|63715.274821|31.110974|41.305943/41.057734/41.065437|1965932106|
|Clean production combination|63912.805443|31.207425|41.096361/40.981333/40.969948|1965932106|

The ratio1.003100208 is approximately0.31%, within ordinary repeat variation.
It establishes no material regression at this width, rather than a separate
significant throughput improvement. The [earlier parallel-fan pair](NATIVE_CONTACT_TAIL_STORAGE_20261012.md)
has1.017149258x on full sample storage, and the corresponding intrusive audit
reduces hard fan calls from roughly107ms to4.3–4.4ms. This observable tail
improvement is retained. Storage/cache and fan gains are not additive.

Each repeated route has2323 resets,3142 successful completed lifts, zero
failed lifts/numerical invalid steps/overflows. Condition banks agree fieldwise;
all per-environment rigid contact/grasp/descent counters and final positions
agree exactly. Sampled assigned-device peaks are7919→7883MiB at200ms,
including setup and trajectory; these are not exact transient maxima.

`native-selected-storage-fans-checkpoint-v1` contains544 entries,132 unique
blobs/17215396 stored bytes. Its allocated CPU publisher verifies original,
stored and decompressed hashes, source before/after closures, reports, all
recorded numerical fields, condition banks, telemetry and per-environment
work/outcomes. Manifest SHA256:
`3c30674f49cfa43ef2959577cb4c14366a8b8180d9fb0200cbc638467593d99e`.

Final-source wider matched four scopes/separated seeds, dynamic profiling,
remaining raw-contact storage decisions and a throughput plateau remain
pending. The nonconvex goal is active; physical convergence and complete RL
training throughput are not claimed.
