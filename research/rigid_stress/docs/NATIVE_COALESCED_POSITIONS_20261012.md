# Sample-contiguous cached positions

The qualified candidate stores current-step physical sample positions in a
scalar `(sample, contact, environment, coordinate)` tensor with layout
`(2,1,0,3)`, alongside sample-contiguous profile/index layouts `(2,1,0)`.
Neighboring fitting lanes therefore access neighboring sample coordinates.
Geometry, physical sample count/order, local integration and the load law stay
unchanged. Position getters reconstruct the same three-component vector;
producer stores and exact-growth copies retain every coordinate.
No tensor count or native byte reduction is claimed.

The candidate is selected for clean production integration. Native CUDA
counted/cached fan storage uses the new layout; existing noncached storage and
CPU sequential sampling retain their semantics. The temporary prototype
selection flag is removed during production cleanup.

Two same-GPU changing-contact pairs use separate seed cohorts on RTX PRO6000
Blackwell Server Edition, FP64/Q10/s8/body_source_prior/max,dt0.01/substep1,
mesh64/GJK patch-on. Each route has600warmup and three1280-step repeats and
includes observations, contact parameter updates, resets, metrics, history,
fallback and every environment.

| Batch /seed | Baseline env-step/s | Candidate env-step/s | Ratio | Baseline repeat seconds | Candidate repeat seconds |
|---|---:|---:|---:|---|---|
|16384 /623102|105364.386|107260.429|1.017995104|199.379210 /198.837970 /198.896917|195.563993 /195.498299 /195.496615|
|8192 /734001|99219.652|100801.845|1.015946367|105.882615 /105.646603 /105.517652|104.393171 /103.756363 /103.920946|

Repeat ranges do not overlap in either cohort. All condition-bank fields,
per-environment rigid/contact/grasp/descent counters and final positions agree
exactly. All repeats have zero numerical failures, overflows and failed lift
outcomes. Complete records, including slow environments, remain in the archive.
These isolated ratios cannot be added to other optimizations.

CPU31 cases pass in481.412s and CUDA31 in684.241s. Across96 independently
checked full fields, worst original-point force/moment error is7.386596158e-10,
complete residual budget fraction0.0458562252 and global peak difference
8.222174074e-6Pa. All six gauge rows and full tensor/von Mises/global maximum
acceptance remain checked. Eight native CUDA fixtures compare each selected
cached position to independent direct chart reconstruction with exact zero
difference, including growth/refinement cases. Lifecycle tests pass.

This differs from the earlier unselected scalar profile/index-only B2048
experiment: that experiment kept vector position storage unchanged, whereas
this candidate also coalesces all cached physical coordinates and has replicated
B8192/B16384 live benefits.

Archive `native-coalesced-positions-qualification-checkpoint-v1` retains both
full regression reports, numerical fields, four live results, complete condition
banks, telemetry, source closures/patches, launchers and checker.
Original/stored/decompressed hashes and bounded large-JSON reconstruction pass.
Manifest SHA256:
`b9583a1c0d4cf6df1505f7eb02344e81c3cb9d01bf532c8777bb0d803aabd406`.

Clean combined production qualification and final-source throughput/scaling
remain pending; the nonconvex extension is active.
