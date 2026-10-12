# Complete-sample storage compaction

This prototype counts every positive physical Q10 sample independently of
storage capacity, grows before fitting and replays the complete producer.
Growth copies successful unaffected profiles before selective disk/fan replay.
The fan refinement bound stays the full physical quadrature size. No loads,
contacts, radii or samples are discarded. The initial one-slot allocation is
the smallest tensor Quadrants accepts, rather than a physical sampling limit.
CPU uses its existing full storage; CUDA frontier sampling uses observed
sample capacity. The physical footprint/profile law and final budgets stay
fixed.

The original B4096 profile/index arrays occupy8257536000 of10923073118
native stress bytes despite observed maxima around1156 samples. Matched
compaction pairs use baselinea6dc0fbc, FP64/Q10/s8, the low full hollow shell,
max output, `body_source_prior`, mesh64/GJK patch-on and dt0.01/substep1. Each
route has600 warmup and three1280-step changing trajectories, including timed
metrics, observations, contact updates, resets and all history/fallback work.
Each pair runs sequentially on one RTX PRO6000 Blackwell GPU,97887MiB,
driver595.71.05/container1_26, Genesis1.4.3/Quadrants1.3.3.

| B/seed | Original env-step/s | Compact env-step/s | Ratio | Native bytes saved |
|---|---:|---:|---:|---:|
|2048/623102|59871.884945|58013.670385|0.968963487|3929899004|
|8192/734001|91482.523882|90558.610412|0.989900656|15715467260|

Every repeat has zero invalid environment steps/overflows and zero failed
completed lifts. Rigid positions agree exactly and every per-environment
rigid grasp/contact/descent counter agrees. Native stress bytes drop
5498093150→1568194146 atB2048 and21773033054→6057565794 atB8192. The B8192
assigned-device200ms monitor records maximum41863→21095MiB, including setup
and dynamic trajectories. This is a sampled peak, not an exact transient
maximum. The B2048 monitor failed because of an unprefixed UUID; no sampled
peak is claimed there. B8192 uses a separated seed cohort.

Expanded independent validation passes31 CPU cases in480.37s and31 CUDA
cases in671.53s. Both manual and actual native contact pipelines are checked
against an independent CPU FP64 traction/nodal/elastic oracle:96 full fields,
maximum normalized wrench error7.386598105e-10, complete residual budget
fraction0.0458562252 and global peak difference8.137525583e-6Pa. Contact birth/
death, group growth, partial reset, recovery and physical incompatibility
semantics retain their existing checks.

All failed first attempts are preserved. Quadrants rejects the initial
zero-length tensor in B16v1; the minimum legal placeholder fixes allocation.
The expanded oracle harness initially fed already aggregated forces into the
native raw-contact entrypoint, and used a CUDA cached scatter entrypoint with
disabled CPU arrays. Those8 CPU/5 CUDA fixture failures are preserved. Fresh
raw source forces and the correct backend entrypoint fix the harness; final
acceptance thresholds are unchanged.

This memory prototype is not selected for production yet. Three additional
capacity reductions/host reads hurt B2048 throughput. A follow-up fuses
capacity statistics into the producers and avoids disk-capacity reads once
the900-sample disk rule fits. Its B16 smoke and31 CUDA cases pass. On one
GPU, the three-repeat B2048 pair reports60148.641796→58797.647578 env-step/s
(0.977539074x), saving3929899008 native bytes. This reduces the observed
regression but does not establish a throughput gain; cross-allocation ratios
are separate paired decisions rather than a direct comparison of elapsed
times. Its48 independent full fields have maximum normalized wrench error
7.386595923e-10, residual budget fraction0.0456061170 and peak difference
8.192051610e-6Pa. CUDA31 completes in718.38s.

The fused-capacity archive has248 entries/115 unique blobs/16238850 bytes,
manifest SHA256:
`7dd7c7c94e1af8f0d0af2db85383e29127d293011c08a3aefba79e074dde1ea8`.
All source/original/stored/decompressed hashes and complete paired counters
pass allocated CPU verification. Both v2 raw telemetry streams are valid;
an inherited limitation sentence in an earlier derived scalar checker is a
metadata error and does not describe these v2 runs. No exact transient peak
is inferred from the monitor.

This follow-up remains a prototype. A current-step position cache is now
being evaluated on counted storage with the selected eight-sample-lane retry.
It verifies cached positions against direct chart reconstruction before the
independent force/nodal/stress oracle. Memory savings may enable larger
batches, which still need final-source measurement.

Archive `native-compact-traction-profiles-checkpoint-v1` contains828 entries/
151 unique blobs/71292207 stored bytes. Allocated CPU publication verifies
all source/original/stored/decompressed hashes, both31-case reports, all96
fields, every failed-run record and both complete dynamic pairs. Manifest SHA:
`0efd5efbec6841450db68bf096cd558f6eb94944dceb005c5a2358c46fcd0b26`.
The patch, launchers, raw results and device monitor are reproducible.
Combined final-source batches/scopes, further scheduling/layout/reuse work
and a plateau remain pending. No physical convergence or RL training claim.
