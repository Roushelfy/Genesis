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
the900-sample disk rule fits. Its B16 smoke passes; expanded CUDA qualification
and paired live evaluation are being completed. Memory savings may also
enable larger batches, which still need measurement.

Archive `native-compact-traction-profiles-checkpoint-v1` contains828 entries/
151 unique blobs/71292207 stored bytes. Allocated CPU publication verifies
all source/original/stored/decompressed hashes, both31-case reports, all96
fields, every failed-run record and both complete dynamic pairs. Manifest SHA:
`0efd5efbec6841450db68bf096cd558f6eb94944dceb005c5a2358c46fcd0b26`.
The patch, launchers, raw results and device monitor are reproducible.
Combined final-source batches/scopes, further scheduling/layout/reuse work
and a plateau remain pending. No physical convergence or RL training claim.
