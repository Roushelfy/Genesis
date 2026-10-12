# Current-step fan position reuse

The complete counted-sample prototype stores each physical sample position
when its producer evaluates the intrinsic footprint. Dual fitting and final
acceptance reuse that position. Normals still come from the shared material
face and signed P2 basis values still use the original chart. Initial, disk
and fan producers write their complete cloud; exact capacity growth copies
unaffected profiles and positions, then replays before fitting. No previous
step geometry is assumed valid. Radius, friction, topology, charts and final
wrench/residual/stress budgets stay fixed.

This experiment combines selected eight-sample-lane retry with the qualified
producer-counted storage prototype. Its pair changes only position reuse.
The low full hollow shell uses FP64/Q10/s8, max output, `body_source_prior`,
mesh64/GJK patch-on and dt0.01/substep1. Both routes run sequentially on one
RTX PRO6000 Blackwell GPU66aad496,97887MiB, driver595.71.05,
container1_26, Genesis1.4.3/Quadrants1.3.3. Seed623102/B2048 has600 warmup
and three1280-step varied live repeats. Timing includes observations, contact
updates, metrics, resets, history and fallback for every environment.

| Route | Valid env-step/s | Batch step/s | Native stress bytes | Sampled device peak MiB |
|---|---:|---:|---:|---:|
| Counted storage, direct reconstruction |58401.350142|28.516284|1568194146|7567|
| Counted storage, cached positions |62744.159094|30.636796|1965932106|7919|

The isolated cache gain is1.074361448x. Repeat wall seconds are
45.456843/44.124843/45.078214 versus41.903454/41.720687/41.715334.
Every repeat has zero invalid environment steps/overflows,2323 resets,
3142 successful and zero failed completed lifts. All per-environment rigid
grasp/contact/descent counters agree and final rigid positions agree within
1e-12m. Assigned-device memory is sampled every200ms and includes setup and
trajectory; these values do not establish exact transient maxima.

CPU31 cases pass in480.63s and CUDA31 in693.83s. Manual and native-entry
contact paths provide96 independent CPU FP64 traction/nodal/elastic full
fields. Maximum normalized wrench error is7.386598249e-10, full residual
budget fraction0.0458562252 and global peak difference8.101487765e-6Pa.
All eight native CUDA full-field fixtures also compare every selected cached
position with direct chart reconstruction: maximum difference0m at an
unchanged1e-15m check. Lifecycle/group growth, partial resets and physical
incompatibility retain their acceptance checks.

The first implementation failed compilation because a face index declared
inside a nested static branch was unavailable to the normal lookup. CPU and
CUDA each recorded18 failed/13 passed; the baseline benchmark stopped during
warmup and its candidate route never ran. Those failures and immutable source
closures are preserved. Declaring the index in the enclosing branch fixes
the compiler scope; final numerical thresholds are unchanged.

This checkpoint qualifies position reuse on counted storage. It does not
establish the combined storage/cache gain versus production50b1b61c. Those
matched B2048 and separated-seed B8192 live pairs are running separately;
the production source remains unchanged. Parallel fan refinement is another
active candidate. Final-source wider four scopes, additional separated seeds,
remaining candidate decisions and a measured plateau are still required.
No physical convergence or RL training throughput is claimed.

Archive `native-cached-fan-positions-checkpoint-v1` contains617 entries,
132 unique blobs/17145827 stored bytes. Allocated CPU publication verifies
source/original/stored/decompressed hashes, both31-case reports,96 fields,
cached/direct geometry, complete paired counters and failed-run records.
Manifest SHA256:
`2ec2cfb3e63237e2fe765e7adc808158c9539c83ee39e43f9928e0e919ba4b19`.
The archive contains the patches, launchers, raw results and device telemetry.
