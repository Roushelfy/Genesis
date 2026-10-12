# Contact storage and refinement decisions

These matched pairs use the low complete hollow shell, FP64/Q10/s8,
`body_source_prior`, max output, mesh64/GJK patch-on and dt0.01/substep1.
Storage/scheduling changes preserve every physical sample, original contact,
radius, friction and final wrench/full gauge residual/stress budget. Each
route runs sequentially on one RTX PRO6000 Blackwell GPU,97887MiB,
driver595.71.05/container1_26, Genesis1.4.3/Quadrants1.3.3, with600 warmup
and three1280-step changing live repeats. Observations, contact updates,
reset/metrics/history/fallback and all environments remain timed.

## Counted storage plus current-step positions

The combined prototype compares complete counted profiles/indices and cached
positions with full fan storage/direct reconstruction. Both retain selected
eight-sample-lane retry; their disabled capacity counter adds4 native bytes
to the production-equivalent baseline. Source closures preserve the complete
implementation. Seed734001/B8192 is separated from623102/B2048.

| B | Original valid env-step/s | Combined valid env-step/s | Combined batch step/s | Ratio | Native bytes saved |
|---|---:|---:|---:|---:|---:|
|2048|61548.787935|62957.376407|30.740906|1.022885722|3532161048|
|8192|96453.911711|99146.497345|12.102844|1.027915774|14116257816|

Native stress bytes change5498093154→1965932106 and
21773033058→7656775242. The assigned-device200ms sampled peaks,
including setup and trajectory, are11247→7919MiB and41447→23143MiB.
These are sampled values, without an exact transient-maximum claim.
Every repeat has zero invalid environment steps/overflows and zero failed
completed lifts. Final rigid positions agree within1e-12m and every
per-environment rigid grasp/contact/descent counter agrees.

The separately published [position qualification](NATIVE_CACHED_FAN_POSITIONS_20261012.md)
provides31 CPU/31 CUDA cases and96 independent full fields, including fresh
native inputs and exact cached/direct geometry agreement. The combined pair
establishes an end-to-end improvement over full storage rather than relying
on an isolated cache gain. It is selected for production integration; cleanup
and its combination with parallel fan sampling require new qualification.

## Parallel refined fan sampling

Each selected source first constructs the original fan charts. A32-lane warp
then visits the complete physical quadrature range and atomically stores every
eligible sample once. CPU retains sequential sampling. Sample order can change;
topology, profile, footprint and final acceptance do not. All constraints
sharing changed supports retain their existing invalidation/refit semantics.

On GPU0e801d13/B2048/623102, live improves62494.434667→63566.167870
valid env-step/s,1.017149258x, with unchanged5498093150 native bytes.
Original repeat seconds41.952595/42.271805/41.615926 become
41.287939/41.349123/41.081581. The saved condition banks agree fieldwise;
every repeated trajectory has2323 resets,3142 successful and zero failed
completed lifts, zero numerical failures and unchanged per-environment rigid
counters/positions. CPU23 cases pass in302.89s, CUDA23 in426.62s,
with48 recorded independent traction/nodal/elastic fields across backends.

A separate intrusive1280-step dynamic audit on GPU4fdce966 preserves all
three refined hard frames. At ticks468/959/1068, fan refinement is
4.414685/4.335135/4.412363ms; whole fenced steps are
242.144489/251.198436/245.595807ms. The preceding eight-lane audit reported
fan calls around107ms and whole steps343–356ms on another allocation.
Those cross-allocation diagnostics describe the tail; the paired live result
above establishes throughput benefit. Remaining serial refits take
75.553438/87.822292/75.542249ms. Mean fenced step is36.610661ms,
with zero invalid steps and unchanged2323 resets/3142 completed lifts.

An independent CPU stiffness/gauge/factor audit checks160 captured GPU
fields against unchanged complete residual/tensor/von Mises/global peak
budgets. Maximum peak difference is1.768057700e-6Pa and residual budget
fraction0.00561184704. It receives captured balanced RHS rather than
independently reconstructing dynamic contact fitting/relief; separate fixture
oracles cover traction/nodal recovery. Fences and exports perturb scheduling,
so these latencies are not headline rates or individual-environment GPU times.

The first candidate overwrote an underscore loop variable with a returned
vector, causing a compiler assertion. CUDA recorded10 failed/13 passed;
CPU recorded failures followed by a compiler segmentation fault. Warmup and
a targeted CUDA smoke also failed. All closures/raw records are preserved.
Removing the unnecessary loop fixes compilation; CPU/CUDA targeted smokes
and full matrices then pass. This candidate is selected for integration
with counted positions; the production combination remains under validation.

## Small unselected gains

A cooperative refit candidate cold-fits every constraint sharing a changed
fan support, then applies serial fallback only to remaining failures. It
preserves the final serial continuation. CUDA23 cases/24 independent fields
pass in438.70s. The matched B2048 pair has1.002938634x live ratio, zero
failures and unchanged condition banks/per-environment rigid outcomes.

A scalar sample-contiguous profile/index layout keeps all positions, samples
and mathematical reductions unchanged. CUDA31 cases/48 fields pass in679.41s.
Its B2048 matched live ratio is1.003031333x, also with zero failures and
unchanged condition banks/rigid outcomes. Both approximately0.3% effects are
within ordinary repeat variation. Neither is selected for throughput; their
implementation, qualifications and full pairs are retained for review.

## Publication and unfinished work

`native-contact-tail-storage-checkpoint-v1` has1796 entries,221 unique
blobs/116272845 stored bytes. Its allocated CPU verifier checks all source/
original/stored/decompressed hashes, ten complete benchmark routes, condition
banks where recorded, full per-environment counters, four regression reports,
120 independent fields and failed-run/smoke records. Maximum normalized wrench
error is7.386596753e-10, residual budget fraction0.0458562252 and peak
difference8.155802789e-6Pa. Manifest SHA256:
`f9c2d10cb698fa7475dc08f98374d01e7626eea07a74dcc7c1c7644db85fdec5`.

`native-parallel-refined-fan-dynamic-checkpoint-v1` has189 entries,
123 unique blobs/20822155 bytes and the160-field independent audit.
Manifest SHA256:
`2960525ecafffb75913f9713af01f60dfa7466b05ce599f3e7494cfc0e55f8cf`.

The production combination is implemented locally with exact sample growth,
position reuse and parallel refined sampling. Experimental switches are removed
and the capacity maximum is an ordinary typed scratch tensor. Its regression
and matched live comparison are pending. Larger B16384/B32768 prototype probes
do not establish final-source scope measurements. Final wider four scopes,
separated seeds, remaining bottleneck decisions and a plateau remain required;
the nonconvex goal is active. No physical convergence or RL training is claimed.
