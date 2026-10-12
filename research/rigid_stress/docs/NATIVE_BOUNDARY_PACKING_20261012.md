# Native boundary diagnostics and bounded contact packing

This checkpoint selects two native changes after `cd76762e`: a masked
boundary-certificate fallback and stable sparse GPU contact packing. The
componentwise pair-union footprint law and final wrench, friction, complete
equilibrium/gauge and stress budgets remain unchanged. Contact-model repair,
full policy acceptance and large-batch optimization remain incomplete.

## Continuous incompatibility diagnostics

The original sampled separating-ray search and depth8/16,384-node continuous
verification run first. Already established continuous proofs retain their
ray, margin, bound and work counts. Only unresolved contacts enter a second
search, using on-demand witnesses at actual triangle/disk vertices, edge
intersections and circle boundaries. No witness cloud is stored. The second
continuous verifier permits depth12/65,536 nodes. Finite witnesses guide the
search; only the conservative continuous-domain verifier can report physical
incompatibility. Force, moment, source, radius and actual friction are preserved.

Both phases stay masked on the device, including successful and released
contacts. Reported search/verification work includes both phases and is
bounded by1,280 evaluations and81,920 visited nodes per contact. The proof
stack grows from25 to37 entries:159,744 extra bytes at B32, approximately
10 MiB at B2048. There is no added force/sample array.

Actual native qualification passes **56 CPU cases in475.98 s and56 CUDA
cases in668.20 s**. The step602 regression keeps its existing stronger
negative-bound assertion. A saved policy876 regression checks CPU/GPU
strict continuous separation, a deliberately unresolved short proof budget,
an independent all-selected-triangle/disk arc-Lipschitz bound, a feasible
control, release and environment isolation. A proven incompatible point
component remains a diagnosed rejection; this does not solve general legal
grasp recovery or authorize labeling such grasps unsupported.

On one GPU, three complete current-law B32 live repetitions give
**3081.969→3069.690 valid env-step/s**, ratio0.996016. The approximately0.4%
cost includes both masked fallback dispatches. The maximum logged peak
difference is1.287e-6 Pa, and complete residuals consume at most0.006296
of their unchanged budget. These healthy trajectories contain no numerical
failure or overflow. Rare incompatible-contact diagnosis has a larger bounded
tail; its cost is not represented by the healthy live average.

## Stable sparse GPU packing

CUDA collects valid raw indices atomically while clearing all original raw
metadata slots. Lists of at most32 active contacts use stable insertion
sorting back into the exact original order. Larger lists rebuild that order
with the original linear scan, preventing quadratic dense tails. CPU retains
its original scan. The change adds no buffers and drops no valid contacts.

The integrated native version passes **10 CPU lifecycle/diagnostic cases
in119.77 s and10 CUDA cases in137.72 s**. Additional CPU/GPU cases exercise
1,200 raw slots in each of six environments, sparse/mixed/full-density
inputs, release and restoration, including the31/32/33 threshold. Valid
masks, counts, maximum count, exact stable indices, metadata clearing and
force clearing agree. Native CUDA packing costs approximately0.043 ms for
sparse input,0.160 ms for full density and0.193 ms for the mixed restoration
case; these are stage measurements, not whole-step rates.

The isolated hybrid candidate passes10 CUDA cases in113.54 s and gives
approximately1.0255x matched live gain. Actual native confirmation then uses
the same typed workspaces and compares only packing: three full repetitions
give **3067.984→3131.582 valid env-step/s,1.020729x**, or95.875→97.862
batch-step/s. Each repetition includes1,280 steps and37 partial environment
resets after600 warmup steps, with zero invalid environment steps, per-env
failures or overflow. Native buffers stay167,626,674 bytes. Maximum logged
peak difference is8.649e-7 Pa and full residual budget fraction0.006147.
Different comparisons use different allocated GPUs; their gains are not
multiplied to claim a combined end-to-end speedup.

## Copied B1024 four-scope evidence

This separate run captures clean source **`cd76762e`**, before the two new
changes. It duplicates the same32-condition bank, including controller
delays, across all1,024 environments. It supplements scaling evidence and
does not replace diverse conditions, additional seeds or later-source
qualification. All environments and resets remain in the accounting.

| Scope | Valid env-step/s | Batch-step/s | Repetition length / resets |
|---|---:|---:|---|
| Rigid only |226,884.310|221.567|2,211/2,220/2,217 steps;2,784 resets each|
| Recovery only |31,356.677|30.622|1,280 steps each;frozen actual snapshot,no resets|
| Live rigid plus stress |27,796.274|27.145|1,280 steps and1,184 resets each|
| Policy plus rigid plus stress |Unverified|Unverified|Timed repetition aborted on contact recovery|

Rates aggregate three repetitions by actual elapsed time, rather than
averaging individual rates. Rigid's minimum10-second rule extends its
trajectory beyond the other scopes, so these rates are not an exactly
matched trajectory slowdown comparison. Recovery-only snapshot replay is
explicitly separate from live contact updates/reset costs. Recovery and
live each allocate3,092,433,490 native stress-buffer bytes; their measured
device free memory is94,235,066,368 bytes. Rigid-only free memory is
97,693,270,016 bytes. Torch's recorded peak is3,971,584 bytes in all three
completed scopes; that figure excludes Quadrants native allocations.

All three completed scopes have zero invalid environment steps/overflow.
The frozen recovery check preserves original points exactly, has normalized
force/moment errors8.930e-9/4.144e-9, nodal replay difference1.943e-16 N,
peak replay difference8.826e-9 Pa and full residual at most1.151e-11 N.
This is snapshot replay consistency, not a new independent CPU full-field
comparison at every B1024 step.

The policy run preserves failures in32 copied environments, index6 modulo32,
contact2: sampled wrench separation with continuous feasibility unresolved.
Its return code1 and complete logs are retained. No valid policy throughput,
complete policy acceptance or RL training throughput is claimed. The earlier
manifest's "warmup aborted" phase wording is corrected by the
[preserved tick807 input and independent audit](NATIVE_SOURCE_PRIORS_20261012.md);
its original traceback points into the timed repetition. Archived hashes
are retained unchanged.

## Reproduction and limitations

Hardware is NVIDIA RTX PRO6000 Blackwell Server Edition,97,887 MiB,
driver595.71.05, image `genesis:1_26`, Quadrants1.3.3, FP64. The B1024 GPU
UUID is `GPU-325bc5a3-b0fe-9a67-0857-3a4ad14c17fb`; native packing's
matched GPU is `GPU-adb3f72e-e837-07af-76f0-f0838bfdf2ed`. Each comparison
archives its own hardware/source record. The full low shell has480 P2
tetrahedra,810 P2 nodes and2,430 DOFs. Q10 squared-coordinate fans, s8
paths, verified single-face disks/nine-panel retry, dt0.01/one substep,
block4/effective8 and GJK mesh64/contact patches remain fixed. No physical
mesh/integration convergence experiment ran.

The [native checkpoint manifest](../evidence/20261011-nonconvex/native-boundary-packing-checkpoint-v1/manifest.json)
contains648 artifacts/24,130,790 bytes, SHA256
`9e9c93094b07ea37d20a0f9ef313123acd10b6bea45b9a25425ef247e2982fc2`.
Allocated CPU publication verifies all source closures and original/published
artifact hashes, matched physical/numerical settings and live counters.
Frozen numerical source, tests, saved fixture, benchmark/profile JSON, failed
policy logs, candidate/baseline drivers and launchers remain reproducible.
The condition-bank SHA256 is
`156d956bc5fc3827da8df236cf1695a4bf30a9a1e1d939c672f6a57d69d10930`.

The [closed model trials](NATIVE_BOUNDARY_MODEL_TRIALS_20261012.md) preserve
rejected direct-search and quadratic-packing designs and the unselected
aggregate-pair experiment. A general aggregate model must retain local
source loads even when total force/moment cancel. Legal boundary-crossing
integration, robust physical load modeling, CPU/GPU sample-count differences,
policy recovery, diverse1024/2048/larger workloads and an optimization plateau
remain open. This checkpoint does not complete the active goal.
