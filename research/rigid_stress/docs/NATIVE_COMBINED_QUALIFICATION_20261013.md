# Clean combination qualification

The nonconvex extension remains active. The clean combination integrates
coalesced cached coordinates, valid-source capture, FP32 warm Newton Hessians
and previous/current raw diagnostic clearing. Every production computation
remains native Quadrants. Model, sampling and final accuracy budgets match
production92ec3e0c. The numerical source is the recorded81bf64e2 base plus
the exact source snapshots/patches in the checkpoint below.

## Matched complete dynamic comparison

RTX PRO6000 Blackwell Server Edition, GPU
`GPU-5c0d0cca-f892-19d7-d4f3-b68ab73a520b`, B16384/seed991909,
FP64/Q10/s8/body_source_prior/max, mesh64/GJK patch-on,dt.01/substep1,
600warmup and three1280-step changing-contact repeats per route. The full
condition bank is identical. Both routes include all environments, observation,
contact parameter updates, reset, metrics, history and fallback.

| Route | Effective env-step/s | Batch step/s | Repeat seconds | Native bytes | Sampled peak MiB |
|---|---:|---:|---|---:|---:|
|92ec3e0c baseline|106145.289|6.478594|197.801658 /197.396217 /197.523298|16138591818|44551|
|Clean four-candidate combination|113913.060|6.952701|184.101830 /183.937114 /184.264366|16138591818|40339|

The measured ratio is1.073180556 with nonoverlapping repeat ranges. Every
per-environment rigid/contact/grasp/descent counter and final rigid position
matches. Every repeat has18571resets and25124successful lift cycles, with
zero invalid environment steps, scatter overflows and failed lifts. The full
per-environment records and slow work remain retained. Memory telemetry uses
200ms assigned-GPU samples including setup, rather than exact transient peaks.
This pair is a combined result; isolated candidate gains are not added.

Archive `native-final-combined-attempt-checkpoint-v2` preserves this pair,
3CPU/3CUDA hard/lifecycle smoke,33CPU passes and the first CUDA regression's
32passes/1failure. All original/stored/decompressed checks and exact large-JSON
reconstruction pass. Manifest:
`945fe3029b5f8d05ef0cb38929d1a2703d5bff240f042c95d6515611be2b4166`.
The failed assertion is explicitly included; this archive alone does not
establish complete qualification. The v1 regression source list omitted the
unchanged shared-lifecycle test. Its exact recorded Git-base blob and hash are
provided and verified in the supplement. Later runners include it explicitly.
The first publication attempt's missing supplement source-list error is also
retained; publicationv2 supplies the verified supplement as explicit artifacts.

## Isolation comparison and final budgets

The original failure compares a warmed result with the first cold result using
node-force RTOL1e-12/ATOL1e-15N. Its largest violating difference is
3.48909721e-11N. A new independent equal-history control already differs by
2.116992692e-11N during the initial calculation, before releasing any
environment. A same-GPU diagnostic forcing only Hessians back toFP64 passes
the strict comparison. This identifies arithmetic/convergence-path repeat
precision as the immediate assertion cause.

The corrected test keeps exact released/reset observations and compares
independent controls against the actual physical budget. It additionally
checks the original aggregate wrench about a force-weighted application-point
reference and propagates each original-point force/moment allowance to that
reference. Independent CPU FP64 stiffness checks include every equilibrium
row and the six gauge rows. Final acceptance remains wrench1e-8, residual
max(1e-11N,1e-8||rhs||), stress RTOL1e-4/ATOL1e-3Pa. The source-prior
full-field/cone oracle matrix retains those exact gates.

Two CUDA shared-lifecycle cases pass in93.45s. Across their complete release,
reactivation and streaming paths, worst unaffected-environment nodal-force
difference normalized by physical load is5.254869784e-10, original wrench
error5.386640765e-10 and complete residual budget fraction0.001664682001.
All comparison diagnostics, forced-FP64 results and failed versions are
preserved by `native-final-isolation-budget-checkpoint-v1`.
All original/stored/decompressed hashes verify; its manifest SHA256 is
`97371d5c405d8d2fd70394a2226dd5f4dcd1c3eefbee1375c7b387692df90046`.

The additional native max/full checkpoint test checks restore invalidation,
recomputation and partial reset in an actual low-shell contact scene. Its
initial test incorrectly expected displacement to survive restore's declared
invalidation. Those failures are preserved. The corrected test compares
separate one-/two-step branches and checks unaffected observations exactly
across partial reset. It is included in the35-case CPU/CUDA combined matrix
now passing on both CPU and CUDA.

The complete35-case matrices pass: CPU570.217s, CUDA818.219s. Their96
independent full-field cases have worst original wrench error7.388121062e-10,
complete residual budget fraction0.0458562252 and global peak difference
8.385381079e-6Pa. The standalone cached/direct geometry checks remain exact.
The complete pair and same-numerical-source CPU/CUDA snapshots qualify in
`native-final-combined-qualification-checkpoint-v1`, manifest
`f669160222ba05ab2c1f628a6d652502a7811e3555de56c758fa77ea229453fc`.
Every original/stored/decompressed hash and large-JSON reconstruction passes.

## Current dynamic profiling and independent oracles

The B16384/623102 changing trajectory includes600warmup and1280instrumented
steps. It retains all per-environment work and outcomes:18571resets,
25124successful lift cycles, zero failed lifts or invalid steps. Serialized
fenced mean is147.438508ms; step[minimum,median,p90,p99,maximum] is
24.112455/161.758372/175.542341/309.684237/341.323833ms. Retry-frame mean
is312.371326ms versus144.554150ms without retry. Per-environment maximum
fit evaluation[minimum,median,p90,p99,maximum] is5/11/16/146/401.
These instrumented latencies are separate from headline throughput. Their
different GPU and scheduling also preclude an exact speedup comparison with
the baseline instrumented mean.

| Stage | Mean ms per instrumented step |
|---|---:|
|Initial fan sampling|43.100137|
|Warm cooperative fit|25.410934|
|Rigid constraints|14.441849|
|Signed nodal scatter|10.235963|
|Blocked inverse application|10.240678|
|Surface ports|6.424756|
|Complete elastic residual|5.554941|
|Global peak|3.458618|
|Source resolution|3.098264|
|Raw contact packing|0.486218|
|Source capture|0.104961|

Nested refined-fan chart/sample timings overlap their wrapper and are not
added. The profile retains22retry frames, original attempts and complete tail
environment records. CPU FP64 independently checks464captured complete fields:
worst residual budget fraction0.00621422449, global peak difference
1.226493623e-6Pa. Archive `native-final-combined-dynamic-checkpoint-v1`,
verified manifest
`45dc02aa5a4efe5802f58e0ee0af7bbd39b990526bb36eff909dc62aa97e24e1`.

An additional B2048/623102 trajectory records original application points,
force/free moments, angular velocity, nodal forces and recovered state.
Independent CPU FP64 checks160fields including its own mass/centrifugal
loading/inertia relief, every stiffness/gauge row and tensor/von Mises/global
peak. Worst original aggregate wrench error is8.623318699e-9, independent
relief RHS difference4.616757194e-8of the residual allowance, complete
residual budget fraction0.005644716894, peak difference1.391832484e-6Pa.
Archive `native-final-combined-wrench-oracle-checkpoint-v1`, verified manifest
`ad1f06bdbede05ffed1bce5c1607577879952d4ed5fd9d751a24099098973d81`.
Both new oracle publications include the35-case CPU qualification. The full
35CUDA lifecycle qualification is retained in the combined archive above.

## Remaining qualification

Final clean-source four-scope and copied/separated-seed scaling measurements,
remaining profile/cost decisions and final publication remain required. The
43.100137ms initial sampler motivates a remaining cost probe for its per-sample
count/reservation/max-distance atomics before selecting or rejecting a warp
aggregation implementation. Raw capture/packing is now0.591179ms combined;
raw-source compaction would need a separate benefit beyond removing these
measured writes to justify its source/index/lifecycle complexity. Neither cost
argument proves a universal upper bound on other memory effects.
The baseline plateau and this single paired improvement establish progress;
they do not complete those remaining requirements. Policy measurements refer
to inference rollout, and physical mesh/quadrature convergence remains outside
this phase.

The counter cost probe now validates six synthetic configurations at the
captured mean/max aggregate-count scales and32768/65536/131072source columns.
All candidate/accepted counts, FP64 maxima and every accepted index permutation
pass. Off/on/off means show2.476–5.390ms savings for warp count/reservation/
max aggregation, with no geometry/profile work in this scope. Raw diagnostics
can duplicate body-union work, so this motivates an actual sampler/rollout
comparison rather than establishing a production bound or saving. Three failed
probe versions are retained: unsigned shuffle signatures were corrected, then
configuration buffers were retained to prevent the observed recycled-buffer
count accumulation. The completed cost archive is
`native-sampler-counter-cost-checkpoint-v1`.
Every original/stored/decompressed hash verifies. Manifest:
`49b56b3d72aa55d756dd96f7cb17ff3c4b179f27f5090264e679ed3440386804`.

The private sampler prototype pads complete32-lane source groups, reduces
candidate/accepted counts and FP64 maximum distance, reserves one complete
sample interval and uses a lane prefix to write every eligible sample. The
original geometry/area/Gaussian/path/radius/friction arithmetic is unchanged.
Its three CUDA hard/growth smoke cases pass in157.13s. A35-case CUDA matrix
and same-GPU B8192/623105 complete changing-contact pair are running/pending.
It remains unselected. Production numerical source `eba3aeb2` is pushed and
keeps the qualified four-candidate implementation while this decision proceeds.
