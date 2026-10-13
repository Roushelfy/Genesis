# Clean combination qualification in progress

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
currently running. Its result is still pending.

## Remaining qualification

The35-case matrices, new B16384 dynamic profiling/full-field capture and
B2048 original-wrench/independent-inertia-relief capture are running. Final
clean-source four-scope and copied/separated-seed scaling measurements,
remaining profile/cost decisions and final publication remain required.
The baseline plateau and this single paired improvement establish progress;
they do not complete those remaining requirements. Policy measurements refer
to inference rollout, and physical mesh/quadrature convergence remains outside
this phase.
