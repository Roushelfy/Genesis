# GPU throughput follow-up execution record

Active specification: [follow-up goal](GPU_THROUGHPUT_FOLLOWUP_GOAL_20261025.md),
remote revision `cb9ae667f`. The accepted starting implementation is frozen as
`gpu-throughput-v0` at that revision. This record describes ongoing work.

Runtime results and receipts use the data location recorded in
[DATA_LOCATION.md](../../../DATA_LOCATION.md), run `20261025-gpu-throughput`.
`meta/start-state.json` retains hashes of all twelve unrelated unfinished files,
the worktree inventory and the initial job list. Scientific work uses Slurm
`rtx-mid`, image `1_26` and the established data-side environment.

## Numerical budgets fixed before candidate tuning

Exact scheduling candidates retain every existing load, residual, acceptance
and lifecycle assertion. Paired controls additionally require relative complete
generalized-load error below `1e-10` and same-sample complete tensor error below
`1e-8`. Contact force/moment preservation remains `1e-8` in the existing scales.
Zero-load cases use the original absolute checks.

Approximate candidates are separate choices. Reduced quadrature requires
relative complete load error below `1e-6`, tensor and sampled-maximum error below
`1e-4` against quadrature 16, in addition to the unchanged acceptance checks.
FP32 response storage or accumulation requires tensor and maximum error below
`1e-4`; the current original-operator residual remains the gate for production
acceptance and exact correction/fallback. Precision is isolated from scheduling.
History prediction requires tensor and maximum error below `1e-5` over sustained
trajectories with predictions feeding subsequent state, plus the same residual
gate. These budgets do not certify continuum discretization accuracy.

Feature-mesh research has a separate spatial-approximation budget of 10% for
common-point tensor norm and 20% for sampled maxima against the covered joint
mesh. Continuity, full footprint coverage, material-side identity and common
frames are additional requirements. A failed budget is retained as a negative
result and cannot justify a production default.

## Initial work

- Fresh graph-enabled module/complete-scene baselines cover B64 and B8192,
  rigid and volume/shell max/full. Separate device profiles cover shell B64,
  shell B8192 and volume B64. Prior small-batch results remain historical.
- Candidate `3bf959e45` builds a conservative current-contact face list in
  shared memory before shell pressure correction. Each iteration retains its
  original integration and convergence logic. A footprint exceeding the local
  list capacity takes the exhaustive path. Volume scheduling is unchanged.
- The candidate ledger, heterogeneous workloads and remaining response,
  precision, atlas, screening, history and condensation controls are pending.

## First measured results

Fresh secondary device profiles reproduce the pressure bottleneck. At B64,
pressure integration/correction is 2.039 ms, scatter 0.569 ms. At B8192 the
corresponding times are 46.714 and 5.339 ms. The face-list candidate reduces
pressure to 1.642 and 42.776 ms respectively. These intrusive stage results
are provisional until paired complete-scene timings finish. Both contact
control routes pass flat, warped, egg and concave-valley cases, including
inner/outer surfaces, friction, slot permutation and release/recontact.

Candidate `d3a79d7f5` also prepares the fixed orthogonal Q4 projection once.
The physical inverse still checks its original position residual and domain.
Both contact routes pass the same controls. Profiles and paired timings are
running on frozen `gpu-throughput-v2`.

`response-v1` evaluates complete nodal-force/moment plus six centrifugal
channels, with relief already included exactly once in the native response.
Shell full output includes all N/M/Q resultants. Pure Quadrants construction
and application pass the same-load tensor comparison. Shell FP64 storage is
320.36 MiB, volume 43.24 MiB for these coarse models. At B1024 shell, existing
complete recovery is 1.827 ms. The best tested FP64 sparse H application takes
3.593 ms for max and 8.878 ms for full. FP32 storage/accumulation gives
1.537/3.744 ms, with field difference 2.87e-7. These H times omit the displacement
and original-operator residual that production must retain, so they are lower
bounds on a complete replacement's cost. FP32 storage with FP64 accumulation
is slower than FP64 in this kernel. Dense products also lose on the tested
small batches. Tile sizes 128/256 are recorded separately.

The conservative sampled-max prototype includes every force/moment/spin column
and uses two 1e-10 upward relative margins in its FP64 bounds. Shell screening
evaluates 2--9 of 60 blocks, agrees with exhaustive output to 2.5e-15, and costs
0.809 ms at B64 and 0.986 ms at B1024. B64 exhaustive H scanning is 0.305 ms.
The volume B64 screen costs 1.994 ms versus 0.219 ms exhaustive H scanning,
despite evaluating only 1--34 blocks. Sample skipping alone is insufficient.
These controls cover the retained loaded bank, not all possible stress fields.

The isolated packed-response precision comparison keeps FP64 factor,
correction inverse, arithmetic and native acceptance while changing only the
stored surface response. It is recorded separately as `packed-precision-v1`.

The first bounded CPU atlas prototype failed during mother-grid construction:
unrounded base coordinates duplicated rounded feature-grid coordinates,
creating near-zero-width cells. `atlas-v1` remains failed. `atlas-v2` rounds the
union using the existing 12-decimal geometry convention before deduplication.
It tests full-footprint coverage/fallback, a three-region junction, common-mother
embedding, per-(mesh, environment) aggregation and both exact hierarchical
Schur directions. This is a planar spatial/algebra control, separate from native
Franka throughput.

## Paired pressure and qualification results

The paired v0/v1/v2 B64 complete-scene medians are 7.274/6.714/6.461 ms;
B8192 is 85.890/78.283/70.049 ms. Frozen module medians are
4.464/4.103/3.828 ms and 65.480/62.005/55.086 ms respectively. All repeats
have zero invalid environment-steps. The v2 pressure device stages are
1.449/36.571 ms. Complete q16 scene fields, N/M/Q, loads and every-step
sampled peaks pass exact paired budgets; a 64-condition heterogeneous cycle
also passes. The maximum paired peak difference is about 2.1e-15 relative.

Quadrature 8 passes the static fixtures but fails the evolving native pressure
fit at frame 108. Quadrature 12 completes the native cycle, but its worst paired
peak difference is 6.901e-4 (frame 476, environment 1), exceeding the predeclared
1e-4 budget. Keep q16. Existing four-column history passes native acceptance,
but fails the separate 1e-5 peak budget (4.548e-5 at frame 545, environment 0).
Its B8192 complete-scene timings are slower: shell 71.303 to 84.282 ms;
volume 47.217 to 69.942 ms. Keep history disabled for the recommended path.

The first new paired audit mistakenly demanded bitwise equality of contact
radii. Its failure is preserved. The corrected audit uses the established
rtol=1e-12 radius contract and independently records the stronger bitwise
observation; all original baseline assertions remain unchanged.

Output transform sharing (v3) passes numerics but does not improve the measured
stress stage (B8192 5.054 to 5.168 ms). It was reverted. Only the measured
pressure changes and device failed-environment compaction remain in v5/v6.
Compaction applies only to packed CUDA exact fallback at B>=1024; the ordinary
history-free fused path retains its schedule. The controlled B8192 shell
1%-failure case falls from 1.261 to 0.658 ms including residual checking.
Final native poisoned-displacement/isolation controls and scene timings are
running. The scratch allocation is 4*B+4 bytes per state, not persistent history.

## Remaining candidate evidence

Isolated FP32 response storage with FP64 arithmetic requires one correction and
is slower (shell B8192 10.993 to 37.533 ms); it is not a recommended default.
A sustained two-frame prototype feeds every accepted prediction into subsequent
history and resets one environment mid-cycle. All 600 steps pass the 1e-5 field
and peak budget; maximum field differences are 1.70e-7 shell and 1.79e-7 volume.
Only 442/2350 loaded environment-steps predict successfully. Module timings are
being collected before drawing a performance conclusion.

Actual cached Kq matches the direct CSR product, but refreshing plus checking
costs more than the direct residual: shell B8192 1.260 versus 1.104 ms; volume
2.877 versus 2.625 ms. Two-buffer rotation also loses at large batches (shell
0.419 versus 0.360 ms, volume 0.697 versus 0.591 ms). No stale RHS-as-Kq shortcut
is used. The installed Quadrants Python API probe found no TF32/WMMA entrypoint;
TF32 was not executed or claimed.

CPU atlas controls cover 3/5 libraries, two refinement widths, continuous blend,
nearest, top-two and narrow overlap, with complete finite-footprint fallback.
The three-library cases meet the spatial budgets, but 19/80 five-library cases
exceed the 10% field budget (worst 15.9%). Nearest and hard top-two exhibit
junction jumps; smooth overlap allows three or more contributors. Both exact
coarse/local Schur directions reproduce the direct solve, but cost roughly
twice as much in these CPU controls. No improved POD basis is justified.

The nested planar common-mother identity also passes an actual Quadrants GPU
recovery prototype. At B1024, three-library independent recovery takes 0.987 ms
versus 0.545 ms common-mother; five-library 1.393 versus 0.773 ms. Selection and
grouping add about 0.205/0.326 ms respectively. These timings use independently
prepared fixture operators/solutions, exclude pressure and solve costs, and
are not native curved-egg end-to-end claims. Native nonnested offset surfaces
lack a validated common embedding; independently refined models also change
lumped inertia relief. The clamped planar identity does not authorize changing
those production semantics. Retain the promising prototype and its limitation.
