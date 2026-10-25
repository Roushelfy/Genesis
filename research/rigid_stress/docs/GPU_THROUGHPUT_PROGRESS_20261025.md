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
