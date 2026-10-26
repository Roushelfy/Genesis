# Matched twelve-feature GPU validation: work ledger

Active goal: [the remote specification](ATLAS_CPU12_GPU_PARITY_GOAL_20261026.md)
at `0e2e819ca`. The independent CPU primary is `a10c5abe3`.
The preceding three/six-feature native measurements remain scoped history.
This ledger records work in progress, not a completion or accuracy claim.

All runtime payloads, snapshots and logs use
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-cpu12-gpu-parity`.
Scientific work runs in Slurm `rtx-mid`, image `1_26`. Existing environments
and compatibility links remain in place. The initial receipt verifies twelve
original unfinished files and twenty-three frozen experimental engine trees.

| Requirement | Implementation / pending evidence |
| --- | --- |
| Frozen CPU inputs and independent numerical oracles | Validated export: job 751986, 242 hashed artifacts, primary sanity results reproduced |
| Same 5196/5190 mother and 3522/3516 spaces | Same actual P/T/K, gauge/mass/relief and 12 spaces verified; four-bank numerical audit passed 1205 checks |
| Same prescribed cap and all positive selector contributors | Native q8/16/24 rule/load/selector and two-side controls passed, including forward/reverse repeated construction |
| Common relief, grouped solves, one stress recovery | Generic PreparedAtlas; original reduced residuals, merged fields and same-state unload/reload/isolation passed (2317 checks) |
| Published43, boundary18, static16/64 | All four frozen banks and all three selectors match CPU fields/peaks/resultants; independent spatial audit records M/Q budget failures in CPU construction itself |
| Matched GPU path and memory/profile measurements | Default serial-factor complete-path timings measured for small/large static banks, max/full and three selectors; bounded cooperative/packed scheduling and memory work underway |
| Native shell and volume adaptations, bounded tuning | Native twelve-direction graph/DPR adaptation frozen at a60f8ec39 and first qualification submitted; finer global shell coarse spaces tested separately |
| Frozen modules and full policy Franka workflows | Pending each native backend; pure rigid and optimized mother, max/full, small/large batches required |

Floating equivalence and spatial errors have separate acceptance decisions.
The initial parity protocol targets FP64 relative `1e-10` for geometry,
interpolation and assembled operators, and `1e-8` for solved fields, with
quantity-specific absolute floors recorded by the driver. Native solve
acceptance remains original unscaled relative `1e-8`, absolute `1e-11`.
A full mother residual of a lifted approximation is only a spatial diagnostic.
No unconditional mother correction is part of the matched construction.

## Observed qualification boundary

Four-bank CPU/GPU equivalence is positive. The largest recorded full-tensor
relative difference is about 2.7e-13; native exact/reduced residuals remain
well inside the original contracts. Same-environment release/reload and
partial grouped-load clearing are tested; this is separate from the pending
complete-scene checkpoint/reset qualification of the native adaptation.

The prescribed CPU construction itself has a spatial negative for applicable
shell resultants. Published43 continuous weighting has worst full field 7.5895%,
peak 0.9738%, N 5.2862%, M 10.3244% and Q 19.9683%; GPU reproduces them.
Nearest has M 12.5722% and Q 19.9685%. The full 10% N/M/Q budget is therefore
not met, even though field and sampled peak budgets are met. No tolerance is
relaxed and no full-mother correction is applied to hide this approximation.

Default B64 nearest max timing is approximately 128.6 ms for the native mother
and 646.6 ms for the native Atlas. These are numerically admitted module
measurements; they do not establish a spatially qualified rate or a complete
Franka gain. Profiling places about 643 ms in projection/solves/lift/recovery,
while cap/selection/coverage cost about 2.2/0.27/0.89 ms. Actual-size cooperative
factor tests pass 183 checks and motivate a bounded scheduling follow-up.
Standalone factor latency is not treated as whole-path performance.

All failures and superseded harness attempts remain at the data run root.
The current machine-readable campaign ledger is `meta/progress.json` there.
