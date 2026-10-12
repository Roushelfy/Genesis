# Timed native rollout outcome collection

The benchmark's optional `--trajectory-metrics` records actual grasp outcomes
and recovery work per environment with Quadrants. It applies to `rigid`,
`live`, `policy` and `policy-rigid` scopes. The public position getter and
device collection kernels execute inside every measured rollout step;
reporting/export occurs after the final timed synchronization. Native metric
buffers cost 106 bytes per environment. Production stress computation and
the final wrench, equilibrium/gauge and stress budgets are unchanged.

A completed lift succeeds when every observed controller phase in
`[0.52,0.60)` has egg height at least 0.05 m above its initial height and
nonzero (>1e-12 N) contacts with both finger links. The cycle is counted at
crossing phase 0.60. Incomplete windows are reported separately. Weak grip
`[0.70,0.76)` and release `[0.92,1)` report descents exceeding 0.001 m over
one dt=0.01 step. These are declared task metrics; numerical recovery success
does not establish grasp success. Phase wrap clears cycle flags while
preserving aggregate counts; each repeat resets metric state.

Recovery statistics include per-environment evaluation sum/max, maximum
sample count, refined-integration frames, elastic correction/fallback work
and invalid stress steps. Percentiles and full environment arrays are saved.
They describe parallel work, rather than attributing GPU wall time to
individual environments. Dynamic critical-path costs still require profiling.

Actual GPU validation uses B16, seed623102, the low complete hollow shell
(480 affine P2 tetrahedra/810 P2 nodes/2430 DOFs), FP64/Q10/s8, explicit
`body_source_prior`, maximum output, mesh64/GJK patch-on, dt0.01/substep1.
Hardware is RTX PRO6000 Blackwell Server Edition,97887MiB, driver595.71.05,
container1_26, Genesis1.4.3/Quadrants1.3.3. Three direct CLI scopes run
sequentially on GPU2653f4d6 with600 warmup and1280 timed steps each.

| Scope | Valid env-step/s | Completed lifts, success/failure | Invalid env-steps | Selected resets |
|---|---:|---:|---:|---:|
| Rigid |3991.525|25/0|0|19|
| Live rigid + stress |1171.462|25/0|0|19|
| Policy + rigid + stress |1109.642|25/0|0|19|

All1235 observed lift-window environment steps pass in each scope. Rigid and
live final positions agree within1e-12 m, and every environment's grasp,
contact and descent counters match exactly, supporting auxiliary-dynamics
isolation. Live maximum fitting work ranges9..23 evaluations and777..1040
samples per environment; integration retries, elastic corrections/fallbacks
and stress failures are zero. Policy uses the declared26→128→128→7 FP32
inference network and passes the existing1e-9 rad reference budget. These
B16 rates validate instrumentation; they do not establish large-batch
throughput, an optimization plateau or full RL training performance.

Recreate the archived `run_rollout_metrics_validation_v3.sh` and sequential
pair wrapper. Run each scope with `--trajectory-metrics --warmup 600
--steps 1280 --repetitions 1 --minimum-seconds 0`. The source epoch is
`ee063542` plus the recorded benchmark instrumentation patch; later selected
port pruning was absent. The earlier external observer v1 API error from
`get_pos(copy=False)` is retained; corrected observer v2 and direct CLI v3
pass, with identical live counters between v2/v3.

Archive `evidence/20261011-nonconvex/native-rollout-metrics-checkpoint-v1/manifest.json`
contains364 entries/99 unique blobs/2,367,358 stored bytes. An allocated CPU
publisher verifies originals, stored/decompressed hashes, complete source
closures, all three scopes, metric buffer bytes, direct/observer agreement,
policy budget and rigid/live isolation. Manifest SHA256:
`bdf34bdfa6be47dc928f0adaed7989877a33b64788a531f311259d277ca752cb`.

The extension goal remains active. Wider batches/seeds, matched four-scope
results, dynamic profiling, actual tail costs, remaining applicable
optimizations and evidence of a plateau remain outstanding.
