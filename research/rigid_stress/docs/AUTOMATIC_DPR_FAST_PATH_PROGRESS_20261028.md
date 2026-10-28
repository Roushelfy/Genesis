# Automatic DPR fast path: execution progress, 2026-10-28

The [remote goal](AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md) at
`d661349b12d76ace126840107d0498562b9f3d76` remains active. The data root is
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261028-automatic-dpr-fast-path`.
Its `meta/requirements-v1.json` records22 complete-scope requirements. This phase
preserves the completed planar delivery and all original/frozen work.

## Actual failure diagnosis

Exact archived volume edge/q16 inputs at frame221 and policy tick606 both have
infeasible sampled nonnegative support. Native residuals are0.240122 and222.267,
with active Hessian rank collapse. Both exhaust80 Newton evaluations without
reaching the40-trial line-search limit. Increasing a near-zero stopping tolerance
cannot fix these discrete support problems. Original status3/errno128 and stale
uncomputed wrench-error fields remain explicit.

The traced selected failing sources match unmodified native outputs byte for
byte. The frame221 trace changes five other coefficient entries by1-7ULP, so
its original full-array failure and RC1 remain retained. Independent LP and
native selected-source diagnostics support the diagnosis without claiming
whole-array telemetry equality.

## Implemented default contracts and runtime checks

Ordinary Options and `.dpr()` now select pressure/patch/Hertz/Hertz. Omitted
intrinsic/resolved settings resolve to their supported manual/dpr context, and
explicit conflicts retain their original errors. API factories, Franka and
benchmark entrypoints preserve omitted values until public Options resolves
them. The benchmark records declared and actual choices separately. Explicit
manual/legacy inputs and SCR remain available. Volume and Q4 shell loading
remain the backend/loading defaults.

Source `150c5d85cf2079838db3c2b7d29e437649d79128` passes20 CPU and20 CUDA option,
public shell, intrinsic and resolved lifecycle cases. The earlier v1 test attempt
failed collection because its sparse helper omitted `shell_cpu`, before running
any test. A new captured dependency-complete helper fixed that test harness;
original outputs remain preserved. Numerical defaults are still provisional
edge/q10 pending the fast/adaptive qualification and selection below.

## Frozen field evidence and admission work

Before candidate selection, `meta/application-budgets-v1.json` freezes5% full
3x3 tensor error,5% global sampled-peak error and2% peak underestimation, with
separate loaded/remote-region and absolute-Pa measures. Every candidate also
retains original strict load, fit, wrench and older field-budget decisions.
Angle/q128 versusq256 reference stability is checked independently.

Initial frozen-input angle/q16 errors are1.315e-6 tensor/2.104e-6 peak for the
conditional frame221 contact control, and3.214e-6/5.652e-7 for the complete tick606
snapshot. Both satisfy the application budgets. Their old1e-6 raw-load comparison
budgets fail and remain recorded. This evidence covers two inputs, not arbitrary
contacts or complete workflows.

Six surrounding frames now retain actual recovery angular velocity, pose and
all contacts. Both rejected sources match their historical10-field inputs
byte for byte. Subsequent full-field work uses those actual mechanical states;
the older frame221 zero-omega comparison retains its conditional scope.

Six complete nearby mechanical snapshots and four predeclared finite-footprint
box/U controls provide60 normal-rule rows. All58 native-valid rows pass the
application budgets. The other two preserve the historical edge/q16 rejections.
The worst measured regional tensor error is0.985458% for the convex-edge angle/q8
control. Its peak underestimation is0.185927%. The authored edge/tip controls
exercise nonlinear pressure correction. No converged-but-inaccurate case has yet
been observed, so false-accept detection remains unexercised.

The initial same-coefficient low-rule admission score falsely rejects accurate
loads. Methodv2 predeclares whole-contact checker mass normalization while
keeping the1e-3 score threshold and application budgets fixed. The new controls
show that normalization alone still falsely rejects all40 angle/q8-64 candidates.
Checker centroid and face-shape discrepancies remain after removing mass bias.
This method is unsuitable as the selected cheap default on present evidence.
The next bounded comparison uses independently fitted low-rule loads, with
additional coarse-rule controls on the existing authored inputs. Original rawv1
scores, thresholds and completed failures stay retained.

An initial CPU postprocess fails its exact raw-score reproduction guard.
The cause is a changed reduction order after NPZ loading changes a noncontiguous
native array view to C-contiguous layout. Restoring the declared original
Face/Contact/Batch/value layout reproduces all36 scores byte for byte. The
repair retains the zero-tolerance guard and original failed receipts.

## Remaining delivery work

The explicit volume adaptive prototype is being implemented with a shared
8/16/64/128 rule bank, source-stable pre-scatter decisions, selective repair and
actual route counters. It must preserve source loads, material support, final
wrench checks, CPU/fused paths, max/full output and lifecycle before a new
numerical default is recommended. Complete scripted/policy accuracy, public
examples, matched timing/profiles/memory and final documentation remain open.

A short public Franka example now accepts a public Options object and reports
max/full shapes and units for volume or fixed-thickness planar-k32 shell. Its
verification worker invokes that same ordinary example and, for viewer checks,
starts an owned Xvfb display. Runtime example qualification is still pending.
The new throughput protocol freezes three paired fresh-process blocks atB4/B8192,
max/full module and complete policy scopes, with separate profiles and memory.
