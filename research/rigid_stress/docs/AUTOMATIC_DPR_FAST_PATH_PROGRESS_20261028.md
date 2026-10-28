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
exercise nonlinear pressure correction. An additional bounded eight-row coarse
control set now contains three converged-but-inaccurate cases: convex-edge
edge/q4 and angle/q2, and U-inner angle/q2. Their worst regional tensor errors
are19.652%,17.741% and6.869%. The original fit, source-wrench and K checks pass,
so solver convergence alone would not detect these integration errors.

The initial same-coefficient low-rule admission score falsely rejects accurate
loads. Methodv2 predeclares whole-contact checker mass normalization while
keeping the1e-3 score threshold and application budgets fixed. The new controls
show that normalization alone still falsely rejects all40 angle/q8-64 candidates.
Checker centroid and face-shape discrepancies remain after removing mass bias.
This method is unsuitable as the selected cheap default on present evidence.
Independent checker fits still overreject when face contributions are compared
before assembly. The selected methodv4 first assembles signed P2 load differences
at each original global node within one source, then takes the L1 norm. Different
sources never cancel each other's errors. With the unchanged1e-3 score threshold,
the frozen44-pair study has no false accepts and11 conservative false rejects;
all three inaccurate coarse controls are rejected. This finite corpus supports
an empirical admission rule, not a universal stress-error certificate. Original
rawv1/v2/v3 scores, thresholds and completed failures stay retained.

An initial CPU postprocess fails its exact raw-score reproduction guard.
The cause is a changed reduction order after NPZ loading changes a noncontiguous
native array view to C-contiguous layout. Restoring the declared original
Face/Contact/Batch/value layout reproduces all36 scores byte for byte. The
repair retains the zero-tolerance guard and original failed receipts.

## Native implementation and complete histories

Explicit volume `pressure_quadrature="adaptive"` now uses a shared8/16/64/128
rule bank and independently fitted candidate/checker pairs16/8,64/16,128/64.
Original1e-9 fit and1e-8 source checks remain in force. Decisions precede the
single final scatter, retain original source indices, invalidate repaired moment
caches and use complete CPU/serial/overflow fallback paths. Preparation and
runtime stay in Quadrants. No q128 evaluation is required merely to screen an
ordinary source.

The v4 loaded suite passes16 CPU/CUDA cases, including material cavities,
cooperative/scalar execution, constrained task capacity, full output, release,
slot changes and partial reset. Actual606 and the authored convex edge pass
cached/uncached/serial/forced-overflow lifecycle controls on CUDA. Nine of12
frozen corpus cases complete successfully; the manual-tip group reaches its
one-hour allocation limit before producing a result, leaving two following
cases unexecuted. That timeout is retained and does not qualify the missing
cases. A subsequent exact traversal optimization is under fresh validation.

The original CPU-versus-GPU byte guard detects differences in16 derived model
arrays. An independent diagnostic keeps all physical/topological inputs exact,
finds relative K difference9.57e-16, and independently solves both original
operators under their original residual budgets. A separately labelled CPU
control then requires every freshly built CPU array to match the diagnostic CPU
model byte for byte. It passes without injecting K or changing any tolerance;
the original GPU-model byte comparison remains false.

Engine `d0ad2ad162928dc49a3d422b893892420712c128` completes all eight1200-step
histories: scripted and actual FP32 policy inference, B4 and heterogeneous B64,
max and full. All326400 environment-frames have valid production, full-mirror
and same-input q128 recovery, with complete600-step cycles, load/release,
recontact, reset and checkpoint checks. Worst original source force and
application-point moment errors are1.44e-15 and1.08e-12. Independently audited
saved K residuals use at most0.00564 of their original budget.

Worst full-tensor weighted error is2.48801e-5 relative; sampled-peak error and
underestimation are at most1.51227e-5. Maximum recorded absolute global RMS and
peak differences are1.73362Pa and5.32470Pa. Original stored-six/displacement/peak
1e-4 and older5/10/5 stress gates pass; the old1e-6 raw-load comparison fails
(worst5.003e-5) and remains separate. All954640 positive-force source records
use the initial angle16 stage with affine fits and zero Newton evaluations.
These histories therefore do not exercise repair or continuous rule switching.
The terminal receipt is `meta/native-trajectory-complete-v1-terminal-audit.json`.

Separate combined-path profiles identify adaptive fit integration as the
remaining dominant cost. The new implementation reuses each source's complete,
validated geometry chain during fitting, avoiding unrelated empty faces while
retaining the same samples, fits and admission thresholds. Its parity and cost
are still under validation; diagnostic profiles from independently advanced
scenes are not strict paired speed comparisons.

## Remaining delivery work

Numerical default promotion, complete frozen-corpus qualification, a nonvacuous
switching control, strict common-input paired timing, process memory/preparation
measurements and final recommendations remain open. Module timing now requires
a shared raw collider/link/geometry snapshot before association and Hertz sizing;
matching seeds or contact counts alone does not establish input equality.

A short public Franka example now accepts a public Options object and reports
max/full shapes and units for volume or fixed-thickness planar-k32 shell. Its
verification worker invokes that same ordinary example and, for viewer checks,
starts an owned Xvfb display. The actual public shell headless B64/max run and
viewer B1/full run both complete730 steps. The viewer uses CUDA physics and a
real512x384 GLX window with software Mesa rendering under the private Xvfb;
the initial inherited EGL-loader failure is retained. Volume examples through
the final numerical default remain pending.
The new throughput protocol freezes three paired fresh-process blocks atB4/B8192,
max/full module and complete policy scopes, with separate profiles and memory.
