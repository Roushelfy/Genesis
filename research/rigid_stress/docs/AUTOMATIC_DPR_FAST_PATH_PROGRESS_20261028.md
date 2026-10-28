# Automatic DPR fast path: execution progress, 2026-10-28

The [remote goal](AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md) at
`d661349b12d76ace126840107d0498562b9f3d76` is complete; see the [final results](AUTOMATIC_DPR_FAST_PATH_RESULTS_20261028.md). The data root is
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
original outputs remain preserved. The numerical-default candidate now selects
adaptive/q16 for ordinary volume Hertz loading, with explicit fixed-order,
manual and shell contexts preserved. Final default/public-example tests pass. Matched cost qualification is complete.

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

The v8 loaded suite passes16 CPU/CUDA cases, including material cavities,
cooperative/scalar execution, constrained task capacity, full output, release,
slot changes and partial reset. Actual606 and the authored convex edge pass
cached/uncached/serial/forced-overflow lifecycle controls on CUDA. All12 final frozen corpus cases pass, including the nonlinear tip and the
separately completed U-inner/U-gap controls. The original manual-tip cooperative attempts time out. An unchanged
GPU serial replay passes all original model, source, independent-stage and
field gates, with28 Newton evaluations and finalq128 selection. A same-runtime
prewarm control completes the convex-edge input and then stalls on the original
tip. Separate synchronized native phase calls complete moment preparation,
initial fitting, original nonlinear correction and source-chain construction;
the stall occurs in the adaptive phase with the GPU busy. Module-load stacks
alone therefore do not establish a compiler/JIT cause. The implemented lane-owned Newton control passes cached/uncached/serial/overflow
tip routes and release/reload, retaining28 Newton evaluations and finalq128.
Twenty CPU/CUDA loaded controls pass. One lane owns the complete iteration state
and broadcasts it before the next collective. The exact compiler-level cause of
the original stall remains unproven. Original timeouts and the initial unsigned-lane
argument compilation failure remain retained.

The original CPU-versus-GPU byte guard detects differences in16 derived model
arrays. An independent diagnostic keeps all physical/topological inputs exact,
finds relative K difference9.57e-16, and independently solves both original
operators under their original residual budgets. A separately labelled CPU
control then requires every freshly built CPU array to match the diagnostic CPU
model byte for byte. It passes without injecting K or changing any tolerance;
the original GPU-model byte comparison remains false.

Final numerical engine `f0fd49a01b65c4ed0715dae18d8ff2d3d6878c0a` completes all eight1200-step
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
The final receipt is `meta/native-trajectory-complete-v3-terminal-audit.json`;
earlier E6 and E13 results remain source-specific controls.

Separate combined-path profiles identify adaptive fit integration as the
remaining dominant cost. The new implementation reuses each source's complete,
validated geometry chain during fitting, avoiding unrelated empty faces while
retaining the same samples, fits and admission thresholds. Its parity and cost
pass the loaded and lifecycle controls. Earlier diagnostic profiles from
independently advanced scenes are not strict paired speed comparisons.

A41-point authored convex-edge radius continuation now exercises eight actual
16/64/128 switches,13 repair frames, contained-disk exit, mirror roundtrips and
release/reload. Worst regional tensor, peak and underestimation errors are
0.020533%,0.013893% and0.008364%; the largest extra field jump relative toq128 is
0.027029%. All original source/K gates and the selectedq256 reference controls
pass. This qualifies the authored integration mechanism, not an arbitrary
live Hertz-radius trajectory. The final E18 source repeats the same frozen path successfully. See
`meta/authored-continuation-v3-terminal-audit.json`.

Raw collider/link/geometry snapshot consumers now reproduce all12 physical
source columns exactly across five B4 volume/shell routes; the B8192 adaptive
consumer also passes. Original failures exposed nondeterministic atomic vertex
curvature accumulation and a tuple/list metadata mismatch. The accepted
curvature preparation traverses the original face contributions serially in
Quadrants, retaining the exact area weights, projection and normalization.
CPU/CUDA tests independently verify the tensor and eight byte-identical repeats.
Consumers advance zero scene steps and preserve raw inputs through preparation
and measurement. Eight intrusive profiles pass these shared-input controls;
their different GPU UUIDs still preclude primary paired rate claims.

Six E13 memory workers and all four final E18 adaptive workers complete the
730-step warmup and730-step policy interval.
Adaptive B8192 max/full uses sampled own-process peaks13.885/14.729GiB and native
buffers3938.709/4778.709MiB; planar B8192 full reaches14.887GiB. The corresponding
B4 sampled peaks are2.271GiB. These are sampled lower bounds and retain cold
preparation accounting; reset-affected endpoint counters do not establish
complete-interval repair incidence. Receipts are
`meta/module-snapshot-native-qa-v3-terminal-audit.json`,
`meta/common-input-profiles-v1-terminal-audit.json` and
`meta/own-process-memory-v2-terminal-audit.json`. Final adaptive memory/profile
receipts are `meta/own-process-memory-v3-terminal-audit.json` and
`meta/common-input-profiles-v2-terminal-audit.json`.

## Final public verification and delivery

All48 final CPU/CUDA default/lifecycle tests pass. Actual public volume and shell
headless B64/max and viewer B1/full commands complete730 steps. The viewer uses
CUDA physics and a real512x384 GLX window with software Mesa rendering under a
private Xvfb. Manual6mm/legacy and angle128 full CLI comparisons also pass.
The original inherited EGL-loader failure and full-output default-guard failure
remain retained. E19 fixes only the public guard to permit the requested output
mode, leaving the complete E18 numerical engine byte-identical.

The102-process throughput matrix runs three paired fresh-process blocks at
B4/B8192, max/full module and complete policy scopes. Module consumers share the
same raw collider/link/geometry inputs, advance zero scene steps and reproduce
all12 physical-source columns exactly. The final source is
`070cb0c96386250dda3a85db703d5705d5c3fccc`. Device observers retain complete policy
work and repair counts across resets. Intrusive profiles and own-process memory
sampling remain separate. All102 original primary processes are now terminal and audited:90 qualified
measurements and12 retained edge16 policy negatives. The final report and22-clause
packet close the requested implementation/evidence goal.
