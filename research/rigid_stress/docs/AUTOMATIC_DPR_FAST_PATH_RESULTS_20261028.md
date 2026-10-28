# Automatic DPR: selective integration, defaults and measured cost

The numerical implementation, complete-history validation and public examples
are qualified. The final three-block throughput campaign and delivery audit are
still running. The [goal](AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md) remains
open until those finish. Original execution details remain in the
[progress record](AUTOMATIC_DPR_FAST_PATH_PROGRESS_20261028.md).

## Recommended configurations

For volume FEM, use ordinary `RigidStressOptions.dpr(mesh=...)`: pressure loading,
patch transfer, Hertz sizing/profile and adaptive integration starting at angular
q16. An independently fitted q8 checker screens each source. Rejected sources
advance to q64 with q16 checking, then q128 with q64 checking. Failure at the last
stage still raises the original error. The original fit tolerance1e-9, source
force/application-point moment tolerance1e-8 and elastic residual checks remain.

For the thin-shell model, retain Q4 elasticity and explicitly select planar
loading k32/q10/radial/blocks/cap128, fixed thickness0.5mm, FP64 simulation/inverse,
history0 and process-local `QD_GRAPH=0` before imports. Q4 loading remains the
public shell default. Shell and volume represent different physical models;
their speed comparison cannot establish interchangeable contact stress.

The [running guide](RUNNING_AUTOMATIC_DPR.md) provides short complete-grasp commands,
viewer/headless examples, output shapes/units and explicit manual6mm/legacy and
angle128 overrides. Sharp Franka jaws explicitly use `box_edge_policy="planar"`.
This is the documented edge-curvature surrogate with sharp collision geometry.
It does not certify the continuum maximum at an ideal sharp edge.

Measured source is `070cb0c96386250dda3a85db703d5705d5c3fccc` (E19). Numerical
qualification source `f0fd49a01b65c4ed0715dae18d8ff2d3d6878c0a` (E18) has identical
`genesis` code. E19 permits full output in the public default CLI guard. The
nonlinear corpus uses the identical E16/E17 adaptive kernels. Later report and
packaging commits do not change the measured engine.

## Why changing the stopping tolerance was insufficient

The exact historical frame221 and policy tick606/env2/slot0 failures are sampled
nonnegative-support infeasibility. Independent LP controls and native traces
agree. Residuals0.240122 and222.267 are far above the1e-9 fit threshold. Both reach
80 Newton evaluations with active Hessian rank collapse, without exhausting the
40-trial line search. The1e-9/1e-7/1e-5 bounded stopping sweep cannot repair them.
Original status3/errno128, failed traces and conditional zero-omega controls remain
separate from later exact nearby-frame mechanics.

Before choosing a candidate, application budgets were fixed at5% weighted full
3x3 tensor error,5% sampled-peak error and2% peak underestimation. Global, loaded
within2R and remote regions are measured separately, with absolute-Pa errors and
declared1Pa/1%-global-reference-peak denominator floors. Old1e-6 load comparisons,
1e-4 field comparisons and5/10/5 spatial decisions are also retained.

There are58 admitted cases among60 normal-rule controls and two original edge16
failures. Eight additional coarse controls include three converged but inaccurate
loads: convex-edge edge4/angle2 and U-inner angle2, with worst regional tensor
errors19.652%,17.741% and6.869%. Their original fit, wrench and K checks pass.
Convergence alone therefore cannot substitute for integration-error admission.

The selected empirical check assembles signed candidate/checker P2-node load
differences within each original source before taking the normalized L1 norm.
Different sources cannot cancel. Its1e-3 threshold yields zero false accepts and
11 conservative false rejects in44 frozen pairs, rejecting all three inaccurate
coarse controls. Earlier same-coefficient and facewise scores overreject and remain
negative controls. This finite calibration is not a universal stress-error bound.

## Native correctness and complete histories

All production preparation/runtime remains in Quadrants. Final scatter occurs
after source selection, with complete source-local geometric chains reused for
fitting. Overflow and serial paths preserve the original full face traversal.
Repaired moment caches are invalidated, accepted neighbors retain their loads,
and original material, support, friction, F and application-point M are preserved.

The authored nonlinear tip exposed a cooperative iteration hang. One lane now
owns Newton/Armijo state and broadcasts the next state before subsequent
collectives. All four cached/uncached/serial/forced-overflow routes pass, including
release/reload,28 Newton evaluations and q128 selection. Original timeouts and an
initial broadcast-index compile failure are retained. The precise compiler-level
cause of the original hang is not established by these observations.

All12 frozen native cases and20 CPU/CUDA loaded test executions pass. Final
default/lifecycle suites pass24 CPU and24 CUDA tests. These cover explicit
overrides, SCR, intrinsic/resolved contexts, max/full, both backends, reset and
checkpoint behavior. Separate curvature tests verify the original tensor against
an independent calculation and eight identical preparations. Deterministic native
face accumulation preserves the original area weights and projection.

All eight1200-step final-source histories pass: B4/B64, scripted/actual FP32 policy
inference, max/full. They contain326400 environment-frames and960326 source
records, including954640 positive-force sources. Production, full-mirror and
same-input q128 recovery pass original source/K gates, field budgets, load/release,
recontact, partial reset and checkpoint checks.

| Region | Worst weighted tensor relative error | Worst sampled peak relative error | Maximum tensor RMS error | Maximum peak error |
| --- | ---: | ---: | ---: | ---: |
| Global | 0.00248801% | 0.00151227% | 1.73362 Pa | 5.32470 Pa |
| Loaded | 0.00206972% | 0.00151227% | 7.32328 Pa | 5.32470 Pa |
| Remote | 0.00287005% | 0.00153369% | 1.40288 Pa | 5.28815 Pa |

All positive history sources use the initial q16 affine fit with zero Newton
evaluations. These histories do not exercise switching. The separate41-point
authored radius continuation exercises eight16/64/128 switches and13 repaired
points, with roundtrips, contained-disk exit and release/reload. Worst regional
tensor/peak/underestimation errors are0.020533%/0.013893%/0.008364%; the original
reference-stability and continuity gates pass. Its scope is the constructed
integration path, not an arbitrary live Hertz trajectory.

The old strict load comparison still fails, reaching5.003e-5 in the histories.
The original1e-4 field and5/10/5 spatial budgets pass. Reference comparisons
establish the declared discrete model; finite q128/q256 checks and sampled fields
leave continuum and sharp-edge model uncertainty explicit.

## Paired throughput

The primary protocol has102 fresh processes in three serial paired blocks,
B4/B8192, max/full, module and complete policy inference. Each block uses one
RTX PRO6000 Blackwell GPU; all blocks use driver595.71.05. Physics/inverse are
FP64, dt0.01, one substep, history0, mesh64 and GraphOff. The FP32 MLP has
layers26/128/128/7 and fixed seed99173. This measures inference with the existing
controller, not training.

Module inputs are shared raw collider/link/geometry snapshots from tick730.
Consumers advance zero scene steps, preserve the raw state and reproduce all12
physical-source columns across volume/shell routes. Policy runs include730 warmup
steps and at least730 timed steps. Observers count the complete timed interval
across resets and are included in every policy timing. Profiles and memory
sampling run separately. Stress feedback can change later live policy mechanics.

All three B4 pairs show adaptive/angle128 module gains1.773 for max/full and
complete-policy gains1.489/1.492. Fixed edge16 is faster on the frozen valid
module frame but fails the complete policy warmup with the original pressure
error. Failed warmups have no qualified rate. Final B8192 results and the full
rate/range/work table remain pending.

## Memory, preparation and remaining cost

Final E18 adaptive memory workers all complete730 warmup and730 timed steps with
zero invalid work. Sampled own-process peaks at B8192 are13.885GiB max and
14.729GiB full; native stress buffers are3938.709/4778.709MiB. Corresponding
angle128 peaks are11.916/12.729GiB and native buffers1905.648/2745.648MiB.
Planar shell reaches13.197/14.887GiB and3224.256/4964.256MiB. Pure rigid is9.150GiB.
These50ms sampled peaks are lower bounds, not whole-GPU occupancy.

The adaptive rule bank is3488 shared bytes. At B8192, admission storage is
2163310592 bytes and source-chain indices are15728640 bytes. Extra admission
storage is a real cost of the current default. B4 adaptive peaks are2.271GiB;
native max/full buffers are66.452/66.862MiB. Final adaptive cold setup including
JIT is164-184s in these separate memory workers, not per-step recovery cost.

Common-input profiles put the combined anchor/pressure/admission/scatter stage
at4.51/4.48ms for B4 max/full and71.25/71.90ms for B8192. The large-batch linear
solve is about4.9ms. Source-chain reuse follows this measured loading bottleneck;
these intrusive stages do not replace paired complete-path timing. Generic
`integration_retry_frames` includes angular-route use; actual selective repair
must be read from `adaptive_repaired_contacts` and the recorded stage counts.

## Public examples, storage and reproduction

Actual short public volume/shell commands pass730 steps, with B64/max headless and
B1/full viewer runs. The viewer creates a512x384 GLX window on private Xvfb with
Mesa software rendering and CUDA physics. This is functional viewer validation.
The default full CLI, manual6mm/legacy and explicit angle128 commands also pass.
The original default-guard and inherited-EGL failures remain in the records.

New runtime data, complete arrays, logs and profiles stay under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261028-automatic-dpr-fast-path/`.
The original12 unfinished files,64 prior frozen sources,21 campaign sources and
all69 captured source/helper inventories pass preservation checks. No existing
data, environment or compatibility link is migrated.

Final packaging and the22-clause delivery ledger remain pending the primary run.
