# Automatic DPR: selective integration, defaults and measured cost

The [requested goal](AUTOMATIC_DPR_FAST_PATH_DEFAULTS_GOAL_20261028.md) is delivered: qualified selective
volume integration, public automatic-radius defaults, complete histories, matched cost measurements and
runnable examples. All 102 original primary processes are retained: 90 qualified measurements and 12 expected
edge16 policy warmup failures. The [22-clause evidence
packet](../evidence/20261028-automatic-dpr-fast-path-v1/README.md) binds each decision to its source and
records. The [progress record](AUTOMATIC_DPR_FAST_PATH_PROGRESS_20261028.md) preserves the diagnostic
sequence.

## Recommended configurations

For volume FEM, use ordinary `RigidStressOptions.dpr(mesh=...)`: pressure loading, patch transfer, Hertz
sizing/profile and adaptive integration starting at angular q16. An independently fitted q8 checker screens
each source. Rejected sources advance to q64 with q16 checking, then q128 with q64 checking. Failure at the
last stage still raises the original error. The original fit tolerance1e-9, source force/application-point
moment tolerance1e-8 and elastic residual checks remain.

For the thin-shell model, retain Q4 elasticity and explicitly select planar loading
k32/q10/radial/blocks/cap128, fixed thickness0.5 mm, FP64 simulation/inverse, history0 and process-local
`QD_GRAPH=0` before imports. Q4 loading remains the public shell default. Shell and volume represent different
physical models; their speed comparison cannot establish interchangeable contact stress.

The [running guide](RUNNING_AUTOMATIC_DPR.md) provides short complete-grasp commands, viewer/headless
examples, output shapes/units and explicit manual6 mm/legacy and angle128 overrides. Sharp Franka jaws
explicitly use `box_edge_policy="planar"`. This is the documented edge-curvature surrogate with sharp
collision geometry. It does not certify the continuum maximum at an ideal sharp edge.

Measured source is `070cb0c96386250dda3a85db703d5705d5c3fccc` (E19). Numerical qualification source
`f0fd49a01b65c4ed0715dae18d8ff2d3d6878c0a` (E18) has identical `genesis` code. E19 permits full output in the
public default CLI guard. The nonlinear corpus uses the identical E16/E17 adaptive kernels. Later report and
packaging commits do not change the measured engine.

## Why changing the stopping tolerance was insufficient

The exact historical frame221 and policy tick606/env2/slot0 failures are sampled nonnegative-support
infeasibility. Independent LP controls and native traces agree. Residuals0.240122 and 222.267 are far above
the 1e-9 fit threshold. Both reach 80 Newton evaluations with active Hessian rank collapse, without exhausting
the 40-trial line search. The 1e-9/1e-7/1e-5 bounded stopping sweep cannot repair them. Original
status3/errno128, failed traces and conditional zero-omega controls remain separate from later exact
nearby-frame mechanics.

Before choosing a candidate, application budgets were fixed at 5% weighted full 3x3 tensor error,5%
sampled-peak error and 2% peak underestimation. Global, loaded within 2R and remote regions are measured
separately, with absolute-Pa errors and declared1 Pa/1%-global-reference-peak denominator floors. Old1e-6 load
comparisons, 1e-4 field comparisons and 5/10/5 spatial decisions are also retained.

There are 58 admitted cases among 60 normal-rule controls and two original edge16 failures. Eight additional
coarse controls include three converged but inaccurate loads: convex-edge edge4/angle2 and U-inner angle2,
with worst regional tensor errors 19.652%,17.741% and 6.869%. Their original fit, wrench and K checks pass.
Convergence alone therefore cannot substitute for integration-error admission.

The selected empirical check assembles signed candidate/checker P2-node load differences within each original
source before taking the normalized L1 norm. Different sources cannot cancel. Its1e-3 threshold yields zero
false accepts and 11 conservative false rejects in 44 frozen pairs, rejecting all three inaccurate coarse
controls. Earlier same-coefficient and facewise scores overreject and remain negative controls. This finite
calibration is not a universal stress-error bound.

## Native correctness and complete histories

All production preparation/runtime remains in Quadrants. Final scatter occurs after source selection, with
complete source-local geometric chains reused for fitting. Overflow and serial paths preserve the original
full face traversal. Repaired moment caches are invalidated, accepted neighbors retain their loads, and
original material, support, friction, F and application-point M are preserved.

The authored nonlinear tip exposed a cooperative iteration hang. One lane now owns Newton/Armijo state and
broadcasts the next state before subsequent collectives. All four cached/uncached/serial/forced-overflow
routes pass, including release/reload,28 Newton evaluations and q128 selection. Original timeouts and an
initial broadcast-index compile failure are retained. The precise compiler-level cause of the original hang is
not established by these observations.

All 12 frozen native cases and 20 CPU/CUDA loaded test executions pass. Final default/lifecycle suites pass24
CPU and 24 CUDA tests. These cover explicit overrides, SCR, intrinsic/resolved contexts, max/full, both
backends, reset and checkpoint behavior. Separate curvature tests verify the original tensor against an
independent calculation and eight identical preparations. Deterministic native face accumulation preserves the
original area weights and projection.

All eight 1200-step final-source histories pass: B4/B64, scripted/actual FP32 policy inference, max/full. They
contain326400 environment-frames and 960326 source records, including 954640 positive-force sources.
Production, full-mirror and same-input q128 recovery pass original source/K gates, field budgets,
load/release, recontact, partial reset and checkpoint checks.

| Region | Worst weighted tensor relative error | Worst sampled peak relative error | Maximum tensor RMS error | Maximum peak error |
| --- | ---: | ---: | ---: | ---: |
| Global | 0.00248801% | 0.00151227% | 1.73362 Pa | 5.32470 Pa |
| Loaded | 0.00206972% | 0.00151227% | 7.32328 Pa | 5.32470 Pa |
| Remote | 0.00287005% | 0.00153369% | 1.40288 Pa | 5.28815 Pa |

All positive history sources use the initial q16 affine fit with zero Newton evaluations. These histories do
not exercise switching. The separate 41-point authored radius continuation exercises eight 16/64/128 switches
and 13 repaired points, with roundtrips, contained-disk exit and release/reload. Worst regional
tensor/peak/underestimation errors are 0.020533%/0.013893%/0.008364%; the original reference-stability and
continuity gates pass. Its scope is the constructed integration path, not an arbitrary live Hertz trajectory.

The old strict load comparison still fails, reaching 5.003e-5 in the histories. The original 1e-4 field and
5/10/5 spatial budgets pass. Reference comparisons establish the declared discrete model; finite q128/q256
checks and sampled fields leave continuum and sharp-edge model uncertainty explicit.

## Paired throughput

The primary protocol has102 fresh processes in three serial paired blocks, B4/B8192, max/full, module and
complete policy inference. Each block uses one RTX PRO6000 Blackwell GPU; all blocks use driver 595.71.05.
Physics/inverse are FP64, dt0.01, one substep, history0, mesh64 and GraphOff. The FP32 MLP has layers
26/128/128/7 and fixed seed 99173. This measures inference with the existing controller, not training.

Module inputs are shared raw collider/link/geometry snapshots from tick730. Consumers advance zero scene
steps, preserve the raw state and reproduce all 12 physical-source columns across volume/shell routes. Policy
runs include730 warmup steps and at least730 timed steps. Observers count the complete timed interval across
resets and are included in every policy timing. Profiles and memory sampling run separately. Stress feedback
can change later live policy mechanics.

Rates are median environment transitions/s over three original repeats. Module transitions mean one recovery
per environment.

| Scope | B | Output | Adaptive | Angle128 | Edge16 | Planar shell | Pure rigid |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| recovery | 4 | max | 936.4 | 528.2 | 3,601.9 | 2,073.4 | n/a |
| recovery | 4 | full | 932.4 | 525.8 | 3,520.9 | 2,043.2 | n/a |
| recovery | 8192 | max | 99,678.0 | 15,924.6 | 193,467.0 | 302,963.9 | n/a |
| recovery | 8192 | full | 99,225.6 | 15,879.7 | 191,262.6 | 288,007.6 | n/a |
| policy | 4 | max | 585.5 | 391.5 | rejected | 895.0 | 1,295.5 |
| policy | 4 | full | 582.6 | 390.6 | rejected | 890.8 | 1,295.5 |
| policy | 8192 | max | 99,461.2 | 17,696.4 | rejected | 244,200.6 | 764,469.6 |
| policy | 8192 | full | 98,565.8 | 17,630.3 | rejected | 233,629.7 | 764,469.6 |

Adaptive / angle128 paired speed ratios:

| Scope | B | Output | Median | Range |
| --- | ---: | --- | ---: | --- |
| recovery | 4 | max | 1.773 | 1.773-1.774 |
| recovery | 4 | full | 1.773 | 1.771-1.773 |
| recovery | 8192 | max | 6.269 | 6.213-6.271 |
| recovery | 8192 | full | 6.237 | 6.184-6.249 |
| policy | 4 | max | 1.489 | 1.480-1.495 |
| policy | 4 | full | 1.492 | 1.469-1.494 |
| policy | 8192 | max | 5.617 | 5.600-5.620 |
| policy | 8192 | full | 5.577 | 5.566-5.591 |

At B8192 the paired max/full gains over angle128 are 6.269/6.237 for the module and 5.617/5.577 for complete
policy inference. Fixed edge16 is faster on the valid frozen module frame but all 12 complete-policy controls
fail their original warmup gate: B4 at tick607 (failing step606), B8192 at tick222 (failing step221). They
have no qualified rate. All 90 qualified measurements have zero invalid environment-steps. The complete raw
repeats, paired ranges against every comparator, setup times and per-environment work vectors remain in the
packet.

Across the 12 adaptive policy processes, the timed observer records 35,898,480 environment-steps and
88,946,538 source attempts/checks. There are 0 repaired sources, 0 rejected sources and 0 overflow replays.
Candidate/checker Newton evaluations total 0/0. These actual low-cost trajectories do not replace the
nonlinear/refinement corpus and authored switching controls. The source-local admission check still runs for
each attempt.

## Memory, preparation and remaining cost

Final E18 adaptive memory workers all complete 730 warmup and 730 timed steps with zero invalid work. Sampled
own-process peaks at B8192 are 13.885 GiB max and 14.729 GiB full; native stress buffers are 3938.709/4778.709
MiB. Corresponding angle128 peaks are 11.916/12.729 GiB and native buffers1905.648/2745.648 MiB. Planar shell
reaches13.197/14.887 GiB and 3224.256/4964.256 MiB. Pure rigid is 9.150 GiB. These 50 ms sampled peaks are
lower bounds, not whole-GPU occupancy. Adaptive measurements use E18; the fixed-rule/pure-rigid controls use
E8, and the repeated planar-full control uses E13. Their source-specific receipts remain explicit. These
separate memory measurements do not establish paired timing ratios.

The adaptive rule bank is 3488 shared bytes. At B8192, admission storage is 2163310592 bytes and source-chain
indices are 15728640 bytes. Extra admission storage is a real cost of the current default. B4 adaptive peaks
are 2.271 GiB; native max/full buffers are 66.452/66.862 MiB. Final adaptive cold setup including JIT is
164-184 s in these separate memory workers, not per-step recovery cost.

Common-input profiles put the combined anchor/pressure/admission/scatter stage at 4.51/4.48 ms for B4 max/full
and 71.25/71.90 ms for B8192. The large-batch linear solve is about 4.9 ms. Source-chain reuse follows this
measured loading bottleneck; these intrusive stages do not replace paired complete-path timing. Generic
`integration_retry_frames` includes angular-route use; actual selective repair must be read from
`adaptive_repaired_contacts` and the recorded stage counts.

## Public examples, storage and reproduction

Actual short public volume/shell commands pass730 steps, with B64/max headless and B1/full viewer runs. The
viewer creates a512x384 GLX window on private Xvfb with Mesa software rendering and CUDA physics. This is
functional viewer validation. The default full CLI, manual6 mm/legacy and explicit angle128 commands also
pass. The original default-guard and inherited-EGL failures remain in the records.

New runtime data, complete arrays, logs and profiles stay under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261028-automatic-dpr-fast-path/`.
The original 12 unfinished files,64 prior frozen sources,21 campaign sources and all 69 captured source/helper
inventories pass preservation checks. No existing data, environment or compatibility link is migrated.

The compact packet retains every original primary benchmark/result JSON, all 22 requirement decisions,
protocol/source receipts and retained failures. Complete histories and fields remain on data with their
recorded hashes. `meta/primary-throughput-v1-terminal-audit.json` and
`meta/primary-throughput-summary-v1.json` can be decoded from the packet index.

To repeat the primary audit, use `python -m research.rigid_stress.qipc_comparison.audit_dpr_fast_primary` with
`--run <run-root>/primary-throughput-v1`, `--manifest <run-root>/meta/primary-throughput-v4-tasks.json`,
`--source <fresh-checkout-at-070cb0c96>` and a fresh `--output` path. It reads existing terminal artifacts
using the standard library. Scientific reruns use the recorded task arguments and captured worker inside the
declared Slurm environment, with fresh data outputs.
