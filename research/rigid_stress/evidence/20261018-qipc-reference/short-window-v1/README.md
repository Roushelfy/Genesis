# Sparse-keyframe contact-window profile and tolerance controls

This checkpoint implements the user's revised execution order:
**short contact-window profiling and numerical controls first, then new long
experiments with sparse spatial stress observations**. The
[full goal](../../../docs/QIPC_REFERENCE_COMPARISON_GOAL.md) remains active.
This packet does not qualify full-grasp physics, capped actuation, reference
spatial/contact convergence or sparse DPR/SCR/Shell application throughput.
Previous dense packets remain unchanged as supplementary evidence.

All11 short runs start from the original initial state and advance continuously
to1.16s. The timed interval is(1.05,1.16]s:110 physical steps at1ms, including
contact onset and a loaded portion. Full spatial observations are predeclared
at1.10,1.13,1.16s. Geometry, material, friction, contact and target motion are
unchanged; only named numerical settings or observation instrumentation vary.
These are prescribed rigid-motion controls with finite-strength pose DBC,
not the original force-limited Franka protocol.

| Setting | Segmented window seconds, n1 | Provisional keyframe bands | Interpretation |
| --- | ---: | --- | --- |
| Baseline sparse: Newton1e-9, linear1e-8, cap4096, linear-PCG |67.206|Pass|3 stress/contact observations |
| Same settings, dense computation with sparse export |65.604|Pass|1160 observations; same3 exports |
| Newton stopping1e-7 only |58.193|Fail|Net force residual reaches3.865% of egg weight at1.13s, exceeding the declared1% band |
| Linear stopping1e-6 only |61.809|Pass|Candidate for the next timing check |
| Linear budget8192 only |66.450|Pass|A larger cap alone does not improve the measured cost |
| Newton1e-8 plus measured linear1e-6 |62.109|Pass|No measured advantage over the selected linear setting in this n1 profile |
| Original partition-PCG, entry gate off |53.607|Fail|Loaded tangent changes16.63–17.32%, finger force7.60–8.09% |
| Same partition-PCG with native Newton-aware entry gate |60.236|Fail|Loaded tangent changes15.27–16.00%;4892 gate attempts,0 accepts |

Every declared keyframe participates in the checks, including near-zero
reaction at onset. Full absolute vectors and floors remain in the reaction
arrays; the table does not replace those data with a phase mask. The source's
Newton-aware option is an inertia/motion-bound **entry gate**, eligible only
for partition-PCG. It is not an arbitrary relative-residual forcing rule.
Actual native activation and counters are checked rather than assuming the
requested option took effect. A nonfinite last derived certificate ratio is
recorded as unavailable; it is not a nonfinite physical state.

The initial baseline sparse profile spends66.857s in native solver calls over
the110-step window, versus0.152s across its3 stress/contact observer calls.
The latter includes the first sparse observation's lazy activation. Controller,
native/geometry checks, scalar/full-field transfer and compression/I/O are
separate measured phases. The baseline has1,102,522 PCG iterations in the
window. This identifies the solver as the dominant measured cost; a large
reduction in observation count does not establish a corresponding throughput
gain. The single dense/sparse pair has different iteration counts and is not
a statistical speed comparison.

The same-setting observation control passes its separately declared strict
checks at all3 frames: maximum position differences are at most7.71e-11m;
volume-weighted corotated tensor contrasts are at most2.84e-6 with a1Pa floor.
This establishes keyframe agreement, not bitwise identity at every unobserved
step or a temporal-extremum certificate.

An independent zero-step target binding verifies retained robot poses against
the actual shared prescribed targets. All11 runs have represented rigid-vertex
tracking errors below9e-8m at these frames, compared with the20um contact band;
finite DBC rotation/translation errors remain explicit. This is a keyframe
tracking observation, not proof of continuous identical boundaries or an
explanation of the different free-egg contact histories. The companion audit
also retains actual mass COM and fitted egg-axis tilt at each selected frame.

Three fresh-process checks of the provisional linear1e-6 setting use only window
boundary timing, without added per-phase CUDA events/synchronization. Their
window times are59.292/62.896/63.783s, median62.896s, and all provisional
keyframe bands pass. Scene setup and one read-only initial elastic-kernel
activation are measured outside the window; initial positions/velocities are
verified unchanged exactly. The window retains necessary native/finite/J
checks and3 full stress/contact observations. Complete spatial host transfer
remains inside; compression/write is deferred and measured afterwards.
Per-phase values in this mode are host-call spans, not GPU kernel latency.
This remains short-window reference-generation timing, not complete-grasp
primary throughput or an accuracy-qualified practical recommendation.

Code inspection prompted by the user's PCG question confirms that native
`tol_rate` bounds `abs(r^T M^-1 r)` relative to its initial value, rather than
the unsquared Euclidean residual ratio suggested by the configuration's short
description. For a fixed SPD preconditioner,1e-6 therefore means a1e-3
relative preconditioned residual norm; default1e-4 means1e-2. Neither number
specifies final nonlinear force accuracy. The Newton test compares the mapped
vertex direction before CCD against `velocity_tol * dt`; its displacement
criterion still requires independent force-residual checks. The1e-6 candidate
has passed this window but has not been shown necessary. A separately
predeclared, fixed-Newton1e-9 contrast at PCG1e-6/1e-4/1e-3/1e-2 is submitted
as job726725; its pending results are outside this frozen11-run packet.
No adaptive residual forcing policy has been tested by the entry-gate trial.

| Files | Scope |
| --- | --- |
| `profiles/<campaign>/group*/<name>/` | All33 full-spatial keyframe NPZs from11 completed runs, exact rest/input/results, and losslessly encoded full journals/timings |
| `analyses/` | Initial and combined common-keyframe field/resultant, physical finger-reaction and cost/numerical audits; no sparse aggregate is labeled a whole-history stress norm |
| `plans/` | Timestamped keyframes, changes, acceptance bands and timing boundaries |
| `inputs/`, `target-binding/` | Common sampling surface, verified rigid-vertex layout and the independent zero-step prescribed target binding |
| `receipts/`, `source/`, `source-variants/` | Exact tasks, submissions, true return codes, engine/patch/inventory identities and executed analysis variants |
| `corrections/kind-label.json` | The first boundary-timing runs retain a legacy `kind` label; explicit `timing_mode`, arguments and captured implementation are authoritative |
| `selection-provenance.json`, `manifest.json` | Byte-copy and decoded-gzip checksums; old accepted packets are unchanged |

Segmented timing explicitly synchronizes phases, and its CUDA stream spans
can include host dispatch/idle time. They are not summed kernel durations or
uninstrumented latency. The initial segmented sparse observer was not prewarmed;
its first selected call can include lazy activation, while the dense control
has already computed through the prefix. The later boundary-timing repeats
isolate initial elastic activation explicitly. Cold imports precede the scene
setup timer; private per-allocation caches are not shared-cache cold-start
replicates.

Use `source/` with the campaign's `source-variants/<campaign>/` overlay for exact
reproduction. Scientific execution remains Slurm `rtx-mid`,
`GENESIS_IMAGE_VER=1_26 gs-srun`; pinned QIPC/native builds, motion and geometry
assets are identified in the original inputs. Numeric reanalysis can consume
these published keyframes/rest/layout data, while full physics reruns still
need those pinned external assets. `capture.py` only copies existing bytes and
losslessly encodes journals; it calculates no scientific result.

The next required steps are capped-coupling short-window validation, actual
sparse computation for the native/Shell routes, reset/phase-aware complete-grasp
keyframes, selected-observable reference qualification, matching primary sparse
cost measurements, and the final detailed comparison/audit. No new long
parameter sweep is justified by the n1 profile alone. Existing delayed case7
friction task726567_5 remains reversibly held; completed dense results are kept.
