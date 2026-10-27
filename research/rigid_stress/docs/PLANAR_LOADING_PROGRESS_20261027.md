# Planar loading DPR: execution ledger

The active goal is [native planar loading](PLANAR_LOADING_DPR_NATIVE_GOAL_20261027.md)
at `f8fad8ba6`; it remains incomplete. CPU source `79720355` is the independent
reference. Runtime artifacts live under the data-side `20261027-planar-loading-dpr`
run directory recorded in `DATA_LOCATION.md`. Its `meta/error-budgets.json` and
`meta/guard-budgets.json` were frozen before tuning.

The current measured configuration is k32, radial integration q10, cell blocks
and capacity128, with FP64 recovery and history off. Complete four/64-condition
policy trajectories qualify this configuration with process-local `QD_GRAPH=0`.
The public loading default remains Q4. The opt-in k8/capacity32 defaults are a
separate lower-fidelity regime and are not the recommendation for the complete
heterogeneous workflow. Cooperative anchor packing is now enabled internally;
planar k16/k32 uses conservative cell blocks. Public benchmark research switches
can still select the older paths, and their actual execution modes are recorded.

## Correctness evidence

| Control | Captured source / campaign | Result and scope |
| --- | --- | --- |
| CPU oracle | `oracle-v1`, `79720355` | 119 artifacts, original 17 checks pass; q24/q48 Q4 and k8/k16 |
| Q4 operators | `operators-v0` | 36 checks per egg/torus bank; original K, mass, gauge-adapted independent solve and stress |
| Planar primitives | `primitives-v2` | 77 checks pass: original 12 clipped controls, geometry, contained moments, work and rigid motion |
| Full load/field parity | `ac7eff794`, `parity-v3` | 170 checks per egg75/torus8 bank pass, k8/k16 q24; source reversal, release/reload and partial reset |
| Material fallback | `ac7eff794`, `parity-v3` | 1,060 checks pass on a U-shell: both sides, friction, nearby disconnected footprint, original q48 load/full-field reference, four fallback scatter paths and lifecycle |
| Warped miss and ambiguity | `8d8b48955`, `guards-v5` | 682 checks pass: k4/k8/k16, serial/warp, unchanged inputs, original-load/full-field controls and release/reload; overlap ambiguity is a guard primitive, not a disconnected elastic solve |
| Original regressions | `parity-v3` / `parity-v2` | 9 shell / 11 volume checks pass; the earlier shell collection failure lacked a captured CPU module and is retained |
| Production trajectory | `ac7eff794`, `parity-v3` | B4 k8 q16 full600: original spatial budgets pass, but strict aggregate result is negative; see below |
| Application-point contract | `8d8b48955` + research `8165112a5`, `trajectory-contract-v2` | Exact fitted-state single-source scatter validates the point/origin distinction; q16 control and k16/q10 with Q4 q16/q48 references pass |
| Complete delayed cycles | `trajectory-contract-v2` | B4 k16/q10 passes711 observations. B64 completes1192 with zero numerical/lifecycle invalids and original spatial budgets passing, but misses two tighter fidelity targets |
| Exact radial mathematics | `fb084a3ca`, `radial-math-v4` | 267 checks pass, including independent high-precision primitives, clipped/active-pressure integrals and the original q24 port; earlier failed controls remain retained |
| Exact radial runtime | `0d34bfad9`, `radial-runtime-v1` | Egg75/torus8, radial and retained 2D-Gauss controls, all four k8/k16 groups pass their original 170 checks |
| Loading-only k32 | `fb084a3ca`, `oracle-k32` / `k32-validation-v1` | Independent CPU export and four native egg/torus q24/q48 groups pass. The old geometry guard fails the warped control; see below. k32 inherits the existing k16 fidelity budgets |
| Public configuration | `4032d7366`, `public-planar-v1` | Twelve Q4/planar shell tests pass: direct/inverse, max/full, every-step/manual, release, partial reset and checkpoint; no elastic DOF growth |
| D1 after material guards | `0ec2e4a7a`, `d1-banks-v11` / `k32-trajectory-v11` | 1,092 bank checks and complete B4/711 plus B64/1192 scripted trajectories pass; k32/q10 inherits k16 budgets, no fallback/overflow/invalid |
| Cooperative anchors | `175226fa1` / `67acd86cf`, `anchor-contract-v2` / `anchor-completion-v13` | Frozen old serial, refactored serial and warp outputs compare byte-for-byte, including signed ties, absent pools, failed-status recovery and slot changes; downstream banks and complete scripted trajectories pass |
| Actual inference feedback | `1836734e7`, `policy-trajectory-v15` | B4/711 and B64/1192 pass after730-step policy warmup and restart; independent original Q4 low24/16 and resolved48 references, actual FP32 weights/actions and original gates retained |
| Conservative cell blocks | `c82ec833a`, `hierarchical-query-coverage-v3` | 8,300 independent geometry/query checks pass for k4/8/16/32, serial/cells/blocks, positive triangle sets, material rejection, overflow and input bytes; full bank and performance decision still pending |
| Selected workspace | `c82ec833a`, `policy-blocks-cap128-v17` | Blocks/cooperative/radial k32/q10/cap128 passes complete actual-policy B4/711 and B64/1192; true native task peaks50 and63, no truncation/fallback/overflow/invalid |

The CPU bank explicitly selects one physical side to match the CPU oracle.
Production uses both sides. Those are distinct qualification scopes. The Q4
elastic DOF remain 1,452 (1,446 after the native translation gauge), independent
of loading refinement. Source point, original anchor, physical Hertz sizing and
surface-ball pressure law remain unchanged.

The complete B4 k8 q16 trajectory retains two failures. At frame504/env2,
sampled-peak error against original Q4 q48 is 1.085974%, exceeding the frozen
1% CPU-fidelity target; all original spatial budgets pass. At frame438/env0,
the original Q4 q16 reference has aggregate body-origin moment error normalized
by summed force-times-radius of 6.01134e-8, exceeding the research diagnostic's
1e-8 gate, while planar and Q4 q48 pass. This diagnostic can amplify a small
force residual by the source-distance/radius ratio; it is not the production
per-source application-point moment gate. Its definition and per-contact
witnesses have now been independently reviewed and replayed. At frame438, the
largest replayed source-relative force/point-moment errors are5.82e-10/5.39e-10,
both below the unchanged1e-8 gates; source raw loads reconstruct the aggregate.
The old failed result and strict-origin diagnostic remain retained. The run
contains 420 fallback contacts among 9,223 loaded contacts and 66 overflow
environment-steps. Neither failure is silently removed or assigned a looser gate.

The corrected research check also preserves a real original-Q4 q10 rejection
at frame585: errno128, an invalid observation and nonfinite field quantities
still fail. Planar k16/q10 is qualified against the valid original Q4 q16 route
and independently resolved q48, with the two integration orders explicit.

The complete64-condition run spans1192 global observations so that every
delayed environment completes its initial600-step cycle. All218,321 loaded
contacts use the proxy without fallback. Original residual, application-point
load, spatial and lifecycle gates pass. The tighter frozen k16 targets still
reject sampled peak0.323185% (limit0.3%) and M0.783200% (limit0.75%). This is
a measured approximation negative. The bounded k32 loading-only candidate now
passes complete scripted and actual-policy qualification; the elastic DOF and
all existing budgets remain unchanged. On the64-condition policy trajectory,
worst field/peak/N/M/Q errors against Q4 q48 are0.0828%/0.0736%/0.0290%/
0.1622%/0.0804%. These compare the same discrete elastic model, not a resolved
physical continuum/contact peak.

The k32 warped control exposed an admission defect in the previous `.01R`
normal-error guard: raw load error0.0281% passed, but full-field error0.3900%
and sampled-peak error0.3668% exceeded the frozen fallback-equivalence0.1%
limit. The old failed campaign remains intact. The geometry admission threshold
is now `.005R`; the error budgets are unchanged. The new guard campaign passes
892 q48 checks across k4/8/16/32 and213 each for k32 q10/q16. The changed
material path also passes all1,060 original U-shell checks at k32/q48.

Two additional material counterexamples were found during the contact audit.
Folded faces can share an authored hinge outside the contact ball while their
footprints inside it are disconnected; a sufficiently warped single Q4 can
also have two separated ball intersections. Parent adjacency and proxy normal
error alone do not establish material identity. The new candidate checks that
the shared physical edge enters the ball and requires a sufficient Q4
connectedness certificate. Uncertified geometry fails closed with the existing
invalid-load mechanism. Independent folded-face, saddle, extreme-scale and
near-tangent controls supplement the U-shell. An independent readback measures
old parent-only graph leakage of4.44–8.37% of the source force on a folded-face
control, while the new selected component has exactly zero remote exclusive-node
load. The original nearly180-degree fold has an ill-conditioned elastic mother
whose CPU and native solves fail the unchanged residual gate. A separately
identified30-degree opening keeps the disconnected contact-ball geometry and
improves conditioning. The existing serial constrained-pressure path rejects
two cases that the maintained cooperative/moment paths admit. This real
negative is retained. A mathematically equivalent, stable Armijo energy
difference fixes the two sources under all original gates: 22 matched native
checks and71 independent400-digit Decimal cases pass. The corrected opening
then passes all2,737 checks across56 phases/four paths/two capacities, including
original K, full fields and exactly zero remote exclusive-node load. A focused
post-fix serial U-shell replay remains pending. No residual, force/moment or
spatial gate is relaxed.

The volume edge/q16 policy control rejects tick606 during a730-step warmup in
both the untouched goal baseline and the later native source: env2, slot0,
pressure status3 and errno128, while the elastic solve remains valid. The
same q16 scripted workflow passes730; angle/q128 policy also passes730.
This is a retained late-cycle pressure-fit negative, not evidence of a planar
regression. Complete volume comparisons will use the qualified angle/q128
configuration and state its independent elastic/stress scope.

## Optimization ledger in progress

All component times below are intrusive device-duration sums over ten actual
frozen recovery calls, not primary wall-time speedups. Traces include the
original anchor, loading, response, stress and acceptance paths.

| Increment | B4 component | B1024 component | Decision |
| --- | --- | --- | --- |
| Serial baseline `profile-v1` | moments263,060us; query46,800us | moments289,628us; query55,101us | Initial actual bottlenecks |
| Warp moment integration `profile-v2` | moments7,173us | moments86,427us | Numerical banks pass; retained pending complete-path timing |
| Warp query `profile-v3` | query5,002us | query6,071us | Numerical banks pass; retained pending complete-path timing |
| Once-per-task clip preparation `profile-v4` | preparation335us + moments7,118us | preparation503us + moments75,028us | Both numerical banks pass; large-batch integration cost reduced, small-batch sum nearly unchanged |

The v3 large-batch moment kernel still costs 85,129us per ten calls and uses
255 registers per thread (trace occupancy estimate17%). Register count does
not establish local-memory spilling. The next isolated increment publishes
clipping geometry once before warp integration; its measured result appears
above. The warp register count falls from255 to248, with the same17% occupancy
estimate. The small-batch launch-grid hypothesis is also being tested, rather
than inferred from that occupancy estimate alone.

The `tuning-v4` campaign completed20 configurations with zero native invalid
steps. It is a screening campaign, not the final interleaved paired comparison.
At B1024 the q16 k8/capacity32 module takes11.386ms per batch step versus
9.129ms for Q4 q16; k8 q10 takes7.383ms but retains its separate fidelity
negative. The selected-fidelity k16 q10 configuration still needs matched
timing. These outcomes keep integration accuracy and wall performance separate.

The bounded-grid experiment at `a26cc3e93` passes its numerical controls
for512/2048/4096 work warps, but all three complete-module sweeps are negative.
At B1024, k8/q16 takes26.611/16.085/13.796ms versus11.386ms with the original
loops; B4 shows no gain. The production loops and research interfaces were
restored to `8d8b48955`; `a26cc3e93` remains an immutable restoration pin.
A smaller launch grid therefore did not establish an optimization.

The exact radial candidate now has independent mathematical and native-bank
qualification. It integrates the sphere-ball Hertz radial coordinate exactly,
retains angular quadrature and splits active-pressure boundaries without changing
the law or source geometry. In `radial-profile-v1`, B4 k8/q16 total recovery
device time falls from3.469ms (2D Gauss) to2.842ms (radial); B1024 falls from
11.502ms to5.011ms. Radial k32/q10 totals3.677/6.348ms at B4/B1024. These are
intrusive device-duration sums, not primary wall speedups or complete policy
rates. The original anchor alone takes about1.17/1.13ms, now18–41% of the
radial path. This changed bottleneck motivates a bounded cooperative anchor
experiment; strict old-serial/new-serial/warp byte comparisons run before
the whole-path screening decision. Its strict byte tests and complete policy
qualification pass, and the internal CUDA selection is now enabled. In bounded
screening it reduces B4 Q4/q16 module
time3.938→2.868ms and B1024 time9.064→8.264ms; these within-process screening
repeats are not the final fresh-process paired headline. Q4 q24 is faster than
q16 on the small frozen module, so the final comparison must use the tuned Q4
configuration rather than attribute shared improvements only to the proxy.

The post-anchor k32 profile identifies full-cell traversal as the next bottleneck:
query1.469ms versus moments0.064ms at B4, and1.671/1.390ms at B1024. The bounded
four-by-four cell-block candidate stores exact QD min/max bounds of the original
cells, then retains the original cell/triangle tests. Independent coverage and
all1,092 CPU-bank checks pass. Within-process screening reduces k32 module time
2.991→2.164ms at B4 and5.752→4.758ms at B1024; complete policy times also fall.
The optimization is retained for k16/k32; k8 has no meaningful measured gain.
Its earlier unsupported-kernel-default import failure and incorrectly authored
test seed are retained as harness/interface failures, not numerical passes.

Process-local graph-off screening improves both the tuned Q4 denominator and
planar workflow on this GPU/image. The final paired campaign uses graph-off
for every variant. The retained volume reference is angle/q128 with its fused
pipeline; a matched unfused/cooperative-anchor control has no useful complete
workflow advantage. These are measured scheduling selections, not changes to
the external Quadrants library or shared cluster configuration.

| Required family | Current status / remaining evidence |
| --- | --- |
| Fixed geometry and compact J maps | Native QD preparation; compact refined grid plus integer topology and original-parent CSR; shared geometry supported. Preparation/memory measurements pending. |
| Material-aware queries and witness reuse | Warp query and original-parent fallback mask implemented; new physical-edge and single-Q4 connectivity guards require fresh qualification. Temporal witness reuse remains to be assessed against actual profile and invalidation cost. |
| Contained moments, clipping and fitting reuse | Exact analytic moments, positivity checks, once-per-task geometry publication, shared G/H and exact radial integration validated in their stated banks. Whole-path gains pending. |
| Packing, grouping, layout and cooperative work | Loaded sources packed; original-anchor warp and bounded per-environment tasks qualified. Cap128 passes the complete measured policy banks with true peaks50/63. Final batch/plateau and paired timing pending. |
| Response, max/full and fusion | Original optimized Q4 response and acceptance retained. Actual profile captures fused recovery; paired max/full wall timing and bounded scheduling/fusion controls pending. |
| Graphs, device scheduling and copies | Native task scheduling uses device fields; no oracle export inside primary timing. Graph and host scheduling increments require matched complete-path measurement. |
| Fallback buffers and cost | Independent material-mask controls pass; record all routes in production timing. Capacity32 is not assumed optimal. |
| Prior Atlas/H/FP32/screen/history evidence | Preserved with its previous scope; no unchanged expensive sweep reopened. |

## Remaining completion work

Qualify warped/missed and tie guards, actual production defaults and selected
k/capacity/quadrature settings on full small/heterogeneous histories, including
all original gates and independently resolved integration. Complete tuning,
paired frozen-module and full Franka inference workflows against optimized Q4,
volume DPR and pure rigid, small/useful-large/plateau max/full measurements,
preparation/native/shared/per-environment/process/device memory, final focused
regressions, preservation audit, compact runnable evidence and recommendations.
The inference workflow uses a seeded random MLP with the scripted controller;
it does not measure policy training. Source `6dedc7957` binds actual MLP weights,
process IDs, physical assets and canonical frozen-source hashes. The final
campaign has168 fresh processes in16 groups, three paired blocks at each
scope/batch/output, with B4/1024/8192/32768. It is running alongside separate
intrusive memory and final trace campaigns. A complete fresh-process result,
final memory/requirements packet and recommendation remain required for closure.
