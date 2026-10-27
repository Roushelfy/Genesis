# Planar loading DPR: execution ledger

The active goal is [native planar loading](PLANAR_LOADING_DPR_NATIVE_GOAL_20261027.md)
at `f8fad8ba6`; it remains incomplete. CPU source `79720355` is the independent
reference. Runtime artifacts live under the data-side `20261027-planar-loading-dpr`
run directory recorded in `DATA_LOCATION.md`. Its `meta/error-budgets.json` and
`meta/guard-budgets.json` were frozen before tuning.

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
witnesses are being independently audited; the original failed result is retained. The run
contains 420 fallback contacts among 9,223 loaded contacts and 66 overflow
environment-steps. Neither failure is silently removed or assigned a looser gate.

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

The subsequent bounded-grid experiment at `a26cc3e93` passes all numerical
controls for512/2048/4096 work warps. Early profiles show no small-batch benefit
and slower large-batch integration for512/2048. A smaller launch grid therefore
does not establish an optimization. Finish the bounded sweep before deciding
whether to retain or archive it. An exact radial Hertz-moment candidate is
under independent primitive/integration study; it is not yet a runtime winner.

| Required family | Current status / remaining evidence |
| --- | --- |
| Fixed geometry and compact J maps | Native QD preparation; compact refined grid plus integer topology and original-parent CSR; shared geometry supported. Preparation/memory measurements pending. |
| Material-aware queries and witness reuse | Warp query, guarded original-parent footprint mask, warped miss and ambiguity controls validated. Temporal witness reuse remains to be assessed against actual profile and invalidation cost. |
| Contained moments, clipping and fitting reuse | Exact analytic moments and positivity checks validated; clipped G/H shared across fit/scatter. Geometry publication increment pending. |
| Packing, grouping, layout and cooperative work | Loaded sources packed, bounded per-environment tasks and whole-environment overflow fallback validated; k4/capacity/quadrature/batch tuning and plateau pending. |
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
it does not measure policy training.
