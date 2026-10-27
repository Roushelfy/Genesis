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
the original Q4 q16 reference has aggregate original-point moment error
6.01134e-8, exceeding 1e-8, while planar and Q4 q48 pass. Native acceptance
and K residual alone do not hide that extra conservation failure. The run
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

The v3 large-batch moment kernel still costs 85,129us per ten calls and uses
255 registers per thread (trace occupancy estimate17%). Register count does
not establish local-memory spilling. The next isolated increment publishes
clipping geometry once before warp integration; it must preserve sample sets
and be measured including the extra preparation kernel.

| Required family | Current status / remaining evidence |
| --- | --- |
| Fixed geometry and compact J maps | Native QD preparation; compact refined grid plus integer topology and original-parent CSR; shared geometry supported. Preparation/memory measurements pending. |
| Material-aware queries and witness reuse | Warp query validated; guarded original-parent footprint mask validated. Warp miss and ambiguity controls pending. Temporal witness reuse remains to be assessed against actual profile and invalidation cost. |
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
