# Practical application delivery checklist

This checklist implements [the user's active goal](PRACTICAL_VALIDATION_GOAL.md).
The [application report](PRACTICAL_APPLICATION_RESULTS_20261016.md) and
[runnable commands](RUNNING_PRACTICAL_APPLICATION.md) provide the delivery handoff.
It distinguishes earned evidence from unfinished application work. Historical
all-shape convergence requirements do not override this scope.

| Deliverable | Evidence and current acceptance |
|---|---|
| Usable native feature | Implemented in the existing rigid solver; all production numerics use Quadrants. Public API, full/max modes, friction, nonconvex mapping and partial resets are documented in [NATIVE_API.md](NATIVE_API.md). |
| Discrete elastic verification | Accept the already-qualified independent FEM checks within scope. The recent bounded-path regression passes40 CPU/CUDA cases, and matched convex comparisons retain228 accepted fields and12 explicitly unsupported pressure rows. Do not repeat unchanged large campaigns. |
| Egg/Franka resolved load and omitted feedback | [Seven matched snapshots](PHYSICAL_COMPLETE_FEEDBACK_20261015.md) already separate native/independent resolved-load agreement, linear/nonlinear recovery, and rigid/deformable feedback. The largest sampled feedback tensor difference is23.17%; this is not an all-time bound. The480-tet numerical contact reference has geometry/contact uncertainty. |
| Egg/Franka reconstructed loading, M/L and M/R | Packet extraction720564 completes the four selected states11/866/869/2640 with1/2/3/2 actual stencil packets. Source/group wrenches reproduce the qualified feedback exactly, and original force/moment checks pass. Two source-closed JSON-export failures are retained. Native reconstruction720570 and independent analysis720686 close all32 model/radius/state variants. Original per-packet/aggregate wrench contracts, full elastic budgets and raw peaks pass. Material-frame/application-coordinate differences and pressure local-cone violations remain explicit; these are conditional numerical comparisons, not absolute physical accuracy. |
| Nonconvex interior hotspot | Existing objective U resolved-load replay and six model fields give conditional M/L evidence; see [PHYSICAL_NONLINEAR_U_SPATIAL_20261015.md](PHYSICAL_NONLINEAR_U_SPATIAL_20261015.md). Preserve unresolved spatial tails and unsupported free-moment pressure inputs. |
| Small convex control | [Matched native axes](PHYSICAL_MATCHED_NATIVE_AXES_20261016.md) separate numerical integration/distance sensitivity from conditional load-model differences. Existing raw-peak tails remain measured limitations. |
| Actual changing contact, release and reset | Pressure-model diagnostic720434 completes8 environments and9600 timed observed environment steps, with0 invalid steps and8 partial resets. Intrusive trajectory diagnostics establish behavior only, not production throughput. |
| Robust model operating range | Componentwise diagnostic720500 rejects an unchanged contact in environment4 with status36 (friction-cone incompatibility reported by the native mapper). Preserve this unsupported input. Source-prior diagnostic720547 completes the identical condition bank with9600 observed timed environment steps,0 invalid steps and8 partial resets. Both successful models record the same15 completed successful lift cycles and contact/release counters. This is observed eight-condition applicability, not universal feasibility. |
| Selected end-to-end cost | Delivered: jobs720548/720565 and analysis720686 close both four-scope allocations at32 environments. Pressure/source-prior live medians5840.70/2623.09 valid env-step/s, with0 invalid timed steps,96 resets per repeat, explicit memory and2.60/9.36ms incremental batch-step cost. Same-bank,900warmup/3x2400steps, MPR/patch-off and source closures pass. Timings exclude intrusive trajectory observers and remain conditional on stated model/reference limitations. |
| Recommended settings and useful reward proxy | Delivered with scope limits: recommend source-prior when requiring local friction-admissible tractions; retain the faster pressure law's diagnosed physical limitation. Eight controlled reward runs close460800 observed steps with0 invalid values across1N/4N and0.2N/4N choices, two models and two radii. The wider choice has8/8 stable tested lift-window rankings and7/8 lift-window separations beyond the measured band. Full-episode log aggregation finds1/8 separations beyond its band, with actual phase-dependent preferred-choice reversals. All unresolved choices remain. No fracture threshold is inferred. |
| Final report and reproducibility | Delivered report, recommended settings, scope table and runnable commands. Both map packets have inspected previews, editable SVG/PDF and1000dpi exports. Complete compact publication practical-application-v2 verifies791 entries against791 actual runtime originals from23 naturally closed jobs, preserving all failures. Native assets reconstruct exactly from Git rather than being repeatedly republished. Fresh restored-packet handoff720758 also closes:53 files/1994054B restore exactly and all four native robust6mm replays pass unchanged original gates. |

Before interpreting new comparisons, use the following application decisions.
Check finite stress and correct reset behavior across accepted contacts. For a
reward proxy, report per-environment repeatability and preserve rank reversals
under declared radius/profile alternatives; choices inside that measured
variability are unresolved. Measure raw peaks and locations alongside tensor
and traction distributions. Report timings and memory for the same declared
configuration; select a configuration only after seeing its measured cost and
limitations. These criteria do not assert absolute stress or fracture accuracy.

Broad new convex local refinements720237/720238/720241/720242 and their pending
auditors720239/720243, plus shared-U fourth-level720325/720326/720327, were
cancelled after the user's priority change. Confirmed states and partial outputs
remain in `meta/application-first-priority-job-cancellations-v1.json`. Accepted
original requalifications and mesh setup remain; no cancelled solve counts as
a new physical result. Existing older running studies may finish as background
research and do not define the engineering delivery critical path.

The applicable engineering deliverables now have direct evidence; the
[completion audit](PRACTICAL_COMPLETION_AUDIT_20261016.md) maps every requested
track to its supported/conditional scope. Full RL training, universal physical
convergence and experimental fracture calibration remain separate work.
