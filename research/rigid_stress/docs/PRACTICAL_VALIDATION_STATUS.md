# Practical application delivery checklist

This checklist implements [the user's active goal](PRACTICAL_VALIDATION_GOAL.md).
It distinguishes earned evidence from unfinished application work. Historical
all-shape convergence requirements do not override this scope.

| Deliverable | Evidence and current acceptance |
|---|---|
| Usable native feature | Implemented in the existing rigid solver; all production numerics use Quadrants. Public API, full/max modes, friction, nonconvex mapping and partial resets are documented in [NATIVE_API.md](NATIVE_API.md). |
| Discrete elastic verification | Accept the already-qualified independent FEM checks within scope. The recent bounded-path regression passes40 CPU/CUDA cases, and matched convex comparisons retain228 accepted fields and12 explicitly unsupported pressure rows. Do not repeat unchanged large campaigns. |
| Egg/Franka resolved load and omitted feedback | [Seven matched snapshots](PHYSICAL_COMPLETE_FEEDBACK_20261015.md) already separate native/independent resolved-load agreement, linear/nonlinear recovery, and rigid/deformable feedback. The largest sampled feedback tensor difference is23.17%; this is not an all-time bound. The480-tet numerical contact reference has geometry/contact uncertainty. |
| Egg/Franka reconstructed loading, M/L and M/R | Packet extraction720564 completes the four selected states11/866/869/2640 with1/2/3/2 actual stencil packets. Source/group wrenches reproduce the qualified feedback exactly, and original force/moment checks pass. Two source-closed JSON-export failures are retained. Native reconstruction720570 compares both3mm/6mm assumptions with four models, reusing R/L fields and explicit material-frame/application-coordinate conventions; its results remain pending. |
| Nonconvex interior hotspot | Existing objective U resolved-load replay and six model fields give conditional M/L evidence; see [PHYSICAL_NONLINEAR_U_SPATIAL_20261015.md](PHYSICAL_NONLINEAR_U_SPATIAL_20261015.md). Preserve unresolved spatial tails and unsupported free-moment pressure inputs. |
| Small convex control | [Matched native axes](PHYSICAL_MATCHED_NATIVE_AXES_20261016.md) separate numerical integration/distance sensitivity from conditional load-model differences. Existing raw-peak tails remain measured limitations. |
| Actual changing contact, release and reset | Pressure-model diagnostic720434 completes8 environments and9600 timed observed environment steps, with0 invalid steps and8 partial resets. Intrusive trajectory diagnostics establish behavior only, not production throughput. |
| Robust model operating range | Componentwise diagnostic720500 rejects an unchanged contact in environment4 with status36 (friction-cone incompatibility reported by the native mapper). Preserve this unsupported input. Source-prior diagnostic720547 completes the identical condition bank with9600 observed timed environment steps,0 invalid steps and8 partial resets. Both successful models record the same15 completed successful lift cycles and contact/release counters. This is observed eight-condition applicability, not universal feasibility. |
| Selected end-to-end cost | In progress: job720548 runs rigid/live/policy-rigid/policy sequentially on the same GPU,32 environments, shared8-condition bank, explicit MPR/patch-off,900 warmup and three2400-step repeats. Timed repetitions exclude intrusive trajectory observers. Report valid rates, resets, memory, incremental scripted cost and separate inference scope. Source-prior job720565 repeats the same protocol with its own same-GPU baselines. Partial scope results are provisional until all source/configuration and scheduler closures pass. |
| Recommended settings and useful reward proxy | Pending application M/L, radius/profile sensitivity and selected cost. Keep the raw global maximum. Assess repeatability of frozen episodes and whether grasp choices remain distinguishable relative to measured numerical/model variability; do not infer a fracture threshold. |
| Final report and reproducibility | Pending concise settings/accuracy/cost report, selected stress/traction maps, runnable commands, source/configuration identity and supported/conditional/unsupported table. Existing archives supply prior evidence; further giant duplicate publication is not the critical path. |

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

Engineering delivery remains open until the applicable rows above have direct
evidence and the final report states the supported scope. Full RL training,
universal convergence and experimental fracture calibration remain separate.
