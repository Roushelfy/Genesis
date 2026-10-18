# Goal: compare three stress methods against a qualified QIPC reference

> Superseded scope update,2026-10-18: follow
> [the parallel-plate goal](PARALLEL_PLATE_COMPARISON_GOAL.md) for the active
> accuracy benchmark. The user requested a simpler scene instead of complex
> Franka grasping. Full-Franka reference/controller/convergence requirements
> below are previous plans, not current completion gates. Sparse full-spatial
> observations, profiling first and the three-method comparison remain.


User direction, 2026-10-18. Deliver a detailed, reproducible comparison of our
two rigid stress recovery routes and Kashu shell PR22 for **full spatial stress
distributions at a few keyframes during complete Franka grasps of a hollow egg**. Evaluate error,
speed, stability, repeatability, memory and practical configuration choices.
Use our existing cuda-graph-qipc as the principal independent contact/elasticity
reference. Extend the completed study; do not treat its unqualified shell rates
or same-mesh FEM agreement as an already completed common-ground-truth test.

## User update: sparse stress keyframes and profiling first, 2026-10-18

The user explicitly clarified: "只需要偶尔几个关键帧的应力数据就行" and
"先短接触窗口做耗时剖析和容差对照，再完整长实验". This update takes
precedence over earlier dense stress-observation, every-step peak-search and
whole-history stress-integration requirements in this goal and existing plans.

The execution order is now:

1. **Profile a short representative contact window first.** Include contact
   onset and a loaded portion. Preserve the preceding physical/contact history:
   use the original initial state or an independently validated native-state
   checkpoint, not a static pose restart that loses friction/controller state.
   Separate warmed solver-step time, controller time, diagnostic/observer time,
   host transfer/export and checkpoint overhead. Separate cold activation from
   warmed timing; use appropriate GPU timing/synchronization boundaries.
2. **Run a bounded tolerance comparison on that window.** Start with a few
   controlled changes to Newton stopping and PCG stopping/budget. Record actual
   Newton/PCG iteration distributions and cap/exit status, force residuals,
   reactions, penetration/CCD observations and keyframe stresses. Check adaptive
   inner stopping or preconditioning only as explicitly labeled contrasts when
   indicated by the profile. Select a practical setting from measured cost and
   declared numerical/observable acceptance bands, not from speed alone or by
   merely raising iteration caps. Do not change material/contact physics to
   obtain a faster result. Unqualified reference quantities stay unqualified.
3. **Then run the complete grasp with sparse stress observations.** Predeclare
   a small common keyframe set by grasp phase/contact event before inspecting
   candidate stress errors; cover the informative stages such as loaded squeeze,
   lift, lateral/near-slip motion and release/recontact as applicable. Compute
   and export the complete spatial stress field only at those keyframes. On
   unselected steps, avoid expensive stress recovery, stress extrema scans,
   duplicate contact-gradient diagnostics and full-field host exports unless
   the solver itself requires them or a separately justified control needs them.
   Keep continuous mechanics, contact/friction/controller/reset history and
   necessary low-cost numerical failure records. Fewer stress samples must not
   disable native failure detection or admit invalid solves.

"Full" now describes spatial coverage at the selected keyframes. Dense1ms/
10ms stress histories,600/900-observation stress norms and the true maximum
over the entire trajectory are no longer required deliverables. Report common
keyframe field/resultant errors and within-keyframe spatial peaks/hotspots,
with the temporal sampling limitation explicit. Do not label a sparse-keyframe
aggregate as a whole-history norm or its largest sample as the global temporal
peak. Existing dense raw evidence and accepted packets remain unchanged.

Primary cost measurements must reflect this sparse-observation workload and
declare its cadence/payload; retain solver-only and observation/export costs
separately. Existing dense full/max timing remains a labeled supplementary
workload, not the new application's default cost. Reference qualification and
the two complete-grasp protocols below remain required at the selected
observables; the change removes unnecessary temporal observation work.

Do not launch further long parameter sweeps before the short-window results
justify the configuration. Already-running jobs are not automatically canceled
by this update: the executing agent should assess their progress/usefulness at
a safe boundary, preserve results, and acknowledge the revised plan in the
execution record. Do not hot-edit a running worker's captured sources.

## Three methods and clearer names

Use these working names in new reports, plots and configuration summaries:

| Name | Exact current configuration | Meaning |
| --- | --- | --- |
| Directional Patch Recovery (DPR; 定向斑块恢复) | `contact_model="pressure"` | A finite patch with tractions aligned with each supplied contact force, including its tangential component; preserves each admitted contact wrench under its pad-frame contract. |
| Surface-Constrained Recovery (SCR; 曲面约束恢复) | `contact_model="intrinsic", intrinsic_load_partition="body_source_prior"` | Surface-connected support and locally constrained vector tractions; preserves the aggregate link wrench and reports local source redistribution. |
| Deformable Shell | Pinned Kashu7100/Genesis PR22, with declared full-field observer | Deformable membrane plus bending with its own contact and time integration. |

The old word "pressure" can suggest normal-only loading, and body_source_prior
is an internal partition choice rather than a parallel contact_model value.
Give users a short, unambiguous choice between the complete DPR/SCR routes.
These names are descriptive, not an accuracy ranking. Improve them if a clearer
pair emerges, keeping one explicit mapping to old identifiers and all effective
options. Make any public naming change small and consistent with repository
API conventions; separate naming from physics/default changes. Historical
receipts and reports keep their original identifiers and bytes.

Our original assumptions remain: only shape/material are fixed for
precomputation; arbitrary changing contact locations, asymmetric loads,
convex/nonconvex surfaces, normal force and friction are supported. Smooth
contact evolution during a grasp does not eliminate contact birth/death,
release, reset or near-slip cases. Keep the feature in the existing rigid
solver, with production numerical work in Quadrants and max/full options.

## Build a common reference using cuda-graph-qipc

Use [Roushelfy/cuda-graph-qipc](https://github.com/Roushelfy/cuda-graph-qipc),
not the separate early Quadrants-only qipc repository. At goal creation its
main is `fe252bd30acc6c717507b74add31512812f48761`; the native Genesis starting
revision is `48fb8d06e9a51b2418a316534a7ee2d6b69721de`, and PR22 remains
`fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab`.
Inspect current main and relevant qualified integration/fix branches before
selecting the QIPC reference. Pin actual source, patches, build, assets and
environment. Work in isolated checkouts; preserve unmodified external baselines.

Model the egg as a **3D hollow solid FEM**, with rigid robot links and ground.
QIPC has four-node linear tetrahedra; do not assume it already supplies our
P2 discretization. First determine whether selective surface/contact and
through-thickness refinement gives adequate accuracy. Do not make adding
high-order elements or optimizing reference throughput a prerequisite.

Match a common continuum geometry, wall thickness, material, density, mass/
inertia, initial state, gravity, support and finger geometry. Expose residual
faceting/lumping differences. Keep the original-contact-geometry cohort; any
rounded/compliant pad cohort is separately identified, with physical parameters
fixed before evaluating methods. Our reconstruction radius must not prescribe
the reference pressure footprint or be fitted on the held-out evaluation cases.

Choose and document the reference constitutive energy. Verify its small-strain
moduli against the supplied E/nu; Stable Neo-Hookean, StVK-Hencky, shell SVK and
small-strain recovery are not identical at finite strain. Establish a common
small-strain, pre-failure regime and quantify constitutive sensitivity where
it affects conclusions. Keep plasticity/fracture disabled in the shared task.

Recover physical stress from the **actual unmodified elastic energy**, e.g.
F = Ds Dm^-1, P = dPsi/dF, sigma = P F^T / det(F). Account for stored material
parameter conversions, reference volumes, time-step scaling and stress frames.
Projected Hessians, barrier gradients, damage indices and smoothed display
fields are not substitutes for elastic Cauchy stress. Check det(F), units,
signs and stress recovery against existing analytic/independent CPU FEM controls.

QIPC is a candidate numerical reference, not ground truth by name. Qualify its
spatial resolution (including contact surface and thickness), time step,
nonlinear/linear stopping and contact/friction regularization independently
where they affect measured quantities. Inspect the implemented contact model;
do not equate Consistent IPC with Convergent IPC solely from the name.
Check penetration/CCD, actual force balance, friction/slip, energy and actuator
work. Friction-sensitive stopping requires measured validation; defaults and
a converged PCG solve alone do not qualify the reference.

Use a bounded refinement study to estimate reference uncertainty for reactions,
field norms, hotspot location and raw peaks. Initial engineering targets might
be about2% reaction,5% common field and10% peak changes on the final controls;
choose justified tolerances for the practical decision rather than treating
these suggestions as a proof. Validate a complete representative grasp, not
only isolated equilibrium snapshots. If a peak is singular or unresolved, retain
it explicitly and diagnose the cause; do not smooth, clip or discard it.
A highest-resolution run without these checks is not a validated truth.

Reuse prior R/L/M, analytic and independent FEM infrastructure. It is acceptable
to keep independent CPU checks small and QIPC reference batches small. A second
complete robot simulator or a universal all-shape convergence campaign is not
required. Reference uncertainty must remain visible in every accuracy claim.

## Compare complete grasps under two distinct protocols

1. **Prescribed robot-motion protocol.** All three candidates and QIPC use the
   same wrist/finger motion history. The egg is free and deforms only where the
   model allows; all contacts, friction and reactions are solved. Include
   approach, squeeze, lift, lateral motion/incipient slip, release and recontact.
   This isolates contact/stress prediction under common moving boundaries.
   It overrides actuator force caps and is not the original capped-control task.
2. **Matched actuator/control protocol.** Retain the actual Franka controller,
   targets, gains, limits, control cadence and inertial assumptions. Reactions
   feed back into robot motion. Execute a bounded representative set with the
   original force caps and weak/release schedule. Qualify the QIPC robot/contact
   coupling as well as egg elasticity. Different actual trajectories are now
   part of the application prediction error, not an excuse to impose matching
   reaction forces.

Start with the accepted bank and a few discriminating normal/asymmetric/
near-slip cases. Retain full loading and friction history; a static restart at a
chosen pose cannot replace the preceding dynamics. Preserve each method's
actual masses, compliance, friction and contact resolution in the mismatch
table when an exact mapping is unavailable. Do not impose both actual motion
and a chosen contact force, or silently call a prescribed-motion result the
complete force-limited application. Closed-loop policy training is not needed.

Separate **elastic recovery**, **load reconstruction**, and **contact/deformation
feedback** using reference-load replay and full rollouts. Replay spatially
resolved QIPC loads into an independent elastic check and our routes, then
compress them to the point-wrench information our rigid feature actually gets.
Where meaningful, apply the same external loads to shell. Retain physical
application coordinates, moments and inertia/body loads; aggregate wrench
agreement does not make traction fields equivalent. Report these layers
separately rather than adding their scalar error norms as an error budget.

## Error and stability measurements

Use common material positions, observation times and stress measures. Align
frames explicitly. Compare compatible tangential tensors, membrane/bending
resultants and top/bottom stresses for all three methods; additionally compare
full 3D fields where available. State the transverse components a shell omits.
Use one common metric/sampling domain for a three-way ranking; different norms
may be useful supplementary results but must not share an unlabeled ranking.

Report weighted tensor-field errors at the common keyframes, raw VM and
tensile-principal spatial peaks where relevant, hotspot locations and sampled
times, sparse stress observations, contact forces/moments,
contact footprint, slip, deformation and grasp outcome. Include near-zero
denominator handling. Preserve raw spatial peak fields at those keyframes and
adaptive/extremum search limitations; a shared plotting grid alone does not
establish the true spatial maximum, and sparse times do not certify a temporal
maximum. Add physical-volume averaged measures only as explicit supplements.

Measure step-size/tolerance sensitivity, repeated fresh processes, duplicate
environments, batch-size effects, partial resets and state restoration followed
by continuation. Record actual residuals, cap exhaustion, missed contacts,
penetration, nonfinite states, artificial energy growth and invalid-step counts.
Reproduce the already reported shell issues before generalizing them to a new
setting. Preserve stock shell and label any minimal instrumented/repaired
variant separately; a shell with IPC substituted is a different fourth method.
Do not turn this comparison into an unbounded shell solver redesign.

Keep failure cases in the result tables. If no shell setting passes a declared
accuracy/stability band, say so and retain its diagnostic rates. Conversely,
preserve cases where shell deformation feedback is materially more accurate
than the rigid approximation. Do not define success as our method winning.

## Speed, memory and the final deliverable

Primary timing runs complete mechanics and produces **full spatial stress
distributions at the predeclared sparse keyframes**, with declared cadence,
field payloads, numerical checks, controller and resets. Separate solver and
observation/export costs. Keep dense full/max-only measurements as explicitly
labeled supplementary workloads rather than requiring per-step stress output.
On matched hardware (prefer existing RTX PRO6000 Blackwell), report complete
scene latency and aggregate env-step/s across useful batch sizes, marginal
overhead versus each applicable rigid baseline, memory/capacity, cold setup and
warmed runtime, repeated measurements and detailed profiles. Separate normal
timing from host exports, rendering, instrumentation and reference generation.

Compare practical defaults and measured useful tuned configurations. Show
error-versus-cost curves and the fastest setting meeting each stated accuracy/
stability band, with numerical-only validity distinct from physical-reference
qualification. Equal meshes, iteration caps or substep counts are not equal
accuracy. Retain QIPC reference cost for reproducibility without requiring its
large-batch optimization. Explain any throughput changes from prior reports.

Deliver one detailed three-method report with:
- architecture/assumptions and exact new-name/config mapping;
- qualified reference and its uncertainty, by protocol and measured quantity;
- shared-scale keyframe stress/traction maps, sparse phase observations,
  necessary mechanical histories and per-case keyframe error tables;
- stability/repeatability/failure tables and original-versus-repaired ablations;
- speed, memory, scaling, profiles and accuracy-versus-cost figures;
- clear recommendations on when DPR, SCR or shell is preferable, including
  where none supports a reliable stress/damage claim;
- runnable commands, pinned source/input/hardware identities and compact raw/
  derived evidence preserving prior accepted packets and the cleaned layout.

Do not close this goal on plans, existing same-mesh agreement or previous
unqualified rates. Produce actual common-reference error comparisons for all
three routes and identify any unqualified quantities explicitly. Keep work
bounded around decisions useful for stress-aware grasping/RL. Experimental
fracture calibration, full RL training and universal nonlinear failure validation
remain separate work.
