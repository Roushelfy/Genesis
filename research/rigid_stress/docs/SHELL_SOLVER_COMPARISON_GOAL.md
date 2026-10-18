# Goal: compare rigid stress recovery with Kashu shell PR22

User direction, 2026-10-17. Compare our native rigid stress feature with
[Kashu7100/Genesis PR22](https://github.com/Kashu7100/Genesis/pull/22)
in detail for the shared task of computing useful **stress distributions**
during grasping. Determine contact reliability, stress/model accuracy,
consistency, throughput/memory and the operating range of each approach.
Do not assume either method wins or that reported shell failures still occur.

## Pin the implementations and understand the comparison

At goal creation the PR is open, head
`fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab` on `Kashu7100/Genesis:feat/shell-fracture`,
base `02908fe6474d8b073b662347a8794614fa250658`. Recheck the head when execution starts, record the
exact tested revision and keep later updates separate.
Our starting point is the completed
[pressure/intrinsic comparison](PRESSURE_COMPARISON_RESULTS_20261018.md).
Use both default/tuned pressure and intrinsic body_source_prior where relevant;
source-prior is another load model, not a truth oracle.

PR22 implements a deformable triangular shell with SVK membrane, dihedral
bending, damping, one linearized backward-Euler solve per substep, and sequential
rigid coupling through LegacyCoupler. The inspected code corrects free vertex
velocities against rigid SDFs at a half-thickness offset. It also contains
plasticity, fracture and per-environment topology storage.
ShellOptions already has an environment convergence mask and threshold plus
an iteration cap; the Python substep still dispatches the capped PCG loop.
Inspect the actual residual/preconditioning/stopping semantics rather than
describing it as only fixed-iteration integration.

Our method leaves the object rigid and reconstructs auxiliary elastic stress.
Their deformation/contact feedback, plasticity and fracture are additional
capabilities. Explain these distinctions and compare the common pre-failure
elastic task first. Disable optional plasticity/fracture for that baseline
using supported options, record remaining allocated storage/work, and retain
their stock/default configuration as a separately labeled practical baseline.
Do not charge optional fracture capability to a purported minimal elastic
implementation, or remove its code silently to improve its baseline.

## Make stress outputs mechanically comparable

The primary output is the full distribution, with global raw peak/location,
not only a damage index, scalar maximum, rendered color or fracture flag.
Audit stress measure, coordinates, units, thickness and observation timing.
Our affine-P2 tetrahedral corner tensors represent a volume field in Pa.
The inspected shell `func_membrane_stress` returns a2x2 membrane quantity
in N/m. Stiffness scratch uses a positive-eigenvalue projection for numerical
purposes. Neither that clipped tensor nor a fracture criterion can be silently
used as the full physical Cauchy/von-Mises stress.

Document the shell's membrane and bending stress recovery, reference/current
configuration conversions and thickness assumptions. Compare compatible
material-frame components and membrane/bending resultants, and reconstruct
top/bottom-fiber stresses where supported. Match physical positions and wall
thickness when sampling our volume field. State which transverse/through-
thickness components the shell approximation cannot supply. Do not compare
N/m directly with Pa or a membrane-only peak with a volume/bending peak.

Add minimal observation/diagnostic code if the PR lacks the needed outputs,
label it separately from the original implementation, and make both methods
produce the declared full-field product in the primary cost comparison.
Keep computation/device storage, host export and rendering costs explicit.
Max-only measurements may be additional useful configurations.

## Separate mechanics, contact and application behavior

Start with inexpensive controls and reuse existing independent references:

- Prescribed finite loads / simple analytic or independent FEM controls:
  test membrane stretching, bending and a rounded shell/egg or suitable curved
  control. Use the small-deformation regime to establish common elastic
  behavior before interpreting nonlinear or contact differences.
- Contact controls: small rigid sphere/probe over a triangle interior versus
  vertex/edge, an offset sweep, normal pressing then frictional sliding,
  opposing/asymmetric pads and release/recontact. Include a compact concave
  or close-sheet case when it reveals a distinct issue.
- Principal application: Franka grasping the same full hollow egg, with support,
  normal force plus friction, changing/asymmetric contacts and complete
  approach/grip/lift/weak/release/reset history. Reuse the accepted condition
  bank and add only a few discriminating cases.

Match continuum shape/mid-surface, thickness, E/nu/density, mass/inertia,
gripper/pad geometry/compliance, friction, supports, actuation and physical
duration as far as each model permits; expose the remaining mismatches.
Do not impose both a chosen contact force and an incompatible jaw displacement.
Use a short resolution/time-step/tolerance study to identify credible settings.
Equal vertex/DOF or iteration counts alone are not equal fidelity.

Separate identical-load stress recovery from contact-generated load differences.
Where useful, replay the resolved shell contact loads through our recovery and
an independent elastic reference, retaining coordinate/inertia conventions.
A discrepancy in a same-load solve, a footprint reconstruction difference and
a different reaction under matched actuation answer different questions.
Their shell is not automatically a converged contact reference, and our rigid
model cannot reproduce deformation-driven feedback, buckling or fracture.

## Test contact stability and numerical consistency directly

Treat prior reports as reproduction targets, not current facts: missed
triangle-interior contact despite a geometric depth reading; strong peak
dependence on substeps despite PCG convergence; and repeat-run variability.
Use saved identical inputs and repeated restarts to establish which occur on
the pinned PR. Record probe depth, actual active contact features/reactions,
force/moment balance, penetration, friction/slip, stress fields, energy and
peak location/time, with the physical sign/frame definitions made explicit.

Sweep meaningful contact resolution, dt/substeps, PCG tolerance/cap and coarse
update settings independently where necessary. Report the actual residual,
iterations, cap exhaustion and converged-environment fractions.
Distinguish linear convergence from the one-linearization time integration
and sequential contact splitting. Increasing PCG cannot repair an undetected
contact; changing damping or material is not a numerical fix for the same case.

Check repeatability, environment/batch independence, partial resets and
restoration, including whether the same deterministic state/loads give the
same field. Determine whether variability is roundoff, an amplified physical
trajectory, stale state, ordering, a race or an unresolved issue before naming
its cause. Continuous/smooth contact motion does not remove birth/death events.

Preserve the PR's baseline and all failed/unresolved cases. If a small clear
defect blocks a useful comparison, provide a separately labeled repaired
variant with targeted checks and an ablation. Do not replace its contact solver
with IPC or merge a large redesign into the baseline. An independent contact
reference can be used selectively where it resolves an ambiguity.

## Measure the practical tradeoff on the same hardware

Use low meshes for iteration, then a compact set of credible accuracy settings.
Measure complete full-field scene steps and marginal stress/deformation cost,
valid aggregate env-step/s, batch latency, memory/capacity, cold setup and
warmed runtime over useful batch sizes. Include changing contacts, output,
checks, control and resets in each declared scope. Use the same GPU family and
matched or paired allocations; PR throughput quoted on an RTX3080 cannot be
compared directly with our RTX PRO6000 measurements.

Show each native/default practical precision and separately comparable precision
where supported. Report same-workload and error-versus-cost comparisons;
do not force unequal models to use the same PCG budget or relax tolerances.
Expose contact pool/storage, effective substeps, convergence failures and
unsupported scenes. A faster inaccurate/unstable rollout is not an accepted
accuracy-qualified rate, but retain its measurements as explicitly labeled
diagnostics. Policy inference can be a secondary scope; full RL training is
not required.

Profile where cost comes from: rigid dynamics, shell force/Hessian products,
PCG/preconditioner/coarse update, contact, our mapping/recovery, full stress
evaluation/output, dispatch/copies and reset/topology storage. Keep profiler
and export overhead out of warmed headline rates unless explicitly included.

## Deliver a decision, not another broad campaign

Deliver a concise architecture/assumption table, matched stress/traction maps
and error/consistency curves, contact-failure reproductions, scaling/memory and
accuracy-versus-cost plots, exact runnable commands and compact source/input/
hardware evidence. State where each method is preferable for stress-aware
grasping/RL, which defects are reproduced/fixed/unresolved, and when neither
supports a reliable damage claim. Distinguish the shared stress task from
deformation/fracture capabilities that only the shell models.

Keep our production implementation native Quadrants and existing rigid-solver
integration intact. Run the external PR and any instrumented/repaired variants
in isolated pinned worktrees/snapshots, without silently porting it or changing
upstream. Preserve the cleaned branch layout and reuse archives/references.
No universal high-mesh convergence campaign, full fracture validation or
experimental damage calibration is required. Use engineering judgment on
experiment size and stop expanding a sweep when it no longer changes the
practical conclusion. Commit the implemented comparison and results on this
branch; do not count a plan or historical author report as executed evidence.
