# Parallel-plate stress comparison: diagnostic checkpoint

The remote task specification is commit`914a558df`,
[PARALLEL_PLATE_COMPARISON_GOAL.md](PARALLEL_PLATE_COMPARISON_GOAL.md).
The previous Franka qualification campaigns remain stopped. This checkpoint
does **not** complete the new goal or establish a physical accuracy ranking.

The simple scene now reaches verified QIPC and DPR mechanical equilibria.
It exposes three separate limitations: unresolved QIPC point-contact spatial
fields, SCR rejection near a cancelling body wrench, and original PR22 Shell's
contact projection outside its implicit elastic solve. These limitations must
remain visible in subsequent comparisons; a successful subset cannot replace
the rejected original baselines.

## Scene and admission

Full free hollow egg, E=10GPa, nu=0.3, density2000kg/m3, gravity0. Two120mm
square plates are10mm thick; the lower is fixed and the upper50g plate has
one axial slide with50Ns/m damping. No egg vertices or surface symmetry
constraints are imposed. The common normal force ramps over0.1s; separation
is measured. Native mechanics needs2s settling holds, whereas the short QIPC
pilot evaluates0.3s endpoints. They are compared as candidate equilibria,
not as identical transient histories. Matching full histories remains open.

Corrected QIPC setup uses its original finite-stiffness guide, explicit root
world position, collision group3 and authored plate inertials. Initial actual
plate vertex bounds are asserted. Empty QIPC/native plate motion matches the
backward-Euler mass/damping recurrence to the declared floating-point bounds;
native default armature must explicitly be0. Earlier adapter failures and
the initially misplaced/intersecting QIPC campaign are preserved and excluded.

Selected contact-entry diagnostic:20um clearance per plate, equal to QIPC
d_hat; translate the large plate mesh3mm along its plane. This preserves the
local flat surface while moving a triangulation diagonal away from the pole.
The original zero-offset/22um cases remain available. The observed reduction
in free lateral drift suggests sensitivity to contact features; it is not a
proved general cure. Native-equilibria-v2 uses the same20um/3mm geometry.

Native exact mu=0 is unavailable: material and collider enforce mu>=0.01.
QIPC mu=0 and mu=0.01 controls explicitly bound the normal-loading difference.
Shell uses original PR22 kernels, nominal500um wall and its own mass/contact
envelope; these differences are recorded rather than adjusted to match fields.
DPR is`pressure`; SCR is`intrinsic` +`body_source_prior`; radius6mm is declared
before comparison and is not fitted to the FEM reference.

Predeclared equilibrium tests: each actual reaction within1% of command
(1mN scale floor), plate speed<=1um/s, egg inertial resultant<=1% of command,
egg kinetic energy<=1e-10J, and endpoint nodal equation residual<=1e-3 of
contact norm. A stable final50ms window is checked separately. Native uses
a conservative rotational-plus-translational kinetic upper bound; its full
elastic field is evaluated only at endpoints. Intermediate unobserved elastic
residuals are not claimed to have been measured.

## Newton and PCG tolerances are different criteria

The primary short pilot uses Newton velocity tolerance1e-8 and PCG
`tol_rate=1e-4`. In the pinned QIPC linear-PCG source,
`pcg_compute_conv_threshold_kernel` sets the threshold to
`tol_rate * abs(r0.T @ M_inv @ r0)`; termination compares the new preconditioned
residual quadratic form to that threshold. Thus1e-6 is **not** a relative
Euclidean residual norm of1e-6. With an SPD fixed preconditioner it corresponds
to a preconditioned norm ratio1e-3. Newton's outer criterion uses a maximum
displacement correction scaled by`velocity_tol * dt`, not a direct force
residual criterion. Independent force/elastic-equation closure remains necessary.

| Off-diagonal0.1N run | Newton velocity tol | PCG tol_rate | Endpoint equation relative residual | Endpoint result |
| --- | ---: | ---: | ---: | --- |
| Original | 1e-8 | 1e-4 | 1.949e-6 | Pass |
| Repeat | 1e-8 | 1e-4 | 1.482e-5 | Pass |
| Newton tighter | 1e-9 | 1e-4 | 2.863e-3 | Reject equation closure |
| PCG tighter | 1e-8 | 1e-6 | 3.281e-3 | Reject equation closure |
| PCG looser | 1e-8 | 1e-2 | 3.539e-3 | Reject equation closure |
| dt/2 | 1e-8 | 1e-4 | 9.355e-6 | Endpoint pass; window audit separate |
| mu=0.01 | 1e-8 | 1e-4 | 1.876e-6 | Endpoint pass; window audit separate |

The tighter runs repeatedly reach the4096 PCG cap, including their last step,
yet return tiny Newton corrections. Their fields differ from the original by
about0.44%/0.47% in the common tangent metric and fail the stricter equation
gate. Neither their smaller input tolerances nor finite completion qualifies
them. The looser-inner run is faster but also fails equation closure and is
not an accepted speedup. Do not replace the1e-4 baseline by1e-6 without measured
benefit. Total early PCG cap counts are retained; the original's final50ms
window has no observed cap, while the repeat has three sampled cap events.
Neither observation establishes that every unrecorded inner solve converged.

## Discrete elasticity passes; contact-field refinement does not

The original off-diagonal endpoint has actual reactions0.09999999839 and
0.09999999813N, kinetic energy1.15e-16J and no detected edge/triangle
intersection. All final50ms mechanical and equation checks pass. Independent
CPU linear Lamé FEM on the **same measured nodal traction and mesh** differs
by2.42e-6 in complete volume-weighted tensor norm. Its equation residual is
1.71e-12N. The six-mode rigid gauge's balancing correction is reported
(4.83e-10N L2); it supplies no hidden unreported support.

The surface2/wall4 endpoint also reaches mechanical/equation equilibrium,
and its same-traction CPU difference is5.96e-6. However, both surface grids
carry each plate's entire normal load on **one polar FEM vertex**; measured
nodal footprint radius is zero. This is not a resolved pressure footprint.

| Spatial diagnostic at0.1N | Raw von Mises peak | Common tangent change vs surface1/wall4 | Status |
| --- | ---: | ---: | --- |
| surface1/wall4 | 25.574kPa | Baseline | Discrete endpoint admitted; continuum field unresolved |
| surface2/wall4 | 53.012kPa | 61.27% | Discrete endpoint admitted; strong spatial sensitivity |
| surface1/wall2,0.3s | 25.238kPa | 11.52% | Still moving; not a thickness-convergence result |
| surface2/wall8 | 52.839kPa | Pending derived contrast | Equation residual4.65%; not admitted refinement |

The surface1->2 membrane change is59.25% and bending change146.26% on the
shared material columns. Raw peaks are neither clipped nor averaged. These
observations diagnose unresolved polar contact fields; they do not prove an
asymptotic continuum singularity or justify excluding polar regions. No
global mesh sweep or fitted reconstruction radius is used to conceal this.

## DPR equilibria and SCR rejection

Native-equilibria-v1 passes all three final50ms mechanical windows and endpoint
elastic residual checks. The geometry-aligned v2 repeats its fields; v2
endpoints are below. Plate travel includes contact regularization and clearance;
it must not be read as pure elastic egg indentation.

| Command | Actual upper reaction | Upper plate travel | DPR raw VM peak | Elastic equation residual |
| ---: | ---: | ---: | ---: | ---: |
| 0.1N | 0.09999904N | 376.668um | 22.679kPa | 3.45e-11 |
| 0.3N | 0.299999999N | 872.369um | 68.038kPa | 3.31e-11 |
| 1N | 1.000000000N | 1703.447um | 226.793kPa | 3.17e-11 |

The original0.1N DPR/common-QIPC tangent contrast is42.25%, or1687Pa RMS.
The two actual reactions agree within1%, but QIPC's spatial change is61.27%.
Consequently42.25% is a **diagnostic discrete-field contrast**, not a qualified
physical error. The field adapter also exports inner/outer tensors, membrane,
bending, full3D solid-only contrasts and raw peak locations in compatible
material frames. Missing SCR fields remain missing.

SCR rejects its first settled endpoint at step2100 in both geometry variants.
The full v2 rejected state is saved without relaxing engine acceptance.
The local priors are approximately+0.09999904156/-0.09999904121N. Body-source
partitioning retains these priors while replacing auxiliary constraints with
their aggregate wrench. Its net force is only3.5474e-10N, although its retained
group force scale is0.199998N. In`cooperative_scatter.py`, final acceptance
uses1e-8 times the **nonzero net force**, falling back to the larger scale only
when the net force is exactly zero. The resulting moment budget is2.1285e-20Nm;
measured cancellation error is3.3982e-20Nm. Force and local friction checks
pass, but moment acceptance fails with status40 and propagates to both loads.
This is a diagnosed cancellation-sensitive admission failure. It is not
evidence of a large physical friction/wrench violation. The original baseline
remains rejected; no successful field is fabricated and no engine fix is
included in this checkpoint.

## Original Shell fails the coupled equilibrium test

The zero-load control remains near its initial3.3e-6Pa stress and negligible
kinetic energy. At0.1N command, dt1ms,0.3s, original PCG50/tol1e-4 yields
only about0.02746N actual reaction and113.22MPa raw VM stress. Tightening to
PCG300/tol1e-8 produces essentially the same response with no PCG caps and
true linear residual near1e-8. Half dt changes reaction to0.07146N and peak
to75.20MPa; this is not settled equal-force convergence.

Read-only diagnostics isolate the splitting: the unconstrained implicit
elastic step uses A=M+dt²K, then contact projects velocity by delta_v and
reports impulse M delta_v. The frozen linearized coupled equation instead
requires A delta_v; the measured missing term is A delta_v-M delta_v.

| dt | Contact impulse L2 | Frozen elastic/contact defect L2 | Defect/impulse |
| ---: | ---: | ---: | ---: |
| 1ms | 3.880e-5Ns | 1.441Ns | 37154 |
| 0.5ms | 5.052e-5Ns | 0.4813Ns | 9526 |

This is a frozen linearized balance diagnostic, not an exact nonlinear
reaction reconstruction. It explains why an accurate unconstrained PCG solve
does not certify coupled contact equilibrium. Shell fields at these times
remain diagnostic and cannot enter the common-force accuracy ranking.

## Warmed native sparse-field cost

The plate workload continues at a held0.1N after mechanical settling and
field compilation. Each1000-step repeat requests and transfers the complete
stress/VM fields every100steps, retaining actual engine admission and finite
checks. No frame files are written inside timing. All environments' end-window
reactions and plate speeds are checked, including the pure-rigid control;
no-contact runs cannot enter this table. Three warmed repeats on RTX PRO6000
Blackwell give the following median aggregate env-step/s (raw repeats remain
in the packet). Duplicated environments test batching cost, not random-scene
robustness or a larger range of contact conditions.

| Batch | Pure rigid | DPR with sparse complete-field transfers |
| ---: | ---: | ---: |
| 1 | 725 | 697 |
| 8 | 5470 | 5359 |
| 64 | 43511 | 41804 |

Separate intrusive B64 build/warm/profile jobs sample process peaks2130MiB
for rigid and2268MiB for DPR. Their200-step CPU/CUDA traces retain every kernel
name and duration; the largest GPU kernel is collision, not the sparse stress
observation. Profile/kernel sums and memory-instrumented runs are not primary
wall-clock rates. SCR has no admitted baseline field here; original Shell has
no admitted coupled equilibrium. These native rates therefore support an
engineering cost statement, not an accuracy-qualified cross-method speedup.

## Execution, evidence and remaining work

All scientific execution uses rtx-mid Slurm and GENESIS_IMAGE_VER=1_26 gs-srun.
New runs, logs and task JSON live under
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates/`.
The old user-authorized home fallback is retained without new writes.
Every submission captures backend revision/patch/file identities, analysis
source, exact argv, runtime/GPU identity and private cache configuration.
The [compact packet](../evidence/20261018-parallel-plates/checkpoint-v1/README.md)
contains traced receipts, the original SCR failure state and visually reviewed
color/grayscale figures with editable SVG/PDF exports. The figure captions
retain differing actual loads and missing-field status.

Useful entrypoints are`plate_reference`, `plate_native`, `plate_shell`,
`plate_audit`, `plate_failure`, `plate_fields` and `plate_figures` under
`research.rigid_stress.qipc_comparison`. Submit with the established
`research.rigid_stress.shell_comparison.submit` helper, explicit backend and
environment paths, a fresh data output, `--tasks` and `--node-local-cache`.
The saved`meta/*-tasks.json` lists exact runnable arguments; never hot-edit a
captured worker or reuse an existing output directory.

Pending: final audits of remaining localized controls, aligned-field tables,
reference sparse-cost screening, stability/radius analysis and discriminating
tangential-friction/off-axis controls where useful, plus a final requirement
audit. Existing Franka throughput remains separate
engineering evidence. No accuracy-qualified speedup or completed task is
claimed by this interim checkpoint.
