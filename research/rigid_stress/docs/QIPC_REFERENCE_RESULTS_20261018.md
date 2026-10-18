# Three-method QIPC comparison: working results,2026-10-18

**Status: active, incomplete.** Complete prescribed-motion histories now have
actual common-reference field discrepancies for DPR, SCR and PR22 Shell.
The reference fails the current spatial/timestep uncertainty targets, so those
discrepancies remain diagnostic. Capped case7 completes9s, while representative
case0 and its uncertainty controls remain incomplete. A confirmed shared-data
storage outage prevents new durable runs. This report is a checkpoint, not the
goal closeout or an accuracy-qualified ranking.

The [goal](QIPC_REFERENCE_COMPARISON_GOAL.md),
[metric contract](QIPC_REFERENCE_METRICS.md),
[execution record](QIPC_REFERENCE_PROGRESS.md) and
[running guide](RUNNING_QIPC_REFERENCE.md) define the task and remaining work.
The [compact checkpoint](../evidence/20261018-qipc-reference/checkpoint-v1/README.md)
preserves selected values from completed Slurm analyses and their source hashes.
All runtime paths below are relative to the established data run
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/`.

## Methods and what their fields mean

| Method | Exact public choice | Contact/deformation feedback | Stress observation |
| --- | --- | --- | --- |
| Directional Patch Recovery, DPR | `RigidStressOptions.dpr(...)`; `contact_model="pressure"` | Existing rigid contact trajectory; reconstructed stress does not deform collision geometry | Linear3D recovery from vector contact patches, including friction |
| Surface-Constrained Recovery, SCR | `RigidStressOptions.scr(...)`; `contact_model="intrinsic", intrinsic_load_partition="body_source_prior"` | Same rigid mechanics as DPR | Surface-connected vector-traction reconstruction with local redistribution |
| Deformable Shell | Kashu PR22 `fd051fe3f6ed58b52810846d4295d7b6fbb5e2ab` | Actual shell deformation and PR22 impulse/velocity contact coupling | Physical in-plane Cauchy tensor from membrane/bending fields at three fibers |
| QIPC candidate | Original `fe252bd30acc6c717507b74add31512812f48761` for prescribed motion; separately pinned integration candidate for capped motion | Hollow3D P1 tetrahedral FEM with contact and friction | Unprojected Cauchy stress differentiated from the actual elastic energy |

The two new preset names do not change the existing numerical operators or
defaults. Neither recovery route is a normal-only pressure model. PR22 Shell
omits transverse stress; a full3D solid metric is a separate supplement.
The original Shell checkout remains unchanged. Input adapters, read-only
diagnostics and numerical QIPC variants have separate sources/receipts.

## Common physical input and residual mismatches

The shared faceted hollow egg has radii22/22/30mm, taper0.18, normal-offset wall
500um, E10GPa, nu0.3 and density2000kg/m3. Its volume-derived mass is
6.529848210401324g. Refinements subdivide the same level1 faceted domain;
the separately rounded level2 shape is not a mesh-convergence sample.
Plasticity and fracture are disabled. The original Panda XML, support plane,
initial bank case, gravity, normal contact and friction are retained.

Prescribed motion uses the same6s robot-link history, with free egg motion and
solved contacts. It overrides actuator caps. The original capped protocol uses
9s,10ms control ticks, original targets/gains, force limits, weak/release
schedule and the authored reset. QIPC's integration candidate materializes
passive damping/rotor inertia and minimal-coordinate joints. Its public drive
mapping preserves start-frame force clipping and the implicit damping term;
the finger equality mapping uses authored dmin inertia and records its remaining
state-dependent coefficient difference. No reaction force is imposed alongside
the prescribed robot motion.

| Quantity | Actual matching or remaining difference |
| --- | --- |
| Finger geometry | Input XML expands inherited geom attributes; original and expanded MuJoCo models have bitwise-equal compared geometry, pose, inertia, passive and actuator arrays. The reader's former literal-size/default omission is retained as a failed input attempt. |
| Mass/inertia | Solid mass is fixed. Nominal Shell500um uses area-based mass6.946250g at surface0. Volume/area face thickness matches6.529848g; its lumped inertia discrepancies decrease9.87%,2.46%,0.602% at surface0/1/2. |
| Wall bending geometry | Nominal thickness gives about21% normal second-moment discrepancy. Volume/area thickness reduces it to0.344–0.368%, with sampled support gaps up to30.16um. This is not an all-direction collision-error bound. |
| Shell contact thickness | PR22 selects the first fan face at each vertex. It does not average incident faces or use their union. The corrected geometry audit uses the actual built layout. |
| Contact distance | QIPC dhat20um; default Shell softness2mm gives4.60517mm influence cutoff padding beyond half thickness. These have different force laws. |
| Friction | Bank coefficient is explicitly materialized per case for Shell's global coupling coefficient. QIPC uses lagged normal friction and smoothing; PR22 applies velocity projection/restitution and Coulomb impulse limiting. |
| Collision filters | Native neutral-geom exclusions and QIPC adjacent whole-link exclusions are not equivalent. This remains a declared coupling mismatch. |

Shell's separate narrow-cutoff input uses softness8.6858896um so
`softness*log(10)=20um`. Both complete prescribed and capped controls finish,
including negative penetration and high-stress observations. Matching that
cutoff does not make the contact laws equivalent or establish accuracy.

## Reference energy and implemented contact

For classical Lamé parameters mu,lambda, QIPC SNH stores
`mu*=4mu/3`, `lambda*=lambda+5mu/6` and `alpha=1+3mu*/(4lambda*)`.
The actual energy is

\[
\Psi=\frac{\mu_*}{2}\{I_1-3-\log(I_1+1)\}
      +\frac{\lambda_*}{2}(J-\alpha)^2,
\qquad
\sigma=\frac{\mu_*}{J}\left(1-\frac1{I_1+1}\right)FF^T
      +\lambda_*(J-\alpha)I.
\]

Only the energy observer subtracts the constant rest energy. The solver's
energy, gradients and Hessians are unchanged. An independent C++ probe of the
original generated energy/gradient agrees with this observer to7.22e-16 at
E1; rotation and small-strain modulus controls pass. Independent CPU implicit
P1 controls also retain actual dynamics and inertia. Projected Hessians and
barrier gradients are not substituted for elastic stress.

In `consistent_ipc_contact.cu` and `cipc_simplex_scatter.h`, the implemented
barrier is `kappa*(r-1)^2*log(r)^2`, with
`r=(D-xi^2)/(dhat^2+2*dhat*xi)`. Simplex-pair volume weight is
`min(Aa,Ab)*dhat/4`; half-plane weight is `Avertex*dhat`. The assembly includes
dt-squared scaling. Friction uses lagged normal force and a C1 smoothed norm
at `epsilon_velocity*dt`: inside that radius,
`f0(x)=x^2*(-abs(x)/3+eps)/eps^2+eps/3`, and outside it is `abs(x)`.
Area weighting exists, but the name Consistent IPC is not a convergence proof.

Original main can stagnate at floating-point energy resolution. The separately
named prescribed diagnostic permits16 double epsilons times absolute energy
in non-KKT line search. Its capped integration counterpart tightens a former
`1e-12*(1+abs(E0))` KKT allowance to the same16-epsilon scale. These are numerical
variants, not unmodified-baseline results. The five-line BVH barrier placement
change between the integration pins is not evidence of missing contacts in
original main; original main has per-child-round barriers.

## Complete prescribed-motion discrepancies and uncertainty

Every three-way observation uses the same1280 material wall columns and10ms
times. Two Gauss points per union segment integrate the discrete P1/P2
through-wall norms. Exact boundary samples select the one-sided interior cell.
Each method uses its own actual mass-weighted corotation. In-plane centroid
sampling is finite; these are not exact whole-volume or continuum-peak norms.

The complete-history norm integrates all600 finite common observations,
including near-zero/release intervals. The [metric contract](QIPC_REFERENCE_METRICS.md)
retains absolute errors and explicit denominator floors and also requires
per-time/peak/reaction checks. The following entries are dimensionless ratios,
relative to the coarse strict SNH candidate, not physical accuracy scores:

| Method | Tangential field | Inner | Outer | Membrane | Bending |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPR | 0.999500 | 0.999488 | 0.999503 | 0.999497 | 0.999145 |
| SCR | 0.999497 | 0.999485 | 0.999500 | 0.999494 | 0.999133 |
| Stock Shell | 4.033453 | 4.118496 | 3.920339 | 4.066672 | 1.324473 |

The solid-only full3D discrepancies on the same columns are0.999645 for DPR
and0.999643 for SCR. These supplement the common tangent metric; Shell is not
ranked under a different full3D observable. The values expose a large contact/
feedback mismatch, but reference uncertainty prevents attributing all of it to
one method or using the tiny DPR/SCR difference to choose a more accurate route.

| Complete reference contrast | Tangential field | Membrane | Bending | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Surface1 wall2 versus wall4 | 0.124305 | 0.012414 | 0.233756 | Through-wall field/bending unresolved |
| Surface2 wall4 versus surface1 wall4 | 0.410273 | 0.391720 | 0.810806 | Spatial/contact resolution unresolved |
| Surface2 wall2 versus surface1 wall4 | 0.408629 | 0.390954 | 0.905909 | Combined contrast, not isolated thickness error |
| Half timestep versus1ms | 0.229989 | 0.228355 | 0.356276 | Timestep-sensitive history |
| Nonlinear1e-8 versus1e-9 | 8.791e-8 | 8.700e-8 | 9.606e-8 | This stopping contrast is small |
| Hencky, different stopping/DBC | 0.003376 | 0.003080 | 0.007500 | Combined constitutive/numerical contrast |

The5% field target is not met by spatial or timestep contrasts. Same-deformation
Hencky/SNH per-time discrepancies are at most0.676% over defined observations;
corotated linear/SNH at most0.167%. Maximum principal stretch departure is1.14%.
These checks bound a particular constitutive difference on the sampled state;
they do not repair contact/timestep uncertainty or calibrate damage.

Surface3/wall2 completes6s, as does halved friction smoothing. Their final paired
analyses remain pending. Surface2/wall8, surface3/wall4 and half contact band are
incomplete after the storage outage. The61440-tet observer's original cuSOLVER
API failure is independently reproduced with finite analytic tensors; chunked
recovery passes independent NumPy controls without changing physics.

## Loads, outcomes and state history

Complete spatial reference-load replay and actual point-wrench compression
cover600 frames. The independent recovery core accepts all600; measured-inertia
DPR/SCR point-only interface replays reject15/397 frames. Their rigid-inertia
ablations retain the same rejection counts. These are derived interface
failures, not native collider rollout failure rates. Complete force/moment
compression defects are bounded by1.82e-10N/5.13e-13Nm in the derived audit.
Elastic recovery, point-traction reconstruction and full contact feedback are
reported separately; their scalar errors are not added as an error budget.

Actual per-public-link reaction extraction checks physical rigid points against
saved COM/material quaternions. Finger vector norms retain opposing pinch loads;
a small net egg wrench cannot certify individual finger loads. Accepted-state
elastic/inertial closure, last-assembled contact gradients, solver caps, CCD and
work/energy remain separate reference gates. Raw microstep journals retain the
net residual vector as well as the nodal residual norm.

The mass/hand-frame outcome audit reports first lift/hold maximum relative COM
drift0.07351mm for the coarse candidate,2.823mm for DPR/SCR,12.54mm for stock
Shell. Maximum mass-weighted nonrigid RMS is0.1077mm for the candidate and
0.2263mm for stock Shell; the rigid routes have zero geometric deformation by
assumption. Relative COM drift is a macroscopic grasp/slip diagnostic, not a
local stick/slip classification. Final stock Shell COM remains high while the
other recorded histories return toward the support; final outcome differences
remain in the evidence rather than being hidden by a common plotting pose.

Near-slip state controls checkpoint at control tick427 and continue61 ticks
after three public restores. DPR/SCR position changes are about2.08e-10m and
stress changes0.00434Pa; immediate full-field invalidation is expected and the
next step is finite. Stock Shell restore continuation differs about0.68mm and
574MPa, and duplicated inputs can differ13.2cm. Geometry-only variants are kept
separate. These observations preclude treating stock Shell's finite completion
as validated restart/batch reproducibility.

QIPC public dump/recover controls at1.2s show immediate bitwise state restoration;
20ms continuation tensor differences are0.019–0.156Pa, relative4.2e-7–2.66e-6.
Full-reset velocity differences are small in absolute terms but1–2.8% relative,
and both are retained. QIPC batch/partial-reset equivalence is not claimed for
the unowned reference scene. Native physical-step histories match the original
10ms controller/ten-substep histories for all six DPR/SCR case0/2/7 controls;
Shell changing to that outer cadence fails equivalence, so its actual physical
peak observer is a separate read-only callback in the unchanged integrator.

## Matched cost and what can be recommended

New primary measurements use FP64, complete scene mechanics, the common host
controller, full output,900 warmup ticks and three900-tick repeats on RTX PRO6000
Blackwell Server GPUs. Primary timing excludes exports, memory sampling and
profiling. One env-step below is one10ms control interval. Earlier FP32/native
controller rates describe another workload.

| Configuration | Batch | Physics steps per control tick | Aggregate env-step/s, median | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Rigid baseline | 4096 | 10 | 88864 | Same native baseline scope |
| DPR full | 4096 | 10 | 27417 | Numerically valid timed repeats |
| Tuned DPR full | 4096 | 10 | 30558 | Same declared full payload |
| SCR full | 4096 | 10 | 10937 | Numerically valid timed repeats |
| DPR max | 4096 | 10 | 27925 | Secondary max-only scope |
| SCR max | 4096 | 10 | 11025 | Secondary max-only scope |
| Stock surface0 Shell full | 32 | 10 | 357.61 | Diagnostic finite throughput; invalid-step counter unavailable |
| Volume-matched surface2 Shell full | 32 | 10 | 66.53 |300 iterations,1e-8 stopping, coarse every step; diagnostic |

Native1-substep full DPR/tuned/SCR measure229330/257087/94156 env-step/s atB4096.
Shell default1-substep B32 is about3217 env-step/s, while the prior strict
surface1 ten-substep setting is about71.74. Equal substeps, mesh sizes or caps
do not establish equal physical accuracy. There is currently no setting that
can be called the fastest within a qualified physical-reference error band.

Separate process-memory sampling finds B8192 DPR/SCR fit at about12.6/22.2GiB;
that is a tested feasible batch, not maximum capacity. Eight isolated cold-cache
activation groups, all native/Shell primary timing groups, seven memory groups
and six moving loaded profiler groups complete. Their final consolidated audit
is pending. Kernel duration sums may overlap and are not complete-scene latency.
Prepared Shell max-only code has not passed its same-state/nonfinite control
and has no measured rate. It must observe actual physical-step maxima before
the secondary timing table can be completed.

DPR remains the measured faster rigid stress proxy; SCR supplies a different
surface-constrained load model, with its cost visible. Neither has qualified
absolute stress accuracy for this grasp. Shell supplies deformation feedback,
but the tested stock setting has contact/restart/batch limitations and large
diagnostic discrepancies. These observations do not support a stress/damage
threshold recommendation for any method in the unresolved contact regime.

## Storage interruption and remaining deliverables

Shared `/mnt/data` reports800TB used and zero available. A task-list write and
independent write/fsync probe directly fail with errno122. Several long jobs
exit1 near11:46UTC without terminal result/exception receipts; the exact worker
exception cannot be inferred from their last finite accepted state alone.
They are incomplete during a confirmed storage outage, not certified physical
solver failures. No task GPU writer remains active. Only1,816,576,375 bytes of
private regenerable caches from successful terminal groups were removed; all
scientific artifacts and Genesis precomputations remain. Storage remains full.

The [incident journal](QIPC_STORAGE_INCIDENT_20261018.json) and
[affected-run inventory](QIPC_STORAGE_AFFECTED_RUNS_20261018.json) retain exact
paths, Slurm states, last valid rows and checkpoint hashes. Actual capped native
checkpoints through2s pass SHA256 verification for case0, half-dt, surface2 and
strict controls. This does not validate cross-process continuation. Resume has
not been implemented. Non-checkpointed trajectory frames cannot reconstruct
native friction/contact history. Case7 does complete9s with one reset,0 invalid
steps and24 recorded PCG-cap steps, but is not yet reference-qualified.

The goal still requires complete capped case0/uncertainty histories, outstanding
paired field/reaction/friction/energy/peak analyses, Shell max-only verification
and timing, final error/cost/profile/uncertainty figures, full source closure,
the requirement audit and final commit/push. Existing three diagnostic figures
have inspected previews and export QA, but do not close these gates. New durable
experiments require recovered space or another writable persistent data root;
large runtime output has not been diverted into home.
