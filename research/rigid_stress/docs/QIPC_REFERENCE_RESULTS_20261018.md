# Three-method QIPC comparison: working results,2026-10-18

**Status: active, incomplete.** The user's latest update requires full spatial
fields only at a few predeclared keyframes, with short-window profiling and
numerical tolerance comparison before new long experiments. The dense norms,
physical-step temporal peaks and dense full/max timing below are retained
historical supplementary evidence; they are no longer required observation
cadence or the primary application cost. Current execution follows the update
in [the full goal](QIPC_REFERENCE_COMPARISON_GOAL.md).

 Complete prescribed-motion histories now have
actual common-reference field discrepancies for DPR, SCR and PR22 Shell.
The reference fails the current spatial/timestep uncertainty targets, so those
discrepancies remain diagnostic. Capped cases0/2/7 now complete9s; case7 surface,
wall, band and strict controls also complete, while half-dt is still running.
Full9s case7 wall/stopping
contrasts exceed100%; the reference remains unqualified. The shared data mount is
still unwritable; an explicitly documented bounded temporary output root now
allows selected validation and coarse histories to proceed. This report is a checkpoint, not the
goal closeout or an accuracy-qualified ranking.

The [goal](QIPC_REFERENCE_COMPARISON_GOAL.md),
[requirement audit](QIPC_REFERENCE_AUDIT.md),
[metric contract](QIPC_REFERENCE_METRICS.md),
[execution record](QIPC_REFERENCE_PROGRESS.md) and
[running guide](RUNNING_QIPC_REFERENCE.md) define the task and remaining work.
The [bounded fallback execution record](QIPC_REFERENCE_FALLBACK_20261018.md)
adds completed post-outage validation and current long-job identities.
The [compact checkpoint](../evidence/20261018-qipc-reference/checkpoint-v1/README.md)
preserves selected values from completed Slurm analyses and their source hashes.
The [figure checkpoint](../evidence/20261018-qipc-reference/checkpoint-figures-v1/README.md)
contains inspected editable exports, plotted arrays and their original renderer;
its cost snapshot still omits some later-completed measurements.
The [validated checkpointv2](../evidence/20261018-qipc-reference/checkpoint-v2/README.md)
adds the actual post-outage controls, consolidated costs, corrected Shell
static replay, adaptive negative evidence and an inspected uncertainty figure.
The [corrected checkpointv3](../evidence/20261018-qipc-reference/checkpoint-v3/README.md)
adds complete capped2/7 comparisons, full9s wall/strict contrasts, native ET
failure admission and saved-geometry forensics, unloaded/actual-pose controls,
matched Shell B1 timing and the inspected cost/profile figure. Full selected
JSON histories are losslessly retained alongside compact summaries.
Original runtime paths below are relative to the established data run
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-qipc-reference/`.
New fallback campaigns are explicitly labeled with root `F` in the execution record.

## New sparse-keyframe short-window evidence

The [11-run short-window checkpoint](../evidence/20261018-qipc-reference/short-window-v1/README.md)
retains all33 full-spatial keyframe states, executed source variants and
measured analyses. Original-state histories advance continuously to1.16s;
the timed interval(1.05,1.16]s contains110 physical steps and the predeclared
1.10/1.13/1.16s observations. Sparse mode actually skips elastic/contact stress
observation on unselected steps while preserving native failure, finite-state
and geometry-only det(F) checks.

The initial segmented sparse profile measures67.206s, including66.857s native
solver calls and0.152s total for3 stress/contact observations. Its window uses
1,102,522 PCG iterations. The dense-compute control performs1160 observations
but exports the same3 fields. All3 same-setting fields pass strict observation
equivalence: maximum position difference7.71e-11m and volume-weighted corotated
tensor contrast2.84e-6 with a1Pa denominator floor. Their different iteration
counts and n1 measurements do not establish a sparse/dense throughput ratio.

With Newton1e-9m/s fixed, linear1e-6 passes the provisional keyframe bands;
three fresh whole-window checks measure59.292/62.896/63.783s, median62.896s.
Cold elastic activation is outside that window, full-field host transfer is
inside, and compression/I/O is deferred. Relaxing Newton alone to1e-7m/s
fails the declared force-balance band at1.13s: net residual3.865% of egg
weight exceeds1%, despite small field contrasts. Partition-PCG gate off/on
changes loaded tangent fields16.63–17.32%/15.27–16.00%, failing the declared5%
band. The native Newton-aware option makes4892 attempts and0 accepts; it is a
partition-PCG entry gate, not an adaptive inner residual forcing policy.
Independent target bindings show represented robot-vertex tracking errors
below9e-8m at these keys in all11 runs; this does not certify the full boundary
history or explain free-egg/contact-path differences.

**PCG1e-6 remains provisional.** Native code stops when
`abs(r_k^T M^-1 r_k) <= tol_rate * abs(r_0^T M^-1 r_0)` or the iteration cap
is reached. With a fixed SPD preconditioner,1e-6 therefore corresponds to a
relative preconditioned residual norm1e-3; default1e-4 corresponds to1e-2.
The outer Newton displacement test uses `velocity_tol * dt`, rather than a
direct force tolerance. Final force/field checks remain necessary. At the
user's request, job726725 now compares PCG1e-6/1e-4/1e-3/1e-2 with Newton fixed
and otherwise identical short-window physics/timing. Its results are pending
and are not part of the frozen11-run packet. A passing short window does not
qualify complete grasps, capped coupling, spatial/contact uncertainty or a
primary sparse DPR/SCR/Shell cost recommendation.

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

## Corrected native-failure and observation admission

A source/log audit found that native ET detection forces Newton exit independently
of PCG/Newton caps. The old runner missed this status and could label a finite,
unconverged history complete. The original receipts remain unchanged; the
retrospective audit identifies6 forced exits in the adaptive4k-PCG pilot and2
in its16k counterpart. Their first recorded events are1.103/1.107s. The16k
pilot eliminates PCG caps but reaches119.496N maximum net force residual,1865.45
egg weights, and remains invalid. Raising a linear budget does not resolve the
native forced-exit mechanism.

The corrected runner records enabled status, pair ids and counts at every
physical step, stops at the first event and retains a full failure frame,
static contact topology and real native dump. Fresh job726639 stops at1.102s:
all1101 previous per-step counts are zero; the last count is1, invalid is true,
requested completion is false and the process exits1. Newton/PCG caps are zero.
Last displacement86.8um is far above the1pm stopping scale. This verifies the
new admission gate with an actual failure, beyond the15 passing audit controls.
Historical missing ET observations remain unknown; a clean retained log alone
is not a per-step zero counter.

Independent saved-geometry audit726679 validates exact agreement of native
global FEM positions, saved FEM fields and active contact points. All five
vertices of reported edge21424/face4617 belong to the egg FEM body. The80-digit
Decimal plane crossing has segment parameter0.03169956, but triangle
barycentrics[-0.50788197,0.50787908,1.00000288], outside the triangle. The
source-transcribed predicate reports true via its approximate-coplanar branch;
its endpoint signed distances are-1.467nm/+44.821nm. This is a detector false
positive for that saved pair, not demonstrated physical self-intersection.
The native solver nevertheless stopped its solve; the trajectory remains
rejected. No repaired detector, solver continuation or exhaustive CCD
certificate is claimed. Other historical ET pairs are not geometrically
classified. The initial forensic owner assertion mistakenly included the floor
vertex; its failed version and corrected explicit single-FEM-body check are
both retained, with24 independent analytic crossing/outside controls.

The material-coordinate query correction is separately validated: unchanged
coordinates and2e-9 barycentric tolerance,21 legacy arrays exactly unchanged,
actual inside point accepted, outside point rejected and distinct one-sided
cells at the same exact boundary. Six adaptive fallbacks recover14 sparse
observations. New complete-grid controls reject sparse, shifted, missing-middle
and missing-end observations for full-history norms; real600/900 grids pass.
This fixes observation admission, not reference physics or failed native solves.

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
| Surface3 versus surface2, both wall2 | 0.411607 | 0.376924 | 0.713727 | Fixed-wall spatial contrast remains large |
| Halved friction smoothing | 0.018816 | 0.018899 | 0.014541 | Complete smoothing contrast is smaller |

The5% field target is not met by spatial or timestep contrasts. Same-deformation
Hencky/SNH per-time discrepancies are at most0.676% over defined observations;
corotated linear/SNH at most0.167%. Maximum principal stretch departure is1.14%.
These checks bound a particular constitutive difference on the sampled state;
they do not repair contact/timestep uncertainty or calibrate damage.

Surface3/wall2 and halved friction smoothing complete6s and their paired analyses
are now executed. The rebuilt surface2/wall4 metric gives three-way tangent
discrepancies0.999381/0.999376/4.654631 for DPR/SCR/stock Shell, still diagnostic.
Surface2/wall8, surface3/wall4 and half contact band are
incomplete after the storage outage. The61440-tet observer's original cuSOLVER
API failure is independently reproduced with finite analytic tensors; chunked
recovery passes independent NumPy controls without changing physics.

New conforming contact-indicated refinement preserves the exact faceted volume,
mass and continuum inertia. An h0.5mm/wall4 mesh has145344 tets; an h0.25mm
attempt exceeds its declared face budget and is not admitted. Its completed1.16s
pilot retains every physical-step global peak but exhausts PCG696 times; maximum
net force residual is606.5 egg weights. The raw VM reaches647.933MPa at1.13s.
At that time finger force/moment differ57.62%/56.78% from surface2/wall4.
Its14 sparse full frames cannot qualify a full6s history, and the separate
initial field comparison stops at the material-column coverage assertion.
A validated coordinate query fixes that observation defect: all21 legacy arrays
are exactly unchanged, real outside points are rejected, and six adaptive
fallbacks produce14-time prefix diagnostics. At1.13s tangent/membrane/bending
contrasts are0.628771/0.578531/2.407296. The large residual and PCG-cap failures
and the later-discovered ET forced exits
still reject numerical/physical qualification; no whole-history norm is inferred.
The16k-PCG repeat has no PCG caps but2 ET exits. At1.13s its tangent/membrane/
bending contrast is0.648171/0.599350/2.421243; these are post-failure diagnostics.
Input construction and failed bounds
are preserved in the [fallback record](QIPC_REFERENCE_FALLBACK_20261018.md).

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

Executed complete per-finger force/moment contrasts are0.339955/0.334382 for
surface2/wall4,0.096648/0.109692 for half-dt and0.006826/0.006821 for halved
friction smoothing. Maximum net force residual divided by egg weight is
0.002208 coarse,0.006439 surface2/wall4 and0.000555 surface3/wall2; recorded
PCG-cap counts are263/549/0 and Newton-cap counts zero. Small net residual does
not cancel the opposing-finger uncertainty. Prescribed energy-minus-contact-work
is negative in these audited histories; this signed numerical dissipation is
not a complete conservation or physical-reference certificate.

Shell resolved-load static replay now has a tested point-force/couple transfer
that preserves undeformed source-point wrench and rigid virtual work. Actual
nonlinear deformation changes the applied moment: valid nominal replay admits
only114/600 statics, with maximum net contact-moment drift22.387Nm. Its all-finite
tangent discrepancy11.1837 retains rejected statics and is not an accuracy score.
A task-generation mistake made the first supposed volume task another nominal
run; the original bytes and correction remain. Corrected volume job726545
completes with113/600 admitted statics,19.1197Nm maximum net moment drift and
diagnostic tangent discrepancy6.20259. Its actual mass/thickness are verified.
This diagnostic cannot replace the complete contact/friction history
or isolate pure constitutive error.

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

## Complete force-limited cases2 and7

Both stock and volume-surface2 Shell comparisons now cover all900 common times
over9s with the original controller, force caps, delay, weak/release schedule
and one reset. The same coarse reference has tangent RMS1511.70Pa over time,
above the1Pa denominator floor. These are complete diagnostic ratios:

| Method / input | Tangent field | Membrane | Bending |
| --- | ---: | ---: | ---: |
| DPR | 7.14440 | 7.10740 | 16.20497 |
| SCR | 7.17043 | 7.13183 | 16.36589 |
| Stock Shell | 1.83988e4 | 1.83409e4 | 6.16237e3 |
| Volume-surface2 Shell | 1.36409e10 | 1.38755e10 | 3.89621e8 |

The solid-only3D supplements are5.95767 DPR and5.98270 SCR; they do not rank
Shell under a different observable. The reference and stock Shell fail lift
in this case, while volume Shell lifts and has severely different stress fields.
Neither a visual lift nor finite enormous fields establishes better stress
accuracy. Case7 starts its approach after2.59s; the reference's recorded global
VM maximum at1.115s precedes that commanded approach. It must not be labeled
a squeeze peak. Full initial/support, grasp, release and reset intervals remain
in every norm and raw history.

The fresh original-state case2 reference now completes all9000 physical steps,
one reset and9s. Its all900-time tangent reference RMS is8971.386Pa, with
common discrepancies below. Full3D supplements remain separate.

| Case2 method / input | Tangent field | Membrane | Bending |
| --- | ---: | ---: | ---: |
| DPR | 0.988975 | 0.968443 | 2.818253 |
| SCR | 0.990957 | 0.970342 | 2.827556 |
| Stock Shell | 3439.938 | 3429.215 | 1325.717 |
| Volume-surface2 Shell | 1.981454e7 | 1.932920e7 | 1.525881e6 |

Complete case7 reference sensitivity is also now measured over all900 times:

| Full9s contrast against the original coarse case7 | Tangent | Membrane | Bending | Finger force / moment |
| --- | ---: | ---: | ---: | ---: |
| Wall2 versus wall4 | 1.374548 | 1.375348 | 1.413411 | 1.000000 / 1.000000 |
| Stricter stopping plus PCG budget | 1.323348 | 1.325274 | 1.244933 | 1.010078 / 1.009606 |

Strict uses velocity1e-8/linear1e-7/maxPCG8192 versus1e-7/1e-5/4096, so it
is explicitly a combined numerical contrast. Wall2 has0 PCG caps, strict1,
and the baseline24. Finite/cap counts do not qualify a field that changes by
more than100%. The baseline finger-vector force RMS is only0.0012213N and
moment RMS5.12013e-5Nm; the1mN/1e-7Nm floors and absolute errors remain in the
results. Wall2 has no finger reaction over the saved grid. Do not disguise
this small-force/contact-onset sensitivity with a larger denominator floor.

Three complete unloaded9s schedules (cases0/2/7) independently execute all
9000 physical steps and match every command, cap, phase and reset epoch.
Declared bands are1e-4rad arm,1e-5m finger,5e-5m hand position and1e-4rad hand
rotation. All pass with maxima about1.15e-5rad,7.65e-6m,1.052e-5m and1.95e-5rad;
QIPC has no failed-status or PCG-cap steps. A10.5um unloaded hand discrepancy
is still substantial relative to the20um contact band. No loaded trajectory
or contact accuracy follows from this control.

Independent immutable initial frame bindings verify actual public rigid COM/
material orientation versus authored MJCF FK at every900 saved case2/7
observation. Maximum COM errors are1.414e-15/1.543e-15m and rotation errors
2.774e-15/2.783e-15rad, below the predeclared1e-7m/1e-6rad bands. This checks
recorded geometry at actual saved joint states, not unsaved physical extrema.

Fresh case0 and the remaining case7 surface/dt/band/friction groups are still
running or pending. None is accepted on a prefix or submission. The complete
wall/strict contrasts already contradict reference qualification; further
successful execution cannot erase those measurements.

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
and six moving loaded profiler groups complete. The refreshed audit contains112
configuration records, eight cold activation receipts and six full kernel tables.
Cold initialization/build/first-two-step totals atB1 are83.874s rigid baseline,
116.418s DPR,111.870s tuned DPR,107.488s SCR,90.008s stock Shell and93.420s
surface1 Shell. Imports precede these timers; fresh cache activation is separate
from warmed throughput. Kernel duration sums may overlap and are not scene latency.
Shell max-only now passes a loaded two-environment/200-physical-observation
control with independent VM from full Cauchy, maximum difference2.98e-8Pa and
selected-environment nonfinite isolation. Its matched stock B32 full/max pair
measures358.142/356.189 diagnostic env-step/s; the fine pair completes at
65.4668/65.3310. Their B32 full observer payloads are860544/13762944B for
stock/fine, versus592B max-only. Smaller payload does not ensure faster execution.
The max payload covers all physical substeps; full exports the last-substep
field. Original primary rates keep their exact source/observer receipts.
The new inspected cost-scope figure preserves all three paired repeats,
measured batch-specific memory, eight isolated cold activations and the three
largest individual kernels in each of six intrusive traces. It keeps numerical
native rates separate from diagnostic Shell rates; kernel sums are not latency.
AtB4096 the largest SCR traction-fit kernel occupies36.94% of summed kernel
duration, followed by19.95% surface-frontier and10.60% cached-traction scatter.
These are trace attribution fractions, not an additive scene-time model.
The [inspected cost figure](../evidence/20261018-qipc-reference/checkpoint-v3/figure/output-memory-activation-profiles.pdf)
and its actual plotted CSV/export QA are retained in checkpointv3.

Matched B1 Shell full-field timing is now complete for the same case0,FP64,
10-substep configurations used in the B1 error histories: three stock rates
11.5414/11.5635/11.5421 env-step/s, and volume-surface2 rates
2.243133/2.242635/2.243026. The full B32 cost figure retains its own batch.
Because Shell batch equivalence fails, B1 error must not be silently paired
with B32 timing as if it describes one admitted configuration. These are
finite diagnostic timing receipts, with no explicit invalid-step counter.

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
solver failures. At the outage audit no task GPU writer remained active. Only1,816,576,375 bytes of
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

After explicit disclosure, a bounded temporary project output root in home
and private compute-node caches allow selected work to continue. Original large
data remain on the data mount. Eight validation groups finish; fresh9s coarse
case0 is running and case2 is complete; final Shell paired timing completes. Resolved Shell dead-load
replay preserves source wrenches in the fixed reference configuration;
the complete nominal600-time replay rejects486 statics and retains deformed
moment drift. Corrected volume replay also fails static admission. Exact paths and results are in the
[fallback record](QIPC_REFERENCE_FALLBACK_20261018.md).

The goal still requires complete capped/uncertainty histories, final physical
qualification by quantity, full-replay/admission and peak/energy analyses,
final error/cost/profile/uncertainty figures, full
source closure and final audit/commit. Existing inspected diagnostic figures
and bounded fallback results do not close these gates.
