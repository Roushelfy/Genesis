# Qualified stress comparison: in progress

The [prescribed-reference qualification](PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md)
now passes25 checks with31 rehashed inputs (728896). Separate signed tensor
controls pass7 cases (728884); an exact load-map allocation repair passes its
before/after control (728885) after the retained250um Shell OOM (728882).
The corrected Shell replay728888 subsequently passes its static equations and
has its own terminal companion. Native/Shell three-field comparisons (728893/
728918) and fine nonlinear friction histories remain active; their outputs are
not included or treated as admitted final comparisons.

The [dimensional normal update](NORMAL_ABSOLUTE_ERRORS_20261019.md) completes
81 absolute/relative metrics from retained full-column data (728856), with69
rehashed inputs. CPU-only postprocessing now uses the same rtx-mid/container
workflow without a GPU request; original unstarted requests remain recorded.
Prescribed-C2 field comparisons continue as728860, while fine nonlinear friction
histories remain active. Final reference qualification and friction deliverables
remain required; the full goal is not complete.

The active goal is fork `950ed9d3ccfaa9a0192d4675e25bbdb6f45d20c8` and
[the qualified ranking specification](QUALIFIED_STRESS_RANKING_GOAL.md).
The [normal-pad reference and conditional comparison](NORMAL_PAD_QUALIFICATION_20261019.md)
now passes the declared reference-change gates. Independent sustained-friction
refinement and matched primary cost/batching remain incomplete. The preceding
bounded report and original failing variants remain unchanged.

The [common-peak/batch checkpoint](COMMON_PEAK_BATCH_RESULTS_20261019.md)
adds completed common-location peak uncertainty, actual B4 conformance,
admitted B1 every-step full/max wall times and the inspected force/closure
figure. Transient half-dt/contact controls and direct constitutive-correction
equivalence pass. Fine sustained-friction spatial qualification, the remaining
sparse/B4 cost/memory matrix and full stress/traction maps remain in progress.

The [later spatial/cost checkpoint](NORMAL_SPATIAL_COST_20261019.md) completes
the normal full-surface/actual-traction maps and B1 sparse full/max primary
audits. B4/memory and sustained-friction spatial qualification remain open.

The [friction resolution/memory checkpoint](FRICTION_RESOLUTION_MEMORY_20261019.md)
adds exact StVK cache equivalence, true250um Shell friction histories and
complete B1/B4 full/max process-memory profiles. Timestep changes are small,
but500-to250um Shell tangent/bending changes reach35%/56%; the fixed250um
normal comparison is not a mesh-independent Shell accuracy claim. Fine
reference qualification and final cost/cross-case synthesis remain open.

The [complete normal cost checkpoint](COMPLETE_PAD_COST_20261019.md) finishes
the24-row B1/B4 every-step/sparse full/max matrix,12 memory profiles and
three separately admitted synchronized phase profiles. The aggregation
rehashes591 audit inputs and requires exact own-run admission. Fine nonlinear
friction spatial qualification and final cross-case conclusions remain open.

The [friction reference/observation update](FRICTION_REFERENCE_OBSERVATION_20261019.md)
retains the completed500um StVK history and physical momentum audit, its
still-too-large spatial changes, and a small same-mesh0.1N linear/StVK contrast.
DPR every-step friction observation rejects eight zero-normal-force sources
with tiny lagged tangential forces; the rejection is captured without loosening
acceptance. Shell's own1000-step friction cost passes. Finer independent
surface/wall controls and the remaining case-specific comparison remain open.

## Declared acceptance and comparison scope

Reference qualification requires a refinement trend with <=2% change in the
common complete-field metric and <=5% change in raw peaks. Solver and temporal
contributions should each remain below 0.5% and 1%, respectively, in those two
metrics; otherwise refine or report unresolved uncertainty. These are starting
targets, not permission to rank methods whose separation is smaller than the
measured uncertainty. Retain raw extrema, common sampling, full spatial fields,
contact footprints, reactions, and actual equation residuals. No peak trimming,
smoothing, fitted recovery radius, or acceptance based on an exhausted solver.
Same-traction reconstruction, contact error, and complete plate response remain
separate comparisons. Original external variants remain identifiable.

## SCR: independently localized rejection and candidate repair

All paths below are relative to the data-side run root
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261019-qualified-ranking/`.

Slurm `727439`, `scr-cpu-v1/group0/probe/result.json`, independently reconstructed
the retained normal status40 and shear status33 states. Both admit feasible
traction fits. The shear dual Hessian is full rank, condition number about186;
the native saved fit's force error is about7.6e-16 of the uncancelled source
force scale. Its640 evaluations do not imply physical infeasibility. Normal
and shear net forces are about3.55e-10N and6.06e-11N, while both source force
scales are about0.2N. Normalizing by net force created unattainable accuracy
demands despite a physically accurate fit.

The candidate uses, before source cancellation,

    S_F = sum_i max(norm(F_i), norm(M_i_anchor) / radius_i)
    S_M = sum_i [norm(M_i_anchor)
                + (norm(anchor_i - original_point_0) + radius_i) norm(F_i)]

These have units N and Nm and scale homogeneously with physical loads. Moment
residuals are evaluated about the unchanged original application point. Dual
conditioning retains its separate group scale. Fit and final commit use the
same physical budgets, retaining the original1e-8 acceptance and friction
checks. There is no fixed SI absolute floor or source-net-force zero switch.
Ordinary summation roundoff is bounded on uncancelled operands; FP64 epsilon
times their scale is well below the retained relative tolerance. The existing
non-source-prior acceptance convention is preserved.

Slurm `727441`, `scr-regression-v1`, runs the same two complete-field tests
against verified original and candidate engine imports. Original:2 failures;
candidate:2 passes,127.96s. Fixtures retain source inputs rather than native
dual initializers. The tests independently check the CPU fit, local Coulomb
cones, wrench, complete fields and gauge rows. Full logs, JUnit and source
receipts remain on data. This is focused regression evidence, not full-history
qualification.

Further validation submitted: `scr-suite-v2` (`727461`) covers the existing
source-prior suite, lifecycle and incompatible-cone rejection; `scr-history-v2`
(`727462`) repeats the original normal and shear histories with the candidate.
The histories completed6300/2500 steps with zero invalid stress steps and
endpoint equation residuals2.85e-11/2.90e-11. The normal1N peak is180.189kPa;
the shear0.1N peak is21.806kPa. The shear reaction still decays to only16.8uN,
so this does not qualify the required sustained tangential loading case.

`scr-covariance-v2` (`727471`) passes both normal and shear complete-field tests
with force scales1e-6/1/1e6, geometry scales1/0.1/10 and rotated/translated
coordinates. Normalized full-field differences are below5.1e-9. The initial
`scr-covariance-v1` failed because the new test treated the packed six-component
tensor as a3x3 matrix; its failure is retained, and only the observer unpacking
was corrected. The main suite has found failures in the existing roundoff-edge
case; `scr-roundoff-original-v1` (`727485`) tests the same case against the
original backend before deciding how to address it. The original passes.
The four failures were the old moment assertion using net force times only the
first radius:6.2e-12Nm exceeded its4.27e-12Nm budget, while independent CPU fields,
force and Coulomb checks passed. All source-prior tests now compute the declared
uncancelled force/moment budgets uniformly. The1e-8 tolerance, field/oracle
thresholds and cone checks are unchanged. `scr-contract-v3` (`727491`) passes
the four updated roundoff variants plus genuine incompatible-cone rejection.
Together with the38 unchanged passing cases, both transformed/scale tests and
full histories, this verifies the GPU repair. `scr-serial-v1` (`727496`) checks
the serial CPU backend on balanced normal/shear, roundoff and infeasible cases;
all four pass in112.24s.

## Smooth contact resolution preparation

Slurm `727443`, `contact-mesh-v1`, generated three locally refined, conforming
solid meshes. New surface nodes are projected onto the same analytical tapered
egg and its500um normal-offset inner surface. Contact target sizes are50/25/12.5um,
global sizes2/1.5/1mm, and wall layers4/6/8. Counts are195312/466272/1167744
tetrahedra. All inspected volumes are positive and boundary edges have two
incident boundary triangles; minimum surface angles are22.57/19.07/22.51deg.
These checks do not certify volume element conditioning or field convergence.

Hertz half-space estimates give polar radii42--54um at0.1N and90--115um at1N.
They only inform local meshing; hollow-shell compliance and friction still need
the full solve. The estimate follows
[Hale's contact mechanics appendix, equations C.1--C.8](https://ocw.mit.edu/courses/2-76-multi-scale-system-design-fall-2004/a529c6243da095dfd8b9fb80cc421521_lec12cont_read.pdf).

The refined solid masses are7.4774/7.4830/7.4847g. This differs materially from
the old coarse piecewise-planar6.5298g model and nominal Shell6.9463g model.
Smooth geometry projection changes the discrete domain; these meshes cannot
be compared to old fields as a same-domain numerical refinement. Match the
new geometry, wall, material and mass in each compared variant and retain the
old geometry cohort separately. No contact or stress solution on these new
meshes is qualified yet.

## Independent contact FEM and coupled Shell controls

`contact-fem-pilot-v1` (`727465`) solves frictionless, free-body, small-strain
P1 plate contact through condensed elastic compliance and nodal complementarity.
Vertical translation and both tilts are unknowns, along with the loaded upper
plate position. The factorization's six gauge rows are removed from the physical
solution; measured balance corrections are about4e-17 relative. At0.1N, the
coarse mesh's equation residual is1.91e-12, but its upper contact has only three
active vertices and no fully active face. It is not spatially qualified.

`contact-fem-refinement-v1` (`727466`) retains the original three-load/three-grid
control. The medium0.1N initializer reports SLSQP line-search failure despite
the subsequent direct active-KKT solve having full rank, feasible compression,
nonnegative global gaps and contact/equation residuals near1e-11. The initial
receipt remains rejected. The next solver version explicitly identifies SLSQP
as an active-set initializer and the direct KKT solve as the final solver.
It also verifies positive-definite condensed compliance by Cholesky and admits
only complete primal/dual/complementarity checks after that solve. It does not
reinterpret a capped optimizer as convergence.

`contact-ordering-v1` (`727479`) reuses column nested dissection from the existing
P2 reference tooling. It changes matrix ordering only: the same coarse0.1N
peak differs by about1e-14 relatively. Factorization falls from36.28s to7.84s,
total execution55.50s to16.53s, and process peak RSS2.19GiB to1.05GiB. These
are preliminary reference-preparation controls, not admitted end-to-end cost
rankings. `contact-ordered-refinement-v2` (`727481`) applies this ordering to
the medium/fine grids while the first fine-grid factorization remains live.
The original fine-grid control reached its20-minute limit during factorization;
its terminal result is retained. The ordered fine-grid control is still live.
`contact-convergence-first-v2` (`727490`) compares the first two grids using
common angular material rays and piecewise-affine tetrahedra in reference
prisms. The first locator attempt incorrectly used a bilinear physical-prism
map and failed its inside-tetrahedron assertion; no clipping or tolerance
relaxation was used to fix it. At0.1/0.3/1N, common tangent changes are31.2/28.6/26.8%,
membrane changes8.36/8.27/8.17%, and bending changes7.66/7.29/6.78%. Raw von-Mises
changes are10.9/38.4/7.77%. Spatial qualification plainly remains open.

The ordered fine run finished0.1N and0.3N before its15-minute limit; both
complete fields and the condensed contact operator are retained and hashed.
`contact-fine-reuse-v1` (`727504`) verifies that cached operator's mesh identity,
nodes, signs, rigid coordinates and gaps before solving the remaining1N case.
At0.1N, the fine-grid equation residual is1.63e-11, with123 lower and37 upper
force-bearing nodes and nonzero fully active face areas on both sides. Raw
von-Mises peaks across the three grids are14.330/16.078/18.933MPa. This is
finite-area discrete contact, not a converged continuum peak.

`contact-convergence-three-v1` (`727500`) gives a25.2--25.3% medium-to-fine
tangent difference at0.1N, far above2%. Through-wall midpoint sampling also
changes some membrane comparisons appreciably between32 and64 samples. The
next comparison should integrate P1 material-column intervals exactly: within
each sorted-vertex reference prism, the three tetrahedra occupy depth intervals
`[0,w_c]`, `[w_c,w_b+w_c]` and `[w_b+w_c,1]`. Union the intervals from both
meshes for the complete-field error; integrate membrane/bending analytically.
Do not hide this sampling uncertainty behind a spatial convergence claim.

An existing, independently developed Shell contact repair is available at
`9a58a7359`, pinned in the detached `shell-qualified-contact` worktree. It joins
shell elasticity, contact and articulated rigid response in an implicit solve;
it also contains solver/history fixes. This is an explicitly named existing
variant, not a new claim about unmodified PR22. Its original branch and tests
are preserved. The observer recomputes current membrane and hinge elastic
forces in separate storage, checks them against measured contact and inertia,
and does not clear velocity or modify engine caches.

`shell-coupled-pilot-v1` (`727473`) retains the original nominal-thickness
geometry/material and0.1N force history. Default1e-4 linear and velocity
tolerances lose contact by0.3s despite zero solver-failure flags, so that run
is rejected. At linear1e-8 and velocity1e-9, the measured upper reaction is
0.09999774N, dynamic equation relative residual5.34e-5, static residual6.66e-5,
and raw von-Mises peak14.291kPa. This is a passing endpoint diagnostic only;
steady-window and dt/2 controls are running as `shell-coupled-steady-v1`
(`727483`). Both extended histories lose contact later despite zero failure
flags. `shell-coupled-audit-v2` (`727493`) separates rigid motion: the1s states
rotate0.284/0.442rad, while maximum corotated deformation is only1.35/1.68nm.
Their gaps are positive, so the zero contact is not a missed penetrating face.
The unsupported vertical long-axis pose tips under compression; the short
stationary endpoint does not establish stable equilibrium. A separate
horizontal-long-axis control is running as `shell-side-control-v1` (`727495`),
with the same complete egg geometry/material/mass and no pinned vertices.

The first smooth fine-mesh Shell test, `shell-coupled-smooth-v1` (`727489`),
reports solver failures and catastrophic values within20steps. It was stopped
after retaining the input/source/log evidence. This is a separate numerical
failure, not the rigid tipping observed on the coarse mesh. Future failure
captures must stop and retain the full state at the first solver failure.
Geometry matching, spatial refinement and original feature-test verification
remain necessary before any qualified Shell ranking.

The horizontal control also rolls during its first second; its corotated
deformation is about2.2nm. Extending the continuous history to5s in
`shell-side-steady-v1` (`727498`) reaches upper/lower reaction0.1N, plate speed
3.9e-14m/s, kinetic energy4.6e-17J and dynamic/static equation residuals about
7.3e-9, with zero solver-failure flags. The raw peak is12.363kPa. This is the
first promising sustained coupled-Shell configuration on the original coarse
geometry, pending the full-window audit (`727503`), dt sensitivity, geometry
matching and refined-field comparison. The initial pose and settling history
must remain explicit. `shell-smooth-failure-v2` (`727502`) is a short10-step
capture with a5ms ramp, using the revised observer that saves the complete
first failed state and stops; it is not an exact replay of the old ramp.

## Tolerances and remaining work

The [existing tolerance experiment](PARALLEL_PLATE_RESULTS_20261018.md#newton-and-pcg-tolerances-are-different-criteria)
does not support replacing PCG1e-4 with1e-6. The latter is a preconditioned
residual quadratic-form ratio, equivalent to a norm ratio1e-3 with fixed SPD
preconditioning. Newton's maximum displacement correction criterion is not a
direct physical equation residual; both solver exits and independent residuals
remain necessary. Tighter input tolerances do not cure iteration-cap failures.

Remaining required work includes finite-area independent reference qualification, admitted
Shell coupling, sustained nonzero tangential loading, common-field rankings
with uncertainty, and admitted matched every-step/sparse cost and stability.
This progress record does not close any of those requirements.


## Exact thickness integration and contact active sets

`contact-exact-columns-v1` (`727505`) integrates each P1 tensor exactly along
shared material columns. It partitions the wall at the union of tetrahedron
crossings and integrates tensor squared differences and membrane/bending
moments analytically. Angular centroid quadrature and the declared common
material measure remain separate discretizations. At0.1N the coarse/medium
and medium/fine tangent changes are31.833% and25.048%; medium/fine membrane
and bending changes are8.590% and8.392%. Thus the earlier thickness-sampling
alias did not explain the missing spatial qualification.

`reference-regions-land-v1/group0/exact` (`727520_0`) retains the full metric
and adds diagnostic error contributions. For medium/fine, polar radii below
0.5mm contain57.27% of squared error and26.30% of reference squared norm;
0.5–2mm contain21.37% and31.83%; the remaining surface contains21.36% and
41.87%. No region is excluded from admission, and no raw peak is trimmed.
The fine1N cached-operator continuation (`contact-fine-reuse-v1`, `727504`)
completed: equation residual3.63e-11,457/211 positive bottom/top nodes,
plate travel6.074um and raw von Mises40.789MPa. These are discrete equation
results, not continuum qualification.

An additional declared geometry, permitted by the goal, is recorded in
`land-mesh-v2` (`727523`): the complete egg has a2mm-radius flat region at
each pole and a quintic C2 height blend back to the original egg at4mm.
Its inner surface is the500um analytic-normal offset. Numerical derivative
normal validation gives1.56e-9 and the minimum meridian offset Jacobian is
0.667, with closed positive-volume meshes. The two generated meshes have
9880/20482 outer vertices and237072/737280 tetrahedra. Their masses are
7.480741/7.485734g. Material directions are stored explicitly because the
original inverse egg map does not apply to the modified polar height.
The first mesh launch (`727520_1`) retained a syntax failure; the corrected
source passed Ruff and the independent geometry checks before use.

`land-fem-pilot-v1` (`727527`) exposed an invalid SLSQP active-set guess at
0.1N: a one-shot KKT solve produced negative compression fraction-36.52,
which the existing gate rejected. The independent FEM now uses feasible
primal active-set steps, removes blocking tensile variables, admits new
negative-gap variables and checks complete KKT conditions with the existing
physical gates. SLSQP provides a feasible starting point when available;
otherwise a nonnegative force/moment-feasible LP starts the solve. The first
LP-only implementation cycled on a degenerate smooth-case start and remained
rejected (`contact-active-set-v1`, `727529`); its output is preserved.

The corrected `contact-active-set-v2` (`727532`) admits all three land loads
and reproduces the admitted smooth0.1N raw peak14.330180MPa to about2e-14
relative. Land raw peaks are108.206kPa,321.385kPa and1.015694MPa. Crucially,
the load-bearing nodes form an unresolved annular band around the flat cap:
fully active triangle area remains zero. This pilot therefore does not qualify
finite-area contact, and a blind medium/fine sweep is deferred. A resolved
annular refinement or an explicitly matched physical compliant-pad model is
still required. The original smooth hard-plate campaign remains separate.

## Shell first-contact controls on refined geometry

`shell-smooth-tolerances-v1` (`727506`) used relative PCG1e-6 and velocity
bounds1e-7/1e-8 on the original refined smooth geometry. Both first failed at
step46, during top contact, with equation residual14.99 and failure flag1.
Their first rejected complete fields are retained. Absolute-velocity tolerance
alone does not repair that contact solve.

`land-shell-pilot-v1` (`727528`) uses the same additional land geometry and
unchanged pinned Shell9a58a7359, nominal500um wall, E10GPa, nu0.3 and rho2000.
Its mass is7.485182g,0.0594% above the coarse solid volume mass; this finite
geometry difference is explicit. Tight1e-8/1e-9 settings first fail at step46,
with equation residual0.0017925 and9970 accumulated PCG iterations. The shared
failure bit alone does not distinguish an inner cap from a contact-Newton cap.
The observer now records both counters and the last checked linear residual
at the first failure; it does not change solver state or continuation policy.
`land-shell-inner-v1` (`727531`) was rejected before stepping because its hold
was zero. The corrected `land-shell-entry-v2` (`727536`) compares20um clearance
and an initially touching free egg, with relative1e-6, velocity1e-7 and4000
inner iterations. No failed history is admitted by changing its thresholds.

`contact-active-set-fields-v1` (`727539`) verifies complete common material
fields on the same meshes. The original smooth0.1N tangent difference is
1.16e-12; the previously admitted land0.3/1N differences are2.30e-14/2.17e-14.
Membrane, bending and unsmoothed peaks agree at comparable roundoff levels.
The formerly rejected land0.1N now passes all existing discrete gates.

Both `land-shell-entry-v2` runs failed:20um clearance at step47 (equation
residual5.18e-4), and initially touching at step1 (residual1.016). Both report
7 contact Newton iterations, below the32-iteration outer cap, while total
linear iterations are11014/28000. The sticky cap is therefore an inner-solve
failure. The last-linear scratch counter and residual threshold have already
been reset by the final preparation pass; their recorded zeros are not proof
of a zero-residual solve. Read-only first-cap instrumentation would be needed
for that residual history. `land-shell-precondition-v1` (`727543`) compares
zero and12 coarse patches on the same initially touching0.001N first step,
plus one hold step, against the failed48-patch control. This bounded check
addresses the elastic-only coarse operator versus contact-updated block
preconditioning; it changes no material, contact equation or admission gate.

## Optional coarse-space compilation repair

The zero-patch control failed during compilation: `func_pcg_norm_scales`
and `func_pcg_advance` unconditionally read `coarse_vec.shape[0]` although
that optional tensor is unallocated. An isolated variant at
`.worktrees/shell-zero-coarse` guards those accesses with the existing static
`has_coarse_space` condition. It changes no equations, tolerances or default
preconditioner. The exact source/test diff against9a58a7359 is retained in
[`shell_zero_coarse.patch`](../shell_comparison/patches/shell_zero_coarse.patch).
Apply it with `git apply` to that pinned Shell revision to reproduce the variant.

`shell-zero-coarse-tests-v1` (`727547`) runs the same existing40-step impact
momentum/contact regression, parametrized over zero/default12 coarse patches
and unbatched/two-environment scenes. Original source fails both zero-patch
cases during compilation; repaired source passes all four cases (113.87s).
The original assertions and tolerances are unchanged. Ruff and diff checks
pass. This is an observable optional-configuration repair, not evidence of
high-stiffness stress qualification.

`land-shell-zero-fixed-v1` (`727549`) now compiles and reaches its first
physical step, but hits the linear cap with11 contact Newton iterations and
44000 accumulated PCG iterations; equation residual is0.861. The12-patch
control also failed at its first step, with9 Newton iterations and36000 PCG
iterations. Disabling the coarse correction alone therefore does not resolve
the stiff contact solve. The next numerical investigation should capture the
first failed inner system before scratch reset, compare its true residual and
operator with an independent frozen-system solve, and then repair the measured
conditioning/operator defect. More tolerance sweeps alone are unsupported.

The first repaired-source pilot's `start.json` inherited a hardcoded9a58
variant description. Its captured `source.json`/`source.patch` correctly
identify the modified implementation; use those receipts for provenance.
The driver now hashes the loaded solver file and uses a neutral captured-source
description for subsequent variants. Historical receipts remain unmodified.
All runtime assets, fields, test logs and failure captures in this section
remain below the qualified-stage data root. The overall goal stays in progress.


## Frozen coupled system and nonlinear stopping budget

`land-shell-floating-v1` (`727552`) isolates the existing floating-sum setting
(`use_deterministic_algorithms=False`) on the optional-zero-coarse repair
`a86ba81d0`; no constitutive or contact penalty change is made. The zero-patch
arm returns two steps without solver failure, but its equation residual is
0.96699. The 12-patch arm still fails its first linear solve. Returning without
a failure flag is not physical admission.

To distinguish linear error from contact iteration, `shell-linear-capture-v1`
(`727554`) uses a separately retained diagnostic source: apply
`patches/shell_first_linear_capture.patch` under `shell_comparison` to
`a86ba81d0`. This truncates the first contact Newton iteration immediately after
its linear solve and before line search; its returned state is deliberately
not an admissible physical trajectory. The driver checks the exact solver
SHA256 `3f42e2d46c3d45292bba294c84e503fc4d5d04b9c6614ec5da2200f5c2bfb360`
before allowing this capture mode. Cached tangents, RHS, contact Jacobians,
Schur inverse, iterate and recursive residual are saved without observer writes
to simulation state. The retained patch SHA256 is
`10aa4d4ef945ad66462339737f77cf442e1744e1ef56fbbc81f361593939ec01`.

`audit_shell_linear` independently differentiates the membrane stress in NumPy,
assembles the hinge/contact terms and the joint shell/rigid sparse operator,
and solves with SuperLU. The auxiliary rigid RHS is zero because the captured
shell RHS is already reduced; eliminating this auxiliary block reproduces the
captured Schur equation. `shell-linear-audit-v1` (`727558`) reports:

| First frozen solve | Fixed-point sums | Floating sums |
| --- | ---: | ---: |
| PCG products | 4000, capped | 2037, converged |
| Independent direct relative residual | 1.879e-11 | 1.255e-11 |
| Actual iterate relative residual | 1.242e-2 | 2.837e-5 |
| Recursive/true residual disagreement, relative to RHS | 1.230e-2 | 4.780e-12 |
| Mass-RMS velocity error versus direct solution, m/s | 2.411e-7 | 3.672e-13 |

The assembled operators have relative asymmetry below6e-17 and positive rigid
inertia after subtracting contact stiffness. The floating solve satisfies its
absolute mass-based velocity error bound, despite not meeting the relative
criterion. These data localize a fixed-point accumulation precision problem;
they do not establish correctness of the complete nonlinear contact step.

`land-shell-newton-budget-v1` (`727562`) therefore keeps relative PCG tolerance
1e-6 and uses the existing floating sums, zero coarse patches and an8000-product
cap, varying absolute velocity tolerance from1e-7 to1e-10. Both two-step arms
return without failure. At1e-7 the cached contact velocity changes match
`J_shell dv_shell - J_rigid dv_rigid` to3.4e-21m/s, ruling out a cache mismatch
at that observation. However, the local nonlinear impulse-model stopping
budget converts to as much as0.01194N per contact, against a commanded0.001N;
the sum of all local budgets is2.888N. These are allowed budgets, not measured
errors. At1e-10 the final equation residual falls from0.96699 to1.922e-7,
while the final static residual is9.817e-5 and the egg is still transient.
This supports tightening the contact/absolute-velocity criterion for this
stiff, small-load case; it does not justify arbitrary tighter relative PCG.
The new longer load/hold controls must still establish an equilibrium window,
trajectory admission, timestep/repeat sensitivity and reference agreement.

## Isolated wall refinement

`contact-wall-isolation-mesh-v1` (`727556`) and
`contact-wall-isolation-fem-v1` (`727560`) keep the original coarse projected
surface exactly fixed and increase wall layers from4 to8. The mass is unchanged,
and the8-layer contact solve passes all existing algebraic gates, including
4.71e-12 equation residual. Nevertheless, exact full-column tangent change is
31.94%, membrane6.925%, bending5.988%, raw von Mises2.068% and raw tension18.29%.
This is not a qualified reference, and the peak-only von Mises agreement cannot
substitute for the complete-field criterion. The control motivates testing
quadratic elasticity rather than another blind simultaneous P1 refinement.


## Sustained coupled Shell on the declared land geometry

`land-shell-hold-v1` (`727564`) uses the unchanged constitutive/contact model
on the optional-zero-coarse repair, floating sums, relative tolerance1e-6,
absolute velocity tolerance1e-10 and8000 products per inner solve. Each full
free egg starts touching, ramps for50ms and holds for150ms at0.001N or0.1N.
Both200-step histories return without a solver failure. Their final raw von
Mises peaks are1548.53Pa and165388.13Pa. No state or velocity clearing is used.

The independent `contact-order-controls-v1` (`727571`, `shell-windows`) checks
all51 observations in the last50ms against the previously declared reaction,
plate-speed, inertia, kinetic-energy and equation gates. Both windows pass.
At0.1N, the maximum equation residual relative to the *contact force norm*
is2.717e-9, maximum plate speed2.62e-14m/s and maximum kinetic energy5.64e-19J.
The driver originally retained the residual norm and internal/contact
resultants rather than the inertia resultant itself. For these two older
histories the audit uses the rigorous bound
`||sum(f_internal+f_contact)|| + sqrt(n_vertices)*||residual||`;
its maximum is9.762e-9N for0.1N, comfortably below the predeclared1% gate.
New histories directly retain the inertia resultant as well. This is mechanical
admission only; resolved contact geometry, full physical reference uncertainty
and matched three-method error/cost are still required.

The half-dt arm of `land-shell-stability-v1` (`727569_1`) also completes all400
steps without failure, ending at0.1N reaction,9.61e-10 equation residual
(command-force normalization), and165388.131Pa raw von Mises. Complete-field
comparison and its own50ms admission are performed separately. The other arm
repeats the0.1N history, then continues0.3/1/0.3/0.1N without resetting state;
only its completed frames may be compared while the rest of that job runs.


`land-shell-sensitivity-v1` (`727581`) compares complete same-mesh fields with
area weights, rigid corotation and exact Simpson integration of the squared
linear fiber field. The repeated0.1N endpoint differs by7.17e-10 in tangent
norm,5.86e-10 membrane and9.64e-10 bending. The half-dt run passes its own101-
observation mechanical window but changes tangent stress2.260%, membrane0.475%
and bending4.125%. Both raw peaks change by less than1e-7 relatively.
Thus the0.5% field-sensitivity target **fails despite essentially identical
peaks**. This configuration is mechanically settled but not yet admitted for
timestep-independent stress ranking. `land-shell-quarter-dt-v1` tests the next
half-step on the same physical history; its result must not be inferred from
the passing peak comparison.


## Quadratic independent elasticity and continuous surface-gap constraints

The independent reference now has an explicit `--element-order 2` option using
scikit-fem affine P2 tetrahedra, consistent mass and all six rigid gauge modes.
The existing P1 path remains the default. Ordered and unordered constant-stress
and exact pure-bending controls (`727567`, repeated with column interpolation
controls in `727571`) have stress errors below8.5e-14, equation errors below
5.2e-14 and negligible balancing corrections. The gap oracle is checked on
quadratic faces whose six nodal gaps are positive but whose interior penetrates.
Prism-column interpolation is checked at37 depths for1/2/4 layers, including
orientation changes. `contact-order-controls-v1` reruns the original P1 case;
`contact-p2-continuous-v1/group1/p1-unchanged` retains its complete-field
comparison, so the new optional order does not silently replace old results.

`contact-p2-pilot-v1` (`727567`) uses the original coarse projected surface with
one quadratic wall layer. Nodal contact alone has1.24e-12 equation residual but
2.014e-9m penetration inside a quadratic triangle and is **rejected**. The
optional `--continuous-quadratic-contact` path repeatedly adds the exact
most-penetrating point of every affected face, using the same P2 interpolation
for the gap and the dual compressive force. All added compliance rows are exact
interpolations of the retained nodal operator; no additional elasticity solve
or fitted pressure radius is introduced.

The first constraint-generation run (`727572_0`) reduces penetration to
1.166e-15m but fails the unchanged full-rank gate (rank31/dimension32).
`contact-p2-continuous-v2` (`727580`) removes one near-coincident generated
constraint, solves again and rechecks the entire quadratic boundary. Round13
has rank31/dimension31,1.31e-12 equation residual,1.19e-15 relative wrench
error, nonnegative physical multipliers and minimum gap-1.164e-15m. It passes
all algebraic/contact gates without changing thresholds. The raw von Mises
peak is17.07148MPa, raw tensile peak7.52946MPa and plate travel0.782235um.
The failed nodal and rank-deficient variants remain separate retained runs.

`contact-p2-support-v1` (`727602`) independently verifies the P2 dual load
transfer, including moment error below1e-22Nm. Physical load support is defined
by the nonnegative *constraint-point* multipliers, not by signs of consistent
P2 nodal forces. The latter can be negative and cannot identify pressure
support. The older P2 pilot's `fully_active_face_area_m2` is only the area of
subtriangles whose nodal force weights are positive; it is not a resolved
pressure-footprint measurement. Current output names this distinction explicitly.
The support audit finds19 lower and8 upper compressive points, maximum radii
31.86um and8.436um respectively; fully closed quadratic-face area is2.429e-9m²
below and **zero above**. Both reactions equal0.1N. This is not a resolved
finite-area continuum reference despite passing the discrete equations.

`contact-p2-wall-v1` (`727596`) isolates one-to-two quadratic wall layers on the
same projected surface and uses exact two-point Gauss integration on the
unioned tetrahedral column intervals. This integrates squared linear stresses
and their membrane/bending moments exactly through the retained affine elements;
raw extrema are retained at every tetrahedron corner. The separate P1-four-
layer/P2-one-layer comparison changes tangent stress32.08%, membrane24.52% and
bending23.42%, so no cross-order agreement is claimed. Spatial/contact and wall
convergence, finite-deformation checks and all three-method ranks remain open.


## Completed load cycle, relative tolerance and next reference control

`land-shell-unload-audit-v1` (`727603`) verifies all five51-observation
mechanical windows of the continuous0.1/0.3/1/0.3/0.1N history. All1000 steps
returned without solver failure. Returning to0.1N changes the complete tangent
field6.80e-10, membrane5.71e-10 and bending8.91e-10 relative to the independent
initial0.1N run. These are reproducible mechanical/stress-history controls, not
continuum accuracy ranks.

`land-shell-relative-control-v1` (`727604`) changes only relative PCG tolerance
from1e-6 to1e-4, retaining absolute velocity tolerance1e-10, dt1ms and the same
land geometry/load history. `land-shell-final-controls-v1` (`727607`) verifies
its mechanical window and reports tangent/membrane/bending differences
6.86e-10/6.97e-10/6.58e-10, with raw peak changes below8e-11. This controlled
case provides no accuracy justification for1e-6 relative PCG;1e-4 is sufficient
here. The first-linear and nonlinear-budget controls explain why its absolute
contact/velocity criterion still matters. This does not equate Shell's norm
ratio with QIPC's quadratic-form ratio or establish adequacy for every case.

The quarter-dt run (`727591`) completes800 steps and passes its own50ms
mechanical window. Against half-dt, complete tangent/membrane/bending changes
are0.8379%/0.2852%/1.4903%. The trend improves but still fails the predeclared
0.5% field target; essentially identical raw peaks cannot waive it.
`land-shell-eighth-dt-v1` (`727609`) continues the same physical history at
0.125ms with paired1e-4/1e-6 relative tolerances, so timestep and inner-solve
sensitivity can be separated.

`contact-p2-wall-v1` (`727596_0`) reaches small equation/gap residuals but fails
its active-set convergence/rank gate: neighboring faces generated duplicate
edge-point constraints. Its13.07% complete tangent difference from one wall
layer is diagnostic only;97.74% of the squared discrepancy lies inside0.5mm
of a pole, while the full comparison domain remains unchanged. No qualified
wall-refinement result is inferred from that rejected solve.

Constraint generation now deduplicates new interpolation rows and removes a
redundant generated active row before retrying a rank-deficient inner solve.
Every retry still checks the original primal/dual/rank and exact face-gap
criteria. The cheap condensed-operator replays in
`contact-p2-constraint-control-v1` (`727605`) pass both one- and two-layer
controls (rank31/31 and49/49; minimum scaled gaps-5.82e-9 and-8.52e-9).
They reuse the exact saved elasticity operator and do not pretend to recompute
full displacements or gaps outside candidate faces. The full two-layer rerun
`contact-p2-wall-repair-v1` (`727608`) reuses that operator but independently
refactors elasticity, checks the complete boundary and retains the complete
stress field before any admission claim.


## Full two-layer repair and isolated contact refinements

The full `contact-p2-wall-repair-v1` rerun (`727608`) completes successfully.
It passes the unchanged discrete contact/elastic gates: rank49/49,
15 constraint-generation rounds, equation residual4.443e-12, wrench error
8.886e-16 and whole-surface minimum gap-1.702e-15m. This supersedes the failed
solver status of `727596` only for the separately retained repaired run.
The complete one-to-two-layer comparison remains unresolved: tangent13.072%,
membrane2.092%, bending5.090%, raw von Mises0.429% and raw tension18.016%.
Fully closed quadratic-face area remains zero on both poles. Passing algebraic
checks therefore does not establish a finite-area physical reference.

`contact-targeted-mesh-v1` (`727611`) validates optional graded wall depths,
including positive prism volume, affine stress at37 depths, and exact analytic
membrane, bending and squared-stress column integrals. Uniform default meshes
retain their former construction; new mesh receipts store the depth fractions.
The same independent ordered/unordered elasticity controls remain below8.5e-14
stress error and5.2e-14 equation error.

`contact-p2-targeted-v1` (`727612`) separates two controls against the accepted
discrete two-layer field. `surface25` halves the local surface size from50um
to25um while retaining global2mm resolution and two uniform wall layers.
`graded50` retains the50um surface and uses wall fractions
[0,0.05,0.15,0.4,1], corresponding to25/50/125/300um layer depths. Both keep
the same declared smooth geometry, material and0.1N load. Their120um candidate
radius only bounds the contact search; complete-boundary gap checks remain
mandatory. No field/peak convergence is inferred before their results.

The contact-operator reader additionally permits a verified subset of cached
candidate nodes on an identical mesh and element order. It checks node IDs,
signs, rigid coordinates and initial gaps exactly before extracting the
compliance submatrix. `contact-p2-cache-subset-v1` (`727621`) tests200um to120um
search-window reduction on the already solved two-layer mesh and compares the
complete field after a fresh elastic factorization. This is a solver-search
control, not a fitted physical pressure footprint.


The subset control `727621` completes with all original algebraic/global-gap
gates satisfied: rank50/50, equation residual4.427e-12, minimum full-boundary
gap-1.648e-15m. Against the200um search, complete tangent/membrane/bending
changes are8.82e-7/4.26e-7/5.92e-7; raw von Mises changes1.40e-6. These are
well below the0.5%/1% solver targets and do not resolve spatial uncertainty.
The retained physical point supports still have zero fully closed-face area.

## Eighth-step field audit

The1e-4 arm of `land-shell-eighth-dt-v1` completes all1600 steps and passes its
401-observation last50ms mechanical window (`727622`). Maximum equation
residual relative to contact-force norm is2.349e-9; maximum plate speed is
2.47e-13m/s. Against the quarter-step1e-6 field, tangent/membrane/bending change
is0.6175%/0.2001%/1.1023%, so the0.5% full-field target still fails.
The same-dt1e-6 arm must be checked separately before isolating this as pure
timestep sensitivity.

`compare_coupled_shell` now retains complete per-face squared error/reference
densities, areas and centroids for each field metric, without changing its
full-domain metrics or acceptance. The replay `land-shell-difference-map-v1`
(`727624`) reproduces all old scalar metrics exactly. Its diagnostic regions
locate98.17% of squared tangent difference and99.06% of squared bending
difference between0.5mm and4mm polar radius. The rest of the surface remains
in every reported metric; this localization is not a trimmed comparison.

`land-shell-sixteenth-dt-v1` (`727625`) tests0.0625ms with1e-4 relative PCG,
unchanged1e-10 absolute velocity tolerance, unchanged contact model and the
same50ms ramp/150ms hold. It retains every step and runs its mechanical-window
and complete-field comparisons after successful completion. No admission is
inferred from submission or from essentially unchanged raw peaks.


The paired audit `land-shell-eighth-paired-audit-v1` (`727627`) now completes:
both1600-step histories pass their401-sample equilibrium windows with no solver
failure. At fixed0.125ms, changing relative PCG1e-6 to1e-4 changes complete
tangent/membrane/bending fields by3.52e-8/2.70e-8/4.98e-8 and raw peaks by at
most2.44e-9. The same-relative1e-6 quarter-to-eighth comparison reproduces
0.6175%/0.2001%/1.1023%, separating the observed timestep dependence from the
negligible relative-PCG effect. The1ms and0.125ms paired controls both support
using1e-4 for this0.1N land case with absolute velocity tolerance1e-10; they do
not establish a universal tolerance for other loads or materials.

This supports the user's distinction between inner and outer accuracy.
For a Newton linearization, the PCG residual is the error in satisfying the
current linear Newton equation; tightening it beyond what changes the accepted
nonlinear solution needlessly increases work. Conversely, a small Newton
increment alone is not a force-residual guarantee. In the pinned Shell source,
the absolute velocity tolerance participates in both the linear error bound
and contact-model/Newton termination. QIPC's quadratic-form relative test and
its separate Newton displacement test have different units/normalizations;
the same printed tolerance cannot be equated across these solvers. The actual
coupled residual and measured complete-field sensitivities remain the checks.


## Shell timestep sensitivity now passes at0.0625ms

`land-shell-sixteenth-dt-v1` (`727625`) completes all3200 steps, with1e-4
relative PCG and the same1e-10 absolute velocity tolerance. All801 samples in
the final50ms pass the mechanical gates. Maximum residual relative to contact
force is2.532e-9, maximum plate speed4.36e-13m/s and maximum kinetic energy
9.85e-21J. Against0.125ms, complete tangent/membrane/bending changes are
0.01608%/0.01013%/0.02528%; both raw peaks change by less than4.4e-8 relatively.
This passes the0.5%/1% timestep targets after the retained coarser failing
steps. It does not establish continuum accuracy or the three-method ranking.

## Retained hard-contact failures and independent QP certificate

Both `contact-p2-targeted-v1` arms (`727612`) finish but fail discrete contact
admission. The25um surface arm reaches40 rounds with rank154/156 and minimum
scaled gap-9.89e-6. The graded-wall arm stops at18 rounds with a full-rank76/76
basis but exhausted inner active-set iterations and minimum scaled gap-1.01e-6.
Their full-field differences are diagnostic only, despite nonzero closed-face
areas in some patches. The original full-surface and force/equation thresholds
are unchanged.

Two experimental active-basis variants are retained in the immutable analysis
captures `contact-p2-basis-control-v1` (`727632`) and
`contact-p2-scaled-basis-v1` (`727644`). A null-direction reduction fixes a small
constructed redundant multiplier basis (old rank8/11) but cycles on the actual
fine problem. Additional diagonal equilibration also makes the previously
passing two-layer control fail. Neither experimental implementation is retained
in the active source; `plate_contact_fem` and `quadratic_contact` are restored
to the preceding validated commit. These failures cannot be hidden behind the
passing constructed regression.

`audit_contact_qp` instead asks pinned Clarabel0.11.1 to solve the *retained*
convex QP independently, with explicit primal/dual/complementarity checks.
It does not modify the contact driver or waive its original rank/gap gates.
The interface follows the [official quadratic-program specification](https://clarabel.org/stable/python/getting_started_py/).
`contact-independent-qp-v1` (`727698`) reports `AlmostSolved` for the two-layer
and25um retained systems, which remain rejected by this audit. The graded-wall
system returns `Solved`, with wrench error2.65e-15, stationarity7.01e-12 and
maximum complementarity9.19e-14. Its fixed-QP certificate passes, but its exact
candidate-face gap remains-3.89e-8 scaled and therefore needs additional
geometric constraints. This is not a full elasticity or outside-candidate
boundary check. All three complete raw results and solution arrays are retained.

## Independent prescribed finite-area reference

`finite_area_reference` implements the early same-traction comparison required
by the goal, separately from the unsolved hard-contact benchmark. On the
declared polar-land geometry, each pole receives0.1N axial inward traction
proportional to max(1-rho^2/a^2,0)^3 with **a=2mm declared before evaluating
stress**. Each pole is normalized independently; the shape is C2 at its edge.
Independent P2 facet integration retains both physical quadrature point forces
and consistent nodal forces, preserving force, moment and rigid virtual work.
No footprint is fitted to stress. The full egg remains free; reported mass-mode
corrections are only roundoff-sized.

The augmented `validate_p2_contact` checks exact P2 virtual work for a quadratic
displacement and equality of eighth/tenth-order integration for an unclipped
polynomial load. `finite-area-pilot-v1` (`727643`) and the post-rollback control
`finite-area-controls-v1` (`727697`) pass this and all preceding independent
elasticity/gap/column controls. The one-layer land reference passes equation
residual7.83e-12 and wrench error1.61e-16. Changing surface quadrature8 to10
changes full tangent stress1.11e-7, membrane5.46e-8 and bending1.43e-7 relatively;
raw peaks change by at most2.42e-7. The tenth-order peak is323363Pa von Mises
and260528Pa tension.

The two-layer arm of `finite-area-refinement-v1` (`727663_0`) also passes,
with1.49e-11 equation residual and329658Pa/265657Pa raw peaks. The one-to-two
comparison (`727697`) changes complete tangent2.685%, membrane1.590%, bending
1.288% and raw peaks1.910%/1.930%. Complete-field wall convergence is not yet
qualified at the2% target. The simultaneous250um/global1.5mm two-layer arm
(`727663_1`,245760 tetrahedra) fails in SuperLU factorization with `MemoryError`;
its128GB allocation and 43,882,352KiB recorded peak RSS are retained, with no
field or admission inferred.

`finite-area-isolated-mesh-v1` (`727702`) and
`finite-area-isolated-refinement-v1` (`727703`) keep global2mm resolution fixed:
one arm increases wall layers2 to3 on the same9880-vertex surface; the other
halves local size500 to250um on a16738-vertex surface and retains two layers.
They measure quadrature and complete-field differences separately. These jobs
remain pending verification; they are not a completed reference qualification.
The follow-up results below supersede that pending status.

## Finite-area full fields and independent contact certificates

The three-layer finite-area reference (`727703_0`) passes equilibrium with
relative residual2.67e-11. Two-to-three layers changes complete tangent1.535%,
membrane0.614%, bending0.304% and raw Mises/tension0.544%/0.242%. These improve
on the one-to-two trend and satisfy the adjacent wall targets. Surface
qualification remains open: the isolated250um two-layer arm (`727703_1`)
fails SuperLU factorization with `MemoryError` at41,302,968KiB recorded RSS.

An optional material-column graph ordering changes only the factor permutation.
`finite-area-ordering-control-v1` (`727740`) passes constant-stress, bending,
uniform/graded prism and same-mesh full-egg controls. Complete field changes
are3.27e-13/3.32e-13/3.21e-13; peaks change by at most1.24e-13. The ordered
250um retry (`727742`) still fails SuperLU at40,333MiB. Both failed receipts
remain on data. A separate optional CHOLMOD factor control uses the isolated
Cholespy2.2.0 asset installed by `727753`; it preserves the physical matrices,
six rigid gauges and all full-equation checks. Its controls are independently
recorded under `finite-area-cholesky-control-v1` (`727755`).

`prepare_native_mesh` exports the exact retained mesh, outer collider and
volume-integrated mass/inertia. `finite-area-native-inputs-v1` (`727735`)
prepares the4498-outer-vertex mesh and its independently integrated P2 loads.
The direct native pilot (`727736`) times out during scene construction at15min;
it supplies no field or timing rank. The existing native PCG route (`727743`)
completes on the same mesh. Its core, DPR and SCR all pass native admission,
with equation residuals about1.08e-10. Feeding the exact independent nodal
forces to the native core changes maximum stress components by3.13e-12
relative to the independent same-mesh solution. DPR/SCR both have contact
status0/0, and retain their actual reconstructed loads. Radius2mm is fixed
before field evaluation. Cold compilation/solve times remain diagnostic.

`finite-area-shell-pilot-v1` (`727737`) applies those independently declared
tractions to the matching mid-surface through force/couple transfer. Each
quadrature point's force, moment and rigid virtual work are checked. Explicit
velocity-cleared static relaxation converges in3 updates to actual elastic
force residual5.31e-9, with no solver failure. This separately labeled static
method does not qualify the original dynamic plate-contact history.

The shared column comparator now accepts Shell's corotated linear fiber field,
checks midpoint/collider identities and saves inner/outer tangent errors plus
absolute RMS error. Complete thickness integration retains every tetrahedron
partition and both boundary fibers. `finite-area-shell-comparison-v1`
(`727741`) passes analytic Shell/solid fiber controls and compares one/three
surface points per reference face. Further9/25-point positive-weight Duffy
quadrature controls are kept separately from mesh refinement.

`finite-area-three-method-fields-v1` (`727752`) gives the following **provisional
prescribed-load differences**, using three area points per finest face:

| Method/control | Tangent field | Membrane | Bending | Raw Mises | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPR versus independent same mesh | 3.039% | 1.057% | 4.064% | 9.566% | 10.003% |
| SCR versus independent same mesh | 2.773% | 0.612% | 3.775% | 4.359% | 8.062% |
| DPR versus three-layer reference | 9.405% | 7.159% | 8.841% | 6.526% | 6.452% |
| SCR versus three-layer reference | 9.370% | 7.187% | 8.762% | 1.463% | 4.574% |
| Static Shell versus three-layer reference | 32.323% | 8.250% | 43.440% | 5.801% | 12.673% |

The native meshes are coarser than the Shell/reference mesh; their discretization
error remains included in the latter rows. Reference surface uncertainty and
angular integration still require qualification. The close DPR/SCR differences
do not establish a separated ranking. These fields are an early same-load
comparison and leave end-to-end contact, nonzero sustained friction, matched
admitted cost/stability and final figures outstanding.

For the original hard smooth contact, adaptive dual QP replay (`727744`) passes
its first finite constraint set but returns `AlmostSolved` after adding exact
gap constraints. It remains a failed certificate. The mathematically equivalent
whitened primal form writes C=L L^T, v=L y and minimizes
`0.5*y.T@y + target.T@q` subject to `R*(L*y+gap+Q*q)>=0`. Its Hessian avoids
the singular multiplier-space matrix R C R^T while preserving the same contact
problem. `contact-independent-qp-primal-v1` (`727746`) returns `Solved` for
wall2/surface25/graded50 after1/7/3 rounds, with minimum candidate gaps
-8.52e-9/-9.03e-9/-9.92e-9 in the unchanged scaled units. Force/moment,
stationarity and complementarity certificates pass without accepting an
`AlmostSolved` exit or changing the exact face-gap target.

`plate_contact_fem --contact-certificate` checks source/operator/load identities,
recomputes condensed KKT quantities and then freshly solves the complete
elastic body. Primal-dual admission uses force-weighted complementarity, global
quadratic gaps, nonnegative multipliers, wrench and elastic equations; multiplier
uniqueness is unnecessary for this convex certificate. The original active-set
path retains its existing rank and active-gap gates. Tiny positive interior-point
multipliers can occur at open constraints, so their raw positive-count/radius
and maximum positive-multiplier gap are diagnostics, not physical support or
an alternative complementarity test.

The full rerun `contact-primal-certificate-full-v1` (`727749`) admits wall2
and surface25: equation residuals4.45e-12/5.68e-12, global minimum gaps
-1.70e-15/-1.81e-15m, and force-weighted complementarity2.98e-13/1.39e-13.
The fine field retains raw Mises20.131MPa and tension11.232MPa. The graded
rerun also passes: equation residual2.34e-11, minimum global gap-1.98e-15m,
complementarity4.54e-13 and raw peaks25.318MPa/9.297MPa. Passing these discrete
equations alone does not establish a continuum reference.

`contact-primal-physical-audits-v1` (`727764`) independently verifies P2
point-to-node force/moment transfer and exact whole-face gaps. On the25um
surface,99% of each plate force is carried within42.10um/29.82um radius,
while at most2.73e-8 of force lies at constraints outside the existing closure
tolerance. Raw positive radii reach119um because the interior-point solution
retains tiny positive multipliers. They must not be mistaken for the contact
footprint. The prior admitted coarse active-set solution and the independent
primal replay agree to1.73e-10 in complete tangent field and1.53e-10 in raw peaks.
The50-to25um surface change remains11.21% tangent and14.83%/18.24% raw peaks
at three area points per face, so spatial qualification still fails. Higher
angular quadrature and the isolated wall comparison are retained separately
under `contact-primal-refinement-audits-v1`.

The CHOLMOD same-mesh control (`727755`) completes with equation residual
3.69e-11 and tangent/membrane/bending changes1.58e-12/1.77e-12/1.29e-12
against SuperLU. Its measured process RSS is4,132,676KiB. The250um two-layer
retry `finite-area-cholesky-refinement-v1` (`727759`) is a new factor-backend
control of the same previously failed mesh and remains pending verification.
The first control receipt's `cholespy_binary` entry identifies the package
initializer; the actual loaded extension hash is preserved in
`cholespy-install-v1/binary.json`, and subsequent runs record that extension
directly. [Cholespy's primary documentation](https://github.com/rgl-epfl/cholespy)
describes the bundled CHOLMOD CPU factor/solve and double-precision interface.

The250um CHOLMOD retry (`727759`) subsequently completes in15m01s. Factorization
takes860.91s, measured process peak RSS is14,159,820KiB and the elastic residual
is3.25e-10. Its complete field is retained with SHA256
`ac36659dad010246bafef5a5da33eb2e7210091ee5a4fa00c22f67f353f4df92`.
At fixed two-layer thickness,500-to250um changes complete tangent/membrane/
bending by3.070%/2.434%/1.613%, while raw Mises/tension change0.549%/0.152%.
The memory obstruction is resolved, but surface qualification still fails the
predeclared2% field target. The next controlled meshes (`727791`) retain the
same geometry/global spacing:125um/two layers has440112 tetrahedra, and
250um/three layers has301248. Their two isolated factor/comparison jobs
`finite-area-next-refinement-v1` (`727797`) use the validated optional backend;
submission alone supplies no new convergence evidence.

`finite-area-angular-control-v1` (`727758`) passes independent triangle
quadrature moments and exact Shell/solid fiber controls. Increasing9 to25
points per reference face changes the reported relative tangent errors by
3.23e-5/3.23e-5/3.13e-9 for DPR/SCR/Shell; membrane changes by at most6.20e-6
and bending by6.43e-5. The25-point differences against the three-layer reference
are9.514%/9.480%/32.322% tangent,7.404%/7.431%/8.390% membrane and
9.077%/8.999%/43.419% bending. Raw peaks remain unchanged. These are convergence
checks of comparison metrics, separately from reference surface discretization.

## Physical integration measure and direct contact controls

The optional `contact_fem_exact_columns --physical-volume` integrates the
piecewise affine reference tetrahedra's actual Jacobian, including changes
within each material column. It verifies the total against the independent
sum of tetrahedron volumes. Analytic tapered/sheared prisms with both uniform
and graded wall partitions pass at1/3/9/25 angular points (`727771`). The
three-layer reference volume is3.7403704039650184e-6m3. With25 angular points,
DPR/SCR/Shell tangent differences become9.5318%/9.4972%/32.4403%; all spatial
points and raw extrema remain included. The old normalized-area results stay
available for provenance. Membrane/bending resultants still use the explicitly
declared500um normal-material coordinate and physical midpoint areas.

`contact_qp` now exposes the independently certified dual and whitened-primal
forms for direct use. Eight-spring analytic controls check exact forces and
free rigid coordinates, including duplicate constraints whose multiplier
split is nonunique (`727769`). Replaying the retained coarse operator agrees
with its earlier certificate. `plate_contact_fem --qp-solver-path` starts from
the original candidate constraints, refines exact quadratic gaps and freshly
checks the complete elastic solution. It does not rely on a previously fitted
contact solution or remove any hard-contact admission gate.

The direct hard polar-land pilot (`727770`) passes discrete contact and elastic
equations at0.1N: residual9.83e-13, force-weighted complementarity7.55e-13 and
plate travel2.526e-7m. However, both fully closed quadratic-face areas are zero;
99% force radii are2.036mm/1.871mm and the load is concentrated near a ring.
Flattening the polar geometry has not itself produced resolved area contact.
Keep this case distinct from prescribed distributed pressure.

The original hard smooth-egg refinement also remains unqualified. At25 angular
points,50-to25um changes complete tangent stress11.440% with raw peaks changing
14.83%/18.24% (`727765`). A strict solved contact certificate resolves the
optimization failure, not the spatial field error.

## Predeclared physical compliant-pad case

The additional [pad declaration](COMPLIANT_PAD_DECLARATION.md) fixes the
compression-only stiffness at1e9N/m3 before stress evaluation. This authorized
alternative retains the full free egg and solves plate travel and footprint.
It preserves the failed hard-contact spatial qualification and the separate
same-traction controls. A fixed physical series compliance is not a Newton or
PCG tolerance and must not change with timestep or mesh.

The independent QP extension adds positive diagonal pad compliance to the
complementary energy, or equivalent unit-energy pad coordinates to the primal
form. Analytic eight-spring controls cover compliance0.01/3/100 in both forms
and pass (`727801`); all prior hard-contact and duplicate-constraint controls
also pass. The P2 pilot uses positive projected-area facet quadrature, checks
consistent force/moment transfer and the actual compression-only pad law, and
checks exact quadratic gaps across every omitted boundary face. Negative
distance into the pad is physical compression, so the hard-gap gate is not
misapplied to this changed model. Quadrature, spatial convergence and matched
Shell contact histories remain required; a pilot submission does not qualify
the physical reference or close the ranking goal.

The pilot (`727801`) passes both fourth- and sixth-order quadrature solves.
At fourth order its pad-law/elastic residuals are1.30e-11/3.47e-12, plate travel
15.207um, and lower/upper compressed projected areas16.001/14.806mm2. Exact
minimum gap on every omitted face is positive104.87um. These checks confirm
discrete area compression in this declared pad model, separate from the hard
land's narrow-ring support.

`compliant-pad-boundary-control-v1` (`727802`) removes theoretical zero
off-boundary shape functions by mesh topology, rather than admitting interior
nodes through floating-point basis roundoff. Candidate coordinates reduce from
1416 to834; old/new full fields agree within1.98e-14. All fourth/sixth/tenth
order pad solves pass. Fourth-to-sixth changes complete tangent/membrane/bending
by0.2327%/0.0951%/0.3525%; sixth-to-tenth changes0.08334%/0.03772%/0.1248%.
The latter raw Mises/tension changes are0.01243%/0.00728%. Tenth order gives
plate travel15.264um, raw peaks203474Pa/126406Pa and compressed areas
15.616/14.366mm2. The measured process RSS is3,482,900KiB and the three-order
reference takes168.71s. This qualifies the bounded coarse quadrature sensitivity,
not surface/wall uncertainty or a primary timing comparison. The500um/two-layer
pad refinement (`727803`) is a separately captured spatial control.

`finite_area_shell --solve-pad-contact` uses the separately declared static
Shell kinematics to compute pad pressure from the current Shell deformation.
It imports only reference quadrature geometry, area, stiffness and target load
for the contact solve; it does not impose the solid's fitted pressure. The
four-coordinate spring-energy solve has analytic uniform/asymmetric controls,
and the displacement/force transfer checks nonrigid virtual work. An omitted
face check uses the same affine outer-envelope map as the pressure law.
This prospective variant retains the original Shell constitutive kernels,
relative PCG1e-4, absolute velocity1e-10 and actual final force residual; it is
velocity-cleared static relaxation and is not a dynamic-history qualification.

The analytic pad/kinematic controls (`727812`) pass, including recovery of all
four coordinates for asymmetric gaps and rigid-motion reproduction at the
offset outer envelope. The paired Shell run (`727813`) then completes both
branches in3 pseudo-steps without a solver failure. The self-consistent pad
branch has final actual elastic-plus-pad force residual5.79e-9, force/moment
error1.25e-15, plate travel15.254um and minimum omitted-face gap104.85um. Both
plate reactions are0.1N. Its fixed-map static pad equations are admitted. The
separate same-traction replay has force residual3.66e-9. Their local summed
deformed moment drift is about6.80e-10Nm; its net is below3.7e-17Nm.

`compliant-pad-coarse-fields-v1` (`727815`) compares complete physical-volume
fields for both Shell branches and the retained native DPR/SCR results against
the coarse pad solid. The native fields use the same unchanged0.1N axial
point-source pair,2mm radius and exact4498-vertex mesh, so their valid earlier
results are reused rather than recomputed. They remain declared source-wrench
reconstructions, not a new native pad-contact trajectory. The paired Shell
comparison separates elastic-model differences from pad-pressure response.
This is a coarse control; spatial reference uncertainty, continuous histories,
sustained friction and matched admitted cost/stability remain open.

The completed coarse comparison (`727815`, nine angular integration points)
measures the following differences against the same-mesh pad solid. These are
**provisional coarse differences**, not qualified continuum ranks:

| Method | Complete tangent | Membrane | Bending | Raw Mises | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPR, fixed2mm source radius | 33.635% | 8.720% | 52.845% | 73.533% | 124.264% |
| SCR, fixed2mm source radius | 33.427% | 8.997% | 52.434% | 65.285% | 120.307% |
| Static Shell, solved pad | 31.484% | 11.821% | 48.017% | 6.392% | 24.220% |
| Static Shell, solid-traction replay | 31.481% | 11.821% | 48.013% | 6.364% | 24.173% |

The paired solved-pad versus replay Shell field differs by0.01125% tangent,
0.00287% membrane and0.01701% bending; raw peak changes are0.02630%/0.03771%.
For this coarse soft-pad case, elastic/thin-surface and source-profile errors
are much larger than the change caused by solving Shell's own pad pressure.
This observation does not waive reference refinement or justify changing the
native radius to fit the pad stress. The small DPR/SCR separation cannot yet
support a distinct order, and the different metrics need not share one rank.

## Continuous physical-pad Shell and independent reference acceleration

The [selected metadata and regression packet](../evidence/20261019-qualified-ranking/pad-history-controls-v1/manifest.json)
publishes byte-verified result/test receipts. Full fields and captured sources
remain at their hashed data paths. The two continuous-history engine inventories
were verified file by file against the committed variant (809 files each).

The isolated engine variant `41b2a57be` implements the declared work-conjugate
centroid pad map in coupled contact, preserving the original constitutive
kernels and default contact path. Its incremental patch is published with
the running instructions. A free flat-sheet regression checks the analytic
gravity compression19.62um, reaction0.44145N, velocity, action/reaction and
solver flags for1/2ms steps and unbatched/two-environment cases. Initial runs
`727824` and `727838` exposed a compile-time configuration error and invalid
zero-size optional arrays; the final bounded static flag and `maybe_shape`
allocation repair passes all10 selected tests in `727844`, including the
original momentum and state-restore controls. The intentional negative source
in `727865` changes `dt^2*k_A*area` to `dt*k_A*area`; the regression fails its
compression assertion (about0.02um actual versus19.62um required), not a build
or import check. That wrong variant is retained only as negative evidence.

`compliant-pad-shell-history-v1` (`727845`) completes200 continuous steps at
0.1N, dt1ms, PCG1e-4 and velocity tolerance1e-10. The final51-sample window
passes actual reactions, inertia, kinetic energy, plate speed, elastic
equations, pad pressure law and transpose transfer. Net compression is
15.932436um after removing44um initial clearance; raw Mises/tension peaks
are221467.666/162352.434Pa. The contact-scaled residual is about3.76e-9.
No velocity clearing occurs. These are admitted discrete normal histories,
not a claim of continuum stress accuracy.

`compliant-pad-shell-stability-v1` (`727859`) passes the101-sample window at
dt0.5ms and all five51-sample windows of the continuous0.1/0.3/1/0.3/0.1N
cycle. Halving dt changes tangent/membrane/bending by
8.37e-10/8.99e-10/7.47e-10. The independent repeat differs by at most3.59e-10
in those fields; returning to0.1N after the cycle differs by at most5.88e-9.
All histories return without solver failure. Observed wall times include
intrusive per-step diagnostics/IO and are not matched primary cost results.

Quadrature correction: the misleadingly named
`compliant-pad-centroid-control-v1` (`727828`) used library `intorder=1`, which
has three points per face (1152 total), not one centroid. Its immutable
receipt is preserved. `compliant-pad-explicit-centroid-v1` (`727854`) uses the
new explicit order0 rule (384 points), passes the discrete equations and
gives solid travel15.944876um. Against tenth order, complete tangent/membrane/
bending differences are1.92951%/0.76688%/2.87692%; raw peak changes are
1.59848%/3.38211%. One-point integration therefore still exceeds the2%
bending target on this coarse geometry.

`compliant-pad-matched-centroid-shell-v1` (`727906`) solves static Shell with
the same explicit centroid rule and gives travel15.932436um, three relaxation
iterations and force residual4.84e-9. Its continuous/static tangent/membrane/
bending differences are1.14e-9/6.10e-10/1.60e-9; peak differences stay below
2.74e-9. The earlier0.68um travel difference against high-order static Shell
was a quadrature difference, not a dynamic-coupling defect.

`compliant-pad-fixed-point-control-v1` (`727876`) verifies a faster reference
path against independent condensed Clarabel for orders0/6/10. It applies the
full elastic factor at every iterate, retaining free-body projection, the
same pad law, force/moment checks, global omitted-face checks and a fresh
final elastic solve. All three converge in five iterations; full-field
differences are below5.12e-11 and raw peak differences below5.83e-11. Total
runtime45.48s includes40.88s factorization; each contact solve takes about
0.50s. No numerical or physical acceptance threshold is relaxed.

The original500um/two-layer condensed QP (`727803_1`) has completed its
sixth-order solve: pad-law residual1.08e-9, elastic residual7.73e-10 and
travel13.452449um. Its QP alone took1165.75s. The tenth-order solve remained
live when an extension of its50-minute allocation was requested; Slurm denied
the time-limit change. It subsequently timed out at50m04s with no tenth-order
field; the admitted sixth-order field and all partial receipts remain intact. `727904` instead applies the independently checked fixed-point
method to500um and250um two-layer meshes at0.1/0.3/1N, orders0/6/10. Submission
is not admission; inspect all equation, search-extent and quadrature gates.

Separately, the250um prescribed-traction two-to-three-wall comparison
(`727797_1`) changes tangent/membrane/bending by1.58954%/0.18889%/0.27831%,
with raw Mises/tension changes0.61784%/0.00160%. Its301248-tet factor took
2105.67s and peak process RSS was23,747,416KiB. The125um surface refinement
is still a distinct running control. Neither this wall result nor discrete
pad admission proves complete reference convergence. Actual native pad
response, sustained friction, qualified rankings and matched costs remain
required before completion.

The refined continuous Shell campaign `727908` stopped during geometry
preparation, before any scene step: world-coordinate centroid subtraction
introduced a1.23e-13/2.18e-13 rigid-translation defect on500/250um faces, exceeding
the unchanged1e-13 off-diagonal threshold. `prepare_shell_pad` now constructs
and recenters lever arms in local face coordinates before forming the rotational
Gram matrix. This avoids cancellation without changing the physical map or
loosening admission. `727910` retries both geometry controls and the full load
cycles with the revised captured helper; its results must still be inspected.

The125um/two-layer prescribed-traction reference (`727797_0`) has now completed
in60m55s. Its factor took3567.63s, peak process RSS was32,884,204KiB and elastic
residual3.49e-10. Refining250um to125um changes complete tangent by2.02054%,
membrane1.33411% and bending0.80678%, with raw peaks0.00754%/0.40296%. The
tangent change narrowly exceeds the predeclared2% target; it is not rounded
into a pass. The trend improves from3.0699% at500-to250um, but surface and
wall uncertainties still need joint qualification. Both revised fine Shell
geometry controls pass their unchanged rigid-motion/virtual-work assertions;
the continuous load cycles are running.

The500um/two-layer fixed-point arm (`727904_0`) completes all nine
load/order combinations, with5–6 iterations per contact solve (2.29–2.81s),
434.03s factorization and492.49s total. Fresh elastic residuals stay below
7.74e-10; all pad-law, wrench and omitted-boundary checks pass. Sixth-to-tenth
order tangent/membrane/bending changes stay below0.0181% across0.1/0.3/1N.
The independent same-mesh condensed-QP comparison (`727915`) agrees in complete
fields within1.19e-10 and raw peaks within1.42e-10. This validates the fast
path with the CHOLMOD factor at the larger mesh, separately from spatial
convergence. The next pad reference pair (`727916`) reuses the already measured
125um/two-layer and250um/three-layer geometries, so surface and wall effects
can be checked independently for all three loads without another dense QP.

The250um continuous Shell arm (`727910_1`) fails at step40 before carrying
normal load: the elastic PCG reaches its8000-product cap, despite small
observed force residuals. Its rejected full state is retained, and this
history is not admitted. The same-tolerance0.5ms400-step control (`727914`)
tests whether reducing the mass/stiffness conditioning burden resolves the
cap; no solver-failure flag is waived. The500um full load cycle remains live.

## Refined pad fields and captured linear failure

The500um Shell cycle (`727910_0`) completes1000 steps and all five equilibrium
windows at0.1/0.3/1/0.3/0.1N pass. The250um half-step control (`727914`)
instead reaches the same8000-product cap at step76 (38ms); timestep reduction
alone does not repair this configuration. Neither failed trajectory is admitted.

The250um/two-layer P2 pad reference (`727904_1`) completes all nine load/order
cases, with860.21s factorization and965.34s total runtime. Sixth-to-tenth-order
complete tangent changes are0.003398%/0.001514%/0.001121% at0.1/0.3/1N.
However,500-to250um refinement (`727919_0`) changes the complete tangent field
by2.74744%/2.49180%/2.24491%, exceeding the2% target at every load. Membrane
changes are1.90867%/1.76231%/1.58020%; bending changes are2.39301%/1.96504%/
1.69874%. Raw Mises/tensile changes stay below2.529%. The125um/two-layer and
250um/three-layer controls (`727916`) remain necessary; quadrature convergence
is not spatial convergence.

The isolated first-linear capture uses the simulator's real substep index,
not a Python call counter that includes build-time warmup. The initial capture
attempt (`727918`) hit a compiled graph-layout mismatch; the second (`727922`)
captured an already overwritten cache. Both are retained as invalid diagnostic
attempts. The corrected step40 capture (`727926`) retains8000 iterations and
the failure flag, with NPZ SHA256
`134d58c15cca05532d06a6379f149f0709556f78657d08dc35f4c92d75f8a75a`.
Its diagnostic source is separate from the continuous trajectory engine.

Independent sparse assembly/direct solution (`727928_0`) of the50215-unknown
system gives relative residual1.10e-12 and symmetry defect8.51e-17. Captured
PCG's true and recursive relative Euclidean residuals are both0.0546659,
with relative disagreement1.78e-7. Its actual mass-weighted RMS velocity
error is1.7883e-10m/s, exceeding the unchanged1e-10m/s target. This is a real
unconverged linear iterate, not merely a recursive-residual reporting artifact.
The generalized full point-Jacobian auditor also reproduces the earlier
floating-sum frozen-system control (`727928_1`). A bounded200-step pair
(`727949`) tests the existing12-patch coarse preconditioner on250/500um meshes
without changing tolerances, caps, physics or acceptance gates.

## Sustained-friction regression

The [pad declaration](COMPLIANT_PAD_DECLARATION.md) now fixes the sustained
friction extension before evaluating its stress fields: mu0.3, slip smoothing
speed1e-4m/s and a1m/s2 horizontal body acceleration after normal equilibrium.
The free egg has no pinned vertices. The physical-pad branch previously
widened smoothing as2*mu*compression/dt, which changed its physical creep law
under timestep refinement. The isolated friction variant retains fixed slip
speed for physical pads and leaves original contact's widening unchanged.

The analytical free-sheet test checks compression, terminal creep velocity,
normal/tangential forces and action/reaction at1/2ms with unbatched/two-env
scenes. The original engine fails both1ms sustained-shear negative controls
(`727924_1`): velocity is about0.00216m/s instead of1.835034e-5m/s. The repaired
engine (`727924_0`) passes10 cases but fails all four shear cases on the
unloaded y-force gate1e-10N, with discrepancies1.16e-10 to1.01e-9N. It is not
reported as passing. Since the fixed friction slope converts velocity error
into force, `727947` tightens only this analytic shear test's absolute velocity
tolerance from1e-10 to1e-12m/s, retaining every force assertion and the original
relative PCG tolerance. Both positive and old-engine negative tests rerun.
Benchmark trajectory tolerances remain unchanged; full friction admission,
native pad histories, qualified rankings and matched costs remain open.

The tightened analytic control (`727947_0`) now completes all14 selected tests
in232.08s, including the unchanged momentum and state-restore regressions.
Both old-engine negative controls (`727947_1`) still fail on the analytic
terminal creep velocity. The friction engine is pinned at `4ce5542f9` and its
incremental patch is retained. Actual400-step1ms and800-step0.5ms free-egg
normal-then-horizontal-load histories (`727954`) use the unchanged benchmark
PCG1e-4/absolute-velocity1e-10 tolerances. Their observer independently checks
the lagged friction law, current cone, nodal transpose transfer, body-force
balance and external moment; submission is not qualification.

The500um12-patch control (`727949_1`) completes200 steps and passes its final
51-sample equilibrium window. Maximum contact-relative equation residual is
9.17e-9, plate speed2.06e-11m/s and kinetic energy5.20e-20J. The250um arm also
completes200 steps and passes all51 final-window samples: maximum contact-relative
equation residual3.86e-8, plate speed9.65e-12m/s and kinetic energy7.29e-21J,
without solver flags. Its observed trajectory runtime is166.93s including
observation/IO, versus55.83s for500um; these are diagnostic timings, not matched
warm costs. A complete-field comparison (`727958`) checks500um preconditioner
invariance and250um Shell versus500um/solid fields. The full250um five-load
cycle and same-tolerance half-step control (`727956`) remain required beyond
this first admitted hold.
Selected unchanged receipts and the independent linear audit are in the
[refinement/friction packet](../evidence/20261019-qualified-ranking/pad-refinement-friction-v1/README.md).

## Sustained shear and preconditioner invariance

Both actual friction histories (`727954`) finish400/800 steps at1/0.5ms.
All normal and sustained-shear windows pass, with zero solver flags and
51/101 samples per window. At1ms the final tangential resultant is
-0.007480995025N, balancing the0.007480995019N horizontal body force;
both normal reactions remain0.1N. The independently reconstructed lagged-law
error is1.68e-14, current Coulomb-cone excess is zero, and maximum window
contact-relative equation residual is1.60e-8. At0.5ms the corresponding
residual is8.64e-9. These are mechanical admissions, not stress-error rankings.

The complete same-mesh timestep comparison (`727960`) changes tangent,
membrane and bending fields by2.75e-7/2.10e-7/3.44e-7 relatively, with raw
Mises/tension changes4.06e-8/1.14e-8; the unchanged solver/timestep gates pass.
Motion decomposition (`727961`) resolves center velocity6.47e-6m/s, angular
velocity about y of-1.44e-5rad/s and nonrigid RMS velocity1.13e-11m/s in both
shear endpoints. The free egg creeps and slowly rotates; this is not a
zero-velocity static solution. A friction reference must retain that history
or quantify a justified reduction, rather than assume fixed rigid orientation.
The longer loading/unloading repeat and500um friction arm (`727962`) check
repeatability, return fields and spatial sensitivity without pinning the egg.

The500um preconditioner comparison (`727958_0`) passes: changing only0 to12
patches changes tangent/membrane/bending by8.00e-8/6.12e-8/1.02e-7 and raw
peaks by at most4.69e-9. The complete250um Shell comparison (`727958_1`) is
still provisional: versus250um solid, tangent/membrane/bending differences
are25.9687%/5.8716%/41.8727%, while raw Mises/tension differ0.00952%/14.1604%.
Refining Shell500-to250um changes tangent35.0822% and bending55.6825% despite
much smaller raw-peak changes. Thus a nearly matching Mises maximum alone
would seriously misrepresent complete-field agreement. Neither this admitted
Shell configuration nor small timestep uncertainty removes the remaining
solid-reference spatial qualification, native contact-response or cost work.

The250um half-step arm (`727956_1`) now completes400 steps and passes its
50ms mechanical window; complete tangent/membrane/bending changes against
the1ms hold are3.48e-8/2.67e-8/4.46e-8, with peaks below1.85e-8. The full
five-load cycle subsequently completed all1000 steps and all five mechanical
windows (`727956_0`). Sustained friction, motion decomposition
and preconditioner invariance receipts are published in the
[continuous friction checkpoint](../evidence/20261019-qualified-ranking/pad-friction-history-v1/README.md).

## Reference refinement, repeated shear and native pad integration

Both finer independent P2 references (`727916`) completed all nine normal
cases. Surface250-to125um at two wall layers (`728064`) changes the complete
physical-volume-weighted tangential field by1.6974%/1.6071%/1.4781% at
0.1/0.3/1N; raw Mises changes are0.3961%/0.2823%/0.1812% and tension changes
0.4461%/0.2594%/0.0840%. At250um, increasing two to three wall layers (`728001`)
changes tangent fields by1.3093%/1.2555%/1.1683%, with raw peaks below0.203%.
Order6-to10 contact quadrature changes tangent fields by at most6.63e-6 at
125um/two layers and3.43e-5 at250um/three layers. These are separate uncertainty
components; passing adjacent pairs does not establish a bound on their joint
effect or the continuum limit. The660168-element125um/three-layer mesh
(`728043`, SHA256`3d580636702bf2d986f52734f9fbfbc09eed439f73d912a263235f2a24d79bf7`)
is now used by the joint reference `728080`, with unchanged material, pad law
and qualification targets.

The longer Shell sustained-friction cycle (`727962_0`) completed1200 steps
and all six windows. Repeating the initial shear state changes the complete
tangent field by2.20e-9; returning to0.1N after0.3/1N changes it by0.0361%,
bending by0.0453%, and either raw peak by less than0.0081%. The500um friction
arm also passed both windows. These quantify small history changes while
retaining free creep; they do not establish the independent friction reference.

The new research boundary condition `rigid_pad_coupling.py` solves seven
velocities in Quadrants and applies actual equal/opposite pad wrenches through
the native API. Analytical tests (`728008`, extended by `728021_0`) check
normal balance, the declared steady slip at1/2ms, and an independent asymmetric
seven-coordinate linear solve. All five controls pass; the asymmetric solution
differs by1.54e-17. The first two launches failed before physics on array-shape
and kernel-argument declarations; their logs remain retained.

The first complete native journals (`728021`) hit a result-summary key error;
the corrected runs (`728065`) then exposed a more important whole-history
defect. Their final windows passed, but the default free-body midpoint position
update did not match the pad's backward-velocity gap prediction. The maximum
normal-law error reached0.8103 during loading. Explicit pose refresh alone did
not fix it: the matched negative control (`728081_2`) retains82 failed steps
despite passing both endpoint windows. Thus these histories are not admitted.

The declared native variant now explicitly selects the existing `Euler`
semi-implicit position update and retains implicit plate damping. No engine
source or physical tolerance changes. Normal and sustained-friction controls
(`728081_0/1`) each complete400 steps with all per-step pressure, friction,
linear/angular momentum and velocity-prediction gates passing. Maximum
normal-law error is1.98e-11; in shear the maximum friction error is1.36e-8 and
velocity-prediction error1.99e-12m/s. The angular/moment and velocity checks
retain their predeclared thresholds, rather than using endpoint-only admission.

The first attempt to attach observation directly (`728073`) was rejected by
the native guard against unspecified link wrenches. That guard is preserved.
The research adapter now feeds the complete actual surface-source packet to
a separate native observer model, with no observation feedback to the motion.
Both DPR and SCR (`728084`) finish the same400-step normal-then-shear history,
pass all mechanical gates and accept both complete fields with zero errno.
The compressed sources number6 in normal loading and10 in shear, with
within-support wrench defects below6.55e-17 and elastic residual ratios below
1.21e-10. Original centroid forces are retained alongside the reduced packet;
source reduction and the independent observer's costs are explicit limitations
of this research adapter, not an extension of the default collider contract.

The five-load native cycles (`728087_0/1`) also complete1000 steps, pass every
mechanical history/window and accept all ten DPR/SCR fields. They additionally
check the frozen-point moment difference on every step. The same-law half-dt
normal/shear controls and complete-field comparisons remain separate checks.
The coarse native volume mass is0.007472791737kg versus the coarse Shell
mass0.007480995019kg; this finite-geometry/thin-shell difference is disclosed,
not hidden by rescaling density or the horizontal body load.

The common125um/two-layer reference comparison (`728090`, matched friction0
native fields corrected in `728107`) is provisional:

| Method | Complete tangent | Membrane | Bending | Raw Mises | Raw tension |
| --- | ---: | ---: | ---: | ---: | ---: |
| DPR, declared2mm |23.6521%|12.1940%|34.8693%|16.8924%|54.2730%|
| SCR, declared2mm |20.6355%|11.2977%|29.8730%|7.0888%|45.1667%|
| Shell250um, repaired pad |26.0000%|5.7665%|42.0390%|0.3865%|14.6697%|

All entries are errors at0.1N, with complete common material columns and physical
volume weights; raw peaks are unsmoothed and retain each method's sampling.
The earlier native fields in `728090` used friction0.3 while Shell used0.
That difference matters for SCR's admissible reconstructed tractions even at
zero applied shear: its earlier tangent/membrane/bending errors were
23.5095%/12.3258%/34.5306%, with Mises/tension12.2664%/51.5813%. The table now
uses the actual friction0 cycle fields for both native methods.
Reference uncertainty and the source-reduction sensitivity below still prevent
a final resolved ordering. The low Shell Mises difference does not imply a low
complete-field error. Final qualification still requires joint reference
uncertainty, the independent sustained-friction comparison, recovery-radius
controls, matched costs/batching and publication figures. The goal remains open.

Both half-dt native arms (`728087_2/3`) finish800 steps with all mechanical gates
passing. The source-adapter audit (`728094`) verifies unchanged motion with
and without observers, complete-source wrench preservation below9.70e-16,
and endpoint frozen-point moment error below2.88e-8 relative to load times
0.03m. SCR reconstructed-surface wrench differences are below7.36e-10.

Complete fields expose a separate adapter defect (`728095`). Normal timestep
changes are below2.51e-11 and load-cycle return changes below1.06e-9, but the
shear tangent field changes2.0498%/2.0271% for DPR/SCR and bending2.8932%/2.8551%.
Both fail the unchanged0.5% field target. Source diagnostics (`728098`) show
actual full centroid forces change only5.06e-7, while the wrench-only LP
switches one selected support point and changes its sparse load field17.33%.
Conserving six wrench components alone therefore does not give a continuous
stress observation of this already known distributed load.

The directory named `repeat` in `728095` was intended as a repeat check, but its
configuration changes friction0.3 to0. Its7.07095% SCR tangent change is a
friction-parameter sensitivity, not nondeterminism or a same-configuration
repeat failure. Identical source positions and force differences2.85e-15 in
`728098` support that distinction. Preserve the original output/name and audit
the actual task arguments; do not use that pair to qualify repeatability.

The new explicit `--source-reduction none` control (`728100`) supplies every
nonzero actual centroid force through an observer-owned native contact buffer.
It keeps physical motion,2mm radius, friction, solver tolerances and acceptance
unchanged. The previous LP version remains a documented failed shear adapter.
Full-source histories and their field sensitivity must pass before this new
variant enters any ranking or cost comparison; submission is not qualification.

Selected completed receipts, negative controls and executed adapter sources are
published in the [native pad checkpoint](../evidence/20261019-qualified-ranking/native-pad-history-v1/README.md).

## Full source fields and the joint-reference index limit

The full-source histories (`728100`) finish400/800 steps for both methods at
1/0.5ms. Every mechanical history/window passes, and all eight complete fields
are accepted with zero errno. Each observation retains166 actual nonzero
centroid forces in an8992-slot observer-owned input buffer. No physical source
is dropped or replaced by an LP representative.

The matching complete-field comparisons (`728122`) pass. In sustained shear,
DPR tangent/membrane/bending changes are2.4593e-7/1.9673e-7/3.1343e-7; SCR
changes are2.4519e-7/1.9688e-7/3.1189e-7. All raw-peak changes are below4.81e-8.
Normal complete-field changes are below2.43e-11. This isolates the earlier
2% shear sensitivity to sparse source representation, without changing PCG,
Newton, timestep targets or the physical friction law. The full-source normal
five-load cycles (`728127_0/1`) also pass1000 steps and all five windows;
identical-configuration repeats, their field checks and the matched three-load
normal comparisons remain distinct from these completed mechanical controls.

The first joint125um/three-layer reference (`728080`) failed after20m35s with
`CHOLMOD error: problem too large`, then `invalid xtype` and a segmentation
fault. Slurm reports failed exit11:0 and maximum step RSS67,400,120KiB, not an
OOM exit. It produced no usable reference field. The installed Cholespy2.2.0
binary exports the32-bit CHOLMOD entrypoints. Its
[pinned source](https://raw.githubusercontent.com/rgl-epfl/cholespy/v2.2.0/src/cholesky_solver.cpp)
uses integer-index factorization and only checks the not-positive-definite
failure before proceeding. This supports an index-capacity failure; it is
not evidence of failed continuum convergence or inadequate Newton tolerance.

The read-only compute-environment probe (`728125`) finds an existing
CVXOPT1.3.2 installation. Its
[pinned CHOLMOD binding](https://raw.githubusercontent.com/cvxopt/cvxopt/1.3.2/src/C/cholmod.c)
selects the long-index interface on this64-bit platform. The new optional
`--cvxopt-factor` uses that existing dependency, the supplied through-wall
mesh ordering, supernodal factorization and no diagonal shift. It restores
process-local CHOLMOD options after factorization; no active environment,
system package, physical parameter or old default factor is changed.

Independent controls (`728133`) verify8-byte indices, vector and three-column
solutions against SuperLU (relative differences below9.95e-14), residuals
below4.17e-12, and rejection of an indefinite matrix. The complete coarse
physical reference at contact orders0/6/10 passes all equation gates. Against
the original factor, tangent/membrane/bending changes are below5.86e-13 and
raw peaks below6.34e-13. Only after these controls, joint refinement `728139`
is submitted with this adapter and192GB, including joint wall/surface field
comparisons. Its completion and spatial qualification remain open.

The completed full-source and factor controls, including the failed joint
factor log, are retained in the
[full-source/factor packet](../evidence/20261019-qualified-ranking/full-source-factor-control-v1/README.md).


## Full-source repeatability and predeclared radius controls

The identical-configuration full-source repeats and their complete-field
checks (`728127_2/3`, `728141`) are complete. Normal/shear tangent,
membrane and bending differences are below1.71e-11, with raw-peak differences
below2.88e-12. The friction0 five-load cycles return to their initial0.1N
fields within4.56e-11, including raw peaks. The separate friction-parameter
contrast changes SCR tangent/membrane/bending by0.5643%/0.5830%/0.5229%;
DPR changes remain below6.56e-12. This latter pair changes the admissible
reconstruction friction cone and is not a same-configuration repeat.

Before executing the next sensitivity controls, retain the previously declared
2mm recovery radius as the primary configuration. Test fixed1.5mm and2.5mm
radii, a symmetric25% perturbation, for both methods on the same full-source
motion: normal friction0 loads0.1/0.3/1N and friction0.3 normal/shear0.1N.
Keep all physical parameters, source points, solver tolerances, mesh and output
sampling unchanged. Report parameter dependence separately from numerical
uncertainty; do not select a radius by minimizing reference-field errors.
The controls must retain complete fields and pass the existing history,
window and native recovery gates before their errors or costs are interpreted.


## Full-source normal contact comparison (joint uncertainty pending)

The three-load comparisons (`728135`) use actual friction0 full-source
native fields and the admitted250um Shell cycle against the125um/two-wall
P2 reference at contact quadrature10. The complete adapter audit passes.
All table entries below are percentages; complete-field metrics use the
common physical-volume columns, with no clipped peaks or excluded polar region.

| Method | Load N | Tangent L2 | Membrane L2 | Bending L2 | Raw Mises | Raw tension |
|---|---:|---:|---:|---:|---:|---:|
| DPR | 0.1 | 7.9897 | 4.6994 | 10.2963 | 3.0211 | 5.4545 |
| DPR | 0.3 | 7.9969 | 4.5452 | 10.8913 | 0.4415 | 0.6298 |
| DPR | 1.0 | 7.4748 | 4.3289 | 10.5086 | 0.8259 | 1.8506 |
| SCR | 0.1 | 8.0324 | 4.8269 | 10.2733 | 1.3956 | 4.9187 |
| SCR | 0.3 | 7.9738 | 4.6859 | 10.7114 | 1.1397 | 0.0053 |
| SCR | 1.0 | 7.4901 | 4.5898 | 10.2807 | 2.2923 | 2.5147 |
| SHELL | 0.1 | 26.0000 | 5.7665 | 42.0390 | 0.3865 | 14.6697 |
| SHELL | 0.3 | 25.2183 | 5.6565 | 41.9102 | 0.2682 | 14.7863 |
| SHELL | 1.0 | 24.0015 | 5.4697 | 41.7246 | 0.3441 | 15.4841 |

These are provisional errors, not a qualified universal order. They replace
the sparse-adapter values only for this explicitly full-source variant. The
DPR/SCR differences are small relative to the unresolved joint spatial
uncertainty, whereas Shell has substantially larger complete tangent/bending
errors despite small raw Mises differences. Retain metric-specific comparisons
and ties where uncertainty prevents separation. Independent friction reference,
radius sensitivity, matched cost/batching and figures remain outstanding.


## Radius dependence and an independent transient reference

All eight fixed-radius histories (`728155`) complete:600 steps for each
normal three-load arm and400 for each friction arm. Every existing history,
window and native field acceptance gate passes. Their source-adapter audits
(`728160`) pass. Relative to the primary2mm radius, the predeclared1.5/2.5mm
controls change complete tangent fields by2.89--4.21%, membrane by1.68--2.22%,
bending by4.50--6.44%, raw Mises by2.59--4.87%, and raw tension by3.61--6.62%.
These are reconstruction-parameter effects, not solver error or nondeterminism.
The primary radius stays2mm. They reinforce retaining DPR/SCR ties rather than
claiming their small primary-field difference is a robust method advantage.

The new `plate_solid_pad` reference advances the full consistent P2 mass and
elastic operator, including all six free rigid coordinates, with the same
physical pad and lagged smooth friction. No static gauge is fixed in dynamics.
`condensed_pad_dynamics` solves in Cholesky contact-compliance coordinates.
Seven independent point-mass and coupled-normal controls (`728147`) pass,
with velocity differences below2.17e-19. The first full launch (`728152`)
failed while serializing a NumPy integer in its start receipt; no physics
result was produced. Converting the DOF count to a Python integer permits
`728154` to finish400 steps and both51-sample windows. All force-equation,
pressure/friction, velocity-prediction and omitted-boundary checks pass.
The normal field agrees with the independent static reference within5.86e-8
in complete tangent/membrane/bending metrics and1.46e-8 in raw peaks.

A separate current-position moment audit (`728166`) nevertheless rejects the
linear model's shear endpoint: normalized moment5.2081e-6 exceeds the unchanged
1e-7 mechanical target, although material-position moment is3.75e-10. The
normal endpoint passes. The residual is reproduced by the linear elastic
internal force's moment at the deformed positions. This is finite-rotation
inconsistency, not contact-source loss or failed force balance. These linear
fields remain diagnostic and cannot qualify the independent friction reference.

For high-order contact integration, a matrix-free alternative avoids forming
or inverting the possibly rank-deficient contact compliance. Its first
full-body velocity formulation passes24 manufactured controls (`728159`) but
fails line search at first contact after33 returned steps (`728165`); retain
this failed source/log. The replacement uses contact impulses and the SPD
operator I+sqrt(H) J A^-1 J^T sqrt(H), allowing redundant quadrature rows and
avoiding subtraction through the stiff full-body operator. All24 controls
(`728167`), including redundant rows and dense-solver agreement, pass within
6.40e-14 velocity error. The complete-trajectory comparison (`728169`) is
separate and must pass before use in spatial refinement.

The explicitly labeled StVK option retains E=10GPa, nu=0.3 and the same free
mass/contact model, and converges total-Lagrangian elastic-force corrections
inside each time step. It uses Green strain, second Piola stress and its
Cauchy push-forward; see the standard
[finite-strain formulation](https://getfem.org/userdoc/model_nonlinear_elasticity.html).
This is an objective small-strain material extension, not a large-strain
qualification or an unchanged-linear-model claim. Its controls (`728168`,
`728171` analytic task) check rigid rotation, covariance, energy gradient,
current-position moment and the linear small-strain limit. Covariance errors
are below2.79e-12, energy-gradient error1.47e-12, and small-strain force
contrast1.41e-8. Common-column sampling evaluates the nonlinear Cauchy stress
from the exact affine P2 displacement gradient, rather than interpolating
nonlinear corner stress. An independent variable-shear column control passes
within6.46e-16. Depth quadrature2/4/8 is explicit for the required integration
sensitivity. Full nonlinear history, moment, timestep and spatial qualification
remain pending; no reference tolerance or mechanical gate is relaxed.


The completed repeat/radius/normal comparison receipts, failed transient
reference controls and executed objective/impulse solver sources are retained
in the [transient/radius checkpoint](../evidence/20261019-qualified-ranking/transient-radius-controls-v1/README.md).
The existing affine P2/column regression (`728177`) passes after adding nonlinear
sampling; two- versus four-point depth integration changes the complete native
DPR/SCR contrast by at most3.5e-18 absolute. New nonlinear trajectories still
require their own measured depth-integration and physical-momentum checks.


## Prescribed-method completion, wall reference and production-priority launch

Jobs728893/728918/728933 complete the [prescribed-C2 method audit](PRESCRIBED_METHOD_RANKING_20261019.md):27 absolute/relative errors,15 signed tensor splits,12 independent complete resultant-array checks and62 rehashed receipts. Conditional ranks depend on the metric; this static prescribed-load configuration is not joined to full-source pad costs.

Job728464 completes400 steps and independent momentum admission;728950 completes q9/q25 two-to-three-wall comparisons. The [250um wall report](FRICTION_WALL250_20261019.md) retains the full-history maxima and pending125um spatial/time/contact controls728589/728930.728945's0.1/1N static replay attempts fail a zero-resultant assertion and remain unresolved; the0.3N pair completed but is not yet a qualified decomposition.

Remote priority commit19af6c201 is incorporated. [Production pilot728966](PRODUCTION_THROUGHPUT_PILOT_20261019.md) runs fixed-horizon paired rigid/DPR/SCR max/full at B1/B8/B32 without waiting for fine FEM. The first728958 submission was cancelled for an empty source inventory during incomplete worktree checkout. A fail-closed submitter guard and independently rehashed858-file source precede the replacement. No active result is frozen as a completed throughput measurement. The full goal and new module/production obligations remain open.


## Separate scene/module measurements and strict motion diagnosis

The [production/module report](NATIVE_PRODUCTION_PERFORMANCE_20261019.md) records terminal15-case728966 timings and passing receipt audit728986, complete15-case conformance728991 (32/259 checks fail, all egg-motion vs rigid at B8/B32), admitted changing-history B8 module replay728981/728990, and separate profiles728977/728989. Raw rates remain conditional on unresolved motion admission and are never joined to2mm full-source pad errors. Max/full outputs, field reductions, runtime counters and matching-condition batch outputs pass.

Fresh controls728988 and independent comparison728996 show bitwise rigid repeatability; manual/unrequested stress and fused-post-off trajectories exactly match original DPR, including its difference from rigid. Recovery execution is not necessary to reproduce it. Controller-only test728998 also completes: all1200 joint/phase/grip/limit outputs are bitwise identical with stress emission off/on. Construction/constraint arithmetic remains to be localized; no tolerance is relaxed.

The original SCR profile used an obsolete net-force normalization and failed. Only prepare_recovery_snapshot is repaired: independently computed source-force/moment-arm scales match the production scales, preserving the1e-8 gate. Its corrected relative force error5.7256e-9 passes while the obsolete5.3393e-8 ratio reproduces rejection; all other benchmark AST is unchanged. SCR contact processing measures9.7075ms at the profiled B8 snapshot versus0.1854ms for its linear solve. No engine speedup is claimed from observer repair. Fine reference work continues independently.


## MPR contact consistency and matched Shell controls

The [native contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) records the346-array construction comparison, first upstream contact difference at step109,384-portal witness-layout controls, failed reciprocal candidate and minimal persistent-portal-point repair. Isolated engine1ff39b077 is cherry-picked as58bbcdd5b on the task branch; the analysis engine stays unchanged for live references. Combined audit729201 passes349 B1/B8/B32 checks, including complete motion, max/full, matched batches and original stress baselines. Focused job729106 passes three MPR/GJK/lifecycle cases. Original32 strict failures remain retained.

The [Shell load/response report](SHELL_PAD_CONTROL_SEPARATION_20261019.md) records six admitted250um own-pad static/replay controls with explicit tiny mass-projection corrections, nine independent rigid-mass controls and eight external-volume/tensor controls. Audit729114 passes151 checks,56 rehashed receipts and complete signed tensor/resultant decompositions. The same-source250um solid control is separate from the qualified125um normal reference.

B128/B512 production timings729137 and receipt audit729202 complete all ten modes with zero runtime invalid counters, using repeated frozen eight-condition inputs and repaired engine1ff39b077. These rates still need separate larger complete conformance and total-process memory; canonical B1/B8/B32 timing729194 and larger conformance729223 remain separately tracked. Diverse independent condition-bank generation729216 completes, but its one-step construction is not a complete stability or timing result. No collider error inherits the2mm full-centroid physical-pad error.

## Fine-reference terminal observations remain unqualified

The125um two-wall trajectory728589_0 completes400 steps. Its endpoint current-moment and source-transfer checks pass, but whole-history friction-law admission fails at step38 (0.038s): relative error4.7494312e-7 exceeds the unchanged1e-7 gate. Both settled51-sample endpoint windows pass. The moment aggregate remains rejected because it requires admitted whole history; this is not an endpoint moment defect. All transient forces and the failure are retained, without force clipping, changed normalization or tolerance relaxation. No spatial qualification is inferred from these fields. The125um three-wall parent728589_1 remains separately running.

Fine125um contact-order and half-timestep controls728930_0/_1 are PREEMPTED, with cancelled worker steps. Their partial histories are incomplete and provide no passing time/contact comparison. Original output remains in place; no live output or environment is migrated. Fine friction qualification and final ranking/figures remain open.

Canonical repaired-source timing729194 now completes all15 cases; independent
receipt audit729236 is submitted. Larger complete conformance729223 and
separate fresh-process total GPU memory729228 are in progress. Memory sampling
starts before simulator import; its nested benchmark rates are diagnostic.
These new pending tasks do not change any previously frozen evidence.

Audit729236 completes with all15 canonical repaired-source timing receipts
passed. The [canonical timing packet](../evidence/20261019-qualified-ranking/native-canonical-production-v1/README.md) preserves92 verified payloads; the
[contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) now includes its
source-matched B1/B8/B32 rates. Larger conformance/memory remains separate.


## Canonical physical-force evaluation repairs fine-reference admission

The old125um two/three-wall histories both fail the original friction-law gate at step38:4.7494e-7 and3.5113e-7. The contact solver returns force evaluated at an auxiliary modal velocity, while the driver audits the actual full-DOF velocity. This near-detachment mismatch can violate a relative force gate despite a small momentum residual. No outer or inner tolerance needs to change to evaluate the physical law at the actual returned state.

For full velocity V=V_free+A^-1 J^T P_aux, the canonical physical impulse is P_phys=-grad E_contact(J V). Its actual momentum defect is J^T(P_aux-P_phys). The existing A^-1-weighted defect check already bounds this under the original tolerance. The corrected research-only modal solver returns V and P_phys after that check, records the auxiliary defect and response impulse, and retains all normal/friction/source/material parameters. It computes J@(V_free+response) directly, using the same full-DOF evaluation as the driver.

The original detachment fixture729250 rejects the inconsistent force. Its12 cases now pass, alongside24 contact cases. The response-reuse test keeps its1e-14 algebraic assertion for the actual auxiliary RHS and separately recomputes the canonical physical momentum under the original budget. The first test adaptation729252 exposed the old exact-physical-force identity assumption (1.75e-14 difference);729255's divide/multiply exact-equality assertion and729263's CPU-only NVIDIA metadata requirement also fail and remain retained. The corrected729265 completes all36 cases without relaxing a scientific gate. Maximum independent friction discrepancy falls from0.00852253 to2.05960e-16.

The unchanged400-step coarse physical configuration729267 passes every-step equations, pressure/friction, boundary and independent current-moment admission. Both normal/friction q9/depth4 old/new field comparisons pass the predeclared1e-8 target: maximum five-field contrast5.6152e-10 and raw-peak change2.2767e-10. All journal entries use the canonical full velocity; maximum friction-law residual is1.61325e-16. Aggregate729344 passes28 checks and rehashes the actual inputs. This is a numerical/reference repair control, not fine spatial qualification.

New pinned executions729312(two/three-wall125um) and729314(exact fine q6/half-dt controls) capture the corrected source and retain every prior physical/uncertainty target. Old rejected and preempted outputs remain in place; the older live three-wall writer is untouched. Fine q9/q25/depth comparisons, final friction ranks and figures remain pending.

## Larger native conformance, sampled process memory and diverse failures

B128/B512 complete conformance729223/729241 passes159 checks. Ten separate fresh-process memory runs729228 and aggregate729260 match the primary timing configuration and source1ff39b077. B512 sampled NVIDIA process peaks are2656MiB rigid,2928/2992MiB DPR max/full and3410/3442MiB SCR max/full, with interval/command-latency limitations. These are not logical/Torch-only allocations or a proven maximum capacity.

Independently generated128/512-condition banks each pass32 complete rigid/DPR checks in729323, including distinct authored conditions. The deliberate repeated-eight-condition control fails only its diversity gate. All four diverse SCR max/full runs729247 reject during warmup. Timely after-step capture729346 preserves the actual first rejected max states at767/env9 and645/env330; periodic pre-step error checks can observe a latched failure after subsequent physical steps have rewritten the current contact flags/certificate. Original770/647 observed-error records and unsuccessful reader assumptions are retained, not repaired in place.

The independent full-wrench reader729351 verifies strict continuous polar bounds across all original source supports and every recorded failed environment. It uses exact closest-triangle geometry, longest-edge binary subdivision and the original independent1e-8 force/moment budgets. Both captured targets are genuinely incompatible with the declared friction-cone/support model. The [native report](NATIVE_CONTACT_CONSISTENCY_20261019.md) retains the signed bounds and projection budgets. These are first-failure certificates, not whole-history failure-rate estimates, physical-error qualifications or permission to clip loads.

B2048/B8192 primary runs729261 complete; separate timing audit729348, complete conformance729291 and total process memory729349 remain independently checked. Their eight-condition input bank cannot substitute for diverse environment reliability. Sparse/output-cadence delivery and matching collider-error qualification remain open under the full goal.

The [canonical-force packet](../evidence/20261019-qualified-ranking/modal-canonical-force-v1/README.md)
retains143 payloads totaling1,162,186 bytes for the complete coarse repair controls,
original failures, exact source variants and immutable new fine-task inputs.
The [batch/memory/diversity packet](../evidence/20261019-qualified-ranking/native-batch-memory-diversity-v1/README.md)
retains413 payloads totaling3,964,315 bytes. New fine outputs are excluded from
both publications. Portable certificate replay729357 passes both relocated
positive inputs and rejects the altered-byte fixture before numerical evaluation.

Larger B2048/B8192 conformance729371 and complete source-matched memory
aggregation729409 now pass159 checks/ten profiles alongside primary audit729348.
The [native contact report](NATIVE_CONTACT_CONSISTENCY_20261019.md) retains
actual rates, sampled peaks and the repeated-bank/tested-capacity limitations.
Source reconstruction from published6b851ec plus a5KB observer overlay verifies
all858 executed source files and225263082 bytes, independently of private Git
object availability. New two-wall125um execution crosses the old step38 defect;
the early prefix is a live diagnostic, not full-history/spatial admission.

The [capacity8192 packet](../evidence/20261019-qualified-ranking/native-capacity8192-v1/README.md) freezes164 compact payloads/1,291,101 bytes. Raw arrays and large primary JSON remain on data; all required source/timing/conformance/memory checks are terminal. This tested eight-condition capacity does not certify diverse SCR reliability or close pending physical-error, sparse-output and fine-friction requirements.

Publication closure also restores169 manifest-listed tiny logs (24,922 bytes) omitted by Git ignore rules. No prior payload, manifest or hash is rewritten. The [index-repair inventory and reader](../evidence/20261019-qualified-ranking/publication-index-repair-v1/README.md) verify all5,865 copied payloads directly from the published Git tree. Immutable failed-log whitespace remains intact; source/documentation formatting is checked separately.

Sparse-cadence source review corrects the prior diagnostic-clearing attribution: `set_radius` only changes radius values; later recovery/association kernels rewrite current diagnostics while the error remains latched until periodic polling. Immediate after-step captures still preserve the original767/645 failures and their independently verified continuous certificates. Existing archived bytes remain untouched.


## Explicit observation cadence and guarded fine postprocessing

Pilot729460 completes DPR/full/stride10 and SCR/full/stride50 full-horizon output controls, with210/42 requests across warmup+recording and1080/1176 exact unsampled full-field holding/reset checks. Separate pilot primary timing also completes. Paired B8/B128 thirteen-mode conformance729467 and primary timing729470 remain active; independent audit729473 waits with afterany so failed observations remain visible. The [cadence report](NATIVE_OBSERVATION_CADENCE_20261019.md) declares freshness, motion/field budgets, timing scope and retained failure limits. No sparse speedup is admitted before that audit.

The old125um rejected reference nevertheless has a successful process/scheduler receipt: the momentum reader writes passed=false without raising. New explicit parent-admission control729476 passes all ten checks on the admitted coarse parent and intentionally fails the rejected legacy parent. Pending unguarded field jobs729474/729475 were cancelled before any worker ran. A six-parent fail-closed preflight729478 now precedes replacement spatial/time/contact field tasks; dependency success alone never admits an unqualified physical history. Active reference sources and outputs remain untouched.

The [pilot](../evidence/20261019-qualified-ranking/native-observation-cadence-pilot-v1/README.md) and [parent-gate](../evidence/20261019-qualified-ranking/fine-parent-admission-guard-v1/README.md) packets preserve150 compact payloads/893,146 bytes, exact worker variants, terminal positive/negative controls and immutable queued plans. Replacement spatial729479/time-contact729480 wait for the six-parent gate729478; all new fine and paired-cadence live outputs remain outside these frozen packets.


## Sparse-output comparison completes

Conformance729467 and primary729470 complete all26 cases each; independent729473 passes521 checks and rehashes858 source files. All stress-mode motion matches rigid bitwise. Maximum sampled peak/full-tensor changes against every-step are0.000378Pa/0.000246Pa, within predeclared budgets. The [cadence report](NATIVE_OBSERVATION_CADENCE_20261019.md) gives source-matched B8/B128 rates, full/max output, GPU identities and sampling limits. At B128, DPR max adds38.2/5.0/1.9% scene latency and SCR max309.7/34.4/7.8% for strides1/10/50, without policy or host field readback. Diverse and physical every-step failures remain retained; collider errors stay pending.

Spatial comparisons only consume the two q10 fine histories and two250um baselines, so their four-parent preflight729487 waits for729312 independently of the slower q6/half-dt controls. The pending spatial job729479 dependency is updated to729487 with a before/after scheduler journal; its captured worker/source and original archived submission are unchanged. Time/contact729480 still requires the complete six-parent gate729478. Final friction qualification still requires both tracks.

The [terminal cadence packet](../evidence/20261019-qualified-ranking/native-observation-cadence-v1/README.md) retains312 payloads/879,107 bytes and26 external trajectory identities. No live fine output is frozen or admitted; the goal remains in progress.


## Explicit complete friction-reference qualification

[Gate controls729509](FRICTION_REFERENCE_QUALIFICATION_PIPELINE_20261019.md) pass19 actual-data/negative-contract checks, retain the known2.749% surface rejection despite passing peaks, reject an incomplete parent matrix without accepted fine frames, and admit the required historical500um parent through all ten physical-history/frame checks. A new qualifier requires all seven parents, fourteen own-admitted endpoints and36 exact ordered spatial/time/contact/integration comparisons under the original targets. Its eighteen uncertainty summaries remain observed conditional envelopes, with no continuum/statistical-bound claim.

Qualification729513 waits afterany on both fine field arrays and runs its own seven-parent preflight. Only complete qualification success enables18 source-pinned DPR/SCR/Shell field tasks729518 at both0.1N normal/friction endpoints and final actual-force data extraction729519. This prepares the necessary comparisons rather than declaring them complete. Fine histories and final ranks/figures remain open.

The CPU-only plotting reader control729516 passes all five checks and retains1600 history rows/eight traction arrays on data. Only optional GPU-inventory metadata changes; scientific extraction and all force samples remain unchanged. No original reference, material/load parameter, physical gate or tolerance is modified.

The [qualification-pipeline packet](../evidence/20261019-qualified-ranking/friction-qualification-pipeline-v1/README.md) retains92 payloads/503,395 bytes for terminal controls and immutable queued inputs. Original arrays, candidate fields and live fine histories stay on data; no final qualified field or ranking is fabricated from the controls.
