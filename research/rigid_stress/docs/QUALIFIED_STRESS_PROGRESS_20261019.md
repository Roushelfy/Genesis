# Qualified stress comparison: in progress

The active goal is fork `950ed9d3ccfaa9a0192d4675e25bbdb6f45d20c8` and
[the qualified ranking specification](QUALIFIED_STRESS_RANKING_GOAL.md).
No qualified physical ranking or admitted matched cost comparison exists yet.
The preceding bounded report remains unchanged.

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
The two additional native pilot meshes prepared by `727697` remain data-side
assets only (4498/7432 outer vertices); no DPR/SCR result is inferred from them.
