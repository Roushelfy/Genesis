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
