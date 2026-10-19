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
