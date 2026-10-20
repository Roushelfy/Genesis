# Measured load sensitivity and resolved-boundary reference errors

The [engineering-budget clarification](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md#error-judgments-and-practical-acceptance-budgets)
now has two actual tests. The resolved-boundary native variant has18 admitted
physical error measurements against the existing qualified normal/friction
reference. Separately, the two retained SCR first failures have sampled
nearest-load and complete elastic correction measurements. These results do
not justify a relaxed production tolerance.

## Resolved pressure against the qualified reference

The native variant uses the measured external-pad samples with conservative
P2 work transfer on their actual boundary. It removes the second finite-radius
spread and the previously measured approximately42% inner-boundary normal
loading. Its own400-step history and both complete endpoints pass admission.
The original physical source positions/forces agree with the fixed-radius run
within the declared1e-9 relative check; original fields remain unchanged.

The same fine StVK reference, observed resolution envelopes, complete
physical-volume columns, outer/inner fields and raw/common peak measures are
reused. Sampling changes from9 to25 surface points per face pass the existing
measurement targets. The18 new dimensional errors include absolute values and
conditional intervals; they are not certified continuum bounds.

| Friction endpoint metric | Fixed2mm DPR | Resolved boundary |
| --- | ---: | ---: |
| Complete tangential tensor, volume L2 | 7.936% | 10.564% |
| Membrane, area L2 | 4.682% | 6.319% |
| Bending, area L2 | 10.307% | 14.077% |
| Outer tangential tensor, area L2 | 7.563% | 11.056% |
| Inner tangential tensor, area L2 | 9.881% | 11.128% |
| Raw von Mises maximum | 2.848% | 8.586% |
| Raw tensile maximum | 5.577% | 15.826% |
| Common-column von Mises maximum | 2.322% | 10.287% |
| Common-column tensile maximum | 5.543% | 15.792% |

The normal endpoint gives the same qualitative result. Conditional intervals
overlap for the complete tangential tensor, membrane and inner tensor. The
fixed-radius error is conditionally lower for bending, outer tensor and all
four peak measures at both endpoints. Thus the corrected boundary transfer
does not establish an accuracy improvement on this coarse mesh. Its larger
stress values are not evidence of being more physically accurate.

The direct three-field integration is a separate running computation. Its
identity is `new-reference = (new-old) + (old-reference)`, including the signed
cross product; final signed attribution is pending its complete admission.
A new-old transfer contrast can be isolated because its sources
and dynamics match. The remaining new-reference difference still combines
source quadrature, pressure response, geometry/elastic discretization and
reference effects; it is not an isolated elasticity error. Neither subtracting
the two scalar error norms nor returning to incorrect-boundary spreading
resolves those components.

## SCR: predeclared small budgets and a separate stress diagnostic

The original env9/step767 and env330/step645 inputs, strict failures and
continuous certificates are unchanged. Each source retains its original
radius, friction and boundary component. An explicit surface path through
facet centroids and edge midpoints proves every selected point is inside its
original intrinsic support. The path upper bound is conservative; these
sampled optimizations do not establish the continuous minimum correction.

The prior uses the existing compact Gaussian profile, evaluated on this
declared surface path. An unconstrained minimum weighted correction makes
that prior preserve each original source force and anchor moment. It is an
explicit diagnostic transfer: its tractions can violate the physical cones,
and it is not a previously admitted SCR stress or a physical reference.

The numerical problem then minimizes weighted distance from that prior under
actual circular Coulomb cones. Each source separately has force and moment
budgets `|delta F| <= beta |F|` and
`|delta M_anchor| <= beta R |F|`. The predeclared grid is
`beta = 0, 0.001, 0.01, 0.1`. This prevents large mutually cancelling source
changes from hiding behind a small aggregate wrench. It is a declared stronger
constraint than aggregate-wrench preservation alone.

At25 and81 points per facet, all grid cases return sampled
`PrimalInfeasible`; none is accepted. Independent sampled minimax searches
give source budgets30.780%/27.393% for env9 and18.082%/17.326% for env330,
stable to the shown digits between those samplings. These are not continuous
lower bounds or recommended budgets. In particular they do not replace the
original independently verified continuous certificates.

A separately labelled **unbudgeted nearest-prior projection** measures stress
sensitivity outside that grid. Successful SOCP solutions are independently
checked against the closed-form pointwise Euclidean cone projection. Actual
post-projection compression, friction, force/moment changes and unsigned
point-force changes are remeasured. Numerical KKT acceptance stays1e-8,
with the solver itself targeting1e-11; no solver cap or unsuccessful optimizer
is accepted. This diagnostic is never an accepted engineering observation.
At625 points/facet for env9, the1e-11 solver target failed the independent1e-7
coordinate check; tightening to1e-14 returned `AlmostSolved`, also rejected.
The replacement uses the exact pointwise projection, with independently
checked primal-cone, dual-polar and weighted-complementarity conditions.
It does not relabel either optimizer attempt as successful.

## Refined correction response

The original480-tetrahedron,810-P2-node Franka asset uses E=10GPa,
Poisson ratio0.3 and density2000kg/m3. The625-point/facet results are:

| Quantity | Env9 | Env330 |
| --- | ---: | ---: |
| Individual source force changes | 36.770%,38.541% | 32.097%,31.549% |
| Individual anchor-moment changes, scaled by R times source force | 12.523%,16.306% | 14.958%,14.932% |
| Complete3D stress correction, volume RMS | 1929.063Pa | 1656.939Pa |
| Complete3D correction / contact-only baseline | 29.813% | 33.510% |
| Membrane correction / baseline | 30.545% | 34.072% |
| Bending correction / baseline | 29.269% | 35.432% |
| Contact-only raw VM maximum, before → after | 42.836 → 48.938kPa | 39.403 → 44.615kPa |
| Contact-only raw tensile maximum, before → after | 23.304 → 25.206kPa | 16.079 → 16.578kPa |
| Maximum VM norm of correction | 12.842kPa | 8.499kPa |
| Maximum spectral norm of correction | 15.478kPa | 8.576kPa |

The full correction field changes by0.170%/0.316% of its own norm between225
and625 points/facet, or0.0508%/0.1059% of the finer contact-only baseline.
Baseline field changes are0.00271%/0.00318%. This is measured sampling
sensitivity on the same discrete elastic geometry, not a continuum bound.
The env330 contact-only tensile maximum moves from
`(0,-21.504,-0.0654)mm` to `(-16.808,-10.388,9.271)mm`;
the exact raw peak locations for every field remain in the receipts.

Exploratory625-point projection calls take0.124/0.077ms for the two analytic
env9 sources and2.122/2.228ms for the two SOCP env330 sources. These different
algorithms are not a controlled speed comparison. These call timers exclude
sampling/matrix preparation, elastic solves, host exports and process startup.
The independent elastic model setup takes0.520/0.600s. None is production
native throughput or a matched RL observation-cost measurement.

## Stress interpretation and limits

All complete3D tensors use the exact physical-volume norm of the affine P2
stress field. Actual tetrahedral orientation is retained and integration uses
absolute Jacobian determinants. P2 surface edge nodes are explicitly mapped
to the CPU shape convention, with point reproduction and original-wrench
checks. Membrane and bending resultants use exact line integration through
the actual tetrahedra and physical wall lengths. Raw von Mises/tensile maxima
and their actual sample positions are preserved without smoothing.

The captured first-failure files do not contain angular velocity or all
applied inertial inputs. Absolute baseline/candidate stresses therefore use
the explicitly declared contact-only free-link response. They are not the
actual dynamic total stress. The **correction field** nevertheless includes
the change in consistent-mass equilibrium projection, and the unchanged
inertial contribution cancels by linearity. Direct correction solves agree
with subtraction of the two full solutions and satisfy the full elastic
equations. No physical pins are introduced.

The correction's maximum von Mises norm and maximum spectral norm bound
changes of the global von Mises and tensile maxima, respectively, for any
unchanged inertial state in this same linear model. They do not bound peak
location changes, continuum/model error or a change of physical trajectory.

Early controls remain retained: the initial implementation incorrectly
assumed positive tetrahedral orientations, then exposed the differing edge
node order. Both were repaired with unchanged numerical checks. The9-point
sampling was too sparse for one source's unconstrained wrench fit and was
rejected without enlarging the support. Initial25/81-point results showed
7.55%/14.32% changes of the correction field itself, motivating225/625-point
refinement with the original budget grid unchanged.

## Consequence for implementation

The [terminal packet](../evidence/20261019-qualified-ranking/engineering-load-sensitivity-v1/README.md)
retains original failed controls, numerical rejections, captured sources,
18 admitted resolved-transfer errors and the complete local-load sensitivity
study. Full fields stay at externally hashed data-side paths.

This work measures separate numerical convergence, permitted load discrepancy
and stress response. It supports no new global default tolerance. The simple
local projection requires substantial source changes and substantial stress
changes on these conflicts. This does not rule out a different cooperative
nearby-load construction: the current small budgets constrain each source
separately, while native body-source-prior SCR fits an aggregate wrench with
source-prior control. A coupled aggregate-budget study must measure its actual
individual changes and complete stress before deciding practical acceptance.
A useful engineering approximation still needs
an admissible stress budget, source-prior control and its own matched native
history/cost evidence.

The resolved-pressure transfer should be generalized using actual boundary
and source provenance, including legitimate inner/nonconvex contacts. Sparse
collider sources still need the physical support/pressure model, including
the [measured compliance limits](PAD_COMPLIANCE_STUDY_20261020.md).
Native Quadrants integration, moving/friction validation, thickness cases,
batch/reset checks and measured incremental production costs remain open.
