# Independent changing-force friction control

The 16-interval radial history candidate now has an independent changing-load
and vector-friction comparison. On the retained positive-depth point bank,
maximum combined tensor and sampled tensile/VM peak differences are
0.561% and 2.364%/0.0275%. Proportional shear reaches 6.899% and
21.435%/0.3979%. This supports retaining history for native integration;
surface/full-volume/global-peak and hollow-egg qualification remain open.

## Independent contact formulation

The reference solves unknown tangential traction under supplied circular Hertz
normal pressure. It uses a square-cell discretization of the two-dimensional
surface of three-dimensional similar elastic half-spaces. The instantaneous
normal force changes the radius as `a = Fn**(1/3)` in normalized units.
Two-component tangential forces follow a nonproportional path with reversal
and return to zero resultant. Similar materials permit the declared uncoupled
normal contact control; this assumption does not cover general dissimilar
rigid-gripper/egg response. See
[Barber, Davies and Hills (2011)](https://websites.umich.edu/~jbarber/loop.pdf).

The in-plane Cerruti compliance kernel, up to a common positive modulus factor,
is `C_ij(r) = (1-nu)*delta_ij/|r| + nu*r_i*r_j/|r|**3`.
Off-diagonal cell influence uses a 12-point-per-axis rule; the square self
integral is analytical. The assembled operator is symmetric and positive
definite. Its common scale cancels from the force-controlled traction solution.
The half-space traction framework is described by
[Taylor and Chung (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6458418/).

At each increment, cell forces minimize
`0.5 * (q-q_previous).T @ C @ (q-q_previous)` subject to their two-component
force sum and local disks `norm(q_cell) <= mu * integrated_Hertz_pressure`.
The equality multiplier gives a uniform rigid tangential increment. This
incremental energy formulation is our discretization of the elastic
stick/slip condition; its history is independent of the candidate's radial
line projection. Cell capacities use integrated pressure and a recorded small
normal-resultant correction. This is an offline reference, separate from the
production Hertz lookup and regular elastic solve.

CVXOPT solves the vector cone problem. Every retained increment passes force,
local-cone, stationarity and complementarity checks. Independent Cerruti stress
integration differentiates point-force displacement by complex step; it uses
8/16-point-per-axis cell rules with an unchanged 1e-4 change gate. The candidate
uses its independently implemented integrated-potential tensors. The original
4/8-rule pilot failure is retained; integration order is increased rather than
loosening its gate.

## Controls and reference variation

The compact study contains grids17/25 at Poisson ratios0 and0.3, with eight
increments per linear force-path segment; an additional nu0.3/grid17 path uses
four increments. Each control has two monotonic partial-slip endpoints and
six variable-normal/vector endpoints. The path has 48 increments at the finer
time setting. Every comparison uses the same 96 material points at depths
0.15,0.5,1 major radius and in-plane radii0,0.5,0.95,1.2.

The nu0 monotonic control has an exact circular Cattaneo-Mindlin comparison.
Fine-grid tensor differences are 1.617% and 0.700% for force ratios0.25/0.65.
At nu0.3, the circular collinear formula is a declared comparison rather than
an exact general vector-friction reference.

| Reference comparison | Tangent tensor change | Combined tensor change | Tensile peak change | VM peak change |
| --- | ---: | ---: | ---: | ---: |
| nu0, grid17 to25 | 21.78% | 0.3290% | 1.9113% | 0.00767% |
| nu0.3, grid17 to25 | 22.35% | 0.3113% | 1.6243% | 0.00961% |
| nu0.3, four to eight increments | 3.254% | 0.1454% | 0.2559% | 0.00681% |

All combined-field/peak changes meet the predeclared 2% change target.
The additional strict 2% tangent-field target fails, mainly in weak residual
fields at zero final resultant. Observed changes are retained variation
indicators, rather than rigorous continuum uncertainty bounds. Numerical cone
feasibility and common-bank agreement cannot certify unsampled surface stress.

## History versus proportional shear

The fine-grid comparison evaluates the original cone-preserving16-interval
history and `q=(Ft/Fn)*p`, using identical normal fields and actual supplied
resultants. Maxima below span both Poisson ratios and all retained endpoints.

| Candidate | Tangent tensor difference | Combined tensor difference | Tensile peak difference | VM peak difference |
| --- | ---: | ---: | ---: | ---: |
| Radial history16 | 29.49% | 0.5610% | 2.3636% | 0.02752% |
| Proportional shear | 100.0% | 6.8993% | 21.4349% | 0.3979% |

The history candidate meets the preexisting 5% combined-tensor / 10% peak
engineering budgets on every sampled bank. Proportional shear fails them.
The stricter additional 5% tangent-only target remains failed for the history
candidate. Residual stresses persist when the net tangential force returns to
zero, so a resultant-only proportional update loses relevant physical memory.
These controls justify the next native history integration and corresponding
surface/field checks; they do not close general physical friction accuracy.

The first saved-array inspector failed only when serializing a NumPy boolean.
Its actual FAILED state and computed arrays remain intact. The corrected
read-only inspection converts that public JSON value and records dependency
versions; no contact trajectory or physical threshold is changed.

## Reproduction and remaining work

The [terminal packet](../evidence/20261019-qualified-ranking/hard-native-source-bounds-and-friction-reference-v1/README.md)
retains the original pilot failure, all matched studies, source captures and
the failed/corrected inspectors. Complete contact forces, capacities, candidate
lines and independent tensors remain at hashed data paths. Use
`study_tangential_contact_reference --grid 17|25 --poisson 0|0.3
--subdivisions 4|8 --output <fresh-data-path>` and
`inspect_tangential_contact_reference --campaign <captured-campaign>
--output <fresh-data-path>` through the established Slurm/image/data environment.

Next work must connect native tangential tensors and history to current region
identity/frame transport, normal-force changes, substeps, release, pair/feature
changes and partial resets. Surface/global peaks, arbitrary elliptical/curved
and inner contacts, thin-shell response and accepted moving Franka max/full
costs remain requirements of the complete active goal.
