# Slender sharp-edge pressure with transverse shear

An unfitted local pressure estimate now covers the two previously rejected
sharp-box contacts and their reversed transverse loads in the declared
half-space reference problem. Grid72 cell-pressure errors are3.40--5.75%,
positive-depth sampled tensor errors1.32--1.61% and sampled von-Mises peak
errors1.85--2.13%. Native Quadrants scale and pressure evaluation pass CPU/CUDA
checks in32/64-bit precision. This supplies the sharp-edge pressure component;
actual contact binding, complete curved-shell recovery and accepted full-grasp
max/full throughput remain open. The [active goal](../GOAL.md) is unchanged.

## Independent normal/shear reference

The existing normal-only rectangle compliance is augmented by the normal
displacement produced by tangential surface force. For reduced modulus
`E*=E/(1-nu^2)`, inward-positive displacement has kernels

\[
G_{zz}=\frac{1}{\pi E^* r},\qquad
G_{zi}=\frac{1-2\nu}{2(1-\nu)\pi E^*}\frac{r_i}{r^2}.
\]

These coefficients follow the isotropic entries of Eq.86 and Table4 in
[Dean (2023)](https://doi.org/10.1680/jgere.22.00063). Exact rectangle integration
of the second kernel gives a skew matrix on a uniform grid. Prescribing
`q_i=(Ft_i/Fn)*p` therefore produces a nonsymmetric normal-contact equation.
The offline active-set solver uses a general linear solve for this equation
and checks load balance, nonnegative force/gap and complete complementarity.
Its original positive-definite normal path remains the default and reproduces
the archived48/72-grid normal forces exactly.

Independent cell quadrature differs by2.43e-15 relative. A manufactured
nonnegative force/dual pair is recovered within1.95e-16 absolute. Zero self
integral, dimensional scaling, the incompressible coupling limit, skew
reciprocity and positive symmetric part pass. These checks establish the
discrete reference calculation for prescribed proportional traction; they
do not determine a stick/slip history law.

On the captured actual P2 gaps, the observed transverse/normal force ratios
are+0.646 and-0.796; along-edge ratios are+0.00804 and-0.00775. At grid72,
including shear-induced normal displacement changes pressure by18.80/27.45%
and sampled stress by0.850/1.109%. Reversed loads give the same qualitative
conclusion. Full-resultant moment discrepancies remain1.449/7.054`Fn*a` for
the observed directions. Normal/shear displacement coupling does not explain
that captured-wrench discrepancy.

## Local estimate without pressure fitting

Let`x>=0` run inward from the sharp edge and`y` run along it. The material gap
near its edge minimum is approximated by`s*x+k*y^2/2`, with`s,k>0`. Set

\[
\beta=\frac{1-2\nu}{2(1-\nu)},\qquad
\lambda=\frac12+\frac{\arctan(\beta F_{t,x}/F_n)}{\pi}.
\]

The strip singular-integral equation admits

\[
p(x,y)=C\left[\frac{w(1-y^2/b^2)-x}{x}\right]^\lambda,
\quad C=\tfrac12E^*s\sin(\pi\lambda),
\quad 0<x<w(1-y^2/b^2).
\]

This expression is derived from the stated surface kernels. Independent
principal-value quadrature verifies its strip equation within1.13e-12 for
both shear signs and zero shear. Integrating the profile gives
`Fn=2*pi*E* s lambda w b/3`. Center/end surface compatibility in the slender
limit gives the larger root of

\[
b^3=\frac{3F_n}{\pi E^* k}
\left[\log\frac{2b}{w}-\psi(1-\lambda)+\psi(2)
-\frac32-\frac{\pi\beta}{2}\frac{F_{t,x}}{F_n}\right].
\]

Here`psi` is the digamma function. The independent CPU implementation uses
SciPy digamma and a bracketed scalar root. The native implementation uses a
shifted asymptotic digamma evaluation and12 safeguarded scalar Newton steps,
with a precision-scaled residual check. It admits the declared slender regime
`w/b<0.05`. It requires no fitted support width, reference-pressure data,
matrix assembly or PCG solve at runtime.

Independent quadrature of the trial's center/end displacement difference
measures at most0.100% discrepancy from the logarithmic approximation on these four cases. This
checks the scale derivation separately from comparison with the independent
contact reference. Earlier trials that retained energy-fitted normal widths
are also preserved; the final unfitted law replaces that fitting requirement.

| Captured contact / shear | Exponent | Width w (micrometres) | Half-length b (micrometres) | Pressure L2 error | Sample tensor error | Sample VM peak error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / observed | 0.55810 | 1.1228 | 99.854 | 3.402% | 1.320% | 1.850% |
| 1 / reversed | 0.44190 | 1.4390 | 98.398 | 5.648% | 1.458% | 1.923% |
| 2 / observed | 0.42884 | 2.1401 | 92.970 | 5.752% | 1.608% | 2.129% |
| 2 / reversed | 0.57116 | 1.5757 | 94.805 | 3.783% | 1.394% | 1.976% |

The unchanged targets are15% cell-pressure L2,10% sampled tensor and15%
sampled peak. Coupled-reference48-to72 changes are0.136--0.395% for tensors
and0.074--0.199% for sampled peaks. Samples lie at depths0.25--2 times the
original Hertz major radius. The reference retains curved gaps with flat
half-space compliance, constant proportional shear and a finite half-plane.
Broader aspect ratios, significant along-edge shear, changing friction history,
other edges/corners and complete hollow-shell errors need further evidence.

## Native checks and measured component cost

Engine[`88d9e891a`](https://github.com/Roushelfy/Genesis/commit/88d9e891aa321010cbd0dd50777e978b7d84fcc0)
adds`rigid/stress/edge.py`. Each CPU/CUDA32/64-bit control covers270 parameter
combinations plus the four captured physical inputs. Maximum scale relative
differences are2.34e-13 in64-bit and6.74e-7 in32-bit. Tests also cover positive,
zero and unbounded pressure locations; unloaded/negative/NaN inputs; unsuitable
edge regimes; unchanged neighboring regions; and restoration after changes.

| Precision | 64 regions | 2,048 regions | 65,536 regions |
| --- | ---: | ---: | ---: |
| 64-bit | 0.0298ms | 0.0320ms | 0.0789ms |
| 32-bit | 0.0200ms | 0.0202ms | 0.0197ms |

These are warmed synchronized inclusive dispatch medians for resident **scale
updates** on RTX PRO6000 Blackwell Server GPUs. The precision runs use separate
GPU allocations, so their difference is not an isolated precision speedup.
Imports, initialization, source geometry, pressure/stress recovery, transfers,
friction history and scene simulation are outside this timing. No accepted
max/full scene-throughput claim follows from these component measurements.

## Sharp-edge particular stress

The prescribed pressure now has a cheap native stress evaluator, including
transverse and longitudinal proportional shear. This isolates the field
calculation from the independent contact-pressure error above. For a local
strip of width`a=w*(1-y^2/b^2)`, its complex Cauchy transform is

\[
H(\zeta)=\frac{\pi C}{\sin(\pi\lambda)}
\left[1-\left(1-\frac a\zeta\right)^\lambda\right],\qquad
\zeta=x+iz.
\]

The transform and its derivative give the exact infinite-strip Flamant
plane-strain stress; longitudinal traction adds the antiplane shear field.
Independent integration of the three-dimensional Boussinesq/Cerruti
displacement derivatives agrees within2.57e-12 relative, including a nonzero
longitudinal shear control. The prescribed surface traction also agrees.

The composite field adds the finite parabolic line load and subtracts the
infinite local line. Its line density is`3*Fn*(1-y^2/b^2)/(4*b)` and its
transverse coordinate is`(1-lambda)*w*(1-y^2/b^2)/2`, preserving the strip's
force and transverse first moment at every y. A hyperbolic-sine quadrature
resolves the line near each observation. The regular finite-length correction
uses`z>=0.001*b`; the local strip retains its actual depth. At the surface,
the known normal and shear tractions are imposed exactly.

The independent two-dimensional continuous-pressure integration uses
Gauss-Jacobi quadrature across the edge and a stretched rule along it.
Increasing64/48 to96/72 knots changes the sampled tensors by at most4.65e-12
relative on these four cases. Each case samples187 points, down to0.1 transverse
width and out to four half-lengths. These are substantially closer to the
edge than the earlier cell-pressure stress samples.

| Contact / shear | Composite32 tensor error | Maximum point error / maximum reference tensor norm | Sample VM peak error | Sample tensile peak error |
| --- | ---: | ---: | ---: | ---: |
| 1 / observed | 0.1643% | 0.3162% | 0.000158% | 0.000430% |
| 1 / reversed | 0.3162% | 0.5091% | 0.000302% | 0.03699% |
| 2 / observed | 0.5240% | 0.8475% | 0.000785% | 0.04899% |
| 2 / reversed | 0.2360% | 0.4275% | 0.000439% | 0.000928% |

At the smallest depth,0.1 width, tensor errors are0.187--0.586%. The overall
norm above weights the specified samples equally; it is not a volume-weighted
whole-shell error. Pressure uncertainty, contact-history error and the shell
correction are separate from this prescribed-pressure comparison.

Native engine[`e64780805`](https://github.com/Roushelfy/Genesis/commit/e64780805847beb0d8f902d3a05392976ce93cd8)
adds`rigid/stress/edge_fields.py`. CPU/CUDA32/64-bit controls each cover768
positive-depth and surface queries, original Hertz quadrature, invalid input,
explicit sharp-edge singularity status, neighbor isolation and restoration.
The first64-bit CPU/GPU checks and32-bit GPU check exposed loss of the exact
zero-pressure endpoint after subtracting nearly equal values. The repaired
expression preserves that zero before applying the fractional power; the same
unchanged tests then pass. Maximum pointwise differences from the independent
composite implementation are1.46e-10 in64-bit and1.65e-5 in32-bit.

| Precision | 256 queries | 8,192 queries | 65,536 queries |
| --- | ---: | ---: | ---: |
| 64-bit | 1.0397ms | 1.0407ms | 3.7916ms |
| 32-bit | 0.0240ms | 0.0250ms | 0.0380ms |

These are warmed resident **tensor-evaluation** dispatch medians on separate
RTX PRO6000 Blackwell Server allocations. They include32 quadrature knots
per nonempty half-interval. They exclude source geometry, scale estimation,
finite-shell recovery, max reduction, transfers and scene simulation. The two
allocations do not isolate precision as the sole cause of their timing gap.
This component measurement does not establish accepted scene throughput.

## Captured source-line sensitivity with the curved-shell response

The two observed edge loads now run through native all-boundary point-traction
correction, curved P2 elastic recovery and the new edge particular field.
Each source is evaluated separately in two configurations: its pressure
centroid projected onto the actual material, and the captured rigid-force
line intersecting that material. The local pressure shape is transported
without fitting its exponent. The latter is a counterfactual load-placement
approximation, not a new contact solution.

The pressure is expressed in each center's actual material tangent frame,
with normal/shear components transformed from the same captured vector force.
The transverse scale and amplitude account for the local projection. A first
adapter attempt supplied reflected frames and was refused by the existing
proper-rotation check. Correcting the adapter's orientation admits all four
supplied configurations with unchanged engine checks. At boundary orders16/24,
maximum normalized wrench residual falls from1.88e-6 to1.10e-9. The retained
combined sampled fields change by at most6.46e-13 relative between those rules.

| Contact | Center displacement | Coarse volume tensor change | Common near-probe tensor change | Sample VM peak change | Sample tensile peak change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 92.98 micrometres | 1.539% | 141.475% | 0.1687% | 0.3124% |
| 2 | 369.66 micrometres | 62.409% | 141.439% | 0.02445% | 0.7034% |

The two pressure-centroid cases retain moment discrepancies1.449/7.054`Fn*a`.
The translated cases preserve the captured resultant moment to1.27e-13 and
3.54e-14 on that scale. Despite this moment agreement and small changes in
peak magnitude, the fields at common material locations differ substantially.
Simply translating pressure to the rigid force line fails the existing5%
sample-field engineering budget and does not justify relaxing source handling.

The volume metric uses positive27-point quadrature per coarse curved element;
it does not resolve the micron-wide contact by itself. Common material probes
cover both source placements at depths0.1 width, one width and0.1 half-length.
The all-boundary point correction remains an approximation to a distributed
sharp-edge shell response. This diagnostic measures source-placement
sensitivity in that declared complete-field approximation. It does not close
the independent shell/contact accuracy requirement or actual scene admission.

A retained-displacement check also tests whether the two gripper sources'
smooth shell response can explain the along-edge displacement. It removes
mass-weighted rigid translation/rotation from the regular-plus-point
displacement and includes the other gripper source's distant particular field.
At the pressure-centroid configurations, the predicted along-edge shifts are
-0.0217/+0.0195 micrometres, compared with required shifts-64.51/-356.00
micrometres. Required gap tilts are-0.00319/-0.01719 radians, while the computed
smooth tilts are-1.08e-6/+9.40e-7 radians. This first-order, two-source check
does not explain the discrepancy; it is not a coupled contact equilibrium
calculation and does not include other contacts' displacement response.

The [field and source-response packet](../evidence/20261019-qualified-ranking/hard-edge-fields-v1/README.md)
retains continuous references, native tests and timings, original endpoint and
frame-orientation failures, curved-shell response arrays and the displacement
diagnostic. The next source study must examine collision/provenance accuracy
and physically constrained force redistribution, rather than treating a small
change in peak magnitude as sufficient evidence for moving an entire patch.

## Remote field quadrature and paired component cost

For distances at least eight times `max(w,b)`, four equal quadrature forces
replace the strip/line cancellation. Their coordinates are

```text
x = 2*(1-lambda)*w/5 +/- w*sqrt(4*(1-lambda)*(3+2*lambda)/175)
y = +/- b/sqrt(5)
```

They reproduce the pressure's first and second moments. Their purpose is remote
field evaluation; pressure support continues to use the continuous sharp-edge
law. Eleven prescribed-pressure controls include both observed contacts and
nine synthetic profiles with `w/b=0.001,0.01,0.049` and
`lambda=0.2,0.5,0.8`, including longitudinal shear. Independent separable Jacobi
integration at12x20 and24x40 agrees within1.41e-14 relative. At radii8.001,8.01,
16 and64 half-lengths, the maximum point tensor error is0.02690%, below the
predeclared0.2% target. Near-surface directions reach `z/r=1e-4`. The largest
same-point difference between the two branches near the switch is0.05018%.

The native candidate agrees with its CPU formulas within9.18e-11 in64-bit and
1.09e-5 in32-bit. It also retains the surface, invalid/singular isolation,
restoration and original Hertz-quadrature checks. The retained full-shell
coordinate bank has98.86% remote queries. A complete curved-shell rerun keeps
the regular displacement identical; the maximum coarse-volume tensor change
across four source placements is0.00209%, and the maximum point change is
0.00952%. These measure the quadrature change within the existing model.

| 64-bit resident queries | Baseline before | Remote candidate | Baseline after |
| --- | ---: | ---: | ---: |
| 256 | 0.5375ms | 0.07117ms | 0.5378ms |
| 8,192 | 1.0649ms | 0.07200ms | 1.0654ms |
| 65,536 | 2.8290ms | 1.2488ms | 2.8258ms |

All three64-bit variants run sequentially in one GPU allocation on the same
captured volume/probe inputs. At65,536 queries the measured reduction is about
2.26x; that bank contains0.92% near queries. Smaller banks contain only remote
queries. The timing includes dispatch and synchronization, and covers only
particular tensors. Geometry, prediction, boundary correction, elastic solve,
max reduction, copies and actual scenes need their own inclusive measurements.

The32-bit baseline fails the unchanged0.05% numerical point-error check, at
0.1126%. Its worst point lies445 half-lengths away with an approximately2Pa
tensor. Input rounding changes the64-bit evaluation by only1.52e-7 relative;
the composite's model error there is1.27e-6. Cancellation in the remote strip
correction explains the much larger native32-bit error. The new remote branch
passes the same numerical target. On its separate allocation,32-bit candidate
medians are0.02126/0.02067/0.04062ms for the three banks. This is an independent
component timing; the failed baseline supplies no admitted32-bit speedup ratio.

## Fixed-support constrained load allocation

The all-contact diagnostic minimizes the sum of squared relative force changes,
subject to the captured total force/moment, local compression and Coulomb cones.
The friction coefficient0.7191016 comes from the pinned fixture's material and
friction-ratio settings and the collider's maximum-friction rule. All three
captured sources are included. The offline convex allocation is a diagnostic;
it adds no iterative runtime solver.

An optimistic check that keeps the ground at its collision position and uses
collider normals requires at most2.69% force change. Using the edge pressure
centers' actual material normals increases that to4.79%. A subsequent reader
mistakenly used unassigned downstream Hertz cache fields in this refused frame;
its origin-centered ground result is retained as invalid geometry evidence.
The corrected check reconstructs the ground center, normal and curvature
directly from the actual quadratic material and computes its Hertz patch.
The existing captured ground load point is184 micrometres above this material
boundary, so this correction must also enter the wrench allocation.

With all three physical centers, maximum force change is6.22%; the individual
changes are5.21% ground,6.22% contact1 and5.33% contact2. Total scaled wrench
residual is5.18e-18, the minimum normalized cone margin is-1.56e-14 and the
stationarity residual is4.35e-10. These establish the constrained minimum for
this declared objective, without establishing a universally optimal stress
approximation.

The original and adjusted loads then use the same fixed pressure supports,
exponents and centers, with pressure amplitude and tangential traction updated.
Native curved-shell point correction and edge fields include both grippers;
the ground uses the analytical normal/tangential Hertz particular field. All
three recovered fields are superposed at common volume and near-contact points.
Boundary orders16/24 change the per-source sample fields by at most2.08e-9.

| Combined field quantity | Relative change |
| --- | ---: |
| Coarse volume tensor | 4.9935% |
| Common near-contact probe tensor | 5.6731% |
| Sample von-Mises maximum | 4.4928% |
| Sample tensile maximum | 7.9824% |

The near-contact field exceeds the predeclared5% budget. Keep this candidate
unadmitted. The sharp-edge probes retain depths0.1 width, one width and0.1
half-length; the volume rule remains coarse near the footprint. Pressure support
and exponent were held fixed, so a self-consistent updated contact estimate is
still required. The next useful discrimination concerns collision/material
geometry and pressure provenance, with measured stress response for any nearby
load approximation. No runtime source gate is relaxed.

The [remote-field and allocation packet](../evidence/20261019-qualified-ranking/hard-edge-farfield-v1/README.md)
retains sources, terminal jobs, failed32-bit control, invalid cache-reader trial,
corrected geometry, full arrays and measured limits.

## Collision application points and updated support

A raw replay of the original first refused frame106 reproduces all eight
saved input/status arrays exactly. Contact sorting, geom/pair association,
body-local transforms, forces, normals and material-side witnesses agree with
the raw collider capture. The ground application point has local
z=-0.029816409079m, while both its witness and the quadratic material surface
have z=-0.03m. Their183.590921micrometre separation is exactly half the
367.181842micrometre penetration. The plane collider constructs this midpoint,
and the constraint Jacobian uses the same application point for its moment arm.
It is a collision convention, not a ground surface discretization error.
The original two gripper midpoint offsets are93.64/110.68micrometres; their
witness-to-quadratic-surface discrepancies are11.97/11.89micrometres.

The earlier `geometry0106.npz` diagnostic queried the material after integration,
using a saved pre-integration witness. The frame positions differ by up to
0.634mm. Those query distances must not be interpreted as same-contact surface
errors. The current inspector labels these explicitly as post-step queries.
Captured contact inputs and `boundaries0106.npz` retain the actual contact-time
geometry and remain valid inputs to the pressure study.

A targeted division32 collision mesh uses the same quadratic material, frame,
mass and inertia. It contains40,962vertices and81,920triangles; maximum sampled
flat-to-quadratic position error is1.960micrometres, versus31.36micrometres for
division8. This sampled mesh error does not bound every actual MPR/hull witness
error. The refined frame has two samples on each gripper region: contacts1/2
share pair210, and3/4 share pair222. Grouping retains their summed force and
free moment before predicting pressure. The initial geometry-v1 refined
contact1/2 studies are per-source geometry diagnostics, not independent region
pressure results; geometry-v2 and subsequent responses use full grouped loads.

| Collision configuration | Maximum force adjustment | Volume tensor change | Near-probe tensor change |
| --- | ---: | ---: | ---: |
| Division8, one substep, fixed support | 6.2225% | 4.9935% | 5.6731% |
| Division32, one substep, fixed support | 2.9646% | 2.6650% | 2.6754% |
| Division32, four substeps, fixed support | 3.0807% | 2.4247% | 2.8406% |
| Division32, one substep, updated support | 2.9646% | 2.6760% | 1.7940% |

Each allocation minimizes squared relative region-force changes under the
original total wrench and actual material-normal Coulomb cones. Grouping
preserves original free moments. Four substeps reduce the default effective
constraint time constant from0.02s to0.01s and reduce penetration, but do not
improve the fixed-support near-field metric. No four-substep default is adopted.
All three actual native Franka variants still refuse at106 because edge source
binding is absent; those failed jobs are retained, not counted as stress updates.

For division32 with one substep, the updated-support study recomputes edge
width, half-length, exponent and physical center from the adjusted loads, and
updates the ground Hertz axes with the new normal load. One further constrained
allocation followed by another law update reduces the scaled moment residual
to4.97e-11; relative force residual is1.21e-16 and minimum normalized cone
margin is positive1.31e-13. Sample von-Mises/tensile maximum changes are
1.166%/1.996%. Boundary orders16/24 change per-source fields by at most2.17e-9.
The refactored source-response helper reproduces all six original fixed-support
fields and regular displacements exactly.

This is a nearby-load sensitivity result within the existing approximate
complete-shell model. It passes the declared5% volume/near-field and10% sampled
peak budgets for this captured frame; it does not independently qualify the
physical pressure or complete stress field. Near-edge probes cover both old
and updated footprints, retaining depths0.1 width, one width and0.1 half-length.
The coarse volume rule does not resolve the surface singularity. Native grouping,
pressure prediction, load allocation and full recovery still need integration,
changing-contact checks and accepted inclusive max/full scene timing.

The [contact application packet](../evidence/20261019-qualified-ranking/hard-contact-application-v1/README.md)
retains raw captures, source identities, generated-asset hashes, all three native
refusals, grouped controls, fixed-support and updated-law responses, and the
post-integration diagnostic limitation. Full arrays and collision assets stay
in the project's data directory.

## Native edge geometry and force allocation

Experimental native source `6c2e887be` adds two supplied-input components. They
are implemented in Quadrants and are not yet connected to actual scene source
binding. The original collision, region and recovery acceptance remains intact.

The box-edge minimum starts at a resolved material anchor, selects the single
violated finite-face half-plane and follows connected quadratic surface faces.
Bounded scalar Newton updates use analytical derivatives of material height in
the projected Cartesian chart. Differentiating the inverse chart retains the
surface's tangential acceleration term. Corners, failed projections and
nonpositive height curvature remain explicit failures. Runtime geometry uses
local connected walks, with no whole-mesh surface minimization.

On the four division32 collision samples, CPU/GPU64 positions agree with the
independent whole-mesh minimization within0.206nanometres. Analytical slopes
and curvatures differ from the preceding one-sided finite-difference records by
at most2.83e-5 relative; independent central differences agree within6.85e-7.
Rotation, translation, invalid anchors, interior/corner classification,
unaffected-query isolation and restoration pass. This verifies the numerical
geometry component for these cases, not a contact-pressure law.

The force allocator minimizes the same squared relative region-force objective
as the offline convex study. It uses six dual coordinates for total force and
moment, exact Euclidean Coulomb-cone projections, and a small Cholesky solve.
Original free couples enter the supplied total-wrench target. A bounded13
residual-evaluation loop returns failure on a singular dual derivative or
nonconvergence; it introduces no large pressure-field solve. Numerical
convergence alone does not accept the adjusted loads as a physical model.

Four source configurations at load scales1,0.001 and1000, with rigid rotations
and translations, converge in3--4 updates on CPU/GPU64. Native forces match the
independently certified constrained minima within3.36e-10 relative. Independent
reconstruction of total force/moment has scaled residual2.47e-13 or less, and
cone margins stay within2.52e-16 of nonnegative. Invalid input, singular geometry,
unload, environment isolation and restoration pass. The original6.22% force
correction remains visible, while refined candidates retain approximately2.96%.

| Environments, three regions each | Resident allocation median |
| --- | ---: |
| 1 | 0.1983ms |
| 64 | 0.1988ms |
| 1,024 | 0.2173ms |

These warmed GPU64 measurements include Python dispatch and synchronization.
Each uses five blocks of32 calls, verifies admitted numerical outputs and replays
the same supplied three-region configuration across environments. Grouping,
geometry, pressure/support updates, stress recovery, transfers and actual scene
steps are excluded. The table measures a component cost, not useful full-scene
throughput or a paired speedup. Native scene composition, changing-contact and
friction checks, independent complete physical accuracy, max/full handling of
the sharp-edge singularity and matched admitted scene timings remain required.

The [native geometry and allocation packet](../evidence/20261019-qualified-ranking/hard-native-edge-geometry-v1/README.md)
retains all executed sources, CPU/GPU controls, source identities, terminal jobs,
component timing blocks and hashes for complete arrays. The earlier geometry-only
source `f57245943` is preserved as an ancestor of the final experimental source.

## Native hard inputs and complete-field composition

Experimental sources `58c18847e` (input binding), `edcffa58b` (field composition)
and `8324e7d53` (cone-boundary stability) extend the preceding native components.
The native input component now runs on actual scene contact records. The main
Hertz contact acceptance and friction/recovery dispatch still require composition
with this candidate. Franka frame106 continues to fail that original path.

The input component resolves connected material geometry and groups source
samples by geom, opposing pair, pressure profile and local surface feature.
Five captured samples form the ground and two gripper regions. It sums their
forces and original moments before computing Hertz or slender-edge pressure.
Three constrained force updates alternate with four pressure/support updates.
An independent final check preserves the original total force and moment,
checks each material friction cone and enforces the existing5% force-change
budget. Numerical success here is an input certificate, with physical model
error assessed separately.

CPU/GPU64 replay retains pressure centers within0.206nanometres of the offline
material solution, forces within1.92e-8 relative and support/amplitude within
2.83e-5 relative. The final force adjustment is2.96456%. Independently
reconstructed force and moment residuals are below2.40e-13 and7.75e-11,
respectively (moment scaled by the sum of normal force times major extent).
Merged samples with their original free couples give the same result. Unload,
positive-environment isolation, selective reset and restoration pass. The live
Franka capture also accepts the new input component in environment0 at106.
Its original main-path refusal remains retained and excluded from throughput.

World-frame rotations and translations exposed a separate allocator defect.
Small changes in roundoff could switch the active friction-cone branch and
cause an undamped dual Newton step to diverge. Requiring a decreasing residual
alone also failed at this nonsmooth boundary. The repaired solver accepts a
converged/decreasing residual or an Armijo decrease of the convex dual objective.
A projection-based Bregman remainder evaluates the objective change without
subtracting large nearly equal objective values. It retains the original
iteration and residual limits, and reports a line-search failure explicitly.
All four transformed bindings now agree within the predeclared force1e-8 and
geometry/support1e-6 tolerances. The original twelve scaled/rotated constrained
minimum controls still pass on CPU/GPU64 within3.36e-10 relative. The preceding
component timing table describes the earlier allocator, before backtracking.

The complete-field component dispatches each source to its Hertz or edge
particular field, adds the same all-boundary point correction and quadratic
material elastic response, and reduces sampled VM/principal maxima. Smooth
sources retain the radial tangential representation. Edge sources use the
explicit proportional-shear approximation. The replay retains750 source slots
and four identically loaded environments, with three loaded regions each.
These are homogeneous numerical controls, rather than four independent scene
trajectories.

| CPU/GPU64 check | Largest relative difference |
| --- | ---: |
| Volume-weighted complete tensor | 4.87e-6 |
| Common near-contact probe tensors | 1.30e-5 |
| Common-sample VM maximum | 1.23e-5 |
| Common-sample tensile maximum | 8.18e-6 |

These differences compare native composition with the same approximate model
assembled offline. The predeclared tensor target was1e-3. The comparison uses
boundary quadrature order24, the retained quadratic material mesh, a3×3×3
volume rule per element, and common points spanning old and updated footprints.
It validates implementation and input conformance. It does not establish the
independent physical error of the complete hard-contact model. An initial28%
volume discrepancy came from a replay-script error: the ground tangential force
had not been rotated into the Hertz principal frame. The corrected replay checks
force reconstruction before recovery, and the failed run remains retained.

Both CPU and GPU max/full modes agree within the predeclared1e-10 relative peak
tolerance. Each sharp region adds89 material probes: seven transverse positions,
three along-edge positions, depths0.01/0.05/0.2/1 width, and five central depths
2/4/8/16/32 widths. The smallest prescribed edge depth is approximately11nm in
this fixture. Full point coordinates, element locations and tensors are retained.
The roughly4.56GPa sampled VM maximum belongs to that finite-depth probe bank;
it is neither a bounded continuum maximum nor a physically qualified failure
prediction. No radius or cutoff has been introduced to make the sharp-edge
maximum finite.

The [native hard composition packet](../evidence/20261019-qualified-ranking/hard-native-hard-composition-v1/README.md)
retains every executed source, successful and failed controls, original live
refusal, field arrays and scheduler outcomes. Runtime arrays and logs remain
under the project data directory. Main-pipeline admission, changing edge-friction
accuracy, independent complete physical errors and matched admitted scene
throughput remain active work.

## Actual sharp-edge scene integration

The actual main pipeline now consumes the grouped hard-contact candidate.
Separate material-frame friction inputs preserve raw collider forces and the
original region diagnostics. Smooth regions retain radial memory, while edge
regions use their declared proportional shear. Entering or leaving an edge
profile clears the affected radial history; independent environments retain
their histories. Geometry, allocation and friction rejection still invalidate
the observation through the native error path.

Actual frame106 initially exposed two independent numerical failures. Its
ground tangential force is0.990564974 times the Coulomb limit. The original
radial ADMM penalty0.1 reaches the2048-iteration limit with force residual
2.15e-8. The same captured staged history, with penalty1, converges in398
iterations on both CPU and GPU. The force residual is1.14e-11; the maximum
normalized line difference from the independent constrained solution is
1.26e-9. The reference's line-search termination message is unsuccessful,
but its independently checked KKT line-distance bound is2.72e-8 and its
force/cone constraints pass. This is a numerical control of the same radial
approximation, not a physical friction validation. Penalties10 and30 also
fail the unchanged iteration limit; increasing the penalty indiscriminately
is not a remedy. The production variant uses1 and retains all original force,
dual, line and continuous-cone acceptance thresholds.

The second failure is integration of the all-boundary correction:

| Captured frame106 rule | Relative wrench residual | Original1e-4 gate |
| --- | ---: | --- |
| Original complete order10 | 1.2012e-3 | Rejected |
| Near-source order16, far faces order10 | 1.2145e-4 | Rejected |
| Complete order16 | 1.5369e-6 | Accepted |
| Complete order24 | 1.2645e-10 | Accepted |

The local-only variant remains frozen as a negative control. The current
variant precomputes a minimum order16 rule for curved Hertz boundaries and
shares it by shape. It retains the original runtime wrench gate. Against the
order32 control, the accepted order16 nodal-load vector differs by4.67e-7
relative. Rejected low-order loads are cleared by production acceptance and
must not be interpreted as valid field differences. These measurements concern
quadrature, not the physical error of the point boundary correction.

Frame107 then exposed a distinct collision/material geometry discrepancy.
The collision normal root, projected onto the actual quadratic material,
leaves a normal error7.92e-4. Two local2x2 updates on the connected material
surface move its minimum25.6micrometres and reduce the normal error below
2.5e-13. Native CPU/GPU replay admits the resulting mixed contact inputs with
maximum relative regional force change3.625%, below the original5% bound,
and normalized force/moment residuals below8.6e-17/4.4e-14. The original
four rotated/translated frame106 controls remain admitted. No whole-mesh
normal search or pressure Newton-PCG solve is introduced by this refinement.

The public experimental API distinguishes the singular idealization from its
finite sampling. `get_max_stress()` returns positive infinity for an accepted
loaded ideal edge. `get_max_sampled_stress()` returns the maximum over material
corners and positive-depth probes, accumulated over observed physical substeps.
Sharp-edge probes begin at0.01 contact widths; full mode exposes their actual
positions and tensors. Failed observations are NaN. CPU/GPU checks cover
substep accumulation, a failed environment, a subsequent smooth step, getter
copy/selection behavior and selective reset. Independent48-step radial-history,
16-step edge-transition,12-step curved-history and four-case force-line controls
also pass on both backends. The original dump-cache and empty-optional-array
failures are retained beside their repairs.

Both four-environment max/full scenes now admit frames0 through106,
including the first loaded sharp-edge frame. They still reject environment0
at frame107 after its geometric input has passed. The moving frame's complete
boundary wrench residual is0.17938 at order16,0.01954 at24,0.01933 at32 and
0.00878 at48; every value exceeds the unchanged1e-4 gate. This is a separate
moving-source integration limitation. The frame106 quadrature result does not
certify it. Increasing the global quadrature order further is not an established
cheap remedy; local per-face error and near-source mapping need diagnosis.

The paired pure-rigid/max/full timing worker is prepared but has not run because
the requested100–129 window is not admitted. There is no valid continuous-window
throughput number from these trials. Actual scene outcomes are recorded in the
[native scene packet](../evidence/20261019-qualified-ranking/hard-native-hard-scene-v1/README.md).
Passing an individual sharp-contact frame does not establish a complete gripping
cycle or usable continuous-window throughput. Independent complete-field model
errors, changing edge-friction accuracy, matched accepted timing and batch
scaling remain active requirements.

## Loaded-region allocation and moving-boundary diagnosis

The input optimization keeps the existing pressure laws, original wrench target,
Coulomb cones, Newton/backtracking iterations and every numerical acceptance gate.
A single scan validates and clears every supplied slot, then records the loaded
region indices. Iterative residuals and line searches traverse those indices.
Release rebuilds the list; no persistent contact identity or friction history is
changed. CPU/GPU checks retain the12 independent scaled/rotated allocations,
invalid/singular isolation and unload/restoration. A reversed sparse embedding
across750 slots also passes. Maximum independent force discrepancy is3.36e-10.

Separate profiling identifies the actual cost: the original allocator consumes
93.05% of GPU kernel time,15.217ms per call, with three calls per hard-input
update. Loaded-index iteration reduces that kernel to0.906ms per call. These
instrumented profiles establish attribution; the following paired wall-clock
measurements disable profiling and include dispatch and final synchronization:

| Fixed captured input bank | Baseline input update | Loaded-index update | Speedup |
| --- | ---: | ---: | ---: |
| B4,750 slots/environment | 47.781ms | 5.655ms | 8.45x |
| B64,16 repetitions of the B4 bank | 50.392ms | 5.851ms | 8.61x |

Each row uses the same GPU allocation and model, with three warmed timing
intervals. All25 captured output fields match the numerical baseline at1e-9
relative/1e-12 absolute tolerances, including status, geometry, force, support
and allocation certificates. This workload has one active hard-contact
environment per four-environment bank. It excludes collision/region construction,
source resolution, elastic recovery, full fields and dynamics. It is resident
fixed-input component throughput, not independently evolving RL throughput.
Geometry, grouping and repeated target/frame work remain measurable costs.

Boundary replays now retain raw nodal loads before acceptance clears them.
Separating the three sources at frame107 localizes the16th-order failure to the
ground's smooth source: its relative wrench error is0.41394; the two gripper edge
sources give4.56e-7 and4.27e-8. Per-face replays sum back to the original raw
nodal vector within1e-10 relative/1e-13N absolute. Against order48, the largest
order16 nodal-load change is9.83e-4N on adjacent face33, followed by4.50e-5N on
source face74. At order32, those two faces still dominate. The source is58.3um
from face33 and lies near a vertex inside face74. Geometry uses the captured
contact-time material frame, not the later diagnostic world geometry query.

Equal-angle source fans and closest-curved-point fans are retained as failed
controls. Neither admits frame107 at the original order16. A hyperbolic radial
map removes the original rational map's far-end concentration; a hyperbolic edge
coordinate resolves the inverse-distance traction near a fan vertex. Applying
concentration to a numerically collapsed fan exposed a frame106 regression;
the retained failure prevents promotion based only on frame107. Such fans retain
the regular variable map, without dropping their load or changing the wrench
gate. Later replays use the actual production64-bit source epsilon, rather than
the earlier diagnostic1e-12 value. Numerical boundary convergence is separate
from independent pressure and complete-shell physical accuracy.

The resulting fixed near-face rule uses four angular subdivisions and three
radial subdivisions, with the existing complete surface order16. It changes
quadrature coordinates and weights, not the material surface, source positions,
pressure or force model. Original frame106 and moving frame107 pass source-wise
CPU/GPU replay under production source epsilon. At frame107 the ground-source
wrench residual is4.93e-9, and the two gripper sources are5.78e-7 and8.03e-9.
The combined residual is1.22e-7, below the unchanged1e-4 gate; the raw nodal
load vector differs from order32 by8.14e-10 relative. Order24/32 reach
a numerical floor around1e-8; their slight non-monotonicity does not establish
physical accuracy. The failed angle-only, closest-point, insufficient-subdivision
and collapsed-fan trials remain in the packet.

At the preceding boundary-repair checkpoint, both actual four-environment
max/full scenes admitted frames0 through109.
The former frame107 boundary failure is repaired. At frame110 environment0
fails friction history (input status16, contact status55), while all four
boundary recoveries report status0. The other three environments remain admitted.
Exact material inputs and staged friction history are captured. CPU replay
reproduces the ground failure at normalized tangential target0.9999893674593.
Every tested penalty0.1,0.3,1,3,10,30 exhausts the unchanged2048 iterations;
penalty1 leaves residual6.19e-5. A larger fixed penalty alone has not repaired
this near-saturation case.
At that checkpoint, the continuous100--129 timing window remained unadmitted,
so the prepared paired pure-rigid/max/full timing worker was unexecuted. No successful prefix was
counted as useful complete scene throughput. Independent complete physical
errors and changing edge-friction qualification remain required.

The [input/boundary packet](../evidence/20261019-qualified-ranking/hard-native-hard-input-boundary-v1/README.md)
retains source/face loads, paired timings, CPU/GPU controls and every unsuccessful
quadrature trial. Full arrays and logs remain under the project's data directory.

## Near-saturation friction solve

The extended penalty sweep30,60,100,300,1000,3000,10000 also fails the captured
frame110 within2048 iterations. Its normalized target is0.9999893674593;
rounding it to fully sliding would change force by1.063e-5, exceeding the
unchanged1e-9 force gate. The existing fully sliding analytic branch therefore
does not apply.

Experimental engine`fe37f2c78` retains the same16-interval radial-memory
objective, staged history, exact force equality and continuous slope disks.
It eliminates the weighted quadratic and force equality analytically. Writing
line increments as`z = D L`, weights as`W` and the force target as`t` gives

```text
L0 = prior + (t - w.T prior) / sum(w)
z(u) = D L0 - M u
M = D W^-1 D.T - e_last e_last.T / sum(w)
F(u) = z(u) - project_disks(z(u) + u)
```

The generalized Newton matrix is block tridiagonal with16 blocks of size2x2.
Native Quadrants block elimination and residual backtracking produce an initial
value for the existing alternating direction solve. A failed initial-value solve
restores its original initialization. Newton's internal1e-12 residual target is
not a replacement acceptance rule: the original force1e-9, dual1e-10,
line1e-9 and continuous cone1+1e-10 gates still decide admission. The bounded
Newton work uses64 steps; its count is recorded separately from the subsequent
alternating direction iterations. No pressure solve or PCG is introduced.

Exact staged-prior CPU/GPU replay admits both frames106 and110. The loaded region
uses12/44 Newton steps respectively, followed by one original iteration. Frame110
force residual is at most3.34e-15 and line residual2.53e-14; its line differs from
the independent constrained reference by less than6.8e-11. The reference itself
reports its1000-iteration limit and a separate2.064e-8 KKT line-distance bound;
its solver status is retained. This comparison verifies the declared discrete
friction model, not its physical accuracy.

CPU/GPU48-step changing-normal/vector histories, tangent rotations, slot
permutations, selected reset, loss latching and release/reload all pass. Maximum
line/traction differences are1.43e-6/1.81e-5 within the existing1e-5/1e-4 budgets.
The separate16-step sharp/smooth transition and binding controls pass on both
backends. The actual B4 Franka max/full scenes now each admit all130 frames0--129:
520 environment steps, including511 loaded environment steps. This repairs the
former110 blocker; it does not establish a full grasp cycle.

A same-allocation, profiler-off GPU replay of the valid frame106,200 calls per
repeat and three repeats, reduces the median radial-friction solve from3.0695ms
to2.0870ms (1.47x). It includes dispatch, synchronization and unchanged runtime
acceptance, and excludes history interpolation, contact construction, elastic
recovery and dynamics. The original solve uses398 iterations for its loaded
region; the candidate uses12 Newton steps plus one original iteration. Both
admit the same input and agree with the independent reference.

### Admitted paired scene cost

The profiler-off paired timer now completes the predeclared100--129 moving
window, with three repeats after continuous history from frame0. Each mode first
passes a separate0--129 audit; restarts reproduce the final outputs. Pure-rigid,
max and full positions agree to1e-12m. Max/full finite stress observations agree
at relative1e-7/absolute1e-6Pa; their singular maximum masks match exactly.
Controller, native runtime acceptance, dispatch and final synchronization are
included. Audits, host field exports, resets and pre-window steps are outside
the clock. Full fields remain on GPU during production timing.

| Batch | Pure rigid ms/step | Max ms/step | Full ms/step | Added max/full ms/step | Max/full env-step/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.638 | 254.060 | 254.254 | 251.422 / 251.616 | 15.74 / 15.73 |
| 64 | 2.852 | 286.197 | 286.317 | 283.344 / 283.465 | 223.62 / 223.53 |

Both allocations use an RTX PRO6000 Blackwell Server GPU,64-bit Genesis,
dt0.01s, one physical substep, every-step output and surface order16. B64 tiles
the four supplied initial conditions16 times; it is homogeneous condition-bank
scaling, not64 independently randomized conditions. All120/1920 timed
environment steps per repeat are loaded;24/384 include sharp contacts. Three
repeats give360/5760 admitted timed environment steps per stress mode. Max
samples span254.032--254.078ms atB4 and286.101--286.630ms atB64; full samples
span254.108--254.307ms and286.291--286.518ms respectively. Setup and factorization
costs are retained separately in the packet.

The complete path remains expensive. Input/friction component improvements do
not remove this approximately0.25--0.29s scene-step cost. The near-equal max/full
times are measured for this configuration and window, not a general output-mode
equivalence. Larger batches improve aggregate throughput, but full-cycle
stability, broader batch diversity and further complete-module optimization
remain open. A separate original-kernel profile localizes the remaining cost.

The separate B4 max profile over the same30-step window identifies
`kernel_hertz_planar` at193.266ms/call (76.72% of measured device kernel time),
`kernel_bind_source_vertices` at23.808ms (9.45%), boundary residual traction
at17.407ms (6.91%), source resolution at3.996ms (1.59%) and combined fields
at3.826ms (1.52%). Friction averages1.438ms (0.57%), with an11.698ms maximum;
the three force allocations average0.753ms each. The instrumented wall time
is264.230ms/step; use the separate254.060ms uninstrumented result for throughput.
Kernel statistics and overlapping inclusive host calls are not added together.
The next priority is current collision/source provenance and local geometry
reuse in the planar-contact search and source-vertex binding. Their cost was
excluded from the earlier resident hard-input replay, which explains why that
component's8.45x improvement did not remove the complete-scene bottleneck.

The [friction and admitted-scene packet](../evidence/20261019-qualified-ranking/hard-native-friction-newton-v1/README.md)
retains exact source inventories, the unsuccessful penalty sweep, independent
prototype, CPU/GPU controls, complete admitted timings and the separate profile.

## Ordered geometry search acceleration

Experimental engine`cf292de7f` precomputes conservative position and normal
bounds for consecutive64-face blocks and4096-face groups. Runtime closest-face
and normal-root searches visit surviving faces in their original order. Normal
roots retain source-nearest selection across disconnected/nonconvex components.
Source-vertex binding uses the same hierarchy and preserves duplicate-feature
ambiguity, geom identity and the original norm tolerance. All new metadata is
shared fixed-shape data; this change adds no temporal cache or history state.
It changes neither the sharp-edge law nor runtime acceptance thresholds.
The additional64-bit bounds occupy104 bytes per face block/group and48 bytes
per vertex block/group, with exact ceiling-sized arrays independent of batch.

The first candidate`68668460a` squared the vertex-distance tolerance when
pruning blocks. A targeted boundary control found25 mismatches among1928 CPU
queries: rounding excluded equality accepted by the original norm comparison.
The retained negative control motivates`bbe06fc38`, which compares the same
norm as the original test. CPU/GPU each pass1928 queries after this repair.
At the smaller geometry resolution, five GPU root face IDs differ from the
original. Each is proved to lie on a shared vertex/edge within1.15e-16m;
root positions, curvature, residuals and statuses still meet their comparison.

The final hierarchy control spans10240 disconnected-sphere faces plus a
nonconvex torus. CPU/GPU each compare1152 rotated/unrotated normal-root and
closest-point queries with the original engine; all pass, including face IDs
for this mesh. Each backend also passes1960 exhaustive membership/binding
queries over4162 vertices, crossing64/4096 boundaries. Controls include
conflicting/same feature IDs, different geoms, unmapped/ambiguous vertices,
two world-coordinate scales and576 roundoff-boundary points per environment.
The actual B4 max/full scenes each admit all130 frames0--129. Status, contact,
load and trajectory observations are retained for every frame.

The raw52-array comparison retains its uniform relative1e-9/absolute1e-12
criterion and its failures. Full-field corner tensors differ by at most
4.44e-8Pa, with maximum relative L2 difference2.33e-15. An unchanged-engine
repeat itself differs by4.57e-8Pa and2.55e-15, also failing that raw absolute
criterion. In max mode, probe barycentrics differ by3.46e-12 while probe
positions differ by at most1.39e-17m. Other raw fields meet the original
criterion. The diagnostic result explicitly records`all_fields_conformant=false`;
it is not labelled a blanket raw-array pass. Separate cross-engine checks
reuse the existing public timer's position1e-12m and stress relative1e-7/
absolute1e-6Pa criteria, exact validity/load/edge/status masks and infinite
maximum comparison. All130-frame pure-rigid/max/full audits pass atB4 andB64.
This distinguishes numerical conformance from independent physical accuracy.

### Complete scene measurements after geometry pruning

The same profiler-off100--129 window and three repeats retain continuous
history from0 and separate every-step audits. Configuration, four-condition
bank, surface order16,64-bit precision and GPU model match the preceding
measurements. Each pure-rigid/max/full trio shares one allocation. Engine
versions use separate allocations; the ratios below compare matched
configurations rather than a paired same-allocation engine crossover.

| Batch | Pure rigid ms/step | Max ms/step | Full ms/step | Added max/full ms/step | Max/full env-step/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.582 | 51.028 | 51.206 | 48.446 / 48.625 | 78.39 / 78.12 |
| 64 | 2.851 | 72.463 | 72.348 | 69.612 / 69.497 | 883.21 / 884.61 |

Compared with the original254.06/254.25ms atB4 and286.20/286.32ms atB64,
max/full scene rates improve by about4.98/4.97x and3.95/3.96x respectively.
All360/5760 timed environment steps per stress mode are admitted; the
24/384 sharp-contact environment steps per repeat remain represented.
Full stress fields stay on GPU during timing. Setup, reset, prelude, research
exports and audits stay outside the measured window. Max timing samples span
50.969--51.114ms atB4 and72.290--72.527ms atB64; full samples span
51.126--51.209ms and72.294--73.010ms. Preparation timings, including warm/cold
compilation differences, are retained separately and are not attributed solely
to geometry preprocessing.

The first single-level candidate measured51.64ms atB4 and72.09--72.19ms atB64.
It retains its off-scene vertex-boundary defect and is a diagnostic timing
control. The final second level adds little whole-scene benefit on this
workload: B4 improves slightly and B64 is slightly slower. Its separately
profiled binding cost decreases from0.636ms to0.395ms. The large measured
improvement over the original comes from conservative block pruning overall.

The final original-kernel B4 profile reports planar geometry13.323ms (27.13%
of device kernel time) and vertex binding0.395ms (0.81%), versus the original
193.266ms/23.808ms. Boundary residual traction now dominates at17.709ms
(36.06%), followed by source resolution4.070ms (8.29%) and combined fields
3.876ms (7.89%). Friction averages1.464ms and reaches11.918ms. Profiled wall
cost61.591ms is separate from the51.028ms primary timing. These observations
prioritize boundary work and remaining current-feature geometry reuse, rather
than further fixed hierarchy levels without measured benefit. Shared geometry
preprocessing alone has not supplied continuous-contact cache statistics.

The [geometry-search packet](../evidence/20261019-qualified-ranking/hard-native-geometry-search-v1/README.md)
retains all successful and failed controls, raw field differences, unchanged
baseline repeat, source inventories, paired scene timings and separate profiles.
Use its captured tasks and pinned engine sources with the existing Slurm/data
workflow. Full-cycle admission, diverse batch scaling and independent complete
curved-shell physical errors remain open. Sharp-edge global singular maxima
and finite positive-depth sampled values retain their existing interpretation.

## Boundary quadrature scheduling

Engine `ddf776072` compacts currently loaded sources before particular-field and
boundary-traction evaluation. The list is rebuilt on every recovery, with the
original source identities and admission checks. Actual B4 frames contain
0,3,4,6 or8 loaded sources among3000 slots. Compaction alone leaves boundary
traction at17.697ms versus17.709ms and complete max/full at51.70/51.58ms
(B4) and71.45/71.42ms (B64). This ablation supplies little overall benefit.

Engine `06cf62f1c` distributes the same boundary quadrature samples over64
lanes. Samples, weights, geometric rules and acceptance gates stay fixed;
summation order changes. CPU/GPU order16 shell and normal-component controls
pass. All-unloaded, one-environment reload and multi-environment reload checks
verify zero stale loads/tensors and restored fields. GPU every-step/manual,
two-substep and selected-reset friction controls also pass.

The complete0--129 max/full runs admit all520 environment steps per mode.
Separate B4/B64 public observations match the geometry-search baseline under
the existing position1e-12m and stress1e-7 relative/1e-6Pa checks; status,
loaded-region and sharp-edge infinity masks match exactly. The stricter raw
array diagnostic retains its failures: probe barycentric differences reach
3.45e-12, and full corner tensors differ by6.35e-8Pa (relative L2 at most
3.12e-15). Each affects128 frames under the original universal1e-9 relative/
1e-12 absolute criterion. These roundoff failures remain recorded separately
from public conformance. Loaded-source counts match on every compared frame.

Primary timings retain the original profiler-off100--129 window, three repeats,
continuous history from0, order16,64-bit precision, dt0.01 and one substep on
RTX PRO6000 Blackwell. B64 tiles the four initial conditions16 times. Each
pure-rigid/max/full trio shares one allocation; engine versions use separate
allocations with matching configurations.

| Batch | Pure rigid ms/step | Max ms/step | Full ms/step | Added max/full ms/step | Max/full env-step/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.604 | 40.695 | 40.361 | 38.091 / 37.757 | 98.29 / 99.11 |
| 64 | 2.899 | 53.907 | 54.055 | 51.008 / 51.156 | 1187.24 / 1183.98 |

Relative to the preceding geometry-search measurements, complete scene costs
decrease by about20--21% atB4 and25--26% atB64. Max samples span40.235--40.835ms
and53.892--54.046ms; full samples span40.182--40.424ms and53.990--54.143ms.
All360/5760 timed environment steps per stress mode are admitted. The24/384
sharp-contact environment steps per repeat remain represented. Runtime checks,
controller, dispatch and final synchronization are included; full fields stay
on GPU. Audits, exports, setup, reset and prelude are separate. Preparation
times, including compilation, remain in the records.

The separate B4 original-kernel profile measures boundary traction7.369ms
(19.07% of device kernel time), about2.40x faster than17.709ms. Planar geometry
now dominates at13.247ms (34.28%), followed by source resolution4.048ms and
combined fields3.876ms. Friction averages1.456ms and reaches11.847ms. The
instrumented51.909ms wall time has a separate scope from primary40.695ms.

### Single-region wrench failure at frame428

The first requested720-frame full/max runs of the geometry-search baseline
reach the scheduler's eight-minute limit. Their final parseable journals retain
367/382 admitted frames respectively; extra saved frames lack completed journal
admission. The original logs and arrays remain preserved. Compact audit retries
of both that baseline and active-source compaction reject max mode at frame428.
The64-lane candidate reproduces the same refusal. All three retain environment
admission `[false,true,true,true]`, input status `[32,0,0,0]` and allocation
status `[3,0,0,0]`. Full-cycle stress timing remains unavailable.

At frame428, environment0 merges two collision sources into one physical sharp
region. CPU/GPU exact-input allocation replays match the original force,
residual, status and iteration results. An independent singular-value
decomposition of its6-by-3 point-force wrench map has rank3 and certifies an
unrestricted normalized residual lower bound of0.0010036262. Its left-null
certificate has zero measured null residual. This exceeds the existing1e-12
gate even when arbitrary signed forces are allowed at the fixed region center.
The other three environments have rank6 and accepted allocation residuals
below6e-17.

The missing free moment is approximately
`[-7.305e-7,1.940e-6,2.068e-6] N m`. Its perpendicular part corresponds to an
86.90micrometre force-line offset; an axial couple of1.033e-8N m remains.
Translating one point force therefore cannot represent the complete wrench.
More Newton iterations or a tighter solver tolerance cannot remove the proven
defect in this point-region model. A distributed pressure/traction field has
additional first moments whose feasibility and stress response still need
independent checks. The earlier unvalidated pressure-translation shortcut
retains its141% near-field change and remains unadmitted.

The [boundary-scheduling packet](../evidence/20261019-qualified-ranking/hard-native-boundary-lanes-v1/README.md)
retains14 terminal campaigns, exact source captures, both timed-out runs,
all three frame428 refusals, CPU/GPU certificates, ablations, short-window
timings and raw differences. All48 frozen engines and12 original unfinished
files are preserved. This completes a measured scheduling improvement;
distributed-traction moment fidelity, full-cycle admission and independent
complete curved-shell physical errors remain active work.

## Single-region pressure support and traction limits

The frame428 diagnostic now retains the original geometric anchor, edge slope,
curvature and material inputs. Rejection clears the final edge exponent and
Hertz axes. The first two readers incorrectly treated those cleared arrays as
valid law outputs and fail explicitly; both remain archived. Independent scalar
root solving and native CPU replay now rebuild the law from retained inputs,
agreeing within1e-10 relative. Independent quadratic-surface projection recovers
the captured pressure center within1e-12m and normal within1e-9.

For environment0's two-source sharp region, the reconstructed transverse width
is0.5986micrometre, along-edge half-length65.953micrometres and exponent0.42469.
The local force is `[-0.019963,-0.000264,0.027143] N`. On a flat reference
plane, its required normal-pressure centroid is44.929micrometres from the
edge and-79.076micrometres along it. Every nonnegative pressure on the predicted
parabolic support has `x >= 0` and `x/width + (y/half_length)^2 <= 1`; the
required centroid gives76.494. A force-preserving affine pressure tilt reaches
negative multipliers as low as-345.17. Adding these first moments to the fixed
pressure law therefore cannot supply a nonnegative pressure field.

Gauss-Jacobi quadrature independently checks pressure mass, centroid and
variance. On the actual quadratic surface, a proportional tangential field
adjusted to reproduce the original force produces local moment approximately
`[5.985e-12,-4.279e-10,-1.583e-11] N m`, versus required
`[-2.146e-6,-1.216e-6,-1.578e-6] N m`. Orders24/48 change the computed curved
moment by at most1.10e-21N m. That particular field also exceeds the local
friction bound: its shear/normal ratio0.735532 exceeds coefficient0.719102.

A separate continuous bound covers a rectangular superset of the predicted
footprint on the original P2 surface. Positive Bernstein subdivision and
rounded convex-hull exclusions retain every intersecting surface portion;
Bernstein area-vector coefficients bound all normals within retained cells.
No retained cell touches the enclosing height boundary. The local sheet is
face9, enclosed within+/-131.905micrometres normal depth. Flat admissible/
excluded force controls and an analytically bounded paraboloid pass.

| Subdivision levels | Retained cells | Normal-angle upper bound | Enlarged friction-cone half-angle | Target-force angle |
| --- | ---: | ---: | ---: | ---: |
| 8 | 7 | 0.285071 degrees | 36.005043 degrees | 36.335698 degrees |
| 10 | 25 | 0.210560 degrees | 35.930533 degrees | 36.335698 degrees |

Both bounds exclude the original force even with arbitrary nonnegative pressure
and arbitrary Coulomb tangential traction over this rest-surface support. At
level10, the separating direction has target-force projection1.19134e-4N,
while every admissible unit traction projects at most-0.00353572. The enclosing
cone implies a necessary relative force change of at least0.7071%. This is a
lower bound on input-model correction; its stress error remains unmeasured.
It strengthens the earlier point-force rank diagnosis: adding free moments
alone cannot resolve this fixed-support force incompatibility.

Raw collider association, signs and contact-time transforms pass independently.
The two material-side witness offsets are49.929/49.525micrometres along the
collision normal, exactly half the99.857/99.050micrometre penetration to within
2.17e-17m. Witness-to-quadratic-surface distances are0.651/2.165micrometres.
The first raw application point matches the mean of its witnesses to2.8e-17m;
the second differs by0.277micrometre, retained separately. Post-integration
surface queries remain distinct from the captured contact-time geometry.

The [region-moment packet](../evidence/20261019-qualified-ranking/hard-region-moments428-v1/README.md)
retains the repeated native refusal, original reader failures, reconstruction,
quadrature checks, raw application-point audit and continuous cone certificate.
These results concern the predicted support and undeformed-normal model.
Deformed contact geometry, different support or a separately qualified nearby
load model remain possible. Next work should measure a cheap correction's
physical pressure and complete/near-field stress response under the existing
engineering budgets. All production gates and the original failure are retained.

## Contact frames and shell displacement

The next checks test two possible explanations of the frame428 mismatch:
the choice of traction normal and elastic motion of the shell. They retain the
same C40 engine and exact captured forces and moments. The one-face Coulomb
condition uses the force components relative to that face's normal, consistent
with the [Coulomb formulation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAETHERefMap/simathe-c-coulombfric.htm).

The contact-time opposing face normal differs from the material normal by
3.82635 degrees. Expressing the original force in that frame increases its
tangential/normal ratio from0.735533 to0.843922, above friction0.719102.
Changing the normal alone therefore worsens this one-face compatibility check.
The corrected normal-pressure centroid on the opposing plane is
(-38.2412, -80.1174) micrometres from the projected edge origin. It lies
38.2412micrometres beyond the finite face boundary.
Both transformations preserve the complete original wrench. Contact involving
the adjacent box face remains a separate possibility requiring its own geometry
and pressure checks.

The first frame reader omitted the saved `region_gap` when locating that plane.
`region_center` identifies the material normal-root point. Adding
`region_gap * normal` supplies the actual opposing plane. The corrected gap is
-180.124micrometres and the pressure center is101.430micrometres below the face.
Both independently captured opponent witnesses lie on its shared sharp edge
to3.23e-17m in the transverse/normal coordinates. The first packet's opposing
plane distances, moment and centroid are superseded by this correction.
Force ratios, frame rotation and the separate displacement results remain valid.

An untimed replay captures admitted frames419,420,427 immediately before the
same native frame428 refusal. In environment0, all four physical regions have
sharp-edge profiles. The following maxima use the actual accepted regular P2
displacement plus the far displacement of other regions. A mass-weighted rigid
translation/rotation is removed from regular plus nodal particular displacement.
The local self-contact field is excluded from these smooth corrections.

| Frame | Maximum smooth normal rotation (degrees) | Maximum along-edge center shift (micrometres) | Maximum smooth normal displacement (micrometres) |
| --- | ---: | ---: | ---: |
| 419 | 0.000468632 | 0.0567423 | 0.280415 |
| 420 | 0.000245830 | 0.0297313 | 0.144903 |
| 427 | 0.0000532190 | 0.00670722 | 0.0331188 |

These values use four point sources matching the pressure second moments.
Replacing them with a single source changes rotation by at most5.99e-9 degrees
and displacement by1.68e-13m. Mass-projection residuals stay below1.6e-15.
The measured smooth correction cannot account for the approximately79micrometre
longitudinal centroid shift. This comparison bounds neither the self-contact
surface gradient nor a different adjacent-face contact state.

Separately, Boussinesq/Cerruti displacement integration uses the reconstructed
frame428 pressure and proportional shear at its pressure centroid. At positive
depths0.1 and0.01 times the0.598608micrometre width, the displacement norms are
0.146983 and0.153620micrometres, respectively. At the shallower5.986nm probe,
orders48/96/192/384 give0.153788/0.153611/0.153620/0.153620micrometres.
The final vector refinement changes by8.69e-7 relative. Force normalization and
pressure centroid checks pass. This conditional estimate is far smaller than
the approximately50micrometre collider application offset. Its pressure and
proportional-shear assumptions retain the original force-cone incompatibility,
so it supplies a displacement scale rather than a physically admitted solution.
Exact surface limits and sharp-edge stress peaks retain their separate scope.

The [contact-frame and displacement packet](../evidence/20261019-qualified-ranking/edge-shell-transition-v1/README.md)
contains five terminal campaigns, captured analysis sources, the native refusal,
and the quadrature refinements. All elastic exports occur in the untimed audit.
The next physical comparison must check adjacent-face contact or explicitly
qualified nearby load/support estimates. Small smooth shell motion and a simple
normal swap are insufficient explanations. Useful full-cycle admission and
complete-field physical accuracy remain open.

## Adjacent-face wrench limits

The corrected contact-time frame supports a separate two-face necessary check.
The primary rectangle spans transverse0--a and longitudinal+/-b. Its adjacent
rectangle spans normal-depth0--D and the same longitudinal interval. Their
compressive normals are respectively+z and-x, with friction0.719102.
Here a=0.598608micrometre and b=65.9525micrometres come from the reconstructed
pressure law. The rectangles deliberately admit arbitrary nonnegative tractions.
Their elastic gap compatibility remains an additional physical requirement.

For each rectangle, nonnegative bilinear coordinates transfer every distributed
traction to its four corners while preserving the complete wrench and the
convex friction cone. This follows the surface-contact wrench construction
described by [Caron, Pham and Nakamura](https://arxiv.org/abs/1501.04719).
The present check treats both perpendicular faces together. Inner friction-cone
polygons supply feasible atomic corner forces. Separators proposed using outer
polygons are verified directly against the circular cone at every corner.
Concavity extends the separating inequality over each complete rectangle.
The primary-face compression, side-face compression and tensile negative
controls pass. The original force alone is feasible with both faces, while the
single-face force remains excluded.
All16 classifications agree between64 and128 polygon facets. The largest
normalized feasible-wrench residual is3.28e-13. One refined primal solve returns
an unknown HiGHS status. The recorded interior-point retry proves infeasibility,
and the resulting separator independently passes the continuous-cone check.

Two moment targets are kept separate: the original collision midpoint wrench,
and an input approximation placing each unchanged force at its captured opponent
witness on the true sharp edge. This replacement changes the transverse moment
from9.84713e-7Nm to-2.70e-19Nm. Its other moment changes are retained in the raw
result. It leaves the complete force unchanged.

| Side depth D (micrometres) | Original, half-length b | Witness, half-length b | Original, half-length 2b | Witness, half-length 2b |
| ---: | --- | --- | --- | --- |
| 0.598608 | Excluded | Excluded | Excluded | Feasible |
| 9.57773 | Excluded | Excluded | Excluded | Feasible |
| 38.3109 | Excluded | Excluded | Feasible | Feasible |
| 153.244 | Excluded | Excluded | Feasible | Feasible |

An independent bound explains the original transverse-moment obstruction.
Let P and S denote the primary and side normal reactions. Coulomb friction gives
Fx <= mu P - S and Fz >= P - mu S, hence
S <= (mu Fz - Fx)/(1-mu^2) =0.0833443N. Since the primary face contributes
nonpositive transverse moment and the side contributes at most D S, the
positive original moment requires D >=11.8150micrometres, independently of
longitudinal support. This exceeds the predicted width by19.7 times.

For witness loads and D=a, another bound gives b >=77.9547micrometres.
Specifically, Fz-Fx >=(1-mu)(P+S) and
abs(Mx+Mz) <= b(Fz-Fx) + mu a(P+S).
This establishes a longitudinal obstruction even after the transverse couple
is removed, against the predicted b=65.9525micrometres.

These tests distinguish force feasibility from complete-wrench feasibility and
pressure physics. In particular, moving loads onto their real witnesses removes
the midpoint-induced transverse couple but still exceeds the original support's
longitudinal capability. Enlarging support until a conic check passes supplies
no elastic-pressure or stress-error qualification. The next comparison must
test a physical two-face contact state or an explicitly declared nearby load and
support model against a matched pressure/stress reference.

The [corrected-plane and adjacent-face packet](../evidence/20261019-qualified-ranking/box-edge-wrench428-v1/README.md)
retains the original plane-reader evidence, corrected witness checks, initial
integer-normal reader failure and the successful conic controls. The failed
refined primal solve and its verified retry are also retained. Production
behavior and the native frame428 refusal remain unchanged.

## Two-face elastic reference and field sensitivity

A small frictionless two-face control now separates geometric normal reactions
from frictional traction. It uses the captured slope0.0668779, longitudinal
curvature49.3940/m, E=10GPa and Poisson ratio0.3, with primary/side normal
resultants0.0257501/0.0217295N. In the flat material chart its face normals are
(s,0,1)/sqrt(1+s^2) and(-1,0,s)/sqrt(1+s^2). The resulting material-chart force is
(-0.0199628,0,0.0271427)N. A large transverse component therefore arises from
frictionless geometric contact with the adjacent face. Applying a single
undeformed-material-normal friction cone to that entire component needs a
separate model justification.

The local reference minimizes linear elastic energy with nonnegative cell
forces and separately fixed face resultants. Its surface vector compliance
integrates Boussinesq/Cerruti rectangular sources exactly and averages over
Gauss target cells. Off-cell components agree with independent point-kernel
quadrature to1.21e-15 relative. The isotropic surface kernels follow the
[linear half-space formulation](https://doi.org/10.1680/jgere.22.00063), whose
small-displacement geometry assumption limits the physical interpretation.
The normal/contact active set checks force balance, gap and complementarity.
Each outer update assigns cells to the face selected by deformed plane gaps;
loaded cell-mean points must lie within the corresponding finite face.
This controls cell-mean geometry. Pointwise face membership and large surface
gradients require further qualification.

The imposed data are the two face normal loads and local quadratic geometry.
Along-edge force is zero; the original friction0.719102, tangential history and
collision midpoint moments remain separate. Transverse curvature is omitted
in this slender local gap. The reference determines a new rigid approach and
moment, so its errors cannot be assigned to the original complete scene load.
Numerical collision penetration supplies no physical indentation here.

| Cells (two faces) | Longitudinal half-domain | Face updates | Invalid-face force | Remote-boundary force |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 2b | 1 | 18.43% | 3.01% |
| 600 | 3b | 3 | 0 | 0 |
| 1600 | 3b | 4 | 0 | 0 |
| 3840 | 3b | 5 | 0 | 0 |

Here b=65.9525micrometres. The finest grid has48 graded transverse cells and80
longitudinal cells; the smallest transverse width is4.157nm and longitudinal
width4.946micrometres. Its maximum material displacement is0.164944micrometre,
rigid approach vector(-0.106700,0,0.129586)micrometre and transverse moment
-3.3733e-9Nm. Reference-area cell traction maxima grow from7.06 to20.85 to
25.61GPa across the three valid grids. These singular local values exceed the
10GPa modulus and reinforce the unresolved small-strain/geometry limitation.
They supply no finite physical surface-pressure or stress maximum.

An independent circular Hertz control uses known force, radius and modulus;
its contact solve selects pressure and support. On20x20 cells, target orders4/8
give indentation errors0.01818%/0.01839% and cell-pressure L2 errors4.593%/4.614%.
These smooth normal-contact checks validate the numerical operator within
those errors. Sharp two-face physical accuracy remains an additional question.
On the1600-cell corner control, target order4 to8 changes finite-depth tensors
by at most0.0171% and sampled tensile peaks by0.1381%. Source integration order8
to12 changes tensors by at most3.44e-7 relative. Spatial refinement remains the
larger uncertainty:1600 to3840 cells changes tensor norms by4.55%,2.95%,1.86%,
0.90%, and tensile maxima by0.15%,5.97%,0.40%,0.51%, at the four depths below.
These are observed pair differences, not rigorous continuum error bounds.

The stress comparison uses572 identical positive-depth probes:11 transverse
positions over+/-0.75b,13 longitudinal positions over+/-1.5b and four depths.
The transverse/longitudinal probe spacing is9.893/16.488micrometres. Both fields
use the same flat-material Green tensor and resultant force. The existing
single effective-edge profile retains its independently predicted b, exponent
0.424692 and transformed width0.599945micrometre, without fitting this reference.
The following percentages compare against the3840-cell control.

| Depth (micrometres) | Tensor L2 error | Sampled von-Mises peak error | Sampled tensile peak error | Tensor<5%, both peaks<10% |
| ---: | ---: | ---: | ---: | --- |
| 6.595 | 11.94% | 2.33% | 37.85% | Fail |
| 16.488 | 10.35% | 3.02% | 32.48% | Fail |
| 32.976 | 6.62% | 4.65% | 0.69% | Fail |
| 65.953 | 2.77% | 2.78% | 1.18% | Pass on these probes |

The reference-derived parabolic half-lengths sqrt(5*variance_y) are65.750 and
78.324micrometres on the primary and side faces. Their fourth/second-moment
ratios are2.163 and1.896, compared with15/7 for a parabola. The side half-length
still changes from82.884micrometres on the1600-cell target-order8 control,
showing residual spatial sensitivity. Thus separate face profiles are a useful
diagnostic direction, with their independent prediction still to be established.

For that diagnostic only, two parabolic line loads fit each reference force,
centroid and longitudinal second moment. At the same four depths their tensor
errors are5.14%,3.86%,1.91%,0.50%; tensile peak errors are3.00%,6.39%,0.043%,0.165%.
The nearest tensor result still exceeds5%. The first two raw tensile argmax
locations differ by32.976/16.488micrometres; the remaining locations coincide.
These coarse-grid argmax differences include symmetric peak ambiguity and
supply no continuous peak-location qualification. The moment fit uses measured
reference parameters, so its smaller errors establish a shape-sensitivity
result rather than a validated cheap predictor.

The [two-face evidence packet](../evidence/20261019-qualified-ranking/box-corner-elastic428-v1/README.md)
retains all ten terminal campaigns, including the initial invalid-face branch
and the data-volume quota failure before output initialization. A small
replacement campaign uses the prior user-authorized4GB home exception; detailed
paths and measured occupancy are in DATA_LOCATION.md. Original large fields
remain on data. Production engines, acceptance gates and the frame428 refusal
are unchanged. Next work should predict the two face profiles from geometry,
material and loads, quantify original-to-control wrench/history differences,
and check useful finite-depth accuracy without promoting a fitted diagnostic
or this dense offline reference into the default runtime path.

## Independent face scales and surface ordering

The next candidate removes the reference-moment fit. Each face uses the
transverse square-root strip profile, its own normal load F, normal n and
projected gap slopes s=abs(n_x), k=n_z*k_material. Its leading self-energy and
gap work are

```text
U = 3 F^2/(5 pi E* b) [log(2b/w) + D]
G = F (s w/5 + k b^2/10)
D = log(4) + 1/4 - [psi(3)-psi(7/2)]/2 + nu n_x^2/(1-nu)
w b = 3 F/(pi E* s)
b^3 = 3 F/(pi E* k) [log(2b/w) + D - 1].
```

This scalar root and strip-force balance predict widths0.509/0.0100micrometre
and half-lengths65.884/188.622micrometres for the primary/side faces. A geometric
candidate caps the side interval at the predicted primary half-length and
renormalizes its force, giving side width0.01996micrometre. The untruncated
side density remains parabolic on its predicted188.622micrometre scale inside
that capped interval. Cross-face elasticity is omitted from these scale
relations. The cap is a declared approximation requiring its own checks.
Dimensional rescaling and pressure/cell-force normalization pass. Neither the
scales nor the subsequent geometric search read reference pressure moments.

| Model | Tensor error at6.595/16.488/32.976/65.953micrometres | Tensile peak error at the same depths |
| --- | --- | --- |
| Independent strips | 29.63% /28.70% /22.49% /13.83% | 1.41% /0.72% /17.63% /3.54% |
| Primary-length side cap | 5.14% /4.16% /2.67% /1.21% | 10.86% /5.49% /1.01% /0.72% |

The capped model meets the sampled tensor5%/both-peaks10% budgets at the three
deeper levels. It fails both tensor and tensile budgets at6.595micrometres.
Doubling longitudinal source integration from128 to256 changes its complete
probe tensor by5.89e-10 relative; the uncapped change is9.99e-5. These conditional
comparisons use the same earlier flat-material linear reference and572 probes.
Their physical-reference limitation remains essential.

The candidate pressure is independently projected to the reference's evaluation
grid for a displacement check. Load-weighted plane approaches minimize mean
normal mismatch without changing pressure. The uncapped/capped force-weighted
normal-gap RMS values are2.43/5.38nm, while the respective other-face violations
carry23.27%/14.03% of total force. For the capped case,30.65% of side-face force
lies beyond the other plane; the largest excess is4.331nm. Thus the apparently
small displacement mismatch still matters relative to the narrow side support.

A separate eight-point search shortens only the side interval from1.00 down to
0.65 times the predicted primary length. Its objective uses calculated elastic
surface gaps, excluding reference stresses and pressure moments. Every trial
retains positive other-plane gaps; the0.65 endpoint still has10.81% of side-face
force outside. No corrected candidate is admitted. The CPU geometry search
costs3.40s in this offline evaluator and supplies no production timing result.
Further shortening merely to pass a gate is not justified by these data.

The reference itself now receives an independent surface-ordering audit. Across
a single, correctly oriented traversal of this convex corner, both face tangents
have positive material-chart x projection. The finest solution instead contains
8 reversed pairs among268 adjacent loaded-cell pairs. Their pair-force weight
is8.83% of the loaded-pair total; this weight is a diagnostic, not a damaged-area
or physical-contact-force fraction. Its minimum deformed/original transverse
increment ratio is-0.006762. The600/1600-cell solutions have no such reversals;
the1600-cell minimum ratio is already only0.000143.

Reintegrating the finest saved forces at target orders8 and16 retains all8
reversed pairs, with minimum ratios-0.006310 and-0.006112. The largest changes
from saved cell displacements are47.7/59.7pm. These controls hold the pressure
fixed; they do not replace a fully re-solved order16 contact state. The maximum
loaded displacement-gradient norm remains1.944, with finite-difference surface
area ratios approximately0.037--2.943. Averaged central-difference normals have
positive z projection, illustrating why that test alone misses adjacent-cell
ordering failures.

These results demonstrate a discrete orientation problem and large gradients
in the current linear reference. They strengthen its previously stated physical
limitation. Preserved pressure/field differences remain useful numerical and
shape diagnostics; they establish neither continuum sharp-corner accuracy nor
an admitted two-face runtime model. Further physical work must address actual
frictional reactions and consistent deformed geometry, with explicit positive
depths outside any unresolved edge zone. The dense reference remains optional.
Native throughput work can proceed independently on the existing admitted
window and its measured geometry bottleneck, preserving all validity checks.

The [independent-profile and surface-audit packet](../evidence/20261019-qualified-ranking/box-corner-predict428-v1/README.md)
retains four terminal campaigns, final and original workers, the unsuccessful
geometric search and source/target quadrature controls. Original runtime
engines and the frame428 refusal remain unchanged.

## Remaining physical and integration limits

The edge pressure behaves as`x^(-lambda)`. Its pointwise loaded-edge limit is
unbounded, and native pressure evaluation explicitly returns infinity there.
Refined cell-average maxima continue to rise, as expected for this singular
model. A finite cell average or positive-depth sample peak is not a converged
surface/global stress peak; retain sampling scales with every such value.

The pressure law and complete-field component are now connected to the actual
main pipeline. The short closing window is admitted; full-cycle changing-contact
admission and independent complete curved-shell physical validation are still required.
The earlier Franka rejections and original moment mismatch remain evidence.
The updated-support nearby-load candidate passes the declared sensitivity
budget at the captured frame. The native candidate still needs full-cycle
changing-contact validation. Preserve strict source diagnostics and the original
refused frames. Paired pure-rigid/max/full throughput now covers the declared
short window; useful complete-module optimization remains required.

The [evidence packet](../evidence/20261019-qualified-ranking/hard-edge-law-v1/README.md)
retains both coupled-reference stages, the unfitted-law comparison, native
CPU/CUDA controls and timings, exact sources and externally hashed arrays.
