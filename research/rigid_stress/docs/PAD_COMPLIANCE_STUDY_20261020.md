# Physical pressure with local and coupled compliance

The eight-case CPU study now predicts pressure and support using the declared
pad law and measured elastic operators. A rigid-gap model is close for the
original10GPa link/1e9N/m3 pad, but its pressure error reaches24.47% for a1GPa
link/1e10N/m3 pad. A diagonal compliance approximation barely improves that case.
The useful next integration therefore needs both an inexpensive validated regime
and a route that retains regional elastic coupling when it matters.

This is a static normal, coarse linear-elastic study on the original full hollow
egg with500um wall thickness. It extends the [first geometric study](AUTOMATIC_CONTACT_PROGRESS_20261020.md)
and [resolved-pressure transfer](RESOLVED_PAD_AND_SOURCE_INPUTS_20261020.md).
It does not supply moving/friction validation, thickness independence or an
automatic production model. The [extended goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md)
remains active.

## Model contrast and numerical checks

One full elastic factor and the existing834-node contact compliance are reused.
All six free-link rigid modes are retained through mass projection and gauge
alignment; there are no physical pins. Coupled compliance includes interactions
between opposite pads and surface regions. Rigid and diagonal models solve their
own approximate pad laws. Their residual against full elasticity is measured
separately and must not be interpreted as an admitted full physical solution.
The asymmetric case declares a smooth physical stiffness factor
`1+0.5*tanh(x/1mm)` on the pads, rather than a numerical contact penalty.

The48 saved fields cover eight cases, three models and centroid/degree6 contact
quadrature. All declared-model force/moment, pad-law, equilibrium and balance
checks pass. The coupled full-elastic pad-law residual stays below2.75e-9.
Independent full-elastic fixed-point pressure agrees with the condensed solution
to4.69e-11 in nodal force; splitting every source into two half-area samples
changes nodal force by1.54e-11. The original cached nominal complete stress is
reproduced within5.69e-13.

A separate actual search control starts at1mm. All three models detect omitted
compression (about21um) using exact quadratic minima on omitted boundary faces,
then expand to3mm and pass containment. The search extent is numerical and does
not prescribe the pressure-bearing support. The eight primary cases all fit
inside the3mm search. Smaller footprints are not artificially expanded.

The [terminal packet](../evidence/20261019-qualified-ranking/pad-compliance-study-v1/README.md) retains actual sources, full controls,
[32 model-error rows](../evidence/20261019-qualified-ranking/pad-compliance-study-v1/meta/pad-compliance-model-errors-v1.csv),
24 separate quadrature comparisons and externally hashed fields.

## Errors against coupled pressure at degree6 quadrature

Each entry is pressure area-weighted L2 error / complete stress physical-volume
L2 error. Values compare models at the same quadrature and mesh, with coupled
pressure as the declared discrete physical solution; they are not continuum
stress errors.

| Link E (GPa) | Pad stiffness (N/m3) | Load (N) | Pad | Rigid pressure / stress | Diagonal pressure / stress |
| ---: | ---: | ---: | --- | ---: | ---: |
| 10 | 1e+09 | 0.1 | Uniform | 0.355% / 0.104% | 0.347% / 0.102% |
| 10 | 1e+08 | 0.1 | Uniform | 0.054% / 0.020% | 0.053% / 0.019% |
| 10 | 1e+10 | 0.1 | Uniform | 2.694% / 0.696% | 2.636% / 0.681% |
| 1 | 1e+08 | 0.1 | Uniform | 0.536% / 0.197% | 0.528% / 0.194% |
| 1 | 1e+09 | 0.1 | Uniform | 3.525% / 1.040% | 3.448% / 1.018% |
| 1 | 1e+10 | 0.1 | Uniform | 24.472% / 6.684% | 23.914% / 6.535% |
| 10 | 1e+09 | 0.3 | Uniform | 0.446% / 0.147% | 0.437% / 0.144% |
| 1 | 1e+09 | 0.1 | Gradient | 3.523% / 1.113% | 3.426% / 1.078% |

For the nominal case, both rigid and coupled models identify the same sampled
upper/lower areas,14.098/15.704mm2, but their pressure fields still differ.
For the flexible-link/stiff-pad case, upper support areas differ by less than1%
(11.295 versus11.379mm2), while the two-sided pressure error is24.47%. A correct
radius or area alone therefore cannot establish a correct pressure field.
The coupled upper peak is11.702kPa versus9.303kPa for the rigid predictor.
The inherited distribution key `maximum_cell_average_pressure_pa` stores
quadrature-point law pressure in these study records; the separate quadrature
comparison explicitly integrates actual face-average pressures.
In the gradient-pad case, the coupled force centroid moves0.3395mm in x;
the rigid estimate is0.3488mm. The complete records retain both sides and axes.

![Predicted upper-pad pressures under two stiffness combinations](../evidence/20261019-qualified-ranking/pad-compliance-study-v1/pad-compliance-figure-v1/group0/figure/pad-compliance-pressure-preview.png)

The upper row shows close nominal pressure patterns. The lower row shows the
coupled flexible-link/stiff-pad pressure concentrating toward the finite-land
edge; the diagonal model stays close to the rigid prediction. Each row shares
a kPa scale, all panels use common physical coordinates, and every positive
quadrature pressure is shown without smoothing or clipping. The complete lower
pad, tensors and errors remain in the saved data. The183×142mm SVG/PDF/600dpi PNG
and150dpi preview pass direct visual review and text/size/font checks.

## Sampling and cost limits that affect the next implementation

Changing centroid to degree6 source quadrature changes the coupled complete
stress by1.94–2.82% across the eight cases. The nominal change is2.677%, much
larger than its0.104% rigid-versus-coupled model difference. Face-average pressure
changes range3.74–10.44%; these compare integrated cell averages, not unmatched
point samples. Source sampling must therefore be controlled independently of
pressure-model selection. Neither quadrature alone establishes spatial convergence.

The reused834×834 operator occupies5,564,448 bytes. Its original construction
cost is not remeasured here. The new full elastic factor setup takes46.78s.
Exploratory degree6 contact calls take about0.009s for rigid equilibrium,
0.165–0.202s for diagonal compliance and23.64–25.65s for the selected coupled
cases. These single-process research call measurements include current CPU
solver work, have no independent timing repeats and are not production rates.
The diagonal model pays substantial extra work for little measured accuracy gain.

The coupled solve has now been reduced and checked below. Extend the
physical study to thickness, changing nonconvex/asymmetric geometry and
friction/unloading. Then integrate the useful model into the existing native
Quadrants feature, preserving supported batching, output, isolation and resets,
and measure paired module/scene costs including prediction. No Gaussian radius
fit or loose solver tolerance replaces these checks.

## Exact pad elimination and inner/outer solver accuracy

The [completed reduced-solve packet](../evidence/20261019-qualified-ranking/reduced-pad-equilibrium-v1/README.md)
retains48 new fields: direct reduced Newton, fixed-PCG Newton and an adaptive-PCG
control each replay the16 coupled cases above. It reuses the original16 coupled
reference fields and changes only the numerical formulation. Eliminating each
independent compression-only pad spring analytically leaves a piecewise
quadratic energy in the834 elastic coordinates and four free plate/link
coordinates. At degree6 quadrature, this reduces5446 unknowns to838. Regional
elastic coupling, free-link balance and the physical pad law are preserved.

The direct solve differs from the original coupled reference by at most
1.51e-10 in complete stress and2.28e-10 in area-weighted pressure. Full elastic
pad-law, force/moment, equilibrium and exact omitted-face checks retain their
original limits. Independent remeasurement bounds both PCG variants' complete
stress differences from direct Newton by8.21e-12 and their raw tensile/von-Mises
peak differences by1.17e-11. These are implementation checks on the same
discrete physical problem, not new continuum errors or ranks.

Three paired calls per case on one allocated CPU core on `rtx-209-099` give:

| Degree6 case | Full QP median (s) | Reduced Newton including cold initialization (s) | Ratio |
| --- | ---: | ---: | ---: |
| Nominal10GPa link /1e9N/m3 pad | 23.467 | 0.0600 | 391 |
| Flexible1GPa link /1e10N/m3 pad | 23.638 | 0.0924 | 256 |
| Gradient pad /1GPa link | 24.583 | 0.0929 | 265 |

Both paths reuse the saved elastic compliance. Reduced Newton also caches its
factor and quadrature map. Actual factor/map rebuilds take0.014–0.021s at
centroid quadrature and0.083–0.090s at degree6; the independent full FEM setup
takes43.71s in this process. Stored compliance, its factor and the mapped factor
together occupy13,690,944/41,873,472 bytes at the two quadratures. Those counts
exclude interpolation, solve workspace and full FEM storage. The captured
direct-v1 metadata repeats the last setup duration on reused cases; only
cases0,3,6,7 at each quadrature are actual rebuilds. Later workers record zero
on reuse. The paired calls are repetitions inside one process, not independent
process samples, native throughput or end-to-end scene rates.

PCG is an inner linear solve. Inexact Newton controls that solve through a
relative linear-residual forcing term while separately checking the nonlinear
residual; see [Eisenstat and Walker](https://epubs.siam.org/doi/10.1137/0917003).
Here every variant keeps the dimensionless outer gradient target1e-10,
independent pad-law target1e-7 and force/moment target1e-8. Fixed inner1e-6
therefore does not impose a1e-6 final nonlinear residual. The tested adaptive
rule is explicitly `min(0.1, sqrt(current_gradient_norm))`, not the paper's
Eisenstat-Walker update.

| Solver, over16 fields | Newton updates | Total PCG iterations |
| --- | ---: | ---: |
| Direct reduced Newton | 24 | — |
| Fixed inner1e-6 PCG | 36 | 268 |
| Tested adaptive PCG | 74 | 304 |

All pass the same final checks. The tested looser early solves increase total
iterations and supply no reason to change a default; cross-job PCG wall times
are not a paired speed experiment. No production tolerance is changed.
This keeps numerical convergence distinct from load-model discrepancy and
physical stress accuracy.

Each solver also passes four prescribed normal load/upper-gap tilt states,
including loading and unloading, against the full QP. Previous pressure is only
an initial guess; cold/warm results agree, and return to the original load/gap
recovers pressure within1.75e-11. These controls use a fixed candidate region.
They do not establish general moving/nonconvex search, tangential history,
thickness independence or automatic native integration. Those remain the next
physical and production work.
