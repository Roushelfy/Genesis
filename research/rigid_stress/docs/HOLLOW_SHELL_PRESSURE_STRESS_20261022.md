# Independent-pressure hollow-shell stress comparison

Hertz plus one central finite-body regular response has2.188% complete-tensor,
0.634% membrane and0.452% bending error in this normal polar case. Sampled
tensile/VM peak errors are5.023%/0.771%. Adding observed reference variations
gives conditional3.479%/0.861%/0.805% field/resultant envelopes and8.007%/0.819%
peak envelopes, inside the declared5% field and10% peak budgets. These are
offline research-mesh results, not accuracy attached to the level1 native scene.

The [independent pressure study](HOLLOW_SHELL_CONTACT_PRESSURE_20261021.md)
defines the fixed faceted polar-land hollow egg:500um wall, E=10GPa, nu=0.3,
density2000kg/m3, Fn=0.2N and rigid principal curvatures50/80 per m. Free tilt
and consistent-mass inertial relief remain identical. Independently solved
fine97 pressure is held fixed across all three body meshes. No new elastic
factorization or runtime pressure solve is introduced.

## Reference and error separation

Bilinear pressure hats feed an independent Boussinesq tensor integral. Along
each angular ray, pressure is piecewise quadratic in radius; radial integrals
are analytic. Surface terms include the finite principal value and local
pressure term. Near points use512 angles with256 control; far points use a
degree16 Chebyshev source expansion with degree8 control, separated at four
Hertz major radii. This is offline integration, not a full-vector contact basis.

Kernel controls compare complex-step point tensors, independent positive-depth
pressure quadrature, analytic Gaussian surface stress and positive-depth limits.
Point relative error is3.28e-16; pressure-quadrature error is below8.26e-7 of
pressure scale. Gaussian surface error is0.0254%, including bilinear pressure
representation. Actual near256/512 and far9/17 changes are2.10e-5 and3.32e-10
of peak pressure; both original integration gates pass.

Retained25 scalar normal-source responses supply the regular finite-body field.
Affine-gradient readout checks pass on all meshes. The common bank contains
378432 volume points,80800 column points,9611 original probes and5280 new
contact-edge points. Each mesh's own element corners also enter sampled peak
extraction; no nodal stress smoothing is used.

| Reference change | Tensor | Membrane | Bending | Tensile peak | VM peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Medium/fine body | 1.0948% | 0.1468% | 0.1303% | 0.0615% | 0.0173% |
| Pressure65/97 | 0.0231% | 0.00081% | 0.00112% | 1.4633% | 0.0127% |
| Source3/5 | 0.1693% | 0.0770% | 0.2188% | 0.1319% | 0.0183% |
| Lower/higher integration | 0.00429% | 0.00233% | 0.00261% | 0.00219% | 0.000044% |

Fine-reference tension is2.642080MPa at(-70.1511,8.12465,30000)um; VM is
15.608267MPa at axis depth27.9807um. Halving contact-edge sampling changes the
tensile maximum1.32494%. This enters the observed variation sum, not a continuum
bound. Symmetric tensile maxima can switch sides: raw coordinate jumps and
distance to the observed near-maximum set remain explicit. A single raw peak
coordinate is not declared converged.

| Coarse candidate versus fine97 | Tensor | Membrane | Bending | Tensile peak | VM peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hertz + central regular response | 2.1878% | 0.6343% | 0.4518% | 5.0234% | 0.7710% |
| Hertz half-space only | 78.2406% | 94.0901% | 94.0558% | 40.5954% | 5.7202% |

Distributed regular sources weighted by analytic Hertz pressure give2.1854%
coarse tensor error; that small improvement does not justify a richer runtime
model here. Signed tensor decomposition retains discretization, point/distributed
source and pressure/particular changes with their measured cross terms. Scalar
error norms are not subtracted to infer error components.

The [reviewed figure](../evidence/20261019-qualified-ranking/hard-shell-pressure-stress-v1/hard-shell-pressure-stress-figure-v4/group0/figure/shell-pressure-stress.svg)
shows independent pressure, surface principal stress, the surface x-axis and
axis VM profiles. Surface tensile peaks concentrate near the footprint edge;
VM peaks below its center. Gray shading sums observed deterministic reference
variations and is not a confidence interval. The regular finite-shell response
accounts for most of the improvement over the half-space-only field.

## Reproduction and limits

The [terminal packet](../evidence/20261019-qualified-ranking/hard-shell-pressure-stress-v1/README.md)
binds captured sources, inputs, actual job states, inspections and figure data.
Jobs737709/737722 cover kernel/fields;737780/737790 reduce saved arrays.
Figure attempts737816/737819 retain text-QA failures before the corrected export.
Use captured `tasks.json`, public engine alias`d4503b9e4`, `rtx-mid`, image`1_26`,
the existing data-side physical environment and explicit data/private-cache paths.
Complete fields stay on data. The study reuses its exact three pressure parents,
fine97 refinement, mesh study and pressure inspection. Run its saved-array
inspector before plotting. `plot_shell_pressure_stress` takes `--study`,
`--inspection`, `--pressure`, the retained `--style` helper and a new `--output`.

This supplies a conditional independent stress comparison for one normal case.
The old prescribed-pressure direct-reference failure remains unchanged. Moving,
asymmetric/nonconvex contacts, friction history, level1 native fields and matched
accepted max/full scene throughput remain required. The active goal is open.
