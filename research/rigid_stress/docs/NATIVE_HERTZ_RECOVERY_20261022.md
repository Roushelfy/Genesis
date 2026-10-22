# Native normal recovery and far-field cost

The experimental native source now contains a normal Hertz estimate and a
finite-domain regular correction inside the existing rigid `StressModel`.
All runtime numerical work is Quadrants; the shared elastic inverse/factor,
consistent-mass inertial relief and complete residual checks are reused.
No runtime pressure solve or contact stress basis is introduced.

The exact experimental source is
[`67c9de0b2`](https://github.com/Roushelfy/Genesis/commit/67c9de0b2ab02ca4adaa47773a0b6cd17692eebd).
Its parent retains the unaccelerated implementation; all captured intermediate
source patches remain reproducible from those public ancestors.

The supplied-region finite-box control passes CPU/GPU, rotated-frame,
multiple-contact, complete tensor, maximum-principal/VM reduction and max/full
checks. The four-source far rule reduces B1024 module time by about39%.
This is a normal recovery component, not the live automatic estimator: contact
association, automatic moving probes, general curved/inner contacts, physical
friction memory and complete moving-scene integration remain required.

## Configuration and acceptance

Regions carry an actual material anchor, center, tangent frame, normal resultant,
reduced modulus and relative curvature. The frame points into material. The
current implementation checks the anchor and requires the material to lie in
that region's half-space, with four major radii of clearance to noncoplanar
boundaries. Unsupported configurations remain rejected; it does not jump to a
different nonconvex surface. Only FP64 is qualified by these measurements.

For each supported normal region, minus the point Boussinesq traction on the
remaining boundary supplies the regular elastic load. On the actual loaded
plane, prescribed and particular tractions cancel analytically. The recovered
regular stress is combined with finite elliptical Hertz stress at every element
corner and supplied material probe. Numerical recovery omits its intermediate
peak pass, so a regular-only maximum cannot contaminate the combined result.
Both complete and max-only outputs use the same combined tensor calculation.

The extra probes in this control are fixed material queries, not automatically
generated moving contact probes. With no additional queries, only element
corners are sampled; this does not resolve subelement Hertz peaks. That mode
is used for refusal/lifecycle controls, not physical peak qualification.

## Reference failures and numerical repairs

Two different probe-side descriptions on an internal element boundary produce
different unsmoothed P2 stresses. The first control fails at5.36e-5 despite
machine-precision corner agreement. Binding the comparison to saved element
indices/barycentric coordinates removes this ambiguity without changing its
1e-7 numerical target. Raw reselected-side differences remain recorded.

The rotated reference then fails at31% query-tensor error. A read-only comparison
finds native rotation errors below7.1e-8, while3.85e-19m of nominal surface-depth
roundoff changes the old reference by1.225 times peak pressure. Both original
root brackets start much too far from the tiny positive confocal coordinate.
For interior points, the exact bound

```text
lambda <= z^2 / (1 - x^2/a^2 - y^2/b^2)
```

tightens the bracket while preserving the equation and stopping gates. CPU/GPU
controls down to z/a=1e-16 remove75 original native root refusals. Analytic-axis
error is3.33e-16; established resolved-depth tensors change at most1.53e-11 of
pressure scale. Original normal and tangential component regressions also pass.
Failed rotated point-location and NumPy batch-solve setups remain retained.

## Independent prescribed-load comparison

The fixed box has192 affine tetrahedra, two region slots and four independently
loaded environments: one source, two displaced sources, opposing-face sources
and an unloaded environment. The body and source frames are also rotated and
translated. The material is E=10GPa, nu=0.3, density2000kg/m3, with the same
50/80 per m curvature law used by the preceding box study.

An independent P2 solve uses order12 boundary integration and96-point normal
stress evaluation, retaining the actual element side of every query. It
recomputes both complete corner/probe banks and sampled tensile/VM maxima.
The predeclared bounds are0.1% for tensor/volume changes and1% for peaks;
all pass. Observed maximum differences are approximately:

| Quantity | Difference |
| --- | ---: |
| Dense query tensor norm | 0.0023% |
| Volume-weighted tensor norm | 0.01974% |
| Maximum principal stress | 0.0019% |
| Von Mises maximum | 0.00030% |

These are independent same-material-mesh, prescribed-Hertz elasticity controls.
They do not establish continuum convergence or independent unknown pressure.
The [hollow-shell physical comparison](HOLLOW_SHELL_PRESSURE_STRESS_20261022.md)
retains its separate geometry, discretization and uncertainty; its2.188% error
is not assigned to this native box or the level1 Franka scene.

## Far rule and measured cost

At distances of at least eight major radii, four equal point forces at
`(+/-a/sqrt(5), +/-b/sqrt(5))` replace the sixteen-point potential calculation.
They preserve total load and both second pressure moments. Near fields keep
the original sixteen-point calculation. An explicit exact-field control is
retained. Independent12x64/20x96 pressure integrals at axis ratios1/0.73/0.1/0.05
and Poisson ratios0/0.3/0.49 give maximum error1.44e-6 of peak pressure, below
the original1e-5 target; the reference-rule change is1.42e-15.

The extra volume-weighted field change in the full controls is at most0.00278%,
and the sampled tensile-peak change is below0.000077%. CPU/GPU and rotation
checks remain below the1e-7 tensor target. All actual current source/configuration
identities are retained in the [terminal packet](../evidence/20261019-qualified-ranking/hard-native-hertz-recovery-v1/README.md).

Primary costs are warmed inclusive wall times, median of three eight-frame
blocks, on the same allocation. Every update includes input generation, Hertz
estimation, boundary integration/acceptance, regular solve and combined output.
The varying normal loads have independent environment phases and release.
Geometry association and friction are absent. The450 shared extra probes are
retained in both variants; output modes are measured separately.

| Batch | Exact max/full (ms) | Four-source far max/full (ms) |
| ---: | ---: | ---: |
| 1 | 1.14 / 1.15 | 1.14 / 1.14 |
| 64 | 1.95 / 1.99 | 1.64 / 1.65 |
| 256 | 3.95 / 4.01 | 2.71 / 2.74 |
| 1024 | 12.16 / 12.21 | 7.39 / 7.43 |

The terminal receipt gives unrounded final repeats. The large-batch repeat
ranges separate; small-batch improvement is limited by dispatch and the solve.
Separate fixed-input phase diagnostics attribute the original large-batch cost
mainly to combined field evaluation; those diagnostic medians are not summed
to manufacture an inclusive total. Every benchmark saves a separate24-frame
status/force-acceptance/elastic/peak audit. B1024 admits24576/24576 total frames,
including17014/17014 loaded frames; the rest are intentionally unloaded.
These rates are component throughput, not Franka scene or general qualified
stress throughput. They cannot be added to another workload's friction timing.

## Failure behavior and reproduction

A connected L-shaped link retains a supported outer contact while refusing a
real reentrant ledge whose material crosses the tangent plane and an oversized
footprint. Their actual anchors/centers remain unchanged. The low order4 rule
also refuses its asymmetric outer source at9.90e-4 wrench error; order8 resolves
that numerical issue while retaining the1e-4 gate. Earlier failed job states
remain failed. Source motion, NaN input, release and next-step recovery pass.
Invalidity persists through the same mechanical step and is counted once.
Unloaded spinning cases reproduce the original centrifugal stress fields.

The default zero-extra-query path initially exposes scalar empty-array and
static-shape issues. The final source uses the existing rank-zero placeholder
convention and a bounded compile-time presence flag. No extra query is read in
that path. Both CPU and GPU refusal/lifecycle controls pass after these repairs.

Use the pinned public engine source and captured analysis package in the packet,
with the established data-side Python, Slurm `rtx-mid`, image `1_26`, an explicit
new data output and private node-local caches. Run `check_native_hertz_recovery`
with `--backend`, `--reference`, optional `--rotated`/`--far-field`, and `--output`.
`time_native_hertz_recovery --control <completed-base-control> --output <new-data>`
times the exact rule; add `--far-field` for the fast rule. The independent
inspector reads complete arrays and timing audits without resimulating native
states. Captured task lists give exact paths and arguments.

The current public stress options are not redirected to this experimental
component. The next delivery must connect actual contact/region/history and
state-change paths, generate moving material probes, cover required curved and
inner contacts and friction, and measure matching accepted max/full Franka
throughput. The complete active goal remains open.

The [moving-probe follow-up](NATIVE_MOVING_HERTZ_PROBES_20261022.md) now implements
the material probe bank and measures its complete resident moving-input cost.
The original fixed-query evidence and timings above remain bound to their own
source/configuration; automatic live scene integration remains open.
