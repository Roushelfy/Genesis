# Cheap hard-contact estimates: first executed controls

The new priority in remote commit `75d534428` is active: inexpensive Hertz/thin-shell
estimates and native throughput. Further soft-pad, vector-compliance and coupled
Newton-PCG integration is superseded. Existing evidence stays unchanged.

This study supports retaining both principal curvatures in a cheap estimate.
Replacing them by one equivalent circular curvature produces58.21% pressure
error in the rotated elliptical half-space control. A native elliptic lookup
costs approximately21us per update for1–8192 supplied regions on the measured GPU.
One explicit far-field correction also improves a symmetric two-region example.
These are local hard-contact component results, not a finished egg estimator.

## Physical comparison

The deformable material has E=10GPa, nu=0.3 and reduced modulus
E*=E/(1-nu²); the opposing quadratic obstacle is rigid. The total normal load
is0.1N. An independent square-cell Boussinesq boundary-element solve minimizes
complementary energy under nonnegative cell forces and the total-force equality.
It determines support and pressure without a prescribed Hertz footprint.
The influence matrix integrates1/(pi*E*distance) over each constant-pressure
square. Active-set completion requires force balance, nonnegative force/gap
and complementarity, and loaded cells may not touch the search boundary.

The cases are circular curvature(50,50)/m, curvature(20,80)/m rotated0.43rad,
and the minimum of two circular quadratic gaps. The latter has two known
symmetric regions, each carrying0.05N; their initial centers are separated by
three isolated-contact radii. All cases retain36² and54² reference grids.
Analytical estimates use8x8 Gauss cell averages and a reported quadrature-only
load normalization, never a fit to the reference distribution.

| Case / estimate | Pressure L2 error,36² →54² | Subsurface sampled tensor error,36² →54² |
| --- | ---: | ---: |
| Circular Hertz | 1.903% →1.162% | 0.2014% →0.0808% |
| Rotated elliptical Hertz | 3.591% →2.407% | 0.3949% →0.1835% |
| Equivalent circle for elliptical obstacle | 58.040% →58.212% | 25.279% →25.330% |
| Independent Hertz for two regions | 9.678% →9.628% | 5.390% →5.360% |
| One interaction correction for two regions | 3.172% →2.501% | 0.6776% →0.4120% |

Pressure error is the relative area-weighted L2 norm of cell-average pressure
on the same grid. Subsurface stress uses Boussinesq displacement differentiated
by complex step and isotropic Hooke's law, with a separate on-axis point-force
identity check. The15x15x9 observation grid is fixed across both reference
resolutions, at depths0.4a–2a. The tensor metric is an unweighted sample norm,
not a complete-volume norm. Cell resultants are point sources for this
subsurface evaluation; its shallowest depth exceeds two coarse-grid cell widths.
Surface peaks, global continuum maxima, shell membrane/bending and friction
are **not qualified** by these sampled tensors. The retained original tables
also include sampled tensile/von-Mises maxima and locations; their variation
is not a global peak-error bound.

The ellipse semi-axes are85.51um and34.19um; the equivalent-circle radius is
55.46um. Keeping ellipticity has a clear advantage larger than the observed
36²→54² differences. For the circular control the54² indentation differs
from Hertz by0.0163%. For the two-region case independent Hertz misses
approximately14.16% of the indentation scale through neglected interaction.
Reference-grid differences are reported sensitivity, not certified continuum
uncertainty or shell accuracy.

## A bounded inexpensive interaction candidate

For each known region, approximate another region's surface displacement by
`w_j(x)=F_j/(pi*E*|x-c_j|)`. Evaluate its gradient and Hessian at the original
region center. Replace the relative curvature K by K+H and shift the center
by `-(K+H)^-1 grad(w)`, then evaluate Hertz once. No additional contact solve,
iteration or fitted coefficient is introduced. This is a point-resultant
far-field Taylor approximation, not a general shell correction.

The two centers move outwards by1.868um. On54², pressure error drops from9.628%
to2.501% and sampled tensor error from5.360% to0.4120%. The unchanged reference
solves/fields are reused. This positive control assumes known symmetric load
allocation and keeps aggregate moment by symmetry. Arbitrary load allocation,
current-source moment fidelity, nearby/merging islands and shell interactions
remain open; this correction is not a production default.

## Native implementation and measured cost

[`native_hertz.py`](../qipc_comparison/native_hertz.py) is a Quadrants component
with total force, reduced modulus and a symmetric2x2 relative-curvature input
per supplied physical region. It uses the tangent-frame eigenvectors and a
shared257-row table for ellipse aspect ratio b/a in[0.05,1]. Numerical table
construction uses256 transformed angular quadrature points in Quadrants;
each update performs eight lookup comparisons and interpolation. Production
geometry preparation, source grouping and the pressure scatter are not yet
connected, so this component remains in the research package.

The table follows the elastic half-space potential integrals. With a>=b,
q=b/a and s=tan²(theta), its dimensionless curvature integrals are

```text
Jx(q) = 2 integral[0,pi/2] cos²(theta) / sqrt(q²cos²(theta)+sin²(theta)) dtheta
Jy(q) = 2 integral[0,pi/2] cos²(theta) / (q²cos²(theta)+sin²(theta))^(3/2) dtheta
kx/ky = Jx/Jy
a³ = 3 Fn Jx/(2 pi E* kx)
p(x,y) = 3 Fn/(2 pi a b) * sqrt(max(1-x²/a²-y²/b²,0))
```

The circular limit recovers the [standard Hertz force/modulus relation](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-demanalysis.htm).
Independent SciPy adaptive integrals/root solves check101 rotated and
force/material-scaled cases per CPU/GPU backend. Maximum relative differences
are2.10e-6 for axes,2.24e-7 for indentation and4.16e-6 for peak pressure.
Direct pressure sampling, zero/invalid loads, unsupported curvature and
independent-region isolation pass. Nonpositive curvature, out-of-table aspect
ratio and invalid inputs remain explicit statuses, with no invented radius.
This does not yet test live collider source splitting or environment lifecycle.

| Supplied regions/update | GPU median | CPU median | Logical tensor storage |
| ---: | ---: | ---: | ---: |
| 1 | 21.32us | 16.69us | 6,268B |
| 128 | 20.58us | 25.46us | 18,968B |
| 512 | 20.64us | 49.25us | 57,368B |
| 8192 | 21.01us | 531.08us | 825,368B |

Hardware: NVIDIA RTX PRO6000 Blackwell Server Edition; CPU uses the engine's
default one Quadrants thread. Each number is the median of21 groups of100
resident updates, synchronized at the group boundary. Dispatch is included;
input transfers, JIT/imports, pressure-sample evaluation, geometry, grouping,
elasticity and output are excluded. These are estimator-module measurements,
not scene throughput or the cost of a complete observation. P90 timings are
retained in the receipts. Table/allocation preparation with warm kernels is
1.84–2.26ms; storage counts logical tensors, not allocator/driver overhead.

## Pressure maps and reproducibility

![Hard-contact pressure controls](../evidence/20261019-qualified-ranking/hard-hertz-v1/hard-hertz-figure-v1/group0/figure/hard-hertz.png)

Caption: constant-cell pressure on the54² grid. Panels a–c compare the rotated
elliptical obstacle's independent BEM solution, equivalent-circle estimate and
elliptic Hertz estimate. Panels d–f compare the nonconvex two-region solution,
independent Hertz regions and one far-field correction. All maps share the
MPa color scale and show the complete nonzero support. Error labels compare
cell-average pressure with the corresponding BEM reference. No smoothing or
statistical error bars are applied. The [SVG](../evidence/20261019-qualified-ranking/hard-hertz-v1/hard-hertz-figure-v1/group0/figure/hard-hertz.svg)
and [PDF](../evidence/20261019-qualified-ranking/hard-hertz-v1/hard-hertz-figure-v1/group0/figure/hard-hertz.pdf)
retain editable text; the rendered preview has been inspected.

The figure supports retaining elliptical geometry and testing cheap interaction
corrections before a coupled solver. The circular replacement loses the long
axis and loads regions outside the elliptical support. The local correction
improves the two-island example without representing arbitrary nonconvex contact.

Executed campaigns under the existing data run root:

- `hard-hertz-v1`, job733211: native GPU/CPU controls and six independent BEM cases.
- `hard-hertz-interaction-v1`, job733223: two reused-reference correction comparisons.
- `hard-hertz-figure-v1`, job733233: pressure maps and editable exports.

All allocations completed with exit0. Each campaign retains exact task arguments,
captured analysis sources, engine identity, logs and results. Use the archived
`tasks.json` with the existing `research.rigid_stress.shell_comparison.submit`
entrypoint and a fresh explicit data-side output directory; its `--cpu-only`
mode still runs numerical work in Slurm `rtx-mid`. Original arrays stay on data
with manifest hashes; no live source, environment or old evidence was migrated.

## Finite footprints on the witnessed mesh

The [native transfer component](../qipc_comparison/native_hertz_transfer.py)
now feeds supplied Hertz regions into the existing native elastic recovery.
It places96 integration points on each physical ellipse, rather than hoping
that fixed mesh quadrature samples intersect a roughly50um footprint. A
projected triangle walk follows boundary-edge identities from the witnessed
face. It cannot use a nearby disconnected face as a shortcut across a cavity
or the gap of a U-shaped material. It refuses non-facing/open walks.

The normalized Hertz measure uses r=sqrt(1-t²), giving3t² dt dtheta/(2pi).
Three Gauss points in t and32 angular points exactly integrate its constant,
linear and quadratic moments on one affine patch. Every mapped sample uses
signed P2 shape weights on its reached boundary face, preserving discrete
virtual work. Crossing several triangles introduces a measured integration
error; this rule is not a certificate of the entire continuous footprint.
No physical radius or elasticity tolerance is enlarged.

Seven controls pass on each CPU/GPU backend: an interior face, shared edge,
interior mesh vertex, cavity inner wall, U-shaped boundary, coarse egg vertex
and its common rotation/translation. All use three environments and include
release, failure isolation, resultant force/moment checks and independent
all-triangle projection. Planar normal/tangential loads also pass a closed-form
quadratic virtual-work identity. Independent scikit-fem P2 solves reproduce
all same-load complete native fields to1.42e-11 relative error. This is
implementation agreement, not Hertz accuracy on the shell.

A separate12-radial by1024-angular CPU rule measures source-integration
sensitivity on the same mesh. Relative complete tensor differences use volume
weights over all tetrahedron corners:

| Footprint placement |96-point versus dense-rule complete tensor difference |
| --- | ---: |
| Interior flat face | Below5e-14 |
| Shared flat edge |0.1808% |
| Interior flat vertex |0.1989% |
| Cavity inner wall |0.1375% |
| U-shaped flat wall | Below1e-12 |
| Coarse egg vertex, either frame |0.01676% |

Invalid environment indices, a valid and a failed region sharing one environment,
and nonzero shear at zero normal load also pass explicit rejection/isolation
controls in the final source. The original vertex/rotated failure is retained. Its cause was comparing a
barycentric roundoff allowance directly with a segment parameter: very short
footprints have small barycentric derivatives. The correction bounds the
current barycentric coordinate first and clamps only its roundoff-sized
zero-distance crossing; no load or geometry acceptance budget changes. A prior
wrong-environment `skfem` import failure is also retained. Successful reruns
reuse the existing data-side physical-validation environment.

The combined estimate,96-point boundary mapping, wrench checks and nodal
scatter costs below include zeroing the destination nodal load and all native
dispatch. They exclude elastic recovery, grouping/curvature preparation,
input transfers, imports/JIT and actual scene dynamics. Each allocation requests
one CPU; timing uses seven groups of five resident updates with synchronization
at each group boundary. The workload repeats one known coarse-egg vertex
region in each environment, with max-mode elastic state storage.

| Regions/environments | GPU median | CPU median |
| ---: | ---: | ---: |
|1 |0.1728ms |0.1747ms |
|128 |0.1727ms |3.207ms |
|512 |0.1955ms |12.584ms |

These timings include the estimate itself and must not be added to the earlier
lookup-only table. They remain module measurements rather than moving-scene
throughput. No public automatic contact option is connected yet.

## A measured friction/moment limitation

On the faceted egg, proportional tangential traction with Fn=0.2N and
Ft/Fn=0.2 introduces7.951e-7N*m about the witnessed center. The strict transfer
rejects this mismatch and retains unaffected environments. A separate CPU
candidate applies one6x6 weighted minimum-force correction to enforce the
same total force and zero source free moment. It does not change rigid dynamics
or the strict transfer path.

The correction changes the samples by0.02172N in summed magnitudes, or12.20%
in the pressure-weighted force norm. Its mass-relieved complete P2 tensor
change is only0.09582%; raw tensile maximum changes from31,231.48Pa to31,235.37Pa,
and raw von-Mises maximum from93,450.51Pa to93,452.35Pa. The free-link balance
correction is recomputed for each load; it is included in these field changes.
Small aggregate-moment disagreement and local source redistribution therefore
need separate stress-response measurements.

This candidate is **not engineering-admitted**. Against declared mu=0.5 cones
on the actual reference facets,60 of96 samples violate the cone both before
and after correction. Maximum required local mu changes from0.8114 to0.8276;
we do not raise mu to make the candidate pass. A separate diagnostic in the
unchanged Hertz contact-plane frame has no cone violations: maximum shear/normal
ratio changes from0.2000 to0.2483, with positive compression throughout. These
are two explicitly different friction-frame models. The latter does not erase
the reference-facet violation, prove the physically correct current normal, or
supply partial-slip/unloading history. This is evidence of the
normal/geometry/traction-model conflict on the coarse faceted mesh, not proof
that physical smooth-shell contact is impossible. Neither the0.09582% coarse
field change nor exact wrench balance establishes hard-contact accuracy,
friction history or a general practical tolerance. Normal curvature/model
preparation and actual source semantics must be reconciled before promotion.

Executed transfer campaigns are `hard-hertz-transfer-v1`/job733285 (missing
validation dependency), `v2`/733299 (retained geometry failures) and `v3`/733327
(seven successful cases per backend and timing). Final `v4`/733392 repeats
the unchanged geometry checks, adds invalid-index/multiple-region/zero-normal
rejection controls and supplies the matching timings above. `hard-hertz-wrench-v1`/733351
reuses the completed v3 GPU fields for the correction study; `v2`/733398 adds
the explicit contact-plane versus reference-facet cone comparison. Their sources,
receipts, failures and data-side full-array hashes are retained in the
[transfer packet](../evidence/20261019-qualified-ranking/hard-hertz-transfer-v1/README.md).

## Remaining work

Next connect cached rest-curvature/current physical-region inputs to the cheap
native estimate and the now-executed witnessed-boundary transfer. Keep sparse
collider samples distinct from physical regions and preserve complete force
and moment under source splitting. Compare a declared inexpensive tangential
law with monotonic/unloading controls, then test hollow-shell thickness/global
response against a matched targeted hard-contact reference. The observed
51um circular footprint also makes mesh/transfer resolution a concrete issue;
do not enlarge it to a numerical disk. Run matched max/full and pure-rigid
moving-scene timing as soon as the native route is connected. No further
soft-pad campaign or dense elastic-basis preparation is required by this goal.

Correctly transferring a50um pressure footprint does not resolve a coarse mesh's
elastic near field. Before claiming damage-relevant global peaks, test a cheap
Hertz near-field stress estimate/correction alongside the coarse shell response
and check its overlap/double-counting and full-output meaning against a matched
reference. Repeating a point-load peak convergence sweep is not the next step.
