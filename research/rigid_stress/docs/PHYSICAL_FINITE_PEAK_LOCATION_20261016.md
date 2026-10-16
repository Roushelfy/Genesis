# Raw finite-load peak locations and the next geometry control

All90 last-three-level reference peaks now have a separate boundary-location
and prescribed-traction diagnosis. Job719976 naturally completes with matching
original/copied source closures. Every peak value and coordinate remains
exactly identical to the previously independent full-field audits.

## Measured locations

For each recorded raw maximum, the diagnosis identifies its actual mesh vertex,
incident material-boundary facets and shortest distance to the complete triangle
boundary. It retains the original prescribed traction on each incident facet.
The finest opposing-load results are:

| Geometry | Finest tetrahedra | Raw peak distance below boundary |
|---|---:|---:|
| Ellipsoid |315446|756.246um|
| Capsule |326676|551.035um|
| Thin neck |205437|527.783um|
| U |294458|0um|
| Narrow gap |446367|0um|
| Cavity |483087|391.511um|

The ellipsoid/capsule/cavity opposing, bending, asymmetric and multiple-load
peaks are subsurface on their finest meshes. Their near-slip peaks are on the
boundary. All five finest U and narrow-gap peaks occur at boundary vertices
with zero prescribed traction there. The diagnosis retains every case and
every raw global maximum, including these unloaded locations.

Consequently, a local boundary-vertex explanation alone does not explain the
16.75%/14.10% opposing raw-peak changes on ellipsoid/capsule. Stable original
27-point interior norms also cannot certify these unsampled subsurface maxima.
The next control must distinguish mesh/basis approximation from the changing
CAD geometry, with the original raw extrema retained.

## Algebraic boundary check and its limits

If incident outward normals and tractions admit one symmetric stress tensor,
then symmetry requires `n_i dot t_j = n_j dot t_i`. The script measures this
identity and the least-squares residual for one six-component symmetric tensor.
This concerns a continuous tensor value at that vertex. It proves neither
boundedness nor singularity of the full elasticity solution. Zero local defect
also does not rule out geometric or nonlocal stress amplification.

A manufactured symmetric stress validates the component map. Separate
normal-only algebraic controls retain the same amplitudes and normals and
pass at every boundary peak. These controls do not replace any physical load.
For the finest near-slip states, normalized incident-traction fit residuals are
3.190035% for ellipsoid,0.129382% for capsule and2.216694% for cavity. These are
unweighted algebraic traction residuals, not field errors or peak-error budgets.
They cannot be subtracted from the observed stress changes. Zero-traction peak
vertices give a zero local fit residual without validating their stress field.

The [complete immutable diagnosis](../evidence/20261014-physical/extended-finite-peak-boundary-v1/manifest.json)
retains all90 records, original peak coordinates, incident face identities,
normals, original tractions, source masks, fitted components, singular values,
residuals and both controls. All121 archive entries verify against their actual
runtime originals. Manifest SHA256:

```text
e41453c1a855bea2635227eb9893025aeacdefc9c521c6cc25ac439c3526ee44
```

## Controlled follow-up with fixed polyhedral geometry

Separate runs now freeze each original1.25mm ellipsoid/capsule polyhedron and
refine its full3D volume and surface basis with uniform edge midpoints, without
CAD projection. Completed asset jobs720018/720023 preserve volume, area and
centre of mass at their original1e-12 checks. Original14916/15979-tetrahedron
five-case states freshly requalify before the fine producers: jobs720019/720024
complete at unchanged original load, force, corner, quadrature and probe gates.

| Fixed polyhedron | Three tetrahedron counts | Fine producers | Independent15-field auditor |
|---|---|---|---|
| Ellipsoid |14916 /119328 /954624|720020 /720021|720022|
| Capsule |15979 /127832 /1022656|720025 /720026|720027|

The same3mm physical radii, continuous profiles, strengths, directions, friction,
material, angular velocities and original probes remain. This separates CAD
projection from mesh approximation. Volume/surface basis and numerical boundary
quadrature still refine together, so it is not a pure volume-only axis. Their
remaining contributions require separate controls before interpretation.
The family stays separate from all original eight-level CAD curves.

The [complete setup archive](../evidence/20261014-physical/fixed-polyhedron-convex-setup-v1/manifest.json)
retains all six meshes and ten freshly requalified original fields. Its manifest
SHA256 is `be31928e3d118aef258aea89d4204c9a964e85319566ae05c18812bea099eb05`.
Mesh assets and original-state qualification establish zero new fine equilibria
in this archive. The new producers and separate15-field auditors require their
own source/input closure, full-field review and publication before acceptance.
The exact submissions are retained in `meta/fixed-polyhedron-convex-axis-commands-v1.json`
under the established runtime root. Every original independent-FEM, applicable
model/island, resolved-contact, full egg/Franka/interior-hotspot, feedback and
physically qualified warmed-cost requirement remains active.
