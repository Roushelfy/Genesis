# Localize the unresolved covered CAD contact tail

The [three-level field study](PHYSICAL_COVERED_CAD_THREE_LEVEL_20261015.md)
retains a growing 9.68% → 10.72% adjacent tensor discrepancy. New source
audits and localization identify where that discrepancy occurs without
discarding any original probes or replacing the raw global maximum.
The [full objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains active.

## Qualify complete force attribution before interpreting local loads

The contact node maps select original source vertices with one unit entry
per row. Auditor 719369 qualifies a standalone selector-mesh reconstruction
against all five independently audited 1920-tet reference states. The same
closed source then processes all 15 dynamic states of the three covered
CAD models in 719370.

Every original normal and tangential stencil retains its source, exact node
support, stored weight and force vector. Original wide-coordinate position
arithmetic, physical barrier and unmollified normal magnitude for the friction
convention remain unchanged. Maximum all-node reconstruction differences are
2.775558e-16 N normal, 2.844947e-16 N shear and 2.220446e-16 N after full
source-force pullback. Maximum source-pair net force is 1.678343e-16 N.
These checks establish faithful attribution of the retained contact reference,
not independent verification of the contact algorithm's continuum limit.

For each captured finger link 10/11, the audit retains normal/shear/total body
and counterbody force and moment about the same reference origin. Actual
triangle adjacency defines components above 1e-8 N normal force; every force
below that threshold remains in the explicit tail wrench and all-node density
norm. Component-plus-tail force and moment must reconstruct the complete
source. At the final state, each source has one discrete component and zero
threshold-tail wrench. These graph components are not certified continuum
contact islands.

## Fixed source resultants coexist with increasing concentration

|Tetrahedra|Source|Active body samples|Active lumped area (m²)|Max discrete normal density (MPa)|Discrete normal density L2 (N/m)|
|---:|---:|---:|---:|---:|---:|
|31668|1|15|1.358154e-8|65.591641|4351.353749|
|31668|2|17|1.537938e-8|70.489892|4593.832362|
|64716|1|31|7.019198e-9|107.726557|5761.486894|
|64716|2|30|6.784956e-9|105.112106|5703.771702|
|171228|1|64|3.613409e-9|337.225338|9704.171376|
|171228|2|65|3.675047e-9|344.687602|9721.877782|

Density means the norm of a sample force divided by its actual deformed
lumped vertex area. Its L2 diagnostic is `sqrt(sum(|f_i|²/A_i))`, using every
sample; the activity threshold affects only counts/areas/components.
These are discrete diagnostics, not continuous pressures or contact areas.
Sparse reaction interpolation on coarse rigid finger triangles must not be
interpreted as a second, independently resolved pressure field.

For the last mesh pair, source force-vector changes are only 0.001696% and
0.001649% relative to their respective approximately 0.43 N norms. Moments
still change, so the complete local wrench is not certified stable:

|Source|Δ force (N), finer minus coarser|Δ moment about reference origin (N·m)|
|---:|---|---|
|1|(-7.320770e-6, -5.362943e-7, 1.404348e-7)|(-8.800841e-9, -1.119991e-8, -4.506891e-7)|
|2|(7.108558e-6, 5.335865e-7, -1.599087e-7)|(8.471916e-9, -4.651630e-8, 4.508934e-7)|

At the finest state, the moment vectors themselves are approximately
(-7.101354e-6, -9.797529e-7, 1.224729e-5) and
(6.835259e-6, -9.326637e-7, -1.224033e-5) N·m. All six original components
remain available in physical units, including small components.
Finite-gap friction produces measured pair total moments of approximately
(1.421772e-7, 3.751766e-11, -4.050104e-10) and
(-1.236925e-7, 5.22425e-11, 5.35492e-10) N·m; they are not silently removed.

## Identify the actual rigid feature without changing the physical case

Postprocessors 719384 and 719401 retain the original template roles and
all-node normal/shear density hotspots. The latter also records the actual
finger edge endpoints and incident face IDs. At every mesh level, each finger
uses the same approximately 6 mm edge with two incident face planes at an
unsigned acute angle of approximately 90 degrees. This is a measured geometric
crease, not a planar mesh subdivision edge.

Final normal stencil counts are 14/16 edge–edge on the first mesh and 30/29
on the second. The finest model uses 78/80 edge–edge, 5/5 edge–vertex and
2/4 face–vertex stencils. The retained node-group roles distinguish which
side supplies the vertex/edge/face. Increasing discrete density and a sharp
feature motivate a contact-concentration diagnosis; they do not alone prove
a continuum singularity. Rounding that finger edge would define a different
physical case, not numerical refinement of this one.

## Recompose the original full-probe error before localizing it

Analysis 719434 uses every original probe and weight. It reproduces both
complete tensor-change norms to 1e-12 relative tolerance before partitioning.
Hydrostatic/deviatoric contributions and all spatial bins must separately
recompose the complete squared norm. Every nonzero body normal-force sample
from both compared meshes defines the reference-coordinate distance cloud;
the component threshold does not delete a force sample.

|Adjacent pair|Squared change within 250 µm|Squared change beyond 500 µm|Deviatoric fraction of complete squared change|
|---|---:|---:|---:|
|31668 → 64716|95.353223%|0.179734%|73.751461%|
|64716 → 171228|96.169048%|0.193371%|76.861732%|

The 100–250 µm bins alone contain 57.73% and 69.99% of the squared
changes. Beyond 500 µm the region-relative tensor changes are 0.467849%
and 0.534137%, respectively. This is a measured localization of the original
diagnostic; discarding the contact-near field would change the question.

There are **zero original probes within 25 µm** of either union sample cloud.
The finest marked surface edges are approximately 8.6 µm, so this sampling
cannot resolve the innermost contact field. Distances refer to original
sample points, not a continuous patch boundary or the current finger edge.
Seed-volume weights are not exact cross-mesh integration. These limitations
prevent a continuum convergence or singularity claim from this analysis.

## Actual production replay and the next isolated control

Actual Quadrants replays 719344 and 719345 pass all original complete
equilibrium, corner-component, full-tensor and raw-peak gates on 31668/64716
tetrahedra in both full-inertia and rigid-relief modes. Their iteration counts
are 16191/1670 and 17096/2515. Maximum independent force-budget fractions
are 0.029525 and 0.034049; maximum corrected component-budget fractions are
0.001606 and 0.002119. The previously verified extended polynomial reference
is used, with legacy float64 component differences retained separately.
These are actual resolved-load native recoveries, conditional on unresolved
physical contact/spatial settings, not a feedback or warmed-cost measurement.

Finest native replay 719430 is a separately tracked run. Its scheduler state,
closed source and complete gates must be checked before counting it as evidence.
The initial scheduler dependency rejection is retained; completed parents were
independently checked before submitting without those expired dependencies.

Fixed-geometry controls separately declare contact distances 2.5/1.25/0.625 µm
on the identical 31668-tet body and original finger geometry. The two new input
preparations 719416/719419 generate **zero physical states**. They supply only
configuration to fresh original 64-control cold closure and five-step producers
719417/719420. Original failed closure metadata remains visible. Independent
five-state audits 719418/719421 precede complete source attribution 719439 and
fixed-mesh corner/quadrature/material-frame comparison 719440. Pending controls
do not establish regularization or physical convergence, and no material,
geometry, friction, timestep or force budget is tuned to manufacture a pass.

## Figure and reproducible evidence

The six-panel figure compares all three outer-front boundary maps on a common
0–60 MPa scale, with identical 0.4 mm peak-region insets. Face colors show the
maximum of the three original element-side boundary corner values. Raw
whole-volume peaks remain separate starred quantities. Subsequent panels
retain growing adjacent tensor/peak changes, both source concentration curves
and cumulative complete squared-error localization.

Caption: Covered CAD contact refinement at 6.25 ms. (a–c) Matched boundary
corner diagnostics and actual nonzero source-2 normal-force samples (white);
stars mark independently found whole-volume raw maxima. (d) Adjacent tensor,
raw-peak and energy changes; dotted gray line is the predeclared initial 3%
diagnostic. (e) All-node maximum discrete body normal-force density by captured
finger source. (f) Cumulative original weighted squared tensor change versus
distance to the union of original normal-force samples. Every one of the
158340 probes is retained, but none lies within 25 µm. These deterministic
fields have no replicate uncertainty or fitted convergence order.

Placement: beside the three-level convergence table, before interpreting any
conditional linearization/native agreement. The strongest observed stability
is away from the sample clouds; the largest differences remain contact-near,
where the sampling and physical contact tail remain unresolved. This supports
prioritizing separated contact/volume/geometry and regularization controls.

Alt text: Three egg stress maps show a narrowing concentrated contact region
and increasing raw maximum with refinement. Tensor differences exceed the
initial target while energy changes remain small. Discrete contact force
densities increase, and most weighted tensor differences occur within
250 micrometres of normal-force samples.

Figure runs retain the actual first rendering failure, the accepted editable
preview, and the final SVG/PDF/1000-dpi PNG. Rendered text and preview inspection
pass. Final SVG has 88 editable text nodes and no embedded bitmap. PDF embeds
two decoded TrueType font streams and ToUnicode maps; PNG is 7204 × 5944 RGB(A)
pixels at approximately 1000 dpi. Source CSVs, figure contract, caption and
all failed/successful source closures accompany the
[immutable contact-tail publication](../evidence/20261014-physical/covered-cad-contact-tail-v1/manifest.json).
Exact commands are retained in its scheduler records and source/diagnostic/
localization/figure journals. The reproducible entry points are
`scripts/audit_covered_contact_sources.sh`,
`scripts/analyze_covered_contact_sources.sh`,
`scripts/localize_covered_tensor_changes.sh` and
`scripts/plot_covered_cad_tail.sh`.

All original A/B/C finite-load/model/island, physical contact/geometry/volume,
complete egg/interior-hotspot application, feedback and warmed accuracy/cost
requirements remain open. Faithful source attribution and numerical agreement
do not substitute for those physical acceptance gates.
