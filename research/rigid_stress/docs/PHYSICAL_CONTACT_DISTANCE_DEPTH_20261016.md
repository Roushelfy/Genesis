# Covered egg contact-distance controls and shared near-surface tensors

The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains
active. Fixed-mesh contact-distance controls now have complete independent
dynamic force, source and field comparisons. New shared material probes expose
large local differences missed by the original spatial probe cloud. Neither
result establishes a converged contact reference or an overall physical pass.

## What is held fixed and independently checked

The original 31668-tet case uses contact distance 2.5 µm. Producers 719417 and
719420 independently repeat the original 64-control cold initialization and
five dynamic states at distances 1.25 and 0.625 µm. Geometry, material, represented
mass, actual fingers, friction, actuation, supports and 1.25 ms timestep remain.
No physical control or force budget is relaxed. Configuration preparers create
zero equilibrium states; the original closure attempt's failure remains visible.

Auditors 719418/719421 rebuild all five new states' complete original physical
force gates. Jobs 719439/719440 reconstruct all 15 source and field states,
checking exact equality of all geometry/contact-map arrays, full corner and
material-frame integration-point tensors, consistent-mass objective displacement,
energy and raw whole-volume extrema. Source analysis 719503 retains both run
identities and distances even though the tetrahedron counts are identical.
Every listed job is scheduler COMPLETED, wrapper code 0, with matching source
closures. These are short dynamic diagnostics. The existing producer saves only
the final held initializer field, so this is **not** an independent full-field
audit of every one of the 64 initialization controls.

At 6.25 ms the raw global lower bounds are 40.400487, 41.519020 and 42.062803 MPa.
The original upper bounds are retained, approximately 1.7e-7 Pa above their
respective lower bounds. All three extrema occur at the same material position
[2.11469208e-5, 2.22513377e-2, -1.82738433e-3] m. Minimum whole-volume deformation
Jacobian bounds are 0.997601, 0.997552 and 0.997522.

|Final observable|2.5 → 1.25 µm|1.25 → 0.625 µm|
|---|---:|---:|
|Complete volume-weighted tensor change|0.969353%|0.571557%|
|Complete corner tensor change, unweighted|2.405633%|1.393426%|
|Raw global lower-peak change|2.694026%|1.292789%|
|Elastic energy change|0.276369%|0.164414%|
|Consistent-mass objective displacement change|0.028565%|0.013987%|
|Full nodal contact-force vector change|2.135887%|1.261044%|

Volume tensor absolute changes are 1.408464 and 0.831098 Pa m^(3/2). Over all
five states, maximum adjacent volume tensor changes are 0.982041%/0.587914%,
and maximum raw lower-peak changes are 2.722144%/1.340129%. The complete histories
and absolute quantities remain in the archived arrays and JSON, rather than
being replaced by these final-time summaries. Three distances on this coarse
spatial mesh do not establish a hard-contact limit.

## The same 576 near-surface points on every mesh and distance

All original 158340 spatial probes and their original norms remain. None of
those original probes lies within 25 µm of the baseline actual normal-sample
cloud. Job 719502 adds a fixed supplemental set without moving any original
point or replacing the original raw global maximum.

The set uses the original baseline's 32 actual nonzero normal body anchors.
For each anchor Xi, a deterministic incident baseline boundary-facet centroid Cf
and outward normal nf define three tangential mixes α = sqrt(2,3,5)/20 and six
depths d = 1, 2, 5, 10, 20, 50 µm:

`X = (1 − α) Xi + α Cf − d nf`.

The 576 resulting material coordinates are identical on all three meshes;
480 lie within 25 µm of the original baseline normal samples. No point is
filtered, relocated or selected by stress magnitude. Minimum barycentric
coordinates on the three meshes are 0.002567/0.002500/0.004010, so every point
is strictly inside every represented body. Full quadratic-polynomial and
finite-rotation controls qualify each point evaluator. Maximum polynomial
stress error is 2.802e-6 Pa; maximum finite-rotation error is 2.242e-5 Pa.
Both original 158340-point tensor changes are reproduced at the original
1e-12 relative comparison before interpreting the supplemental results.

Job 719513 evaluates the **same** 576 points on all 15 fixed-distance states.
It rechecks exact fixed geometry/maps and reproduces the qualified baseline
final tensor at unchanged 1e-12 relative / 1e-6 Pa absolute comparison. There
are 180 retained source/depth/time comparisons. The table uses both sources,
96 points per depth, at 6.25 ms. These equal-point norms are local diagnostics,
not volume norms. Nominal depth is from the baseline polyhedral facet, not the
exact CAD or current deformed interface.

|Nominal depth|Spatial 31668 → 64716 tets|Spatial 64716 → 171228 tets|Fixed mesh 2.5 → 1.25 µm|Fixed mesh 1.25 → 0.625 µm|
|---|---:|---:|---:|---:|
|1 µm|36.0813%|44.5927%|2.9012%|1.8221%|
|2 µm|36.1585%|44.7308%|2.8991%|1.8197%|
|5 µm|36.4064%|45.1502%|2.8928%|1.8123%|
|10 µm|38.6375%|50.0977%|3.0840%|1.8333%|
|20 µm|36.4875%|68.0848%|2.9773%|1.7518%|
|50 µm|35.0860%|84.3469%|2.9027%|1.6854%|

The largest of all 180 distance comparisons is 3.273467%. Spatial and distance
changes are evaluated at the same points, but the spatial sequence still couples
bulk, represented geometry and contact trace. The coarse distance sensitivity
can interact with spatial error. **Do not add or subtract these changes as a
causal error budget.** Large local differences remain unresolved; no contact-edge
singularity, pressure convergence or acceptable physical tail is inferred.

## Source diagnostics retain local changes beyond net force

At the 1e-8 N diagnostic activity threshold, the final normal body samples are:

|Distance|Source 1 count / lumped area (µm²)|Source 2 count / lumped area (µm²)|Source 1 / 2 maximum discrete density (MPa)|
|---|---:|---:|---:|
|2.5 µm|15 / 13581.5409|17 / 15379.3776|65.5916 / 70.4899|
|1.25 µm|15 / 13581.4985|15 / 13570.1897|67.5530 / 71.9720|
|0.625 µm|14 / 12679.1126|14 / 12665.3508|68.3593 / 72.3943|

These are discrete nodal density and lumped-area diagnostics, **not** continuum
pressure or physical contact area. Every force tail, actual graph component and
normal/shear/total six-component source wrench remains retained. Last adjacent
active-area relative differences are 7.1171%/7.1442%, normalized by the finer
distance's area; these percentages must not be described as coarse-normalized
area decreases. Both fingers still use the original 6 mm crease and 90-degree
incident face planes.

Last adjacent source force changes are approximately 0.014210%/0.014225%, while
the full nodal vector changes 1.261044%. The corresponding source moment changes
about the original reference origin are
[-1.263969e-7, 2.020279e-10, -3.276529e-8] and
[1.256543e-7, -7.997561e-10, 3.277288e-8] N m. Small resultant-force changes do
not certify the full wrench or local load distribution.

## Figure, complete publication and remaining work

Place the [editable three-panel figure](../evidence/20261014-physical/covered-cad-distance-depth-v1/blobs/13d1fb0d3ca730b4ac22a4dc17955254f62d49ae1c6980e9cb934b794e82da63.svg)
after the local depth comparison. Caption: “Fixed material-point tensors and
contact-distance sensitivity at 6.25 ms. (a) Both adjacent spatial changes and
both adjacent fixed-mesh distance changes at the same 576 points, grouped by
baseline-facet depth. (b) Complete volume tensor, raw lower peak, energy and
nodal-force changes on the fixed 31668-tet mesh. (c) Discrete active lumped areas
and sample counts for each source. Equal-point norms and active nodal areas do
not define continuum volume errors or contact areas. The spatial sequence
couples bulk, geometry and contact trace; curves are not additive error terms.”

Alt text: Near-surface spatial tensor changes remain tens of percent, while
fixed coarse-mesh contact-distance changes are a few percent; source active
lumped areas and counts also change with distance. The figure retains separate
symbols and line styles, logarithmic axes with units, and no inferred order,
uncertainty bars or fitted trend.

Figure jobs 719523/719525 retain the inspected preview, source visual-QA record,
three exact data CSVs and figure contract. The 183 × 136 mm SVG has 53 editable
text nodes and no embedded images. The [vector PDF](../evidence/20261014-physical/covered-cad-distance-depth-v1/blobs/0501a862d7c00506a19407af494f40732ef214d0afde431d5f3fcb784de46322.pdf)
contains two decoded TrueType font streams, ToUnicode mappings and no raster
objects. The RGB/RGBA PNG is 7204 × 5354 pixels at approximately 1000 dpi.
Accepted SVG/preview byte identities are preserved through the export run.

The [immutable publication](../evidence/20261014-physical/covered-cad-distance-depth-v1/manifest.json)
has SHA256 `e7e50f37850f96cb890c997bff56220361c40f5ffd5ebc18adac2a88f0282e72`.
Its runtime verifier checks 933 archive entries against 915 actual originals,
reconstructing nine large original files from bounded byte ranges. Parent
publications retain the three spatial fields, source/localization evidence and
the original initializer/reference inputs. No array is omitted to fit a Git
blob limit. Exact scheduler commands, source closures and every physical
configuration are retained through the manifest and runtime run root.

The three spatial contact meshes retain two thickness layers. Their 500 µm wall
therefore has approximately 250 µm radial intervals even where tangential
contact edges are far smaller. A separate fixed-boundary thickness axis is
needed to diagnose the 1–50 µm local response; tangential refinement alone does
not establish radial resolution. This motivates an isolated volume study,
without changing the physical surface or interpreting new assets as solved
states. Full initializer-field coverage, continuous pressure/spatial convergence,
the original A/B/C load-model and island studies, full application/feedback and
physically verified warmed native costs all remain required.
