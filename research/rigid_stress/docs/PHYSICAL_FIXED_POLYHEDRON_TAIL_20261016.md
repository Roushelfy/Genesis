# Fixed-polyhedron finite-load tails and unsmoothed hotspot diagnostics

This checkpoint follows the [2026-10-16 practical goal](PRACTICAL_VALIDATION_GOAL.md).
Its completed checks are accepted within their stated scope; unresolved physical
tails remain limitations or research follow-ups. The full historical matrix
is no longer an engineering-delivery prerequisite.

The [original90-peak diagnosis](PHYSICAL_FINITE_PEAK_LOCATION_20261016.md)
identified subsurface maxima despite unresolved ellipsoid/capsule CAD tails.
This follow-up freezes each original1.25mm polyhedron and refines its volume
and surface basis without CAD projection. All physical3mm profiles, load
strengths/directions, friction, material, pose, angular velocity and original
probes remain. Volume/surface basis and numerical boundary quadrature still
refine together; this is not a pure volume-only study.

## Complete independent fields

Four finer producers720020/720021/720025/720026 and separate three-level
auditors720022/720027 naturally complete. The ten original fields requalify
before the20 new equilibria. All30 original/new fields retain complete fresh
stiffness, mass, inertia-relief, boundary-load, force, corner, quadrature,
probe, energy and raw-maximum checks. Sources and actual recorded inputs close.

Every original nodal reintegration difference is exactly0N. The30 fresh audits
check1116390825 quadrature tensor components. Maximum original force-budget
fractions are0.0001079101 for ellipsoid and0.0001765345 for capsule. No tolerance
or physical input changes.

| Geometry | Three tetrahedron counts | Final original-probe tensor change | Final raw-peak change across five loads | Cases passing all five observables on both pairs |
|---|---|---:|---:|---:|
| Ellipsoid |14916 /119328 /954624|0.28086–0.30728%|1.49131–14.01099%|1/5|
| Capsule |15979 /127832 /1022656|0.46885–0.57794%|0.80179–13.47621%|0/5|

The two-pair3% column remains the previously declared diagnostic, with every
tail explicitly retaining `physical_convergence_accepted=false`. Original
probe positions, validity/outside masks, full raw maxima and energies remain.

For opposing loads, ellipsoid raw peaks are6554.7312,5787.9488 and5168.2029Pa;
capsule values are6082.1083,5983.8745 and5372.1730Pa. The last capsule opposing
change is11.38648%. Thus fixing CAD geometry has not removed the raw-peak
discrepancy, even when the original internal probe norm is below1%. These
results do not establish a continuum peak or explain its cause.

## Actual peak location and cell diagnostics

Separate job720202 checks all30 recorded raw peaks against both the actual
saved whole-field maximum and the separate auditor's exact value/location.
It retains incident boundary normals, original prescribed tractions, algebraic
compatibility controls, every tetrahedron's mean-ratio quality, and every
unsmoothed incident-cell tensor at each peak coordinate.

The finest opposing ellipsoid peak lies519.074um below the material boundary;
capsule lies662.715um below it. Their peak-cell mean-ratio qualities are0.60031
and0.61702. The largest same-coordinate incident-cell tensor jump is9.78166%
and8.58804% of the respective peak tensor norm. All original incident tensors
remain; no mean or quantile replaces the raw maximum.

Near-slip peaks remain on the boundary. The symmetric-traction algebraic
controls and their limits follow the original report: a local compatibility
defect does not prove a singularity, and zero defect does not rule one out.
Cell quality and same-vertex jumps also do not establish a causal explanation
or a continuum error budget. Their measured values motivate a more local
mesh-resolution control without changing the physical load law.

## Fixed-domain local follow-up

Mesh-generation jobs720218/720219 also naturally complete, with matching source
and input-source closures. Marking includes every one of the five audited
finest global maxima and every original physical load anchor. Its3mm numerical
marking radius is a mesh choice; the physical3mm load footprint is retained as
a separate parameter. Each next level uses two longest-edge refinement passes.

| Geometry | Local three-level tetrahedron counts | Largest checked parent-volume relative difference |
|---|---|---:|
| Ellipsoid |954624 /1243613 /3799085|6.67e-14|
| Capsule |1022656 /1315018 /3909062|4.69e-14|

Every child tetrahedron is contained in a declared parent and every parent
volume passes the existing1e-10 relative check. Original vertex coordinates,
total volume and area pass their original1e-12 relative checks, and COM passes
the1e-14m absolute check. Parent identity arrays are saved for every pass.
These mesh assets solve zero new equilibria.

Fresh original-field qualifications720236/720240 now naturally complete with
matching source/input-source closures. All ten original million-cell fields
retain exact original nodal loads and unchanged complete force/corner/
quadrature/probe checks before local producers720237/720238/720241/720242 and
separate auditors720239/720243. After the user changed priorities, these four
new producers and two pending auditors were cancelled; partial outputs remain.
No new finer equilibrium or convergence result is accepted. Their state checks
and reason are in `meta/application-first-priority-job-cancellations-v1.json`.
The actual original probe NPZ is read using
its recorded `probe_field.npz` name; the new path option changes no numerical
algorithm, source value, physical input or tolerance. The
[complete local setup archive](../evidence/20261014-physical/fixed-polyhedron-local-hotspot-setup-v1/manifest.json)
retains all six meshes, every parent map and the ten original requalifications.
All191 entries verify against187 runtime originals. Manifest SHA256:
`90f4cecc07d56f6bec7404c57dcbb645007d3107445c298fc59de009026535ef`.

Exact commands remain under the established runtime root:

```text
meta/fixed-polyhedron-convex-axis-commands-v1.json
meta/fixed-polyhedron-raw-peak-cell-diagnosis-command-v1.json
meta/fixed-polyhedron-local-hotspot-meshes-commands-v1.json
meta/fixed-polyhedron-local-hotspot-field-commands-v1.json
```

The complete30-field and diagnostic archives are byte-verified against actual
runtime originals. Both complete field archives have been pushed; the compact
diagnostic archive accompanies this source checkpoint:

| Archive | Manifest SHA256 |
|---|---|
| [Ellipsoid fields](../evidence/20261014-physical/fixed-polyhedron-ellipsoid-fields-v1/manifest.json) | `03faaca3f06bf9ce8af7cf0d00729f9cfc91e14fa4d9afc1a64fe6d37d11c06e` |
| [Capsule fields](../evidence/20261014-physical/fixed-polyhedron-capsule-fields-v1/manifest.json) | `18b4a44edd673718ae288569f5f04d9d974e6e6700c4232e9f0b131865977d27` |
| [Every peak/cell diagnostic](../evidence/20261014-physical/fixed-polyhedron-peak-cells-v1/manifest.json) | `3eff81a2ce6d5cf21b4272da460baf20f9e8952b17158770316115fdef450431` |

Both field archives retain273 entries and228 actual runtime originals each,
with five reconstructed large original files each. All90 diagnostic entries
match actual runtime originals. No field or incident-cell array is omitted.

The generic unchanged field auditor's historical limitations text mentions CAD
refinement; hashed fixed-polyhedron mesh metadata and the complete geometric
partition checks define this actual family. The original CAD curves remain
separate. Further universal refinement is a research follow-up. Representative
R/L/M reconstruction, feedback, stress/traction maps and measured settings
remain the application-delivery work under the latest practical goal.
