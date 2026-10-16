# Actual fine native models and independently audited load redistribution

The native capacity repair now reaches complete physical fields on the
111286-tetrahedron ellipsoid and115768-tetrahedron capsule. Producers719912/
719916 and separate auditors719913/719917 naturally complete with matching
source and full isolated-snapshot verification. Each retains all five original
finite-load cases and four models, physical3mm radii, material, pose, angular
velocity, source forces/moments, friction,8 ports and10-point native quadrature.

## Complete numerical evidence and scope

Each geometry retains20 model/case rows. Nineteen fields are accepted and
independently audited. Near-slip `pressure` remains explicitly unsupported
because its original free contact moments are nonzero. No moment, radius,
friction, source or original tolerance is changed to obtain acceptance.

The separate auditors freshly assemble stiffness, consistent mass, rigid modes
and complete inertia-relieved loads. Every saved native displacement is checked
against complete equilibrium, full native corner tensors, independent
quadrature, global peak and energy. Original physical sample coordinates/forces
are independently scattered with the quadratic basis. All38 accepted fields
pass unchanged gates, including427088574 quadrature tensor components. Maximum
complete-force budget fractions are0.01042196 and0.01924415, respectively.
Every rejected/unsupported row remains explicit.

## Conditional model differences

The table reports the original same-mesh full-corner tensor statistic relative
to the independently specified finite-traction reference. It is the Frobenius
norm over all cell-corner tensors, without volume weighting. Native-scattered
loads are independently solved for this comparison; actual raw native fields
receive the separate numerical checks above.

| Geometry and load | Pressure | Componentwise | Pair union | Source prior |
|---|---:|---:|---:|---:|
| Ellipsoid opposing |0.02455%|0.58074%|0.58074%|0.58320%|
| Ellipsoid near-slip |Unsupported moment|3.89510%|3.89510%|4.38311%|
| Ellipsoid multiple |0.02370%|0.57360%|6.95565%|0.57635%|
| Capsule opposing |0.01192%|0.52529%|0.52529%|0.52613%|
| Capsule near-slip |Unsupported moment|5.99359%|5.99359%|5.96666%|
| Capsule multiple |0.01155%|0.52837%|5.23571%|0.52997%|

Comparisons against this prescribed reference combine declared traction-model,
intrinsic-distance and integration effects at these resolutions. They are not
isolated continuum model-error estimates. The reference is a specified finite
traction law, not a resolved deformable-contact solution. No universal model
ranking follows from its resemblance to a particular model's profile.

The componentwise/pair-union comparison holds intrinsic distance, native
quadrature and all inputs fixed while changing the declared support law. In the
multiple-contact ellipsoid, pair union moves approximately0.987mN in the x
component and0.612mN in the z component between the first and third destination
disks. Their original forces are0.05N and0.02N in magnitude. Capsule pair union
moves approximately0.479mN in x between those disks. These actual disks are
disjoint; exported sample identities are destination-disk identities after
packing, and each stored sample contributes to the retained per-island result.
All original aggregate/source wrench acceptance still passes.

Corresponding multiple-load raw peak differences are only1.04766%/1.05328%/
1.04798% for ellipsoid componentwise/pair-union/source-prior, and0.99831%/
0.96404%/0.99617% for capsule. Thus a similar scalar maximum does not establish
similar full fields or local load distribution. Source-prior local wrench
shifts remain measured rather than assumed zero. Self-balanced opposing grips
retain nonzero internal stress despite zero resultant force.

## Reproduction, publication and remaining objective

The complete runs are:

```text
extended-native-ellipsoid-level1-models-v2
extended-native-ellipsoid-level1-independent-audit-v2
extended-native-capsule-level1-models-v2
extended-native-capsule-level1-independent-audit-v2
```

Exact isolated-source commands remain in
`meta/extended-convex-native-model-commands-v2.json`. All successful raw loads,
workspaces, traction samples, physical areas/normals, source/patch identities,
corner/displacement/RHS fields, residuals, energies and failed/unsupported inputs
are retained. Both complete first-level full-array archives have been verified
against every actual runtime original and published in two bounded Git packets
per geometry. Each manifest retains1517 entries representing1479 runtime
originals, including exact byte reconstruction of19 large originals.

| Complete first-level archive | Manifest SHA256 |
|---|---|
| [Ellipsoid](../evidence/20261014-physical/extended-native-ellipsoid-level1-fields-v1/manifest.json) | `165de9fca80d696f6db55a7650dc55f3f8bc82c99a0672ece38e0c97d0026e08` |
| [Capsule](../evidence/20261014-physical/extended-native-capsule-level1-fields-v1/manifest.json) | `39468b14331a49cb35092e7d9bbc5a9e08dc341ea9227d32a296eb0ea5e8a1bf` |

Original failed capacity attempts remain in the separate immutable capacity
archive. The315446/326676-tetrahedron producers719914/719918 and fresh
auditors719915/719919 also now naturally complete, with matching source and
full-snapshot verification. Each adds19 accepted fields and one unsupported
pressure row. Their fresh audits check593353926 and614477556 quadrature
components. Across all four fine runs,76 fields and1634920056 quadrature
components pass the original gates, with all four unsupported rows retained.
Both finest-level full-array archives are now verified and pushed in five
bounded Git packets per geometry. Each retains1619 manifest entries from1479
actual runtime originals and39 exactly reconstructed large original files.
Their manifest hashes are:

| Finest full-array archive | Manifest SHA256 |
|---|---|
| [Ellipsoid](../evidence/20261014-physical/extended-native-ellipsoid-level2-fields-v1/manifest.json) | `4eb8d590dff3e333aeeff254141228b03bf762a8161e81e37a7bd5e197f8c7da` |
| [Capsule](../evidence/20261014-physical/extended-native-capsule-level2-fields-v1/manifest.json) | `6e2d543c11788f678315e8183140c1ad18bd033128913d8e92cf70c7afa1f6fe` |

The conditional comparison table above still describes only the first added
level and its original unweighted statistic.

Matched old-finest spatial runs and same-mesh ports4/8/16, quadrature10/14
controls now complete with separate auditors. The
[matched-axis report](PHYSICAL_MATCHED_NATIVE_AXES_20261016.md) records all228
accepted native fields across twelve runs, plus every unsupported input and
new exact volume-norm comparisons. Exact commands remain in
`meta/native-convex-matched-distance-quadrature-commands-v1.json`. They supply no
physical convergence by themselves. Original CAD opposing raw-peak changes remain
16.75%/14.10%, so these model comparisons do not establish a physical operating
range. [Fixed-polyhedron controls](PHYSICAL_FINITE_PEAK_LOCATION_20261016.md)
preserve the original five-case qualification. Volume/geometry/trace/quadrature
and distance tails, resolved contact, complete egg/Franka/interior-hotspot
histories, feedback separation and physically qualified warmed costs remain
required. Recorded native timings have different cold-compilation and analysis
scopes; they do not establish warmed application performance.
