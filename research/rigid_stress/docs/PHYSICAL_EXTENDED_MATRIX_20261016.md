# Fresh finite-load field audits and two additional CAD levels

The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains
active. This checkpoint freshly audits 30 previously published finest reference
fields and adds 12 meshes for the unresolved convex/nonconvex tails. The old
six-level curves, original raw maxima and all five original observables remain.
Mesh generation and qualification create zero new equilibrium states; finer
physical conclusions require the separately scheduled solves and audits below.

## Close the original audit's actual scope

The [original fine-reference publication](PHYSICAL_FINE_REFERENCES_20261014.md)
reconstructs raw corner von Mises arrays, extrema and the work identity, and
retains the producer's complete residual. It does not independently reassemble
stiffness to verify that residual. The new qualification closes that gap for
the finest existing level of each of the six original shape families. Its scope
is these 30 fields, rather than every field on every historical level.

Jobs 719747–719752 use isolated source snapshots with no production numerical
backend. They freshly assemble scikit-fem stiffness, consistent mass, six rigid
modes and inertia-relieved RHS; regenerate every original continuous finite
traction vector; check the complete original force budget; and reconstruct every
corner and integration-point tensor. Original physical probes, validity masks,
von Mises fields and consistently gauge-aligned displacement are also checked.
The unchanged original `finite_traction_reference.py` and `skfem_reference.py`
must match the producer snapshots byte-for-byte before qualification.

|Original finest shape|Tetrahedra|Qualified cases|Largest original force-budget fraction|
|---|---:|---:|---:|
|Ellipsoid|41566|5|0.000094061|
|Capsule|43481|5|0.000098495|
|Thin neck|27356|5|0.002426765|
|Rounded U|40824|5|0.026137195|
|Narrow gap|62513|5|0.005532498|
|Cavity shell|67035|5|0.000094633|

All 30 full nodal vectors are reproduced exactly. Every original physical
probe remains inside its original finest mesh; no point is moved, removed or
replaced. The 139973625 explicit quadrature tensor components pass unchanged
1e-10 relative / 1e-6 Pa absolute comparisons against the affine corner field;
maximum discrepancy is 7.4081874e-8 Pa. Worst complete equilibrium residual
consumes 0.026138 of its original budget. Qualification reads saved displacements
and solves no new equilibrium equation, so numerical rechecking does not count
as a new physical mesh result or a converged tail.

## Extend the existing continuum families, loads and probes

Jobs 719739–719744 construct two finer meshes for every family using the
**byte-identical original CAD generator**. All geometries retain their original
full three-dimensional dimensions and construction, including cavity shell
layers. Every tetrahedron has positive volume; minimum recorded scaled Jacobian
across the 12 assets is 0.171463. These mesh diagnostics do not certify their
stress accuracy.

|Shape|Original finest tetrahedra|New level 1 tetrahedra|New level 2 tetrahedra|New Gmsh size parameters|
|---|---:|---:|---:|---|
|Ellipsoid|41566|111286|315446|0.000625 / 0.0004375 m|
|Capsule|43481|115768|326676|0.000625 / 0.0004375 m|
|Thin neck|27356|73158|205437|0.000625 / 0.0004375 m|
|Rounded U|40824|109020|294458|0.0004375 / 0.0003125 m|
|Narrow gap|62513|166209|446367|0.0004375 / 0.0003125 m|
|Cavity shell|67035|208428|483087|0.04 / 0.03 normalized sphere coordinates|

Each field uses the same five original cases: opposing grip, bending,
asymmetric force with original angular velocity, near slip, and multiple patches.
All original physical centers, 3 mm radius, compact profile, strengths,
friction, material, density, pose and omega remain. Original probe coordinates
and outside indices remain explicit. Independent CAD remeshing changes the
represented volume/mass approximation; it does not change the continuum density.
These curves couple volume and geometry, so the separate fixed-domain volume,
geometry, quadrature and intrinsic-distance requirements still apply.

The isolated pure-FEM entry point `extend_finite_load_reference.py` must pass
the closed five-case qualification on the matching original mesh before finer
use. Qualified input hashes and relevant helper sources must match. New solves
reuse the previously qualified AMG-CG reference method, retain every full
displacement/load/RHS/corner/von Mises array and physical traction sample, and
check the original complete force and corner/quadrature budgets. This is an
independent reference; production arithmetic remains in Quadrants.

## Declared interpretation and separately scheduled independent audits

The original initial engineering diagnostic remains 3% for original common
full-tensor/von-Mises probes, consistent-mass gauge displacement, raw unaveraged
global peak and strain energy. The new three-level tail consists of the old
finest mesh and both additions. Both latest adjacent changes are recorded for
all five observables, alongside the earlier six-level curves, hotspot positions,
absolute errors near zero and any nonmonotone behavior. A single last adjacent
pass or stable peak is insufficient; a diagnostic pass alone is not a bound on
continuum error, load-model accuracy or a physically verified native cost.

|Shape|Five-case producer jobs, levels 1 / 2|Fresh three-level, 15-field auditor|
|---|---|---|
|Ellipsoid|719755 / 719756|719757|
|Capsule|719758 / 719759|719760|
|Thin neck|719761 / 719762|719763|
|Rounded U|719764 / 719765|719766|
|Narrow gap|719767 / 719768|719769|
|Cavity shell|719770 / 719771|719772|

The auditors separately reassemble all three levels, regenerate every continuous
boundary load and original RHS, check complete original force/corner/quadrature
gates, reproduce all probes and recompute energy/raw maxima and their locations.
They retain complete equilibrium residual vectors and force-reintegration
differences. Producers and auditors must have actual successful scheduler,
wrapper and source closures before new fields are interpreted. Running or
dependency-pending jobs in this table are scheduled work only.

Actual native follow-ups 719781/719782 and 719783/719784 cover ellipsoid and
capsule at both added levels, respectively. They wait for the corresponding
three-level reference auditor, then compare all five original inputs across
pressure/componentwise/pair_union/body_source_prior on the actual Quadrants
path. Ports8, native quadrature10 and original residual/tensor gates remain;
the iteration allowance is100000. Unsupported or rejected inputs remain
classified. Initial runtime measurements include cold setup and do not establish
warmed native throughput. Commands are in
`meta/extended-convex-native-model-commands-v1.json`; they were submitted after
the qualification publication and remain separately pending.

## Reproducibility and remaining objective

The [immutable qualification publication](../evidence/20261014-physical/extended-six-shape-qualification-v1/manifest.json)
retains all 12 closed mesh/qualification jobs, complete mesh assets, the 30-field
audit results, exact scheduler commands, pinned package/hardware details and
source/configuration closures. Its six independently reverified parent manifests
preserve the original full fields and probe arrays. Runtime results and large
arrays stay under the established data root.
Manifest SHA256 is
`5244fc67f41f9ce8de36181019d697f4d3d079d9cc4bc91b7fb0aec74079a58b`;
all 411 archive entries verify against 411 actual runtime originals.

Commands and the predeclared physical plan are in
`meta/extended-six-shape-mesh-commands-v1.json`,
`meta/extended-six-shape-original-field-qualification-commands-v1.json`,
`meta/extended-six-shape-full-fields-commands-v1.json` and
`meta/extended-six-shape-physical-plan-v1.json`. Entry points are
`scripts/generate_physical_matrix_meshes.sh`,
`scripts/extend_finite_load_reference.sh` and
`scripts/audit_finite_load_extension.sh`.

The full egg has its distinct ongoing fixed-boundary thickness and contact
studies; this matrix does not replace it. Native pressure/componentwise/
pair_union/body_source_prior and local-island sensitivity, resolved egg/Franka
and interior-hotspot contact, original full histories, feedback separation,
separated error axes and physically verified warmed costs remain required.
No overall completion is claimed.
