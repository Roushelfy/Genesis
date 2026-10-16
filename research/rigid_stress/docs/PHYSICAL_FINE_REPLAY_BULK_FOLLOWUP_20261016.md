# Finest native replay and the next fixed-traction bulk level

The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains
active. Actual native resolved-load recovery now passes its unchanged numerical
gates on all three covered CAD meshes, including 171228 tetrahedra. This closes
the finest conditional native replay; it does not close the independently
measured 10.72% spatial/contact discrepancy.

## Actual finest Quadrants recovery

Job 719430 completes with scheduler/wrapper code 0 and matching source closures.
Both modes use the same independently resolved final state at 6.25 ms, original
material/geometry and original numerical tolerances. The iteration cap remains
100000. Original reference RHS balance corrections remain explicit in the
[three-level linearization report](PHYSICAL_COVERED_CAD_THREE_LEVEL_20261015.md).
Full-inertia and rigid-relief loads define separate declared linear problems.

|171228-tet mode|Iterations|Independent complete residual (N)|Force error / original budget|Largest corner component error / original budget|Native/independent complete corner tensor difference|
|---|---:|---:|---:|---:|---:|
|Full inertia|18688|4.600759e-11|0.044182|0.003922|1.448810e-11|
|Rigid relief|3511|2.759528e-11|0.026501|0.003920|8.598068e-12|

Every original force/gauge, cached-corner component, whole-field tensor and
raw-global-peak gate passes. The component oracle is the previously qualified
extended exact-polynomial/cofactor evaluator; legacy float64 comparisons remain
separate retained outputs. Both raw native peaks are approximately 55.337980 MPa,
versus the nonlinear raw lower bound of 55.382574 MPa. The native/nonlinear
complete volume tensor discrepancy is approximately 0.146786%, and the raw peak
discrepancy is approximately 0.080520%.

Together with the [two earlier actual native replays](PHYSICAL_COVERED_CAD_CONTACT_TAIL_20261015.md),
this gives six successful original-load-mode replays across 31668/64716/171228
tetrahedra. Compilation, reference reconstruction and the numerical solves are
included in these job runtimes; no warmed production throughput is inferred.
Same-load agreement and conditional linearization do not establish continuous
pressure, physical spatial convergence or rigid/deformable feedback.

## Separate frozen-traction bulk evidence and its remaining tail

The three-level fixed-domain/fixed-traction runtime study now completes both
516287/554574-tet producers 718587/718588 and independent auditor 718627.
Together with the original 502230-tet field, all 15 complete fields pass fresh
stiffness/consistent-mass/RHS, full force budgets, corner fields,
original 36 probes, energies and raw extrema. Maximum original force-budget
fraction is 0.053535. The independent nested-P2 prolongation is checked against
all ten quadratic monomials. Exact common-volume affine tensor integrals and
nested energy identities retain explicit residual-work bounds.

|Load case|First exact-volume tensor change|Last exact-volume tensor change|Last original-probe tensor change|Last raw-peak change|Last energy change|
|---|---:|---:|---:|---:|---:|
|Opposing|2.498647%|2.012212%|2.222124%|0.435113%|0.042820%|
|Bending|2.514568%|2.007569%|2.344858%|0.745328%|0.042574%|
|Asymmetric|2.578322%|2.025002%|2.719377%|0.905914%|0.043343%|
|Near slip|2.498845%|2.002964%|2.092069%|1.154417%|0.042275%|
|Multiple|2.306961%|1.857740%|2.174069%|0.112320%|0.036321%|

These initial changes are below 3%, but several original probe differences
increase, and some raw peaks are nonmonotone. They do not yet establish a
credible physical tail or justify replacing the original probe/peak criteria
with the supplemental volume norm. The geometry is the unchanged polyhedral
domain and the physical traction quadrature remains byte-identical. Continuous
geometry, finite-footprint integration and solved contact remain separate axes.

An audit-scope review finds that 718627 does not explicitly compare all finer
integration-point stress components against the retained affine corner field.
The original five-case baseline load qualification does perform that check,
but its scope must not be extended to the ten newer fields by implication.
New auditor 719487 will recheck all 15 saved fields with direct scikit-fem
quadrature gradients and the unchanged 1e-10 relative / 1e-6 Pa absolute
tensor comparison, alongside all existing gates. Until it closes successfully,
explicit complete quadrature coverage of this 15-field study remains pending.

This checkpoint retains the independently reverified runtime summary and input
closure metadata for that 15-field study. Its complete large-array immutable
publication remains outstanding; the original producer/auditor arrays remain
intact on the data mount. Metadata alone is not represented as a full published
reference dataset.

## A fourth level preserves the original three exactly

Mesh job 719471 adds a 653088-tet local level at the original numerical 3 mm
marking radius and physical anchors. Actual levels 0/1/2 reproduce the original
502230/516287/554574-tet NPZ files byte-for-byte. All positive refinement levels
pass child containment and per-parent volume preservation; the fourth maximum
parent-volume relative difference is 1.439824e-13. The original seed has no
parent partition, a metadata distinction explicitly retained in the check.
Geometry, surface area, material, mass and the physical load law stay fixed.

The auditor CLI now accepts two or more refined levels and explicitly checks
the complete quadrature tensors described above. Qualified `prolongation`,
`volume_squared` functions and independent FEM source remain exactly the
qualified versions. Existing live source closures were checked before editing;
the four-level auditor is still dependency-pending and will snapshot the
extended source at launch. These reference checks change no production engine
code or original acceptance tolerance.

Producer 719476 starts the new fixed-traction solve only after the mesh's actual
successful closure and original-prefix verification. Dependent auditor 719477
will freshly verify all 20 complete fields, every original probe/raw maximum and
the nested common-volume/energy comparisons. Pending jobs do not establish a
fourth accepted equilibrium field or a converged tail.

The fixed 31668-tet contact-distance controls and their independent complete
source/field comparisons remain separately tracked as jobs
719417/719418/719420/719421/719439/719440. Their original 64-control cold closures,
five dynamic states, geometry, friction, timestep and force budgets remain.
No partial held initializer or pending analysis counts as physical acceptance.

## Reproduction and scope

The [immutable finest-native follow-up](../evidence/20261014-physical/covered-cad-finest-native-v1/manifest.json)
retains both full native fields, original source closures, complete four-level
mesh generation, exact old/new prefix identities and actual scheduler commands.
Immediate parent publications retain the matched independent nonlinear/linear
fields and preceding source/localization/figure evidence. Large files use exact
bounded byte ranges verified against the actual runtime originals.

Commands are in `meta/covered-cad-native-linearization-level3-commands-v1.json`,
`meta/frozen-traction-fourth-local-bulk-mesh-commands-v1.json` and
`meta/frozen-traction-fourth-local-bulk-commands-v1.json` under the existing
physical-validation run root. Entry points are
`scripts/replay_resolved_linearization.sh`,
`scripts/refine_thickness_tail_regions.sh`,
`scripts/frozen_traction_bulk_reference.sh` and
`scripts/audit_frozen_traction_bulk_axis.sh`.

Every original A/B/C finite-load model/local-island, physical contact/geometry/
volume/history, egg/interior-hotspot application, feedback and warmed
accuracy/cost requirement remains active. No overall completion is claimed.
