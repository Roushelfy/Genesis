# Fixed-domain, fixed-traction bulk fields and independent checks

The local bulk study retains the same physical polyhedral egg domain, material,
mass and original finite traction quadrature. The 516287/554574-tet producers
718587/718588 extend the original 502230-tet field without moving a quadrature
point, changing the load profile, or replacing any of the original 36 probes.
The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains active.

The [complete producer-field publication](../evidence/20261014-physical/frozen-traction-bulk-fields-v1/manifest.json)
retains every displacement, full force/RHS, full corner tensor/von Mises field,
raw global maximum/location, energy and original physical-probe field for all
ten new fields. It references the [full input qualification](PHYSICAL_FIXED_TRACTION_INPUTS_20261016.md)
and reverified original thickness/local-mesh parents. The successful original
traction-preservation qualification precedes both actual finer solves.

Independent audit 718627 reconstructs stiffness, consistent mass, balanced
loads, complete residuals, all corner fields/probes, exact common-volume tensor
integrals and nested energy identities. Follow-up 719487 closes the identified
finer-quadrature coverage gap on every one of the same 15 saved fields:
778680045 individual tensor components are directly reconstructed at integration
points and compared against the affine corner field at the unchanged
1e-10 relative / 1e-6 Pa absolute comparison budget. Maximum absolute
quadrature difference is 3.536356e-7 Pa; maximum original force-budget fraction
is 0.053535. All ten complete nested-difference NPZ files are byte-identical
between the two audits. The new checks preserve the original numerical results.

|Load case|First exact-volume tensor change|Last exact-volume tensor change|Last original-probe tensor change|Last raw-peak change|
|---|---:|---:|---:|---:|
|Opposing|2.498647%|2.012212%|2.222124%|0.435113%|
|Bending|2.514568%|2.007569%|2.344858%|0.745328%|
|Asymmetric|2.578322%|2.025002%|2.719377%|0.905914%|
|Near slip|2.498845%|2.002964%|2.092069%|1.154417%|
|Multiple|2.306961%|1.857740%|2.174069%|0.112320%|

Common-volume tensor integration is exact for the retained affine stress
differences on the nested finer tetrahedra. Complete quadratic-polynomial
prolongation checks and explicit residual-work bounds qualify the nested
energy comparison. These supplementary volume/energy metrics do not replace
the declared original probe/peak criteria. Several original probe differences
increase and some raw peaks are nonmonotone, despite changes below the initial
3% diagnostic; the physical tail is not accepted.

The [independent large-array audit publication](../evidence/20261014-physical/frozen-traction-volume-audit-v1/manifest.json)
retains both original and explicit-quadrature audit outputs, the manufactured
6/12-tet control, closed sources and commands. All 20 complete old/new large
files reconstruct from exact bounded chunks; identical arrays share stored
bytes without removing either original identity. Manifest SHA256 is
`8e57b63ded4a0aaca49cf37656446518b82e00c75dd76ee889ac1523144b8c9f`.
Neither a producer residual nor publication of a mesh alone substitutes for
that independent scope. The 653088-tet fourth local level, with the original
three meshes byte-preserved, remains under producer 719476 and subsequent
20-field audit 719477. Their pending states count as scheduled work only.

Exact submitted commands are retained in
`meta/frozen-traction-local-bulk-commands-v2.json`,
`meta/frozen-traction-local-bulk-audit-commands-v1.json` and
`meta/frozen-traction-explicit-quadrature-audit-commands-v2.json` under the
existing physical-validation run root. The opt-in wrappers are
`scripts/frozen_traction_bulk_reference.sh` and
`scripts/audit_frozen_traction_bulk_axis.sh`.

This study isolates elastic bulk/trace basis refinement with fixed physical
quadrature on the same represented domain. It does not improve that domain's
CAD approximation, establish continuous traction integration or solved-contact
pressure convergence, measure load-model/local-island bias, complete application
feedback, or establish warmed native cost at demonstrated accuracy. All original
A/B/C requirements remain open where not separately verified.
