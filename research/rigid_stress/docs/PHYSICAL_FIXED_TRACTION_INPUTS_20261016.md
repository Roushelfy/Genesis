# Complete frozen-traction qualification inputs

This checkpoint fills the full-array publication gap for the fixed-domain
bulk study. The [complete physical objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active; preserving the original traction law is a prerequisite for
interpreting the new bulk fields, not physical convergence by itself.

Qualification 718586 uses the original 502230-tet hollow-egg mesh and all five
original cases: opposing grip, bending, asymmetric load, near-slip friction and
multiple patches. Physical quadrature positions, area weights, tractions and
all 36 original probes remain unchanged. Boundary ownership is resolved
without moving a quadrature point. All original nodal loads, full RHS, complete
force budgets, corner/von Mises fields and complete quadrature tensors pass
before any finer bulk solve. The original generic element-finder failure
718574 retains its actual failed wrapper, log and closed source.

Maximum original load/RHS difference is 4.336809e-18 N; maximum complete
quadrature tensor difference is 7.071936e-8 Pa. The largest complete
force-budget fraction is 0.028813477. Maximum original-point boundary-plane
distance is 3.602424e-18 m, with maximum quadratic virtual-work difference
1.016440e-20 N·m². These are numerical preservation checks of the declared
finite quadrature, with no redundant same-matrix reference solve.

The [immutable input publication](../evidence/20261014-physical/frozen-traction-inputs-v1/manifest.json)
retains all five complete qualified field arrays, source snapshots, actual
failure and successful scheduler commands, versions and hash closures.
The independently reverified five-level-thickness and local-mesh parents
retain the original reference fields, physical traction samples and
partition-checked geometry. Large arrays reconstruct from exact bounded byte
ranges and are checked against their actual runtime originals.

Use the recorded `meta/frozen-traction-local-bulk-commands-v2.json` command
journal under `runs/20261014-physical-validation/`, with
`scripts/frozen_traction_bulk_reference.sh --qualify-only` and the original
parent mesh/FEM/probe paths. The immutable journal includes the full positional
wrapper arguments and scheduler/container invocation.

The finer 516287/554574-tet producer fields, complete common-volume/energy
audit and added explicit finer quadrature audit have their own publication
steps. The new 653088-tet solve and dependent 20-field audit remain separately
tracked. Fixed physical quadrature does not establish continuum traction
integration, geometry, solved contact, load-model/local-island accuracy, full
application feedback or warmed cost at demonstrated accuracy. Every original
A/B/C requirement remains in scope.
