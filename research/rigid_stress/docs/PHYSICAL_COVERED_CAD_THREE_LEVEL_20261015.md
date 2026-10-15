# Three covered CAD levels: a growing physical field discrepancy

The [full physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. All three covered CAD egg references now have independently
audited five-state dynamic windows. At the common final time of 6.25 ms,
the adjacent tensor change increases from 9.68% to 10.72%, and the raw
whole-volume peak change increases from 13.89% to 15.28%. This sequence
does not pass the predeclared 3% initial diagnostic or establish a credible
physical tail. Small energy and displacement changes do not close it.

## Original fine equations and independent evidence

Producer 719100 continues the closed 171228-tet case from its original held
initializer, using the previously qualified exact eight-coordinate Schur
factorization. Every initializer/geometry/map file copied from the original
failed factorization case retains its original SHA256. There were no saved
dynamic prefix states to replace. All five original dynamic steps complete;
their physical timestep, material, density, friction, jaw control, Newton
target and original force budgets remain unchanged.

Independent auditor 719101 reconstructs current normal/friction forces, body
inertia, gravity, jaw actuation and complete constrained balances for every
saved state. Final body/mapped residuals are 5.426195e-14/1.443373e-13 N.
Coverage audit 719102 finds all 129 final active body samples inside the
declared 500 µm marking caps, with final marked edge-length percentiles
approximately 8.595/8.623/8.690 µm. These are contact-sample coverage and
numerical mesh diagnostics, not independently converged contact pressures.

The original 64 held initializer increments have accepted producer records
and a byte-preserved final closure. Full initializer fields were not saved
at every increment, so this checkpoint does not claim an independent
all-increment field audit. The original failed full factorization remains
retained in the parent evidence.

## Every original common probe and raw maximum remains

Comparison 719103 retains all 158340 original physical probes and their
original five-per-seed-tetrahedron volume weights. Each mesh's point evaluator
passes its quadratic-polynomial and finite-rotation qualification before
evaluating the nonlinear stress in its independently fitted material frame.
The weights define a seed-volume diagnostic norm, not exact integration of
the discontinuous cross-mesh difference. Full element-side corner tensors,
global extrema bounds, locations, objective displacement, force arrays and
energies remain archived.

|Tetrahedra|Raw whole-volume peak lower–upper bound (MPa)|Nonlinear energy (µJ)|Active body normal samples|
|---:|---:|---:|---:|
|31668|40.400487–40.400487|0.963624|32|
|64716|46.917667–46.917667|0.967558|61|
|171228|55.382574–55.415414|0.974115|129|

|Adjacent meshes|Tensor diagnostic change|von Mises diagnostic change|Raw lower-peak change|Energy change|Objective displacement change|
|---|---:|---:|---:|---:|---:|
|31668 → 64716|9.680122%|6.339263%|13.890674%|0.406612%|0.021481%|
|64716 → 171228|10.720876%|6.271904%|15.284424%|0.673161%|0.032006%|

The absolute weighted tensor differences are 13.248508 and 14.596638
Pa·m^(3/2). Bulk, body-surface geometry and contact trace vary together.
Density is fixed; represented mass and reference geometry are measured
separately. Neither three levels nor the more stable scalar observables
establish isolated mesh convergence or continuum contact accuracy.

## Independent linearization is a separate, much smaller conditional error

Audit 719220 independently assembles the linear FEM stiffness, consistent
mass, rigid modes and resolved full-inertia loads at each final state.
Complete volume-quadrature tensor fields are compared against the objective
nonlinear fields. Frozen-objective constitutive changes and rigid-relief
loads remain separate diagnostics rather than interchangeable problems.

|Tetrahedra|Resolved linear/nonlinear volume tensor change|Frozen-objective constitutive change|Resolved linear raw peak (MPa)|Complete linear residual (N)|
|---:|---:|---:|---:|---:|
|31668|0.143684%|0.313597%|40.363746|4.773035e-12|
|64716|0.145668%|0.313633%|46.873744|4.857297e-12|
|171228|0.146786%|0.313471%|55.337980|4.700027e-12|

The finest absolute volume tensor difference is 0.214095 Pa·m^(3/2).
The finest raw linear/nonlinear peak difference is 0.080520%. Reference RHS
balance corrections of approximately 8.4e-9 N remain explicit outputs;
the independently solved residual is reported after that declared correction.
The full-inertia/rigid-relief volume tensor differences are approximately
6.8–7.4e-10 relative. None of these small conditional discrepancies resolves
the 10.72% spatial/contact difference or measures rigid/deformable feedback.

## Reproduce and assess the checkpoint

Exact scheduler commands, environments, source closures and actual input
hashes are in the [immutable publication](../evidence/20261014-physical/covered-cad-three-level-v1/manifest.json).
Its immediate parents retain the preceding covered fields and qualified
reference-solver attempts. Every newly archived array is retained, with
large originals reconstructed from exact bounded byte ranges and checked
against the actual runtime originals.

For the comparison and linearization, use the recorded command journals
`meta/fine-schur-paired-continuation-commands-v1.json` and
`meta/covered-cad-three-level-linearization-commands-v1.json` under
`runs/20261014-physical-validation/`. Their wrapper entry points are
`scripts/compare_covered_cad_fields.sh` and
`scripts/audit_resolved_linearization.sh`; the continuation uses
`scripts/run_schur_metis_paired_reference.sh` with the retained final closure.

The next physical evidence must explain the contact/spatial discrepancy.
Source-resolved normal/shear forces, all original local wrench components,
contact feature geometry, and fixed-mesh contact-distance controls address
that question. Pending controls are excluded from this accepted checkpoint.
All original A/B/C load-model/island, physical/contact/geometry/history,
egg/interior-hotspot application, native replay versus feedback, and warmed
cost at demonstrated accuracy requirements remain open.
