# Fixed-domain local bulk refinement and retained contact histories

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. Four useful2/3/4/5-layer levels now independently retain20 full
FEM fields with identical physical boundaries, mass and traction samples. Their
last tensor changes remain3.18–4.52%. An unchanged-probe diagnosis locates most
of this difference in five physical probe positions. Three local bulk meshes
are built and independently checked as partitions of the same domain; their
solved-field convergence is not yet established.

## Twenty independently checked fixed-boundary fields

Five-layer producer716835 and four-level auditor718556 complete at their actual
zero return codes with matching final source identities. The501k-scale mesh
contains502230 tetrahedra. Together with the previously completed2/3/4 layers,
all five original opposing, bending, asymmetric, near-slip and multiple-patch
loads retain the same original36 material probes, full corner tensors, global
unaveraged maxima, energy and complete FEM force budgets.

| Case |4→5 full tensor change|4→5 von Mises change|4→5 raw global peak change|4→5 energy change|
|---|---:|---:|---:|---:|
|Opposing|3.392043%|1.591384%|0.001342%|0.005737%|
|Bending|3.788795%|1.536118%|0.135353%|0.005729%|
|Asymmetric|3.641747%|3.426246%|0.044497%|0.005757%|
|Near slip|4.521452%|2.072228%|0.048093%|0.005668%|
|Multiple|3.176461%|1.545369%|0.002141%|0.005236%|

Every case still fails the previously declared3% full-observable diagnostic.
Small energy or peak changes do not override the tensor discrepancy. The
separate six-layer producer and subsequent five-level comparison remain
outstanding. The current checks separate thickness resolution from boundary
geometry/mass and traction-quadrature changes; they do not establish complete
three-dimensional bulk convergence or the full contact application.

## Original probes identify the unresolved region

Diagnosis718569 preserves every original probe and component, exporting all180
case/probe rows plus complete3×3 tensors and their differences. It reproduces
the original full-tensor norm exactly. The five largest probe differences
account for97.4253%,98.1422%,98.4114%,98.6695% and98.4885% of the squared
Frobenius difference for opposing/bending/asymmetric/near-slip/multiple loads.
The same five positions dominate all cases, with their ordering changing for
the asymmetric case. No probe is removed from convergence assessment.

The deviatoric contribution to the squared tensor difference is89.1986%,
90.4934%,81.9121%,94.0391% and88.0123%, respectively. Hydrostatic/deviatoric
parts are orthogonal in the tensor Frobenius product; these fractions do not
form an additive decomposition of bulk, geometry or contact error. This result
rules out describing the observed discrepancy as predominantly hydrostatic.
Distances to the retained nonzero traction samples are recorded as diagnostics,
not continuum contact-boundary distances or singularity proofs.

## Three local meshes preserve the complete physical domain

Mesh job718572 applies scikit-fem longest-edge bisection to tetrahedra whose
centers lie within a declared3mm numerical marking radius of the five diagnosed
positions. That radius changes numerical marking, not the physical traction
footprint. Every original seed vertex remains byte-identical. Each child
tetrahedron must lie inside a declared parent, and the accumulated child volume
must recover each parent's volume. Positive orientation, total area, volume and
COM are checked independently.

| Local bulk level |Tetrahedra|Vertices|Maximum parent-volume relative difference|
|---|---:|---:|---:|
|0|502230|100458|Seed|
|1|516287|103376|5.342855e-14|
|2|554574|110449|1.007422e-13|

All levels retain volume3.74246385632928e-6m³ and surface area
0.014973144006272036m² within the recorded arithmetic gates. Density and mass
are unchanged. Boundary triangles can subdivide, so geometric preservation
alone does not establish boundary integration identity.

The new opt-in [frozen-traction reference](../frozen_traction_bulk_reference.py)
therefore reuses the original physical traction positions, area weights and
traction vectors byte-for-byte. It scatters their work into the refined P2
basis, retaining all original36 probes and complete raw fields. Before fine
solves, qualification must reproduce all five original nodal loads, full RHS,
original complete force budgets and full corner/quadrature tensors. No redundant
same-matrix reference re-solve is needed for this qualification.

First attempt718574 fails before a qualified case because the generic volume
element finder rejects a retained boundary quadrature point. Its true failure
and source closure are retained; both unstarted success-only dependents are
cancelled. Replacement718586 uses explicit boundary-triangle ownership while
evaluating the original physical coordinates without relocation. It checks
boundary-plane distance, total force/moment and all18 quadratic virtual-work
components. The same original five-case field/force qualification still gates
fine jobs718587/718588. These jobs are not successful evidence in this report.

## Contact-history recording closes a separate evidence gap only after audit

The earlier U final-contact audit explicitly lacked successful previous friction
states. The isolated [recording patch](../reference_contact_history.patch) and
[recording helper](../reference_contact_history.py) now retain each original
accepted initialization/load stage's current/previous displacement, contact
force, complete gradient, external force, constraint map and exact source/data
hashes. They do not add physical timesteps or change the force equations.

Qualification producer718566 repeats the original845-tet/pad5 physical case.
The first346 recorded residuals match the old records bit-for-bit; differences
begin at347 at approximately1e-13N scale. Their cause is not established. A
promise of residual/iteration bit identity is therefore rejected, not inferred
from unchanged equations. The unstarted strict-identity audit718567 is cancelled
with its accounting retained. No original physical gate is relaxed.

Replacement independent audit718584 must check every retained current normal
and friction force, all original1e-9N constrained balances, complete corner/
quadrature fields and raw whole-volume peak bounds. Original final arrays are
compared explicitly; stress fields retain the previously declared independent
RTOL1e-4/ATOL1e-3Pa criteria. Geometry remains exact and actual differences are
reported. This qualification is unfinished here. No nonexistent old intermediate
full fields are claimed, and quasi-static load-stage records are not physical
time samples. History/load-increment and contact-regularization convergence remain
separate requirements after recording is qualified.

## Reproducibility and scope

The [diagnostic/mesh evidence packet](../evidence/20261014-physical/thickness-local-diagnosis-v1/manifest.json)
passes independent verification of151 archive entries against150 actual runtime
originals. Manifest SHA256:
`1f4644d9ef3628e7041c290e941c256bff9fd98cf02a0e2d7f266d6e577a0dc6`.
It contains all new meshes, all180 diagnostic rows/tensors, input traction samples,
exact commands and closed audit/source records. Complete large original FEM
fields remain at their recorded immutable runtime paths for the separate full
thickness publication; this small packet does not claim to contain those fields.
The packet's unfinished-job observations are historical scheduling records,
not producer completion certificates.

Scripts keep scientific execution on Slurm with explicit data-side output,
temporary and cache locations. New local/frozen-load/history methods are opt-in
reference paths. Production Quadrants and rigid dynamics are unchanged.
The full A/B/C, load-model, physical/contact/geometry/regularization/history,
matched full application/feedback/native replay and warmed accuracy/cost
requirements remain open.
