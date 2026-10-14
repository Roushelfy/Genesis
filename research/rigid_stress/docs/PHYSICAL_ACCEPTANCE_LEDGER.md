# Active physical-validation acceptance ledger

This ledger follows [LOAD_MODEL_PHYSICAL_VALIDATION.md](LOAD_MODEL_PHYSICAL_VALIDATION.md)
without narrowing its scope. Historical native performance completion does not
establish the new physical stage. Dated reports preserve their observations;
running or pending jobs count as scheduled work only.

| Requirement | Evidence needed for completion | Current evidence and remaining work |
|---|---|---|
| A: independent elasticity and boundary loads | Independent K, consistent mass, six rigid modes, boundary quadrature, full equilibrium/energy/gauge displacement/tensor/peak; common physical probes across independent meshes | Independent scikit-fem assembly, analytic fields, native CPU/CUDA algebra and continuous-boundary transfer are qualified. Seven-shape physical reference sweep completed. Representative continuum tails remain incomplete; algebra agreement alone does not satisfy A. |
| B: identical-input model sensitivity | All applicable pressure/componentwise/pair_union/body_source_prior comparisons with fixed points, six-component wrenches, pose, omega, material, radius and friction; per-source/island loads, support areas, concentration, energy and full stress | Finite-traction matrix, fixed-wrench alternative profiles, changing histories and native port sweeps are published. Guarded resolved-U replay adds actual contact source wrenches and measured redistribution. Broader converged resolved references and remaining physical operating ranges are required. Unsupported pressure moments and near-zero aggregate arithmetic failures remain visible. |
| C: coupled deformable contact | Converged unilateral/frictional controls followed by full hollow egg/Franka and an interior-hotspot case, matching geometry, compliance, supports and actuation | Controlled references and free-body U pilots solve their complete original equilibrium. U joint refinement is unresolved:44.6% tensor-probe change across two useful levels. Spatial/regularization/history tails, nonlinear pad comparison, full egg acceptance and actual Franka matching remain incomplete. |
| C: feedback separation and assumptions | Native replay of resolved loads plus corresponding rigid/deformable actuation/load comparisons; deformation bounded relative to thickness/gap/footprint | Actual native nodal replay matches U contact-body fields; finite-footprint replay quantifies mapping bias. Body strains are small, pad strains reach7.5%. Nonlinear compliance and corresponding rigid-contact feedback comparisons remain required. |
| Geometry and load coverage | Full3D smooth convex controls, hollow egg, rounded U, thin neck, cavity/narrow gap; opposing/asymmetric/shear/near-slip/count/island histories | Seven independent geometry assets and finite-load sweeps exist. Full resolved contact coverage remains concentrated in controls and U; egg/Franka is the principal unfinished application. Interior peak search remains global. |
| Convergence and extrema | Fixed continuum geometry/load/radius/profile; separated volume, geometry, quadrature, distance and contact regularization axes; credible trends/tails, full fields and raw global extrema | Six-level FEM sweeps, fixed-polyhedron axes and numerical port/quadrature comparisons are published. U seed-crease peak growth is distinguished from rounded CAD refinement. Contact axes and remaining admissible physical tails still need completion. |
| Accuracy/cost range | Warmed native costs measured at verified physical accuracy, declared representative settings and limitations; preserve qualified fast paths | Native2/4/8-port costs and numerical limits are measured, but contact-model/physical errors are not fully qualified. Do not infer a physical operating range from the previous large-batch rates. |
| Deliverables | Reusable assets/snapshots/scripts, successful and failed runs, exact source/config/package/hardware identity, editable matched maps/traction/island charts, exact commands, commit/push | Immutable evidence and reports are incrementally published. New resolved-model figures and guarded observers are added; live jobs are excluded. Future accepted results require the same evidence closure. |

For smooth fixed finite-load cases, the established3% adjacent-change target is
an initial engineering diagnostic for full tensor, von Mises probes, gauge
displacement, energy and raw peak. At least three useful levels and a credible
tail are required; a single adjacent pass or stable peak is insufficient.
Complete algebra/contact residuals must meet their original case budgets and
remain below discretization changes. A singular raw peak needs a demonstrated
geometric/loading explanation, with the original peak retained. These numerical
diagnostics are separate from measured load-model and omitted-feedback errors;
no universal3% claim or tolerance relaxation is implied.

Next evidence priorities are the scheduled separated U contact axes and third
level, nonlinear pad/all-part comparison with objective finite-rotation checks,
resolved full egg followed by actual Franka
matching, regularization/history tails, and corresponding rigid-contact feedback.
The latest completed load-model replay checkpoint is
[PHYSICAL_RESOLVED_MODELS_20261014.md](PHYSICAL_RESOLVED_MODELS_20261014.md).
The [nonlinear bulk qualification](PHYSICAL_NONLINEAR_REFERENCE_20261014.md)
adds derivative/objectivity/whole-volume bounds and two live same-protocol
references. Evaluating a different law at saved positions is diagnostic evidence;
it does not establish its contact equilibrium or physical accuracy.
The later [objective replay/application-input checkpoint](PHYSICAL_OBJECTIVE_FRANKA_20261014.md)
closes those nonlinear pilots and gives3.09337e-5 objective tensor difference
at one useful level. Actual complete Franka input snapshots and temporal axes
are verified, but they retain angular motion and rank5 point-force support.
They supply corresponding inputs; matched deformable application contact and
the original convergence/feedback/cost acceptance remain incomplete.
