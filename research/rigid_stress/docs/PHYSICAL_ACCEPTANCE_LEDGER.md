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
The [actual Panda mechanics checkpoint](PHYSICAL_FRANKA_MECHANICS_20261014.md)
adds the real rigid compound fingers and jaw potential, unchanged-budget
same-held-row contact requalification and strict corresponding fixture geometry.
Both free-body release solvers remain unqualified; their net force/moment and
raw stress diagnostics are explicit. Three identical52s rigid fixed-hand
histories are compared at80 exact common physical times. Normal-force changes
are small, but orientation differences are nonmonotone and angular motion
persists. Contact regularization/support matching, solved deformable application
contact, actual full-arm matching and the original acceptance remain required.
Subsequent source inspection identifies the temporary weld's automatic2*dt
time constant: the earlier integration sweep also changes initializer compliance.
It is a combined numerical observation, not pure fixed-support temporal evidence.
The contact-only regularization study now records actual mixed contact parameters,
penetration and runtime weld parameters at a fixed integration interval.
The [contact-regularization/inertial checkpoint](PHYSICAL_CONTACT_REGULARIZATION_20261015.md)
measures default rigid penetration near92% of wall thickness and contact-law
load changes7.841%/3.010% at fixed dt/weld parameters. Independent actual-finger
inertial references solve140 steps at three dt values with unchanged complete
dynamic gates; their temporal tensor/velocity/energy diagnostics are below3%
at one coarse spatial/contact setting. Actual native nodal replay agrees with
independent linear FEM to7.77e-11 and that nonlinear dynamic field to4.80e-5.
Physical spatial/contact tails, matched supports/longer/full-arm histories,
load-model/feedback separation and the full operating-range acceptance remain.
The [finer contact/full-arm/refinement checkpoint](PHYSICAL_FRANKA_CONTACT_TAIL_20261015.md)
adds fixed-weld contact-law force tails0.810%/0.206%, actual complete full-arm
input/control audits and exact sparse-inertia equation/83-array-archive equivalence.
Spatial qualifications are not spatial convergence; finer coupled studies are
live. The full-egg lagged initialization attempt fails at increment384. Same-state
current-force Newton recovers that increment to4.11e-12N at the unchanged1e-9N
budget, but both continuation guards fail. The second restores the saved held
rows and exact shoe meshes. Completed same-state audit finds exactly matching
bulk operators and4.11e-12N residuals in both reconstructed systems; the earlier
force discrepancy remains unexplained. Strict continuation714503 passes its
saved-increment guard and advances into the remaining original initialization.
Complete egg contact acceptance remains unverified. Original fine
qualification714444_2 isTIMEOUT, regardless of its wrapper returncode0.
All original full physical gates remain.

The [articulated/spatial checkpoint](PHYSICAL_ARTICULATED_SPATIAL_20261015.md)
adds2400 common material probes and exact equal-constructor controls, full-arm
contact-force tails0.824%/0.223%, and corrected physical inertia checks on all
captured motion frames. Original working mass contains implicit damping; the
near-stationary audit alone did not establish physical inertia. The new reader
and explicit readouts remove that ambiguity without modifying dynamics.
Fixed-contact coarse controls preserve83 original array archives exactly;
the first completed480/3840 comparison changes full tensor probes56.2%, raw
peak30.8% and energy35.2%, despite contact resultants changing only0.0071%.
Original force budgets and componentwise postprocessor gates remain unchanged.
Fixed-contact volume and third-level reference studies remain live; no spatial
tail or singularity acceptance is inferred from these first two levels.
Full coupled application, contact/stress convergence and every original gate remain.

The [independent contact-discretization checkpoint](PHYSICAL_CONTACT_DISCRETIZATION_20261015.md)
solves three collision-only levels at fixed480-tet bulk geometry and2970 source
unknowns. Exact P2 trace/current-friction pullback passes, including an independent
negative-weight patch. Coarse-body tensor changes0.002419%/0.005042% do not
establish contact convergence: two active nodes persist, their lumped area quarters,
and maximum normal/shear force-per-area approximately quadruples at each level.
Raw surface forces, areas and per-island six-component wrenches are retained.
Reference-only cached current-Newton preconditioning matches all80 original states
and full corner tensors at unchanged complete1e-9N gates; the fine control is
scheduled. These qualify numerical tools and expose an unresolved physical axis.
Separated/coupled spatial tails, matched full-arm/egg/U contact, load-model and
feedback separation, verified accuracy/cost and complete deliverables remain open.

The [local-refinement checkpoint](PHYSICAL_LOCAL_REFINEMENT_20261015.md) extends
reference numerical equivalence to all80 finer3840-tet states and corner tensors
at unchanged original force budgets. Four qualified local meshes preserve the
fixed polyhedron's volume/centre/boundary area. Coupled/frozen-contact physical
studies are live; four failed mesh diagnostics include a rejected topology-
changing tie candidate. These supply verified tools and scheduled coverage,
not convergence or a singularity outcome. The first solved480/1099 comparison
changes tensors43.03% and reaches562kPa on coupled/fixed-contact paths; two
active nodes and a moving peak retain unresolved concentration. Full application matching, solved
egg/U/contact tails, model/feedback separation and warmed physical accuracy/cost
remain required.
