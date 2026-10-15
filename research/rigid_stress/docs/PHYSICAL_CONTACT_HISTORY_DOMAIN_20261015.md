# Recorded U histories and covered-domain field sensitivity

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. All744 original U initialization/load stages now have independent
current normal/friction force, original balance and complete field audits.
The broad egg marking-domain reference also closes, but its full tensor still
differs4.27% despite a0.00555% raw peak change. Original U initialization
coordinate choices differ across body meshes; their final effect requires a
separate controlled comparison. These results do not establish physical
convergence or the requested accuracy/cost operating range.

## All744 retained stages pass original current-friction gates

Recorded producer718566 and independent auditor718584 complete with actual
wrapper0 and unchanged source identities. The original845-tet/pad5 case keeps
its geometry, material, mass,0.05N force per shoe, gravity, friction,2.5µm
barrier distance and1e-8m tangential mollifier. The retained sequence contains
12 approach,604 contact-initialization,64 held-force and64 free-body stages.
These are original quasi-static loading increments, not physical time samples.

Every stage reconstructs individual current normal and friction stencils from
the saved current/previous displacement. The complete gradient is independently
assembled from nonlinear body/shoe bulk, actual external force and reconstructed
contact. Maximum contact-force reconstruction difference is5.551115e-17N;
maximum complete constrained residual is9.550504e-10N, inside the original
1e-9N gate. Every stage retains full corner/quadrature tensors, raw global
lower/upper stress bounds and individual shoe wrenches about the current body COM.

| Final original-case qualification | Measured difference or residual |
|---|---:|
|Rest positions, surface triangles, applied force, body permutation|Bit-identical|
|Displacement maximum difference|4.543485e-16m|
|Contact force maximum difference|6.255118e-12N|
|Full corner tensor maximum difference|5.124086e-6Pa|
|Complete quadrature tensor relative difference|6.618341e-11|
|Independent body residual|3.101533e-10N|
|Independent original mapped residual|4.796677e-10N|

Residual/iteration records differ in bits and remain visible. Final arrays and
complete fields pass the previously declared independent-field tolerances;
no original physical gate is weakened. The cause of the bit differences is not
established. No nonexistent original intermediate full fields are claimed for
a matched-field comparison. The final raw global interval remains
112413.887876–112414.743121Pa and is not replaced by a smoothed maximum.

## A stable raw peak does not establish a stable covered-domain field

Broad-domain producer716992, five-state auditor716993, coverage auditor716994
and full comparison718622 all complete with their actual original gates.
The150µm/500µm values are numerical marking radii, not physical footprint radii.
Both cases keep the100µm CAD edge target, original held closure and physical
contact law/controls. Declared CAD parameters, identical seed hashes and all
eight geometry-constructor Python sources are checked across their distinct
provenance roots. Their represented geometry and mass still differ explicitly.

| Final6.25ms comparison |150µm marking domain|500µm marking domain|
|---|---:|---:|
|Body tetrahedra|31668|90240|
|Represented body mass, kg|0.007424160816030722|0.0074241744582911945|
|Raw global lower peak, Pa|40400486.643241|40398245.894106|

All158340 original common physical probes and volume weights remain. The
coarse/fine objective tensor difference is4.266503%, von Mises1.818691%,
objective displacement0.173335%, energy0.172649% and raw lower peak0.005547%.
No probe or component is dropped or renormalized to improve those changes.

The broad final state has30 normal stencils and32 active body samples, all
inside the declared numerical caps. Its original body/mapped residuals are
6.242433e-14/1.729612e-13N. Active lumped area is2.896084e-8m² and maximum
discrete normal-force/lumped-area density is70.489345MPa. These are discrete
trace diagnostics, not continuum pressure/contact area or identified islands.
The raw hotspot remains at[2.114692e-5,0.022251337694,-0.001827384333]m.

Coverage and a matching peak do not resolve the tensor discrepancy. Bulk,
surface geometry and contact-trace approximation change together here; two
marking domains are neither a separated bulk axis nor a credible physical tail.
The five saved states are a short stationary window, not the complete application.

## Numerical observables pass independent manufactured checks

Qualification718623 checks the nested P2 transfer and exact affine-tensor
volume-norm functions before local bulk results are interpreted. A6/12-tet
unit-cube control reproduces all ten quadratic nodal polynomials. The affine
tensor's squared-volume integral is20/3Pa²m³ both analytically and using an
independent four-point tetrahedral quadrature. A nonzero nested Galerkin energy
error4.531064927069364e-5J matches twice the discrete energy increase within
6.437451e-19J; both complete force budgets pass.

The pending [local bulk auditor](../audit_frozen_traction_bulk_axis.py) must match
the exact qualified numerical functions and independent FEM source. It will
independently reconstruct all15 original/fine fields at their original force
budgets, retain all36 declared probes and raw peak positions, and supplement
them with exact common-volume tensor norms and explicit residual-work energy
bounds. Supplemental norms do not replace the original criteria. This small
manufactured control is not a finite-footprint physical convergence result.

## Initialization coordinates are an additional controlled variable

Audits718726/718741 recover the actual held body coordinate/component choices
from saved original maps. The old2339/5669-tet cases do not share all six
coordinates with the845-tet case. The largest component-matched distances to
the nearest baseline constraint point are20.576mm and18.219mm, respectively.
Those distances describe coordinate selection, not body displacement or stress
error. All body modes are restored after release. The final effect on contact
history remains unmeasured, so these selections cannot be silently treated as
an isolated bulk/geometry change.

Twenty-three P2 positions are common to the three meshes. A rank-six anchor
set selected only from that intersection has scaled rigid-mode condition2.82914.
Controlled case718746 and its independent audit718747 change only the temporary
initialization coordinate anchors on the same845-tet/pad5 case. They retain all
12/604/64/64 stages and original physical/numerical parameters. The actual anchor
coordinates must be independently recovered from the new map. This is a
declared initialization-sensitivity control, not a replacement of the old setup
or a device for manufacturing convergence. Results remain outstanding here.

Separately, barrier-distance jobs718718/718720 and independent audits718719/
718721 compare2.5/1.25/0.625µm while keeping original coordinates, geometry,
material, mass, friction, force, all loading-stage counts and tangential mollifier
fixed. They quantify a contact-regularization axis rather than requiring
different barrier problems to have identical solutions. Objective final tensor
norms use finite material frames. The first two startup attempts718703/718705
fail before FEM because a frozen import was missing; real code1/source closure
and the cancelled unstarted audits remain retained. Corrected cases are running,
not accepted results.

## Evidence and remaining objective

- [Recorded contact history](../evidence/20261014-physical/recorded-contact-history-v1/manifest.json):
 2354 archive entries,1597 actual runtime originals; SHA256
 `7f9accb7d15815fd1e73614c43e456071668a19cf0fe7d23494a0f66d9674e87`.
- [Covered marking-domain fields](../evidence/20261014-physical/covered-marking-domain-v1/manifest.json):
 374 entries,360 actual originals, one exact reconstructed large field; SHA256
 `5a46ff656089cf87caa8b5234d2aa691d91485d94ee40e8ec55f4ddb975e3610`.
- [Observable methods and actual failures](../evidence/20261014-physical/contact-observable-methods-v1/manifest.json):
 242 entries/originals; SHA256
 `974df7244a09a21c70de77bb51b48745023bad20c80708d4dcaec83532edc0ad`.

Every packet passes original/archive/source and exact-byte reconstruction checks.
Counts include provenance files and are not numbers of independent experiments.
Exact commands, versions, hardware, source identities, failures and full raw
fields remain reproducible; live cases are excluded from successful evidence.
Large runtime studies stay under the declared data root and outside routine tests.

The full A/B/C objective remains: credible body/geometry/contact/quadrature/
history/regularization tails, applicable finite-load model and local redistribution
accuracy, converged full egg/Franka and nonconvex interior-hotspot references,
resolved native replay versus feedback at that accuracy, and verified warmed
accuracy/cost tradeoffs. The [complete original coarse histories](PHYSICAL_COMPLETE_FEEDBACK_20261015.md)
remain qualified within their stated scope. Production Quadrants and rigid
dynamics are unchanged.
