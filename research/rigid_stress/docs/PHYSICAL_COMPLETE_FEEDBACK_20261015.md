# Complete original rigid/deformable feedback comparison

Both same-law original protocols now cover all330 controls,2640 physical states
and3.3s. The rigid counterpart has an independent all-state force/moment, fresh
actual robot, contact and motion audit. Complete-history source-wrench processing,
seven material-frame tensor comparisons and14 actual native resolved-load
replays have closed successfully. This completes the original coarse counterpart
and comparison histories, not the [full physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).

## Original controls and all-state physical gates

Producer718525, independent auditor718526, feedback processor718527 and actual
native GPU replay718528 all have scheduler COMPLETED, actual wrapper0 and
unchanged before/after source identities. The rigid history preserves its2129
qualified prefix files byte-for-byte and appends the remaining original physical
states. Its11 additional lagged-friction recoveries are numerical procedures,
not extra physical time samples. Final acceptance always uses the original
current-friction equation, fresh actual Panda step and original motion gates.

The independently reconstructed maximum body force-equivalent residual over
all2640 states is9.976702e-10N; the maximum fresh actual robot effort residual
is9.881476e-10. Both remain inside the original1e-9 gates. The final body's
net force/moment residuals are2.968868e-10N/2.470525e-12Nm; final contact
pullback difference is0N. All original approach, ground support and lift
controls are included.

The rigid and deformable references share the represented480-tet polyhedron,
mass0.006529848210401344kg, original full Panda model, ground, material,
actuation, contact/friction law, regularization and dt0.00125s. This preserves
the original declared comparison. It does not repair the coarse geometry's
known volume/mass error or unresolved contact concentration.

## Feedback differs from elastic recovery error

The processor independently reconstructs each finger and ground source's normal
and shear contact forces at the actual body application points at every physical
state, and their moments about the current body COM. All source sums reproduce
the saved original contact force. Physical source identities are not established
continuum contact islands.

At seven declared times, full-inertia resolved-load independent linear FEM
separates rigid/deformable load-history feedback from linear-versus-nonlinear
recovery on the deformable history. Both comparisons use proper finite material
frames and the actual positive nonlinear volume-quadrature weights. Relative
errors below divide by the stated reference tensor L2 norm; absolute norms in
Pa·m^(3/2) and denominators remain in the archived report.

|Time, s|State|Rigid/deformable resolved-load feedback tensor difference|Deformable linear/nonlinear tensor difference|
|---:|---:|---:|---:|
|0.01375|11|0.231340%|0.000291746%|
|0.1|80|2.08996e-8%|0.000317425%|
|1.0825|866|7.810383%|0.032167450%|
|1.08625|869|23.169286%|0.003865348%|
|1.8|1440|0.031628%|0.004826095%|
|2.5|2000|0.006849%|0.004809466%|
|3.3|2640|0.013338%|0.004807413%|

The23.17% value is the largest of these seven tensor snapshots, not a claim
about the maximum tensor error over every physical state. At869, the absolute
feedback norm is9.653922802Pa·m^(3/2), with reference norm41.666898794.
The independent linear/nonlinear recovery norm is0.001610612Pa·m^(3/2),
with reference norm41.667959377. These distinct error norms are not additive.

Actual contact/source differences accompany the transient. At866, the rigid
reference still has0.020024663N upward ground normal force, while the deformable
reference has no active ground stencil. At869, both have ground contact, but
finger1 normal y forces are0.401864721N/0.333504135N and finger2 normal y
forces are-0.385363163N/-0.305279414N for rigid/deformable respectively.
All normal/shear vectors and individual source moments remain available,
including finite-gap friction moments and changing support contact.

Later stress differences are much smaller under the same controls. This
time dependence prevents a single settled snapshot from representing the
approach/lift transient. The largest whole-history COM separation is94.684µm;
it measures differing rigid-pose histories and is not an elastic-strain measure.
These results quantify feedback in the original coarse problem. Physical
contact/geometry/space/time convergence is still required before interpreting
them as continuum accuracy or a production operating range.

## Fourteen actual native replays pass their numerical scope

Each declared time has both full-inertia and rigid-relief native Quadrants
resolved nodal replays. These are two separate physical RHS definitions.
All14 pass their original complete numerical budgets; maximum budget fraction
is0.032045772. Maximum native/independent complete corner-tensor relative
array-norm difference is1.588382e-10. Native raw global maxima and the
independent full corner fields remain unaveraged.

These numerical comparisons establish the actual native solve at those resolved
loads. They do not establish pressure/componentwise/pair-union/body-source-prior
finite-footprint accuracy, convergence of continuum contact islands, or warmed
native cost at physically verified resolution. The native array norm here is
also distinct from the volume-weighted material-frame norm used for feedback.

## Fixed-load local bulk studies are separately qualified

The [thickness/local-axis checkpoint](PHYSICAL_THICKNESS_LOCAL_AXIS_20261015.md)
still retains3.18–4.52% tensor changes. Replacement frozen-traction qualification
718586 now passes all five original502230-tet cases: maximum nodal load/RHS
changes are4.336809e-18N, maximum full-quadrature tensor difference is
7.071936e-8Pa, and maximum complete original FEM budget fraction is0.028813477.
The unchanged quadrature points lie within3.602424e-18m of their boundary planes;
maximum quadratic virtual-work difference is1.016440e-20N·m². No physical
coordinate is relocated. Original generic-finder failure718574 remains rejected.

Only after this full qualification do516287/554574-tet local bulk jobs
718587/718588 start with identical physical traction positions, weights and
vectors. Their solved fields and convergence are still outstanding. The recorded
U history/current-friction audit is another unfinished axis, and neither study
narrows the full objective.

## Reproduction and completion limits

The [complete original feedback packet](../evidence/20261014-physical/complete-original-feedback-v1/manifest.json)
retains the full rigid producer, all-state audit, full source-wrench/scalar history,
seven resolved-field snapshots,14 actual native fields, exact submitted commands,
source identities and hardware. It references the previously published
[complete deformable parent](../evidence/20261014-physical/complete-original-history-v1/manifest.json)
by immutable manifest SHA256; every actual parent input hash is reverified.
The inherited deformable artifacts are not unnecessarily duplicated.

Independent runtime verification passes all5826 archive entries against5826
actual runtime originals. Manifest SHA256:
`c33983a75048467f4dfcc9ec4e743a7385f4cc6784e4e3152a8da594786aae44`.
These counts include provenance/source files and are not numbers of experiments.

The scientific scope remains the full A/B/C objective: independent finite-load
verification and applicable load-model/island sensitivity, physically credible
geometry/bulk/contact/quadrature/history/regularization tails for convex and
nonconvex cases, converged egg/Franka and interior-hotspot contact references,
and warmed native cost at that established accuracy. This checkpoint closes
the complete original coarse histories and their declared numerical/feedback
comparisons. It does not close those remaining physical/model/cost gates.
Production Quadrants and rigid dynamics remain unchanged.
