# Actual Panda mechanics and incomplete free-body contact

This checkpoint continues the complete
[physical-validation goal](LOAD_MODEL_PHYSICAL_VALIDATION.md) after the
[objective/application-input checkpoint](PHYSICAL_OBJECTIVE_FRANKA_20261014.md).
It qualifies actual compound finger geometry and one supported contact state.
It does not qualify a converged deformable Franka grasp. Complete original
requirements remain in the [acceptance ledger](PHYSICAL_ACCEPTANCE_LEDGER.md).

## Actual rigid fingers, objective egg and actuation

`franka_contact_reference.py` builds independent scikit-fem P2 elasticity for
the full level1 hollow egg and an objective neo-Hookean body. Material stays
E=10GPa, nu=0.3, density2000kg/m3 and mass0.0065298482104kg. Each actual15g
Panda finger retains its captured convex mesh and five box collision geoms.
All surface vertices translate rigidly along the captured joint axis; no
fictitious shoe modulus is assigned. An unconnected COM node supplies rigid
mass/gravity virtual work and the jaw spring. The actual contact coefficient
is0.6. Each jaw has its own100N/m spring,18mm target and4N force cap.
Static spring saturation uses the corresponding Huber potential.

The hand is fixed at the selected captured lifted pose. Six body displacement
rows temporarily initialize closure; they are removed before free-body
acceptance. This declared controlled problem omits the application's seven
arm PD joints and soft finger-joint equality. It requires its corresponding
rigid fixture, history checks and an explicit comparison with the full arm.
Static equilibrium does not depend on the original velocity gains, passive
damping or armature; time-dependent comparisons must retain those parameters.

The source-coordinate qualification714213 passes exact rigid jaw translation,
axis, mass and gravity/spring virtual-work checks. No-contact energy/gradient
finite differences agree below8.2e-8; gradient/Hessian differences are below
3.2e-11. The selected centred qualification714249 repeats those controls.
These checks concern kinematics and derivatives, not reference convergence.

## Same-state numerical floor and original held rows

The first actual-geometry solve714216 fails during closure target56, at
q_target=20.75mm. Its full constrained residual stalls at1.1804362e-9N,
above the unchanged1e-9N budget. Original positions, previous friction state,
all six held rows, log, configuration and executable source are retained.

`StableNeoHookeanBulk` evaluates the same first Piola stress as
P=[mu*(B-I)+lambda*log(J)*I]*F^(-T), using H=grad(u) to compute
B-I=H+H.T+H*H.T and log1p(det(I+H)-1). This avoids subtracting nearly equal
identity tensors. The inherited quadrature and analytic tangent remain the
same; the selected small-strain constitutive gradient differs from the
conventional expression by1.4186e-12 relative. This change alone does not
resolve the source-coordinate contact floor:714227 still fails near1.168e-9N.

The complete fixture can instead be translated to a coordinate origin at the
captured body COM. World-axis orientation, relative geometry, gravity, control,
friction and material remain unchanged. The recorded world-origin offset is
[0.65123842329217,-0.00028621341113715,0.14800936850090216]m; geometry files
store it explicitly. Coordinate round-trip error is1.11e-16m.

Automatic QR pivoting initially chooses different six held rows after the
translation. The first centred qualification714229 therefore diagnoses a
different supported fixture and cannot prove identical-state repair. Its
outputs are preserved exactly and excluded from that claim. The selected
v2 and separate714233 audit keep the original interleaved rows
[25,31,75,76,84,473], including the saved previous friction displacement.
They also assert that the original failed residual is reproduced.

| Identical saved state and held rows | Complete constrained residual (N) |
|---|---:|
| Original world coordinates and bulk expression |1.18043621e-9|
| World coordinates, equivalent stable bulk expression |1.16818840e-9|
| Centred coordinates, before further Newton update |1.11077764e-9|
| Centred coordinates, after one further Newton update |1.68895704e-11|

The original1e-9N tolerance is unchanged. The centre translation permits a
successful update for this supported state; it does not establish free-body
equilibrium or a universal numerical improvement at other scales.

## Closed fingers still require free-body equilibrium

The centred cold solve714230 completes all64 jaw target increments. At its
last held state, q is about21.9968mm and spring forces are about0.3997N per
jaw. The temporary support still carries gravity and moment. When that
support is released, the complete direct Newton solve fails after100
iterations. An explicitly selected damped least-squares solve714237 starts
from the same retained closure, preserves loads and geometry, and also fails.
Neither solver relaxes the original residual budget.

`audit_franka_release.py` records all nonlinear body nodal forces, gravity and
moments about the deformed consistent-mass COM. Original failed stresses and
whole-volume bounds are retained as diagnostics, not accepted grasp results.

| Incomplete released state | Complete free-map residual (N) | Net force/moment residual, moment scaled by20mm (N) | Peak lower/upper (Pa) |
|---|---:|---:|---:|
| Direct Newton |0.05355336|0.05019731|155633.0302 /155633.0305|
| Damped least squares |1.8968053e-5|5.4092489e-4|156553.4773 /156553.4776|
| Elastic/rigid coordinate basis, direct Newton |0.05605661|0.05029207|155632.4263 /155632.4266|

Both retain only two active normal stencils. In the latter state, objective
elastic displacement is0.352um, about0.0704% of the0.5mm wall and0.00586% of
the nominal6mm footprint. Small deformation and a narrow nonlinear peak
interval do not prove equilibrium. The observed residuals are not proof that
the global grasp is impossible; weak contact modes, current friction history
and nonlinear solve conditioning require further diagnosis. The actual rigid
application also retains angular motion, as documented previously.

An opt-in invertible coordinate basis now separates elastic coordinates from
all six restored rigid modes; rotations use a declared20mm length scale.
It checks reconstruction of an arbitrary full body vector, and constrains no
free-body mode. The same-closure direct solve714252 also fails to converge
within its original iteration budget. Its complete failed state and714262
force/moment audit are retained in the later fixture checkpoint. The table
uses the same original full free-map residual for every solver; different
coordinate scaling is not counted as physical accuracy.

## Corresponding rigid fixture geometry

`franka_fixture_geometry.py` constructs a fixed-hand two-jaw URDF from the
fully closed actual input capture. Primitive boxes remain boxes; the original
mesh geoms remain meshes. Independent rigid fits reconstruct every geom in
its finger frame. Runtime matching714247 verifies all12 geoms, their types,
actual world vertices, jaw poses and masses. Largest bidirectional vertex
distance is1.259e-16m against the original1e-10m guard.

The URDF zero pose is open. Physical Panda q equals fixture q plus40mm;
limits and PD targets undergo the same constant offset. This preserves the
physical trajectory and prevents construction from initially closing around
the egg. The original v1 fixture714236 built at closed q=0 and triggers a
native incompatible-load rejection before its geometry audit. That failed
attempt remains visible; its error is not bypassed or counted as a pass.

The rigid fixture retains100/10 jaw PD gains,4N caps and the actual source
class's0.1kg generalized joint armature and1N*s/m passive damping. Runtime
readouts verify both passive parameters. The controlled64-target closure with
0.5s at each target and20s released hold714253 completes at2.5ms integration
interval. A temporary compliant weld initializes the rigid body and is removed
before its final phase. All576 captured physical substeps have unique records.

The strict history audit714279 finds temporary weld position/orientation changes
up to27.30um/0.000575rad. The original v1 audit omitted its output directory and
fails before writing; its log/source are preserved, and v2 fixes that ordinary
artifact-writing defect without changing mechanics or diagnostic criteria.
Released tail samples have speed RMS2.0287e-5m/s and angular-speed RMS0.02886rad/s.
The last angular speed is0.03526rad/s. Last normal forces are0.353079/0.352981N,
at fixture geoms4/11 corresponding to actual geoms16/23. Net static-wrench and
fixed-point force-span residuals are both about1.6281e-5N with20mm moment scale;
the point-force wrench span has rank5. These are static diagnostics on a
time-dependent rigid history, not complete dynamic-equilibrium errors or
proof that the global grasp cannot settle.

The temporary weld differs from the independent held-row initializer, and the
rigid constraint law has its own contact regularization. Thus the normal-force
difference from the unaccepted nonlinear closure cannot yet be attributed to
deformation feedback. Controls with5ms/1.25ms integration intervals complete
as714277_0/1, retaining the same physical64-target/52s protocol and geometry.
Support matching, contact regularization, time/history convergence and actual
full-arm equivalence remain required. This is controlled fixed-hand evidence.

The three temporal inputs contain288/576/1152 unique captured substeps.
Audit714294 compares80 identical released-phase physical times from32.24 to
51.99s, using physical_substep0 at the same macro ticks and no interpolation.
Source geometry/order, frozen controls and all non-temporal configuration are
checked. Normal forces are compared by the same two runtime source geoms.

| Integration change | Normal-force L2 difference | Angular-velocity L2 difference | Maximum position difference | Maximum orientation difference |
|---|---:|---:|---:|---:|
|5ms to2.5ms|0.0019493%|0.24336%|78.66um|0.16114mrad|
|2.5ms to1.25ms|0.0044284%|0.053675%|19.11um|0.33631mrad|

Small normal-force differences are useful temporal diagnostics, not a monotone
physical convergence claim. The orientation difference grows, and position
relative errors would be dominated by the shared world-origin offset; absolute
translation/rotation changes are therefore retained. Sampled agreement does
not resolve angular settling, contact-law regularization or deformation
feedback, and does not qualify a physical operating range.

## Reproduction and remaining acceptance

Use `scripts/franka_contact_reference.sh RUN qualify|solve --capture INPUT`.
`--failed-run` is a qualification argument for the original source-coordinate
failure. `--restart-failed-release` restores the verified original completed
closure for solver comparisons; it does not claim a new cold history.
`--free-least-squares` and `--rigid-mode-coordinates` are explicit solver
choices. Complete original free-body force budgets remain required.

Use `scripts/audit_franka_coordinate_frame.sh RUN --capture INPUT --failed-run
FAILED`, `scripts/audit_franka_release.sh RUN --run FAILED`, and
`scripts/capture_franka_fixture.sh RUN --capture INPUT [--geometry-only]`.
Fresh runtime directories, Slurm submissions, package/hardware/source identity
and exact successful/failed outputs are retained under the established data
root. `scripts/publish_franka_mechanics_checkpoint.sh` publishes the immutable
`franka-mechanics-checkpoint-v1`; live tasks are excluded.
Its443 entries reconstruct443 exact originals with no prefix substitutions.
Manifest SHA256:
`8a3f9c81be8aba424d207ad22ef8cc05a6759603507c5626711e8645636acb0d`.
The later `scripts/publish_franka_fixture_checkpoint.sh` retains the completed
rigid history, successful/failed history audits and the failed rigid-mode
reference in `franka-fixture-checkpoint-v1`. It excludes live temporal axes.
That initial fixture archive verifies766 exact originals with manifest SHA256
`ab35fc8328ae1b1d22f586e01d97019e75dfd1e74ef936aef2541d57f57f1499`.
The later `franka-fixture-checkpoint-v2` includes both completed temporal runs
and the common-time audit, preserving the earlier failed/verified records.
Run `scripts/publish_franka_fixture_checkpoint.sh v2` to reproduce its selection.
It verifies2433 exact original entries without prefix substitutions. Manifest
SHA256: `f42255a5e95c086050b3ddefb28a3495528ef0bb644a6e237e7d51b37bd65d60`.

The original U third-level and full-egg references714003_0/3 remain live.
Actual matched deformable application contact, spatial/quadrature/
regularization/history tails, rigid/deformable feedback separation, all
remaining A/B/C coverage, matched maps and verified physical accuracy/cost
remain required. This checkpoint does not complete or narrow the goal.
