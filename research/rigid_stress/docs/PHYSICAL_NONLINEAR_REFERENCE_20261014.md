# Nonlinear reference qualification and body-rotation diagnostic

This continues [resolved contact/model validation](PHYSICAL_RESOLVED_MODELS_20261014.md)
under the complete [physical-validation requirement](LOAD_MODEL_PHYSICAL_VALIDATION.md).
It qualifies nonlinear bulk assembly and changes the next reference experiment;
it does not establish converged nonlinear contact, Franka equivalence or stage completion.

## Independent material and assembly

`neohookean_reference.py` integrates the compressible neo-Hookean law

W(F) = mu/2 (tr(F^T F)-3) - mu log J + lambda/2 (log J)^2,

with F=I+grad u, J=det F and the original Lamé parameters derived from E and nu.
The [DOLFINx hyperelasticity demo](https://docs.fenicsproject.org/dolfinx/main/cpp/demos/demo_hyperelasticity.html)
states this energy and the Cauchy relation sigma=P F^T/J. This implementation uses
the installed scikit-fem affine P2 basis, topology, quadrature and gradients;
NumPy contracts element residual/tangent tensors and SciPy assembles the sparse
matrix. DOLFINx is a formulation reference, not the installed numerical backend.
Geometry, density, E and nu are retained without parameter retuning.

Two natural-close qualifications,714051_0 and714063_0, complete successfully.
The latter extends the first with whole-element peak bounds and both complete
body/pad block configurations. Its actual source snapshots and assertions qualify:

| Check | Pad parameters: E1MPa, nu0 | Body parameters: E10GPa, nu0.3 |
|---|---:|---:|
| Rest tangent versus independent linear K, relative norm |2.257e-16|1.884e-16|
| Largest finite-difference energy-gradient error |2.937e-8|1.193e-10|
| Largest finite-difference gradient-tangent error |2.576e-11|7.730e-11|
| Tangent symmetry error |5.810e-17|6.233e-17|
| Analytic7% compression versus separate boundary integration |2.087e-15|2.022e-15|

Derivative checks use two independent perturbation sizes1e-8m and3e-9m.
Three translated rigid rotations,0.003/0.6/1.2rad, retain zero energy/force to
floating precision and positive deformation determinant across every element.
The complete pad-only and all-part replacement configurations also pass separate
total-energy/force/tangent finite-difference checks: energy errors5.87e-9/1.61e-10
and tangent errors4.71e-12/5.33e-11. These are mathematical assembly checks;
their separated cubes have no active contact and do not qualify contact physics.

For numerical stability, energy is evaluated from singular stretches, an
algebraically equivalent expression that avoids cancellation of rigid-rotation
energy. The force and exact tangent use the first Piola derivative of the same
law. Inverted quadrature states are rejected; trial failures cause line-search
reduction rather than determinant clamping or a changed material.

## Whole-volume determinant and global-peak bounds

On affine P2 tetrahedra, F is affine and J is a complete cubic barycentric
polynomial. The implementation expands its three columns at the four vertices,
groups equal monomials and obtains the cubic Bernstein coefficients. A positive
minimum coefficient bounds J away from zero throughout the element, including
points outside the quadrature/corner samples.

Deviatoric Kirchhoff stress is mu*dev(F F^T), a quadratic barycentric tensor.
Its Bernstein coefficient norms bound its norm by convexity. Dividing by the
positive determinant lower bound gives a per-element Cauchy von Mises upper
bound. Actual unsmoothed corner samples give a lower bound. A small FP64
rounding allowance is included in the reported upper bound. All corner samples
and200 independently generated barycentric locations per control element pass
the bound check; the polynomial argument supplies the whole-element bound.
The nonlinear driver reports the full global interval and its relative width.
It does not silently treat the four-corner sample maximum as an exact nonlinear
maximum. No interelement averaging or stress smoothing is introduced.

## Saved-position audit changes the physical next step

The existing U states are evaluated under the new law without claiming to
have solved its equilibrium. The reported norm compares complete assembled
nonlinear versus linear bulk force at the same saved nodal positions.

| Saved linear U case | Body force difference / linear norm | Left pad | Right pad |
|---|---:|---:|---:|
|845 body tets, pad grid5|60.4365|0.0128166|0.0112119|
|2339 body tets, pad grid7|16.2301|0.0114941|0.0175771|

The body energy changes from4.38268e-8 to5.40749e-7J in the first state and
from4.07383e-8 to7.98060e-8J in the second. After the original infinitesimal
mass-gauge removal, corresponding body force differences are0.0113146 and
0.0095173. This reveals sensitivity to interpreting the old infinitesimal
rotation as a finite nodal geometry. It is not a60-times physical stress error:
the new law is evaluated at a state that is not its equilibrium, the force norm
depends on mesh/state, and gauge removal changes actual finite geometry.

Pad force changes are about1–2% at these positions despite5–7.5% local strain.
Positive determinant certificates for the finer pads are0.93371 and0.91616.
The result motivates both a pad-only comparison and an objective all-part
reference. A small linear body strain alone does not close finite-rotation validity.
Solved nonlinear states still need finite rigid-frame/pose fitting before
material-coordinate displacement and native stress comparisons.

## Same-protocol nonlinear experiments

Array714067_0/1 runs the845-tet U with the same pad grid5, initial gap1mm,
thickness1mm, edge lift0.1mm, actuation0.05N per jaw, friction0.8, dhat2.5um,
64 force increments, updated friction, gravity(0,-9.81,0), temporary initialization
fixture and subsequent fully free body. The two cases change only the declared
bulk material models: nonlinear pads with linear body, and nonlinear body/pads.
They start from the original cold protocol rather than a retained converged pose.
Original complete constrained/body residual budgets remain1e-9N. Common-x gauge
fixing retains the original coordinate-map acceptance and zero applied x force.

`scripts/run_neohookean_grasp.sh` records canonical input sources and their hashes,
copies a small executable source snapshot into the data run, applies the tracked
`neohookean_grasp_driver.patch` only there, and separately hashes the actual
executed sources before and after the producer. Empty package initializers and
an explicit module-origin check ensure Python imports the copied driver.
The live canonical reference scripts and main checkout are unchanged.
This snapshot is a runtime evidence artifact, not a new data-side Git checkout.

The producer retains both input/executed source identities, exact arguments,
environment/hardware, partial/final fields and natural success/failure closure.
While these tasks are live their results are excluded from this immutable
qualification checkpoint. Original linear third-level/separated-axis tasks714003
also continue; the full-egg retry remains scheduled behind its array limit.

Reproduce controls with `scripts/qualify_neohookean.sh RUN --run ORIGINAL ...`;
retained `arguments.txt` specifies the original state directories. Reproduce
nonlinear experiments using `scripts/run_neohookean_grasp.sh RUN` with the same
grasp arguments and explicit `--pad-material neo_hookean`, optionally
`--body-material neo_hookean`. All runtime data stays in the established
`runs/20261014-physical-validation/` directory.

Next requirements remain solved nonlinear contact comparisons, finite rigid
frame/native-load replay, separate contact spatial/quadrature/regularization/
history tails, full egg and actual Franka geometry/compliance/control matching,
rigid-contact feedback, representative physical accuracy/cost and publication.
See [the full acceptance ledger](PHYSICAL_ACCEPTANCE_LEDGER.md).

## Current native application matching inventory

The actual qualified workload is
[`examples/rigid/franka_egg_stress.py`](../../../examples/rigid/franka_egg_stress.py),
with [`egg_stress_controller.py`](../../../examples/rigid/egg_stress_controller.py)
and `genesis/assets/xml/franka_emika_panda/panda.xml`. The older research
`franka_egg.py` is not the current production workload reference.
Inspection identifies concrete matching work before an egg pilot can support
a Franka approximation claim:

| Input | Current native application | Existing coupled pilot |
|---|---|---|
| Gripper collision geometry | Finger mesh plus five box classes per finger; main box XML size0.0085/0.004/0.0085m | Analytically conforming6mm-wide curved shoes |
| Finger actuation | Position PD: kp100, kv10; close target0.018m; ordinary force cap4N; weakened cap0.055N during phase0.7–0.76 | Prescribed0.05N per shoe, solved jaw travel |
| Material friction inputs | Robot0.1, egg0.6 with per-condition egg ratios | Declared contact coefficient0.8 |
| Native modeled footprint | Base radius6mm, phase modulation15%, optional varied bank | Model replay uses fixed3mm radius |
| Rigid/body mass | Level1 asset metadata0.006529848kg,480 tets | Independent h0.15 egg about0.007426kg |
| Support/loading history | Approach, grip, lift, hold, lateral motion, weakened grip and release; initial plane contact | Temporary fixture, force establishment, fixture removal, gravity |

The native material-friction inputs do not by themselves prove the effective
contact coefficient; it must be captured from the actual same-substep contact.
Actuator caps likewise do not identify each contact resultant. Required next
inputs are corresponding hand/finger/egg poses, velocities, controller targets,
generalized actuation, original contacts/wrenches and actual friction from one
chosen phase, followed by a reference with the same geometry/mass and actuation.
Changing geometry resolution and mass belongs to a separate convergence axis.
The level1 shell metadata retains thickness0.5mm, E10GPa, nu0.3 and density2000.
These parameter matches alone do not establish geometry/control equivalence.

The nonlinear qualification publication verifies46 exact entries/46 runtime
originals. Manifest SHA256:
`cede63776cb22807c908344932d036e2183ab5a6b017b9452a6803929ef6037a`.
Its source/result archive is
`research/rigid_stress/evidence/20261014-physical/neohookean-qualification-v1`.
