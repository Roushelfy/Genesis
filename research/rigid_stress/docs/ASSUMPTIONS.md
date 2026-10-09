# Model assumptions and required scope

## What the user permits

The reference egg geometry is fixed. Its world pose may translate and rotate.
Every contact's location, direction, normal/tangential force, area and activity
can change. There is no symmetry restriction, fixed finger placement, fixed
number of contacts, or guarantee that the maximum remains near its previous
location. Table, fingers, palm and any other contacting geoms must be included.
Spatial indexing caches geometry, never the next contact pattern.

During ordinary grasp stages, contacts can vary continuously/smoothly. Exploit
this with displacement/history prediction. Contact birth/death, slipping,
impact, release and asynchronous resets may invalidate prediction; correction
is selected per environment using the current complete RHS, never by assuming
a frame must be close to the previous one.

Friction is part of the required physics. Do not discard tangential components
or infer them from normal force. The actual rigid contact solver supplies them.
The synthetic CPU friction-cone examples do not prove real grasp consistency.

## Additional explicit mechanical inputs

Factor reuse requires a fixed discretization and a fixed linear constitutive
operator. The benchmark starts with homogeneous isotropic linear elasticity:
E=10 GPa, nu=0.3, density=2000 kg/m^3, wall thickness=0.5 mm, outer dimensions
approximately 44 x 44 x 60 mm with the tapered egg surface in `egg_mesh`.
These are assumed parameters, not calibrated biological egg properties.

Material and mass are declared inputs, not deductions from fixed shape. If a
training task changes E, nu, density or topology, update/scale operators where
mathematically justified or group environments by operator identity. Never
share an incompatible factor. A common scalar E change at fixed nu scales K;
it does not authorize arbitrary heterogeneous materials to share the same K.

The recovery solves small elastic strain quasi-statically in a moving rigid
frame. Rigid motion itself can be large. Contact geometry stays undeformed;
elastic displacement does not alter Genesis contact response. Elastic waves,
buckling, cracking, yielding, finite-strain response and geometry/material
change require a different model. Violating these assumptions is not a solver
optimization. The egg is hollow for stiffness/mass but rigid for collision.

## Requested output

Default metric is the global unaveraged maximum **von Mises** stress in Pa.
This is the metric used in the existing experiments. It is not a calibrated
failure probability. If principal tensile stress is later requested, introduce
it as a separate output and validate its extrema rather than relabeling VM.
Internal displacement is necessary for recovery; full stress tensors need not
be stored or copied out. Argmax element/corner may be retained for QA.

The four-corner exact maximum applies to P2 displacement on **straight affine
tetrahedra**, fixed linear material and unaveraged stress. Curved P2 geometry,
nonlinear constitutive laws or nodal smoothing invalidate that shortcut.

## Finite contact area

A rigid solver's point and resultant do not uniquely determine local pressure
or contact area. A finite traction footprint, pad/contact compliance law or
measured footprint is an additional modeling choice. The algorithm must accept
changing footprints and integrate them accurately. A fixed radius in the demo
is a demo input, not an algorithm restriction or physical fact.

Do not shrink a patch with mesh size and call the growing point-load singular
peak a convergence failure. Conversely, do not widen patches only to make
peaks numerically easier. Hold the physical footprint law fixed during mesh
convergence, report radius/model sensitivity, and include the minimum admitted
footprint radius in benchmark configuration.

## Proposed accuracy profiles, not previously user-specified tolerances

The user has not set a numerical error budget. Start with these explicit,
editable acceptance profiles; measure and publish both:

| Profile | Equation residual target | Empirical peak error vs same-mesh FP64 direct |
|---|---|---|
| Reference | FP64 direct, aim 1e-8 where attainable | oracle |
| Strict | full nonzero RHS relative residual <=1e-6 | max <=1e-4 (0.01%) |
| Throughput | full nonzero RHS relative residual <=1e-3 | max <=1e-2 (1%) |

For reference peak <=1 Pa, additionally publish absolute Pa error instead of
claiming a relative guarantee near zero. Publish every exception and attainable
roundoff floor; never silently add the historical script's factor-of-five
check margin to a production acceptance budget. A residual tolerance alone
does **not** certify a peak error on unseen loads. Calibration is empirical;
rigorous output error certification is a separate optional project.

Physical discretization acceptance: require at least three successively finer
full-model meshes and surface quadrature refinement with unchanged finite load
models; initial proposed criterion is <=2% peak change in the final refinement
for a representative asymmetric frictional suite. Report monotonicity, hot
locations and unresolved cases. This is not proven by the coarse CPU timings.
