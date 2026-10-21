# Independent normal pressure on the hollow polar land

Hertz agrees with the independently solved normal pressure within the declared
5% pressure/moment-axis targets for this one fixed case. The fine-body reference
gives 1.062% pressure L2 error, 0.319% peak-pressure error and
0.248%/0.099% moment-axis errors. This is a conditional pressure result;
complete/peak stress, friction and the moving native estimator remain unqualified.

## Physical problem and independent solve

Reuse the exact faceted hollow egg and three meshes from the
[particular/residual control](HERTZ_HOLLOW_SHELL_RESIDUAL_20261021.md): 500 um
wall, 300 um flat polar land, E=10 GPa, nu=0.3 and density=2000 kg/m3.
A rigid paraboloid with principal curvatures 50/80 per metre applies 0.2 N
normal load. Its normal position and two small tilts are solved with zero
first moments; the free shell uses the same consistent-mass inertial relief.
This is a normal single-pole control, not a two-finger frictional grip.

Continuous bilinear positive pressure hats use the integrated half-space Green
function plus a finite-body displacement correction. A 5x5 table of normal
unit-source responses includes the outer boundary and inner cavity; a 3x3
subset tests source interpolation. This reuses the existing P2 elastic inverse,
without preparing a full vector contact basis. Pressure and discrete support
are complementarity outputs. Hertz only sizes the search window, whose boundary
must remain inactive; neither Hertz pressure nor its support is prescribed.

The three body meshes have 60,600/94,608/139,488 tetrahedra on the same physical
domain. Pressure grids 17/33/65 are retained, including their failed 33-to-65
2% refinement gates. A targeted 97 grid resolves that remaining comparison.
All contact nodes can enter or leave the active set. Completed half-space
solutions supply warm starts; medium/fine 97 grids reuse the completed
97-grid half-space seed. Force, first moments, positivity, gap and
complementarity keep their original numerical gates.

## Separate measured changes

All pressure comparisons use the same 129x129 observation grid. Errors below
use the independently solved pressure as denominator. Moment-equivalent axes
come from integrated second moments and are distinct from support edges.

| Body mesh | Pressure grid 65→97 | Source table 3→5 | Hertz pressure L2 error | Hertz peak-pressure error | Curvature-corrected pressure error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coarse | 1.1306% | 0.00365% | 1.0899% | 0.4738% | 0.9794% |
| Medium | 1.1275% | 0.00321% | 1.0631% | 0.3569% | 0.9751% |
| Fine | 1.1274% | 0.00317% | 1.0623% | 0.3185% | 0.9747% |

At pressure grid 97, coarse/fine body-mesh change is 0.1003%; medium/fine is
0.00875%. The half-space-only fields are identical across the three body meshes.
The tested Green-kernel 8/12-point rule change is 0.00717%, below its original
0.02% target. Independent singular Duffy and nonsingular Gauss integration
agree within 5.2e-15 relative error; the original square kernels remain equal
in both aligned and rotated controls.

The fine-body pressure maximum is 26.6177 MPa, versus Hertz 26.7025 MPa.
Its moment-equivalent axes are 70.1065/51.1880 um, versus Hertz
69.9328/51.1374 um. The pressure-hat support bounds are approximately
±71.97/52.63 um; those quantized bounds are not certified continuum contact edges.
Independent cell integration and force coefficients both recover 0.2 N.

Adding the fine pressure error, observed grid change, source-table change and
medium/fine body-mesh change gives a 2.202% conditional engineering envelope.
It is an observed variation sum, not a certified continuum or statistical bound.
The displacement-based single-pass curvature correction improves the pressure
error by only 0.088 percentage points, below the observed grid variation. It is
not selected as an additional runtime correction on this evidence.

## Execution, cost and limits

Three initial runs fail because writes to a scikit-fem `DiscreteField` subclass
do not perform the intended coordinate conversion. An independent four-rule
input check reproduces that failure. Explicit NumPy conversion repairs it;
all boundary depths and original source/elastic checks then pass. The earlier
plane-coordinate hypothesis and all failed captured workers remain retained.

Jobs 737601 and 737606 complete the three body studies; 737614 and 737619
complete 97-grid refinements; 737624 independently reduces saved full pressure,
force and gap arrays. Job 737633 checks the Green kernel. The compact
[pressure packet](../evidence/20261019-qualified-ranking/hard-shell-contact-pressure-v1/README.md)
retains all eleven campaigns and exact inputs. The current refinement worker
only clarifies its warm-start description relative to the executed variant;
the complete remaining AST matches, and both source hashes/diff are recorded.
Captured workers and actual seed receipts remain authoritative for replay.

These are offline reference costs: elastic construction takes 81/155/243 s,
and the 25 normal-source responses take 42/55/70 s. The first 97-grid half-space
solve takes 358 s; later meshes reuse its completed seed and construct only
the scalar P2 probing basis, taking 0.67/1.01 s for that preparation. These
costs are excluded from native scene throughput. No coupled pressure solve
is added to the runtime estimator.

This case supports retaining the cheap normal Hertz estimate in its tested
small smooth polar-contact regime. It does not qualify general curvature,
nonconvex/asymmetric moving contacts, tangential memory, complete/membrane/
bending stresses or damage-relevant global stress maxima. The earlier
prescribed-pressure stress reference failures remain unchanged. Matching native
shell/near-field accuracy and useful accepted scene throughput are still required.
