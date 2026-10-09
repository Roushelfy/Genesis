# Finite pad traction model

The live adapter accepts the complete point position, force, inward contact
normal, effective solver friction coefficient and an explicitly supplied
finite radius at every contact. All finger, table and other egg contacts enter
the same mapper. Rolling and torsional contact friction are disabled in this
scene. The radius is an assumed pad input, rather than a rigid collision output.

The production law first places the footprint on the exterior intersection of
the source force line. Write c = p - t F/|F| and choose the outgoing full-shell
face intersection. The complete source wrench is unchanged because
(c-p) x F = 0. This explicit anchor is applied to every active force, including
penetrated contact points. It neither changes the force nor widens the footprint.
The raw-point anchor remains available as a distinct comparison law. An absent
intersection is rejected with status 6.

For exterior quadrature positions x_q, center c and radius r, the prior weights
are the surface integration weights times

```text
exp(-0.5 |x_q-c|^2 / (0.45 r)^2) max(0, 1-|x_q-c|^2/r^2)^2.
```

Normalize the prior to w_q. The selected compliant pad law uses nonnegative
pressure coefficients a_q and a constant tangential-to-normal force ratio:
f_q = a_q F. Minimize sum (a_q-w_q)^2/(2 w_q), subject to

```text
sum a_q = 1
sum a_q (x_q-p) x F = 0
a_q >= 0.
```

Two orthonormal directions transverse to F reduce these constraints to three
independent equations. Their convex dual uses
`a_q = w_q max(0, 1-h_q^T lambda)`. A damped Newton solve preserves the
complete point wrench with nonnegative pressure. The consistent six-node
quadratic face shape functions transfer these physical quadrature forces to
the volume nodes. Some consistent nodal weights are signed.

Compression and Coulomb friction refer to the **supplied contact pad normal**.
A finite compliant pad's contact frame can differ from the normals of the
undeformed, faceted reference shell. The assumed traction law keeps this frame
fixed within each supplied footprint. This is an additional physical model.
The stress result depends on that model and its radius. It requires radius
sensitivity and physical mesh convergence before accuracy acceptance.

The alternative vector-traction fit in `WrenchFit` imposes the separate local
shell-face friction cones and can preserve pure moments. Its dual solves six
equations. Synthetic unilateral/friction/pure-moment cases pass. One actual
sliding wrench at step 424 remains incompatible or unresolved under that law
even with surface quadrature orders 10, 16 and 24. That diagnostic remains
published. The live pad law is selected explicitly, rather than used as a
hidden failure fallback.

For live FP32 collision data, the source force may exceed the supplied circular
friction cone by rounding. The adapter retains the original force and allows
at most `16 * eps(source precision) * |F|` in the cone inequality. It reports
the actual excess and allowance separately. This allowance applies to source
admissibility. Integrated force and moment must still satisfy the independent
FP64 normalized `1e-8` conservation targets. Larger violations are rejected.
The integration scene explicitly selects the elliptic solver friction cone.

`PadPressureGPU` offers both a complete exterior scan and an exact geometry-only
uniform grid. The grid enumerates every candidate quadrature sample in the
current radius bounding box, then applies the same distance test, dual solve
and consistent scatter. There is no fixed contact basis or per-footprint sample
cap. Both modes support changing positions, directions, counts and radii, and
allocate no contact-by-surface tensor. A
failed input, empty/rank-deficient footprint or failed conservation check
sets a device status. Callers must gate observations with mapping and recovery
acceptance. The diagnostic CLI reads these flags and rejects failure.

Contact forces and positions are produced before rigid integration. Recovery
therefore uses that solve's pre-integration authored pose and angular velocity.
`Scene` exposes pre/post physical-substep observers. The adapter captures the
pre-integration authored pose, then observes every completed contact solve;
the policy receives the maximum over all substeps. Plane/egg GPU tests at one
and four substeps pass independent CPU full-field peak checks. The default
Panda performance profile uses dt=0.01 s and one substep.

The current Panda profile explicitly uses FP64 rigid arithmetic, convex contact
resolution, the elliptic friction cone, 100 iterations, tolerance 1e-12, no
contact pruning and disabled spin/rolling friction. Earlier Signorini source
forces exceeding the supplied cone by more than roundoff remain rejected;
changing resolution is a declared scene input, not an expanded admissibility
cone. Finite Newton-Euler imbalances match the solver gradient in checked
trajectories and are reported separately from auxiliary equilibrium error.

Full levels 4/5/6, unchanged radii including the 4.08 mm minimum, and quadrature
6/10/16 pass the proposed 2% final-refinement criterion for 12 asymmetric and
six actual nominal grasp snapshots. The largest final surface-mesh change is
1.365%; an independent one/two/four wall-layer check at surface level 4 changes
by at most 1.063% in the final wall refinement. These empirical checks do not
prove a universal physical error bound or identify unique contact compliance.
