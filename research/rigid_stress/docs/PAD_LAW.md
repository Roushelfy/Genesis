# Finite pad traction model

The live adapter accepts the complete point position, force, inward contact
normal, effective solver friction coefficient and an explicitly supplied
finite radius at every contact. All finger, table and other egg contacts enter
the same mapper. Rolling and torsional contact friction are disabled in this
scene. The radius is an assumed pad input, rather than a rigid collision output.

For exterior quadrature positions x_q, center p and radius r, the prior weights
are the surface integration weights times

```text
exp(-0.5 |x_q-p|^2 / (0.45 r)^2) max(0, 1-|x_q-p|^2/r^2)^2.
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

`PadPressureGPU` scans the complete exterior quadrature domain with one block
per padded contact. This correctness baseline supports changing positions,
directions, counts and radii. It allocates no contact-by-surface tensor. A
failed input, empty/rank-deficient footprint or failed conservation check
sets a device status. Callers must gate observations with mapping and recovery
acceptance. The diagnostic CLI reads these flags and rejects failure.

Contact forces and positions are produced before rigid integration. Recovery
therefore uses that solve's pre-integration authored pose and angular velocity.
The present scene uses one substep per transition. Multiple-substep sampling
still requires an explicit integration hook and validation.
