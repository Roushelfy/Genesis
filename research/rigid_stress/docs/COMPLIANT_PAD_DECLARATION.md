# Declared compliant-pad plate case

This additional case is authorized by the qualified-ranking goal. Its parameters
are fixed here before evaluating pad-loaded stress fields. Preserve the original
hard smooth-egg and hard polar-land contact results and their unresolved spatial
qualification. The prescribed C2 pressure replay remains a separate control.

Use the existing full polar-land egg, E=10GPa, nu=0.3, density2000kg/m3 and500um
normal wall. Both horizontal plates carry a compression-only normal spring
foundation with stiffness per projected reference area **1e9N/m3**. This is an
ideal bed of independent axial columns, each with E=1MPa and height1mm; lateral
coupling of a homogeneous three-dimensional rubber layer is outside this
declared constitutive model. The constitutive law is
`p = k_A * max(-g, 0)` and stored energy per area is
`0.5 * k_A * max(-g, 0)^2`. Here g is the signed distance to the undeformed pad
surface and compression corresponds to negative g. The pad exerts no tensile
traction. It is physical compliance, held fixed during spatial/timestep studies.

Start with common normal force0.1N and zero friction, then use0.3N and1N after
the first case passes. Plate travel and contact footprint are solved quantities.
The3mm candidate radius is a numerical search region, checked against the full
outer boundary and enlarged if load reaches its edge; it is never imposed as
the pressure support or fitted against stress error. Use the same pad law,
normal geometry, area convention and measured force in solid and Shell runs.
Retain the distinct Shell contact-envelope/thin-surface reduction and quantify it.

Independent P2 facet quadrature applies axial forces and transfers them through
the same shape functions. For a point of projected area w, pad compliance is
`1/(k_A*w)` in m/N. This adds a positive diagonal term to the complementary
contact energy. Verify force/moment, rigid work, pad constitutive residual,
nonnegative compression forces and the complete elastic equations. Mesh,
quadrature and constitutive checks are separate; negative distance into the
undeformed pad is physical compression and is not a hard-contact gap failure.

The existing2% complete-field/5% raw-peak reference targets and smaller solver
and timestep targets remain in force. Report complete physical-volume-weighted
stress, normal-material membrane/bending resultants and raw unsmoothed extrema.
An admitted normal case does not waive the required sustained friction case,
full mechanical histories, matched costs, stability or final publication.

The first Shell control is explicitly static, using the existing velocity-cleared
elastic relaxation kernels. At each iterate it solves four free rigid/plate
coordinates and recomputes pad pressure from its own displacement. It uses the
same solid-side reference quadrature points, projected areas and initial outer
gaps. The prior force/couple transfer defines a fixed linear map from Shell
vertex displacement to each outer point's axial displacement; its transpose
transfers pressure forces, including the local moment from the thickness offset.
This preserves force, moment and virtual work and makes the pad energy and
forces consistent. The linear through-thickness map is an explicit Shell
kinematic reduction, not a claim that its nonrigid displacement equals solid P2.
The same affine face map checks every omitted boundary face at its vertices.
Original hard-contact Shell histories remain separate, and this static control
cannot qualify continuous dynamics or sustained friction on its own.

The continuous Shell variant uses the same fixed linear map at one outer-face
centroid per triangle, with the original projected face area. Its isolated
engine source is `41b2a57be`, based on the optional-coarse repair `a86ba81d0`;
the [incremental patch](../shell_comparison/patches/shell_physical_pad.patch)
includes the engine and observable regression. Both the point velocity and
the transpose force transfer use this map, including the offset moment, in
the effective compliance, Newton products and preconditioner. The stored
incremental stiffness is `dt^2 * k_A * area`; nonlinear impulse acceptance
uses effective contact mass rather than the physical spring stiffness.
Relative PCG tolerance remains1e-4 in benchmark trajectories, with absolute
velocity tolerance1e-10 and independent actual-equation admission.

This is an explicitly small-deformation, fixed-topology envelope reduction.
The variant requires floating reductions; original contact remains the default.
One centroid is a quadrature approximation, not equivalent to high-order
solid contact integration. In the reference CLI, order0 explicitly selects
one centroid, whereas the library's order1 uses three points. Continuous
trajectories, matched-centroid static controls and high-order reference
comparisons must retain those distinctions. Normal histories do not yet
qualify sustained friction or a spatially resolved reference.

For the sustained-friction extension, fix Coulomb coefficient0.3 and smoothed
slip speed `epsilon=1e-4m/s` before evaluating fields. The isolated engine pin
is `4ce5542f9`, based on normal-pad pin `41b2a57be`; its
[incremental patch](../shell_comparison/patches/shell_pad_friction.patch)
includes the analytical regression. Retain the existing
potential shape: tangential traction magnitude is `mu*p*(2*s-s^2)` for
`s=|v_t|/epsilon < 1`, and `mu*p` otherwise, opposing slip. The normal pressure
bound is lagged by one substep as in the original coupled solver; verify its
equilibrium consistency and timestep dependence separately. The physical-pad
branch must keep epsilon independent of dt. Its old stiffness-based widening
would give `epsilon=max(1e-4, 2*mu*compression/dt)`, since pad stiffness is
stored as `dt^2*k_A*area`; that is a different, timestep-dependent friction law.
Keep the original hard-contact rule unchanged and identify the new pad variant.

Use uniform horizontal gravity `g_x=1m/s^2` on the free egg, starting from the
normal0.1N hold, with stationary horizontal plate coordinates. This supplies
a sustained tangential load without vertex pins or an artificial torque
constraint. Apply the same mass-proportional body load in each reference and
recovery method, and verify net force/moment, local friction law, actual
nonzero shear reactions and inertial terms. Small steady creep is expected
under the declared smooth friction law; it is not zero-velocity equilibrium.
Admit its measured stationary response only with explicit drift/rotation,
kinetic and timestep checks. Extend to other loads after the first admitted
case. This declaration does not claim that the friction extension has passed.

A free flat sheet with horizontal/vertical gravity ratio `r < mu` supplies an
analytic control: normal compression is `rho*h*g_z/k_A`, horizontal reaction
is `-mass*g_x`, and steady slip speed is
`epsilon*(1-sqrt(1-r/mu))`, independent of timestep. Compare1ms and2ms with
the same physical data, retaining the old widening rule as a negative control.

The force-per-area spring convention is documented in
[Spring Foundation and Thin Elastic Layer](https://doc.comsol.com/6.3/doc/com.comsol.help.sme/sme_ug_theory.06.068.html).
This declaration specifies the benchmark's ideal constitutive law; no COMSOL
implementation or result is used as numerical ground truth.
