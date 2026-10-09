# Equations and implementation

## 1. Fixed offline operators

Build the entire shell-wall tetrahedral mesh. Let x_i be authored reference
coordinates, r_i=x_i-COM and u elastic displacement in the same frame. P2 has
four vertices and six edge nodes; use exact degree-two stiffness quadrature
and consistent mass integration. Cache:

```text
K = integral B^T D B dV                 # ndof x ndof, sparse symmetric PSD
M = integral rho N^T N dV              # ndof x ndof, consistent mass
R_i = [I_3, -skew(r_i)]                # six rigid modes, ndof x 6
MR = M R
G = R^T M R                           # 6 x 6 SPD
surface quadrature positions, weights, P2 connectivity/shapes
barycentric gradients and volume element/node connectivity
```

K R=0 in exact arithmetic. The six null modes are translation/rotation, not
elastic strain. Choose six independent scalar rows `pins` with nonsingular
R[pins,:]; this fixes a coordinate gauge. Factor K_ff once on the remaining
rows. This does not attach the egg to six physical supports. Contact movement
changes the RHS, never these pins. Verify the residual of the **full** equation
after solve, including pinned rows. A compatible load then has no artificial
reactions there. Compare peaks under an alternative independent gauge.

Shared geometry/operator identity means one K and one factor per group, not
one factor per environment. Cache G's small Cholesky as well.

## 2. Contacts and complete external forces

Transform all world positions and vectors to the authored material frame:

```text
p_body = Q^T (p_world - t_world)
F_body = Q^T F_world
omega_body = Q^T omega_world
```

No position translation applies to force vectors. Preserve the force applied
to the egg, not the force on the opposite geom. Assemble finite vector
traction samples with consistent surface P2 shapes:

```text
f_a += sum_q area_weight_q * N_a(q) * traction_q
```

P2 corner shape integrals may be negative; replacing them with positive-only
lumped weights changes the FE load. A mapper must check resultant force and
moment against the **chosen traction law**, and then check consistency with
the rigid contact wrench. A finite patch with constant force direction usually
has moment `centroid_patch x F`, not necessarily `p_contact x F`. The seed
reports this discrepancy explicitly. A corrected traction model may require
a constrained force/moment fit; check the friction cone and positivity after
correction. Do not add arbitrary couples and claim Coulomb admissibility.

Include gravity, table/support contacts, applied wrenches, drag or other
external loads. Pure spin/rolling moments require a distributed couple or
consistent equivalent traction; the public linear contact-force getter does
not include those moments. Their solver rows must be exported if enabled.

## 3. Moving-frame inertia relief

For inferred rigid acceleration from complete loads, first subtract the known
centrifugal acceleration field:

```text
c_i = omega x (omega x r_i)
f0 = f_external - M c
eta = G^-1 R^T f0
b = f0 - MR eta
```

Thus R^T b=0 and solve K u=b. `eta` contains linear/angular acceleration
coefficients needed to balance net wrench. Translation/angular acceleration
are in the rigid-mode span; centrifugal acceleration generally is not and
must not be forgotten. Gravity in free fall cancels by inertia relief, while
nonuniform contact loading leaves elastic deformation.

Alternative measured-acceleration mode uses Genesis's classical acceleration
at the authored link origin, angular acceleration and omega, converted into
the material frame. Transport linear acceleration to COM with
`a_COM=a_origin+alpha x d+omega x (omega x d)` and use
`a_i=a_COM+alpha x r_i+omega x (omega x r_i)`.
Then `b=f_external-M a`. Diagnose R^T b from all loads and matching substep;
do not silently project an inconsistent measured state and call it equivalent.
Report the wrench mismatch before any deliberately selected inertia-relief
fallback. Rigid mass/COM/inertia must match FEM's shell mass moments.

Elastic inertia and elastic velocity terms are neglected by the quasi-static
recovery model. This is an auxiliary stress estimate, not coupled dynamics.

## 4. Batch direct baseline

For B environments stack b as [ndof,B], gather free rows, solve one shared
factor with multiple RHS, scatter u to full rows with gauge coordinates zero.
CPU uses SciPy SuperLU `factor.solve(B)`; GPU target uses sparse shared
Cholesky/LU plus multi-RHS solves. Validate permutations, transpose modes,
scaling and leading dimension against separate CPU solves.

Factorization/analysis belong offline. Do not refactor per frame. Never replace
K by its dense inverse in the scalable backend.

## 5. Previous-frame prediction and residual projection

Each environment maintains its own last two displacement fields. With uniform
sample interval, predict `u_pred=2 u_last-u_before`; on first reuse hold u_last,
and on cold start use zero. Variable sample intervals require time-scaled
extrapolation. Recompute the current complete residual `r=b-K u_pred`.

Maintain at most h recent **correction directions**, Q=[q_1,...,q_h], and
A=KQ. For residual-metric projection:

```text
Gram = A^T A
coefficient = solve(Gram, A^T r)
u_pred += Q coefficient
r = b - K u_pred                     # recompute, not r - cached A coefficient
```

The energy-metric alternative uses Q^T KQ and Q^T r. Residual projection was
used in the latest batch/history study. Check conditioning/rank and reject
unstable directions; fallback must remain correct.

Compare every environment's complete residual with `atol+rtol*||b||` or the
documented max-form threshold, including gauge rows. Zero-load environments
receive zero stress. A residual pass is a numerical equation test, not a
general mathematical bound on peak-stress error.

## 6. Compact only failed environments

```text
failed_e = ||r_e|| > tolerance_e
ids = select(failed)
delta_free = shared_factor.solve(r_free[:, ids])
u[:, ids] += scatter_free(delta_free)
```

An accepted environment never pays triangular solve work. A failed environment
is corrected even if all others pass. Handle 0, 1 and B failures, and arbitrary
ID order. Recompute final full residual and refine/FP64-correct if roundoff
prevents acceptance. Reset state for only the environments that reset.

When failure density/strict precision makes history more expensive than direct
batch solves, select the direct path automatically using past measured costs.
Do not obtain future acceptance information from the correctness oracle.

## 7. Low-overhead history updates

Only corrected environments append delta. Keep FIFO slot indices rather than
physically shifting [ndof,h] columns. Drop the oldest slot, compute `a=K delta`,
perform two orthogonalization passes in the chosen metric, normalize, and
write q/a into the slot. Recompute only the affected Gram row and column:

```text
Gram[:,slot] = A^T A[:,slot]
Gram[slot,:] = Gram[:,slot]^T
```

Inactive slots must not enter the small solve. Rank checks should not destroy
an old useful slot before a new direction is accepted. The historical CPU
version is the starting point, not a proof of robustness under all precision
or singular-history cases. Periodically check A=KQ to detect cache drift;
current acceptance always uses fresh K u.

## 8. Global peak without storing stress distribution

On a straight tetrahedron P2 shape gradients are affine, so linear elastic
stress is affine in barycentric position. VM is a norm of a linear transform
of stress and hence convex. For barycentric lambda:

```text
VM(sigma(lambda)) <= sum_c lambda_c VM(sigma(c)) <= max_c VM(sigma(c))
```

Therefore evaluate exactly four corners of **every** element and reduce to
the maximum. This is an exact property of the discrete affine P2 field; it
does not remove mesh error. No contact-dependent shortlist is allowed in the
baseline.

For isotropic material, with engineering strains exx,eyy,ezz,gxy,gyz,gxz,
the hydrostatic lambda contribution cancels from VM:

```text
VM^2 = mu^2 * [2((exx-eyy)^2+(eyy-ezz)^2+(ezz-exx)^2)
               +3(gxy^2+gyz^2+gxz^2)]
```

Reuse each tetrahedron's vertex contribution across corners, add its corner
vertex and three incident midpoint terms, and subtract one element-local
displacement before arithmetic to reduce rigid-translation cancellation.
Cache 4x3 barycentric gradients instead of 4x10x3 corner gradients when memory
traffic dominates. Compare against `baseline_numpy` constructing full D B u.

## Sources and provenance

Equations above are the mathematical model implemented/validated in the
imported `rigid_stress_reference.py`, `eggshell_convergence_cpu.py`,
`general_peak.py`, `temporal_benchmark.py` and `benchmark.py`; the convexity
argument is given explicitly here rather than attributed to an unrelated paper.
For sparse solve contracts see SciPy's official
[SuperLU documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.SuperLU.html).
Genesis integration and NVIDIA implementation references are in the adjacent docs.
