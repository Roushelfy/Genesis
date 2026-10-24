# Locally refined feature-mesh banks for rigid stress recovery

This document derives the method proposed on 2026-10-24. The derivations below
are elementary consequences of linear variational elasticity and Galerkin
projection; they are not claims that an arbitrary atlas is automatically accurate
or faster. Numerical results must be supplied separately.

The intended input is an already specified physical distributed traction from
the existing finite-pressure DPR law. Geometry, material and thickness are fixed;
contact position, force, radius, direction, count and side may change. Elastic
recovery does not feed deformation back into the rigid contact solve.

## 1. One physical problem, several discretizations

Let a connected bounded body occupy a fixed domain Omega in R^d. For volume
elasticity define

\[
 \epsilon(u)=\tfrac12(\nabla u+\nabla u^T),\qquad
 a(u,v)=\int_\Omega \epsilon(v):C:\epsilon(u)\,dx.
\]

Assume the elasticity tensor is bounded and strictly positive on symmetric
strains. With homogeneous essential constraints sufficient to remove rigid
motion, use the corresponding subspace of H^1. For a free body use

\[
 V=H^1(\Omega;\mathbb R^d)/\mathcal R,
 \qquad \|v\|_a=\sqrt{a(v,v)},
\]

where R is the finite-dimensional rigid-motion space. Korn's inequality makes
this an energy norm on the quotient. A particular gauge merely chooses a
representative of each equivalence class. It must not add physical clamps.
Bounded load functionals are required: finite-area integrable pressures with
appropriate trace regularity satisfy this, whereas ideal point loads can fall
outside V' and need not have finite-energy solutions. Bounds are not asserted
uniformly as a fixed-force pressure footprint shrinks to a point.

For linear shells replace a by the consistent membrane/bending/shear bilinear
form and quotient out its rigid modes. All subsequent operator algebra is the
same. Its stress output remains the stress of that shell theory, not the
complete 3D stress tensor near a contact.

All feature meshes must approximate this same problem. Sharing only the nominal
mesh dimensions is insufficient if their material surfaces, thickness, geometry
maps, inertia relief or actual applied pressure differ.

### 1.1 Common linear inertia relief

An isolated contact load usually has a nonzero resultant, even if several
contacts together balance. Its free-body static response needs a common linear
relief rule. Let r_1,...,r_m span R and let m be a fixed physical mass bilinear
form, positive on R. Define

\[
 G_{ij}=m(r_i,r_j),\quad b_i(\ell)=\ell(r_i),\quad
 \alpha(\ell)=G^{-1}b(\ell),
\]
\[
 (P\ell)(v)=\ell(v)-\sum_j\alpha_j(\ell)m(r_j,v).
\]

Then P is linear and (P ell)(r_i)=0. Consequently

\[
 P\!\left(\sum_c\ell_c\right)=\sum_cP\ell_c.
\]

Use one physical mass/inertia model across the bank. If each mesh independently
changes the mass quadrature, it implements a slightly different P_j; the
resulting load mismatch is another error term, not a harmless gauge change.
Balanced external loads have zero total relief, so their individual relief
contributions must cancel to the tested numerical accuracy.

In the rest of the document ell_c denotes the already balanced functional
P ell_c where needed. Common body loads can be assigned to a separate source;
they must not be repeated once per contact.

## 2. Exact superposition and its mixed-mesh approximation

For a supplied load functional ell in V', let T ell in V satisfy

\[
 a(T\ell,v)=\ell(v)\qquad\text{for all }v\in V.
\]

The operator T is linear. With fixed constitutive law, the stress map
S u=C:epsilon(u) is linear as well. Hence

\[
 u=T\sum_c\ell_c=\sum_cT\ell_c,\qquad
 \sigma=ST\sum_c\ell_c=\sum_cST\ell_c.
\tag{1}
\]

This requires supplied tractions, not a claim that a coupled nonlinear contact
problem is linear. Contact prediction can be nonlinear; once its current loads
are fixed, the elastic recovery considered here is linear.

### 2.1 Definition of the feature-mesh bank

Construct conforming spaces V_j, each resolving a different material region.
For a given contact choose an index j(c) whose fine region covers its current
footprint with adequate margin. Let T_j ell in V_j satisfy

\[
 a(T_j\ell,v_j)=\ell(v_j)\qquad\text{for all }v_j\in V_j.
\]

The proposed approximation is

\[
 \widehat u=\sum_cT_{j(c)}\ell_c,\qquad
 \widehat\sigma(x)=\sum_cST_{j(c)}\ell_c(x).
\tag{2}
\]

Each term is a **full-body response to its assigned load**. It can be evaluated
at an arbitrary common physical point even though its underlying mesh differs.
No universal union mesh or new mesh construction is required online.

If several contacts choose the same mesh, combine their loads before solving:

\[
 \widehat u=\sum_jT_j\left(\sum_{c:j(c)=j}\ell_c\right).
\]

Thus the number of online solves is the number of active bank entries, not
necessarily the number of contacts. Do not apply the full load to every mesh
and then sum the responses; that would multiply the external loading.

### 2.2 Energy-error bound

**Proposition.** Under the common-form, conforming assumptions above,

\[
 \left\|T\sum_c\ell_c-\widehat u\right\|_a
 \leq\sum_c\inf_{v_j\in V_{j(c)}}\|T\ell_c-v_j\|_a.
\tag{3}
\]

**Proof.** Galerkin orthogonality gives
`a(T ell_c - T_j ell_c, v_j)=0`. Therefore T_j ell_c is the a-orthogonal
projection of T ell_c and its energy error equals the best approximation error
in V_j. Apply (1), subtract (2), and use the triangle inequality. QED.

The bound justifies combining individually accurate responses. It does not say
that local refinement always improves the total error: errors can cancel or
reinforce. Nor does it require two contacts to be far apart. Nearby contacts
still obey (1); their source responses simply need sufficient accuracy.

If each source has relative energy error at most epsilon_c, the relative error
of the sum obeys only

\[
 \frac{\|u-\widehat u\|_a}{\|u\|_a}
 \leq \frac{\sum_c\epsilon_c\|T\ell_c\|_a}
             {\|T\sum_c\ell_c\|_a}
 \leq (\max_c\epsilon_c)\,
       \frac{\sum_c\|T\ell_c\|_a}{\|T\sum_c\ell_c\|_a}.
\tag{3a}
\]

The last ratio is a response-cancellation factor and can be large or undefined
for a zero response. Cancellation of **elastic fields** is the relevant issue;
a zero net rigid resultant alone does not imply such cancellation, as balanced
opposing forces can create a substantial compressive field. Report absolute
errors and the normalization scale for near-cancelling tests rather than
inheriting per-source relative tolerances for the total field or scalar peak.

### 2.3 Mixed-mesh summation is not a union-space Galerkin solve

Let X contain all selected V_j. In general the residual

\[
 r(v)=\ell(v)-a(\widehat u,v)
     =\sum_c[\ell_c(v)-a(T_{j(c)}\ell_c,v)]
\tag{4}
\]

does not vanish for all v in X. Each term vanishes only on its own test space.
The mixed-mesh sum is therefore not generally identical to the solution on a
mesh refined near all contacts simultaneously.

A two-dimensional algebraic counterexample is enough. Take

\[
 K=\begin{bmatrix}2&1\\1&2\end{bmatrix},\quad
 f_1=(1,0)^T,\quad f_2=(0,1)^T,
\]

with V_1=span(e_1), V_2=span(e_2). Individual Galerkin responses sum to
`u_hat=(1/2,1/2)`, while the full solution is `(1/3,1/3)`. The full residual is
`f-K u_hat=(-1/2,-1/2)`. It would be incorrect to advertise (2) as an exact
direct solver on the joint-refined discretization.

For a finite-dimensional common reference space and SPD stiffness K, the exact
energy error satisfies

\[
 \|u-\widehat u\|_K^2=r^TK^{-1}r.
\tag{5}
\]

This identity is useful for CPU verification. A cheap online approximate
preconditioned residual is an indicator unless spectral equivalence constants
make it a proven bound. A residual measured only in the original coarse test
space cannot detect every missing fine-scale error.

### 2.4 Quadrature, geometry and load-model errors

Physical pressure must be integrated on each actual selected material surface.
Identical resultant force alone does not mean identical traction. Use the same
pressure law, footprint, application wrench and source-side identity, then
discretize its virtual work consistently.

If a lifted discretization instead solves
`a_j(u_j,v_j)=ell_j(v_j)`, define

\[
 \eta_{\ell,j}=\sup_{v_j\ne0}
 \frac{|\ell(v_j)-\ell_j(v_j)|}{\|v_j\|_a},\qquad
 \eta_{a,j}=\sup_{v_j\ne0}
 \frac{|a_j(u_j,v_j)-a(u_j,v_j)|}{\|v_j\|_a}.
\]

Comparing u_j to the exact a-Galerkin projection and testing their difference
gives

\[
 \|T\ell-u_j\|_a
 \leq\inf_{v_j\in V_j}\|T\ell-v_j\|_a
      +\eta_{\ell,j}+\eta_{a,j}.
\tag{6}
\]

Sum these bounds for the sources. Curved meshes with different geometric
domains first need well-defined lifts to a common domain; geometry-Jacobian
and traction transformations contribute to the consistency terms. Without
those lifts, reporting (3) as a theorem for the actual meshes is unjustified.
This is why an exactly common planar/polyhedral geometry is a valuable first
CPU control, followed separately by smooth curved geometry.

## 3. Common coordinates and stress evaluation

Evaluate every contribution at the same physical location x. Locate x in each
selected mesh, evaluate the FE gradient there, and transform to a common frame.
For a shell with local tangent frame R_j use `sigma_global=R_j sigma_local R_j^T`
before adding; also match the same physical inner/outer surface.

Nodal stress averaging or matching arrays by index is not a substitute for this
operation. At a material interface or element boundary specify the trace or
averaging convention. The numerical comparison set must be identical across
methods. Peak values over different per-mesh Gauss grids mix field error and
sampling error.

For volume elasticity the energy error controls the compliance-weighted L2
stress error exactly:

\[
 \int_\Omega\delta\sigma:C^{-1}:\delta\sigma\,dx
   =\|u-\widehat u\|_a^2.
\tag{7}
\]

It does **not** supply a mesh-independent pointwise or maximum-stress bound.
Local regularity and sufficiently resolved evaluation regions are additional
requirements. A small displacement, energy or global L2 error can coexist with
a large local peak error.

## 4. Coarse response plus refinement corrections

Let T_0 be a common coarse solver. By linearity,

\[
 \widehat\sigma
 =ST_0\!\left(\sum_c\ell_c\right)
  +\sum_cS(T_{j(c)}-T_0)\ell_c.
\tag{8}
\]

Equation (8) is exactly (2), not a new approximation, provided the complete
corrections and consistent source maps are retained. It avoids counting the
coarse response once for every source. The difference operators may be
precomputed or compressed. They are numerical differences between FEM
approximations of one physical problem, not analytic particular-stress
enrichment or an independent Hertz stress field pasted onto a coarse solve.

### 4.1 Local truncation is an additional approximation

Replacing each correction stress delta_sigma_j by
`chi_j(x) delta_sigma_j(x)` discards a tail. Its direct norm error is bounded by
the sum of the discarded-tail norms, but no assumption above proves those tails
small. A thin shell, U-shaped ligament or bending-dominated body can transmit
nonlocal corrections.

Moreover,

\[
 \operatorname{div}(\chi\,\delta\sigma)
 =\chi\operatorname{div}(\delta\sigma)+\delta\sigma\nabla\chi.
\]

The second term is an artificial equilibrium residual in the transition zone.
If instead a displacement correction is localized, the constitutively
consistent strain is

\[
 \epsilon(\chi\,\delta u)
 =\chi\epsilon(\delta u)+\operatorname{sym}(\delta u\otimes\nabla\chi).
\]

Dropping that derivative term is inconsistent. Localized bases in a common
variational problem provide a principled alternative to blind tensor clipping.

## 5. Selecting and blending feature meshes

The bank centre is a coverage label, not a replacement contact position. Put
the actual pressure footprint at its actual location within the chosen fine
region. Fine-region coverage controls centre spacing; fine element size controls
the ability to resolve the pressure and stress gradient. They are different
length scales.

Let p collect contact parameters and choose nonnegative scalar weights
`w_j(p)`, summing to one over valid bank entries. For one contact define

\[
 \widehat u(p)=\sum_jw_j(p)T_j\ell(p).
\tag{9}
\]

If each T_j has error E_j(p), convexity gives

\[
 \|T\ell(p)-\widehat u(p)\|_a\leq\sum_jw_j(p)E_j(p).
\tag{10}
\]

With continuous weights and continuous ell(p) in V', (9) is continuous in the
energy norm. A hard switch i to j at fixed p has jump bounded by E_i(p)+E_j(p).
The same statements hold for a uniform stress norm when its E_j bounds have
been established independently. Continuity alone is not accuracy.

Here w_j depends on source parameters and is constant with respect to the
spatial observation coordinate x for each source. Spatially blending solution
pieces is a different construction and incurs the gradient terms of Section 4.1.
Negative interpolation weights replace sum(w_j E_j) by sum(|w_j| E_j), which
can amplify error.

Simply interpolating snapshots for loads applied at different centres need not
preserve the intended load, wrench or sharp peak location. It approximates both
the load manifold and its response. A nearby finer mesh evaluated under the
true current pressure is the safer first implementation.

## 6. Force-dependent radius and friction

Linearity holds with respect to the distributed load after it has been built.
For a fixed position and fixed shape of traction, three force-direction basis
responses suffice if the load mapping itself is linear in the force vector.
However, Hertz sizing gives

\[
 a(F_n)=\left(\frac{3F_n}{4E^*\bar\kappa}\right)^{1/3}.
\]

The complete map `F -> S T ell(p,a(F_n),F)` is therefore not generally linear
in F. The mature DPR anchor and nonnegative pressure fit may also depend on
force direction. Scaling one unit-force contact snapshot is not automatically
correct even at a fixed nominal centre.

Caching K_j factors or exact nodal-load response columns avoids this problem:
construct the actual finite traction and nodal generalized load at runtime,
then apply the linear response. A snapshot atlas must explicitly cover radius,
position, direction-dependent pressure shape and other changing input parameters.

For shells the generalized virtual-work map includes both surface force and
eccentric couple. With side s=+/-1 and nodal director d_i,

\[
 f_i=\int N_i t\,dA,\qquad
 m_i=\int N_i\,(s h d_i/2)\times t\,dA.
\]

Preserve these terms and source-side provenance when generating or combining
responses. The method does not infer local stick/slip traction from a rigid
resultant unless that is already part of the supplied load law.

## 7. Exact static condensation and a shared enriched model

Response summation is a convenient first approximation. A more mechanically
consistent common model can retain interface coupling while removing local
fine unknowns offline. For one selected discretization partition DOFs into
local interiors I and retained interface/coarse DOFs Gamma:

\[
 \begin{bmatrix}K_{\Gamma\Gamma}&K_{\Gamma I}\\
 K_{I\Gamma}&K_{II}\end{bmatrix}
 \begin{bmatrix}u_\Gamma\\u_I\end{bmatrix}
 =\begin{bmatrix}f_\Gamma\\f_I\end{bmatrix}.
\]

Assuming K_II is invertible,

\[
 S_\Gamma=K_{\Gamma\Gamma}-K_{\Gamma I}K_{II}^{-1}K_{I\Gamma},\quad
 g_\Gamma=f_\Gamma-K_{\Gamma I}K_{II}^{-1}f_I,
\]
\[
 S_\Gamma u_\Gamma=g_\Gamma,\qquad
 u_I=K_{II}^{-1}(f_I-K_{I\Gamma}u_\Gamma).
\tag{11}
\]

This is an exact algebraic reduction of that selected FE system. It does not
remove the need to represent the interior load or recover the desired stress.
Interior factors, load maps, interface maps and stress maps can be prepared
offline; online work operates on retained ports and active loads.

For disjoint subdomain interiors the Schur contributions assemble through a
common interface space. Overlapping refinements must be handled as a common
compatible space, a partitioned domain or a selected union; summing independent
overlapping element stiffnesses double-counts material.

### 7.1 Shared coarse factor with selected enrichment modes

In a common conforming representation write

\[
 u=V_0q+W_Az,
\]

where V_0 is a shared coarse basis and W_A contains the selected local modes.
Galerkin projection gives the block system

\[
 \begin{bmatrix}K_{00}&K_{0A}\\K_{A0}&K_{AA}\end{bmatrix}
 \begin{bmatrix}q\\z\end{bmatrix}
 =\begin{bmatrix}b_0\\b_A\end{bmatrix}.
\]

Remove rigid modes and dependent enrichment columns first. The projected
matrix is then SPD if the underlying physical form is SPD. Eliminating q gives

\[
 (K_{AA}-K_{A0}K_{00}^{-1}K_{0A})z
   =b_A-K_{A0}K_{00}^{-1}b_0.
\tag{12}
\]

The coarse factor and candidate cross-response blocks can be precomputed.
Online only a small active Schur system changes. Importantly, even disjoint
local supports can have nonzero condensed cross-coupling through the global
coarse inverse. Dropping off-diagonal blocks is another approximation requiring
an error test; proximity is not the only determinant of interaction.

This consistent enriched Galerkin model is distinct from the independent
per-source summation (2). It can be used as an intermediate or stronger
reference in the CPU study. It also has a different offline-memory/online-solve
trade-off, so its speed cannot be inferred from the response-bank timings.

## 8. Maximum stress and sampled outputs

Full stress tensors must be added before computing scalar invariants. For
symmetric tensors A and B,

\[
 |\operatorname{VM}(A)-\operatorname{VM}(B)|
 \leq\sqrt{3/2}\,\|A-B\|_F,
\]

because VM(A)=sqrt(3/2)||dev(A)||_F. Weyl's inequality gives

\[
 |\lambda_{\max}(A)-\lambda_{\max}(B)|
 \leq\|A-B\|_2\leq\|A-B\|_F.
\]

The positive maximum principal stress `max(0,lambda_max)` has the same bound.
For a common sample set X,

\[
 |\max_{x\in X}q(\widehat\sigma(x))-
   \max_{x\in X}q(\sigma(x))|
 \leq L_q\max_{x\in X}\|\widehat\sigma(x)-\sigma(x)\|_F.
\tag{13}
\]

This is a sampled-maximum bound. A continuum maximum requires a uniform stress
error bound and sampling control. If q(sigma) is Lipschitz on a region with
constant L_x and X has fill distance delta, its continuum-minus-sampled maximum
is at most L_x delta. Such regularity may fail at corners, interfaces or ideal
point contacts, so no unqualified continuum maximum is asserted here.

Scanning only a source's own fine patch misses a possible peak at a remote
neck, reentrant corner or thin ligament. Evaluate the summed tensor on a global
coverage set plus appropriate local samples, and refine candidate regions.
Taking the largest individual-source maximum is not equivalent to (13).

## 9. What local refinement can and cannot fix

1. It reduces spatial discretization error where the FE space actually becomes
   capable of representing the response. It cannot repair an inaccurate radius,
   traction profile, application wrench or reference geometry.
2. Pure shell refinement cannot restore the 3D contact-normal component omitted
   by shell theory. Local volume refinement is the appropriate controlled test
   when the target is a 3D near-contact peak. A shell/solid hybrid additionally
   needs a mechanically consistent coupling and its own validation.
3. Finite footprint removes the ideal point-force load singularity on a smooth
   regular boundary, but does not guarantee a bounded stress at a sharp geometric
   or boundary-condition singularity, or at a discontinuous traction edge.
   State geometry, pressure regularity, physical rounding, depth, sample set
   and peak definition explicitly.
4. A small energy/global-field error does not certify the maximum. Accuracy and
   continuity of the peak require direct tests at the target scale.
5. Every new bank entry must cover a region, not merely one source centre.
   Coverage and approximation indicators must reject out-of-bank inputs or
   choose a complete fallback; silently snapping the source changes the problem.

## 10. Required CPU verification matrix

Separate implementation agreement, atlas approximation, discretization error,
load-model validity and output-sampling error.

| Control | What must be compared |
|---|---|
| Algebraic superposition | Same-matrix per-source sum versus combined solve; should agree to solver roundoff |
| Mixed-space counterexample | Equation (4) need not vanish on the joint space; do not falsely claim exactness |
| Common-load work | Force, moment, virtual work and shared inertia relief across meshes |
| Coarse-plus-delta | Equation (8) versus direct mixed-mesh sum before any truncation/compression |
| Joint refinement | Mixed-mesh bank versus a mesh refined near all current sources |
| Refined reference | At least two suitable reference resolutions; retain reference uncertainty |
| Moving source | Continuous sweep through interiors, overlap and hard-switch boundaries |
| Radius and force | Vary current radius and force; detect incorrect unit-snapshot scaling |
| Friction and side | Rotate/reverse tangential load; preserve shell eccentricity and inner/outer identity |
| Multiple sources | Nearby/far contacts, changing count, release; compare summed tensors and true invariants |
| Nonconvex structure | Remote neck/reentrant-region peaks, not only contact neighbourhoods |
| Condensation | Equation (11)/(12) against the corresponding uncondensed selected FE system |
| Truncation/compression | Full correction versus retained local/low-rank version, including far-field tail |
| Cost | Same output points, same error budget, setup/time/memory and active-bank grouping |

Use cached factors and shared bank storage. Report actual numbers of unique
bank entries, active entries and DOFs; excluded setup work must be recorded
separately. Dense response storage costs O(sum_j n_output n_input,j), while
factor storage costs sum_j nnz(L_j). Neither is automatically small. The atlas
may be faster than a globally refined model at matched error while remaining
slower than the original coarse model. No GPU throughput follows from CPU data.

## 11. Related primary work and relation to this proposal

- D. L. James and D. K. Pai, *Multiresolution Green's Function Methods for
  Interactive Simulation of Large-scale Elastostatic Objects*, author preprint:
  https://www.cs.cornell.edu/~djames/papers/JamesPai02TOG.pdf . This supports
  precomputed linear responses, multiresolution representations and explicit
  storage/latency trade-offs. It does not validate this study's contact peaks.
- I. Babuska and J. M. Melenk, *The Partition of Unity Method*, IJNME 40 (1997),
  727-758: https://ivo100.oden.utexas.edu/assets/papers/ijnme97.pdf . This provides
  a framework for conforming spaces with locally chosen approximation functions;
  arbitrary stress blending is not the same construction.
- D. B. P. Huynh, D. J. Knezevic and A. T. Patera, *A Static Condensation Reduced
  Basis Element Method: Approximation and a Posteriori Error Estimation*, ESAIM
  M2AN 47 (2013), 213-251, DOI 10.1051/m2an/2012022:
  https://www.numdam.org/item/M2AN_2013__47_1_213_0/ . This develops offline
  component libraries and online port coupling, including elasticity examples.
  Its certified reduced-basis assumptions must not be presumed for an arbitrary
  hand-built feature mesh bank.

The application-specific proposal here is a finite-pressure, changing-contact
stress-recovery bank with measured complete-field/peak errors and native batch
execution. Its usefulness requires the numerical experiments above; the basic
linear superposition and condensation ideas are established mathematics.
