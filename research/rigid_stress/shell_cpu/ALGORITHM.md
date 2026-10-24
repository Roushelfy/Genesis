# CPU automatic-radius shell DPR: algorithm and migration notes

## 1. Scope and implementation status

This package is a **CPU research prototype**, not a delivered Genesis shell
feature. It uses NumPy/SciPy to test a fixed linear shell recovery operator.
Rigid dynamics supplies current forces and application points; elastic
displacement is an observation and is not fed back into collision or motion.
Only reference shape, thickness and material are fixed. Contact locations,
force directions, friction, symmetry, active count and inner/outer side may vary.
Production migration must stay in the existing Genesis rigid solver and use
Quadrants for numerical work. No GPU throughput has been measured here.

Files and retained evidence:

| File | Role |
|---|---|
| `core.py` | Linear MITC4-like curved-reference shell, factors and stress maps |
| `geometry.py` | Plates, sphere/egg quad meshes and analytical curvature |
| `contacts.py` | Hertz sizing, projected finite traction and checked fallback |
| `performance.py` | Batched transfer and exact response-operator experiments |
| `study.py` | Egg resolution/load cases and a nonconvex torus control |
| `review_checks.py` | Rigid modes, patches, analytical shells and load controls |
| `reference.py` | Independent pyfe3d and analytical comparisons |
| `solid_comparison.py` | Matched prescribed-load comparison with volume FEM |
| `contact_reference.py` | Independent half-space contact control, not shell contact |
| `precision_study.py` | Solve/response precision experiments and residual checks |
| `results/performance_confirmation.json` | Bounded interleaved CPU timing |
| `review_results/`, `reference_results/`, `solid_results/` | Accuracy evidence |

## 2. Fixed shell operator

Each Q4 node has global translation and infinitesimal rotation:
`q_i = (u_x,u_y,u_z,theta_x,theta_y,theta_z)`. The director increment is
`delta_d_i = theta_i x d_i`. Interpolate the reference midsurface and directors;
include the reference-Jacobian thickness derivative in the bending operator.
This is important for six exact rigid modes on warped/curved-reference quads.

Assemble once, in the rigid body's reference frame:

    K = integral(Bm.T A Bm + Bb.T D Bb + Bs.T S Bs + Kdrill) dA
    A = h C_plane_stress; D = h^3 C_plane_stress / 12
    S = (5/6) G h I

Covariant edge-midpoint assumed shear reduces the elementary Mindlin locking
mechanism. A weak consistent drilling term controls independent director spin.
The prototype is not a claim of locking-free behavior on every distorted mesh.
Patch, thin-plate, sphere, cylindrical-bending and drilling-sensitivity controls
are therefore retained rather than relying on a single egg example.

Share connectivity, reference geometry, stiffness, stress maps and factors
across matching environments. Recompute only current loads and observations.
Uniform rigid translation/rotation changes the body-to-world transform, not K.

For a free object, remove the six rigid-mode resultant loads by inertia relief:

    R = six rigid displacement modes
    f_bal = f - M R (R.T M R)^(-1) R.T f

Here M is the declared lumped translation/rotary mass. Contact transfer first
preserves its supplied wrench; inertia relief is a separate elastic equilibrium
step. It must not be mistaken for changing the contact traction model.
Fix a gauge for the six null modes, factor the remaining K once and reuse it.
Do not introduce physical clamps merely to make the factorization invertible.

Historical timings use FP64 and the `legacy` gauge. A separate distributed
translation-only gauge changes conditioning without changing K or inertia
relief. See `review_results/gauge_checks.json` and precision reports before
choosing a GPU gauge/precision; old timing is not a new precision benchmark.

## 3. Current contact size and finite traction

The input normal n is the **outward material-surface normal**. Compression is
`Fn = -F dot n > 0`; F includes the supplied tangential force. For two positive
combined relative principal curvatures, use the cheap circular estimate:

    k_mean = (k1+k2)/2
    E_star = E/(1-nu^2)             # rigid counterface, unless explicitly supplied
    a = (3 Fn/(4 E_star k_mean))^(1/3)
    p(r) = 3 Fn/(2 pi a^2) sqrt(1-r^2/a^2), 0 <= r <= a

Use the current force even when geometry is cached. Release returns exact zero;
tensile/invalid inputs, nonpositive relative curvature and friction-cone
violations are explicit failures. Manual radii provide a separate control.
Signed curvature and actual source-side provenance matter on nonconvex objects;
no contact symmetry or fixed contact location is assumed.

**This prototype's pressure model differs from native curved-surface DPR.**
It defines a Hertz disk in the supplied normal plane at the original contact
point and projects samples onto the offset Q4 surface **along F**. Genesis's
current native DPR instead weights its reference surface inside a support ball.
The two coincide in appropriate planar controls, but curved-surface results
cannot be presented as a numerical-only replacement of the native load law.
No independent shell-contact solution is inferred from the Hertz formula.

For side `s = +/-1`, the actual offset surface is
`X_s = sum_i N_i (X_i + s h d_i/2)`. Integrate its shape weights w_i over
the projected disk, then transfer both generalized force and couple:

    f_i = w_i F
    m_i = w_i (s h/2 d_i) x F

This follows virtual work for translation and director rotation. Dropping m_i
would lose the eccentric friction load. Projection along F preserves the
original point wrench when the finite distribution has its required centroid.
Check integrated force/moment, normalized footprint mass and inversion quality;
never increase a to fit an elastic element or silently truncate the support.

## 4. Exact and approximate transfer paths

The baseline first checks a current collision-face witness. It bypasses spatial
search only if the full disk is contained on that valid face, with matching
side/normal and a bounded actual force-line intersection. A stale or missing
witness takes the spatial/local-sheet path. Neighbor candidates remain tied to
the actual contacted material sheet rather than a nearby disconnected wall.

- **Exact affine complete disk:** Q4 inverse coordinates are affine. Its shape
  moments need only zeroth/first/second Hertz moments, with covariance `a^2/5`.
- **Baseline nonaffine/clipped support:** integrate face-clipped polar regions,
  split angular intervals at vertices and disk/edge intersections, and invert
  each projected bilinear Q4. Adaptive higher order retains failed attempts.
- **Opt-in `fast`:** a complete nonaffine disk compares degree-7/11 cubatures
  with 16/36 samples. If their nodal-weight difference is below the indicator
  threshold, use the 36-point result; otherwise retain the baseline rule.
  Polynomial degree is not an exactness proof for nonpolynomial inverse shapes.
- **Opt-in `taylor`:** a second-order local inverse-shape expansion. Its errors
  are measured separately; it is not silently enabled by `auto`.

`batched_witness_transfer` vectorizes the same `fast` rule across environments.
It retains current-witness, inversion, mass, embedded-error and wrench checks.
Clipped, unsupported, unloaded or unresolved cases use the original scalar
path, including its failures. `results/batched_transfer_checks.json` exercises
edge/vertex, inner-side, unload and stale-witness fallbacks explicitly.

## 5. Recovery and observations

One reused factor solve gives q. Recover membrane N, bending M and shear Q.
For homogeneous linear material the reported in-plane surface tensors satisfy:

    sigma_parallel(z) = N/h + 12 z M/h^3
    sigma_top/bottom = N/h +/- 6 M/h^2

`stress_matrix('local')` maps q to both surfaces, four in-plane points per quad,
and `[sigma_xx,sigma_yy,tau_xy]`. Fixed basis rotations give six symmetric
body-frame tensor components. These are **shell in-plane tensors**, not the
complete three-dimensional near-contact stress. VM/principal maxima here
range over those declared shell samples and surfaces.

An exact alternative precomputes `H = S K_gauge^-1 P`, including inertia relief.
Then `sigma = H f` is the same discrete model, not analytical stress enrichment.
On-demand response columns allow changing contact locations, but cold misses
cost solves and dense active-column unions may erase a warm-cache advantage.
Storage scales as `O(n_stress * n_dof)`, approximately quadratically with mesh
size; use a budget and fall back to the complete prefactorized solve.

For point forces on one physical side, fold eccentric couples into a compact
map `T_side: 3N -> 6N`, and precompute `H_side = S K^-1 P T_side`.
This halves response columns for that side without deleting friction moments.
Mixed inner/outer contacts sum both maps; additional independent couples need
the general operator. At 336/588/1,308 DOFs, one-side response storage is
1.74/5.42/27.12 MB, with measured build costs about 13/27/254 ms.

## 6. Measured CPU optimization, not GPU throughput

See `results/performance_summary.md` for complete scope, ranges and sources.
Single-thread EPYC 9V74, 588 DOFs, seven interleaved confirmation repeats:

| B128 operation | Median ms |
|---|---:|
| Per-column reused LU + full local field | 23.122 |
| Multi-RHS reused LU + full local field | 10.392 |
| Compact response + full local field | 4.128 |
| Scalar checked-witness fast transfer | 153.316 |
| Batched checked-witness fast transfer | 9.030 |
| Batched transfer + reused LU + full local field | 19.863 |
| Batched transfer + compact response + full local field | 12.889 |
| Batched transfer + compact response + VM peak | 13.685 |

Compact-response field error is `8.55e-12` versus LU; batched/scalar transfer
load difference is `2.38e-16`. The timed disks are inside current witnessed
faces; fallback correctness is checked separately, not timed as a clipped bank.
Max-only has no confirmed CPU speed gain because this implementation still
forms stresses before reduction. Dense-response and tile-loop negative cases
remain in the broader `results/performance.json`; stage times are not additive
whole-scene costs and none of these numbers includes Genesis rigid dynamics.

## 7. Proposed Quadrants migration

1. Add an optional shell discretization backend inside existing stress recovery;
   retain solid FEM, contact provenance, max/full options and rigid lifecycle.
2. Build reference shell operators/factors once in Quadrants. Extend DOF/block
   handling deliberately; the current solid backend's 3-by-3 assumptions do not
   automatically support shell translation/rotation blocks.
3. Keep forces, curvature/witness lookup, work queues, residuals and observations
   device-resident. Batch current contacts and pack only active work; validate
   release/slot changes and partial resets without per-environment host loops.
4. Fuse common complete-disk moment/load work. Compact exceptional clipped
   contacts into the validated fallback, counting packing and overflow costs.
5. Budget shared exact response operators and sparse selected-column products;
   compare them with batched factor application rather than assuming dense wins.
6. Stream stress tiles into exact maxima when full output is not requested;
   keep complete field work/output for full mode. Measure both end to end.
7. Start from FP64. Test gauge/scaling, lower-precision storage and mixed solves
   with physical residuals and full fields. TF32/tensor-core speed is unmeasured;
   rounded response output alone does not validate a recovered displacement.

These are migration recommendations, not implemented GPU optimizations.
Keep the numerical load/stiffness model fixed while profiling. Physical shell
accuracy, finite support resolution and contact-model accuracy remain separate.
Tiny Hertz patches can be far smaller than element size or shell thickness;
neither analytical load integration nor factor reuse creates missing local
stress resolution, and a shell cannot claim a full 3D contact-stress maximum.

## 8. Reproduction after placement in Genesis

From the repository root, use fresh output paths and one BLAS thread:

```bash
mkdir -p /tmp/dpr_shell_cpu_reproduce
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=research/rigid_stress \
  python -m shell_cpu.study --resolutions 3,5,8 --torus --out /tmp/dpr_shell_cpu_reproduce/egg
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=research/rigid_stress \
  python -m shell_cpu.performance --resolution 3 4 --batches 1 8 32 128 --output /tmp/dpr_shell_cpu_reproduce/performance.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=research/rigid_stress \
  python -m shell_cpu.results.performance_confirm --output /tmp/dpr_shell_cpu_reproduce/confirmation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=research/rigid_stress \
  python -m shell_cpu.reference --suite plate --levels 4,8,16 --output /tmp/dpr_shell_cpu_reproduce/independent_plate.json
```

`review_checks.py` and `precision_study.py` currently use CWD-relative source and
output paths and have no argument parser. Run them from `research/rigid_stress`
(`python -m shell_cpu.review_checks`, `python -m shell_cpu.precision_study`),
after setting `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. They execute suites on
invocation, including `--help`, and write their default result directories.
Preserve published JSON before rerunning; the existing results are frozen
evidence, not files that documentation-only work should regenerate.

Methodological references: [MOOSE shell elements](https://mooseframework.inl.gov/moose/modules/solid_mechanics/ShellElements.html)
and [Code_Aster plate/shell models](https://code-aster.org/doc/v17/manuals/man_r/r3/r3.07.03/index.html).
The independent pyfe3d comparisons and their discretization differences are
reported in `reference_results/`; no Code_Aster run or full shell-contact
qualification is claimed by the availability probe or prescribed-load checks.

## 9. Supplementary native-DPR pressure-law control

`native_dpr_control.py` directly imports the existing unchanged
`wrench.py::PadPressureFit`. It keeps the native 3D surface-ball Hertz density,
actual offset-Q4 surface area, nonnegative affine wrench fit, and consistent
shell force/couple scatter. It is a selected-shell-side extension of the
native load law, not a run of the production Quadrants/P2 boundary code.

Twelve bounded plane/n4/n8-egg normal/friction controls pass the original
wrench budgets. Complete sampled shell-stress differences versus the projected
variant are <=4.63e-8 for the tiny automatic radius and <=2.664e-4 (0.02664%)
for a manual 2 mm radius. These are **model differences**, not exact numerical
equivalence or physical error. The metric is the unweighted Frobenius relative
norm over the same mesh's two surfaces and four Gauss samples. Native-law
q24/q48 differences are separately reported, up to about 1.03e-6 in stress.
The supplied curvatures 50/70 per metre are identical in both paths to isolate
the transfer law; this test does not independently validate curvature lookup.

This supports an integration that keeps mature native DPR pressure loading
and replaces only the elastic recovery backend. Primary performance timings
remain for the projected variant; they are not relabeled native-DPR timings.

From `research/rigid_stress`, reproduce the bounded control with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m shell_cpu.native_dpr_control --output /tmp/shell-native-dpr.json
```
