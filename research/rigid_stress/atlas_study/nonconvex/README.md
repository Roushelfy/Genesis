# Nonconvex 2D P2 feature-mesh atlas control

This is an **executed 2D plane-stress** numerical experiment, not a 3D egg,
physical contact solve, native Genesis feature, or GPU benchmark.

`study.py` constructs an exact axis-aligned dogbone: x=0..4, outer half-width1,
neck x=1.5..2.5 with half-width0.2. The left edge is physically clamped and finite
Hertz-line pressure intervals act on the right edge, including tangential force.
Unit-load responses are used to test the linear operator; loads can be scaled
for a small-strain physical regime. Pressure size is explicitly supplied, not a
new automatic shell/3D contact-radius estimator.

## Numerical construction

- Conforming tensor-knot triangular meshes with quadratic (P2) displacement.
  Filtered cells respect all exact domain boundaries, including reentrant corners.
- Three representative contact y positions: -0.65,0,0.65. The mesh bank refines
  y windows of half-width0.36 plus right-edge x knots. Tensor-product grading
  creates extra fine **strips**, so it is not a minimal isotropic local remesh.
- Compare contact-only feature meshes with feature meshes also containing fixed
  neck refinement. This tests whether contact-driven refinement misses distant
  stress concentrations, rather than assuming stress peaks stay near contacts.
- Every actual load position/radius is retained. Exact zeroth/first/second
  moments of the finite square-root pressure integrate the P2 edge weights;
  resultants and moments are preserved without point substitution or patch growth.
- Use common physical stress probes. Full tensors, including shear, are added
  before VM peaks are taken. The field norm is an **unweighted sampled tensor
  Frobenius norm**, not a rigorous continuum-domain L2 error estimate.
- Sharp reentrant corners have singular continuum maxima. The fixed probe set
  excludes radius0.105 around the four such corners and starts at x=0.113 away
  from the clamp. These are reported sampled-region maxima, not continuum
  global maxima. The displayed white corner gaps are excluded probes, not
  rounded physical geometry.
- The merged knot lists have a minimum-spacing quality control, preserving
  geometry-defining coordinates exactly. The known-SPD clamped stiffness uses
  diagonal scaling, symmetric ordering and no threshold row pivoting; one
  FP64 residual correction is included in the timed solve.

## Completed cases and results

Seventeen cases cover 11 moving positions, three prescribed radii, distant and
nearby pairs, and an asymmetric three-source load. The finer reference has
84,294 DOFs. A union-refined mesh and an intermediate reference are retained.
The last observed reference change is **1.15–2.33%** in the same sampled field
norm; this is reference sensitivity, not a certified continuum-error bound.

Against that finer discrete reference:

| Method | Sampled full-field discrepancy |
|---|---:|
| Contact-only atlas | 15.1–24.4% |
| Contact + fixed-neck atlas | 3.32–4.76% |

The maximum sampled neck-region discrepancy is 14.43% for contact-only and
1.19% after fixed-neck refinement. For the distant pair, the largest sampled
VM occurs near x=1.632,y=0.190, far from the actual sources on x=4 and outside
our explicit corner exclusion. This is evidence for preserving geometry-driven
refinement as well as contact-driven refinement; it is not a claim that the
sharp physical corner maximum is finite or resolved.

Thirteen of seventeen hard-selection cases have their entire pressure interval
inside the selected fine y window. Four deliberately cross its edge; their
fine-window margin and coverage are recorded in `hard_fine_coverage`. Physical
loads are still integrated on the complete mesh, with no hidden support clipping.

The algebra controls are strong:

- Same-mesh per-source response superposition: relative discrepancy9.48e-12.
- Full coarse + fine-minus-coarse reconstruction: 2.41e-12.
- Wrongly applying the complete load to each chosen feature mesh and summing
  produces approximately99.8% field discrepancy, as expected from double loading.
- Force and moment errors are below1e-9 (typically floating-point roundoff).

For an infinitesimal contact movement across the two atlas-selection boundaries,
hard selection jumps by **3.44% /3.09%**. Partition-of-unity blending reduces
changes to1.10e-7 /3.00e-8, matching actual reference physical-motion changes
1.10e-7 /3.01e-8. Blending costs additional solves and does not erase the
underlying spatial approximation error.

## Numerical precision: keep the negative control visible

`selected_checks_pass` is deliberately **false** because the original strict
raw relative residual budget1e-9 is missed: the worst recorded
`||K u-f||/||f||` is1.944e-9. That check has not been loosened or relabeled.
All superposition, coarse/delta and load checks pass; the normwise backward
error is at most6.04e-19.

`precision_diagnostic.py` independently revisits the worst raw-residual case
using extended-precision residual accumulation on the **same FP64 assembled K**.
The resulting correction changes the complete sampled field by9.21e-11 and
VM peak by8.09e-11. After that correction, extended residual is6.11e-10 while
the FP64 sparse-matmul readout remains around1.4e-9. This quantifies the actual
stress impact and supports a cancellation-floor interpretation; it does not
silently convert the strict primary residual failure to a pass.

## CPU cost and storage

Single-thread EPYC CPU,5 repeats after warmup; module cost includes actual
force integration, cached solve and all common stress samples. Other CPU
studies were running; ranges are retained and these are not production/GPU
throughput figures. The timed load is the distant pair.

| Method | Median online ms |
|---|---:|
| Coarse | 0.382 |
| Feature atlas, hard | 2.817 |
| Feature atlas, blend | 4.108 |
| Union-refined | 2.344 |
| Finer reference | 34.184 |

The atlas is much cheaper than the finer reference but **not faster than the
union-refined comparator** in this small bank. Their errors are not identical,
so no same-accuracy speedup is established here. Offline setup is separate.

Stored LU arrays for all three contact-and-neck feature meshes total13.53 MB,
versus7.70 MB for the union mesh and122.86 MB for the finer reference. These
are LU arrays only, not process peak memory. Feature-bank duplication is real.
The SPD-aware factorization reduced a diagnostic97k-DOF LU from74M to12M
nonzeros and factorization from15.1 to0.51 seconds; that is a separate solver
configuration experiment, not an atlas-only gain.

## Reproduction and evidence

From a directory containing the `atlas_study` package (namespace packages work):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m atlas_study.nonconvex.study
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m atlas_study.nonconvex.precision_diagnostic
python -m atlas_study.nonconvex.plot_results
```

The first two commands write their result files next to source. Preserve the
recorded evidence before rerunning. Dependencies: NumPy, SciPy, scikit-fem12,
threadpoolctl, Matplotlib; exact executed versions are in`environment.json`.

`results.json` binds the final source hash and records source stability. Large
field NPZ is reproducible and can be excluded from Git; `plot_results.py`
reconstructs the clean field figure from it. `meshes.png` displays the actual
coarse/feature triangulations.

Historical JSON files preserve early diagnostics: an initially incorrect jump
normalization (fixed in final), the strict residual failure, and changes caused
by reference refinement/pivot/knot quality. `resource_failure.json` separately
records an OS memory kill during an unbounded common-probe lookup; batching
point location fixed this without changing the physical domain or loads.
