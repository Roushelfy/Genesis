# CPU thin-shell automatic-radius + DPR study (2026-10-24)

This is a **CPU research prototype**, not a native Genesis solver feature.
It evaluates replacing the volume elastic-recovery discretization with a
fixed linear shell while retaining rigid dynamics and finite contact loads.
Production integration must remain in the rigid stress feature and use
Quadrants. No GPU or complete Franka rollout was timed in this study.

**Result:** the shell mechanics and several exact execution optimizations are
validated on bounded controls. The small Hertz footprint on the egg is **not
stress-converged**, and shell in-plane stress is not a full 3D contact stress.
Do not replace the current volume backend by default on the strength of these
results.

See [ALGORITHM.md](ALGORITHM.md) for equations, runtime design, measured and
proposed optimizations, and Quadrants migration guidance.

## Implemented algorithm

- Curved fixed-reference MITC4-type shell: three translations and three small
  rotations per node, membrane/bending/shear operators, consistent weak
  drilling stabilization, six exact rigid modes and physical rotary inertia.
- Geometry/material/thickness-dependent stiffness assembled and factorized
  once; arbitrary current normal/friction loads are scattered each update.
- Cheap circular Hertz radius from force, reduced modulus and combined local
  principal curvatures. No compliant gripper pad is assumed.
- Conservative finite pressure transfer includes the actual inner/outer
  surface eccentricity and tangential-force moment. Six-mode inertia relief
  is explicit for unbalanced supplied contact loads.
- Complete contained disks use exact moments or checked embedded cubature;
  clipped disks retain converged quadrature and explicit failure. Radius is
  never inflated to accommodate the mesh.
- Changing collision witnesses are checked, stale witnesses fall back, and
  local material-sheet identity is preserved on nonconvex controls.
- Multiple RHS, exact precomputed stress responses, on-demand response columns,
  compact single-side `3N` surface responses, batched transfer, full/maximum
  output comparisons, and gauge/scaling/precision experiments are implemented.

The main `contacts.py` load variant is **projected-Hertz DPR**, not the exact
native curved-surface support-ball pressure fit. This is a disclosed modeling
difference, not a numerically equivalent port. Plane controls agree; native
integration must explicitly preserve the selected load law. The source does
not contain an analytic particular stress field or a boundary-traction stress
correction.

The supplementary [native load-law control](native_dpr_control.py) imports
the unchanged existing `wrench.py::PadPressureFit` and extends the native
surface-ball Hertz law to the chosen physical Q4 shell side. All 12 bounded
controls pass. Relative full sampled stress differences from the projected
variant are <=4.63e-8 for automatic tiny patches and <=2.664e-4 (0.02664%)
for 2 mm patches. These are unweighted Frobenius norms on a common mesh.
This supports keeping mature DPR's pressure law during backend integration;
it is not production runtime parity or independent physical validation.
See `native_dpr_results/control.json` for geometry, quadrature differences,
source hashes, and the selected-side versus all-volume-boundary distinction.

## What was actually checked

| Evidence | Main result | Limit |
|---|---|---|
| `review_results/review.json` | All 13 selected controls pass; six null modes; membrane/bending energy; thin plate, sphere, curved cylinder; contact invariance and U-wall identity | Linear shell controls, not contact ground truth |
| `reference_results/` | Executed independent pyfe3d 0.10.0 Quad4; sphere n16 field difference 1.60%; plate analytical field error 2.38% at n32 | Prescribed load; not a Code_Aster run |
| `contact_reference_results/` | Independent unilateral halfspace BEM recovers Hertz contact radius within 0.365% and continuous pressure L2 within 1.27% at finest tested grid | Halfspace/parabolic contact only, not hollow-egg contact |
| `results/egg_final/summary.json` | 24 normal/friction/asymmetric/manual-control load cases pass numerical transfer/solve checks | Peaks and energy remain mesh dependent |
| `solid_results/` | Compared with existing CPU P2 volume oracle and 3D Lame control; broad-load shell/volume differences decrease | Different geometry interpolation; neither local coarse field is ground truth |
| `results/performance_confirmation.json` | Seven interleaved repeats after other heavy jobs finished; exact execution errors below 1e-8 | Single-thread CPU recovery module only |
| `native_dpr_results/control.json` | Existing native pressure-fit law extended to Q4 shell, all 12 bounded cases pass | Separate load-law check; no production or contact-ground-truth claim |
| `review_results/gauge_checks.json` | Translation-only gauge preserves stress; diagonal scaling cuts LU fill and permits accurate mixed precision | Separate supplementary experiment, not the gauge used in primary timing |

Code_Aster was unavailable in this environment. Installed OpenSees binaries
failed during MPI initialization, so neither supplied an executed reference.
The pyfe3d element uses an independent mixed-integration shell assembly, rather
than calling this prototype's MITC operators.

## Egg results: unresolved small contact

Authored outer half-axes are 22/22/30 mm, taper -0.18; the midsurface is a
half-thickness inward normal offset. Thickness is 0.5 mm, E=10 GPa, nu=0.3.
Two opposing loads each have Fn=0.05 N; automatic radius is 45.472 micrometres,
or a/h=0.09094. The friction case has Ft/Fn=0.25.

| Shell DOF | Normal VM peak, kPa | Friction VM peak, kPa |
|---:|---:|---:|
| 336 | 6.691 | 6.360 |
| 912 | 12.793 | 12.294 |
| 2,316 | 29.537 | 30.916 |
| 5,196 | 62.980 | 63.262 |
| 9,228 | 77.767 | 76.408 |
| 20,748 | 128.841 | 129.777 |

These are maxima over four Gauss points per element and both surfaces, not
continuous maxima. Normal peak changes by +65.7% from the previous mesh
(39.6% when normalized by the finer result); strain energy changes by 27.8%
using the finer result as denominator. Increasing integration accuracy alone
does not add missing FE stress resolution. The 2 mm and 5 mm footprints in
controls are **manual** broad loads, not adjustments to the physical radius.

Output tensors contain **in-plane membrane plus bending** stresses. They omit
the 3D Hertz near-contact normal stress. The independent halfspace control
cannot certify the shell radius model once global shell compliance matters.
The independent-reference egg fixture uses different, explicitly recorded
geometry and force; do not merge its numerical peaks with this table.

## Confirmed CPU performance

AMD EPYC 9V74, one BLAS/OMP thread, 588 DOF / 96 quads. Milliseconds per batch:

| Operation | B1 | B128 |
|---|---:|---:|
| Reused LU + full local field | 0.189 | 10.392 |
| Compact exact surface response + full local field | 0.099 | 4.128 |
| Scalar checked-witness transfer | 1.300 | 153.316 |
| Batched checked-witness transfer | 0.727 | 9.030 |
| Batched transfer + LU + full local field | 0.973 | 19.863 |
| Batched transfer + compact response + full local field | 0.845 | 12.889 |
| Batched transfer + compact response + VM maximum | 0.885 | 13.685 |

The compact response field differs from LU by 8.55e-12; batched transfer
differs from scalar by 2.38e-16. Single-side response storage at 336/588/1308
DOF is 1.74/5.42/27.12 MB, shared by identical environments. Memory grows
approximately quadratically. Max-only did **not** give a CPU latency gain.

A separate matched-surface sphere experiment compares cached solve plus common
in-plane stress outputs: at n8, shell 2,316 DOF versus volume P2 13,842 DOF,
B32 costs 40.71 versus 99.10 ms. This is not an accuracy-matched speedup and
excludes transfer and full volumetric output; details are in
`solid_results/matched_cpu_cost.json`.

An important supplementary improvement is `gauge="translation"` plus diagonal
equilibration. At 2,316 DOF / eight RHS it reduces original-unit FP64 solve and
stress from 11.97 to 3.14 ms, principally through reduced pivot/fill. One FP32
factor correction is accurate but slower (5.55 ms). This gain is **not**
multiplied into the old-gauge primary performance table. See
[precision notes](review_results/PRECISION_NOTES.md).

## Reproduction and evidence

Install NumPy, SciPy, Matplotlib, threadpoolctl and optional pyfe3d==0.10.0.
For scripts using relative source/evidence paths, work from this package's
parent directory:

```bash
cd research/rigid_stress
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m shell_cpu.study --resolutions 3,5,8 --torus --out /tmp/shell-egg
python -m shell_cpu.reference --suite plate --levels 4,8,16,32 --output /tmp/shell-plate.json
python -m shell_cpu.reference --suite sphere --levels 2,4,8,16 --output /tmp/shell-sphere.json
python shell_cpu/results/performance_confirm.py --output /tmp/shell-performance.json
python -m shell_cpu.matched_cpu_cost --help
```

`review_checks.py`, `precision_study.py` and the gauge check execute tests
when invoked and overwrite their default evidence paths. They have no help
parser. Preserve the recorded JSON or use a separate checkout before running:

```bash
python -m shell_cpu.review_checks
python -m shell_cpu.precision_study
PYTHONPATH=. python shell_cpu/review_results/check_translation_gauge.py
```

`make_report.py` builds the Chinese self-contained HTML report once the field
NPZs have been regenerated at `shell_cpu/results/egg_final` with
`study --resolutions 3,5,8,12,16,24 --torus --out shell_cpu/results/egg_final`.
The n24 factorization took about 203 seconds on this CPU; use coarse cases for
routine implementation checks. Large generated field arrays are not committed.

Source hashes bind evidence to the executed versions. The old core used in
primary timings is preserved byte-for-byte at
`review_results/source_at_primary_runs/core.py`; the current core adds only
an optional gauge and keeps legacy as default. Initial failed quadrature,
stationary mixed-precision refinement and contact-reference integration runs
remain explicitly historical, rather than being deleted or relabeled passes.
