# CPU P2-solid cross-checks

These are prescribed-load numerical experiments, **not Code_Aster runs, not an independent contact solve, and not converged local-contact ground truth**.

`solid_comparison.py` imports the existing Genesis research `P2Shell` volume oracle. The MITC4 midsurface vertices are extruded along their spherical directors, split into triangles and then P2 tetrahedra. Thus the two methods share authored surface vertices and thickness but have different interpolated geometry: curved Q4/director shell versus straight-sided tetrahedral geometry.

## Uniform internal pressure

R=20 mm, h=0.5 mm, E=10 GPa, nu=0.3, inner pressure=100 Pa; exact continuum control is the 3D Lame solution. `spheres.json` covers n=3,6,8 subdivisions on each cube-sphere face and one/two thickness layers. At n=8: shell 2,316 DOF versus solid 13,842/23,070 DOF; shell midpoint radial displacement difference from Lame 2.263%, mean hoop stress difference 1.236%; solid base radial displacement L2 7.111/7.125%, volume hoop-stress L2 2.455/2.652%. Increasing thickness layers alone does not fix the coarsely approximated surface geometry.

## Opposing finite disks

Two balanced oblique pressure disks act at sphere antipodes; Fn=0.05 N each. The physical Hertz circular radius is 41.036 micrometres (a/h=0.082). `disks.json` retains n=3,6,8 at that automatic radius and at an explicitly manual 2 mm radius. Those coarse stress fields differ materially; no physical qualification is claimed.

`wide_disks_diagnostics.json` checks an explicitly manual 5 mm disk (a/h=10) at n=8,12,16. Midpoint displacement differences decrease 9.40%, 4.63%, 2.53%; common sampled in-plane stress component-vector L2 differences decrease 34.23%, 21.95%, 15.70%. At n=16, sampled in-plane von Mises peaks are 30,335 Pa shell and 31,303 Pa solid (3.09% difference). The field comparison is still not converged and cannot certify a full 3D peak.

All disk comparisons use area-weighted equal-component **Voigt-vector L2** over `(xx, yy, xy)`; this is not the Frobenius tensor norm and not directly comparable with other reports using that metric. The far-field diagnostic excludes points closer than max(2 sqrt(Rh), 2a) to either load center and never replaces the complete field metric.

P2 disk forces use independent radial/angular quadrature and directional ray projection to the actual triangulated outer surface. Both transfer and solve preserve source wrenches to numerical precision; the largest normalized disk moment residual was below 1e-13. This isolates discretization/model differences without introducing a fitted or analytic particular stress field.

Reproduce from the experiment folder with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shell_cpu.solid_comparison --n 3 6 8 --layers 1 2`. The disk cases are exposed as `run_disk_case(n, layers=1, Fn=.05, manual_radius=None)`; pass `.005` only for the declared broad-footprint control, not as an automatic-radius correction.

## Bounded matched authored-surface CPU cost

`matched_cpu_cost.json` and `matched_cpu_cost.py` add a single-thread AMD EPYC 9V74 measurement, five interleaved repeats after warmup. The explicit 5 mm footprint is a broad-load control. Online scope is cached sparse-LU solve (including free-body load balance) plus a precomputed sparse map to the **same number** of outer/inner in-plane stress samples. Assembly, factorization, contact transfer, volume-wide 3D stress output and any GPU/Franka work are excluded. RHS amplitudes vary at fixed patch locations; no fixed-contact assumption is used by the solve.

| Face subdivisions n | Batch | Shell ms/batch | Solid P2 ms/batch | Solid/shell |
| --- | ---: | ---: | ---: | ---: |
| 4 (588 vs 3,474 DOF) | 1 | 0.267 | 1.082 | 4.05 |
| 4 | 32 | 3.746 | 18.300 | 4.89 |
| 8 (2,316 vs 13,842 DOF) | 1 | 3.371 | 8.349 | 2.48 |
| 8 | 32 | 40.712 | 99.102 | 2.43 |

Setup/factor took 0.696/0.098 s (shell/solid, n4) and 3.201/0.663 s (n8). The Python shell assembly is slower during this one-time setup. Solid map versus separate direct stress contraction differs less than 2.1e-14 relative; shell map differs less than 1e-15. **These are different discretizations with different errors and geometric interpolation; these factors are not accuracy-matched speedups.** Raw repeated times are retained.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m shell_cpu.matched_cpu_cost --n 4 8`.
