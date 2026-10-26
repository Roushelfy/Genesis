# Feature selection and boundary reconstruction

The source reconstructs the reported algorithm from the surviving report.
The old exact case coordinates, force conventions, tangent choices and source
files were unavailable. The new deterministic corpus was frozen before its
errors were evaluated. No position, weight or force was adjusted to reproduce
the historical 4.02/4.14/4.08 percent table.

## Model and corpus

The fixed mother has 864 Q4 elements and 5,196 coordinates. The authored egg
at the coarse nodes is its midsurface. Bilinear refinement preserves that
physical geometry. The common coarse space has 336 coordinates, six gauge
coordinates are removed, and each M12 feature has 3,516 independent solve
coordinates (3,522 including the gauge). It contains the common global coarse
space and 531 locally independent fine nodes over 37 active macro elements.

`fixtures.py` defines 42 loaded cases and one complete release: 24 independent
Fibonacci sphere locations, six pole/seam/corner locations, two double-contact
cases, a three-contact case and nine smooth single-contact frames. Radius is
one, two or three nominal mm in the fixed material-cap law. Friction fractions
are zero, 0.3 or 0.6. `results/selection/fixtures.json` contains all positions,
forces and metadata. Its SHA-256 is recorded in the result JSON.
The nine `smooth_path` frames specify a smooth geometric path and amplitudes.
Their default tangent convention can change the global friction direction.
They are independent spatial samples, not a friction-force continuity test.
Fixed-force boundary controls and the separate history trajectory provide
the actual temporal continuity checks.

The M12 centers are standard icosahedral vertices. Support is 1.2 times their
exact spherical covering radius. The refinement core adds the largest cap
angle, 3/22 radians, and the model adds one vertex-adjacency macro halo.
`build_bank` accepts other center counts as an exploratory Fibonacci library.
Those additional libraries are not the historical M24 Lloyd-relaxed bank.

## Selection rules

- `continuous`: normalized compact Wendland weights at angular distance/support.
- `nearest`: one nearest feature per source.
- `narrow`: continuous weights multiplied by the same compact kernel evaluated
  at `(distance - nearest_distance)/(0.12*support)`. All tied contributors stay.
- `top2`: diagnostic hard truncation of the continuous weights.
- `wrong_all_loads`: negative control applying every contact to each selected
  feature and averaging, instead of assigning a feature only its own source.

The actual force remains at the current physical material point. Equal
`(feature, environment)` RHSs merge before solving. Inertia relief is applied
using the mother model. Stress tensors are combined at the common points
before von Mises and maximum evaluation. The numerical study caches source
solutions for independent superposition checks, not for performance claims.

## Final numerical result

For the 42 loaded cases, order 8, on the declared final midsurface:

| Strategy | Mean full-field error | Worst full-field error | Mean solve groups |
| --- | ---: | ---: | ---: |
| Continuous | 4.4524% | 7.5895% | 2.0714 |
| Nearest one | 4.5979% | 9.7410% | 1.0714 |
| Narrow transition | 4.4734% | 7.5896% | 1.3333 |
| Hard top two | 4.5206% | 8.5424% | 1.8571 |
| Wrong all-loads average | 6.3714% | 48.2770% | 1.0714 |

Full-field error is the area-weighted Frobenius tensor difference, including
top and bottom and counting shear twice. Sampled peak errors are separate
in the JSON. All correctly assigned finite-source supports are covered. The
wrong all-loads negative control has five uncovered source/feature pairs.
The maximum reduced numerical residual is 3.20e-14. Merged RHS and independent
source superposition differ by at most 5.16e-14 in displacement norm.

The qualitative tradeoff reproduces. The mean errors and continuous group
count do not exactly reproduce the historical table on this reconstructed
corpus. The nearest/narrow group counts agree. These are distinct frozen
experiments, not an exact rerun of the missing source archive.

## Switching continuity

`boundary_study.py` crosses a true Voronoi pair boundary, a second/third rank
exchange and a triple junction at 1e-4, 1e-5 and 1e-6 radians. Each crossing
keeps one global force and a safe material-cap tangent frame fixed, isolating
selection from tangent-frame branch changes. The frame override was introduced
for this diagnostic and leaves the 42-case corpus unchanged.

At epsilon 1e-6, nearest-one full-field jumps are 2.9083% at the pair and
6.5754% at the triple junction. Their sampled-maximum changes are 0.000641%
and 0.018955%, respectively. These different quantities must remain separate.
Top-two has a 0.2892% field jump at the second/third exchange and 3.2879% at
the triple junction. Continuous and narrow changes scale with the crossing
step. Narrow uses all three contributors at the triple junction.

## Reproduce

From the repository root, with a fresh explicit output path:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m research.rigid_stress.atlas_study.cpu_repro.selection_study --output /path/to/new/selection
python -m research.rigid_stress.atlas_study.cpu_repro.boundary_study --output /path/to/new/boundary
python -m research.rigid_stress.atlas_study.cpu_repro.plot_stress --output /path/to/new/stress-maps
```

The plots show every mother stress point on each surface, with common scales,
contact markers and complete-tensor error maps. Derived arrays are ignored by
Git and reconstructed by the command.

`results/initial_outer_offset/` retains an initial interpretation that shifted
the authored coarse surface inward by half the thickness. The final convention
uses the report's stated authored midsurface. This choice followed the model
description rather than the closeness of either numerical error table.
The first boundary run failed only while serializing NumPy integer metadata.
Its [original diagnostic log](results/diagnostics/boundary_initial_serialization_failure.txt)
is retained separately from the final numerical results.

These scripts provide numerical evidence. Formal latency comparisons are
owned by the separate pipeline benchmark and include current load assembly.
