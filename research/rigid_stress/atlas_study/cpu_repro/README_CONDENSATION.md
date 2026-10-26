# Exact coarse/local condensation control

`condensation.py` implements both elimination directions on the same selected
coarse-plus-local Galerkin space. It supports general right-hand sides in both
blocks, including distributed loads produced by the common inertia relief.
All cross-patch coupling is retained.

For the reduced stiffness and current projected load,

```text
[A C ] [u] = [f_c]
[C' D] [v]   [f_l]
```

coarse-first elimination uses `Z=A^-1 C` and `H=D-C'Z`:

```text
v = H^-1 (f_l - C' A^-1 f_c)
u = A^-1 f_c - Z v
```

local-first elimination uses `W=D^-1 C'` and `H_c=A-CW`:

```text
u = H_c^-1 (f_c - C D^-1 f_l)
v = D^-1 f_l - W u
```

The implementations apply symmetric diagonal equilibration before forming the
blocks and again to the dense Schur complement. Results are transformed back
to the original reduced coordinates. Numerical checks use the original,
unscaled projected operator and all common top/bottom stress samples.

Coarse-first forms a dense local Schur complement and its dense factor.
Local-first instead retains a sparse local factor, a dense harmonic map and
a coarse Schur factor. Their storage and multiple-right-hand-side performance
can differ substantially even though both reproduce the same Galerkin solve.
The code does not assume that condensed local blocks decouple.

## Reproduce

Run from the Genesis checkout root with NumPy/SciPy and threadpoolctl installed:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.rigid_stress.atlas_study.cpu_repro.condensation_study --output /absolute/new-run/condensation
```

This creates `condensation_results.json`. It compares a single patch, two
distant patches, two nearby patches and three asymmetric patches. The selected
space is the union of their actual active macros and compatible local bubbles.
Each case contains its finite-cap normal/friction load plus three general
distributed loads, balanced by the same mother mass and rigid modes.

The negative control discards the coarse-first Schur complement's coupling
between assigned patch groups. Its field difference and original residual are
recorded separately as an approximation, never as an exact condensation result.

After other CPU workers are idle, run the separate timing phase:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.rigid_stress.atlas_study.cpu_repro.condensation_study --output /absolute/new-run/condensation --timing --repeats 5
```

This creates `condensation_timing.json`. Each B1/B16/B64 measurement starts from
an already projected reduced load and includes solve plus the complete common
local stress operator. Source construction, relief, projection, body-frame
nine-component conversion and maximum reduction are excluded. It is a stage
comparison, not full stress-recovery throughput or a GPU benchmark.

The four RHS columns are tiled for larger batches. They are one physical cap
load and three distributed load controls, rather than independent grasp scenes.
Methods run in forward/reverse order on alternating repeats, with wall and
process CPU time retained. Checksums are computed outside the measured interval.
Source hashes are checked before and after each run. Construction timings in
the numerical phase are diagnostic and can overlap other reconstruction work.

`payload_bytes` counts retained factor, permutation, scaling, coupling and
harmonic-map numerical arrays owned by that solve method. It excludes the
shared original K, T, S and mother model, Python/SciPy object overhead and other
process allocations. It is not resident-set memory. Dense layouts count their
full allocated triangular-factor arrays, including unused triangle storage.

The comparison checks exact algebra within an identical chosen space. It does
not establish spatial convergence of the egg or accuracy of the material-cap
contact model. Earlier CPU timings are historical targets and are not inserted
into these new run records.

## Geometry provenance

The initial numerical run used an outer-surface-minus-half-thickness convention.
Its JSON and log are retained under `condensation_results/initial_outer_offset/`.
The final reconstruction follows the historical description's frozen bilinear
midsurface, with physical contact sides offset from that surface by half the
thickness. The old geometry snapshot is retained in
`diagnostics/initial_outer_offset_geometry.py.txt`; its hash matches the initial
JSON. Final comparisons keep the same patch definitions, general RHSs and
Schur algorithms rather than choosing patches to match historical values.
