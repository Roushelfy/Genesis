# Planar triangular loading proxies on unchanged Q4 elasticity

This CPU study keeps the existing shell stiffness, mass, gauge, inertia relief
and stress recovery. Only the physical loading surface and its displacement
interpolation are approximated by offline planar triangles.

Read [the measured Chinese report](REPORT_ZH.md),
[accuracy results](results/summary.json), [paired timing results](results/timing_summary.json)
and [independent checks](results/checks.json). This is a CPU research reference.
The maintained Genesis runtime and defaults remain unchanged.

## Reproduce

Use a checkout containing the source and native level-1 shell asset. NumPy,
SciPy, threadpoolctl and Matplotlib are required. No Genesis runtime import,
Quadrants, CUDA or Numba is required for this CPU study.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
python -m research.rigid_stress.triangular_load_cpu.study \
  --phase all --output /absolute/fresh/output
python -c 'from pathlib import Path; from research.rigid_stress.triangular_load_cpu.geometry_controls import run_geometry_controls; run_geometry_controls(Path("/absolute/fresh/output"))'
python -m research.rigid_stress.triangular_load_cpu.plot_results \
  --output /absolute/fresh/output
```

Use a fresh destination to preserve the published measurements. Numeric and
benchmark phases can run separately. Avoid other scientific CPU workloads
during benchmark measurement. Each timing method prepares fresh current loads.

## Source map

- `reference.py`: original physical Q4 surface-ball loading, closed-form Q4
  inverse and original nonnegative affine pressure/wrench fit.
- `proxy.py`: offline inner/outer triangular surfaces, consistent transpose
  load transfer, exact contained-disk moments and clipped integration.
- `study.py`: frozen 75-case bank, full-field/peak/resultant comparison and
  paired B16 complete load/solve/stress timings.
- `checks.py`: 17 independent work, geometry, analytical, inversion,
  quadrature and original-stiffness controls.
- `geometry_controls.py`: prescribed-footprint nonconvex torus diagnostics.
- `plot_results.py`: accuracy/cost and complete-surface stress illustrations.

Existing CPU elasticity is imported unchanged from `shell_cpu/core.py`.
The pressure fitting algebra is imported unchanged from `wrench.py`.
Its historical class name `PadPressureFit` introduces no deformable pad here.
The prescribed Hertz base density replaces the helper's generic input weights.

## Evidence

`results/protocol.json` freezes the mesh identity, case inputs, unchanged model,
quadrature and budgets. `fixtures.json` stores every original source. Field
norms include both sides and all four Gauss samples, with shear multiplicity two.
`numeric.json.gz` preserves all per-case outcomes and load diagnostics.
`timings.json` preserves all seven alternating-order paired repeats, with
stage timings, zero-load cases, sample counts and rejection counters.
`source_all.json` records source hashes before and after the scientific run.
`selected_fields.npz` contains the asymmetric two-source comparison fields.
`MANIFEST.json` records artifact bytes and hashes.

Two preliminary runs remain under `diagnostics/`. The first found eight
invalid cross-edge input forces. The final fixture uses one common force
based on the two face-normal bisector, retaining the original friction limit.
All final 75 cases are retained and valid. The second preliminary timing bank
used only outer-side loads. The final B16 bank includes both sides and is the
sole primary performance result.

The tiny warped-footprint miss is an explicit retained counterexample. Exact
force/moment preservation does not establish full-field equivalence. Same-Q4
agreement does not establish independent physical contact accuracy. This
delivery makes no GPU, Franka dynamics or policy-training throughput claim.
