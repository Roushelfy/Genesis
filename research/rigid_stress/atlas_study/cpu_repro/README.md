# Reconstructed common-mother CPU atlas reference

This is a new executable reconstruction of the retained `cpu_next` report,
not the recovered original source. The historical unpushed commit
`3ac2e3a66203c9287de2dff428bb9ef4e8be506f` is unavailable. Historical claims
are preserved in [HISTORICAL_TARGETS.json](HISTORICAL_TARGETS.json); new results
and their qualifications are in [REPORT_ZH.md](REPORT_ZH.md).

The model is an asymmetric egg midsurface with 866 nodes, 864 MITC4 Q4
elements and 5,196 raw shell DOFs. Twelve genuine locally enriched spaces
have 3,522 raw / 3,516 free coordinates each. They share the same frozen
physical geometry, coarse interpolation, mass/inertia relief, six-coordinate
gauge and stress samples. No online remeshing or factorization is required.
The unchanged enclosing `shell_cpu` finite-element core is the dependency.

## Source map

| Modules | Purpose |
| --- | --- |
| `geometry.py`, `model.py`, `loads.py` | Mother geometry, `T=[P,B_local]`, factors, finite-cap force and eccentric friction loads |
| `fixtures.py`, `selection.py`, `selection_study.py` | Frozen 42-loaded-case bank, continuous / nearest / narrow source-wise selection and grouped RHSs |
| `boundary_study.py`, `plot_stress.py` | Pair/triple selector continuity and whole-domain top/bottom stress maps |
| `pipeline.py`, `benchmark.py`, `audit.py` | Current loads through complete field or sampled maximum, profiling and independent element controls |
| `response.py`, `response_study.py` | All-column response inventory, FP32, whole-domain maximum bounds and retained POD failure |
| `history*.py` | Residual-gated prediction, failure compaction, rotating state, real cached `Kq`, lifecycle and motion-rate controls |
| `condensation.py`, `condensation_study.py` | Both exact Schur directions and deliberately dropped-coupling controls |

See the individual `CORE_README.md` and `README_*.md` files for API and scope.
The original protocol, geometry correction and negative pilot evidence are
retained. Superseded `initial_outer_offset` results are diagnostics, not the
geometry used for the final tables or formal timings.

## Reproduce in a fresh output directory

Install [requirements.txt](requirements.txt) in a research environment.
No Genesis engine initialization, GPU, scikit-fem or Gmsh is needed. From the
repository root, this runs numerical validation and then exclusive timings
sequentially, preserving stdout/stderr and command receipts:

```bash
python research/rigid_stress/atlas_study/reproduce_cpu_repro.py \
  --scope all --output /absolute/path/to/a/new/cpu-atlas-run
```

Use `--scope numerics` or `--scope timing` for a subset. The output directory
must be empty. The wrapper sets BLAS/OMP/MKL threads to one. Stop other CPU
numerical jobs for timing. Expect several minutes and a few GB of RAM; the
large response matrices are rebuilt in memory and are not committed.

The retained numerical result locations are `results/`, `history_results.json`,
`history_operator_controls.json`, `history_speed_results.json`,
`history_slow_results.json` and `condensation_results/`. Formal measurements are under `formal/`, including
raw repeats, command receipts and logs. Timing files record source hashes and
the actual one-thread numerical environment. The measured static B64 is four
tiles of 16 frozen conditions, not 64 independent grasp trajectories.

The retained full motion-case stream is losslessly compressed as
`history_speed_cases.jsonl.gz`. `gzip -dk` restores the JSONL with the exact
`cases_sha256` recorded in `history_speed_results.json`. New runs emit plain
JSONL in their fresh output directory.

## Interpretation

Each contact contributes only to its selected feature(s). Contributions
sharing `(feature, environment)` are summed before solving. The displacement
contributions are lifted into the mother coordinates and summed before one
stress recovery. Tensor components are combined before any von Mises maximum.
The sum of different feature responses is generally not the joint-refined
Galerkin solution. Its spatial approximation error is measured separately
from linear-solve residual and response-inventory roundoff.

The finite footprint is a prescribed normalized material-sphere cap with
Hertz-shaped radial density, normal/friction force and top-surface eccentricity.
Its nominal radius is not a certified physical contact radius. These runs do
not establish continuum peak stress, independent contact physics, arbitrary
nonconvex mapping, native Quadrants integration or Franka/RL/GPU throughput.
Native integration remains governed by the enclosing branch goal.
