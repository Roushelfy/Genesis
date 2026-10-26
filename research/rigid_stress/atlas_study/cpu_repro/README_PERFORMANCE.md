# Complete CPU pipeline measurement

`pipeline.py` rebuilds the current material-cap quadrature, surface location,
shape-function loads and eccentric couples on every invocation. Feature
selection acts on each source. Loads sharing a `(feature, environment)` pair
are combined before a multiple-RHS solve. Lifted feature displacements are
summed in the common mother space and the sparse stress map is applied once.
The independent-recovery control sums each feature's stress instead.

The mother uses one common mass/rotary-inertia relief. The projected-RHS
variant evaluates `T.T @ raw - (T.T @ mass_modes) @ relief_coefficients @ raw`
before its solve. It has the same load model as balancing full source vectors
before projection. The variant is named explicitly in benchmark results.
Every selector is measured with both full balancing and projected-RHS
preparation: continuous, nearest and narrow. These paired controls separate
selector/group-count savings from projected-RHS savings. Comparing optimized
nearest with balanced continuous measures both changes together.

## Outputs and cases

- `full` returns both surfaces at all four Gauss points of all 864 elements,
  expressed as complete body-frame 3x3 tensors. Symmetric entries are present.
- `max` computes von Mises stress at all 6,912 authored output points and
  returns one maximum per environment.
- `local` exposes the 20,736 common in-plane scalar stress rows for numerical
  controls. It has a smaller output cost than `full`.

The timing tile contains 16 frozen cases: 15 loaded cases, one release and
19 sources. B16 evaluates this tile together. B64 repeats it four times,
giving 76 sources and 16 independent conditions. B1 is the arithmetic mean
of 16 separate single-environment calls. B64 therefore does not represent
64 independent conditions. The larger 42-loaded-case accuracy corpus belongs
to the separate selection study.
The mother compresses zero aggregate RHS columns before relief, factor
solving and stress recovery, then scatters results into the complete batch.
Its inactive/release outputs remain exactly zero. Thus its B64 factor solve
has 60 loaded RHS, matching the active-environment convention used by the
feature paths. Complete requested outputs still contain all 64 environments.

Each immutable case's global force is evaluated once at its declared center
and recorded explicitly in the frozen manifest. The current cap quadrature,
material positions, nodal forces and moments are still rebuilt inside every
timed call. This is the explicit-force front-end optimization from the earlier
experiment. There is no per-case displacement or stress-response cache.

The complete H alternative is a shared all-column linear operator prepared
offline. It includes the common relief and takes raw top-force inputs.
Application includes input validation and packing. H32 uses FP32 inventory,
input coordinates and accumulation. Body-frame tensor conversion uses the
same FP64 geometry basis as the other complete-field paths, so the full9
H32 path is mixed precision rather than an entirely FP32 pipeline.

## Numerical controls

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.rigid_stress.atlas_study.cpu_repro.audit \
  --output research/rigid_stress/atlas_study/cpu_repro/results/audit.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.rigid_stress.atlas_study.cpu_repro.benchmark \
  --output research/rigid_stress/atlas_study/cpu_repro/results/pipeline_numerics.json \
  --orders 4 8 --include-response --independent-control --accuracy --numeric-only
```

`audit.py` checks closed topology, rigid modes, geometry interpolation, common
relief virtual work, original reduced-system residuals, full finite-load
shape support, projected-RHS consistency and grouped superposition. It also
compares the sparse stress map and body-frame tensors with the element
constitutive evaluator. A separate 3D deviatoric tensor calculation checks
the plane-stress von Mises reduction. These are discrete numerical controls.

The accuracy file compares the exact same 16 explicit-force cases at q4/q8
against mother q24. It additionally records differences from the mother at
the same quadrature, separating integration sensitivity from spatial or
floating-point differences. Complete-field errors use common physical
Gauss-area weights and the Frobenius shear factor two. Sample maxima are
reported separately. The integration model is the declared material angular
cap, rather than an independently solved physical contact patch.

## Quiet performance window

Stop other numerical workers before this command. The runner itself applies
`threadpool_limits(1)` and records the actual BLAS thread counts.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.rigid_stress.atlas_study.cpu_repro.benchmark \
  --output research/rigid_stress/atlas_study/cpu_repro/results/benchmark.json \
  --orders 4 8 --batches 1 16 64 --repetitions 5 \
  --include-response --independent-control
```

Each method warms once. Timed repetitions alternate forward and reverse
method order. Both wall and process CPU times are retained, with median,
minimum, maximum and every repeat. Checksums execute after both clocks stop.
Current-load preparation, feature selection/grouping, relief/projection,
solves/response, lifting, stress and requested output are included. Offline
assembly, factors and response construction are recorded separately.

A single additional instrumented call records source, grouping/preparation,
projection, solve, merging, stress and output stages where exposed. Its stage
durations are diagnostic and cannot replace or be added to paired headline
medians. Source hashes before/after and the frozen case-manifest digest are
part of each result. A modified source set invalidates a quiet timing claim.
`recovery_seconds` includes solve, merge and stress sub-stages; it must not be
added to those same sub-stages when aggregating diagnostic costs.

The process RSS peak includes the simultaneous mother, feature factors,
H64/H32 inventories and maximum-bound data when enabled. It is not the
allocation of one selected production configuration. Inventory byte counts
remain separate. Timings are CPU stress-module costs and do not include
rigid dynamics, Franka control, GPU execution or RL policy inference.

The initial geometric interpretation with an outer-surface offset is retained
under `results/initial_outer_offset/`. Final results use the declared authored
coarse midsurface. The change was selected from the report's geometry
definition before performance timing. No old absolute timing or accuracy
number is a target for the reconstructed implementation.
