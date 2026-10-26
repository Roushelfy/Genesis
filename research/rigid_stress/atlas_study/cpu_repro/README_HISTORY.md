# Batch history reconstruction

These files reconstruct the reported two-frame prediction, failed-column
compaction, buffer rotation and actual `Kq` caching experiment. The underlying
mother shell and its output samples are shared with this reconstruction's
static experiments. The old source and complete trajectory parameters were
unavailable. The trajectory below is a new frozen deterministic control, not a
claim to recover the old 1.21x speedup or 0.96% maximum field difference.

## Algorithm

`RecoverySystem` uses the gauged mother's free stiffness, stress map and one
equilibrated factor. Both exact and predictive recovery compact active columns
and clear releases. Prediction uses the preceding two accepted displacements
and complete local-three-component stress fields. Approximate accepted fields
feed subsequent history. The current-load gate is

```
||D (K q_pred - f_current)||_2 / max(||D f_current||_2, 1e-30)
D = diag(K)^(-1/2)
```

Cold, reset or failed environments take one compact multiple-RHS solve and
stress recovery. A tolerance of 0.01 or 0.05 is a numerical residual gate,
not a guaranteed stress-error percentage. Field and sampled-maximum errors
are evaluated independently against the exact current-load mother solve.

The copy and rotation variants have identical logical two-frame histories.
Rotation reuses three work/history arrays for displacement and three for
stress. Cached products additionally retain three `g=Kq` arrays. Accepted
predictions extrapolate the actual products. Exact failures refresh them with
`K @ q_solved`, rather than substituting the RHS. The numerical study also
evaluates direct `K @ q_pred` shadows, records decision differences, product
drift and the nearest tested threshold gap. It does not claim an arbitrary
long-duration rounding guarantee or a rigorous stress-error bound.
The reconstructed cache additionally recomputes the actual product whenever
its gate ratio lies within `1e-10` of the threshold. Recheck counts are retained.

## Frozen trajectory

`sources_at` defines two independent asymmetric contacts per loaded environment.
Environment phase uses the golden-ratio fractional sequence. Azimuth rates
vary by environment/contact, latitudes and force magnitudes evolve smoothly,
tangential friction varies between 0.2 and 0.4, and the nominal material-cap
radius scales with normal force to the one-third power. Initial nominal radii
are 1.6 and 1.8 mm. The force is explicitly prescribed as
`Fn * (-direction + friction * tangent)` in the fixed frame. This avoids an
extra facet-normal query and keeps force smooth across macro-chart seams.
Here `Fn` names the prescribed radial compression scale. This is supplied shell loading, not a deformable contact
solution or automatic physical Hertz contact-area validation.

At frames 24--27, environments divisible by four release both contacts.
Frame 35 resets indices congruent to zero modulo five. Frame 61 resets indices
congruent to two modulo seven. At frame 42, indices congruent to one modulo
four receive an azimuth and normal-force shock on their first contact.
All other motion remains continuous. The numerical JSON includes per-frame
errors, accepted counts and exact fallback IDs plus a SHA256 of the complete
balanced force trace.

Every benchmark starts with fresh history and traverses the entire trace.
It includes cold history, releases, reset and shock work. Reset isolation is
checked on private state clones, so an exact oracle never replaces actual
predictor histories during the qualifying trace.

## Commands

Run from the repository root with one BLAS/OpenMP thread:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.history_study \
  --batches 16 64 --frames 80 --order 8 --tolerances .01 .05 \
  --output research/rigid_stress/atlas_study/cpu_repro/history_results.json
```

Run formal performance alone, after other numerical workers stop:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.history_benchmark \
  --batches 16 64 --frames 80 --order 8 --repeats 3 \
  --output research/rigid_stress/atlas_study/cpu_repro/history_timings.json
```

`recovery` receives the same pre-generated balanced loads and includes gate,
compression, solve, stress evaluation, scatter and history updates. `complete`
regenerates the current contacts/load/relief independently for each method
inside timing. It compares full recovery, direct-gate rotation and cached-gate
rotation at tolerance 0.01. Recovery additionally tests copy/rotation/cache at
0.05. Exact recovery receives the same release/active compaction conveniences.

All outputs contain the complete 20,736-row local stress field. Global 3x3
tensor export is excluded for every history variant. These complete local
field timings cannot be compared as absolute values with a nine-component
body-tensor benchmark. Checksums and truth comparisons are outside timing.
Source hashes before and after the timed phase must match. Raw paired/reversed
repeat wall and process CPU times are retained, with separate history storage.

## Predeclared smooth-time sensitivity

`history_speed_study.py` is a separate numerical control with speeds
`{0.25, 0.5, 1, 2}`, B16, 80 frames, order8 and gate0.01. Every smooth
azimuth/latitude, force, friction and radius expression uses
`smooth_time = speed * frame`. Release at24--27, resets at35/61 and the
frame42 shock remain at their original frame numbers. The shock keeps its
0.18-radian increment and force multiplier1.6, with the radius responding
through the same one-third-power law. The entire speed-one balanced-force
trace must have the same SHA256 as the saved main numerical trace.

Each speed covers a different total smooth path over80 frames. This measures
sensitivity to per-frame evolution, not the cost of completing the same
motion faster. All four results are retained and no favorable speed replaces
the frozen main result. `history_speed_cases.jsonl` contains every actual
source, active environment, event and smooth time used. The results contain
full per-frame numerical diagnostics, acceptance and direct-cache controls.
No timing is performed by this control.
The inherited diagnostic setup duration includes case-file serialization and
is not a throughput measurement.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.history_speed_study
```

`history_scaled_trajectory.py` reuses the frozen sensitivity source generator
for the separately declared speed0.25 B64 control. At speed1 it calls the
original main force/preparation functions directly. The benchmark option
`--speed` defaults to1, so the main command above retains its force trace.
The speed and output scope are written in the timing JSON. No source-case
serialization is included in its complete-path timing.

Before timing speed0.25 at B64, run this independent numerical qualification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.history_slow_study
```

`--speed-results path/to/history_speed_results.json` selects a separately
regenerated sensitivity input; the input SHA is captured and checked too.

It freezes B64,80frames,speed0.25,order8,gate0.01; reproduces the frozen B16
force SHA with the shared generator, then checks the B64 first16 columns to
roundoff tolerance (mass-relief matrix reductions can differ with batch size);
and records every
frame's field/peak errors, exact fallback IDs, per-environment direct/cache
gate values and decisions. It emits `history_slow_results.json`, with source
and complete force SHA256, rather than duplicating source-case JSONL. It
performs no timing. Only after this qualification, measure the separately
declared control in a quiet window:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.history_benchmark \
  --batches 64 --frames 80 --order 8 --repeats 3 --speed .25 --scopes complete \
  --output research/rigid_stress/atlas_study/cpu_repro/history_slow_timings.json
```
