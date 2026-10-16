# Matched native distance/integration controls and volume stress norms

This checkpoint follows the [2026-10-16 practical goal](PRACTICAL_VALIDATION_GOAL.md).
Its completed checks are accepted within their stated scope; unresolved physical
tails remain limitations or research follow-ups. The full historical matrix
is no longer an engineering-delivery prerequisite.

Eight additional producers720132/134/136/138/140/142/144/146 and their eight
separate auditors720133/135/137/139/141/143/145/147 now naturally complete.
Together with the four previously audited fine native runs, these are12
five-load/four-model runs:228 accepted native fields and12 explicitly unsupported
near-slip pressure inputs. Independent audits check2274813684 quadrature tensor
components at the original force, physical-point scatter, corner, energy and
source gates. No original moment or numerical tolerance is changed.

The separate comparison720212 also completes with source and input-source
closure. It retains120 numerical-control rows and120 model/reference rows,
including every unsupported row. All366 recorded input files are rehashed.
The [full physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains open.

## Controlled inputs and observable

At the original finest mesh, compare ports4/8/16 with native quadrature10,
then quadrature10/14 at ports8. The ellipsoid has41566 tetrahedra and the
capsule43481. Every frozen-input NPZ byte is identical within each numerical
comparison, including physical force points, forces, free moments, collision
normals,3mm radii, friction and angular velocity. Geometry, material and pose
also match. The four declared load/support models remain separate.

The additional spatial levels are111286/315446 ellipsoid tetrahedra and
115768/326676 capsule tetrahedra. Each native field is compared with the
separately full-field-audited prescribed finite-traction FEM field on its own
mesh. These per-mesh differences are not cross-mesh field-convergence estimates.

For a linear symmetric tensor on an affine tetrahedron of volume V, integrate
its squared Frobenius norm exactly from the four unsmoothed corner tensors:

```text
integral ||sigma||_F^2 = V/20 * (sum_i ||sigma_i||_F^2 + ||sum_i sigma_i||_F^2).
```

Native components use xx/yy/zz/xy/xz/yz, with shear multiplicity two in this
norm. A manufactured arbitrary tensor field is checked independently with the
degree-two four-point tetrahedral rule; a constant-tensor control also passes.
The difference norm is divided by the target field's volume norm. This is a
newly explicit volume L2 statistic; previous unweighted full-corner statistics
remain retained. It does not average or replace any raw global maximum.

## Same-mesh numerical sensitivity

Each maximum spans all19 accepted case/model fields for that geometry. Tensor
and peak maxima can belong to different fields. The three comparisons isolate
the stated distance or integration setting while keeping all physical inputs.

| Geometry | Setting change | Largest volume tensor change | Largest raw-peak change |
|---|---|---:|---:|
| Ellipsoid | Ports4 to8, quadrature10 |1.47130%|2.91088%|
| Ellipsoid | Ports8 to16, quadrature10 |0.88725%|1.55195%|
| Ellipsoid | Quadrature10 to14, ports8 |0.013452%|0.019820%|
| Capsule | Ports4 to8, quadrature10 |1.72819%|2.99544%|
| Capsule | Ports8 to16, quadrature10 |0.76650%|1.15774%|
| Capsule | Quadrature10 to14, ports8 |0.012802%|0.016669%|

The largest tensor change in each row belongs to asymmetric loading with the
source prior. Integration sensitivity is smaller in this declared experiment;
the remaining intrinsic-distance sensitivity is measured explicitly. Neither
three port levels nor a small last change supplies a geodesic or continuum
error bound. These controls do not close contact-reference spatial/geometry
convergence, feedback or an application operating range.

## Finest conditional model/reference differences

Values below use actual raw native volume tensor L2 relative to the separately
audited prescribed-traction reference on the finest315446/326676-tet meshes.
The declared reference uses a Euclidean finite profile while the intrinsic
models use their own distance/support law. Consequently these are conditional
model/distance/integration differences, not pure model error or resolved-contact
truth. There is no universal ranking of contact models.

| Geometry and load | Pressure | Componentwise | Pair union | Source prior |
|---|---:|---:|---:|---:|
| Ellipsoid opposing |0.00687%|0.46218%|0.46218%|0.46544%|
| Ellipsoid near-slip |Unsupported moment|3.06619%|3.06619%|3.77140%|
| Ellipsoid multiple |0.00671%|0.45583%|5.95376%|0.45984%|
| Capsule opposing |0.00578%|0.40120%|0.40120%|0.40189%|
| Capsule near-slip |Unsupported moment|4.88957%|4.88957%|4.87236%|
| Capsule multiple |0.00597%|0.40703%|4.56277%|0.41116%|

The [previous first-added-level report](PHYSICAL_FINE_NATIVE_MODELS_20261016.md)
retains actual destination-disk redistribution and its original unweighted
statistics. This new volume statistic supplies another full-field comparison
without hiding the original peak or local resultants. Every numerical-control
row also retains changes in each destination disk's force. Those identities
are exported destination disks, not inferred continuum contact islands.

These finest-model differences and the numerical-control table concern
different resolutions and observables. They cannot be subtracted to construct
an error budget. The original ellipsoid/capsule CAD raw-peak tails still change
16.75%/14.10%, so numerical/model agreement has not established continuum
accuracy. [Fixed-polyhedron controls](PHYSICAL_FIXED_POLYHEDRON_TAIL_20261016.md)
also retain unresolved raw peaks.

## Reproduction and remaining deliverables

The comparison uses `scripts/compare_native_finite_axes.sh NEW_RUN --runtime-root P`.
Exact Slurm commands, complete inputs and source identities are recorded under
the established runtime root in:

```text
meta/native-convex-matched-distance-quadrature-commands-v1.json
meta/native-convex-matched-numerical-axes-comparison-command-v1.json
native-convex-matched-numerical-axes-comparison-v1/results/summary.json
native-convex-matched-numerical-axes-comparison-v1/results/numerical_axes.csv
native-convex-matched-numerical-axes-comparison-v1/results/model_bias.csv
```

Complete field and comparison archives are byte-verified against actual runtime
originals. Both matched field archives have been pushed. Both complete fine native
levels are already pushed, as recorded in the fine-field report. Each new
matched-field geometry archive retains5904 entries and5904 actual runtime
originals. The comparison archive uses these reverified parents:

| Archive | Manifest SHA256 |
|---|---|
| [Ellipsoid matched fields](../evidence/20261014-physical/native-ellipsoid-matched-axes-fields-v1/manifest.json) | `2b13d27905511733f300c9dbba0c4da46160f76be39082f44a1c41799efc8bd7` |
| [Capsule matched fields](../evidence/20261014-physical/native-capsule-matched-axes-fields-v1/manifest.json) | `58c612d0c65ef2ce614bdadc687198d81f510976703278b1fdb1f48cd406a582` |
| [Complete comparison](../evidence/20261014-physical/native-convex-matched-axes-summary-v1/manifest.json) | `99453f01a982909e1596946c810f348a980ed06c5fb016a4e729548e5721a02b` |

All raw arrays, unsupported inputs and original failures stay.
Representative stress/traction maps, physical profile assumptions, resolved
contact reconstruction and separate feedback effects remain application work.
Further full-matrix convergence is a research follow-up. Recorded cold timings
do not establish warmed application cost.
