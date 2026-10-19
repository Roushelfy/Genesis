# Dimensional normal-pad errors

The completed `pad-absolute-errors-cpu-v2` audit (728856) recovers absolute
membrane/bending RMS errors from all retained material columns and consolidates
81 absolute/relative metrics for nine admitted normal comparisons. All checks
pass;69 input receipts were rehashed, including the actual field and mesh files.
The [evidence packet](../evidence/20261019-qualified-ranking/normal-absolute-errors-v1/README.md)
retains the exact81-row CSV, complete peak locations/definitions, independent
3D-only component errors, source and terminal receipts. Full arrays stay on data.

This fills the dimensional reporting gap in the
[normal qualification](NORMAL_PAD_QUALIFICATION_20261019.md). It does not alter
the previously declared conditional uncertainty envelopes or ranks, establish
a new continuum bound, or complete the [active goal](QUALIFIED_STRESS_RANKING_GOAL.md).
The fixed250um repaired Shell, native full-source2mm patches,125um/three-wall
reference and physical compliant-pad model remain unchanged.

## Complete tensor RMS errors

For membrane N and bending M, the absolute error is
`sqrt(sum_q area_q * ||actual_q-reference_q||_F^2 / sum_q area_q)`.
Membrane units are N/m; bending moment per unit length has units N. Resultants
retain the original500um material-thickness integral. Tangent stress uses
physical volume weighting; outer/inner fibers use the complete mid-surface
area. Every column is included, with no polar trimming or thresholding.
Field integration is q9/depth2. Common peaks below use the separate q25
sampling; the two quadratures are not silently merged into one error norm.

| Load N | Method | Tangent RMS kPa | Membrane RMS N/m | Bending RMS N | Outer RMS kPa | Inner RMS kPa |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | DPR | 0.722758 | 0.170177 | 4.11081e-05 | 0.962154 | 1.14104 |
| 0.1 | SCR | 0.726653 | 0.17484 | 4.10165e-05 | 0.955655 | 1.14658 |
| 0.1 | Shell | 2.36668 | 0.20896 | 0.000167732 | 4.07163 | 4.0674 |
| 0.3 | DPR | 2.11917 | 0.48923 | 0.000123747 | 2.92006 | 3.50421 |
| 0.3 | SCR | 2.11288 | 0.504538 | 0.000121706 | 2.83161 | 3.49908 |
| 0.3 | Shell | 6.71945 | 0.609351 | 0.00047584 | 11.5508 | 11.5433 |
| 1 | DPR | 6.35939 | 1.53021 | 0.000366011 | 8.67373 | 10.3396 |
| 1 | SCR | 6.37222 | 1.62321 | 0.000358074 | 8.36851 | 10.3237 |
| 1 | Shell | 20.5312 | 1.93485 | 0.00145213 | 35.2558 | 35.2409 |

The recomputed membrane/bending relative errors match the published values
within1e-11 relative/1e-14 absolute tolerance. Tangent RMS is independently
recovered from saved volume-weighted squared errors; the boundary-fiber RMS
values reuse their original direct integration. Their reference RMS is derived
from the stored absolute/relative pair, not separately reintegrated.
Reference normalizations agree across all three methods within1e-9 relative
tolerance. All are positive and finite; no near-zero floor is introduced.
DPR/SCR remain unresolved ties for the field metrics under the existing
reference envelopes; these dimensional values do not create a new ranking.

## Unsmoothed peak-magnitude absolute errors

Entries are absolute differences between each actual and reference peak
magnitude, in kPa. These are not pointwise errors at the reference maximizer.
Raw peaks retain each method's native sampling and full3D definition where
available; common peaks use matched tangent-plane tensors on q25 material
columns. Raw and common definitions must remain distinct.

| Load N | Method | Raw Mises kPa | Raw tension kPa | Common Mises kPa | Common tension kPa |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.1 | DPR | 5.48291 | 6.35348 | 4.61883 | 6.30393 |
| 0.1 | SCR | 2.38353 | 5.74646 | 4.69117 | 5.69675 |
| 0.1 | Shell | 0.459501 | 16.7931 | 6.0774 | 16.7986 |
| 0.3 | DPR | 1.63534 | 2.43463 | 1.73327 | 2.30931 |
| 0.3 | SCR | 6.92138 | 0.491757 | 2.27427 | 0.366355 |
| 0.3 | Shell | 0.697885 | 46.481 | 17.0724 | 46.4966 |
| 1 | DPR | 15.8577 | 15.345 | 27.5821 | 15.6851 |
| 1 | SCR | 40.151 | 21.3471 | 29.9439 | 21.6863 |
| 1 | Shell | 3.52697 | 141.337 | 47.8301 | 141.384 |

The JSON retains both actual/reference common Mises and tensile locations and
wall depths. Its historical raw position is specifically the raw Mises
location; it must not be relabeled as the raw tensile location. Full3D normal
stress/shear remain separate components, with no invented Shell counterpart.
Their retained absolute Pa errors prevent a large relative error in a small
component from being confused with a dominant tangent-field error.

## Reproduction and resource scope

Run `python -m research.rigid_stress.qipc_comparison.pad_absolute_errors
--data-root RUN_ROOT --output NEW_DATA_PATH` through the captured submission
helper. This completed CPU-only analysis uses the same `rtx-mid` partition and
`GENESIS_IMAGE_VER=1_26 gs-srun` container, one CPU and8GB requested memory;
Slurm reports37s elapsed and7,948,516KiB maximum step RSS. Prefer16GB for
reproduction headroom. This is postprocessing cost, not simulator throughput.

`submit --cpu-only` requests no GPU; its default still requests one GPU.
CPU-only metadata explicitly records unavailable GPU inventory, while required
GPU inventory still raises on failures. Four mocked metadata controls pass.
Original GPU request728846 was cancelled while PENDING with no executed data;
its exact submission and cancellation receipt remain retained.
The comparator now also writes `membrane_rms_error_n_per_m` and
`bending_rms_error_n` for future field comparisons without changing integration.

The four prescribed-C2 fine-grid field groups were likewise still PENDING as
728823. They were replaced by CPU-only728860 at a fresh data path after
rechecking the admitted reference hashes. The same physical inputs, q9/q25
sampling and acceptance targets are retained. Only immutable submission and
cancellation records enter this packet; these active results and all active
friction writers are excluded. Their eventual spatial qualification remains
a separate required outcome.
