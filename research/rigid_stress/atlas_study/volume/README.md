# Independent 3D volume FEM feature-mesh control

This is CPU feasibility evidence for combining full-body responses computed on
several offline locally refined meshes. It is **not** a hollow-egg/Franka run,
a native Quadrants feature, or a converged continuum-contact reference.

## Model and source loading

- Exact common box: 20 x 20 x 10 mm, E=10 GPa, nu=0.3; every displacement
  component is clamped on the bottom z=0 face for all meshes.
- Affine tetrahedral geometry, quadratic P2 displacement; the volume elastic
  energy is integrated with an exact degree-two rule. scikit-fem 12.0.2 is the
  independent assembler. Geometry and material are unchanged between meshes.
- Actual finite Hertz disks on the top face, with true arbitrary x/y position,
  normal and tangential resultant, radii 0.55/0.7/1.0/1.3 mm. The footprint is
  not snapped to an anchor. Normal force is typically 0.05 N. These are supplied
  finite-pressure inputs, not a coupled contact or radius-estimation solve.
- Three offline anchors x=-3/0/+3 mm, y=0. Conforming longest-edge bisection
  targets h=1.1 mm inside a 2.2 mm radius and gradually coarsens outside.
  The nominal target core completely contains 14 of the 20 queried pressure
  disks; remaining disks extend into graded transition regions. Exact coverage
  margins are recorded in `switch_control.json`.
- The original main suite has 16 load cases: a moving shear trajectory, several
  radii, near/far double contacts, and three asymmetric contacts. Each contact
  is solved only on its selected feature mesh; global stress tensor responses
  are evaluated at common physical points and summed. Smooth interpolation
  combines responses with weights depending on **source** position, constant
  over the observation coordinates.

## Files and provenance

| File | Status |
|---|---|
| `results_bounded/results.json` | Complete original 16-case suite; all selected force/moment, residual, linear superposition and coarse-plus-delta identity controls pass |
| `model_primary_general_pivot.py` | Byte-exact model used for the original complete suite, SHA256 b8bcac9e897e687097f9f7cc0930292916de81eeb4dacdc09a9ed00452448879 |
| `model.py` | Current model adds SPD-aware LU; geometry, loading and stress mapping unchanged; generic pivoting remains selectable |
| `refined_reference/results.json` | Further h=0.75 mm union-reference run: 15 complete row records persisted; final numerical summary flags did not persist |
| `refinement_summary.json` | Historical integrity audit of the missing record; retained unchanged |
| `finer_case16_receipt.json`, `refinement_complete.json` | Freshly recomputed last case, original partial-report hash and actual residual/q48-q96 checks; complete combined 16-case finer-reference record |
| `switch_control.json` | Dedicated fixed-force/radius/y switch controls and coverage diagnostics |
| `invariants.json` | Current-source exact affine/quadratic stress-map, rigid-mode, reciprocal-work and independent scatter controls |
| `volume_accuracy.png`, `volume_fields.png` | Figures directly from recorded fields and comparisons; inspected visually |
| `model_initial_unbatched.py`, `initial_unbatched_locator.log` | Historical pre-results locator attempt; global query batching was too expensive and interrupted |
| `run_chunked.log` | Another large setup attempt, interrupted before its first model result; no numerical claims derive from it |

All current runtime work in this directory is research NumPy/SciPy, not
production integration. Source hashes in each JSON identify which version ran.
Do not relabel old generic-LU timings as current SPD-aware timing. The separate
performance study measures the current source after other heavy jobs finish.

## Numerical checks

- Maximum free-DOF residual in the complete primary suite: 4.87e-14.
- Same-mesh per-contact response sum versus total-load solve: <=8.59e-15.
- Whole-domain coarse-plus-full-correction identity: <=6.82e-16.
- Current affine and quadratic constitutive stress-map errors: 1.23e-15 and
  4.53e-16; rigid zero-stiffness residuals <=5.42e-18.
- Reciprocity error 1.79e-16; manually implemented P2 scatter versus independent
  scikit-fem probe interpolation error 3.00e-16.
- All selected force/moment checks pass, but this alone does not certify
  pressure integration. The reference q24->q48 tensor-field changes reach
  0.114%; the smallest-radius finer q48->q96 check changes 0.00491%.
- Deliberately applying the **entire** load to every selected mesh produces
  large double-counting errors (78.4% for two far contacts, 175.3% for three).
  Two near contacts select only one mesh, so that particular negative-control
  construction has no duplicate loading and is not a counterexample.

The common field metric is an **unweighted Frobenius norm over point samples**:
847 regular global samples plus 2,975 denser near-field samples. It is not an
integrated volume L2 norm. Individual global and near-field values are also
recorded. Peaks are samples, with nearest depth 50 micrometres; they are not
continuum, whole-boundary or certified global stress maxima. Clamp corners
remain sharp, and their ideal continuum maxima are not qualified.

## Accuracy findings and limitations

The coarse model has 1,701 DOF. Feature meshes have 13,812--13,872 DOF each;
the h=1.1 mm union model has 24,432; reference models have 33,858 (h=0.9 mm)
and 55,281 (h=0.75 mm).

Against the h=0.75 mm discrete reference, the completed 16 comparisons have:

- coarse common-point tensor differences about 73.8--90.0%;
- feature-mesh differences about 11.6--31.7%;
- comparable joint-refinement differences, about 11.4--31.9%.

The h=0.9->0.75 mm reference changes are still about 7.2--19.1% in this norm.
Thus the atlas strongly improves the coarse field on the tested samples and
can approach the joint-refinement model, but these percentages are **not
qualified physical stress errors**. The smallest footprints remain poorly
resolved. Their actual 45-micrometre egg counterpart was not tested here.

Hard nearest-mesh switching has a same-source discontinuity of 8.08--8.71%.
At a fixed force/radius/y, moving only 0.2 micrometres across a switch produces
about 8% hard-switch change, versus 0.0247--0.0249% for smooth source weights,
matching the joint-mesh physical-motion change. At 2 micrometres, the blended
change is about ten times larger. Smoothness repairs the artificial jump; it
is not an independent equilibrium or error certification.

## Reproduction

Install `numpy scipy scikit-fem==12.0.2 matplotlib threadpoolctl`. Gmsh was
unavailable because its native library needed libXft; the actual experiments
use scikit-fem's conforming adaptive tetrahedral mesh, without Gmsh.

From the directory containing `atlas_study`, under a separate checkout/output
folder if recorded evidence must remain frozen:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m atlas_study.volume.invariants
python -m atlas_study.volume.study --fine-h .0009 --out atlas_study/volume/results_bounded
python -m atlas_study.volume.refinement
python -m atlas_study.volume.complete_case16  # only to reproduce the historical partial-record completion
python -m atlas_study.volume.switch_control
python -m atlas_study.volume.plot_results
```

If rerunning refinement.py now persists all16 rows, its report is already
complete; the historical completion helper intentionally expects the retained
15-row snapshot and must not overwrite a fresh full report.

The current source uses SPD-aware LU by default; use
`BoxModel(mesh, symmetric_factor=False)` or the archived model for old generic
factor timing. Generated NPZ arrays can be reproduced; they need not all be
tracked in Git. Keep JSON, source pins, failed attempts and scope limitations.
