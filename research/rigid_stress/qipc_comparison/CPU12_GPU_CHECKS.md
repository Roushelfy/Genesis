# Native CPU12 numerical checks

The scientific driver `check_cpu12_gpu.py` runs only in Slurm `rtx-mid`, image
`1_26`. It reads a successful, hashed `genesis.cpu12-reference.v1` export and
keeps generated assets, matrices, fields, checks and source receipts under its
explicit project data output directory. It does not report performance timings.

```sh
python -m research.rigid_stress.qipc_comparison.check_cpu12_gpu \
  --reference /path/to/frozen/reference \
  --output /path/to/new/data/check \
  --git-source /path/to/actual/git/checkout \
  --bank published43 --stage all
```

`--git-source` supplies Git identity when the executed research files are in a
captured source directory without `.git`. The actual imported native package
and executing driver have independent before/after source hashes.

## Stages and banks

- `--stage operators` checks the complete mother and twelve feature spaces:
  bilinear P, T, physical K, explicit gauge/free masks, local nodes and active
  macros, center ordering, geometry, MITC4 B operators, mass, G/MG/Gram, common
  relief and projected common relief. The six deleted CPU scalar columns map
  explicitly to the native 3522 raw-coordinate masks.
- `--stage loads` checks the original Gauss-Jacobi material cap, exact ray/Q4
  samples, source resultants, raw loads, three selectors, source contributions,
  environment groups and common mother relief. It builds the same native
  mother; it does not allocate twelve feature factors.
- `--stage all`, the default, includes both stages and compares every native
  feature's projected RHS, coordinates and unscaled physical residual with the
  independent CPU Ki. It compares merged displacement, side-major local3 and
  body-frame full9 tensors, sampled peak, N/M/Q, exact-zero release and the
  separate field/peak spatial errors against a matching native mother solve.
- `--bank` accepts `published43`, `boundary18`, `static16`, `static64` or `all`.
  The static banks gather the CPU oracle's explicitly frozen column aliases;
  native sources and environment groups are independently prepared each time.

Each selector calls the native material-cap and selector implementation, with
all positive contributors. The parent raw force is the device sum of native
feature forces. Atlas output is captured before calling the exact mother with
that same raw force. CPU forces, matrices, selector outputs and solutions are
never uploaded into this native path. SciPy reconstructs exported native sparse
arrays only for untimed comparison, and applies the independent original Ki
only to audit reduced residuals.

## Tolerances and interpretation

`checks.json` states every comparison's tolerance and measured norm. Floating
comparisons use whole-array L2/Frobenius error bounded by
`rtol * reference_norm + absolute_l2_floor`; geometry/operators/work default
to `rtol=1e-10`, coordinates/fields to `1e-8`, with explicit small absolute
floors. Connectivity, masks, counts and sample identities compare exactly.
Nonfinite comparisons fail. Complete native `valid` flags and release zeros
are checked in addition to CPU agreement.

Feature acceptance checks `norm(Ki*z-rhs)/max(norm(rhs),1e-30) <= 1e-8`, using
the original unscaled free-coordinate Ki. Only the separately solved exact
mother is checked against its full free residual. An approximate Atlas field's
mother residual is not a numerical acceptance condition or an instruction to
perform a mother correction.

Native stress samples are element-major, two sides, six symmetric components.
The driver explicitly reorders sides before reconstructing full9 tensors and
projecting local3/N/M/Q through the native Gauss axes. The weighted field norm
counts shear components twice; the sampled von Mises maximum is evaluated after
all source contributions have been summed.

## Initial driver scope

The first runnable version checks complete solved stress maps through the native
MITC4 constituents and full recovery; it does not export a separately assembled
native sparse S. It preserves exact sample IDs as diagnostics; a legitimate
neighboring-cell alias at a fine-grid seam still needs an explicit equivalence
audit if the first GPU run reveals one. Finite-footprint coverage counts and
boundary-jump summary aggregation remain follow-up audit additions. The first
version saves the latest mother output under each bank; per-selector raw loads
and Atlas/spatial outputs remain separate. A subsequent revision should retain
all three paired mother output files. These scope notes do not qualify any
complete-scene throughput, memory benefit or native DPR adaptation.
