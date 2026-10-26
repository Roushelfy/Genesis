# Complete nodal responses and global sampled maxima

`response.py` implements a complete fixed-mother top-force inventory. For each
fine node, the three input forces also induce the physical top-surface couple
`h/2 * (director cross force)`. With the common mother gauge and inertia relief,

```text
B g = raw generalized top load
H = S K_g^-1 relief(B)
stress = H g
```

All 2,598 nodal force columns are retained. Positions, radii, source count and
friction can change online. The current forces are integrated before response
application. Relief is included once in construction, so `apply` accepts the
raw unrelieved generalized load. It checks the top force/couple relationship.
Explicit bottom-side loads and independent nodal couples are rejected. Those
channels require their own inventory or a more general force/moment basis.

The output is the original mother stress ordering: top/bottom, element,
Gauss point, local `xx/yy/xy`. It contains 20,736 scalar rows at 6,912 samples.
It represents in-plane membrane-plus-bending stress. Body-frame tensor output
and criterion reductions can be performed from these common components.

The inventory is shared across environments and rebuilt in RAM. Large H
arrays are neither committed nor stored as numerical-result archives.

## Reproduce

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m research.rigid_stress.atlas_study.cpu_repro.response_study \
  --output /absolute/new-output/response
```

This executes numerical controls, global maximum bounds and energy POD.
`--no-pod` or `--no-bounds` omits those optional studies. Use a fresh output
directory to preserve a previous run. Source preparation and inventory
construction are recorded separately from the formal throughput benchmark.
The latter is owned by the common pipeline timing driver.

## Reconstructed numerical results

The final geometry uses the authored egg as the middle surface. The independent
response bank freezes 36 Fibonacci directions at phase 0.371, six axes, two
asymmetric multi-source cases and one release: 45 cases, 44 loaded. This bank
is distinct from the selection study's 42 loaded cases and from the old
unrecoverable source's exact fixture details.

| Quantity | Reconstructed value |
| --- | ---: |
| H dimensions | 20,736 by 2,598 |
| FP64 H payload | 430,977,024 bytes |
| FP32 H payload | 215,488,512 bytes |
| FP64 complete-field worst relative difference | 3.6234e-15 |
| FP32 complete-field worst relative difference | 7.9366e-8 |
| FP32 sampled-peak worst relative difference | 2.3281e-7 |
| Raw top-couple absolute difference | 1.9843e-19 |
| Release and untouched-environment differences | exactly zero |

`results/response/response_results.json` retains all cases. Field differences
use common physical Gauss areas and symmetric-tensor Frobenius weights
`[1,1,2]`. FP32 tests quantize both inventory and current input, with FP32
matrix accumulation. This is separate from a full FP32 stiffness solve or
TF32. The current CPU result reproduces the inventory dimensions and the
small additional floating-point error, rather than claiming the old exact
case values were recovered.

The same control also covers 80 moving frames with two independent environment
identities, radius changes, asymmetric friction, partial release/reset,
position shock and alternating batch-column order. Worst FP64/FP32 field
differences are 3.4873e-15 / 9.1430e-8. Maximum screening differs by at most
4.1306e-16. Every released field is exactly zero.

## Conservative sampled maximum

For each block of elements, `MaximumBounds` stores the maximum von Mises norm
of every input column. The triangle inequality bounds the total field by the
sum of absolute current inputs times these column bounds. Candidate blocks
are evaluated exactly until all remaining bounds are below the best sample.
Every output block participates in the bound, including regions far from
contact. This is a maximum over the declared finite sampling set.

| Elements per block | Blocks | Loaded static blocks skipped |
| --- | ---: | ---: |
| 6 | 144 | 97.680% |
| 24 | 36 | 93.939% |
| 72 | 12 | 85.795% |

All three layouts match exhaustive H maxima within 2.868e-16 relative.
These counts are work measurements, not speedup factors. Formal timing
includes input packing, bound products, ordering, exact candidates and output.

Optional temporal state is private to each explicit environment identity.
Its bound is tightened by the previous bound plus the response-column bound
on input changes. Previously evaluated blocks retain their actual block
maxima with a roundoff allowance. Partial reset clears only selected states.
Repeated-identical-frame skip counts in the static JSON are labeled as such
and are not used as continuous-trajectory throughput evidence.

Bounds use FP64 inventory and empirical upward floating-point margins. These
controls verify the implementation over the recorded cases and trajectory.
They do not provide a directed-rounding interval proof or an FP32 bounds
implementation.

## Global energy POD control

`EnergyPOD` builds a Galerkin displacement basis from 216 training snapshots:
24 different directions, three radii and three Cartesian force directions.
Online coefficients come from each current force. The held-out phase differs
from the training phase. The full current-load FEM solution supplies only the
independent error measurement.

| Rank | Loaded mean complete-field difference | Worst complete-field difference |
| --- | ---: | ---: |
| 32 | 91.746% | 99.883% |
| 64 | 87.520% | 98.267% |
| 128 | 79.165% | 95.093% |
| 192 | 74.631% | 92.907% |

The old report recorded 73.582% mean difference at rank 192. The rebuilt
training/held-out bank retains the same negative conclusion and similar
scale without fitting seeds or radii to that number. This result concerns
this global energy basis, not every local or hierarchical reduction method.

The initial geometry implementation offset the authored surface inward by
h/2. Those superseded numerical results remain under
`results/initial_outer_offset/response/`, with an explicit diagnostic label.
Only the final authored-middle-surface results above enter reconstruction
comparisons.
