# Proposed GPU implementation and optimization plan

> Superseded architecture, 2026-10-10: follow
> [NATIVE_QUADRANTS_PLAN.md](NATIVE_QUADRANTS_PLAN.md) and `../GOAL.md`.
> All production numerical work now uses Quadrants inside the rigid solver.
> The external-library backends and tuning order below are historical reference
> material; they do not define the new implementation or require fine-mesh runs.

This is a design to implement and measure, not a measured GPU result. The
CUDA-capable Torch code in `gpu_prototype.py` was checked on CPU only. Its dense
factor is deliberately restricted to small debug meshes and is not scalable.

## First correct device path

One operator group owns immutable K/M/MR/G, a shared factor, free/pin indices,
surface geometry/connectivity and barycentric gradients. Per-environment state
owns current/previous u, optional Q/KQ, small Gram, slot order/mask and output
maximum. Use exact built scene dimensions plus explicitly chosen history/tile
capacity; never duplicate a factor B times.

The intended graph is:

```text
rigid contact solve (all envs)
  -> masked contacts and authored-frame transforms
  -> finite traction / wrench-consistent sparse scatter
  -> body loads + centrifugal inertia + 6x6 relief
  -> prediction / cached-history projection / fresh full residual
  -> select failed envs / pack free RHS
  -> shared sparse factor solve / scatter / final residual
  -> update only corrected histories
  -> all-element P2 corner VM + max reduction
  -> [B] device stress observation
```

The first baseline skips prediction/history and solves every active RHS. It
must be correct before temporal approximation is enabled. No per-env Python
loop, `.cpu()`, `.numpy()` or tensor scalar `.item()` belongs in the GPU hot
path. CPU preprocessing and offline correctness replay are allowed.

## Shared sparse factor backend

Preferred options to compare, not assumptions of equal speed:

1. **cuDSS SPD multi-RHS**: analyze/factor K_ff once, solve many dense RHS
   columns repeatedly. Use one matrix with many RHS, not B independent copies
   of the same matrix. Confirm the installed library's supported types,
   descriptor/leading-dimension rules, CUDA stream and graph restrictions.
2. **Offline CPU factor + cuSPARSE SpSM**: export compatible sparse triangular
   factors and explicit permutations, upload once, run analyzed forward/back
   multi-RHS solves. CPU SuperLU LU is a reproducible initial export, although
   a SPD Cholesky can reduce factor storage/work. Verify permutation identities
   numerically; do not guess `perm_r/perm_c` gather direction.
3. **Sparse PCG with a shared reusable preconditioner**: compare if sparse
   triangular solves have poor parallelism. Use true residual stopping and
   per-env active convergence masks. This is not automatically faster or a
   maximum-stress certificate. Include mixed precision refinement/fallback.

Cache SpSM analysis/workspace for the actual repeated shapes. Changing the
number of compact RHS may need different descriptors/analysis depending on
backend; measure that overhead. Distinguish shared-A/multi-RHS from library
"uniform batch" of different matrices. Their restrictions differ.

Never form full K^-1 or a dense surface-to-stress response matrix for the
arbitrary-contact production path. Dense debug Cholesky in `gpu_prototype.py`
is only an oracle for small ndof, and errors above its explicit size cap.

Primary API references (consult installed-version docs):

- [cuSPARSE SpSM](https://docs.nvidia.com/cuda/cusparse/index.html#cusparsespsm)
- [cuDSS functions](https://docs.nvidia.com/cuda/cudss/functions.html)
- [cuDSS types and batch restrictions](https://docs.nvidia.com/cuda/cudss/types.html)
- [NVIDIA sparse library examples](https://github.com/NVIDIA/CUDALibrarySamples)

## Layout, chunking and VRAM

Benchmark both environment-contiguous AoSoA and column-major free-RHS storage.
Library solves often prefer [nfree,nrhs] with column-major leading dimension;
contact and peak kernels often benefit from neighbouring lanes handling envs
on a shared node/tet. Pack/reorder once where needed and include the cost.

Memory model, with s bytes per scalar, n DOFs, h history and B environments:

```text
shared factor/K/M/geometric buffers                  O(nnz_factor + nnz_K)
u / before / rhs / residual / correction scratch     O(B n)
Q and KQ                                            2 B n h s
Gram + slot masks                                   O(B h^2)
```

For n=9630, h=8, B=4096, history alone is 2.52 GB at FP32 or 5.05 GB at FP64
(decimal units), excluding every other allocation. Peak scan must not create
[B,tet,corner,10,3] gradients/displacements for all tets. Tile or fuse.
Large converged meshes can dominate VRAM; report maximum safe B rather than
extrapolating the coarse model's capacity to fine meshes.

Sweep chunk sizes and B independently. It can be faster to process a large
resident environment set in shared-factor solve tiles. Overlapping tiles
requires correct stream/event ownership of contacts/history and measurements.
Do not double-count overlap by summing isolated stage throughput.

## Contact scatter on fixed geometry

CPU cKDTree is not a GPU hot-path algorithm. Preprocess a spatial grid or BVH
over fixed recovery-surface faces/quadrature, with sparse adjacency. For each
current contact/footprint, traverse only relevant geometry, integrate P2
tractions, and scatter nodal contributions. Current centre, area and direction
are runtime inputs. A Gaussian demo is not the only supported traction law.

Compare atomic scatter against contact-local reduction / sorted segmented
reduction. High contact count and concurrent loads on the same nodes may make
atomics contention expensive. Deterministic and fast scatter modes must pass
the same published error profile. Detect contact/patch buffer overflow; never
drop loads to preserve FPS. Measure mapping and search separately.

Keep actual force/torque and friction diagnostics. Pruning adequate for rigid
net wrench is not necessarily adequate for local maximum stress; compare
pruned/unpruned configurations with a fixed physical footprint law.

## Full-domain peak kernel

Implement the `general_peak.py` on-the-fly algorithm in CUDA/Triton/Quadrants.
Reuse tetrahedral vertex terms across four corners and calculate VM^2 directly.
Return one maximum (optional argmax for QA) per environment; do not materialize
six-component stresses for the whole mesh.

Compare a warp/tet with env-lane tiling, a block/env with tet lanes, and a tiled
two-stage reduction. Gather 10 P2 nodes once, subtract the local origin,
reuse gradients, reduce block maxima, then reduce per-env. Stress hydrostatic
lambda terms cancel only for isotropic VM; make this restriction explicit.
Shared geometry does not imply shared displacements. Benchmark recomputing
gradients from 4x3 barycentric values vs caching 4x10x3 corner operators.

Temporal **certified** group exclusion is an optional later optimization:
precompute bounds on a group stress operator, use actual displacement change
and conservative margins to upper-bound its current maximum, and compare
with a current evaluated lower bound. Keep an unconditional full scan fallback.
Do not port the old small fixed-contact response-basis bounds as if they applied
to arbitrary contact. Bounds must hold for the full displacement space and
numerical errors, otherwise label them approximate and validate separately.

## Per-env history and failed-env selection

Start with h in {0,4,8}; h=16 only if measured benefit exceeds storage/bandwidth.
Perform projection and small Gram solves batched. Only append for corrected
envs; circular slots avoid shifting columns. Cache/update affected Gram rows,
twice orthogonalize in FP32/FP64 as selected, test rank and refresh KQ/Gram
when numerical drift is detected. Reset masks must clear all per-env state.

Avoid Python `torch.nonzero` as the final selection implementation if its
dynamic output synchronization dominates. Choices to measure:

- device selection + count, with one explicitly timed host count transfer;
- fixed/bucketed RHS capacity with masked padding and graphs;
- tiled solves selected by device masks, including the extra dummy work;
- dense solve at high failure density.

The exact selection count, workspace allocation and graph compatibility are
backend constraints, not facts solved by a masked tensor. Maintain one mapping
from compact column to env ID through pack/solve/scatter/history. Include
selection, gathering, scattering and descriptor changes in timings.

Direct-vs-temporal scheduling uses past failure fraction and measured stage
costs. Estimate predictor+projection+residual+history overhead against saved
solve work. History may lose at strict tolerance, abrupt contacts or small B.
Use hysteresis to prevent oscillating configuration. No future reference
trajectory or truth peaks may choose the runtime path.

## Precision and Tensor Cores

Offline FP64 assembly is allowed. Compare runtime FP64, FP32, diagonal-scaled
FP32, FP32 with FP64 residual/refinement, and FP32 history with selective FP64
small Gram/residual arithmetic. Scale K via S=diag(K_ff)^(-1/2): solve
`(S K_ff S) y=S b`, then `u=S y`. Gauge and load balance must remain consistent.

Refinement uses the true higher-precision residual from the original equation
and an FP32 factor for the correction. Stop on the declared profile; fallback
if refinement stagnates, rather than claiming all low-precision solves pass.
Full stress comparisons are also required; relative residual alone is insufficient.

TF32 is not a storage type or a cuSPARSE SpSM mode. Consider it only for dense
tiles/GEMM, history projections or an approximate preconditioner after testing.
TF32 product precision can damage thin-shell conditioning/history small solves;
never silently enable it for an accuracy oracle. Any Tensor Core path must
show actual kernel utilization, accuracy and net throughput gain. Quantizing
inputs on CPU is not a hardware TF32 test.

See the official [CUDA floating-point guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
and installed CUDA library datatype documentation. CPU FP32 evidence is
summarized in `CPU_EVIDENCE.md`; there is no measured TF32 result here.

## Graphs, integration and tuning order

Remove Python/env loops and host transfers first; then test CUDA Graph capture
for shape-stable paths. Analysis/factorization stay outside capture. Use
supported stream-ordered memory/workspace behavior for library calls; verify
capture support for the installed versions. Dynamic failed count can conflict
with graph replay; benchmark buckets/padding rather than assuming a graph
automatically saves cost.

Tune in order: correct device direct solve -> fused peak -> contact mapping ->
multi-RHS/chunks -> FP32/refinement -> temporal prediction/compaction/history ->
graphs/layout -> adaptive path -> optional hierarchical output bounds.
Select configurations on calibration seeds, freeze them, and validate/time
held-out seeds. Keep a GPU direct baseline at matching precision/accuracy.

RTX 6000 Ada 48 GB and RTX PRO 6000 Blackwell are different devices. Discover
and record the actual model, compute capability and VRAM. Do not silently apply
an Ada FP64 estimate to Blackwell or treat theoretical FLOPS as measured FPS.
