# Native low-shell optimization coverage

This inventory concerns the complete L1 hollow shell, fixed Q10 plus the
declared seven-triangle local contact integration, unchanged final numerical
budgets and actual combined friction variation. It does not claim physical
mesh convergence or the absence of unknown optimizations. Measurements and
rejected attempts are in the [ledger](OPTIMIZATION_LEDGER_20261010.md) and
the [hashed evidence manifest](../evidence/20261010-contact-repair/manifest.json).

| Candidate family | Evaluated route and decision | Evidence and practical limit |
|---|---|---|
| Contact representation | Retain supplied-normal constant-ratio pad with local integration retry. Reject the incomplete surface-normal cone trial. | Original 132-point apex LP is infeasible; local partition makes its exact wrench feasible. All three captured fixtures recover. The alternative cone trial rejects another actual sliding contact and is not selected. |
| Internal fit stopping | Retain 1e-9 normalized gradient, with original final wrench/full residual/stress checks. | 1e-10 still rejects a numerically stagnant but wrench-accurate captured fit. Independent CPU fit remains stricter. Final-v8 432 full-field samples pass. |
| Shared operator and solve route | Retain exact complete-exterior FP64 response, with native full-load correction and factor fallback. | All 162 exterior P2 nodes and three force components, six centrifugal coefficients, complete 810-node displacement and all gauge rows. It uses the complete load space; there is no assumed contact basis or symmetry. |
| Precision and history | Retain FP64 and h=0. Reject measured mixed precision/residual correction and h=4 prediction. | Actual changing-contact measurements are slower after accounting for correction, update and history costs. No precision or residual budget is relaxed to obtain a rate. |
| Matrix tiling and layout | Retain sparse nonzero boundary-column packing and the measured block scheduling. Reject extra environment/matrix tiles and alternate traversal. | Shared response is 9,565,128 bytes and fits the measured device cache. Extra block-512 traversal saves at most 0.0641 ms in the measured large snapshot. External BLAS is outside the production contract; unsupported tensor-core tile APIs are not substituted. |
| Scatter and reduction | Retain face reduction, conservative face bounds, compact bounded face tasks and complete device overflow. | No partial task list contributes a load. Forced overflow and local partition compare with CPU FP64. Shared/further scatter and alternate reduction trials are recorded in the ledger. |
| Current-footprint reuse | Retain Gram plus P2 load moments at large batches; keep original route below B=8192. Reject derived-Gram construction. | Strictly positive affine fits reuse current-step moments. Constrained/refined/overflowing fits integrate completely. Native inventory includes 217 bytes per task capacity. Derived Gram is slower at both tested sizes. |
| Rigid-mode projection | Retain large-batch integrated-wrench reuse with complete nodal overflow fallback. | Actual integrated contact wrench, COM conversion and centrifugal wrench are used; complete residual remains mandatory. Both native interaction ablations lose throughput when this route is disabled. |
| Complete residual scheduling | Retain large-batch coalesced row scheduling. | Same products, order and all gauge rows. Two-seed native interaction ablations prefer it. |
| Failure selection | Retain device conditional correction/factor fallback. Reject the measured extra failure-environment compaction. | Extra compaction gives only about 0.13%; ordinary audited trajectories need no correction/factor fallback. All failure environments remain in acceptance and throughput counts. |
| Peak and full field | Retain cached seven-node gradients and complete corner peak reduction. | Alternate peak traversal and additional scratch clearing have no useful end-to-end gain. Full fields are optional storage, not reduced sampling; final matched output-cost measurements are running. |
| Kernel and graph fusion | Retain the validated serial full stress pipeline and existing native rigid graphs. Reject elastic parallel graph and extra stress fastcache. | Parallel trial raises an illegal address during actual warmup. Serial elastic-only graph gains about 0.14%; whole stress pipeline is the selected route. Stress fastcache slows the two actual small-batch trajectories by about 0.59%/0.06%. Final profiler additionally separates contact postprocessing and rigid integration dispatch. |
| Configuration dispatch | Retain resident Quadrants phase/target/grip/limit/radius calculation through ordinary public setters. | Independent CPU configuration oracle, full-field trajectory oracles and matched small/large policy composition. Scalar dtype and FP32 residual multiplication match the original. |
| Control and reset indices | Retain ordinary range selectors for Panda joints and bounded resident reset selectors. | Two-seed B=1024 combinations improve about 9.4%/9.8%. Device selectors preserve initial and untouched rigid state, max/full-field invalidation and phase. LRU capacity is at most B int64 entries; hit/miss/eviction maintenance is timed. Large native composition remains in measurement. |
| State restoration and invalidation | Retain saved public Scene state, per-link subscribers and duplicate-notice suppression. Reject alternative native copy kernels. | Original/global subscriber fails the independent link-isolation regression. Duplicate geometry+dynamics clears cost about 0.9%/1.1% at B=1024. Replacing contiguous Torch state copies is slower in both actual rollouts and reset microbenchmarks. |
| Observation dispatch | Keep existing getters. Fused FP64 and compact-argument prototypes are evaluated. | All 26 columns match exactly over two 2400-step trajectories. The full-state prototype is slightly slower on both small-batch seeds; compact arguments save isolated submission cost but give opposite end-to-end results across seeds. Large full-state gain is only about 0.055%. |
| Policy inference graph | Keep eager FP32 as the current selection; retain reproducible policy-only graph probe. | Small gains 0.24%/0.52%, large gain 0.12%; two are within repeat variation. Policy graph is Torch inference only. No stress backend or RL training rate is inferred. |
| Batch scaling and tails | Final same-source 1024/2048/32768/49152/61440 two-seed four-scope sweep is running. | Largest existing valid workload is 61440. The original rigid flattened-buffer i32 limit rejects 65536; it is not an OOM measurement. All attempted/valid counts, reset costs, whole-card memory and per-environment work remain disclosed. |

The current known bottlenecks are native rigid work, complete exterior inverse
application and contact integration/scatter. Final same-source profiles,
repeated batch scaling, max/full output cost and every-environment completed
episode audits remain required before declaring a throughput plateau. An
individual environment's GPU latency is not separable from these shared SIMT
kernels; the audits disclose work-count tails and the profiler discloses
whole-batch intervals rather than claiming a per-environment latency.
