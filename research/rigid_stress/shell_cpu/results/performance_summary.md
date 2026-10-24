# CPU shell-recovery performance experiments

These are **single-thread CPU research-kernel** measurements, not Genesis
scene-step timings, GPU predictions or RL transitions.  Complete fields here
mean top/bottom **in-plane shell stresses** at four in-plane points per quad.
They do not include a three-dimensional near-contact stress field.

Hardware: AMD EPYC 9V74, NumPy/SciPy OpenBLAS restricted to one thread.
The broad scan used 336, 588 and 1,308 DOFs, batches 1/8/32/128, five repeats.
It overlapped other CPU validation jobs, so `performance.json` keeps its full
ranges but the narrower, interleaved confirmation is the primary timing source.

## Confirmed 588-DOF egg model

`performance_confirmation.json` uses seven interleaved repeats, reversing the
method order on alternating repeats, after other long CPU tasks finished.
Source hashes and an unchanged-during-run guard are recorded.

| Operation, ms per batch | B1 median | B128 median | B128 min–max |
|---|---:|---:|---:|
| Reused sparse LU + full local field | 0.1888 | 10.3924 | 10.0636–11.9401 |
| Per-column reused LU + full local field | 0.2192 | 23.1215 | 22.4319–24.9056 |
| Exact compact surface response + full local field | 0.0986 | 4.1282 | 3.9595–4.3124 |
| Scalar checked-witness fast transfer | 1.2997 | 153.3155 | 151.4013–166.7643 |
| Batched checked-witness fast transfer | 0.7271 | 9.0298 | 8.5648–9.8070 |
| Batched transfer + reused LU + full local field | 0.9725 | 19.8628 | 18.8516–21.7030 |
| Batched transfer + compact response + full local field | 0.8454 | 12.8893 | 12.5834–16.7051 |
| Batched transfer + compact response + VM peak | 0.8845 | 13.6849 | 12.9957–18.6378 |
| Reused LU + full body-frame symmetric tensors | 0.3178 | 11.4787 | 11.1242–13.6919 |
| Compact response + full body-frame symmetric tensors | 0.1499 | 4.8681 | 4.4239–5.0490 |

The max-only CPU implementation does not show a confirmed speed gain over
full local fields: the stress calculation still occurs, and the maximum adds
a reduction.  The tiled maximum avoids allocating the complete output, but
Python tile-loop overhead and the dense response storage remain.  No speedup
is assumed merely because a maximum is requested.

## Actual algorithm experiments

1. The fixed shell stiffness is factorized once.  The broad scan explicitly
   measures the much more expensive refactor-per-load baseline.  Multiple RHS
   are faster than a Python loop over scalar RHS.
2. The exact operator `H = S K^-1 P` can be precomputed, where `P` includes
   the same rigid-mode inertia relief as the direct solve.  An on-demand column
   cache works for arbitrary changing load locations; no grasp symmetry is
   assumed.  It helps sparse small batches, but broad active-column unions and
   dense operator traffic can remove the benefit at larger batches.  The
   negative/neutral cases remain in `performance.json`.
3. Surface point forces have a known couple `(+/- h/2) director x force`.
   Folding that exact mapping into the response gives `H_surface` with `3N`
   force columns per physical side instead of `6N` general force/couple
   columns.  It preserves tangential-load eccentricity.  Mixed inner/outer
   contacts use both side operators; arbitrary additional couples still need
   the general operator.  For 336/588/1,308 DOFs, one-side response storage is
   1.74/5.42/27.12 MB; broad-scan construction costs are about
   13.0/26.7/253.9 ms.  These setup costs are reported separately.
4. The complete-disk transfer batches the **same** checked current witness,
   projected-Hertz pressure law and embedded 16/36-point rule.  Missing,
   clipped, invalid or unresolved cases retain the conservative scalar path.
   It reduces B128 transfer time from 153.3 to 9.03 ms in the confirmation.
   This is an execution optimization, not a changed pressure model.

## Numerical checks and coverage

- All broad-scan recovery and fast-transfer comparisons pass their existing
  `1e-8` accuracy-based threshold.  At B128 in the confirmation, compact
  response versus direct LU field difference is `8.55e-12`, batched versus
  scalar checked transfer load difference is `2.38e-16`, and compact VM peak
  difference is `6.88e-13`.
- `batched_transfer_checks.json` explicitly covers contained disks, crossed
  edges, a shared vertex, complete unload, the inner face and a stale witness:
  two use the fast path and four use the original fallback.  The combined
  load difference is `1.09e-16`; unload is exactly zero.
- The timed egg bank changes contact faces, interior positions, normal load
  and shear direction.  Its disks remain inside their current witnessed
  faces; it is not a clipped-contact throughput claim.  Curvature is supplied
  from immutable geometry.  Radius computation is measured separately.
- Taylor, embedded cubature, clipped quadrature and unhinted geometry-query
  measurements are retained as separately labelled candidates.  The compact
  response and batched execution do not alter source loads or shell stiffness.

Reproduce the broad scan with `python -m shell_cpu.performance`, and the bounded
confirmation with `python shell_cpu/results/performance_confirm.py`, under
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` from the project root.
