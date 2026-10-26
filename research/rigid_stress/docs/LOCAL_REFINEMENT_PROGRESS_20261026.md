# Native local-refinement atlas execution ledger

Active source specification: `4bc4ada25`,
[review and goal](ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md),
[CPU guide](ATLAS_CPU_REFERENCE_GUIDE.md). Primary algorithm reference:
`atlas_study/cpu_repro`, reconstructed in `a10c5abe3`.
The previous complete-response partition remains a scoped historical experiment.

## Required decisions

| Candidate family | Shell | Volume |
|---|---|---|
| Genuine coarse plus local feature spaces | Open: native construction and independent controls | Open: native construction and independent controls |
| Per-feature contact aggregation and multiple RHS | Open | Open |
| Common displacement then one full-domain recovery | Open | Open |

Completion requires actual smaller independent coordinates/operators, numerical
and spatial qualification, targeted tuning, paired module and complete Franka
max/full timing, and setup/shared/per-environment/peak memory for each backend.
Preserve the earlier H/FP32/screen/history results without unchanged reruns.

## Budgets declared before selection

- Preserve original pressure/radius/friction, force/moment checks and exact-mode
  numerical acceptance. Feature solves use the same relative `1e-8` and absolute
  `1e-11` tolerance on the unscaled reduced operator. The complete mother
  residual is a separate spatial-error diagnostic for approximate solves.
- Verify common geometry, gauge, rigid-mode reproduction, projected relief and
  virtual work to `1e-10` relative, with dimensional absolute floors stated in
  each control. An all-bubbles space must recover the exact mother solution.
- Every admitted approximate case must have combined complete tensor error at
  most 10% and whole-object sampled maximum error at most 5%; target mean field
  error at most 5%. Use area-weighted shell and volume-weighted volume tensor
  Frobenius norms, including symmetric shear multiplicities. Check shell
  membrane/bending/transverse resultants separately and record peak positions.
- Cancellation cases retain the total-field relative metric and additionally
  report absolute and source-norm-normalized errors. Single-source agreement
  cannot qualify the independently solved source sum.
- Numerical/spatial budgets are distinct from runtime admission. Report every
  fallback and its reason/cost. No unconditional mother correction may replace
  the actual reduced solve being evaluated.

## Construction and evidence plan

Use `T_i=[P,B_i]`, project the same mother stiffness and relieved source load,
solve smaller `K_i`, and merge lifted displacement before nonlinear criteria.
Volume uses its P1 vertex subspace plus local P2 edge enrichment, including
through-thickness nodes. Shell retains its actual geometry/directors and uses
a geometry-aware translation/rotation prolongation. Retain mother pin nodes
in the shared coarse space; no feature chooses independent mass or gauge.
Local bubbles must have complete element support inside the covered region.
Provide shared fixed enrichment for an independently checked remote neck.

Compare coarse-only, individual feature, source-wise grouped atlas, joint
enrichment union and mother on identical native loads and output samples.
Record dimensions/rank, operator nonzeros, factor fill, core/halo/transition,
active groups, preparation/storage and fallback rates. Check rigid modes,
virtual work, common relief, spin exactly once, both material sides, friction,
distant/cancelling loads, junction continuity, release/reset/checkpoint and
environment isolation. Include curved egg and true nonconvex neck controls.

Sweep a bounded range of region counts/core sizes/transition widths, then
profile projection/grouping, reduced solves, lift, stress output and gates.
Measure complete evolving Franka max/full, small and useful large batches,
heterogeneous conditions, matched accuracy/cost and memory. A tuned negative
is acceptable only after evaluating this actual local-refinement mechanism.

All numerical preparation/runtime remains in Quadrants. CPU scientific code
is an independent oracle executed through Slurm `rtx-mid`, image `1_26`.
Runtime artifacts and stdout/stderr use
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261026-local-refinement-atlas/`.
The start receipt preserves twelve original unfinished files and the prior
frozen sources. Higher-resolution continuum sharp-edge peaks remain outside
this integration/throughput phase.

## First native feature-space controls

Source `7cad6e363` and the corrected captured worker pass both scheduled GPU
controls (`space-controls-v1c`, job 751053). Shell coarse/local/all-fine
coordinate counts are 252/402/1452; volume counts are 378/474/2430, before
removing the common six pins. Independent sparse controls pass cardinal/rank,
rigid-mode reproduction, Galerkin K/RHS, virtual work and original unscaled
residual checks. The all-bubbles solution/complete fields recover the mother.
These prove genuine coordinate reduction; moving-contact combined spatial
qualification and complete-scene cost decisions remain open.

The first attempt exposed an empty scalar boundary-buffer allocation in the
new shared state factory; the second exposed a missing explicit activity
argument in the control worker. Both failed records remain retained, together
with preflight rejections before any Slurm submission. Neither is counted as
a passed numerical control. Frozen experiment trees are not hot-edited.
