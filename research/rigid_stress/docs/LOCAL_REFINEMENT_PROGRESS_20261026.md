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

## First moving-contact spatial pilot

Frozen source `1eaa439d0`, campaign `atlas-pilot-v3` (job 751094), completes
all 140 frames in both backends with the original numerical/load gates.
Each backend has 730 loaded contacts and zero coverage fallbacks. Volume
maximum weighted field/peak errors are 4.9939%/0.0981%, within the declared
budgets. Shell maximum field/peak errors are 22.4337%/5.7135%; its separate
membrane/bending/shear errors also reject this configuration. Shell feature
sizes are 1110/1080/966 scalar coordinates versus the 1452-coordinate mother.
These are spatial results, not a reason to relax reduced solve tolerances.
Full trajectories, lifecycle, geometry controls and timing remain open.

The first runtime integration attempt (`atlas-pilot-v2`, job 751065) failed
during compilation because a scatter template was flattened as a dataclass.
The opaque scatter container in `1eaa439d0` resolves that interface failure;
both records are retained.

The next bounded tuning varies shell core/halo and coarse resolution, and
tests sparse boundary responses of actual smaller feature operators. Coverage
selection now fades each feature conservatively before its complete pressure
footprint reaches an unsupported face; the remaining weight goes to the exact
mother. This preserves load and avoids a hard spatial-model switch. Its
continuity and fallback costs require the new geometry controls.

## Bounded tuning and complete trajectories

Source `a17fc82b7`, `atlas-tuning-v4` (751224), passes moving-contact numerical
and grouped-load checks. Increasing shell core from 1.0 to 1.2 or adding one
halo still fails spatial accuracy. Core 1.4 passes 140 frames (3.7055% tensor,
0.3172% peak), while retaining 1236--1428 scalar coordinates per feature
against the 1452-coordinate mother. Level-2 core 1.0 reduces features to
3708--4152 versus 5772 mother coordinates and passes tensor/peak budgets,
but fails the separate transverse-resultant budget (13.5694%).
The coarse-only control is strongly inaccurate. Volume response/three-region
fields match the factor pilot and pass 140 frames; six narrower regions fail
(20.5437% tensor). These are targeted local-space trials, not repeated old H
or history campaigns.

`atlas-full-v4` (751239) demonstrates why the short pilot cannot qualify the
method: shell core 1.4 fails the complete 600-frame trajectory at 12.0542%
tensor and 5.1857% peak; volume core 1.0 fails at 17.3490% tensor. Shell
resultants also reject the partial-reset and 64-condition trajectories.
All frames and rejected configurations remain retained. Further bounded
core/transition tuning and matched module/scene timing are running.

The first curved-egg and true U-neck controls (`atlas-geometry-v4`, 751226)
pass source numerical/load, permutation, release/restore, grouped versus
separate-source plus once-only spin, and decreasing-perturbation continuity
checks in both backends. Their spatial errors reject these narrow libraries,
including the fixed-neck enrichment. The volume junction/partial-footprint
probe limitations are recorded separately. A shell Gauss-position ordering
error in the research witness and dimensional wrench-reporting thresholds
were found after this run; corrected native-directed probes and output
positions are captured for the next control. Prior raw results are preserved.

The independent shell keyframe audit (`joint-audit-v4`, 751238) passes complete
mother reconstruction, projected load/operator checks and saved source-sum
reconstruction. Joint-union and coarse controls remain explicitly distinct
from the independent source sum. Optional matched native keyframe timing and
an actual full-recovery CUDA profiler are now available. The first profiler
run exposed a research-only weight-axis readback error; the corrected worker
retains the original runtime gates and the failed record.

## Accuracy frontier and profiling-directed follow-up

`atlas-wide-v4` (751260) obtains complete 600-frame accuracy at shell core 1.6
and volume core 1.4. However, their block dimensions are respectively
484/484/452 versus mother 484, and 810/810/736 versus mother 810: two of
three features have reached the complete mother dimension. This is a
degenerating accuracy/cost frontier, not a claim that every feature remains
smaller. Narrower transition does not repair the rejected configurations.
Matched timing of this accuracy corner, heterogeneous and lifecycle checks
is separate from timing of the earlier spatially rejected libraries.

Actual-recovery traces (`profile-v5`, 751280) pass original runtime gates.
Shell/1024 selection costs about 1.86 ms in intrusive device task sums,
including 1.80 ms in the serial per-contact footprint-face scan. Shared lift
is only about 0.075 ms, and volume/1024 lift about 0.116 ms; optimizing shared
coarse lifting first is not supported by this profile. These diagnostic
device sums are distinct from uninstrumented module/scene latency.

A bounded shell coarse-space alternative adds a director-normal quadratic
bending correction: midpoint coefficient 1/8 and triangle-centroid coefficient
1/18 on rotation differences crossed with parent edges. It preserves cardinal
rows, constant rotations and the existing curved rigid-motion correction.
Planar analytic kinematics and the all-bubbles mother solution must pass
before interpreting curved-egg spatial results. The independent oracle uses
parent pairs, distinct from native per-parent coefficient assembly.

The planar bending proof (`bending-proof-v6`, 751300) passes horizontal and
inclined patches, including rigid/quadratic kinematics, virtual work and the
all-bubbles solution. Four full curved-egg trajectories (`bending-full-v6`,
751311) all reject spatial accuracy. Level-1 core 1.0/core 1.2/six-region
maximum tensor errors are 26.3744%/18.2499%/29.4144%; level 2 has 7.3138%
tensor and 0.8028% peak but 19.6479% transverse-resultant error. This bounded
mechanics-based alternative does not rescue curved-shell qualification.

The wider accuracy corner passes both lifecycle and 64-condition 600-frame
controls (`qualified-full-v4`, 751303), with the documented two-full-size
features in each library. All 32 matched module/scene outputs and 128 repeats
in `qualified-paired-v4` (751291) pass runtime acceptance and remain slower
than the optimized mother. Their spatial/dimension qualifications are bound
to these same configurations.

Corrected narrow-library controls (`remaining-controls-v6`, 751316) now
observe real partial footprints in both backends, complete grouped-load
preservation and continuous mixed mother weights. Shell's selected footprint
has mother share 0.02739; volume's intersecting footprint fully falls back,
while other real frames have mixed shares 0.32984--0.37647. Fixed-neck shell
bending improves the peak but still fails full-field/resultant budgets.

Source `a971d61d1` changes only selector scheduling: a CUDA warp shares each
contact's complete unsupported-face scan and reduces FP64 clearances. The
same-source scalar comparison is retained; numerical/whole-field and actual
runtime validation remain pending. A separate four-path component replay
isolates source aggregation at fixed one-stress output and merged recovery
at fixed grouped solves. Its optional signed raw-load cancellation control
is an algebra diagnostic, not physical contact qualification.
