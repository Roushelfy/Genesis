# Physical-pad Shell load/response controls, 2026-10-19

All six250um Shell static controls at0.1/0.3/1N now pass their unchanged
static-equation gates; all three own-pad solves also pass pressure/boundary
admission. The independent summary passes151 checks and rehashes56 receipts.
Nine rigid-mass-projection and eight signed-volume/tensor controls pass.
The original pre-solver failures remain in the earlier packet and this record.

## The controlled equations and domains

C is the admitted continuous250um Shell history, S its self-consistent static
pad solve, R its replay of the independent solid's exact centroid point loads,
and F that same250um two-wall P2 solid source solve. C/S/R share the declared
Shell mesh, material, physical pad,12 coarse patches, PCG1e-4, velocity1e-10,
8000-product cap and the complete material domain. Static velocity clearing is
an explicitly separate relaxation procedure. Full fields are compared with
q9/q25 angular sampling and depth4, preserving all volume/area contributions.

F is the admitted coarse centroid-pressure source, not the final qualified
125um reference. Its pressure is held fixed for R; the source point forces and
force/couple transfer remain unchanged. The existing
[normal rankings](NORMAL_PAD_QUALIFICATION_20261019.md) and
[dimensional errors](NORMAL_ABSOLUTE_ERRORS_20261019.md) continue to use the
qualified fine reference and its observed uncertainty. These controls identify
response/load differences within a fixed configuration; they do not replace
those physical-error estimates.

The explicit free-body equation is elastic_force + mapped_contact_force -
rigid_mass_correction =0. The optional --inertia-relief path uses the same
six-mode mass projection as the existing free solid/static replay; it saves
mapped contact and body correction arrays separately. Every iteration rejects
a correction norm above1e-8 of mapped-load norm and retains the original
1e-12 zero-wrench and1e-7 static-residual gates. Actual correction ratios are
at most2.132e-11 for these replays and2.374e-16 for the own-pad solves. No point
force is clipped or changed to manufacture equilibrium.

The0.3N before/after controls change complete field metrics by at most
1.004e-10 relatively and raw peaks by at most2.015e-11, below the predeclared
1e-8 numerical-control target. The new path fixes the0.1/1N free-static
solvability issue without loosening the original gates or claiming dynamics
from a velocity-cleared solve.

## Measured response separation

C-S complete tangent differences are3.798e-8,4.263e-9 and3.659e-9 relatively.
All five field and both raw-peak continuous/static targets pass. Thus static
relaxation and the matched pad map are measured controls for this normal case.
S-R tangent changes are0.013895%,0.016828% and0.021201%. Different pressure
profiles are a measured physical effect, with no imposed equivalence threshold.

The direct C/R/F decomposition below uses F's full physical volume. Percentages
are all normalized by the same F tangent norm; RMS values are Pa.

| Load N | C-R % | R-F % | C-F % | C-R RMS | R-F RMS | C-F RMS | Cross correlation |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.013958 | 25.966337 | 25.969255 | 1.27542231 | 2372.70962 | 2372.9762 | 0.208753239 |
| 0.3 | 0.016906 | 25.191181 | 25.194818 | 4.51764754 | 6731.51911 | 6732.49092 | 0.21479482 |
| 1 | 0.021313 | 23.985357 | 23.990140 | 18.2669787 | 20557.8447 | 20561.9448 | 0.224028955 |

At this fixed250um configuration, the same-load response difference dominates
the measured full response contrast. R-F includes Shell kinematics, spatial
elastic discretization and force/couple transfer. The measured
[500-to250um Shell field changes](FRICTION_RESOLUTION_MEMORY_20261019.md)
remain large, so these data do not isolate a continuum thin-shell-theory error
or establish mesh-independent Shell accuracy.

## Complete signed tensor checks

For C/S/R and C/R/F, the code integrates each tensor difference and signed
cross product directly on one unioned partition. It verifies ||A+B||² =
||A||² + ||B||² +2<A,B>, without subtracting scalar error norms. All15 shared
C-R absolute metric checks agree across the two triples. Twelve independent
full-array membrane/bending checks pass, including the signed cross terms.
The JSON retains all dimensional field/resultant errors and six decompositions.

An explicit solid volume carrier now allows three selected Shell tensors to
use the same declared P2 volume. Its tetrahedral breaks join the integration
partition; the selected third tensor still supplies the stress norm. The
external-carrier analytic control deliberately changes carrier stress and
verifies that tensor norms remain unchanged. The original last-field guard
failures are retained; the successful replacement adds the actual solid carrier
rather than discarding the full-volume requirement.

## Reproduction and evidence

The [packet](../evidence/20261019-qualified-ranking/shell-pad-load-separation-v1/README.md)
retains175 verified payloads totaling1,248,122 bytes: terminal parents,
q9/q25 results, exact sources/task files, failed guard records and the151-check
summary. Full frame/column arrays remain on data with exact hashes. Reproduce
with its tasks through the documented Slurm/container runner. The aggregate is:

```bash
python -m research.rigid_stress.qipc_comparison.audit_shell_pad_controls \
  --data-root "$RANKING_DATA" --output "$NEW_DATA_AUDIT"
```

These controls advance the required load/response separation. Native physical
reconstruction controls, qualified final friction ranks, production scaling and
final figures remain subject to the full completion checklist.
