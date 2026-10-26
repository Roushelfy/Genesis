# CPU source reference for the local-refinement atlas

The earlier CPU code is already present in
[`atlas_study/`](../atlas_study/README.md). This guide links the actual source
rather than duplicating a second implementation. It supports the
[active review/goal](ATLAS_LOCAL_REFINEMENT_REVIEW_AND_GOAL.md).

## Published reference implementations

| Source | What is implemented | Migration boundary |
| --- | --- | --- |
| [volume/model.py](../atlas_study/volume/model.py), make_mesh and BoxModel | Conforming local tetrahedral bisection, separate P2 stiffness/factor for each feature, real finite disk loads | Bottom-clamped box; independent meshes rather than a common-mother embedding |
| [volume/study.py](../atlas_study/volume/study.py) | Coarse, feature, joint-refined and finer models; moving pressure/friction, common physical stress probes and tensor superposition | CPU reference with supplied radius; no native Franka or automatic-pressure performance claim |
| [shell/plate_atlas.py](../atlas_study/shell/plate_atlas.py), refined_knots and Plate | Actual different Q4 shell meshes/DOFs/operators, eccentric friction and switching | Clamped plate; tensor-product refinement also creates fine strips |
| [shell/egg_supplement.py](../atlas_study/shell/egg_supplement.py) | Actual different curved Q4 egg meshes and common material probes | Balanced normal pairs, no inertia relief, projected-Hertz adapter and retained accuracy failures |
| [nonconvex/study.py](../atlas_study/nonconvex/study.py) | Nonconvex neck, contact-only versus fixed-neck refinement, remote hot-region checks | 2D plane stress; not a 3D nonconvex shell qualification |
| [freebody_control.py](../atlas_study/freebody_control.py) | Consistent mass, inertia relief and load superposition on one P2 model | Does not establish common relief across independently meshed features |
| [performance/condensation_check.py](../atlas_study/performance/condensation_check.py) | Exact full-interface static condensation and coupled recovery | Clamped body with its stated load assumptions; not active-port or cross-feature approximation |

The actual refinement loop in volume/model.py includes:

```python
goal = np.minimum(.009, hmin + .65 * np.maximum(distances - core, 0.))
marked = np.flatnonzero(h > 1.18 * goal)
mesh = mesh.refined(marked)
```

Offline anchors affect the mesh. Online contacts keep their real physical
position and full finite footprint. Each resulting BoxModel assembles and
factors its own K. This is the relevant distinction from selecting different
columns of one unchanged complete response.

Read [the CPU report](../atlas_study/REPORT.md),
[the Chinese theory](../atlas_study/THEORY_ZH.md) and the suite READMEs before
porting. Keep their negative results. Independent meshes differ in spatial
error and can differ in geometry/mass; common physical output points alone
do not make them one common Galerkin model.

## Reproduction entrypoints

Dependencies are NumPy, SciPy, Matplotlib, scikit-fem and threadpoolctl.
These commands are optional research runs, not a requirement to repeat old
expensive studies before implementing a small native path. Use a fresh
checkout/output directory under the current project-data policy.

```bash
cd research/rigid_stress
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ATLAS_REF_RUN=/absolute/path/under/project-data/new-atlas-reference
python -m atlas_study.volume.study --fine-h .0009 --out "$ATLAS_REF_RUN/volume"
python -m atlas_study.shell.plate_atlas --quick --output "$ATLAS_REF_RUN/shell"
python -m atlas_study.shell.egg_supplement --output "$ATLAS_REF_RUN/egg"
```

The listed three entrypoints have argument parsers. Other runners, including
nonconvex.study and some algebra/condensation controls, execute default work
without a safe output/parser interface; invoking --help can run the study.
Use a scratch worktree or introduce safe explicit outputs before running
them. Preserve committed historical evidence. Inspect factory functions to
construct smaller targeted controls when appropriate.

The shell suite imports the enclosing checkout's shell_cpu implementation.
Verify import paths and use private compile caches for native worktrees as
specified in root engineering instructions.

## Later common-mother CPU construction: provenance and algorithm

A later separately reported CPU experiment used a fixed 5,196-DOF curved
shell mother, a shared coarse interpolation and local fine-node enrichment.
Its unpushed cpu_next source was removed by temporary-workspace maintenance;
the retained report is available to the user, but that unpublished source
archive was not recovered in this task. The published modules above are
older, different experiments. They must not be relabeled the missing version
or inherit its measured errors/timings.

The transferable construction from that report is:

**New executable reconstruction:** [cpu_repro/](../atlas_study/cpu_repro/README.md)
now implements this common-mother construction on the 5,196-DOF shell model,
with explicit new fixtures and independent numerical controls. Read its
[measured comparison](../atlas_study/cpu_repro/REPORT_ZH.md) before using the
historical numbers. The original cpu_next source remains unrecovered; the
new reconstruction does not establish identical historical inputs or timings.

```python
# Algorithm sketch, not a recovered executable from the missing experiment.
# All maps are defined on the same physical mother model within one backend.
T_i = sparse.hstack([P, B_i]).tocsr()
K_i = (T_i.T @ K_m @ T_i).tocsc()
factor_i = factorize(K_i)           # offline
b_i = T_i.T @ balanced_source_i     # current source; common mother relief
z_i = factor_i(b_i)                 # one RHS after same-feature aggregation
q_i = T_i @ z_i
q = sum(q_i_for_selected_sources)
stress = S_m @ q
```

P carries global coarse behavior. B_i adds local independent fine directions;
it is not the full K inverse restricted to regional load columns. Its support
must respect element incidence, interfaces and shell director/rotation
semantics so that the trial space is conforming. Constrain the common gauge
consistently and use the mother mass/relief before feature projection.

Building T_i and K_i for the native shell and volume models remains part of
the current work. The published independently meshed CPU examples supply
real-refinement, finite-load and common-probe controls; they do not already
implement this common-mother construction. Any reconstructed code should be
identified as new and independently checked.

For each model, verify numerical residual in its selected space and separately
measure combined spatial field/peak error against the mother. Source-wise
mixing is not automatically a joint Galerkin solve. Check arbitrary positions,
nonconvex identity, remote peaks and continuous selector junctions.

## How to use the references in the native work

1. Reproduce a small real-refinement control with visibly different mesh/DOF
   counts. Keep a coarse, local-feature and union/reference comparison.
2. Choose a validated compatible geometry and reduced-space construction
   for each backend, with common physical loads, gauge and inertia relief.
3. Port preparation and runtime to Quadrants, reusing native loading,
   selection, grouping and lifecycle machinery.
4. Validate actual source and combined fields, then measure the complete
   accuracy/time/memory tradeoff. CPU module results remain CPU evidence.

The older modules and their evidence remain unchanged. New CPU experiments
are reported separately in cpu_repro; this guide does not claim native GPU
completion or new physical-contact convergence.
