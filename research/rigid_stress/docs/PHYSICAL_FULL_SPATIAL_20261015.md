# Complete fine spatial history and measured linearization discrepancy

This advances the complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
The precision-aware8433-tet reference now completes all80 dynamic steps, and
every step passes an independent force/kinematic audit at the original budgets.
The three complete local-refinement histories remain physically unresolved:
the finest adjacent tensor/energy changes over ten common times are23.94%/49.66%. Independent linear recovery
of resolved32.5ms loads also exposes a5.17% tensor discrepancy on the finest mesh.
These findings do not establish contact convergence or stage completion.

## Complete80-step reference and independent audit

Slurm715870_0 completes in46m55s;715877_0 independently audits all80 states
in4m16s. The producer restarts from the byte-identical original64-target held
closure with zero velocity and resolves the entire0.1s history. It does not
copy the original25 dynamic states. The already published25-state comparison
establishes the numerical precision correction's full-array and full-tensor
equivalence on that original prefix.

The physical geometry, material, mass, friction, contact parameters, actuation,
dt1.25ms and original1e-9N force budgets are unchanged. This is the explicit
fixed-arm/two-jaw control, not the full original approach/lift/support history.
The independent audit reconstructs centered normal/friction stencils, P2
pullback, consistent body/finger inertia, armature and capped jaw PD. It checks
every previous displacement/velocity and acceleration, body/mapped equilibrium,
Coulomb constraints and unrestricted whole-volume field bounds.

Maximum audited body/mapped residuals are5.5381164e-11/9.8007450e-11N.
At step80 the audit retains18 active body nodes,2.8301354e-9m² lumped support,
and1.4599041GPa maximum unsmoothed normal force per lumped area. This area is
a quadrature diagnostic, not an independently solved physical contact boundary.
The raw global stress interval is[2022676378.8721902,2022676378.8721905]Pa,
and the whole-volume deformation determinant lower bound is0.797373079.
Its hotspot remains at reference coordinates
[-2.26834868e-5,-0.0219618335,0.00208660682]m. These are field enclosures,
not bounds on continuum or load-model error.

## Three complete spatial histories

715965_0 exports immutable state/geometry links and freshly derived final fields
without resolving dynamics or altering the producer.715966_0 then compares
480/3577/8433-tet histories at2400 common material probes and ten physical times
0.01–0.1s. Each tensor uses its proper finite body frame. Seed-volume probe
weights define diagnostic norms rather than exact continuum volume norms.

| Observable, normalized by finer result |480→3577|3577→8433|
|---|---:|---:|
| Complete probe tensor |41.1587%|23.9432%|
| Probe von Mises |20.7651%|8.15881%|
| Objective displacement |17.7018%|6.65699%|
| Velocity |0.639479%|9.32416%|
| Nonlinear body energy |79.5391%|49.6634%|
| Contact force |0.0930742%|0.0520455%|
| Contact moment |0.0539121%|0.0999734%|
| Final raw global peak |99.9442%|86.1305%|

Final raw peak lower bounds are0.156447/280.534/2022.676MPa. The comparison
preserves the wide-coordinate audited fine interval separately from its
ordinary-FP64 derived-field enclosure. Larger active contact sets and small
resultant changes do not establish a physical stress tail. Refinement changes
both bulk and collision resolution within the fixed polyhedron. This is not
a separated-volume or smooth-CAD geometry-convergence result. The independently
balanced32.5ms fields in the [previous checkpoint](PHYSICAL_MATCHED_SPATIAL_20261015.md)
remain a separate single-time comparison with its own recovered fine state.

## Independent resolved-load linearization

715999_0 independently reconstructs linear FEM on each of those three balanced
32.5ms states. It rotates resolved contact/gravity/full-inertia loads into the
proper material frame and uses independent scikit-fem elasticity/mass assembly.
The reference rigid-mode balance correction is explicit:1.31630e-10,
3.51844e-8 and5.60772e-8N. It includes the reference-coordinate wrench change
and must not be confused with a failed original dynamic force budget.

| Body tetrahedra | Resolved linear/nonlinear tensor difference | Linear law at frozen objective deformation | Raw peak difference |
|---|---:|---:|---:|
|480|0.00480425%|0.0123485%|0.00263126%|
|3577|0.726501%|0.818433%|0.490960%|
|8433|5.17456%|5.40253%|2.99613%|

Tensor differences are volume-weighted L2 diagnostics at the identical
order-six quadrature points. The resolved linear/nonlinear difference combines
finite-geometry and constitutive linearization, with the balance correction
reported separately. It is not a pure constitutive-error measurement. The
frozen-deformation column changes only the stress evaluation at that objective
deformation; it is not another equilibrated solution. Linear raw global peaks
are0.156446/279.148/1982.962MPa, compared with the original nonlinear raw peak
lower bounds0.156450/280.525/2044.209MPa.

The finest local Green-Lagrange strain norm near0.20 already showed that small
wall-relative displacement does not ensure a small-strain hotspot. The new
comparison quantifies that limitation. It is scoped to an unresolved physical
setting and cannot define a validated operating range. The near-equality of
full-inertia and rigid-relief linear fields is likewise not a corresponding
solved rigid/deformable feedback comparison.

Actual native CUDA recovery of both declared loads on all three meshes completes
in716014_0/1/2. All six replays pass the original complete elastic residual
budget `max(1e-11N,1e-8*||rhs||)`, including all six gauge rows. Independently
assembled full residuals consume at most0.479106 of that budget. Full corner
tensors agree with independent linear FEM to1.43117e-10 relative L2 difference;
native-versus-nonlinear quadrature differences reproduce the table above.
The fine full-inertia replay reaches20000 PCG iterations and still passes the
original final acceptance budget; no tolerance is relaxed. The subsequent
rigid-relief mode reuses that displacement as an initial guess. These iteration
counts are numerical diagnostics, not a warmed throughput or accuracy/cost study.

The first comparison attempt716010_0 fails because
its postprocessor used order-four linear quadrature against order-six nonlinear
samples; remaining tasks were cancelled after diagnosing that mismatch.
The corrected postprocessor checks matching sample shapes before native build.
The failed/cancelled attempts remain archived and do not count as accepted replays.

## Original-history completeness classification

`original_history_status.py` now requires consecutive saved states, a matching
summary, the complete requested count and zero producer code before labeling
a requested history complete. Incomplete-prefix audits remain explicit opt-ins.
This prevents the observed cluster behavior of a TIMEOUT wrapper writing zero
from being treated as completed application output. Authoritative Slurm state
and source closure remain separate required evidence.

The metadata qualification checks the actual completed80-state original startup,
the actual failed865-state source, a counterfactual supplied zero code on those
same865 files, malformed/count-mismatched summaries and a gapped prefix. Original
returncode files are not changed. This only qualifies classification; full
physical state/force audits remain necessary.

## Reproduction and remaining acceptance

Run the reference scripts inside the existing1_26 image and declared physical
validation environment. Each output must be fresh; these commands show the
already recorded physical inputs with fresh reproduction destinations:

```bash
P=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation
bash research/rigid_stress/scripts/run_precise_spatial_history.sh "$P/franka-precise-spatial-history-reproduction" --run "$P/franka-tip-coupled-15-v1" --qualification "$P/franka-precise-stencil-state-audit-v1"
bash research/rigid_stress/scripts/audit_precise_spatial_history.sh "$P/franka-precise-spatial-audit-reproduction" --run "$P/franka-precise-spatial-history-reproduction"
bash research/rigid_stress/scripts/export_precise_spatial_fields.sh "$P/franka-precise-spatial-export-reproduction" --run "$P/franka-precise-spatial-history-reproduction" --audit-run "$P/franka-precise-spatial-audit-reproduction"
bash research/rigid_stress/scripts/compare_spatial_franka.sh "$P/franka-precise-spatial-tail-reproduction" --axis local_levels --run "$P/franka-tip-coupled-0-v1" "$P/franka-tip-coupled-12-v1" "$P/franka-precise-spatial-export-reproduction"
bash research/rigid_stress/scripts/audit_resolved_linearization.sh "$P/franka-resolved-linearization-reproduction" --comparison "$P/franka-matched-recovered-spatial-v3"
bash research/rigid_stress/scripts/replay_resolved_linearization.sh "$P/franka-resolved-linearization-native-0-reproduction" --audit-run "$P/franka-resolved-linearization-reproduction" --case 0 --backend cuda
bash research/rigid_stress/scripts/replay_resolved_linearization.sh "$P/franka-resolved-linearization-native-1-reproduction" --audit-run "$P/franka-resolved-linearization-reproduction" --case 1 --backend cuda
bash research/rigid_stress/scripts/replay_resolved_linearization.sh "$P/franka-resolved-linearization-native-2-reproduction" --audit-run "$P/franka-resolved-linearization-reproduction" --case 2 --backend cuda
```

Complete physical contact/volume/geometry/regularization tails, the nonlinear U
tail, original3.3s Franka history, remaining finite-footprint sensitivities and
feedback separation, and warmed native cost at verified physical accuracy all
remain required by the [acceptance ledger](PHYSICAL_ACCEPTANCE_LEDGER.md).
The editable matched maps from the preceding checkpoint remain valid for their
explicit32.5ms inputs; they are not plots of these new complete-history results.
