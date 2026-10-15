# Fixed-surface egg thickness axis and original-state recovery

This advances the complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
An independent mesh audit establishes an unchanged-boundary2/3/4-layer egg
volume axis. The new thickness FEM fields are still running. Separately, all
1936 original-history states and global fields are audited, and original1937
is recovered and independently checked at the unchanged physical budgets.
Full original history, physical convergence, remaining model/feedback and
verified warmed accuracy/cost requirements remain active.

## The thickness study fixes both actual material boundaries

The existing finite-traction egg has200892 tetrahedra at surface size0.03 in
unit-sphere coordinates and two thickness layers. All five finite-load cases
still have7.64–9.61% last-change tensor diagnostics, despite a0.091% opposing
raw-peak change. Those earlier curves vary surface geometry and volume together.
They do not isolate through-thickness error.

`refine_egg_thickness.py` uses the actual saved outer and inner vertex arrays,
retains their triangle connectivity exactly, and places only new interior
layers on the existing radial segments. It reconstructs the seed prism
connectivity and checks every boundary face before starting the FEM solves.
No CAD projection, material/density change or mass compensation is introduced.

716370_0 independently re-enumerates tetrahedron faces, checks positive
volumes and exact material boundaries, and reconstructs volume and COM from
oriented boundary-surface fluxes. Compensated tetrahedron-volume sums provide
a second geometric measure. The completed auditor retains hashes of every
mesh/metadata input before and after its calculation.

| Thickness layers | Tetrahedra | Boundary triangles | Independent surface-flux volume (m³) | Discrete mass at rho2000 (kg) |
|---|---:|---:|---:|---:|
|2|200892|66964|3.7424638563292785e-6|0.007484927712658562|
|3|301338|66964|3.7424638563292785e-6|0.007484927712658562|
|4|401784|66964|3.7424638563292785e-6|0.007484927712658562|

The independent surface-flux COM is identical on all three meshes:
[-7.35052400334e-9,9.94216313123e-9,-0.00196336709990]m. Both boundary vertex
arrays remain byte-identical. These are geometry-only results:716343/716344
are still solving the five unchanged finite-traction FEM cases.716353 must
check their complete corner peaks/work, common-point tensors/displacements,
mass/COM, frozen physical inputs and every boundary quadrature traction sample.
Its initial3% diagnostics remain separate from physical accuracy acceptance.
The existing affine CAD approximation is fixed by this study, not corrected.

This is the independently declared finite-traction egg case. Its7.48493g mass
is distinct from the original Franka application's6.52985g case; the two cases
are not used as a matched feedback comparison.

## The finer two-jaw prefix preserves every original saved field

The [fine-prefix qualification packet](../evidence/20261014-physical/fine-prefix-qualification-v1/README.md)
retains the interrupted30298-tet source and two naturally closed qualification
runs.716329_0 re-solves all five original dynamic states from the byte-identical
held initializer in44m15s.716330_0 independently reconstructs every state and
compares all saved arrays and complete corner/quadrature tensors with the
original five-state data. Maximum body/mapped residuals are4.84244e-11/8.80326e-11N;
maximum corner/quadrature tensor changes are6.04881e-11/3.21242e-11.

At6.25ms, the186 active body nodes have3.43003e-9m² lumped support. The maximum
unsmoothed nodal normal force/area diagnostic is3.61394GPa; the raw nonlinear
global stress interval is[3.79054690,5.94033052]GPa. The whole-volume determinant
lower bound is0.653992, which is not a measured strain norm. These retain the
unresolved concentration and field-bound width; they do not establish a physical
contact tail or small-strain recovery validity.

716331_0 continues all80 original control steps only after that numerical and
independent force qualification passes. Its five copied prefix files are
byte-identical. The pending job's walltime is increased from12h to24h based on
the measured reference cost, without changing physical parameters.716332_0 is
its independent all-state audit. This fixed-arm/two-jaw control remains separate
from the full original nine-joint approach/lift/support history. The packet
verifies280 actual-runtime entries and202 source-identity records; no live
continuation result is included.

## All1936 original states and global fields are preserved

716302_0 naturally exits FAILED after saving1936 physical states and rejecting
original1937. Its return code is1 and its natural source/input closure records
are present.716303_0 independently checks every saved state, original control,
ground contact and actual robot-step balance through2.42s. All original1e-9
physical gates pass. The producer remains an incomplete2640-step history.

716319_0 reconstructs raw whole-volume stress bounds for every1936 state.
The unrestricted maximum over that prefix is1.06873070MPa; the final raw
interval is[156365.30748023745,156365.30748039758]Pa. Its whole-volume
determinant lower bound is0.999991884 at that final state. The objective maximum
displacement is0.351674um. These extrema retain their actual global search.

716320_0 executes12 actual native CUDA replays at six declared times, with
full-inertia and rigid-relief loads kept separate. Maximum complete elastic
budget fraction is0.0375514 and maximum native/independent tensor difference
is1.97918e-10. At the final state, the nonlinear/full-inertia-linear tensor
difference is4.87449e-5. These qualify the declared coarse replay and model
diagnostics, not spatial convergence.716321_0 additionally checks8147520
independent interior/corner samples against retained volume bounds.

## Pseudo-stage residuals are distinguished from the physical equation

The saved direct1937 candidate does not satisfy the original body/robot gates.
716350_0 rechecks it and fails accordingly. A small residual in the subsequent
artificial-defect equations is not a physical state: at fraction0.0625 the
continued residual is tiny but the original body residual is0.0552528N and
the original robot effort residual is501.196 in the recorded joint units.

`original_residual_stage_budget.py` separately checks a saved numerical stage
when the unchanged internal1e-10 Newton target cannot be reached. Continued
equation residuals must be finite and at most1e-9. Intermediate stages remain
numerical iterates. At fraction1, the entire artificial defect is removed;
the original body equation, actual robot equation and fresh robot step must
all satisfy the unchanged1e-9 physical gates. Geometry, controls, predictor,
previous friction state,1.25ms dt and physical loads remain fixed.

716352_0 reaches fraction1.716354_0 independently audits all1937 physical
states, including the recovered2.42125s state:

| Recovered original1937 check | Independent result |
|---|---:|
| Body dynamic residual |2.01445098951e-10N|
| Actual robot-step effort residual |9.87870473752e-12|
| Contact pullback difference |0N|
| Contact groups |Two actual fingers; no ground reaction|

Across all1937 states, maximum body and actual-robot residuals are
2.09789e-10N and9.91075e-11. No artificial continuation stage is added to
that physical history. The failed1936-state producer is retained as failed.

716355_0 also applies the method to the previously recovered original869
with the same868-state source.716356_0 compares sealed complete arrays,
every corner/quadrature tensor and raw corner maximum. All differences are
zero under the existing numerical-comparison budgets. It does not claim
equivalence of uncomputed future trajectories. A later finite-value guard
only rejects nonfinite stage norms; the qualified finite branch is unchanged.

716357_0 initially resumes the330-macro history after both the1937 independent
audit and869 numerical qualification pass. All1937 prior state files are copied
byte-identically. The next qualification compares checking original physical
budgets on every Newton candidate against that unchanged baseline.

## Candidate physical-budget checks pass a complete-state qualification

716371_0 completes the243-macro short history, preserving every1937 copied
prefix file byte-identically.716372_0 independently checks all1944 states at
the unchanged original budgets; maximum body/actual-robot residuals are
8.33512e-10N and5.42449e-10. This is short-history completion, not330 macros.

Read-only sealed observations retain1938–1944 from the still-live baseline
and completed candidate, with every saved array and original mesh/XML/config
identity. Only requested future history length differs outside their common
window.716377_0 checks all seven common states and retains the existing
1e-5 relative/1e-12 absolute array budgets and1e-6 tensor/peak budgets:

| Complete field comparison | Maximum relative difference |
|---|---:|
| All corner tensors |1.58123e-11|
| All quadrature tensors |1.30579e-11|
| Raw global corner peak |3.77542e-12|

The physical tolerance, controls, dt, loads and constitutive equations do not
change. The internal Newton target remains recorded separately; a candidate
can exit through the already declared physical gates only after a fresh robot
step also passes. The [qualification packet](../evidence/20261014-physical/original-budget-qualification-v1/README.md)
retains the complete short producer/audit, sealed common states and comparison.
Its8249 actual-runtime entries and262 source-identity records are verified
separately from the preceding6821-entry thickness/recovery packet.

716378_0 starts the full330-macro continuation from those1944 independently
audited states. Every copied prefix file is additionally rehashed. Its full
state/global-field/native/interior-bound chain is716379/716380/716381/716382.
Only after that startup and numerical qualification is the redundant716357_0
baseline cancelled. Its actual wrapper code is0, Slurm isCANCELLED, and its
summary is missing.716358_0 independently checks all1964 saved states and
correctly classifies the producer as incomplete. The cancelled run and its
source EXIT records remain intact. No completion is inferred from wrapper0.

## Reproduction and remaining acceptance

Exact scheduler commands, source identities, failed attempts and closed results
are retained in the [immutable evidence packet](../evidence/20261014-physical/thickness-original-checkpoint-v1/README.md).
It verifies6821 actual-runtime entries and466 source-identity records. The geometry-only audit
explicitly records that the two FEM producers are live and does not infer their
completion. The new thickness meshes are independently audited data inputs,
not completed FEM reference fields.

```bash
P=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation
bash research/rigid_stress/scripts/run_egg_thickness_axis.sh "$P/egg-thickness-reproduction" "$P/fine-matrix-mesh-egg-v1/meshes/h0.03.npz" 3
bash research/rigid_stress/scripts/continue_budgeted_original_residual.sh "$P/original1937-reproduction" --run "$P/franka-original-audited-prefix-full-history-v1"
```

Commands require the pinned image1_26 and existing physical-validation
environment through the documented Slurm launcher. Full contact/geometry and
volume tails, the original complete approach/lift/support history, resolved-load
replay versus actual feedback, remaining load models and verified physical
accuracy/cost remain required by the [acceptance ledger](PHYSICAL_ACCEPTANCE_LEDGER.md).
