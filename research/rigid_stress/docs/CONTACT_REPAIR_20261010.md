# Contact robustness development, 2026-10-10

The active goal prioritizes legal contact recovery before performance tuning.
This report records development results. Final trajectory acceptance and
repeated throughput measurements remain in progress.

## Discrete model diagnosis

The original B=2048, tick 632, environment 1996, slot 1 input has 132 positive
Q10 samples in its original 5.998121586-mm footprint. The constant-ratio pad
force has tangential/normal ratio exactly 1 at effective friction 1. A
nonnegative coefficient linear program is infeasible. The transverse target
is 10.92801351 micrometres outside the sampled convex hull. More iterations
of that discrete pressure fit cannot recover this input.

A six-component surface-normal Coulomb traction fit on the same samples
does recover the apex wrench. Independent CPU FP64 force and moment errors
are 9.25e-18 N and 2.91e-21 Nm. The native semismooth Newton trial passes
17 GPU numerical and lifecycle cases. Near equilibrium, residual-based line
search removes a serial objective-resolution stall. However, the actual
varied-friction B=2048 rollout rejects another sliding contact at tick 120,
environment 957. This trial is retained as an ablation and is insufficient
as a general replacement for the compliant pad model.

## Local integration repair

Retain the pad's supplied normal and actual friction coefficient, compact
Gaussian/bump prior, finite radius, constant tangential/normal traction ratio
and original force/moment acceptance. The first attempt uses the shared fixed
surface Q10 samples. A failed sampled fit triggers device-side local
integration on the exterior face containing the force-line anchor.

For that face, let its corner positions be v_i and the anchor's barycentric
coordinates be b_i. Define q_i = c + b_i (v_i-c). Their centroid is c because
sum b_i(v_i-c)=0. The central triangle (q_0,q_1,q_2) and six surrounding
triangles partition the original face. Apply the original Q10 Duffy/Gauss
rule separately on each triangle. Replace the original face's samples rather
than adding its area twice. Evaluate quadratic shape functions in the parent
face coordinates. All other eligible faces retain their fixed samples.

This changes local integration explicitly. The force, radius, pressure
objective, pad friction constraints and wrench budget remain identical.
The low mesh is fixed. This is a contact sampling repair, separate from
physical mesh or global integration convergence.

The apex now has 561 eligible samples. The independent nonnegative LP is
feasible and the CPU fit takes 12 evaluations, with force/moment errors
4.44e-18 N and 4.95e-22 Nm. Both native serial and warp paths pass the apex
regression, nodal loads, complete equilibrium including gauge rows and full
maximum-stress comparison. Six targeted GPU mapping cases pass in 45.44 s.

The native retry set stays on device. The implementation recomputes actual
loads independently for every environment and every recovery substep. All
integration and fit work belongs in the new baseline's timing. The initial
development evaluation counter records the final fit pass only. Retry flags
identify affected contacts. Cumulative first-pass work will be added before
final tail-cost reporting.

## Workload change and pending validation

The example gives the plane and Panda material friction 0.1. Egg friction
remains 0.6 times the independent varied ratios [0.7,1.4], making the actual
combined egg contact coefficients [0.42,0.84] instead of the previous fixed 1.
The benchmark now accepts a seed and records its load model. Warmup evidence
includes per-environment integration retries, final-pass fit iterations and
contact force/moment errors. The original apex at effective friction 1 remains
an exact regression fixture.

Full CPU/GPU tests, actual varied-friction trajectories, multiple seeds and
all measurement scopes are being evaluated. Neither the unsuccessful
surface-normal trial nor targeted apex acceptance establishes completed
trajectory robustness or a final throughput result.

Runtime evidence is under
`$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair/`.
