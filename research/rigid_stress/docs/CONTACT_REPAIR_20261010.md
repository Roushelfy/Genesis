# Contact robustness development, 2026-10-10

The active goal prioritizes legal contact recovery before performance tuning.
This report records validated development checkpoints. Broad trajectory
coverage and final repeated throughput selection remain in progress.

## Internal stopping budget update

The B=32768 seed-623001 trajectory rejects environment 13456, slot 0 at
tick 943 under the historical 2e-12 dual threshold. Its fixed 82 samples
are LP-infeasible; the local 474 samples are feasible. The CPU FP64 pad
fit preserves the original wrench after 80 evaluations despite numerical
stagnation. This is separate from the earlier sampling inadequacy.

Following the user's error-budget instruction, the internal normalized
dual threshold is now 1e-9, one tenth of the independent 1e-8 final wrench
budget. Both the initial Gram fit and the serial/warp constrained fits use
it. The final scatter still checks relative resultant force and moment
divided by force magnitude times the original radius against 1e-8.
Complete equilibrium still includes all six gauge rows and uses
max(1e-11 N, 1e-8 times RHS norm). Independent maximum-stress consistency
still uses rtol=1e-4, atol=1e-3 Pa. Friction admission, geometry, sample
selection, footprint radius, pressure law and Newton limits are unchanged.

A trial at 1e-10 still rejects the cooperative GPU fit: its measured
normalized residual is about 1.3e-10, while the final force/moment checks
pass. At 1e-9, serial and warp GPU paths accept the captured input in
106 total evaluations, including the failed fixed-sample attempt. Their
force errors are 9.37e-14 / 2.17e-13 N and moment errors
3.32e-15 / 2.59e-15 Nm. These diagnostic coefficients are also covered
by the unchanged CPU-oracle nodal-load, complete-residual and peak checks
in the native regression.

CPU and GPU each pass all 27 regression cases (134.58 / 391.40 s).
Three additional CUDA fused-contact-graph variants pass in 55.33 s,
including the B=32768 captured input in 106 evaluations. They repeat the
same unchanged force, moment, nodal-load, full-gauge and peak assertions.
Independent full-field trajectories pass 192 samples per seed for
510000 and 623001, plus 96 samples with four substeps. Saved-state versus
individual-setter reset comparisons also pass both seeds for 2400 steps
and 48 partial environment resets each. The full fields, wrench frames,
actual friction and complete gauge residual checks remain mandatory.
The oracle retains its independent stopping algorithm. These checks
select the internal threshold; they do not establish a throughput optimum.

## Bounded native face scheduling checks

The production scatter can schedule complete eligible contact/face tasks
independently, preserving the same fixed Q10 and local replacement samples.
Its build-time buffer is bounded. Capacity overflow triggers a complete
native contact-warp integration on the device. Incomplete task lists never
contribute partial loads. Wrench checks retain the original 1e-8 budgets.

The integrated v2 source passes 27 CPU cases (12 CUDA-only skips), all
39 CUDA cases, and 672 independent FP64 full-field oracle snapshots.
The snapshots cover both seeds, four substeps and forced task overflow.
The captured B=2048, B=16384 and B=32768 fixtures pass serial, cooperative,
graph, face, forced-overflow and uncached-face tests. A local-variable
naming cleanup repeats all 18 CUDA fixture cases successfully.

Two additional v3 every-step audits each validate 67,200 observations:
normal live recovery with zero overflow, and policy plus full-field output
with 2096 deliberately forced complete-scatter fallback calls. They include
contact changes and partial resets. Final full-residual, force/moment and
friction checks are unchanged. These development checks establish numerical
and lifecycle correctness. Final large-batch throughput and policy coverage
remain in progress, as recorded in the optimization ledger.

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
integration and fit work belongs in the new baseline's timing. Retry flags
identify affected contacts. Source e1fbf937 adds cumulative evaluations
covering both initial and retry fits; the initial development counter recorded
only the final pass.

## Workload change and pending validation

The example gives the plane and Panda material friction 0.1. Egg friction
remains 0.6 times the independent varied ratios [0.7,1.4], making the actual
combined egg contact coefficients [0.42,0.84] instead of the previous fixed 1.
The benchmark now accepts a seed and records its load model. Warmup evidence
includes per-environment integration retries, final-pass fit iterations and
contact force/moment errors. The original apex at effective friction 1 remains
an exact regression fixture.

Checkpoint 6494f7f7 passes 17 GPU cases (199.84 s) and 15 CPU numerical cases
(89.11 s). Actual B=2048 live measurement reaches 57,175.96 env-step/s and
27.91795 batch steps/s over 1,200 timed steps, retaining all 2,048 resets.
Warmup meets the bilateral-hold criterion in 1,880/2,048 environments.
Remaining environments are retained in timing. Maximum sampled complete
residual is 1.6744e-10 N. A separate every-step audit covers 4,300,800
observations with zero invalid environment steps. This workload's sampled
contacts need no local retries. The apex regression exercises the retry
and complete elastic solve.

## Complete exterior operator checkpoint

Balance and elasticity are linear in the complete exterior P2 nodal force
space and six centrifugal coefficients at the declared fixed geometry,
material and mass. Construct their exact shared responses in Quadrants.
This uses all 162 exterior nodes and three force components, retaining the
complete 810-node displacement and global stress scan. Complete equilibrium
checks and full-load correction/factor fallback remain active. Arbitrary
interior-load validation uses the full inverse.

At B=1024, native application microtiming falls from 7.2782 to 1.5147 ms.
Relative displacement difference is 7.40e-15, maximum peak difference
6.21e-8 Pa, and independent CPU FP64 complete residual at most 6.30e-12 N.
Checkpoint e1fbf937 passes 18 GPU cases (305.01 s) and 16 CPU numerical cases
(94.41 s), including arbitrary exterior loads, rotation and partial reset.

On one RTX PRO 6000 Blackwell allocation (GPU UUID
GPU-f30bd82d-756d-28ed-24af-a6dec8bbf9bc), B=1024 full inverse and complete
exterior operator live rates are 41,310.81 and 54,137.61 env-step/s
(one-repeat development comparison, +31.05%). The latter reaches 84,456.04
at B=2048. Extra shared operator storage is 9,565,128 bytes.

## Face reduction checkpoint

Reduce all eligible sample loads within each complete exterior face before
writing its six nodes. Conservative triangle bounds select faces, and every
eligible Q10 sample, including seven-triangle local retries, is evaluated.
Global force/moment checks remain unchanged.

An actual B=1024 frozen snapshot measures scalar scatter 2.6750 ms versus
face reduction 1.1553 ms. Nodal force difference is at most 3.33e-16 N.
Source e0b6714c passes 18 GPU cases in 218.54 s. Its separate policy audit
at B=2048 records zero invalid observations in 4,300,800 environment steps.
On one GPU, live B=1024 scalar and face rates are 53,820.81 and 58,316.33
(+8.35%, one development repeat). B=2048 reaches 86,897.68 env-step/s.

An independent seed 623001 at B=4096 completes the full 1,200-step trajectory
at 120,705.13 env-step/s (29.469 batch steps/s). Seed 510000 at B=8192 reaches
150,361.28 env-step/s (18.355 batch steps/s). These are single-repeat
development runs on separate allocations, not a selected maximum.

The optimized B=1024 wall profile measures pressure 1.3950 ms, scatter
1.2100 ms, exterior application 1.5650 ms and global peak 1.1174 ms. A
separate 50-step CUPTI trace contains 22,098 GPU kernels and 5,496 copies;
GPU kernel duration totals 544.24 ms and copies 2.75 ms. It records 3,296
stream synchronizations and 9,598 runtime kernel launches. Most copies are
small pageable host-to-device scalar transfers. This intrusive trace is
excluded from throughput and motivates remaining scheduling/fusion work.

These are validated development checkpoints. Final scopes, multiple seeds,
larger batches, repeated measurements and additional applicable candidates
remain required by the active goal. Original JSON, test logs, rejected trial
and sampled whole-card memory are archived with hashes in
`../evidence/20261010-contact-repair/`.

The surface-normal cone's circumscribed 32-direction LP remains feasible
for the rejected sliding input, leaving circular-cone feasibility undecided.
That trial remains an unresolved fit/model failure.

Runtime evidence is under
`$RIGID_STRESS_DATA_ROOT/runs/20261010-contact-repair/`.
