# Warm Hessian precision and raw diagnostic clearing

Two additional candidates qualify unchanged-budget numerical checks and
replicated changing-contact live benefits. They are selected for clean
integration alongside coalesced positions and valid-source capture.
Clean combined qualification and final-source measurements remain pending;
the nonconvex goal is active.

Both experiments use RTX PRO6000 Blackwell Server Edition, FP64/Q10/s8,
body_source_prior/max, mesh64/GJK patch-on,dt0.01/substep1,600 changing warmup
and three1280-step repeats per route. Within each pair, the GPU and full saved
condition bank match. Observation, contact updates, reset, metrics, history,
fallback and every environment are included. All per-environment rigid/contact/
grasp/descent counters and final positions agree exactly; all measured invalid
steps, overflows and failed lifts are zero. The isolated gains are not additive.

## FP32 warm Hessian accumulation

CUDA warm Newton steps accumulate only the21 independent Hessian entries in
FP32, including their small operator products and lane reductions. Physical
sample positions/weights, Coulomb projection, forces, energy, gradients, duals,
search-direction solve and final acceptance remain in scene precision. Cold
and retry Hessians retain scene precision. FP64 is the precision of all
reported stress workloads. Existing FP32 scenes retain their scene precision.

The Hessian is converted back before the direction solve. Approximate search
directions still have to pass the original force/moment/local-cone checks;
all complete residual/gauge/stress checks and full fallback remain enabled.
No physical load law or stopping/acceptance budget is changed.

| Batch /seed | Baseline env-step/s | Candidate env-step/s | Ratio | Baseline repeat seconds | Candidate repeat seconds |
|---|---:|---:|---:|---|---|
|16384 /623102|105580.347|109136.683|1.033683688|198.655205 /198.562222 /198.675290|192.314720 /192.334476 /191.825720|
|8192 /991909|99687.052|101682.341|1.020015526|105.260157 /105.082524 /105.217659|103.302459 /103.140634 /102.925080|

Both pairs have nonoverlapping repeat ranges and unchanged native buffer bytes.
CPU31 cases pass in485.557s, CUDA31 in686.186s. Across96 independent full
fields, worst wrench error is7.388619086e-10, complete residual budget fraction
0.0458562252 and peak difference8.086841262e-6Pa. Full tensor/von Mises,
six gauge rows, local friction and signed P2 scatter are checked against CPU
FP64 references. Eight cached/direct-position checks remain exact.

Stable `qd.types.matrix` factories supply the kernel template, avoiding
Quadrants' non-weak-referenceable C++ datatype objects. The initial two
signature/compilation failures and corrected hard-fixture smoke are retained.
They are implementation failures, not rejected physical contacts.

The expanded source/CPU/CUDA/pair archive is
`native-fp32-warm-jacobian-qualification-checkpoint-v2`; all original/stored/
decompressed hashes and complete large-JSON reconstruction pass. Manifest:
`e6a18dd6fd5737cd8d3fefd154e96a55624226fa2187c1ccedc380e807911b9a`.
The first-pair archive is retained asv1.

## Previous/current raw diagnostic clearing

Raw statuses, wrench errors, evaluation/sample counters and certificates are
initialized once, then cleared through the previous complete active list and
every currently valid contact. Clearing the old list happens before collecting
the new one, so disappearing contacts and partial resets also lose stale
diagnostics. Never-active slots remain initialized to zero. Original global
contact indices, radius storage/API and every physical input are retained.

The prototype uses build-time field fills; clean production fuses them into
one native initialization kernel. The same diagnostic function serves initial,
previous and current entries. The lifecycle regression injects stale sample
counts, then verifies every cleared diagnostic on released environments.

| Batch /seed | Baseline env-step/s | Candidate env-step/s | Ratio | Baseline repeat seconds | Candidate repeat seconds |
|---|---:|---:|---:|---|---|
|8192 /991909|100632.947|101542.484|1.009038163|104.442582 /104.067468 /104.084193|103.693459 /103.117932 /102.982881|
|16384 /734001|105553.747|106649.468|1.010380685|198.860590 /198.493214 /198.689079|196.823388 /196.587180 /196.508552|

Valid-source-only initialization is enabled in both routes of these pairs.
Both pairs have nonoverlapping repeat ranges. Native bytes are unchanged:
7,662,968,394 and16,129,154,634. CPU31 cases pass in485.825s and CUDA31
in695.253s. Across96 independent full fields, worst wrench error is
7.386597204e-10, complete residual budget fraction0.0458562252 and peak
difference8.235838322e-6Pa. Partial release/reset, workspace growth, invalid
members and environment isolation pass.

The first-pair archive is
`native-active-raw-diagnostics-qualification-checkpoint-v1`, manifest
`2e02b785cda5e1c5ea339f538caee8ef25fbf02fa45bfc8f3cabc67fb973bfe6`.
Its expanded two-cohortv2 publication passes all original/stored/decompressed
hash checks and exact large-JSON reconstruction. Manifest:
`410c3f8a297abe8a0917f12a1245fda7e9e770d3442d84a96f72240ef5465e1b`.
Clean combined qualification is separately tracked in
[the combined report](NATIVE_COMBINED_QUALIFICATION_20261013.md).
All complete results and slow environments remain retained.

## Remaining-path squared distance gate: not selected

This separate candidate evaluates the squared remaining path allowance before
taking a port-to-target square root, with a conservative scene-precision
roundoff margin. It retains the same physical radius, exact adjacent-triangle
unfolding, verified planar paths and final minimum. Baseline/candidate both
enable coalesced positions and valid-source capture.

B8192/seed623105 three-repeat rates are101809.507→101692.107 env-step/s,
ratio0.998846865, with overlapping repeat ranges. Baseline seconds:
103.209385 /102.714809 /103.057561; candidate:
102.950275 /103.323439 /103.064749. CPU31/CUDA31 cases pass in
479.621/678.892s and all complete live outcomes remain valid, but there is no
end-to-end benefit. The candidate is not retained in production.

Archive `native-squared-port-bound-qualification-checkpoint-v2` passes full
publication verification; manifest
`829a5c21358e32ee611fd2f65fb311ea2f21382f0a126d344abca4a40dcf61d2`.
The separate inactive-walk exit candidate passes31CPU/31CUDA cases in
483.78/688.66s. Same-GPU B8192/623105 rates107923.558→107961.431 env-step/s,
ratio1.000350925, have overlapping repeat ranges and no useful measured gain.
It is not selected. Both routes already contain the four clean candidates,
retain the same geometry/law and all outcomes, and have zero invalid/failed
lifts. Baseline repeat seconds97.147838/97.081713/97.247867; candidate
97.445942/96.987667/96.941559.
Archive `native-segment-early-exit-qualification-checkpoint-v1` verifies all
original/stored/decompressed hashes and complete chunk reconstruction;
manifest `db56887c5fe0d13b8dc556c9ea777af88ec1478ac077a27f20a5127ff402cbc5`.
