# Precision-aware spatial recovery and original-history field checks

This continues the complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
The8433-tet spatial control's rejected step26 now passes an independently
reconstructed complete balance without changing its original physical inputs
or1e-9N force budget. Its complete80-step re-solve remains live. Separately,
all80 original ground-supported startup states retain global stress intervals,
with independent interior checks and actual native CUDA nodal replays. These
are reference qualifications, not completed spatial/contact/application acceptance.

## Measured contact arithmetic and unchanged force law

Probe715836 reconstructs all18 normal and18 tangential stencils at the saved
rejected local-level15 state. For each tested face/vertex stencil,
`abs(weight) * normal_force_magnitude` equals the body normal resultant.
Per-stencil assembly reproduces the original global normal gradient exactly.
Centering stencil coordinates before evaluating the same IPC barrier changes
the complete collision-space normal force by4.62503e-10N. Directly interpolating
`u-u_previous` changes its relative increment by at most3.46945e-18m compared
with the original rest-position subtraction route; the corresponding shear-force
change is1.31992e-11N. These are measured arithmetic differences, not new loads.

`PreciseStencilSpatial` retains the original mesh and P2 contact-node map. It
evaluates each normal stencil after subtracting a wide-coordinate local origin,
updates tangential normal-force magnitudes using the independently checked
weight/resultant relation, and maps relative displacement increments directly.
Material, mass, physical geometry, all finger meshes, dt, friction, barrier
stiffness/distance and velocity mollifier remain fixed. It currently accepts
the face/vertex body/finger stencil types in this control and rejects others.
This scope does not limit the production feature's arbitrary-contact contract.

The original tangent serves as a numerical Newton approximation. It is not
declared the exact derivative of every modified floating-point force operation.
Complete mapped directional differences at perturbations1e-11/3e-12/1e-12m
are4.68330e-11/1.37317e-10/3.10287e-10. Those random full-field checks do not
replace weak-mode checks or the complete final force gates.

## Independently recovered spatial step26

At the original failed iterate, centered force evaluation initially increases
the mapped residual from1.01144e-9 to1.51056e-9N. Producer715849 then converges
the unchanged step with a stricter1e-10 internal target. Independent audit715866
does not import or call that precision-aware solve. It separately reconstructs
the unsmoothed normal/current-friction stencils, consistent body/finger inertia,
authored0.1 armature,100/10 PD gains,4N caps and1 passive damping term.

| Independent original step26 check | Result |
|---|---:|
| Complete mapped dynamic residual |9.61108e-13N|
| Complete body dynamic residual |4.36708e-13N|
| Maximum force pullback difference |5.55112e-17N|
| Normal stencils |18|
| Maximum Coulomb violation |negative, all tested stencils admissible|

Expanded independent audit715888 confirms18 active body contact nodes at this
recovered state, using the existing1e-8N activation diagnostic. Their current
lumped support area is0.00283025mm² and maximum unsmoothed normal force/area is
1.45993GPa. This is a different resolution/time diagnostic from the earlier
two-node final states. It is not a completed temporal or spatial tail, and the
lumped area does not identify an independently resolved physical contact boundary.
Normal/shear surface fields and areas remain available without nodal smoothing.

This recovers one rejected equation; it does not prove that contact arithmetic
was the sole cause of every previous failure. Producer715870 re-solves all80
original dynamic steps from the byte-identical64-target held closure and zero
initial velocity. No old dynamic prefix is substituted. The same-state audit
guards startup, and terminal independent history audit715877 is scheduled.
Until both close successfully, no complete finer trajectory or spatial tail is
accepted. This remains the fixed-arm/two-jaw control, not the full nine-joint
Franka application.

## Original support/history stress bounds and native replays

The original first0.1s has already passed80 independent force/control audits.
Field audit715853 reconstructs every one of its global unsmoothed corner stress
fields using saved wide body coordinates. It expands the affine P2 deformation
gradient into cubic determinant and quadratic deviatoric Kirchhoff Bernstein
coefficients in long double, retaining a positive whole-volume determinant
certificate and a global nonlinear peak lower/upper interval at every time.
The raw peak is never replaced by a smoothed or quadrature-only maximum.

At0.1s the global interval is22887.917912901–22887.917913061Pa and the J lower
bound is0.999999065053. The peak remains at the bottom support, reference
position`[0,0,-0.0283639758]`m. FP64 rounded-field bounds and the maximum measured
corner rounding difference4.56046e-5Pa are recorded separately. A tight computed
field bound does not measure continuum or contact accuracy.

Qualifications715862/715886 check397440 independent corner/interior samples across
12 saved states,69 barycentric samples per element. It directly evaluates
deviatoric Cauchy stress from F F^T, separately recording its cancellation
allowance. The later version normalizes barycentric coordinates in long double,
with exactly zero measured partition error and no negative sampled-J/bound margin.
Sampling verifies the stored certificates; the Bernstein construction
supplies the whole-volume argument. Reported J scalars are FP64: their comparison
retains an explicit4-epsilon scalar conversion allowance. It leaves the positive
J requirement and every physical force budget unchanged.

CUDA job715855 performs four actual native Quadrants replays at steps11/80,
each with full-inertia and rigid-relief inputs. Maximum native-versus-independent
full-corner tensor error is1.04748e-10; maximum complete residual-budget fraction
is0.0174588. Full-inertia versus rigid-relief and nonlinear reference differences
are separately measured modeling diagnostics. They do not establish matched
rigid/deformable feedback or finite-footprint load-model accuracy.

The first field-report attempt715842 completes its80 field computations but
fails on an absent historical `complete_requested_history` report key. Version2
derives the original requested count from the recorded protocol and explicitly
distinguishes the completed80-step prefix from a complete330-macro application.
First interior check715858 compares an FP64 reported J bound using long-double
epsilon and rejects arithmetic at a few FP64 ulps. Version715862 records the
scalar allowance;715886 additionally removes the FP64 barycentric partition
error rather than attributing that entire discrepancy to reported-J conversion.
Both failed reports remain visible and count as failed attempts.

Full original-history force audit715817, field audit715856, native replay715878
and bound qualification715879 are scheduled behind producer715815. Their
live/pending states are not accepted evidence. Complete3.3s history and physical
temporal/spatial/contact/geometry convergence remain required.

## Natural uncached timeout and remaining acceptance

Uncached30720-tet job714462_1 ends in Slurm TIMEOUT after8h00m, despite a wrapper
returncode file containing0. It completes49 held closure targets, saves no
dynamic states and has no complete summary. Slurm is authoritative; this does
not count as a successful original-versus-cached fine trajectory comparison.
The completed cached producer and its still-running common-point field audit
715757 remain separate evidence. No automatic restart is launched for this
timeout. Failed and successful source snapshots stay immutable.

The terminal `stencil-precision-checkpoint-v1` contains10 runs,577 file entries
and408 source-hash records. Manifest SHA256:
`5e9b6e4431109c52041670a945f7a4f199b6451177062e471cdb32eb5eafb815`.
Every stored entry is verified against the complete original runtime bytes;
live jobs are excluded. Reproduce the archive with
`scripts/publish_stencil_precision_checkpoint.sh`.
The expanded patch audit is retained separately by
`scripts/publish_stencil_patch_checkpoint.sh`:76 entries, manifest SHA256
`c94c759efa8d65756554ab437e6b9182f97ff9ed1044fd01e2ba22dcce96a2ab`.
It is outside the earlier manifest.

Reproduce with `scripts/probe_local_contact_precision.sh`,
`scripts/precise_stencil_spatial.sh`, `scripts/audit_precise_spatial_state.sh`,
`scripts/run_precise_spatial_history.sh`, `scripts/audit_precise_spatial_history.sh`,
`scripts/audit_original_history_fields.sh`, `scripts/replay_original_history.sh`
and `scripts/qualify_original_history_bounds.sh`; exact arguments and Slurm
commands are retained with the runs. All original A/B/C, geometry, finite-load
model/feedback and physically verified warmed accuracy/cost requirements remain.
