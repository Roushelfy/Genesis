# Captured COM motion and controlled initial-velocity sensitivity

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. This checkpoint corrects the reference point of captured linear
velocity and introduces a corresponding initial-motion control. It does not
reproduce the original approach/lift/support history or qualify physical
spatial/contact accuracy.

## Velocity reference and independent observation

`capture_franka_reference.py` records `workload.link.get_vel(relative=False)`.
The rigid-link getter and solver default measure world velocity at the internal
link origin, rather than the mass COM. The corresponding COM velocity is

```
v_COM = v_origin + omega_world cross (x_COM - x_origin).
```

Observer job715153 repeats the original32-substep application capture on the
same RTX PRO6000 Blackwell node, adding direct solver getters with
`ref=gs.link_ref_frame.link_COM`. It retains all original576 capture records
exactly and verifies16173 original NPY payloads across1165 NPZ archives byte for
byte. The additional observation does not change the original dynamics or
record schema. Direct COM velocity and independently transported origin
velocity differ by at most1.38778e-17m/s over all576 frames.

At the selected final stationary-lift-hold frame, tick359/substep31:

| Quantity | World-coordinate vector |
|---|---|
| Origin velocity, m/s | `[-8.09737e-10, 2.93847e-9, -1.65070e-6]` |
| COM minus origin, m | `[1.14878e-5, -1.78818e-5, -1.63589e-3]` |
| Angular velocity, rad/s | `[2.32465e-5, 8.61216e-2, -9.41346e-4]` |
| Actual COM velocity, m/s | `[-1.40902746e-4, 3.01530736e-8, -2.64046213e-6]` |

A small origin velocity therefore does not establish negligible COM motion.
The first motion pilots715129_0/1 used the wrong COM interpretation. Their
equation solves complete, but their intended physical input is invalid; they
are retained as diagnostics and excluded from corrected motion evidence.
Audit715186 independently rejects these inputs at the original1e-12m/s
origin-velocity check, with maximum component mismatch1.409e-4m/s. This expected
failure verifies that the corrected audit detects the wrong reference point.

## Declared moving control

`--captured-initial-velocity` applies the transported COM velocity and captured
world angular velocity to the current solved body closure. Initial internal
elastic velocity is zero. All nine robot joint velocities match the selected
capture. The closed finger positions differ from the captured rigid positions
by approximately4.77um, so this is initial-velocity matching at an independent
closure, rather than exact pose/strain/history matching.

An independent audit mass-projects the saved nodal velocity onto translations
and current-position rotations using the full consistent mass matrix. It
checks the inferred COM velocity transported back to the original link origin,
the captured angular velocity, all joint velocities and zero internal velocity.
The generation helper is not imported by this check. Corrected8-step pilots
715142_0/1 and independent audit715154 complete all original body/joint/fresh-step
gates. Their initial kinetic energy is9.67755e-9J.

The audit also reconstructs body contact torque and angular momentum about the
current mass COM. Its discrete momentum balance retains the moving-position
transport term, rather than treating endpoint positions as fixed:

```
(L_new - L_old)/dt = torque_contact + torque_gravity
                    + sum(((r_new - r_old)/dt) cross (M v_old)).
```

All original force and moment gates remain unchanged. The complete contact
moment, including regularization effects, stays in the retained output.

## Full-hold attempt and numerical precision diagnostic

Rigid-body job715175_1 completes all80 steps at1.25ms. The corresponding FEM
job715175_0 completes48 accepted steps, then rejects step49 when residual-based
line search stalls at1.00159372e-9N, above the original1e-9N gate. The failed
iterate, all previous states and natural nonzero return code are retained.
No complete80-step moving-feedback result is established by that attempt.

At exactly that rejected state, diagnostic715187 evaluates the unchanged
neo-Hookean Piola force using63-mantissa-bit long-double contractions. The
original and extended force contractions differ by3.51280e-10N in vector norm;
the extended residual is initially1.01640756e-9N. Two further Newton corrections
using the original tangent reach9.95050e-10N, with fresh robot-step residual
3.28739e-11. This recovers one rejected state without changing constitutive law,
quadrature, physical parameters or force gates. It does not qualify a complete
history or prove that this arithmetic change eliminates every later precision
floor. Its original energy/tangent remain FP64.

An isolated extended-force80-step producer715190 and independent audit715192
are scheduled/running. Both other original-arithmetic moving time-step producers
also fail naturally. Their rejected states remain above the original gate:

| dt, ms | Accepted steps | Rejected step/time, s | Rejected body residual, N |
|---|---:|---|---:|
|2.5|22|23 / 0.0575|1.01281709e-9|
|1.25|48|49 / 0.06125|1.00159372e-9|
|0.625|105|106 / 0.06625|1.05118401e-9|

These failures prevent a complete0.1s temporal comparison. Failed downstream
dependencies are explicitly cancelled in the archived journal. No unfinished
retry is counted as accepted evidence.

The zero-initial-velocity comparison baseline is independently re-audited by
715182, using the exact closure arrays selected for the corrected moving
experiments. All160 rigid/deformable states satisfy original budgets. Its
angular-momentum balance residual is at most9.896e-12Nm. Final body angular
velocity has y component0.0262681rad/s for the FEM case, so even this control's
initially zero velocity does not imply a stationary subsequent body.

## Spatial evidence remains unresolved

The50um numerical refinement band preserves the original physical geometry,
material, force protocol and activation distance. Its completed3577-tetrahedron
case715083_1 solves the original64-target closure and80 dynamic steps. The
independent traction audit715124 reconstructs the final saved force exactly.

| Numerical local level | Tetrahedra | Active body nodes | Active lumped area, mm² | Normal force/area maximum, MPa | Raw global peak, MPa |
|---|---:|---:|---:|---:|---:|
|0|480|2|96.5136|0.00828321|0.156447|
|12|3577|2|0.00408089|196.055|280.534097|

Each contact island still contains one active body node. These areas and
densities are unsmoothed collision quadrature diagnostics, rather than resolved
physical footprint areas or a continuum pressure field. The nonlinear
whole-volume lower/upper peak bounds agree within0.000335Pa at level12; that
tight bound concerns the computed field and does not establish continuum
accuracy. Finer levels15/18 remain separate live work. The concentrated-load
diagnosis in [the preceding report](PHYSICAL_CONTACT_LIMIT_20261015.md) is
retained without claiming that the complete nonlinear contact problem has been
proved singular.

## Reproduction and remaining scope

Use `scripts/capture_com_velocity.sh` for the read-only capture observer,
`scripts/run_articulated_franka.sh` and `scripts/run_rigid_egg_franka.sh` with
`--captured-initial-velocity` for the corresponding controls, and
`scripts/audit_rigid_egg_feedback.sh` for the independent balances and load replay.
`scripts/compare_initial_motion.sh` requires array-identical closures before
comparing zero and captured initial velocities. The temporal audit also checks
identical initial-velocity arrays across its integration intervals.

Runtime data use the existing `runs/20261014-physical-validation/` directory.
Original approach, support release, changing actuation and longer motion remain
required, along with credible separated spatial/geometry/contact tails, native
finite-footprint model comparisons, feedback separation and physically verified
warmed costs. Controlled initial-motion evidence does not complete these items.

The immutable `captured-motion-checkpoint-v1` publishes14 terminal runs,
3036 exact file entries and636 source identities. Independent verification
checks every stored artifact against its original runtime bytes. Manifest SHA256:
`1f4925eb5b208238e203cc31a3d76e07c32ec72e2d8f989e10eaff6dcb1c77c9`.
The subsequently finished fine-time-step failure is retained separately in
`motion-failure-supplement-v1`; it is not retroactively added to the first
checkpoint or treated as a successful temporal tail.
