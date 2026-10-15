# Original-step recovery and fourth resolved U level

This advances the complete [physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
Original step866 now satisfies an independent reconstruction at the original
1.25ms time step. The next direct attempt rejects869, subsequently recovered by
fixed-dt residual continuation and independently audited. Four resolved U levels
reduce tensor changes to7.38%, still above the declared3% diagnostic. Neither
result completes original history or physical spatial/contact acceptance.

## Original equation recovered without accepting pseudo-history

Job715762 holds the physical previous state865, original controller target,
effort caps, geometry, mass, material, friction, barrier and contact distance fixed.
It uses shorter pseudo-timesteps only to initialize Newton at the original
dt=0.00125s. Accepted pseudo-fractions are0.0625,0.15625,0.296875,0.40234375,
0.560546875,0.7978515625 and1. A rejected intermediate attempt remains saved.
These states are solver iterates, not extra steps in the physical trajectory.
Only the final original equation supplies recovered physical state866.

Independent audit715772 reconstructs every original state1–865 and the recovered
866 using the same unchanged actual controls and previous physical states.
It rebuilds current normal/friction forces, body bulk/inertia balance, complete
finger kinematics and the actual nine-joint MuJoCo step.

| Independent check at866, t=1.0825s | Result |
|---|---:|
| Complete body dynamic residual |8.28893e-11N|
| Actual robot-step effort residual |8.89253e-11 in joint Nm/N units|
| Maximum contact pullback difference |0N|
| Active opposing groups |left and right fingers|
| Ground reaction |0N|

All866 states meet the unchanged1e-9 budgets. Maximum body/robot residuals across
the entire audit are9.88327e-11/9.91075e-11. This verifies the recovered equation;
it does not establish uniqueness, time convergence or physical contact accuracy.

`run_original_franka_prefix.py` can resume only after a successful independent
recovery audit. It verifies the same capture/protocol/material configuration,
copies accepted physical states byte for byte, checks their hashes, and records
each source path and hash. Pseudo-stages are excluded. Producer715775 accepts
867 and868, then rejects869 with body/joint residuals0.347398/0.144919. The
original3.3s approach/grasp/lift is therefore still incomplete.

`globalized_original_franka.py` applies the same pseudo-timestep strategy after
a rejected direct solve, requiring the original endpoint equation, exact original
predictor/previous joint state/targets/caps and fresh robot-step checks. It retains
failed iterates and all pseudo-stage evidence separately. Its full-history job
715800 subsequently rejects869: even its0.0625 pseudo-step stops at1.334e-10N,
above the stricter internal1e-10 target, and smaller pseudo-dt attempts also fail.
It accepts no new physical states. No tolerance or physical parameter is relaxed.

The fixed-dt `continue_original_residual.py` instead continues
`F(z) - (1-lambda) F(z0) = 0`, where `z0` is the previous accepted pose with
zero endpoint joint velocity. The previous physical state, original dt, contact
history, target and caps stay fixed; the defect is a numerical globalization
device. Intermediate lambda states do not satisfy the original dynamics and are
never accepted as physical-history steps. At lambda1 the defect is exactly
removed, and all original body/joint/fresh-step gates apply.

Job715811 recovers original869 with body residual4.18838e-11N and fresh robot
effort residual4.02186e-11. Independent audit715813 reconstructs all869 physical
states including867/868 and recovered869, with zero contact-pullback difference
at869. It confirms the original equation rather than the artificial continuation
equations. `residual_globalized_original.py` supplies this fallback after later
direct Newton rejections. Full3.3s continuation715815 and its independently
scheduled terminal audit715817 remain live/pending, not completed evidence.

## Four resolved linear U levels

Producer715021 completes the original cold force-controlled U protocol with
16146 body tetrahedra and pad grid11. Independent audit715776 reconstructs all
four body fields, raw element-corner stresses,75 unchanged physical probes,
raw-coordinate and gauge-projected body balances, and every retained original
accepted-step residual. The finest531 accepted-step records all meet1e-9N;
their maximum is9.97745e-10N. Its final raw-coordinate body balance is7.750e-12N.

| Body tetrahedra / pad grid | Raw global peak (kPa) | Body energy (J) |
|---|---:|---:|
|845 /5|112.410639|4.38267944e-8|
|2339 /7|100.653754|4.07383026e-8|
|5669 /9|95.544657|4.00580795e-8|
|16146 /11|95.630777|3.94841313e-8|

| Adjacent change, normalized by finer result |845→2339|2339→5669|5669→16146|
|---|---:|---:|---:|
| Full tensor at common probes |44.5788%|12.2066%|7.37687%|
| Probe von Mises |18.8720%|5.15818%|2.89370%|
| Gauge displacement |9.16828%|2.55186%|4.06743%|
| Raw global peak |11.6805%|5.34734%|0.0900543%|
| Body energy |7.58130%|1.69809%|1.45362%|

The peak stays on the inner bend, but its physical location changes sides to
`[-0.001125786,-0.004889939,-0.000326847]`m. Peak stability does not close the
remaining tensor/displacement tail. Body geometry and pad discretization change
together; nonlinear compliance/finite rotation and separated physical axes remain
independent requirements. A fifth40824-tet/pad13 cold reference is running as
715794, without changing the physical geometry definition or loading law.

## Unsmoothed traction and nonlinear initialization evidence

Audit715758 retains exactly two active body nodes on each of the480/3840/30720
uniform egg meshes. Their lumped support areas are96.5136/24.1282/6.03194mm²,
with maximum normal force/area8.283/33.131/132.514kPa and zero force-pullback
error. This is unresolved concentration, not a resolved finite contact footprint.
The separate common-point tensor audit715757 is still live. Aggregate force or
moment stability does not replace its full-field comparison.

Finer all-part neo-Hookean U attempts715781_1/2 reject a nonpositive determinant
while evaluating the first boundary-only approach iterate, before producing a
qualified contact solution. These failures remain visible. The initializer first
moves back-plane nodes while the interior stays at rest; its Newton guess can
compress or invert a thin terminal layer even in a collision-free approach.

The isolated `rigid_nonlinear_approach.patch` initializes that contact-free solve
with each shoe's prescribed rigid translation, already qualified by the existing
CCD fraction. Body and shoe laws, final prescribed boundaries, contact protocol
and force budgets stay fixed. New source snapshots715802_0/1/2 test this numerical
initialization at the same three mesh/pad levels. The original coarse cached
control715781_0 and its final-field equivalence715792 remain separate. All these
live runs are unverified here; further source/force/field and history checks are
required before interpreting their physical tails.

## Reproduction and remaining scope

Use `scripts/continue_original_contact_state.sh` for rejected-state continuation,
`scripts/audit_original_franka_prefix.sh --allow-incomplete-prefix --recovery-run`
for its independent original-history audit, and
`scripts/run_original_franka_prefix.sh --resume-run --recovery-run --recovery-audit`
to resume only from verified physical states. Runtime argument files and Slurm
commands specify full values. `scripts/audit_resolved_u_tail.sh --run ...` accepts
both original `solution/` and cached `results/` producers. The tracked numerical
patches apply only to hashed executable snapshots, preserving live canonical
reference sources.

The terminal cases are archived by
`scripts/publish_contact_recovery_checkpoint.sh`:8 runs,2153 file entries and274
source-hash records. Manifest SHA256:
`49652e19905f08e969bda12e14d0e907beac93a3f97b927f548cff6babc08659`.
Every stored entry is checked against its complete original runtime bytes.
Live results are excluded.
The subsequent fixed-dt recovery, its independent audit and failed pseudo-dt
history are separately retained in `residual-recovery-checkpoint-v1`:3 runs,
1929 entries and132 source-hash records. Manifest SHA256:
`4115f62a21116c08ab4fe87b406ab91468115e90a6cb389c300ef670710a2aee`.
They are not part of the earlier manifest. Reproduce this archive with
`scripts/publish_residual_recovery_checkpoint.sh`.
Complete original history, physical egg/U contact/geometry/volume tails,
remaining applicable load-model and feedback comparisons, and warmed native
costs at verified physical accuracy remain open. All original A/B/C coverage,
raw global extrema and reproducibility requirements remain in force.
