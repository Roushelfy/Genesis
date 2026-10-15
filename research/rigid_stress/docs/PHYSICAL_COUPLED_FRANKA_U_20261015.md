# Solved full-arm contact hold and third U reference level

This checkpoint advances the complete
[physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md).
An independent FEM/contact body and nine-joint Panda now solve a short coupled
hold with the actual finger surfaces, inertia, PD control and joint equality.
The third resolved U level reduces its tensor change to12.2%, still above the
declared3% diagnostic. Neither result establishes full application acceptance.

## Coupled robot and body equations

`ArticulatedContact` uses the independently assembled P2 consistent body mass,
stable neo-Hookean bulk and current unilateral/frictional contact. Actual rigid
compound fingers follow nonlinear forward kinematics of all seven arm joints
and two finger slides. Robot dynamics use the already
[qualified authored Panda skeleton](PHYSICAL_LOCAL_TAIL_ARTICULATED_20261015.md).
All nine authored armature/damping terms, gravity, original finger equality and
captured application PD gains remain. Applied effort caps are
`[87,87,87,87,12,12,12,4,4]` in each joint's native Nm/N unit.

At each time step, nine independent unit-effort MuJoCo steps construct the current
velocity response `v=b+S*tau`. The coupled residuals are the complete FEM body
balance and `S^-1*(v-b)-PD(q,v)+J(q)^T*g_contact`, with
`q=q_previous+dt*v`. Contact, endpoint kinematics and PD forces are updated in
Newton iterations. The full unsymmetric body/joint tangent uses a sparse body
factorization and nine-variable Schur solve. Each accepted state must also
reproduce a fresh complete MuJoCo step at the original effort residual budget.
This old-state robot response with implicit endpoint forces is a declared
integration scheme; temporal evidence remains required.

Initialization closes the jaw through64 targets with six original body gauge
rows held. Both dynamic cases reuse its identical saved closure, release all
body coordinates and start with zero body/joint velocities. The full-arm case
has all nine joint variables free. The explicit support control fixes only the
seven arm joints at closure and records the required external reaction torques;
its two finger slides remain dynamic. Original approach/lift and captured angular
history have not yet been matched.

## Solved hold and independent verification

Pilot714971 solves64 closure targets and8 dynamic steps at1.25ms. Independent
audit715004 reconstructs all8 retained states. Held/current-contact and dynamic
tangent checks decrease to1.81735e-5 and2.08760e-6 relative directional error.
Their declared1e-3 tangent diagnostic is separate from the original force gates.

Array714999 solves80 steps through0.1s for the full-arm and fixed-arm cases.
Audits715005 and715019 independently rebuild normal/current-friction loads,
complete FEM body balance and joint contact efforts from resultant finger
wrenches. The later audit also reconstructs body velocity/acceleration from saved
positions, checks joint integration, independently checks every finger pose,
and retains raw whole-volume peak intervals at every step. It imports no coupled
Newton implementation. All160 accepted states satisfy the original budgets.

| Maximum over80 dynamic states | Full arm | Fixed-arm control |
|---|---:|---:|
| Complete body residual (N) |4.35231e-10|2.99214e-10|
| Joint effort residual (joint Nm/N coordinates) |3.26676e-10|4.78579e-11|
| Fresh robot-step effort residual |3.26811e-10|4.78555e-11|
| Contact reconstruction maximum difference (N) |0|0|
| Explicit seven-arm reaction norm (Nm) |0|0.0424781|
| Objective elastic nodal displacement (m) |3.51699e-7|3.51697e-7|

Both complete body and joint residual limits remain1e-9; fresh independent robot
steps use the same1e-9 effort limit. At0.1s the full-arm maximum joint movement
from closure is6.32166e-6rad, with nonzero arm velocity. Fixed-arm positions and
velocities remain exactly prescribed and their reactions are explicitly applied
in the fresh-step check. Body displacement is about0.0704% of the0.5mm wall.

Across all80 matched times, support-axis differences reach0.0045574% for
volume-weighted objective Cauchy tensors,0.00048475% for elastic energy,
0.115538% for body contact resultant force and0.098349% for its moment.
Forces and moments have separate norms and denominators. Final raw global peak
lower bounds are156447.322Pa/156447.200Pa; the largest peaks anywhere in the
histories are156460.825Pa/156462.849Pa. Peak upper/lower gaps remain about
0.000329Pa. These bound computed fields, not contact or continuum error.
Small support sensitivity in this short, nearly stationary hold does not measure
rigid/deformable feedback or validate the original angular-motion history.

## Failed attempts and numerical repair

Both714984 attempts fail and remain archived. The full-arm solver stops at
step31 with body/joint residuals4.89485e-9/2.78441e-9 above their original limits.
The fixed-arm attempt reaches step69's fresh robot-step assertion after its
Newton residuals pass. Its old driver saved fields only through68; no saved
failed69 state is claimed. Later drivers preserve fresh-step failures explicitly.

The successful version evaluates the same compiled joint tree in centered
`longdouble` arithmetic before casting FEM nodal displacements/Jacobians to FP64,
and tightens MuJoCo's internal solver tolerance from1e-8 to1e-14. Every evaluated
pose is independently checked against MuJoCo at1e-12 absolute tolerance; the
largest observed positional discrepancy in the full hold is1.034e-15m.
These changes leave physical inputs and outer force budgets fixed. Their combined
repair is verified; this checkpoint does not separately attribute the two failures
to one change or claim a higher-precision physical reference than the residuals.

## Three resolved U levels

Original job714003_0 completes its unchanged free-body force-controlled protocol
in7h36m. Audit715016 reconstructs all three original linear body fields, raw
four-corner stresses, complete body balance and75 unchanged physical probes.
All original accepted-step residuals pass1e-9, including744/598/562 records.

| Body tetrahedra / pad grid | Raw global peak (kPa) | Body elastic energy (J) |
|---|---:|---:|
|845 /5|112.410639|4.38267944e-8|
|2339 /7|100.653754|4.07383026e-8|
|5669 /9|95.544657|4.00580795e-8|

| Adjacent change, normalized by finer result |845→2339|2339→5669|
|---|---:|---:|
| Complete probe tensor |44.5788%|12.2066%|
| Probe von Mises |18.8720%|5.15818%|
| Gauge displacement |9.16828%|2.55186%|
| Raw global peak |11.6805%|5.34734%|
| Body energy |7.58130%|1.69809%|

The finest raw peak remains on the inner bend, at
`[0.001455524,-0.004795046,0.000257711]`m. Body geometry and pad discretization
change together along this curve. The final level uses the previously qualified
common-x gauge/cache numerical option; loads and physical constraints remain
fixed. Raw-coordinate and gauge-projected force reconstructions both satisfy
the original1e-9N budget. The first audit715011 failed an additional1e-12
comparison between those differently rounded arithmetic forms. Corrected715016
retains that componentwise check in a consistent gauge and separately verifies
raw-coordinate balance. The failed audit and exact source remain visible.

The tensor, von Mises and raw peak tails do not pass3%. Linear pad/body laws,
finite-rotation assumptions and combined geometry/contact changes remain explicit;
this curve cannot establish the nonlinear physical-contact operating range.

## Reproduction, evidence and next work

Use `scripts/run_articulated_franka.sh RUN --capture CAPTURE --closure-run PILOT
--dt 0.00125 --steps 80`. Add `--fixed-arm`
for the support control. `scripts/audit_articulated_contact.sh RUN --run FULL
FIXED` independently verifies all saved states.
`scripts/audit_resolved_u_tail.sh RUN --run COARSE MIDDLE FINE` audits the U curve.
Runs require the established allocated CPU/container environment and physical
validation venv. Runtime outputs belong under the project data root.

The [immutable checkpoint](../evidence/20261014-physical/articulated-u-checkpoint-v1/manifest.json)
contains13 terminal runs, including three failures,1083 exact original files
and285 checked input/executable source identities. The manifest SHA256 is
`732f4e04567290c6871947940681eb6a76761d614f30e6f1cd28481463c1429a`.
Its adjacent verification checks exact original bytes and archived inputs.
Earlier capture provenance remains in the linked articulated-spatial dependency;
the selected actual geometry/input files are included here. Exact commands,
package/hardware identities and Slurm terminal states accompany the evidence.

Array715012 schedules40/80/160 steps at2.5/1.25/0.625ms through the same0.1s
and byte-identical closure arrays. The authored equality time constant stays5ms;
a guard prevents MuJoCo's `2*dt` clamp from changing it. Its common-time audit715017
is dependent work, not yet accepted evidence. U fourth-level job715021 schedules
16146 tetrahedra and pad grid11 with the qualified current-Newton preconditioner.
Live studies are excluded from this checkpoint. Physical spatial/contact/
regularization and original-history tails, actual original egg/Franka matching,
resolved-load replay versus feedback, applicable model comparisons, warmed native
accuracy/cost and all original A/B/C deliverables remain open.
