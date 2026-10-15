# Complete original protocol and stationary spatial histories

The [full physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. The original330-control,3.3s deformable reference now closes
naturally with2640 consecutive physical states and independent force/control
audits. A separate four-mesh stationary comparison now spans all80 physical
states through0.1s on each mesh. Its last spatial pair still changes tensor
15.52%, energy9.15% and raw lower peak45.42%; local finite strain remains large.
These completed histories establish their stated temporal coverage, without
establishing spatial/contact convergence or rigid/deformable feedback accuracy.
The predeclared [execution protocol](PHYSICAL_VALIDATION_PROTOCOL_20261014.md)
retains the initial3% physical tail target for peak, energy and common-probe
stress norms, with at least three useful levels and a stable hotspot region.

## Original approach, support, grasp and lift

Jobs716378–716382 are terminalCOMPLETED with natural producer exit and matching
before/after source identities. The independent audit checks all2640 states
against the original330 approach/lift targets, finger effort caps, initial
zero body velocity, actual ground support and fresh old-state Panda steps.
The timestep remains1.25ms, with eight physical substeps per10ms macro control.
The original480-tet polyhedral egg, E=10GPa, Poisson ratio0.3, density2000kg/m³,
represented mass0.006529848210401328kg, friction and barrier law remain fixed.
The separately measured continuum-geometry/mass deficit is not corrected by
changing density in this case.

| Verified observable | Result |
|---|---:|
| Original macro controls / physical states |330 /2640|
| Physical horizon |3.3s|
| Maximum independently assembled body residual |9.922684454e-10N|
| Maximum fresh actual robot effort residual |8.124247982e-10|
| Whole-history raw global peak lower/upper |1068730.6968747915 /1068730.6968749515Pa|
| Actual native resolved-load replays |14 at seven times, two inertia inputs|
| Maximum native/full-reference tensor relative L2 |1.501098796e-10|
| Maximum native complete elastic budget fraction |0.04740965|
| Saved replay snapshots / interior-and-corner samples |334 /11062080|

The force gates remain1e-9 in their original units. There are696 explicitly
recorded misses of the stricter internal1e-10 Newton target that pass the
original physical body and fresh actual robot gates. This count uses explicit
recorded flags; missing historical flags are not inferred to be misses.
Numerical recovery stages are never counted as physical time steps.

All2640 states retain independently reconstructed nonlinear global fields,
objective-frame data and both full-inertia and rigid-inertia-relief linear
replays. The whole-volume upper bounds use independently reconstructed
Bernstein certificates. Interior/corner sampling checks those certificates;
sampling alone is not a global-maximum proof. Near-zero startup tensor relative
errors can approach one while absolute discrepancies are about1.79e-5Pa;
absolute errors remain available and must accompany those relative numbers.
The final producer scalar near156445Pa is not the whole-history maximum.

The actual CUDA replays use resolved nodal loads to isolate elastic algebra.
They do not establish native finite-footprint model accuracy, deformation
feedback or physically verified warmed production cost. Some legacy producer
and audit prose still calls the run a prefix. Completion of this scoped history
is instead established by the explicit completion flags, gapless2640-state
counts,330 matching controls, terminal scheduler state and independent audit.

## Four stationary histories through0.1s

Jobs716331/716332 complete the finest30298-tet history and its separate all-state
audit. The byte-identical held initializer is retained. All five previously
saved original prefix states pass the declared complete-array and full
corner/quadrature comparisons; the remaining75 are newly solved physical
states. No equivalence against nonexistent original states is claimed.

Job717369 compares the480/3577/8433/30298-tet histories at10/20/…/100ms,
retaining40 independently reconstructed full fields and2400 identical
physical probes. All four histories have80 consecutive saved states and
identical physical configurations after excluding explicit refinement/output
metadata. Independently integrated volume, mass at unchanged density and COM
agree: volume3.264924105200664e-6m³ and mass0.006529848210401329kg. This is the
fixed original polyhedron, rather than the separate CAD-geometry sequence.

The finest80 states pass independent body and mapped force gates, with maxima
4.842437643e-11N and8.803253547e-11N. Each of the40 matched field states also
passes both independent gates. Four times80 saved states is not a claim that
this checkpoint independently audits all320 states.

| Final100ms change,8433→30298 tetrahedra | Finer-normalized change |
|---|---:|
| Common-point full tensor |15.5218821%|
| Common-point von Mises |4.5313796%|
| Objective displacement |2.3578201%|
| Nonlinear energy |9.1458761%|
| Raw global lower peak |45.4201437%|

Across all ten common times, the last-pair tensor change remains
15.5218821–15.5218967%, energy9.1458761–9.3370838% and raw lower peak
44.7236121–46.0896347%. Extending the complete time window therefore does not
resolve the spatial discrepancy observed in the earlier6.25ms comparison.

At100ms the finest raw whole-volume peak interval is
3.705902719–5.812471033GPa; over all80 finest states it reaches
3.793462911–5.943853804GPa. The observed maximum corner Green–Lagrange strain
norm is0.451305 at100ms, while objective nodal displacement is only4.49694µm,
or0.899388% of the0.5mm wall. Small wall-relative displacement therefore does
not establish local small strain. These are observed corner strains, not
certified whole-volume strain maxima.

The100ms peak reference location is
(-2.268348685e-5,-0.0219618335450,0.00208660682403)m. An independent direct
coordinate check puts the peak0.0339470m from the nearest of the six held
initializer points. The observed peak is not located at those pins. Growing
concentration, active contact counts or these distances alone do not prove a
contact singularity; contact law, geometry, integration and field regularity
still require independent diagnosis.

## Matched rigid counterpart remains under qualification

`run_original_rigid_franka.py` retains the same initial geometry, consistent
mass, ground, full Panda model, original controls, contact/friction law,
regularization and timestep, with the egg constrained to six finite-rotation
coordinates. Its acceptance uses actual net force and moment/radius, plus a
fresh actual old-state Panda step. Elastic nodal residuals are not substituted
for the constrained rigid body's six equilibrium equations.

Pilot717371 saves856 accepted states through1.07s before its first finger
approach failure. Independent auditor717438 reconstructs all856 states without
importing the coupled rigid solver. Maximum body force-equivalent, actual net
moment and actual robot effort residuals are3.840549523e-10N,
3.762809761e-18Nm and9.881475567e-10. The failed pilot remains incomplete.

Recovery717450 finds small original residual roots for step857, but all three
candidate endpoints intersect the left finger and egg. Independent CCD
diagnosis717488 locates exactly one rejected face–vertex candidate, involving
egg vertex36 and left-finger face773. Default safe fraction0.4296875 tends to
0.4340186 under tighter geometric CCD tolerances; each endpoint intersects.
Those candidates remain rejected. The measured robot tangent agrees with
central finite differences to about5e-10 relative, so this evidence does not
support a robot-tangent defect as the cause.

The velocity prediction can itself enter contact before Newton starts.
A collision-qualified numerical initializer is now used in the new rigid
reference only; it scales the numerical starting velocity and preserves the
actual previous state, timestep, controls, inertia predictor and equations.
Recovery717491 then finds a nonintersecting original-equation root. Its
single-query default CCD fraction is0.859375 and approaches0.868880 with
tighter CCD tolerances; it is still rejected by that recovery attempt.
The [pinned IPC1.6.0 source](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/ccd/tight_inclusion_ccd.cpp#L29-L66)
shows that default CCD initially protects an effective separation based on the
query's initial distance. A returned fraction below one therefore does not
alone prove that the endpoint or entire trajectory intersects.

Recovery717500 certifies that same straight vertex trajectory using four
contiguous default CCD queries. Every interval returns1.0; endpoints are also
intersection-free. Neither the trajectory, physical timestep, original force
equation, CCD tolerance nor conservative-rescaling parameter is changed.
These are geometric verification intervals, not additional dynamic substeps.
Independent auditor717501 separately reconstructs all857 states and that
motion path. At857, actual body force-equivalent and fresh robot effort
residuals are1.694700124e-10N and8.475112582e-12; actual net moment residual is
2.229062402e-12Nm. All856 retained prefix files remain byte-identical. The
maximum body/robot residuals over all857 states remain3.840549523e-10N and
9.881475567e-10. Code0 for this recovery is not completion of880 requested
states or2640 full-original states.

Continuation717502 retains that qualified prefix and saves eight further
accepted physical states before Newton stalls at866. Its first active-contact
tangent check reaches8.16233e-7 relative difference at the smallest declared
perturbation. Independent audit717589 now passes all865 states. Recovery717590
retains its realFAILED1:0: direct Newton, robot-only roots and full15-variable
trust-region attempts do not pass the original physical residuals. Its
success-only audit717591 never starts and is cancelled. Numerical residual
continuation717624 now terminatesFAILED1:0 from the exact865 anchor. It retains
56 numerical stages and13362 original-residual evaluations, reaching artificial
parameter0.3074008636 without reaching the original equation at parameter1.
Its success-only independent audit717625 never starts and is cancelled. The
intermediate equations remove a recorded artificial residual offset; only zero
final offset, all original physical gates and motion checks could qualify a
recovered state. No intermediate stage is a physical time step, and zero new
physical states are accepted. This failure does not prove nonexistence of an
original-equation solution. The full rigid continuation remains incomplete,
and no rigid/deformable feedback result is accepted yet.

The separate171228-tet corrected-contact job717295 also terminatesFAILED1:0:
SuperLU reports `MemoryError` in its first held-initializer Newton factorization.
Slurm records peak RSS89576316KiB with256GiB requested. This is a numerical
resource failure, not evidence of physical convergence or a contact singularity.
Before/after source closure and the real failure are retained; no initializer
or dynamic history from this attempt is accepted. Its unstarted success-only
audits are cancelled. Finer body/contact resolution remains required.

## Fixed-pad U body and geometry axis

Producer716333 and independent final-field audit716334 complete naturally.
The845/2339/5669-tet U bodies use identical resolution5 pad geometry, verified
by separate left/right geometry-array hashes. Both pads retain384 tetrahedra,
their original material/compliance, force0.05N per shoe, friction0.8,
gravity(0,-9.81,0)m/s²,2.5µm barrier distance and64 force increments.
All body material parameters remain E=10GPa, Poisson ratio0.3 and
density2000kg/m³. The independent reconstruction checks every recorded
original accepted-stage residual and the complete final constrained/body
equilibrium, plus all original final corner stresses and global bounds.
Recorded numerical stages are not asserted to be additional physical times.

| Body tetrahedra | Represented mass (kg) | Raw global lower/upper (Pa) | Body energy (J) |
|---:|---:|---:|---:|
|845|0.003987816969|112413.887877 /112414.743122|4.383008777e-8|
|2339|0.004185568092|100634.579990 /100634.990467|4.046305948e-8|
|5669|0.004265647367|95606.439058 /95606.780009|4.007763417e-8|

Represented body mass changes because these are different geometry/mesh
approximations at fixed density. This axis isolates the pads, not volume
discretization from surface-geometry approximation.

| Adjacent body levels | Tensor (%) | von Mises (%) | Objective displacement (%) | Energy (%) | Raw lower peak (%) |
|---|---:|---:|---:|---:|---:|
|845→2339|44.864985|19.095150|8.940356|8.321240|11.705030|
|2339→5669|12.327608|5.125018|2.175467|0.961697|5.259207|

Tensor/von Mises/displacement comparisons use75 identical material probes,
with all probes inside every body. Relative changes normalize by the finer
field/value. Full unsmoothed fields and raw global extrema are retained.
The finest final body/mapped residuals are4.281938231e-11N and4.441689696e-11N;
all616 original recorded accepted stages stay below1e-9N. Its body whole-volume
J lower bound is0.9999951825 and objective displacement1.108606µm.

The finest raw peak is at(0.001455523921,-0.004795046354,0.000257710854)m,
different from the medium-mesh hotspot. Three useful levels and a sub-percent
energy change therefore still do not establish physical convergence: tensor,
von Mises, raw peak and hotspot stability remain unresolved. The separately
running fixed-body pad9 and coupled finer references address different axes;
these measured changes are not an additive decomposition of their errors.

## Immutable evidence and reproduction

Runtime root is
`$RIGID_STRESS_DATA_ROOT/runs/20261014-physical-validation`. Complete source,
configuration, all accepted states, field arrays, source identity records,
versions/hardware and exact Slurm commands are archived. Counts below are
per-packet checks rather than unique experiments.

| Packet | Manifest SHA256 | Entries / actual originals verified | Source identity records |
|---|---|---:|---:|
|[complete-original-history-v1](../evidence/20261014-physical/complete-original-history-v1/manifest.json)|`d2fe098b040884cb736371858e281f3bab3854ee0f7e361f87d3112dc884d8fa`|9972 /7332|222|
|[complete-spatial-histories-v1](../evidence/20261014-physical/complete-spatial-histories-v1/manifest.json)|`ca66aab42c6a9a391a26bb30778562e6697db6045ae419e6dc9a55539b8b4dd2`|926 /926|342|
|[original-rigid-qualification-v1](../evidence/20261014-physical/original-rigid-qualification-v1/manifest.json)|`518737c3fd8a20eb8e48eb6e9ff453ddc95ff103be94419fe2219d84c3c5039e`|3935 /3935|390|
|[closed-physical-followup-v1](../evidence/20261014-physical/closed-physical-followup-v1/manifest.json)|`57cce4b12ef1574a8d5479a96718ac6d60a2301ff95acb642659b4d998465239`|2252 /2245|322|

The runtime verifier checks actual producer bytes as well as publication
staging. The original-history packet includes repeated entries referencing
the same physical states; its7332 originals are deduplicated by actual path.
No scientific state is reduced or omitted.

Reusable entry points are:

```bash
bash research/rigid_stress/scripts/compare_complete_spatial_histories.sh FRESH_RUN --run FOUR_CLOSED_RUNS --fine-audit FINE_ALL_STATE_AUDIT
bash research/rigid_stress/scripts/run_original_rigid_franka.sh FRESH_RUN --matched-deformable CLOSED_ORIGINAL_HISTORY --matched-audit ORIGINAL_ALL_STATE_AUDIT --macro-steps 330 --check-jacobian
bash research/rigid_stress/scripts/audit_original_rigid_franka.sh FRESH_RUN --run CLOSED_RIGID_HISTORY
python3 -m research.rigid_stress.scripts.publish_complete_original_history --runtime-root RUNTIME_RUNS --destination-root NEW_ARCHIVE_ROOT
python3 -m research.rigid_stress.scripts.publish_complete_spatial_histories --runtime-root RUNTIME_RUNS --destination-root NEW_ARCHIVE_ROOT
```

The exact original producer, field, native replay and certificate commands
are retained in the first packet's job command record. Failed/incomplete
prefix auditing explicitly requires `--allow-incomplete-prefix`; a successful
audit of a prefix does not turn it into a complete requested history.

Remaining acceptance includes explained spatial/geometry/contact/time and
regularization tails, the converged egg/Franka and nonconvex interior-hotspot
references, applicable load-model sensitivities, resolved-load native replay
versus same-law rigid/deformable feedback, and warmed native costs at physically
verified accuracy. Production Quadrants kernels and rigid dynamics are unchanged.
