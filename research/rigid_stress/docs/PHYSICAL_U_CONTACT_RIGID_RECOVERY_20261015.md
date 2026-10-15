# U contact traces and original rigid step2129 recovery

The [complete physical-validation objective](LOAD_MODEL_PHYSICAL_VALIDATION.md)
remains active. Three coupled nonlinear U levels now have independently checked
final fields, but their last tensor change is12.21%. A separate normal-contact
audit shows why the fixed-body pad sequence's1.20% tensor change cannot establish
contact-pressure convergence: its left maximum discrete force density still
increases25.48%. The original rigid counterpart independently qualifies2129
physical states through2.66125s after recovering the unchanged step2129 equation.
This checkpoint establishes neither physical convergence nor the full2640-state
rigid application.

## Three coupled U levels retain a spatial discrepancy

Jobs716308/716316 close the5669-tet/pad9 solve and independent three-level field
comparison. All cases retain the original force-controlled0.05N per shoe,
gravity[0,-9.81,0]m/s², friction0.8,2.5µm barrier distance,1mm pad thickness,
100µm edge lift and64 force increments. The body remains Neo-Hookean with
E10GPa, Poisson0.3 and density2000kg/m³; shoes retain E1MPa, Poisson0 and
density1000kg/m³. Materials are not adjusted to equalize represented mass.

| Coupled level | Body tetrahedra | Pad resolution | Raw global lower/upper peak, Pa |
|---|---:|---:|---:|
| 1 |845|5|112413.887877 /112414.743122|
| 2 |2339|7|100650.015242 /100650.427385|
| 3 |5669|9|95543.837903 /95544.178410|

At75 common material probes, the2339→5669 pair changes full tensor12.206254%,
von Mises5.156873%, objective displacement2.550669%, nonlinear body energy
1.694397% and raw lower peak5.344329%. The finest independent complete body/
original mapped residuals are3.08918e-10/6.11152e-10N. Every one of its562
retained accepted-stage residual records passes the original1e-9N gate.
Only final complete fields were saved; these records do not reconstruct missing
friction histories.

The finest raw hotspot is[0.001455523921,-0.004795046354,0.000257710854]m in
reference coordinates, away from the shoe centers at y0.018m. It moves between
levels. The reported peak comes from the full body, with independent nonlinear
whole-volume upper bounds. Neither a narrow lower/upper interval on one mesh
nor a small displacement supplies spatial convergence. Body approximation,
represented mass and pad geometry/contact resolution change together here.

## Independently rebuilt final normal contact exposes an unresolved trace axis

Job718505 reconstructs the original area-weighted IPC normal collisions on all
six distinct closed final states covered by the coupled, fixed-body pad and
fixed-pad body audits. Actual saved rest positions and surface triangles must
match exactly, endpoints must be intersection-free, and the sum of all individual
normal-stencil forces must reproduce the full normal gradient. Maximum absolute
pullback difference is1.73473e-18N. Both physical shoe sources retain their body
force/moment, counterbody balance, full nodal arrays and stencil types.

The fixed2339-tet body and mass are identical for the following pad sequence:

| Pad resolution | Normal stencils | Left/right active lumped area, mm² | Left/right max discrete normal density, kPa | Left/right density L2, N/m |
|---|---:|---:|---:|---:|
|5|15|2.062428 /2.975627|44.959253 /27.505767|40.147387 /33.678239|
|7|37|5.343245 /5.106581|21.442886 /22.542173|27.940897 /29.767660|
|9|47|4.659111 /4.682340|26.906560 /21.757288|29.101754 /28.095412|

Here each density is the magnitude of the actual nodal normal force divided by
its incident current-triangle lumped area; its L2 is the square root of the sum
of nodal force magnitude squared divided by that area. These are discrete
diagnostics, not continuum pressure or measured continuum contact area. All
forces contribute to norms. A fixed1e-8N threshold affects only active counts
and area reporting.

For pad7→9, left maximum density increases25.480122%, while left/right normal
density L2 changes+4.154688%/-5.617668% and active lumped area changes
-12.803723%/-8.307732%. The already audited1.195351% tensor and0.017624% raw
peak changes therefore establish only conditional fixed-body field stability.
They do not establish a converged pressure field or the complete U reference.
The fixed-pad body/geometry axis separately retains12.33% tensor change.
These coupled/separated norms are not an additive error decomposition.

Final shear is retained as the saved total force minus the independently rebuilt
normal force. The old successful solves did not save their previous friction
states, so this subtraction is explicitly not an independent friction-law audit.
Disjoint body-node stencil supports permit exact attribution to each physical
shoe; ambiguous attribution is rejected. Shoe identities are not established
continuum contact islands. Normal action/reaction moments are about the same
declared reference COM; finite-gap shear moments remain measured outputs.

## Original step2129 is recovered at the unchanged physical gates

Full rigid continuation717843 saves2128 physical states before failure at2129.
Independent audit718517 passes all2128 original body force/moment, actual Panda
step, contact pullback and original straight-path CCD checks. Its largest body
force-equivalent/fresh robot residuals are9.95861e-10/9.88148e-10. The wrapper's
actual return code is1; no missing full-history summary is invented.

The frozen-friction inner solve stalls at1.010763e-9N body residual, just above
its nominal1e-9 threshold. The isolated
[reference patch](../inexact_lagged_rigid_inner.patch) permits a stalled frozen
inner iterate below1e-8 to seed the next outer numerical iteration. This changes
no contact equation, physical timestep, old state, control, predictor or final
physical gate. Its first true current-friction residual is9.14046e-6N and is
correctly rejected as a physical state.

Recovery718523 takes26 outer numerical stages, with only the first using the
inexact frozen seed. Final acceptance restores the original current friction,
requires the original1e-9 gates, verifies a fresh actual robot step and certifies
the same physical straight vertex paths with default CCD. The26 stages do not
advance physical time. Independent audit718524 checks all2129 physical states:

| Recovered physical state2129 | Independently reconstructed value |
|---|---:|
| Physical time |2.66125s|
| Body force-equivalent residual |3.628109e-10N|
| Body net force residual |9.117590e-11N|
| Body net moment residual |1.110955e-11Nm|
| Fresh actual robot effort residual |4.701612e-11|
| Contact pullback maximum difference |0N|
| Default CCD intervals for unchanged straight path |1|

All2128 prefix files remain byte-identical. Full continuation718525 starts only
after this independent qualification and exact patched-helper source matching.
Full audit718526, full same-law feedback718527 and actual native resolved-load
replay718528 are success-dependent work, not completed evidence here. Failed
producer717843's three unstarted success-only dependents were cancelled and their
original accounting/commands retained before replacement.

## Reproduction and retained scope

Opt-in scripts are
[the U contact audit](../scripts/audit_nonlinear_u_contact_trace.sh),
[the isolated rigid recovery/continuation wrapper](../scripts/run_inexact_lagged_rigid.sh)
and the existing independent original-history auditor. Scientific runs use the
recorded Slurm/container environment and data-side output paths. Exact submitted
commands, dependencies, versions, hardware, original source hashes, successful
results and actual failure records are retained with the evidence.

- [U contact evidence](../evidence/20261014-physical/u-contact-trace-v1/manifest.json)
  contains all six producers, three field audits and the new contact audit,
  including every raw final field and force/area array.
- [Rigid recovery evidence](../evidence/20261014-physical/rigid-inexact-recovery-v1/manifest.json)
  contains the failed full continuation, all2128-state independent audit,
  recovered2129 prefix and its independent audit, with numerical stages separate.

Independent runtime verification passes all493/9025 archive entries against
448/9025 actual runtime originals respectively. The U manifest SHA256 is
`40d7a10955d53f100b15798c652edee389ebd06a8e2bf9ca84a71e1ec79d07ef`;
the rigid-recovery manifest SHA256 is
`b3716f0e0164ef3df92651a935e91e48d0bea6f1d13e13d4c164afc143478152`.
Counts include source/provenance files and are not numbers of experiments.

The checkpoint still leaves the full A/B/C objective open: credible body,
geometry/contact and regularization/history tails; applicable finite-traction
load-model sensitivity and local redistribution; physically converged full
egg/Franka and nonconvex interior-hotspot references; full feedback versus native
resolved replay; and warmed costs at verified physical accuracy. Production
Quadrants and rigid dynamics are unchanged.
