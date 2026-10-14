# Physical error separation and force-control progress

The [complete physical goal](LOAD_MODEL_PHYSICAL_VALIDATION.md) remains active.
This checkpoint separates several sources of error that the initial joint
mesh/geometry curves could not distinguish. Production engine `f63d0931` is
unchanged; the new contact work is an independent reference prototype.

## Frozen geometry versus volume resolution

Scikit-fem12.0.2 uniformly refines the seed tetrahedra with edge midpoints;
there is no CAD projection. See the package's
[refinement API](https://scikit-fem.readthedocs.io/en/stable/api.html).
Volume, boundary area and centre of mass remain equal to the seed within the
recorded1e-12 relative checks. Both studies retain five fixed continuum loads,
physical radii/friction and common physical probes. Separate quadrature checks
bound the prescribed-traction integration error.

| Opposing load | Tetrahedra | Raw global peak | Energy | Adjacent full-tensor probe change |
|---|---:|---:|---:|---:|
| Frozen ellipsoid |316|4.863kPa|5.246e-11J|—|
| Frozen ellipsoid |2528|6.446kPa|6.227e-11J|16.21%|
| Frozen ellipsoid |20224|6.329kPa|6.553e-11J|6.235%|
| Frozen U |637|140.542kPa|4.574e-8J|—|
| Frozen U |5096|169.623kPa|4.639e-8J|13.74%|
| Frozen U |40768|205.414kPa|4.665e-8J|5.325%|

The U raw peak grows while energy changes become much smaller: the last energy
change is0.567%. Every reported peak lies on a reentrant edge of the seed
polyhedron. The coarse and finest hotspots return to the same31.63-degree
crease near the inner bend, about18mm from the contact centres. The intermediate
peak is on another6.58-degree seed crease. Distances to those edges are recorded
without hotspot smoothing. This is evidence of polygonal geometry sensitivity;
it does not prove convergence of the intended smooth CAD geometry or a limiting
peak. The independently generated CAD curve decreases over its initial levels,
whereas the frozen polygonal peak grows.

All30 reference fields satisfy the original complete equilibrium budget. The
two studies still fail their complete initial3% tail diagnostic. The next CAD
meshes are independently generated for all seven shapes:21 additional assets,
including U5669/16146/40824 tetrahedra and full egg51768/114228/200892 tetrahedra.
Their21 five-load reference sweeps are submitted as job array712795 and must be
polled to authoritative completion. Long references remain opt-in.

## Reference integration and native numerical axes

Independent triangle subdivision levels2/3/4 vary the prescribed-traction
quadrature on fixed U/capsule meshes. The largest level3-to4 whole-field tensor
change across five loads is4.43e-9 for U and9.80e-8 for capsule. Nodal-force
changes are3.28e-8 and1.17e-7. These are much smaller than the observed volume
and geometry changes; force/profile/radius have not been changed to reduce them.

Native quadratures6/10/14 and intrinsic ports4/8/16 use the fixed finest meshes
from the first matrix. Every compared model receives byte-identical frozen
inputs, verified against the original q10/p8 snapshots. The8 runs contain160
model rows:150 are numerically qualified,8 pressure free-moment inputs are
unsupported and2 intrinsic fits are rejected. Full native workspaces, loads,
fields, original-wrench data and per-island observables remain archived.

| Geometry/variant versus q10/p8 | Maximum whole-field tensor change | Maximum raw-peak change |
|---|---:|---:|
| U q6/p8 |0.00729%|0.00339%|
| U q14/p8 |0.00422%|0.00243%|
| U q10/p4 |0.306%|0.236%|
| U q10/p16 |0.0836%|0.0426%|
| Capsule q6/p8 |0.0927%|0.0379%|
| Capsule q14/p8 |0.0309%|0.0540%|
| Capsule q10/p4 |2.140%|3.032%|
| Capsule q10/p16 |2.102%|2.234%|

Maximum changes use mutually accepted models/cases. Capsule q10/p4 rejects the
multiple-patch `body_source_prior` fit; q10/p16 rejects the bending `pair_union`
fit. More ports do not establish monotone acceptance or accuracy. These cases
remain visible, and their rejection is distinct from an elastic solve failure.
No native force/moment/cone or elastic tolerance was changed.

## Radius is a different physical load

Separate radius1/2/4mm sweeps retain fixed geometry/material and compare all four
models within each newly declared physical snapshot. The original3mm baseline
remains the baseline, rather than being relabelled as a refinement. For opposing
capsule loads the independent raw peak is23.254/12.144/5.837/3.340kPa at
radius1/2/3/4mm. At the U internal bend the corresponding peak remains about
96.605/96.606/96.607/96.607kPa; the local contact distribution changes while the
interior bending hotspot dominates this particular load. Full other-load and
per-island results are preserved. Mesh error is still unresolved, so these
numbers do not establish continuum radius sensitivity. Profile-law changes and
changing-count/history comparisons remain required.

## Force-controlled unilateral-contact reference

The independent contact solver now accepts a sparse coordinate map that ties
back-face normal displacements to one rigid-backplate translation. Prescribed
transverse guides carry reactions; applied actuator force enters the potential
and its gradient. The final jaw/backplate displacement is solved, not prescribed
alongside the force. Ordinary displacement-controlled calls retain their
previous equations and residual budget.

The full3D two-block controls have E1MPa, nu0,100mm² area,10mm total height,
friction0 and actuator force0.5N. A contact state is initialized before switching
to force control. n3/4/5 all recover0.5N axial interface force with complete
constrained residuals1.62e-12/2.63e-12/3.20e-12N. The solved back translations
are about−1.047534mm, versus−1.05mm in the analytic zero-gap limit; dhat2.5µm
remains finite. Raw global peaks are5.098/5.126/5.141kPa. These peaks are kept,
including the upward spatial trend.

The original force-control assertion also required both transverse interface
forces to be exactly zero. Only the axial coordinate is force controlled;
transverse guides supply reactions. That failed assertion is retained. The
corrected check enforces the exact axial target, rigid-backplate equality and
complete force balance. Transverse force relative to axial decreases from
8.82e-5 to2.47e-5 to9.62e-6 as the mesh changes. It is reported as a discretization
observable, not removed by rounding. The prior prescribed-friction control and
the subsequent force-control regression pass the unchanged1e-9N criterion.

## Full U contact prototype and remaining limitations

The prototype contains the full U plus two declared compliant contact shoes,
E1MPa/nu0 shoes, E10GPa/nu0.3 link, friction0.8, force0.05N per backplate and
consistent-mass gravity. Shoe surfaces follow the declared CAD wall with a smooth
edge lift; complete3D meshes and all parameters are retained. Initialization
supports are temporary. Jaw translations and contact forces are solved in the
force-controlled phase, and the final link has no prescribed displacement.
Actual Franka geometry/compliance/protocol matching remains unverified.

The first close initial placement has mesh intersections. The independent
[IPC intersection/CCD interface](https://raw.githubusercontent.com/ipc-sim/ipc-toolkit/v1.6.0/python/src/ipc.cpp)
requires an intersection-free CCD start. The invalid initial state and failed
iterates remain archived. A larger initialization separation and an explicit
intersection audit fix that initialization problem without changing force,
material, friction or the final force-controlled equations.

The next attempt stalls at about1e-8N because the energy line-search bound omits
coordinate cancellation. Retained diagnostics show that a full Newton step lowers
the residual below1e-12N while increasing the computed energy by about2.5e-21J.
The line search now includes a force-times-coordinate floating-work scale, in
addition to the energy magnitude, and still requires residual decrease. The
stopping threshold remains1e-9N. The force-controlled normal regression passes.
This is a numerical error bound, not a looser mechanical acceptance gate.

The corrected prototype completes its displacement initialization. It still
fails the updated friction equilibrium at the first free-body force/gravity step.
The later attempt establishes both force-controlled contacts before removing the
support; job712813 also fails the first free-body updated friction step and is
tracked separately. None of these failed references is
counted as a converged deformable grasp. Small-strain checks, full equilibrium,
spatial/regularization convergence, resolved-load native replay and interior
hotspot/egg comparisons remain required.

## Reproduction and publication

```bash
bash research/rigid_stress/scripts/refine_fixed_geometry.sh "$NEW_MESH_RUN" \
  --seed "$SEED_MESH" --levels 3
bash research/rigid_stress/scripts/run_load_models.sh "$NEW_REFERENCE_RUN" \
  --mesh "$MESH" --geometry u --reference-only --reference-solver amg-cg
bash research/rigid_stress/scripts/run_load_models.sh "$NEW_AXIS_RUN" \
  --mesh "$MESH" --geometry u --backend cuda --method pcg \
  --native-quadrature 14 --ports 8
bash research/rigid_stress/scripts/run_force_control_blocks.sh "$NEW_CONTROL_RUN" \
  --resolution 5 --force 0.5 --dhat 2.5e-6
```

The [immutable progress manifest](../evidence/20261014-physical/physical-axes-checkpoint-v1/manifest.json)
verifies3462 entries/3462 exact originals. SHA256:
`0dc73cf51894647d6ea1d76558d54942f7f743a3055b2540692893d55a2d8669`.
It retains fields, samples, force-control coordinate maps, failures, source
snapshots, exact commands, initial geometry diagnostics and all selected scans.
Publisher verification/source-after records are derived companions. Ongoing
fine CAD references and later coupled attempts are outside this checkpoint.

The original goal's physical convergence, profile/count/history sensitivity,
resolved full egg/Franka and interior-hotspot contact, per-island maps and warmed
accuracy/cost acceptance remain required. This checkpoint does not replace any
of those requirements.
