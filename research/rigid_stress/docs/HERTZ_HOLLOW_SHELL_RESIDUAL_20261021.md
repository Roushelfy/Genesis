# Prescribed Hertz pressure on a fixed hollow shell

The particular-stress correction passes its internal refinement targets on a
full hollow polar-land egg. The direct-pressure reference has not converged
enough to admit an independent physical stress-error claim. Both outcomes are
retained; this control does not predict the pressure or qualify a scene estimator.

The [study](../qipc_comparison/study_hertz_shell_residual.py) applies one given
normal ellipse at the upper pole: Fn = 0.2 N, axes 69.93/51.14 um and peak
pressure 26.702 MPa. Material is E = 10 GPa, nu = 0.3, density 2000 kg/m3,
with a 500 um wall. The original polar-land construction has flat/blend radii
300/600 um. Its fixed faceted outer and inner boundaries remain identical
through all subsequent edge and wall refinements; volume remains
3.730390838297287e-6 m3. No rigid symmetry reduction or pad is introduced.

The solution is sigma_Hertz plus a P2 residual. Prescribed and particular
tractions cancel analytically on the entire loaded plane, including a footprint
smaller than a face. Every other boundary, including the inner cavity, carries
minus the particular traction. Both direct and residual solves retain the
original consistent-mass free-link inertial relief. Adding a Hertz tensor to the
original loaded P2 field would count part of the load twice; this study subtracts
the particular boundary contribution first.

The comparison uses 378,432 common volume-weighted queries, eight depth points
in each of 10,100 common material columns, and 9,611 near-field probes. Membrane
and bending are tangential material-column stress resultants in the same fixed
frames. Every element-corner tensor and displacement is retained as well.
Reported maxima include all corners and common probes, without smoothing;
they are sampled maxima, not certified continuum global extrema.

| Mesh | Tetrahedra | P2 scalar nodes | Direct tensile / VM MPa | Corrected tensile / VM MPa |
| --- | ---: | ---: | ---: | ---: |
| Coarse | 60,600 | 101,010 | 2.318 / 7.824 | 2.777 / 15.730 |
| Medium | 94,608 | 147,182 | 9.833 / 19.465 | 2.771 / 15.659 |
| Fine | 139,488 | 209,250 | 6.152 / 19.025 | 2.773 / 15.662 |

| Comparison | Complete tensor L2 | Membrane L2 | Bending L2 |
| --- | ---: | ---: | ---: |
| Coarse corrected / fine corrected | 2.189% | 0.658% | 0.495% |
| Medium corrected / fine corrected | 1.099% | 0.148% | 0.131% |
| Coarse direct / fine direct | 33.58% | 4.979% | 7.130% |
| Medium direct / fine direct | 9.823% | 0.783% | 0.975% |
| Coarse corrected / fine direct | 6.068% | 0.952% | 1.004% |
| Fine direct / fine corrected | 5.625% | 0.923% | 0.708% |

The original internal targets are 5% complete/resultant differences and 10%
peak differences, with 2%/5% reference field/peak changes. Corrected coarse/fine
tensile and VM peak changes are 0.159%/0.436%; medium/fine changes are
0.0644%/0.0173%. Those internal targets pass. The fine corrected tensile maximum
is at (69.93 um, 0, 30 mm); its VM maximum is on the axis at 27.98 um depth.
The same common probes attain the corrected maxima at all three resolutions;
this does not independently certify peak-location convergence.

The [retained-field inspection](../qipc_comparison/inspect_hertz_shell_residual.py)
also evaluates the separate direct-reference gates. Its medium/fine complete
field change is 9.823% and its tensile peak changes 59.83%, exceeding the original
2%/5% reference budgets. The VM peak change alone is 2.313%. Coarse corrected
versus fine direct peak differences are 54.86% tensile and 17.32% VM, with
sampled peak-position distances 147.1/12.36 um. These are differences against an
unconverged reference, not qualified physical-error estimates. The direct
reference, candidate-versus-direct and conditional prescribed-pressure gates
all remain false; no threshold was enlarged.

Complete particular boundary force/moment errors stay below 1.4e-14 relative,
and elastic residuals below 9.2e-11 relative. Coarse boundary quadrature 8/12
changes the corrected complete field by 1.13e-9 relative; remote Hertz evaluation
16/96 changes it by 2.23e-6. Doubling the fine pressure rule from 24x512 to
48x1024 changes the direct complete field by 0.1185%. These checks bound the
tested numerical components; they do not turn the remaining reference gap into
a physical error bound.

Construction takes 248/393/252 s and individual CPU solves 0.56-1.48 s before
field readback/interpolation. These are reference workloads, with substantially
more nodes than the separately timed level 1 native scene. Their costs and
accuracy cannot be assigned to the B1024 scene's measured rate. No new dense
contact basis or default coupled pressure/Newton-PCG solve is introduced.

Slurm CPU study 737295 and read-only inspection 737336 complete with exit 0 in
`rtx-mid`, image `1_26`, using the pinned scientific Python and one core. The
[packet](../evidence/20261019-qualified-ranking/hard-shell-residual-v1/README.md)
retains 55 compact payloads and hashes six data-side mesh/full-field archives.
Replay the captured tasks and original conditions with fresh explicit data-side
output/cache paths and public source alias `d4503b9e4`, as verified by the
[public binding](../evidence/20261019-qualified-ranking/hard-scene-public-source-binding-v1/README.md).
The cheap native shell correction, pressure/support prediction, arbitrary
boundary contacts and friction history remain active work.
