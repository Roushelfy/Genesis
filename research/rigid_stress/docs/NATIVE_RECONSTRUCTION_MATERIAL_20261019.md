# Actual reconstructed loads, independent elastic cores and material controls

All10 selected native states pass independent integration and same-RHS elastic
checks. Separate500um normal material-model controls at0.3/1N pass the existing
reporting bands. These results strengthen error attribution, but do not complete
the [goal](QUALIFIED_STRESS_RANKING_GOAL.md): fine friction reference qualification,
the prescribed-load spatial control and final cross-case deliverables remain open.

The [packet](../evidence/20261019-qualified-ranking/native-reconstruction-material-v1/README.md)
retains362 verified payloads totaling4,010,250 bytes. A metadata audit rehashes
109 distinct inputs/outputs. All complete fields, reconstructed quadrature and
runtime histories remain on data. See the
[working completion checklist](QUALIFIED_STRESS_COMPLETION_CHECKLIST.md) for
requirements that these checks do and do not establish.

## Actual native integration and independent same-domain solve

`replay_native_pad_tractions` reads the exact original journal, its admitted
stress-frame hash and the corresponding mechanical frame. It replays original
world forces, rotation and angular velocity through unchanged native recovery.
The saved material forces must agree with the original input before recovery;
complete stress, force and RHS must reproduce the original within1e-8 afterward.
No approximate state or new mechanical trajectory is substituted.

`native_traction_snapshot` calls the actual native sample, force and P2 shape
functions without mutating the force/stress arrays. Its complete nonzero sample
set must reproduce the signed P2 nodal load and wrench within1e-10. These are
integrated quadrature forces, not fabricated pressure from nodal dual areas.
DPR's0.1N control retains2,405,189 samples; SCR retains1,751,298. No new force
cutoff, footprint fit or acceptance relaxation is introduced.

The first diagnostic failed because older stress NPZs store validity in the
journal and world state in the companion mechanical frame. The second failed
on the observer's Quadrants array annotations. Both failures are retained.
A newly submitted SCR diagnostic with a known wrong annotation was cancelled
after27s; its corrected retry passed. These are diagnostic implementation
failures, distinct from the retained physical every-step friction rejections.
Production recovery kernels and active reference jobs were unchanged.

The completed native captures are728746_0,728750_0 and all four groups of728762.
The independent single-state CPU checks728755/728761 pass. The final joint audit
728773 reuses one independent factor for all10 states: three normal loads per
method, plus the normal and sustained-friction endpoints at mu0.3 per method.
All have their own original complete-history admission and valid requested
observations. Earlier failed every-step stress histories remain unadmitted.

`audit_native_reconstruction` derives barycentric coordinates from actual point
positions, evaluates independent P2 face functions and assembles the surface
load without using the exported native interpolation weights. It then solves
the exact saved native RHS with independent scikit-fem stiffness and the
verified CVXOPT factor. Maxima across all10 states are:

| Check | Relative difference/residual |
| --- | ---: |
| Independent surface integration versus native nodal load | 3.041e-15 |
| Complete independent elastic equation | 5.094e-12 |
| Additional rigid-gauge load correction | 4.134e-15 |
| Complete corner stress versus native | 7.662e-12 |
| Maximum stress-component difference | 1.937e-11 |

This verifies numerical integration and the elastic core on the same discrete
domain and assembled RHS. It does not independently validate native inertial
terms, continuum discretization, contact footprint or material reduction.
Percent-level benchmark differences cannot be assigned solely to contact by
subtracting these scalar norms from an end-to-end error.

## Measured redistribution and footprint broadening

The explicit [body source-prior law](BODY_SOURCE_PRIOR_MODEL.md) conserves the
link's aggregate wrench; individual disk or plate-side resultants may differ.
The newly captured SCR quadrature measures that behavior directly. For the
frictionless normal states, lower/upper recovered axial forces are essentially
equal; their values and reduction from the supplied side force are:

| Supplied force per side N | SCR reconstructed force per side N | Reduction % |
| --- | ---: | ---: |
| 0.1 | 0.0992419703 | 0.75803 |
| 0.3 | 0.296376767 | 1.20774 |
| 1.0 | 0.983433214 | 1.65668 |

DPR preserves these side resultants to roundoff. SCR's opposite reductions can
cancel in the aggregate equilibrium check; global balance is not proof that
each plate-side recovered force equals its measured input. This is a measured
load-model effect under the declared body-prior contract, not a PCG residual.
Actual mechanical plate reactions remain the separate measured comparison load.
At mu0.3 the redistribution is different, so the frictionless percentages must
not be transferred to the sustained-friction case.

At0.1N and mu0, axial-force-weighted RMS radii in the material frame are:

| Load representation | Lower radius mm | Upper radius mm |
| --- | ---: | ---: |
| Actual physical centroid sources | 1.47280 | 1.39738 |
| DPR reconstructed quadrature | 1.68929 | 1.62308 |
| SCR reconstructed quadrature | 1.68344 | 1.60720 |

The complete nonzero radial extents also grow from2.286/2.113mm at the sources
to4.244/3.974mm for DPR and4.229/3.938mm for SCR. These maxima retain arbitrarily
small nonzero forces and are not99%-force footprint radii. They are radial
distances, distinct from the earlier map's maximum absolute x/y coordinate.
The force-weighted radii show broadening without trimming those small samples.
No radius is changed in response to these observations.

## Normal material-model control

`solid-pad-stvk-normal-load-controls-v1` (728615) completes400 steps, both
equilibrium windows, analytic checks and independent current-position momentum.
It uses the same500um two-wall mesh and q10 pads as the retained linear static
reference. Physical normal loads are0.3/1N; there is no imposed shear. These are
same-mesh linear-static versus objective-StVK-dynamic contrasts, not measured
finest-mesh material-error bounds. All q9/q25 endpoint inputs are independently
rehashed against their own admission receipts.

Values below are **percent changes** at q25. The largest field includes tangent,
membrane, bending and both surfaces; it is bending at both loads.

| Load N | Largest field | Raw VM | Raw tension | Common VM | Common tension |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3 | 0.0756960 | 0.0646376 | 0.126779 | 0.0625169 | 0.126667 |
| 1.0 | 0.248046 | 0.214383 | 0.423290 | 0.207780 | 0.422952 |

Both pass the predeclared0.5% field/1% peak reporting bands. Angular sampling
changes these field-contrast metrics by at most4.723e-9 in absolute relative
units. The actual400-step dynamics take3557.36s in this one execution; this
control supplies no primary performance comparison or statistical speedup.

## Prescribed finite-area control at this checkpoint

The later [reference qualification](PRESCRIBED_REFERENCE_QUALIFICATION_20261019.md)
passes the fixed125um/three-wall spatial/integration gates. The measurements
below retain this earlier checkpoint's scope, including its failed two-wall
surface control. Same-load method comparisons remain separately required.

The independent prescribed C2 pressure retains0.1N and its predeclared2mm
radius. The verified long-index adapter control728702 agrees with the retained
same-mesh field and raw/common peaks within3.130e-13. It changes only the factor
backend and now fails the process if the actual elastic equations fail.

Common q9/q25 comparisons728738 re-evaluate the completed prescribed-load
fields with physical-volume columns and aligned common peaks. At q25,
500-to250um two-wall tangent/membrane changes are3.14818%/2.70947%;250-to125um
changes are2.08330%/1.50925%. The250um two-to-three-wall tangent change is
1.62895%. The final two-wall surface gate therefore still fails2%; stable raw
peaks do not override it. The earlier2.0205% value used q3 and is superseded
for this comparison, not rounded down into a pass.

The same coarse mesh, two-source DPR/SCR comparisons against the independently
prescribed C2 profile give3.05032%/2.78256% tangent and4.05682%/3.76736% bending
differences. They describe source-profile reconstruction on the matched coarse
domain; they are not the later full-source physical-pad benchmark or qualified
continuum rankings. The125um three-wall prescribed reference728727 was active
at this packet's cutoff. Only its immutable submission and fixed follow-up
tasks are included, with its active results excluded.

After that cutoff,728727 completes both q8/q10 prescribed-load cases with
admitted elastic equations in25m30s. Setup/factorization takes1443.67s and
process peak RSS is53,042,636KiB. The independent wall/surface/joint/quadrature
comparisons are now submitted as728823; submission does not establish their
spatial gates. Their fresh preflight and tasks remain on data under
`finite-area-prescribed-final-fields-v1/` and its adjacent `meta/` receipts.

Reproduce the captured tasks through the established rtx-mid / gs-srun workflow.
`audit_native_reconstruction --capture FIRST --additional-captures OTHER ...
--output NEW_DATA_PATH` shares one independent factor across matching meshes.
The metadata script `meta/audit-native-reconstruction-material-v1.py` rechecks
terminal status, own admission and hashes; it does not recompute the solves.
Runtime data and active writers remain under the existing data root.
