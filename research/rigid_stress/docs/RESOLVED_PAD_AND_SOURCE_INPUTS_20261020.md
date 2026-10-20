# Resolved pad transfer and distinct source-input failures

The [extended goal](AUTOMATIC_CONTACT_AND_THROUGHPUT_GOAL_20261020.md) remains
active. The [first CPU study](AUTOMATIC_CONTACT_PROGRESS_20261020.md) now has an
actual Quadrants counterpart: work-conjugate outer-centroid transfer completes
400 normal/friction steps with independently admitted mechanics and complete
fields. A separately bounded source-roundoff variant completes both original
DPR/SCR1000-step every-step observation paths with their own independent audits.
Neither result supplies an automatic production contact-region model or a new
fine-reference error ranking.

Terminal evidence retains the [resolved-pressure native controls](../evidence/20261019-qualified-ranking/resolved-pad-native-v1/README.md),
[source-roundoff and wrench contrasts](../evidence/20261019-qualified-ranking/source-roundoff-and-wrench-attribution-v1/README.md)
and [completed125um main/contact parents](../evidence/20261019-qualified-ranking/canonical125-final-contact-terminal-v1/README.md).
Full arrays stay hashed on data; earlier failed controls and baseline packets
remain immutable.

## Known physical pressure: Quadrants transfer and continuous history

`plate_native_pad --method dpr --source-reduction none --load-transfer
resolved_outer_p2` is an explicitly labelled **research adapter**. The physical
pad solver determines its own forces from the declared gaps, spring law and
lagged friction history. The observer checks that law and momentum before
transferring every actual outer-face centroid force through the six P2 shape
functions. Runtime force assembly, free-link balance, elasticity, maximum/full
output and rejection use Quadrants. The native engine itself is unchanged.

This is the work-conjugate transfer for the declared centroid quadrature, not
exact integration of a continuous pressure profile. Signed P2 corner loads are
valid. Source-sampling and elastic-resolution errors still require comparison
against the independent fine reference. The new path adds no second finite disk
and has no reconstruction-radius parameter in its transfer calculation.

Controls731439/731507 preserve the original fixed-radius replay and compare
normal/friction complete fields against the independent CPU transfer. Eight
final checks include both max outputs without full-field host export, zero-load
clearing, missing physical-admission refusal, nonfinite-source rejection and
recovery on the next valid input. Complete-stress differences are below1.24e-11;
maximum-only peak differences are below1.36e-12. The maximum clears to exactly
zero after loading. A deliberately nonfinite source produces native errno128.

Actual continuous run731442 completes400 steps. Independent audit731480 checks
both complete equilibrium windows, every mechanical row and all400 stress
observations. Compared with the retained original physical trajectory, maximum
differences are2.74e-16m/s plate speed and2.73e-14N plate reaction. The source
pressure and motion remain its own measured quantities.

Four complete material-column comparisons731485 cover both endpoints with q9
and q25 angular sampling, physical-volume weighting and four depth nodes.
Tangent/membrane/bending differences are below6.11e-11. Separate full3D normal
and transverse-shear errors stay below1.21e-10. Raw/common peak comparisons and
unsmoothed outer/inner fields are retained. This is same-discretization numerical
conformance, not spatial or continuum qualification.

The original manual2mm Gaussian DPR baseline and its inner-boundary spreading
remain intact. This resolved-pressure path does not repair SCR by substituting
another method. Its present research interface requires one environment and
already physical centroid forces; unknown collider pressures and production
batching remain separate work.

## Tiny detachment residues: bounded input reconciliation

The old DPR37/SCR38 failures remain unchanged. An opt-in
`--source-roundoff` variant applies to the retained fixed-radius/full-source
path. It first uses the existing local16-epsilon cone test. For rejected rows
only, it removes incompatible tangential roundoff while preserving meaningful
normal compression. Aggregate correction must stay below
`16*source_epsilon*max(command_force,unsigned_source_force_sum)`.
Larger conflicts raise an error before recovery, preserving their actual input.
No cone, friction coefficient or native solver tolerance is enlarged.

| Actual failed input | Changed sources | Force-change bound | Moment-change bound | Independent maximum principal-stress perturbation |
| --- | ---: | ---: | ---: | ---: |
| DPR37 |8|3.803e-21N|1.143e-22Nm|5.313e-15Pa|
| SCR38 |46|1.765e-20N|5.297e-22Nm|7.626e-15Pa|

The physical force budgets are2.630e-16/2.701e-16N. The stress perturbation is
the independently solved linear P2 response of the actual correction through
the known outer-centroid transfer, including free-link mass projection. It is
not a bound for arbitrary continuum placement or a stress field admitted by
the original failed finite-radius fit. Full equation residuals stay below
1.98e-12, against1e-7; the predeclared absolute stress-perturbation target is
1e-6Pa. Substantial incompatible pure shear remains rejected.

The controls test force scales0.001/1/1000, preserve already-admitted inputs
bitwise and check a nontrivial coordinate rotation. That rotation exposes a
positive normal dot product of order1e-38N on a mathematically zero-normal
load. The final helper restores zero within the same local dot-product
allowance before checking the aggregate physical budget. The actual two
unrotated corrections remain bitwise equal to those used by the submitted
trajectory variant. Original failed frame controls and their exact sources
remain recorded; valid running histories are not restarted.

Actual DPR/SCR runs731447_0/1 each complete1000/1000 valid observations, including
the formerly rejected37/38 states. Independent731574/731575 mechanical and
primary-cost audits pass each complete history and all eight cost gates.
Only observations37/38 change inputs. The actual new trajectory corrections
are at most2.513e-20N/7.544e-22Nm for DPR and1.778e-20N/5.336e-22Nm for SCR;
they are distinct from the original captured inputs in the table above.
All physical steps retain the original gap/friction/momentum
checks, with no velocity clearing or dropped sources beyond the recorded
bounded correction.

Their median complete-workflow costs are1885.276/3381.136ms/step over three warm200-step
intervals after both normal and friction loading. This includes host field
readback, mechanics/admission and I/O; it is not a production-solver rate.
It cannot be compared as a speedup against the old75.925ms **sparse** DPR cost.
The original failing every-step paths still have no admitted cost.

At zero normal load, step38 retains numerical von Mises tails of9.647e-8Pa
(DPR) and7.278e-8Pa (SCR), with unchanged native acceptance. Dividing its tiny
equation residual by the zero/tiny right-hand side produces a diagnostic near
1e17. That ratio does not demonstrate relative accuracy at zero load; the
complete mechanical audit and numerical absolute tail remain explicit. This
is not a claim that all trajectory stresses become exactly zero after unloading.

## Diverse SCR banks: geometry/normal and wrench contrasts

The exact first-failure states remain B128 step767/env9 and B512 step645/env330.
Their original independent continuous **full-wrench** infeasibility proofs
remain valid. New CPU contrasts use the captured mesh, source radii/friction,
normals and application points. Candidate point forces have explicit paths on
the actual surface, within the stated intrinsic radius and anchor component;
inner and outer boundary layers are never joined.

The source-to-anchor-face normal differences are15.956–17.983degrees in env9
and17.560–17.809degrees in env330. Application-to-anchor offsets are55.26–71.59um
and258.83–259.83um, respectively. Both source force packets satisfy their own
declared source-frame cones within their original floating-point allowance.

For env330, a new independently checked **force-only** polar witness covers
every P1 facet intersecting the conservative ball on each anchor's connected
component. Its rounded maximum polar bound is-5.932e-11, target-force projection
4.120e-6N, and unchanged force-acceptance budget1.466e-9N. A moment change alone
therefore cannot resolve this original-radius force conflict. Keeping the
actual surface points and original full wrench while using source-frame normals
does yield a nonnegative construction within the1e-8 residual budget. That is
a diagnostic change of traction law, not admitted SCR physical accuracy.

For env9, the compact force-only construction is unresolved and the force-only
search supplies no certificate. The full-wrench proof remains authoritative.
A joint source-normal/anchor-application contrast admits a construction; the
individual compact contrasts do not prove which change is necessary. The1.5x
radius probes are diagnostic only. A failed sampled LP never proves continuous
infeasibility, and a larger patch never supplies a repaired SCR accuracy result.

The witness search uses ordinary convex-solver stopping criteria, followed by
independent strict rounded polar/projection checks. Earlier failed metadata and
overly tight witness-search controls remain retained. Constructive LP outputs
are made nonnegative and their actual residual is rechecked; a solver-success
flag alone does not admit a load. These diagnoses distinguish input/model
conflict from an iteration-budget explanation without changing production code.

## Remaining delivery

The resumed125um/three-wall/q10 reference730978 and q6 contact control730979
each complete400 steps. Four-parent admission731010 passes40 checks; q6/q10
pair admission731014 passes20. Subsequently half-dt730980 completes800 steps
and the complete seven-parent qualification,54-error/18-rank and matched-cost
chain passes. Its separately generated final figures still require visual review
and publication; this packet retains the earlier main/contact terminal boundary.

Next compare rigid-gap, local approximations and reusable coupled compliance
across declared stiffness/thickness/asymmetry/nonconvex/motion cases. Predict
support and pressure jointly, check numerical search containment, then compare
complete stresses and peaks. Migrate useful validated models into the existing
rigid feature with GPU-resident inputs, supported batching/resets and measured
module versus scene cost. Keep unknown pressure, unresolved geometry, friction
history and sub-mesh resolution visible. The current prototype, completed DPR
path and new certificate do not complete the extended goal.
