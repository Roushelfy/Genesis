# Native finite surface-fan recovery checkpoint

The subsequent [source-robustness and profiling record](NATIVE_SOURCE_PROFILE_PROGRESS_20261011.md)
adds the first new-law profile and the B=16 producer-force diagnosis.

This continues the nonconvex contract introduced by
`1d6babf77848c46ba616da8bbb7062260a7ccd09` from source base
`463ac6c010d0c1b3e4cf58d599f727ce16966f9b`. The new contact law passes the
saved slipping-contact regressions and a complete B=4, 640-step live trajectory
on the low hollow shell. The nonconvex goal remains active: general producer
coverage, local refinement and the new-law throughput/optimization campaign
are unfinished. The earlier pressure-law throughput plateau does not establish
a plateau for this extension.

## Declared integration and distance

The intrinsic default now partitions every boundary face into three finite
triangles about that face's closest actual point to the witnessed anchor.
Each triangle uses the same positive Q10 area quadrature, Gaussian width
`0.45 R` and squared compact taper. Profiles use topology-connected port paths,
verified fully covered plane segments and an additional exact adjacent-face
unfolding path. Every accepted distance is the length of an actual path on
the material boundary. Euclidean distance only prunes candidates. The original
point, force, free moment, radius and actual combined friction remain inputs.

The new baseline uses eight subdivisions of each shared topological edge.
This is an evidence-selected baseline, not a claim that eight is universally
optimal or that graph distances are continuous geodesics. The earlier
anchor-only integration remains selectable with
`intrinsic_integration="anchor_fan"`. Surface-fan coordinates, material normals
and signed P2 basis values are regenerated from immutable shared face geometry
and one closest center per face/contact/environment. Positive profiles retain
their complete original sample indices; packing does not truncate them.
The allocation accommodates all 24,000 possible samples on the 80-face shell.
This deliberately conservative first baseline still needs memory and local
candidate optimization.

At saved tick 147, the previous anchor-only, coarse-distance model has no
sampled feasible solution. Increasing fit iterations cannot repair that
representation. Independent inner Coulomb LPs construct feasible loads after
both nearby-face finite integration and finer paths are provided. The new
native regression preserves the actual original wrench and succeeds on CPU
and CUDA. The GJK diagnostic does not justify attributing this failure to MPR:
the finite representation was also insufficient with the alternative producer.
No rigid collision solver was changed by this repair.

## Internal stopping and arithmetic

The internal normalized dual stopping value is now `9e-9`; independent final
original-point force and moment checks remain `1e-8`. Tick 427 was previously
rejected by an `8e-9` internal threshold despite fitting inside the final
budget. Tick 454 exposes a different numerical issue: the old dual-energy
comparison allowance ignores large cancelling multiplier/target products.
CPU fitting stagnates near `1.71e-8` even with 2,048 evaluations, while a CUDA
run needs 331 evaluations to pass. Its sampled inner LP is feasible.

The corrected comparison allowance includes the absolute products of both
current and proposed dual multipliers with the target, in addition to the
energies. A roundoff-scale energy increase can only be accepted when the
actual squared wrench residual improves. Holding law, samples, ridge and
stopping threshold fixed, saved tick 454 now fits in 18 evaluations on CPU
and CUDA, with normalized force errors below `2.5e-14` and moment errors below
`5.9e-15`. This is a matched arithmetic repair. It changes no final acceptance
budget or physical input.

## Independent checks and limitations

The new regression independently rebuilds closest centers, area quadrature,
topological distances, profiles and signed P2 loads for saved ticks 147, 427
and 454. Each also tests actual friction and two distinct friction variants.
Its CPU FP64 oracle minimizes the convex Coulomb dual with a trust-region
method and independently checks the original wrench. The older residual
least-squares oracle can stop at a nonzero-residual stationary point on these
cone-boundary cases; those failed development outputs remain archived.
The independent solver receives no native fitted forces or dual multipliers.

The broad suite passes 40 CPU and 40 CUDA cases. Same-mesh comparisons retain
the complete equilibrium matrix, including all six gauge rows, at
`max(1e-11 N, 1e-8 ||rhs||)`, and full tensor/von Mises/global peak acceptance
at `rtol=1e-4, atol=1e-3 Pa`. The B=4 live run uses seed 623001, FP64 collision
geometry, every-step source/traction/wrench checks, 33 independent CPU captures
(132 full fields), changing contact slots, slip/release and two environment
resets. Actual combined contact coefficients are approximately
`0.512724, 0.653102, 0.674688, 0.763502`. This run includes diagnostic and oracle
overhead and supplies no throughput measurement.

Across the 640 steps, the largest normalized force and moment errors are
`8.99673e-9` and `7.52132e-9`; the largest independent complete residual is
`1.43532e-11 N` and passes its unchanged per-input relative/absolute budget.
The run records 138 contact-slot changes, at most 59 fit evaluations and at
most 1,200 positive samples for a contact.

An independent rigid-only versus recovered two-scene comparison exceeds its
existing `1e-12` state bound at tick 466 by one value of approximately
`1.05e-12`; it is retained as a failed comparison. A rigid-only twin control
completes 640 steps with identical states. A direct audit surrounds the actual
pre-integration recovery call and compares all rigid checkpoint fields
byte-for-byte. All 213 fields match at all 17 audited calls across 640 steps,
including two resets. Its results are recorded in the accompanying raw evidence;
they do not erase the independent-scene comparison's failure or establish its
cause. The initial extra recovery replay after `scene.step()` is an invalid
audit timing: it combines solved witnesses with a later integrated pose and
correctly fails the source projection bound. The native ordinary recovery
call remains at the existing pre-integration boundary.

The benchmark's existing frame restoration matches a frozen solved snapshot
to its pre-integration reference frame. The new intrinsic benchmark route
checks replayed original positions, forces, final wrench errors, complete
residual and stress peak before timing. Its profile distinguishes source
capture and the intrinsic contact pipeline from elastic recovery; further
contact-stage subdivision and end-to-end large-batch evidence remain pending.

Production stress numerics remain entirely in Quadrants. Hardware for these
CUDA checks is NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
driver 595.71.05, Quadrants 1.3.3, Genesis 1.4.3, image `1_26`.
Only the existing low shell (810 P2 nodes, 480 tetrahedra, 80 loaded boundary
faces) and inexpensive contact fixtures were used. This checkpoint does not
include a physical mesh or material-quadrature convergence experiment.

The next work includes primitive/SDF/decomposition/aggregate provenance
coverage, diagnosed integration/distance refinement, larger varied live
trajectories and profiling-driven contact scheduling/packing/reuse. The new
four-scope B=1024/2048/larger throughput campaign and optimization plateau
remain unverified.

## Reproduction and immutable evidence

Run `research/rigid_stress/scripts/check_nonconvex_contact.sh cpu` and `cuda`
inside the configured allocated compute environment. For the live check, run
`research/rigid_stress/probe_nonconvex_online.py --envs 4 --steps 640
--seed 623001 --contact-block-size 2 --egg-geometry mesh64 --output <data-path>`.
Set data/cache/log paths explicitly with `scripts/cluster_env.sh` and keep
source files unchanged until a Quadrants process exits.

The [complete capsule manifest](../evidence/20261011-nonconvex/online-checkpoint-v3/manifest.json)
records 1,009 artifacts (18,398,327 bytes), including failed development runs,
saved original inputs, exact source snapshots, launchers, before/after source
hashes, raw device logs and independent CPU outputs. Every original and copied
artifact passes SHA-256 verification. Manifest SHA-256:
`a16ae6f7de3697dcdd7e2d96b821ded519a1d2a305db0b608db2118909e79e90`.
The [generated validation reduction](../evidence/20261011-nonconvex/online-checkpoint-v3/raw/surface-fan-checkpoint-summary.json)
preserves the aggregate values reported above without replacing the full records.
