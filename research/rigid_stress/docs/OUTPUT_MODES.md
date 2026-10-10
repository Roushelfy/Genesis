# Build-time stress output modes

User requirement, 2026-10-10: let the caller choose before simulation
between the complete stress field and its global maximum. This is an
implemented build-time option in the existing native rigid solver feature.
Both output APIs and their numerical/lifecycle checks are available. Matched
ordinary live/policy throughput and memory measurements on source `e6094ff9`
complete; final optimized-source measurements remain required.

At B=1024 on one RTX PRO 6000 Blackwell, three 2400-step repeats after 900
warmup steps give the following valid env-step/s, with zero invalid steps:

| Seed | Scope | Maximum only | Full field |
| --- | --- | ---: | ---: |
| 510000 | live | 73909.92 | 72895.53 |
| 510000 | policy inference | 73521.04 | 72750.94 |
| 623001 | live | 73250.64 | 72622.83 |
| 623001 | policy inference | 72625.25 | 72220.09 |

Full output allocates 107520 bytes/environment for 480 tetrahedra, four
corners, six tensor components and one von Mises value, all FP64. The extra
110100480 bytes at B=1024 raises counted native storage from 291118453 to
401218933 bytes. The four ordinary paired throughput costs are 0.56–1.37%.
Exact repeats, GPU UUID, source hashes and whole-card memory samples appear
in `evidence/20261010-contact-repair/mode-cost-e609-*` and the combined
`native-face-prod-and-mode-e609-summary.json`. Policy scope is inference
rollout, not full RL training.

## Configuration and API

Add a validated build-time option to `RigidStressOptions`, preferably
`output_mode: Literal["max", "full"] = "max"`. Configure each stressed link
before `scene.build()`; select allocations and numerical paths at build.
A mode change after build requires rebuilding rather than silently
reallocating or changing the observation contract.

- `max`: retain the existing per-environment global maximum von Mises
  observation and its current substep aggregation. Do not allocate or write
  persistent full-field stress buffers. Keep this the default for throughput.
- `full`: expose the complete recovered stress field on the scene device,
  while continuing to provide the existing maximum observation. Provide the
  symmetric stress tensor and corresponding von Mises values across the
  entire configured elastic mesh, with an accessor consistent with Genesis
  observation conventions (`copy=False` read-only device views).

The full representation must have documented sample locations, dimensions,
units (Pa), coordinate frame, tensor component order, and validity/lifetime.
For the current affine P2 elements, unaveraged stresses at all four corners
of every tetrahedron determine the element's entire affine tensor field.
This is an appropriate complete representation; one averaged value per
element or only a contact-region field is insufficient. A recommended
batched layout is `[B, n_tets, 4, 6]` for symmetric tensor components plus
`[B, n_tets, 4]` for von Mises values. Choose final accessor names and
layout according to Genesis conventions, and document unbatched behavior.

## Time and lifecycle semantics

Define the full field as a snapshot of the most recent recovery substep.
Preserve the existing maximum observation's maximum over all physical
substeps in a scene step. At a single matching recovery substep, reducing
the full von Mises field must agree with max-only mode. With multiple
substeps, a last-substep snapshot need not have the same maximum as the
scene-step temporal maximum; document and test this distinction.

Full-field state follows existing per-link/per-environment reset, invalid
solve/contact and checkpoint-restoration semantics. A retained buffer from
before invalidation must not appear as a valid new observation. Partial
reset preserves other environments. Immutable operators may be shared by
links with different output modes; mutable observations stay independent.

## Numerical and performance acceptance

Keep both modes inside the existing rigid solver and use Quadrants for
all production stress arithmetic and reductions. Use the same loads,
elastic model, complete equilibrium residual budgets and rigid dynamics.
The mode selects output work, not a different physics approximation.

On the current low-resolution complete hollow egg, compare both modes
against an independent CPU FP64 tensor/von Mises oracle. Verify max-only
and full-field reduction agree for the same recovered state, including
changing frictional contacts, rotation, multiple substeps, batched mode,
partial resets, and restoration/invalidation.

Benchmark max-only and full on matched trajectories and hardware, reporting
end-to-end live/policy env-step/s, batch step time, stress-stage time and
per-environment/total field storage. Demonstrate that max-only retains its
allocation advantage and avoids full-field writes; measure any regression.
Full-field output may cost more and must have its cost reported explicitly.
Continue low-mesh throughput tuning and same-mesh consistency checks; this
requirement does not introduce high-resolution physical convergence work.
