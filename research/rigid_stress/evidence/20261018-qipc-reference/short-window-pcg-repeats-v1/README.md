# Three fresh-process checks of PCG1e-2

The [fixed-Newton contrast](../short-window-pcg-v1/README.md) identified
PCG1e-2 as a faster candidate satisfying all declared short-window bands.
Job726727 repeats that setting in three fresh processes, sequentially in one
allocation. Job726728 executes the complete three-keyframe field, actual
finger-reaction, residual/iteration and target/egg-pose audits.

| Run | Window seconds | All three keyframes pass | PCG-cap steps |
| --- | ---: | --- | ---: |
|r0|35.001181|Yes|0|
|r1|35.489127|Yes|0|
|r2|34.600375|Yes|0|

The median is35.001181s. Newton remains1e-9m/s, PCG1e-2, standard linear-PCG,
double MAS and cap4096. Every run starts from the original state and advances
continuously to1.16s, with the same geometry/material/contact/friction and
prescribed rigid targets. The timed interval is(1.05,1.16]s with110 physical
steps. Full spatial stress is computed/exported only at the predeclared
1.10/1.13/1.16s. Each process separately measures initial read-only elastic
activation outside the window. Full host transfer remains inside, compression
and file writing outside. Native failure, finite/J and per-step robot-tracking
checks remain enabled.

These repeats support using1e-2 for the next validation stage. They retain the
same provisional numerical reference and bands as the exploratory contrast.
They do not qualify the full trajectory, capped actuation or spatial/contact/
timestep uncertainty. The active goal and primary sparse candidate-cost work
remain open.

All9 complete spatial keyframe NPZs and full journals/timings are included.
`analyses/` retains actual common-field/reaction and pose results, with no
whole-history stress norm claimed for three observations. `plan.json` was
declared before these repeats. `receipts/`, lossless `logs/` and exact source
inventories reconstruct each executed worker from
[the first packet's source](../short-window-v1/source/) plus this packet's
campaign-specific `source-variants/` overlay. The source identities are checked
file by file. Shared stricter reference fields, layout/surface and zero-step
target binding remain in that first packet. `capture.py` performs only byte
copies, hashing and lossless compression. Scientific calculations ran in the
captured Slurm/container workers.
