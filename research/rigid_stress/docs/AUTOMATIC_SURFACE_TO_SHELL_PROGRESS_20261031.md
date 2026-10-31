# Automatic surface-to-shell progress

The [October 31 goal](AUTOMATIC_SURFACE_TO_SHELL_GOAL_20261031.md) is active and incomplete. The starting remote
revision is `7c7cdf6cd`; prototype geometry integration is `694be1506`. Original unrelated QIPC changes are preserved
in place; their starting hashes are recorded in the data-side `meta/start-state.json`.

The prototype splits each triangle into three conforming Q4s without moving the midsurface. It joins only exactly
coincident coordinates, checks material connectivity/manifoldness/intersections, normalizes directors consistently
with the native solver, constructs both thickness sides and open-edge caps, and rejects collapsed/inverted offsets.
Thickness participates in the cache key. Custom shell collision uses the material boundary without convexification
or watertight wrapping. This implementation has not yet passed the full geometry, mechanics or scene qualification.

Pending work: focused geometry/cache tests; independent CPU/native fields and eccentric two-sided loads;
membrane/bending/mechanism controls and bounded refinement; public reset/isolation; real automatic entrypoint and
viewer color demonstration; cold/warm build profiles; small/large GPU module and complete-scene max/full timings,
memory and comparisons with explicit generated and authored egg assets; measured tuning; final usage/limits/report.

Runtime root: `/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261031-automatic-surface-shell/`.
No experiment outputs are committed. The first frozen worktree is
`/mnt/data/zhaofeng/projects/workspace/Genesis/worktrees/automatic-shell-geometry-v1` (session October 31,
geometry validation at `694be1506`, planned removal after its jobs finish, by November 1). Subsequent frozen worktrees
must be recorded here or in a later handoff and removed promptly after use.
