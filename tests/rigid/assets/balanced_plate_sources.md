# Balanced plate source fixtures

`balanced_plate_normal.npz` and `balanced_plate_shear.npz` retain only the
anchors, original application points, source forces, radii, friction,
anchor-face indices and anchor moments of two failed complete-egg plate
observations. They contain no fitted dual, expected native field or solver
initializer. The independent extraction is implemented in
`research/rigid_stress/qipc_comparison/scr_balanced_probe.py`.

Original observations, relative to
`/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261018-parallel-plates/`:

| Fixture | Original | SHA256 |
| --- | --- | --- |
| normal | `native-equilibria-v2/group1/scr/rejected-state.npz` | `e79ec59323caa33d84b72ee476e8d9280e712fb25b0b983705ed9231965a84db` |
| shear | `shear-native-shell-v1/group1/scr/rejected-state.npz` | `0da4edd7dd540eca57a5a9a0b76162db76d1aafbe454f01af8a995cacb82b965` |

The independent extraction/conditioning result is retained under
`runs/20261019-qualified-ranking/scr-cpu-v1/group0/probe/` in the same project.
Both originals fail against engine950ed9d3c; the regression checks feasibility,
local friction cones, independently computed force/moment budgets, independent
CPU traction/complete-field agreement, and native recovery. The covariance
test also checks rotated/translated geometries and changes of load/length scale.
