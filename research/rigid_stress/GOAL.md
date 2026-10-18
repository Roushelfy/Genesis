# Active goal: parallel-plate stress comparison

Follow [docs/PARALLEL_PLATE_COMPARISON_GOAL.md](docs/PARALLEL_PLATE_COMPARISON_GOAL.md).

Use two parallel rigid plates compressing a full hollow egg as the primary
DPR/SCR/Shell accuracy benchmark against qualified cuda-graph-qipc solid FEM.
Begin with quasi-static common-force normal loading without gravity/friction,
then add bounded friction/tangential and off-axis controls. Remove Franka
robot/controller complexity from primary error qualification.

Retain occasional predeclared full-spatial stress fields, short-window profiling
and tolerance selection before larger runs. Compare field/peak errors,
reactions/indentation/contact footprints, speed/memory and stability. Equal
plate separation and equal reaction are different protocols; do not impose both.

This replaces full-Franka physical-reference qualification as an active gate.
Reuse existing [QIPC work](docs/QIPC_REFERENCE_RESULTS_20261018.md),
[sparse sampling evidence](evidence/20261018-qipc-reference/sparse-keyframes-v1/README.md)
and prior FEM/replay controls. Keep Franka throughput as separate engineering
evidence. Preserve native Quadrants integration, original external baselines,
failure records, accepted evidence and the cleaned branch.
