#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Quadrature index required}}
task_geometries=(u capsule)
test "$task_index" -ge 0 && test "$task_index" -lt 6
task_geometry=${task_geometries[task_index / 3]}
task_level=$((2 + task_index % 3))
task_size=0.00175
if test "$task_geometry" = capsule; then task_size=0.0025; fi
bash research/rigid_stress/scripts/run_load_models.sh \
    "$task_root/ref-quad-$task_geometry-l$task_level-v1" \
    --mesh "$task_root/matrix-mesh-$task_geometry-v1/meshes/h$task_size.npz" \
    --geometry "$task_geometry" --reference-only --reference-quadrature-level "$task_level"
