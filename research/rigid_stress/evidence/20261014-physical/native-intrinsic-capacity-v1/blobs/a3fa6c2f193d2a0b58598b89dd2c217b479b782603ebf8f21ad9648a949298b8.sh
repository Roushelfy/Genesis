#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Shape index required}}
task_geometries=(ellipsoid capsule u thin_neck cavity narrow_gap egg)
test "$task_index" -ge 0 && test "$task_index" -lt 7
task_geometry=${task_geometries[task_index]}
case "$task_geometry" in
    ellipsoid|capsule|thin_neck) task_sizes=(0.00175 0.00125 0.000875) ;;
    u|narrow_gap) task_sizes=(0.00125 0.000875 0.000625) ;;
    cavity) task_sizes=(0.11 0.08 0.06) ;;
    egg) task_sizes=(0.06 0.04 0.03) ;;
esac
bash research/rigid_stress/scripts/generate_physical_matrix_meshes.sh \
    "$task_root/fine-matrix-mesh-$task_geometry-v1" "$task_geometry" "${task_sizes[@]}"
