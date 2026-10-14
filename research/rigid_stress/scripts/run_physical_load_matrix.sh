#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical-validation runtime root required}
task_version=${2:?Fresh version suffix required}
task_index=${3:-${SLURM_ARRAY_TASK_ID:?Matrix index required}}
task_geometries=(ellipsoid capsule u thin_neck cavity narrow_gap egg)
test "$task_index" -ge 0 && test "$task_index" -lt 21
task_geometry=${task_geometries[task_index / 3]}
task_level=$((task_index % 3))
case "$task_geometry" in
    ellipsoid|capsule|thin_neck) task_sizes=(0.005 0.0035 0.0025) ;;
    u|narrow_gap) task_sizes=(0.0035 0.0025 0.00175) ;;
    cavity) task_sizes=(0.28 0.20 0.15) ;;
    egg) task_sizes=(0.15 0.11 0.08) ;;
esac
task_mesh="$task_root/matrix-mesh-$task_geometry-v1/meshes/h${task_sizes[task_level]}.npz"
task_run="$task_root/load-matrix-$task_geometry-l$task_level-$task_version"
task_reference=superlu
if test "$task_geometry" = egg; then task_reference=amg-cg; fi
bash research/rigid_stress/scripts/run_load_models.sh "$task_run" \
    --geometry "$task_geometry" --mesh "$task_mesh" --backend cuda --method pcg \
    --reference-solver "$task_reference"
bash research/rigid_stress/scripts/sample_load_fields.sh "$task_run-probes" \
    --run "$task_run/results" --geometry "$task_geometry" --mesh "$task_mesh"
