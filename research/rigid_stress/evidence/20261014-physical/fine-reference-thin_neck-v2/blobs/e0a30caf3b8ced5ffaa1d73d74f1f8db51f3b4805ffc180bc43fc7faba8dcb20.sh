#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
task_root=${1:?Physical runtime root required}
task_index=${2:-${SLURM_ARRAY_TASK_ID:?Axis index required}}
test "$task_index" -ge 0 && test "$task_index" -lt 8
task_geometries=(u capsule)
task_geometry=${task_geometries[task_index / 4]}
task_variant=$((task_index % 4))
task_quadratures=(6 14 10 10)
task_ports=(8 8 4 16)
task_size=0.00175
if test "$task_geometry" = capsule; then task_size=0.0025; fi
task_mesh="$task_root/matrix-mesh-$task_geometry-v1/meshes/h$task_size.npz"
task_run="$task_root/axis-$task_geometry-q${task_quadratures[task_variant]}-p${task_ports[task_variant]}-v1"
bash research/rigid_stress/scripts/run_load_models.sh "$task_run" \
    --mesh "$task_mesh" --geometry "$task_geometry" --backend cuda --method pcg \
    --native-quadrature "${task_quadratures[task_variant]}" --ports "${task_ports[task_variant]}"
bash research/rigid_stress/scripts/sample_load_fields.sh "$task_run-probes" \
    --run "$task_run/results" --mesh "$task_mesh" --geometry "$task_geometry"
