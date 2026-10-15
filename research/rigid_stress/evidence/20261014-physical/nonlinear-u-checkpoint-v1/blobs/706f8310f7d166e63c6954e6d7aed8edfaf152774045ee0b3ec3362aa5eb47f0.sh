#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
task_sources=(research/rigid_stress/{replay_objective_contact,audit_objective_frame,replay_resolved_grasp,neohookean_reference,contact_reference,skfem_reference,summarize_fine_references,sample_load_fields,validation_meshes,profile_reference_sensitivity,load_model_validation,native_load_samples,finite_traction_reference,spherical_shell_validation,qualify_reference_scaling}.py
    research/rigid_stress/scripts/{replay_objective_contact,cluster_env}.sh)
"$task_env/bin/python" - <<'PY' > "$task_run/additional-sources.txt"
from pathlib import Path
paths = sorted(Path("genesis/engine/solvers/rigid/stress").rglob("*.py"))
assert len(paths) > 10
print("\n".join(str(path) for path in paths))
PY
while IFS= read -r task_source; do task_sources+=("$task_source"); done < "$task_run/additional-sources.txt"
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
"$task_env/bin/python" -m research.rigid_stress.replay_objective_contact --output "$task_run/results" "$@" \
    > "$task_run/replay.log" 2>&1
