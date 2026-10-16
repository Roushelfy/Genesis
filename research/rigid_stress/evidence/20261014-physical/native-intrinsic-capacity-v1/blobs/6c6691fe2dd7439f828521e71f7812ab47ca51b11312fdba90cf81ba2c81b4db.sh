#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
shift
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
task_sources=(research/rigid_stress/{run_articulated_franka,original_franka_ground,predicted_wide_franka,wide_coordinate_franka,run_articulated_franka,captured_reference_motion,precise_panda_kinematics,articulated_franka_contact,continue_original_contact_state,franka_mujoco_reference,franka_contact_reference,franka_rigid_mode_coordinates,audit_franka_reference,contact_reference,neohookean_reference,stable_neohookean_reference,skfem_reference,summarize_fine_references}.py
    research/rigid_stress/scripts/{continue_original_contact_state,cluster_env}.sh genesis/assets/xml/franka_emika_panda/panda.xml)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
touch "$task_run/source/research/__init__.py" "$task_run/source/research/rigid_stress/__init__.py"
task_actual=("${task_sources[@]}" research/__init__.py research/rigid_stress/__init__.py)
(cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_actual[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
printf '%s\n' "$*" > "$task_run/arguments.txt"
hostname > "$task_run/node.txt"
lscpu > "$task_run/cpu.txt"
"$task_env/bin/python" - <<'PY' > "$task_run/environment.json"
import importlib.metadata, json, platform, sys
print(json.dumps({"python":sys.version,"platform":platform.platform(),
                  "packages":{n:importlib.metadata.version(n) for n in ["numpy","scipy","mujoco","scikit-fem","ipctk"]}},indent=2))
PY
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -m research.rigid_stress.continue_original_contact_state --output "$task_run/results" \
        --mjcf "$task_run/source/genesis/assets/xml/franka_emika_panda/panda.xml" "$@" \
        > "$task_run/reference.log" 2>&1)
