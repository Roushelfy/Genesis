#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_qualification=${2:?Closed sparse-ordering qualification required}
task_module=${3:?run_paired_egg_contact or resume_paired_egg_contact required}
shift 3
case "$task_module" in
    run_paired_egg_contact|resume_paired_egg_contact) ;;
    *) exit 2 ;;
esac
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_LINEAR_SOLVER_STATS_PATH="$task_run/linear_solver_stats.json"
task_sources=(research/rigid_stress/{run_paired_egg_contact,precise_stencil_spatial,audit_original_history_fields,audit_articulated_contact,audit_objective_frame,qualify_articulated_precision,reference_motion_audit,audit_precise_spatial_state,resume_paired_egg_contact,wide_spatial_franka,wide_coordinate_franka,precise_panda_kinematics,articulated_franka_contact,franka_mujoco_reference,localized_franka_mesh,qualify_local_franka_mesh,cached_reference_newton,refined_franka_contact,frozen_franka_contact,dynamic_franka_reference,refine_franka_reference,franka_contact_reference,franka_rigid_mode_coordinates,audit_franka_reference,contact_reference,neohookean_reference,stable_neohookean_reference,skfem_reference,summarize_fine_references,sparse_franka_inertia}.py
    research/rigid_stress/{symmetric_reference_ordering,generic_stencil_types,localized_franka_driver,mapped_contact_vertices,cached_reference_newton,sparse_franka_driver}.patch
    research/rigid_stress/scripts/{run_ordered_paired_reference,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
for task_patch in mapped_contact_vertices cached_reference_newton generic_stencil_types symmetric_reference_ordering; do
    patch -p1 -d "$task_run/source" < "research/rigid_stress/$task_patch.patch" >> "$task_run/patch.log"
done
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
                  "packages":{n:importlib.metadata.version(n) for n in ["numpy","scipy","scikit-fem","ipctk"]}},indent=2))
PY
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" -c 'from pathlib import Path; from research.rigid_stress import localized_franka_mesh as m; assert Path(m.__file__).resolve().is_relative_to(Path.cwd()); print(m.__file__)' > "$task_run/driver-origin.txt"
    "$task_env/bin/python" - "$task_qualification" <<'QUAL' > "$task_run/ordering-qualification.json"
import hashlib,json,sys
from pathlib import Path
from research.rigid_stress.summarize_fine_references import closed_source
q=Path(sys.argv[1]);closed_source(q)
assert int((q/'returncode.txt').read_text())==0
f=q/'results/summary.json';s=json.loads(f.read_text())
assert s['all_original_arrays_and_complete_fields_qualified'] and s['states_checked']==len(s['rows'])>=5
assert all(r['complete_body_dynamic_residual_N']<=1e-9 and r['complete_mapped_dynamic_residual_N']<=1e-9 for r in s['rows'])
patch=Path('research/rigid_stress/symmetric_reference_ordering.patch')
sha=hashlib.sha256(patch.read_bytes()).hexdigest()
assert hashlib.sha256((q/'source'/patch).read_bytes()).hexdigest()==sha
print(json.dumps({'qualification':str(q),'summary_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'ordering_patch_sha256':sha,'all_saved_arrays_and_fields_qualified':True},indent=2))
QUAL
    "$task_env/bin/python" -m "research.rigid_stress.$task_module" --output "$task_run/results" "$@" \
        > "$task_run/reference.log" 2>&1)
