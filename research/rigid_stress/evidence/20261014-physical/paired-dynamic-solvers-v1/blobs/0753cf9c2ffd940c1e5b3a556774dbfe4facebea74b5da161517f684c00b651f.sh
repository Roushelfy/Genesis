#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_metis_setup=${2:?Closed isolated METIS dependency setup required}
task_qualification=${3:?Full-state METIS qualification or dash for small qualification candidate required}
task_module=${4:?run_paired_egg_contact or resume_paired_egg_contact required}
shift 4
case "$task_module" in
    run_paired_egg_contact|resume_paired_egg_contact) ;;
    *) exit 2 ;;
esac
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_LINEAR_SOLVER_STATS_PATH="$task_run/linear_solver_stats.json"
export REFERENCE_METIS_STATS_PATH="$task_run/metis_factor_stats.json"
task_sources=(research/rigid_stress/{metis_reference_factor,run_paired_egg_contact,precise_stencil_spatial,audit_original_history_fields,audit_articulated_contact,audit_objective_frame,qualify_articulated_precision,reference_motion_audit,audit_precise_spatial_state,resume_paired_egg_contact,wide_spatial_franka,wide_coordinate_franka,precise_panda_kinematics,articulated_franka_contact,franka_mujoco_reference,localized_franka_mesh,qualify_local_franka_mesh,cached_reference_newton,refined_franka_contact,frozen_franka_contact,dynamic_franka_reference,refine_franka_reference,franka_contact_reference,franka_rigid_mode_coordinates,audit_franka_reference,contact_reference,neohookean_reference,stable_neohookean_reference,skfem_reference,summarize_fine_references,sparse_franka_inertia}.py
    research/rigid_stress/{metis_reference_ordering,symmetric_reference_ordering,generic_stencil_types,localized_franka_driver,mapped_contact_vertices,cached_reference_newton,sparse_franka_driver}.patch
    research/rigid_stress/scripts/{run_metis_paired_reference,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
for task_patch in mapped_contact_vertices cached_reference_newton generic_stencil_types metis_reference_ordering; do
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
    "$task_env/bin/python" - "$task_metis_setup" "$task_qualification" "$task_run" "$@" <<'QUAL' > "$task_run/ordering-qualification.json"
import hashlib,json,sys
from pathlib import Path
import numpy as np
from research.rigid_stress.summarize_fine_references import closed_source
setup,q,run=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
closed_source(setup); assert int((setup/'returncode.txt').read_text())==0
sf=setup/'qualification.json'; dependency=json.loads(sf.read_text())
assert dependency['pymetis']=='2025.2.2' and dependency['full_residual']<=1e-12
root=Path(dependency['dependency_root'])
for name,expected in dependency['dependency_sha256'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
(run/'dependency-path.txt').write_text(str(root/'site-packages')+'\n')
record={'dependency_setup':str(setup),'dependency_summary_sha256':hashlib.sha256(sf.read_bytes()).hexdigest(),
        'pymetis':'2025.2.2','production_or_active_environment_modified':False}
if q=='-':
 args=sys.argv[4:]; geom=Path(args[args.index('--geometry-run')+1]); mode=args[args.index('--mode')+1]; level=int(args[args.index('--level')+1])
 with np.load(geom/f'results/{mode}-level{level}.npz') as saved: count=len(saved['tetrahedra'])
 assert count<=2000, 'Unqualified candidate is restricted to the small original-state qualification mesh.'
 record.update(physical_qualification_pending=True,qualification_candidate_tetrahedra=count)
else:
 q=Path(q);closed_source(q);assert int((q/'returncode.txt').read_text())==0
 f=q/'results/summary.json';s=json.loads(f.read_text())
 assert s['all_original_arrays_and_complete_fields_qualified'] and s['states_checked']==len(s['rows'])>=5
 assert all(max(r['complete_body_dynamic_residual_N'],r['complete_mapped_dynamic_residual_N'])<=1e-9 for r in s['rows'])
 candidate=Path(s['candidate']);closed_source(candidate)
 for name in ['metis_reference_ordering.patch','metis_reference_factor.py']:
  local=Path('research/rigid_stress')/name
  assert hashlib.sha256(local.read_bytes()).digest()==hashlib.sha256((candidate/'source'/local).read_bytes()).digest()
 record.update(qualification=str(q),qualification_summary_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),all_original_arrays_and_fields_qualified=True)
print(json.dumps(record,indent=2))
QUAL
    IFS= read -r task_metis_path < "$task_run/dependency-path.txt"
    export PYTHONPATH="$task_run/source:$task_metis_path"
    "$task_env/bin/python" -c 'from pathlib import Path; from research.rigid_stress import localized_franka_mesh as m; assert Path(m.__file__).resolve().is_relative_to(Path.cwd()); print(m.__file__)' > "$task_run/driver-origin.txt"
    "$task_env/bin/python" -m "research.rigid_stress.$task_module" --output "$task_run/results" "$@" \
        > "$task_run/reference.log" 2>&1)
