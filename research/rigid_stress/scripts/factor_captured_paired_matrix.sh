#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_setup=${2:?Closed isolated METIS dependency setup required}
shift 2
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export REFERENCE_METIS_STATS_PATH="$task_run/metis_factor_stats.json" REFERENCE_BORDERED_METIS_STATS_PATH="$task_run/bordered_metis_factor_stats.json"
export REFERENCE_SCHUR_METIS_STATS_PATH="$task_run/schur_metis_factor_stats.json"
task_sources=(research/rigid_stress/{factor_captured_paired_matrix,bordered_metis_reference_factor,schur_metis_reference_factor,metis_reference_factor,franka_rigid_mode_coordinates,summarize_fine_references}.py
    research/rigid_stress/scripts/{factor_captured_paired_matrix,cluster_env}.sh)
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
"$task_env/bin/python" - <<'ENV' > "$task_run/environment.json"
import importlib.metadata, json, platform, sys
print(json.dumps({"python":sys.version,"platform":platform.platform(),
                  "packages":{name:importlib.metadata.version(name) for name in ["numpy","scipy"]}},indent=2))
ENV
(cd "$task_run/source"
    export PYTHONPATH="$task_run/source"
    "$task_env/bin/python" - "$task_setup" "$task_run" <<'DEP' > "$task_run/dependency-qualification.json"
import json,hashlib,sys
from pathlib import Path
from research.rigid_stress.summarize_fine_references import closed_source
setup,run=map(Path,sys.argv[1:]);closed_source(setup);assert int((setup/'returncode.txt').read_text())==0
f=setup/'qualification.json';d=json.loads(f.read_text());assert d['pymetis']=='2025.2.2' and d['full_residual']<=1e-12
root=Path(d['dependency_root'])
for name,sha in d['dependency_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
(run/'dependency-path.txt').write_text(str(root/'site-packages')+'\n')
print(json.dumps({'setup':str(setup),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'all_pinned_dependency_files_verified':True}))
DEP
    IFS= read -r task_metis_path < "$task_run/dependency-path.txt"
    export PYTHONPATH="$task_run/source:$task_metis_path"
    "$task_env/bin/python" -m research.rigid_stress.factor_captured_paired_matrix --output "$task_run/results" "$@" \
        > "$task_run/audit.log" 2>&1)
