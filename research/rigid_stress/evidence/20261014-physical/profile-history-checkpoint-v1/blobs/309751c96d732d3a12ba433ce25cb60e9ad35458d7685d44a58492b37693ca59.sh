#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_root="$RIGID_STRESS_DATA_ROOT/runs"
task_epoch="$task_root/20261014-physical-validation"
task_version=${1:-v1}
task_tools="$task_epoch/profile-history-publication-$task_version"
test ! -e "$task_tools"
mkdir -p "$task_tools/source"
task_sources=(research/rigid_stress/{profile_reference_sensitivity,summarize_profile_references,plot_profile_references,native_physical_sequence,load_model_validation}.py
    genesis/engine/solvers/rigid/stress/{model,solve}.py tests/rigid/test_stress.py
    research/rigid_stress/scripts/publish_profile_history_checkpoint.sh
    research/rigid_stress/scripts/{publish_nonconvex_checkpoint,verify_nonconvex_publication}.py)
sha256sum "${task_sources[@]}" > "$task_tools/source.sha256"
cp --parents "${task_sources[@]}" "$task_tools/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"; printf "%s\n" "$task_status" > "$task_tools/returncode.txt"; cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"; exit "$task_status"' EXIT
git rev-parse HEAD > "$task_tools/base-revision.txt"
task_names=(profile-u-q3-v1 profile-u-q4-v1 profile-capsule-q3-v1 profile-capsule-q4-v1
    profile-summary-v1 profile-plots-v1 profile-plots-v2 fine-all-summary-v1 fine-all-plots-v1
    native-sequence-u-pilot-v2 native-sequence-u-pilot-v3 native-sequence-u-pilot-v4 native-sequence-u-pilot-v5
    grasp-u-self-balanced-v1 pcg-release-before-cpu-v1 pcg-release-after-cpu-v1 pcg-release-after-gpu-v1)
for task_backend in cpu gpu; do
    task_name="pcg-final-stress-suite-$task_backend-v1"
    if test -f "$task_epoch/$task_name/returncode.txt"; then task_names+=("$task_name"); fi
done
TASK_PROFILE_EPOCH="$task_epoch" TASK_PROFILE_NAMES="${task_names[*]}" "$task_env/bin/python" - <<'PY' > "$task_tools/summary.json"
import hashlib
import json
import os
from pathlib import Path
root=Path(os.environ["TASK_PROFILE_EPOCH"])
records=[]
for name in os.environ["TASK_PROFILE_NAMES"].split():
    run=root/name
    before=(run/"source.sha256").read_bytes()
    assert before==(run/"source-after.sha256").read_bytes(),name
    for line in before.decode().splitlines():
        expected,relative=line.split("  ",1)
        assert hashlib.sha256((run/"source"/relative).read_bytes()).hexdigest()==expected,(name,relative)
    records.append({"name":name,"returncode":int((run/"returncode.txt").read_text())})
profiles=json.loads((root/"profile-summary-v1/results/summary.json").read_text())
sequence=json.loads((root/"native-sequence-u-pilot-v5/results/summary.json").read_text())
print(json.dumps({"scope":"Fixed-wrench physical profile sensitivity, complete seven-shape FEM refinement diagnostics and native changing-contact/PCG-release qualification",
                  "source_closures":records,"profiles":profiles,"native_sequence":sequence,
                  "goal_completion":False},indent=2))
PY
task_runs=()
for task_name in "${task_names[@]}"; do task_runs+=("$task_epoch/$task_name"); done
task_runs+=("$task_tools")
task_extra=("$task_epoch"/fine-egg-time-extension-*.txt)
while IFS= read -r task_path; do task_extra+=("$task_path"); done < <("$task_env/bin/python" -c 'from pathlib import Path; import sys; print("\n".join(str(p) for p in Path(sys.argv[1]).rglob("*") if p.is_file()))' "$task_epoch/native-sequence-u-pilot-v1")
task_jobs=(712795 712878 712892 713125 713191 713192 713193 713194 713373 713334
    713361 713362 713382 713409 713417 713446 713452 713455 713456 713461 713499 713500)
task_ids=$(IFS=,; printf '%s' "${task_jobs[*]}")
sacct -X -j "$task_ids" -o JobIDRaw,State,ExitCode,Elapsed -n -P > "$task_tools/jobs.txt"
sacct -X -j "$task_ids" -o JobIDRaw,SubmitLine%8000 -n -P > "$task_tools/job-commands.txt"
for task_job in "${task_jobs[@]}"; do
    if test "$task_job" = 713499 || test "$task_job" = 713500; then continue; fi
    for task_path in "$task_epoch"/*"$task_job"*.log "$task_epoch"/*"$task_job"*.err; do
        if test -f "$task_path"; then task_extra+=("$task_path"); fi
    done
done
task_dest="research/rigid_stress/evidence/20261014-physical/profile-history-checkpoint-$task_version"
"$task_env/bin/python" research/rigid_stress/scripts/publish_nonconvex_checkpoint.py \
    --runtime-root "$task_root" --destination "$task_dest" --staging-root "$task_epoch/publication-staging" \
    --run "${task_runs[@]}" --extra "${task_extra[@]}" --summary "$task_tools/summary.json" \
    --limitations "Alternative traction profiles preserve each original six-component wrench and byte-identical rigid inputs; local inadmissibility stays explicit. All seven FEM reference sweeps complete, with full fields separately archived. PCG release and absolute stopping-margin repair preserves the complete acceptance tolerances; original release failures, sampler/capacity and overly strict replay assertion failures remain exact. Native single-environment changing-contact fields and warmed scope are retained. Finer native sweeps, converged resolved egg/Franka and interior-hotspot contact, complete physical operating ranges and remaining traction maps are outside this checkpoint."
sha256sum "${task_sources[@]}" > "$task_tools/source-after.sha256"
cmp "$task_tools/source.sha256" "$task_tools/source-after.sha256"
"$task_env/bin/python" research/rigid_stress/scripts/verify_nonconvex_publication.py \
    --manifest "$task_dest/manifest.json" --originals --output "$task_tools/verification.json"
