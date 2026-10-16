#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_parent=${2:?Closed original capture required}
test ! -e "$task_run"
mkdir -p "$task_run/input-source" "$task_run/source" "$task_run/tmp"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
export GS_CACHE_FILE_PATH="$task_run/cache/genesis" QD_OFFLINE_CACHE_FILE_PATH="$task_run/cache/quadrants"
python3 - "$task_parent" <<'PY'
import hashlib,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert int((p/'returncode.txt').read_text())==0
assert (p/'source.sha256').read_bytes()==(p/'source-after.sha256').read_bytes()
for line in (p/'source.sha256').read_text().splitlines():
    expected,relative=line.split('  ',1)
    assert hashlib.sha256((p/'source'/relative).read_bytes()).hexdigest()==expected,relative
    assert hashlib.sha256(pathlib.Path(relative).read_bytes()).hexdigest()==expected,relative
PY
mapfile -t task_sources < <(sed 's/^[^ ]*  //' "$task_parent/source.sha256")
task_sources+=(research/rigid_stress/captured_com_velocity_observer.patch research/rigid_stress/scripts/capture_com_velocity.sh
    genesis/engine/entities/rigid_entity/rigid_link.py genesis/constants.py)
sha256sum "${task_sources[@]}" > "$task_run/input-source.sha256"
cp --parents "${task_sources[@]}" "$task_run/input-source"
cp -a "$task_run/input-source/." "$task_run/source/"
patch -p1 -d "$task_run/source" < research/rigid_stress/captured_com_velocity_observer.patch > "$task_run/patch.log"
(cd "$task_run/source"; sha256sum "${task_sources[@]}") > "$task_run/source.sha256"
trap 'task_status=$?; (cd "$task_run/source"; sha256sum "${task_sources[@]}") > "$task_run/source-after.sha256"; sha256sum "${task_sources[@]}" > "$task_run/input-source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; cmp "$task_run/input-source.sha256" "$task_run/input-source-after.sha256"; exit "$task_status"' EXIT
hostname > "$task_run/node.txt"
cp "$task_parent/arguments.txt" "$task_run/arguments.txt"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$task_run/hardware.csv"
"$task_env/bin/python" - "$task_run" "$task_parent" <<'PY' > "$task_run/capture.log" 2>&1
import importlib.util,json,pathlib,shlex,sys,zipfile
run,parent=map(pathlib.Path,sys.argv[1:])
path=run/'source/research/rigid_stress/capture_franka_reference.py'
spec=importlib.util.spec_from_file_location('research.rigid_stress.capture_franka_com_observer',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
sys.argv=[str(path),'--output',str(run/'results'),*shlex.split((parent/'arguments.txt').read_text())]
module.main()
original=json.loads((parent/'results/summary.json').read_text())
observed=json.loads((run/'results/summary.json').read_text())
assert observed['captures']==original['captures']
archives=0;payloads=0
for source in sorted((parent/'results').glob('*.npz')):
    target=run/'results'/source.name
    with zipfile.ZipFile(source) as a,zipfile.ZipFile(target) as b:
        assert a.namelist()==b.namelist(),source
        for name in a.namelist():assert a.read(name)==b.read(name),(source,name)
        payloads+=len(a.namelist())
    archives+=1
result={'passed':True,'original_rows_identical':len(observed['captures']),'original_npz_archives':archives,'original_npy_payloads_identical':payloads}
(run/'observer-equivalence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
PY
