#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-figure-preview-v2
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/tmp" "$task_run/results"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/analysis-source/final-throughput-figure-preview-v2/* "$task_run/source/"
cd "$task_run"
sha256sum source/* > source-before.sha256
/usr/bin/python3 - <<'SCHEDULER' > metadata-preflight.log
from pathlib import Path
import subprocess,json
p2=Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput");run=Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-figure-preview-v2")
required=["725216","725251","725252"]
text=subprocess.check_output(["sacct","-j",",".join(required),"--format=JobID,State,ExitCode","-P"],text=True)
assert all(f"{j}_0|COMPLETED|0:0" in text for j in required),text
ids=sorted({j["stdout"].strip().split(";")[0] for f in (p2/"meta").glob("*-command.json") if (j:=json.loads(f.read_text())).get("returncode")==0 and j.get("stdout","").strip().split(";")[0].isdigit()},key=int)
(run/"scheduler.txt").write_text(subprocess.check_output(["sacct","-j",",".join(ids),"--format=JobID,State,ExitCode,Elapsed,NodeList","-P"],text=True))
print(json.dumps(dict(required_terminal_jobs=required,job_records=len(ids))))
SCHEDULER
set +u
source /mnt/data/shared/config/env.sh
set +e
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "$task_run/source/analysis.sh"
task_status=$?
printf '%s\n' "$task_status" > batch-returncode.txt
exit "$task_status"
