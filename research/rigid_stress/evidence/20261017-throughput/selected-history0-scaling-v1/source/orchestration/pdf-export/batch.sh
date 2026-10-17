#!/usr/bin/env bash
set -euo pipefail
task_run=/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-pdf-v1
test ! -e "$task_run"
mkdir -p "$task_run/source" "$task_run/input" "$task_run/tmp" "$task_run/results"
cp /mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/analysis-source/final-throughput-pdf-v1/* "$task_run/source/"
cd "$task_run"
/usr/bin/python3 - <<'PREFLIGHT' > preflight.log
from pathlib import Path
import json,hashlib,shutil,subprocess
p=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput');run=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261017-practical-throughput/final-throughput-pdf-v1')
scheduler=subprocess.check_output(['sacct','-j','725302,725303','--format=JobID,State,ExitCode,Elapsed,NodeList','-P'],text=True)
assert all(f'{job}_0|COMPLETED|0:0|' in scheduler for job in ['725302','725303'])
(run/'upstream-scheduler.txt').write_text(scheduler)
for name in ['final-throughput-figure-preview-v2','final-throughput-comparison-preview-v3']:
 root=p/name
 assert all((root/f).read_text().strip()=='0' for f in ['program-returncode.txt','returncode.txt','batch-returncode.txt'])
 assert (root/'source-before.sha256').read_bytes()==(root/'source-after.sha256').read_bytes()
for label in ['history4','history0','comparison']:
 f=p/'analysis'/f'final-{label}-visual-review-v1.json';d=json.loads(f.read_text())
 assert d['preview_inspected'] and d['visual_result']=='pass' and not d['remaining_visual_issues']
 assert hashlib.sha256(Path(d['preview_path']).read_bytes()).hexdigest()==d['preview_sha256']
 shutil.copy2(f,run/'input'/f.name)
for history in [4,0]:
 a=p/f'final-throughput-figure-preview-v2/results/history{history}.json'
 b=p/f'final-throughput-comparison-preview-v3/input/history{history}.json'
 assert a.read_bytes()==b.read_bytes()
 shutil.copy2(a,run/'input'/a.name)
print(json.dumps({'accepted_preview_histories':[4,0],'accepted_comparison':True,'upstream_terminal0':True}))
PREFLIGHT
sha256sum source/* input/* > source-before.sha256
set +u
source /mnt/data/shared/config/env.sh
set +e
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 bash "$task_run/source/analysis.sh"
task_status=$?
printf '%s\n' "$task_status" > batch-returncode.txt
exit "$task_status"
