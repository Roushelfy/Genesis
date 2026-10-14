#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
sha256sum research/rigid_stress/scripts/setup_physical_amg.sh > "$task_run/source.sha256"
cp research/rigid_stress/scripts/setup_physical_amg.sh "$task_run/source/launcher.sh"
"$task_env/bin/python" -m pip install --no-deps --disable-pip-version-check \
    --report "$task_run/pip-install.json" pyamg==5.3.0
"$task_env/bin/python" - "$task_run" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import pyamg

source = Path(pyamg.__file__)
result = {"version": importlib.metadata.version("pyamg"), "source": str(source),
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
Path(sys.argv[1], "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
PY
sha256sum research/rigid_stress/scripts/setup_physical_amg.sh > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
