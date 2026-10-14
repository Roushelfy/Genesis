#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
sha256sum research/rigid_stress/scripts/setup_physical_contact.sh > "$task_run/source.sha256"
cp research/rigid_stress/scripts/setup_physical_contact.sh "$task_run/source/launcher.sh"
"$task_env/bin/python" -m pip install --no-deps --disable-pip-version-check \
    --report "$task_run/pip-install.json" ipctk==1.6.0
"$task_env/bin/python" - "$task_run" <<'PY'
import hashlib
import importlib.metadata
import inspect
import json
import sys
from pathlib import Path

import ipctk

source = Path(inspect.getfile(ipctk))
result = {"version": importlib.metadata.version("ipctk"), "source": str(source),
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
Path(sys.argv[1], "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
print(ipctk.BarrierPotential.__init__.__doc__)
print(ipctk.FrictionPotential.__init__.__doc__)
print(ipctk.TangentialCollisions.build.__doc__)
print(ipctk.CollisionMesh.build_from_full_mesh.__doc__)
print(ipctk.CollisionMesh.to_full_dof.__doc__)
print(ipctk.FrictionPotential.gradient.__doc__)
PY
sha256sum research/rigid_stress/scripts/setup_physical_contact.sh > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
