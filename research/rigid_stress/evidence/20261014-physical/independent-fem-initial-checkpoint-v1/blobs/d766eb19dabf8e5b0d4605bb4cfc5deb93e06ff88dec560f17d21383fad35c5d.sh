#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_run=${1:?New runtime directory required}
task_env=${2:?New reference environment required}
test ! -e "$task_run"
test ! -e "$task_env"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
sha256sum research/rigid_stress/scripts/setup_physical_env.sh > "$task_run/source.sha256"
cp research/rigid_stress/scripts/setup_physical_env.sh "$task_run/source/launcher.sh"
git rev-parse HEAD > "$task_run/base-revision.txt"
"$RIGID_STRESS_ENV/bin/python" -m venv --system-site-packages "$task_env"
task_site=$("$task_env/bin/python" -c 'import site; print(site.getsitepackages()[0])')
task_base_site=$("$RIGID_STRESS_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
printf '%s\n' "$task_base_site" > "$task_site/rigid_stress_reference_base.pth"
"$task_env/bin/python" -m pip install --no-deps --disable-pip-version-check \
    --report "$task_run/pip-install.json" \
    scikit-fem==12.0.2 gmsh==4.14.1 meshio==5.3.5 pygeodesic==0.1.11
"$task_env/bin/python" - "$task_run" <<'PY'
import hashlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path

import gmsh
import meshio
import numpy
import pygeodesic.geodesic
import scipy
import skfem

result = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(), "packages": {}}
for module, name in ((skfem, "scikit-fem"), (gmsh, "gmsh"), (meshio, "meshio"),
                     (pygeodesic.geodesic, "pygeodesic"), (numpy, "numpy"), (scipy, "scipy")):
    path = Path(inspect.getfile(module))
    result["packages"][name] = {"version": importlib.metadata.version(name), "source": str(path),
                               "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
Path(sys.argv[1], "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
sha256sum research/rigid_stress/scripts/setup_physical_env.sh > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
