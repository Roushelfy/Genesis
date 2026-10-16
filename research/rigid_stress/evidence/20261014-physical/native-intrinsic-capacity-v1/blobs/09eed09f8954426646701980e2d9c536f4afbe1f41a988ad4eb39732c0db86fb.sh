#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_run=${1:?New runtime directory required}
test ! -e "$task_run"
mkdir -p "$task_run/tmp" "$task_run/source"
export TMPDIR="$task_run/tmp" PYTHONUNBUFFERED=1
sha256sum research/rigid_stress/scripts/inspect_physical_env.sh > "$task_run/source.sha256"
cp research/rigid_stress/scripts/inspect_physical_env.sh "$task_run/source/launcher.sh"
git rev-parse HEAD > "$task_run/base-revision.txt"
"$RIGID_STRESS_ENV/bin/python" - "$task_run" <<'PY'
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path

result = {"python": sys.version, "platform": platform.platform(), "hostname": platform.node(), "packages": {}}
for module, distribution in (
    ("numpy", "numpy"), ("scipy", "scipy"), ("skfem", "scikit-fem"), ("gmsh", "gmsh"),
    ("meshio", "meshio"), ("matplotlib", "matplotlib"), ("pygeodesic", "pygeodesic"),
    ("sfepy", "sfepy"), ("dolfinx", "fenics-dolfinx"), ("quadrants", "quadrants"),
):
    spec = importlib.util.find_spec(module)
    if spec is None:
        result["packages"][module] = {"available": False}
        continue
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "distribution metadata unavailable"
    source = Path(spec.origin)
    result["packages"][module] = {
        "available": True, "version": version, "module_source": str(source),
        "module_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
Path(sys.argv[1], "environment.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
sha256sum research/rigid_stress/scripts/inspect_physical_env.sh > "$task_run/source-after.sha256"
cmp "$task_run/source.sha256" "$task_run/source-after.sha256"
