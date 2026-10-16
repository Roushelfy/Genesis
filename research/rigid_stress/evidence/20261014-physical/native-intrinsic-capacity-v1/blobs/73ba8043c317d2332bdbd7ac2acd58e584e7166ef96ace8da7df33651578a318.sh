#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source research/rigid_stress/scripts/cluster_env.sh
task_env=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
task_run=${1:?Fresh runtime directory required}
task_assets=${2:?Fresh data-side dependency directory required}
test ! -e "$task_run"
test ! -e "$task_assets"
mkdir -p "$task_run/source" "$task_run/tmp" "$task_assets/wheels" "$task_assets/site-packages"
export TMPDIR="$task_run/tmp" PIP_CACHE_DIR=/mnt/data/zhaofeng/pip_cache
task_sources=(research/rigid_stress/scripts/{setup_reference_metis,cluster_env}.sh)
sha256sum "${task_sources[@]}" > "$task_run/source.sha256"
cp --parents "${task_sources[@]}" "$task_run/source"
trap 'task_status=$?; sha256sum "${task_sources[@]}" > "$task_run/source-after.sha256"; printf "%s\n" "$task_status" > "$task_run/returncode.txt"; cmp "$task_run/source.sha256" "$task_run/source-after.sha256"; exit "$task_status"' EXIT
hostname > "$task_run/node.txt"
"$task_env/bin/python" -m pip download --no-deps --only-binary=:all: --dest "$task_assets/wheels" pymetis==2025.2.2 > "$task_run/download.log" 2>&1
task_wheels=("$task_assets"/wheels/*.whl)
test "${#task_wheels[@]}" -eq 1
sha256sum "${task_wheels[@]}" > "$task_run/dependency-wheel.sha256"
"$task_env/bin/python" -m pip install --no-deps --no-index --target "$task_assets/site-packages" "${task_wheels[@]}" > "$task_run/install.log" 2>&1
export PYTHONPATH="$task_assets/site-packages"
"$task_env/bin/python" - "$task_assets" <<'PY' > "$task_run/qualification.json"
import hashlib, importlib.metadata, json, sys
from pathlib import Path
import numpy as np
import pymetis
import scipy
from scipy import sparse
from scipy.sparse.linalg import splu

assert importlib.metadata.version('pymetis') == '2025.2.2'
n = 125
matrix = sparse.diags([-np.ones(n-1), 4*np.ones(n), -0.7*np.ones(n-1)], [-1,0,1], format='csc')
graph = (abs(matrix) + abs(matrix.T)).tocsr()
graph.setdiag(0); graph.eliminate_zeros(); graph.sort_indices()
dtype = pymetis.zero_copy_dtype()
adjacency = pymetis.CSRAdjacency(np.asarray(graph.indptr,dtype=dtype), np.asarray(graph.indices,dtype=dtype))
permutation, inverse = [np.asarray(x,dtype=np.int64) for x in pymetis.nested_dissection(adjacency, options=pymetis.Options(seed=20261015))]
np.testing.assert_array_equal(permutation[inverse], np.arange(n))
known = np.linspace(-0.4,0.7,n); rhs = matrix @ known
factor = splu(matrix[permutation][:,permutation].tocsc(), permc_spec='NATURAL')
solved = np.empty(n); solved[permutation] = factor.solve(rhs[permutation])
np.testing.assert_allclose(solved, known, rtol=1e-13, atol=1e-14)
assert np.linalg.norm(matrix @ solved-rhs) <= 1e-12
root = Path(sys.argv[1]); identities={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
print(json.dumps({'scope':'Pinned isolated dependency and nonsymmetric permutation round-trip; no physical reference qualification.',
 'dependency_root':str(root),'python':sys.version,'pymetis':importlib.metadata.version('pymetis'),
 'numpy':np.__version__,'scipy':scipy.__version__,'graph_index_dtype':str(dtype),
 'known_solution_max_error':float(abs(solved-known).max()),'full_residual':float(np.linalg.norm(matrix@solved-rhs)),
 'dependency_sha256':identities,'production_or_active_environment_modified':False,'goal_completion':False},indent=2))
PY
