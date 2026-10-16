#!/usr/bin/env bash
source research/rigid_stress/scripts/cluster_env.sh
mkdir -p "$RIGID_STRESS_DATA_ROOT/cache" "$RIGID_STRESS_DATA_ROOT/runs/20261009-baseline"
if [[ ! -x "$RIGID_STRESS_ENV/bin/python" ]]; then
    python -m venv --system-site-packages "$RIGID_STRESS_ENV"
fi
"$RIGID_STRESS_ENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$RIGID_STRESS_ENV/bin/python" -m pip install \
    -r research/rigid_stress/requirements-gpu.txt -r research/rigid_stress/requirements-checks.txt \
    -r research/rigid_stress/requirements-cholmod.txt -r research/rigid_stress/requirements-cudss.txt
"$RIGID_STRESS_ENV/bin/python" -c 'import sys, torch, scipy, cupy; print(sys.executable, torch.__version__, scipy.__version__, cupy.__version__); print(cupy.cuda.runtime.getDeviceProperties(0))'
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.check_cpu --output "$RIGID_STRESS_DATA_ROOT/runs/20261009-baseline/cpu.json"
"$RIGID_STRESS_ENV/bin/python" -m research.rigid_stress.check_gpu --output "$RIGID_STRESS_DATA_ROOT/runs/20261009-baseline/gpu-direct.json"
"$RIGID_STRESS_ENV/bin/python" -m pip install -e .
"$RIGID_STRESS_ENV/bin/python" -c 'import genesis; print(genesis.__file__)'
