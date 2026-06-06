#!/usr/bin/env bash
# Fallback runner: run an IPC_Solver example in the gs-gym-internal uv environment.
#
# Normally you can run these examples directly from the Genesis repo:
#   cd ~/work/Genesis && uv run examples/IPC_Solver/<script.py> [args...]
# The Genesis uv env now carries the libuipc fork (uipc 0.9.0) + cu129 torch via the
# default `ipc` dependency group, and gs-nyx resolves from public PyPI (no jfrog 401).
#
# Use this script only to reuse the *prebuilt* gs-gym-internal venv as-is (e.g. to skip
# any sync entirely). `--no-sync` uses that env without re-resolving or regenerating the
# editable finder, so it keeps any editable-finder redirect to /home/zhaofeng/work/Genesis.
#
# Usage:
#   examples/IPC_Solver/run.sh ipc_monolithic_grasp_tune.py --vis --realtime
#   examples/IPC_Solver/run.sh ipc_robot_grasp_cube.py --coup_type ipc_monolithic --abd
#   GS_GYM_DIR=/path/to/gs-gym-internal examples/IPC_Solver/run.sh <script.py> [args...]
set -euo pipefail

GS_GYM="${GS_GYM_DIR:-$HOME/work/gs-gym-internal}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <example.py> [args...]" >&2
    exit 2
fi
script="$1"; shift

if [[ ! -d "$GS_GYM" ]]; then
    echo "error: gs-gym-internal not found at '$GS_GYM' (set GS_GYM_DIR)" >&2
    exit 1
fi

( cd "$GS_GYM" && exec uv run --no-sync python "$HERE/$script" "$@" )
