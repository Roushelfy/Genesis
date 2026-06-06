#!/usr/bin/env bash
# Run an IPC_Solver example in the gs-gym-internal uv environment.
#
# Why not `uv run` from the Genesis repo directly?
#   - The Genesis project pulls the optional `nyx` extra -> `gs-nyx` from a private
#     jfrog index that needs auth (401), so a fresh `uv run`/`uv sync` here fails.
#   - The Genesis repo also doesn't depend on `uipc` (the IPC backend) at all.
# The gs-gym-internal venv already has everything built: the libuipc fork (uipc 0.9.0),
# cu129 torch, and this Genesis checkout (via the editable-finder redirect). `--no-sync`
# uses that prebuilt env as-is, so it neither re-resolves (no 401) nor regenerates the
# editable finder (keeps the redirect to /home/zhaofeng/work/Genesis).
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
