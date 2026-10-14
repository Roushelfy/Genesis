#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export RIGID_STRESS_ENV=${PHYSICAL_VALIDATION_ENV:-/mnt/data/zhaofeng/venvs/rigid-stress-physical}
bash research/rigid_stress/scripts/check_nonconvex_final.sh "$@"
