#!/usr/bin/env bash
set -eo pipefail
source /mnt/data/shared/config/env.sh
GENESIS_IMAGE_VER=1_26 gs-srun --ntasks=1 /mnt/data/zhaofeng/venvs/rigid-stress-physical/bin/python -c 'import ipctk; print(ipctk.CollisionMesh.__init__.__doc__); print(ipctk.CollisionMesh.build_from_full_mesh.__doc__); print(ipctk.CollisionMesh.vertices.__doc__); print(ipctk.CollisionMesh.to_full_dof.__doc__)'
