"""Exact rigid-data audit around the actual pre-integration recovery call."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd

import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.engine.solvers.rigid.stress.recovery import RigidStressRecovery
from genesis.utils.array_class import CHECKPOINT_KINDS
from genesis.utils.misc import qd_to_numpy


ROOT = Path("/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex/rigid-immutability-v2")
ORIGINAL_RECOVER = RigidStressRecovery.recover
CURRENT_TICK = -1
RECORDS = []
ITEMS = []
WORKLOAD = None


def audited_recover(self, i_substep):
    selected = CURRENT_TICK % 40 == 0 or CURRENT_TICK == 639
    if not selected:
        return ORIGINAL_RECOVER(self, i_substep)
    qd.sync()
    before = {item.name: qd_to_numpy(item.value, transpose=True, copy=True) for item in ITEMS}
    result = ORIGINAL_RECOVER(self, i_substep)
    qd.sync()
    after = {item.name: qd_to_numpy(item.value, transpose=True, copy=True) for item in ITEMS}
    for name in before:
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    digest_before = hashlib.sha256(b"".join(before[item.name].tobytes() for item in ITEMS)).hexdigest()
    digest_after = hashlib.sha256(b"".join(after[item.name].tobytes() for item in ITEMS)).hexdigest()
    assert digest_before == digest_after
    record = {
        "tick": CURRENT_TICK,
        "substep": i_substep,
        "rigid_data_fields": len(ITEMS),
        "before_sha256": digest_before,
        "after_sha256": digest_after,
        "reset_environments": WORKLOAD.reset_count,
    }
    RECORDS.append(record)
    print(json.dumps(record), flush=True)
    return result


ROOT.mkdir(exist_ok=True)
gs.init(backend=gs.cuda, precision="64", logging_level="warning")
WORKLOAD = FrankaEgg(4, seed=623001, varied=True, contact_model="intrinsic", contact_block_size=2, egg_geometry="mesh64")
SOLVER = WORKLOAD.scene.rigid_solver
ITEMS = [item for item in SOLVER.data if item.kind in CHECKPOINT_KINDS and not item.name.startswith("stress.")]
completed = 0
try:
    with patch.object(RigidStressRecovery, "recover", audited_recover):
        for CURRENT_TICK in range(640):
            WORKLOAD.step()
            SOLVER.check_errno()
            completed += 1
finally:
    (ROOT / "summary.json").write_text(
        json.dumps(
            {
                "scope": __doc__,
                "completed_steps": completed,
                "records": RECORDS,
                "rigid_data_names": [item.name for item in ITEMS],
                "reset_environments": WORKLOAD.reset_count,
            },
            indent=2,
        ) + "\n"
    )
