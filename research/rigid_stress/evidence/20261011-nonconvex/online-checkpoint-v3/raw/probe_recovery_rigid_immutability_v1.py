"""Exact same-scene rigid data comparison across ordinary native recovery replay."""
import hashlib
import json
from pathlib import Path
import numpy as np
import quadrants as qd
import genesis as gs
from examples.rigid.franka_egg_stress import FrankaEgg
from genesis.utils.array_class import CHECKPOINT_KINDS
from genesis.utils.misc import qd_to_numpy

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261011-nonconvex/rigid-immutability-v1')
root.mkdir(exist_ok=True)
gs.init(backend=gs.cuda, precision='64', logging_level='warning')
workload = FrankaEgg(4, seed=623001, varied=True, contact_model='intrinsic', contact_block_size=2, egg_geometry='mesh64')
solver = workload.scene.rigid_solver
items = [item for item in solver.data if item.kind in CHECKPOINT_KINDS and not item.name.startswith('stress.')]
records = []
completed = 0
try:
    for tick in range(640):
        workload.step()
        solver.check_errno()
        if tick % 40 == 0 or tick == 639:
            qd.sync()
            before = {item.name: qd_to_numpy(item.value, transpose=True, copy=True) for item in items}
            solver.stress_recovery.recover(0)
            qd.sync()
            solver.check_errno()
            after = {item.name: qd_to_numpy(item.value, transpose=True, copy=True) for item in items}
            for name in before:
                np.testing.assert_array_equal(after[name], before[name], err_msg=name)
            digest_before = hashlib.sha256(b''.join(before[item.name].tobytes() for item in items)).hexdigest()
            digest_after = hashlib.sha256(b''.join(after[item.name].tobytes() for item in items)).hexdigest()
            assert digest_before == digest_after
            record = {'tick': tick, 'rigid_data_fields': len(items), 'before_sha256': digest_before, 'after_sha256': digest_after, 'reset_environments': workload.reset_count}
            records.append(record)
            print(json.dumps(record), flush=True)
        completed += 1
finally:
    (root / 'summary.json').write_text(json.dumps({'scope': __doc__, 'completed_steps': completed, 'records': records, 'rigid_data_names': [item.name for item in items]}, indent=2) + '\n')
