"""Untimed repeated-recovery diagnosis; preserve small original charts and all failed-result fields."""
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import quadrants as qd

import genesis as gs
from genesis.utils.array_class import V, V_VEC, V_MAT, iter_data
from genesis.utils.misc import qd_to_numpy
import examples.speed_benchmark.rigid_stress as benchmark

destination = Path(sys.argv[sys.argv.index("--output") + 1]).parent / "failure"
original_prepare = benchmark.prepare_recovery_snapshot

@qd.kernel
def kernel_copy_environment(source: qd.Tensor, target: qd.Tensor, environment: int):
    for index in qd.grouped(target):
        original = qd.Vector.zero(gs.qd_int, len(source.shape))
        for dimension in qd.static(range(len(source.shape))):
            original[dimension] = index[dimension]
        original[len(source.shape) - 1] = environment
        target[index] = source[original]

def snapshot(value, environment, n_envs):
    if math.prod(value.shape) == 0:
        return None
    if value.shape and value.shape[-1] == n_envs:
        shape = (*value.shape[:-1], 1)
        if isinstance(value, qd.VectorTensor):
            target = V_VEC(*value.element_shape, dtype=value.dtype, shape=shape)
        elif isinstance(value, qd.MatrixTensor):
            target = V_MAT(*value.element_shape, dtype=value.dtype, shape=shape)
        else:
            target = V(dtype=value.dtype, shape=shape)
        kernel_copy_environment(value, target, environment)
        return qd_to_numpy(target, transpose=True, copy=True)
    return qd_to_numpy(value, transpose=True, copy=True)

def prepare(workload):
    entry = workload.scene.rigid_solver.stress_recovery.links[0]
    n_envs = workload.n_envs
    items = []
    for name, state in (("contacts", entry.contacts), ("state", entry.state), ("raw_sources", entry.intrinsic.raw_sources), ("fit_contacts", entry.intrinsic.contacts), ("fit_sources", entry.intrinsic.sources), ("tractions", entry.intrinsic.tractions), ("footprints", entry.intrinsic.footprints), ("packing", entry.intrinsic.packing)):
        items.extend(iter_data(state, name))
    before = {}
    for item in items:
        value = item.value
        if math.prod(value.shape) == 0 or (len(value.shape) > 1 and value.shape[0] > 1000):
            continue
        before[item.name] = qd_to_numpy(value, transpose=True, copy=True)
    try:
        return original_prepare(workload)
    except AssertionError:
        peak = qd_to_numpy(entry.state.peak, transpose=True, copy=True)
        original_peak = before["state.peak"]
        differences = abs(peak - original_peak)
        ids = np.flatnonzero(differences > 1e-3 + 1e-4 * abs(original_peak))
        assert len(ids)
        environment = int(ids[np.argmax(differences[ids])])
        destination.mkdir(parents=True, exist_ok=False)
        values = {"environment": np.array(environment)}
        for item in items:
            if item.name in before:
                array = before[item.name]
                if item.value.shape and item.value.shape[-1] == n_envs:
                    array = array[environment:environment + 1]
                values["before." + item.name] = array
            array = snapshot(item.value, environment, n_envs)
            if array is not None:
                values["after." + item.name] = array
        np.savez_compressed(destination / "before-after-recovery.npz", **values)
        metadata = {"scope": __doc__, "command": sys.argv, "selected_environment": environment,
                    "failed_environments": ids.tolist(), "before_peak_Pa": original_peak[ids].tolist(),
                    "after_peak_Pa": peak[ids].tolist(), "peak_difference_Pa": differences[ids].tolist(),
                    "limitations": "Before retains small charts,dual,prior and nodal state,not large full-batch sample profiles. After retains every selected-environment field. Failed consistency diagnosis;no qualified throughput."}
        (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata), flush=True)
        raise

with patch.object(benchmark, "prepare_recovery_snapshot", prepare):
    benchmark.main()
