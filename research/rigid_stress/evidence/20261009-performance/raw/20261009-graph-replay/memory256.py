"""Untimed six-frame calibration probe of the unchanged graph/eager reference replay buffers."""

import json
from pathlib import Path
from time import perf_counter

import cupy as cp
from threadpoolctl import threadpool_limits

from research.rigid_stress.cpu import EggConfig, EggRecoveryCPU
from research.rigid_stress.device_pressure import PadPressureGPU
from research.rigid_stress.graph_gpu import CapturedDirectRecoveryGPU
from research.rigid_stress.replay import ContactReplayGPU, ReplayRecoveryPipeline
from research.rigid_stress.sparse_gpu import EggRecoveryGPU

root = Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs')
with threadpool_limits(limits=1), cp.cuda.Stream(non_blocking=True):
    start = perf_counter()
    model = EggRecoveryCPU(256, EggConfig(level=6, ordering='column-nd', factor_backend='none'), direct=True, history=0)
    recovery = CapturedDirectRecoveryGPU(model.fem, 256, inertia='quadratic', body_products='fused', sparse_layout='C')
    source = ContactReplayGPU(root / '20261009-panda-reusable/varied32-grid-temporal.contacts.npz', 256)
    pipeline = ReplayRecoveryPipeline(
        source, PadPressureGPU(model.surface, anchor_to_surface=True, sampling='grid', scatter='atomic'), recovery
    )
    print(json.dumps({'setup_seconds': perf_counter()-start}), flush=True)
    errors = []
    for tick in (0, 100, 200, 300, 450, 550):
        out = pipeline.step(tick)
        exact = EggRecoveryGPU.recover(recovery, pipeline.last_rhs)
        exact_peak, exact_accepted = exact.peak_pa, exact.is_accepted
        assert cp.all(out.is_accepted & exact_accepted & pipeline.mapping_accepted).item()
        error = float((abs(out.peak_pa-exact_peak)/cp.maximum(exact_peak, 1)).max())
        assert error < 1e-4
        errors.append(error)
        print(json.dumps({'tick': tick, 'peak_error': error}), flush=True)
        del exact
    free, total = cp.cuda.runtime.memGetInfo()
    (root/'20261009-graph-replay/memory256.json').write_text(json.dumps({
        'passed': True, 'scope': 'Untimed B=256 graph replay with shared-operator eager reference; no FPS',
        'frames': [0, 100, 200, 300, 450, 550], 'peak_errors': errors,
        'device_memory_after_bytes': total-free, 'device_memory_total_bytes': total,
    }, indent=2)+'\n')
