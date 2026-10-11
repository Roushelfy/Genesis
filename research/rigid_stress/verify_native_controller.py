"""Independent CPU FP64 oracle for Quadrants controller configuration inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import quadrants as qd
import torch

import genesis as gs
from research.rigid_stress import probe_native_controller_resident as controller


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=1211)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gs.init(backend=gs.gpu, precision="64", logging_level="warning")
    random = np.random.default_rng(631935)
    delays = random.integers(0, 600, args.envs, dtype=np.int64)
    delays[:7] = [0, 1, 89, 180, 420, 539, 599]
    joints = [random.uniform(-2.0, 2.0, (args.envs, 9)) for _ in range(4)]
    base_radius = random.uniform(0.004, 0.008, args.envs)
    arm = np.arange(7, dtype=np.int64)
    inputs = [torch.as_tensor(value, device=gs.device) for value in (delays, *joints, base_radius, arm)]
    ticks = [-1, 0, 1, 89, 90, 119, 179, 180, 200, 299, 300, 419, 420, 455, 456, 539, 540, 599, 600, 943, 1799, 2399]
    reports = []
    scalar_dtype = np.float32 if torch.get_default_dtype() == torch.float32 else np.float64
    for action_dtype in (None, np.float32, np.float64):
        action = random.normal(size=(args.envs, 7)).astype(action_dtype or np.float64)
        action_device = torch.as_tensor(action, device=gs.device)
        output = [
            torch.empty(args.envs, device=gs.device, dtype=torch.float64),
            torch.empty((args.envs, 9), device=gs.device, dtype=torch.float64),
            torch.empty(args.envs, device=gs.device, dtype=torch.get_default_dtype()),
            torch.empty(args.envs, device=gs.device, dtype=torch.get_default_dtype()),
            torch.empty(args.envs, device=gs.device, dtype=torch.float64),
        ]
        maxima = np.zeros(5)
        for tick in ticks:
            controller.kernel_inputs(
                tick, *inputs, action_device, *output, action_dtype is not None, action_dtype == np.float32, True
            )
            phase = np.where(tick >= delays, ((tick - delays) % 600).astype(np.float64) / 599.0, 0.0)
            approach, lift, slide = (
                np.clip(phase / 0.15, 0.0, 1.0),
                np.clip((phase - 0.3) / 0.2, 0.0, 1.0),
                np.clip((phase - 0.6) / 0.15, 0.0, 1.0),
            )
            approach, lift, slide = [value * value * (3.0 - 2.0 * value) for value in (approach, lift, slide)]
            initial, pick, lift_joints, slide_joints = joints
            target = initial * (1.0 - approach[:, None]) + pick * approach[:, None]
            target += (lift_joints - pick) * lift[:, None] + (slide_joints - lift_joints) * slide[:, None]
            if action_dtype is not None:
                target[:, arm] += action * (np.float32(1e-4) if action_dtype == np.float32 else np.float64(1e-4))
            grip = np.where((phase < 0.15) | (phase > 0.9), 0.04, 0.018).astype(scalar_dtype)
            limit = np.where((phase >= 0.7) & (phase < 0.76), 0.055, 4.0).astype(scalar_dtype)
            radius = base_radius * (1.0 + 0.15 * np.sin(2.0 * np.pi * phase))
            qd.sync()
            for i, (actual, expected) in enumerate(zip(output, (phase, target, grip, limit, radius))):
                actual = actual.cpu().numpy()
                maxima[i] = max(maxima[i], float(np.max(np.abs(actual - expected))))
                if i in (2, 3):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9 if i == 1 else 3e-15)
        reports.append(
            {
                "action_dtype": "none" if action_dtype is None else np.dtype(action_dtype).name,
                "max_abs_errors": maxima.tolist(),
            }
        )
    result = {
        "envs": args.envs,
        "ticks": ticks,
        "reports": reports,
        "columns": ["phase", "target_rad", "grip_m", "force_limit_N", "radius_m"],
        "control_error_budget_rad": 1e-9,
        "radius_error_budget_m": 3e-15,
        "kernel_source_sha256": hashlib.sha256(Path(controller.__file__).read_bytes()).hexdigest(),
        "oracle_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Synthetic independent CPU NumPy FP64 configuration-input oracle, including phase boundaries, resets, delayed starts, FP32/FP64 residuals and exact scalar-dtype matching. This is not a throughput or stress convergence measurement.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
