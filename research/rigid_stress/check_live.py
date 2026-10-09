"""Check all-substep device observations against identical-RHS CPU FP64 direct peaks and finite loads."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import genesis as gs

from .assets import write_egg_assets
from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .device_pressure import PadPressureGPU
from .live import StressSubstepObserver
from .oracle import reference
from .sparse_gpu import EggRecoveryGPU


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(2, EggConfig(level=1), history=0, direct=True, anchor_to_surface=True)
        urdf = write_egg_assets(model, args.output.parent / "assets")
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        rows = []
        for substeps in (1, 4):
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=0.02, substeps=substeps),
                rigid_options=gs.options.RigidOptions(
                    friction_cone=gs.friction_cone.elliptic,
                    contact_resolution=gs.contact_resolution.convex,
                    contact_pruning_tolerance=None,
                    enable_torsional_friction=False,
                    enable_rolling_friction=False,
                    use_hibernation=False,
                ),
                show_viewer=False,
            )
            scene.add_entity(gs.morphs.Plane())
            egg = scene.add_entity(gs.morphs.URDF(file=str(urdf), pos=(0.0, 0.0, 0.12), align=False, decimate=False))
            scene.build(n_envs=2)
            mapper = PadPressureGPU(model.surface, anchor_to_surface=True)
            recovery = EggRecoveryGPU(model.fem, environments=2)
            radii = cp.full((2, 1), 0.009)
            substep_truth = []
            errors, moment_errors = [], []

            def diagnostic(local, mapped, recovered, index):
                if index == 0:
                    substep_truth.clear()
                contacts = ContactBatch(
                    local.position_m.cpu().numpy(),
                    local.force_n.cpu().numpy(),
                    np.full(tuple(local.is_valid.shape), 0.009),
                    local.is_valid.cpu().numpy(),
                    local.friction.cpu().numpy(),
                    local.inward_normal.cpu().numpy(),
                    np.finfo(np.float32).eps,
                )
                loads = model.map_contacts(contacts)
                np.testing.assert_allclose(cp.asnumpy(mapped.nodal_force_n), loads.nodal_force_n, atol=1e-10, rtol=1e-8)
                rhs = recovery.compatible_rhs(
                    mapped.nodal_force_n + recovery.mass_modes[:, :3] @ cp.from_dlpack(local.gravity_m_s2).T,
                    cp.from_dlpack(local.omega_rad_s),
                )
                expected = np.zeros((model.fem.ndof, 2))
                expected[model.fem.free] = model.fem.factor.solve(cp.asnumpy(rhs)[model.fem.free])
                truth = np.array([reference.peak(model.fem, expected[:, i]) for i in range(2)])
                actual = cp.asnumpy(recovered.peak_pa)
                errors.append(float(np.max(abs(actual - truth) / np.maximum(truth, 1))))
                moment_errors.append(
                    float(np.linalg.norm(loads.resultant_moment_nm - loads.input_moment_nm, axis=1).max())
                )
                substep_truth.append(truth)
                assert cp.all(mapped.is_accepted & recovered.is_accepted).item()

            observer = StressSubstepObserver(
                scene, egg, mapper, recovery, radii, substeps, dt=0.02, diagnostic=diagnostic
            )
            maximum_difference = 0.0
            for step in range(30):
                if step == 19:
                    scene.reset(envs_idx=[1])
                    observer.reset(cp.asarray([1]))
                scene.step(update_visualizer=False)
                expected = np.max(substep_truth, axis=0)
                np.testing.assert_allclose(cp.asnumpy(observer.peak_pa), expected, atol=1e-5, rtol=1e-4)
                maximum_difference = max(
                    maximum_difference, float(cp.max(observer.peak_pa - observer.substep_peak_pa[-1]).item())
                )
                torch.testing.assert_close(observer.observation(), torch.from_dlpack(observer.peak_pa))
            if substeps > 1:
                assert maximum_difference > 1
            rows.append(
                {
                    "substeps": substeps,
                    "steps": 30,
                    "recoveries": 30 * substeps * 2,
                    "peak_relative_error_max": max(errors),
                    "moment_error_max_nm": max(moment_errors),
                    "maximum_peak_missed_by_last_substep_pa": maximum_difference,
                    "partial_reset_ids": [1],
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        args.output.write_text(json.dumps({"passed": True, "GPU_tested": True, "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
