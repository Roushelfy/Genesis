"""Record a real rigid Panda grasp and diagnose auxiliary CPU stress recovery.

Approach, clamp, lift, hold, lateral motion with weakened gripping, and release use actual rigid contacts. Per-step
diagnostics and optional video include host transfers and compilation, so this command is not a throughput benchmark.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from threadpoolctl import threadpool_limits

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils.misc import tensor_to_array

from .assets import write_egg_assets
from .cpu import ContactBatch, EggConfig, EggRecoveryCPU

try:
    import cupy as cp

    from .device_pressure import PadPressureGPU
    from .sparse_gpu import EggRecoveryGPU
except ModuleNotFoundError as import_error:
    if import_error.name != "cupy":
        raise
    cp = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--stress", choices=("cpu", "gpu", "off"), default="cpu")
    parser.add_argument("--verify-gpu", action="store_true")
    parser.add_argument("--save-contacts", action="store_true")
    parser.add_argument("--envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--patch-radius-m", type=float, default=0.006)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.envs < 1 or args.steps < 100:
        raise ValueError("Positive environment count and at least 100 steps required")
    if args.stress == "gpu" and (args.backend != "cuda" or cp is None):
        raise ValueError("GPU stress requires the CUDA rigid backend and CuPy")
    if args.verify_gpu and args.stress != "gpu":
        raise ValueError("GPU verification requires GPU stress")
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(args.envs, EggConfig(level=args.level), rtol=args.rtol, history=0, direct=True)
        assets = args.output.parent / "assets"
        urdf = write_egg_assets(model, assets)
        gs.init(backend=gs.gpu if args.backend == "cuda" else gs.cpu, precision="32", seed=0)
        gpu_mapper = PadPressureGPU(model.surface) if args.stress == "gpu" else None
        gpu_recovery = EggRecoveryGPU(model.fem, args.envs, rtol=args.rtol) if args.stress == "gpu" else None
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=args.dt,
                substeps=1,
            ),
            rigid_options=gs.options.RigidOptions(
                friction_cone=gs.friction_cone.elliptic,
                contact_pruning_tolerance=None,
                enable_torsional_friction=False,
                enable_rolling_friction=False,
                use_hibernation=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.9, -0.35, 0.36),
                camera_lookat=(0.65, 0.0, 0.15),
            ),
            show_viewer=args.vis,
        )
        scene.add_entity(
            gs.morphs.Plane(),
        )
        franka = scene.add_entity(
            gs.morphs.MJCF(
                file="xml/franka_emika_panda/panda.xml",
            ),
            vis_mode="collision",
        )
        egg = scene.add_entity(
            gs.morphs.URDF(
                file=str(urdf.resolve()),
                pos=(0.65, 0.0, 0.031),
                fixed=False,
                align=False,
                convexify=True,
                decimate=False,
            ),
            material=gs.materials.Rigid(
                friction=0.6,
            ),
            vis_mode="collision",
        )
        camera = None
        if args.video is not None:
            args.video.parent.mkdir(parents=True, exist_ok=True)
            camera = scene.add_camera(
                res=(640, 480),
                pos=(0.9, -0.35, 0.36),
                lookat=(0.65, 0.0, 0.15),
                fov=40,
                GUI=False,
            )
        scene.build(n_envs=args.envs, env_spacing=(1.0, 1.0))
        np.testing.assert_allclose(tensor_to_array(egg.get_mass()), model.fem.mass, rtol=1e-6)
        np.testing.assert_allclose(tensor_to_array(egg.get_links_COM()), model.fem.com[None], rtol=1e-6, atol=1e-9)
        inertia_eigenvalues = np.linalg.eigvalsh(tensor_to_array(egg.get_links_inertia()))
        shell_eigenvalues = np.linalg.eigvalsh(model.fem.gram[3:, 3:])
        assert np.max(abs(inertia_eigenvalues - shell_eigenvalues) / shell_eigenvalues) <= 1e-6
        arm = [franka.get_joint(f"joint{i}").dofs_idx_local[0] for i in range(1, 8)]
        fingers = [franka.get_joint(f"finger_joint{i}").dofs_idx_local[0] for i in (1, 2)]
        franka.set_dofs_kp([4500, 4500, 3500, 3500, 2000, 2000, 2000], arm)
        franka.set_dofs_kv([450, 450, 350, 350, 200, 200, 200], arm)
        franka.set_dofs_kp([100.0, 100.0], fingers)
        franka.set_dofs_kv([10.0, 10.0], fingers)
        franka.set_dofs_force_range([-4.0, -4.0], [4.0, 4.0], fingers)
        initial = torch.tensor(
            [-1.0124, 1.5559, 1.3662, -1.6878, -1.5799, 1.7757, 1.4602, 0.04, 0.04], dtype=gs.tc_float, device=gs.device
        ).repeat(args.envs, 1)
        franka.set_qpos(initial)
        # Different centres/amplitudes are allowed inputs, never an algorithm
        # dependency. The complete production suite must also vary orientations,
        # patch sizes, friction, contact transitions and environment phases.
        centers = torch.tensor([0.65, 0.0, 0.031], dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1)
        centers[:, 1] += torch.linspace(-0.006, 0.006, args.envs, device=gs.device) if args.envs > 1 else 0
        egg.set_pos(centers)
        hand = franka.get_link("hand")
        orientation = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1)
        pickup = centers.clone()
        pickup[:, 2] = 0.135
        raised = pickup.clone()
        raised[:, 2] += 0.12
        q_pick = franka.inverse_kinematics(link=hand, pos=pickup, quat=orientation)
        q_lift = franka.inverse_kinematics(link=hand, pos=raised, quat=orientation)
        translated = raised.clone()
        translated[:, 1] += 0.025
        q_slide = franka.inverse_kinematics(link=hand, pos=translated, quat=orientation)
        if camera is not None:
            camera.start_recording(save_to_filename=str(args.video), fps=50)
        records = []
        contact_records = []
        trace = args.output.with_suffix(".frames.jsonl").open("w")
        start = perf_counter()
        for step in range(args.steps):
            phase = step / (args.steps - 1)
            blend = np.clip((phase - 0.3) / 0.2, 0.0, 1.0)
            blend = blend**3 * (10 - 15 * blend + 6 * blend**2)
            target = q_pick * (1 - blend) + q_lift * blend
            slide = np.clip((phase - 0.6) / 0.15, 0.0, 1.0)
            slide = slide**3 * (10 - 15 * slide + 6 * slide**2)
            target += (q_slide - q_lift) * slide
            franka.control_dofs_position(target[:, arm], arm)
            grip = 0.04 if phase < 0.15 or phase > 0.9 else 0.018
            finger_limit = 0.055 if 0.7 <= phase < 0.76 else 4.0
            franka.set_dofs_force_range(-finger_limit, finger_limit, fingers)
            franka.control_dofs_position(grip, fingers)
            solve_position = egg.get_pos().clone()
            solve_quat = egg.get_quat().clone()
            solve_omega = egg.get_ang().clone()
            links_com = scene.rigid_solver.get_links_pos(ref=gs.link_ref_frame.link_COM).clone()
            links_velocity = scene.rigid_solver.get_links_vel(ref=gs.link_ref_frame.link_COM).clone()
            links_angular = scene.rigid_solver.get_links_ang().clone()
            scene.step()
            contacts = egg.get_contacts(is_padded=True)
            on_a = (contacts["geom_a"] >= egg.geom_start) & (contacts["geom_a"] < egg.geom_end)
            force = torch.where(on_a[..., None], contacts["force_a"], contacts["force_b"])
            force = torch.where(contacts["valid_mask"][..., None], force, 0)
            quat = egg.get_quat()
            position = egg.get_pos()
            normal_force = (force * contacts["normal"]).sum(dim=-1, keepdim=True) * contacts["normal"]
            tangent = torch.linalg.vector_norm(force - normal_force, dim=-1)
            opposite_link = torch.where(on_a, contacts["link_b"], contacts["link_a"])
            is_grasp = (opposite_link >= franka.link_start) & (opposite_link < franka.link_end)
            env_ids = torch.arange(args.envs, device=gs.device)[:, None]
            link_a = contacts["link_a"].clamp(min=0)
            link_b = contacts["link_b"].clamp(min=0)
            velocity_a = links_velocity[env_ids, link_a] + torch.cross(
                links_angular[env_ids, link_a], contacts["position"] - links_com[env_ids, link_a], dim=-1
            )
            velocity_b = links_velocity[env_ids, link_b] + torch.cross(
                links_angular[env_ids, link_b], contacts["position"] - links_com[env_ids, link_b], dim=-1
            )
            relative_velocity = torch.where(on_a[..., None], velocity_a - velocity_b, velocity_b - velocity_a)
            tangent_velocity = (
                relative_velocity
                - (relative_velocity * contacts["normal"]).sum(dim=-1, keepdim=True) * contacts["normal"]
            )
            com_offset = gu.transform_by_quat(
                torch.as_tensor(model.fem.com, dtype=gs.tc_float, device=gs.device), solve_quat
            )
            solve_com = solve_position + com_offset
            angular_acceleration = egg.get_links_acc_ang()[:, 0]
            origin_acceleration = egg.get_links_acc()[:, 0]
            # The public getter transports cached spatial acceleration with post-integration velocities and positions.
            # Undo that transport before pairing the cached acceleration with the contact solve's pre-integration state.
            post_com_offset = gu.transform_by_quat(
                torch.as_tensor(model.fem.com, dtype=gs.tc_float, device=gs.device), quat
            )
            spatial_acceleration = (
                origin_acceleration
                + torch.cross(angular_acceleration, post_com_offset, dim=-1)
                - torch.cross(egg.get_ang(), egg.get_vel(), dim=-1)
            )
            measured_com_acceleration = spatial_acceleration + torch.cross(
                solve_omega, links_velocity[:, egg.link_start], dim=-1
            )
            force_sum = force.sum(dim=-2)
            torque_sum = torch.cross(contacts["position"] - solve_com[:, None], force, dim=-1).sum(dim=-2)
            gravity_world = torch.tensor([0.0, 0.0, -9.81], dtype=gs.tc_float, device=gs.device)
            balance_force = force_sum + model.fem.mass * (gravity_world - measured_com_acceleration)
            omega_body = gu.inv_transform_by_quat(solve_omega, solve_quat)
            alpha_body = gu.inv_transform_by_quat(angular_acceleration, solve_quat)
            inertia_body = torch.as_tensor(model.fem.gram[3:, 3:], dtype=gs.tc_float, device=gs.device)
            balance_torque = gu.inv_transform_by_quat(torque_sum, solve_quat) - (
                alpha_body @ inertia_body.T + torch.cross(omega_body, omega_body @ inertia_body.T, dim=-1)
            )
            net_force_error = force_sum - egg.get_links_net_contact_force()[:, 0]
            record = {
                "step": step,
                "phase": phase,
                "position_m": tensor_to_array(position).tolist(),
                "com_position_m": tensor_to_array(
                    position
                    + gu.transform_by_quat(torch.as_tensor(model.fem.com, dtype=gs.tc_float, device=gs.device), quat)
                ).tolist(),
                "hand_position_m": tensor_to_array(hand.get_pos()).tolist(),
                "egg_velocity_m_s": tensor_to_array(egg.get_vel()).tolist(),
                "hand_velocity_m_s": tensor_to_array(hand.get_vel()).tolist(),
                "contact_count": tensor_to_array(contacts["valid_mask"].sum(dim=-1)).tolist(),
                "grasp_contact_count": tensor_to_array((is_grasp & contacts["valid_mask"]).sum(dim=-1)).tolist(),
                "contact_force_n": tensor_to_array(force.sum(dim=-2)).tolist(),
                "tangent_force_norm_sum_n": tensor_to_array(tangent.sum(dim=-1)).tolist(),
                "egg_as_a_contact_count": tensor_to_array((on_a & contacts["valid_mask"]).sum(dim=-1)).tolist(),
                "egg_as_b_contact_count": tensor_to_array((~on_a & contacts["valid_mask"]).sum(dim=-1)).tolist(),
                "pre_solve_grasp_tangent_speed_max_m_s": tensor_to_array(
                    torch.where(
                        is_grasp & contacts["valid_mask"], torch.linalg.vector_norm(tangent_velocity, dim=-1), 0
                    )
                    .max(dim=-1)
                    .values
                ).tolist(),
                "pre_solve_friction_power_w": tensor_to_array(
                    ((force - normal_force) * tangent_velocity).sum(dim=(-2, -1))
                ).tolist(),
                "net_contact_force_difference_n": tensor_to_array(
                    torch.linalg.vector_norm(net_force_error, dim=-1)
                ).tolist(),
                "newton_balance_force_error_n": tensor_to_array(
                    torch.linalg.vector_norm(balance_force, dim=-1)
                ).tolist(),
                "euler_balance_torque_error_nm": tensor_to_array(
                    torch.linalg.vector_norm(balance_torque, dim=-1)
                ).tolist(),
                "measured_com_acceleration_m_s2": tensor_to_array(measured_com_acceleration).tolist(),
                "inferred_com_acceleration_m_s2": tensor_to_array(force_sum / model.fem.mass + gravity_world).tolist(),
            }
            if args.stress in ("cpu", "gpu") or args.save_contacts:
                # Keep filtering/transforms batched on device; transfer once per
                # selected field into this explicitly CPU-only reference path.
                local_position = gu.inv_transform_by_quat(
                    contacts["position"] - solve_position[:, None], solve_quat[:, None]
                )
                local_force = gu.inv_transform_by_quat(force, solve_quat[:, None])
                inward_normal = torch.where(on_a[..., None], contacts["normal"], -contacts["normal"])
                local_normal = gu.inv_transform_by_quat(inward_normal, solve_quat[:, None])
                gravity = gu.inv_transform_by_quat(
                    torch.tensor([0.0, 0.0, -9.81], dtype=gs.tc_float, device=gs.device).repeat(args.envs, 1),
                    solve_quat,
                )
                local_omega = gu.inv_transform_by_quat(solve_omega, solve_quat)
            if args.save_contacts:
                ids = torch.nonzero(contacts["valid_mask"], as_tuple=False)
                contact_records.append(
                    (
                        step,
                        tensor_to_array(ids),
                        tensor_to_array(local_position[contacts["valid_mask"]]),
                        tensor_to_array(local_force[contacts["valid_mask"]]),
                        tensor_to_array(local_normal[contacts["valid_mask"]]),
                        tensor_to_array(contacts["friction"][contacts["valid_mask"]]),
                        tensor_to_array(local_omega),
                        tensor_to_array(gravity),
                    )
                )
            if args.stress == "cpu" or args.verify_gpu:
                cpu_contacts = ContactBatch(
                    tensor_to_array(local_position),
                    tensor_to_array(local_force),
                    np.full(contacts["valid_mask"].shape, args.patch_radius_m),
                    tensor_to_array(contacts["valid_mask"]),
                    tensor_to_array(contacts["friction"]),
                    tensor_to_array(local_normal),
                    np.finfo(np.float32).eps,
                )
                try:
                    loads = model.map_contacts(cpu_contacts)
                except ValueError:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        args.output.with_suffix(".failed-contact.npz"),
                        step=step,
                        position_m=cpu_contacts.position_m,
                        force_n=cpu_contacts.force_n,
                        radius_m=cpu_contacts.radius_m,
                        valid=cpu_contacts.valid,
                        friction=cpu_contacts.friction,
                        inward_normal=cpu_contacts.inward_normal,
                        contact_normal_world=tensor_to_array(contacts["normal"]),
                        quat=tensor_to_array(solve_quat),
                    )
                    raise
                gravity_cpu = tensor_to_array(gravity)
                raw = loads.nodal_force_n.copy(order="F")
                for env in range(args.envs):
                    raw[:, env] += model.fem.m @ np.tile(gravity_cpu[env], len(model.fem.xyz))
                omega = tensor_to_array(local_omega)
                cpu_rhs = model.compatible_rhs(raw, omega)
                result = model.recover(cpu_rhs)
                record.update(
                    {
                        "peak_pa": result.peak_pa.tolist(),
                        "relative_residual_nonzero": result.relative_residual.tolist(),
                        "finite_patch_point_moment_difference_nm": np.linalg.norm(
                            loads.resultant_moment_nm - loads.input_moment_nm, axis=1
                        ).tolist(),
                        "source_cone_excess_n_max": [
                            max((d.source_cone_excess_n for d in env), default=0.0) for env in loads.contact_diagnostics
                        ],
                        "full_residual_absolute_n": np.linalg.norm(
                            model.fem.k @ result.displacement_m - cpu_rhs, axis=0
                        ).tolist(),
                        "rhs_norm_n": np.linalg.norm(cpu_rhs, axis=0).tolist(),
                    }
                )
            if args.stress == "gpu":
                mapped = gpu_mapper.map(
                    cp.from_dlpack(local_position),
                    cp.from_dlpack(local_force),
                    cp.full(contacts["valid_mask"].shape, args.patch_radius_m),
                    cp.from_dlpack(local_normal),
                    cp.from_dlpack(contacts["friction"]),
                    cp.from_dlpack(contacts["valid_mask"]),
                    np.finfo(np.float32).eps,
                )
                external = (
                    mapped.nodal_force_n
                    + gpu_recovery.mass @ cp.tile(cp.from_dlpack(gravity).astype(np.float64), (1, len(model.fem.xyz))).T
                )
                rhs = gpu_recovery.compatible_rhs(external, cp.from_dlpack(local_omega))
                recovered = gpu_recovery.recover(rhs)
                if not cp.all(mapped.is_accepted & recovered.is_accepted).item():
                    raise ArithmeticError("Live GPU mapping or complete recovery residual failed")
                if args.verify_gpu:
                    np.testing.assert_allclose(cp.asnumpy(rhs), cpu_rhs, atol=1e-11, rtol=1e-8)
                    peak_error = np.abs(cp.asnumpy(recovered.peak_pa) - result.peak_pa)
                    assert np.max(peak_error / np.maximum(result.peak_pa, 1)) <= 1e-4
                    record["gpu_peak_error_pa"] = peak_error.tolist()
                    record["gpu_peak_error_relative_above_1pa"] = (peak_error / np.maximum(result.peak_pa, 1)).tolist()
                record.update(
                    peak_pa=cp.asnumpy(recovered.peak_pa).tolist(),
                    full_residual_relative=cp.asnumpy(recovered.relative_residual).tolist(),
                    full_residual_absolute_n=cp.asnumpy(recovered.absolute_residual_n).tolist(),
                    rhs_norm_n=cp.asnumpy(cp.linalg.norm(rhs, axis=0)).tolist(),
                    mapping_diagnostics_max=cp.asnumpy(mapped.contact_diagnostics.max(axis=1)).tolist(),
                )
            records.append(record)
            trace.write(json.dumps(record) + "\n")
            if step % 50 == 0:
                trace.flush()
            if camera is not None and step == args.steps // 2:
                rgb, _, _, _ = camera.render()
                Image.fromarray(rgb).save(args.video.with_suffix(".png"))
        if camera is not None:
            camera.stop_recording()
        trace.close()
        if args.backend == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - start
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.save_contacts:
            offsets = np.r_[0, np.cumsum([len(item[1]) for item in contact_records])]
            np.savez_compressed(
                args.output.with_suffix(".contacts.npz"),
                offsets=offsets,
                ids=np.concatenate([item[1] for item in contact_records]),
                position_m=np.concatenate([item[2] for item in contact_records]),
                force_n=np.concatenate([item[3] for item in contact_records]),
                inward_normal=np.concatenate([item[4] for item in contact_records]),
                friction=np.concatenate([item[5] for item in contact_records]),
                omega_rad_s=np.stack([item[6] for item in contact_records]),
                gravity_m_s2=np.stack([item[7] for item in contact_records]),
                radius_m=args.patch_radius_m,
                dt=args.dt,
                source_epsilon=np.finfo(np.float32).eps,
            )
        args.output.write_text(
            json.dumps(
                {
                    "status": "integration_seed_not_accepted_benchmark",
                    "backend": args.backend,
                    "stress_backend": args.stress,
                    "envs": args.envs,
                    "steps": args.steps,
                    "dt": args.dt,
                    "substeps": 1,
                    "friction_cone": "elliptic",
                    "footprint_law": "Gaussian nonnegative pad pressure with constant rigid-frame traction ratio",
                    "friction_roundoff_allowance": "16 times FP32 epsilon times each complete contact force norm",
                    "seconds_with_first_compile": elapsed,
                    "env_steps_per_second_with_first_compile": args.steps * args.envs / elapsed,
                    "stress_mesh_converged": False,
                    "material_calibrated_to_real_egg": False,
                    "sampling": "sole rigid contact solve, paired with pre-integration authored pose and angular velocity",
                    "limitations": "validate grasp success, wrench/footprint consistency and impacts before acceptance",
                    "records": records,
                },
                indent=2,
            )
        )
        print(args.output)


if __name__ == "__main__":
    main()
