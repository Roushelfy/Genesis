"""Validate reusable full Panda grasps, asynchronous resets and every-substep device recovery."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import torch
from threadpoolctl import threadpool_limits

import genesis as gs
from genesis.utils import geom as gu
from genesis.utils.misc import qd_to_torch

from .cpu import ContactBatch, EggConfig, EggRecoveryCPU
from .panda_scene import PandaConfig, PandaEggScene
from .sparse_gpu import EggRecoveryGPU
from .temporal_gpu import TemporalRecoveryGPU
from .timing import source_hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--nominal", action="store_true")
    parser.add_argument("--synchronous", action="store_true")
    parser.add_argument("--temporal", action="store_true")
    parser.add_argument("--seed", type=int, default=510000)
    parser.add_argument("--rigid-iterations", type=int, default=100)
    parser.add_argument("--rigid-tolerance", type=float, default=1e-12)
    parser.add_argument("--partial-reset-step", type=int)
    parser.add_argument("--cpu-factor", choices=("superlu", "cholmod"), default="superlu")
    parser.add_argument("--factor-backend", choices=("spsm", "cudss"), default="spsm")
    parser.add_argument("--record-only", action="store_true", help="Record inputs without same-RHS CPU oracle solves")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--inertia", choices=("sparse", "quadratic"), default="sparse")
    parser.add_argument("--body-products", choices=("cublas", "fused"), default="cublas")
    parser.add_argument("--scatter", choices=("atomic", "warp"), default="atomic")
    args = parser.parse_args()
    hashes = source_hashes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        model = EggRecoveryCPU(
            args.envs,
            EggConfig(
                level=args.level,
                ordering="column-nd",
                factor_backend="none" if args.record_only else args.cpu_factor,
            ),
            history=0,
            direct=True,
            anchor_to_surface=True,
        )
        gs.init(backend=gs.gpu, precision="64", seed=args.seed, logging_level="warning")
        recovery = (
            TemporalRecoveryGPU(
                model.fem,
                args.envs,
                history=4,
                factor_backend=args.factor_backend,
                inertia=args.inertia,
                body_products=args.body_products,
            )
            if args.temporal
            else EggRecoveryGPU(
                model.fem,
                args.envs,
                factor_backend=args.factor_backend,
                inertia=args.inertia,
                body_products=args.body_products,
            )
        )
        scene = PandaEggScene(
            model,
            args.output.parent / "assets",
            PandaConfig(
                environments=args.envs,
                seed=args.seed,
                varied=not args.nominal,
                asynchronous=not args.synchronous,
                substeps=args.substeps,
                iterations=args.rigid_iterations,
                tolerance=args.rigid_tolerance,
                video_path=None if args.video is None else str(args.video),
                scatter=args.scatter,
            ),
            recovery,
        )
        observer = scene.observer
        inertia_rotation = gu.quat_to_R(scene.egg.links[0].desc.inertial_quat)
        authored_inertia = inertia_rotation @ scene.egg.get_links_inertia().cpu().numpy()[0] @ inertia_rotation.T
        np.testing.assert_allclose(authored_inertia, model.fem.gram[3:, 3:], rtol=1e-7, atol=1e-14)
        np.testing.assert_allclose(scene.egg.get_dofs_armature().cpu().numpy(), 0, atol=0)
        np.testing.assert_allclose(scene.egg.get_dofs_damping().cpu().numpy(), 0, atol=0)
        rows, contacts, offsets = [], [], [0]
        errors, full_residuals, moment_errors, force_errors, torque_errors = [], [], [], [], []
        near_zero = []
        gradient_force_errors, gradient_torque_errors = [], []
        force_relative_errors, torque_relative_errors = [], []
        body_inertia = torch.as_tensor(model.fem.gram[3:, 3:], dtype=gs.tc_float, device=gs.device)

        def diagnostic(local, mapped, recovered, substep):
            snapshot = observer.adapter.snapshot
            full_residuals.append(cp.asnumpy(recovered.relative_residual))
            cpu_contacts = ContactBatch(
                local.position_m.cpu().numpy(),
                local.force_n.cpu().numpy(),
                np.broadcast_to(scene.radius_m.cpu().numpy(), tuple(local.is_valid.shape)),
                local.is_valid.cpu().numpy(),
                local.friction.cpu().numpy(),
                local.inward_normal.cpu().numpy(),
                np.finfo(np.float64).eps,
            )
            if not args.record_only:
                expected = np.zeros((model.fem.ndof, args.envs))
                external = mapped.nodal_force_n + recovery.gravity_load(cp.from_dlpack(local.gravity_m_s2))
                rhs = recovery.compatible_rhs(external, cp.from_dlpack(local.omega_rad_s))
                rhs_cpu = cp.asnumpy(rhs)
                expected[model.fem.free] = model.fem.factor.solve(rhs_cpu[model.fem.free])
                truth = np.array([model.peak(expected[:, i])[0] for i in range(args.envs)])
                actual = cp.asnumpy(recovered.peak_pa)
                errors.append(abs(actual - truth) / np.maximum(truth, 1))
                near_zero.extend(abs(actual - truth)[truth <= 1].tolist())
                loads = model.map_contacts(cpu_contacts)
                np.testing.assert_allclose(cp.asnumpy(mapped.nodal_force_n), loads.nodal_force_n, atol=1e-10, rtol=1e-8)
                moment_errors.append(np.linalg.norm(loads.resultant_moment_nm - loads.input_moment_nm, axis=1))
            else:
                moment_errors.append(cp.asnumpy(mapped.contact_diagnostics[:, :, 1].sum(axis=1)))
            if not cp.all(mapped.is_accepted & recovered.is_accepted).item():
                failure = args.output.with_name(f"{args.output.stem}.failure-{scene.tick:06d}-{substep}.npz")
                np.savez_compressed(
                    failure,
                    position_m=cpu_contacts.position_m,
                    force_n=cpu_contacts.force_n,
                    inward_normal=cpu_contacts.inward_normal,
                    valid=cpu_contacts.valid,
                    friction=cpu_contacts.friction,
                    radius_m=cpu_contacts.radius_m,
                    omega_rad_s=local.omega_rad_s.cpu().numpy(),
                    gravity_m_s2=local.gravity_m_s2.cpu().numpy(),
                    contact_status=mapped.contact_status.get(),
                    contact_diagnostics=mapped.contact_diagnostics.get(),
                    full_relative_residual=recovered.relative_residual.get(),
                    full_absolute_residual_n=recovered.absolute_residual_n.get(),
                    rhs_norm_n=recovered.rhs_norm_n.get(),
                    peak_pa=recovered.peak_pa.get(),
                    is_accepted=recovered.is_accepted.get(),
                )
                raise ArithmeticError(
                    f"Rejected complete observation at frame {scene.tick}, substep {substep}: {failure}"
                )
            post_offset = gu.transform_by_quat(scene.com_body, scene.egg.get_quat())
            alpha_world = scene.egg.get_links_acc_ang()[:, 0]
            spatial = (
                scene.egg.get_links_acc()[:, 0]
                + torch.cross(alpha_world, post_offset, dim=-1)
                - torch.cross(scene.egg.get_ang(), scene.egg.get_vel(), dim=-1)
            )
            old_offset = gu.transform_by_quat(scene.com_body, snapshot.quaternion)
            old_com_velocity = snapshot.linear_velocity_m_s + torch.cross(
                snapshot.angular_velocity_rad_s, old_offset, dim=-1
            )
            measured_com_acc = spatial + torch.cross(snapshot.angular_velocity_rad_s, old_com_velocity, dim=-1)
            measured_body_acc = gu.inv_transform_by_quat(measured_com_acc, snapshot.quaternion)
            measured_body_alpha = gu.inv_transform_by_quat(alpha_world, snapshot.quaternion)
            force_balance = local.force_n.sum(dim=1) + model.fem.mass * (local.gravity_m_s2 - measured_body_acc)
            torque = torch.cross(local.position_m - scene.com_body, local.force_n, dim=-1).sum(dim=1)
            torque_balance = torque - (
                measured_body_alpha @ body_inertia.T
                + torch.cross(local.omega_rad_s, local.omega_rad_s @ body_inertia.T, dim=-1)
            )
            force_errors.append(torch.linalg.vector_norm(force_balance, dim=-1).cpu().numpy())
            torque_errors.append(torch.linalg.vector_norm(torque_balance, dim=-1).cpu().numpy())
            gradient = qd_to_torch(
                scene.scene.rigid_solver.constraint_solver.grad,
                transpose=True,
                copy=True,
            )[:, scene.egg.dof_start : scene.egg.dof_end]
            force_gradient_body = gu.inv_transform_by_quat(gradient[:, :3], snapshot.quaternion)
            torque_origin_balance = torque_balance + torch.cross(
                scene.com_body.expand_as(force_balance), force_balance, dim=-1
            )
            gradient_force_errors.append(
                torch.linalg.vector_norm(force_gradient_body + force_balance, dim=-1).cpu().numpy()
            )
            gradient_torque_errors.append(
                torch.linalg.vector_norm(gradient[:, 3:] + torque_origin_balance, dim=-1).cpu().numpy()
            )
            force_scale = torch.linalg.vector_norm(local.force_n.sum(dim=1), dim=-1) + model.fem.mass * (
                torch.linalg.vector_norm(local.gravity_m_s2, dim=-1)
                + torch.linalg.vector_norm(measured_body_acc, dim=-1)
            )
            torque_scale = torch.linalg.vector_norm(torque, dim=-1) + torch.linalg.vector_norm(
                measured_body_alpha @ body_inertia.T, dim=-1
            )
            force_relative_errors.append(
                (torch.linalg.vector_norm(force_balance, dim=-1) / force_scale.clamp(min=1e-8)).cpu().numpy()
            )
            torque_relative_errors.append(
                (torch.linalg.vector_norm(torque_balance, dim=-1) / torque_scale.clamp(min=1e-8)).cpu().numpy()
            )
            valid = local.is_valid.cpu().numpy()
            ids = np.column_stack(np.nonzero(valid))
            contacts.append(
                (
                    ids,
                    cpu_contacts.position_m[valid],
                    cpu_contacts.force_n[valid],
                    cpu_contacts.inward_normal[valid],
                    cpu_contacts.friction[valid],
                    cpu_contacts.radius_m[valid],
                    local.omega_rad_s.cpu().numpy(),
                    local.gravity_m_s2.cpu().numpy(),
                    local.is_egg_a.cpu().numpy()[valid],
                )
            )
            offsets.append(offsets[-1] + len(ids))

        observer.diagnostic = diagnostic
        if scene.camera is not None:
            scene.camera.start_recording(save_to_filename=str(args.video), fps=round(1 / scene.config.dt_s))
        with args.output.with_suffix(".frames.jsonl").open("w") as trace:
            for step in range(args.steps):
                if step == args.partial_reset_step:
                    if args.envs < 2:
                        raise ValueError("Partial-reset diagnostic requires at least two environments")
                    scene.reset(np.array([1]))
                scene.step()
                raw = scene.egg.get_contacts(is_padded=True)
                egg_a = (raw["geom_a"] >= scene.egg.geom_start) & (raw["geom_a"] < scene.egg.geom_end)
                opposite = torch.where(egg_a, raw["link_b"], raw["link_a"])
                grasp = (opposite >= scene.robot.link_start) & (opposite < scene.robot.link_end) & raw["valid_mask"]
                position = scene.egg.get_pos() + gu.transform_by_quat(scene.com_body, scene.egg.get_quat())
                row = {
                    "step": step,
                    "phase": scene.phase.cpu().tolist(),
                    "com_height_m": position[:, 2].cpu().tolist(),
                    "grasp_contacts": grasp.sum(dim=1).cpu().tolist(),
                    "peak_pa": cp.asnumpy(observer.peak_pa).tolist(),
                    "accepted": cp.asnumpy(observer.is_accepted).tolist(),
                    "reset_count": scene.reset_count,
                    "newton_force_error_n": np.max(force_errors[-args.substeps :], axis=0).tolist(),
                    "euler_torque_error_nm": np.max(torque_errors[-args.substeps :], axis=0).tolist(),
                }
                rows.append(row)
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                if step % 100 == 0:
                    print(json.dumps({"step": step, "resets": scene.reset_count}), flush=True)
        height, phase = np.array([row["com_height_m"] for row in rows]), np.array([row["phase"] for row in rows])
        grasp = np.array([row["grasp_contacts"] for row in rows])
        success = []
        for environment in range(args.envs):
            hold = (phase[:, environment] >= 0.5) & (phase[:, environment] < 0.6)
            success.append(
                bool(np.any(hold) and np.all(height[hold, environment] > 0.11) and np.all(grasp[hold, environment] > 0))
            )
        if not args.record_only:
            assert float(np.max(errors)) <= 1e-4
            assert max(near_zero, default=0) <= 1e-3
        if scene.camera is not None:
            scene.camera.stop_recording()
        recorded_force = np.concatenate([item[2] for item in contacts])
        recorded_normal = np.concatenate([item[3] for item in contacts])
        normal_force = np.einsum("ij,ij->i", recorded_force, recorded_normal)
        tangent = recorded_force - normal_force[:, None] * recorded_normal
        recorded_a = np.concatenate([item[8] for item in contacts])
        report = {
            "GPU_tested": True,
            "equations_passed": True,
            "envs": args.envs,
            "steps": args.steps,
            "substeps": args.substeps,
            "seed": args.seed,
            "varied": not args.nominal,
            "asynchronous": not args.synchronous,
            "temporal": args.temporal,
            "mesh_level": args.level,
            "source_sha256_at_start": hashes,
            "tangential_force_max_n": float(np.linalg.norm(tangent, axis=1).max(initial=0)),
            "egg_as_a_contacts": int(recorded_a.sum()),
            "egg_as_b_contacts": int(len(recorded_a) - recorded_a.sum()),
            "physical_mesh_converged": False,
            "CPU_direct_verified": not args.record_only,
            "peak_relative_error_max": None if args.record_only else float(np.max(errors)),
            "full_relative_residual_max": float(np.max(full_residuals)),
            "near_zero_absolute_peak_error_max_pa": None if args.record_only else max(near_zero, default=0),
            "moment_error_max_nm": float(np.max(moment_errors)),
            "newton_force_error_max_n": float(np.max(force_errors)),
            "euler_torque_error_max_nm": float(np.max(torque_errors)),
            "newton_force_relative_error_max": float(np.max(force_relative_errors)),
            "euler_torque_relative_error_max": float(np.max(torque_relative_errors)),
            "force_gradient_difference_max_n": float(np.max(gradient_force_errors)),
            "torque_gradient_difference_max_nm": float(np.max(gradient_torque_errors)),
            "reset_count": scene.reset_count,
            "grasp_success_fraction": sum(success) / len(success),
            "grasp_success": success,
            "records": rows,
        }
        np.savez_compressed(
            args.output.with_suffix(".contacts.npz"),
            offsets=np.array(offsets),
            ids=np.concatenate([item[0] for item in contacts]),
            position_m=np.concatenate([item[1] for item in contacts]),
            force_n=np.concatenate([item[2] for item in contacts]),
            inward_normal=np.concatenate([item[3] for item in contacts]),
            friction=np.concatenate([item[4] for item in contacts]),
            radius_m=np.concatenate([item[5] for item in contacts]),
            omega_rad_s=np.stack([item[6] for item in contacts]),
            gravity_m_s2=np.stack([item[7] for item in contacts]),
            reset_delay=scene.delay,
            dt_s=scene.config.dt_s / args.substeps,
            source_epsilon=np.finfo(np.float64).eps,
            is_egg_a=recorded_a,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
