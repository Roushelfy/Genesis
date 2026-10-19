"""Independent physical-pad states advanced by one batched native scene step."""

import json

import numpy as np

import genesis as gs
from genesis.utils.geom import quat_to_R
from genesis.utils.misc import qd_to_numpy, tensor_to_array

from .artifacts import file_id
from .plates import configure_native_plate, plate_xml
from .rigid_pad_coupling import RigidPadState, carry_normal, prepare_geometry, solve


def motions(link, center_local):
    position, quaternion, velocity, angular = map(
        tensor_to_array, (link.get_pos(), link.get_quat(), link.get_vel(), link.get_ang())
    )
    rotation = np.stack([quat_to_R(q) for q in quaternion])
    offset = rotation @ center_local
    return position, rotation, position + offset, velocity + np.cross(angular, offset), angular


class NativeBatchPad:
    def __init__(self, args):
        self.args = args
        assets = json.loads((args.assets / "result.json").read_text())
        assert file_id(args.mesh)["sha256"] == assets["source"]["sha256"]
        self.center_local = np.array(assets["com_m"])
        self.inertia_local = np.array(assets["exact_volume_inertia_kg_m2"])
        with np.load(args.mesh) as saved:
            self.outer, self.faces = saved["outer_vertices"], saved["mid_faces"]
        triangle = self.outer[self.faces]
        self.points = triangle.mean(axis=1)
        self.area = np.abs(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])[:, 2]) / 2
        self.top = self.points[:, 2] > 0
        self.sign = np.where(self.top, -1.0, 1.0)
        self.lower, self.upper = (
            float(self.outer[:, 2].min() - args.clearance),
            float(self.outer[:, 2].max() + args.clearance),
        )
        self.xml = plate_xml(self.lower, self.upper)
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(gravity=(0.0, 0.0, 0.0), dt=args.dt),
            rigid_options=gs.options.RigidOptions(
                use_hibernation=False, enable_collision=False, integrator=gs.integrator.Euler
            ),
            show_viewer=False,
        )
        self.plate = scene.add_entity(
            morph=gs.morphs.MJCF(file=self.xml, default_armature=0.0), material=gs.materials.Rigid()
        )
        self.egg = scene.add_entity(
            morph=gs.morphs.URDF(
                file=args.assets / "egg_shell.urdf", align=False, decimate=False, preserve_mesh_precision=True
            ),
            material=gs.materials.Rigid(),
        )
        scene.build(n_envs=args.envs)
        configure_native_plate(self.plate, args)
        self.scene, self.link = scene, self.egg.base_link
        mass = tensor_to_array(self.link.get_mass())
        np.testing.assert_allclose(mass, assets["mass_kg"], rtol=1e-12)
        self.mass_kg = float(assets["mass_kg"])
        actual_inertia = tensor_to_array(scene.rigid_solver.get_links_inertia([self.link.idx])).reshape(-1, 3, 3)
        for tensor in actual_inertia:
            np.testing.assert_allclose(np.linalg.eigvalsh(tensor), np.linalg.eigvalsh(self.inertia_local), rtol=1e-10)
        self.pads = [RigidPadState.allocate(self.points, self.area) for _ in range(args.envs)]

    def advance(self, commands, step):
        args = self.args
        assert commands.shape == (args.envs,)
        old = motions(self.link, self.center_local)
        position, rotation, center, velocity, angular = old
        q, plate_velocity = [
            tensor_to_array(v)[:, 0] for v in (self.plate.get_dofs_position(), self.plate.get_dofs_velocity())
        ]
        inertia = rotation @ self.inertia_local @ rotation.transpose(0, 2, 1)
        gravity = args.gravity_x * min(max((step * args.dt - args.gravity_start) / args.gravity_ramp, 0), 1)
        body_force = np.array([self.mass_kg * gravity, 0.0, 0.0])
        forces, worlds, predicted, previous_normal, iterations, solve_residuals = [], [], [], [], [], []
        for env, pad in enumerate(self.pads):
            mass = np.zeros((7, 7))
            mass[:3, :3] = self.mass_kg * np.eye(3)
            mass[3:6, 3:6] = inertia[env] / 0.03**2
            mass[6, 6] = 0.05 + args.dt * args.damping
            free = np.r_[
                velocity[env] + args.dt * body_force / self.mass_kg,
                0.03
                * (
                    angular[env]
                    - args.dt * np.linalg.solve(inertia[env], np.cross(angular[env], inertia[env] @ angular[env]))
                ),
                (0.05 * plate_velocity[env] + args.dt * commands[env]) / mass[6, 6],
            ]
            prepare_geometry(position[env], rotation[env], center[env], self.lower, self.upper - q[env], pad)
            count, residual = solve(pad, mass, free, args.dt, 1e9, args.friction, commands[env])
            iterations.append(count)
            solve_residuals.append(residual)
            forces.append(qd_to_numpy(pad.force, transpose=True))
            worlds.append(qd_to_numpy(pad.world, transpose=True))
            predicted.append(qd_to_numpy(pad.velocity, transpose=True))
            previous_normal.append(qd_to_numpy(pad.previous_normal, transpose=True))
        forces, worlds = np.array(forces), np.array(worlds)
        total_force = forces.sum(axis=1)
        torque = np.cross(worlds - center[:, None], forces).sum(axis=1)
        self.link.apply_external_wrench(force=total_force + body_force, torque=torque, ref=gs.link_ref_frame.link_COM)
        for name, mask in (("upper", self.top), ("lower", ~self.top)):
            link = self.plate.get_link(name)
            plate_center = tensor_to_array(link.get_pos())
            link.apply_external_wrench(
                force=-forces[:, mask].sum(axis=1),
                torque=-np.cross(worlds[:, mask] - plate_center[:, None], forces[:, mask]).sum(axis=1),
                ref=gs.link_ref_frame.link_COM,
            )
        self.plate.control_dofs_force(commands[:, None])
        self.scene.step()
        self.scene.rigid_solver.check_errno()
        self.scene.rigid_solver.update_forward_pos()
        new_position, new_rotation, new_center, new_velocity, new_angular = motions(self.link, self.center_local)
        new_q, new_plate_velocity = [
            tensor_to_array(v)[:, 0] for v in (self.plate.get_dofs_position(), self.plate.get_dofs_velocity())
        ]
        rows, frames = [], []
        for env, pad in enumerate(self.pads):
            world = self.points @ new_rotation[env].T + new_position[env]
            force = forces[env]
            gap = np.where(self.top, self.upper - new_q[env] - world[:, 2], world[:, 2] - self.lower)
            normal = 1e9 * self.area * np.maximum(-gap, 0)
            relative_velocity = new_velocity[env] + np.cross(new_angular[env], world - new_center[env])
            relative_velocity[self.top, 2] += new_plate_velocity[env]
            speed = np.linalg.norm(relative_velocity[:, :2], axis=1)
            ratio = np.minimum(speed / 1e-4, 1)
            tangent = (
                -(args.friction * previous_normal[env] * (2 * ratio - ratio**2))[:, None]
                * relative_velocity[:, :2]
                / np.maximum(speed[:, None], 1e-30)
            )
            inertial = self.mass_kg * (new_velocity[env] - velocity[env]) / args.dt
            new_inertia = new_rotation[env] @ self.inertia_local @ new_rotation[env].T
            angular_rate = (new_inertia @ new_angular[env] - inertia[env] @ angular[env]) / args.dt
            scale = max(np.linalg.norm(force), 1e-12)
            rows.append(
                {
                    "step": step,
                    "time_s": step * args.dt,
                    "env": env,
                    "command_force_n": float(commands[env]),
                    "compression_coordinate_m": float(new_q[env]),
                    "plate_speed_m_per_s": float(new_plate_velocity[env]),
                    "upper_reaction_n": float(-force[self.top, 2].sum()),
                    "lower_reaction_n": float(force[~self.top, 2].sum()),
                    "body_force_n": body_force.tolist(),
                    "tangential_resultant_n": total_force[env, :2].tolist(),
                    "egg_inertia_resultant_norm_n": float(np.linalg.norm(inertial)),
                    "egg_kinetic_j": float(
                        0.5 * self.mass_kg * np.dot(new_velocity[env], new_velocity[env])
                        + 0.5 * new_angular[env] @ new_inertia @ new_angular[env]
                    ),
                    "equation_residual_relative": float(
                        np.linalg.norm(total_force[env] + body_force - inertial) / max(commands[env], 1e-3)
                    ),
                    "angular_equation_residual_nm": float(np.linalg.norm(torque[env] - angular_rate)),
                    "frozen_point_moment_defect_nm": float(
                        np.linalg.norm(np.cross(world - worlds[env], force).sum(axis=0))
                    ),
                    "native_prediction_velocity_error_m_per_s": float(
                        np.max(
                            np.abs(
                                np.r_[new_velocity[env], 0.03 * new_angular[env], new_plate_velocity[env]]
                                - predicted[env]
                            )
                        )
                    ),
                    "normal_law_relative_error": float(np.linalg.norm(self.sign * force[:, 2] - normal) / scale),
                    "friction_law_relative_error": float(np.linalg.norm(force[:, :2] - tangent) / scale),
                    "newton_iterations": iterations[env],
                    "newton_momentum_residual": solve_residuals[env],
                }
            )
            frames.append(
                dict(
                    points_world_m=world,
                    applied_points_world_m=worlds[env],
                    contact_force_n=force,
                    projected_area_m2=self.area,
                    body_gap_m=gap,
                    center_m=new_center[env],
                    rotation=new_rotation[env],
                    velocity_m_per_s=new_velocity[env],
                    angular_velocity_rad_per_s=new_angular[env],
                    previous_normal_n=previous_normal[env],
                )
            )
            carry_normal(pad)
        return rows, forces, new_rotation, new_angular, frames
